#include "resource_allocator.hpp"

#include "ast.hpp"
#include "aux.hpp"
#include "context.hpp"
#include "io.hpp"

using namespace Choreo;

namespace Choreo {

Option<bool> dma_alloc_mode(
    OptionKind::User, "-fdma-alloc", "", true,
    "Enable liveness-driven DMA future allocation: color FUTURE handles into "
    "a minimal set of completion slots. Futures are only pooled where the "
    "target provides a pooled completion path (e.g. a DTE pool); where no "
    "such path is enabled this flag has no observable effect. Disable with "
    "-fdma-alloc=false to fall back to monotonic slot assignment.");

Option<std::string> event_alloc_mode(
    OptionKind::User, "-fevent-alloc", "", "simple",
    "Select event allocation: color EVENT handles into a minimal barrier set. "
    "off = monotonic fallback; simple = color scalar events (default); "
    "full = scalar-replace event arrays then color (planned).",
    "", /*requires_arg=*/true);

EventAllocMode ParseEventAllocMode() {
  const std::string mode = ToLower(event_alloc_mode.GetValue());
  if (mode == "off") return EventAllocMode::Off;
  if (mode == "full") return EventAllocMode::Full;
  if (mode == "simple") return EventAllocMode::Simple;
  errs() << "warning: unknown -fevent-alloc mode '" << mode
         << "'; using 'simple'\n";
  return EventAllocMode::Simple;
}

}  // namespace Choreo

bool DmaResourceAllocator::RunOnProgramImpl(AST::Node& root) {
  DmaResourcePlan::Reset();

  la.SetLevelPrefix("  ");
  la.SSTab().UpdateGlobal(SymTab());
  if (!la.RunOnProgram(root)) return la.Status();

  root.accept(*this);

  return !HasError();
}

bool DmaResourceAllocator::BeforeVisitImpl(AST::Node& n) {
  if (isa<AST::ChoreoFunction>(&n)) {
    cur_dev_fname = CurrentFunctionName();
  } else if (auto pb = dyn_cast<AST::ParallelBy>(&n)) {
    if (pb->IsDeviceEntry()) cur_dev_fname = SSTab().ScopeName();
  }
  return true;
}

bool DmaResourceAllocator::AfterVisitImpl(AST::Node& n) {
  if (auto pb = dyn_cast<AST::ParallelBy>(&n)) {
    if (pb->IsDeviceEntry()) {
      if (dma_alloc_mode)
        AllocateClass(LivenessAnalyzer::ResourceClass::FUTURE, cur_dev_fname);
      if (EventAllocEnabled())
        AllocateClass(LivenessAnalyzer::ResourceClass::EVENT, cur_dev_fname);
      cur_dev_fname = CurrentFunctionName();
    }
  }
  return true;
}

const LivenessAnalyzer::HBGraph*
DmaResourceAllocator::FindHBGraph(const std::string& df_name) const {
  const auto& graphs = la.HBGraphs();
  auto it = graphs.find(df_name);
  if (it != graphs.end()) return &it->second;
  // Fallback: most specific graph whose scope is a prefix of df_name.
  const LivenessAnalyzer::HBGraph* best = nullptr;
  size_t best_len = 0;
  for (const auto& [scope, graph] : graphs) {
    if (PrefixedWith(df_name, scope) && scope.size() > best_len) {
      best_len = scope.size();
      best = &graph;
    }
  }
  return best;
}

void DmaResourceAllocator::AllocateClass(LivenessAnalyzer::ResourceClass rc,
                                         const std::string& df_name) {
  auto& plan = DmaResourcePlan::Plans()[df_name];

  HeapSimulator::Chunks chunks;
  for (const auto& [name, ranges] : la.ResourceRanges(rc)) {
    if (!PrefixedWith(name, df_name)) continue;
    // SHARED-storage futures are routed through named CDTE contexts, not the
    // SDTE pool, so they must not consume a pool slot.  This mirrors the
    // storage half of the codegen `use_pool = use_dte_pool &&
    // sto != Storage::SHARED` condition; the target-specific `use_dte_pool`
    // switch is applied at the codegen consumption site, not here.
    if (rc == LivenessAnalyzer::ResourceClass::FUTURE && !la.IsPoolFuture(name))
      continue;
    HeapSimulator::Chunk chunk;
    chunk.size = 1; // each DMA handle occupies one completion slot
    chunk.ranges = ranges.Values();
    chunk.buffer_id = name;
    chunk.Sort();
    chunks.push_back(std::move(chunk));
  }

  if (chunks.empty()) return;

  const LivenessAnalyzer::HBGraph* hb = FindHBGraph(df_name);
  HeapSimulator::HBOverride hb_override = nullptr;
  HeapSimulator::HBMustInterfere hb_must_interfere = nullptr;
  if (hb) {
    hb_override = [hb](const std::string& a, const std::string& b) {
      return hb->CanOverlap(a, b);
    };
    hb_must_interfere = [hb](const std::string& a, const std::string& b) {
      return hb->IsUnsafeMultiInstanceOverlap(a, b);
    };
  }

  HeapSimulator simulator;
  HeapSimulator::Result result = simulator.Allocate(
      chunks, /*alignment=*/0, hb_override, hb_must_interfere);

  auto& slots = (rc == LivenessAnalyzer::ResourceClass::FUTURE)
                    ? plan.future_slots
                    : plan.event_slots;
  for (const auto& [name, offset] : result.chunk_offsets) slots[name] = offset;

  if (rc == LivenessAnalyzer::ResourceClass::FUTURE)
    plan.future_slot_count = result.heap_size;
  else
    plan.event_slot_count = result.heap_size;
}
