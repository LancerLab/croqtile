#include "pipeline.hpp"
#include "active_threads.hpp"
#include "codegen.hpp"
#include "codegen_prepare.hpp"
#include "target_utils.hpp"
#include "colors.hpp"
#include "earlysema.hpp"
#include "interval.hpp"
#include "latenorm.hpp"
#include "liveness_analysis.hpp"
#include "loop_vectorize.hpp"
#include "mem_reuse.hpp"
#include "memcheck.hpp"
#include "normalize.hpp"
#include "semacheck.hpp"
#include "shapeinfer.hpp"
#include "symbexpr.hpp"
#include "typeinfer.hpp"
#include "visualize.hpp"
#include <chrono>
#include <iomanip>

extern Choreo::AST::Program root;

void Choreo::CompilationContext::SavePreSemaRoot(AST::Node& r) {
  pre_sema_root_ = r.Clone();
}

// --- TargetCompilationState implementation ---

struct Choreo::CompilationContext::TargetCompilationState::CGIHolder {
  std::unique_ptr<CodeGenInfo> cgi;
};

Choreo::CompilationContext::TargetCompilationState::TargetCompilationState() = default;
Choreo::CompilationContext::TargetCompilationState::~TargetCompilationState() = default;
Choreo::CompilationContext::TargetCompilationState::TargetCompilationState(TargetCompilationState&&) noexcept = default;
Choreo::CompilationContext::TargetCompilationState&
Choreo::CompilationContext::TargetCompilationState::operator=(TargetCompilationState&&) noexcept = default;

Choreo::CompilationContext::TargetCompilationState
Choreo::CompilationContext::SaveTargetState() {
  TargetCompilationState s;
  s.target = std::move(compile_target);
  s.archs = archs;
  s.output_kind = out_kind;
  s.sym_tab = sym_tab;
  s.anonymous_count = SymbolTable::anonymous_count;
  s.anon_pb_count = SymbolTable::anon_pb_count;
  s.cgi_holder = std::make_unique<TargetCompilationState::CGIHolder>();
  s.cgi_holder->cgi = std::move(CodeGenInfo::instance);
  s.pl_depth_map = std::move(pl_depth_map_);
  return s;
}

void Choreo::CompilationContext::RestoreTargetState(TargetCompilationState&& s) {
  compile_target = std::move(s.target);
  archs = s.archs;
  out_kind = s.output_kind;
  sym_tab = s.sym_tab;
  SymbolTable::anonymous_count = s.anonymous_count;
  SymbolTable::anon_pb_count = s.anon_pb_count;
  CodeGenInfo::instance = std::move(s.cgi_holder->cgi);
  pl_depth_map_ = std::move(s.pl_depth_map);
}

using namespace Choreo;

std::once_flag ASTPipeline::init_flag;
std::unique_ptr<ASTPipeline> ASTPipeline::instance;

void ASTPipeline::Dump() const {
  dbgs() << "++ Pipeline Stages\n";
  for (auto& ps : pl)
    if ((ps.pred && ps.pred()) || !ps.pred)
      if (ps.v) dbgs() << " |-" << ps.v->GetName() << "\n";

  dbgs() << "++ END Pipeline\n";
}

void ASTPipeline::PrintPassTimings(const std::vector<PassTimingEntry>& timings,
                                   double total_ms) const {
  const char* sep =
      "===-------------------------------------------------------------------"
      "----===";
  errs() << "\n"
         << sep << "\n"
         << "                      ... Pass Execution Timing ...\n"
         << sep << "\n";
  errs() << std::right << std::setw(10) << "Time (ms)" << "  " << std::setw(6)
         << "  %" << "  " << "Pass\n";
  errs() << std::string(40, '-') << "\n";
  for (const auto& e : timings) {
    double pct = total_ms > 0 ? (e.ms / total_ms) * 100.0 : 0.0;
    errs() << std::right << std::setw(10) << std::fixed << std::setprecision(2)
           << e.ms << "  " << std::setw(5) << std::fixed << std::setprecision(1)
           << pct << "%"
           << "  " << e.name << "\n";
  }
  errs() << std::string(40, '-') << "\n";
  errs() << std::right << std::setw(10) << std::fixed << std::setprecision(2)
         << total_ms << "  " << std::setw(5) << "100.0"
         << "%"
         << "  " << "Total\n";
  errs() << sep << "\n";
}

void ASTPipeline::ValidatePassNames() const { Visitor::ValidatePassEnvVars(); }

bool ASTPipeline::RunOnProgram(AST::Node& root) {
  if (debug) Dump();
  // verify the input
  if (CCtx().VerifyVisitors()) vf.RunOnProgram(root);

  const bool do_time = CCtx().TimePasses();
  std::vector<PassTimingEntry> timings;
  auto wall_start = std::chrono::steady_clock::now();
  bool pass_failed = false;

#if CHOREO_ENABLE_SBE_STATS
  // Reset SBE counters only when stats output is requested and profiling is
  // compiled in.
  if (CCtx().PrintStats()) sbe::SBEProfiler::Get().Reset();
#endif
  if (CCtx().PrintStats()) sbe::IntervalProfiler::Get().Reset();

  for (auto& ps : pl) {
    if ((ps.pred && ps.pred()) || !ps.pred) {
      if (ps.v) {
        auto t0 = std::chrono::steady_clock::now();
        bool pass_ok = ps.v->RunOnProgram(root);
        if (do_time) {
          auto t1 = std::chrono::steady_clock::now();
          double ms =
              std::chrono::duration<double, std::milli>(t1 - t0).count();
          timings.push_back({ps.v->GetName(), ms});
        }
        if (!pass_ok) {
          state = ps.v->Status();
          pass_failed = true;
          break;
        }
        symtab = ps.v->SymTab();
        // TODO: force abend when failing on verifiers
        if (CCtx().VerifyVisitors()) vf.RunOnProgram(root);
      }
      if (ps.cond_action) ps.cond_action(*this);
    }

    if (abend) break;
    if (ps.action) ps.action(*this);
  }

  if (do_time) {
    auto wall_end = std::chrono::steady_clock::now();
    double total_ms =
        std::chrono::duration<double, std::milli>(wall_end - wall_start)
            .count();
    PrintPassTimings(timings, total_ms);
  }

  if (abend || pass_failed) return false;

  if (CCtx().PrintStats()) {
    const auto& s = CCtx().GetAssessmentStats();
    const char* sep =
        "===-------------------------------------------------------------------"
        "----===";
    errs() << "\n"
           << color::err(color::kBold) << sep << "\n"
           << "                      ... Assessment Statistics ...\n"
           << sep << color::err(color::kReset) << "\n";
    auto row = [&](size_t n, const char* desc) {
      errs() << color::err(color::kBold) << std::right << std::setw(6) << n
             << color::err(color::kReset) << "  assess  - " << desc << "\n";
    };
    row(s.total, "Assessments evaluated");
    row(s.static_true, "Resolved at compile time (static-true)");
    row(s.static_false, "Proven false at compile time (static-false)");
    row(s.runtime_total, "Runtime assertions generated");
    row(s.runtime_entry, "Runtime assertions (entry cost)");
    row(s.runtime_low, "Runtime assertions (low cost)");
    row(s.runtime_medium, "Runtime assertions (medium cost)");
    row(s.runtime_high, "Runtime assertions (high cost)");
    row(s.runtime_enabled, "Runtime assertions enabled");
    row(s.runtime_disabled, "Runtime assertions disabled by cost filter");
    errs() << color::err(color::kDim) << "  ---" << color::err(color::kReset)
           << "\n";
    row(s.unclassified_total, "Assessments (unclassified)");
    row(s.shape_compat_total, "Assessments (shape-compatibility)");
    row(s.elem_access_total, "Assessments (element-access)");
    row(s.loop_bound_total, "Assessments (loop-bound)");
    row(s.hw_constraint_total, "Assessments (hw-constraint)");
    row(s.unclassified_runtime, "Runtime assertions (unclassified)");
    row(s.shape_compat_runtime, "Runtime assertions (shape-compatibility)");
    row(s.elem_access_runtime, "Runtime assertions (element-access)");
    row(s.loop_bound_runtime, "Runtime assertions (loop-bound)");
    row(s.hw_constraint_runtime, "Runtime assertions (hw-constraint)");

#if CHOREO_ENABLE_SBE_STATS
    // SBE stats live under --stats so they follow the same colored reporting
    // path as the rest of the assessment summary.
    errs() << color::err(color::kDim) << "  ---" << color::err(color::kReset)
           << "\n";
    auto ss = sbe::SBEProfiler::Get().Snapshot();
    errs() << color::err(color::kBold) << sep << "\n"
           << "                          ... SBE Statistics ...\n"
           << sep << color::err(color::kReset) << "\n";
    auto sbe_row = [&](uint64_t n, const char* desc) {
      errs() << color::err(color::kBold) << std::right << std::setw(6) << n
             << color::err(color::kReset) << "  sbe     - " << desc << "\n";
    };
    sbe_row(ss.expression_created, "Symbolic expressions created");
    sbe_row(ss.symbolic_value_created, "Symbolic values created");
    sbe_row(ss.unary_operation_created, "Unary operations created");
    sbe_row(ss.binary_operation_created, "Binary operations created");
    sbe_row(ss.ternary_operation_created, "Ternary operations created");
    sbe_row(ss.normalize_calls, "Normalize() calls");
    sbe_row(ss.normalize_iterations, "Normalize() loop iterations");
    sbe_row(ss.hash_calls, "Hash() calls");
#endif

    {
      errs() << color::err(color::kDim) << "  ---" << color::err(color::kReset)
             << "\n";
      auto is = sbe::IntervalProfiler::Get().Snapshot();
      errs() << color::err(color::kBold) << sep << "\n"
             << "                    ... Interval Analysis Statistics ...\n"
             << sep << color::err(color::kReset) << "\n";
      auto iv_row = [&](uint64_t n, const char* desc) {
        errs() << color::err(color::kBold) << std::right << std::setw(6) << n
               << color::err(color::kReset) << "  intval  - " << desc << "\n";
      };
      iv_row(is.project_calls, "ProjectConstraint() calls");
      iv_row(is.eval_pred_calls, "EvalPredInterval() calls");
      iv_row(is.proven_true, "Predicates proven true");
      iv_row(is.proven_false, "Predicates proven false");
      iv_row(is.unknown, "Predicates with unknown result");
    }

    const auto& vs = CCtx().GetVectorizerStats();
    if (vs.loops_analyzed > 0) {
      errs() << color::err(color::kBold) << sep << "\n"
             << "                      ... Vectorizer Statistics ...\n"
             << sep << color::err(color::kReset) << "\n";
      auto vec_row = [&](size_t n, const char* desc) {
        errs() << color::err(color::kBold) << std::right << std::setw(6) << n
               << color::err(color::kReset) << "  vec     - " << desc << "\n";
      };
      vec_row(vs.loops_analyzed, "Loops analyzed");
      vec_row(vs.loops_vectorized, "Loops vectorized");
      vec_row(vs.loops_rejected, "Loops rejected");
      vec_row(vs.loops_hinted, "Loops with vectorize() hints");
      vec_row(vs.max_vector_factor, "Max vector factor chosen");
      vec_row(vs.masks_generated, "Masks generated");
    }

    const auto& ms = CCtx().GetMemReuseStats();
    if (ms.buffers_analyzed > 0) {
      errs() << color::err(color::kBold) << sep << "\n"
             << "                    ... Memory Reuse Statistics ...\n"
             << sep << color::err(color::kReset) << "\n";
      auto mr_row = [&](size_t n, const char* desc) {
        errs() << color::err(color::kBold) << std::right << std::setw(6) << n
               << color::err(color::kReset) << "  mreuse  - " << desc << "\n";
      };
      mr_row(ms.buffers_analyzed, "Buffers analyzed");
      mr_row(ms.static_buffers, "Static-size buffers");
      mr_row(ms.dynamic_buffers, "Dynamic-size buffers (JIT heap sim)");
      mr_row(ms.device_functions, "Device functions with reuse");
      mr_row(ms.total_buffer_bytes, "Total buffer bytes (before reuse)");
      mr_row(ms.total_static_heap_bytes, "Static heap bytes (after reuse)");
    }

    errs() << color::err(color::kBold) << sep << color::err(color::kReset)
           << "\n";
  }

  return !abend;
}

ASTPipeline& ASTPipeline::PlanSemanticRoutine() {
  // For heterogeneous compilation, save a pre-sema clone of the AST so that
  // device offload functions can be compiled from a clean (artifact-free) AST.
  if (CCtx().NeedPreSemaClone())
    AddAction([](ASTPipeline&) { CCtx().SavePreSemaRoot(root); });

  // Initialize the common ast pipeline.
  // apply early semantics check without knowing type details
  AddStage<EarlySemantics>();
  // minor AST change: desugar for canonicalized AST
  AddStage<Normalizer>();
  // perform shape inference of mdspans, future, etc.
  AddStage<ShapeInference>();
  // inference all the unknown types - decls
  AddStage<TypeInference>();
  AddAction([](ASTPipeline& p) {
    if (CCtx().ShowInferredTypes() || CCtx().TraceValueNumbers()) p.SetAbend();
    CCtx().SetGlobalSymbolTable(p.LastSymTab());
  });
  // late normalize
  AddStage<LateNorm>();
  AddAction([](ASTPipeline& p) {
    CCtx().SetGlobalSymbolTable(p.LastSymTab());
    // debug: dump the symbol table
    if (std::getenv("DUMP_SYMTAB") || CCtx().DumpSymtab())
      CCtx().GetGlobalSymbolTable()->Print(dbgs());
  });

  // to visualize the dma
  if (std::getenv("VISUALIZE") || CCtx().Visualize())
    AddStageWithPost<Visualizer>([](ASTPipeline& p) { p.SetAbend(); });

  // early loop vectorizer for the certain target
  if (!CCtx().NoVectorize() && CCtx().TargetSupportVectorize()) {
    AddStage<LoopVectorizer>();
    if (CCtx().TraceVectorize())
      AddAction([](ASTPipeline& p) { p.SetAbend(); });
    AddAction(
        [](ASTPipeline& p) { CCtx().SetGlobalSymbolTable(p.LastSymTab()); });
  }

  if (CCtx().TargetSupportMemAlloc() && CCtx().MemReuse())
    AddStage<MemoryReuse>();

  // compute active thread counts for inthreads scopes (before semacheck
  // so the checker can validate events with thread count info)
  AddStage<ActiveThreadsAnalysis>();

  // apply the semantic check
  AddStage<SemaChecker>();
  return *this;
}

ASTPipeline& ASTPipeline::PlanCodeGenRoutine() {
  // do not generate code
  if (CCtx().NoCodegen()) AddAction([](ASTPipeline& p) { p.SetAbend(); });

  AddStage<CodegenPrepare>();

  // delegate to target for codegen plan
  if (!CCtx().GetTarget().PlanCodeGenStages(*this)) {
    errs() << "Failed to initialize codegen for target '" << CCtx().TargetName()
           << "'\n";
    abend = true;
  }
  return *this;
}

ASTPipeline& ASTPipeline::Get() {
  std::call_once(init_flag,
                 []() { instance = std::make_unique<ASTPipeline>(); });
  return *instance;
}
