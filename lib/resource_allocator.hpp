#ifndef __CHOREO_RESOURCE_ALLOCATOR_HPP__
#define __CHOREO_RESOURCE_ALLOCATOR_HPP__

/// DmaResourceAllocator -- interference coloring of DMA completion resources.
///
/// Opted-in private contexts use completion occupancy, independently of buffer
/// liveness. Other resource protocols retain their existing liveness analysis.
/// The shared HeapSimulator colors the interference relation into a static
/// plan, keyed by scoped handle identity (or anonymous synchronous DMA site).
/// Greedy coloring is deterministic but does not guarantee the minimum count.
///
/// Resource classes:
///   FUTURE -- device DMA completion slot (DTE), GPU TMA mbarrier
///   EVENT  -- GPU named barrier / scalar mbarrier, device sync-point
/// BUFFER is handled by MemReuse and is out of scope here.

#include "codegen.hpp"
#include "dma_completion_analysis.hpp"
#include "heap_simulator.hpp"
#include "liveness_analysis.hpp"
#include "options.hpp"

#include <map>
#include <string>

namespace Choreo {

// How EVENT handles are allocated.  `simple` (default) colors scalar events
// into a minimal barrier set; `full` additionally scalar-replaces event arrays
// before coloring (planned; see the dma-resource-allocation design doc); `off`
// disables liveness coloring and falls back to monotonic slot assignment.
enum class EventAllocMode {
  Off,
  Simple,
  Full,
};

// Enable/disable liveness-driven coloring of FUTURE handles (DMA completion
// slots).  Futures are only pooled where the target provides a pooled
// completion path (e.g. a DTE pool); otherwise the flag has no observable
// effect.  Replaces the old greedy -fdte-merge slot assignment.
extern Option<bool> dma_alloc_mode;
extern Option<bool> dma_completion_report;

// Select how EVENT handles (named barrier / mbarrier) are colored.  This is a
// parameterized option rather than a bare bool so the upcoming scalar-
// replacement pass can be selected with -fevent-alloc=full.
extern Option<std::string> event_alloc_mode;

// Parse -fevent-alloc into a strongly-typed mode (unknown -> Simple + warning).
EventAllocMode ParseEventAllocMode();

// True unless -fevent-alloc=off.
inline bool EventAllocEnabled() {
  return ParseEventAllocMode() != EventAllocMode::Off;
}

// ---------------------------------------------------------------------------
// DmaResourcePlan -- static store of per-device-function allocation results.
// ---------------------------------------------------------------------------

struct DmaResourcePlan {
  // Allocation result for one device function.
  struct DevicePlan {
    // Scoped handle name -> slot index.
    std::map<std::string, size_t> future_slots;
    std::map<const AST::DMA*, size_t> synchronous_slots;
    std::map<std::string, size_t> event_slots;
    // Total number of slots (colors) needed per class.
    size_t future_slot_count = 0;
    size_t event_slot_count = 0;
  };

  // Static store keyed by device-function scope name (e.g. "::co::paraby_0::").
  // Static so it survives the pass instance and can be queried by codegen.
  static std::map<std::string, DevicePlan>& Plans() {
    static std::map<std::string, DevicePlan> store;
    return store;
  }

  static void Reset() { Plans().clear(); }

  static const DevicePlan* Lookup(const std::string& dev_fname) {
    auto it = Plans().find(dev_fname);
    return it != Plans().end() ? &it->second : nullptr;
  }
};

// ---------------------------------------------------------------------------
// DmaResourceAllocator -- the pass.
// ---------------------------------------------------------------------------

struct DmaResourceAllocator : public VisitorWithSymTab {
  explicit DmaResourceAllocator(bool use_completion_occupancy = false)
      : VisitorWithSymTab("dma-alloc"),
        use_completion_occupancy_(use_completion_occupancy) {}
  ~DmaResourceAllocator() override = default;

private:
  bool use_completion_occupancy_;
  LivenessAnalyzer la;
  DMACompletionAnalysis completion;
  std::string cur_dev_fname;

  bool RunOnProgramImpl(AST::Node& root) override;
  bool BeforeVisitImpl(AST::Node& n) override;
  bool AfterVisitImpl(AST::Node& n) override;

  // Find the most specific HB graph whose scope covers df_name (exact match
  // first, then longest-prefix fallback).  Mirrors MemReuse::ProtoType.
  const LivenessAnalyzer::HBGraph*
  FindHBGraph(const std::string& df_name) const;

  // Color one resource class (FUTURE or EVENT) for one device function and
  // store the result in DmaResourcePlan.
  void AllocateClass(LivenessAnalyzer::ResourceClass rc,
                     const std::string& df_name);
  bool AllocateCompletions(const std::string& df_name);
};

// VisitorGroup wrapper so the pass can be added as a pipeline stage.
class DmaResourceAllocation : public VisitorGroup {
private:
  DmaResourceAllocator dra;

public:
  explicit DmaResourceAllocation(bool use_completion_occupancy = false)
      : VisitorGroup("DmaResourceAllocation", dra),
        dra(use_completion_occupancy) {}
};

} // namespace Choreo

#endif // __CHOREO_RESOURCE_ALLOCATOR_HPP__
