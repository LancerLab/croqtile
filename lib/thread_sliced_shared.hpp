#ifndef __CHOREO_THREAD_SLICED_SHARED_HPP__
#define __CHOREO_THREAD_SLICED_SHARED_HPP__

#include "ast.hpp"
#include "context.hpp"
#include "types.hpp"
#include "visitor.hpp"

namespace Choreo {

// Thread-sliced shared buffer analysis (GPU/CuTe target).
//
// A `shared` buffer whose content is thread-varying — i.e. any DMA moving data
// to or from the buffer indexes either side with a thread-level parallel
// variable (e.g. `dma.copy rhs.chunkat(k_tile, q#n_tile) => shared`) — cannot
// be a single block-common allocation: every thread stages its own slice.
// Without this analysis the CuTe codegen emitted one block-common buffer and
// loaded it under `__CHOREO_BLOCK_SINGLE__` (thread 0 only), so all threads
// computed from thread 0's slice — a silent miscompile.
//
// This pass records, for every such buffer, the number of per-thread slices
// (the bound of the enclosing thread-level parallel-by).  Consumers:
//   - MemAnalyzer (mem_reuse.cpp) scales the buffer footprint by the slice
//     count so the shared spm holds the union of all slices;
//   - MemReuse::ApplyMemOffset annotates the decl with the per-thread slice
//     size ("tsliced" note) so codegen can add the per-thread addend
//     `__choreo_vtid_x * <slice bytes>` to the buffer base pointer;
//   - DMAPlan forces naive (per-thread) copies for DMAs touching the buffer
//     and CuteCodeGen drops the `__CHOREO_BLOCK_SINGLE__` guard for them.
struct ThreadSlicedShared : public VisitorWithSymTab {
public:
  ThreadSlicedShared() : VisitorWithSymTab("tsliced") {}

  // scoped buffer name -> number of per-thread slices (thread count).
  static std::unordered_map<std::string, ValueItem>& Store() {
    static std::unordered_map<std::string, ValueItem> store;
    return store;
  }

  static const ValueItem* Lookup(const std::string& scoped_name) {
    auto& s = Store();
    auto it = s.find(scoped_name);
    return it == s.end() ? nullptr : &it->second;
  }

  static bool Enabled() {
    return CCtx().MemReuse() && CCtx().TargetName() == "cute";
  }

private:
  std::unordered_map<std::string, ParallelLevel> pv_levels_;
  // bounds of enclosing thread-level parallel-bys (one 1-D level supported).
  std::vector<ValueList> thread_bounds_;

private:
  bool BeforeVisitImpl(AST::Node& n) override {
    if (isa<AST::Program>(&n)) {
      Store().clear();
      pv_levels_.clear();
      thread_bounds_.clear();
    }
    if (auto pb = dyn_cast<AST::ParallelBy>(&n)) {
      pv_levels_[InScopeName(pb->BPV()->name)] = pb->GetLevel();
      for (auto id : pb->AllSubPVs())
        pv_levels_[InScopeName(cast<AST::Identifier>(id)->name)] =
            pb->GetLevel();
      if (pb->GetLevel() == ParallelLevel::THREAD)
        thread_bounds_.push_back(pb->BoundValues());
    }
    return true;
  }

  bool AfterVisitImpl(AST::Node& n) override {
    if (auto pb = dyn_cast<AST::ParallelBy>(&n)) {
      if (pb->GetLevel() == ParallelLevel::THREAD && !thread_bounds_.empty())
        thread_bounds_.pop_back();
    }
    return true;
  }

  // Does the chunkat reference any symbol deeper than block level?
  // Sets `group_varying` when the reference is at (warp-)group level, which
  // has its own (MMA) machinery and is out of scope for thread slicing.
  bool IsThreadVarying(AST::ChunkAt* ca, bool& group_varying) const {
    std::set<std::string> syms;
    if (ca->indices) {
      auto s = ReferredSymbols(ca->indices.get(), this);
      syms.insert(s.begin(), s.end());
    }
    auto s = ReferredSymbols(ca, this);
    syms.insert(s.begin(), s.end());
    syms.erase(InScopeName(ca->RefSymbol()));
    bool varying = false;
    for (auto& sym : syms) {
      auto it = pv_levels_.find(sym);
      if (it == pv_levels_.end()) continue;
      if (it->second == ParallelLevel::THREAD)
        varying = true;
      else if (it->second == ParallelLevel::GROUP ||
               it->second == ParallelLevel::GROUPx4)
        group_varying = true;
    }
    return varying;
  }

  bool Visit(AST::DMA& n) override {
    if (!Enabled()) return true;
    if (n.operation == ".any") return true;

    auto fty = GetSpannedType(n.from->GetType());
    auto tty = GetSpannedType(n.to->GetType());
    if (!fty || !tty) return true;
    bool from_shared = fty->GetStorage() == Storage::SHARED;
    bool to_shared = tty->GetStorage() == Storage::SHARED;
    if (!from_shared && !to_shared) return true;

    bool group_varying = false;
    bool from_varying = false, to_varying = false;
    if (auto ca = dyn_cast<AST::ChunkAt>(n.from.get()))
      from_varying = IsThreadVarying(ca, group_varying);
    if (auto ca = dyn_cast<AST::ChunkAt>(n.to.get()))
      to_varying = IsThreadVarying(ca, group_varying);
    bool varying = from_varying || to_varying;
    if (varying && group_varying)
      Error1(n.LOC(),
             "thread-sliced shared buffers do not yet support mixing "
             "thread-level and group-level parallel variables in DMA chunk "
             "indexing.");
    // Group-level-only variance uses the warp-level (MMA) machinery; leave
    // the existing lowering untouched.
    if (!varying || group_varying) return true;

    // Slicing is needed only when the DMA stages per-thread data INTO the
    // shared buffer (thread-varying source: each thread's chunk is distinct
    // data that must coexist), or writes per-thread results OUT to distinct
    // global-memory regions (S2G with a thread-varying destination, which
    // implies the buffer held per-thread data).  A shared -> local copy with
    // a thread-varying address is a distributed read of block-common data
    // and needs no slicing.
    bool slice_to = to_shared && from_varying;
    bool slice_from = from_shared && to_varying &&
                      (tty->GetStorage() == Storage::GLOBAL ||
                       tty->GetStorage() == Storage::DEFAULT || to_shared);
    if (!slice_to && !slice_from) return true;
    if (thread_bounds_.size() != 1 || thread_bounds_.front().size() != 1)
      Error1(n.LOC(),
             "thread-sliced shared buffers do not yet support nested or "
             "multi-dimensional thread-level parallel-bys.");
    ValueItem count = thread_bounds_.front().front();
    if (!VIIsInt(count))
      Error1(n.LOC(),
             "thread-varying DMA into shared storage requires an enclosing "
             "thread-level parallel-by with a static bound.");
    auto mark = [this, &count](AST::Node* side, bool shared_side) {
      if (!shared_side) return;
      auto ca = dyn_cast<AST::ChunkAt>(side);
      if (!ca) return;
      auto sname = InScopeName(ca->RefSymbol());
      auto& s = Store();
      if (!s.count(sname)) s.emplace(sname, count);
    };
    mark(n.from.get(), slice_from);
    mark(n.to.get(), slice_to);
    return true;
  }
};

} // namespace Choreo

#endif // __CHOREO_THREAD_SLICED_SHARED_HPP__
