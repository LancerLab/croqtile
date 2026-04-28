#ifndef __CHOREO_DMA_PLAN_HPP__
#define __CHOREO_DMA_PLAN_HPP__

/// DMAPlan -- pre-compute all DMA lowering decisions before codegen.
///
/// This pass walks every DMA node and annotates it with a DMALoweringDecision
/// that captures all strategy choices that codegen would otherwise compute
/// inline.  Centralising this logic here makes CuteCodeGen a pure emitter
/// with no strategic reasoning.
///
/// Decisions captured per DMA node:
///   - Rank / element type of source and destination spans
///   - Direction (g2s, s2g, l2l, ...)
///   - DMA flavour: naive copy, threads-collective tiled copy, TMA bulk
///   - For tiled copy: predicated vs. unpredicated, thread/value layouts,
///     alignment, swizzle, cp.async (zero-fill) flag
///   - Box shape (static 2-D tile that the copy covers)
///   - Whether the operation is asynchronous and/or has future/trigger
///
/// The pass runs in PlanCodeGenStages() after GPUAdaptor (which fills in
/// target-specific type/shape info) and before CuteCodeGen.

#include "ast.hpp"
#include "codegen.hpp"
#include "dmaconf.hpp"
#include "types.hpp"

#include <cstddef>
#include <optional>
#include <unordered_map>

namespace Choreo {

// ---------------------------------------------------------------------------
// DMAStrategy -- which lowering path to use for a dma.copy node.
// [dma|tma].[copy|transp][.async][.zfill] A => B;
// ---------------------------------------------------------------------------

enum class DMAStrategy {
  UNKNOWN,
  NAIVE_COPY,        // scalar / single-thread fallback
  TILED_COPY_UNPRED, // choreo::tiled_copy without predicate
  TILED_COPY_PRED,   // choreo::tiled_copy with predicate (tail / zfill)
  TMA,               // TMA bulk-copy (async, sm_90+)
};

inline const std::string STR(DMAStrategy s) {
  switch (s) {
  case DMAStrategy::UNKNOWN: return "unknown";
  case DMAStrategy::NAIVE_COPY: return "naive_copy";
  case DMAStrategy::TILED_COPY_UNPRED: return "tiled_copy_unpred";
  case DMAStrategy::TILED_COPY_PRED: return "tiled_copy_pred";
  case DMAStrategy::TMA: return "tma";
  }
  return "?";
}

// ---------------------------------------------------------------------------
// DMADirection -- source and destination memory levels.
// ---------------------------------------------------------------------------

enum class DMADirection {
  UNKNOWN,
  G2S, // global  -> shared
  S2G, // shared  -> global
  G2L, // global  -> local/register
  L2G, // local   -> global
  S2S, // shared  -> shared
  OTHER,
};

enum class CUDA_COPY_ATOM {
  TMA_ATOM,
  CP_ASYNC_128B,
  CP_ASYNC_64B,
  CP_ASYNC_32B,
  VEC_128B,
  VEC_64B,
  VEC_32B,
  VEC_16B,
  ASYNC_COPY,
  UNIVERSAL_COPY,
  UNKNOWN,
};

inline const std::string STR(DMADirection d) {
  switch (d) {
  case DMADirection::UNKNOWN: return "unknown";
  case DMADirection::G2S: return "g2s";
  case DMADirection::S2G: return "s2g";
  case DMADirection::G2L: return "g2l";
  case DMADirection::L2G: return "l2g";
  case DMADirection::S2S: return "s2s";
  case DMADirection::OTHER: return "other";
  }
  return "?";
}

inline const std::string STR(CUDA_COPY_ATOM a) {
  switch (a) {
  case CUDA_COPY_ATOM::TMA_ATOM: return "tma_atom";
  case CUDA_COPY_ATOM::CP_ASYNC_128B: return "cp.async.128b";
  case CUDA_COPY_ATOM::CP_ASYNC_64B: return "cp.async.64b";
  case CUDA_COPY_ATOM::CP_ASYNC_32B: return "cp.async.32b";
  case CUDA_COPY_ATOM::VEC_128B: return "vec.128b";
  case CUDA_COPY_ATOM::VEC_64B: return "vec.64b";
  case CUDA_COPY_ATOM::VEC_32B: return "vec.32b";
  case CUDA_COPY_ATOM::VEC_16B: return "vec.16b";
  case CUDA_COPY_ATOM::ASYNC_COPY: return "async_copy";
  case CUDA_COPY_ATOM::UNIVERSAL_COPY: return "universal_copy";
  case CUDA_COPY_ATOM::UNKNOWN: return "unknown";
  }
  return "?";
}

// ---------------------------------------------------------------------------
// TiledCopyParams -- thread/value tile layout for one tiled-copy entry.
// ---------------------------------------------------------------------------

struct TiledCopyParams {
  // CUDA COPY ATOM
  ValueList box_shape;  // (rows, cols) of the tile box (static or dynamic)
  ValueList thr_layout; // (rows, cols) of threads in the collective tile
  ValueList val_layout; // (rows, cols) of elements per thread
  bool need_pred = false;
  bool need_iteration = false;
  ValueList prediction;
  ValueList iterations;
  size_t align_bits = 1;
};

// ---------------------------------------------------------------------------
// BatchDimInfo -- one outer dimension peeled into a codegen loop.
// For rank-3 there is one entry (dim-0); for rank-4/5 there may be
// two or three entries, outermost first.  Codegen emits nested loops.
// ---------------------------------------------------------------------------

struct BatchDimInfo {
  ValueItem size;        // extent of this dimension
  ValueItem from_stride; // source stride along this dimension
  ValueItem to_stride;   // dest stride along this dimension
};

// ---------------------------------------------------------------------------
// DynBoxTileInfo -- fixed-tile loop parameters for dynamic box shapes.
// ---------------------------------------------------------------------------
// When the DMA subspan has runtime dimensions, the box shape cannot be
// determined at compile time.  Instead of falling back to naive_copy, we
// choose a fixed compile-time tile (TILE_M=1, TILE_N=threads*vecfactor)
// and emit an explicit loop in codegen that iterates over the runtime
// extents with predication on the last tile of each row.
//
// Codegen emits:
//   for (rm = 0; rm < dyn_extent_m; ++rm)
//     for (rn = 0; rn < dyn_extent_n; rn += TILE_N) {
//       pred_n = min(TILE_N, dyn_extent_n - rn);
//       // full tile -> use fast atom, no pred
//       // last tile -> use slow atom + pred
//     }

struct DynBoxTileInfo {
  ValueItem dyn_extent_m; // runtime M extent (outer loop bound)
  ValueItem dyn_extent_n; // runtime N extent (inner loop bound)
  size_t tile_n = 0;      // compile-time TILE_N
};

// ---------------------------------------------------------------------------
// DMALoweringDecision -- all lowering choices for one DMA node.
// ---------------------------------------------------------------------------

struct DMALoweringDecision {
  // -- identity --------------------------------------------------------------
  DMAStrategy strategy = DMAStrategy::UNKNOWN;
  DMADirection direction = DMADirection::UNKNOWN;
  CUDA_COPY_ATOM atom = CUDA_COPY_ATOM::UNKNOWN;

  // -- element / rank --------------------------------------------------------
  BaseType elem_type = BaseType::UNKNOWN;
  int rank = -1; // -1 = not yet determined

  // -- flags -----------------------------------------------------------------
  bool is_async = false; // dma.copy.async
  bool is_zfill = false; // dma.copy.*.zfill  -> cp.async zero-fill semantic
  bool has_pred = false; // predicated tiled copy needed (tail handling)
  bool use_tma = false;  // TMA bulk-copy path

  // -- swizzle ---------------------------------------------------------------
  SwizMode swizzle_mode = SwizMode::NONE;
  Shape from_ca_shape;
  Shape to_ca_shape;

  Shape from_parent_shape; // after span_as
  Shape to_parent_shape;   // after span_as

  ValueList from_strides;
  ValueList to_strides;

  size_t threads_count = 0;
  // -- tiled copy parameters -------------------------------------------------
  std::optional<TiledCopyParams> tiled_params;

  // -- runtime-dispatched fast path (unknown-stride optimization) -----------
  // When a stride is unknown at compile time, tiled_params uses conservative
  // alignment (element-width vectorization).  tiled_params_fast holds an
  // optimistic 128-bit-aligned variant.  Codegen emits a runtime branch:
  //   if (stride is 128b-aligned) -> use tiled_params_fast / atom_fast
  //   else                        -> use tiled_params / atom
  std::optional<TiledCopyParams> tiled_params_fast;
  CUDA_COPY_ATOM atom_fast = CUDA_COPY_ATOM::UNKNOWN;

  // -- batch loops (rank >= 3 with non-contiguous outer dims) --------------
  // When a high-rank DMA cannot be fully flattened to 2D, the outer
  // dimensions are peeled into nested codegen loops.  Each entry
  // represents one loop level, outermost first.  The tiled copy operates
  // on the innermost 2D slice.
  std::vector<BatchDimInfo> batch_dims;

  // -- dynamic box tiling loop ------------------------------------------
  // When the 2D box shape has dynamic dimensions, the tiled copy uses a
  // fixed compile-time tile [1, TILE_N] with an explicit loop.  Codegen
  // emits the loop and switches between vectorized (no-pred) for full
  // tiles and predicated for the last partial tile.
  std::optional<DynBoxTileInfo> dyn_box;

  bool IsResolved() const { return strategy != DMAStrategy::UNKNOWN; }
  bool IsTiledDMA() const {
    return strategy == DMAStrategy::TILED_COPY_UNPRED ||
           strategy == DMAStrategy::TILED_COPY_PRED;
  }
  bool IsPredTiledDMA() const {
    return strategy == DMAStrategy::TILED_COPY_PRED;
  }
  bool IsUnpredTiledDMA() const {
    return strategy == DMAStrategy::TILED_COPY_UNPRED;
  }
  bool IsNaiveDMA() const { return strategy == DMAStrategy::NAIVE_COPY; }
  bool IsTMA() const { return strategy == DMAStrategy::TMA; }

  void Print(std::ostream& os) const {
    os << "DMALoweringDecision{"
       << " strategy=" << STR(strategy) << " dir=" << STR(direction)
       << " rank=" << rank << " async=" << is_async << " zfill=" << is_zfill
       << " pred=" << has_pred << " tma=" << use_tma
       << " swiz=" << STR(swizzle_mode) << " atom=" << STR(atom) << " }";
  }
  Shape GetBoxOfFrom() const {
    if (IsTiledDMA()) { return Shape(tiled_params->box_shape); }
    // For naive copy, use destination shape as the effective copy extent so
    // source and destination tensors have matching dimensions.  The dest
    // view determines how many elements to copy; the source just needs to
    // be at least that large.
    return to_ca_shape;
  }
  Shape GetBoxOfTo() const {
    if (IsTiledDMA()) { return Shape(tiled_params->box_shape); }
    return to_ca_shape;
  }
  bool IsValid() const {
    if (strategy == DMAStrategy::UNKNOWN ||
        direction == DMADirection::UNKNOWN || rank == -1) {
      return false;
    }
    if (IsTiledDMA() && !tiled_params.has_value()) { return false; }
    if (atom == CUDA_COPY_ATOM::UNKNOWN) { return false; }
    return true;
  }
};

// ---------------------------------------------------------------------------
// DMAPlan -- pass that populates a DMALoweringDecision per DMA node.
// ---------------------------------------------------------------------------

struct DMAPlan : public CodeGenerator {
  DMAPlan() : CodeGenerator("dma-plan") {}
  ~DMAPlan() override = default;

  // -- Visitor interface -----------------------------------------------------
  bool Visit(AST::DMA& n) override;

  // -- lookup (called by CuteCodeGen after this pass has run) ----------------
  //
  // Usage:
  //   DMAPlan& plan = DMAPlan::Instance();
  //   const DMALoweringDecision* dec = plan.GetDecision(&dma_node);
  //
  const DMALoweringDecision* GetDecision(const AST::DMA* n) const {
    auto it = decisions().find(n);
    return it != decisions().end() ? &it->second : nullptr;
  }

  // -- global lookup (callable from CuteCodeGen after this pass has run)
  // -------
  //
  // The decision table is stored as a static so it survives the pass
  // instance being destroyed and can be queried by the subsequent codegen pass.
  //
  static const DMALoweringDecision* Lookup(const AST::DMA* n) {
    auto it = decisions().find(n);
    return it != decisions().end() ? &it->second : nullptr;
  }

  // Reset between compilations (call before each new source file).
  static void Reset() { decisions().clear(); }

private:
  // -- required by VisitorWithScope ------------------------------------------
  bool BeforeVisitImpl(AST::Node& n) override {
    if (isa<AST::Program>(&n)) {
      Reset();
      pv_levels_.clear();
    }
    if (auto pb = dyn_cast<AST::ParallelBy>(&n)) {
      pv_levels_[InScopeName(pb->BPV()->name)] = pb->GetLevel();
      for (auto id : pb->AllSubPVs())
        pv_levels_[InScopeName(cast<AST::Identifier>(id)->name)] =
            pb->GetLevel();
    }
    return true;
  }
  bool AfterVisitImpl(AST::Node&) override { return true; }

  // -- analysis helpers ------------------------------------------------------
  DMADirection ResolveDirection(const AST::DMA& n) const;
  void ResolveDMADecision(const AST::DMA& n, DMALoweringDecision& dec) const;
  bool HasInnerBlockPV(const AST::ChunkAt& ca) const;
  size_t ResolveParticipatingThreadCount() const;
  std::vector<std::vector<size_t>>
  ResolveCandidateThreadLayouts(const AST::DMA& n,
                                const DMALoweringDecision& dec) const;
  void ResolveTiledParams(const AST::DMA& n, DMALoweringDecision& dec,
                          const Shape& unceiled_box) const;
  void ResolvePrediction(const AST::DMA& n, DMALoweringDecision& dec,
                         TiledCopyParams& param) const;
  ValueList ResolveStrides(const AST::ChunkAt& ca,
                           const std::vector<size_t>& tc = {}) const;

  // -- shared decision store (static so it outlives the pass instance) --------
  static std::unordered_map<const AST::DMA*, DMALoweringDecision>& decisions() {
    static std::unordered_map<const AST::DMA*, DMALoweringDecision> store;
    return store;
  }

  std::unordered_map<std::string, ParallelLevel> pv_levels_;
};

} // namespace Choreo

#endif // __CHOREO_DMA_PLAN_HPP__
