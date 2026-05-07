#include "dma_plan.hpp"
#include "ast.hpp"
#include "types.hpp"

using namespace Choreo;

namespace Choreo {
extern Option<bool> assume_aligned_global;
}

// ---------------------------------------------------------------------------
// DMAPlan::Visit -- entry point for each DMA node in the AST.
// ---------------------------------------------------------------------------

bool DMAPlan::Visit(AST::DMA& n) {
  if (isa<PlaceHolderType>(NodeType(n))) return true;
  DMALoweringDecision dec;
  ResolveDMADecision(n, dec);
  assert(dec.IsValid() && "DMA decision should be valid after resolution");
  decisions()[&n] = std::move(dec);
  return true;
}

// ---------------------------------------------------------------------------
// Direction resolution -- inspect from/to span storage qualifiers.
// ---------------------------------------------------------------------------
DMADirection DMAPlan::ResolveDirection(const AST::DMA& n) const {
  auto from_sty = GetSpannedType(NodeType(*n.GetFrom()));
  auto to_sty = GetSpannedType(NodeType(*n.GetTo()));
  if (!from_sty || !to_sty) return DMADirection::UNKNOWN;

  const Storage from_sto = ProjectStorage(from_sty->GetStorage());
  const Storage to_sto = ProjectStorage(to_sty->GetStorage());

  if (from_sto == Storage::GLOBAL && to_sto == Storage::SHARED)
    return DMADirection::G2S;
  if (from_sto == Storage::SHARED && to_sto == Storage::GLOBAL)
    return DMADirection::S2G;
  if (from_sto == Storage::GLOBAL && to_sto == Storage::LOCAL)
    return DMADirection::G2L;
  if (from_sto == Storage::LOCAL && to_sto == Storage::GLOBAL)
    return DMADirection::L2G;
  if (from_sto == Storage::SHARED && to_sto == Storage::SHARED)
    return DMADirection::S2S;
  return DMADirection::OTHER;
}

// ---------------------------------------------------------------------------
// Strategy resolution -- choose the lowering flavour for this DMA op.
//
// Policy (to be filled in during the codegen refactor):
//
//   TMA
//     - sm_90+ only
//
//   TILED_COPY_UNPRED
//    - Eligible if the copy is between global and shared memory, 2D, and the
//      tile evenly covers the runtime extent
//
//   TILED_COPY_PRED
//     - Same eligibility as TILED_COPY_UNPRED but requires a predicated tail
//       pass because the tile doesn't evenly cover the runtime extent
//
//   NAIVE_COPY
//     - Fallback for everything else
// Forward declarations of static helpers (defined below ResolveDMADecision).
static size_t DetectAlignBits(const ValueItem& stride, size_t elem_bits);
static CUDA_COPY_ATOM SelectCopyAtom(size_t align_bits, bool cp_async);

struct TiledCopySearchResult {
  bool found = false;
  size_t vf = 0;
  ValueList box_shape;
  ValueList thr_layout;
  ValueList val_layout;
  bool pred_needed = false;
};

static TiledCopySearchResult SearchTiledConfig(size_t box_m, size_t box_n,
                                               size_t threads_count,
                                               size_t max_vf, size_t elem_bits,
                                               bool cp_async, bool allow_pred);

// Compare two shapes, returning true if they might differ at runtime.
// Conservative: returns true (potential mismatch) if any dimension is dynamic.
static bool StaticShapeMismatch(const Shape& a, const Shape& b) {
  if (a.Rank() != b.Rank()) return true;
  for (size_t i = 0; i < a.Rank(); ++i) {
    auto av = VIInt(a.ValueAt(i));
    auto bv = VIInt(b.ValueAt(i));
    if (!av || !bv) return true;
    if (*av != *bv) return true;
  }
  return false;
}

static ValueList ShapeToValueList(const Shape& shape) {
  ValueList values;
  for (size_t i = 0; i < shape.Rank(); ++i) values.push_back(shape.ValueAt(i));
  return values;
}

static ValueList ProjectPredDims(const ValueList& values) {
  if (values.empty()) return {sbe::nu(0), sbe::nu(0)};
  if (values.size() == 1) return {sbe::nu(1), values[0]};
  return {values[values.size() - 2], values[values.size() - 1]};
}

static ValueList ResolveNodeValues(const ptr<AST::MultiValues>& values) {
  if (!values) return {};
  if (values->Opts().HasVals()) return values->Opts().GetVals();

  ValueList resolved;
  for (auto node : values->AllValues()) {
    auto expr = dyn_cast<AST::Expr>(node);
    if (!expr || !expr->Opts().HasVal()) return {};
    resolved.push_back(expr->Opts().GetVal());
  }
  return resolved;
}

static ValueList PermuteValueList(const ValueList& values,
                                  const std::vector<size_t>& order) {
  if (values.size() != order.size()) return values;

  ValueList permuted;
  permuted.reserve(order.size());
  for (auto dim : order) {
    if (dim >= values.size()) return values;
    permuted.push_back(values[dim]);
  }
  return permuted;
}

static ValueList ResolveTileCoordOffsets(const AST::ChunkAt& ca) {
  size_t rank = ca.GetBlockShape().Rank();
  for (auto& op : ca.AllOperations()) {
    if (!isa<AST::SOP::Reshape>(op)) {
      rank = op->GetBlockShape().Rank();
      break;
    }
  }

  ValueList offsets(rank, sbe::nu(0));
  for (auto& op : ca.AllOperations()) {
    if (isa<AST::SOP::Reshape>(op)) continue;

    if (isa<AST::SOP::View>(op)) {
      auto off = ResolveNodeValues(op->GetOffsets());
      for (size_t i = 0; i < off.size() && i < offsets.size(); ++i)
        offsets[i] = (offsets[i] + off[i])->Normalize();
      continue;
    }

    auto idx = ResolveNodeValues(op->GetIndices());
    if (idx.empty()) continue;

    auto blk = op->GetBlockShape();
    ValueList steps;
    if (auto step_mv = op->GetSteps()) steps = ResolveNodeValues(step_mv);

    for (size_t i = 0; i < idx.size() && i < offsets.size() && i < blk.Rank();
         ++i) {
      auto stride = (i < steps.size()) ? steps[i] : blk.ValueAt(i);
      offsets[i] = (offsets[i] + idx[i] * stride)->Normalize();
    }
  }
  return offsets;
}

static ValueItem BuildPrediction(const ValueItem& outer_bound,
                                 const ValueItem& offset,
                                 const ValueItem& box_dim) {
  auto remaining = (outer_bound - offset)->Normalize();
  return sbe::sel(sbe::oc_lt(remaining, box_dim)->Normalize(), remaining,
                  box_dim)
      ->Normalize();
}

// ---------------------------------------------------------------------------
void DMAPlan::ResolveDMADecision(const AST::DMA& n,
                                 DMALoweringDecision& dec) const {
  auto debug_naive_fallback = [&](const char* reason) {
    VST_DEBUG(dbgs() << "DMAPlan fallback to naive: " << reason << "\n"
                     << "  op: " << n.operation << "\n"
                     << "  from: " << PSTR(n.GetFrom()) << "\n"
                     << "  to: " << PSTR(n.GetTo()) << "\n"
                     << "  dir: " << STR(dec.direction) << "\n";);
    Warning(n.LOC(), std::string("dma") + n.operation +
                         " falls back to naive copy (" + reason + ").");
  };

  // basic attributes
  dec.direction = ResolveDirection(n);
  dec.is_async = n.IsAsync();
  dec.is_zfill = n.IsOOBZeroFill();
  dec.swizzle_mode = n.GetSwizzleMode();
  dec.use_tma = n.IsTMA();

  auto from_ca = dyn_cast<AST::ChunkAt>(n.GetFrom().get());
  auto to_ca = dyn_cast<AST::ChunkAt>(n.GetTo().get());
  auto from_sty = GetSpannedType(NodeType(*n.GetFrom()));
  auto to_sty = GetSpannedType(NodeType(*n.GetTo()));
  assert(from_ca && to_ca && from_sty && to_sty);
  auto from_sym = from_ca->data->name;
  auto to_sym = to_ca->data->name;
  auto from_sym_sty = GetSpannedType(GetSymbolType(from_sym));
  auto to_sym_sty = GetSpannedType(GetSymbolType(to_sym));
  assert(from_sym_sty && to_sym_sty);
  auto from_parent_shape = from_sym_sty->GetShape();
  auto to_parent_shape = to_sym_sty->GetShape();
  // span_as
  if (auto idx = from_ca->IndexOfLastSpanAs())
    from_parent_shape = from_ca->OpAt(*idx)->GetBlockShape();
  if (auto idx = to_ca->IndexOfLastSpanAs())
    to_parent_shape = to_ca->OpAt(*idx)->GetBlockShape();

  dec.from_parent_shape = from_parent_shape;
  dec.to_parent_shape = to_parent_shape;

  const Shape& f_ca_shape = from_ca->GetBlockShape();
  const Shape& t_ca_shape = to_ca->GetBlockShape();

  dec.from_ca_shape = from_ca->GetBlockShape();
  dec.to_ca_shape = to_ca->GetBlockShape();

  std::vector<size_t> transp_config;
  if (n.operation == ".transp")
    transp_config = cast<TransposeConfig>(n.GetConfig())->dim_values;
  dec.from_strides = ResolveStrides(*from_ca, transp_config);
  dec.to_strides = ResolveStrides(*to_ca);
  dec.rank = static_cast<int>(from_ca->GetBlockShape().Rank());
  dec.elem_type = from_sty->ElementType();

  if (n.IsTMA()) {
    dec.strategy = DMAStrategy::TMA;
    dec.atom = CUDA_COPY_ATOM::TMA_ATOM;

    // TMA descriptors are configured on the host side. The box shape
    // must be computable from function parameters -- kernel-runtime
    // values (blockIdx, loop indices, computed variables) are not
    // available in the host wrapper.
    for (size_t i = 0; i < f_ca_shape.Rank(); ++i) {
      auto vi = f_ca_shape.ValueAt(i);
      if (VIIsInt(vi)) continue;
      for (const auto& sym : GetSymbols(vi)) {
        auto sym_name = VISym(sym);
        if (sym_name && sym_name->find("::paraby") != std::string::npos) {
          Error1(n.LOC(),
                 "TMA box shape must be host-computable; it depends on a "
                 "kernel-runtime value (blockIdx / loop index). "
                 "Use chunkat/subspan.at or switch to dma.copy.");
          return;
        }
      }
    }

    // TMA shape-mismatch / zfill diagnostics.
    // Account for .transp permutation when comparing shapes.
    Shape tma_effective_from = f_ca_shape;
    if (n.operation == ".transp") {
      auto tc = cast<TransposeConfig>(n.GetConfig())->dim_values;
      tma_effective_from =
          Shape(PermuteValueList(ShapeToValueList(f_ca_shape), tc));
    }
    const bool tma_is_pad = (n.operation == ".pad");
    const bool tma_mismatch =
        tma_is_pad ? false
                   : StaticShapeMismatch(tma_effective_from, t_ca_shape);
    const bool tma_explicit_zfill = n.IsOOBZeroFill();

    if (tma_mismatch && !tma_explicit_zfill) {
      Warning(n.LOC(),
              "TMA source and destination tiles have different shapes; "
              "consider adding .zfill to avoid out-of-bounds access.");
      dec.is_zfill = true;
    } else if (tma_explicit_zfill && !tma_mismatch) {
      Warning(n.LOC(),
              "TMA .zfill is redundant because source and destination tiles "
              "have the same shape.");
    }
    return;
  }

  // set fallback dma strategy
  dec.strategy = DMAStrategy::NAIVE_COPY;
  size_t thr_count = ResolveParticipatingThreadCount();
  // first and most important eligibility check: if we can't determine the
  // thread count, we can't make informed decisions about tiled copy parameters,
  // so we bail out to naive copy.
  if (thr_count == 0 || thr_count < 2) {
    debug_naive_fallback("participating thread count is unknown or invalid");
    if (dec.is_async)
      dec.atom = CUDA_COPY_ATOM::ASYNC_COPY;
    else
      dec.atom = CUDA_COPY_ATOM::UNIVERSAL_COPY;
    return;
  }
  dec.threads_count = thr_count;

  // set default copy atom
  if (dec.is_async)
    dec.atom = CUDA_COPY_ATOM::ASYNC_COPY;
  else
    dec.atom = CUDA_COPY_ATOM::UNIVERSAL_COPY;

  if (dec.direction == DMADirection::OTHER) {
    debug_naive_fallback("unsupported DMA direction");
    return;
  }
  if (HasInnerBlockPV(*from_ca) || HasInnerBlockPV(*to_ca)) {
    debug_naive_fallback("chunk indexing depends on inner-than-block PV");
    return;
  }

  if (n.operation != ".copy" && n.operation != ".transp" &&
      n.operation != ".pad") {
    debug_naive_fallback("operation is not .copy, .transp, or .pad");
    return;
  }

  if (dec.direction != DMADirection::G2S &&
      dec.direction != DMADirection::S2G) {
    debug_naive_fallback("only g2s and s2g are eligible for tiled lowering");
    return;
  }

  if (IsSubByteType(dec.elem_type)) {
    debug_naive_fallback("sub-byte element type not eligible for tiled copy");
    return;
  }

  if (f_ca_shape.Rank() < 1 || f_ca_shape.Rank() > 5 || t_ca_shape.Rank() < 1 ||
      t_ca_shape.Rank() > 5) {
    debug_naive_fallback("only rank 1-5 DMA is eligible for tiled lowering");
    return;
  }
  if (f_ca_shape.Rank() != t_ca_shape.Rank()) {
    debug_naive_fallback("from and to rank mismatch for tiled lowering");
    return;
  }

  // For rank-1 DMA, reshape to rank-2 [1, N] so all downstream 2D tiled-copy
  // logic applies unchanged.
  Shape eff_f_shape = f_ca_shape;
  Shape eff_t_shape = t_ca_shape;
  if (f_ca_shape.Rank() == 1) {
    eff_f_shape = Shape({sbe::nu(1), f_ca_shape.ValueAt(0)});
    eff_t_shape = Shape({sbe::nu(1), t_ca_shape.ValueAt(0)});
    dec.from_ca_shape = eff_f_shape;
    dec.to_ca_shape = eff_t_shape;
    dec.from_strides.insert(dec.from_strides.begin(), f_ca_shape.ValueAt(0));
    dec.to_strides.insert(dec.to_strides.begin(), t_ca_shape.ValueAt(0));
    dec.rank = 2;
  }

  // For rank >= 3 DMA, reduce to 2D by repeatedly trying two operations
  // (innermost-first):
  //   1. Flatten: merge the two innermost dims if they are contiguous
  //      (s[R-2] == D[R-1] * s[R-1] for both src and dst).
  //   2. Peel: strip the outermost dim into a batch loop.
  // Continue until only 2 dims remain.
  if (eff_f_shape.Rank() >= 3) {
    // Work on mutable copies of shapes/strides.
    ValueList f_dims, t_dims, f_str, t_str;
    for (size_t i = 0; i < eff_f_shape.Rank(); ++i)
      f_dims.push_back(eff_f_shape.ValueAt(i));
    for (size_t i = 0; i < eff_t_shape.Rank(); ++i)
      t_dims.push_back(eff_t_shape.ValueAt(i));
    f_str = dec.from_strides;
    t_str = dec.to_strides;

    // Track parent shapes through the same rank reduction so that
    // ResolvePrediction sees consistent 2D bounds for guard masking.
    ValueList fp_dims, tp_dims;
    bool track_parent = from_parent_shape.Rank() == eff_f_shape.Rank() &&
                        to_parent_shape.Rank() == eff_t_shape.Rank();
    if (track_parent) {
      for (size_t i = 0; i < from_parent_shape.Rank(); ++i)
        fp_dims.push_back(from_parent_shape.ValueAt(i));
      for (size_t i = 0; i < to_parent_shape.Rank(); ++i)
        tp_dims.push_back(to_parent_shape.ValueAt(i));
    }

    auto can_flatten_pair = [](const ValueList& strides, const ValueList& dims,
                               size_t pos) {
      // Check if dim[pos] and dim[pos+1] are contiguous:
      // stride[pos] == dim[pos+1] * stride[pos+1]
      if (pos + 1 >= dims.size() || pos + 1 >= strides.size()) return false;
      auto s_outer = strides[pos];
      auto s_inner = strides[pos + 1];
      auto d_inner = dims[pos + 1];
      if (!VIIsInt(s_outer) || !VIIsInt(s_inner) || !VIIsInt(d_inner))
        return false;
      return *VIInt(s_outer) == *VIInt(d_inner) * *VIInt(s_inner);
    };

    while (f_dims.size() > 2) {
      size_t R = f_dims.size();
      // Try to flatten the two innermost dims (pos = R-2).
      if (can_flatten_pair(f_str, f_dims, R - 2) &&
          can_flatten_pair(t_str, t_dims, R - 2)) {
        auto f_merged = (f_dims[R - 2] * f_dims[R - 1])->Normalize();
        auto t_merged = (t_dims[R - 2] * t_dims[R - 1])->Normalize();
        f_dims[R - 2] = f_merged;
        t_dims[R - 2] = t_merged;
        f_dims.erase(f_dims.begin() + R - 1);
        t_dims.erase(t_dims.begin() + R - 1);
        if (track_parent) {
          fp_dims[R - 2] = (fp_dims[R - 2] * fp_dims[R - 1])->Normalize();
          fp_dims.erase(fp_dims.begin() + R - 1);
          tp_dims[R - 2] = (tp_dims[R - 2] * tp_dims[R - 1])->Normalize();
          tp_dims.erase(tp_dims.begin() + R - 1);
        }
        // Keep the inner stride, drop the outer one of the pair.
        f_str.erase(f_str.begin() + R - 2);
        t_str.erase(t_str.begin() + R - 2);
      } else {
        // Can't flatten -> peel dim-0 as a batch loop.
        dec.batch_dims.push_back(BatchDimInfo{f_dims[0], f_str[0], t_str[0]});
        f_dims.erase(f_dims.begin());
        t_dims.erase(t_dims.begin());
        if (track_parent) {
          fp_dims.erase(fp_dims.begin());
          tp_dims.erase(tp_dims.begin());
        }
        f_str.erase(f_str.begin());
        t_str.erase(t_str.begin());
      }
    }

    // Commit the reduced 2D shape and strides.
    eff_f_shape = Shape(f_dims);
    eff_t_shape = Shape(t_dims);
    dec.from_ca_shape = eff_f_shape;
    dec.to_ca_shape = eff_t_shape;
    if (track_parent) {
      dec.from_parent_shape = Shape(fp_dims);
      dec.to_parent_shape = Shape(tp_dims);
    }
    dec.from_strides = f_str;
    dec.to_strides = t_str;
    dec.rank = 2;
  }

  // without ceiling to mulitple
  bool cannot_determine_box = false;
  bool n_dim_is_dynamic = false;
  auto unceiled_box = [&]() {
    // For .pad the tiled copy only moves the source (inner) data, so the box
    // must be the FROM shape.  For .copy/.transp we prefer the TO dimension
    // because it is the actual allocation extent.
    auto choose_base = [&](const ValueItem& f_dim, const ValueItem& t_dim,
                           int default_dim, bool is_n) -> int64_t {
      if (n.operation == ".pad") {
        if (VIIsInt(f_dim)) return static_cast<int64_t>(*VIInt(f_dim));
        if (VIIsInt(t_dim)) return static_cast<int64_t>(*VIInt(t_dim));
      } else {
        if (VIIsInt(t_dim)) return static_cast<int64_t>(*VIInt(t_dim));
        // TO is dynamic -- box is bounded by the dynamic dest extent even
        // if FROM has a static size.  Must flag as undetermined so the
        // dynamic-box or naive-copy path handles the real runtime bounds.
        cannot_determine_box = true;
        if (is_n) n_dim_is_dynamic = true;
        if (VIIsInt(f_dim)) return static_cast<int64_t>(*VIInt(f_dim));
      }
      cannot_determine_box = true;
      if (is_n) n_dim_is_dynamic = true;
      return static_cast<int64_t>(default_dim);
    };

    int64_t base_m =
        choose_base(eff_f_shape.ValueAt(0), eff_t_shape.ValueAt(0), 1, false);
    int64_t base_n = choose_base(eff_f_shape.ValueAt(1), eff_t_shape.ValueAt(1),
                                 dec.threads_count, true);

    ValueList box_shape;
    box_shape.push_back(sbe::nu(static_cast<int>(base_m)));
    box_shape.push_back(sbe::nu(static_cast<int>(base_n)));
    return Shape(box_shape);
  }();

  if (cannot_determine_box) {
    // Dynamic box: choose a fixed [1, TILE_N] tile and emit an explicit loop
    // over dynamic dimensions.  Predication handles partial tail tiles.
    size_t elem_bits = SizeOf(dec.elem_type) * 8;
    size_t threads_count = dec.threads_count;
    size_t box_n_val = static_cast<size_t>(*VIInt(unceiled_box.ValueAt(1)));
    size_t tile_n = threads_count * (128 / elem_bits);
    if (tile_n == 0) tile_n = threads_count;
    // Cap tile_n to the known box N bound (from the static side, if any)
    // to avoid oversized tiled-copy configurations.
    if (tile_n > box_n_val) tile_n = box_n_val;

    // Retrieve the runtime extents for the loop bounds.
    // For .copy/.transp the TO dimension is the actual destination bound;
    // always use it so the loop iterates over the real runtime extent.
    auto pick_extent = [&](int dim) -> ValueItem {
      auto f = eff_f_shape.ValueAt(dim);
      auto t = eff_t_shape.ValueAt(dim);
      if (n.operation == ".pad") return VIIsInt(f) ? f : t;
      return t;
    };
    auto dyn_m = pick_extent(0);
    auto dyn_n = pick_extent(1);

    // Build primary params with ELEMENT-LEVEL alignment.
    // AutoVectorizingCopy<128> + Pred=true crashes for non-CP_ASYNC atoms,
    // so the slow (predicated) path must use element-width alignment.
    // IMPORTANT: CP_ASYNC_ZFILL atoms write zeros to the destination for
    // predicated-out elements, which causes OOB shared writes when the
    // tile is larger than the actual data.  Disable cp.async for the
    // slow (predicated) path -- use synchronous AutoVectorizingCopy instead.
    size_t slow_align = elem_bits;
    bool enable_cp_async = false;

    size_t slow_vf = slow_align / elem_bits;
    auto slow_cfg = SearchTiledConfig(1, tile_n, threads_count, slow_vf,
                                      elem_bits, enable_cp_async, false);
    if (!slow_cfg.found) {
      dec.strategy = DMAStrategy::NAIVE_COPY;
      dec.atom = dec.is_async ? CUDA_COPY_ATOM::ASYNC_COPY
                              : CUDA_COPY_ATOM::UNIVERSAL_COPY;
      debug_naive_fallback("dynamic box: no tiled config found");
      return;
    }

    TiledCopyParams param;
    param.box_shape = slow_cfg.box_shape;
    param.thr_layout = slow_cfg.thr_layout;
    param.val_layout = slow_cfg.val_layout;
    param.align_bits = slow_cfg.vf * elem_bits;
    param.need_pred = true;
    ValueList tile_pred;
    tile_pred.push_back(sbe::nu(1));
    tile_pred.push_back(sbe::nu(static_cast<int>(tile_n)));
    param.prediction = tile_pred;

    dec.strategy = DMAStrategy::TILED_COPY_PRED;
    dec.has_pred = true;
    dec.atom = SelectCopyAtom(param.align_bits, enable_cp_async);
    if (dec.atom == CUDA_COPY_ATOM::UNKNOWN && enable_cp_async)
      dec.atom = SelectCopyAtom(param.align_bits, false);
    if (dec.atom == CUDA_COPY_ATOM::UNKNOWN) {
      dec.strategy = DMAStrategy::NAIVE_COPY;
      dec.atom = dec.is_async ? CUDA_COPY_ATOM::ASYNC_COPY
                              : CUDA_COPY_ATOM::UNIVERSAL_COPY;
      debug_naive_fallback("dynamic box: no suitable copy atom");
      return;
    }
    dec.tiled_params = param;

    // Build optimistic fast path (128-bit vectorized, Pred=false).
    // Full tiles don't need predication, so 128-bit AutoVectorizing is safe.
    bool fast_cp_async =
        dec.direction == DMADirection::G2S && threads_count % 32 == 0;
    size_t fast_max_vf = 128 / elem_bits;
    auto fast_cfg = SearchTiledConfig(1, tile_n, threads_count, fast_max_vf,
                                      elem_bits, fast_cp_async, false);
    if (fast_cfg.found && fast_cfg.vf * elem_bits > param.align_bits) {
      TiledCopyParams fast_param;
      fast_param.box_shape = param.box_shape;
      fast_param.thr_layout = fast_cfg.thr_layout;
      fast_param.val_layout = fast_cfg.val_layout;
      fast_param.align_bits = fast_cfg.vf * elem_bits;
      fast_param.need_pred = false;
      CUDA_COPY_ATOM fast_atom =
          SelectCopyAtom(fast_param.align_bits, fast_cp_async);
      if (fast_atom != CUDA_COPY_ATOM::UNKNOWN) {
        dec.tiled_params_fast = fast_param;
        dec.atom_fast = fast_atom;

        // Rewrite slow path's thread layout to match fast path (avoid bloat).
        param.thr_layout = fast_param.thr_layout;
        param.val_layout = fast_cfg.val_layout;
        dec.tiled_params = param;
      }
    }

    dec.dyn_box = DynBoxTileInfo{dyn_m, dyn_n, tile_n};
  } else {
    ResolveTiledParams(n, dec, unceiled_box);
  }
  if (dec.IsNaiveDMA()) return; // ResolveTiledParams fell back to NAIVE
  assert(dec.IsResolved() && "tiled copy should be resolved at this point");
  assert(dec.IsTiledDMA() && dec.tiled_params.has_value() &&
         "tiled copy should have tiled params");
}

bool DMAPlan::HasInnerBlockPV(const AST::ChunkAt& ca) const {
  auto collect = [this](AST::ChunkAt* c) {
    std::set<std::string> syms;
    if (c->indices) {
      auto s = ReferredSymbols(c->indices.get(), this);
      syms.insert(s.begin(), s.end());
    }
    auto s = ReferredSymbols(c, this);
    syms.insert(s.begin(), s.end());
    syms.erase(InScopeName(c->RefSymbol()));
    return syms;
  };
  for (auto& sym : collect(const_cast<AST::ChunkAt*>(&ca))) {
    auto it = pv_levels_.find(sym);
    if (it != pv_levels_.end() &&
        PlDepthMap::Get().ToDepth(it->second) >
            PlDepthMap::Get().ToDepth(ParallelLevel::BLOCK))
      return true;
  }
  return false;
}

size_t DMAPlan::ResolveParticipatingThreadCount() const {
  auto& lcs = cgi.GetFunctionLaunches(CurrentFunctionName());
  if (lcs.empty()) return 0;

  const auto& lconfig = lcs.front();
  auto inner_thr =
      lconfig.thread_count.x * lconfig.thread_count.y * lconfig.thread_count.z;
  auto group_cnt = lconfig.group_count.x * lconfig.group4_count.x *
                   lconfig.group_count.y * lconfig.group_count.z;
  bool has_warpspec =
      cgi.GetFunctionTrait(CurrentFunctionName()).has_warpspec_pattern;
  auto thr_count_vi = has_warpspec ? inner_thr : inner_thr * group_cnt;
  if (!VIIsInt(thr_count_vi)) return 0;
  if (*VIInt(thr_count_vi) == 0) return 0;
  return static_cast<size_t>(*VIInt(thr_count_vi));
}

// ---------------------------------------------------------------------------
// Shared helpers for ResolveTiledParams
// ---------------------------------------------------------------------------

/// Detect the maximum alignment (in bits) that a stride provides.
static size_t DetectAlignBits(const ValueItem& stride, size_t elem_bits) {
  if (!VIIsInt(stride)) return elem_bits;
  size_t stride_bits = static_cast<size_t>(*VIInt(stride)) * elem_bits;
  for (size_t a : {size_t{128}, size_t{64}, size_t{32}, size_t{16}})
    if (stride_bits % a == 0) return a;
  return 1;
}

/// Map alignment bits to the appropriate CuTe copy atom.
static CUDA_COPY_ATOM SelectCopyAtom(size_t align_bits, bool cp_async) {
  if (cp_async) {
    switch (align_bits) {
    case 128: return CUDA_COPY_ATOM::CP_ASYNC_128B;
    case 64: return CUDA_COPY_ATOM::CP_ASYNC_64B;
    case 32: return CUDA_COPY_ATOM::CP_ASYNC_32B;
    default: return CUDA_COPY_ATOM::UNKNOWN;
    }
  }
  switch (align_bits) {
  case 128: return CUDA_COPY_ATOM::VEC_128B;
  case 64: return CUDA_COPY_ATOM::VEC_64B;
  case 32: return CUDA_COPY_ATOM::VEC_32B;
  case 16: return CUDA_COPY_ATOM::VEC_16B;
  default: return CUDA_COPY_ATOM::UNIVERSAL_COPY;
  }
}

/// Search for the best 2D thread-layout configuration for a tiled copy.
///
/// Iterates vector factors from \p max_vf down to 1, trying thread layouts
/// with increasing thr_n (column threads) for better coalescing.
///
/// \param allow_pred  When true, box_m may be ceiled to satisfy thr_m
///                    divisibility (slow-path predication).  When false,
///                    configurations requiring such ceiling are skipped
///                    (fast-path -- must divide exactly).
static TiledCopySearchResult SearchTiledConfig(size_t box_m, size_t box_n,
                                               size_t threads_count,
                                               size_t max_vf, size_t elem_bits,
                                               bool cp_async, bool allow_pred) {
  TiledCopySearchResult result;

  for (size_t vf = max_vf; vf >= 1; vf /= 2) {
    if (box_n % vf != 0) continue;

    ValueList best_box = {sbe::nu(box_m), sbe::nu(box_n)};
    ValueList best_thr, best_val;
    bool cfg_found = false;
    bool cfg_pred = false;

    for (size_t thr_n = 1; thr_n <= threads_count && thr_n * vf <= box_n;
         thr_n *= 2) {
      if (threads_count % thr_n != 0 || box_n % thr_n != 0) continue;
      size_t thr_m = threads_count / thr_n;
      auto val_n_i = static_cast<int64_t>(box_n) / static_cast<int64_t>(thr_n);
      // val_n must be a multiple of vf so that each thread's column offset
      // (c * val_n) is vf-aligned, preserving the alignment assumption for
      // both cp.async and AutoVectorizingCopyWithAssumedAlignment.
      if (val_n_i == 0 || val_n_i % static_cast<int64_t>(vf) != 0) continue;

      size_t eff_box_m = box_m;
      bool this_pred = false;
      if (box_m % thr_m != 0) {
        if (!allow_pred) continue;
        this_pred = true;
        eff_box_m = ((box_m + thr_m - 1) / thr_m) * thr_m;
      }

      auto val_n = sbe::nu(val_n_i);
      ValueList val;
      if (cp_async) {
        size_t max_vm = std::max(
            size_t{1}, (128 / elem_bits) / static_cast<size_t>(*VIInt(val_n)));
        size_t vm = max_vm;
        while (vm > 1 && eff_box_m % (thr_m * vm) != 0) vm /= 2;
        if (eff_box_m % (thr_m * vm) != 0) continue;
        val = {sbe::nu(vm), val_n};
      } else {
        val = {sbe::nu(eff_box_m / thr_m), val_n};
      }

      best_box = {sbe::nu(eff_box_m), sbe::nu(box_n)};
      best_thr = {sbe::nu(thr_m), sbe::nu(thr_n)};
      best_val = val;
      cfg_found = true;
      cfg_pred = this_pred;
    }

    if (!cfg_found) continue;
    result.found = true;
    result.vf = vf;
    result.box_shape = best_box;
    result.thr_layout = best_thr;
    result.val_layout = best_val;
    result.pred_needed = cfg_pred;
    if (!cfg_pred) break; // optimal: no predication needed
  }
  return result;
}

// ---------------------------------------------------------------------------
// dma prediction rules to avoid oob:
// 1. user should set zfill for dma with
// different shape from and to, if user forget to set this zfill, gives warning
// and compiler auto-set implicit zfill active , if user set zfill but from and
// to are same, compiler should also gives waning to let user know this zfill is
// useless.
// 2. When no zfill are set(include explicit zfill and implicit zfill,
// equal to "from == to"). compiler should add auto-mask to avoid oob, like get
// a sub box [TM, TK] from [M, K] the guard masking should be [min(TM,
// M-OFFSET_M, K-OFFSET_K)]. These two rules are exclusive. This means
// choreo(our compiler) auto-avoid oob if no explicit/implicit zfill are used.
// This rules let user easily write safe and flexible dma.
void DMAPlan::ResolvePrediction(const AST::DMA& n, DMALoweringDecision& dec,
                                TiledCopyParams& param) const {
  auto from_ca = dyn_cast<AST::ChunkAt>(n.GetFrom().get());
  assert(from_ca && "ResolvePrediction requires a source ChunkAt");

  const bool explicit_zfill = n.IsOOBZeroFill();

  // For .transp, permute source shape before comparison so that
  // from[M,N] transposed as <1,0> correctly matches to[N,M].
  Shape effective_from = dec.from_ca_shape;
  if (n.operation == ".transp") {
    auto tc = cast<TransposeConfig>(n.GetConfig())->dim_values;
    ValueList from_vals = ShapeToValueList(dec.from_ca_shape);
    ValueList permuted = PermuteValueList(from_vals, tc);
    effective_from = Shape(permuted);
  }

  const bool is_pad = (n.operation == ".pad");
  const bool shape_mismatch =
      StaticShapeMismatch(effective_from, dec.to_ca_shape);
  const bool implicit_zfill = shape_mismatch && !explicit_zfill;

  // Suppress shape-mismatch warning for .pad -- padding inherently produces
  // different source/destination shapes; cooperative_fill handles pad values.
  if (implicit_zfill && !is_pad) {
    Warning(n.LOC(),
            "DMA source and destination tiles have different shapes; "
            "implicitly enabling zfill to avoid out-of-bounds access.");
  } else if (explicit_zfill && !shape_mismatch && !is_pad) {
    Warning(n.LOC(),
            "DMA zfill is redundant because source and destination tiles "
            "have the same shape; falling back to guard masking only.");
  }

  dec.is_zfill = explicit_zfill || implicit_zfill;
  if (explicit_zfill && !shape_mismatch) dec.is_zfill = false;

  auto outer_bounds = ShapeToValueList(dec.from_parent_shape);
  auto inner_bounds = ShapeToValueList(dec.from_ca_shape);
  auto offsets = ResolveTileCoordOffsets(*from_ca);

  if (n.operation == ".transp") {
    auto tc = cast<TransposeConfig>(n.GetConfig())->dim_values;
    outer_bounds = PermuteValueList(outer_bounds, tc);
    inner_bounds = PermuteValueList(inner_bounds, tc);
    offsets = PermuteValueList(offsets, tc);
  }

  auto pred_outer = ProjectPredDims(outer_bounds);
  auto pred_inner = ProjectPredDims(inner_bounds);
  auto pred_offsets = ProjectPredDims(offsets);

  auto pred_matches_box = [&](const ValueList& prediction) {
    for (size_t i = 0; i < 2 && i < prediction.size(); ++i) {
      auto pred_v = VIInt(prediction[i]);
      auto box_v = VIInt(param.box_shape[i]);
      if (!pred_v || !box_v || *pred_v != *box_v) return false;
    }
    return true;
  };

  if (dec.is_zfill) {
    // For .pad, padding is handled by cooperative_fill before the tiled copy.
    // The tiled copy should only copy valid source elements with predication;
    // ZFill would overwrite pad values with zeros.
    if (n.operation == ".pad") dec.is_zfill = false;

    if (pred_matches_box(pred_inner)) {
      param.need_pred = false;
      param.prediction.clear();
      dec.has_pred = false;
      dec.is_zfill = false;
    } else {
      param.need_pred = true;
      param.prediction = pred_inner;
      dec.has_pred = true;
    }
    return;
  }

  bool full_tiles = true;
  for (size_t i = 0; i < 2; ++i) {
    auto outer_dim = VIInt(pred_outer[i]);
    auto box_dim = VIInt(param.box_shape[i]);
    if (!outer_dim || !box_dim || *box_dim == 0 ||
        (*outer_dim % *box_dim) != 0) {
      full_tiles = false;
      break;
    }
  }

  if (full_tiles) {
    param.need_pred = false;
    param.prediction.clear();
    dec.has_pred = false;
    return;
  }

  ValueList prediction = {
      BuildPrediction(pred_outer[0], pred_offsets[0], param.box_shape[0]),
      BuildPrediction(pred_outer[1], pred_offsets[1], param.box_shape[1])};

  if (pred_matches_box(prediction)) {
    param.need_pred = false;
    param.prediction.clear();
    dec.has_pred = false;
    return;
  }

  param.need_pred = true;
  param.prediction = prediction;
  dec.has_pred = true;
}

void DMAPlan::ResolveTiledParams(const AST::DMA& n, DMALoweringDecision& dec,
                                 const Shape& unceiled_box) const {
  auto elem_type = dec.elem_type;
  size_t elem_bits = SizeOf(elem_type) * 8;
  size_t threads_count = dec.threads_count;
  assert(unceiled_box.Rank() == 2);
  size_t box_m = *VIInt(unceiled_box.ValueAt(0));
  size_t box_n = *VIInt(unceiled_box.ValueAt(1));

  // Trivially small boxes (fewer elements than threads) gain nothing from
  // tiled copy and the thread-layout ceiling would bloat the box well beyond
  // the actual allocation, causing OOB shared-memory writes.
  if (box_m * box_n < threads_count) {
    dec.strategy = DMAStrategy::NAIVE_COPY;
    dec.atom = dec.is_async ? CUDA_COPY_ATOM::ASYNC_COPY
                            : CUDA_COPY_ATOM::UNIVERSAL_COPY;
    return;
  }

  // 1. Detect alignment from parent strides.
  size_t from_align = DetectAlignBits(dec.from_strides[0], elem_bits);
  size_t to_align = DetectAlignBits(dec.to_strides[0], elem_bits);

  if (assume_aligned_global) {
    if (dec.direction == DMADirection::G2S) from_align = 128;
    if (dec.direction == DMADirection::S2G) to_align = 128;
  }
  // Single-row tile (includes 1D reshaped to [1, N]): the row stride is
  // never used for address computation, so it cannot limit alignment.
  if (box_m == 1) {
    from_align = 128;
    to_align = 128;
  }
  size_t align_bits = std::min(from_align, to_align);

  // 2. Determine whether cp.async is eligible.
  bool enable_cp_async = (dec.direction == DMADirection::G2S &&
                          align_bits >= 32 && threads_count % 32 == 0);
  if (enable_cp_async && (threads_count % 32 != 0)) {
    VST_DEBUG(dbgs() << "DMAPlan disabling cp.async: participating thread "
                        "count is not warp-multiple ("
                     << threads_count << ")\n"
                     << "  from: " << PSTR(n.GetFrom()) << "\n"
                     << "  to: " << PSTR(n.GetTo()) << "\n";);
    enable_cp_async = false;
  }

  // 3. Search for the best tiled copy configuration (slow path).
  size_t max_vf = (align_bits >= elem_bits) ? align_bits / elem_bits : 1;
  auto cfg = SearchTiledConfig(box_m, box_n, threads_count, max_vf, elem_bits,
                               enable_cp_async, /*allow_pred=*/true);
  if (!cfg.found) {
    dec.strategy = DMAStrategy::NAIVE_COPY;
    dec.atom = dec.is_async ? CUDA_COPY_ATOM::ASYNC_COPY
                            : CUDA_COPY_ATOM::UNIVERSAL_COPY;
    return;
  }

  // Swizzle requires power-of-2 val dimensions.
  if (dec.swizzle_mode != SwizMode::NONE) {
    for (auto& v : cfg.val_layout) {
      auto vi = VIInt(v);
      if (vi && (*vi & (*vi - 1)) != 0) {
        dec.strategy = DMAStrategy::NAIVE_COPY;
        dec.atom = dec.is_async ? CUDA_COPY_ATOM::ASYNC_COPY
                                : CUDA_COPY_ATOM::UNIVERSAL_COPY;
        return;
      }
    }
  }

  // 4. Fill the primary tiled copy params.
  TiledCopyParams param;
  param.align_bits = cfg.vf * elem_bits;
  param.box_shape = cfg.box_shape;
  param.thr_layout = cfg.thr_layout;
  param.val_layout = cfg.val_layout;

  ResolvePrediction(n, dec, param);

  dec.strategy = dec.has_pred ? DMAStrategy::TILED_COPY_PRED
                              : DMAStrategy::TILED_COPY_UNPRED;
  dec.atom = SelectCopyAtom(param.align_bits, enable_cp_async);

  // If cp_async was enabled but the chosen alignment is too small for any
  // cp_async atom, fall back to a synchronous vector atom instead.
  if (dec.atom == CUDA_COPY_ATOM::UNKNOWN && enable_cp_async) {
    dec.atom = SelectCopyAtom(param.align_bits, /*cp_async=*/false);
  }
  if (dec.atom == CUDA_COPY_ATOM::UNKNOWN) {
    dec.strategy = DMAStrategy::NAIVE_COPY;
    dec.atom = dec.is_async ? CUDA_COPY_ATOM::ASYNC_COPY
                            : CUDA_COPY_ATOM::UNIVERSAL_COPY;
    return;
  }

  dec.tiled_params = param;

  // 5. Compute an optimistic fast path for unknown-stride cases.
  //    When a stride is a runtime value, the slow path uses narrow
  //    vectorization.  Here we compute a 128-bit variant and codegen emits a
  //    runtime branch on the actual stride(s).
  bool from_stride_unknown =
      !dec.from_strides.empty() && !VIIsInt(dec.from_strides[0]);
  bool to_stride_unknown =
      !dec.to_strides.empty() && !VIIsInt(dec.to_strides[0]);
  bool any_stride_unknown = from_stride_unknown || to_stride_unknown;

  if (any_stride_unknown && !assume_aligned_global && param.align_bits < 128) {
    size_t fast_align;
    if (from_stride_unknown && to_stride_unknown) {
      fast_align = 128; // both checked at runtime
    } else {
      size_t known_side_align = from_stride_unknown ? to_align : from_align;
      fast_align = std::min(size_t{128}, known_side_align);
    }
    if (fast_align <= param.align_bits) fast_align = 0;
    bool fast_cp_async = dec.direction == DMADirection::G2S &&
                         fast_align >= 32 && threads_count % 32 == 0;

    size_t slow_box_m = static_cast<size_t>(*VIInt(param.box_shape[0]));
    size_t slow_box_n = static_cast<size_t>(*VIInt(param.box_shape[1]));
    size_t fast_max_vf = fast_align / elem_bits;

    auto fast_cfg =
        SearchTiledConfig(slow_box_m, slow_box_n, threads_count, fast_max_vf,
                          elem_bits, fast_cp_async, /*allow_pred=*/false);

    if (fast_cfg.found && fast_cfg.vf * elem_bits > param.align_bits) {
      TiledCopyParams fast_param;
      fast_param.box_shape = param.box_shape;
      fast_param.need_pred = param.need_pred;
      fast_param.prediction = param.prediction;
      fast_param.align_bits = fast_cfg.vf * elem_bits;
      fast_param.thr_layout = fast_cfg.thr_layout;
      fast_param.val_layout = fast_cfg.val_layout;

      CUDA_COPY_ATOM fast_atom =
          SelectCopyAtom(fast_param.align_bits, fast_cp_async);
      if (fast_atom != CUDA_COPY_ATOM::UNKNOWN) {
        dec.tiled_params_fast = fast_param;
        dec.atom_fast = fast_atom;

        // Rewrite the slow path's thr/val layout to match the fast path's
        // thread decomposition, preventing code bloat from divergent templates.
        size_t fast_thr_m =
            static_cast<size_t>(*VIInt(fast_param.thr_layout[0]));
        size_t fast_thr_n =
            static_cast<size_t>(*VIInt(fast_param.thr_layout[1]));
        if (slow_box_m % fast_thr_m == 0 && slow_box_n % fast_thr_n == 0) {
          param.thr_layout = fast_param.thr_layout;
          param.val_layout = {sbe::nu(slow_box_m / fast_thr_m),
                              sbe::nu(slow_box_n / fast_thr_n)};
          dec.tiled_params = param;
        }
      }
    }
  }

  VST_DEBUG(auto from_ca = dyn_cast<AST::ChunkAt>(n.GetFrom().get());
            auto to_ca = dyn_cast<AST::ChunkAt>(n.GetTo().get());
            dbgs() << "Resolved tiled copy params for DMA: " << PSTR(from_ca)
                   << " => " << PSTR(to_ca) << std::endl;
            dbgs() << "  from shape: " << STR(dec.from_ca_shape) << std::endl;
            dbgs() << "  to shape: " << STR(dec.to_ca_shape) << std::endl;
            dbgs() << "  from strides: " << STR(dec.from_strides) << std::endl;
            dbgs() << "  to strides: " << STR(dec.to_strides) << std::endl;
            dbgs() << "  unceiled box shape: " << STR(unceiled_box)
                   << std::endl;
            dbgs() << "  box shape: " << STR(param.box_shape) << std::endl;
            dbgs() << "  thread layout: " << STR(param.thr_layout) << std::endl;
            dbgs() << "  value layout: " << STR(param.val_layout) << std::endl;
            dbgs() << "  copy atom: " << STR(dec.atom) << std::endl;
            dbgs() << "  predication needed: " << param.need_pred << std::endl;
            if (param.need_pred) dbgs()
            << "  predication shape: " << STR(param.prediction) << std::endl;
            if (dec.tiled_params_fast) dbgs()
            << "  FAST PATH: atom=" << STR(dec.atom_fast)
            << " thr=" << STR(dec.tiled_params_fast->thr_layout)
            << " val=" << STR(dec.tiled_params_fast->val_layout)
            << " align=" << dec.tiled_params_fast->align_bits << std::endl;);
}

ValueList DMAPlan::ResolveStrides(const AST::ChunkAt& ca,
                                  const std::vector<size_t>& tc) const {
  auto sty = GetSpannedType(NodeType(ca));
  if (!sty) return {};
  auto strides = sty->GetStrides();

  // When SubSpan has explicit .step(), shapeinfer bakes the step factor into
  // the type's strides (inter-tile strides).  DMA needs the intra-tile strides
  // -- the original memory layout of the data -- so fall back to the root
  // symbol's strides.
  for (auto& op : ca.AllOperations()) {
    if (auto subspan = dyn_cast<AST::SOP::SubSpan>(op);
        subspan && subspan->GetSteps()) {
      auto orig_sty = GetSpannedType(GetSymbolType(ca.RefSymbol()));
      if (orig_sty) strides = orig_sty->GetStrides();
      break;
    }
  }

  if (tc.empty()) return strides;
  if (tc.size() != strides.size()) return strides;
  ValueList t_strds = strides;
  for (size_t i = 0; i < strides.size(); ++i) t_strds[i] = strides[tc[i]];
  return t_strds;
}
