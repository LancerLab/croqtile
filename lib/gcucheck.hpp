#ifndef __CHOREO_GCU_CHECK_INFO_HPP__
#define __CHOREO_GCU_CHECK_INFO_HPP__

// This apply the GCU target specific check and information annotation

#include "ast.hpp"
#include "visitor.hpp"

namespace Choreo {

extern Storage GCUDeviceParallelLevel(int);
extern int GCUDeviceParallelDepth(Storage);

struct GCUCheck : public VisitorWithSymTab {
private:
  std::unordered_map<std::string, AST::Parameter*> cur_params;
  int pl_depth = 0;
  int max_pl_depth = 0;
  int kernel_launch_count = 0;
  int local_level = 0;
  std::string cur_fname;
  std::string cur_arch;

  std::map<std::string, std::function<bool(size_t, size_t)>> op_map = {
      {">", [](size_t l, size_t r) { return l > r; }},
      {"<", [](size_t l, size_t r) { return l < r; }},
      {"==", [](size_t l, size_t r) { return l == r; }},
      {"!=", [](size_t l, size_t r) { return l != r; }},
      {">=", [](size_t l, size_t r) { return l >= r; }},
      {"<=", [](size_t l, size_t r) { return l <= r; }},
  };

private:
  bool BeforeVisitImpl(AST::Node& n) override {
    TraceEachVisit(n, "(pre)");
    if (auto cf = dyn_cast<AST::ChoreoFunction>(&n)) {
      local_level = 0;
      cur_params.clear();
      cur_fname = cf->name;
      kernel_launch_count = 0;
    } else if (auto pb = dyn_cast<AST::ParallelBy>(&n)) {
      if (pb->GetLevel() == Storage::NONE) {
        pl_depth++;
        pb->SetLevel(PLevel(*pb, pl_depth));
      } else
        pl_depth = PDepth(*pb, pb->GetLevel());
      if (pb->GetLevel() == Storage::SHARED) ++kernel_launch_count;
      max_pl_depth = pl_depth;
    }
    return true;
  }

  bool AfterVisitImpl(AST::Node& n) override {
    TraceEachVisit(n, "(post)");
    if (auto pb = dyn_cast<AST::ParallelBy>(&n)) {
      std::string append_note = ":";
      assert(pb->GetLevel() != Storage::NONE);
      pl_depth = PDepth(*pb, pb->GetLevel());
      if (CCtx().GetTarget() == CompileTarget::Topscc)
        append_note += PBLevelString(*pb);
      else
        append_note += std::to_string(max_pl_depth - pl_depth);
      auto pty = cast<BoundedITupleType>(NodeType(*pb->bpv));
      pty->AppendNote(append_note);
      for (auto& symbol : pb->cmpt_bpvs->AllValues())
        cast<BoundedITupleType>(NodeType(*symbol))->AppendNote(append_note);

      pl_depth--;
      assert(pl_depth >= 0 && "Unexpected parallel level");
      if (pl_depth == 0) {
        max_pl_depth = 0;
        local_level = 0;
      }
    }

    // mask stmts that are possible to be shared
    else if (auto c = dyn_cast<AST::Call>(&n))
      if (c->is_stmt) n.SetLevel(PLevel(n, pl_depth));

    return true;
  }

  void TraceEachVisit(AST::Node& n, std::string sup = "") {
    if (trace_visit) dbgs() << n.TypeNameString() << sup << "\n";
  }

private:
  Storage PLevel(AST::Node& n, int depth) {
    auto lvl = GCUDeviceParallelLevel(depth);
    if (lvl == Storage::NONE) {
      Error(n.LOC(), "the parallel level (depth: " + std::to_string(pl_depth) +
                         ") is not supported by current GCU architecture (" +
                         cur_arch + ").");
      ++error_count;
    }
    return lvl;
  }

  int PDepth(AST::Node& n, Storage l) {
    auto depth = GCUDeviceParallelDepth(l);
    if (depth == -1) {
      Error(n.LOC(), "the parallel level (" + STR(l) +
                         ") is not supported by  current GCU architecture (" +
                         cur_arch + ").");
      ++error_count;
    }
    return depth;
  }

public:
  const std::string PBLevelString(AST::ParallelBy& n) {
    if (max_pl_depth == 1) {
      switch (pl_depth) {
      case 0: return "global"; break;
      case 1: return "local"; break;
      default:
        choreo_unreachable(
            "unsupported parallel level: " + std::to_string(pl_depth) + ".");
      }
    } else
      return STR(PLevel(n, pl_depth));
    return "";
  }

  bool IsHost() const { return pl_depth == 0; }

  void CheckDMA(AST::DMA& n) {
    if (n.operation == ".any") return;

    assert(isa<AST::ChunkAt>(n.from));
    auto f_ca = cast<AST::ChunkAt>(n.from);
    auto f_name = f_ca->RefSymbol();
    auto f_sty = GetSpannedType(GetSymbolType(f_name));
    auto f_shape = f_sty->GetShape();
    auto f_rank = f_shape.Rank();

    assert(isa<AST::ChunkAt>(n.to));
    auto t_ca = cast<AST::ChunkAt>(n.to);
    auto t_name = t_ca->RefSymbol();
    auto t_sty = GetSpannedType(GetSymbolType(t_name));
    auto t_shape = t_sty->GetShape();
    auto t_rank = t_shape.Rank();

    // common limitation (currently guarded by memcheck)
    for (auto& sty : {f_sty, t_sty}) {
      if (sty->RuntimeShaped()) {
        std::string bs = sty->ByteSizeExpression(true);
        FCtx(cur_fname).AppendRtCheck(
            {bs,
             "<",
             std::to_string(1ULL << 32) + "ULL",
             n.LOC(),
             "The size of data transferred by DMA cannot exceed 2^32",
             {}});
      } else {
        if (sty->ByteSize() >= (1ULL << 32)) {
          Error(n.LOC(), "On " + cur_arch +
                             ", the size of data transferred by "
                             "DMA cannot exceed 2^32.");
          error_count++;
        }
      }
    }

    auto IsLinearCopy = [&]() -> bool {
      return f_ca->positions == nullptr && t_ca->positions == nullptr;
    };
    auto IsSlice = [&]() -> bool {
      return f_ca->positions != nullptr && t_ca->positions == nullptr;
    };
    auto IsDeslice = [&]() -> bool {
      return f_ca->positions == nullptr && t_ca->positions != nullptr;
    };
    auto RankLE5 = [&](const std::string& dma_op) {
      if (f_rank > 5) {
        Error(n.LOC(), "On " + cur_arch + ", the rank in " + dma_op +
                           " must be in range [1, 5].");
        error_count++;
      }
    };

    if (CCtx().GetArch() == TargetArch::GCU3 ||
        CCtx().GetArch() == TargetArch::GCU4) { // TODO: check this for GCU400
      // linear copy
      // omitted

      // transpose
      if (n.operation == ".transp" && IsLinearCopy()) {
        RankLE5("dma.transp(not slice nor deslice)");
        for (size_t idx = 1; idx < f_rank; ++idx)
          CheckDimSize(f_shape, idx, "<", 1 << 24, n.from->LOC());
        for (size_t idx = 1; idx < t_rank; ++idx)
          CheckDimSize(t_shape, idx, "<", 1 << 24, n.to->LOC());
        auto bpe = sbe::nu((int)(SizeOf(f_sty->f_type)));
        auto value = (f_shape.ValueAt(0) * bpe + sbe::nu(127)) / sbe::nu(128) *
                     sbe::nu(128);
        CheckValue(value, "<", 1 << 24, n.from->LOC(),
                   "CeilTo128Byte(src_dim0_size * bpe) < 2^24.");
        value = (t_shape.ValueAt(0) * bpe + sbe::nu(127)) / sbe::nu(128) *
                sbe::nu(128);
        CheckValue(value, "<", 1 << 24, n.to->LOC(),
                   "CeilTo128Byte(dst_dim0_size * bpe) < 2^24.");
        for (size_t idx = 1; idx < t_rank; ++idx)
          value = value * t_shape.ValueAt(idx);
        CheckValue(value, "<", 1ULL << 32, n.to->LOC(),
                   "CeilTo128Byte(bpe * dst dim0) * dim1 * dim2 "
                   "* dim3 * dim4 < 4GB.");
      }

      // pad
      if (n.operation == ".pad" && IsLinearCopy()) {
        RankLE5("dma.pad");
        auto pc = cast<PadConfig>(n.config);
        assert(f_rank == pc->pad_low.size());

        for (auto v : pc->pad_low) {
          if (v > (1 << 11)) {
            Error(n.LOC(), "On GCU300, the value of padding_low in "
                           "dma.pad must be in range [0, 2^11].");
            error_count++;
          }
        }

        for (auto v : pc->pad_high) {
          if (v > (1 << 11)) {
            Error(n.LOC(), "On " + cur_arch +
                               ", the value of padding_high in dma.pad must be "
                               "in range [0, 2^11].");
            error_count++;
          }
        }
        // padding_mid
        for (size_t idx = 0; idx < f_rank; ++idx) {
          size_t v = pc->pad_mid[idx];
          if (idx == f_rank - 1) {
            if (v != 0) {
              Error(n.LOC(),
                    "On " + cur_arch +
                        ", the value of padding_mid[rank-1] in dma.pad "
                        "must be 0 (mid padding of dim[rank-1] is not "
                        "supported by "
                        "the hardware).");
              error_count++;
            }
          } else if (v > (1 << 10)) {
            Error(n.LOC(), "On " + cur_arch +
                               ", the value of padding_mid in "
                               "dma.pad must be in range [0, 2^10].");
            error_count++;
          }
        }
        if (f_rank == 5) {
          if (pc->pad_low[0] != 0) {
            Error(n.LOC(), "On " + cur_arch +
                               ", dma.pad does not support 5-dimensional "
                               "array (if dim is 5, pad_low[0] must be 0).");
            error_count++;
          }
          if (pc->pad_high[0] != 0) {
            Error(n.LOC(), "On " + cur_arch +
                               ", dma.pad does not support 5-dimensional "
                               "array (if dim is 5, pad_high[0] must be 0).");
            error_count++;
          }
          if (pc->pad_mid[0] != 0) {
            Error(n.LOC(), "On " + cur_arch +
                               ", dma.pad does not support 5-dimensional "
                               "array (if dim is 5, pad_mid[0] must be 0).");
            error_count++;
          }
        }

        if (pc->value.t != f_sty->f_type) {
          Error(n.from->LOC(), "On " + cur_arch +
                                   ", Data type of pad value is "
                                   "inconsistent with that of data in dma: " +
                                   STR(pc->value.t) + " vs. " +
                                   STR(f_sty->f_type) + ".");
          error_count++;
        }
      }

      // slice
      if (n.operation == ".copy" && IsSlice()) {
        RankLE5("dma.copy(slice)");
        for (size_t idx = 0; idx < f_rank; ++idx)
          CheckDimSize(f_shape, idx, "<", 1 << 24, n.from->LOC());
        for (size_t idx = 0; idx < t_rank; ++idx)
          CheckDimSize(t_shape, idx, "<", 1 << 24, n.to->LOC());
        // TODO: offset limitation: [0, 2^24)
        if (f_rank == 5) {
          auto first = f_ca->positions->ValueAt(0);
          auto t = dyn_cast<BoundedITupleType>(first->GetType());
          assert(t != nullptr);
          if (VIIsInt(t->ubounds.ValueAt(0))) {
            if (!IsValueItemEqual(1, t->ubounds.ValueAt(0))) {
              Error(n.LOC(), "On " + cur_arch +
                                 ", dma.copy(slice) does not "
                                 "support 5-dimensional "
                                 "array (if dim is 5, offsets[0] must be 0).");
              error_count++;
            }
          } else {
            choreo_unreachable("unexpected situation");
            // TODO
            // Is that the case?
          }
        }
        // TODO: check for auto padding
      }

      // deslice
      if (n.operation == ".copy" && IsDeslice()) {
        RankLE5("dma.copy(deslice)");
        for (size_t idx = 0; idx < f_rank; ++idx)
          CheckDimSize(f_shape, idx, "<", 1 << 24, n.from->LOC());
        for (size_t idx = 0; idx < t_rank; ++idx)
          CheckDimSize(t_shape, idx, "<", 1 << 24, n.to->LOC());
        // TODO: offset limitation: [0, 2^24)
        if (t_rank == 5) {
          auto first = t_ca->positions->ValueAt(0);
          auto t = dyn_cast<BoundedITupleType>(first->GetType());
          assert(t != nullptr);
          if (VIIsInt(t->ubounds.ValueAt(0))) {
            if (!IsValueItemEqual(1, t->ubounds.ValueAt(0))) {
              Error(n.LOC(), "On " + cur_arch +
                                 ", dma.copy(deslice) does not "
                                 "support 5-dimensional "
                                 "array (if dim is 5, offsets[0] must be 0).");
              error_count++;
            }
          } else {
            choreo_unreachable("unexpected situation");
            // TODO
            // Is that the case?
          }
        }
      }

      // slice transpose
      if (n.operation == ".transp" && IsSlice()) {
        RankLE5("dma.transp(slice then transpose)");
        for (size_t idx = 1; idx < f_rank; ++idx)
          CheckDimSize(f_shape, idx, "<", 1 << 24, n.from->LOC());
        for (size_t idx = 1; idx < t_rank; ++idx)
          CheckDimSize(t_shape, idx, "<", 1 << 24, n.to->LOC());
        auto bpe = sbe::nu(SizeOf(f_sty->f_type));
        auto value = (f_shape.ValueAt(0) * bpe + sbe::nu(127)) / sbe::nu(128) *
                     sbe::nu(128);
        CheckValue(value, "<", 1 << 24, n.from->LOC(),
                   "CeilTo128Byte(src_dim0_size * bpe) < 2^24.");
        value = (t_shape.ValueAt(0) * bpe + sbe::nu(127)) / sbe::nu(128) *
                sbe::nu(128);
        CheckValue(value, "<", 1 << 24, n.to->LOC(),
                   "CeilTo128Byte(dst_dim0_size * bpe) < 2^24.");
        for (size_t idx = 1; idx < t_rank; ++idx)
          value = value * t_shape.ValueAt(idx);
        CheckValue(value, "<", 1ULL << 32, n.to->LOC(),
                   "CeilTo128Byte(bpe * dst dim0) * dim1 * dim2 "
                   "* dim3 * dim4 < 4GB.");
      }

      // transpose deslice
      if (n.operation == ".transp" && IsDeslice()) {
        RankLE5("dma.transp(transpose then deslice)");
        for (size_t idx = 1; idx < f_rank; ++idx)
          CheckDimSize(f_shape, idx, "<", 1 << 24, n.from->LOC());
        for (size_t idx = 1; idx < t_rank; ++idx)
          CheckDimSize(t_shape, idx, "<", 1 << 24, n.to->LOC());
        auto bpe = sbe::nu((int)(SizeOf(f_sty->f_type)));
        auto value = (f_shape.ValueAt(0) * bpe + sbe::nu(127)) / sbe::nu(128) *
                     sbe::nu(128);
        CheckValue(value, "<", 1 << 24, n.from->LOC(),
                   "CeilTosbe::nu(128)Byte(src_dim0_size * bpe) < 2^24.");
        value = (t_shape.ValueAt(0) * bpe + sbe::nu(127)) / sbe::nu(128) *
                sbe::nu(128);
        CheckValue(value, "<", 1 << 24, n.to->LOC(),
                   "CeilTo128Byte(dst_dim0_size * bpe) < 2^24.");
        for (size_t idx = 1; idx < t_rank; ++idx)
          value = value * t_shape.ValueAt(idx);
        CheckValue(value, "<", 1ULL << 32, n.to->LOC(),
                   "CeilTo128Byte(bpe * dst dim0) * dim1 * dim2 "
                   "* dim3 * dim4 < 4GB.");
      }

      return;
    }

    if (CCtx().GetArch() == TargetArch::GCU20 ||
        CCtx().GetArch() == TargetArch::GCU21) {
      // linear copy
      // omitted

      // transpose
      if (n.operation == ".transp" && IsLinearCopy()) {
        RankLE5("dma.transp(not slice nor deslice)");
        if (f_rank == 5) {
          CheckDimSize(f_shape, 0, "<", 1 << 24, n.from->LOC());
          for (size_t idx = 1; idx < f_rank; ++idx)
            CheckDimSize(f_shape, idx, "<", 1 << 16, n.from->LOC());
        } else {
          for (size_t idx = 0; idx < f_rank; ++idx)
            CheckDimSize(f_shape, idx, "<", 1 << 16, n.from->LOC());
        }

        if (SizeOf(t_sty->f_type) == 4) {
          for (size_t idx = 0; idx < t_rank; ++idx) {
            CheckDimSize(t_shape, idx, ">", 1, n.to->LOC());
            CheckDimSize(t_shape, idx, "<", (1 << 16) - 32, n.to->LOC());
          }
        }

        auto tc = cast<TransposeConfig>(n.config);
        if (f_rank == 5 && tc->dim_values[0] != 0) {
          Error(
              n.LOC(),
              "On " + cur_arch +
                  ", dma.transp(not slice nor deslice) does not "
                  "support 5-dimensional array (if dim is 5, layout[0] must be "
                  "0).");
          error_count++;
        }
      }

      // pad
      if (n.operation == ".pad" && IsLinearCopy()) {
        RankLE5("dma.pad");
        if (f_rank == 5) {
          CheckDimSize(f_shape, 0, "<", 1 << 24, n.from->LOC());
          for (size_t idx = 1; idx < f_rank; ++idx)
            CheckDimSize(f_shape, idx, "<", 1 << 16, n.from->LOC());
        } else {
          for (size_t idx = 0; idx < f_rank; ++idx)
            CheckDimSize(f_shape, idx, "<", 1 << 16, n.from->LOC());
        }

        // shape of n.to is the same as n.from's
        // so the check of n.to is omitted

        auto pc = cast<PadConfig>(n.config);
        assert(f_rank == pc->pad_low.size());

        for (auto v : pc->pad_low) {
          if (v > (1 << 11)) {
            Error(n.LOC(), "On " + cur_arch +
                               ", the value of padding_low in "
                               "dma.pad must be in range [0, 2^11].");
            error_count++;
          }
        }

        for (auto v : pc->pad_high) {
          if (v > (1 << 11)) {
            Error(n.LOC(), "On " + cur_arch +
                               ", the value of padding_high in "
                               "dma.pad must be in range [0, 2^11].");
            error_count++;
          }
        }
        // padding_mid
        for (size_t idx = 0; idx < f_rank; ++idx) {
          size_t v = pc->pad_mid[idx];
          if (idx == f_rank - 1) {
            if (v != 0) {
              Error(n.LOC(),
                    "On " + cur_arch +
                        ", the value of padding_mid[rank-1] in dma.pad "
                        "must be 0 (mid padding of dim[rank-1] is not "
                        "supported by "
                        "the hardware).");
              error_count++;
            }
          } else if (v > (1 << 10)) {
            Error(n.LOC(), "On " + cur_arch +
                               ", the value of padding_mid in "
                               "dma.pad must be in range [0, 2^10].");
            error_count++;
          }
        }
        if (f_rank == 5) {
          if (pc->pad_low[0] != 0) {
            Error(n.LOC(), "On " + cur_arch +
                               ", dma.pad does not support 5-dimensional "
                               "array (if dim is 5, pad_low[0] must be 0).");
            error_count++;
          }
          if (pc->pad_high[0] != 0) {
            Error(n.LOC(), "On " + cur_arch +
                               ", dma.pad does not support 5-dimensional "
                               "array (if dim is 5, pad_high[0] must be 0).");
            error_count++;
          }
          if (pc->pad_mid[0] != 0) {
            Error(n.LOC(), "On " + cur_arch +
                               ", dma.pad does not support 5-dimensional "
                               "array (if dim is 5, pad_mid[0] must be 0).");
            error_count++;
          }
        }

        if (pc->value.t != f_sty->f_type) {
          Error(n.from->LOC(), "On " + cur_arch +
                                   ", Data type of pad value is "
                                   "inconsistent with that of data in dma: " +
                                   STR(pc->value.t) + " vs. " +
                                   STR(f_sty->f_type) + ".");
          error_count++;
        }
      }

      // slice
      if (n.operation == ".copy" && IsSlice()) {
        RankLE5("dma.copy(slice)");
        for (size_t idx = 0; idx < f_rank; ++idx)
          CheckDimSize(f_shape, idx, "<", 1 << 24, n.from->LOC());

        for (size_t idx = 0; idx < t_rank; ++idx)
          CheckDimSize(t_shape, idx, "<", 1 << 16, n.to->LOC());
        if (f_rank == 5) {
          auto first = f_ca->positions->ValueAt(0);
          auto t = dyn_cast<BoundedITupleType>(first->GetType());
          assert(t != nullptr);
          if (VIIsInt(t->ubounds.ValueAt(0))) {
            if (!IsValueItemEqual(1, t->ubounds.ValueAt(0))) {
              Error(n.LOC(), "On " + cur_arch +
                                 ", dma.copy(slice) does not "
                                 "support 5-dimensional "
                                 "array (if dim is 5, offsets[0] must be 0).");
              error_count++;
            }
          } else {
            choreo_unreachable("unexpected situation");
            // TODO
            // Is that the case?
          }
        }
      }

      // deslice
      if (n.operation == ".copy" && IsDeslice()) {
        RankLE5("dma.copy(deslice)");
        for (size_t idx = 0; idx < f_rank; ++idx)
          CheckDimSize(f_shape, idx, "<", 1 << 16, n.from->LOC());

        for (size_t idx = 0; idx < t_rank; ++idx)
          CheckDimSize(t_shape, idx, "<", 1 << 24, n.to->LOC());
      }

      // slice transpose
      if (n.operation == ".transp" && IsSlice()) {
        RankLE5("dma.transp(slice then transpose)");
        if (f_rank == 5) {
          CheckDimSize(f_shape, 0, "<", 1 << 24, n.from->LOC());
          for (size_t idx = 1; idx < f_rank; ++idx)
            CheckDimSize(f_shape, idx, "<", 1 << 16, n.from->LOC());
        } else {
          for (size_t idx = 0; idx < f_rank; ++idx)
            CheckDimSize(f_shape, idx, "<", 1 << 16, n.from->LOC());
        }

        if (SizeOf(t_sty->f_type) == 4) {
          for (size_t idx = 0; idx < t_rank; ++idx) {
            CheckDimSize(t_shape, idx, ">", 1, n.to->LOC());
            CheckDimSize(t_shape, idx, "<", (1 << 16) - 32, n.to->LOC());
          }
        }

        if (f_rank == 5) {
          auto first = f_ca->positions->ValueAt(0);
          auto t = dyn_cast<BoundedITupleType>(first->GetType());
          assert(t != nullptr);
          if (VIIsInt(t->ubounds.ValueAt(0))) {
            if (!IsValueItemEqual(1, t->ubounds.ValueAt(0))) {
              Error(n.LOC(), "On " + cur_arch +
                                 ", dma.transp(slice then "
                                 "transpose) does not support 5-dimensional "
                                 "array (if dim is 5, offsets[0] must be 0).");
              error_count++;
            }
          } else {
            // TODO
            // Is that the case?
          }
        }

        auto tc = cast<TransposeConfig>(n.config);
        if (f_rank == 5 && tc->dim_values[0] != 0) {
          Error(n.LOC(), "On " + cur_arch +
                             ", dma.transp(slice then transpose) "
                             "does not support 5-dimensional "
                             "array (if dim is 5, layout[0] must be 0).");
          error_count++;
        }
      }

      // transpose deslice
      if (n.operation == ".transp" && IsDeslice()) {
        RankLE5("dma.transp(transpose then deslice)");
        if (f_rank == 5) {
          CheckDimSize(f_shape, 0, "<", 1 << 24, n.from->LOC());
          for (size_t idx = 1; idx < f_rank; ++idx)
            CheckDimSize(f_shape, idx, "<", 1 << 16, n.from->LOC());
        } else {
          for (size_t idx = 0; idx < f_rank; ++idx)
            CheckDimSize(f_shape, idx, "<", 1 << 16, n.from->LOC());
        }

        if (SizeOf(t_sty->f_type) == 4) {
          for (size_t idx = 0; idx < t_rank; ++idx) {
            CheckDimSize(t_shape, idx, ">", 1, n.to->LOC());
            CheckDimSize(t_shape, idx, "<", (1 << 16) - 32, n.to->LOC());
          }
        }

        if (t_rank == 5) {
          auto first = t_ca->positions->ValueAt(0);
          auto t = dyn_cast<BoundedITupleType>(first->GetType());
          assert(t != nullptr);
          if (VIIsInt(t->ubounds.ValueAt(0))) {
            if (!IsValueItemEqual(1, t->ubounds.ValueAt(0))) {
              Error(n.LOC(), "On " + cur_arch +
                                 ", dma.transp(transpose then "
                                 "deslice) does not support 5-dimensional "
                                 "array (if dim is 5, offsets[0] must be 0).");
              error_count++;
            }
          } else {
            // TODO
            // Is that the case?
          }
        }

        auto tc = cast<TransposeConfig>(n.config);
        if (t_rank == 5 && tc->dim_values[0] != 0) {
          Error(n.LOC(), "On " + cur_arch +
                             ", dma.transp(transpose then deslice) "
                             "does not support 5-dimensional "
                             "array (if dim is 5, layout[0] must be 0).");
          error_count++;
        }
      }
      return;
    }

    choreo_unreachable("Unsupported target architecture: " + cur_arch);
  }

  void CheckDimSize(const Shape& s, size_t idx, const std::string& op,
                    size_t limit, const location& loc) {
    assert(idx < s.Rank());
    std::string message = "the " + Ordinal(idx + 1) + " dim " +
                          ValueItemAsString(s.ValueAt(idx)) + " " + op + " " +
                          std::to_string(limit) + ".";

    CheckValue(s.ValueAt(idx), op, limit, loc, message);
  }

  void CheckValue(const ValueItem& vi, const std::string& op, size_t limit,
                  const location& loc, std::string message = "") {
    VST_DEBUG(dbgs() << "[GCUCHECK] Generated check at " << loc << ": "
                     << ValueItemAsString(vi) << " " << op << " "
                     << std::to_string(limit) + "ULL"
                     << "\n\twith message: " << message << "\n");
    message = "On " + cur_arch + ", must satisfy: " + message;
    if (auto vi_int = VIInt(vi); vi_int && op_map.count(op)) {
      if (!op_map[op](*vi_int, limit)) {
        Error(loc, message);
        error_count++;
      }
    } else {
      auto vi_str = ValueItemAsString(vi, true);
      VST_DEBUG(dbgs() << "[GCUCHECK] Generated runtime check at " << loc
                       << ": " << vi_str << " " << op << " "
                       << std::to_string(limit) + "ULL"
                       << "\n\twith message: " << message << "\n");
      FCtx(cur_fname).AppendRtCheck(
          {vi_str, op, std::to_string(limit) + "ULL", loc, message, {}});
    }
  }

public:
  GCUCheck()
      : VisitorWithSymTab("gcu", CCtx().GetGlobalSymbolTable()),
        cur_arch(STR(CCtx().GetArch())) {}
  ~GCUCheck() {}

  bool Visit(AST::MultiNodes& n) override {
    TraceEachVisit(n);
    return true;
  }
  bool Visit(AST::MultiValues& n) override {
    TraceEachVisit(n);
    return true;
  }
  bool Visit(AST::IntLiteral& n) override {
    TraceEachVisit(n);
    return true;
  }
  bool Visit(AST::FloatLiteral& n) override {
    TraceEachVisit(n);

    if (CCtx().GetTarget() == CompileTarget::Factor) {
      if (!n.IsFloat32()) {
        Error(n.LOC(), "Factor backend in Choreo does not support " +
                           PSTR(n.GetType()) + " float-point number yet!");
        error_count++;
      }
    }

    return true;
  }
  bool Visit(AST::Boolean& n) override {
    TraceEachVisit(n);
    return true;
  }
  bool Visit(AST::Expr& n) override {
    TraceEachVisit(n);
    return true;
  }
  bool Visit(AST::MultiDimSpans& n) override {
    TraceEachVisit(n);
    return true;
  }
  bool Visit(AST::NamedTypeDecl& n) override {
    TraceEachVisit(n);
    return true;
  }
  bool Visit(AST::NamedVariableDecl& n) override {
    TraceEachVisit(n);
    auto ty = GetSymbolType(n.name_str);
    if (isa<EventType>(ty) || isa<EventArrayType>(ty))
      if (CCtx().GetArch() == TargetArch::GCU20 ||
          CCtx().GetArch() == TargetArch::GCU21) {
        Error(n.LOC(), "Event is not supported on " + cur_arch + ".");
        error_count++;
      }

    if (isa<AST::Select>(n.init_expr))
      if (IsHost() && CCtx().GetTarget() != CompileTarget::Factor) {
        Error(n.LOC(), "select in host is not supported.");
        error_count++;
      }

    if (!isa<SpannedType>(ty)) return true;
    auto sty = cast<SpannedType>(ty);

    auto st = sty->GetStorage();
    switch (st) {
    case Storage::GLOBAL:
      if (pl_depth != 0) {
        Error(n.LOC(), "global variable '" + n.name_str +
                           "` mustn't be declared inside parallel-by.");
        error_count++;
      }
      break;
    case Storage::SHARED:
      if (pl_depth == 0) {
        Error(n.LOC(), "shared variable '" + n.name_str +
                           "` must be declared inside parallel-by.");
        error_count++;
      }
      if (sty->RuntimeShaped() && !CCtx().MemReuse()) {
        Error(n.LOC(), "GCU forbids shared variable '" + n.name_str +
                           "` to be dynamically shaped (by " +
                           STR(sty->GetShape()) + ").");
        error_count++;
      }
      break;
    case Storage::LOCAL:
      if (pl_depth == 0) {
        Error(n.LOC(), "local variable '" + n.name_str +
                           "` must be declared inside parallel-by.");
        error_count++;
      } else if (local_level == 0)
        local_level = pl_depth;
      if (sty->RuntimeShaped() && !CCtx().MemReuse()) {
        Error(n.LOC(), "GCU forbids local variable '" + n.name_str +
                           "` to be dynamically shaped (by " +
                           STR(sty->GetShape()) + ").");
        error_count++;
      }
      break;
    default:
      Error(n.LOC(), "can not declare variable '" + n.name_str + "` as " +
                         STR(st) + " inside choreo function.");
      error_count++;
      break;
    }
    return true;
  }
  bool Visit(AST::IntTuple& n) override {
    TraceEachVisit(n);
    return true;
  }
  bool Visit(AST::Assignment& n) override {
    TraceEachVisit(n);
    return true;
  }
  bool Visit(AST::IntIndex& n) override {
    TraceEachVisit(n);
    return true;
  }
  bool Visit(AST::DataType& n) override {
    TraceEachVisit(n);
    return true;
  }
  bool Visit(AST::Identifier& n) override {
    TraceEachVisit(n);
    return true;
  }
  bool Visit(AST::Parameter& n) override {
    TraceEachVisit(n);
    if (n.sym) cur_params.emplace(InScopeName(n.sym->name), &n);
    return true;
  }
  bool Visit(AST::ParamList& n) override {
    TraceEachVisit(n);
    return true;
  }
  bool Visit(AST::ParallelBy& n) override {
    TraceEachVisit(n);
    if (CCtx().GetTarget() == CompileTarget::Factor) {
      auto shape = GetShape(NodeType(n));
      if (shape.IsDynamic()) {
        Error(n.LOC(),
              "symbolic bound value is not supported for Factor backend yet.");
        error_count++;
      }
    }
    return true;
  }
  bool Visit(AST::WhereBind& n) override {
    TraceEachVisit(n);
    return true;
  }
  bool Visit(AST::WithIn& n) override {
    TraceEachVisit(n);
    return true;
  }
  bool Visit(AST::WithBlock& n) override {
    TraceEachVisit(n);
    return true;
  }
  bool Visit(AST::Memory& n) override {
    TraceEachVisit(n);
    return true;
  }
  bool Visit(AST::SpanAs& n) override {
    TraceEachVisit(n);
    return true;
  }
  bool Visit(AST::DMA& n) override {
    TraceEachVisit(n);

    // Check DMA first.
    CheckDMA(n);

    // The user does not have to explicitly claim a global memory that requires
    // direct copy from host to device. Here Choreo judge if a spanned memory is
    // shadowed from the data movement. Later, codegen handles such a shadow.
    if (!isa<AST::ChunkAt>(n.from)) return true;
    auto f_name = cast<AST::ChunkAt>(n.from)->RefSymbol();

    auto sty = GetSpannedType(GetSymbolType(f_name));

    // storage level must be specified
    if (sty->GetStorage() == Storage::NONE) return false;

    // not referencing the parameter
    if (!cur_params.count(InScopeName(f_name))) return false;

    auto annotate_by_storage = [this, &f_name](Storage st) {
      switch (st) {
      case Storage::GLOBAL:
      case Storage::SHARED:
      case Storage::LOCAL: {
        auto p = cur_params[InScopeName(f_name)];
        if (p->attr == ParamAttr::NONE) p->attr = ParamAttr::SHADOW_TO_GLOBAL;
        break;
      }
      default: break;
      }
    };
    if (auto to = dyn_cast<AST::ChunkAt>(n.to))
      annotate_by_storage(
          cast<SpannedType>(GetSymbolType(to->RefSymbol()))->GetStorage());
    else if (auto m = dyn_cast<AST::Memory>(n.to))
      annotate_by_storage(m->st);

    return true;
  }
  bool Visit(AST::ChunkAt& n) override {
    TraceEachVisit(n);
    return true;
  }
  bool Visit(AST::Wait& n) override {
    TraceEachVisit(n);
    return true;
  }
  bool Visit(AST::Call& n) override {
    TraceEachVisit(n);
    if (n.is_arith_bif && (CCtx().GetArch() == TargetArch::GCU20 ||
                           CCtx().GetArch() == TargetArch::GCU21)) {
      Error(n.LOC(), "Arithmetic built-in function is not supported on GCU2.");
      error_count++;
    }
    return true;
  }
  bool Visit(AST::Rotate& n) override {
    TraceEachVisit(n);
    return true;
  }

  bool Visit(AST::Synchronize& n) override {
    TraceEachVisit(n);

    auto pl2s = [this]() {
      switch (pl_depth) {
      case 0: return Storage::GLOBAL;
      case 1: return Storage::SHARED;
      case 2: return Storage::LOCAL;
      case 3: return Storage::SUB;
      default: break;
      }
      return Storage::NONE;
    };

    switch (n.scope->Get()) {
    case Storage::GLOBAL:
      if (pl_depth != 0) {
        Error(n.LOC(), "unsupported: " + PSTR(n.scope) +
                           " synchronization in " + STR(pl2s()) + " scope.");
        error_count++;
      }
      break;
    case Storage::SHARED:
      if (pl_depth == 0) {
        Error(n.LOC(), "unsupported: " + PSTR(n.scope) +
                           " synchronization in " + STR(pl2s()) + " scope.");
        error_count++;
      }
      break;
    case Storage::LOCAL:
      if (CCtx().GetArch() != TargetArch::GCU4) {
        Error(n.LOC(), STR(CCtx().GetArch()) + " does not support " +
                           PSTR(n.scope) + " synchronization.");
        error_count++;
      } else if (pl_depth != 3) {
        Error(n.LOC(), "unsupported: " + PSTR(n.scope) +
                           " synchronization in " + STR(pl2s()) + " scope.");
        error_count++;
      }
      break;
    default:
      Error(n.scope->LOC(),
            "unsupported synchronization: " + PSTR(n.scope) + ".");
    }
    return true;
  }

  bool Visit(AST::Select& n) override {
    TraceEachVisit(n);
    return true;
  }
  bool Visit(AST::Return& n) override {
    TraceEachVisit(n);
    return true;
  }
  bool Visit(AST::LoopRange& n) override {
    TraceEachVisit(n);
    return true;
  }
  bool Visit(AST::ForeachBlock& n) override {
    TraceEachVisit(n);
    return true;
  }
  bool Visit(AST::FunctionDecl& n) override {
    TraceEachVisit(n);
    return true;
  }
  bool Visit(AST::ChoreoFunction& n) override {
    TraceEachVisit(n);
    return true;
  }
  bool Visit(AST::CppSourceCode& n) override {
    TraceEachVisit(n);
    return true;
  }
  bool Visit(AST::Program& n) override {
    TraceEachVisit(n);
    return true;
  }

  bool HasError() override {
    if (error_count)
      dbgs() << "Totally " << error_count << " errors have been detected.\n";
    return error_count != 0;
  }
};

} // end namespace Choreo

#endif // __CHOREO_GCU_CHECK_INFO_HPP__
