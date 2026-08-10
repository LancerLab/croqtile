#ifndef __CHOREO_GCU_ADAPT_HPP__
#define __CHOREO_GCU_ADAPT_HPP__

// GCU target-specific validation and MMA adaptation pass.

#include "assess.hpp"
#include "ast.hpp"
#include "gcu_mma_limit.hpp"
#include "lower_libcall.hpp"
#include "target_utils.hpp"
#include "visitor.hpp"

namespace Choreo {

inline size_t GCUVLdStAlignment(ptr<VectorType> vt) {
  if (CCtx().GetArch() != "gcu400") return SizeOf(*vt);
  return 1;
}

struct GCUAdaptor : public VisitorWithSymTab {
private:
  std::unordered_map<std::string, AST::Parameter*> cur_params;
  AST::Node* cur_fnode;
  std::string cur_arch;
  std::stack<ParallelLevel> levels;
  std::stack<bool> cooperative_stack;

private:
  ParallelLevel Level() const {
    assert(levels.size() > 0);
    return levels.top();
  }

  bool IsInCooperativeBlock() const {
    return !cooperative_stack.empty() && cooperative_stack.top();
  }

  bool Assess(const ValueItem& pred, const std::string& message,
              AST::Node& node, AST::Node* emit_node = nullptr,
              UsageType uty = UsageType::HardwareConstraint) {
    return FCtx(fname)
        .GetAssessor(*this)
        .Assess(AssessPolicy::Error, pred, message, uty, AssessType::USE_SITE,
                node.LOC(), &node,
                ((emit_node == nullptr) ? cur_fnode : emit_node))
        .passed;
  }

private:
  bool BeforeVisitImpl(AST::Node& n) override {
    TraceEachVisit(n, "(pre)");
    if (isa<AST::Program>(&n)) {
      if (CCtx().MaxLocalMemCapacity() > 0)
        Error1(n.LOC(), "Local memory capacity cannot be set manually on GCU.");
    } else if (isa<AST::ChoreoFunction>(&n)) {
      cur_params.clear();
      cur_fnode = &n;
      levels.push(ParallelLevel::SEQ);
      cooperative_stack.push(false);
    } else if (auto pb = dyn_cast<AST::ParallelBy>(&n)) {
      if (!ValidLevel(*pb, pb->GetLevel())) return false;
      levels.push(pb->GetLevel());
      cooperative_stack.push(pb->IsCooperative());
      VST_DEBUG(pb->InlinePrint(dbgs());
                dbgs() << ": level << " << STR(pb->GetLevel()) << " / "
                       << STR(TargetMaxLevel()) << "\n");
    }
    return true;
  }

  bool AfterVisitImpl(AST::Node& n) override {
    TraceEachVisit(n, "(post)");
    if (auto pb = dyn_cast<AST::ParallelBy>(&n)) {
      std::string append_note;
      levels.pop();
      cooperative_stack.pop();
      append_note = STR(pb->GetLevel());
      auto pty = cast<BoundedITupleType>(NodeType(*pb->BPV()));
      pty->AddNote("pv", append_note);
      for (auto& symbol : pb->AllSubPVs())
        NodeType(*symbol)->AddNote("pv", append_note);
    }

    // mask stmts that are possible to be shared
    else if (auto c = dyn_cast<AST::Call>(&n))
      if (!c->IsExpr()) n.SetLevel(Level());

    return true;
  }

  void TraceEachVisit(AST::Node& n, std::string sup = "") {
    if (trace_visit) dbgs() << n.TypeNameString() << sup << "\n";
  }

public:
  bool ValidLevel(AST::Node& n, ParallelLevel pl) {
    if (pl == ParallelLevel::NONE) {
      Error1(n.LOC(), "internal error: the parallel level is not inferenced.");
      return false;
    } else if (CCtx().GetArch() == "gcu300" && pl == ParallelLevel::GROUP) {
      Error1(n.LOC(), "the parallel level (" + STR(pl) +
                          ") is not supported by current GCU architecture (" +
                          cur_arch + ").");
      return false;
    }
    return true;
  }

  bool IsHost() const { return Level() == ParallelLevel::SEQ; }

  void CheckDMA(AST::DMA& n) {
    if (n.operation == ".any") return;

    if (!isa<AST::ChunkAt>(n.from) || !isa<AST::ChunkAt>(n.to)) return;

    auto f_ca = cast<AST::ChunkAt>(n.from);
    auto f_name = f_ca->RefSymbol();
    auto f_sty = GetSpannedType(NodeType(*f_ca));
    auto f_shape = f_sty->GetShape();
    auto f_rank = f_shape.Rank();

    auto t_ca = cast<AST::ChunkAt>(n.to);
    auto t_name = t_ca->RefSymbol();
    auto t_sty = GetSpannedType(NodeType(*t_ca));
    auto t_shape = t_sty->GetShape();
    auto t_rank = t_shape.Rank();

    // common limitation (currently guarded by memcheck)
    for (auto& sty : {GetSpannedType(GetSymbolType(f_name)),
                      GetSpannedType(GetSymbolType(t_name))}) {
      if (sty->RuntimeShaped()) {
        auto bs = sty->ByteSizeValue();
        if (!IsComputable(bs)) continue; // TODO: dst shape should be computable
        constexpr size_t limit = 1ULL << 32;
        auto msg = "The size of data transferred by DMA cannot exceed 2^32.";
        auto asrt = sbe::oc_lt(bs, sbe::nu(limit))->Normalize();
        Assess(asrt, msg, n, &n);
      } else {
        if (sty->ByteSize() >= (1ULL << 32))
          Error1(n.LOC(), "On " + cur_arch +
                              ", the size of data transferred by "
                              "DMA cannot exceed 2^32.");
      }
    }
    auto IsLinearCopy = [&]() -> bool {
      return f_ca->NoTilingOperation() && t_ca->NoTilingOperation();
    };
    auto IsSlice = [&]() -> bool {
      return f_ca->HasTilingOperation() && t_ca->NoTilingOperation();
    };
    auto IsDeslice = [&]() -> bool {
      return f_ca->NoTilingOperation() && t_ca->HasTilingOperation();
    };
    auto RankLE5 = [&](const std::string& dma_op) {
      if (f_rank > 5)
        Error1(n.LOC(), "On " + cur_arch + ", the rank in " + dma_op +
                            " must be in range [1, 5], but got " +
                            std::to_string(f_rank) + ".");
    };

    auto msb_is_zero = [](const ptr<AST::SpannedOperation>& so) {
      if (auto indices = so->GetIndices()) {
        auto val = indices->Opts().GetVals()[0];
        if (VIIsInt(val) && !IsValueItemEqual(0, val)) return false;
      }
      if (auto tf = so->GetTilingFactors()) {
        auto val = tf->Opts().GetVals()[0];
        if (VIIsInt(val) && !IsValueItemEqual(1, val)) return false;
      }
      return true;
    };

    if (CCtx().GetArch() == "gcu300" ||
        CCtx().GetArch() == "gcu400") { // TODO: check this for GCU400
      // linear copy
      // omitted

      // transpose
      if (n.operation == ".transp" && IsLinearCopy()) {
        RankLE5("dma.transp(not slice nor deslice)");
        for (size_t idx = 1; idx < f_rank; ++idx)
          CheckDimSize(f_shape, idx, "<", 1 << 24, *n.GetFrom(), &n);
        for (size_t idx = 1; idx < t_rank; ++idx)
          CheckDimSize(t_shape, idx, "<", 1 << 24, *n.GetTo(), &n);
        auto bpe = sbe::nu((int)(SizeOf(f_sty->e_type)));
        auto value = (f_shape.ValueAt(0) * bpe + sbe::nu(127)) / sbe::nu(128) *
                     sbe::nu(128);
        CheckValue(value, "<", 1 << 24, *n.GetFrom(), &n,
                   "CeilTo128Byte(src_dim0_size * bpe) < 2^24.");
        value = (t_shape.ValueAt(0) * bpe + sbe::nu(127)) / sbe::nu(128) *
                sbe::nu(128);
        CheckValue(value, "<", 1 << 24, *n.GetTo(), &n,
                   "CeilTo128Byte(dst_dim0_size * bpe) < 2^24.");
        for (size_t idx = 1; idx < t_rank; ++idx)
          value = value * t_shape.ValueAt(idx);
        CheckValue(value, "<", 1ULL << 32, *n.GetTo(), &n,
                   "CeilTo128Byte(bpe * dst dim0) * dim1 * dim2 "
                   "* dim3 * dim4 < 4GB.");
      }

      // pad
      if (n.operation == ".pad" && IsLinearCopy()) {
        RankLE5("dma.pad");
        auto pc = cast<PadConfig>(n.config);
        assert(f_rank == pc->pad_low->Count());

        for (const auto& mv : {pc->pad_low, pc->pad_high}) {
          for (auto v : mv->AllValues()) {
            auto e = cast<AST::Expr>(v);
            if (!e->Opts().HasVal()) continue;
            auto val = e->Opts().GetVal();
            if (VIIsInt(val)) {
              if (sbe::clt(val, sbe::nu(0)) || sbe::cgt(val, sbe::nu(1 << 11)))
                Error1(e->LOC(), "On GCU300, the config in "
                                 "dma.pad must be in range [0, 2^11].");
            } else {
              auto msg = "On GCU300, the config in "
                         "dma.pad must be in range [0, 2^11]";
              auto asrt = sbe::cmp(">=", val, sbe::nu(0));
              Assess(asrt, msg, *e, &n);
              asrt = sbe::cmp("<=", val, sbe::nu(1 << 11));
              Assess(asrt, msg, *e, &n);
            }
          }
        }
        // padding_mid
        for (size_t idx = 0; idx < f_rank; ++idx) {
          auto e = cast<AST::Expr>(pc->pad_mid->ValueAt(idx));
          if (!e->Opts().HasVal()) continue;
          auto val = e->Opts().GetVal();
          if (VIIsInt(val)) {
            if (idx == f_rank - 1 && sbe::cne(val, sbe::nu(0)))
              Error1(e->LOC(),
                     "On " + cur_arch +
                         ", the value of padding_mid[rank-1] in dma.pad must "
                         "be 0 (mid padding of dim[rank-1] is not supported by "
                         "the hardware).");
            else if (sbe::cgt(val, sbe::nu(1 << 10)))
              Error1(e->LOC(), "On " + cur_arch +
                                   ", the value of padding_mid in dma.pad must "
                                   "be in range [0, 2^10].");
          } else {
            if (idx == f_rank - 1) {
              auto asrt = sbe::cmp("==", val, sbe::nu(0));
              Assess(
                  asrt,
                  "On " + cur_arch +
                      ", the value of padding_mid[rank-1] in dma.pad must be 0 "
                      "(mid padding of dim[rank-1] is not supported by the "
                      "hardware)",
                  *e, &n);
            } else {
              auto asrt = sbe::cmp("<=", val, sbe::nu(1 << 10));
              Assess(
                  asrt,
                  "On " + cur_arch +
                      ", the value of padding_mid in dma.pad must be in range "
                      "[0, 2^10]",
                  *e, &n);
            }
          }
        }
        if (f_rank == 5) {
          for (const auto& mv : {pc->pad_low, pc->pad_high, pc->pad_mid}) {
            auto v = mv->ValueAt(0);
            auto e = cast<AST::Expr>(v);
            if (!e->Opts().HasVal()) continue;
            auto val = e->Opts().GetVal();
            if (VIIsInt(val)) {
              if (sbe::cne(val, sbe::nu(0)))
                Error1(e->LOC(),
                       "On " + cur_arch +
                           ", dma.pad does not support 5-dimensional "
                           "array (if dim is 5, pad_config[0] must be 0).");
            } else {
              auto asrt = sbe::cmp("==", val, sbe::nu(0));
              Assess(
                  asrt,
                  "On " + cur_arch +
                      ", dma.pad does not support 5-dimensional array (if dim "
                      "is 5, pad_config[0] must be 0)",
                  *e, &n);
            }
          }
        }
      }
      // slice
      if (n.operation == ".copy" && IsSlice()) {
        RankLE5("dma.copy(slice)");
        for (size_t idx = 0; idx < f_rank; ++idx)
          CheckDimSize(f_shape, idx, "<", 1 << 24, *n.GetFrom(), &n);
        for (size_t idx = 0; idx < t_rank; ++idx)
          CheckDimSize(t_shape, idx, "<", 1 << 24, *n.GetTo(), &n);
        // TODO: offset limitation: [0, 2^24)
        if (f_rank == 5) {
          for (auto so : f_ca->AllOperations()) {
            if (!msb_is_zero(so))
              Error1(n.LOC(), "On " + cur_arch +
                                  ", dma.copy(slice) does not "
                                  "support 5-dimensional "
                                  "array (if dim is 5, offsets[0] must be 0).");
          }
        }
        // TODO: check for auto padding
      }

      // deslice
      if (n.operation == ".copy" && IsDeslice()) {
        RankLE5("dma.copy(deslice)");
        for (size_t idx = 0; idx < f_rank; ++idx)
          CheckDimSize(f_shape, idx, "<", 1 << 24, *n.GetFrom(), &n);
        for (size_t idx = 0; idx < t_rank; ++idx)
          CheckDimSize(t_shape, idx, "<", 1 << 24, *n.GetTo(), &n);
        // TODO: offset limitation: [0, 2^24)
        if (t_rank == 5) {
          for (auto so : t_ca->AllOperations()) {
            if (!msb_is_zero(so))
              Error1(n.LOC(), "On " + cur_arch +
                                  ", dma.copy(deslice) does not "
                                  "support 5-dimensional "
                                  "array (if dim is 5, offsets[0] must be 0).");
          }
        }
      }

      // slice transpose
      if (n.operation == ".transp" && IsSlice()) {
        RankLE5("dma.transp(slice then transpose)");
        for (size_t idx = 1; idx < f_rank; ++idx)
          CheckDimSize(f_shape, idx, "<", 1 << 24, *n.GetFrom(), &n);
        for (size_t idx = 1; idx < t_rank; ++idx)
          CheckDimSize(t_shape, idx, "<", 1 << 24, *n.GetTo(), &n);
        auto bpe = sbe::nu(SizeOf(f_sty->e_type));
        auto value = (f_shape.ValueAt(0) * bpe + sbe::nu(127)) / sbe::nu(128) *
                     sbe::nu(128);
        CheckValue(value, "<", 1 << 24, *n.GetFrom(), &n,
                   "CeilTo128Byte(src_dim0_size * bpe) < 2^24.");
        value = (t_shape.ValueAt(0) * bpe + sbe::nu(127)) / sbe::nu(128) *
                sbe::nu(128);
        CheckValue(value, "<", 1 << 24, *n.GetTo(), &n,
                   "CeilTo128Byte(dst_dim0_size * bpe) < 2^24.");
        for (size_t idx = 1; idx < t_rank; ++idx)
          value = value * t_shape.ValueAt(idx);
        CheckValue(value, "<", 1ULL << 32, *n.GetTo(), &n,
                   "CeilTo128Byte(bpe * dst dim0) * dim1 * dim2 "
                   "* dim3 * dim4 < 4GB.");
      }

      // transpose deslice
      if (n.operation == ".transp" && IsDeslice()) {
        RankLE5("dma.transp(transpose then deslice)");
        for (size_t idx = 1; idx < f_rank; ++idx)
          CheckDimSize(f_shape, idx, "<", 1 << 24, *n.GetFrom(), &n);
        for (size_t idx = 1; idx < t_rank; ++idx)
          CheckDimSize(t_shape, idx, "<", 1 << 24, *n.GetTo(), &n);
        auto bpe = sbe::nu((int)(SizeOf(f_sty->e_type)));
        auto value = (f_shape.ValueAt(0) * bpe + sbe::nu(127)) / sbe::nu(128) *
                     sbe::nu(128);
        CheckValue(value, "<", 1 << 24, *n.GetFrom(), &n,
                   "CeilTo128Byte(src_dim0_size * bpe) < 2^24.");
        value = (t_shape.ValueAt(0) * bpe + sbe::nu(127)) / sbe::nu(128) *
                sbe::nu(128);
        CheckValue(value, "<", 1 << 24, *n.GetTo(), &n,
                   "CeilTo128Byte(dst_dim0_size * bpe) < 2^24.");
        for (size_t idx = 1; idx < t_rank; ++idx)
          value = value * t_shape.ValueAt(idx);
        CheckValue(value, "<", 1ULL << 32, *n.GetTo(), &n,
                   "CeilTo128Byte(bpe * dst dim0) * dim1 * dim2 "
                   "* dim3 * dim4 < 4GB.");
      }

      return;
    }

    if (CCtx().GetArch() == "gcu200" || CCtx().GetArch() == "gcu210") {
      // linear copy
      // omitted

      // transpose
      if (n.operation == ".transp" && IsLinearCopy()) {
        RankLE5("dma.transp(not slice nor deslice)");
        if (f_rank == 5) {
          CheckDimSize(f_shape, 0, "<", 1 << 24, *n.GetFrom(), &n);
          for (size_t idx = 1; idx < f_rank; ++idx)
            CheckDimSize(f_shape, idx, "<", 1 << 16, *n.GetFrom(), &n);
        } else {
          for (size_t idx = 0; idx < f_rank; ++idx)
            CheckDimSize(f_shape, idx, "<", 1 << 16, *n.GetFrom(), &n);
        }

        if (SizeOf(t_sty->e_type) == 4) {
          for (size_t idx = 0; idx < t_rank; ++idx) {
            CheckDimSize(t_shape, idx, ">", 1, *n.GetTo(), &n);
            CheckDimSize(t_shape, idx, "<", (1 << 16) - 32, *n.GetTo(), &n);
          }
        }

        auto tc = cast<TransposeConfig>(n.config);
        if (f_rank == 5 && tc->dim_values[0] != 0)
          Error1(
              n.LOC(),
              "On " + cur_arch +
                  ", dma.transp(not slice nor deslice) does not "
                  "support 5-dimensional array (if dim is 5, layout[0] must be "
                  "0).");
      }

      // pad
      if (n.operation == ".pad" && IsLinearCopy()) {
        RankLE5("dma.pad");
        if (f_rank == 5) {
          CheckDimSize(f_shape, 0, "<", 1 << 24, *n.GetFrom(), &n);
          for (size_t idx = 1; idx < f_rank; ++idx)
            CheckDimSize(f_shape, idx, "<", 1 << 16, *n.GetFrom(), &n);
        } else {
          for (size_t idx = 0; idx < f_rank; ++idx)
            CheckDimSize(f_shape, idx, "<", 1 << 16, *n.GetFrom(), &n);
        }

        // shape of n.to is the same as n.from's
        // so the check of n.to is omitted
        auto pc = cast<PadConfig>(n.config);
        assert(f_rank == pc->pad_low->Count());

        for (const auto& mv : {pc->pad_low, pc->pad_high}) {
          for (auto v : mv->AllValues()) {
            auto e = cast<AST::Expr>(v);
            if (!e->Opts().HasVal()) continue;
            auto val = e->Opts().GetVal();
            if (VIIsInt(val)) {
              if (sbe::clt(val, sbe::nu(0)) || sbe::cgt(val, sbe::nu(1 << 11)))
                Error1(
                    e->LOC(),
                    "On " + cur_arch +
                        ", the config in dma.pad must be in range [0, 2^11].");
            } else {
              auto msg = "On " + cur_arch +
                         ", the config in dma.pad must be in range [0, 2^11]";
              Assess(sbe::cmp(">=", val, sbe::nu(0)), msg, *e, &n);
              Assess(sbe::cmp("<=", val, sbe::nu(1 << 11)), msg, *e, &n);
            }
          }
        }

        // padding_mid
        for (size_t idx = 0; idx < f_rank; ++idx) {
          auto e = cast<AST::Expr>(pc->pad_mid->ValueAt(idx));
          if (!e->Opts().HasVal()) continue;
          auto val = e->Opts().GetVal();
          if (VIIsInt(val)) {
            if (idx == f_rank - 1 && sbe::cne(val, sbe::nu(0)))
              Error1(e->LOC(),
                     "On " + cur_arch +
                         ", the value of padding_mid[rank-1] in dma.pad must "
                         "be 0 (mid padding of dim[rank-1] is not supported by "
                         "the hardware).");
            else if (sbe::cgt(val, sbe::nu(1 << 10)))
              Error1(e->LOC(), "On " + cur_arch +
                                   ", the value of padding_mid in dma.pad must "
                                   "be in range [0, 2^10].");
          } else {
            if (idx == f_rank - 1) {
              auto asrt = sbe::cmp("==", val, sbe::nu(0));
              Assess(
                  asrt,
                  "On " + cur_arch +
                      ", the value of padding_mid[rank-1] in dma.pad must be 0 "
                      "(mid padding of dim[rank-1] is not supported by the "
                      "hardware)",
                  *e, &n);
            } else {
              auto asrt = sbe::cmp("<=", val, sbe::nu(1 << 10));
              Assess(
                  asrt,
                  "On " + cur_arch +
                      ", the value of padding_mid in dma.pad must be in range "
                      "[0, 2^10]",
                  *e, &n);
            }
          }
        }
        if (f_rank == 5) {
          for (const auto& mv : {pc->pad_low, pc->pad_high, pc->pad_mid}) {
            auto v = mv->ValueAt(0);
            auto e = cast<AST::Expr>(v);
            if (!e->Opts().HasVal()) continue;
            auto val = e->Opts().GetVal();
            if (VIIsInt(val)) {
              if (sbe::cne(val, sbe::nu(0)))
                Error1(e->LOC(),
                       "On " + cur_arch +
                           ", dma.pad does not support 5-dimensional "
                           "array (if dim is 5, pad_config[0] must be 0).");
            } else {
              auto asrt = sbe::cmp("==", val, sbe::nu(0));
              Assess(
                  asrt,
                  "On " + cur_arch +
                      ", dma.pad does not support 5-dimensional array (if dim "
                      "is 5, pad_config[0] must be 0)",
                  *e);
            }
          }
        }
      }

      // slice
      if (n.operation == ".copy" && IsSlice()) {
        RankLE5("dma.copy(slice)");
        for (size_t idx = 0; idx < f_rank; ++idx)
          CheckDimSize(f_shape, idx, "<", 1 << 24, *n.GetFrom(), &n);

        for (size_t idx = 0; idx < t_rank; ++idx)
          CheckDimSize(t_shape, idx, "<", 1 << 16, *n.GetTo(), &n);
        if (f_rank == 5) {
          for (auto so : f_ca->AllOperations()) {
            if (!msb_is_zero(so))
              Error1(n.LOC(), "On " + cur_arch +
                                  ", dma.copy(slice) does not "
                                  "support 5-dimensional "
                                  "array (if dim is 5, offsets[0] must be 0).");
          }
        }
      }

      // deslice
      if (n.operation == ".copy" && IsDeslice()) {
        RankLE5("dma.copy(deslice)");
        for (size_t idx = 0; idx < f_rank; ++idx)
          CheckDimSize(f_shape, idx, "<", 1 << 16, *n.GetFrom(), &n);

        for (size_t idx = 0; idx < t_rank; ++idx)
          CheckDimSize(t_shape, idx, "<", 1 << 24, *n.GetTo(), &n);
      }

      // slice transpose
      if (n.operation == ".transp" && IsSlice()) {
        RankLE5("dma.transp(slice then transpose)");
        if (f_rank == 5) {
          CheckDimSize(f_shape, 0, "<", 1 << 24, *n.GetFrom(), &n);
          for (size_t idx = 1; idx < f_rank; ++idx)
            CheckDimSize(f_shape, idx, "<", 1 << 16, *n.GetFrom(), &n);
        } else {
          for (size_t idx = 0; idx < f_rank; ++idx)
            CheckDimSize(f_shape, idx, "<", 1 << 16, *n.GetFrom(), &n);
        }

        if (SizeOf(t_sty->e_type) == 4) {
          for (size_t idx = 0; idx < t_rank; ++idx) {
            CheckDimSize(t_shape, idx, ">", 1, *n.GetTo(), &n);
            CheckDimSize(t_shape, idx, "<", (1 << 16) - 32, *n.GetTo(), &n);
          }
        }

        if (f_rank == 5) {
          for (auto so : f_ca->AllOperations()) {
            if (!msb_is_zero(so))
              Error1(n.LOC(), "On " + cur_arch +
                                  ", dma.transp(slice then "
                                  "transpose) does not support 5-dimensional "
                                  "array (if dim is 5, offsets[0] must be 0).");
          }
        }

        auto tc = cast<TransposeConfig>(n.config);
        if (f_rank == 5 && tc->dim_values[0] != 0)
          Error1(n.LOC(), "On " + cur_arch +
                              ", dma.transp(slice then transpose) "
                              "does not support 5-dimensional "
                              "array (if dim is 5, layout[0] must be 0).");
      }

      // transpose deslice
      if (n.operation == ".transp" && IsDeslice()) {
        RankLE5("dma.transp(transpose then deslice)");
        if (f_rank == 5) {
          CheckDimSize(f_shape, 0, "<", 1 << 24, *n.GetFrom(), &n);
          for (size_t idx = 1; idx < f_rank; ++idx)
            CheckDimSize(f_shape, idx, "<", 1 << 16, *n.GetFrom(), &n);
        } else {
          for (size_t idx = 0; idx < f_rank; ++idx)
            CheckDimSize(f_shape, idx, "<", 1 << 16, *n.GetFrom(), &n);
        }

        if (SizeOf(t_sty->e_type) == 4) {
          for (size_t idx = 0; idx < t_rank; ++idx) {
            CheckDimSize(t_shape, idx, ">", 1, *n.GetTo(), &n);
            CheckDimSize(t_shape, idx, "<", (1 << 16) - 32, *n.GetTo(), &n);
          }
        }

        if (t_rank == 5) {
          for (auto so : t_ca->AllOperations()) {
            if (!msb_is_zero(so))
              Error1(n.LOC(), "On " + cur_arch +
                                  ", dma.transp(transpose then "
                                  "deslice) does not support 5-dimensional "
                                  "array (if dim is 5, offsets[0] must be 0).");
          }
        }

        auto tc = cast<TransposeConfig>(n.config);
        if (t_rank == 5 && tc->dim_values[0] != 0)
          Error1(n.LOC(), "On " + cur_arch +
                              ", dma.transp(transpose then deslice) "
                              "does not support 5-dimensional "
                              "array (if dim is 5, layout[0] must be 0).");
      }
      return;
    }

    choreo_unreachable("Unsupported target architecture: " + cur_arch);
  }

  void CheckDimSize(const Shape& s, size_t idx, const std::string& op,
                    size_t limit, AST::Node& n, AST::Node* en) {
    assert(idx < s.Rank());
    std::string message = "the " + Ordinal(idx + 1) + " dim " +
                          s.ValueAt(idx)->ToString() + " " + op + " " +
                          std::to_string(limit) + ".";

    CheckValue(s.ValueAt(idx), op, limit, n, en, message);
  }

  void CheckValue(const ValueItem& vi, const std::string& op, size_t limit,
                  AST::Node& n, AST::Node* en, std::string message = "") {
    if (!IsComputable(vi)) {
      VST_DEBUG(dbgs() << "[GCUCHECK] Not Checking " << STR(vi) << " " << op
                       << " " << limit << ".\n");
      return;
    }

    message = "On " + cur_arch + ", must satisfy: " + message;
    Assess(sbe::cmp(op, vi, sbe::nu(limit)), message, n, en);
  }

  // Check that the total bound of a parallel-by node at a given level does
  // not exceed the architecture limit (from GetMaxParallelByCount).  If the
  // bound is a compile-time constant the check is static; if it is symbolic a
  // runtime assertion is emitted.
  void CheckPBRange(AST::ParallelBy& pb) {
    auto lvl = pb.GetLevel();
    auto max_count = CCtx().GetMaxParallelByCount(lvl);
    if (max_count == 0) return; // level unconstrained on this architecture

    // Compute total bound as product of all component bounds.
    // For a simple `parallel p by N`, cmpt_bounds is empty and we fall back
    // to BoundValue() which holds the merged expression directly.
    // For `parallel {x,y} by [M,N]`, cmpt_bounds has the originals with
    // Opts set; BoundValue() would use a synthetic clone without Opts.
    ValueItem total;
    auto bvs = pb.BoundValues();
    if (!bvs.empty()) {
      total = sbe::nu(1);
      for (auto bv : bvs) total = (total * bv)->Normalize();
    } else {
      total = pb.BoundValue();
    }
    if (!IsValidValueItem(total)) return; // bound not yet inferred

    auto pred = sbe::cmp("<=", total, sbe::nu((int64_t)max_count))->Normalize();
    // Skip if trivially safe -- avoids cluttering --show-assess report.
    if (auto bv = VIBool(pred); bv && bv.value()) return;
    auto msg = "On " + cur_arch + ", the total " + STR(lvl) +
               " parallel-by count must not exceed " +
               std::to_string(max_count) + ".";
    Assess(pred, msg, pb, &pb);
  }

public:
  GCUAdaptor()
      : VisitorWithSymTab("gcu"), cur_arch(ToUpper(CCtx().GetArch())) {}
  ~GCUAdaptor() {}

  bool Visit(AST::FloatLiteral& n) override {
    TraceEachVisit(n);

    return true;
  }

  bool Visit(AST::NamedVariableDecl& n) override {
    TraceEachVisit(n);
    auto ty = GetSymbolType(n.name_str);
    if (isa<EventType>(ty) || isa<EventArrayType>(ty))
      if (!CCtx().TargetSupportEvent())
        Error1(n.LOC(), "Event is not supported on " + cur_arch + ".");

    if (isa<AST::Select>(n.init_expr))
      if (IsHost()) Error1(n.LOC(), "select in host is not supported.");

    if (!isa<SpannedType>(ty)) {
      auto mem = n.GetMemory();
      if (!mem) return true;

      auto st = mem->Get();
      if (n.init_expr != nullptr) {
        if (st == Storage::SHARED || st == Storage::LOCAL)
          Error1(n.LOC(), "initialization is not supported for " + STR(st) +
                              " variables");

        if (st == Storage::GLOBAL)
          Error1(n.LOC(), "'global' attribute only applies to functions ");
      }
      return true;
    }

    auto sty = cast<SpannedType>(ty);

    auto st = sty->GetStorage();

    if (auto e = dyn_cast<AST::Expr>(n.init_expr))
      if (isa<AST::SpanAs>(e->GetReference()))
        if (st == Storage::GLOBAL)
          Error1(n.LOC(), "declare reference by span_as cannot be applied to "
                          "global variable.");

    switch (st) {
    case Storage::GLOBAL:
      if (Level() != ParallelLevel::SEQ && Level() != ParallelLevel::DEVICE)
        Error1(n.LOC(), "global variable '" + n.name_str +
                            "` mustn't be declared inside parallel-by.");
      break;
    case Storage::SHARED:
      if (Level() == ParallelLevel::SEQ)
        Error1(n.LOC(), "shared variable '" + n.name_str +
                            "` must be declared inside parallel-by.");
      else if (Level() == ParallelLevel::DEVICE)
        Error1(n.LOC(), "shared variable '" + n.name_str +
                            "` must be declared inside a kernel parallel-by "
                            "(': block' or inner), not at device scope.");
      if (sty->RuntimeShaped() && !CCtx().MemReuse())
        Error1(n.LOC(), "GCU forbids shared variable '" + n.name_str +
                            "` to be dynamically shaped (by " +
                            STR(sty->GetShape()) + ").");
      break;
    case Storage::LOCAL:
      if (Level() == ParallelLevel::SEQ)
        Error1(n.LOC(), "local variable '" + n.name_str +
                            "` must be declared inside parallel-by.");
      else if (Level() == ParallelLevel::DEVICE)
        Error1(n.LOC(), "local variable '" + n.name_str +
                            "` must be declared inside a kernel parallel-by "
                            "(': block' or inner), not at device scope.");
      if (sty->RuntimeShaped() && !CCtx().MemReuse())
        Error1(n.LOC(), "GCU forbids local variable '" + n.name_str +
                            "` to be dynamically shaped (by " +
                            STR(sty->GetShape()) + ").");
      break;
    default:
      Error1(n.LOC(), "can not declare variable '" + n.name_str + "` as " +
                          STR(st) + " inside choreo function.");
      break;
    }
    return true;
  }

  bool Visit(AST::DataAccess& n) override {
    TraceEachVisit(n);
    if ((CCtx().GetArch() == "gcu200" || CCtx().GetArch() == "gcu210" ||
         (CCtx().GetArch() == "gcu300" && !IsInCooperativeBlock())) &&
        n.indices != nullptr) {
      if (auto sty = GetSpannedType(GetSymbolType(n.data->name)))
        if (sty->GetStorage() == Storage::GLOBAL ||
            cur_params.count(InScopeName(n.data->name)))
          Error1(n.LOC(),
                 "global data access '" + STR(n) + "` is not allowed.");
    }
    return true;
  }
  bool Visit(AST::Parameter& n) override {
    TraceEachVisit(n);
    if (n.sym) cur_params.emplace(InScopeName(n.sym->name), &n);
    return true;
  }
  bool Visit(AST::ParallelBy& n) override {
    TraceEachVisit(n);
    CheckPBRange(n);
    CheckStreamBinding(&n);
    return true;
  }

  void CheckStreamBinding(AST::ParallelBy* pb) {
    if (!pb->HasStream()) return;
    auto lvl = pb->GetLevel();
    if (lvl != ParallelLevel::BLOCK) {
      Error1(pb->StreamExpr()->LOC(),
             "stream binding is only allowed on block-level parallel; '" +
                 STR(lvl) + "' level cannot have its own stream.");
      return;
    }
    auto se = pb->StreamExpr();
    if (!se->IsReference()) {
      Error1(se->LOC(),
             "stream binding in parallel<...> must be a stream variable.");
      return;
    }
    auto id = dyn_cast<AST::Identifier>(se->GetR().get());
    if (!id) {
      Error1(se->LOC(),
             "stream binding in parallel<...> must be a stream variable.");
      return;
    }
    auto ty = GetSymbolType(id->name);
    if (!isa<StreamType>(ty))
      Error1(se->LOC(), "'" + id->name +
                            "' is not a stream type; parallel<...> requires "
                            "a stream variable.");
  }

  bool Visit(AST::DMA& n) override {
    TraceEachVisit(n);

    // GCU2 does not support slice+pad (dma.pad with chunkAt source).
    // slice+transpose and transpose+deslice work fine on GCU2.
    if ((CCtx().GetArch() == "gcu200" || CCtx().GetArch() == "gcu210") &&
        n.operation == ".pad" && isa<AST::ChunkAt>(n.from))
      Error1(n.LOC(), "slice+pad DMA (dma.pad with chunkAt source) is not "
                      "supported on GCU2.");

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

  bool Visit(AST::Call& n) override {
    TraceEachVisit(n);
    if (n.IsArith() &&
        (CCtx().GetArch() == "gcu200" || CCtx().GetArch() == "gcu210"))
      Error1(n.LOC(), "Arithmetic built-in function is not supported on GCU2.");

    if (CCtx().GetArch() == "gcu300" && n.function->name != "print" &&
        n.function->name != "println") {
      for (auto& arg : n.GetArguments()) {
        if (auto sty = GetSpannedType(arg->GetType()))
          if (sty->GetStorage() == Storage::GLOBAL)
            Error1(n.LOC(), "function call '" + n.function->name +
                                "` with global data '" + STR(arg) +
                                "` is not allowed.");
        if (auto id = AST::GetIdentifier(arg))
          if (cur_params.count(InScopeName(STR(id))) &&
              isa<SpannedType>(GetSymbolType(id->name)))
            Error1(n.LOC(), "function call '" + n.function->name +
                                "` with global data '" + STR(arg) +
                                "` is not allowed.");
      }
    }

    // Validate acore:: library calls
    if (!n.IsBIF() && PrefixedWith(n.function->name, "acore::")) {
      auto error_fn = [this](const auto& loc, const std::string& msg) {
        Error1(loc, msg);
      };
      CheckAcoreCall(n, error_fn, CCtx().GetArch());
    }

    // Validate __lib_* calls through the target interface
    if (n.IsLibCall()) {
      auto error_fn = [this](const auto& loc, const std::string& msg) {
        Error1(loc, msg);
      };
      auto warn_fn = [this](const auto& loc, const std::string& msg) {
        Warning(loc, msg);
      };
      auto assess_fn = [this](auto expr, const std::string& msg,
                              AST::Node& node) { Assess(expr, msg, node); };
      ValidateLibCall(n, error_fn, warn_fn, assess_fn, CCtx().UseTargetLib());
    }

    return true;
  }

  bool Visit(AST::Synchronize& n) override {
    TraceEachVisit(n);

    switch (n.Resource()) {
    case Storage::GLOBAL:
      if (Level() != ParallelLevel::SEQ && Level() != ParallelLevel::DEVICE)
        Error1(n.LOC(), "unsupported: " + STR(n.Resource()) +
                            " synchronization in " + STR(Level()) + " scope.");
      break;
    case Storage::SHARED:
      if (Level() == ParallelLevel::SEQ || Level() == ParallelLevel::DEVICE)
        Error1(n.LOC(), "unsupported: " + STR(n.Resource()) +
                            " synchronization in " + STR(Level()) + " scope.");
      break;
    case Storage::LOCAL:
      if (!TargetHasLevel(ParallelLevel::GROUP))
        Error1(n.LOC(), ToUpper(CCtx().GetArch()) + " does not support " +
                            STR(n.Resource()) + " synchronization.");
      else if (Level() == ParallelLevel::SEQ ||
               Level() == ParallelLevel::DEVICE ||
               Level() == ParallelLevel::BLOCK)
        Error1(n.LOC(), "unsupported: " + STR(n.Resource()) +
                            " synchronization in " + STR(Level()) + " scope.");
      break;
    default:
      Error1(n.LOC(),
             "unsupported synchronization: " + STR(n.Resource()) + ".");
    }
    return true;
  }

  bool Visit(AST::MMA& n) override {
    TraceEachVisit(n);
    if (!CCtx().TargetSupportMMAUKernel()) return true;

    auto& op = *n.GetOperation();

    switch (op.Tag()) {
    case AST::MMAOperation::Fill: break;

    case AST::MMAOperation::Load: {
      if (op.IsAsync())
        Error1(n.LOC(), "async MMA load is not supported on GCU target.");
      if (op.GetSwizzleMode() != SwizMode::NONE)
        Error1(n.LOC(), "swizzled MMA load is not supported on GCU target.");
      auto src = op.LoadFrom();
      if (src) {
        auto src_ty = src->GetType();
        if (src_ty) {
          if (auto sty = dyn_cast<SpannedType>(src_ty)) {
            auto sto = sty->GetStorage();
            if (sto != Storage::LOCAL && sto != Storage::NONE)
              Error1(n.LOC(),
                     "MMA load on GCU requires source in LOCAL (L1) "
                     "storage, but found " +
                         STR(sto) +
                         ". Use dma.copy to move data to local first.");
          }
        }
      }
    } break;

    case AST::MMAOperation::LoadR: {
      Error1(n.LOC(), "MMA register load is not supported on GCU target.");
    } break;

    case AST::MMAOperation::Exec: {
      auto& a_sym = AST::FragName(op.ExecOperand(1));
      auto& b_sym = AST::FragName(op.ExecOperand(2));
      auto& c_sym = AST::FragName(op.ExecOperand(0));
      auto a_sty = GetSpannedType(op.ExecOperand(1)->GetType());
      auto b_sty = GetSpannedType(op.ExecOperand(2)->GetType());
      auto c_sty = GetSpannedType(op.ExecOperand(0)->GetType());

      if (!a_sty || !b_sty)
        Error1(n.LOC(), "MMA exec operands must have spanned types.");

      auto a_ty = a_sty->ElementType();
      auto method = op.GetMethod();

      if (method != AST::MMAOperation::ROW_COL &&
          method != AST::MMAOperation::ROW_ROW)
        Error1(n.LOC(), "GCU acore only supports row.col (MK_KN) and "
                        "row.row (MK_NK) MMA layouts.");

      if (op.IsSparse())
        Error1(n.LOC(), "sparse MMA is not supported on GCU target.");

      if (op.HasScale())
        Error1(n.LOC(), "scaled MMA is not supported on GCU target.");

      auto a_shape = a_sty->GetShape();
      int static_M = -1;
      if (a_shape.Rank() >= 1) {
        auto m_vi = a_shape.Value()[0];
        if (auto mv = VIInt(m_vi)) static_M = (int)mv.value();
      }

      AcoreMMAConfig cfg;
      cfg.lhs_type = a_ty;
      cfg.rhs_type = b_sty->ElementType();
      cfg.acc_type = c_sty ? c_sty->ElementType() : BaseType::UNKNOWN;
      cfg.M = static_M > 0 ? static_M : 0;
      cfg.method = method;

      if (!IsValidAcoreMMAConfig(cfg))
        Error1(n.LOC(), "MMA config [" + AcoreMMAConfigStr(cfg) +
                            "] is not supported by acore.");

      FCtx(fname).SetFragMMAType(InScopeName(a_sym), MMAType::UKERNEL);
      FCtx(fname).SetFragMMAType(InScopeName(b_sym), MMAType::UKERNEL);
      FCtx(fname).SetFragMMAType(InScopeName(c_sym), MMAType::UKERNEL);
    } break;

    case AST::MMAOperation::Store: break;

    case AST::MMAOperation::Commit:
      choreo_unreachable("mma.commit should not reach GCU adaptor.");
      break;

    case AST::MMAOperation::Scale:
      Error1(n.LOC(), "mma.scale is not supported on GCU target.");
      break;

    default: break;
    }

    return true;
  }
};

} // end namespace Choreo

#endif // __CHOREO_GCU_ADAPT_HPP__
