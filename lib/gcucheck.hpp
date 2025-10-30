#ifndef __CHOREO_GCU_CHECK_INFO_HPP__
#define __CHOREO_GCU_CHECK_INFO_HPP__

// This apply the GCU target specific check and information annotation

#include "ast.hpp"
#include "visitor.hpp"

namespace Choreo {

extern int GetMaxParallelLevelFromNote(AST::ParallelBy& n);

inline size_t GCUVLdStAlignment(ptr<VectorType> vt) {
  if (CCtx().GetArch() != TargetArch::GCU4) return SizeOf(*vt);
  return 1;
}

struct GCUCheck : public VisitorWithSymTab {
private:
  std::unordered_map<std::string, AST::Parameter*> cur_params;
  std::vector<int> pl_depths;
  int pl_depth = 0;
  int max_pl_depth = 0;
  std::string cur_fname;
  std::string cur_arch;

private:
  bool BeforeVisitImpl(AST::Node& n) override {
    TraceEachVisit(n, "(pre)");
    if (auto cf = dyn_cast<AST::ChoreoFunction>(&n)) {
      cur_params.clear();
      cur_fname = cf->name;
    } else if (auto pb = dyn_cast<AST::ParallelBy>(&n)) {
      if (pb->GetLevel() == Storage::NONE) {
        pl_depth++;
      } else
        pl_depth = PDepth(pb->GetLevel()); // depth from annotation
      pl_depths.push_back(pl_depth);

      max_pl_depth = GetMaxParallelLevelFromNote(*pb);
      assert(max_pl_depth >= 2);

      if (!ValidDepth(*pb, pl_depth)) return false;
      if (!ValidDepth(*pb, max_pl_depth)) return false;

      pb->SetMaxLevel(PLevel(max_pl_depth));
      if (pb->GetLevel() == Storage::NONE) pb->SetLevel(PLevel(pl_depth));

      VST_DEBUG(pb->InlinePrint(dbgs());
                dbgs() << ": level << " << STR(pb->GetLevel()) << " / "
                       << STR(pb->GetMaxLevel()) << "\n");
    }
    return true;
  }

  bool AfterVisitImpl(AST::Node& n) override {
    TraceEachVisit(n, "(post)");
    if (auto pb = dyn_cast<AST::ParallelBy>(&n)) {
      std::string append_note = ":";
      //      assert(pb->GetLevel() != Storage::NONE);
      pl_depth = pl_depths.back();
      if (ValidDepth(*pb, pl_depth)) {
        if (CCtx().GetTarget() == CompileTarget::Topscc) {
          auto sto = PBLevel();
          // set the storage level when it is not explicitly set
          if (pb->GetLevel() == Storage::NONE) pb->SetLevel(sto);
          VST_DEBUG(dbgs() << "In function '" << fname << "': ";
                    pb->InlinePrint(dbgs());
                    dbgs() << ", level: " << STR(pb->GetLevel()) << " / "
                           << STR(pb->GetMaxLevel()) << "\n");
          append_note += STR(sto);
        } else
          append_note += std::to_string(max_pl_depth - pl_depth);
        auto pty = cast<BoundedITupleType>(NodeType(*pb->BPV()));
        pty->AppendNote(append_note);
        for (auto& symbol : pb->AllSubPVs())
          cast<BoundedITupleType>(NodeType(*symbol))->AppendNote(append_note);
      }
      pl_depths.pop_back();
      // pl_depth may not be increasing in a sequential manner
      // so need a container to store the state.
      pl_depth = pl_depths.back();
      assert(pl_depth >= 0 && "Unexpected parallel level");
      if (pb->IsOuter()) { max_pl_depth = 0; }
    }

    // mask stmts that are possible to be shared
    else if (auto c = dyn_cast<AST::Call>(&n))
      if (!c->IsExpr()) n.SetLevel(PLevel(pl_depth));

    return true;
  }

  void TraceEachVisit(AST::Node& n, std::string sup = "") {
    if (trace_visit) dbgs() << n.TypeNameString() << sup << "\n";
  }

private:
  Storage PLevel(int depth) {
    auto lvl = GCUDeviceParallelLevel(depth);
    assert(lvl != Storage::NONE);
    return lvl;
  }

  int PDepth(Storage l) {
    auto depth = GCUDeviceParallelDepth(l);
    assert(depth != -1);
    return depth;
  }

public:
  bool ValidDepth(const AST::Node& n, int depth) {
    if ((CCtx().GetArch() == TargetArch::GCU3 && depth >= 3) ||
        (CCtx().GetArch() == TargetArch::GCU4 && depth >= 4)) {
      Error1(n.LOC(), "the parallel level (depth: " + std::to_string(depth) +
                          ") is not supported by current GCU architecture (" +
                          cur_arch + ").");
      return false;
    }
    return true;
  }

  Storage PBLevel() {
    if (max_pl_depth == 1) {
      switch (pl_depth) {
      case 0: return Storage::GLOBAL; break;
      case 1: return Storage::LOCAL; break;
      default:
        choreo_unreachable(
            "unsupported parallel level: " + std::to_string(pl_depth) + ".");
      }
    } else
      return PLevel(pl_depth);
    return Storage::NONE;
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

    // consider reshapes that may change the rank
    if (f_ca->HasOperation()) {
      f_rank = f_ca->AllOperations().back()->GetRank();
      f_shape = f_ca->AllOperations().back()->GetBlockShape();
    }
    if (t_ca->HasOperation()) {
      t_rank = t_ca->AllOperations().back()->GetRank();
      t_shape = t_ca->AllOperations().back()->GetBlockShape();
    }

    // common limitation (currently guarded by memcheck)
    for (auto& sty : {f_sty, t_sty}) {
      if (sty->RuntimeShaped()) {
        auto bs = sty->ByteSizeValue();
        if (!IsComputable(bs)) continue; // TODO: dst shape should be computable
        constexpr size_t limit = 1ULL << 32;
        auto msg = "The size of data transferred by DMA cannot exceed 2^32.";
        auto asrt = sbe::oc_lt(bs, sbe::nu(limit))->Normalize();
        if (auto b = VIBool(asrt)) {
          if (b.value() == false) Error1(n.LOC(), msg);
        } else
          FCtx(cur_fname).InsertAssertion(asrt, n.LOC(), msg);
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
        auto bpe = sbe::nu((int)(SizeOf(f_sty->e_type)));
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
              FCtx(cur_fname).InsertAssertion(asrt, e->LOC(), msg);
              asrt = sbe::cmp("<=", val, sbe::nu(1 << 11));
              FCtx(cur_fname).InsertAssertion(asrt, e->LOC(), msg);
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
              FCtx(cur_fname).InsertAssertion(
                  asrt, e->LOC(),
                  "On " + cur_arch +
                      ", the value of padding_mid[rank-1] in dma.pad must be 0 "
                      "(mid padding of dim[rank-1] is not supported by the "
                      "hardware)");
            } else {
              auto asrt = sbe::cmp("<=", val, sbe::nu(1 << 10));
              FCtx(cur_fname).InsertAssertion(
                  asrt, e->LOC(),
                  "On " + cur_arch +
                      ", the value of padding_mid in dma.pad must be in range "
                      "[0, 2^10]");
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
              FCtx(cur_fname).InsertAssertion(
                  asrt, e->LOC(),
                  "On " + cur_arch +
                      ", dma.pad does not support 5-dimensional array (if dim "
                      "is 5, pad_config[0] must be 0)");
            }
          }
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
          for (auto tsi : f_ca->AllOperations()) {
            if (tsi->SpecifyReshape()) continue;
            auto first = tsi->Positions()->ValueAt(0);
            auto t = dyn_cast<BoundedITupleType>(first->GetType());
            assert(t != nullptr);
            if (VIIsInt(t->ubounds.ValueAt(0))) {
              if (!IsValueItemEqual(1, t->ubounds.ValueAt(0)))
                Error1(n.LOC(),
                       "On " + cur_arch +
                           ", dma.copy(slice) does not "
                           "support 5-dimensional "
                           "array (if dim is 5, offsets[0] must be 0).");
            } else {
              choreo_unreachable("unexpected situation");
              // TODO
              // Is that the case?
            }
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
          for (auto tsi : t_ca->AllOperations()) {
            if (tsi->SpecifyReshape()) continue;
            auto first = tsi->Positions()->ValueAt(0);
            auto t = dyn_cast<BoundedITupleType>(first->GetType());
            assert(t != nullptr);
            if (VIIsInt(t->ubounds.ValueAt(0))) {
              if (!IsValueItemEqual(1, t->ubounds.ValueAt(0)))
                Error1(n.LOC(),
                       "On " + cur_arch +
                           ", dma.copy(deslice) does not "
                           "support 5-dimensional "
                           "array (if dim is 5, offsets[0] must be 0).");
            } else {
              choreo_unreachable("unexpected situation");
              // TODO
              // Is that the case?
            }
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
        auto bpe = sbe::nu(SizeOf(f_sty->e_type));
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
        auto bpe = sbe::nu((int)(SizeOf(f_sty->e_type)));
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

        if (SizeOf(t_sty->e_type) == 4) {
          for (size_t idx = 0; idx < t_rank; ++idx) {
            CheckDimSize(t_shape, idx, ">", 1, n.to->LOC());
            CheckDimSize(t_shape, idx, "<", (1 << 16) - 32, n.to->LOC());
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
              auto asrt = sbe::cmp(">=", val, sbe::nu(0));
              FCtx(cur_fname).InsertAssertion(asrt, e->LOC(), msg);
              asrt = sbe::cmp("<=", val, sbe::nu(1 << 11));
              FCtx(cur_fname).InsertAssertion(asrt, e->LOC(), msg);
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
              FCtx(cur_fname).InsertAssertion(
                  asrt, e->LOC(),
                  "On " + cur_arch +
                      ", the value of padding_mid[rank-1] in dma.pad must be 0 "
                      "(mid padding of dim[rank-1] is not supported by the "
                      "hardware)");
            } else {
              auto asrt = sbe::cmp("<=", val, sbe::nu(1 << 10));
              FCtx(cur_fname).InsertAssertion(
                  asrt, e->LOC(),
                  "On " + cur_arch +
                      ", the value of padding_mid in dma.pad must be in range "
                      "[0, 2^10]");
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
              FCtx(cur_fname).InsertAssertion(
                  asrt, e->LOC(),
                  "On " + cur_arch +
                      ", dma.pad does not support 5-dimensional array (if dim "
                      "is 5, pad_config[0] must be 0)");
            }
          }
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
          for (auto tsi : f_ca->AllOperations()) {
            if (tsi->SpecifyReshape()) continue;
            auto first = tsi->Positions()->ValueAt(0);
            auto t = dyn_cast<BoundedITupleType>(first->GetType());
            assert(t != nullptr);
            if (VIIsInt(t->ubounds.ValueAt(0))) {
              if (!IsValueItemEqual(1, t->ubounds.ValueAt(0)))
                Error1(n.LOC(),
                       "On " + cur_arch +
                           ", dma.copy(slice) does not "
                           "support 5-dimensional "
                           "array (if dim is 5, offsets[0] must be 0).");
            } else {
              choreo_unreachable("unexpected situation");
              // TODO
              // Is that the case?
            }
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

        if (SizeOf(t_sty->e_type) == 4) {
          for (size_t idx = 0; idx < t_rank; ++idx) {
            CheckDimSize(t_shape, idx, ">", 1, n.to->LOC());
            CheckDimSize(t_shape, idx, "<", (1 << 16) - 32, n.to->LOC());
          }
        }

        if (f_rank == 5) {
          for (auto tsi : f_ca->AllOperations()) {
            if (tsi->SpecifyReshape()) continue;
            auto first = tsi->Positions()->ValueAt(0);
            auto t = dyn_cast<BoundedITupleType>(first->GetType());
            assert(t != nullptr);
            if (VIIsInt(t->ubounds.ValueAt(0))) {
              if (!IsValueItemEqual(1, t->ubounds.ValueAt(0)))
                Error1(n.LOC(),
                       "On " + cur_arch +
                           ", dma.transp(slice then "
                           "transpose) does not support 5-dimensional "
                           "array (if dim is 5, offsets[0] must be 0).");
            } else {
              // TODO
              // Is that the case?
            }
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
          CheckDimSize(f_shape, 0, "<", 1 << 24, n.from->LOC());
          for (size_t idx = 1; idx < f_rank; ++idx)
            CheckDimSize(f_shape, idx, "<", 1 << 16, n.from->LOC());
        } else {
          for (size_t idx = 0; idx < f_rank; ++idx)
            CheckDimSize(f_shape, idx, "<", 1 << 16, n.from->LOC());
        }

        if (SizeOf(t_sty->e_type) == 4) {
          for (size_t idx = 0; idx < t_rank; ++idx) {
            CheckDimSize(t_shape, idx, ">", 1, n.to->LOC());
            CheckDimSize(t_shape, idx, "<", (1 << 16) - 32, n.to->LOC());
          }
        }

        if (t_rank == 5) {
          for (auto tsi : t_ca->AllOperations()) {
            if (tsi->SpecifyReshape()) continue;
            auto first = tsi->Positions()->ValueAt(0);
            auto t = dyn_cast<BoundedITupleType>(first->GetType());
            assert(t != nullptr);
            if (VIIsInt(t->ubounds.ValueAt(0))) {
              if (!IsValueItemEqual(1, t->ubounds.ValueAt(0)))
                Error1(n.LOC(),
                       "On " + cur_arch +
                           ", dma.transp(transpose then "
                           "deslice) does not support 5-dimensional "
                           "array (if dim is 5, offsets[0] must be 0).");
            } else {
              // TODO
              // Is that the case?
            }
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
                    size_t limit, const location& loc) {
    assert(idx < s.Rank());
    std::string message = "the " + Ordinal(idx + 1) + " dim " +
                          s.ValueAt(idx)->ToString() + " " + op + " " +
                          std::to_string(limit) + ".";

    CheckValue(s.ValueAt(idx), op, limit, loc, message);
  }

  void CheckValue(const ValueItem& vi, const std::string& op, size_t limit,
                  const location& loc, std::string message = "") {
    if (!IsComputable(vi)) {
      VST_DEBUG(dbgs() << "[GCUCHECK] Not Checking " << STR(vi) << " " << op
                       << " " << limit << ".\n");
      return;
    }

    message = "On " + cur_arch + ", must satisfy: " + message;

    // try to evaluate at compilation time
    auto cmp = sbe::cmp(op, vi, sbe::nu(limit))->Normalize();
    if (auto tf = VIBool(cmp)) {
      if (tf && tf.value() == false) {
        VST_DEBUG(dbgs() << "[GCUCHECK] Generated check at " << loc << ": "
                         << vi->ToString() << " " << op << " "
                         << std::to_string(limit) + "ULL"
                         << "\n\twith message: " << message << "\n");
        Error1(loc, message);
      } else
        VST_DEBUG(dbgs() << "[GCUCHECK] Check at " << loc
                         << " is statisfied: " << vi->ToString() << " " << op
                         << " " << std::to_string(limit) + "ULL"
                         << "\n\twith message: " << message << "\n");
      return;
    }

    // or else, generate the runtime check
    VST_DEBUG(dbgs() << "[GCUCHECK] Generated runtime check at " << loc << ": "
                     << vi->ToString("ULL") << " " << op << " "
                     << std::to_string(limit) + "ULL"
                     << "\n\twith message: " << message << "\n");
    auto asrt = sbe::cmp(op, vi, sbe::nu(limit))->Normalize();
    if (auto b = VIBool(asrt)) {
      if (b.value() == false) Error1(loc, message);
    } else
      FCtx(cur_fname).InsertAssertion(asrt, loc, message);
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
      if (!n.IsFloat32())
        Error1(n.LOC(), "Factor backend in Choreo does not support " +
                            PSTR(n.GetType()) + " float-point number yet!");
    }

    return true;
  }
  bool Visit(AST::BoolLiteral& n) override {
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
          CCtx().GetArch() == TargetArch::GCU21)
        Error1(n.LOC(), "Event is not supported on " + cur_arch + ".");

    if (isa<AST::Select>(n.init_expr))
      if (IsHost() && CCtx().GetTarget() != CompileTarget::Factor)
        Error1(n.LOC(), "select in host is not supported.");

    if (!isa<SpannedType>(ty)) {
      auto mem = n.GetMemory();
      if (!mem) return true;

      auto st = mem->Get();
      if (n.init_expr != nullptr) {
        if (st == Storage::SHARED || st == Storage::LOCAL || st == Storage::SUB)
          Error1(n.LOC(), "initialization is not supported for " + STR(st) +
                              " variables");

        if (st == Storage::GLOBAL)
          Error1(n.LOC(), "'global' attribute only applies to functions ");
      }
      return true;
    }

    auto sty = cast<SpannedType>(ty);

    auto st = sty->GetStorage();
    switch (st) {
    case Storage::GLOBAL:
      if (pl_depth != 0)
        Error1(n.LOC(), "global variable '" + n.name_str +
                            "` mustn't be declared inside parallel-by.");
      break;
    case Storage::SHARED:
      if (pl_depth == 0)
        Error1(n.LOC(), "shared variable '" + n.name_str +
                            "` must be declared inside parallel-by.");
      if (sty->RuntimeShaped() && !CCtx().MemReuse())
        Error1(n.LOC(), "GCU forbids shared variable '" + n.name_str +
                            "` to be dynamically shaped (by " +
                            STR(sty->GetShape()) + ").");
      break;
    case Storage::LOCAL:
      if (pl_depth == 0)
        Error1(n.LOC(), "local variable '" + n.name_str +
                            "` must be declared inside parallel-by.");
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
    if ((CCtx().GetArch() == TargetArch::GCU20 ||
         CCtx().GetArch() == TargetArch::GCU21 ||
         CCtx().GetArch() == TargetArch::GCU3) &&
        n.indices != nullptr) {
      if (auto sty = GetSpannedType(GetSymbolType(n.data->name)))
        if (sty->GetStorage() == Storage::GLOBAL ||
            cur_params.count(InScopeName(n.data->name)))
          Error1(n.LOC(),
                 "global data access '" + STR(n) + "` is not allowed.");
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
      if (shape.IsDynamic())
        Error1(n.LOC(),
               "symbolic bound value is not supported for Factor backend yet.");
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
    if (n.IsArith() && (CCtx().GetArch() == TargetArch::GCU20 ||
                        CCtx().GetArch() == TargetArch::GCU21))
      Error1(n.LOC(), "Arithmetic built-in function is not supported on GCU2.");

    if (CCtx().GetArch() == TargetArch::GCU3 && n.function->name != "print" &&
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
      if (pl_depth != 0)
        Error1(n.LOC(), "unsupported: " + PSTR(n.scope) +
                            " synchronization in " + STR(pl2s()) + " scope.");
      break;
    case Storage::SHARED:
      if (pl_depth == 0)
        Error1(n.LOC(), "unsupported: " + PSTR(n.scope) +
                            " synchronization in " + STR(pl2s()) + " scope.");
      break;
    case Storage::LOCAL:
      if (CCtx().GetArch() != TargetArch::GCU4)
        Error1(n.LOC(), STR(CCtx().GetArch()) + " does not support " +
                            PSTR(n.scope) + " synchronization.");
      else if (pl_depth != 3)
        Error1(n.LOC(), "unsupported: " + PSTR(n.scope) +
                            " synchronization in " + STR(pl2s()) + " scope.");
      break;
    default:
      Error1(n.scope->LOC(),
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
