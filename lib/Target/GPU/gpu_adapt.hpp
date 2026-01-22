#ifndef __CHOREO_GPU_ADAPTION_HPP__
#define __CHOREO_GPU_ADAPTION_HPP__

// This apply the GPU target specific check and information annotation

#include "ast.hpp"
#include "codegen.hpp"
#include "target_utils.hpp"

#define STRINGIFY2(x) #x
#define STRINGIFY(x) STRINGIFY2(x)

namespace Choreo {

struct ParallelSymbols {
  std::map<ParallelLevel, std::set<std::string>> all_pvs;

  void AddLevelPV(ParallelLevel pl, const std::string& sym) {
    if (!all_pvs.count(pl)) all_pvs.emplace(pl, std::set<std::string>{});
    all_pvs[pl].insert(sym);
  }

  const std::set<std::string> GetLevelPVs(ParallelLevel pl) const {
    if (!all_pvs.count(pl)) return {};
    return all_pvs.at(pl);
  }

  const std::set<std::string> GetInnerPVs(ParallelLevel pl) const {
    std::set<std::string> inner;
    auto& pld = PlDepthMap::Get();
    for (auto& lvl_pvs : all_pvs) {
      if (pld.ToDepth(lvl_pvs.first) > pld.ToDepth(pl)) {
        inner.insert(lvl_pvs.second.begin(), lvl_pvs.second.end());
      }
    }
    return inner;
  }

  ParallelLevel GetPVLevel(const std::string& sym) {
    for (auto& lvl_pvs : all_pvs)
      if (lvl_pvs.second.count(sym)) return lvl_pvs.first;
    return ParallelLevel::UNKNOWN;
  }

  void Show(std::ostream& os, ParallelLevel pl) const {
    os << STR(pl) << " parallel variables: { ";
    if (all_pvs.count(pl))
      for (auto& pv : all_pvs.at(pl)) os << pv << " ";
    os << "}";
  }

  void Reset() { all_pvs.clear(); }
};

struct GPUAdaptor : public CodeGenerator {
private:
  std::unordered_map<std::string, AST::Parameter*> cur_params;
  std::stack<ParallelLevel> levels;
  std::string cur_fname;
  std::string cur_arch;

  ParallelSymbols ps;

  bool enforced_group = false;
  bool enforced_4x_group = false;

private:
  ParallelLevel Level() const { return levels.top(); }

private:
  bool BeforeVisitImpl(AST::Node& n) override {
    if (auto cf = dyn_cast<AST::ChoreoFunction>(&n)) {
      ps.Reset();
      cur_params.clear();
      cur_fname = cf->name;
      levels.push(ParallelLevel::SEQ);
    } else if (auto pb = dyn_cast<AST::ParallelBy>(&n)) {
      auto lvl = pb->GetLevel();
      levels.push(lvl);

      pb->SetMaxLevel(TargetMaxLevel());

      ps.AddLevelPV(lvl, InScopeName(pb->BPV()->name));
      for (auto id : pb->AllSubPVs())
        ps.AddLevelPV(lvl, InScopeName(cast<AST::Identifier>(id)->name));

      VST_DEBUG(ps.Show(dbgs(), lvl); dbgs() << "\n";);
      CheckPBSettings(pb);
    }
    return true;
  }

  bool AfterVisitImpl(AST::Node& n) override {
    if (auto pb = dyn_cast<AST::ParallelBy>(&n)) {
      if (pb->GetLevel() == ParallelLevel::GROUP)
        enforced_group = false;
      else if (pb->GetLevel() == ParallelLevel::GROUPx4)
        enforced_4x_group = false;
      levels.pop();
    }

    // mask stmts that are possible to be shared
    else if (auto c = dyn_cast<AST::Call>(&n))
      if (!c->IsExpr()) n.SetLevel(Level());

    return true;
  }

public:
  bool IsHost() const { return Level() == ParallelLevel::SEQ; }

  void CheckPBSettings(AST::ParallelBy* pb) {
    if (pb->GetLevel() == ParallelLevel::THREAD) {
      // Calculate total thread dimension (product of all dimensions)
      auto total_threads = sbe::nu(1);
      for (auto bv : pb->BoundValues()) {
        total_threads = (total_threads * bv)->Normalize();
      }

      auto ppb = cgi.GetPBTree(fname).GetParent(pb);
      assert(ppb->GetLevel() == ParallelLevel::GROUP);
      if (ppb->IsEnforced()) {
        auto asrt = sbe::oc_eq(total_threads, sbe::nu(32));
        auto msg = "The total thread dimension must be a multiple of 32 "
                   "when 'group' exists for " +
                   ToUpper(CCtx().GetArch()) + ".";
        if (auto b = VIBool(asrt)) {
          if (b.value() == false) Error1(pb->LOC(), msg);
        } else
          FCtx(cur_fname).InsertAssertion(asrt, pb->LOC(), msg);
      }

      if (TargetHasLevel(ParallelLevel::GROUPx4)) {
        auto gppb = cgi.GetPBTree(fname).GetParent(ppb);
        assert(gppb->GetLevel() == ParallelLevel::GROUPx4);
        if (ppb->IsEnforced() && gppb->IsEnforced())
          Error1(ppb->LOC(), "explicit 'group' inside the 'group-4' "
                             "parallel-by is not supported by " +
                                 ToUpper(CCtx().GetArch()) + ".");
        else if (gppb->IsEnforced()) {
          auto asrt = sbe::oc_eq(total_threads, sbe::nu(128));
          auto msg = "The total thread dimension must be a multiple of "
                     "128 when 'group-4' exists for " +
                     ToUpper(CCtx().GetArch()) + ".";
          if (auto b = VIBool(asrt)) {
            if (b.value() == false) Error1(pb->LOC(), msg);
          } else
            FCtx(cur_fname).InsertAssertion(asrt, pb->LOC(), msg);
        }
      }
    }
  }

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
#if 0 // TODO: dma size limitation in GPU
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
#endif
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

    // TODO: to confirm about the GPU value
#if 1
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
    if (n.operation == ".pad") {
      auto pc = cast<PadConfig>(n.config);
      for (const auto& v : pc->pad_mid->AllValues())
        if (auto il = AST::GetIntLiteral(v); !il || il->Val() != 0)
          Error1(v->LOC(), "dma.pad with pad_mid is not supported for CuTe "
                           "backend(must set pad_mid to 0).");
    }
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
              Error1(e->LOC(), "On " + CCtx().GetArch() +
                                   ", the config in "
                                   "dma.pad must be in range [0, 2^11].");
          } else {
            auto msg = "On " + CCtx().GetArch() +
                       ", the config in "
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
              Error1(n.LOC(), "On " + cur_arch +
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
              Error1(n.LOC(), "On " + cur_arch +
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
#endif
  }

  void CheckTMA(AST::DMA& n) {
    if (n.operation == ".any") return;

    assert(isa<AST::ChunkAt>(n.from));
    auto f_ca = cast<AST::ChunkAt>(n.from);
    auto f_name = f_ca->RefSymbol();
    auto f_sty = GetSpannedType(GetSymbolType(f_name));
    auto f_shape = f_sty->GetShape();

    assert(isa<AST::ChunkAt>(n.to));
    auto t_ca = cast<AST::ChunkAt>(n.to);
    auto t_name = t_ca->RefSymbol();
    auto t_sty = GetSpannedType(GetSymbolType(t_name));
    auto t_shape = t_sty->GetShape();
    auto elem_bits = [](BaseType bt) -> size_t {
      if (IsFloatSubByteType(bt)) return 8;
      switch (bt) {
      case BaseType::F4_E2M1: return 4;
      case BaseType::F6_E2M3:
      case BaseType::F6_E3M2: return 6;
      default: return SizeOf(bt) * 8;
      }
    };
    auto tma_align_bits = [&](BaseType bt) -> size_t {
      if (IsFloatSubByteType(bt)) return 16 * 8;
      return 16 * 8;
    };
    if ((f_sty->GetStorage() == Storage::GLOBAL ||
         f_sty->GetStorage() == Storage::DEFAULT) &&
        t_sty->GetStorage() == Storage::SHARED) {
      // none-interleave tma requires the `leading-dimension * elementsize` be a
      // multiple of 16
      if (!(t_shape.LeadingValue()->IsSymbolic())) {
        auto bits =
            (t_shape.LeadingValue() * sbe::nu(elem_bits(t_sty->ElementType())))
                ->Normalize();
        auto rem = bits % sbe::nu(tma_align_bits(t_sty->ElementType()));
        if (!sbe::ceq(rem->Normalize(), sbe::nu(0)))
          Error(f_ca->LOC(), "GPU TMA requires boxDim[0] * elementSizeInBits "
                             "be aligned to the TMA requirement. (got: " +
                                 STR(bits) + ")");
      } else {
        // TODO: emit runtime assessment
      }
    } else if ((t_sty->GetStorage() == Storage::GLOBAL ||
                t_sty->GetStorage() == Storage::DEFAULT) &&
               f_sty->GetStorage() == Storage::SHARED) {
      if (!(f_shape.LeadingValue()->IsSymbolic())) {
        auto bits =
            (f_shape.LeadingValue() * sbe::nu(elem_bits(f_sty->ElementType())))
                ->Normalize();
        auto rem = bits % sbe::nu(tma_align_bits(f_sty->ElementType()));
        if (!sbe::ceq(rem->Normalize(), sbe::nu(0)))
          Error(t_ca->LOC(), "GPU TMA requires boxDim[0] * elementSizeInBits "
                             "be aligned to the TMA requirement. (got: " +
                                 STR(bits) + ")");
      } else {
        // TODO: emit runtime assessment
      }
    }

    // check if the chunkat utilize invalid parallel variable
    auto intersection = [](const std::set<std::string>& set1,
                           const std::set<std::string>& set2) {
      std::set<std::string> intersection;

      std::set_intersection(set1.begin(), set1.end(), set2.begin(), set2.end(),
                            std::inserter(intersection, intersection.begin()));

      return intersection;
    };

    auto& pvs = ps.GetInnerPVs(ParallelLevel::BLOCK);
    auto& f_syms = ReferredSymbols(f_ca.get(), this);
    auto& t_syms = ReferredSymbols(t_ca.get(), this);

    auto format_string = [&](const std::set<std::string>& syms) {
      std::ostringstream oss;
      int i = 0;
      for (auto& sym : syms) {
        if (i++ != 0) oss << ", ";
        oss << "`" << UnScopedName(sym) << "'(" << STR(ps.GetPVLevel(sym))
            << ")";
      }
      return oss.str();
    };
    if (auto f_int = intersection(f_syms, pvs); !f_int.empty())
      Error1(f_ca->LOC(), "TMA is " + STR(ParallelLevel::BLOCK) +
                              "-wise that parallel variable " +
                              format_string(f_int) +
                              " can not be used as indices/tiling-factors.");
    if (auto t_int = intersection(t_syms, pvs); !t_int.empty())
      Error1(t_ca->LOC(), "TMA is " + STR(ParallelLevel::BLOCK) +
                              "-wise that parallel variable " +
                              format_string(t_int) +
                              " can not be used as indices/tiling-factors.");
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
      VST_DEBUG(dbgs() << "[GPUCHECK] Not Checking " << STR(vi) << " " << op
                       << " " << limit << ".\n");
      return;
    }

    message = "On " + cur_arch + ", must satisfy: " + message;

    // try to evaluate at compilation time
    auto cmp = sbe::cmp(op, vi, sbe::nu(limit))->Normalize();
    if (auto tf = VIBool(cmp)) {
      if (tf && tf.value() == false) {
        VST_DEBUG(dbgs() << "[GPUCHECK] Generated check at " << loc << ": "
                         << vi->ToString() << " " << op << " "
                         << std::to_string(limit) + "ULL"
                         << "\n\twith message: " << message << "\n");
        Error1(loc, message);
      } else
        VST_DEBUG(dbgs() << "[GPUCHECK] Check at " << loc
                         << " is statisfied: " << vi->ToString() << " " << op
                         << " " << std::to_string(limit) + "ULL"
                         << "\n\twith message: " << message << "\n");
      return;
    }

    // or else, generate the runtime check
    VST_DEBUG(dbgs() << "[GPUCHECK] Generated runtime check at " << loc << ": "
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
  GPUAdaptor() : CodeGenerator("gpu"), cur_arch(ToUpper(CCtx().GetArch())) {}
  ~GPUAdaptor() {}

  bool Visit(AST::NamedVariableDecl& n) override {
    auto ty = GetSymbolType(n.name_str);

    if (!isa<SpannedType>(ty)) {
      auto mem = n.GetMemory();
      if (!mem) return true;

      auto st = mem->Get();
      if (n.init_expr != nullptr) {
        if (st == Storage::SHARED)
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
      if (Level() != ParallelLevel::SEQ)
        Error1(n.LOC(), "global variable '" + n.name_str +
                            "` mustn't be declared inside parallel-by.");
      break;
    case Storage::SHARED:
      if (Level() == ParallelLevel::SEQ)
        Error1(n.LOC(), "shared variable '" + n.name_str +
                            "` must be declared inside parallel-by.");
      if (sty->RuntimeShaped() && !CCtx().MemReuse())
        Error1(n.LOC(), "GPU forbids shared variable '" + n.name_str +
                            "` to be dynamically shaped (by " +
                            STR(sty->GetShape()) + ").");
      break;
    case Storage::LOCAL:
      if (Level() == ParallelLevel::SEQ)
        Error1(n.LOC(), "local variable '" + n.name_str +
                            "` must be declared inside parallel-by.");
      if (sty->RuntimeShaped() && !CCtx().MemReuse())
        Error1(n.LOC(), "GPU forbids local variable '" + n.name_str +
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

  bool Visit(AST::Parameter& n) override {
    if (n.sym) cur_params.emplace(InScopeName(n.sym->name), &n);
    return true;
  }

  bool Visit(AST::DMA& n) override {
#if CHOREO_CUDA_VERSION < 12040
    if (n.IsTMA()) {
      Error1(n.LOC(),
             "TMA is not supported by current CUDA. "
             "(Version " STRINGIFY(CHOREO_CUDA_VERSION_MAJOR) "." STRINGIFY(
                 CHOREO_CUDA_VERSION_MINOR) " < 12.9+).");
      return false;
    }
#endif

    if (n.IsTMA() && !CCtx().TargetSupportTMA()) {
      Error1(n.LOC(), "TMA is not supported by current architecture: " +
                          CCtx().GetArch() + ".");
      return false;
    }

    if (n.operation == ".any") return true;

    // DMA directions check:
    // GPU's DMA is mainly serve for GMEM -> SMEM
    auto fty = GetSpannedType(n.from->GetType());
    auto tty = GetSpannedType(n.to->GetType());
    assert(fty && tty);
    auto fst = fty->GetStorage();
    auto tst = tty->GetStorage();

    // Many restriction on dma
    if (n.IsTMA()) {
      if (!(fst == Storage::SHARED &&
            (tst == Storage::GLOBAL || tst == Storage::DEFAULT)) &&
          !((fst == Storage::GLOBAL || fst == Storage::DEFAULT) &&
            tst == Storage::SHARED))
        Error1(n.LOC(), "GPU does not allow the TMA " + n.operation.substr(1) +
                            " (" + STR(fst) + " -> " + STR(tst) + ").");
    } else if (n.IsAsync()) {
      if ((fst == Storage::SHARED && tst == Storage::GLOBAL) ||
          (fst == Storage::SHARED && tst == Storage::SHARED) ||
          (fst == Storage::SHARED && tst == Storage::LOCAL) ||
          (fst == Storage::GLOBAL && tst == Storage::LOCAL) ||
          (fst == Storage::DEFAULT && tst == Storage::LOCAL) ||
          (fst == Storage::GLOBAL && tst == Storage::GLOBAL))
        Error1(n.LOC(), "GPU does not allow the async " +
                            n.operation.substr(1) + " (" + STR(fst) + " -> " +
                            STR(tst) + ").");
    } else {
      if ((fst == Storage::GLOBAL) && (tst == Storage::GLOBAL))
        Error1(n.LOC(), "GPU does not allow the sync " + n.operation.substr(1) +
                            " (" + STR(fst) + " -> " + STR(tst) + ").");
    }

    if (fst == Storage::GLOBAL && tst == Storage::SHARED && n.IsTMA())
      n.SetLevel(ParallelLevel::BLOCK); // single instance in a block
    else
      n.SetLevel(ParallelLevel::THREAD); // threads-cooperative

    if (n.IsTMA())
      CheckTMA(n);
    else
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
      case Storage::SHARED: {
        auto p = cur_params[InScopeName(f_name)];
        if (p->attr == ParamAttr::NONE) p->attr = ParamAttr::SHADOW_TO_GLOBAL;
        break;
      }
      default: break;
      }
    };
    if (auto to = dyn_cast<AST::ChunkAt>(n.to)) {
      annotate_by_storage(
          cast<SpannedType>(GetSymbolType(to->RefSymbol()))->GetStorage());
    } else if (auto m = dyn_cast<AST::Memory>(n.to))
      annotate_by_storage(m->st);
    return true;
  }

  bool Visit(AST::MMA& n) override {
    auto& op = *n.GetOperation();
    ValueList mma_shape;
    switch (op.Tag()) {
    case AST::MMAOperation::Fill: break;
    case AST::MMAOperation::Load: break;
    case AST::MMAOperation::Exec: {
      auto& a_sym = op.ExecOperand(1);
      auto& b_sym = op.ExecOperand(2);
      auto& c_sym = op.ExecOperand(0);
      auto a_sty = GetSpannedType(GetSymbolType(a_sym));
      auto b_sty = GetSpannedType(GetSymbolType(b_sym));
      auto c_sty = GetSpannedType(GetSymbolType(c_sym));
      auto a_shape = a_sty->GetShape();
      auto b_shape = b_sty->GetShape();
      switch (op.GetMethod()) {
      case AST::MMAOperation::ROW_ROW:
        mma_shape.push_back(a_shape.ValueAt(0));
        mma_shape.push_back(b_shape.ValueAt(0));
        if (op.IsSparse())
          mma_shape.push_back(
              sbe::bop(OpCode::MULTIPLY, a_shape.ValueAt(1), sbe::nu(2))
                  ->Normalize());
        else
          mma_shape.push_back(a_shape.ValueAt(1));
        break;
      case AST::MMAOperation::ROW_COL:
        mma_shape.push_back(a_shape.ValueAt(0));
        mma_shape.push_back(b_shape.ValueAt(1));
        if (op.IsSparse())
          mma_shape.push_back(
              sbe::bop(OpCode::MULTIPLY, a_shape.ValueAt(1), sbe::nu(2))
                  ->Normalize());
        else
          mma_shape.push_back(a_shape.ValueAt(1));
        break;
      case AST::MMAOperation::COL_ROW:
        mma_shape.push_back(a_shape.ValueAt(1));
        mma_shape.push_back(b_shape.ValueAt(0));
        if (op.IsSparse())
          mma_shape.push_back(
              sbe::bop(OpCode::MULTIPLY, a_shape.ValueAt(0), sbe::nu(2))
                  ->Normalize());
        else
          mma_shape.push_back(a_shape.ValueAt(0));
        break;
      case AST::MMAOperation::COL_COL:
        mma_shape.push_back(a_shape.ValueAt(1));
        mma_shape.push_back(b_shape.ValueAt(1));
        if (op.IsSparse())
          mma_shape.push_back(
              sbe::bop(OpCode::MULTIPLY, a_shape.ValueAt(0), sbe::nu(2))
                  ->Normalize());
        else
          mma_shape.push_back(a_shape.ValueAt(0));
        break;
      default: choreo_unreachable("unsupported mma execution method.");
      }
      auto MMAShapeSTR = [](ValueList s) {
        assert(s.size() == 3);
        std::ostringstream oss;
        oss << "m" << STR(s[0]) << "n" << STR(s[1]) << "k" << STR(s[2]);
        return oss.str();
      };
#if 1
      auto a_ty = a_sty->ElementType();
      auto b_ty = b_sty->ElementType();
      auto c_ty = c_sty->ElementType();
      auto d_ty = c_ty;
      auto scale_ty = BaseType::UNKNOWN;
      auto arch = CCtx().ArchNum();

      if (a_ty == BaseType::F32) a_ty = BaseType::TF32;
      if (b_ty == BaseType::F32) b_ty = BaseType::TF32;

      auto sparsity = op.IsSparse() ? MMALimit::SPARSE : MMALimit::DENSE;
      MMALimit::MMAConfig mma_config{sparsity, a_ty,     b_ty,     c_ty,
                                     d_ty,     scale_ty, mma_shape};
      if (!IsValidMMAConfig(mma_config, arch))
        Error1(n.LOC(), "MMA [" + STR(a_ty) + "(a)" + STR(b_ty) + "(b)" +
                            (scale_ty != BaseType::UNKNOWN
                                 ? ":" + STR(scale_ty) + "(scale)"
                                 : "") +
                            STR(c_ty) + "(c)" + STR(d_ty) +
                            "(d): " + MMAShapeSTR(mma_shape) +
                            "] is not support by current architecture(" +
                            ToUpper(CCtx().GetArch()) + ").");

      auto mma_ty = GetMMAType(mma_config);

      // WGMMA requires SM90+ architecture
      if (mma_ty == MMAType::WGMMA && arch < 90)
        Error1(n.LOC(), "WGMMA [" + MMAShapeSTR(mma_shape) +
                            "] requires SM90+ architecture, "
                            "but target is SM" +
                            std::to_string(arch) + ".");

      FCtx(cur_fname).SetFragMMAType(InScopeName(a_sym), mma_ty);
      FCtx(cur_fname).SetFragMMAType(InScopeName(b_sym), mma_ty);
      FCtx(cur_fname).SetFragMMAType(InScopeName(c_sym), mma_ty);
      if (op.IsSparse() && !op.ExecOperand(3).empty()) {
        FCtx(cur_fname).SetFragMMAType(InScopeName(op.ExecOperand(3)), mma_ty);
      }

      // TODO: consider to merge predicate
      if (mma_ty == MMAType::CTMMA) {
        std::string mma_policy = MMALimit::MMAConfig2CuteMMAName(mma_config);
        FCtx(cur_fname).SetMMAPolicyOfFrag(InScopeName(a_sym), mma_policy);
        FCtx(cur_fname).SetMMAPolicyOfFrag(InScopeName(b_sym), mma_policy);
        FCtx(cur_fname).SetMMAPolicyOfFrag(InScopeName(c_sym), mma_policy);
        if (op.IsSparse() && !op.ExecOperand(3).empty()) {
          FCtx(cur_fname).SetMMAPolicyOfFrag(InScopeName(op.ExecOperand(3)),
                                             mma_policy);
        }
      } else if (mma_ty == MMAType::WGMMA) {
        std::string mma_policy = MMALimit::MMAConfig2WGMMAName(mma_config);
        FCtx(cur_fname).SetMMAPolicyOfFrag(InScopeName(c_sym), mma_policy);
      }
#else
      auto ety = a_ty->ElementType();
      switch (ety) {
      case BaseType::F16:
        if (!sbe::ceq(mma_shape[0], sbe::nu(16)) ||
            !sbe::ceq(mma_shape[1], sbe::nu(16)) ||
            !sbe::ceq(mma_shape[2], sbe::nu(16)))
          Error1(n.LOC(), "MMA [" + STR(ety) + ": " + MMAShapeSTR(mma_shape) +
                              "] is not support by current architecture(" +
                              CCtx().GetArch() + ").");
        break;
      case BaseType::F32:
        if (sbe::ceq(mma_shape[0], sbe::nu(16)) ||
            sbe::ceq(mma_shape[1], sbe::nu(16)) ||
            sbe::ceq(mma_shape[2], sbe::nu(8)))
          return true;
        else if (sbe::ceq(mma_shape[0], sbe::nu(16)) ||
                 sbe::ceq(mma_shape[1], sbe::nu(16)) ||
                 sbe::ceq(mma_shape[2], sbe::nu(16))) {
          // support tf32 mma here
          Error1(n.LOC(), "MMA [" + STR(ety) + ": " + MMAShapeSTR(mma_shape) +
                              "] is yet to support.");
          return true;
        } else
          Error1(n.LOC(), "MMA [" + STR(ety) + ": " + MMAShapeSTR(mma_shape) +
                              "] is not support by current architecture(" +
                              CCtx().GetArch() + ").");
        break;
      default:
        choreo_unreachable(STR(ety) + " is not supported by current MMA");
        break;
      }
#endif
      VST_DEBUG(dbgs() << STR(n) << ", mma_size: " << MMAShapeSTR(mma_shape)
                       << "\n");
    } break;
    case AST::MMAOperation::Store: break;
    default: choreo_unreachable("unsupported mma operation.");
    }
    return true;
  }

  bool Visit(AST::Call& n) override {
    if (n.IsArith())
      Error1(n.LOC(),
             "Arithmetic built-in function is yet to supported on CUDA.");

    return true;
  }

  bool Visit(AST::Synchronize& n) override {
    auto pl = Level();

    switch (n.Resource()) {
    case Storage::GLOBAL:
      if ((pl == ParallelLevel::BLOCK) || (pl == ParallelLevel::GROUP) ||
          (pl == ParallelLevel::THREAD) || (pl == ParallelLevel::TERM))
        Error1(n.LOC(), "unsupported: " + STR(n.Resource()) +
                            " synchronization in " + STR(pl) + " scope.");
      break;
    case Storage::SHARED:
    case Storage::LOCAL:
      if ((pl == ParallelLevel::SEQ) || (pl == ParallelLevel::TERM) ||
          (pl == ParallelLevel::DEVICE))
        Error1(n.LOC(), "unsupported: " + STR(n.Resource()) +
                            " synchronization in " + STR(pl) + " scope.");
      break;
    case Storage::NODE:
    default:
      Error1(n.LOC(),
             "unsupported synchronization: " + STR(n.Resource()) + ".");
    }
    return true;
  }
};

} // end namespace Choreo

#endif // __CHOREO_GPU_ADAPTION_HPP__
