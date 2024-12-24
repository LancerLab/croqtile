#ifndef __CHOREO_FACTOR_CHECK_HPP__
#define __CHOREO_FACTOR_CHECK_HPP__

// This apply the GCU target specific check and information annotation

#include "ast.hpp"
#include "visitor.hpp"

namespace Choreo {

struct FactorCheck : public VisitorWithSymTab {
private:
  std::string cur_fname;

private:
  bool BeforeVisitImpl(AST::Node& n) override {
    TraceEachVisit(n, "(pre)");
    if (auto cf = dyn_cast<AST::ChoreoFunction>(&n)) { cur_fname = cf->name; }
    return true;
  }

  bool AfterVisitImpl(AST::Node& n) override {
    TraceEachVisit(n, "(post)");
    return true;
  }

  void TraceEachVisit(AST::Node& n, std::string sup = "") {
    if (trace_visit) dbgs() << n.TypeNameString() << sup << "\n";
  }

  void CheckDimSize(const Shape& s, size_t idx, const std::string& op,
                    size_t limit, const location& loc) {
    assert(idx < s.Rank());
    std::string message = "[FactorCheck] The " + Ordinal(idx + 1) + " dim " +
                          ValueItemAsString(s.ValueAt(idx)) +
                          " should satisfy: d " + op + " " +
                          std::to_string(limit);

    std::map<std::string, std::function<bool(size_t, size_t)>> op_map = {
        {">", [](size_t l, size_t r) { return l > r; }},
        {"<", [](size_t l, size_t r) { return l < r; }},
        {"==", [](size_t l, size_t r) { return l == r; }},
        {"!=", [](size_t l, size_t r) { return l != r; }},
        {">=", [](size_t l, size_t r) { return l >= r; }},
        {"<=", [](size_t l, size_t r) { return l <= r; }},
    };

    if (auto vi_int = dyn_cast<int>(&s.ValueAt(idx));
        vi_int && op_map.count(op)) {
      if (!op_map[op](*vi_int, limit)) {
        Error(loc, message);
        error_count++;
      }
    } else {
      auto vi_str = ValueItemAsString(s.ValueAt(idx), true);
      FCtx(cur_fname).AppendRtCheck(
          {vi_str, op, std::to_string(limit) + "ULL", loc, message, {}});
    }
  }

public:
  FactorCheck(const ptr<SymbolTable> s_tab)
      : VisitorWithSymTab("ftchk", s_tab) {}
  ~FactorCheck() {}

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
    return true;
  }
  bool Visit(AST::ParamList& n) override {
    TraceEachVisit(n);
    return true;
  }
  bool Visit(AST::ParallelBy& n) override {
    TraceEachVisit(n);
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

    if (CCtx().GetArch() != TargetArch::GCU20 &&
        CCtx().GetArch() != TargetArch::GCU21)
      return true;
    // TODO: GCU3

    if (n.operation == ".any") return true;

    // http://docs.enflame.cn/sw/manuals/factor_user_guide/TopsFactor_User_Guide.html#dte

    assert(isa<AST::ChunkAt>(n.from));
    auto f_ca = cast<AST::ChunkAt>(n.from);
    auto f_name = f_ca->RefSymbol();
    auto f_sty = GetSpannedType(GetSymbolType(f_name));
    auto f_sto = f_sty->GetStorage();
    auto f_shape = f_sty->GetShape();
    auto f_rank = f_shape.Rank();

    assert(isa<AST::ChunkAt>(n.to));
    auto t_ca = cast<AST::ChunkAt>(n.to);
    auto t_name = t_ca->RefSymbol();
    auto t_sty = GetSpannedType(GetSymbolType(t_name));
    auto t_sto = t_sty->GetStorage();
    auto t_shape = t_sty->GetShape();
    auto t_rank = t_shape.Rank();

    // common limitation (currently guarded by memcheck)
    for (auto& sty : {f_sty, t_sty}) {
      if (sty->RuntimeShaped()) {
        std::string bs = f_sty->ByteSizeExpression(true);
        FCtx(cur_fname).AppendRtCheck(
            {bs,
             "<",
             std::to_string(1ULL << 32) + "ULL",
             n.LOC(),
             "The size of data transferred by DMA cannot exceed 2^32",
             {}});
      } else {
        if (sty->ByteSize() >= (1ULL << 32)) {
          Error(n.LOC(), "[FactorCheck] The size of data transferred by "
                         "DMA cannot exceed 2^32");
          error_count++;
        }
      }
    }

    auto IsLinearCopy = [&]() -> bool {
      return f_ca->positions == nullptr && t_ca->positions == nullptr;
    };
    auto IsSlice = [&]() -> bool {
      if (f_ca->positions != nullptr && t_ca->positions == nullptr)
        if (MemLevel(f_sto) > MemLevel(t_sto)) return true;
      return false;
    };
    auto IsDeslice = [&]() -> bool {
      if (f_ca->positions == nullptr && t_ca->positions != nullptr)
        if (MemLevel(f_sto) < MemLevel(t_sto)) return true;
      return false;
    };

    // linear copy
    // omitted

    // transpose
    if (n.operation == ".transp" && IsLinearCopy()) {
      if (f_rank > 5) {
        Error(n.LOC(), "[FactorCheck] The rank in dma.transp(not slice "
                       "nor deslice) must be in range [1, 5]");
        error_count++;
      }
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
        Error(n.LOC(),
              "[FactorCheck] dma.transp(not slice nor deslice) does not "
              "support 5-dimensional array (if dim is 5, layout[0] must be 0)");
        error_count++;
      }
    }

    // pad
    if (n.operation == ".pad" && IsLinearCopy()) {
      if (f_rank > 5) {
        Error(n.LOC(), "[FactorCheck] The rank in dma.pad should "
                       "be in range [1, 5]");
        error_count++;
      }
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
          Error(n.LOC(), "[FactorCheck] The value of padding_low in "
                         "dma.pad must be in range [0, 2^11]");
          error_count++;
        }
      }

      for (auto v : pc->pad_high) {
        if (v > (1 << 11)) {
          Error(n.LOC(), "[FactorCheck] The value of padding_high in "
                         "dma.pad must be in range [0, 2^11]");
          error_count++;
        }
      }
      // padding_mid
      for (size_t idx = 0; idx < f_rank; ++idx) {
        size_t v = pc->pad_mid[idx];
        if (idx == f_rank - 1) {
          if (v != 0) {
            Error(n.LOC(),
                  "[FactorCheck] The value of padding_mid[rank-1] in dma.pad "
                  "must be 0 (mid padding of dim[rank-1] is not supported by "
                  "the hardware)");
            error_count++;
          }
        } else if (v > (1 << 10)) {
          Error(n.LOC(), "[FactorCheck] The value of padding_mid in "
                         "dma.pad must be in range [0, 2^10]");
          error_count++;
        }
      }
      if (f_rank == 5) {
        if (pc->pad_low[0] != 0) {
          Error(n.LOC(), "[FactorCheck] dma.pad does not support 5-dimensional "
                         "array (if dim is 5, pad_low[0] must be 0)");
          error_count++;
        }
        if (pc->pad_high[0] != 0) {
          Error(n.LOC(), "[FactorCheck] dma.pad does not support 5-dimensional "
                         "array (if dim is 5, pad_high[0] must be 0)");
          error_count++;
        }
        if (pc->pad_mid[0] != 0) {
          Error(n.LOC(), "[FactorCheck] dma.pad does not support 5-dimensional "
                         "array (if dim is 5, pad_mid[0] must be 0)");
          error_count++;
        }
      }

      if (pc->value.t != f_sty->f_type) {
        Error(n.from->LOC(), "[FactorCheck] Data type of pad value is "
                             "inconsistent with that of data in dma: " +
                                 STR(pc->value.t) + " vs. " +
                                 STR(f_sty->f_type));
        error_count++;
      }
    }

    // 根据factor文档 （均不考虑数据搬运方向：L1->L2, L2->L1, ...）
    // linear copy 指的是整块数据搬运
    // transpose 指的是对整块数据transp
    // pad 是对 from 做 padding
    // slice 是 大块 到 小块
    // deslice 相反
    // slice transpose 为 slice, then transp
    // transpose deslice 为 transp, then deslice
    // 目前 choreo dma 的 copy 操作,

    // slice
    if (n.operation == ".copy" && IsSlice()) {
      if (f_rank > 5) {
        Error(n.LOC(), "[FactorCheck] The rank in dma.copy(slice) must "
                       "be in range [1, 5]");
        error_count++;
      }
      for (size_t idx = 0; idx < f_rank; ++idx)
        CheckDimSize(f_shape, idx, "<", 1 << 24, n.from->LOC());

      for (size_t idx = 0; idx < t_rank; ++idx)
        CheckDimSize(t_shape, idx, "<", 1 << 16, n.to->LOC());
      if (f_rank == 5) {
        auto first = f_ca->positions->ValueAt(0);
        auto t = dyn_cast<BoundedITupleType>(first->GetType());
        assert(t != nullptr);
        if (isa<int>(&t->ubounds.ValueAt(0))) {
          if (!IsValueItemEqual(1, t->ubounds.ValueAt(0))) {
            Error(n.LOC(), "[FactorCheck] dma.copy(slice) does not "
                           "support 5-dimensional "
                           "array (if dim is 5, offsets[0] must be 0)");
            error_count++;
          }
        } else {
          // TODO
          // Is that the case?
        }
      }
    }

    // deslice
    if (n.operation == ".copy" && IsDeslice()) {
      if (f_rank > 5) {
        Error(n.LOC(), "[FactorCheck] The rank in dma.copy(deslice) must "
                       "be in range [1, 5]");
        error_count++;
      }
      for (size_t idx = 0; idx < f_rank; ++idx)
        CheckDimSize(f_shape, idx, "<", 1 << 16, n.from->LOC());

      for (size_t idx = 0; idx < t_rank; ++idx)
        CheckDimSize(t_shape, idx, "<", 1 << 24, n.to->LOC());
    }

    // slice transpose
    if (n.operation == ".transp" && IsSlice()) {
      if (f_rank > 5) {
        Error(n.LOC(), "[FactorCheck] The rank in dma.transp(slice then "
                       "transpose) must be in range [1, 5]");
        error_count++;
      }
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
        if (isa<int>(&t->ubounds.ValueAt(0))) {
          if (!IsValueItemEqual(1, t->ubounds.ValueAt(0))) {
            Error(n.LOC(), "[FactorCheck] dma.transp(slice then "
                           "transpose) does not support 5-dimensional "
                           "array (if dim is 5, offsets[0] must be 0)");
            error_count++;
          }
        } else {
          // TODO
          // Is that the case?
        }
      }

      auto tc = cast<TransposeConfig>(n.config);
      if (f_rank == 5 && tc->dim_values[0] != 0) {
        Error(n.LOC(), "[FactorCheck] dma.transp(slice then transpose) "
                       "does not support 5-dimensional "
                       "array (if dim is 5, layout[0] must be 0)");
        error_count++;
      }
    }

    // transpose deslice
    if (n.operation == ".transp" && IsDeslice()) {
      if (f_rank > 5) {
        Error(n.LOC(), "[FactorCheck] The rank in dma.transp(transpose "
                       "then deslice) must be in range [1, 5]");
        error_count++;
      }
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
        if (isa<int>(&t->ubounds.ValueAt(0))) {
          if (!IsValueItemEqual(1, t->ubounds.ValueAt(0))) {
            Error(n.LOC(), "[FactorCheck] dma.transp(transpose then "
                           "deslice) does not support 5-dimensional "
                           "array (if dim is 5, offsets[0] must be 0)");
            error_count++;
          }
        } else {
          // TODO
          // Is that the case?
        }
      }

      auto tc = cast<TransposeConfig>(n.config);
      if (t_rank == 5 && tc->dim_values[0] != 0) {
        Error(n.LOC(), "[FactorCheck] dma.transp(transpose then deslice) "
                       "does not support 5-dimensional "
                       "array (if dim is 5, layout[0] must be 0)");
        error_count++;
      }
    }

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
    return true;
  }
  bool Visit(AST::Rotate& n) override {
    TraceEachVisit(n);
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

#endif // __CHOREO_FACTOR_CHECK_HPP__
