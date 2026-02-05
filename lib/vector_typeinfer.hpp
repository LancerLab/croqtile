#ifndef __VECTOR_TYPEINFER_HPP__
#define __VECTOR_TYPEINFER_HPP__

#include "ast.hpp"
#include "aux.hpp"
#include "context.hpp"
#include "diversity_analysis.hpp"
#include "io.hpp"
#include "loop_utils.hpp"
#include "symtab.hpp"
#include "types.hpp"
#include <cstddef>

namespace Choreo {

struct VectorTypeInfer final : LoopVisitor {
private:
  ptr<LoopInfo> li;
  ptr<DiversityInfo> di;
  std::unordered_map<std::string, ptr<Type>> with_types;
  std::string indent = "    ";

  bool Skip(ptr<Type> ty, DiversityShape shape) {
    if (!InLoop()) return true;
    if (!cur_loop->CanVectorize()) return true;
    if (shape.Uniform()) return true;
    if (IsActualVectorType(ty)) return true;

    return false;
  }

  bool VTypeEq(const ptr<Type>& rhs, const ptr<Type>& lhs) {
    if (!IsActualVectorType(rhs) || !IsActualVectorType(lhs)) return false;
    if (ElementCount(rhs) != ElementCount(lhs)) return false;
    if (ElementType(rhs) != ElementType(lhs)) return false;
    return true;
  }

  void AssignSymVType(const location& loc, const std::string& sym,
                      const ptr<Type>& ty) {
    if (!IsActualVectorType(ty)) {
      Error1(loc, "symbol: " + sym + " is not associated with a vector type.");
    }
    if (!SymTab()->Exists(sym)) {
      Error1(loc, "The symbol `" + sym + "' has not been defined.");
    } else {
      auto symbol = SymTab()->GetSymbol(sym);
      symbol->SetType(ty);
    }
  }

  ptr<Type> GetSymVType(const location& loc, const std::string& sym) {
    assert(SymTab()->Exists(sym));
    if (!SymTab()->Exists(sym)) {
      Error1(loc, "The symbol `" + sym + "' has not been defined.");
      return nullptr;
    }
    auto vty = SymTab()->GetSymbol(sym);
    return vty->GetType();
  }

public:
  VectorTypeInfer(const ptr<SymbolTable> s_tab, ptr<LoopInfo> l,
                  ptr<DiversityInfo> d)
      : LoopVisitor(s_tab, "vector-type-infer"), li(l), di(d) {}

  bool Visit(AST::WithIn& n) {
    TraceEachVisit(n);
    if (n.with)
      with_types.emplace(InScopeName(n.with->name), n.with->GetType());

    return true;
  }

  bool Visit(AST::ForeachBlock& n) {
    TraceEachVisit(n);
    ptr<AST::Call> c = nullptr;

    if (!cur_loop->CanVectorize()) return true;

    auto iv_sym = cur_loop->IVSym();
    bool with_found = false;
    std::string with_sym;
    ptr<Type> with_ty = nullptr;

    for (auto item : within_map) {
      if (with_types.count(item.first) == 0) continue;
      auto ivs = item.second;
      if (ivs[ivs.size() - 1] == iv_sym) {
        with_found = true;
        with_sym = item.first;
        with_ty = with_types[with_sym];
        break;
      }
    }

    if (with_found) {
      assert(isa<BoundedITupleType>(with_ty) && "with type should be valid.");
      auto bit = dyn_cast<BoundedITupleType>(with_ty);

      assert(bit && "with type should be BoundedITupleType.");
      auto lb = bit->GetLowerBounds();
      auto ub = bit->GetUpperBounds();
      auto s = bit->GetSteps();
      auto widths = bit->GetWidths();
      widths[widths.size() - 1] = cur_loop->GetVectorFactor();
      auto vty = MakeBoundedITupleType(lb, ub, s, widths);
      if (debug_visit)
        dbgs() << indent << "IV:   " << with_sym << ", Type: " << PSTR(vty)
               << "\n";
      AssignSymVType(n.LOC(), with_sym, vty);
    }

    auto iv_ty = cur_loop->GetIVType();

    auto lb = dyn_cast<BoundedITupleType>(iv_ty)->GetLowerBounds();
    auto ub = dyn_cast<BoundedITupleType>(iv_ty)->GetUpperBounds();
    auto s = dyn_cast<BoundedITupleType>(iv_ty)->GetSteps();
    IntegerList widths(iv_ty->Dims(), cur_loop->GetVectorFactor());
    auto vty = MakeBoundedITupleType(lb, ub, s, widths);
    cur_loop->SetIVType(vty);

    if (debug_visit)
      dbgs() << indent << "IV:   " << cur_loop->IVSym()
             << ", Type: " << PSTR(vty) << "\n";
    AssignSymVType(n.LOC(), cur_loop->IVSym(), vty);

    return true;
  }

  bool Visit(AST::Identifier& n) {
    TraceEachVisit(n);
    auto nty = GetSymVType(n.loc, InScopeName(n.name));
    assert(nty);
    auto scoped_sym = InScopeName(n.name);
    if (!IsActualVectorType(nty)) return true;

    n.SetType(nty);
    return true;
  }

  bool Visit(AST::DataAccess& n) {
    TraceEachVisit(n);
    auto nty = n.GetType();
    auto nds = n.GetDiversityShape();
    if (Skip(nty, nds)) return true;
    if (!n.AccessElement()) {
      auto name = n.data->name;
      if (di->IsDefiniteUniform(InScopeName(name))) {
        // if the array is definitely uniform, we do not need to widen its type
        n.SetType(n.data->GetType());
        return true;
      }
      auto vty = GetSymVType(n.LOC(), InScopeName(n.GetDataName()));
      n.SetType(vty);
    } else {
      auto da_ty = n.GetType();
      auto ety = da_ty->GetBaseType();
      for (auto indice : n.GetIndices()) {
        auto indice_ty = indice->GetType();
        if (IsActualVectorType(indice_ty)) {
          auto ElemCount = ElementCount(indice_ty);
          auto da_vty = MakeVectorType(ety, ElemCount);
          n.SetType(da_vty);
          if (debug_visit) {
            dbgs() << indent << "DA:   " << InScopeName(n.GetDataName());
            for (auto& idx : n.GetIndices()) dbgs() << "[" << PSTR(idx) << "]";
            dbgs() << ", Type: " << PSTR(da_vty) << "\n";
          }
        }
        auto indice_ds = indice->GetDiversityShape();
        if (indice_ds.Stride(1)) {
          n.AddNote("VLDST"); // load/store
        } else if (indice_ds.Divergent() || indice_ds.Stride()) {
          n.AddNote("VGZST"); // getter/scatter
        }
      }
    }
    return true;
  }

  bool Visit(AST::Expr& n) {
    TraceEachVisit(n);
    auto nty = n.GetType();
    auto nds = n.GetDiversityShape();
    if (Skip(nty, nds)) return true;
    if (n.IsReference()) {
      auto ref = n.GetReference();
      auto ref_ty = ref->GetType();
      if (IsActualVectorType(ref_ty)) { n.SetType(ref_ty); }
    } else if (n.IsBinary()) {
      auto op = n.GetOp();
      if (op == "dimof") {
        choreo_unreachable("dimof expr can not be vectorized.");
      }
      auto lhs = n.GetL();
      auto rhs = n.GetR();
      assert(lhs && rhs && "binary expr should have two operands.");
      auto lhs_ty = lhs->GetType();
      auto rhs_ty = rhs->GetType();
      if (IsActualVectorType(lhs_ty) && IsActualVectorType(rhs_ty)) {
        assert(VTypeEq(lhs_ty, rhs_ty) &&
               "vector types of two operands should be the same.");
        auto ecount = ElementCount(lhs_ty);
        auto etype = ElementType(lhs_ty);
        auto v_ty = MakeVectorType(etype, ecount);
        n.SetType(v_ty);
      } else if (IsActualVectorType(lhs_ty) && !IsActualVectorType(rhs_ty)) {
        // broadcast rhs to vector
        auto ecount = ElementCount(lhs_ty);
        auto etype = ElementType(lhs_ty);
        rhs->AddNote("broadcast", std::to_string(ecount));
        if (debug_visit)
          dbgs() << indent << "Expr: broadcast `" << STR(rhs)
                 << "`, type: " << PSTR(MakeVectorType(etype, ecount)) << "\n";
        auto v_ty = MakeVectorType(etype, ecount);
        n.SetType(v_ty);
      } else if (!IsActualVectorType(lhs_ty) && IsActualVectorType(rhs_ty)) {
        // broadcast lhs to vector
        auto ecount = ElementCount(rhs_ty);
        auto etype = ElementType(rhs_ty);
        lhs->AddNote("broadcast", std::to_string(ecount));
        if (debug_visit)
          dbgs() << indent << "Expr: broadcast `" << STR(lhs)
                 << "`, type:  " << PSTR(MakeVectorType(etype, ecount)) << "\n";
        auto v_ty = MakeVectorType(etype, ecount);
        n.SetType(v_ty);
      } else {
        choreo_unreachable("at least one operand should be vector type.");
      }
    } else if (n.IsTernary()) {
    }
    return true;
  }

  bool Visit(AST::CastExpr& n) {
    TraceEachVisit(n);
    auto nty = n.GetType();
    auto nds = n.GetDiversityShape();
    if (Skip(nty, nds)) return true;
    auto r_expr = dyn_cast<AST::Expr>(n.GetR());
    assert(r_expr && "Only Expr can be R of CastExpr.");
    auto r_ty = r_expr->GetType();
    if (IsActualVectorType(r_ty)) {
      auto ec = ElementCount(r_ty);
      auto new_vty = MakeVectorType(nty->GetBaseType(), ec);
      n.SetType(new_vty);
      n.SetFrom(n.FromType(), ec);
      n.SetTo(n.ToType(), ec);
      if (debug_visit)
        dbgs() << indent << "cast: `" << PSTR(nty) << "` -> " << PSTR(new_vty)
               << "\n";
    }
    return true;
  }

  bool Visit(AST::Assignment& n) {
    TraceEachVisit(n);
    auto nty = n.GetType();
    auto nds = n.GetDiversityShape();
    if (Skip(nty, nds)) return true;

    auto from = n.value;
    auto from_ty = from->GetType();
    auto from_ds = from->GetDiversityShape();
    auto to = n.da;
    auto to_ty = to->GetType();
    auto to_ds = to->GetDiversityShape();
    assert(!to_ds.Unknown() && !from_ds.Unknown() &&
           "diversity shape should not be unknown in Assignment.");
    // if this is an array element assignment
    if (n.AssignToDataElement()) {
      // TODO: support reduction operation
      // first, we must ensure the dest array element cannot be uniform if the
      // from is varying e.g., a[0] = varying_value; where a[0] is a uniform
      // access.
      if (from_ds.Varying() && to_ds.Uniform()) {
        Error1(n.LOC(),
               "cannot assign a varying value to a uniform array element: " +
                   STR(n));
        return false;
      }
      // uniform -> varying
      // e.g., a.at[i] = 0;
      else if (from_ds.Uniform() && to_ds.Varying()) {
        from->AddNote("broadcast", std::to_string(cur_loop->GetVectorFactor()));
        n.SetType(to_ty);
      } else {
        // varying -> varying
        // e.g., a.at[i] = b.at[i]
        if (!VTypeEq(from_ty, to_ty)) {
          Error1(n.LOC(), "type mismatch in assignment: " + STR(n) + ".");
          return false;
        }
        n.SetType(from_ty);
      }

    } else {
      // e.g., val = a.at[i]
      // TODO: if val is masked as varying,
      if (n.IsDecl()) {
        assert(IsActualVectorType(from_ty));
        if (debug_visit)
          dbgs() << indent << "asgn:   " << InScopeName(n.GetName()) << " = "
                 << STR(n.value) << ", type: " << PSTR(from_ty) << "\n";
        n.SetType(from_ty);
      } else {
        // 1. if to is a scalar type and from is a vector type, there exist a
        // reduce operation(TODO)
        if (!IsActualVectorType(to_ty) && IsActualVectorType(from_ty)) {
          choreo_unreachable(
              "currently we do not support reduce operation in assignment.");
        }
        // 2. if to is a vector type and from is a scalar type, there exist a
        // broadcast operation
        if (IsActualVectorType(to_ty) && !IsActualVectorType(from_ty)) {
          from->AddNote("broadcast",
                        std::to_string(cur_loop->GetVectorFactor()));
          if (debug_visit)
            dbgs() << indent << "Asgn: broadcast `" << STR(from)
                   << "`, type: " << PSTR(to_ty) << "\n";
        }
        n.SetType(to_ty);
      }
    }

    return true;
  }

  bool Visit(AST::NamedVariableDecl& n) {
    TraceEachVisit(n);
    auto nty = n.GetType();
    auto nds = n.GetDiversityShape();
    if (Skip(nty, nds)) return true;
    auto init_expr = n.init_expr;
    auto init_ty = init_expr->GetType();

    ptr<Type> vty = nullptr;
    if (!IsActualVectorType(init_ty)) {
      init_expr->AddNote("broadcast",
                         std::to_string(cur_loop->GetVectorFactor()));
      vty = MakeVectorType(init_ty->GetBaseType(), cur_loop->GetVectorFactor());
      if (debug_visit)
        dbgs() << indent << "Expr: " << "broadcast `" << PSTR(init_expr)
               << "`, type: " << PSTR(vty) << "\n";
    } else
      vty = init_ty;

    if (debug_visit)
      dbgs() << indent << "Decl: " << InScopeName(n.name_str) << " = "
             << STR(init_expr) << ", type: " << PSTR(vty) << "\n";
    n.SetType(vty);
    AssignSymVType(n.LOC(), InScopeName(n.name_str), vty);
    return true;
  }

  bool Visit(AST::Call& n) {
    TraceEachVisit(n);
    auto nty = n.GetType();
    auto nds = n.GetDiversityShape();
    if (Skip(nty, nds)) return true;
    bool need_widen = false;
    for (auto arg : n.GetArguments()) {
      if (IsActualVectorType(arg->GetType()) ||
          !arg->GetDiversityShape().Uniform()) {
        need_widen = true;
      }
    }
    if (!need_widen) return true;

    auto vty = nty;
    if (IsScalarType(nty->GetBaseType())) {
      vty = MakeVectorType(nty->GetBaseType(), cur_loop->GetVectorFactor());
      n.SetType(vty);
      n.AddNote("widen", std::to_string(cur_loop->GetVectorFactor()));
    }
    if (debug_visit) {
      dbgs() << indent << "call:  widen `" << n.function->name << "(";
      size_t i = 0;
      for (auto arg : n.GetArguments()) {
        dbgs() << PSTR(arg);
        if (i++ < n.GetArguments().size() - 1) dbgs() << ", ";
      }
      dbgs() << ")`, type: ";
      dbgs() << (IsActualVectorType(vty) ? PSTR(vty) : "void");
      dbgs() << "\n";
    }

    return true;
  }
};

} // namespace Choreo
#endif
