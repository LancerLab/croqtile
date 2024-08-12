#ifndef __CHOREO_NORMALIZATION_HPP__
#define __CHOREO_NORMALIZATION_HPP__

// This applies 'normalization' or 'canonicalization' of AST for easier handling
// in later visiting passes.

#include <iostream>

#include "types.hpp"
#include "visitor.hpp"

namespace Choreo {

struct Normalizer : public Visitor {
 private:
  std::ostream &os;
  bool trace = false;

 private:
  bool changed = false;

  std::string old;
  size_t count = 0;  // name suffix of runtime int values

  bool handle_parameter = false;
  ptr<AST::Expr> list_ref = nullptr;
  void SetListReference(const location &l, const std::string &r) {
    list_ref = AST::Make<AST::Expr>(l, AST::Make<AST::Identifier>(l, r));
  }
  void ResetListReference() { list_ref = nullptr; }

  std::string GetInternalValueString() { return "$" + std::to_string(count++); }

 public:
  // it does not require a symbol table
  Normalizer(std::ostream &o)
      : Visitor(nullptr), os(o), trace(std::getenv("TRACE_NORM")) {}

  bool BeforeVisit(AST::Node &n) override {
    if (auto *b = dyn_cast<AST::MultiDimSpans>(&n)) {
      if (b->ref_name != "") {
        SetListReference(n.LOC(), b->ref_name);
        return true;
      }

      if (b->list) return true;
      if (!handle_parameter) return true;

      assert(IsValidRank(b->Rank()));

      // append the node that have multiple dynamic values
      auto mvals = AST::Make<AST::MultiValues>(n.LOC());
      for (size_t i = 0; i < b->Rank(); ++i)
        mvals->Append(AST::Make<AST::IntLiteral>(n.LOC()));
      b->list = mvals;
      changed = true;
    } else if (auto *b = dyn_cast<AST::IntTuple>(&n)) {
      if (b->ref_name != "") SetListReference(n.LOC(), b->ref_name);
    } else if (auto p = dyn_cast<AST::Parameter>(&n)) {
      handle_parameter = true;
      old = AST::STR(*p->type);
      changed = false;
    }
    return true;
  }

  bool AfterVisit(AST::Node &n) override {
    if (auto *b = dyn_cast<AST::MultiDimSpans>(&n)) {
      ResetListReference();
      b->ref_name = "";  // no reference is required
    } else if (isa<AST::IntTuple>(&n)) {
      ResetListReference();
    } else if (auto p = dyn_cast<AST::Parameter>(&n)) {
      handle_parameter = false;
      if (changed && trace)
        os << "Name dims of `" << STR(*p->sym) << "': " << old << " ---> "
           << STR(*p->type) << "\n";
      old.clear();
      changed = false;
    } else if (isa<AST::ChoreoFunction>(&n)) {
      count = 0;
    }
    return true;
  }

  bool Visit(AST::MultiNodes &) override { return true; }

  bool Visit(AST::MultiValues &n) override {
    if (list_ref) {  // desugar the list reference
      for (size_t i = 0; i < n.values.size(); ++i) {
        if (auto expr = dyn_cast<AST::Expr>(n.values[i])) {
          if (auto ref = expr->GetReference()) {
            if (isa<AST::IntIndex>(ref.get())) {
              // apply desugaring a {(0), 1} -> {a(0), 1}
              auto new_expr =
                  AST::Make<AST::Expr>(expr->LOC(), "dimof", list_ref, ref);
              if (trace)
                os << "Desugar ref: " << STR(*expr) << " ---> "
                   << STR(*new_expr) << "\n";
              n.values[i] = new_expr;
            }
          }
        }
      }
    }

    if (handle_parameter) {  // make runtime values of "?" to be named
      for (size_t i = 0; i < n.values.size(); ++i) {
        if (auto il = dyn_cast<AST::IntLiteral>(n.values[i])) {
          if (!IsUnKnownInteger(il->value)) continue;
          auto new_il =
              AST::Make<AST::Identifier>(il->LOC(), GetInternalValueString());

#if 0
          if (trace) {
            il->Print(os);
            os << " --->";
            new_il->Print(os);
            os << "\n";
          }
#endif
          changed = true;
          n.values[i] = new_il;
        }
      }
    }

    return true;
  }

  bool Visit(AST::IntLiteral &) override { return true; }
  bool Visit(AST::Boolean &) override { return true; }
  bool Visit(AST::Expr &n) override {
    if (list_ref) {  // could be with syntax sugar
      auto Apply = [this](AST::Expr *expr) -> ptr<AST::Expr> {
        if (!expr) return nullptr;
        if (auto ref = expr->GetReference()) {
          if (!isa<AST::IntIndex>(ref.get())) return nullptr;

          // apply desugaring a {(0), 1} -> {a(0), 1}
          auto ret = AST::Make<AST::Expr>(expr->LOC(), "dimof", list_ref, ref);

          if (trace) {
            os << "Desugaring expression node: ";
            expr->Print(os);
            os << " --->";
            ret->Print(os);
            os << "\n";
          }

          changed = true;

          return ret;
        }
        return nullptr;
      };

      if (auto new_value = Apply(n.GetC().get())) n.SetC(new_value);
      if (auto lv = dyn_cast<AST::Expr>(n.GetL()))
        if (auto new_value = Apply(lv)) n.SetL(new_value);
      if (auto rv = dyn_cast<AST::Expr>(n.GetR())) {
        if (auto new_value = Apply(rv)) n.SetR(new_value);
      }

      return true;
    }

    return true;
  }

  bool Visit(AST::MultiDimSpans &) override { return true; }
  bool Visit(AST::NamedTypeDecl &) override { return true; }
  bool Visit(AST::NamedVariableDecl &n) override {
    if (n.mem && (n.mem->Get() == Storage::DEFAULT)) {
      // Should this be set by target?
      n.mem->Set(Storage::GLOBAL);

      if (trace)
        os << "Place storage of '" << n.name_str << "': DEFAULT ---> GLOBAL\n";
    }

    return true;
  }
  bool Visit(AST::IntTuple &) override { return true; }
  bool Visit(AST::Assignment &) override { return true; }
  bool Visit(AST::IntIndex &) override { return true; }
  bool Visit(AST::DataType &) override { return true; }
  bool Visit(AST::Identifier &) override { return true; }
  bool Visit(AST::Parameter &) override { return true; }
  bool Visit(AST::ParamList &) override { return true; }
  bool Visit(AST::ParallelBy &) override { return true; }
  bool Visit(AST::WhereBind &) override { return true; }

  bool Visit(AST::WithIn &n) override {
    if (n.with_matchers) return true;
    assert(n.with && "must have with statement.");

    auto wty = n.with->GetType();
    assert(isa<BoundedITupleType>(wty) && "expect a bounded ituple type.");

    // comment to support new feature
    // if (wty->Dims() == 1) return true;

    auto mval = AST::Make<AST::MultiValues>(n.LOC(), ",");
    // fill the with-matchers
    for (size_t i = 0; i < wty->Dims(); ++i) {
      mval->Append(AST::Make<AST::Identifier>(
          n.with->LOC(), n.with->name + "__elem__" + std::to_string(i)));
      auto bity = cast<BoundedITupleType>(wty);
      if (bity->HasValidBound())
        mval->ValueAt(i)->SetType(MakeBoundedIntegerType(bity->GetBound(i)));
      else
        mval->ValueAt(i)->SetType(MakeUnknownBoundedIntegerType());
    }

    n.with_matchers = mval;

    if (trace)
      os << "Generate with-matchers for '" << n.with->name << "': " << STR(mval)
         << "\n";

    return true;
  }

  bool Visit(AST::WithBlock &) override { return true; }
  bool Visit(AST::Memory &) override { return true; }
  bool Visit(AST::DMA &) override { return true; }
  bool Visit(AST::ChunkAt &) override { return true; }
  bool Visit(AST::Wait &) override { return true; }
  bool Visit(AST::Call &) override { return true; }
  bool Visit(AST::Select &) override { return true; }
  bool Visit(AST::Return &) override { return true; }
  bool Visit(AST::ForeachBlock &) override { return true; }
  bool Visit(AST::FunctionDecl &) override { return true; }
  bool Visit(AST::ChoreoFunction &) override { return true; }
  bool Visit(AST::CppSourceCode &) override { return true; }
  bool Visit(AST::Program &) override { return true; }

 private:
  bool SetCurrentType(AST::Node &, const std::string &);
};

}  // end namespace Choreo

#endif  // __CHOREO_NORMALIZATION_HPP__
