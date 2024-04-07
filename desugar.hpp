#ifndef __CHOREO_DESUGARING_HPP__
#define __CHOREO_DESUGARING_HPP__

#include <iostream>

#include "types.hpp"
#include "visitor.hpp"

namespace Choreo {

struct DeSugaring : public Visitor {
 private:
  bool trace = false;
  ptr<AST::Expr> list_ref = nullptr;
  void SetListReference(const location &l, const std::string &r) {
    list_ref = AST::Make<AST::Expr>(l, AST::Make<AST::Identifier>(l, r));
  }
  void ResetListReference() { list_ref = nullptr; }

 public:
  DeSugaring() : trace(std::getenv("TRACE_DESUGAR")) {}

  bool BeforeVisit(AST::Node &n) override {
    if (auto *b = dyn_cast<AST::MultiDimSpans>(&n)) {
      if (b->ref_name != "") SetListReference(n.LOC(), b->ref_name);
    } else if (auto *b = dyn_cast<AST::IntTuple>(&n)) {
      if (b->ref_name != "") SetListReference(n.LOC(), b->ref_name);
    }
    return true;
  }

  bool AfterVisit(AST::Node &n) override {
    if (isa<AST::MultiDimSpans>(&n) || isa<AST::IntTuple>(&n)) {
      ResetListReference();
    }
    return true;
  }

  bool Visit(AST::MultiNodes &) override { return true; }

  bool Visit(AST::MultiValues &n) override {
    if (!list_ref) return true;  // no syntax sugar

    ptr<AST::Expr> new_expr = nullptr;
    size_t i = 0;
    for (; i < n.values.size(); ++i) {
      auto pv = n.values[i];
      if (auto expr = dyn_cast<AST::Expr>(pv.get())) {
        if (auto ref = expr->GetReference()) {
          if (!isa<AST::IntIndex>(ref.get())) continue;

          // apply desugaring a {(0), 1} -> {a(0), 1}
          new_expr = AST::Make<AST::Expr>(expr->LOC(), "dimof", list_ref, ref);

          if (trace) {
            std::cout << "Desugaring node: ";
            expr->Print(std::cout);
            std::cout << " --->";
            new_expr->Print(std::cout);
            std::cout << "\n";
          }

          break;
        }
      }
    }

    if (new_expr) n.values[i] = new_expr;

    return true;
  }
  bool Visit(AST::IntLiteral &) override { return true; }
  bool Visit(AST::Expr &n) override {
    if (!list_ref) return true;  // no syntax sugar

    auto Apply = [this](AST::Expr *expr) -> ptr<AST::Expr> {
      if (!expr) return nullptr;
      if (auto ref = expr->GetReference()) {
        if (!isa<AST::IntIndex>(ref.get())) return nullptr;

        // apply desugaring a {(0), 1} -> {a(0), 1}
        auto ret = AST::Make<AST::Expr>(expr->LOC(), "dimof", list_ref, ref);

        if (trace) {
          std::cout << "Desugaring expression node: ";
          expr->Print(std::cout);
          std::cout << " --->";
          ret->Print(std::cout);
          std::cout << "\n";
        }

        return ret;
      }
      return nullptr;
    };

    if (auto new_value = Apply(n.value_c.get())) n.value_c = new_value;
    if (auto new_value = Apply(n.value_l.get())) n.value_l = new_value;
    if (isa<AST::Expr>(n.value_r.get())) {
      if (auto new_value = Apply(cast<AST::Expr>(n.value_r.get())))
        n.value_r = new_value;
    }

    return true;
  }
  bool Visit(AST::MultiDimSpans &) override { return true; }
  bool Visit(AST::NamedTypeDecl &) override { return true; }
  bool Visit(AST::NamedVariableDecl &) override { return true; }
  bool Visit(AST::IntTuple &) override { return true; }
  bool Visit(AST::Assignment &) override { return true; }
  bool Visit(AST::IntIndex &) override { return true; }
  bool Visit(AST::DataType &) override { return true; }
  bool Visit(AST::Identifier &) override { return true; }
  bool Visit(AST::Parameter &) override { return true; }
  bool Visit(AST::ParamList &) override { return true; }
  bool Visit(AST::ParallelBy &) override { return true; }
  bool Visit(AST::RequireBind &) override { return true; }
  bool Visit(AST::WithIn &) override { return true; }
  bool Visit(AST::WithBlock &) override { return true; }
  bool Visit(AST::Memory &) override { return true; }
  bool Visit(AST::DMA &) override { return true; }
  bool Visit(AST::ChunkAt &) override { return true; }
  bool Visit(AST::Wait &) override { return true; }
  bool Visit(AST::Call &) override { return true; }
  bool Visit(AST::ForeachBlock &) override { return true; }
  bool Visit(AST::FunctionDecl &) override { return true; }
  bool Visit(AST::ChoreoFunction &) override { return true; }
  bool Visit(AST::CppSourceCode &) override { return true; }
  bool Visit(AST::Program &) override { return true; }

 private:
  bool SetCurrentType(AST::Node &, const std::string &);
};

}  // end namespace Choreo

#endif  // __CHOREO_DESUGARING_HPP__
