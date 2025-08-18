#ifndef __CHOREO_DIVERSITY_ANALYSIS_HPP__
#define __CHOREO_DIVERSITY_ANALYSIS_HPP__

#include "ast.hpp"
#include "context.hpp"
#include "loop_utils.hpp"
#include "symbexpr.hpp"
#include "symvals.hpp"
#include "utils.hpp"
#include <ostream>
#include <stack>
#include <string>
#include <unordered_map>
namespace Choreo {

// Forward declarations
class DiversityShape;

// Utility functions
inline std::string STR(const DiversityShape& ds) {
  switch (ds.shape) {
  case DiversityShapeKind::UNIFORM:
    assert(ds.value);
    if (ds.value->Computable() && ds.value->IsNumeric())
      return "uniform(" + STR(ds.value) + ")";
    else
      return "uniform";
  case DiversityShapeKind::STRIDE: return "stride(" + STR(ds.stride) + ")";
  case DiversityShapeKind::DIVERGENT: return "divergent";
  case DiversityShapeKind::UNKNOWN: return "unknown";
  default: choreo_unreachable("unknown diversity shape kind.");
  }
}

inline ValueItem GetExprVal(const AST::Expr& expr) {
  if (expr.Opts().HasVal()) { return expr.Opts().GetVal(); }
  return UncomputableValueItem();
}

inline DiversityShape ComputeDiversityShape(const DiversityShape& lhs,
                                            const DiversityShape& rhs,
                                            std::string op = "") {
  using Kind = DiversityShapeKind;
  assert(lhs.shape != Kind::UNKNOWN && rhs.shape != Kind::UNKNOWN &&
         "cannot compute diversity shape with unknown shape.");
  if (lhs.Uniform() && rhs.Uniform()) return DiversityShape(Kind::UNIFORM);
  if (op.empty()) return lhs < rhs ? rhs : lhs;

  if (lhs.Divergent() || rhs.Divergent())
    return DiversityShape(Kind::DIVERGENT);

  if (lhs.Stride() && rhs.Uniform()) {
    if (op == "+" || op == "-")
      return DiversityShape(lhs);
    else if (op == "*") {
      if (IsValidValueItem(lhs.stride) && IsValidValueItem(rhs.value)) {
        auto stride = lhs.stride * rhs.value;
        return DiversityShape(Kind::STRIDE, stride);
      } else
        return DiversityShape(Kind::DIVERGENT);
    } else if (op == "/") {
      if (rhs.value == 0)
        choreo_unreachable("division by zero in diversity shape.");
      if (IsValidValueItem(lhs.stride) && IsValidValueItem(rhs.value)) {
        auto stride = lhs.stride / rhs.value;
        return DiversityShape(Kind::STRIDE, stride);
      } else
        return DiversityShape(Kind::DIVERGENT);
    } else
      return DiversityShape(Kind::DIVERGENT);
  } else if (lhs.Uniform() && rhs.Stride()) {
    if (op == "+" || op == "-")
      return DiversityShape(rhs);
    else if (op == "*") {
      if (IsValidValueItem(lhs.value) && IsValidValueItem(rhs.stride)) {
        auto stride = lhs.value * rhs.stride;
        return DiversityShape(Kind::STRIDE, stride);
      } else
        return DiversityShape(Kind::DIVERGENT);
    } else if (op == "/") {
      if (lhs.value == 0)
        choreo_unreachable("division by zero in diversity shape.");
      if (IsValidValueItem(lhs.value) && IsValidValueItem(rhs.stride)) {
        auto stride = lhs.value / rhs.stride;
        return DiversityShape(Kind::STRIDE, stride);
      } else
        return DiversityShape(Kind::DIVERGENT);
    } else
      return DiversityShape(Kind::DIVERGENT);
  } else if (lhs.Stride() && rhs.Stride()) {
    if (IsValidValueItem(lhs.stride) && IsValidValueItem(rhs.stride)) {
      if (op == "+")
        return DiversityShape(Kind::STRIDE, lhs.stride + rhs.stride);
      else if (op == "-")
        return DiversityShape(Kind::STRIDE, lhs.stride - rhs.stride);
      return DiversityShape(Kind::DIVERGENT);
    } else
      return DiversityShape(Kind::DIVERGENT);
  }
  return DiversityShape(Kind::UNKNOWN);
}

struct DiversityInfo {
  std::unordered_map<std::string, DiversityShape>
      shapes; // k: scoped symbol name, v: shape

  void Dump(std::ostream& os) const;

  bool IsDefinedSymbol(const std::string& name) const {
    return shapes.count(name) > 0;
  }

  DiversityShape GetSymbolShape(const std::string& name) const {
    if (shapes.count(name)) { return shapes.at(name); }
    return DiversityShape(DiversityShapeKind::UNKNOWN);
  }

  void DefineSymbolShape(const std::string& name, const DiversityShape& shape) {
    shapes[name] = shape;
  }

  void ModifySymbolShape(const std::string& name, const DiversityShape& shape) {
    if (shapes.count(name)) { shapes[name] = shape; }
  }

  bool AssignSymbolShape(const std::string& sym, const DiversityShape& shape) {
    if (!IsDefinedSymbol(sym)) {
      DefineSymbolShape(sym, shape);
      return true;
    } else {
      auto existing_shape = GetSymbolShape(sym);
      if (existing_shape > shape || existing_shape.ApprxEqual(shape))
        return false; // no change
      ModifySymbolShape(sym, shape);
      return true; // changed
    }
  }
};

inline DiversityShape ExprDShape(const ptr<AST::Expr> e, ptr<DiversityInfo>) {
  if (!e) return DiversityShape(DiversityShapeKind::UNKNOWN);

  auto expr_val = GetExprVal(*e);
  DiversityShape shape;

  if (e->IsReference()) {
    if (AST::GetIdentifier(*e)) {
      // For identifier references, we need the symbol table context
      // This is a simplified version - in practice would need symbol lookup
      shape = DiversityShape(DiversityShapeKind::UNIFORM, sbe::nu(0), expr_val);
    } else if (auto call = AST::GetCall(e->GetReference())) {
      shape = call->GetDiversityShape();
      shape.value = UncomputableValueItem();
    } else {
      auto ref = e->GetReference();
      auto ref_shape = ref->GetDiversityShape();
      if (ref_shape.Unknown()) {
        shape =
            DiversityShape(DiversityShapeKind::UNIFORM, sbe::nu(0), expr_val);
      } else {
        shape = ref_shape;
        if (shape.Uniform()) shape.value = expr_val;
      }
    }
  } else if (e->IsUnary()) {
    shape = e->GetR()->GetDiversityShape();
    if (shape.Uniform()) shape.value = expr_val;
  } else if (e->IsBinary()) {
    auto lhs_shape = e->GetL()->GetDiversityShape();
    auto rhs_shape = e->GetR()->GetDiversityShape();
    shape = ComputeDiversityShape(lhs_shape, rhs_shape, e->op);
    if (shape.Uniform()) shape.value = expr_val;
  } else if (e->IsTernary()) {
    auto cond_shape = e->GetC()->GetDiversityShape();
    auto lhs_shape = e->GetL()->GetDiversityShape();
    auto rhs_shape = e->GetR()->GetDiversityShape();
    if (cond_shape.Uniform()) {
      shape = ComputeDiversityShape(lhs_shape, rhs_shape);
    } else {
      shape = DiversityShape(DiversityShapeKind::DIVERGENT);
    }
  } else {
    shape = DiversityShape(DiversityShapeKind::UNIFORM, sbe::nu(0), expr_val);
  }

  return shape;
}

struct DiversityAnalysis final : public LoopVisitor {
private:
  ptr<LoopInfo> li;
  ptr<DiversityInfo> di;
  std::stack<DiversityShape> scope_shapes;

  bool NeedAnalyze();

public:
  bool changed = false;
  DiversityAnalysis(const ptr<SymbolTable> s_tab, ptr<LoopInfo> l,
                    ptr<DiversityInfo> d);

  bool Visit(AST::Expr& n) override;
  bool Visit(AST::NamedVariableDecl& n) override;
  bool Visit(AST::Identifier& n) override;
  bool Visit(AST::DataAccess& n) override;
  bool Visit(AST::Assignment& n) override;
  bool Visit(AST::Call& n) override;
  bool Visit(AST::ForeachBlock& n) override;
  bool Visit(AST::IfElseBlock& n) override;
  bool BeforeAfterVisitImpl(AST::Node& n) override;
};

struct DiversityAnalysisHandler final : public LoopVisitor {
  ptr<LoopInfo> li;
  ptr<DiversityInfo> di;

  DiversityAnalysisHandler(const ptr<SymbolTable> s_tab, ptr<LoopInfo> li);
  bool RunOnProgram(AST::Node& root) override;
  ptr<DiversityInfo> GetDiversityAnalysis() const;
};

} // namespace Choreo

#endif // __CHOREO_DIVERSITY_ANALYSIS_HPP__