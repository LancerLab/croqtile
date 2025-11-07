#ifndef __CHOREO_DIVERSITY_ANALYSIS_HPP__
#define __CHOREO_DIVERSITY_ANALYSIS_HPP__

#include "ast.hpp"
#include "context.hpp"
#include "infra_utils.hpp"
#include "loop_utils.hpp"
#include "symbexpr.hpp"
#include "symvals.hpp"
#include "visitor.hpp"
#include <ostream>
#include <stack>
#include <string>
#include <unordered_map>
namespace Choreo {

// Refering from "Divergence Analysis
// and Optimizations," doi: 10.1109/PACT.2011.63, a variable v is divergent if,
// and only if, one of these conditions holds:
// 1) v = tid
// 2) v is defined atomically, e.g.: atominc(v, vx).
// 3) v is data dependent on some divergent variable.
// 4) v is sync dependent on some divergent variable.
class DiversityShape;

// Utility functions
inline std::string STR(const DiversityShape& ds) {
  switch (ds.shape) {
  case UNIFORM:
    assert(ds.value);
    if (ds.value->Computable() && ds.value->IsNumeric())
      return "uniform(" + STR(ds.value) + ")";
    else
      return "uniform";
  case STRIDE: return "stride(" + STR(ds.stride) + ")";
  case DIVERGENT: return "divergent";
  case UNKNOWN: return "unknown";
  default: choreo_unreachable("unknown diversity shape kind.");
  }
}

inline ValueItem GetExprVal(const AST::Expr& expr) {
  if (expr.Opts().HasVal()) { return expr.Opts().GetVal(); }
  return UncomputableValueItem();
}

// Compute the diversity shape of two shapes with an operator
// if the op is empty, it means lhs && rhs.
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
    // e.g., 0,2,4,6 + 1, then it is still stride = 2
    if (op == "+" || op == "-")
      return DiversityShape(lhs);
    else if (op == "*") {
      // e.g., 0,2,4,6 * 2, then its stride = 4
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
    // e.g., 0,2,4,6 + 1,3,5,7 = 1,5,9,13, which is still stride = 2 + 2 = 4
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
      shapes;                        // k: scoped symbol name, v: shape
  std::vector<std::string> uni_syms; // symbols that are uniform

  void Dump(std::ostream& os) const;

  bool IsDefinedSymbol(const std::string& name) const {
    return shapes.count(name) > 0;
  }

  bool IsDefiniteUniform(const std::string& name) const {
    return std::find(uni_syms.begin(), uni_syms.end(), name) != uni_syms.end();
  }

  void AddDefiniteUniformSymbol(const std::string& name) {
    if (!IsDefiniteUniform(name)) uni_syms.push_back(name);
  }

  DiversityShape GetSymbolShape(const std::string& name) const {
    if (shapes.count(name)) { return shapes.at(name); }
    return DiversityShape(UNKNOWN);
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

// Diversity shape is deduced according to two dependence:
// 1. data dependence
// 2. cfg dependence
// if (cond)
//    a = b;
// DShape(a) = DShape(cond) && DShape(b);
struct DiversityAnalysis final : public LoopVisitor {
private:
  ptr<LoopInfo> li;
  ptr<DiversityInfo> di;
  std::stack<DiversityShape> scope_shapes;
  std::set<std::string> with_syms; // with symbols defined in with-in blocks
  std::string indent = "    ";

  bool NeedAnalysis();

  // Get the diversity shape of an expression
  DiversityShape ExprDShape(const ptr<AST::Expr> e);

public:
  // when changed is true, we need to re-run the analysis
  bool changed = false;
  DiversityAnalysis(const ptr<SymbolTable> s_tab, ptr<LoopInfo> l,
                    ptr<DiversityInfo> d);

  bool Visit(AST::Expr& n) override;
  bool Visit(AST::CastExpr& n) override;
  bool Visit(AST::NamedVariableDecl& n) override;
  bool Visit(AST::Identifier& n) override;
  bool Visit(AST::DataAccess& n) override;
  bool Visit(AST::Assignment& n) override;
  bool Visit(AST::Call& n) override;
  bool Visit(AST::WithIn& n) override;
  bool Visit(AST::ForeachBlock& n) override;
  bool Visit(AST::IfElseBlock& n) override;
  bool BeforeAfterVisitImpl(AST::Node& n) override;
};

struct DiversityAnalysisHandler final : public LoopVisitor {
  ptr<LoopInfo> li;
  ptr<DiversityInfo> di;

  DiversityAnalysisHandler(const ptr<SymbolTable> s_tab, ptr<LoopInfo> li);
  bool RunOnProgramImpl(AST::Node& root) override;
  ptr<DiversityInfo> GetDiversityAnalysis() const;
};

} // namespace Choreo

#endif // __CHOREO_DIVERSITY_ANALYSIS_HPP__
