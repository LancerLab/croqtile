#ifndef __CHOREO_DIVERSITY_ANALYSIS_HPP__
#define __CHOREO_DIVERSITY_ANALYSIS_HPP__

#include "ast.hpp"
#include "aux.hpp"
#include "context.hpp"
#include "io.hpp"
#include "loop_utils.hpp"
#include "symbexpr.hpp"
#include "symvals.hpp"
#include "utils.hpp"
#include <string>
#include <unordered_map>

namespace Choreo {
static std::string STR(const DiversityShape& ds) {
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

static ValueItem GetExprVal(const AST::Expr& expr) {
  if (expr.Opts().HasVal()) { return expr.Opts().GetVal(); }
  return UncomputableValueItem();
}

static DiversityShape ComputeDiversityShape(const DiversityShape& lhs,
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
  void Dump(std::ostream& os) const {
    os << "Diversity Shapes:\n";
    for (auto& item : shapes)
      os << "  " << item.first << " : " << STR(item.second) << "\n";
  }
};

struct DiversityAnalysis final : public LoopVisitor {
private:
  ptr<LoopInfo> li;
  ptr<DiversityInfo> di;
  // DiversityShape scope_shape =
  //     DiversityShape(UNIFORM); // shape of the current scope
  std::stack<DiversityShape> scope_shapes;

  bool NeedAnalyze() {
    auto loop = li->GetLoop(lname);
    if (!loop) return false;
    if (AST::NeedVectorize(*loop->loop)) return true;
    return false;
  }

  bool IsDefinedSymbol(const std::string& name) {
    return di->shapes.count(name) > 0;
  }

  void DefineSymbolShape(const std::string& name, const DiversityShape& shape) {
    if (debug_visit)
      dbgs() << "Define shape for symbol `" << name << "`: " << STR(shape)
             << "\n";

    if (di->shapes.count(name)) {
      di->shapes[name] = shape;
    } else {
      di->shapes[name] = shape;
    }
  }

  void ModifySymbolShape(const std::string& name, const DiversityShape& shape) {
    if (debug_visit)
      dbgs() << "Modify shape for symbol `" << name << "`: " << STR(shape)
             << "\n";

    if (di->shapes.count(name)) {
      di->shapes[name] = shape;
    } else {
      choreo_unreachable("symbol `" + name +
                         "` is not defined in diversity analysis.");
    }
  }

  void AssignSymbolShape(const std::string& sym, const DiversityShape& shape) {
    if (!IsDefinedSymbol(sym))
      DefineSymbolShape(sym, shape);
    else {
      auto existing_shape = di->shapes[sym];
      if (existing_shape > shape || existing_shape.ApprxEqual(shape))
        return; // no change
      changed = true;
      ModifySymbolShape(sym, shape);
    }
  }

  DiversityShape GetSymbolShape(const std::string& name) {
    DiversityShape shape;
    if (di->shapes.count(name)) {
      shape = di->shapes[name];
    } else {
      Error1(location(),
             "symbol `" + name + "` is not defined in diversity analysis.");
      shape = DiversityShape(DiversityShapeKind::UNKNOWN);
    }
    return shape;
  }

public:
  bool changed = false;
  DiversityAnalysis(const ptr<SymbolTable> s_tab, ptr<LoopInfo> l,
                    ptr<DiversityInfo> d)
      : LoopVisitor(s_tab, "diversity_analysis"), li(l), di(d) {
    assert(s_tab != nullptr);
  }

  bool Visit(AST::Expr& n) override {
    TraceEachVisit(n);

    if (!NeedAnalyze()) return true;
    DiversityShape shape;
    auto expr_val = GetExprVal(n);
    if (n.IsReference()) {
      if (auto id = AST::GetIdentifier(n)) {
        auto sym_name = InScopeName(id->name);
        if (IsDefinedSymbol(sym_name)) {
          shape = GetSymbolShape(sym_name);
          if (shape.Uniform()) shape.value = expr_val;
        } else {
          shape =
              DiversityShape(DiversityShapeKind::UNIFORM, sbe::nu(0), expr_val);
          AssignSymbolShape(sym_name, shape);
        }
      } else if (auto call = AST::GetCall(n.GetReference())) {
        shape = call->GetDiversityShape();
        shape.value = UncomputableValueItem();
      } else {
        auto ref = n.GetReference();
        auto ref_shape = ref->GetDiversityShape();
        if (ref_shape.Unknown()) {
          shape =
              DiversityShape(DiversityShapeKind::UNIFORM, sbe::nu(0), expr_val);
        } else {
          shape = ref_shape;
          if (shape.Uniform()) shape.value = expr_val;
        }
      }
    } else if (n.IsUnary()) {
      shape = n.GetR()->GetDiversityShape();
      if (shape.Uniform()) shape.value = expr_val;
    } else if (n.IsBinary()) {
      auto lhs_shape = n.GetL()->GetDiversityShape();
      auto rhs_shape = n.GetR()->GetDiversityShape();
      shape = ComputeDiversityShape(lhs_shape, rhs_shape, n.op);
      if (shape.Uniform()) shape.value = expr_val;
    } else if (n.IsTernary()) {
      auto cond_shape = n.GetC()->GetDiversityShape();
      auto lhs_shape = n.GetL()->GetDiversityShape();
      auto rhs_shape = n.GetR()->GetDiversityShape();
      if (cond_shape.Uniform()) {
        shape = ComputeDiversityShape(lhs_shape, rhs_shape);
      } else {
        shape = DiversityShape(DiversityShapeKind::DIVERGENT);
      }
    } else {
      Error1(n.LOC(),
             "unknown expression type in diversity analysis: " + STR(n));
      return false;
    }

    assert(shape.shape != DiversityShapeKind::UNKNOWN);
    if (!n.GetDiversityShape().ApprxEqual(shape)) {
      if (debug_visit)
        dbgs() << "[expr] `" << STR(n) << "` " << STR(n.GetDiversityShape())
               << " -> " << STR(shape) << "\n";
      n.SetDiversityShape(shape);
      changed = true;
    }
    return true;
  }

  bool Visit(AST::NamedVariableDecl& n) override {
    TraceEachVisit(n);
    if (!NeedAnalyze()) return true;
    if (n.IsArray()) {
      Error1(n.LOC(), "array variable declaration in diversity analysis: " +
                          STR(n) + ".");
      assert(false);
    }
    DiversityShape shape =
        dyn_cast<AST::Expr>(n.init_expr)->GetDiversityShape();

    auto scope_shape = scope_shapes.top();
    shape = ComputeDiversityShape(scope_shape, shape);
    AssignSymbolShape(InScopeName(n.name_str), shape);
    return true;
  }

  bool Visit(AST::Identifier& n) override {
    TraceEachVisit(n);
    if (!NeedAnalyze()) return true;

    auto sym_name = InScopeName(n.name);
    if (IsDefinedSymbol(sym_name)) {
      auto shape = GetSymbolShape(sym_name);
      n.SetDiversityShape(shape);
    } else {
      // if not defined
      n.SetDiversityShape(DiversityShape(UNIFORM));
      AssignSymbolShape(sym_name, n.GetDiversityShape());
    }
    return true;
  }

  bool Visit(AST::DataAccess& n) override {
    TraceEachVisit(n);
    if (!NeedAnalyze()) return true;

    if (n.AccessElement()) {
      auto indices = n.GetIndices();
      bool all_uniform = true;
      for (auto& idx : indices) {
        assert(!idx->GetDiversityShape().Unknown() &&
               "index shape should not be unknown.");
        if (!idx->GetDiversityShape().Uniform()) all_uniform = false;
      }
      DiversityShape val_shape = DiversityShape(DiversityShapeKind::DIVERGENT);
      if (all_uniform)
        // if all indices are uniform, we can compute the shape
        val_shape = DiversityShape(UNIFORM);
      n.SetDiversityShape(val_shape);
    } else {
      n.SetDiversityShape(n.data->GetDiversityShape());
    }

    return true;
  }

  bool Visit(AST::Assignment& n) override {
    TraceEachVisit(n);
    if (!NeedAnalyze()) return true;
    auto da = n.da;
    // divergent control flow denpendence
    auto scope_shape = scope_shapes.top();
    auto val_shape =
        ComputeDiversityShape(scope_shape, n.value->GetDiversityShape());
    if (val_shape.Uniform() && isa<AST::Expr>(n.value))
      val_shape.value = GetExprVal(*dyn_cast<AST::Expr>(n.value));

    if (!da->AccessElement()) {
      // example: a = b.at[j];
      auto iv_name = da->GetDataName();

      AssignSymbolShape(InScopeName(iv_name), val_shape);
      da->SetDiversityShape(val_shape);
    } else {
      // example: a.at[i] = b.at[j];
    }

    return true;
  }

  bool Visit(AST::Call& n) override {
    TraceEachVisit(n);
    if (!NeedAnalyze()) return true;

    auto args = n.GetArguments();
    bool all_uniform = true;
    for (auto& arg : args) {
      assert(!arg->GetDiversityShape().Unknown() &&
             "argument shape should not be unknown.");
      if (!arg->GetDiversityShape().Uniform()) all_uniform = false;
    }
    DiversityShape val_shape = DiversityShape(DiversityShapeKind::DIVERGENT);
    if (all_uniform)
      // if all arguments are uniform, we can compute the shape
      val_shape = DiversityShape(UNIFORM);
    n.SetDiversityShape(val_shape);

    return true;
  }

  bool Visit(AST::ForeachBlock& n) override {
    TraceEachVisit(n);
    if (!NeedAnalyze()) return true;
    auto loop = li->GetLoop(lname);
    auto iv_ty = loop->GetIVType();
    auto iv_name = loop->IVName();

    AssignSymbolShape(InScopeName(iv_name),
                      DiversityShape(DiversityShapeKind::STRIDE,
                                     sbe::nu(iv_ty->GetStride())));

    assert(scope_shapes.empty() &&
           "scope_shapes should be empty before entering scoped loop.");
    scope_shapes.push(DiversityShape(DiversityShapeKind::UNIFORM));
    return true;
  }

  bool Visit(AST::IfElseBlock& n) override {
    TraceEachVisit(n);
    if (!NeedAnalyze()) return true;
    auto pred = n.GetPred();
    auto pred_shape = pred->GetDiversityShape();
    auto scope_shape = scope_shapes.top();
    scope_shape = ComputeDiversityShape(scope_shape, pred_shape);
    scope_shapes.push(scope_shape);
    return true;
  }

  bool BeforeAfterVisitImpl(AST::Node& n) override {
    if (NeedAnalyze()) {
      if (isa<AST::IfElseBlock>(&n)) {
        assert(!scope_shapes.empty());
        scope_shapes.pop();
      }
    }
    return true;
  }
};

struct DiversityAnalysisHandler final : public LoopVisitor {
  ptr<LoopInfo> li;
  ptr<DiversityInfo> di;
  DiversityAnalysisHandler(const ptr<SymbolTable> s_tab, ptr<LoopInfo> li)
      : LoopVisitor(s_tab, "diversity"), li(li),
        di(AST::Make<DiversityInfo>()) {}

  bool RunOnProgram(AST::Node& root) override {
    if (!isa<AST::Program>(&root)) {
      Error(root.LOC(), "Not running a choreo program.");
      return false;
    }

    int times = 0;
    bool stop = false;
    while (!stop) {
      DiversityAnalysis da(SymTab(), li, di);
      da.SetDebugVisit(debug_visit);
      da.SetTraceVisit(trace_visit);
      root.accept(da);
      if (da.HasError() || abend_after) return false;
      if (debug_visit) {
        dbgs() << "\n[diversity analysis] iteration " << ++times
               << " finished.\n";
        di->Dump(dbgs());
      }
      stop = !da.changed;
    }

    return true;
  }

  ptr<DiversityInfo> GetDiversityAnalysis() const { return di; }
};
} // namespace Choreo

#endif // __CHOREO_DIVERSITY_ANALYSIS_HPP__