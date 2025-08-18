#include "diversity_analysis.hpp"
#include "io.hpp"

using namespace Choreo;
void DiversityInfo::Dump(std::ostream& os) const {
  os << "[diversity] Symbol Shape:\n";
  for (auto& item : shapes)
    os << "  " << item.first << " : " << STR(item.second) << "\n";
}

DiversityAnalysis::DiversityAnalysis(const ptr<SymbolTable> s_tab,
                                     ptr<LoopInfo> l, ptr<DiversityInfo> d)
    : LoopVisitor(s_tab, "diversity_analysis"), li(l), di(d) {
  assert(s_tab != nullptr);
}

bool DiversityAnalysis::NeedAnalyze() {
  auto loop = li->GetLoop(lname);
  if (!loop) return false;
  if (AST::NeedVectorize(*loop->loop)) return true;
  return false;
}

bool DiversityAnalysis::Visit(AST::Expr& n) {
  TraceEachVisit(n);

  if (!NeedAnalyze()) return true;

  DiversityShape shape;
  auto expr_val = GetExprVal(n);

  if (n.IsReference()) {
    if (auto id = AST::GetIdentifier(n)) {
      auto sym_name = InScopeName(id->name);
      if (di->IsDefinedSymbol(sym_name)) {
        shape = di->GetSymbolShape(sym_name);
        if (shape.Uniform()) shape.value = expr_val;
      } else {
        shape =
            DiversityShape(DiversityShapeKind::UNIFORM, sbe::nu(0), expr_val);
        if (di->AssignSymbolShape(sym_name, shape)) changed = true;
      }
    } else {
      // Use ExprDShape for other reference types
      shape = ExprDShape(AST::Make<AST::Expr>(n), di);
    }
  } else {
    // Use ExprDShape for non-reference expressions
    shape = ExprDShape(AST::Make<AST::Expr>(n), di);
  }

  assert(shape.shape != DiversityShapeKind::UNKNOWN);
  if (!n.GetDiversityShape().ApprxEqual(shape)) {
    if (debug_visit)
      dbgs() << "[diversity] [expr] `" << STR(n) << "` "
             << STR(n.GetDiversityShape()) << " -> " << STR(shape) << "\n";
    n.SetDiversityShape(shape);
    changed = true;
  }
  return true;
}

bool DiversityAnalysis::Visit(AST::NamedVariableDecl& n) {
  TraceEachVisit(n);
  if (!NeedAnalyze()) return true;
  if (n.IsArray()) {
    Error1(n.LOC(),
           "array variable declaration in diversity analysis: " + STR(n) + ".");
    assert(false);
  }
  DiversityShape shape = dyn_cast<AST::Expr>(n.init_expr)->GetDiversityShape();

  auto scope_shape = scope_shapes.top();
  shape = ComputeDiversityShape(scope_shape, shape);
  if (di->AssignSymbolShape(InScopeName(n.name_str), shape)) changed = true;
  return true;
}

bool DiversityAnalysis::Visit(AST::Identifier& n) {
  TraceEachVisit(n);
  if (!NeedAnalyze()) return true;

  auto sym_name = InScopeName(n.name);
  if (di->IsDefinedSymbol(sym_name)) {
    auto shape = di->GetSymbolShape(sym_name);
    n.SetDiversityShape(shape);
  } else {
    // if not defined
    n.SetDiversityShape(DiversityShape(UNIFORM));
    if (di->AssignSymbolShape(sym_name, n.GetDiversityShape())) changed = true;
  }
  return true;
}

bool DiversityAnalysis::Visit(AST::DataAccess& n) {
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

bool DiversityAnalysis::Visit(AST::Assignment& n) {
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

    if (di->AssignSymbolShape(InScopeName(iv_name), val_shape)) changed = true;
    da->SetDiversityShape(val_shape);
  } else {
    // example: a.at[i] = b.at[j];
  }

  return true;
}

bool DiversityAnalysis::Visit(AST::Call& n) {
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

bool DiversityAnalysis::Visit(AST::ForeachBlock& n) {
  TraceEachVisit(n);
  if (!NeedAnalyze()) return true;
  auto loop = li->GetLoop(lname);
  auto iv_ty = loop->GetIVType();
  auto iv_name = loop->IVName();

  if (di->AssignSymbolShape(InScopeName(iv_name),
                            DiversityShape(DiversityShapeKind::STRIDE,
                                           sbe::nu(iv_ty->GetStride())))) {
    changed = true;
  }

  assert(scope_shapes.empty() &&
         "scope_shapes should be empty before entering scoped loop.");
  scope_shapes.push(DiversityShape(DiversityShapeKind::UNIFORM));
  return true;
}

bool DiversityAnalysis::Visit(AST::IfElseBlock& n) {
  TraceEachVisit(n);
  if (!NeedAnalyze()) return true;
  auto pred = n.GetPred();
  auto pred_shape = pred->GetDiversityShape();
  auto scope_shape = scope_shapes.top();
  scope_shape = ComputeDiversityShape(scope_shape, pred_shape);
  scope_shapes.push(scope_shape);
  return true;
}

bool DiversityAnalysis::BeforeAfterVisitImpl(AST::Node& n) {
  if (NeedAnalyze()) {
    if (isa<AST::IfElseBlock>(&n)) {
      assert(!scope_shapes.empty());
      scope_shapes.pop();
    }
  }
  return true;
}

DiversityAnalysisHandler::DiversityAnalysisHandler(const ptr<SymbolTable> s_tab,
                                                   ptr<LoopInfo> li)
    : LoopVisitor(s_tab, "diversity"), li(li), di(AST::Make<DiversityInfo>()) {}

bool DiversityAnalysisHandler::RunOnProgram(AST::Node& root) {
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
    if (debug_visit && da.changed) {
      dbgs() << "[diversity] iteration " << ++times << " finished.\n";
      di->Dump(dbgs());
      dbgs() << "\n";
    }
    stop = !da.changed;
  }

  return true;
}

ptr<DiversityInfo> DiversityAnalysisHandler::GetDiversityAnalysis() const {
  return di;
}
