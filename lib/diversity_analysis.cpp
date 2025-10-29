#include "diversity_analysis.hpp"

using namespace Choreo;
void DiversityInfo::Dump(std::ostream& os) const {
  os << "[diversity] Diversity Shape:\n";
  for (auto& item : shapes)
    os << "    " << item.first << " : " << STR(item.second) << "\n";
}

DiversityAnalysis::DiversityAnalysis(const ptr<SymbolTable> s_tab,
                                     ptr<LoopInfo> l, ptr<DiversityInfo> d)
    : LoopVisitor(s_tab, "diversity_analysis"), li(l), di(d) {
  assert(s_tab != nullptr);
}

// we only analyze diversity in vectorized loops(loops with vectorization hint)
bool DiversityAnalysis::NeedAnalysis() {
  if (!InLoop()) return false;
  if (li->IsInnermostLoop(cur_loop->LoopName())) return true;
  return false;
}

// compute diversity shape for an expr node
DiversityShape DiversityAnalysis::ExprDShape(const ptr<AST::Expr> e) {
  if (!e) return DiversityShape(UNKNOWN);

  auto expr_val = GetExprVal(*e);
  DiversityShape shape;

  if (e->IsReference()) {
    if (auto id = AST::GetIdentifier(e)) {
      auto sym_name = InScopeName(id->name);
      // if the sym is defined, we use its shape directly
      if (di->IsDefinedSymbol(sym_name)) {
        shape = di->GetSymbolShape(sym_name);
        if (shape.Uniform()) shape.value = expr_val;
      }
      // if not, we use it as UNIFORM, because the identifer is not
      // defined/assigned in the vectorized loop
      else {
        shape = DiversityShape(UNIFORM, sbe::nu(0), expr_val);
        if (di->AssignSymbolShape(sym_name, shape)) changed = true;
      }
    } else if (auto call = AST::GetCall(e->GetReference())) {
      shape = call->GetDiversityShape();
      shape.value = UncomputableValueItem();
    } else {
      auto ref = e->GetReference();
      auto ref_shape = ref->GetDiversityShape();
      if (ref_shape.Unknown()) {
        shape = DiversityShape(UNIFORM, sbe::nu(0), expr_val);
      } else {
        shape = ref_shape;
        if (shape.Uniform()) shape.value = expr_val;
      }
    }
  } else if (e->IsUnary()) {
    shape = e->GetR()->GetDiversityShape();
    if (shape.Uniform()) shape.value = expr_val;
  } else if (e->IsBinary()) {
    if (e->op == "dimof") {
      shape = DiversityShape(UNIFORM, sbe::nu(0), expr_val);
      return shape;
    }
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
      shape = DiversityShape(DIVERGENT);
    }
  } else {
    shape = DiversityShape(UNIFORM, sbe::nu(0), expr_val);
  }

  return shape;
}

bool DiversityAnalysis::Visit(AST::CastExpr& n) {
  TraceEachVisit(n);

  if (!NeedAnalysis()) return true;
  DiversityShape shape = n.GetR()->GetDiversityShape();
  if (shape.Unknown()) shape = DiversityShape(DIVERGENT);
  if (!n.GetDiversityShape().ApprxEqual(shape)) {
    if (debug_visit)
      dbgs() << indent << "cast: `" << STR(n) << "` "
             << STR(n.GetDiversityShape()) << " -> " << STR(shape) << "\n";
    n.SetDiversityShape(shape);
    changed = true;
  }
  return true;
}

bool DiversityAnalysis::Visit(AST::Expr& n) {
  TraceEachVisit(n);

  if (!NeedAnalysis()) return true;
  DiversityShape shape = ExprDShape(AST::Make<AST::Expr>(n));
  if (shape.Unknown()) shape = DiversityShape(DIVERGENT);
  if (!n.GetDiversityShape().ApprxEqual(shape)) {
    if (debug_visit)
      dbgs() << indent << "expr: `" << STR(n) << "` "
             << STR(n.GetDiversityShape()) << " -> " << STR(shape) << "\n";
    n.SetDiversityShape(shape);
    changed = true;
  }
  return true;
}

// a symbol defined in the front lines may be reassigned in later lines.
// like:
//       #1 s32 val = 0;
//       #2 if (cond)
//       #3   val = varying_value;
// val defined in #1 is uniform, but it is reassigned in #3 to be varying_value,
// which is divergent. So we need to update the diversity shape in shape table
// and update this NamedVariableDecl node in next iteration.
bool DiversityAnalysis::Visit(AST::NamedVariableDecl& n) {
  TraceEachVisit(n);
  if (!NeedAnalysis()) {
    auto sym = InScopeName(n.name_str);
    di->AssignSymbolShape(sym, DiversityShape(UNIFORM));
    di->AddDefiniteUniformSymbol(sym);
    return true;
  }
  if (n.IsArray()) {
    Error1(n.LOC(),
           "array variable declaration in diversity analysis: " + STR(n) + ".");
    assert(false);
  }
  DiversityShape shape = n.GetDiversityShape();
  DiversityShape init_shape =
      n.init_expr ? n.init_expr->GetDiversityShape() : DiversityShape(UNIFORM);

  auto name = n.name_str;
  auto sym_shape = di->GetSymbolShape(InScopeName(name));
  if (!sym_shape.Unknown()) {
    // the symbol has been defined in previous iterations.
    shape = sym_shape;
  } else {
    auto scope_shape = scope_shapes.top();
    shape = ComputeDiversityShape(scope_shape, init_shape);
  }
  if (!n.GetDiversityShape().ApprxEqual(shape)) {
    if (debug_visit)
      dbgs() << indent << "decl: `" << n.name_str << " = " << STR(n.init_expr)
             << "` " << STR(n.GetDiversityShape()) << " -> " << STR(shape)
             << "\n";
    changed = true;
    n.SetDiversityShape(shape);
  }
  // assign the shape to symbol table
  if (di->AssignSymbolShape(InScopeName(n.name_str), shape)) changed = true;
  return true;
}

bool DiversityAnalysis::Visit(AST::Identifier& n) {
  TraceEachVisit(n);
  if (!NeedAnalysis()) return true;

  auto sym_name = InScopeName(n.name);
  if (di->IsDefinedSymbol(sym_name)) {
    auto shape = di->GetSymbolShape(sym_name);
    n.SetDiversityShape(shape);
  } else {
    n.SetDiversityShape(DiversityShape(UNIFORM));
    if (di->AssignSymbolShape(sym_name, n.GetDiversityShape())) changed = true;
  }
  return true;
}

bool DiversityAnalysis::Visit(AST::DataAccess& n) {
  TraceEachVisit(n);
  if (!NeedAnalysis()) return true;

  if (n.AccessElement()) {
    auto indices = n.GetIndices();
    bool all_uniform = true;
    for (auto& idx : indices) {
      assert(!idx->GetDiversityShape().Unknown() &&
             "index shape should not be unknown.");
      if (!idx->GetDiversityShape().Uniform()) all_uniform = false;
    }
    DiversityShape val_shape = DiversityShape(DIVERGENT);
    if (all_uniform)
      // if all indices are uniform, we can compute the shape
      val_shape = DiversityShape(UNIFORM);
    n.SetDiversityShape(val_shape);
  } else {
    n.SetDiversityShape(n.data->GetDiversityShape());
  }

  return true;
}

// assignmet may be a declaration, and the lhs of assignment may be a memory
// access.
bool DiversityAnalysis::Visit(AST::Assignment& n) {
  TraceEachVisit(n);
  if (!NeedAnalysis()) {
    if (n.IsDecl() && !n.AssignToDataElement()) {
      auto sym = InScopeName(n.da->GetDataName());
      di->AssignSymbolShape(sym, DiversityShape(UNIFORM));
      di->AddDefiniteUniformSymbol(sym);
    }
    return true;
  }
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
    // if this assignment is a declaration, we need to consider the symbol shape
    // computed in previous iterations.
    if (n.IsDecl()) {
      auto sym_shape = di->GetSymbolShape(InScopeName(iv_name));
      auto shape = val_shape;
      if (!sym_shape.Unknown() && sym_shape > val_shape) { shape = sym_shape; }
      da->SetDiversityShape(shape);
      n.SetDiversityShape(shape);
      return true;
    }
    // if this assignment is a plain assignment, the symbol must be defined
    // before.
    if (di->IsDefinedSymbol(InScopeName(iv_name))) {
      if (di->AssignSymbolShape(InScopeName(iv_name), val_shape)) {
        if (debug_visit)
          dbgs() << indent << "asgn: `" << PSTR(n.da) << " = " << PSTR(n.value)
                 << "`, diversity: " << STR(val_shape) << "\n";
        changed = true;
      }

      da->SetDiversityShape(val_shape);
      n.SetDiversityShape(val_shape);
    } else {
      Error1(n.LOC(),
             "The symbol `" + iv_name +
                 "' has not been defined before assignment: " + STR(n) + ".");
      assert(false);
    }
  } else {
    // example: a.at[0] = varying_value;
    auto da_shape = da->GetDiversityShape();
    auto val_shape = n.value->GetDiversityShape();
    auto asgn_shape = ComputeDiversityShape(da_shape, val_shape);
    n.SetDiversityShape(asgn_shape);
  }

  return true;
}

bool DiversityAnalysis::Visit(AST::Call& n) {
  TraceEachVisit(n);
  if (!NeedAnalysis()) return true;

  auto args = n.GetArguments();
  bool all_uniform = true;
  for (auto& arg : args) {
    assert(!arg->GetDiversityShape().Unknown() &&
           "argument shape should not be unknown.");
    if (!arg->GetDiversityShape().Uniform()) all_uniform = false;
  }
  DiversityShape val_shape = DiversityShape(DIVERGENT);
  if (all_uniform)
    // if all arguments are uniform, we can compute the shape
    val_shape = DiversityShape(UNIFORM);
  n.SetDiversityShape(val_shape);

  return true;
}

bool DiversityAnalysis::Visit(AST::WithIn& n) {
  TraceEachVisit(n);
  if (n.with) { with_syms.insert(InScopeName(n.with->name)); }
  return true;
}

bool DiversityAnalysis::Visit(AST::ForeachBlock& n) {
  TraceEachVisit(n);
  if (!NeedAnalysis()) return true;
  auto lname = SSTab().ScopeName();
  auto loop = n.loop;
  auto iv_ty = loop->GetIVType();
  auto iv_sym = loop->IVSym();
  int stride = 1;
  assert(IsActualBoundedIntegerType(iv_ty));

  if (auto bit = dyn_cast<BoundedIntegerType>(iv_ty)) {
    stride = bit->GetStride();
  } else if (auto bit = dyn_cast<BoundedITupleType>(iv_ty)) {
    stride = bit->GetStride(0);
  }

  if (di->AssignSymbolShape(iv_sym, DiversityShape(STRIDE, sbe::nu(stride)))) {
    changed = true;
  }

  bool with_found = false;
  std::string with_sym;
  for (auto item : within_map) {
    if (with_syms.count(item.first) == 0) continue;
    auto ivs = item.second;
    if (ivs[ivs.size() - 1] == iv_sym) {
      with_found = true;
      with_sym = item.first;
      break;
    }
  }

  if (with_found) {
    if (di->AssignSymbolShape(with_sym,
                              DiversityShape(STRIDE, sbe::nu(stride)))) {
      changed = true;
    }
  }

  assert(scope_shapes.empty() &&
         "scope_shapes should be empty before entering scoped loop.");
  scope_shapes.push(DiversityShape(UNIFORM));
  return true;
}

bool DiversityAnalysis::Visit(AST::IfElseBlock& n) {
  TraceEachVisit(n);
  if (!NeedAnalysis()) return true;
  auto pred = n.GetPred();
  auto pred_shape = pred->GetDiversityShape();
  auto scope_shape = scope_shapes.top();
  scope_shape = ComputeDiversityShape(scope_shape, pred_shape);
  scope_shapes.push(scope_shape);
  return true;
}

bool DiversityAnalysis::BeforeAfterVisitImpl(AST::Node& n) {
  if (NeedAnalysis()) {
    if (isa<AST::IfElseBlock>(&n)) {
      assert(!scope_shapes.empty());
      scope_shapes.pop();
    } else if (isa<AST::ForeachBlock>(&n)) {
      assert(!scope_shapes.empty());
      scope_shapes.pop();
      assert(scope_shapes.empty() &&
             "scope_shapes should be empty after exiting scoped loop.");
    }
  }
  return true;
}

DiversityAnalysisHandler::DiversityAnalysisHandler(const ptr<SymbolTable> s_tab,
                                                   ptr<LoopInfo> li)
    : LoopVisitor(s_tab, "diversity"), li(li), di(AST::Make<DiversityInfo>()) {}

bool DiversityAnalysisHandler::RunOnProgramImpl(AST::Node& root) {
  if (!isa<AST::Program>(&root)) {
    Error(root.LOC(), "Not running a choreo program.");
    return false;
  }

  // it may take multiple iterations to reach a fixed point
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
