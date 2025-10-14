#include "scalar_evolution.hpp"

namespace Choreo {

bool ScalarEvolutionAnalysis::InLoop() { return cur_loop != nullptr; }

// op is one of "+", "-", "*", "/"
ptr<SCEV> ScalarEvolutionAnalysis::ComputeARSCEV(ptr<SCEV> lhs, ptr<SCEV> rhs,
                                                 std::string op) {
  if (isa<SCEVAddRecExpr>(lhs) && isa<SCEVVal>(rhs)) {
    auto lhs_ar = dyn_cast<SCEVAddRecExpr>(lhs);
    auto rhs_val = dyn_cast<SCEVVal>(rhs);
    auto ar_loop = lhs_ar->loop;
    if (op == "+" || op == "-") {
      if (rhs_val->IsLoopInVariant(ar_loop)) {
        auto new_base = ComputeARSCEV(lhs_ar->base, rhs_val, op);
        return MakeSCEVAddRecExpr(new_base, lhs_ar->step, ar_loop);
      } else
        return nullptr;
    } else if (op == "*" || op == "/") {
      auto new_base = ComputeARSCEV(lhs_ar->base, rhs_val, op);
      auto new_step = ComputeARSCEV(lhs_ar->step, rhs_val, op);
      return MakeSCEVAddRecExpr(new_base, new_step, ar_loop);
    } else
      return nullptr;
  } else if (isa<SCEVVal>(lhs) && isa<SCEVAddRecExpr>(rhs)) {
    auto lhs_val = dyn_cast<SCEVVal>(lhs);
    auto rhs_ar = dyn_cast<SCEVAddRecExpr>(rhs);
    auto ar_loop = rhs_ar->loop;
    if (op == "+" || op == "-") {
      if (lhs_val->IsLoopInVariant(ar_loop)) {
        auto new_base = ComputeARSCEV(lhs_val, rhs_ar->base, op);
        return MakeSCEVAddRecExpr(new_base, rhs_ar->step, ar_loop);
      } else
        return nullptr;
    } else if (op == "*" || op == "/") {
      auto new_base = ComputeARSCEV(lhs_val, rhs_ar->base, op);
      auto new_step = ComputeARSCEV(lhs_val, rhs_ar->step, op);
      return MakeSCEVAddRecExpr(new_base, new_step, ar_loop);
    } else
      return nullptr;
  } else if (isa<SCEVAddRecExpr>(lhs) && isa<SCEVAddRecExpr>(rhs)) {
    auto lhs_ar = dyn_cast<SCEVAddRecExpr>(lhs);
    auto rhs_ar = dyn_cast<SCEVAddRecExpr>(rhs);
    if (op == "+" || op == "-") {
      if (lhs_ar->loop == rhs_ar->loop) {
        auto new_base = ComputeARSCEV(lhs_ar->base, rhs_ar->base, op);
        auto new_step = ComputeARSCEV(lhs_ar->step, rhs_ar->step, op);
        return MakeSCEVAddRecExpr(new_base, new_step, lhs_ar->loop);
      }
      if (lhs_ar->loop->HasLoop(rhs_ar->loop->loop_name)) {
        // {a, +, b} <L1> op {c, +, d} <L2> , L2 is nested in L1
        // we treat {c, +, d} <L2> as invariant in L1, we get {{a, +, b} <L1>
        // op c, +, b} <L2>
        auto new_base = ComputeARSCEV(lhs_ar, rhs_ar->base, op);
        return MakeSCEVAddRecExpr(new_base, rhs_ar->step, rhs_ar->loop);
      } else if (rhs_ar->loop->HasLoop(lhs_ar->loop->loop_name)) {
        // {a, +, b} <L1> op {c, +, d} <L2> , L1 is nested in L2
        // we treat {a, +, b} <L1> as invariant in L2, we get {{c, +, d} <L2>
        // op a, +, d} <L1>
        auto new_base = ComputeARSCEV(lhs_ar->base, rhs_ar, op);
        return MakeSCEVAddRecExpr(new_base, lhs_ar->step, lhs_ar->loop);
      } else {
        choreo_unreachable("invalid AddRecExprs in different loops.");
      }
    } else if (op == "*" || op == "/") {
      // we do not support multiply/divide two AddRec right now
      return nullptr;
    } else
      return nullptr;
  } else if (isa<SCEVVal>(lhs) && isa<SCEVVal>(rhs)) {
    auto lhs_val = dyn_cast<SCEVVal>(lhs);
    auto rhs_val = dyn_cast<SCEVVal>(rhs);
    if (op == "+") {
      auto new_val = lhs_val->value + rhs_val->value;
      return MakeSCEVVal(new_val);
    } else if (op == "-") {
      auto new_val = lhs_val->value - rhs_val->value;
      return MakeSCEVVal(new_val);
    } else if (op == "*") {
      auto new_val = lhs_val->value * rhs_val->value;
      return MakeSCEVVal(new_val);
    } else if (op == "/") {
      auto new_val = lhs_val->value / rhs_val->value;
      return MakeSCEVVal(new_val);
    }
  }
  return nullptr;
}

bool ScalarEvolutionAnalysis::Visit(AST::Expr& n) {
  TraceEachVisit(n);
  if (!NeedAnalyze(n.GetType())) return true;
  auto valno = n.Opts().HasVal() ? n.Opts().GetVal() : sbe::sym(STR(n));
  auto scev_val = MakeSCEVVal(valno);
  auto loop_name = InLoop() ? LoopName() : NoLoopName();
  auto op = n.op;
  if (n.IsReference()) {
    if (auto id = AST::GetIdentifier(n)) {
      auto iv_name = id->name;
      auto iv_sym = SymName(iv_name);
      if (!IsAssignedSym(iv_sym)) {
        AssignSCEVToSym(SymName(iv_name), scev_val, loop_name);
        n.SetSCEV(scev_val);
      } else {
        auto scev = GetSCEVOfSym(iv_sym);
        n.SetSCEV(scev);
      }
    } else {
      // for other reference types, we just make it a sym
      n.SetSCEV(scev_val);
    }
  } else if (op == "dimof") {
    n.SetSCEV(scev_val);
  } else if (n.IsBinary()) {
    auto lhs = cast<AST::Expr>(n.GetL());
    auto rhs = cast<AST::Expr>(n.GetR());
    auto lhs_ty = lhs->GetType();
    auto rhs_ty = rhs->GetType();
    auto lhs_scev = lhs->GetSCEV();
    auto rhs_scev = rhs->GetSCEV();
    if (!isa<SCEVAddRecExpr>(lhs_scev) && !isa<SCEVAddRecExpr>(rhs_scev)) {
      // both sides are not AddRec, we just make it a sym
      n.SetSCEV(scev_val);
    } else {
      auto bin_scev = ComputeARSCEV(lhs_scev, rhs_scev, n.op);
      n.SetSCEV(bin_scev ? bin_scev : scev_val);
    }
  } else {
    choreo_unreachable("unsupported expr in scalar evolution: " + STR(n) + ".");
  }

  return true;
}

bool ScalarEvolutionAnalysis::Visit(AST::NamedVariableDecl& n) {
  TraceEachVisit(n);
  if (!NeedAnalyze(n.GetType())) return true;

  auto init_expr = dyn_cast<AST::Expr>(n.init_expr);
  assert(init_expr && "Only Expr can be init_expr of NamedVariableDecl.");
  // named variable declaration outside any loop
  auto iv_name = n.name_str;
  auto init_scev = init_expr->GetSCEV();
  if (debug_visit)
    dbgs() << "decl:  `" << iv_name << "` -> " << STR(init_scev) << "\n";
  AssignSCEVToSym(SymName(iv_name), init_scev, LoopName());

  return true;
}

bool ScalarEvolutionAnalysis::Visit(AST::Identifier& n) {
  TraceEachVisit(n);
  return true;
}

bool ScalarEvolutionAnalysis::Visit(AST::Assignment& n) {
  TraceEachVisit(n);
  if (!NeedAnalyze(n.GetType())) return true;
  if (n.da->AccessElement()) return true;
  auto name = n.GetName();
  auto sym_name = SymName(name);
  if (!IsAssignedSym(sym_name)) {
    // if this aassignment is a variable declaration in inner scope, we should
    // regard it as a named variable declaration
    auto expr = dyn_cast<AST::Expr>(n.value);
    assert(expr && "Only Expr can be rhs of Assignment.");
    auto expr_scev = expr->GetSCEV();
    if (debug_visit)
      dbgs() << "asgn:  `" << name << "` -> " << STR(expr_scev) << "\n";
    AssignSCEVToSym(sym_name, expr_scev, LoopName());
  } else {
    // else, we will invalidate the scev of the symbol this assignment assigns
    // to, since we cannot track the scev for re-assignment variables.
    if (debug_visit)
      dbgs() << "asgn:  `" << name
             << "` is re-assigned, invalidate its scev.\n";
    AssignSCEVToSym(sym_name, nullptr, LoopName());
  }
  return true;
}

bool ScalarEvolutionAnalysis::Visit(AST::Call& n) {
  TraceEachVisit(n);
  return true;
}

bool ScalarEvolutionAnalysis::Visit(AST::ForeachBlock& n) {
  TraceEachVisit(n);
  cur_loop = n.loop;
  int vector_width = 1;
  if (cur_loop->NeedVectorize()) { vector_width = cur_loop->vector_width; }

  // we register iv's scev of all loops, instead of only vectorized loops
  auto iv_ty = cur_loop->GetIVType();
  auto iv_name = cur_loop->IVName();
  auto iv_sym = SymName(iv_name);
  auto upper_bound = GetSingleUpperBound(iv_ty);
  auto stride = GetSingleStride(iv_ty);
  auto step = sbe::nu(stride * vector_width);
  auto ar_expr = MakeSCEVAddRecExpr(sbe::nu(0), step, cur_loop);

  if (debug_visit)
    dbgs() << "iv:    `" << iv_name << "` -> " << STR(ar_expr) << "\n";
  AssignSCEVToSym(SymName(iv_name), ar_expr, LoopName());
  return true;
}

bool ScalarEvolutionAnalysis::Visit(AST::ParallelBy& n) {
  TraceEachVisit(n);
  auto loop_name = InLoop() ? LoopName() : NoLoopName();
  for (auto pb : n.AllSubPVs()) {
    auto pb_id = AST::GetIdentifier(pb);
    auto se_val = MakeSCEVVal(sbe::sym(SymName(pb_id->name)), cur_loop);
    if (debug_visit)
      dbgs() << "pi:    `" << pb_id->name << "` -> " << STR(se_val) << "\n";
    AssignSCEVToSym(SymName(pb_id->name), se_val, loop_name);
  }
  return true;
}

bool ScalarEvolutionAnalysis::Visit(AST::Parameter& n) {
  TraceEachVisit(n);
  if (auto sty = dyn_cast<SpannedType>(n.type->GetType())) {
    auto syms = sty->GetShape().GetDynamicSymbols();
    for (auto s : syms) {
      auto se_val = MakeSCEVVal(s);
      if (IsAssignedSym(s->ToString())) continue;
      if (debug_visit)
        dbgs() << "param: `" << s << "` -> " << STR(se_val) << "\n";
      AssignSCEVToSym(s->ToString(), se_val, NoLoopName());
    }
  } else if (NeedAnalyze(n.GetType())) {
    auto se_val = MakeSCEVVal(sbe::sym(n.sym->name));
    if (debug_visit)
      dbgs() << "param: `" << n.sym->name << "` -> " << STR(se_val) << "\n";
    AssignSCEVToSym(SymName(n.sym->name), se_val, NoLoopName());
  }

  return true;
}
} // namespace Choreo