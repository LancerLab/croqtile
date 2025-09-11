#include "scalar_evolution.hpp"
#include "ast.hpp"
#include "aux.hpp"
#include "io.hpp"
#include "symbexpr.hpp"
#include "types.hpp"
#include "utils.hpp"

namespace Choreo {

bool ScalarEvolutionAnalysis::InVectorizedLoop() {
  auto loop = li->GetLoop(lname);
  if (!loop) return false;
  if (AST::NeedVectorize(*loop->loop)) return true;
  return false;
}

bool ScalarEvolutionAnalysis::InLoop() {
  auto loop = li->GetLoop(lname);
  if (!loop) return false;
  return true;
}

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
      if (lhs_ar->loop->HasLoop(rhs_ar->loop->lname)) {
        // {a, +, b} <L1> op {c, +, d} <L2> , L2 is nested in L1
        // 1. we treat {c, +, d} <L2> as invariant in L1, we get {{a, +, b} <L1>
        // op c, +, b} <L2>
        auto new_base = ComputeARSCEV(lhs_ar, rhs_ar->base, op);
        return MakeSCEVAddRecExpr(new_base, rhs_ar->step, rhs_ar->loop);
      } else if (rhs_ar->loop->HasLoop(lhs_ar->loop->lname)) {
        // {a, +, b} <L1> op {c, +, d} <L2> , L1 is nested in L2
        // 1. we treat {a, +, b} <L1> as invariant in L2, we get {{c, +, d} <L2>
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
  if (InAnno) return true;
  auto valno = n.Opts().HasVal() ? n.Opts().GetVal() : sbe::sym(STR(n));
  auto scev_val = MakeSCEVVal(valno);
  auto loop_name = InLoop() ? lname : NoLoopName();
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
    } else if (auto da = dyn_cast<AST::DataAccess>(n.GetReference())) {
      // todo: handle data access
    } else if (auto call = AST::GetCall(n.GetReference())) {
      // todo: handle call
    } else if (auto intlit = AST::GetIntLiteral(n.GetReference())) {
      auto se_val = MakeSCEVVal(valno);
      n.SetSCEV(se_val);
    } else {
      Error1(n.LOC(),
             "unsupported reference expr in scalar evolution: " + STR(n) + ".");
      return false;
    }
    return true;
  } else if (op == "dimof") {
    n.SetSCEV(scev_val);
    return true;
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
      return true;
    } else {
      auto bin_scev = ComputeARSCEV(lhs_scev, rhs_scev, n.op);
      n.SetSCEV(bin_scev ? bin_scev : scev_val);
      return true;
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
    dbgs() << "[scev][decl]: " << iv_name << " -> " << STR(init_scev) << "\n";
  AssignSCEVToSym(SymName(iv_name), init_scev, lname);

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
      dbgs() << "[scev][assign]: " << name << " -> " << STR(expr_scev) << "\n";
    AssignSCEVToSym(sym_name, expr_scev, lname);
  } else {
    // else, we will invalidate the scev of the symbol this assignment assigns
    // to, since we cannot track the scev for re-assignment variables.
    if (debug_visit)
      dbgs() << "[scev][assign]: " << name
             << " is re-assigned, invalidate its scev.\n";
    AssignSCEVToSym(sym_name, nullptr, lname);
  }
  return true;
}

bool ScalarEvolutionAnalysis::Visit(AST::Call& n) {
  TraceEachVisit(n);
  if (n.IsAnno()) InAnno = true;
  return true;
}

bool ScalarEvolutionAnalysis::Visit(AST::ForeachBlock& n) {
  TraceEachVisit(n);
  // we register iv's scev of all loops, instead of only vectorized loops
  auto loop = li->GetLoop(lname);
  auto iv_ty = loop->GetIVType();
  auto iv_name = loop->IVName();
  auto iv_sym = SymName(iv_name);
  auto upper_bound = GetSingleUpperBound(iv_ty);
  auto stride = GetSingleStride(iv_ty);
  auto width = GetSingleWidth(iv_ty);
  auto step = sbe::nu(stride * width);
  auto ar_expr = MakeSCEVAddRecExpr(sbe::nu(0), step, loop);

  if (debug_visit)
    dbgs() << "[scev][iv]: " << iv_name << " -> " << STR(ar_expr) << "\n";
  AssignSCEVToSym(SymName(iv_name), ar_expr, lname);
  return true;
}

bool ScalarEvolutionAnalysis::Visit(AST::ParallelBy& n) {
  TraceEachVisit(n);
  auto loop_name = InLoop() ? lname : NoLoopName();
  auto loop = li->GetLoop(lname);
  for (auto pb : n.AllSubPVs()) {
    auto pb_id = AST::GetIdentifier(pb);
    auto se_val = MakeSCEVVal(sbe::sym(SymName(pb_id->name)), loop);
    if (debug_visit)
      dbgs() << "[scev][pid]: " << pb_id->name << " -> " << STR(se_val) << "\n";
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
        dbgs() << "[scev][param]: " << s << " -> " << STR(se_val) << "\n";
      AssignSCEVToSym(s->ToString(), se_val, NoLoopName());
    }
  } else if (NeedAnalyze(n.GetType())) {
    auto se_val = MakeSCEVVal(sbe::sym(n.sym->name));
    if (debug_visit)
      dbgs() << "[scev][param]: " << n.sym->name << " -> " << STR(se_val)
             << "\n";
    AssignSCEVToSym(SymName(n.sym->name), se_val, NoLoopName());
  }

  return true;
}

bool ScalarEvolutionAnalysis::BeforeAfterVisitImpl(AST::Node& n) {
  if (auto f = dyn_cast<AST::Call>(&n)) {
    if (f->IsAnno()) InAnno = false;
  }
  return true;
}

bool ScalarEvolutionAnalysis::AfterBeforeVisitImpl(AST::Node& n) {
  if (auto f = dyn_cast<AST::Call>(&n)) {
    if (f->IsAnno()) InAnno = true;
  }
  return true;
}
} // namespace Choreo