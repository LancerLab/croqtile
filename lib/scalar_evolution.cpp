#include "scalar_evolution.hpp"
#include "ast.hpp"
#include "io.hpp"
#include "loop_utils.hpp"
#include "symvals.hpp"

namespace Choreo {

bool ScalarEvolutionAnalysis::InLoop() { return cur_loop != nullptr; }

bool ScalarEvolutionAnalysis::InAppointedLoop() {
  if (appointed_loop.empty()) return true;
  if (!InLoop()) return false;
  return cur_loop->LoopName() == appointed_loop;
}

// op is one of "+", "-", "*", "/"
ptr<SCEV> ScalarEvolutionAnalysis::ComputeARSCEV(ptr<SCEV> lhs, ptr<SCEV> rhs,
                                                 std::string op) {
  if (isa<SCEVAddRecExpr>(lhs) && isa<SCEVVal>(rhs)) {
    auto lhs_ar = dyn_cast<SCEVAddRecExpr>(lhs);
    auto rhs_val = dyn_cast<SCEVVal>(rhs);
    auto ar_loop = lhs_ar->GetLoop();
    if (op == "+" || op == "-") {
      if (rhs_val->IsLoopInVariant(ar_loop)) {
        auto new_base = ComputeARSCEV(lhs_ar->GetBase(), rhs_val, op);
        return MakeSCEVAddRecExpr(new_base, lhs_ar->GetStep(), ar_loop);
      } else
        return nullptr;
    } else if (op == "*" || op == "/") {
      auto new_base = ComputeARSCEV(lhs_ar->GetBase(), rhs_val, op);
      auto new_step = ComputeARSCEV(lhs_ar->GetStep(), rhs_val, op);
      return MakeSCEVAddRecExpr(new_base, new_step, ar_loop);
    } else
      return nullptr;
  } else if (isa<SCEVVal>(lhs) && isa<SCEVAddRecExpr>(rhs)) {
    auto lhs_val = dyn_cast<SCEVVal>(lhs);
    auto rhs_ar = dyn_cast<SCEVAddRecExpr>(rhs);
    auto ar_loop = rhs_ar->GetLoop();
    if (op == "+" || op == "-") {
      if (lhs_val->IsLoopInVariant(ar_loop)) {
        auto new_base = ComputeARSCEV(lhs_val, rhs_ar->GetBase(), op);
        return MakeSCEVAddRecExpr(new_base, rhs_ar->GetStep(), ar_loop);
      } else
        return nullptr;
    } else if (op == "*" || op == "/") {
      auto new_base = ComputeARSCEV(lhs_val, rhs_ar->GetBase(), op);
      auto new_step = ComputeARSCEV(lhs_val, rhs_ar->GetStep(), op);
      return MakeSCEVAddRecExpr(new_base, new_step, ar_loop);
    } else
      return nullptr;
  } else if (isa<SCEVAddRecExpr>(lhs) && isa<SCEVAddRecExpr>(rhs)) {
    auto lhs_ar = dyn_cast<SCEVAddRecExpr>(lhs);
    auto rhs_ar = dyn_cast<SCEVAddRecExpr>(rhs);
    if (op == "+" || op == "-") {
      if (lhs_ar->GetLoop() == rhs_ar->GetLoop()) {
        auto new_base = ComputeARSCEV(lhs_ar->GetBase(), rhs_ar->GetBase(), op);
        auto new_step = ComputeARSCEV(lhs_ar->GetStep(), rhs_ar->GetStep(), op);
        return MakeSCEVAddRecExpr(new_base, new_step, lhs_ar->GetLoop());
      }
      if (lhs_ar->GetLoop()->HasLoop(rhs_ar->GetLoop()->LoopName())) {
        // {a, +, b} <L1> op {c, +, d} <L2> , L2 is nested in L1
        // we treat {c, +, d} <L2> as invariant in L1, we get {{a, +, b} <L1>
        // op c, +, b} <L2>
        auto new_base = ComputeARSCEV(lhs_ar, rhs_ar->GetBase(), op);
        return MakeSCEVAddRecExpr(new_base, rhs_ar->GetStep(),
                                  rhs_ar->GetLoop());
      } else if (rhs_ar->GetLoop()->HasLoop(lhs_ar->GetLoop()->LoopName())) {
        // {a, +, b} <L1> op {c, +, d} <L2> , L1 is nested in L2
        // we treat {a, +, b} <L1> as invariant in L2, we get {{c, +, d} <L2>
        // op a, +, d} <L1>
        auto new_base = ComputeARSCEV(lhs_ar->GetBase(), rhs_ar, op);
        return MakeSCEVAddRecExpr(new_base, lhs_ar->GetStep(),
                                  lhs_ar->GetLoop());
      } else {
        choreo_unreachable("invalid AddRecExprs in different loops.");
      }
    } else if (op == "*" || op == "/") {
      // we do not support multiply/divide two AddRec right now
      return nullptr;
    } else
      return nullptr;
  } else if (isa<SCEVVal>(lhs) && isa<SCEVVal>(rhs)) {

    if (op == "+") {
      auto new_val = lhs->GetValue() + rhs->GetValue();
      return MakeSCEVVal(new_val);
    } else if (op == "-") {
      auto new_val = lhs->GetValue() - rhs->GetValue();
      return MakeSCEVVal(new_val);
    } else if (op == "*") {
      auto new_val = lhs->GetValue() * rhs->GetValue();
      return MakeSCEVVal(new_val);
    } else if (op == "/") {
      auto new_val = lhs->GetValue() / rhs->GetValue();
      return MakeSCEVVal(new_val);
    }
  }
  return nullptr;
}

bool ScalarEvolutionAnalysis::Visit(AST::Program& n) {
  if (!InAppointedLoop()) return true;
  if (debug_visit) dbgs() << "\n[scev] Initialize scalar evolution analysis.\n";
  root_ptr = AST::Make<AST::Program>(n.LOC(), n.nodes);
  return true;
}

bool ScalarEvolutionAnalysis::Visit(AST::Expr& n) {
  TraceEachVisit(n);
  if (!InAppointedLoop()) return true;
  if (!NeedAnalyze(n.GetType())) return true;
  auto valno = n.Opts().HasVal() ? n.Opts().GetVal() : sbe::sym(STR(n));
  auto scev_val = MakeSCEVVal(valno);
  auto loop_name = InLoop() ? LoopName() : NoLoopName();
  auto op = n.op;
  if (n.IsReference()) {
    if (auto id = AST::GetIdentifier(n)) {
      auto iv_sym = SymName(id->name);
      if (!IsAssignedSym(iv_sym)) {
        AssignSCEVToSym(SymName(iv_sym), scev_val, loop_name);
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

bool ScalarEvolutionAnalysis::Visit(AST::CastExpr& n) {
  TraceEachVisit(n);
  if (!InAppointedLoop()) return true;
  if (!NeedAnalyze(n.GetType())) return true;
  auto r_expr = dyn_cast<AST::Expr>(n.GetR());
  assert(r_expr && "Only Expr can be R of CastExpr.");
  auto r_scev = r_expr->GetSCEV();
  n.SetSCEV(r_scev);
  if (debug_visit)
    dbgs() << indent << "cast:  `" << STR(n) << "` -> " << STR(r_scev) << "\n";
  return true;
}

bool ScalarEvolutionAnalysis::Visit(AST::NamedVariableDecl& n) {
  TraceEachVisit(n);
  if (!InAppointedLoop()) return true;
  if (!NeedAnalyze(n.GetType())) return true;

  auto init_expr = dyn_cast<AST::Expr>(n.init_expr);
  assert(init_expr && "Only Expr can be init_expr of NamedVariableDecl.");
  // named variable declaration outside any loop
  auto iv_name = n.name_str;
  auto init_scev = init_expr->GetSCEV();
  if (debug_visit)
    dbgs() << indent << "decl:  `" << iv_name << "` -> " << STR(init_scev)
           << "\n";
  AssignSCEVToSym(SymName(iv_name), init_scev, LoopName());

  return true;
}

bool ScalarEvolutionAnalysis::Visit(AST::Identifier& n) {
  TraceEachVisit(n);
  if (!InAppointedLoop()) return true;
  return true;
}

bool ScalarEvolutionAnalysis::Visit(AST::Assignment& n) {
  TraceEachVisit(n);
  if (!InAppointedLoop()) return true;
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
      dbgs() << indent << "asgn:  `" << name << "` -> " << STR(expr_scev)
             << "\n";
    AssignSCEVToSym(sym_name, expr_scev, LoopName());
  } else {
    // else, we will invalidate the scev of the symbol this assignment assigns
    // to, since we cannot track the scev for re-assignment variables.
    if (debug_visit)
      dbgs() << indent << "asgn:  `" << name
             << "` is re-assigned, invalidate its scev.\n";
    AssignSCEVToSym(sym_name, nullptr, LoopName());
  }
  return true;
}

bool ScalarEvolutionAnalysis::Visit(AST::Call& n) {
  TraceEachVisit(n);
  if (!InAppointedLoop()) return true;
  return true;
}

bool ScalarEvolutionAnalysis::Visit(AST::DataAccess& n) {
  TraceEachVisit(n);
  if (!InAppointedLoop()) return true;
  if (!n.AccessElement()) return true;
  auto indices = n.GetIndices();
  if (indices.size() == 1) {
    auto idx_name = AST::GetIdentifier(indices[0])->name;
    if (with_syms.count(InScopeName(idx_name))) {
      auto with_scev = GetSCEVOfSym(SymName(idx_name));
      n.scev = with_scev;
      if (debug_visit)
        dbgs() << indent << "da:    `" << STR(n) << "` -> " << STR(with_scev)
               << "\n";
      return true;
    }
  }

  ValueItem stride = sbe::nu(1);
  ptr<SCEV> ptr_scev = nullptr;

  for (int i = indices.size() - 1; i >= 0; i--) {
    if (auto idx = dyn_cast<AST::Expr>(indices[i])) {
      ptr<SCEV> scev = idx->GetSCEV();

      if (!ptr_scev) {
        ptr_scev = scev;
      } else {
        scev = ComputeARSCEV(scev, MakeSCEVVal(stride), "*");
        ptr_scev = ComputeARSCEV(ptr_scev, scev, "+");
      }

      // accumulate the stride
      if (auto loop = scev->GetLoop()) {
        auto iv_ty = loop->GetIVType();
        auto loop_bound = GetSingleUpperBound(iv_ty);
        stride = stride * loop_bound;
      }
    }
  }
  n.scev = ptr_scev;
  if (debug_visit)
    dbgs() << indent << "da:    `" << STR(n) << "` -> " << STR(ptr_scev)
           << "\n";

  return true;
}

bool ScalarEvolutionAnalysis::Visit(AST::WithIn& n) {
  TraceEachVisit(n);
  if (!InAppointedLoop()) return true;
  if (n.with) { with_syms.insert(InScopeName(n.with->name)); }
  return true;
}

bool ScalarEvolutionAnalysis::Visit(AST::ForeachBlock& n) {
  TraceEachVisit(n);
  if (!InAppointedLoop()) return true;
  // cur_loop = n.loop;
  int vector_width = 1;
  if (cur_loop->NeedVectorize()) { vector_width = cur_loop->GetVectorFactor(); }

  // we register iv's scev of all loops, instead of only vectorized loops
  auto iv_ty = cur_loop->GetIVType();
  auto iv_sym = cur_loop->IVSym();
  auto upper_bound = GetSingleUpperBound(iv_ty);
  auto stride = GetSingleStep(iv_ty);
  auto step = sbe::nu(stride * vector_width);
  auto ar_expr = MakeSCEVAddRecExpr(sbe::nu(0), step, cur_loop);

  if (debug_visit)
    dbgs() << indent << "iv:    `" << UnScopedName(iv_sym) << "` -> "
           << STR(ar_expr) << "\n";
  AssignSCEVToSym(iv_sym, ar_expr, LoopName());

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
    auto matchers = within_map[with_sym];
    ptr<SCEV> with_scev = nullptr;
    ValueItem stride = sbe::nu(1);
    for (int i = matchers.size() - 1; i >= 0; i--) {
      auto matcher = matchers[i];
      auto scev_ar = GetSCEVOfSym(matcher);
      auto loop = li->GetLoopOfIV(matcher);
      assert(loop && "loop should exist.");
      if (!with_scev) {
        with_scev = scev_ar;
      } else {
        scev_ar = ComputeARSCEV(scev_ar, MakeSCEVVal(stride), "*");
        with_scev = ComputeARSCEV(with_scev, scev_ar, "+");
      }

      auto iv_ty = loop->GetIVType();
      auto loop_bound = GetSingleUpperBound(iv_ty);
      stride = stride * loop_bound;
    }

    if (debug_visit)
      dbgs() << indent << "iv:    `" << UnScopedName(with_sym) << "` -> "
             << STR(with_scev) << "\n";
    AssignSCEVToSym(with_sym, with_scev, LoopName());
  }

  return true;
}

bool ScalarEvolutionAnalysis::Visit(AST::ParallelBy& n) {
  TraceEachVisit(n);
  if (!InAppointedLoop()) return true;
  auto loop_name = InLoop() ? LoopName() : NoLoopName();
  for (auto pb : n.AllSubPVs()) {
    auto pb_id = AST::GetIdentifier(pb);
    auto se_val = MakeSCEVVal(sbe::sym(SymName(pb_id->name)), cur_loop);
    if (debug_visit)
      dbgs() << indent << "pi:    `" << pb_id->name << "` -> " << STR(se_val)
             << "\n";
    AssignSCEVToSym(SymName(pb_id->name), se_val, loop_name);
  }
  return true;
}

bool ScalarEvolutionAnalysis::Visit(AST::Parameter& n) {
  TraceEachVisit(n);
  if (!InAppointedLoop()) return true;
  if (auto sty = dyn_cast<SpannedType>(n.type->GetType())) {
    auto syms = sty->GetShape().GetDynamicSymbols();
    for (auto s : syms) {
      auto se_val = MakeSCEVVal(s);
      if (IsAssignedSym(s->ToString())) continue;
      if (debug_visit)
        dbgs() << indent << "param: `" << s << "` -> " << STR(se_val) << "\n";
      AssignSCEVToSym(s->ToString(), se_val, NoLoopName());
    }
  } else if (NeedAnalyze(n.GetType())) {
    auto se_val = MakeSCEVVal(sbe::sym(n.sym->name));
    if (debug_visit)
      dbgs() << indent << "param: `" << n.sym->name << "` -> " << STR(se_val)
             << "\n";
    AssignSCEVToSym(SymName(n.sym->name), se_val, NoLoopName());
  }

  return true;
}

void ScalarEvolutionAnalysis::ReComputeInLoop(const std::string& lname,
                                              bool debug) {
  bool original_debug = debug_visit;
  debug_visit = debug && debug_visit;
  if (debug_visit)
    dbgs() << "\n[scev] Re-compute SCEV expressions in loop `" << lname
           << "`.\n";
  appointed_loop = lname;
  root_ptr->accept(*this);
  appointed_loop = "";
  debug_visit = original_debug;
}
} // namespace Choreo
