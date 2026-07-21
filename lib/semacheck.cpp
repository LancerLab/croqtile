#include "semacheck.hpp"
#include "interval.hpp"
#include "types.hpp"

using namespace Choreo;

namespace {

bool ScopeContainsPredicate(const ValueItem& scope_pred,
                            const ValueItem& target_pred) {
  if (!IsValidValueItem(scope_pred) || !IsValidValueItem(target_pred))
    return false;

  auto scope = scope_pred->Normalize();
  auto target = target_pred->Normalize();
  if (*scope == *target) return true;
  if (auto eq = VIBool(sbe::oc_eq(scope, target)->Normalize());
      eq && eq.value())
    return true;
  if (auto b = VIBool(scope); b && b.value()) return true;

  if (auto bop = dyn_cast<sbe::BinaryOperation>(scope)) {
    if (bop->GetOpCode() == OpCode::AND)
      return ScopeContainsPredicate(bop->GetLeft(), target) ||
             ScopeContainsPredicate(bop->GetRight(), target);
  }
  return false;
}

bool SameSymbolicExpr(const ValueItem& lhs, const ValueItem& rhs) {
  if (!IsValidValueItem(lhs) || !IsValidValueItem(rhs)) return false;
  auto l = lhs->Normalize();
  auto r = rhs->Normalize();
  if (*l == *r) return true;
  if (auto lsym = dyn_cast<sbe::SymbolicValue>(l)) {
    if (auto rsym = dyn_cast<sbe::SymbolicValue>(r)) {
      auto canon = [](std::string s) {
        auto pos = s.find("__elem__");
        if (pos != std::string::npos) s = s.substr(0, pos);
        return s;
      };
      if (canon(lsym->Value()) == canon(rsym->Value())) return true;
    }
  }
  if (auto eq = VIBool(sbe::oc_eq(l, r)->Normalize()); eq && eq.value())
    return true;
  return false;
}

bool ScopeImpliesUpperBound(const ValueItem& scope_pred, const ValueItem& var,
                            const ValueItem& ub) {
  if (!IsValidValueItem(scope_pred) || !IsValidValueItem(var) ||
      !IsValidValueItem(ub))
    return false;

  auto scope = scope_pred->Normalize();
  if (auto bop = dyn_cast<sbe::BinaryOperation>(scope)) {
    auto lhs = bop->GetLeft();
    auto rhs = bop->GetRight();
    switch (bop->GetOpCode()) {
    case OpCode::AND:
      return ScopeImpliesUpperBound(lhs, var, ub) ||
             ScopeImpliesUpperBound(rhs, var, ub);
    case OpCode::LT:
      if (SameSymbolicExpr(lhs, var) && sbe::cle(rhs, ub)) return true;
      break;
    case OpCode::LE:
      if (SameSymbolicExpr(lhs, var) && sbe::clt(rhs, ub)) return true;
      break;
    case OpCode::GT:
      if (SameSymbolicExpr(rhs, var) && sbe::cle(lhs, ub)) return true;
      break;
    case OpCode::GE:
      if (SameSymbolicExpr(rhs, var) && sbe::clt(lhs, ub)) return true;
      break;
    default: break;
    }
  }
  return false;
}

struct ExprBounds {
  ValueItem lb = GetInvalidValueItem();
  ValueItem ub = GetInvalidValueItem();

  bool IsValid() const { return IsValidValueItem(lb) && IsValidValueItem(ub); }
};

std::string CanonicalScopedSymbol(std::string sym) {
  auto pos = sym.find("__elem__");
  if (pos != std::string::npos) sym = sym.substr(0, pos);
  return sym;
}

ValueItem ToInclusiveUpperBound(const ValueItem& ub) {
  if (!IsValidValueItem(ub)) return GetInvalidValueItem();
  return (ub - sbe::nu(1))->Normalize();
}

ExprBounds InferExprBounds(SemaChecker* sc, const ValueItem& expr) {
  if (!IsValidValueItem(expr)) return {};

  auto norm = expr->Normalize();
  if (auto iv = VIInt(norm)) {
    auto v = sbe::nu(*iv);
    return {v, v};
  }

  if (auto sym = VISym(norm)) {
    auto scoped = CanonicalScopedSymbol(*sym);
    if (!PrefixedWith(scoped, "::")) return {};

    auto ty = sc->GetScopedSymbolType(scoped);
    if (auto bit = dyn_cast<BoundedIntegerType>(ty);
        bit && bit->HasValidBound()) {
      return {bit->GetLowerBound(),
              ToInclusiveUpperBound(bit->GetUpperBound())};
    }
    if (auto bitt = dyn_cast<BoundedITupleType>(ty);
        bitt && bitt->HasValidBound() && bitt->Dims() == 1) {
      return {bitt->GetLowerBound(0),
              ToInclusiveUpperBound(bitt->GetUpperBound(0))};
    }
    return {};
  }

  if (auto bop = VIBop(norm)) {
    auto lhs = InferExprBounds(sc, bop->GetLeft());
    auto rhs = InferExprBounds(sc, bop->GetRight());
    if (!lhs.IsValid() || !rhs.IsValid()) return {};

    switch (bop->GetOpCode()) {
    case OpCode::ADD:
      return {(lhs.lb + rhs.lb)->Normalize(), (lhs.ub + rhs.ub)->Normalize()};
    case OpCode::SUBTRACT:
      return {(lhs.lb - rhs.ub)->Normalize(), (lhs.ub - rhs.lb)->Normalize()};
    case OpCode::MULTIPLY:
      if (sbe::cge(lhs.lb, sbe::nu(0)) && sbe::cge(rhs.lb, sbe::nu(0)))
        return {(lhs.lb * rhs.lb)->Normalize(), (lhs.ub * rhs.ub)->Normalize()};
      return {};
    default: break;
    }
  }

  return {};
}

// Build an interval constraint environment from BoundedType information
// and scope predicates, then evaluate whether a predicate is provably
// true or false using interval analysis.
std::optional<bool> TryProveWithIntervals(SemaChecker* sc,
                                          const ValueItem& pred,
                                          const ValueItem& scope_pred) {
  if (!IsValidValueItem(pred)) return std::nullopt;

  auto symbols = GetSymbols(pred);
  if (symbols.empty()) return std::nullopt;

  sbe::ConstraintEnv env;

  for (auto& sym_vi : symbols) {
    auto sym_name = VISym(sym_vi);
    if (!sym_name.has_value()) continue;

    auto scoped = CanonicalScopedSymbol(*sym_name);
    if (!PrefixedWith(scoped, "::")) continue;

    // Start with the type-declared bounds.
    sbe::IntervalSet iv = sbe::IntervalSet::MakeUniverse();
    auto ty = sc->GetScopedSymbolType(scoped);
    if (auto bit = dyn_cast<BoundedIntegerType>(ty);
        bit && bit->HasValidBound()) {
      auto lb = VIInt(bit->GetLowerBound());
      auto ub = VIInt(bit->GetUpperBound());
      if (lb.has_value() && ub.has_value())
        iv = sbe::IntervalSet(sbe::Interval::HalfOpen(*lb, *ub));
    } else if (auto bitt = dyn_cast<BoundedITupleType>(ty);
               bitt && bitt->HasValidBound() && bitt->Dims() == 1) {
      auto lb = VIInt(bitt->GetLowerBound(0));
      auto ub = VIInt(bitt->GetUpperBound(0));
      if (lb.has_value() && ub.has_value())
        iv = sbe::IntervalSet(sbe::Interval::HalfOpen(*lb, *ub));
    }

    // Narrow by scope predicates.
    if (IsValidValueItem(scope_pred)) {
      auto scope_iv = sbe::ProjectConstraint(scope_pred, *sym_name, env);
      iv = Intersect(iv, scope_iv);
    }

    env[*sym_name] = iv;
  }

  if (env.empty()) return std::nullopt;
  return sbe::EvalPredInterval(pred, env);
}

} // namespace

ValueItem SemaChecker::ActiveScopePredicate() const {
  ValueItem pred = sbe::bl(true);
  for (const auto& p : scope_pred_stack) {
    assert(IsValidValueItem(p));
    pred = sbe::bl_and(pred, p);
  }
  return pred;
}

bool SemaChecker::ExpressionIsConstrained(const ValueItem& expr) const {
  auto syms = GetSymbols(expr);
  auto constrained_syms = GetSymbols(ActiveScopePredicate());
  for (auto sym : syms)
    for (auto c_sym : constrained_syms)
      if (sbe::ceq(sym, c_sym)) return true;
  return false;
}

static int64_t GetForeachStaticExtent(const AST::ForeachBlock& fe) {
  auto& ranges = fe.GetRanges();
  if (ranges.empty()) return -1;
  auto rng = dyn_cast<AST::LoopRange>(ranges.front());
  if (!rng || !rng->GetRV()) return -1;
  auto iv_ty = dyn_cast<BoundedType>(rng->GetRV()->GetType());
  if (!iv_ty) return -1;
  auto ub = iv_ty->GetUpperBound();
  if (!ub || !ub->IsNumeric()) return -1;
  return dyn_cast<sbe::NumericValue>(ub.get())->Value();
}

static ValueItem GetForeachDynamicExtent(const AST::ForeachBlock& fe) {
  auto& ranges = fe.GetRanges();
  if (ranges.empty()) return nullptr;
  auto rng = dyn_cast<AST::LoopRange>(ranges.front());
  if (!rng || !rng->GetRV()) return nullptr;
  auto iv_ty = dyn_cast<BoundedType>(rng->GetRV()->GetType());
  if (!iv_ty) return nullptr;
  return iv_ty->GetUpperBound();
}

// Count MMA Exec operations in the direct body of a foreach (not nested
// foreachs) that precede the given mma.wait node.  Each Exec on a distinct
// accumulator generates a warpgroup_commit_batch(), so the maximum number of
// batches in flight per iteration equals the number of Exec ops.
static int CountMMAExecsInBody(const ptr<AST::Node>& node,
                               const AST::Node* stop_at = nullptr,
                               bool* found_stop = nullptr) {
  if (!node) return 0;
  int count = 0;
  bool dummy = false;
  if (!found_stop) found_stop = &dummy;

  auto walk = [&](auto&& self, const ptr<AST::Node>& n) -> void {
    if (!n || *found_stop) return;
    if (stop_at && n.get() == stop_at) {
      *found_stop = true;
      return;
    }
    if (auto mma = dyn_cast<AST::MMA>(n)) {
      if (auto op = mma->GetOperation()) {
        if (op->Tag() == AST::MMAOperation::Exec) count++;
      }
      return;
    }
    if (dyn_cast<AST::ForeachBlock>(n)) return;
    if (auto mn = dyn_cast<AST::MultiNodes>(n)) {
      for (auto& item : mn->values) {
        self(self, item);
        if (*found_stop) return;
      }
      return;
    }
    if (auto ie = dyn_cast<AST::IfElseBlock>(n)) {
      self(self, ie->GetBody());
      if (*found_stop) return;
      if (ie->HasElse()) self(self, ie->GetElseBody());
      return;
    }
    if (auto block = dyn_cast<AST::Block>(n)) {
      self(self, block->GetBody());
      return;
    }
  };
  walk(walk, node);
  return count;
}

bool SemaChecker::BeforeVisitImpl(AST::Node& n) {
  if (isa<AST::ChoreoFunction>(&n)) {
    pending_async.clear();
    waited_async.clear();
    scope_pred_stack.clear();
    shared_tensor_producers.clear();
  }
  if (auto block = dyn_cast<AST::PredBlock>(&n))
    scope_pred_stack.push_back(block->GetScopePredicate());
  else if (auto fe = dyn_cast<AST::ForeachBlock>(&n)) {
    scope_pred_stack.push_back(fe->GetScopePredicate());
    foreach_stack.push_back(fe);
    int unroll_factor = 0;
    if (AST::HasUnrollHint(*fe, unroll_factor) && unroll_factor > 0) {
      int64_t extent = GetForeachStaticExtent(*fe);
      if (extent > 0 && unroll_factor > extent)
        Error1(n.LOC(), "unroll factor (" + std::to_string(unroll_factor) +
                            ") exceeds the loop extent (" +
                            std::to_string(extent) + ").");
    }
  }
  return true;
}

bool SemaChecker::AfterVisitImpl(AST::Node& n) {
  if (isa<AST::PredBlock>(&n) || isa<AST::ForeachBlock>(&n))
    scope_pred_stack.pop_back();
  if (isa<AST::ForeachBlock>(&n)) foreach_stack.pop_back();

  if (isa<AST::ChoreoFunction>(&n)) {
    for (auto n : waited_async) pending_async.erase(n);
    if (!pending_async.empty())
      Error1(n.LOC(), "some asyncs are not explicitly waited: " +
                          DelimitedString(pending_async) + ".");
  }
  return true;
}

bool SemaChecker::InMidVisitImpl(AST::Node& n) {
  if (auto block = dyn_cast<AST::IfElseBlock>(&n)) {
    if (block->HasElse()) {
      scope_pred_stack.pop_back();
      scope_pred_stack.push_back(block->GetElseScopePredicate());
    }
  }
  return true;
}

bool SemaChecker::VisitNode(AST::IntLiteral& n) {
  if (!ReportUnknown(n, __FILE__, __LINE__)) return false;
  return true;
}

bool SemaChecker::VisitNode(AST::FloatLiteral& n) {
  if (!ReportUnknown(n, __FILE__, __LINE__)) return false;
  return true;
}

bool SemaChecker::VisitNode(AST::BoolLiteral& n) {
  if (!ReportUnknown(n, __FILE__, __LINE__)) return false;
  return true;
}

bool SemaChecker::VisitNode(AST::Expr& n) {
  if (!ReportUnknown(n, __FILE__, __LINE__)) return false;

  if (input_deps.Contains(n.GetR()))
    input_deps.Add(n);
  else if (input_deps.Contains(n.GetL()))
    input_deps.Add(n);
  else if (input_deps.Contains(n.GetC()))
    input_deps.Add(n);

  if (local_deps.Contains(n.GetR()))
    local_deps.Add(n);
  else if (local_deps.Contains(n.GetL()))
    local_deps.Add(n);
  else if (local_deps.Contains(n.GetC()))
    local_deps.Add(n);

  // check out-of-bound for the elemof operation in wait or trigger.
  // note: elemof in chunkat is not Expr node, so we do not check it here.
  if (n.op == "elemof") {
    auto arr_sym = GetArrayBaseSymbol(n);
    size_t subscription_level = GetSubScriptLevel(n);
    // access: events[a][b][c]
    // level:         1  2  3
    auto ty = NodeType(*arr_sym);
    auto arr_ty = cast<ArrayType>(ty);
    assert(arr_ty && "expect the array symbol to be an array type.");

    size_t arr_rank = arr_ty->ArrayRank();
    if (subscription_level > arr_rank) {
      Error1(n.LOC(), "Invalid array access: expected " +
                          std::to_string(arr_rank) + " dimensions, but " +
                          std::to_string(subscription_level) + " were used.");
      return false;
    }

    auto dims = arr_ty->Dimensions();
    const ValueItem& bound_vi = arr_ty->Dimension(subscription_level - 1);
    int bound = *VIInt(bound_vi);
    auto idx = n.GetR();

    // parallel p by 2 { xxx; dma arr[p] => local; }
    // if the index is a bounded var, hard to determine if it is out of bound
    if (isa<BoundedType>(NodeType(*idx))) return true;

    auto expr = cast<AST::Expr>(idx);
    if (!expr->Opts().HasVal()) {
      VST_DEBUG(dbgs() << "Expression: " << PSTR(expr)
                       << " does not have a value!\n");
      return false;
    }
    auto index = expr->Opts().GetVal();

    // skip checking the one with 'nil' value though
    if (!IsComputable(index)) {
      Error1(expr->LOC(), "The " + Ordinal(subscription_level) +
                              " subscription index can not be evaluated.");
      return true;
    }

    // 0 <= index < bound
    auto asrt0 = sbe::bop(OpCode::LT, index, bound_vi)->Normalize();
    auto asrt1 = sbe::bop(OpCode::GE, index, sbe::nu(0))->Normalize();
    assert(IsValidValueItem(asrt0) && IsValidValueItem(asrt1));

    auto message = "Index " + STR(index) + " is out of bounds of the " +
                   Ordinal(subscription_level) + " dimension of array '" +
                   PSTR(arr_sym) + "', where the valid range is [0, " +
                   std::to_string(bound) + ")";

    CreateAssessment(asrt0, message, expr->LOC(), expr,
                     UsageType::ElementAccess);
    CreateAssessment(asrt1, message, expr->LOC(), expr,
                     UsageType::ElementAccess);
  } else if (n.IsBinary() && n.IsArith()) {
    auto lty = NodeType(*n.GetL());
    auto rty = NodeType(*n.GetR());
    auto lsty = dyn_cast<SpannedType>(lty);
    auto rsty = dyn_cast<SpannedType>(rty);

    // non-spanned types are checked already
    if (!lsty && !rsty) return true;

    // Allows `tensor <op> scalar`
    if ((lsty && isa<ScalarType>(rty)) || (rsty && isa<ScalarType>(lty)))
      return true;

    auto lshape = lsty->GetShape();
    auto rshape = rsty->GetShape();

    assert(lshape.IsValid());
    assert(rshape.IsValid());

    bool compatible = true;
    bool shape_error_reported = false;
    if (lshape.Rank() == rshape.Rank()) {
      if (lshape != rshape) compatible = false;
    } else {
      // only allow broadcasting of msb dimenisons
      auto min_rank = std::min(lshape.Rank(), rshape.Rank());
      assert(min_rank >= 1);
      for (size_t i = 1; i <= min_rank; ++i) {
        auto lidx = lshape.Rank() - i;
        auto ridx = rshape.Rank() - i;
        auto err_message = "inconsistent shapes for spanned-operation `" +
                           n.op + "`(" + STR(lshape) + " v.s. " + STR(rshape) +
                           ").";
        auto warn_message =
            "shapes may be inconsistent for spanned-operation `" + n.op + "`(" +
            STR(lshape) + " v.s. " + STR(rshape) + ").";
        auto res = FCtx(fname).GetAssessor(*this).Assess(
            AssessPolicy::ErrWarn, AssessRelation::EQ, lshape.ValueAt(lidx),
            rshape.ValueAt(ridx), err_message, warn_message,
            UsageType::ShapeCompatibility, AssessType::ENTRY, n.LOC(), &n);
        if (!res.passed) {
          compatible = false;
          shape_error_reported = true;
          break;
        }
        if (res.warned) { break; }
      }
    }
    if (!compatible) {
      if (!shape_error_reported)
        Error1(n.LOC(), "inconsistent shapes for spanned-operation `" + n.op +
                            "`(" + STR(lshape) + " v.s. " + STR(rshape) + ").");
    }
  }

  return true;
}

bool SemaChecker::VisitNode(AST::MultiDimSpans& n) {
  if (!ReportUnknown(n, __FILE__, __LINE__)) return false;
  if (n.list)
    if (auto mv = dyn_cast<AST::MultiValues>(n.list))
      for (const auto& v : mv->AllValues())
        if (local_deps.Contains(v)) {
          local_deps.Add(*n.list);
          break;
        }
  return true;
}

bool SemaChecker::VisitNode(AST::NamedTypeDecl& n) {
  if (!ReportUnknown(n, __FILE__, __LINE__)) return false;
  return true;
}

bool SemaChecker::VisitNode(AST::NamedVariableDecl& n) {
  if (!ReportUnknown(n, __FILE__, __LINE__)) return false;

  auto ty = NodeType(n);
  auto s = GetShape(ty);
  if (s.IsValid())
    for (auto sv : s.Value())
      if (*sv == *sbe::nu(0))
        Error1(n.LOC(), "found 0-dimension within the shape of variable `" +
                            n.name_str + "'.");

  if (n.init_expr && input_deps.Contains(n.init_expr)) input_deps.Add(n);

  local_deps.Add(InScopeName(n.name_str));

  return true;
}

bool SemaChecker::VisitNode(AST::IntTuple& n) {
  if (!ReportUnknown(n, __FILE__, __LINE__)) return false;
  return true;
}

bool SemaChecker::VisitNode(AST::DataAccess& n) {
  if (!ReportUnknown(n, __FILE__, __LINE__)) return false;
  if (n.AccessElement() &&
      !ReportUnknownSymbol(n.GetDataName(), n.LOC(), __FILE__, __LINE__))
    return false;

  // Runtime bound check for element access: data.at(idx0, idx1, ...)
  // Each index must satisfy 0 <= idx_i < dim_i.
  if (n.AccessElement()) {
    auto dty = GetSymbolType(n.GetDataName());
    auto sty = GetSpannedType(dty);
    if (sty) {
      auto shape = sty->GetShape();
      if (shape.IsValid()) {
        auto& indices = n.GetIndices();
        size_t ndims = std::min(indices.size(), shape.Rank());
        for (size_t d = 0; d < ndims; ++d) {
          // The index might be wrapped in an IntIndex or be a direct Expr.
          ptr<AST::Node> val_node;
          if (auto ii = dyn_cast<AST::IntIndex>(indices[d]))
            val_node = ii->Val();
          else
            val_node = indices[d];

          // Try to get the symbolic value from the index expression.
          ValueItem index_val;
          ptr<AST::Node> class_node; // node used for classification

          if (auto expr = dyn_cast<AST::Expr>(val_node)) {
            if (expr->Opts().HasVal()) index_val = expr->Opts().GetVal();
            class_node = expr;
          } else if (auto id = dyn_cast<AST::Identifier>(val_node)) {
            index_val = sbe::sym(InScopeName(id->name));
            class_node = id;
          }

          if (!IsValidValueItem(index_val)) continue;
          if (!IsComputable(index_val)) continue;

          auto dim_bound = shape.ValueAt(d);
          auto expr_bounds = InferExprBounds(this, index_val);

          auto idx_str = PSTR(val_node);
          auto data_str = n.GetDataName();
          auto nty = NodeType(*val_node);

          if (isa<BoundedType>(nty)) {
            auto c = ExpressionIsConstrained(index_val);
            if (IsValidValueItem(expr_bounds.ub) && !c)
              CreateAssessment(
                  sbe::oc_lt(expr_bounds.ub, dim_bound)->Normalize(),
                  "The " + Ordinal(d + 1) + " index `" + idx_str +
                      "` of element access '" + data_str +
                      "' should be less than " + STR(dim_bound),
                  val_node->LOC(), class_node, UsageType::ElementAccess, &n);
            else {
              CreateAssessment(sbe::oc_ge(index_val, sbe::nu(0))->Normalize(),
                               "The " + Ordinal(d + 1) + " index `" + idx_str +
                                   "` of element access '" + data_str +
                                   "' should be greater than or equal to 0",
                               val_node->LOC(), class_node,
                               UsageType::ElementAccess, &n);
              CreateAssessment(sbe::oc_lt(index_val, dim_bound)->Normalize(),
                               "The " + Ordinal(d + 1) + " index `" + idx_str +
                                   "` of element access '" + data_str +
                                   "' should be less than " + STR(dim_bound),
                               val_node->LOC(), class_node,
                               UsageType::ElementAccess, &n);
            }
          } else {
            CreateAssessment(sbe::oc_ge(index_val, sbe::nu(0))->Normalize(),
                             "The " + Ordinal(d + 1) + " index `" + idx_str +
                                 "` of element access '" + data_str +
                                 "' should be greater than or equal to 0",
                             val_node->LOC(), class_node,
                             UsageType::ElementAccess, &n);
            CreateAssessment(sbe::oc_lt(index_val, dim_bound)->Normalize(),
                             "The " + Ordinal(d + 1) + " index `" + idx_str +
                                 "` of element access '" + data_str +
                                 "' should be less than " + STR(dim_bound),
                             val_node->LOC(), class_node,
                             UsageType::ElementAccess, &n);
          }
        }
      }
    }
  }

  return true;
}

bool SemaChecker::VisitNode(AST::Assignment& n) {
  if (!ReportUnknown(n, __FILE__, __LINE__)) return false;
  return false;

  if ((*NodeType(n) != *NodeType(*n.value))) {
    Error1(n.LOC(), "inconsistent types are found in the assignment: " +
                        STR(*NodeType(*n.value)) + " vs. " + STR(*NodeType(n)) +
                        ".");
    return false;
  }

  if (n.value && input_deps.Contains(n.value)) input_deps.Add(n);

  if (n.IsDecl()) local_deps.Add(InScopeName(n.da->GetDataName()));

  return true;
}

bool SemaChecker::VisitNode(AST::IntIndex& n) {
  if (!ReportUnknown(n, __FILE__, __LINE__)) return false;
  if (!isa<ScalarIntegerType>(n.value->GetType())) {
    Error1(n.LOC(), "Expect `" + PSTR(n.value) + "' to be a integer type.");
    return false;
  }
  return true;
}
bool SemaChecker::VisitNode(AST::DataType& n) {
  // TODO: figure out if we could check SufficientInfo
  if (!ReportUnknown(n, __FILE__, __LINE__, true)) return false;
  return true;
}

bool SemaChecker::VisitNode(AST::Identifier& n) {
  if (PrefixedWith(n.name, "$")) return true; // do not check internal symbols
  if (n.name == "_") return true;             // ignore unit bpv
  if (!ReportUnknownSymbol(n.name, n.LOC(), __FILE__, __LINE__)) return false;
  return true;
}

bool SemaChecker::VisitNode(AST::Parameter& n) {
  if (!ReportUnknown(n, __FILE__, __LINE__)) return false;

  if (n.sym) input_deps.Add(InScopeName(n.sym->name));

  return true;
}

bool SemaChecker::VisitNode(AST::IfElseBlock& n) {
  (void)n;
  return true;
}

bool SemaChecker::VisitNode(AST::ParallelBy& n) {
  if (auto shape = GetShape(NodeType(n)); shape.IsDynamic()) {
    int index = 1;
    for (auto& dim : shape.Value()) {
      auto spv = n.SubPVs()->ValueAt(index - 1);
      auto& loc = spv->LOC();
      if (!IsComputable(dim)) {
        Error1(loc, "The parallel count (" + Ordinal(index) +
                        "th) can not be evaluated.");
        continue;
      }
      auto message =
          "The " + Ordinal(index) +
          " bound item of parallelby is invalid: should be greater than 0";
      auto asrt = sbe::cmp(">", dim, sbe::nu(0))->Normalize();
      assert(IsValidValueItem(asrt));

      // The assertion "dim > 0" is about the BOUND (a parameter expression),
      // not the parallel variable itself.  The SubPV has BoundedType, so
      // CreateAssessment would escalate to USE_SITE - bypass it and force
      // ENTRY.
      FCtx(fname).GetAssessor(*this).Assess(
          AssessPolicy::Error, asrt, message, UsageType::LoopBound,
          AssessType::ENTRY, loc, spv.get(), &n);
      ++index;
    }
  }

  int setreg_count = 0;
  if (auto body = n.GetBody()) {
    for (const auto& stmt : body->values) {
      auto call = dyn_cast<AST::Call>(stmt);
      if (!call || !call->IsBIF()) continue;
      if (call->function->name == "croq::cuda::setreg_inc" ||
          call->function->name == "croq::cuda::setreg_dec") {
        setreg_count++;
      }
    }
  }

  if (setreg_count > 0 && n.GetLevel() != ParallelLevel::BLOCK) {
    Error1(n.LOC(),
           "croq::cuda::setreg_inc/dec is only allowed in parallel-by blocks "
           "lowered to CUDA kernels (level : block).");
  }

  if (n.HasLaunchBounds()) {
    if (n.GetLevel() != ParallelLevel::BLOCK)
      Error1(n.LOC(),
             "[[launch_bounds]] is only valid on block-level parallel-by.");
    auto& lb_args = n.GetLaunchBoundsArgs();
    for (size_t i = 0; i < lb_args->Count(); ++i) {
      auto arg = dyn_cast<AST::Expr>(lb_args->ValueAt(i));
      if (!arg || !arg->Opts().HasVal() || !arg->Opts().GetVal()->IsNumeric()) {
        Error1(lb_args->ValueAt(i)->LOC(),
               "[[launch_bounds]] argument must be a compile-time "
               "integer constant.");
      } else if (auto val = VIInt(arg->Opts().GetVal());
                 !val || val.value() < 0) {
        Error1(lb_args->ValueAt(i)->LOC(),
               "[[launch_bounds]] argument must be a non-negative integer.");
      } else if (i > 0 && val && val.value() == 0) {
        Error1(lb_args->ValueAt(i)->LOC(),
               "[[launch_bounds]] minBlocks/maxCluster must be positive "
               "(use 0 only for maxThreadsPerBlock to auto-compute).");
      }
    }
  }

  if (n.HasMaxnreg()) {
    if (n.GetLevel() != ParallelLevel::BLOCK)
      Error1(n.LOC(), "[[maxnreg]] is only valid on block-level parallel-by.");
    auto arg = n.GetMaxnregArg();
    if (!arg->Opts().HasVal() || !arg->Opts().GetVal()->IsNumeric()) {
      Error1(arg->LOC(),
             "[[maxnreg]] argument must be a compile-time integer constant.");
    } else if (auto val = VIInt(arg->Opts().GetVal());
               !val || val.value() <= 0) {
      Error1(arg->LOC(), "[[maxnreg]] argument must be a positive integer.");
    }
  }

  return true;
}

bool SemaChecker::VisitNode(AST::WithIn& n) {
  if (auto shape = GetShape(NodeType(*n.in)); shape.IsDynamic()) {
    int index = 1;
    for (auto& dim : shape.Value()) {
      if (!IsComputable(dim))
        continue; // not reporting error since there could be no use of the
                  // value
      std::string message = "zero is detected for the " + Ordinal(index) +
                            " dim of the mdspan inside the with-in statement";
      auto asrt = sbe::cmp("!=", dim, sbe::nu(0))->Normalize();
      assert(IsValidValueItem(asrt));

      // The assertion "dim != 0" is about the span BOUND (a parameter
      // expression), not the with-in iterator variable itself.  Since n.in
      // has BoundedType, CreateAssessment would escalate to USE_SITE - bypass
      // it and force ENTRY.
      FCtx(fname).GetAssessor(*this).Assess(
          AssessPolicy::Error, asrt, message, UsageType::LoopBound,
          AssessType::ENTRY, n.in->LOC(), n.in.get());
      ++index;
    }
  }
  return true;
}

bool SemaChecker::VisitNode(AST::SpanAs& n) {
  if (!ReportUnknownSymbol(n.id->name, n.LOC(), __FILE__, __LINE__))
    return false;

  auto ity = GetSymbolType(n.id->name);

  if (!(isa<SpannedType>(ity) || isa<FutureType>(ity))) {
    Error1(n.LOC(), "Expect symbol `" + n.id->name + "' to be a spanned type.");
    return false;
  }

  if (!(AST::istypeof<SpannedType>(&n))) {
    Error1(n.LOC(), "Invalid type of span_as expression.");
    return false;
  }

  auto sty = GetSpannedType(ity);
  auto nty = cast<SpannedType>(NodeType(n));

  if (sty->ElementType() != nty->ElementType()) {
    Error1(n.LOC(), "Inconsistent element type: (" + STR(nty->ElementType()) +
                        " = span_as(" + STR(sty->ElementType()) + ".");
    return false;
  }

  auto p = (sty->IsDense() & nty->IsDense());
  if (p == Modality::NOT)
    Error1(n.LOC(), "span_as is applied on non-contiguous spanned data.");
  else if (p == Modality::MAY)
    Warning(n.LOC(),
            "span_as could be applied on non-contiguous spanned data.");

  if (!sty->RuntimeShaped() && !nty->RuntimeShaped()) {
    // check if the shape size are same
    if (sty->ElementCount() != nty->ElementCount()) {
      Error1(n.LOC(), "Inconsistent mdspan size: " + n.id->name + "(" +
                          STR(sty->ElementCount()) + ") = spanas (" +
                          n.nid->name + "(" + STR(nty->ElementCount()) + ")).");
      return false;
    }
  }

  return true;
}

bool SemaChecker::VisitNode(AST::DMA& n) {
  bool IsDummy = (n.operation == ".any");
  auto ty = n.GetType();

  if (IsDummy) {
    if (n.future.empty()) {
      Error1(n.LOC(), "A dummy/async DMA must be named.");
      return false;
    }
    if (!isa<PlaceHolderType>(ty) ||
        (cast<PlaceHolderType>(ty)->GetBaseType() != BaseType::FUTURE)) {
      Error1(n.LOC(), "Expect a placeholder type but got '" + PSTR(ty) + "'.");
      return false;
    }
    return true;
  }

  // Check swizzle: only report error if swizzle is explicitly specified
  // and we're not in a WGMMA context
  auto swiz_set = CCtx().TargetSwizzleModes();
  if (!swiz_set.empty() && !swiz_set.count(n.GetSwizzleMode())) {
    // For now, we just validate the swizzle value is valid (128, 64, or 32)
    // The WGMMA context check will be done in codegen phase
    Error1(n.LOC(), "Invalid swizzle mode: " + STR(n.GetSwizzleMode()) + ".");
    return false;
  }

  if (n.IsSparse()) {
    extern Option<bool> sim_sparse;
    if (!sim_sparse) {
      Warning(n.LOC(),
              "Sparse DMA is enabled without -sim; this path is experimental.");
    }
    if (n.operation != ".copy") {
      Error1(n.LOC(), "Sparse DMA only supports dma.copy.sp currently.");
      return false;
    }
    auto sp = n.GetSparsePattern();
    if (!(sp.first == 2 && sp.second == 4)) {
      Error1(n.LOC(), "Only 2:4 structured sparsity is supported; got " +
                          std::to_string(sp.first) + ":" +
                          std::to_string(sp.second) + ".");
      return false;
    }
  }

  if (!ReportUnknown(n, __FILE__, __LINE__)) return false;

  if (!isa<FutureType>(ty))
    Error1(n.LOC(), "Expect the DMA to produce a FutureType, but got '" +
                        PSTR(n.GetType()) + "'.");

  if (cast<FutureType>(ty)->IsAsync() && n.future.empty() && !n.HasEvent())
    Error1(n.LOC(),
           "A async DMA must have a named future or an event to wait on.");

  if (!n.future.empty() && cast<FutureType>(ty)->IsAsync())
    pending_async.insert(InScopeName(n.future));
  if (!n.chain_from.empty()) waited_async.insert(InScopeName(n.chain_from));

  if (!isa<AST::ChunkAt>(n.from) || !isa<SpannedType>(n.from->GetType()))
    Error1(n.LOC(),
           "The 'from' of DMA is not as expected: " + n.from->TypeNameString() +
               "(" + PSTR(n.from->GetType()) + ").");

  if (!isa<AST::ChunkAt>(n.to) || !isa<SpannedType>(n.to->GetType()))
    Error1(n.LOC(),
           "The 'to' of DMA is not as expected: " + n.to->TypeNameString() +
               "(" + PSTR(n.from->GetType()) + ").");

  auto& fty = n.from->GetType();
  auto& tty = n.to->GetType();

  auto sfty = cast<SpannedType>(fty);
  auto stty = cast<SpannedType>(tty);
  auto f_shape = sfty->GetShape();
  auto t_shape = stty->GetShape();

  if (n.IsSparse()) {
    if (!n.GetSrc()->NoTilingOperation() || !n.GetDst()->NoTilingOperation()) {
      Error1(n.LOC(), "Sparse DMA currently requires symbol-to-symbol copy "
                      "with no tiling.");
      return false;
    }
    if ((f_shape.Rank() != 2 && f_shape.Rank() != 3) ||
        (t_shape.Rank() != 2 && t_shape.Rank() != 3) || f_shape.IsDynamic() ||
        t_shape.IsDynamic()) {
      Error1(n.LOC(),
             "Sparse DMA currently requires static rank-2 or rank-3 tensors.");
      return false;
    }
  }

  if (n.operation == ".transp") {
    // no transposed shape need to be generated
    // do LogicalEqual() manually
    auto tc = cast<TransposeConfig>(n.config);
    if (sfty->e_type != stty->e_type || !f_shape.SameRankAs(t_shape)) {
      Error1(n.LOC(), "Type inconsistent between DMA 'from'(" + PSTR(fty) +
                          ") with " + PSTR(tc) + " and 'to'(" + PSTR(tty) +
                          ").");
    } else {
      auto& dim_values = tc->dim_values;
      for (size_t i = 0; i < dim_values.size(); ++i) {
        auto eq =
            sbe::oc_eq(f_shape.ValueAt(dim_values[i]), t_shape.ValueAt(i));
        assert(IsValidValueItem(eq));
        auto message = "Type inconsistent between DMA 'from'(" + PSTR(fty) +
                       ") with " + PSTR(tc) + " and 'to'(" + PSTR(tty) +
                       ") at the " + Ordinal(i + 1) + " dim.";
        CreateAssessment(eq, message, n.from->LOC(), n.from,
                         UsageType::ShapeCompatibility, &n);
      }
    }
  } else if (n.operation == ".pad") {
    // no padded shape need to be generated
    // do LogicalEqual() manually
    auto pc = cast<PadConfig>(n.config);
    if (sfty->e_type != stty->e_type || !f_shape.SameRankAs(t_shape)) {
      Error1(n.LOC(), "Type inconsistent between DMA 'from'(" + PSTR(fty) +
                          ") with " + PSTR(pc) + " and 'to'(" + PSTR(tty) +
                          ").");
    } else {
      size_t dim_count = f_shape.DimCount();
      for (size_t i = 0; i < dim_count; ++i) {
        auto h_i = cast<AST::Expr>(pc->pad_high->ValueAt(i));
        auto l_i = cast<AST::Expr>(pc->pad_low->ValueAt(i));
        auto m_i = cast<AST::Expr>(pc->pad_mid->ValueAt(i));
        if (!h_i->Opts().HasVal()) {
          VST_DEBUG(dbgs() << "Expression: " << PSTR(pc->pad_high->ValueAt(i))
                           << " does not have a value!\n");
          return false;
        }
        if (!l_i->Opts().HasVal()) {
          VST_DEBUG(dbgs() << "Expression: " << PSTR(pc->pad_low->ValueAt(i))
                           << " does not have a value!\n");
          return false;
        }
        if (!m_i->Opts().HasVal()) {
          VST_DEBUG(dbgs() << "Expression: " << PSTR(pc->pad_mid->ValueAt(i))
                           << " does not have a value!\n");
          return false;
        }

        auto pad_length =
            h_i->Opts().GetVal() + l_i->Opts().GetVal() + m_i->Opts().GetVal();
        if (!IsValueItemEqual(f_shape.ValueAt(i) + pad_length,
                              t_shape.ValueAt(i))) {
          Error1(n.LOC(), "Type inconsistent between DMA 'from'(" + PSTR(fty) +
                              ") with " + PSTR(pc) + " and 'to'(" + PSTR(tty) +
                              ").");
          break;
        }
      }
    }
  } else if (!(cast<SpannedType>(fty)->LogicalEqual(*tty))) {
    // check: to-buffer size must be larger or equal than from
    auto asrt = sbe::bop(OpCode::LE, f_shape.ElementCountValue(),
                         t_shape.ElementCountValue())
                    ->Normalize();
    assert(IsValidValueItem(asrt));

    auto message = "DMA to-buffer is too small (" +
                   STR(f_shape.ElementCountValue()) + " > " +
                   STR(t_shape.ElementCountValue()) + ")";
    CreateAssessment(asrt, message, n.LOC(), n.from,
                     UsageType::ShapeCompatibility, &n);

    bool emit_error = true;
    std::string msg;
    if (f_shape != t_shape && f_shape.IsValid() && t_shape.IsValid()) {
      // ignore the symbolic value inconsistance
      for (size_t i = 0; i < f_shape.Rank(); ++i) {
        auto& fv = f_shape.ValueAt(i);
        auto& tv = t_shape.ValueAt(i);
        if (!IsValueItemEqual(fv, tv)) {
          if (tv->IsSymbolic() || fv->IsSymbolic()) {
            emit_error = false;
            msg += " 'from[" + std::to_string(i) + "](" + STR(fv) +
                   ")' v.s. 'to[" + std::to_string(i) + "](" + STR(tv) + ")";
            continue;
          } else {
            emit_error = true;
            break;
          }
        }
      }
    }
    auto f_sp_ty = cast<SpannedType>(fty);
    auto t_sp_ty = cast<SpannedType>(tty);
    bool supported_conditional_dma = false;
    if (f_shape.Rank() == 2 && t_shape.Rank() == 2 && f_shape.IsValid() &&
        t_shape.IsValid() && VIIsInt(f_shape.ValueAt(1)) &&
        VIIsInt(t_shape.ValueAt(1)) &&
        *VIInt(f_shape.ValueAt(1)) == *VIInt(t_shape.ValueAt(1))) {
      auto f_sto = f_sp_ty->GetStorage();
      auto t_sto = t_sp_ty->GetStorage();
      supported_conditional_dma =
          ((f_sto == Storage::GLOBAL || f_sto == Storage::DEFAULT) &&
           t_sto == Storage::SHARED) ||
          (f_sto == Storage::SHARED &&
           (t_sto == Storage::GLOBAL || t_sto == Storage::DEFAULT));
    }

    bool smaller_source_dma = false;
    if (supported_conditional_dma && sfty->e_type == stty->e_type &&
        f_shape.SameRankAs(t_shape)) {
      smaller_source_dma = true;
      for (size_t i = 0; i < f_shape.Rank(); ++i) {
        auto dim_le =
            sbe::bop(OpCode::LE, f_shape.ValueAt(i), t_shape.ValueAt(i))
                ->Normalize();
        assert(IsValidValueItem(dim_le));
        auto dim_le_bool = VIBool(dim_le);
        if (!dim_le_bool || !dim_le_bool.value()) {
          smaller_source_dma = false;
          break;
        }
      }
    }

    if (emit_error && !smaller_source_dma) {
      Error1(n.LOC(), "Type inconsistent between DMA 'from'(" + PSTR(fty) +
                          ") and 'to'(" + PSTR(tty) + ").");
    }
  }

  for (const auto& ca_node : {n.from, n.to}) {
    const auto& ca = cast<AST::ChunkAt>(ca_node);
    auto sty = GetSpannedType(GetSymbolType(ca->RefSymbol()));
    assert(sty);
    Shape s = sty->GetShape();
    ValueList strd = sty->GetStrides();

    for (auto& op : ca->AllOperations()) {
      if (auto rop = dyn_cast<AST::SOP::Reshape>(op)) {
        switch (IsContiguous(s, strd)) {
        case Modality::NOT:
          Warning(rop->LOC(),
                  "'span_as' is applied to non-contiguous spanned data (" +
                      STR(s) + " {" + STR(strd) + "}).");
          break;
        case Modality::MAY:
          Warning(rop->LOC(),
                  "'span_as' may be applied to non-contiguous spanned data (" +
                      STR(s) + " {" + STR(strd) + "}).");
          break;
        case Modality::MUST:
        default: break;
        }
      }
      s = op->GetBlockShape();
      strd = op->GetBlockStrides();
    }
  }
#if 0
  // Check if the spanned operations are valid
  for (const auto& ca_node : {n.from, n.to}) {
    const auto& ca = cast<AST::ChunkAt>(ca_node);
    if (ca->NoTilingOperation()) continue;
    Shape original_shape = GetShape(GetSymbolType(ca->RefSymbol()));
    bool has_noncontiguous = false;
    size_t last_tiling = 0;
    for (size_t i = 0; i < ca->OpCount(); ++i)
      if (!isa<AST::SOP::Reshape>(ca->OpAt(i))) last_tiling = i;
    bool has_reshape = false;
    for (size_t i = 0; i < ca->OpCount(); ++i) {
      const auto& sop = ca->OpAt(i);
      if (isa<AST::SOP::Reshape>(ca->OpAt(ca->OpCount() - i - 1)))
        has_reshape = true;
      auto is_contiguous = AST::IsContiguousSOp(*sop, original_shape);
      if (auto val = std::get_if<bool>(&is_contiguous); val && *val == false) {
        has_noncontiguous = true;
        if (i != last_tiling)
          Warning(sop->LOC(), "There are more than one tiling executed in "
                              "noncontiguous manner.");
      } else if (!val) {
        if (i != last_tiling)
          Warning(sop->LOC(), "There may have more than one tiling executed in "
                              "noncontiguous manner.");
        if (has_reshape)
          Warning(sop->LOC(),
                  "The reshape operation inside DMA expression maybe is "
                  "executed on a noncontiguous tiling result.");
      }
      if (has_noncontiguous && isa<AST::SOP::Reshape>(sop))
        Warning(sop->LOC(), "The reshape operation inside DMA expression is "
                            "executed on a noncontiguous tiling result.");
      original_shape = sop->GetBlockShape();
    }
  }
#endif

  if (auto dst = dyn_cast<AST::ChunkAt>(n.GetTo())) {
    auto fsty = GetSpannedType(n.GetFrom()->GetType());
    auto tsty = GetSpannedType(dst->GetType());
    if (fsty && tsty && tsty->GetStorage() == Storage::SHARED &&
        (fsty->GetStorage() == Storage::GLOBAL ||
         fsty->GetStorage() == Storage::DEFAULT)) {
      shared_tensor_producers[InScopeName(dst->RefSymbol())] = &n;
    }
  }

  return true;
}

bool SemaChecker::VisitNode(AST::MMA& n) {
  auto& op = *n.GetOperation();
  switch (op.Tag()) {
  case AST::MMAOperation::Fill: break;
  case AST::MMAOperation::LoadR: break;
  case AST::MMAOperation::Desc: break;
  case AST::MMAOperation::Load: {
    // Keep explicit mma.load swizzles consistent with the DMA/TMA that fills
    // the referenced shared-memory tensor.
    auto load_from = op.LoadFrom();
    if (load_from && isa<AST::ChunkAt>(load_from)) {
      auto mma_swizzle = op.GetSwizzleMode();
      auto swiz_set = CCtx().TargetSwizzleModes();
      if (!swiz_set.empty() && !swiz_set.count(mma_swizzle)) {
        Error1(n.LOC(),
               "Invalid swizzle value in MMA load: " + STR(mma_swizzle) + ".");
        return false;
      }

      auto it =
          shared_tensor_producers.find(InScopeName(load_from->RefSymbol()));
      if (op.HasExplicitSwizzle()) {
        if (it != shared_tensor_producers.end()) {
          auto* dma = it->second;
          auto tensor_name = load_from->RefSymbol();
          auto dma_label = dma->IsTMA() ? "TMA" : "DMA";
          if (!dma->HasExplicitSwizzle() &&
              dma->GetSwizzleMode() == SwizMode::NONE) {
            dma->SetSwizzleMode(mma_swizzle);
            dma->AddNote("swizzle_inferred_from_mma", STR(mma_swizzle));
            Note(n.LOC(), std::string("inferred ") + dma_label + " swizzle '" +
                              STR(mma_swizzle) + "' for tensor '" +
                              tensor_name +
                              "' from explicit mma.load swizzle.");
          } else if (dma->GetSwizzleMode() != mma_swizzle) {
            auto origin =
                dma->HasExplicitSwizzle()
                    ? (std::string("explicit ") + dma_label + " swizzle '")
                    : (std::string("previously inferred ") + dma_label +
                       " swizzle '");
            Warning(n.LOC(), std::string("explicit mma.load swizzle '") +
                                 STR(mma_swizzle) + "' conflicts with " +
                                 origin + STR(dma->GetSwizzleMode()) +
                                 "' for tensor '" + tensor_name + "'.");
            Note(dma->LOC(), std::string(dma_label) + " affecting tensor '" +
                                 tensor_name + "' is here.");
          }
        }
      } else if (it != shared_tensor_producers.end()) {
        auto* dma = it->second;
        auto dma_swizzle = dma->GetSwizzleMode();
        if (dma_swizzle != SwizMode::NONE) {
          op.SetSwizzleMode(dma_swizzle);
          auto tensor_name = load_from->RefSymbol();
          auto dma_label = dma->IsTMA() ? "TMA" : "DMA";
          Note(n.LOC(), std::string("inferred mma.load swizzle '") +
                            STR(dma_swizzle) + "' from " + dma_label +
                            " swizzle for tensor '" + tensor_name + "'.");
        }
      }

      // Provide guidance on TILE_K constraints
      // Note: These are recommendations, not enforced constraints
      // Users should set TILE_K accordingly:
      // - swizzle(128): TILE_K = 64 (default)
      // - swizzle(64):  TILE_K = 32
      // - swizzle(32):  TILE_K = 16
      if (mma_swizzle == SwizMode::B64) {
        VST_DEBUG(dbgs() << "MMA load with swizzle(64): Consider setting "
                            "TILE_K = 32 for optimal performance\n");
      } else if (mma_swizzle == SwizMode::B32) {
        VST_DEBUG(dbgs() << "MMA load with swizzle(32): Consider setting "
                            "TILE_K = 16 for optimal performance\n");
      }
    }
    break;
  }
  case AST::MMAOperation::Exec: {
    auto& a_sym = AST::FragName(op.ExecOperand(1));
    auto& b_sym = AST::FragName(op.ExecOperand(2));
    auto& c_sym = AST::FragName(op.ExecOperand(0));
    auto a_ty = GetSpannedType(GetSymbolType(a_sym));
    auto b_ty = GetSpannedType(GetSymbolType(b_sym));
    auto c_ty = GetSpannedType(GetSymbolType(c_sym));
    bool old_ec = error_count;
    if (a_ty == nullptr)
      Error1(n.LOC(), "Expect `" + a_sym + "' to contain a spanned data.");
    if (b_ty == nullptr)
      Error1(n.LOC(), "Expect `" + b_sym + "' to contain a spanned data.");
    if (c_ty == nullptr)
      Error1(n.LOC(), "Expect `" + c_sym + "' to contain a spanned data.");
    if (old_ec != error_count) return false;

    auto a_shape = a_ty->GetShape();
    auto b_shape = b_ty->GetShape();
    auto c_shape = c_ty->GetShape();
    if ((a_shape.Rank() != 2) || a_shape.IsDynamic())
      Error1(n.LOC(), "Expect `" + a_sym +
                          "' to be a matrix with fixed size, but got: " +
                          STR(a_shape) + ".");
    if ((b_shape.Rank() != 2) || b_shape.IsDynamic())
      Error1(n.LOC(), "Expect `" + b_sym +
                          "' to be a matrix with fixed size, but got: " +
                          STR(b_shape) + ".");
    if ((c_shape.Rank() != 2) || c_shape.IsDynamic())
      Error1(n.LOC(), "Expect `" + c_sym +
                          "' to be a matrix with fixed size, but got: " +
                          STR(c_shape) + ".");
    if (old_ec != error_count) return false;

    bool shape_match = true;
    bool sparse_packed_match = false;
    ValueList cs_vals;
    switch (op.GetMethod()) {
    case AST::MMAOperation::ROW_ROW:
      if (!sbe::ceq(a_shape.ValueAt(1), b_shape.ValueAt(1)))
        shape_match = false;
      if (op.IsSparse() && !shape_match) {
        auto a_k = a_shape.ValueAt(1);
        auto b_k = b_shape.ValueAt(1);
        auto a_k2 = a_k * sbe::nu(2);
        if (sbe::ceq(a_k2, b_k)) sparse_packed_match = true;
      }
      cs_vals.push_back(a_shape.ValueAt(0));
      cs_vals.push_back(b_shape.ValueAt(0));
      break;
    case AST::MMAOperation::ROW_COL:
      if (!sbe::ceq(a_shape.ValueAt(1), b_shape.ValueAt(0)))
        shape_match = false;
      if (op.IsSparse() && !shape_match) {
        auto a_k = a_shape.ValueAt(1);
        auto b_k = b_shape.ValueAt(0);
        auto a_k2 = a_k * sbe::nu(2);
        if (sbe::ceq(a_k2, b_k)) sparse_packed_match = true;
      }
      cs_vals.push_back(a_shape.ValueAt(0));
      cs_vals.push_back(b_shape.ValueAt(1));
      break;
    case AST::MMAOperation::COL_ROW:
      if (!sbe::ceq(a_shape.ValueAt(0), b_shape.ValueAt(1)))
        shape_match = false;
      if (op.IsSparse() && !shape_match) {
        auto a_k = a_shape.ValueAt(0);
        auto b_k = b_shape.ValueAt(1);
        auto a_k2 = a_k * sbe::nu(2);
        if (sbe::ceq(a_k2, b_k)) sparse_packed_match = true;
      }
      cs_vals.push_back(a_shape.ValueAt(1));
      cs_vals.push_back(b_shape.ValueAt(0));
      break;
    case AST::MMAOperation::COL_COL:
      if (!sbe::ceq(a_shape.ValueAt(0), b_shape.ValueAt(0)))
        shape_match = false;
      if (op.IsSparse() && !shape_match) {
        auto a_k = a_shape.ValueAt(0);
        auto b_k = b_shape.ValueAt(0);
        auto a_k2 = a_k * sbe::nu(2);
        if (sbe::ceq(a_k2, b_k)) sparse_packed_match = true;
      }
      cs_vals.push_back(a_shape.ValueAt(1));
      cs_vals.push_back(b_shape.ValueAt(1));
      break;
    default: choreo_unreachable("unsupported mma execution method.");
    }
    if (!shape_match && !sparse_packed_match) {
      Error1(n.LOC(), "MMA: matrix shapes do not match: `" + a_sym + "'(" +
                          STR(a_shape) + ") v.s. `" + b_sym + "'(" +
                          STR(b_shape) + ").");
      return false;
    }

    if (op.IsSparse()) {
      ValueItem k_dim;
      switch (op.GetMethod()) {
      case AST::MMAOperation::ROW_ROW:
      case AST::MMAOperation::ROW_COL: k_dim = a_shape.ValueAt(1); break;
      case AST::MMAOperation::COL_ROW:
      case AST::MMAOperation::COL_COL: k_dim = a_shape.ValueAt(0); break;
      default: break;
      }
      if (auto kv = VIInt(k_dim)) {
        if ((*kv % 4) != 0) {
          Error1(
              n.LOC(),
              "Sparse MMA requires K dimension to be a multiple of 4. Got: " +
                  STR(k_dim));
          return false;
        }
      } else {
        Warning(n.LOC(), "Sparse MMA expects K dimension multiple of 4; unable "
                         "to verify at compile time.");
      }
    }
  } break;
  case AST::MMAOperation::Scale: {
    auto& c_sym = AST::FragName(op.ScaleAccumulator());
    auto c_ty = GetSpannedType(GetSymbolType(c_sym));
    auto a_ty = GetSpannedType(op.ScaleA()->GetType());
    bool old_ec = error_count;
    if (c_ty == nullptr)
      Error1(n.LOC(), "Expect `" + c_sym + "' to contain a spanned data.");
    if (a_ty == nullptr)
      Error1(n.LOC(), "Expect MMA scale A to contain a spanned data.");
    if (!isa<ScalarType>(op.ScaleB()->GetType()))
      Error1(n.LOC(), "Expect MMA scale B to be a scalar expression.");
    if (old_ec != error_count) return false;

    auto c_shape = c_ty->GetShape();
    auto a_shape = a_ty->GetShape();
    if ((c_shape.Rank() != 2) || c_shape.IsDynamic())
      Error1(n.LOC(), "Expect `" + c_sym +
                          "' to be a matrix with fixed size, but got: " +
                          STR(c_shape) + ".");
    if ((a_shape.Rank() != 2) || a_shape.IsDynamic())
      Error1(
          n.LOC(),
          "Expect MMA scale A to be a matrix tile with fixed size, but got: " +
              STR(a_shape) + ".");
    if (old_ec != error_count) return false;

    bool is_standard = sbe::ceq(a_shape.ValueAt(0), c_shape.ValueAt(0)) &&
                       sbe::ceq(a_shape.ValueAt(1), sbe::nu(1));
    bool is_transposed = sbe::ceq(a_shape.ValueAt(0), sbe::nu(1)) &&
                         sbe::ceq(a_shape.ValueAt(1), c_shape.ValueAt(0));
    if (!is_standard && !is_transposed)
      Error1(n.LOC(), "MMA scale A must have shape [M, 1] or [1, M].");
  } break;
  case AST::MMAOperation::Store: break;
  case AST::MMAOperation::Commit: break;
  case AST::MMAOperation::Wait: {
    int depth = op.WaitDepth();
    if (depth > 0 && !foreach_stack.empty()) {
      auto* fe = foreach_stack.back();
      int execs_per_iter = CountMMAExecsInBody(fe->GetBody(), &n);
      if (execs_per_iter < 1) execs_per_iter = 1;
      int64_t extent = GetForeachStaticExtent(*fe);
      if (extent > 0 && depth >= extent * execs_per_iter) {
        Error1(n.LOC(), "mma.wait<" + std::to_string(depth) +
                            "> depth must be less than the enclosing loop "
                            "extent (" +
                            std::to_string(extent) +
                            ") * commits per iteration (" +
                            std::to_string(execs_per_iter) + ").");
      } else if (extent < 0) {
        auto ub = GetForeachDynamicExtent(*fe);
        if (ub) {
          auto limit = (execs_per_iter > 1) ? ub * sbe::nu(execs_per_iter) : ub;
          auto cond = sbe::oc_lt(sbe::nu(depth), limit);
          FCtx(fname).GetAssessor(*this).Assess(
              AssessPolicy::Error, cond,
              "mma.wait<" + std::to_string(depth) +
                  "> depth must be less than the enclosing loop extent"
                  " * commits per iteration (" +
                  std::to_string(execs_per_iter) + ").",
              UsageType::ShapeCompatibility, AssessType::ENTRY, n.LOC(), &n);
        }
      }
    }
  } break;
  default: choreo_unreachable("unsupported mma operation.");
  }
  return true;
}

bool SemaChecker::VisitNode(AST::ChunkAt& n) {
  if (!ReportUnknown(n, __FILE__, __LINE__)) return false;

  if (n.indices && n.indices->Count()) {
    auto ty = NodeType(*n.data);
    auto arr_ty = cast<ArrayType>(ty);
    assert(arr_ty && "expect the array symbol to be an array type.");

    size_t rank = arr_ty->ArrayRank();
    size_t idx_cnt = n.indices->Count();
    // need exactly `rank` indices to access the array!
    if (idx_cnt != rank) {
      Error1(n.LOC(), "Invalid array access: expected " + std::to_string(rank) +
                          " dimensions, but " + std::to_string(idx_cnt) +
                          " were used.");
      return false;
    }

    // check if any indices are out of bound
    for (size_t i = 0; i < rank; ++i) {
      int bound = *VIInt(arr_ty->Dimension(i));
      auto dim_bound = arr_ty->Dimension(i);

      auto expr = n.indices->ValueAt(i);
      ValueItem index = GetInvalidValueItem();
      if (auto e = dyn_cast<AST::Expr>(expr)) {
        if (!e->Opts().HasVal()) {
          VST_DEBUG(dbgs() << "Expression: " << PSTR(expr)
                           << " does not have a value!\n");
          return false;
        }
        index = e->Opts().GetVal();
      } else if (auto id = dyn_cast<AST::Identifier>(expr)) {
        index = sbe::sym(InScopeName(id->name));
      } else if (auto lit = dyn_cast<AST::IntLiteral>(expr)) {
        index = sbe::nu(lit->Val());
      }

      if (!IsValidValueItem(index)) {
        Error1(expr->LOC(), "The " + Ordinal(i + 1) +
                                " subscription index can not be evaluated.");
        return true;
      }

      auto asrt0 = sbe::bop(OpCode::LT, index, dim_bound)->Normalize();
      auto asrt1 = sbe::bop(OpCode::GE, index, sbe::nu(0))->Normalize();
      assert(IsValidValueItem(asrt0) && IsValidValueItem(asrt1));

      auto expr_bounds = InferExprBounds(this, index);
      auto active_guard = ActiveScopePredicate();
      auto constrained = ExpressionIsConstrained(index);
      bool upper_safe = IsValidValueItem(expr_bounds.ub) &&
                        sbe::clt(expr_bounds.ub, dim_bound);
      bool lower_safe = IsValidValueItem(expr_bounds.lb) &&
                        sbe::cge(expr_bounds.lb, sbe::nu(0));
      if (!upper_safe)
        upper_safe = ScopeContainsPredicate(active_guard, asrt0) ||
                     ScopeImpliesUpperBound(active_guard, index, dim_bound);
      if (!lower_safe) lower_safe = ScopeContainsPredicate(active_guard, asrt1);

      if (lower_safe && upper_safe) continue;

      // skip checking the one with 'nil' value though
      if (!expr_bounds.IsValid() && !IsComputable(index)) {
        Error1(expr->LOC(), "The " + Ordinal(i + 1) +
                                " subscription index can not be evaluated.");
        return true;
      }

      auto message = "Index " + STR(index) + " is out of bounds of the " +
                     Ordinal(i + 1) + " dimension of array '" + PSTR(n.data) +
                     "', where the valid range is [0, " +
                     std::to_string(bound) + ")";
      // Array subscript index bounds are pre-conditions on the index value,
      // not the foreach iteration variable itself.  Bypass CreateAssessment
      // to force ENTRY placement: static violations become compile errors,
      // and runtime assertions are placed in the host wrapper (not inside the
      // kernel body).  See similar fix in ParallelBy and WithIn visitors.
      auto upper_pred = expr_bounds.IsValid()
                            ? sbe::oc_lt(expr_bounds.ub, dim_bound)->Normalize()
                            : asrt0;
      auto lower_pred =
          expr_bounds.IsValid() && !constrained
              ? sbe::oc_ge(expr_bounds.lb, sbe::nu(0))->Normalize()
              : asrt1;
      if (!upper_safe)
        FCtx(fname).GetAssessor(*this).Assess(
            AssessPolicy::Error, upper_pred, message, UsageType::ElementAccess,
            AssessType::ENTRY, expr->LOC(), expr.get());
      if (!lower_safe)
        FCtx(fname).GetAssessor(*this).Assess(
            AssessPolicy::Error, lower_pred, message, UsageType::ElementAccess,
            AssessType::ENTRY, expr->LOC(), expr.get());
    }
  }

  return true;
}

bool SemaChecker::VisitNode(AST::Trigger& n) {
  for (auto& f : n.GetEvents()) {
    auto fty = NodeType(*f);
    if (!isa<EventType>(fty)) {
      Error1(n.LOC(),
             "trigger a non-event type " + PSTR(f) + " (" + PSTR(fty) + ").");
      continue;
    }
    if (auto id = AST::GetIdentifier(*f))
      pending_async.insert(InScopeName(id->name));
    else if (auto e = dyn_cast<AST::Expr>(f)) {
      if (e->op != "elemof") {
        Error1(n.LOC(),
               "expect a element-of operation but got " + e->op + ").");
        continue;
      }
      auto bid = GetArrayBaseSymbol(*e);
      pending_async.insert(InScopeName(bid->name));
    }
  }
  return true;
}

bool SemaChecker::VisitNode(AST::Wait& n) {
  for (auto& f : n.GetTargets()) {
    auto fty = NodeType(*f);
    if (!isa<FutureType>(fty) && !isa<EventType>(fty)) {
      Error1(n.LOC(),
             "wait for a non-async type " + PSTR(f) + " (" + PSTR(fty) + ").");
      continue;
    }
    if (auto id = AST::GetIdentifier(*f))
      waited_async.insert(InScopeName(id->name));
    else if (auto e = dyn_cast<AST::Expr>(f)) {
      if (e->op != "elemof") {
        Error1(n.LOC(),
               "expect a element-of operation but got " + e->op + ").");
        continue;
      }
      auto bid = GetArrayBaseSymbol(*e);
      waited_async.insert(InScopeName(bid->name));
    }
  }

  return true;
}

bool SemaChecker::VisitNode(AST::Call& n) {
  if (n.template_args) {
    size_t count = 0;
    for (auto& v : n.template_args->AllValues()) {
      count++;
      auto ty = NodeType(*v);
      auto expr = cast<AST::Expr>(v);
      if (expr->IsReference() && isa<AST::DataType>(expr->GetReference())) {
        continue;
      }
      // must be a scalar type
      if (!CanYieldAnInteger(ty))
        Error1(n.LOC(),
               "The " + Ordinal(count) + " template argument of type '" +
                   PSTR(ty) +
                   "` can not be used to instantiate the kernel function.");
      // fail if the template argument can not be evaluated as a compile-time
      // constant
      if (!expr->Opts().HasVal() || !expr->Opts().GetVal()->IsNumeric())
        Warning(n.LOC(), "The " + Ordinal(count) +
                             " template argument of type '" + PSTR(ty) +
                             "` can not be evaluated at choreo compile time.");
    }
  }

  if (n.IsBIF()) {
    const auto func_name = n.function->name;
    if (func_name == "assert") {
      auto cmp = n.arguments->ValueAt(0);
      if (auto cexpr = dyn_cast<AST::Expr>(cmp);
          cexpr && cexpr->Opts().HasVal()) {
        if (auto bv = VIBool(cexpr->Opts().GetVal());
            bv && (bv.value() == false)) {
          std::string msg;
          if (auto str = dyn_cast<AST::StringLiteral>(n.arguments->ValueAt(1)))
            msg = str->value;
          else
            choreo_unreachable(
                "choreo assertion requires a string message as the second.");
          Error1(n.LOC(), "choreo assertion abort: " + msg);
        }
      }
    } else if (func_name == "croq::cuda::setreg_inc" ||
               func_name == "croq::cuda::setreg_dec") {
      if (n.arguments->Count() != 1) {
        Error1(n.LOC(), "'" + func_name + "' expects exactly one argument.");
      } else {
        auto arg = dyn_cast<AST::Expr>(n.arguments->ValueAt(0));
        if (!arg) {
          Error1(n.LOC(),
                 "'" + func_name + "' expects an integer expression argument.");
        } else {
          auto aty = NodeType(*arg);
          if (!isa<ScalarIntegerType>(aty)) {
            Error1(n.LOC(), "'" + func_name +
                                "' expects an integer argument but got '" +
                                PSTR(aty) + "'.");
          } else if (!arg->Opts().HasVal() ||
                     !arg->Opts().GetVal()->IsNumeric()) {
            Error1(n.LOC(),
                   "'" + func_name +
                       "' argument must be a compile-time integer constant.");
          } else if (auto reg_limit = VIInt(arg->Opts().GetVal());
                     !reg_limit || reg_limit.value() <= 0) {
            Error1(n.LOC(),
                   "'" + func_name + "' argument must be a positive integer.");
          }
        }
      }
    }
    if (n.IsArith()) {
      auto pty = NodeType(*n.arguments->ValueAt(0));
      if ((n.function->name == "__min" || n.function->name == "__max") &&
          CanYieldAnInteger(pty)) {
        for (size_t i = 1; i < n.arguments->Count(); ++i) {
          auto sty = NodeType(*n.arguments->ValueAt(i));
          if (!CanYieldAnInteger(sty))
            Error1(n.LOC(),
                   "expect the " + std::to_string(i) +
                       "th argument to be an integer-compatible type.");
        }
        return true;
      }

      if (!(isa<ScalarFloatType>(pty) || isa<VectorType>(pty)))
        Error1(
            n.LOC(),
            "expect the argument to be a float type or vector type but got '" +
                PSTR(pty) + "'.");

      for (size_t i = 1; i < n.arguments->Count(); ++i) {
        auto sty = NodeType(*n.arguments->ValueAt(i));
        if (!sty->ApprxEqual(*pty))
          Error1(n.LOC(),
                 "expect the " + std::to_string(i) +
                     "th argument to be the same type as the first one.");
      }
    }
    // Atomic op/type/storage validation is handled in earlysema.
  }

  if (n.IsBIF()) return true;

  // resolve device functions
  if (!resolve_fns) return true;
  if (n.device_functions.empty()) return true;

  bool function_found = false;
  auto function_name = n.function->name;
  ptr<AST::DeviceFunctionDecl> matched_function = nullptr;
  std::string mismatch_msg = "";
  for (size_t i = 0; i < n.device_functions.size(); i++) {
    auto device_function = n.device_functions[i];
    bool arg_match = true;
    for (size_t param_index = 0; param_index < n.arguments->Count();
         param_index++) {
      auto pnode = n.arguments->ValueAt(param_index);
      auto arg_ty = NodeType(*pnode);
      auto param_ty = device_function->param_types[param_index];

      std::string attr = param_ty->attr;
      if (auto spanned_ty = dyn_cast<SpannedType>(arg_ty)) {
        auto m_ty = spanned_ty->GetStorage();
        if ((attr.find("__private__") != std::string::npos) ||
            (attr.find("__attribute__((address_space(5)))") !=
             std::string::npos)) {
          if (m_ty != Storage::LOCAL)
            arg_match = false;
          else
            pnode->AddNote("annotate_as"); // annotate the addrspace
        } else if ((attr.find("__shared__") != std::string::npos) ||
                   (attr.find("__attribute__((shared))") !=
                    std::string::npos)) {
          if (m_ty != Storage::SHARED)
            arg_match = false;
          else
            pnode->AddNote("annotate_as");
        }

        if (!arg_match) {
          mismatch_msg = "the type of " + std::to_string(param_index + 1) +
                         "th argument '" + PSTR(arg_ty) + "' is not " +
                         STR(m_ty) + ".";
          break;
        }
      }
    }

    if (arg_match) {
      matched_function = device_function;
      function_found = true;
    }
  } // end of device function loop

  if (!function_found) {
    Warning(n.LOC(), "unable to find a device function '" + function_name +
                         "', because " + mismatch_msg);
  } else if (debug_visit) {
    if (matched_function->IsTemplated())
      dbgs() << "Find instantiated device function '";
    else
      dbgs() << "Find matched device function '";
    dbgs() << matched_function->name << "' -> "
           << PSTR(matched_function->ret_type) << " (";
    for (size_t i = 0; i < matched_function->param_types.size(); ++i) {
      auto& pt = matched_function->param_types[i];
      if (i > 0) dbgs() << ", ";
      dbgs() << PSTR(pt);
    }
    dbgs() << ")\n";
  }

  return true;
}

bool SemaChecker::VisitNode(AST::Rotate& n) {
  size_t index = 0;
  for (auto s : n.ids->AllValues()) {
    if (auto id = AST::GetIdentifier(*s))
      waited_async.insert(InScopeName(id->name));

    if (index == 0) {
      index++;
      continue;
    }
    auto lty = NodeType(*n.ids->ValueAt(index - 1));
    auto rty = NodeType(*n.ids->ValueAt(index));
    if (*lty != *rty)
      Error1(n.LOC(), "swapping values of different types (" + PSTR(lty) +
                          " vs. " + PSTR(rty));

    auto lid = AST::GetIdentifier(*n.ids->ValueAt(index - 1));
    auto rid = AST::GetIdentifier(*n.ids->ValueAt(index));
    assert(lid && rid && "no identifier is found.");
    auto l_scope = GetScope(InScopeName(lid->name));
    auto r_scope = GetScope(InScopeName(rid->name));
    if (l_scope != r_scope)
      Error1(n.LOC(),
             "swapping values defined in different scopes is forbidden (" +
                 InScopeName(lid->name) + " vs. " + InScopeName(rid->name));

    index++;
  }
  return true;
}

bool SemaChecker::VisitNode(AST::Select& n) {
  size_t ec = error_count;

  if (!CanYieldAnInteger(n.select_factor->GetType()))
    Error1(n.select_factor->LOC(),
           "Expect " + PSTR(n.select_factor) +
               " to be an integer type but got " +
               NodeType(*n.select_factor)->TypeNameString() + ".");

  auto expr_list = n.expr_list;
  auto expr0 = expr_list->ValueAt(0);
  if (!isa<FutureType>(NodeType(*expr0)) &&
      !isa<SpannedType>(NodeType(*expr0))) {
    Error1(expr0->LOC(),
           "Expect " + PSTR(expr0) + " to be a future/spanned type.");
    return false;
  }

  for (auto expr : expr_list->AllValues()) {
    if (auto id = AST::GetIdentifier(*expr))
      waited_async.insert(InScopeName(id->name)); // can not check statically

    if (*NodeType(*expr) == *NodeType(*expr0)) continue;

    Error1(expr->LOC(), "Type mismatch inside SELECT: " + PSTR(expr) + "(" +
                            TYPE_STR(expr) + ") vs. " + PSTR(expr0) + "(" +
                            TYPE_STR(expr0) + ").");
  }

  int64_t select_value_cnt = static_cast<int64_t>(expr_list->Count());
  if (auto il = AST::GetIntLiteral(*n.select_factor)) {
    if (il->Val() < 0 || il->Val() >= select_value_cnt)
      Error1(il->LOC(), "The select factor `" + PSTR(il) +
                            "` is not in bound [0, " +
                            std::to_string(select_value_cnt) + ")");
  } else {
    if (n.select_factor->Opts().HasVal()) {
      auto v = n.select_factor->Opts().GetVal();

      // Pass &n (the Select node) as emit_node because Expr::accept does not
      // call AfterVisit, so the select_factor pointer would never match
      // in emitted assessments. Select::accept does call AfterVisit.
      CreateAssessment(sbe::oc_ge(v, sbe::nu(0)),
                       "The select factor `" + PSTR(n.select_factor) +
                           "` should be greater than or equal to 0",
                       n.select_factor->LOC(), n.select_factor,
                       UsageType::ElementAccess, &n);
      CreateAssessment(
          sbe::oc_lt(v, sbe::nu(select_value_cnt)),
          "The select factor `" + PSTR(n.select_factor) +
              "` should be less than " + std::to_string(select_value_cnt) +
              ", which is the count of values in the select statement",
          n.select_factor->LOC(), n.select_factor, UsageType::ElementAccess,
          &n);
    }
  }

  for (auto& v : n.expr_list->AllValues())
    if (input_deps.Contains(v)) {
      input_deps.Add(n);
      break;
    }

  return ec == error_count;
}

bool SemaChecker::VisitNode(AST::Return& n) {
  if (n.value) {
    auto vty = NodeType(*n.value);
    if (!(isa<SpannedType>(vty) || isa<ScalarType>(vty))) {
      Error1(n.LOC(),
             "returning value with type '" + PSTR(vty) + "' is not supported.");
      return false;
    }
  }

  return true;
}

bool SemaChecker::ReportUnknownSymbol(const std::string& name,
                                      const location& loc, const char* file,
                                      int line) {
  if (isa<UnknownType>(GetSymbolType(name))) {
    Error1(loc, "failed to obtain the type of `" + name + "'.");
    VST_DEBUG(dbgs() << file << ":" << line << "\n");
    return false;
  }
  return true;
}

bool SemaChecker::ReportUnknown(AST::Node& n, const char* file, int line,
                                bool ignore_detail) {
  if (STR(n) == "_") return true; // ignore built-in unit iv.
  if (isa<UnknownType>(NodeType(n))) {
    Error1(n.LOC(), "failed to obtain a type.");
    VST_DEBUG(dbgs() << file << ":" << line << ", " << STR(n) << "\n");
    return false;
  }

  if (!ignore_detail && !NodeType(n)->HasSufficientInfo()) {
    Error1(n.LOC(), "failed to obtain a type with sufficient info.");
    VST_DEBUG(dbgs() << file << ":" << line << ", " << STR(n) << "("
                     << PSTR(NodeType(n)) << ")\n");
    return false;
  }

  return true;
}

void SemaChecker::CreateAssessment(const ValueItem& pred,
                                   const std::string& message,
                                   const location& l, const ptr<AST::Node>& n,
                                   UsageType uty, AST::Node* emit_node) {
  // Classification lattice: ENTRY < HOIST_SITE < USE_SITE.
  // Start at ENTRY and escalate upward as needed.
  auto aty = AssessType::ENTRY;

  // When the node references locally-defined names (buffer objects that can be
  // redefined), the assertion is hoistable but must stay after the latest
  // relevant definition site.
  if (local_deps.Contains(n)) aty = AssessType::HOIST_SITE;

  // When the node has a BoundedType (loop iteration variable from foreach /
  // with-in), the value varies between iterations and the assertion must be
  // checked at each use site - escalate to USE_SITE.
  // NOTE: Callers whose assertions are about bounds (not the variable itself)
  // must bypass this function and call Assess directly with ENTRY - see the
  // ParallelBy and WithIn visitors.
  if (isa<BoundedType>(NodeType(*n))) aty = AssessType::USE_SITE;

  // Log when the assertion is unrelated to any input - for diagnostic purposes
  // only; the assertion still gets ENTRY type to preserve runtime checking.
  if (aty == AssessType::ENTRY && !input_deps.Contains(n))
    if (isa<AST::Expr>(n) || isa<AST::NamedVariableDecl>(n) ||
        isa<AST::Assignment>(n))
      VST_DEBUG(dbgs() << "questionable: check is not related to input: "
                       << PSTR(n) << ".\n");

  auto active_guard = ActiveScopePredicate();
  // Conservative: if we are inside a conditional scope, don't allow this
  // assertion to be hoisted all the way to function entry.  Escalating from
  // ENTRY to HOIST_SITE bypasses the early-return in HoistAssertions and lets
  // the hoisting pass decide the proper placement conservatively.
  if (IsValidValueItem(active_guard) && aty == AssessType::ENTRY)
    aty = AssessType::HOIST_SITE;

  // Try to prove the predicate using interval analysis on variable ranges
  // narrowed by both BoundedType declarations and active scope predicates.
  auto effective_pred = pred;
  if (!VIBool(pred) && IsValidValueItem(active_guard)) {
    auto proven = TryProveWithIntervals(this, pred, active_guard);
    if (proven.has_value()) effective_pred = sbe::bl(*proven);
  }

  FCtx(fname).GetAssessor(*this).Assess(AssessPolicy::Error, effective_pred,
                                        message, uty, aty, l, n.get(),
                                        emit_node, active_guard);
}
