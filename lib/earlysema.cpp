#include "earlysema.hpp"
#include "target_utils.hpp"
#include "types.hpp"

// Note: Early semantics lacks shape details. Therefore, any semantic analysis,
// type inference, etc. related to shapes, are pulled off.

using namespace Choreo;

bool EarlySemantics::BeforeVisitImpl(AST::Node& n) {
  if (isa<AST::Program>(&n)) {
    type_equals.Reset();
  } else if (isa<AST::ChoreoFunction>(&n)) {
    VST_DEBUG(dbgs() << "Before " << GetName() << " - " << STR(FBInfo())
                     << "\n");
    requires_return = false;
    return_deduction = false;
    found_return = false;
    pl_depth = 0;
    explicit_pl = false;
    inthreads_levels.clear();
    inthreads_levels.push_back(0);
  } else if (auto pb = dyn_cast<AST::ParallelBy>(&n)) {
    if (pl_depth == 0) pb->SetOuter(true);
    pl_depth++;
    pl_depths.push_back(pl_depth);
    inthreads_levels.push_back(0);
    assert(inthreads_levels.size() == (unsigned)pl_depth + 1);
  } else if (auto it = dyn_cast<AST::InThreadsBlock>(&n)) {
    ++inthreads_levels[pl_depth];
    if (inthreads_levels[pl_depth] > 1)
      it->outer = false; // it is a inner inthreads
  } else if (isa<AST::Parameter>(&n)) {
    in_decl = true;
    allow_named_dim = true; // tolerate repeated symbols inside mdspan params
  } else if (auto a = dyn_cast<AST::Assignment>(&n)) {
    if (!a->AssignToDataElement()) assign_id = a->GetName();
  } else if (isa<AST::ForeachBlock>(&n) || isa<AST::WhileBlock>(&n)) {
    inside_loop = true;
  } else if (auto call = dyn_cast<AST::Call>(&n)) {
    if (call->template_args) {
      in_template_param = true; // enter template param visit
    }
  }

  return true;
}

bool EarlySemantics::AfterVisitImpl(AST::Node& n) {
  if (auto f = dyn_cast<AST::ChoreoFunction>(&n)) {
    VST_DEBUG(dbgs() << "After " << GetName() << " - " << STR(FBInfo())
                     << "\n");
    if (return_deduction) {
      // maybe this can be moved to type inference
      if (!found_return && f->f_decl.ret_type->IsUnknown()) {
        f->f_decl.ret_type->base_type = BaseType::VOID;
        SetNodeType(*f->f_decl.ret_type, MakeVoidType());
      }
      // anything is ok
    } else if (requires_return && !found_return) {
      Error1(n.LOC(), "non-void function '" + f->name +
                          "` does not contain a return statement.");
    } else if (!requires_return && found_return) {
      Error1(n.LOC(),
             "return statement found in void function '" + f->name + "`.");
    }
  } else if (isa<AST::ParallelBy>(&n)) {
    assert(pl_depth > 0);
    assert(inthreads_levels.size() == (unsigned)pl_depth + 1);
    inthreads_levels.pop_back();
    if (n.GetLevel() != ParallelLevel::NONE) explicit_pl_stk.pop();
    pl_depth--;
    if (pl_depth == 0) explicit_pl = false;
  } else if (isa<AST::InThreadsBlock>(&n)) {
    --inthreads_levels[pl_depth];
  } else if (isa<AST::WithBlock>(&n)) {
    with_syms.clear();
  } else if (isa<AST::Parameter>(&n)) {
    in_decl = false;
    allow_named_dim = false;
  } else if (isa<AST::Assignment>(&n)) {
    assign_id = "";
  } else if (isa<AST::ForeachBlock>(&n) || isa<AST::WhileBlock>(&n)) {
    inside_loop = false;
  }

  return true;
}

bool EarlySemantics::Visit(AST::MultiNodes& n) {
  TraceEachVisit(n);
  return true;
}

bool EarlySemantics::Visit(AST::MultiValues& n) {
  TraceEachVisit(n);

  if (n.None()) return true;

  if (in_template_param) in_template_param = false; // exit template param visit

  if (auto ty = NodeType(*n.ValueAt(0)); ty && isa<BoundedType>(ty)) {
    size_t dims = 0;
    for (auto v : n.AllValues()) {
      auto vty = NodeType(*v);
      if (!isa<BoundedType>(ty))
        Error1(n.LOC(), PSTR(v) + "is not bounded value.");
      dims += vty->Dims();
    }
    SetNodeType(n, MakeBoundedITupleType(Shape(dims)));
  }

  if (n.HasNote("array_dims")) {
    for (const auto& d : n.AllValues()) {
      auto dty = d->GetType();
      if (!isa<ScalarIntegerType>(dty))
        Error1(d->LOC(), "Dimension of array can only be const integer "
                         "value, but got value of type " +
                             dty->TypeNameString() + ": " + PSTR(dty) + ".");
    }
    SetNodeType(n, MakeRankedArrayType(n.Count()));
  }

  return true;
}

bool EarlySemantics::Visit(AST::IntLiteral& n) {
  TraceEachVisit(n);
  return true;
}

bool EarlySemantics::Visit(AST::FloatLiteral& n) {
  TraceEachVisit(n);
  if (n.IsFloat32()) {
    assert(std::holds_alternative<float>(n.value) &&
           "unexpected floating-point type.");
    assert(!IsUnKnownFloatPoint(n.Val_f32()) &&
           "floating-point number can only used as literal for now.");
    SetNodeType(n, MakeF32Type());
  } else if (n.IsFloat64()) {
    assert(std::holds_alternative<double>(n.value) &&
           "unexpected floating-point type.");
    assert(!IsUnKnownFloatPoint(n.Val_f64()) &&
           "floating-point number can only used as literal for now.");
    SetNodeType(n, MakeF64Type());
  } else
    choreo_unreachable("unexpected floating-point type.");

  return true;
}

bool EarlySemantics::Visit(AST::StringLiteral& n) {
  TraceEachVisit(n);
  SetNodeType(n, MakeStringType());
  return true;
}

bool EarlySemantics::Visit(AST::BoolLiteral& n) {
  TraceEachVisit(n);
  SetNodeType(n, MakeBooleanType());
  return true;
}

bool EarlySemantics::Visit(AST::Expr& n) {
  TraceEachVisit(n);

  if (in_template_param) {
    SetNodeType(n, MakeIntegerType());
    return true;
  }

  if (auto ref = n.GetReference()) {
    auto rty = NodeType(*ref);
    assert(!isa<UnknownType>(rty) && "reference type is unknown.");
    SetNodeType(n, rty);
    if (diverges.Contains(dyn_cast<AST::Identifier>(ref))) diverges.Add(n);
  } else if (n.op == "dataof" || n.op == "mdataof") {
    auto ty = NodeType(*n.GetR());
    if (!isa<FutureType>(ty)) {
      Error1(n.LOC(), "in operation \"" + n.op +
                          "\": expect a future type but got `" + PSTR(ty) +
                          "'.");
      SetNodeType(n, MakeUnknownType());
      return false;
    }
    if (auto sym = cast<AST::Expr>(n.GetR())->GetSymbol())
      SetNodeType(n, GetSymbolType(sym->name +
                                   (n.op == "mdataof" ? ".mdata" : ".data")));
    else
      SetNodeType(n, MakeDummySpannedType());
  } else if (n.op == "addrof") {
    auto ty = NodeType(*n.GetR());
    if (!isa<SpannedType>(ty) && !isa<AST::DataAccess>(n.GetR())) {
      Error1(n.LOC(), "in operation \"" + n.op +
                          "\": expect a data type but got `" + PSTR(ty) + "'.");
      SetNodeType(n, MakeUnknownType());
      return false;
    }
    SetNodeType(n, MakeAddrType());
  } else if (n.op == "sizeof") {
    auto ty = NodeType(*n.GetR());
    if (!GetMDSpanType(ty)) {
      Error1(n.LOC(), "in operation \"" + n.op +
                          "\": expect a mdspan type but got `" + PSTR(ty) +
                          "'.");
      SetNodeType(n, MakeUnknownType());
      return false;
    }
    SetNodeType(n, MakeIntegerType());
  } else if (n.op == "dimof") {
    auto lty = NodeType(*n.GetL());
    auto rty = NodeType(*n.GetR());
    if (!isa<MDSpanType>(lty) && !isa<ITupleType>(lty) &&
        !isa<BoundedType>(lty)) {
      Error1(n.LOC(), "in operation \"" + n.op +
                          "\": expect a indexable type but got `" + PSTR(lty) +
                          "'.");
      SetNodeType(n, MakeUnknownType());
      return false;
    }
    if (!isa<IndexType>(rty)) {
      Error1(n.LOC(), "in operation \"" + n.op +
                          "\": expect a index type but got `" + PSTR(rty) +
                          "'.");
      SetNodeType(n, MakeUnknownType());
      return false;
    }
    if (isa<BoundedType>(lty)) {
      // disambiguate subscription into bounded ituple and getith of bounded
      // integer
      n.op = "getith";
      cast<AST::IntIndex>(n.GetR())->UseBracket();
      SetNodeType(n, lty);
    } else
      SetNodeType(n, MakeIntegerType());
  } else if (n.op == "ubound") {
    auto ty = NodeType(*n.GetR());
    if (!isa<BoundedType>(ty)) {
      Error1(n.LOC(), "in operation \"" + n.op +
                          "\": expect a bounded type but got `" + PSTR(ty) +
                          "'.");
      SetNodeType(n, MakeUnknownType());
      return false;
    }
    if (isa<BoundedIntegerType>(ty))
      SetNodeType(n, MakeIntegerType());
    else if (isa<BoundedITupleType>(ty))
      SetNodeType(n, MakeITupleType(ty->Dims()));
    else
      choreo_unreachable("unexpect");
  } else if ((n.op == "++") || (n.op == "--")) {
    auto ty = NodeType(*n.GetR());
    auto sty = dyn_cast<ScalarIntegerType>(ty);
    assert(!isa<BooleanType>(ty) &&
           "increment/decrement operation on boolean is not allowed.");
    if (!sty || !sty->IsMutable()) {
      Error1(n.LOC(), "in operation \"" + n.op +
                          "\": expect a mutable scalar type but got `" +
                          PSTR(ty) + "'.");
      SetNodeType(n, MakeUnknownType());
      return false;
    }
    SetNodeType(n, ty->Clone());
  } else if ((n.op == "+") || (n.op == "-") || (n.op == "*") || (n.op == "/") ||
             (n.op == "%") || (n.op == "cdiv")) {
    auto lty = NodeType(*n.GetL());
    auto rty = NodeType(*n.GetR());
    if (!lty || !rty)
      choreo_unreachable("expect the both types to be not nullptr.");
    bool is_mutable = IsMutable(*lty) || IsMutable(*rty);
    if (isa<NoValueType>(lty)) {
      Error1(n.GetL()->LOC(),
             "Can not evaluate the expression without a value.");
      return false;
    } else if (isa<NoValueType>(rty)) {
      Error1(n.GetR()->LOC(),
             "Can not evaluate the expression without a value.");
      return false;
    } else if ((isa<MDSpanType>(lty) && isa<ITupleType>(rty)) ||
               (isa<MDSpanType>(rty) && isa<ITupleType>(lty))) {
      // mdspan + ituple
      if (!CompatibleRank(lty->Dims(), rty->Dims())) {
        Error1(n.LOC(), "in operation \"" + n.op +
                            "\": dimension inconsistent (" +
                            std::to_string(lty->Dims()) + " vs. " +
                            std::to_string(rty->Dims()) + ").");
        return false;
      }
      MutateNodeType(n, MakeRankedMDSpanType(lty->Dims()), is_mutable);
    } else if ((isa<ITupleType>(lty) && isa<ScalarIntegerType>(rty)) ||
               (isa<MDSpanType>(lty) && isa<ScalarIntegerType>(rty))) {
      SetNodeType(n, lty->Clone());
    } else if ((isa<ITupleType>(rty) && isa<ScalarIntegerType>((lty))) ||
               (isa<MDSpanType>(rty) && isa<ScalarIntegerType>(lty))) {
      SetNodeType(n, rty->Clone());
    } else if ((isa<ITupleType>(lty) && isa<ITupleType>(rty))) {
      // ituple + ituple
      if (lty->HasSufficientInfo() && rty->HasSufficientInfo()) {
        if (!CompatibleRank(lty->Dims(), rty->Dims())) {
          Error1(n.LOC(), "in operation \"" + n.op +
                              "\": dimension inconsistent (" +
                              std::to_string(lty->Dims()) + " vs. " +
                              std::to_string(rty->Dims()) + ").");
          return false;
        }
        SetNodeType(n, lty->Clone());
      } else
        SetNodeType(n, MakeUninitBoundedITupleType());
    } else if ((isa<BoundedType>(lty) && isa<BoundedType>(rty))) {
      if (IsActualBoundedIntegerType(lty) && IsActualBoundedIntegerType(rty)) {
        // decay the type to be mutable int
        SetNodeType(n, MakeIntegerType(true));
      } else {
        Error1(n.LOC(), "in operation \"" + n.op +
                            "\": unable to apply to the types (" + PSTR(lty) +
                            " vs. " + PSTR(rty) + ").");
        SetNodeType(n, MakeUnknownType());
        return false;
      }
    } else if ((IsActualBoundedIntegerType(lty) &&
                isa<ScalarIntegerType>(rty)) ||
               (IsActualBoundedIntegerType(rty) &&
                isa<ScalarIntegerType>(lty))) {
      // this is promissing, simply allow it
      if (IsActualBoundedIntegerType(lty))
        if (cast<AST::Expr>(n.GetL())->op == "getith") {
          Error1(n.LOC(),
                 "in operation \"" + n.op +
                     "\": unable to apply to the getith bounded variable (" +
                     PSTR(n.GetL()) + ").");
          SetNodeType(n, MakeUnknownType());
          return false;
        } else if (cast<ScalarIntegerType>(rty)->IsMutable()) {
          // decay bounded to be non-bounded when mutable
          SetNodeType(n, MakeScalarIntegerType(rty->GetBaseType(), true));
        } else
          SetNodeType(n, lty->Clone());
      else {
        if (cast<AST::Expr>(n.GetR())->op == "getith") {
          Error1(n.LOC(),
                 "in operation \"" + n.op +
                     "\": unable to apply to the 'getith' bounded variable (" +
                     PSTR(n.GetR()) + ").");
          SetNodeType(n, MakeUnknownType());
          return false;
        } else if (cast<ScalarIntegerType>(lty)->IsMutable()) {
          // decay bounded to be non-bounded when mutable
          SetNodeType(n, MakeScalarIntegerType(lty->GetBaseType(), true));
        } else
          SetNodeType(n, rty->Clone());
      }
    } else if ((isa<BoundedITupleType>(lty) && isa<ITupleType>(rty)) ||
               (isa<BoundedITupleType>(rty) && isa<ITupleType>(lty))) {
      if (!CompatibleRank(lty->Dims(), rty->Dims())) {
        Error1(n.LOC(), "in operation \"" + n.op +
                            "\": dimension inconsistent (" +
                            std::to_string(lty->Dims()) + " vs. " +
                            std::to_string(rty->Dims()) + ").");
        return false;
      }
      SetNodeType(n, MakeITupleType(lty->Dims()));
    } else if (isa<MDSpanType>(lty) && isa<MDSpanType>(rty)) {
      // only allow div/mod operations
      if ((n.op != "/") && (n.op != "%") && (n.op != "cdiv")) {
        Error1(n.LOC(), "in operation \"" + n.op +
                            "\": unable to apply to the types (" + PSTR(lty) +
                            " vs. " + PSTR(rty) + ").");
        SetNodeType(n, MakeUnknownType());
        return false;
      }
      if (!CompatibleRank(lty->Dims(), rty->Dims())) {
        Error1(n.LOC(), "in operation \"" + n.op +
                            "\": dimension inconsistent (" +
                            std::to_string(lty->Dims()) + " vs. " +
                            std::to_string(rty->Dims()) + ").");
        SetNodeType(n, MakeUnknownType());
        return false;
      }
      SetNodeType(n, MakeITupleType(lty->Dims()));
    } else if ((isa<MDSpanType>(lty) && isa<ScalarIntegerType>(rty)) ||
               (isa<MDSpanType>(rty) && isa<ScalarIntegerType>(lty)) ||
               (isa<ITupleType>(lty) && isa<ScalarIntegerType>(rty)) ||
               (isa<ITupleType>(rty) && isa<ScalarIntegerType>(lty))) {
      if (isa<MDSpanType>(lty) || isa<ITupleType>(lty))
        SetNodeType(n, lty->Clone());
      else
        SetNodeType(n, rty->Clone());
    } else if ((isa<ITupleType>(lty) && isa<ITupleType>(rty)) ||
               (isa<BooleanType>(lty) && isa<BooleanType>(rty)) ||
               (isa<IndexType>(lty) && isa<IndexType>(rty))) {
      Error1(n.LOC(), "in operation \"" + n.op +
                          "\": unable to apply to the types (" + PSTR(lty) +
                          " vs. " + PSTR(rty) + ").");
      SetNodeType(n, MakeUnknownType());
      return false;
    } else if ((isa<ScalarIntegerType>(lty) && isa<IndexType>(rty)) ||
               (isa<ScalarIntegerType>(rty) && isa<IndexType>(lty))) {
      // when desugaring of ituple/mdspan has not been applied, we have to deal
      // with nodes like:
      //   a {1 + (1)}
      SetNodeType(n, lty->Clone());
    } else if (isa<SpannedType>(lty) || isa<SpannedType>(rty)) {
      // For spanned arithmetics, shape consistent check is deferred
      auto lety = BaseType::UNKSCALAR;
      auto rety = BaseType::UNKSCALAR;
      bool no_support = false;
      if (auto lsty = dyn_cast<SpannedType>(lty))
        lety = lsty->ElementType();
      else if (isa<ScalarType>(lty))
        lety = lty->GetBaseType();
      else
        no_support = true;

      if (auto rsty = dyn_cast<SpannedType>(rty))
        rety = rsty->ElementType();
      else if (isa<ScalarType>(rty))
        rety = rty->GetBaseType();
      else
        no_support = true;

      if (lety != BaseType::UNKSCALAR && rety != BaseType::UNKSCALAR &&
          lety != rety)
        no_support = true;

      if (no_support)
        Error1(n.LOC(), "in operation \"" + n.op +
                            "\": unable to apply to the types (" + PSTR(lty) +
                            " vs. " + PSTR(rty) + ").");

      if (auto lsty = dyn_cast<SpannedType>(lty))
        SetNodeType(n, lsty->Clone());
      else if (auto rsty = dyn_cast<SpannedType>(rty))
        SetNodeType(n, rsty->Clone());

    } else if (!lty->ApprxEqual(*rty)) {
      Error1(n.LOC(), "in operation \"" + n.op +
                          "\": unable to apply to the types (" + PSTR(lty) +
                          " vs. " + PSTR(rty) + ").");
      SetNodeType(n, MakeUnknownType());
      return false;
    } else {
      SetNodeType(n, lty->Clone());
    }
    if (diverges.Contains(n.GetL()) || diverges.Contains(n.GetR()))
      diverges.Add(n);
  } else if (n.op == "&" || n.op == "|" || n.op == "^") {
    auto lty = NodeType(*n.GetL());
    auto rty = NodeType(*n.GetR());
    if (!(CanYieldAnInteger(lty) && CanYieldAnInteger(rty))) {
      Error1(n.LOC(), "in operation \"" + n.op +
                          "\": unable to apply to the types (" + PSTR(lty) +
                          " vs. " + PSTR(rty) + ").");
      SetNodeType(n, MakeUnknownType());
      return false;
    }
    bool is_mutable = IsMutable(*lty) || IsMutable(*rty);
    SetNodeType(n, MakeIntegerType(is_mutable));
    if (diverges.Contains(n.GetL()) || diverges.Contains(n.GetR()))
      diverges.Add(n);
  } else if (n.op == "~") {
    assert(n.IsUnary() && "bitwise negation operator must be unary.");
    auto rty = NodeType(*n.GetR());
    if (!CanYieldAnInteger(rty)) {
      Error1(n.LOC(), "in operation \"" + n.op +
                          "\": unable to apply to the type (" + PSTR(rty) +
                          ").");
      SetNodeType(n, MakeUnknownType());
      return false;
    }
    SetNodeType(n, MakeIntegerType(IsMutable(*rty)));
    if (diverges.Contains(n.GetR())) diverges.Add(n);
  } else if (n.op == "<<" || n.op == ">>") {
    auto lty = NodeType(*n.GetL());
    auto intr = AST::GetIntLiteral(*n.GetR());
    if (!intr) {
      Error1(n.LOC(), "in operation \"" + n.op +
                          "\": expect an integer literal as the right operand "
                          "but got `" +
                          PSTR(n.GetR()) + "'.");
      SetNodeType(n, MakeUnknownType());
      return false;
    }
    if (!CanYieldAnInteger(lty)) {
      Error1(n.LOC(), "in operation \"" + n.op +
                          "\": unable to apply to the type (" + PSTR(lty) +
                          ").");
      SetNodeType(n, MakeUnknownType());
      return false;
    }
    SetNodeType(n, MakeIntegerType(IsMutable(*lty)));
    if (diverges.Contains(n.GetL()) || diverges.Contains(n.GetR()))
      diverges.Add(n);
  } else if (n.op == "#") {
    // allow only # operator for cartesian products on two bounded-vars
    // a # b => a * (#b) + b
    auto lty = NodeType(*n.GetL());
    auto rty = NodeType(*n.GetR());
    if (IsActualBoundedIntegerType(lty) && IsActualBoundedIntegerType(rty)) {
      SetNodeType(n, MakeBoundedITupleType(Shape(1)));
    } else {
      // TODO: computation of multi-dim bounded vars is not supported yet.
      Error1(n.LOC(), "in operation \"" + n.op +
                          "\": unable to apply to the types (" + PSTR(lty) +
                          " vs. " + PSTR(rty) + ").");
      SetNodeType(n, MakeUnknownType());
      return false;
    }
    if (diverges.Contains(n.GetL()) || diverges.Contains(n.GetR()))
      diverges.Add(n);
  } else if ((n.op == "#-") || (n.op == "#+")) {
    auto lty = NodeType(*n.GetL());
    auto rty = NodeType(*n.GetR());
    if ((IsActualBoundedIntegerType(lty) && isa<ScalarIntegerType>(rty))) {
      SetNodeType(n, MakeBoundedITupleType(Shape(1)));
    } else {
      // TODO: computation of multi-dim bounded vars is not supported yet.
      Error1(n.LOC(), "in operation \"" + n.op +
                          "\": unable to apply to the types (" + PSTR(lty) +
                          " vs. " + PSTR(rty) + ").");
      SetNodeType(n, MakeUnknownType());
      return false;
    }
  } else if ((n.op == "#*") || (n.op == "#/") || (n.op == "#%")) {
    auto lty = NodeType(*n.GetL());
    auto rty = NodeType(*n.GetR());
    Error1(n.LOC(), "in operation \"" + n.op +
                        "\": unable to apply to the types (" + PSTR(lty) +
                        " vs. " + PSTR(rty) + ").");
    return false;
  } else if ((n.op == "<") || (n.op == ">") || (n.op == "==") ||
             (n.op == "!=") || (n.op == "<=") || (n.op == ">=")) {
    auto lty = NodeType(*n.GetL());
    auto rty = NodeType(*n.GetR());
    // only support ScalarIntegerType currently
    if (!(CanYieldAnInteger(lty) && CanYieldAnInteger(lty))) {
      Error1(n.LOC(), "in operation \"" + n.op +
                          "\": unable to apply to the types (" + PSTR(lty) +
                          " vs. " + PSTR(rty) +
                          "). (Only the values that can produce integers are "
                          "supported by now.)");
      SetNodeType(n, MakeUnknownType());
      return false;
    }
    if ((IsActualBoundedIntegerType(lty) && ConvertibleToInt(rty)) ||
        (IsActualBoundedIntegerType(rty) && ConvertibleToInt(lty)) ||
        (ConvertibleToInt(lty) && ConvertibleToInt(rty))) {
      // this is acceptable
    } else if (!(lty->ApprxEqual(*rty))) {
      Error1(n.LOC(), "in operation \"" + n.op +
                          "\": unable to apply to the types (" + PSTR(lty) +
                          " vs. " + PSTR(rty) + ").");
      SetNodeType(n, MakeUnknownType());
      return false;
    }

    auto ld = diverges.Contains(n.GetL());
    auto rd = diverges.Contains(n.GetR());

    // comparing two divergent values, like p < q - 1 makes no sense in
    // inthreads pred.
    if ((ld && !rd) || (!ld && rd)) diverges.Add(n);

    SetNodeType(n, MakeBooleanType());
  } else if ((n.op == "&&") || (n.op == "||")) {
    auto lty = NodeType(*n.GetL());
    auto rty = NodeType(*n.GetR());
    if (isa<BooleanType>(lty) && isa<BooleanType>(rty))
      SetNodeType(n, MakeBooleanType());
    else if (isa<EventType>(lty) && isa<EventType>(rty) && (*lty == *rty))
      SetNodeType(n, lty);
    else {
      Error1(n.LOC(), "in operation \"" + n.op +
                          "\": unable to apply to the types (" + PSTR(lty) +
                          " vs. " + PSTR(rty) + ").");
      SetNodeType(n, MakeUnknownType());
      return false;
    }

    // inthreads requires both operations are divergent
    if (diverges.Contains(n.GetL()) && diverges.Contains(n.GetR()))
      diverges.Add(n);
  } else if (n.op == "!") {
    auto rty = NodeType(*n.GetR());
    if (isa<BooleanType>(rty) || isa<EventType>(rty)) {
      SetNodeType(n, rty);
    } else {
      Error1(n.LOC(), "in operation \"" + n.op +
                          "\": unable to apply to the type (" + PSTR(rty) +
                          ").");
      SetNodeType(n, MakeUnknownType());
      return false;
    }
    if (diverges.Contains(n.GetR())) diverges.Add(n);
  } else if (n.op == "?") {
    auto cty = NodeType(*n.GetC());
    auto lty = NodeType(*n.GetL());
    auto rty = NodeType(*n.GetR());
    if (!isa<BooleanType>(cty)) {
      Error1(n.LOC(),
             "in operation \"" + n.op +
                 "\": expect an expr of bool type as condition but got " +
                 PSTR(cty) + " type.");
      SetNodeType(n, MakeUnknownType());
    }
    // allow BoundedInteger to participate in scalar computation.
    if (CanYieldAnInteger(lty) && CanYieldAnInteger(rty)) {
      SetNodeType(n, lty->Clone());
      if (isa<ScalarType>(rty)) SetNodeType(n, rty->Clone());
    } else {
      if (!lty->ApprxEqual(*rty)) {
        Error1(n.LOC(), "in operation \"" + n.op +
                            "\": unable to apply to the types (" + PSTR(cty) +
                            ") " + PSTR(lty) + " : " + PSTR(rty) + ").");
        SetNodeType(n, MakeUnknownType());
        return false;
      }
      SetNodeType(n, lty->Clone());
    }
  } else if (n.op == "concat") {
    auto lty = NodeType(*n.GetL());
    auto rty = NodeType(*n.GetR());
    if (!((isa<MDSpanType>(lty) || isa<ITupleType>(lty)) &&
          (isa<MDSpanType>(rty) || isa<ITupleType>(rty)))) {
      Error1(n.LOC(), "in operation \"" + n.op +
                          "\": unable to apply to the types " + PSTR(lty) +
                          " and " + PSTR(rty) + ".");
      SetNodeType(n, MakeUnknownType());
      return false;
    }
    if ((!IsValidRank(lty->Dims())) || (!IsValidRank(rty->Dims())))
      SetNodeType(n, MakeUninitMDSpanType());
    else
      SetNodeType(n, MakeRankedMDSpanType(lty->Dims() + rty->Dims()));
  } else if (n.op == "elemof") {
    auto lty = NodeType(*n.GetL());
    auto rty = NodeType(*n.GetR());
    auto old_ec = error_count;
    if (!isa<ArrayType>(lty))
      Error1(n.LOC(), "in operation \"" + n.op +
                          "\": expect an array but got " + PSTR(lty) + ".");
    if (!CanYieldAnInteger(rty))
      Error1(n.LOC(), "in operation \"" + n.op +
                          "\": expect an integer index expression but got " +
                          PSTR(rty) + ".");

    auto aty = cast<ArrayType>(lty);
    SetNodeType(n, aty->SubScriptType(1));

    if (error_count != old_ec) return false;
  } else
    choreo_unreachable("operation '" + n.op + "' in expression " + STR(n) +
                       "is not supported yet.");

  if (mutables.Contains(n.GetL()) || mutables.Contains(n.GetR()) ||
      mutables.Contains(n.GetC()))
    mutables.Add(n);

  return true;
}

bool EarlySemantics::Visit(AST::CastExpr& n) {
  TraceEachVisit(n);
  choreo_unreachable("AST::CastExpr should not appear at EarlySemantics.");
  return true;
}

bool EarlySemantics::Visit(AST::AttributeExpr& n) {
  TraceEachVisit(n);
  if (n.AttrName() == "vectorize") {
    if (n.AttrValueCount() != 2)
      Error1(n.LOC(),
             "suffix expresion 'vectorize' should have 2 attribute values");
    auto loop_iv = AST::GetIdentifier(n.AttrValueAt(0));
    auto vec_len = AST::GetIntLiteral(n.AttrValueAt(1));
    if (!loop_iv)
      Error1(loop_iv->LOC(),
             "the first attribute value of `vectorize` should be an "
             "identifier for the loop induction variable");

    if (!SSTab().IsDeclared(loop_iv->name))
      Error1(loop_iv->LOC(), "the loop induction variable '" + loop_iv->name +
                                 "' is not declared in the current scope");

    if (!within_map.count(InScopeName(loop_iv->name)) ||
        within_map[InScopeName(loop_iv->name)].size() != 1)
      Error1(loop_iv->LOC(),
             "vectorization should be applied on an loop induction variable");

    auto iv_ty = GetSymbolType(loop_iv->name);
    if (iv_ty) {
      if (auto bit = dyn_cast<BoundedITupleType>(iv_ty)) {
        if (bit->Dims() != 1)
          Error1(loop_iv->LOC(), "the loop induction variable '" +
                                     loop_iv->name +
                                     "' is a bounded ituple and its dim is not "
                                     "1, vectorization is not supported yet");
      } else if (!isa<BoundedIntegerType>(iv_ty)) {
        Error1(loop_iv->LOC(), "the loop induction variable '" + loop_iv->name +
                                   "' should be a bounded integer type");
      }
    }
    if (!vec_len)
      Error1(n.AttrValueAt(1)->LOC(),
             "the second attribute value of `vectorize` should be an integer "
             "literal for the vector length");
    if (vec_len && vec_len->Val() <= 0 && !IsUnKnownInteger(vec_len->Val()))
      Error1(vec_len->LOC(),
             "the vector length of `vectorize` should be a positive integer");
  }
  return true;
}

bool EarlySemantics::Visit(AST::MultiDimSpans& n) {
  TraceEachVisit(n);
  size_t rank = GetInvalidRank();

  // transform [a.span, b.span] to be an expr of concat(a.span, b.span)
  if (auto mvals = dyn_cast<AST::MultiValues>(n.list)) {
    // concat can not work with syntactic sugar
    if (n.ref_name.empty()) {
      std::vector<ptr<AST::Node>> wl;
      wl.push_back(nullptr);
      for (auto& v : mvals->AllValues()) {
        if (isa<MDSpanType>(v->GetType()) || isa<ITupleType>(v->GetType())) {
          if (wl.back() == nullptr)
            wl.back() = v;
          else
            wl.push_back(v);
          wl.push_back(nullptr); // accepting new values
          continue;
        }

        // normal values are added to the mdspan
        if (wl.back() == nullptr)
          wl.back() = AST::Make<AST::MultiDimSpans>(
              v->LOC(), "" /*anon*/, AST::Make<AST::MultiValues>(v->LOC()));
        cast<AST::MultiValues>(cast<AST::MultiDimSpans>(wl.back())->list)
            ->Append(v);
      }

      if (wl.back() == nullptr) wl.pop_back();

      if (wl.size() > 1) { // mdspan inside
        ptr<AST::Node> last = wl[0];
        for (size_t i = 1; i < wl.size(); ++i) {
          auto concat =
              AST::Make<AST::Expr>(last->LOC(), "concat", last, wl[i]);
          last = concat;
        }
        if (debug_visit)
          dbgs() << "Transform: " << PSTR(n.list) << " to be " << PSTR(last)
                 << "\n";
        n.list = last;
        n.list->accept(*this); // go evaluate the concatenation
      }
    }
  }

  // check the elements type
  if (auto mvals = dyn_cast<AST::MultiValues>(n.list)) {
    if (n.ref_name.empty()) {
      for (auto& v : mvals->AllValues()) {
        auto ty = NodeType(*v);
        if (!isa<ScalarIntegerType>(ty) && !isa<MDSpanType>(ty) &&
            !isa<ITupleType>(ty) && !isa<NoValueType>(ty) &&
            !isa<BoundedType>(ty)) {
          Error1(v->LOC(),
                 "unexpected data type '" + PSTR(ty) + "' is found in mdspan.");
        }
      }
    }
  }

  // try to figure out the dimensions
  if (auto mvals = dyn_cast<AST::MultiValues>(n.list)) {
    size_t elem_count = 0;
    for (auto& v : mvals->AllValues()) {
      auto ty = NodeType(*v);
      elem_count += ty->Dims();
    }
    rank = elem_count;
  } else if (isa<AST::Expr>(n.list))
    rank = n.list->GetType()->Dims();

  if (!IsValidRank(n.Rank()))
    n.SetRank(rank);
  else if (!IsValidRank(rank))
    rank = n.Rank();
  else {
    if (n.Rank() != rank) {
      Warning(n.LOC(), "mdspan is initialized with a rank of " +
                           std::to_string(rank) + " but declared as rank of " +
                           std::to_string(n.Rank()) + ".");
      Warning(n.LOC(),
              "assume the mdspan as a rank of " + std::to_string(rank) + ".");
      if (debug_visit)
        dbgs() << "Warning in " << __FILE__ << ", line: " << __LINE__ << ".\n";
      n.SetRank(rank);
    }
  }

  // check if the int literals are valid
  if (auto mvals = dyn_cast<AST::MultiValues>(n.list))
    for (auto& v : mvals->AllValues())
      if (auto il = AST::GetIntLiteral(*v))
        if (il->Val() <= 0 && !IsUnKnownInteger(il->Val())) {
          Error1(v->LOC(), "The mdspan size \"" + std::to_string(il->Val()) +
                               "\" is invalid!");
        }

  // check if it use mutable values that can not be inferred
  if (auto mvals = dyn_cast<AST::MultiValues>(n.list)) {
    for (auto& v : mvals->AllValues()) {
      bool is_mutable = false;
      if (mutables.Contains(v)) is_mutable = true;
      if (is_mutable) mutables.Add(*v);
    }
  }

  SetNodeType(n, MakeRankedMDSpanType(rank));

  return true;
}

bool EarlySemantics::Visit(AST::NamedTypeDecl& n) {
  TraceEachVisit(n);
  assert(n.init_expr && "missing init expr.");
  auto ety = NodeType(*n.init_expr);
  auto nty = (IsValidRank(n.rank)) ? MakeRankedMDSpanType(n.rank)
                                   : MakeUninitMDSpanType();
  // check for the type consistency
  if (!ety->ApprxEqual(*nty)) {
    Error1(n.LOC(), "`" + n.name_str + "' is declared as \"" + PSTR(nty) +
                        "\" but initialized as \"" + PSTR(ety) + "\".");
    // keep processing
  }

  if (mutables.Contains(n.init_expr)) {
    Error1(n.init_expr->LOC(),
           "mdspan/ituple can not be initialized with mutable values.");
  }

  ReportErrorWhenViolateODR(n.LOC(), n.name_str, __FILE__, __LINE__, ety);
  SetNodeType(n, ety);
  return true;
}

bool EarlySemantics::CheckInitializerType(const ptr<Type>& ty,
                                          const std::string& sym,
                                          const location& loc) {
  if (isa<UnknownType>(ty)) {
    Error1(loc, "unable to inference the type from `" + sym +
                    "'s initialization expression.");
    return false;
  } else if (isa<NoValueType>(ty)) {
    Error1(loc, "unable to evaluate the initialization expression of `" + sym +
                    "'.");
    return false;
  } else if (isa<StringType>(ty)) {
    Error1(loc, "string variables are not supported yet.");
    return false;
  } else if (isa<AddrType>(ty)) {
    Error1(loc, "pointer variables are not supported.");
    return false;
  } else if (isa<VoidType>(ty)) {
    Error1(loc, "can not initialize `" + sym + "' with a void type.");
    return false;
  } else if (isa<DeviceDataType>(ty)) {
    Error1(loc, "can not initialize `" + sym + "' with a device data type.");
    return false;
  }
  return true;
}

bool EarlySemantics::Visit(AST::NamedVariableDecl& n) {
  TraceEachVisit(n);

  ptr<Type> tty = nullptr; // type from the annotation
  ptr<Type> ety = nullptr; // type from the initialization expression

  bool force_mutable = false;

  if (n.type) tty = n.type->GetType();
  if (n.init_expr) {
    ety = n.init_expr->GetType();
    if (!isa<AST::Call>(n.init_expr))
      assert(!isa<UnknownType>(ety) && "no type for an init expression.");
    else {
      force_mutable = true;
      if (resolve_fns) {
        ety = n.init_expr->GetType();
        if (isa<UnknownType>(ety)) {
          Warning(n.LOC(), "can not infer the type of `" + n.name_str +
                               "' from the initialization expression '" +
                               STR(n.init_expr) + "'.");
          ety = nullptr;
        }
      } else
        ety = nullptr;
    }
  }

  if (!ety) {
    // The initializer expression does not render a type. In this case, the type
    // is deduced directly from type annotation
    assert(tty && "the annotation type must exist.");

    if (isa<UnknownType>(tty)) {
      Error1(n.LOC(), "unable to deduce the type of `" + n.name_str + "'.");
      return false;
    }

    if (force_mutable && !n.IsMutable())
      Error1(n.LOC(),
             "`" + n.name_str + "' must be annotated as a mutable type.");

    // update the scope/storage for event types
    if (auto evty = dyn_cast<EventArrayType>(tty)) {
      tty = MakeEventArrayType(n.mem->Get(), evty->Dimensions());
      SetNodeType(*n.type, tty);
    } else if (isa<EventType>(tty)) {
      tty = MakeEventType(n.mem->Get());
      SetNodeType(*n.type, tty);
    }

    // event type is purely declarative
    if (auto et = dyn_cast<EventType>(tty)) {
      if (inthreads_levels[pl_depth] > 0)
        Error1(n.LOC(),
               "the event should not be declared inside a inthreads block.");
      if (et->GetStorage() == Storage::GLOBAL)
        if (pl_depth != 0)
          Error1(n.LOC(), "global event can only be declared at host side.");
      if (et->GetStorage() == Storage::SHARED ||
          et->GetStorage() == Storage::LOCAL)
        if (pl_depth == 0)
          Error1(n.LOC(), STR(et->GetStorage()) +
                              " event can only be declared at device side.");
    }

    ReportErrorWhenViolateODR(n.LOC(), n.name_str, __FILE__, __LINE__, tty);
    SetNodeType(n, tty);
    if (force_mutable) SetNodeType(*n.init_expr, tty);
  } else {
    // The initializer expression DOES render a type
    if (!CheckInitializerType(ety, n.name_str, n.init_expr->LOC()))
      return false;

    // since the type annotation exists, it requires to check its consistance
    // with the initializer
    if (isa<MDSpanType>(ety)) {
      if (isa<ITupleType>(tty))
        Error1(n.LOC(), "must use '{' and '}' to initialize an ituple.");
      else
        Error1(n.LOC(), "use ':' instead of '=' to define the \"" + PSTR(ety) +
                            "\" type variable.");
      VST_DEBUG(dbgs() << "Error in " << __FILE__ << ", line: " << __LINE__
                       << ".\n");
      // keep working
    } else if (isa<PlaceHolderType>(ety) && n.init_expr &&
               isa<AST::Expr>(n.init_expr) &&
               cast<AST::Expr>(n.init_expr)->GetSymbol()) {
      // forbid to directly initialize a placeholder with a placeholder
      Error1(n.LOC(), "can not initialize variable `" + n.name_str +
                          "' with a placeholder.");
    }

    assert(tty && "no expression type.");

    // check for type consistency between annotation and init expr.
    if (!isa<UnknownType>(tty) && !tty->ApprxEqual(*ety)) {
      if (!n.IsMutable())
        Error1(n.LOC(), "`" + n.name_str + "' is declared as \"" +
                            PSTR(n.type->GetType()) +
                            "\" but initialized as \"" +
                            PSTR(n.init_expr->GetType()) + "\".");
      // keep working
    }

    if (n.IsMutable()) {
      // update with the mutable attributes
      if (auto sty = dyn_cast<ScalarType>(ety)) {
        ety = sty->Clone(n.IsMutable());
      } else if (IsActualBoundedIntegerType(ety)) {
        // decay a bounded integer to be integer when it is mutable
        ety = MakeIntegerType(true);
      } else {
        Error1(n.LOC(), "`" + n.name_str + "' with a type of \"" + PSTR(ety) +
                            "\" can not be declared as 'mutable'.");
      }
    }

    // use tty first if it is not UnknownType
    if (isa<UnknownType>(tty) || !isa<ScalarType>(tty)) {
      ReportErrorWhenViolateODR(n.LOC(), n.name_str, __FILE__, __LINE__, ety);
      SetNodeType(n, ety);
      // also set the type of type annotation
      SetNodeType(*n.type, ety);
    } else {
      ReportErrorWhenViolateODR(n.LOC(), n.name_str, __FILE__, __LINE__, tty);
      SetNodeType(n, tty);
      // also set the type of type annotation
      SetNodeType(*n.type, tty);
    }

    if (diverges.Contains(n.init_expr)) diverges.Add(InScopeName(n.name_str));
  }

  if (isa<SpannedType>(n.GetType())) {
    auto mds = dyn_cast<AST::MultiDimSpans>(n.type->mdspan_type);
    if (mds && mds->list) {
      if (auto e = dyn_cast<AST::Expr>(mds->list)) {
        auto bt = dyn_cast<BoundedType>(e->GetType());
        if (bt)
          Error1(e->LOC(), "Dimension of span can only be const integer "
                           "value, but got value of type " +
                               e->GetType()->TypeNameString() + ": " +
                               PSTR(e->GetType()) + ".");
        if (mutables.Contains(e))
          Error1(e->LOC(), "Dimension of span can only be const integer "
                           "value, but got value of type " +
                               e->GetType()->TypeNameString() + ": " +
                               PSTR(e->GetType()) + ".");

      } else {
        auto mv = cast<AST::MultiValues>(mds->list);
        for (const auto& v : mv->AllValues()) {
          auto bt = dyn_cast<BoundedType>(v->GetType());
          if (bt)
            Error1(v->LOC(), "Dimension of span can only be const integer "
                             "value, but got value of type " +
                                 v->GetType()->TypeNameString() + ": " +
                                 PSTR(v->GetType()) + ".");
          if (mutables.Contains(v))
            Error1(v->LOC(), "Dimension of span can only be const integer "
                             "value, but got value of type " +
                                 v->GetType()->TypeNameString() + ": " +
                                 PSTR(v->GetType()) + ".");
        }
      }
    }
  }

  // check the type of init_value of span
  if (isa<SpannedType>(n.GetType()) && n.init_value) {
    auto ty = dyn_cast<SpannedType>(n.GetType());
    auto iv_ty = n.init_value->GetType();
    if (!isa<ScalarType>(iv_ty)) {
      Error1(n.LOC(),
             "'" + n.name_str +
                 "' is declared as a span but has an initialization value "
                 "which is not of scalar type: '" +
                 PSTR(iv_ty) + "'.");
      return false;
    }
  }

  // now handle the associated symbol
  if (auto ty = dyn_cast<SpannedType>(n.GetType())) {
    ReportErrorWhenViolateODR(n.LOC(), n.name_str + ".span", __FILE__, __LINE__,
                              ty->GetMDSpanType());
  } else if (auto ty = dyn_cast<FutureType>(n.GetType())) {
    ReportErrorWhenViolateODR(n.LOC(), n.name_str + ".span", __FILE__, __LINE__,
                              MakeRankedMDSpanType(ty->Dims()));
    ReportErrorWhenViolateODR(n.LOC(), n.name_str + ".data", __FILE__, __LINE__,
                              MakeRankedSpannedType(ty->Dims()));
  }

  if (auto sty = dyn_cast<ScalarType>(n.GetType()))
    if (sty->IsMutable()) mutables.Add(InScopeName(n.name_str));

  return true;
}

bool EarlySemantics::Visit(AST::IntTuple& n) {
  TraceEachVisit(n);
  size_t dim_count = 0;
  for (auto& v : n.GetValues()->AllValues())
    if (auto itt = dyn_cast<ITupleType>(v->GetType()))
      dim_count += itt->dim_count;
    else
      ++dim_count;

  for (auto& v : n.GetValues()->AllValues()) {
    bool is_mutable = false;
    if (mutables.Contains(v)) {
      Error1(v->LOC(),
             "mutable values can not be used for the ituple declaration.");
      is_mutable = true;
    }
    if (is_mutable) mutables.Add(*v);
  }

  SetNodeType(n, MakeITupleType(dim_count));

  return true;
}

bool EarlySemantics::Visit(AST::DataAccess& n) {
  TraceEachVisit(n);

  auto dsym = RemoveSuffix(n.GetDataName(), ".data");

  if (!n.AccessElement()) {
    SetNodeType(n, GetSymbolType(dsym));
    return true;
  }

  if (!SSTab().IsDeclared(dsym)) {
    Error1(n.LOC(), "unable to access an undeclared variable '" + dsym + "'.");
    return false;
  }

  auto dty = GetSymbolType(n.GetDataName());
  auto sty = dyn_cast<SpannedType>(dty);

  if (!sty) {
    Error1(n.LOC(), "expect '" + n.GetDataName() + "' a spanned type but got " +
                        PSTR(dty) + ".");
    return false;
  }

  size_t idx_count = 0;
  for (auto idx : n.GetIndices()) {
    auto ity = NodeType(*idx);
    if (!CanYieldIndex(ity)) {
      Error1(n.LOC(), "expect '" + PSTR(idx) + "' to yield indices but got " +
                          PSTR(ity) + ".");
    }
    idx_count += NodeType(*idx)->Dims();
  }

  if (!CompatibleRank(sty->Dims(), idx_count)) {
    Error1(n.LOC(),
           "accessing an spanned data (rank: " + std::to_string(sty->Dims()) +
               ") with " + std::to_string(idx_count) + " indices.");
  }

  // data element is considered as mutable
  SetNodeType(n, MakeScalarType(sty->ElementType(), true));

  return true;
}

bool EarlySemantics::Visit(AST::Assignment& n) {
  TraceEachVisit(n);

  // assign to the array element
  if (n.AssignToDataElement()) {
    if (!SSTab().IsDeclared(n.GetDataArrayName()))
      Error1(n.da->LOC(),
             "unable to access element of an undeclared variable " +
                 n.GetName() + ".");

    auto ety = NodeType(*n.da);
    auto vty = NodeType(*n.value);

    if (!IsMutable(*ety)) {
      Error1(n.da->LOC(), "must assign to a mutable value.");
    }

    if (!ety->ApprxEqual(*vty)) {
      // consider taking value of bounded variables
      if (!(isa<ScalarIntegerType>(ety) && CanYieldAnInteger(vty))) {
        Error1(n.da->LOC(), "type inconsistent: assign " + PSTR(vty) + " to " +
                                PSTR(ety) + ".");
      }
    }

    SetNodeType(n, ety);
    n.SetDecl(false);
    return true;
  }

  // Decide if it is a declaration or assignment
  if (!SSTab().IsDeclared(n.GetName()))
    n.SetDecl();
  else {
    // We must distiguish new decl with assignment
    //
    // (0)  sym = ...;
    // (1)   {  sym = ...; }
    //
    // if 'symbol' is mutable, (1) is an assignment. Or else, it is creating a
    // new symbol with identical nameof the outer scope.
    auto ety = NodeType(*n.da);
    if (IsMutable(*ety))
      n.SetDecl(false);
    else {
      if (!SSTab().DeclaredInScope(n.GetName())) {
        n.SetDecl(true); // immutables in inner-scope: new decls
        Note(n.LOC(), "new declaration shadows outer variable '" + n.GetName() +
                          "' (not an assignment).");
      } else {
        Error1(n.LOC(),
               "only mutables can be re-assigned (" + n.GetName() + ").");
        return false;
      }
    }
  }

  if (n.IsDecl()) {
    auto sty = NodeType(*n.value);
    assert(sty && "internal error: failed to find the type.");
    if (!CheckInitializerType(sty, n.GetName(), n.value->LOC())) return false;

    if (isa<MDSpanType>(sty)) {
      Error1(loc, "use ':' to define the \"" + STR(*sty) + "\" type variable.");
      return false;
    }

    if (auto e = dyn_cast<AST::Expr>(n.value);
        e && isa<AST::ChunkAt>(e->GetReference()))
      n.AddNote("ref");

    ReportErrorWhenViolateODR(n.LOC(), n.GetName(), __FILE__, __LINE__,
                              ShadowTypeStorage(sty));

    if (auto ty = dyn_cast<SpannedType>(sty)) {
      ReportErrorWhenViolateODR(n.LOC(), n.GetName() + ".span", __FILE__,
                                __LINE__, MakeRankedMDSpanType(ty->Dims()));
    }
    if (auto ty = dyn_cast<FutureType>(sty)) {
      ReportErrorWhenViolateODR(n.LOC(), n.GetName() + ".span", __FILE__,
                                __LINE__, MakeRankedMDSpanType(ty->Dims()));
      ReportErrorWhenViolateODR(n.LOC(), n.GetName() + ".data", __FILE__,
                                __LINE__,
                                ShadowTypeStorage(ty->GetSpannedType()));
    }

    if (diverges.Contains(n.value)) diverges.Add(InScopeName(n.GetName()));

    return true;
  }

  auto vty = GetSymbolType(n.GetName()); // variable type
  auto ety = NodeType(*n.value);         // assignment expression type

  // placeholder can be reassigned
  if (isa<PlaceHolderType>(vty)) {
    assert(GeneralFutureType(vty));

    // check for type consistent
    if (!vty->ApprxEqual(*ety)) {
      Error1(n.LOC(), "`" + n.GetName() + "' of type '" + STR(*vty) +
                          "' is assigned as " + STR(*ety) + ".");
      if (debug_visit)
        dbgs() << "Error in " << __FILE__ << ", line: " << __LINE__ << ".\n";
      return false;
    }

    if (isa<FutureType>(ety)) {
      ModifySymbolType(n.GetName(), ety);
      ModifySymbolType(n.GetName() + ".span",
                       MakeRankedMDSpanType(ety->Dims()));
    } else
      choreo_unreachable("Expect a future type but got '" + PSTR(ety) + "'.");
  }

  // Allow re-assignment only for mutables
  if (!IsMutable(*vty)) {
    Error1(n.LOC(), "only mutables can be re-assigned (" + n.GetName() + ").");
    SetNodeType(n, MakeUnknownType());
    return false;
  }

  if (isa<AST::Call>(n.value)) {
    SetNodeType(*n.value, vty);
  } else if (!vty->ApprxEqual(*ety)) {
    Error1(n.LOC(), "`" + n.GetName() + "' of type \"" + STR(*vty) +
                        "\" can not be re-assigned as \"" + STR(*ety) + "\".");
    SetNodeType(n, MakeUnknownType());
    return false;
  }

  return true;
}

bool EarlySemantics::Visit(AST::IntIndex& n) {
  TraceEachVisit(n);
  SetNodeType(n, MakeIndexType());
  return true;
}

bool EarlySemantics::Visit(AST::DataType& n) {
  TraceEachVisit(n);

  allow_named_dim = false; // no duplicated symbol is allowed except for mdspan

  if (n.infer_span) {
    SetNodeType(n, MakeUnRankedSpannedType(n.getBaseType()));
  } else if (isa<SpannedType>(n.GetType())) {
    // The sema type has been generated. refine with dims
    if (auto sty = dyn_cast<MDSpanType>(n.mdspan_type->GetType()))
      SetNodeType(n, MakeRankedSpannedType(sty->Dims(), n.base_type));
  }
  return true;
}

bool EarlySemantics::Visit(AST::Identifier& n) {
  TraceEachVisit(n);

  // to check in parent node
  if (n.name == assign_id) return true;

  if (in_decl) {
    if (allow_named_dim) {
      if (!SSTab().DeclaredInScope(n.name))
        ReportErrorWhenViolateODR(n.LOC(), n.name, __FILE__, __LINE__,
                                  MakeIntegerType()); // named dim is integer
    } else
      ReportErrorWhenViolateODR(n.LOC(), n.name, __FILE__, __LINE__);
  } else {
    if (in_template_param && !SSTab().IsDeclared(n.name)) {
      Warning(n.LOC(), "symbol `" + n.name + "' is used before declaration.");
      SetNodeType(n, MakeIntegerType());
    } else
      ReportErrorWhenUseBeforeDefine(n.LOC(), n.name);
  }
  return true;
}

bool EarlySemantics::Visit(AST::Parameter& n) {
  TraceEachVisit(n);
  assert(n.type);

  auto nty = n.type->GetType()->Clone();

  if (n.GetAttr() == ParamAttr::GLOBAL_INPUT) {
    if (auto sty = dyn_cast<SpannedType>(nty))
      sty->SetStorage(Storage::GLOBAL);
    else
      Error1(n.LOC(),
             "Unable to pass in a global with the type: " + PSTR(nty) + ".\n");
  } else {
    if (auto mds = dyn_cast<AST::MultiDimSpans>(n.type->mdspan_type))
      if (auto mv = dyn_cast<AST::MultiValues>(mds->list))
        for (auto e : mv->AllValues())
          if (isa<NoValueType>(e->GetType()))
            Error1(e->LOC(),
                   "The parameter with an unbounded dimension must be global.");
  }

  // check the validity of pass-by-ref
  if (!n.HasSymbol())
    if (n.pass_by_ref)
      Error1(n.LOC(),
             "Unnamed parameter is not allowed to be passed by reference.");
  if (isa<ScalarType>(nty))
    if (n.pass_by_ref)
      Error1(n.LOC(), "Scalar parameter is not supported to be passed by "
                      "reference for now.");
  if (isa<SpannedType>(nty) && n.GetAttr() == ParamAttr::GLOBAL_INPUT)
    if (n.pass_by_ref)
      Error1(
          n.sym->LOC(),
          "The parameter '" + n.sym->name +
              "' is global, which is not allowed to be passed by reference.");

  if (n.HasSymbol()) {
    if (auto ty = dyn_cast<SpannedType>(nty)) {
      ReportErrorWhenViolateODR(n.LOC(), n.sym->name + ".span", __FILE__,
                                __LINE__,
                                ty->GetMDSpanType()); // named dim is integer
    }
    ModifySymbolType(n.sym->name, nty);
  }

  SetNodeType(n, nty);

  return true;
}

bool EarlySemantics::Visit(AST::ParamList& n) {
  TraceEachVisit(n);
  return true;
}

bool EarlySemantics::Visit(AST::ParallelBy& n) {
  TraceEachVisit(n);

  const std::string pl_anno_msg =
      "For nested parallel-by, it is better to either explicitly specify all "
      "parallelism levels, or none (Choreo to automatically infer the levels).";
  // The pb is specified with parallel level explicitly.
  if (n.GetLevel() != ParallelLevel::NONE) {
    // current is specified, but outer not.
    if (pl_depth > 1 && !explicit_pl) Warning(n.LOC(), pl_anno_msg);
    explicit_pl = true;
  } else {
    // outer is specified, but current not.
    if (explicit_pl) Warning(n.LOC(), pl_anno_msg);
  }

  if (pl_depth > 1 && n.IsAsync())
    Error1(n.LOC(), "inner parallel-by level can not be asynchronous.");

  auto bty = NodeType(*n.BoundExpr());
  if (!SupportIntListCollapse(bty))
    Error1(n.BoundExpr()->LOC(),
           "the parallel bound requires integers but got '" + PSTR(bty) + "'.");
  else if (!CanYieldAnInteger(bty) && !n.IsBracketed())
    Error1(n.BoundExpr()->LOC(),
           "must use mdspan instead of ituple to define the parallel bound.");

  if (n.HasSubPVs()) {
    for (auto sb : n.AllBoundExprs()) {
      auto sbty = NodeType(*sb);
      if (!SupportIntListCollapse(sbty))
        Error1(sb->LOC(), "the parallel bounds require integers but got '" +
                              PSTR(sbty) + "'.");
      else if ((n.SubPVCount() == 1) && isa<ITupleType>(sbty) &&
               !n.IsBracketed())
        Error1(
            n.BoundExpr()->LOC(),
            "must use mdspan instead of ituple to define the parallel bound.");
    }
    auto ub_count = CountMultiValues(n.BoundExprs());
    if (ub_count > 3)
      Error1(n.LOC(),
             "The number of parallel dimensions is limited to 3 (x, y, z).");

    SetNodeType(*n.BPV(), MakeBoundedITupleType(Shape(ub_count), "pv"));
  } else
    SetNodeType(*n.BPV(), MakeBoundedIntegerType(n.BPV()->name));

  ReportErrorWhenViolateODR(n.LOC(), n.BPV()->name, __FILE__, __LINE__,
                            n.BPV()->GetType());

  if (n.HasSubPVs()) {
    for (auto& sym : n.AllSubPVs()) {
      auto sname = cast<AST::Identifier>(sym)->name;
      auto mty = MakeBoundedIntegerType(sname);
      ReportErrorWhenViolateODR(n.LOC(), sname, __FILE__, __LINE__, mty);
      SetNodeType(*sym, mty);
    }
    SetNodeType(*n.SubPVs(), MakeBoundedITupleType(n.BoundExprs()->Count()));

    auto pv_count = CountMultiValues(n.SubPVs());
    auto ub_count = CountMultiValues(n.BoundExprs());
    if (pv_count > ub_count)
      Error1(n.LOC(), "parallel variables are more than their bounds (" +
                          std::to_string(pv_count) + " vs. " +
                          std::to_string(ub_count) + ").");
    else if (pv_count < ub_count)
      Error1(n.LOC(), "parallel variables are less than their bounds (" +
                          std::to_string(pv_count) + " vs. " +
                          std::to_string(ub_count) + ").");
  }

  // simple integer value check
  if (auto il = AST::GetIntLiteral(*n.BoundExpr()); il && (il->Val() <= 0))
    Error1(n.BPV()->LOC(),
           "bound " + STR(n.BoundExpr()) +
               " in parallelby is invalid: should be greater than 0.");

  for (auto& bv : n.AllBoundExprs())
    if (auto il = AST::GetIntLiteral(*bv); il && (il->Val() <= 0))
      Error1(n.LOC(),
             "bound item " + STR(bv) +
                 " in parallelby is invalid: should be greater than 0.");

  diverges.Add(InScopeName(n.BPV()->name));
  for (auto& v : n.AllSubPVs()) {
    auto name = AST::GetName(*v);
    assert(name.has_value() && "expect a name.");
    diverges.Add(InScopeName(name.value()));
  }

  return true;
}

bool EarlySemantics::Visit(AST::WhereBind& n) {
  TraceEachVisit(n);
  if (!isa<AST::Identifier>(n.lhs)) {
    Error1(n.lhs->LOC(), "expect an identifier.");
    return false;
  }
  if (!isa<AST::Identifier>(n.rhs)) {
    Error1(n.rhs->LOC(), "expect an identifier.");
    return false;
  }
  auto lname = cast<AST::Identifier>(n.lhs)->name;
  auto rname = cast<AST::Identifier>(n.rhs)->name;
  if (with_syms.count(lname) == 0) {
    Error1(n.lhs->LOC(),
           "symbol `" + lname + "' is not defined inside the with statement.");
  }
  if (with_syms.count(rname) == 0) {
    Error1(n.rhs->LOC(),
           "symbol `" + rname + "' is not defined inside the with statement.");
  }
  return true;
}

bool EarlySemantics::Visit(AST::WithIn& n) {
  TraceEachVisit(n);
  in_decl = true;

  auto ity = NodeType(*n.in);

  size_t rank = 0;
  if (auto itty = dyn_cast<ScalarIntegerType>(ity)) {
    rank = 1;
  } else if (auto mdst = dyn_cast<MDSpanType>(ity)) {
    rank = mdst->Dims();
  } else {
    Error1(n.in->LOC(), "expect a span type or int type, but got the " +
                            PSTR(ity) + " type.");
  }

  // `with {x} in 8 {}`
  if (!n.with && n.with_matchers && isa<ScalarIntegerType>(ity))
    Error1(n.in->LOC(),
           "expect a span type but got the " + PSTR(ity) + " type.");

  // check the if rank equal between with-in and with-matcher
  if (n.with_matchers && n.with_matchers->Count() != rank &&
      isa<MDSpanType>(ity))
    Error1(n.in->LOC(), "un-matched with-matcher-count(" +
                            std::to_string(n.with_matchers->Count()) +
                            ") and mdspan rank(" + std::to_string(rank) + ").");

  if (n.with && n.with->name == "_")
    Error1(n.LOC(),
           "_ is not allowed as a with variable. Can only be used in chunkat.");

  if (n.with_matchers)
    for (auto v : n.with_matchers->AllValues())
      if (auto id = dyn_cast<AST::Identifier>(v); id->name == "_") {
        Error1(v->LOC(),
               "_ is not allowed as a with variable. Can only be used "
               "in chunkat.");
        continue;
      }

  // infer the type of bounded variable
  if (n.with) {
    n.with->accept(*this); // make the symbol be defined
    with_syms.insert(n.with->name);
    ptr<Type> wty;
    if (isa<AST::IntLiteral>(n.in))
      wty = MakeBoundedIntegerType(n.with->name);
    else
      wty = MakeBoundedITupleType(Shape(rank));
    ModifySymbolType(n.with->name, wty);
    SetNodeType(*n.with, wty);
  }

  if (n.with_matchers) {
    n.with_matchers->accept(*this); // make the symbol be defined
    for (auto v : n.with_matchers->AllValues()) {
      // only id are accepted in with-matcher
      if (!isa<AST::Identifier>(v)) {
        Error1(v->LOC(), "expect an identifier.");
        continue;
      }
      auto sname = cast<AST::Identifier>(v)->name;
      if (with_syms.count(sname)) {
        Note(v->LOC(),
             "symbol `" + sname +
                 "' has been defined inside the with statement already.");
        continue;
      }
      with_syms.insert(sname);
      auto mty = MakeBoundedIntegerType(sname);
      ModifySymbolType(sname, mty);
      SetNodeType(*v, mty);
    }
    if (n.with)
      SetNodeType(*n.with_matchers, n.with->GetType());
    else
      SetNodeType(*n.with_matchers,
                  MakeBoundedITupleType(n.with_matchers->Count()));
  }
  in_decl = false;

  SetNodeType(n, ity);
  return true;
}

bool EarlySemantics::Visit(AST::WithBlock& n) {
  TraceEachVisit(n);
  return true;
}

bool EarlySemantics::Visit(AST::Memory& n) {
  TraceEachVisit(n);
  return true;
}

bool EarlySemantics::Visit(AST::SpanAs& n) {
  TraceEachVisit(n);

  for (auto val : n.list->AllValues())
    if (mutables.Contains(val))
      Error1(val->LOC(),
             "the mutable value can not used for mdspan declaration.");

  auto sty = GetSpannedType(NodeType(*n.id));
  if (!sty) {
    Error1(n.LOC(), "span-as operation operates on a non-mdspan type.");
    return false;
  }

  size_t cnt = n.list->Count();
  // handle `y = x.span_as([...])`
  if (n.list->Count() == 1)
    if (auto e = dyn_cast<AST::Expr>(n.list->ValueAt(0)))
      if (auto mds = dyn_cast<AST::MultiDimSpans>(e->GetReference()))
        cnt = mds->rank;

  auto asty = MakeRankedSpannedType(cnt, (BaseType)sty->e_type, sty->m_type);
  ReportErrorWhenViolateODR(n.LOC(), n.nid->name, __FILE__, __LINE__, asty);
  SetNodeType(n, asty);

  // TODO: set the proper type
  return true;
}

bool EarlySemantics::Visit(AST::DMA& n) {
  TraceEachVisit(n);

  if (n.operation == ".any") { // skip the place holder
    assert(!n.future.empty());
    ReportErrorWhenViolateODR(n.LOC(), n.future, __FILE__, __LINE__,
                              MakePlaceHolderFutureType());
    ReportErrorWhenViolateODR(n.LOC(), n.future + ".span", __FILE__, __LINE__,
                              MakePlaceHolderMDSpanType());
    ReportErrorWhenViolateODR(n.LOC(), n.future + ".data", __FILE__, __LINE__,
                              MakePlaceHolderSpannedType());
    return true;
  }

  auto sty = GetSpannedType(NodeType(*n.from));

  ptr<SpannedType> tty = nullptr;
  if (!isa<AST::Memory>(n.to)) {
    tty = dyn_cast<SpannedType>(NodeType(*n.to));
    if (!tty) {
      Error1(n.to->LOC(),
             "The DMA destination is neither storage identifier nor span.");
      return true;
    }
  }

  if (pl_depth == 0) {
    if (auto m = dyn_cast<AST::Memory>(n.to)) {
      if ((m->Get() != Storage::GLOBAL) && (m->Get() != Storage::DEFAULT)) {
        Error1(n.LOC(),
               "`" + STR(m->Get()) +
                   "' can not be DMA destination outside parallel-by.");
        if (debug_visit)
          dbgs() << "Error in " << __FILE__ << ", line: " << __LINE__ << ".\n";
      }
    } else if ((tty->GetStorage() != Storage::GLOBAL) &&
               (tty->GetStorage() != Storage::DEFAULT)) {
      Error1(n.LOC(), "`" + STR(tty->GetStorage()) +
                          "' can not be DMA destination outside parallel-by.");
      if (debug_visit)
        dbgs() << "Error in " << __FILE__ << ", line: " << __LINE__ << ".\n";
    }
  }

  if (!n.future.empty()) {
    Storage sto = Storage::NONE;
    if (auto m = dyn_cast<AST::Memory>(n.to))
      sto = m->Get();
    else
      sto = tty->GetStorage();

    // rewrite the placeholder type
    if (SSTab().IsDeclared(n.future)) {
      auto ptype = dyn_cast<PlaceHolderType>(GetSymbolType(n.future));
      if (!ptype || ptype->GetBaseType() != BaseType::FUTURE) {
        Error1(n.LOC(), "symbol `" + n.future + "' has been declared already.");
        VST_DEBUG(dbgs() << "Error in " << __FILE__ << ", line: " << __LINE__
                         << ".\n");
      }
      ReportErrorWhenUseBeforeDefine(n.LOC(), n.future + ".span");
      ReportErrorWhenUseBeforeDefine(n.LOC(), n.future + ".data");
      auto spanned_ty = MakeRankedSpannedType(sty->GetShape().Rank(),
                                              sty->ElementType(), sto);
      ModifySymbolType(n.future + ".data", spanned_ty);
      ModifySymbolType(n.future + ".span",
                       spanned_ty->GetMDSpanType()->Clone());
      if (n.IsSparse()) ModifySymbolType(n.future + ".mdata", spanned_ty);
      ModifySymbolType(n.future, MakeFutureType(spanned_ty, n.IsAsync()));
    } else {
      auto spanned_ty = MakeRankedSpannedType(sty->GetShape().Rank(),
                                              sty->ElementType(), sto);
      ReportErrorWhenViolateODR(n.LOC(), n.future + ".data", __FILE__, __LINE__,
                                spanned_ty);
      ReportErrorWhenViolateODR(n.LOC(), n.future + ".span", __FILE__, __LINE__,
                                spanned_ty->GetMDSpanType()->Clone());
      if (n.IsSparse())
        ReportErrorWhenViolateODR(n.LOC(), n.future + ".mdata", __FILE__,
                                  __LINE__, spanned_ty);
      ReportErrorWhenViolateODR(n.LOC(), n.future, __FILE__, __LINE__,
                                MakeFutureType(spanned_ty, n.IsAsync()));
    }

    // set the buffer kind
    auto from_kind =
        (cast<AST::ChunkAt>(n.from)->NoOperation()) ? DOK_SYMBOL : DOK_CHUNK;
    auto to_kind = DOK_UNKNOWN;
    if (!isa<AST::ChunkAt>(n.to))
      to_kind = DOK_SYMBOL;
    else
      to_kind =
          (cast<AST::ChunkAt>(n.to)->NoOperation()) ? DOK_SYMBOL : DOK_CHUNK;
    auto to_sym = n.ToSymbol();
    if (!to_sym.empty()) to_sym = InScopeName(to_sym);
    FCtx(fname).GetFutureBufferInfo().emplace(
        InScopeName(n.future), DMABufferInfo{to_sym, from_kind, to_kind});
  } else if (n.IsAsync())
    Error1(n.LOC(), "forbid to associated async dma without a named future.");

  if (isa<AST::ChunkAt>(n.from) && isa<AST::ChunkAt>(n.to))
    if (cast<AST::ChunkAt>(n.from)->HasTilingOperation() &&
        cast<AST::ChunkAt>(n.to)->HasTilingOperation() &&
        (CCtx().HasFeature(ChoreoFeature::DSDMA))) {
      Error1(n.LOC(),
             "slice and deslice in single DMA statement is not supported yet.");
    }

  if (!isa<AST::Memory>(n.to)) {
    if (!CompatibleRank(sty->Dims(), tty->Dims()) && !allow_auto_threading) {
      Error1(n.LOC(),
             "DMA statement has a rank mismatch between 'from' and 'to': " +
                 std::to_string(sty->Dims()) +
                 " != " + std::to_string(tty->Dims()) + ".");
    }
    if (sty->ElementType() != tty->ElementType()) {
      Error1(n.LOC(),
             "DMA statement contains a type mismatch: the element types of "
             "the 'from'(" +
                 STR(sty->ElementType()) + ") and 'to'(" +
                 STR(tty->ElementType()) + ") are inconsistent.");
    }
  }

  // dma.pad specific check
  if (n.operation == ".pad") {
    auto pcfg = dyn_cast<PadConfig>(n.config);
    if (!pcfg) {
      Error1(n.LOC(), "The DMA PAD config is incorrect. The correct form: "
                      "dma.pad(.async)<{pad_highs}, {pad_lows}, {pad_mids}, "
                      "padding value>.");
    } else if (pcfg->pad_high->Count() != pcfg->pad_low->Count() ||
               (pcfg->pad_low->Count() != pcfg->pad_mid->Count())) {
      if (pcfg->pad_high->Count() != pcfg->pad_low->Count())
        Error1(n.LOC(), "DMA PAD statement has a rank mismatch between the "
                        "pad-low and pad-high: " +
                            std::to_string(pcfg->pad_low->Count()) + " != " +
                            std::to_string(pcfg->pad_high->Count()) + ".");
      if (pcfg->pad_low->Count() != pcfg->pad_mid->Count())
        Error1(n.LOC(), "DMA PAD statement has a rank mismatch between the "
                        "pad-low and pad-mid: " +
                            std::to_string(pcfg->pad_low->Count()) + " != " +
                            std::to_string(pcfg->pad_mid->Count()) + ".");
    } else if (!CompatibleRank(NodeType(*n.from)->Dims(),
                               pcfg->pad_high->Count())) {
      Error1(n.LOC(), "DMA PAD statement has a rank mismatch: " +
                          std::to_string(NodeType(*n.from)->Dims()) + " != " +
                          std::to_string(pcfg->pad_high->Count()) + ".");
    }

    if (!isa<ScalarType>(NodeType(*pcfg->GetPadValue())))
      Error1(pcfg->GetPadValue()->LOC(),
             "The padding value should be of scalar type.");
    for (const auto& mv : {pcfg->pad_high, pcfg->pad_low, pcfg->pad_mid})
      for (const auto& v : mv->AllValues())
        if (!isa<ScalarIntegerType>(NodeType(*v)))
          Error1(v->LOC(),
                 "The padding config should be of scalar integer type.");
  }

  // dma.transp specific check
  if (n.operation == ".transp") {
    auto tcfg = dyn_cast<TransposeConfig>(n.config);
    if (!tcfg) {
      Error1(n.LOC(), "DMA TRANSPOSE config is incorrect. The correct form: "
                      "dma.transp(.async)<dim0, dim1, ...>");
    } else {
      auto dim_values = tcfg->dim_values;
      if (!CompatibleRank(dim_values.size(), sty->Dims())) {
        Error1(n.LOC(), "DMA TRANSPOSE statement has a rank mismatch between "
                        "the 'transpose "
                        "layout' and 'from': " +
                            std::to_string(dim_values.size()) +
                            " != " + std::to_string(sty->Dims()) + ".");
      }
      if (!isa<AST::Memory>(n.to)) {
        if (!CompatibleRank(dim_values.size(), sty->Dims())) {
          Error1(n.LOC(),
                 "DMA TRANSPOSE statement has a rank mismatch: the 'transpose "
                 "layout' and 'to': " +
                     std::to_string(dim_values.size()) +
                     " != " + std::to_string(sty->Dims()) + ".");
        }
      }
      std::sort(dim_values.begin(), dim_values.end());
      for (size_t i = 0; i < dim_values.size(); ++i) {
        if (dim_values[i] != i) {
          Error1(n.LOC(), "The DMA statement contains an error: the transpose "
                          "layout is invalid.");
          break;
        }
      }
    }
  }

  return true;
}

bool EarlySemantics::Visit(AST::MMA& n) {
  auto old_ec = error_count;
  auto& op = *n.GetOperation();
  switch (op.Tag()) {
  case AST::MMAOperation::Fill: {
    // MMA is a 2D operation
    std::string fill_sym = FragName(op.FillingTo());
    if (op.FillingIsDecl()) {
      // `mc[1] = mma.fill 0;` is invalid.
      assert(!AST::FragIsArrayElem(op.FillingTo()));
      if (op.FillingArrayDims()) {
        auto arr_type = cast<ArrayType>(op.FillingArrayDims()->GetType());
        ReportErrorWhenViolateODR(
            n.LOC(), fill_sym, __FILE__, __LINE__,
            MakeRankedSpannedArrayType(2, arr_type->dims));
      } else {
        ReportErrorWhenViolateODR(n.LOC(), fill_sym, __FILE__, __LINE__,
                                  MakeRankedSpannedType(2));
      }
      ReportErrorWhenViolateODR(n.LOC(), fill_sym + ".span", __FILE__, __LINE__,
                                MakeRankedMDSpanType(2));
      if (!isa<ScalarType>(op.FillingValue()->GetType()))
        Error1(n.LOC(), "Expect a scalar value for MMA fill.");
    } else {
      ReportErrorWhenUseBeforeDefine(n.LOC(), fill_sym);
    }
  } break;
  case AST::MMAOperation::Load: {
    std::string fut_sym = AST::FragName(op.GetFuture());
    assert(!AST::FragIsArrayElem(op.GetFuture()) &&
           "For now, frag with indices is only supported for mc.");
    auto sty = dyn_cast<SpannedType>(op.LoadFrom()->GetType());
    if (!sty) Error1(n.LOC(), "Expected a spanned buffer for MMA load.");
    ReportErrorWhenViolateODR(
        n.LOC(), fut_sym, __FILE__, __LINE__,
        MakeFutureType(cast<SpannedType>(sty->Clone()), op.IsAsync()));
    ReportErrorWhenViolateODR(n.LOC(), fut_sym + ".span", __FILE__, __LINE__,
                              cast<SpannedType>(sty->Clone()));
  } break;
  case AST::MMAOperation::Exec: {
    std::string op0_sym = AST::FragName(op.ExecOperand(0));
    std::string op1_sym = AST::FragName(op.ExecOperand(1));
    std::string op2_sym = AST::FragName(op.ExecOperand(2));
    ReportErrorWhenUseBeforeDefine(n.LOC(), op0_sym);
    ReportErrorWhenUseBeforeDefine(n.LOC(), op1_sym);
    ReportErrorWhenUseBeforeDefine(n.LOC(), op2_sym);
    if (op.IsSparse() && op.ExecOperand(3)) {
      std::string op3_sym = AST::FragName(op.ExecOperand(3));
      ReportErrorWhenUseBeforeDefine(n.LOC(), op3_sym);
    }
    if (isa<ArrayType>(GetSymbolType(op0_sym)) &&
        !AST::FragIsArrayElem(op.ExecOperand(0)))
      Error1(op.ExecOperand(0)->LOC(),
             "Cannot use the whole fragment array in mma exec operation.");
  } break;
  case AST::MMAOperation::Store: {
    std::string sto_from_sym = AST::FragName(op.StoreFrom());
    ReportErrorWhenUseBeforeDefine(n.LOC(), sto_from_sym);
    auto sty = op.StoreTo()->GetType();
    if (!isa<SpannedType>(sty))
      Error1(n.LOC(), "Expected a spanned buffer for MMA store.");
  } break;
  default: break;
  }
  return error_count == old_ec;
}

bool EarlySemantics::Visit(AST::ChunkAt& n) {
  TraceEachVisit(n);

  n.data->accept(*this);
  auto nty = NodeType(*n.data);
  if (n.sa) nty = NodeType(*n.sa);

  if (!isa<SpannedType>(nty) && !isa<FutureType>(nty)) {
    Error1(n.LOC(),
           "expect '" + n.data->name + "` of a spanned data or future type.");
    return true;
  }

  auto sty = GetSpannedType(nty);

  if (IsUnknownRank(sty->Dims())) {
    SetNodeType(n,
                MakeUnRankedSpannedType(sty->ElementType(), sty->GetStorage()));
    return true;
  }

  auto rank = sty->GetShape().Rank();

  for (auto op : n.AllOperations()) {
    op->accept(*this);

    if (auto rop = dyn_cast<AST::SOP::Reshape>(op)) {
      size_t r_count = 0;
      for (auto v : rop->GetNewSpan()->AllValues()) {
        auto ty = NodeType(*v);
        if (!CanYieldDimension(ty) && !isa<MDSpanType>(ty))
          Error1(v->LOC(), "the value (" + PSTR(ty) +
                               ") can not used for declaring a mdspan.");
        if (IsValidRank(ty->Dims()) && IsValidRank(r_count))
          r_count += ty->Dims();
      }
      if (IsValidRank(r_count)) rank = r_count;
      // can not check further util shapeinfer is done
      continue;
    }

    for (auto& v : op->IndexNodes()) {
      auto ty = NodeType(*v);
      if (!CanYieldIndex(ty))
        Error1(v->LOC(), "expect '" + PSTR(v) +
                             "` to yield an index (but got " + PSTR(ty) + ").");
      SetNodeType(*v, ty);
    }
    for (auto& v : op->TilingFactorNodes()) {
      auto ty = NodeType(*v);
      if (!IntegersOnly(ty))
        Error1(v->LOC(), "expect '" + PSTR(v) +
                             "` be a type with only integers (but got " +
                             PSTR(ty) + ").");
      SetNodeType(*v, ty);
    }
    for (auto& v : op->SubSpanNodes()) {
      auto ty = NodeType(*v);
      if (!isa<ScalarIntegerType>(ty) && !isa<ITupleType>(ty) &&
          !isa<MDSpanType>(ty))
        Error1(
            v->LOC(),
            "expect '" + PSTR(v) +
                "` to be either an integer, ituple, or mdspan type (but got " +
                PSTR(ty) + ").");
      SetNodeType(*v, ty);
    }
    for (auto& v : op->OffsetNodes()) {
      auto ty = NodeType(*v);
      if (!isa<ScalarIntegerType>(ty) && !isa<ITupleType>(ty) &&
          !isa<MDSpanType>(ty) && !isa<BoundedType>(ty))
        Error1(v->LOC(), "expect '" + PSTR(v) +
                             "` to be either an integer, ituple, mdspan type "
                             "or bounded type (but got " +
                             PSTR(ty) + ").");
      SetNodeType(*v, ty);
    }
  }

  SetNodeType(
      n, MakeRankedSpannedType(rank, sty->ElementType(), sty->GetStorage()));
  return true;
}

bool EarlySemantics::Visit(AST::Wait& n) {
  TraceEachVisit(n);

  for (auto& v : n.targets->AllValues()) {
    if (!AST::IsSymbolOrArrayRef(*v))
      Error1(n.LOC(),
             "expect a symbol/array reference but got '" + AST::STR(*v) + "'.");

    auto ty = NodeType(*v);

    if (auto fty = dyn_cast<FutureType>(ty)) {
      if (!fty->IsAsync())
        Error1(n.LOC(), "non-async future '" + AST::GetName(*v).value() +
                            "` can not be waited.");
      continue;
    } else if (auto pty = dyn_cast<PlaceHolderType>(ty)) {
      if (pty->GetBaseType() != BaseType::FUTURE)
        Error1(n.LOC(), "'" + AST::GetName(*v).value() + "` of type \"" +
                            PSTR(ty) + "\" can not be waited.");
    } else if (isa<EventArrayType>(ty)) {
      if (inthreads_levels[pl_depth] == 0)
        Warning(v->LOC(), "Be careful to wait event outside inthreads block, "
                          "which may lead to parallelism issues.");
    } else if (!isa<EventType>(ty)) {
      Error1(n.LOC(), "'" + STR(n) + "` of type \"" + PSTR(ty) +
                          "\" can not be waited.");
    }
  }
  return true;
}

bool EarlySemantics::Visit(AST::Trigger& n) {
  TraceEachVisit(n);

  for (auto& v : n.targets->AllValues()) {
    if (!AST::IsSymbolOrArrayRef(*v))
      Error1(v->LOC(),
             "expect a symbol/array reference but got '" + AST::STR(*v) + "'.");
    auto ty = NodeType(*v);
    if (isa<EventArrayType>(ty)) {
      if (inthreads_levels[pl_depth] == 0)
        Warning(v->LOC(),
                "Be careful to trigger event outside inthreads block, "
                "which may lead to parallelism issues.");
    }
    if (!isa<EventType>(ty))
      Error1(v->LOC(),
             "expect `" + PSTR(v) + "' an event but got '" + PSTR(ty) + "'.");
  }

  return true;
}

bool EarlySemantics::Visit(AST::Break& n) {
  TraceEachVisit(n);

  if (!inside_loop) {
    Error1(n.LOC(), "unable to break outside a loop.");
    return false;
  }

  return true;
}

bool EarlySemantics::Visit(AST::Call& n) {
  TraceEachVisit(n);

  size_t ec = error_count;

  if ((pl_depth == 0) && !n.IsBIF()) {
    Error1(n.LOC(),
           "unable to call kernel function outside the parallel-by block(s).");
    return false;
  }

  if (n.IsBIF()) {
    if (n.template_args)
      Error1(n.LOC(), "the built-in functions are not function templates.");
    const auto func_name = n.function->name;
    if (func_name == "assert") {
      auto pty = NodeType(*n.arguments->ValueAt(0));
      if (!isa<BooleanType>(pty))
        Error1(n.LOC(), "expect a predicate but got '" + PSTR(pty) + "'.");
      auto sty = NodeType(*n.arguments->ValueAt(1));
      if (!isa<StringType>(sty))
        Error1(n.LOC(), "expect a string but got '" + PSTR(sty) + "'.");
    } else if (func_name == "print" || func_name == "println") {
      auto Printable = [](ptr<Type> ty) -> bool {
        if (isa<StringType>(ty) || isa<ScalarIntegerType>(ty) ||
            isa<EventType>(ty) || isa<F32Type>(ty) || isa<F64Type>(ty) ||
            isa<ITupleType>(ty) || isa<MDSpanType>(ty) ||
            isa<BoundedType>(ty) || isa<F16Type>(ty) || isa<BF16Type>(ty) ||
            isa<AddrType>(ty) || isa<BooleanType>(ty))
          return true;
        // half8 is invalid.
        return false;
      };
      for (const auto& arg : n.GetArguments()) {
        const auto aty = NodeType(*arg);
        if (n.CompileTimeEval()) {
          if (isa<ScalarType>(aty)) {
            if (isa<BooleanType>(aty) || isa<ScalarFloatType>(aty))
              Warning(arg->LOC(), "compile-time evaluation of type '" +
                                      PSTR(aty) + "' is yet to support.");
          } else if (isa<BoundedType>(aty) || isa<EventType>(aty) ||
                     isa<SpannedType>(aty) || isa<AsyncType>(aty) ||
                     isa<AddrType>(aty)) {
            Warning(arg->LOC(), "compile-time evaluation of type '" +
                                    PSTR(aty) +
                                    "' is impossible since its value is "
                                    "determined at runtime.");
          }
        } else if (!Printable(aty))
          Error1(arg->LOC(), "the argument of type '" + PSTR(aty) +
                                 "' is not supported for printing.");
      }
    } else if (n.IsArith()) {
      auto pty = NodeType(*n.arguments->ValueAt(0));
      SetNodeType(n, pty);
    } else if (func_name == "__alignup" || func_name == "__aligndown") {
      if (n.arguments->Count() != 2)
        Error1(n.LOC(), "expect 2 arguments but got " +
                            std::to_string(n.arguments->Count()) + ".");
      for (size_t i = 0; i < n.arguments->Count(); ++i) {
        auto arg_ty = NodeType(*n.arguments->ValueAt(i));
        if (!isa<ScalarIntegerType>(arg_ty))
          Error1(n.LOC(), "expect the " + std::to_string(i) +
                              "th argument to be a integer type but got '" +
                              PSTR(arg_ty) + "'.");
      }
      auto pty = NodeType(*n.arguments->ValueAt(0));
      SetNodeType(n, pty);
    } else
      choreo_unreachable("unsupported bif '" + func_name + "'.");

    return ec == error_count;
  }

  if (resolve_fns && !n.IsBIF()) {
    auto function_name = n.function->name;

    ptr<AST::DeviceFunctionDecl> matched_function = nullptr;
    ptr<AST::DeviceFunctionDecl> candidate_function = nullptr;
    std::vector<ptr<DeviceDataType>> real_param_types;
    ptr<DeviceDataType> real_ret_type = nullptr;
    std::string mismatch_msg;
    for (auto& f : device_functions) {
      if (f->name != n.function->name) continue;
      candidate_function = f;
      mismatch_msg = "";
      std::unordered_map<std::string, BaseType> template_param_map;

      size_t param_count_uninitized = 0;
      for (auto& param : f->param_types) {
        if (!param->Initized())
          param_count_uninitized++;
        else
          break;
      }

      if (!(candidate_function->param_types.size() >= n.arguments->Count() &&
            n.arguments->Count() >= param_count_uninitized)) {
        mismatch_msg = "the function '" + function_name + "' requires " +
                       std::to_string(param_count_uninitized) +
                       " parameters, but " +
                       std::to_string(n.arguments->Count()) +
                       " arguments are "
                       "provided.";
        continue;
      }
      // check the template arguments
      bool template_match = true;
      using TemplateParam = AST::DeviceFunctionDecl::DeviceTemplateParam;

      if (candidate_function->IsTemplated() && n.template_args) {
        auto templ_params = candidate_function->template_params;
        // note:the number of template arguments may not equal the number of
        // template parameters, since there may be some template parameters with
        // default values
        size_t count = std::min(templ_params.size(), n.template_args->Count());
        // deduce the types of template parameters by the template arguments
        for (size_t i = 0; i < count; i++) {
          auto& templ_param = templ_params[i];
          auto templ_arg = dyn_cast<AST::Expr>(n.template_args->ValueAt(i));
          auto templ_arg_ty = templ_arg->GetType();

          if (templ_param.kind == TemplateParam::UNKNOWN) {
            template_match = false;
            break;
          } else if (templ_param.kind == TemplateParam::VALUE) {
            auto arg_bt = templ_arg_ty->GetBaseType();
            auto device_type_match = [](BaseType lhs, std::string str) {
              using BT = BaseType;
              if (!(IsIntegralType(lhs) || lhs == BT::BOUNDED_INT ||
                    lhs == BT::INDEX)) {
                choreo_unreachable(
                    "Unexpected base type for device type match: " + STR(lhs));
              }
              auto rhs = DSTR2BT(str);
              if (rhs == BaseType::UNKNOWN) { return false; }
              return IsValuePreservingCast(lhs, rhs) ||
                     IsReinterpretiveCast(lhs, rhs);
            };

            if (!device_type_match(arg_bt, templ_param.type_name)) {
              auto it = template_param_map.find(templ_param.type_name);
              if (it != template_param_map.end()) {
                if (it->second != arg_bt) {
                  template_match = false;
                  break;
                }
              } else {
                template_match = false;
                break;
              }
            }
          } else if (templ_param.kind == TemplateParam::TYPE) {
            if (templ_arg->IsReference() &&
                isa<AST::DataType>(templ_arg->GetReference())) {
              auto arg_dt = dyn_cast<AST::DataType>(templ_arg->GetReference());
              auto param_name = templ_param.param_name;
              template_param_map[param_name] = arg_dt->getBaseType();
            } else {
              template_match = false;
              break;
            }
          }
        }
      }
      if (!template_match)
        Warning(n.LOC(), "unmatched template arguments of the device "
                         "function '" +
                             n.function->name + "'.");

      // check the argument types
      bool arg_match = true;
      for (size_t param_idx = 0; param_idx < n.arguments->Count();
           ++param_idx) {
        auto arg_ty = NodeType(*n.arguments->ValueAt(param_idx));
        auto param_ty = candidate_function->param_types[param_idx];
        auto param_name = param_ty->PlainName();
        auto real_param_type = dyn_cast<DeviceDataType>(param_ty->Clone());
        // update param type with template arguments
        if (template_param_map.find(param_name) != template_param_map.end())
          real_param_type->SetDataType(template_param_map[param_name]);
        real_param_types.push_back(real_param_type);
        // We allow the argument type promoted to parameter type, which means
        // IsValuePreservingCast(arg_ty, param_ty) should return true. For
        // example, a float argument can be passed to a double parameter. Other
        // type cast is not allowed.
        if (!real_param_type->ApprxEqual(*arg_ty)) {
          arg_match = false;
          mismatch_msg = "the type of " + std::to_string(param_idx + 1) +
                         "th argument '" + PSTR(arg_ty) +
                         "' is not compatible with the parameter type '" +
                         PSTR(real_param_type) + "'.";
          break;
        }
      }
      // if all arguments match, we will use this function
      if (arg_match) {
        matched_function =
            dyn_cast<AST::DeviceFunctionDecl>(candidate_function->Clone());
        matched_function->param_types = real_param_types;
        real_ret_type =
            dyn_cast<DeviceDataType>(matched_function->ret_type->Clone());

        if (template_param_map.find(real_ret_type->PlainName()) !=
            template_param_map.end()) {
          real_ret_type->SetDataType(
              template_param_map[real_ret_type->PlainName()]);
          matched_function->ret_type = real_ret_type;
        }
      }
    } // for each device function

    // if no matched function, we will report a warning
    // and suggest the candidate function.
    if (!matched_function) {
      Warning(n.LOC(),
              "unable to find a device function '" + function_name + "'.");
      if (candidate_function)
        Warning(candidate_function->LOC(),
                "candidate function '" + candidate_function->name + "' with " +
                    std::to_string(candidate_function->param_types.size()) +
                    " parameters is found, but " + mismatch_msg);
    } else {
      n.device_functions.push_back(matched_function);
      if (real_ret_type) {
        auto call_ty = MakeChoreoDataType(real_ret_type);
        SetNodeType(n, call_ty);
      }
    }
  } // end of resolve_fns

  size_t count = 0;
  for (auto& v : n.arguments->AllValues()) {
    count++;
    auto ty = NodeType(*v);
    // must be a callable type
    if (!CanYieldAnInteger(ty) && !isa<SpannedType>(ty) && !isa<AddrType>(ty))
      Error1(v->LOC(), "(" + Ordinal(count) + ") argument of type '" +
                           PSTR(ty) +
                           "` can not be passed to the kernel function.");
  }

  if (n.template_args) {
    size_t count = 0;
    for (auto& v : n.template_args->AllValues()) {
      count++;
      auto expr = cast<AST::Expr>(v);
      if (expr->IsReference() && isa<AST::DataType>(expr->GetReference())) {
        continue;
      }
      auto ty = NodeType(*v);
      // must be a scalar type
      if (ty && !ConvertibleToInt(ty))
        Error1(n.LOC(),
               "(" + std::to_string(count) + "th) template argument of type '" +
                   PSTR(ty) +
                   "` can not be used to instantiate the kernel function.");
    }
  }

  return true;
}

bool EarlySemantics::Visit(AST::Synchronize& n) {
  TraceEachVisit(n);

  switch (n.Resource()) {
  case Storage::GLOBAL:
  case Storage::SHARED:
  case Storage::LOCAL: break;
  default:
    Error1(n.LOC(), "Unsupported synchronization: " + STR(n.Resource()) + ".");
    break;
  }
  return true;
}

bool EarlySemantics::Visit(AST::Rotate& n) {
  TraceEachVisit(n);

  ptr<Type> lty = nullptr;
  for (size_t index = 0; index < n.ids->Count(); ++index) {
    auto pnode = n.ValueAt(index);
    auto cname = cast<AST::Identifier>(pnode)->name;
    auto cty = NodeType(*pnode);
    if (!GeneralFutureType(*cty)) {
      Error1(n.LOC(), "only support swapping of 'future'. (" +
                          n.IdAt(index)->name + ": " + PSTR(lty) + ").");
      return false;
    }

    // Avoid to swap a 'chunkat' target where no explicit buffer symbol is
    // associated.
    if (FCtx(fname).GetFutureBufferInfo()[InScopeName(cname)].to_kind ==
        DOK_CHUNK) {
      Error1(n.LOC(), "rotate/swap a 'future' referring a buffer chunk has not "
                      "been supported yet.");
      return false;
    }

    if (index < 1) continue;
    lty = NodeType(*n.ValueAt(index - 1));

    if (!lty->ApprxEqual(*cty)) {
      Error1(n.LOC(), "rotate/swap data of different types (" + PSTR(lty) +
                          " vs. " + PSTR(cty));
      return false;
    }
  }

  auto fty = type_equals.ResolveEqualFutures(*n.ids);

  if (!fty) {
    Error1(n.LOC(), "Fail to resolve types for swap/rotate.");
    return false;
  } else if (isa<PlaceHolderType>(fty))
    return true; // do not apply placeholders

  // add the missing ".span" type
  for (size_t index = 0; index < n.ids->Count(); ++index) {
    if (isa<PlaceHolderType>(NodeType(*n.ValueAt(index)))) {
      auto lname = AST::GetName(*n.ValueAt(index));
      assert(lname.has_value());
      ModifySymbolType(*lname + ".span", MakeRankedMDSpanType(fty->Dims()));
    }
  }

  return true;
}

bool EarlySemantics::Visit(AST::Select& n) {
  TraceEachVisit(n);

  size_t ec = error_count;

  if (auto sf_type = NodeType(*n.select_factor);
      !isa<BoundedIntegerType>(sf_type) && !isa<ScalarIntegerType>(sf_type))
    Error1(n.LOC(), "expect `" + PSTR(n.select_factor) +
                        "` to be a (bounded) integer type, but got " +
                        sf_type->TypeNameString() + ".");

  // check value types in val_list are the same
  assert(n.expr_list->Count() > 0);
  const auto& v0 = n.expr_list->AllValues()[0];
  auto v0ty = NodeType(*v0);

  if (!GeneralFutureType(v0ty) && !isa<SpannedType>(v0ty)) {
    Error1(v0->LOC(),
           "expect `" + PSTR(v0ty) + "` to be a future/spanned type.");
    return ec == error_count;
  }

  ptr<Type> sel_fty = nullptr;
  for (auto& v : n.expr_list->AllValues()) {
    auto nty = NodeType(*v);
    if (!nty->ApprxEqual(*v0ty))
      Error1(v->LOC(), "expect `" + PSTR(v) + "`(" + PSTR(nty) +
                           ") to be the same type as `" + PSTR(v0) + "`(" +
                           PSTR(v0ty) + ").");

    if (isa<FutureType>(nty)) sel_fty = nty;
  }

  if (sel_fty) {
    auto rank = cast<FutureType>(sel_fty)->Dims();
    // some elements could be placeholders, propagate the type
    for (auto& v : n.expr_list->AllValues()) {
      SetNodeType(*v, sel_fty);
      if (auto name = AST::GetName(*v)) {
        ModifySymbolType(*name, sel_fty);
        ModifySymbolType(*name + ".span", MakeRankedMDSpanType(rank));
      }
    }
  }

  SetNodeType(n, ShadowTypeStorage(NodeType(*n.expr_list->AllValues()[0])));

  bool all_diverge = true;
  for (auto& v : n.expr_list->AllValues())
    if (diverges.Contains(v)) {
      all_diverge = false;
      break;
    }

  if (all_diverge) diverges.Add(n);

  return true;
}

bool EarlySemantics::Visit(AST::Return& n) {
  TraceEachVisit(n);
  found_return = true;
  if (pl_depth != 0) {
    Error1(n.LOC(), "unable to return inside the parallel-by block(s).");
    return false;
  }

  if (n.value) {
    if (auto rexp = dyn_cast<AST::Expr>(n.value)) {
      if (isa<AST::ChunkAt>(rexp->GetR())) {
        Error1(n.LOC(), "illegal: chunkat is used in return expression.");
        return false;
      }
    }

    auto vty = NodeType(*n.value);
    if (!(isa<SpannedType>(vty) || isa<ScalarType>(vty))) {
      Error1(n.LOC(),
             "returning value with type '" + PSTR(vty) + "' is not supported.");
      return false;
    }

    if (CCtx().HasFeature(ChoreoFeature::NSVR)) {
      if (isa<ScalarType>(vty)) {
        Error1(
            n.LOC(),
            "returning scalar value in Factor backend is not supported yet.");
        return false;
      }
    }
  }

  return true;
}

bool EarlySemantics::Visit(AST::LoopRange& n) {
  TraceEachVisit(n);

  if (n.lbound && !isa<ScalarIntegerType>(NodeType(*n.lbound)))
    Error1(n.lbound->LOC(), "the lower bound is not an integer.");
  if (n.ubound && !isa<ScalarIntegerType>(NodeType(*n.ubound)))
    Error1(n.ubound->LOC(), "the upper bound is not an integer.");

  return true;
}

bool EarlySemantics::Visit(AST::ForeachBlock& n) {
  TraceEachVisit(n);

  for (auto& i : n.GetRanges()) {
    if (auto id = dyn_cast<AST::LoopRange>(i)->iv) {
      if (id->name == "_") {
        Error1(id->LOC(), "_ is not allowed as an iteration variable.");
        continue;
      }
      auto ity = NodeType(*id);
      if (!(isa<BoundedType>(ity))) {
        Error1(id->LOC(), "expect a bounded type for iteration variable '" +
                              id->name + "' but got '" + PSTR(ity) + "'.");
        continue;
      }
      std::string scope_name = GetScope(InScopeName(id->name));
      auto scopes = SplitStringByDelimiter(scope_name, "::");
      if (!PrefixedWith(scopes.back(), "within_")) {
        std::string error_msg = "expect the bounded variable '" + id->name +
                                "' to be declared by 'within' block";
        if (PrefixedWith(scopes.back(), "paraby_"))
          error_msg += " instead of 'parallel-by' block";
        Error1(id->LOC(), error_msg + ".");
      }
    } else {
      auto ity = i->GetType();
      Error1(i->LOC(),
             "expect a range expression but got '" + PSTR(ity) + "'.");
    }
  }
  return true;
}

bool EarlySemantics::Visit(AST::InThreadsBlock& n) {
  TraceEachVisit(n);
  if (!isa<BooleanType>(NodeType(*n.pred)))
    Error1(n.pred->LOC(), "requires a predication expression but got '" +
                              PSTR(NodeType(*n.pred)) + "'.");

  if (pl_depth == 0)
    Error1(n.pred->LOC(), "inthreads can not be declared in global scope.");

  if (n.async && !n.outer)
    Error1(n.pred->LOC(), "inner inthreads can not be declared as async.");

  if (!diverges.Contains(n.pred))
    Error1(n.pred->LOC(), "inthreads' predicate must be strictly divergent.");

  return true;
}

bool EarlySemantics::Visit(AST::WhileBlock& n) {
  TraceEachVisit(n);

  if (!isa<EventType>(NodeType(*n.pred)))
    Error1(n.pred->LOC(), "requires an event predication expression but got '" +
                              PSTR(NodeType(*n.pred)) + "'.");

  return true;
}

bool EarlySemantics::Visit(AST::IfElseBlock& n) {
  TraceEachVisit(n);
  if (isa<AST::Call>(n.pred)) return true; // can not derive function call

  if (!isa<BooleanType>(NodeType(*n.pred))) {
    Error1(n.pred->LOC(), "requires a predication expression but got '" +
                              PSTR(NodeType(*n.pred)) + "'.");
  }

  return true;
}

bool EarlySemantics::Visit(AST::IncrementBlock& n) {
  TraceEachVisit(n);
  for (auto& iv : n.GetIterationVars()) {
    auto ity = NodeType(*iv);
    if (!(isa<BoundedType>(ity)))
      Error1(n.LOC(), "expect a bounded type but got '" + PSTR(ity) + "'.");
    if (auto id = AST::GetIdentifier(*iv)) {
      if (id->name == "_")
        Error1(n.LOC(), "_ is not allowed as an iteration variable.");
    }
  }

  auto pty = NodeType(*n.GetPredicate());
  if (!isa<BooleanType>(pty))
    Error1(n.LOC(),
           "expect the a boolean-typed predicate but got '" + PSTR(pty) + "'.");

  return true;
}

bool EarlySemantics::Visit(AST::FunctionDecl& n) {
  TraceEachVisit(n);

  if (n.ret_type->IsVoid())
    requires_return = false;
  else if (n.ret_type->IsUnknown())
    return_deduction = true;
  else
    requires_return = true;

  return true;
}

bool EarlySemantics::Visit(AST::ChoreoFunction& n) {
  TraceEachVisit(n);
  return true;
}
bool EarlySemantics::Visit(AST::CppSourceCode& n) {
  TraceEachVisit(n);
  return true;
}

bool EarlySemantics::ParseTemplateParams(
    std::string input, std::vector<DeviceTemplateParam>& template_params) {

  auto trim = [](const std::string& str) {
    size_t first = str.find_first_not_of(" \t");
    size_t last = str.find_last_not_of(" \t");
    return (first == std::string::npos || last == std::string::npos)
               ? ""
               : str.substr(first, last - first + 1);
  };
  std::regex re(R"(template\s*<(.*)>)");
  std::smatch match;
  if (!std::regex_search(input, match, re) || match.size() < 2) {
    return false;
  }
  std::vector<std::string> params;
  std::string inner = match[1].str();

  std::string current;
  int depth = 0;
  for (char c : inner) {
    if (c == '<') {
      depth++;
      current += c;
    } else if (c == '>') {
      depth--;
      current += c;
    } else if (c == ',' && depth == 0) {
      params.push_back(trim(current));
      current.clear();
    } else {
      current += c;
    }
  }

  if (!current.empty()) { params.push_back(trim(current)); }
  if (params.empty()) {
    return false; // No valid template parameters found
  }
  for (auto param : params) {
    DeviceTemplateParam tp;
    std::regex type_re(R"((\w+)\s*(\w+)\s*(=\s*([^,>]+))?)");
    std::smatch type_match;
    if (std::regex_match(param, type_match, type_re)) {
      auto type_name = type_match[1].str();
      auto param_name = type_match[2].str();
      auto default_value = type_match[4].str();
      if (type_name.empty() || param_name.empty()) {
        template_params.push_back(tp);
        continue;
      }
      if (type_name == "typename" || type_name == "class") {
        tp.kind = DeviceTemplateParam::TYPE; // This is a type parameter
      } else {
        tp.kind = DeviceTemplateParam::VALUE; // This is a value parameter
      }
      tp.param_name = param_name;
      tp.type_name = type_name;
      tp.default_value = default_value;
    }
    template_params.push_back(tp);
  }
  return true;
}

bool EarlySemantics::Visit(AST::DeviceFunctionDecl& n) {
  TraceEachVisit(n);
  if (resolve_fns) {
    bool initized = false;
    bool change_type = false;
    for (size_t param_idx = 0; param_idx < n.param_types.size(); param_idx++) {
      auto param_ty = n.param_types[param_idx];
      if (param_ty->Initized())
        initized = true;
      else if (param_idx > 0 && initized) {
        Error1(n.LOC(), "Missing default argument on " +
                            std::to_string(param_idx + 1) + "th parameter of " +
                            "device function: " + STR(n));
        return false;
      }

      // if the parameter type is still UNKNOWN, we will try to resolve it
      auto plain_name = param_ty->PlainName();
      auto known_type = DSTR2BT(plain_name);
      if (known_type != BaseType::UNKNOWN && param_ty->pointer_count <= 1) {
        param_ty->SetDataType(known_type);
        change_type = true;
      }
    }

    if (change_type && debug_visit) dbgs() << "<resolved> " << STR(n) << "\n";

    // parse template parameters according to the template string
    if (!n.templates.empty()) {
      std::vector<DeviceTemplateParam> template_params;
      if (ParseTemplateParams(n.templates, template_params))
        n.template_params = template_params;
      else
        Warning(n.LOC(), "unparsed template parameters in device function: " +
                             n.templates);
    }
  }
  auto df = dyn_cast<AST::DeviceFunctionDecl>(n.Clone());
  device_functions.push_back(df);
  return true;
}

bool EarlySemantics::Visit(AST::Program& n) {
  TraceEachVisit(n);
  return true;
}

bool EarlySemantics::ReportErrorWhenUseBeforeDefine(const location& loc,
                                                    const std::string& name) {
  if (!SSTab().IsDeclared(name)) {
    Error1(loc, "symbol `" + name + "' is used before declaration.");
    return false;
  }
  return true;
}

bool EarlySemantics::ReportErrorWhenViolateODR(const location& loc,
                                               const std::string& name,
                                               const char* file, int line,
                                               const ptr<Type>& type) {
  if (SSTab().DeclaredInScope(name)) {
    Error1(loc, "symbol `" + name + "' has been declared already.");
    VST_DEBUG(dbgs() << "Error in " << file << ", line: " << line << ".\n");
    return false;
  }
  SSTab().DefineSymbol(name, type); // TODO: improve the type
  VST_DEBUG(dbgs() << "Define Symbol '" << InScopeName(name)
                   << "' as: " << PSTR(type) << ".\n");
  return true;
}
