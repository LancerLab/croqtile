#include "earlysema.hpp"

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
    parallel_level = 0;
  } else if (isa<AST::ParallelBy>(&n)) {
    parallel_level++;
  } else if (isa<AST::Parameter>(&n)) {
    in_decl = true;
    allow_named_dim = true; // tolerate repeated symbols inside mdspan params
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
      Error(n.LOC(), "non-void function '" + f->name +
                         "` does not contain a return statement.");
      error_count++;
    } else if (!requires_return && found_return) {
      Error(n.LOC(),
            "return statement found in void function '" + f->name + "`.");
      error_count++;
    }
  } else if (isa<AST::ParallelBy>(&n)) {
    assert(parallel_level > 0);
    parallel_level--;
  } else if (isa<AST::WithBlock>(&n)) {
    with_syms.clear();
  } else if (isa<AST::Parameter>(&n)) {
    in_decl = false;
    allow_named_dim = false;
  }

  return true;
}

bool EarlySemantics::Visit(AST::MultiNodes& n) {
  TraceEachVisit(n);
  return true;
}

bool EarlySemantics::Visit(AST::MultiValues& n) {
  TraceEachVisit(n);
  return true;
}

bool EarlySemantics::Visit(AST::IntLiteral& n) {
  TraceEachVisit(n);
  SetNodeType(n, MakeIntegerType());
  return true;
}

bool EarlySemantics::Visit(AST::Boolean& n) {
  TraceEachVisit(n);
  SetNodeType(n, MakeBooleanType());
  return true;
}

bool EarlySemantics::Visit(AST::Expr& n) {
  TraceEachVisit(n);
  if (auto ref = n.GetReference()) {
    auto rty = NodeType(*ref);
    assert(!isa<UnknownType>(rty) && "reference type is unknown.");
    SetNodeType(n, rty);
  } else if (n.op == "dataof") {
    auto ty = NodeType(*n.GetR());
    if (!isa<FutureType>(ty)) {
      Error(n.LOC(), "in operation \"" + n.op +
                         "\": expecting a future type but got `" + PSTR(ty) +
                         "'.");
      error_count++;
      return false;
    }
    SetNodeType(n, MakeDummySpannedType());
  } else if (n.op == "sizeof") {
    auto ty = NodeType(*n.GetR());
    if (!GetMDSpanType(ty)) {
      Error(n.LOC(), "in operation \"" + n.op +
                         "\": expecting a mdspan type but got `" + PSTR(ty) +
                         "'.");
      error_count++;
      return false;
    }
    SetNodeType(n, MakeIntegerType());
  } else if (n.op == "dimof") {
    auto lty = NodeType(*n.GetL());
    auto rty = NodeType(*n.GetR());
    if (!isa<MDSpanType>(lty) && !isa<ITupleType>(lty) && !IsBoundedType(lty)) {
      Error(n.LOC(), "in operation \"" + n.op +
                         "\": expecting a indexable type but got `" +
                         PSTR(lty) + "'.");
      error_count++;
      return false;
    }
    if (!isa<IndexType>(rty)) {
      Error(n.LOC(), "in operation \"" + n.op +
                         "\": expecting a index type but got `" + PSTR(rty) +
                         "'.");
      error_count++;
      return false;
    }
    if (isa<BoundedType>(lty)) {
      // disambiguite subscription into bounded ituple and getith of bounded
      // integer
      n.op = "getith";
      cast<AST::IntIndex>(n.GetR())->UseBracket();
      SetNodeType(n, lty);
    } else
      SetNodeType(n, MakeIntegerType());
  } else if (n.op == "ubound") {
    auto ty = NodeType(*n.GetR());
    if (!IsBoundedType(ty)) {
      Error(n.LOC(), "in operation \"" + n.op +
                         "\": expecting a bounded type but got `" + PSTR(ty) +
                         "'.");
      error_count++;
      return false;
    }
    SetNodeType(n, MakeIntegerType());
  } else if ((n.op == "+") || (n.op == "-") || (n.op == "*") || (n.op == "/") ||
             (n.op == "%") || (n.op == "cdiv")) {
    auto lty = NodeType(*n.GetL());
    auto rty = NodeType(*n.GetR());
    if ((isa<MDSpanType>(lty) && isa<ITupleType>(rty)) ||
        (isa<MDSpanType>(rty) && isa<ITupleType>(lty))) {
      // mdspan + ituple
      if (lty->Dims() != rty->Dims()) {
        Error(n.LOC(), "in operation \"" + n.op +
                           "\": dimension inconsistent (" +
                           std::to_string(lty->Dims()) + " vs. " +
                           std::to_string(rty->Dims()) + ").");
        error_count++;
        return false;
      }
      SetNodeType(n, MakeRankedMDSpanType(lty->Dims()));
    } else if ((isa<ITupleType>(lty) && isa<IntegerType>(rty)) ||
               (isa<MDSpanType>(lty) && isa<IntegerType>(rty))) {
      SetNodeType(n, lty);
    } else if ((isa<ITupleType>(rty) && isa<IntegerType>(lty)) ||
               (isa<MDSpanType>(rty) && isa<IntegerType>(lty))) {
      SetNodeType(n, rty);
    } else if ((isa<ITupleType>(lty) && isa<ITupleType>(rty))) {
      // ituple + ituple
      if (lty->HasSufficientInfo() && rty->HasSufficientInfo()) {
        if (lty->Dims() != rty->Dims()) {
          Error(n.LOC(), "in operation \"" + n.op +
                             "\": dimension inconsistent (" +
                             std::to_string(lty->Dims()) + " vs. " +
                             std::to_string(rty->Dims()) + ").");
          error_count++;
          return false;
        }
        SetNodeType(n, lty);
      } else
        SetNodeType(n, MakeUninitBoundedITupleType());
    } else if ((isa<BoundedType>(lty) && isa<BoundedType>(rty))) {
      Error(n.LOC(), "in operation \"" + n.op +
                         "\": unable to apply to the types (" + PSTR(lty) +
                         " vs. " + PSTR(rty) + ").");
      return false;
    } else if ((IsActualBoundedIntegerType(lty) && isa<IntegerType>(rty)) ||
               (IsActualBoundedIntegerType(rty) && isa<IntegerType>(lty))) {
      // this is promissing, simply allow it
      if (IsActualBoundedIntegerType(lty))
        if (cast<AST::Expr>(n.GetL())->op == "getith") {
          Error(n.LOC(),
                "in operation \"" + n.op +
                    "\": unable to apply to the getith bounded variable (" +
                    PSTR(n.GetL()) + ").");
          return false;
        } else
          SetNodeType(n, lty);
      else {
        if (cast<AST::Expr>(n.GetR())->op == "getith") {
          Error(n.LOC(),
                "in operation \"" + n.op +
                    "\": unable to apply to the 'getith' bounded variable (" +
                    PSTR(n.GetR()) + ").");
          return false;
        } else
          SetNodeType(n, rty);
      }
    } else if ((isa<BoundedITupleType>(lty) && isa<ITupleType>(rty)) ||
               (isa<BoundedITupleType>(rty) && isa<ITupleType>(lty))) {
      if (lty->Dims() != rty->Dims()) {
        Error(n.LOC(), "in operation \"" + n.op +
                           "\": dimension inconsistent (" +
                           std::to_string(lty->Dims()) + " vs. " +
                           std::to_string(rty->Dims()) + ").");
        error_count++;
        return false;
      }
      SetNodeType(n, MakeITupleType(lty->Dims()));
    } else if (isa<MDSpanType>(lty) && isa<MDSpanType>(rty)) {
      // only allow div/mod operations
      if ((n.op != "/") && (n.op != "%") && (n.op != "cdiv")) {
        Error(n.LOC(), "in operation \"" + n.op +
                           "\": unable to apply to the types (" + PSTR(lty) +
                           " vs. " + PSTR(rty) + ").");
        return false;
      }
      if (lty->Dims() != rty->Dims()) {
        Error(n.LOC(), "in operation \"" + n.op +
                           "\": dimension inconsistent (" +
                           std::to_string(lty->Dims()) + " vs. " +
                           std::to_string(rty->Dims()) + ").");
        error_count++;
        return false;
      }
      SetNodeType(n, MakeITupleType(lty->Dims()));
    } else if ((isa<MDSpanType>(lty) && isa<IntegerType>(rty)) ||
               (isa<MDSpanType>(rty) && isa<IntegerType>(lty)) ||
               (isa<ITupleType>(lty) && isa<IntegerType>(rty)) ||
               (isa<ITupleType>(rty) && isa<IntegerType>(lty))) {
      if (isa<MDSpanType>(lty) || isa<ITupleType>(lty))
        SetNodeType(n, lty);
      else
        SetNodeType(n, rty);
    } else if ((isa<ITupleType>(lty) && isa<ITupleType>(rty)) ||
               (isa<BooleanType>(lty) && isa<BooleanType>(rty)) ||
               (isa<IndexType>(lty) && isa<IndexType>(rty))) {
      Error(n.LOC(), "in operation \"" + n.op +
                         "\": unable to apply to the types (" + PSTR(lty) +
                         " vs. " + PSTR(rty) + ").");
      error_count++;
      return false;
    } else if ((isa<IntegerType>(lty) && isa<IndexType>(rty)) ||
               (isa<IntegerType>(rty) && isa<IndexType>(lty))) {
      // when desugaring of ituple/mdspan has not been applied, we have to deal
      // with nodes like:
      //   a {1 + (1)}
      SetNodeType(n, lty);
    } else if (!lty->ApprxEqual(*rty)) {
      Error(n.LOC(), "in operation \"" + n.op +
                         "\": unable to apply to the types (" + PSTR(lty) +
                         " vs. " + PSTR(rty) + ").");
      error_count++;
      return false;
    } else
      SetNodeType(n, lty);
  } else if (n.op == "#") {
    // allow only # operator for catesian products on two bounded-vars
    // a # b => a * (#b) + b
    auto lty = NodeType(*n.GetL());
    auto rty = NodeType(*n.GetR());
    if (IsActualBoundedIntegerType(lty) && IsActualBoundedIntegerType(rty)) {
      SetNodeType(n, MakeBoundedITupleType(Shape(1)));
    } else {
      // TODO: computation of multi-dim bounded vars is not supported yet.
      Error(n.LOC(), "in operation \"" + n.op +
                         "\": unable to apply to the types (" + PSTR(lty) +
                         " vs. " + PSTR(rty) + ").");
      return false;
    }
  } else if ((n.op == "<") || (n.op == ">") || (n.op == "==") ||
             (n.op == "!=") || (n.op == "<=") || (n.op == ">=")) {
    auto lty = NodeType(*n.GetL());
    auto rty = NodeType(*n.GetR());
    // assert(false);
    // only support IntegerType currently
    assert(isa<IntegerType>(lty) && isa<IntegerType>(rty));
    if (!(lty->ApprxEqual(*rty))) {
      Error(n.LOC(), "in operation \"" + n.op +
                         "\": unable to apply to the types (" + PSTR(lty) +
                         " vs. " + PSTR(rty) + ").");
      error_count++;
      return false;
    }
    SetNodeType(n, MakeBooleanType());
  } else if ((n.op == "&&") || (n.op == "||")) {
    auto lty = NodeType(*n.GetL());
    auto rty = NodeType(*n.GetR());
    assert(false);
    if (!isa<BooleanType>(lty) || !isa<BooleanType>(rty)) {
      Error(n.LOC(), "in operation \"" + n.op +
                         "\": unable to apply to the types (" + PSTR(lty) +
                         " vs. " + PSTR(rty) + ").");
      error_count++;
      return false;
    }
    SetNodeType(n, MakeBooleanType());
  } else if (n.op == "!") {
    auto rty = NodeType(*n.GetR());
    if (!isa<BooleanType>(rty)) { // TODO: will we allow integer?
      Error(n.LOC(), "in operation \"" + n.op +
                         "\": unable to apply to the type (" + PSTR(rty) +
                         ").");
      error_count++;
      return false;
    }
    SetNodeType(n, MakeBooleanType());
  } else if (n.op == "?") {
    auto cty = NodeType(*n.GetC());
    auto lty = NodeType(*n.GetL());
    auto rty = NodeType(*n.GetR());
    if (!isa<BooleanType>(cty) || (!lty->ApprxEqual(*rty))) {
      Error(n.LOC(), "in operation \"" + n.op +
                         "\": unable to apply to the types (" + PSTR(cty) +
                         ") " + PSTR(lty) + " : " + PSTR(rty) + ").");
      error_count++;
      return false;
    }
    SetNodeType(n, lty);
  } else if (n.op == "concat") {
    auto lty = NodeType(*n.GetL());
    auto rty = NodeType(*n.GetR());
    if (!((isa<MDSpanType>(lty) || isa<ITupleType>(lty)) &&
          (isa<MDSpanType>(rty) || isa<ITupleType>(rty)))) {
      Error(n.LOC(), "in operation \"" + n.op +
                         "\": unable to apply to the types " + PSTR(lty) +
                         " and " + PSTR(rty) + ".");
      error_count++;
      return false;
    }
    if ((!IsValidRank(lty->Dims())) || (!IsValidRank(rty->Dims())))
      SetNodeType(n, MakeUninitMDSpanType());
    else
      SetNodeType(n, MakeRankedMDSpanType(lty->Dims() + rty->Dims()));
  } else
    choreo_unreachable("operation in expression is not supported yet.");
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
        n.list->accept(*this); // go evaluate the concatanation
      }
    }
  }

  // check the elements type
  if (auto mvals = dyn_cast<AST::MultiValues>(n.list)) {
    if (n.ref_name.empty()) {
      for (auto& v : mvals->AllValues()) {
        auto ty = NodeType(*v);
        if (!isa<IntegerType>(ty) && !isa<MDSpanType>(ty) &&
            !isa<ITupleType>(ty)) {
          Error(v->LOC(),
                "unexpected data type '" + PSTR(ty) + "' is found in mdspan.");
          error_count++;
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
      if (auto il = dyn_cast<AST::IntLiteral>(v))
        if (il->value <= 0 && il->value != GetUnKnownInteger()) {
          Error(v->LOC(), "The mdspan size \"" + std::to_string(il->value) +
                              "\" is invalid!");
          error_count++;
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
    Error(n.LOC(), "`" + n.name_str + "' is declared as \"" + PSTR(nty) +
                       "\" but initialized as \"" + PSTR(ety) + "\".");
    error_count++;
    // keep processing
  }
  ReportErrorWhenViolateODR(n.LOC(), n.name_str, __FILE__, __LINE__, ety);
  SetNodeType(n, ety);
  return true;
}

bool EarlySemantics::Visit(AST::NamedVariableDecl& n) {
  TraceEachVisit(n);

  ptr<Type> tty = nullptr;
  ptr<Type> ety = nullptr;
  if (n.type) tty = n.type->GetType();
  if (n.init_expr) {
    ety = n.init_expr->GetType();
    assert(!isa<UnknownType>(ety) && "no type for an init expression.");
  }

  if (!ety) {
    // in this case, the type is deduced from type annotation
    if (isa<UnknownType>(tty)) {
      Error(n.LOC(), "unable to deduce the type of `" + n.name_str + "'.");
      error_count++;
      return false;
    }
    ReportErrorWhenViolateODR(n.LOC(), n.name_str, __FILE__, __LINE__, tty);
    SetNodeType(n, tty);
  } else {
    // in this case, the type is deduced from initialize expression
    if (isa<MDSpanType>(ety)) {
      Error(n.LOC(), "use ':' instead of '=' to define the \"" + PSTR(ety) +
                         "\" type variable.");
      error_count++;
      if (debug_visit)
        dbgs() << "Error in " << __FILE__ << ", line: " << __LINE__ << ".\n";
      // keep working
    } else if (isa<PlaceHolderType>(ety) && n.init_expr &&
               isa<AST::Expr>(n.init_expr) &&
               cast<AST::Expr>(n.init_expr)->GetSymbol()) {
      // forbid to directly initialize a placeholder with a placeholder
      Error(n.LOC(), "can not initialize vairable `" + n.name_str +
                         "' with a placeholder.");
      error_count++;
      if (debug_visit)
        dbgs() << "Error in " << __FILE__ << ", line: " << __LINE__ << ".\n";
    }
    // check for type consistency between annotation and init expr.
    if (!isa<UnknownType>(tty) && !tty->ApprxEqual(*ety)) {
      Error(n.LOC(), "`" + n.name_str + "' is declared as \"" +
                         PSTR(n.type->GetType()) + "\" but initialized as \"" +
                         PSTR(n.init_expr->GetType()) + "\".");
      error_count++;
      // keep working
    }
    ReportErrorWhenViolateODR(n.LOC(), n.name_str, __FILE__, __LINE__, ety);
    SetNodeType(n, ety);
    // also set the type of type annotation
    if (tty && isa<UnknownType>(tty)) SetNodeType(*n.type, ety);
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

  return true;
}

bool EarlySemantics::Visit(AST::IntTuple& n) {
  TraceEachVisit(n);
  SetNodeType(n, MakeITupleType(n.GetValues()->Count()));
  return true;
}

bool EarlySemantics::Visit(AST::Assignment& n) {
  TraceEachVisit(n);

  if (!SSTab().DeclaredInScope(n.name)) {
    // This is a definition rather than an assignment. The parser fails to make
    // it correct
    auto sty = NodeType(*n.value);
    assert((sty && !isa<UnknownType>(sty)) &&
           "internal error: failed to find the type.");
    if (isa<MDSpanType>(sty)) {
      Error(n.LOC(),
            "use ':' to define the \"" + STR(*sty) + "\" type variable.");
      ++error_count;
      if (debug_visit)
        dbgs() << "Error in " << __FILE__ << ", line: " << __LINE__ << ".\n";
      return false;
    }
    ReportErrorWhenViolateODR(n.LOC(), n.name, __FILE__, __LINE__,
                              ShadowTypeStorage(sty));
    if (auto ty = dyn_cast<SpannedType>(sty)) {
      ReportErrorWhenViolateODR(n.LOC(), n.name + ".span", __FILE__, __LINE__,
                                MakeRankedMDSpanType(ty->Dims()));
    }
    if (auto ty = dyn_cast<FutureType>(sty)) {
      ReportErrorWhenViolateODR(n.LOC(), n.name + ".span", __FILE__, __LINE__,
                                MakeRankedMDSpanType(ty->Dims()));
      ReportErrorWhenViolateODR(n.LOC(), n.name + ".data", __FILE__, __LINE__,
                                ShadowTypeStorage(ty->GetSpannedType()));
    }
    return true;
  }

  auto vty = SSTab().LookupSymbol(n.name); // variable type
  auto ety = NodeType(*n.value);           // assignment expression type

  // placeholder can be reassigned
  if (isa<PlaceHolderType>(vty)) {
    assert(GeneralFutureType(vty));

    // check for type consistent
    if (!vty->ApprxEqual(*ety)) {
      Error(n.LOC(), "`" + n.name + "' of type '" + STR(*vty) +
                         "' is assigned as " + STR(*ety) + ".");
      ++error_count;
      if (debug_visit)
        dbgs() << "Error in " << __FILE__ << ", line: " << __LINE__ << ".\n";
      return false;
    }

    if (isa<FutureType>(ety)) {
      ModifySymbolType(n.name, ety);
      ModifySymbolType(n.name + ".span", MakeRankedMDSpanType(ety->Dims()));
    } else
      choreo_unreachable("Expect a future type but got '" + PSTR(ety) + "'.");
  }

  // For now, we have to keep the single assignment
  {
    if (vty->ApprxEqual(*ety))
      Error(n.LOC(), ToUpper(vty->Name()) + " re-assignment (" + n.name +
                         ") is not supported.");
    else
      Error(n.LOC(), "`" + n.name + "' of type \"" + STR(*vty) +
                         "\" can not be re-assigned as \"" + STR(*ety) + "\".");
    ++error_count;
    if (debug_visit)
      dbgs() << "Error in " << __FILE__ << ", line: " << __LINE__ << ".\n";
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

  // sema type has been generated at construction ast. refine with dims
  if (isa<SpannedType>(n.GetType())) {
    if (auto sty = dyn_cast<MDSpanType>(n.mdspan_type->GetType())) {
      SetNodeType(n, MakeRankedSpannedType(sty->Dims(), n.base_type));
    }
    return true;
  }
  return true;
}

bool EarlySemantics::Visit(AST::Identifier& n) {
  TraceEachVisit(n);
  if (in_decl) {
    if (allow_named_dim) {
      if (!SSTab().DeclaredInScope(n.name))
        SSTab().DefineSymbol(n.name,
                             MakeIntegerType()); // named dim is integer
    } else
      ReportErrorWhenViolateODR(n.LOC(), n.name, __FILE__, __LINE__);
  } else {
    // to avoid the expected error
    if (n.name == "__choreo_tile_one" && !SSTab().DeclaredInScope(n.name)) {
      auto bit = MakeBoundedIntegerType(1);
      SSTab().DefineSymbol(n.name, bit);
      SSTab().DefineSymbol("@" + n.name, MakeIntegerType());
    }
    ReportErrorWhenUseBeforeDefine(n.LOC(), n.name);
  }
  return true;
}

bool EarlySemantics::Visit(AST::Parameter& n) {
  TraceEachVisit(n);
  if (n.sym) {
    ModifySymbolType(n.sym->name, n.type->GetType());
    if (auto ty = dyn_cast<SpannedType>(n.type->GetType())) {
      SSTab().DefineSymbol(n.sym->name + ".span", ty->GetMDSpanType());
    }
  }
  return true;
}

bool EarlySemantics::Visit(AST::ParamList& n) {
  TraceEachVisit(n);
  return true;
}

bool EarlySemantics::Visit(AST::ParallelBy& n) {
  TraceEachVisit(n);
  ReportErrorWhenViolateODR(n.LOC(), n.biv, __FILE__, __LINE__,
                            MakeBoundedITupleType(Shape(1, n.biv), "pv"));
  return true;
}

bool EarlySemantics::Visit(AST::WhereBind& n) {
  TraceEachVisit(n);
  if (!isa<AST::Identifier>(n.lhs)) {
    Error(n.lhs->LOC(), "expecting an indentifier.");
    error_count++;
    return false;
  }
  if (!isa<AST::Identifier>(n.rhs)) {
    Error(n.rhs->LOC(), "expecting an indentifier.");
    error_count++;
    return false;
  }
  auto lname = cast<AST::Identifier>(n.lhs)->name;
  auto rname = cast<AST::Identifier>(n.rhs)->name;
  if (with_syms.count(lname) == 0) {
    Error(n.lhs->LOC(),
          "symbol `" + lname + "' is not defined inside the with statement.");
    error_count++;
  }
  if (with_syms.count(rname) == 0) {
    Error(n.rhs->LOC(),
          "symbol `" + rname + "' is not defined inside the with statement.");
    error_count++;
  }
  return true;
}

bool EarlySemantics::Visit(AST::WithIn& n) {
  TraceEachVisit(n);
  in_decl = true;

  auto ity = NodeType(*n.in);
  if (!isa<MDSpanType>(ity)) {
    Error(n.in->LOC(),
          "expecting a span type but got the " + PSTR(ity) + " type.");
    error_count++;
  }

  // check the if rank equal between with-in and with-matcher
  if (n.with_matchers &&
      n.with_matchers->Count() != cast<MDSpanType>(ity)->Dims()) {
    Error(n.in->LOC(),
          "un-matched with-matcher-count(" +
              std::to_string(n.with_matchers->Count()) + ") and mdspan rank(" +
              std::to_string(cast<MDSpanType>(ity)->Dims()) + ").");
    error_count++;
  }

  if (n.with && n.with->name == "_") {
    Error(n.LOC(),
          "_ is not allowed as a with variable. Can only be used in chunkat.");
    error_count++;
  }

  if (n.with_matchers) {
    for (auto v : n.with_matchers->AllValues()) {
      if (auto id = dyn_cast<AST::Identifier>(v); id->name == "_") {
        Error(v->LOC(), "_ is not allowed as a with variable. Can only be used "
                        "in chunkat.");
        error_count++;
        continue;
      }
    }
  }

  // infer the type of bounded variable
  if (n.with) {
    n.with->accept(*this); // make the symbol be defined
    with_syms.insert(n.with->name);
    auto wty = MakeBoundedITupleType(Shape(cast<MDSpanType>(ity)->Dims()));
    ModifySymbolType(n.with->name, wty);
    SetNodeType(*n.with, wty);
  }

  if (n.with_matchers) {
    n.with_matchers->accept(*this); // make the symbol be defined
    for (auto v : n.with_matchers->AllValues()) {
      // only id are accepted in with-matcher
      if (!isa<AST::Identifier>(v)) {
        Error(v->LOC(), "expecting an identifier.");
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

  auto sty = GetSpannedType(NodeType(*n.id));
  if (!sty) {
    Error(n.LOC(), "span-as operation operates on a non-mdspan type.");
    error_count++;
    return false;
  }

  auto asty = MakeRankedSpannedType(n.list->Count(), (BaseType)sty->f_type,
                                    sty->m_type);
  SSTab().DefineSymbol(n.nid->name, asty);
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
  if (!isa<AST::Memory>(n.to)) tty = cast<SpannedType>(NodeType(*n.to));

  if (!n.future.empty()) {
    size_t rank = sty->Dims();
    assert(IsValidRank(rank));

    Storage sto = Storage::NONE;
    if (auto m = dyn_cast<AST::Memory>(n.to))
      sto = m->Get();
    else
      sto = tty->GetStorage();

    // rewrite the placeholder type
    if (SSTab().IsDeclared(n.future)) {
      ReportErrorWhenUseBeforeDefine(n.LOC(), n.future + ".span");
      ReportErrorWhenUseBeforeDefine(n.LOC(), n.future + ".data");
      ModifySymbolType(n.future + ".span", MakeRankedMDSpanType(rank));
      auto spanned_ty = MakeRankedSpannedType(rank, sty->ElementType(), sto);
      ModifySymbolType(n.future + ".data", spanned_ty);
      ModifySymbolType(n.future, MakeFutureType(spanned_ty, n.async));
    } else {
      ReportErrorWhenViolateODR(n.LOC(), n.future + ".span", __FILE__, __LINE__,
                                MakeRankedMDSpanType(rank));
      auto spanned_ty = MakeRankedSpannedType(rank, sty->ElementType(), sto);
      ReportErrorWhenViolateODR(n.LOC(), n.future + ".data", __FILE__, __LINE__,
                                spanned_ty);
      ReportErrorWhenViolateODR(n.LOC(), n.future, __FILE__, __LINE__,
                                MakeFutureType(spanned_ty, n.async));
    }

    // set the buffer kind
    auto from_kind = (cast<AST::ChunkAt>(n.from)->SymbolicBufferName())
                         ? DOK_SYMBOL
                         : DOK_CHUNK;
    auto to_kind = DOK_UNKNOWN;
    if (!isa<AST::ChunkAt>(n.to))
      to_kind = DOK_SYMBOL;
    else
      to_kind = (cast<AST::ChunkAt>(n.to)->SymbolicBufferName()) ? DOK_SYMBOL
                                                                 : DOK_CHUNK;
    auto to_sym = n.ToSymbol();
    if (!to_sym.empty()) to_sym = InScopeName(to_sym);
    FCtx(fname).GetFutureBufferInfo().emplace(
        InScopeName(n.future), DMABufferInfo{to_sym, from_kind, to_kind});
  } else {
    if (n.async) {
      Error(n.LOC(), "forbid to associated async dma without a named future.");
      error_count++;
    }
  }

  if (!isa<AST::Memory>(n.to)) {
    if (sty->Dims() != tty->Dims() && !allow_auto_threading) {
      Error(n.LOC(),
            "The DMA statement contains a rank mismatch: the 'from' and 'to' "
            "arrays have inconsistent dimensions.");
      error_count++;
    } else if (sty->ElementType() != tty->ElementType()) {
      Error(n.LOC(),
            "The DMA statement contains a type mismatch: the element types of "
            "the 'from'(" +
                STR(sty->ElementType()) + ") and 'to'(" +
                STR(tty->ElementType()) + ") arrays are inconsistent.");
      error_count++;
    }
  }

  // dma.pad specific check
  if (auto pcfg = dyn_cast<PadConfig>(n.config)) {
    if (!((pcfg->pad_high.size() == pcfg->pad_low.size()) &&
          (pcfg->pad_low.size() == pcfg->pad_mid.size()))) {
      Error(n.LOC(),
            "The DMA statement contains a rank mismatch: the paddings have "
            "inconsistent ranks.");
      error_count++;
    } else if (NodeType(*n.from)->Dims() != pcfg->pad_high.size()) {
      Error(n.LOC(),
            "The rank of the data to transfer is inconsistent with the DMA "
            "padding settings.");
      error_count++;
    }
  }

  // dma.transp specific check
  if (auto tcfg = dyn_cast<TransposeConfig>(n.config)) {
    auto dim_values = tcfg->dim_values;
    if (dim_values.size() != sty->Dims()) {
      Error(n.LOC(),
            "The DMA statement contains a rank mismatch: the 'transpose "
            "layout' and 'from' arrays have inconsistent dimensions.");
      error_count++;
    }
    if (!isa<AST::Memory>(n.to)) {
      if (dim_values.size() != sty->Dims()) {
        Error(n.LOC(),
              "The DMA statement contains a rank mismatch: the 'transpose "
              "layout' and 'to' arrays have inconsistent dimensions.");
        error_count++;
      }
    }
    std::sort(dim_values.begin(), dim_values.end());
    for (size_t i = 0; i < dim_values.size(); ++i) {
      if (dim_values[i] != i) {
        Error(n.LOC(), "The DMA statement contains an error: the transpose "
                       "layout is invalid.");
        error_count++;
        break;
      }
    }
  }

  return true;
}

bool EarlySemantics::Visit(AST::ChunkAt& n) {
  TraceEachVisit(n);

  if (n.positions)
    for (auto& v : n.positions->AllValues())
      if (auto expr = cast<AST::Expr>(v); expr->IsReference())
        if (auto id = dyn_cast<AST::Identifier>(expr->GetReference()))
          if (id->name == "_") id->name = "__choreo_tile_one";

  n.data->accept(*this);
  auto nty = NodeType(*n.data);
  if (n.sa) nty = NodeType(*n.sa);

  if (!isa<SpannedType>(nty) && !isa<FutureType>(nty)) {
    Error(n.LOC(),
          "expecting '" + n.data->name + "` of a spanned data or future type.");
    error_count++;
  }

  auto sty = GetSpannedType(nty);

  if (n.positions) {
    n.positions->accept(*this);
    size_t rank = sty->Dims();
    size_t r_count = 0;
    for (auto& v : n.positions->AllValues()) {
      auto ty = NodeType(*v);
      if (!IsBoundedType(ty)) {
        Error(n.LOC(), "expecting '" + PSTR(v) +
                           "` be a bounded type (but got " + PSTR(ty) + ").");
        error_count++;
      }
      r_count += ty->Dims();
      SetNodeType(*v, ty);
    }
    // report error when the ranks do not match
    if (rank != r_count) {
      Error(n.LOC(), "un-matched ranks between spanned data (" +
                         std::to_string(rank) + ") and bounded variables (" +
                         std::to_string(r_count) + ").");
      error_count++;
    }
  }
  assert(IsValidRank(nty->Dims()));
  SetNodeType(n, MakeRankedSpannedType(nty->Dims(), sty->ElementType(),
                                       sty->GetStorage()));
  return true;
}

bool EarlySemantics::Visit(AST::Wait& n) {
  TraceEachVisit(n);

  for (auto& v : n.targets->AllValues()) {
    auto id = dyn_cast<AST::Identifier>(v);
    if (!id) Error(n.LOC(), "expecting symbol but got '" + AST::STR(*v));

    auto ty = NodeType(*v);

    if (auto fty = dyn_cast<FutureType>(ty)) {
      if (!fty->IsAsync()) {
        Error(n.LOC(),
              "non-async future '" + id->name + "` can not be waited.");
        error_count++;
      }
      continue;
    } else if (auto pty = dyn_cast<PlaceHolderType>(ty)) {
      if (pty->Category() != TypeCategory::FUTURE) {
        Error(n.LOC(), "'" + id->name + "` of type \"" + PSTR(ty) +
                           "\" can not be waited.");
        error_count++;
      }
    } else {
      Error(n.LOC(), "'" + id->name + "` of type \"" + PSTR(ty) +
                         "\" can not be waited.");
      error_count++;
    }
  }
  return true;
}

bool EarlySemantics::Visit(AST::Call& n) {
  TraceEachVisit(n);

  if (parallel_level == 0) {
    Error(n.LOC(),
          "unable to call kernel function outside the parallel-by block(s).");
    error_count++;
    return false;
  }

  size_t count = 0;
  for (auto& v : n.arguments->AllValues()) {
    count++;
    auto ty = NodeType(*v);
    // must be a callable type
    if (!CanYieldAnInteger(ty) && !isa<SpannedType>(ty)) {
      Error(n.LOC(), "(" + std::to_string(count) + "th) argument of type '" +
                         PSTR(ty) +
                         "` can not be passed to the kernel function.");
      error_count++;
    }
  }

  if (n.template_args) {
    size_t count = 0;
    for (auto& v : n.template_args->AllValues()) {
      count++;
      auto ty = NodeType(*v);
      // must be a scalar type
      if (!ConvertibleToInt(ty)) {
        Error(n.LOC(),
              "(" + std::to_string(count) + "th) template argument of type '" +
                  PSTR(ty) +
                  "` can not be used to instantiate the kernel function.");
        error_count++;
      }
    }
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
      Error(n.LOC(), "only support swapping of 'future'. (" +
                         n.IdAt(index)->name + ": " + PSTR(lty) + ").");
      error_count++;
      return false;
    }

    // Avoid to swap a 'chunkat' target where no explicit buffer symbol is
    // associated.
    if (FCtx(fname).GetFutureBufferInfo()[InScopeName(cname)].to_kind ==
        DOK_CHUNK) {
      Error(n.LOC(), "rotate/swap a 'future' referring a buffer chunk has not "
                     "been supported yet.");
      error_count++;
      return false;
    }

    if (index < 1) continue;
    lty = NodeType(*n.ValueAt(index - 1));

    if (!lty->ApprxEqual(*cty)) {
      Error(n.LOC(), "rotate/swap data of different types (" + PSTR(lty) +
                         " vs. " + PSTR(cty));
      error_count++;
      return false;
    }
  }

  auto fty = type_equals.ResolveEqualFutures(*n.ids);

  if (!fty) {
    Error(n.LOC(), "Fail to resolve types for swap/rotate.");
    error_count++;
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

  // TODO(wsj) isa<IntegerType>(rty)?
  if (!isa<BoundedIntegerType>(NodeType(*n.select_factor)) &&
      !isa<IntegerType>(NodeType(*n.select_factor))) {
    Error(n.LOC(), "expect `" + PSTR(n.select_factor) +
                       "` to be a (bounded) integer type.");
    error_count++;
  }

  // TODO(wsj) assert bound <= span_val_list.count ?

  // check value types in val_list are the same
  assert(n.expr_list->Count() > 0);
  const auto& v0 = n.expr_list->AllValues()[0];
  auto v0ty = NodeType(*v0);

  if (!GeneralFutureType(v0ty) && !isa<SpannedType>(v0ty)) {
    Error(v0->LOC(),
          "expect `" + PSTR(v0ty) + "` to be a future/spanned type.");
    error_count++;
    return ec == error_count;
  }

  ptr<Type> sel_fty = nullptr;
  for (auto& v : n.expr_list->AllValues()) {
    auto nty = NodeType(*v);
    if (!nty->ApprxEqual(*v0ty)) {
      Error(v->LOC(), "expect `" + PSTR(v) + "`(" + PSTR(nty) +
                          ") to be the same type as `" + PSTR(v0) + "`(" +
                          PSTR(v0ty) + ").");
      error_count++;
    }

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

  return true;
}

bool EarlySemantics::Visit(AST::Return& n) {
  TraceEachVisit(n);
  found_return = true;
  if (parallel_level != 0) {
    Error(n.LOC(), "unable to return inside the parallel-by block(s).");
    error_count++;
    return false;
  }

  if (n.value) {
    auto vty = NodeType(*n.value);
    if (!(isa<SpannedType>(vty) || isa<ScalarType>(vty))) {
      Error(n.LOC(),
            "returning value with type '" + PSTR(vty) + "' is not supproted.");
      error_count++;
      return false;
    }
  }

  return true;
}

bool EarlySemantics::Visit(AST::LoopRange& n) {
  TraceEachVisit(n);

  return true;
}

bool EarlySemantics::Visit(AST::ForeachBlock& n) {
  TraceEachVisit(n);
  for (auto& i : n.GetRanges()) {
    if (auto id = dyn_cast<AST::LoopRange>(i)->iv) {
      if (id->name == "_") {
        Error(n.LOC(), "_ is not allowed as an iteration variable.");
        error_count++;
        continue;
      }
      auto ity = NodeType(*id);
      if (!(IsBoundedType(ity))) {
        Error(n.LOC(), "expecting a bounded type for iteration variable '" +
                           id->name + "' but got '" + PSTR(ity) + "'.");
        error_count++;
      }
    } else {
      auto ity = i->GetType();
      if (!(IsBoundedType(ity))) {
        Error(n.LOC(), "expecting a bounded type but got '" + PSTR(ity) + "'.");
        error_count++;
      }
    }
  }
  return true;
}

bool EarlySemantics::Visit(AST::IncrementBlock& n) {
  TraceEachVisit(n);
  for (auto& iv : n.GetIterationVars()) {
    auto ity = NodeType(*iv);
    if (!(IsBoundedType(ity))) {
      Error(n.LOC(), "expect a bounded type but got '" + PSTR(ity) + "'.");
      error_count++;
    }
    if (auto id = AST::GetIdentifier(*iv)) {
      if (id->name == "_") {
        Error(n.LOC(), "_ is not allowed as an iteration variable.");
        error_count++;
      }
    }
  }

  auto pty = NodeType(*n.GetPredicate());
  if (!isa<BooleanType>(pty)) {
    Error(n.LOC(),
          "expect the a boolean-typed predicate but got '" + PSTR(pty) + "'.");
    error_count++;
  }

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
bool EarlySemantics::Visit(AST::Program& n) {
  TraceEachVisit(n);
  return true;
}

bool EarlySemantics::ReportErrorWhenUseBeforeDefine(const location& loc,
                                                    const std::string& name) {
  if (!SSTab().IsDeclared(name)) {
    Error(loc, "symbol `" + name + "' is used before declaration.");
    ++error_count;
    return false;
  }
  return true;
}

bool EarlySemantics::ReportErrorWhenViolateODR(const location& loc,
                                               const std::string& name,
                                               const char* file, int line,
                                               const ptr<Type>& type) {
  if (SSTab().DeclaredInScope(name)) {
    Error(loc, "symbol `" + name + "' has been declared already.");
    ++error_count;
    if (debug_visit)
      dbgs() << "Error in " << file << ", line: " << line << ".\n";
    return false;
  }
  SSTab().DefineSymbol(name, type); // TODO: improve the type
  if (debug_visit)
    dbgs() << "Define Symbol '" << name << "' as: " << PSTR(type) << ".\n";
  return true;
}

bool EarlySemantics::HasError() {
  if (error_count > 0) {
    dbgs() << "Totally " << error_count << " errors have been detected.\n";
    return true;
  }
  return false;
}
