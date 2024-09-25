#include "earlysema.hpp"

using namespace Choreo;

#define __TRACE_EACH_VISIT__(n)       \
  if (trace_visit) {                  \
    os << n.TypeNameString() << ": "; \
    os << "\n";                       \
  }

bool EarlySemantics::BeforeVisit(AST::Node& n) {
  if (isa<AST::Program>(&n)) {
    SSTab().EnterScope("");  // global scope
  } else if (auto f = dyn_cast<AST::ChoreoFunction>(&n)) {
    SSTab().EnterScope(f->name);
    requires_return = false;
    return_deduction = false;
    found_return = false;
    parallel_level = 0;
  } else if (isa<AST::ParallelBy>(&n)) {
    static size_t count = 0;
    SSTab().EnterScope("paraby_" + std::to_string(count++));
    parallel_level++;
  } else if (isa<AST::WithBlock>(&n)) {
    static size_t count = 0;
    SSTab().EnterScope("within_" + std::to_string(count++));
  } else if (isa<AST::ForeachBlock>(&n)) {
    static size_t count = 0;
    SSTab().EnterScope("foreach_" + std::to_string(count++));
  }

  if (isa<AST::Parameter>(&n)) {
    in_decl = true;
    allow_named_dim = true;  // tolerate repeated symbols inside mdspan params
  }

  return true;
}

bool EarlySemantics::AfterVisit(AST::Node& n) {
  if (isa<AST::Program>(&n) || isa<AST::ChoreoFunction>(&n) ||
      isa<AST::ParallelBy>(&n) || isa<AST::WithBlock>(&n) ||
      isa<AST::ForeachBlock>(&n)) {
    SSTab().LeaveScope();
  }

  if (auto f = dyn_cast<AST::ChoreoFunction>(&n)) {
    if (return_deduction) {
      // maybe this can be moved to type inference
      if (!found_return && f->f_decl.ret_type->IsUnknown()) {
        f->f_decl.ret_type->base_type = BaseType::VOID;
        f->f_decl.ret_type->SetType(MakeVoidType());
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
  }

  if (isa<AST::Parameter>(&n)) {
    in_decl = false;
    allow_named_dim = false;
  }
  return true;
}

bool EarlySemantics::Visit(AST::MultiNodes& n) {
  __TRACE_EACH_VISIT__(n)
  return true;
}

bool EarlySemantics::Visit(AST::MultiValues& n) {
  __TRACE_EACH_VISIT__(n)
  return true;
}

bool EarlySemantics::Visit(AST::IntLiteral& n) {
  __TRACE_EACH_VISIT__(n)
  SetNodeType(n, MakeIntegerType());
  return true;
}

bool EarlySemantics::Visit(AST::Boolean& n) {
  __TRACE_EACH_VISIT__(n)
  SetNodeType(n, MakeBooleanType());
  return true;
}

bool EarlySemantics::Visit(AST::Expr& n) {
  __TRACE_EACH_VISIT__(n)
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
    if (!isa<MDSpanType>(ty)) {
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
    } else if ((isa<BoundedITupleType>(lty) && isa<BoundedIntegerType>(rty)) ||
               (isa<BoundedIntegerType>(lty) && isa<BoundedITupleType>(rty)) ||
               (isa<BoundedIntegerType>(lty) && isa<BoundedIntegerType>(rty))) {
      // allow only * operator for catesian products on two bounded-vars
      // currently, only support boundedituple * boundedint or boundedint *
      // boundedint os << STR(n.GetL()) << "lty = " << PSTR(lty) << "; rty = "
      // << PSTR(rty);
      if ((n.op != "*")) {
        Error(n.LOC(), "in operation \"" + n.op +
                           "\": unable to apply to the types (" + PSTR(lty) +
                           " vs. " + PSTR(rty) + ").");
        return false;
      }
    } else if ((isa<BoundedIntegerType>(lty) && isa<IntegerType>(rty)) ||
               (isa<BoundedIntegerType>(rty) && isa<IntegerType>(lty))) {
      // this is promissing, simply allow it
      SetNodeType(n, MakeUnknownBoundedIntegerType());
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
    if (!isa<BooleanType>(rty)) {  // TODO: will we allow integer?
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
  __TRACE_EACH_VISIT__(n)
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
          wl.push_back(nullptr);  // accepting new values
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

      if (wl.size() > 1) {  // mdspan inside
        ptr<AST::Node> last = wl[0];
        for (size_t i = 1; i < wl.size(); ++i) {
          auto concat =
              AST::Make<AST::Expr>(last->LOC(), "concat", last, wl[i]);
          last = concat;
        }
        if (trace_visit)
          os << "Transform: " << PSTR(n.list) << " to be " << PSTR(last)
             << "\n";
        n.list = last;
        n.list->accept(*this);  // go evaluate the concatanation
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
          Error(v->LOC(), "unexpected data type '" + PSTR(v->GetType()) +
                              "' is found in mdspan.");
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
      if (trace_visit)
        os << "Warning in " << __FILE__ << ", line: " << __LINE__ << ".\n";
      n.SetRank(rank);
    }
  }

  SetNodeType(n, MakeRankedMDSpanType(rank));
  return true;
}

bool EarlySemantics::Visit(AST::NamedTypeDecl& n) {
  __TRACE_EACH_VISIT__(n)
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
  __TRACE_EACH_VISIT__(n)
  if (isa<UnknownType>(n.type->GetType())) {
    if (!n.init_expr) {
      Error(n.LOC(), "unable to deduce the type of `" + n.name_str + "'.");
      error_count++;
      // keep working
    } else {
      if (isa<MDSpanType>(n.init_expr->GetType())) {
        Error(n.LOC(), "use ':' instead of '=' to define the \"" +
                           STR(*n.init_expr->GetType()) + "\" type variable.");
        error_count++;
        if (trace_visit)
          os << "Error in " << __FILE__ << ", line: " << __LINE__ << ".\n";
        // keep working
      } else if (isa<PlaceHolderType>(n.init_expr->GetType())) {
        Error(n.LOC(), "can not initialize vairable `" + n.name_str +
                           "' with a placeholder.");
        error_count++;
        if (trace_visit)
          os << "Error in " << __FILE__ << ", line: " << __LINE__ << ".\n";
      }
      // sometimes the parser can not decide the type. We need to figure out
      // from the initialization expression
      SetNodeType(*n.type, n.init_expr->GetType());
    }
  }

  // check for type consistency between annotation and init expr.
  if (n.init_expr &&
      (!n.type->GetType()->ApprxEqual(*n.init_expr->GetType()))) {
    Error(n.LOC(), "`" + n.name_str + "' is declared as \"" +
                       PSTR(n.type->GetType()) + "\" but initialized as \"" +
                       PSTR(n.init_expr->GetType()) + "\".");
    error_count++;
    // keep working
  }

  ReportErrorWhenViolateODR(n.LOC(), n.name_str, __FILE__, __LINE__,
                            n.type->GetType());
  if (auto ty = dyn_cast<SpannedType>(n.type->GetType())) {
    ReportErrorWhenViolateODR(n.LOC(), n.name_str + ".span", __FILE__, __LINE__,
                              ty->GetMDSpanType());
  }
  return true;
}

bool EarlySemantics::Visit(AST::IntTuple& n) {
  __TRACE_EACH_VISIT__(n)
  SetNodeType(n, MakeITupleType(n.GetValues()->Count()));
  return true;
}

bool EarlySemantics::Visit(AST::Assignment& n) {
  __TRACE_EACH_VISIT__(n)
  // SSTab().Dump();
  if (!SSTab().DeclaredInScope(n.name)) {
    // It is a definition instead of assignment. parsing fails to distiguish
    // them
    auto sty = NodeType(*n.value);
    assert((sty && !isa<UnknownType>(sty)) &&
           "internal error: failed to find the type.");
    if (isa<MDSpanType>(sty)) {
      Error(n.LOC(),
            "use ':' to define the \"" + STR(*sty) + "\" type variable.");
      ++error_count;
      if (trace_visit)
        os << "Error in " << __FILE__ << ", line: " << __LINE__ << ".\n";
      return false;
    }
    ReportErrorWhenViolateODR(n.LOC(), n.name, __FILE__, __LINE__, sty);
    if (auto ty = dyn_cast<SpannedType>(sty)) {
      ReportErrorWhenViolateODR(n.LOC(), n.name + ".span", __FILE__, __LINE__,
                                MakeRankedMDSpanType(ty->Dims()));
      if (trace_visit)
        os << "Error in " << __FILE__ << ", line: " << __LINE__ << ".\n";
    }
    return true;
  }

  auto vty = SSTab().LookupSymbol(n.name);  // variable type
  auto ety = NodeType(*n.value);            // assignment expression type
  // ituples/mdspan/spanned can not be assigned after initialization
  if (isa<ITupleType>(ety) || isa<MDSpanType>(ety) || isa<SpannedType>(ety) ||
      isa<PlaceHolderType>(ety)) {
    if (vty->ApprxEqual(*ety))
      Error(n.LOC(), "`" + n.name + "' of type '" + vty->Name() +
                         "' can not be re-assigned.");
    else
      Error(n.LOC(), "`" + n.name + "' of type '" + STR(*vty) +
                         "' can not be re-assigned as '" + STR(*ety) + "'.");
    ++error_count;
    if (trace_visit)
      os << "Error in " << __FILE__ << ", line: " << __LINE__ << ".\n";
    return false;
  }

  // check for type consistent
  if (!vty->ApprxEqual(*ety)) {
    Error(n.LOC(), "`" + n.name + "' of type '" + STR(*vty) +
                       "' is assigned as " + STR(*ety) + ".");
    ++error_count;
    if (trace_visit)
      os << "Error in " << __FILE__ << ", line: " << __LINE__ << ".\n";
    return false;
  }

  // For now, we have to keep the single assignment
  {
    Error(n.LOC(), "current compiler does not support re-assignment of '" +
                       vty->Name() + "'.");
    ++error_count;
    if (trace_visit)
      os << "Error in " << __FILE__ << ", line: " << __LINE__ << ".\n";
    return false;
  }

  return true;
}

bool EarlySemantics::Visit(AST::IntIndex& n) {
  __TRACE_EACH_VISIT__(n)
  SetNodeType(n, MakeIndexType());
  return true;
}

bool EarlySemantics::Visit(AST::DataType& n) {
  __TRACE_EACH_VISIT__(n)

  allow_named_dim = false;  // no duplicated symbol is allowed except for mdspan

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
  __TRACE_EACH_VISIT__(n)
  if (in_decl) {
    if (allow_named_dim) {
      if (!SSTab().DeclaredInScope(n.name))
        SSTab().DefineSymbol(n.name,
                             MakeIntegerType());  // named dim is integer
    } else
      ReportErrorWhenViolateODR(n.LOC(), n.name, __FILE__, __LINE__);
  } else
    ReportErrorWhenUseBeforeDefine(n.LOC(), n.name);
  return true;
}

bool EarlySemantics::Visit(AST::Parameter& n) {
  __TRACE_EACH_VISIT__(n)
  if (n.sym) {
    SSTab().ModifySymbolType(n.sym->name, n.type->GetType());
    if (auto ty = dyn_cast<SpannedType>(n.type->GetType())) {
      SSTab().DefineSymbol(n.sym->name + ".span", ty->GetMDSpanType());
    }
  }
  return true;
}

bool EarlySemantics::Visit(AST::ParamList& n) {
  __TRACE_EACH_VISIT__(n)
  return true;
}

bool EarlySemantics::Visit(AST::ParallelBy& n) {
  __TRACE_EACH_VISIT__(n)
  ReportErrorWhenViolateODR(n.LOC(), n.biv, __FILE__, __LINE__,
                            MakeBoundedITupleType(Shape(1), "pv"));
  return true;
}

bool EarlySemantics::Visit(AST::WhereBind& n) {
  __TRACE_EACH_VISIT__(n)
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
  __TRACE_EACH_VISIT__(n)
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

  // infer the type of bounded variable
  if (n.with) {
    n.with->accept(*this);  // make the symbol be defined
    with_syms.insert(n.with->name);
    auto wty = MakeBoundedITupleType(Shape(cast<MDSpanType>(ity)->Dims()));
    SSTab().ModifySymbolType(n.with->name, wty);
    n.with->SetType(wty);
  }

  if (n.with_matchers) {
    n.with_matchers->accept(*this);  // make the symbol be defined
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
      SSTab().ModifySymbolType(sname, mty);
      v->SetType(mty);
    }
    if (n.with)
      n.with_matchers->SetType(n.with->GetType());
    else
      n.with_matchers->SetType(MakeBoundedITupleType(n.with_matchers->Count()));
  }
  in_decl = false;

  return true;
}

bool EarlySemantics::Visit(AST::WithBlock& n) {
  __TRACE_EACH_VISIT__(n)
  return true;
}

bool EarlySemantics::Visit(AST::Memory& n) {
  __TRACE_EACH_VISIT__(n)
  return true;
}

bool EarlySemantics::Visit(AST::SpanAs& n) {
  __TRACE_EACH_VISIT__(n)
  auto sty = dyn_cast<SpannedType>(NodeType(*n.id));

  if (!sty) {
    Error(n.LOC(), "span-as operation operates on a non-mdspan type.");
    error_count++;
    return false;
  }

  auto asty = MakeRankedSpannedType(n.list->Count(), (BaseType)sty->f_type,
                                    sty->m_type);
  SSTab().DefineSymbol(n.nid->name, asty);
  n.SetType(asty);

  // TODO: set the proper type
  return true;
}

bool EarlySemantics::Visit(AST::DMA& n) {
  __TRACE_EACH_VISIT__(n)

  if (n.operation == ".none") {  // skip the place holder
    assert(!n.future.empty());
    ReportErrorWhenViolateODR(n.LOC(), n.future, __FILE__, __LINE__,
                              MakePlaceHolderFutureType());
    ReportErrorWhenViolateODR(n.LOC(), n.future + ".span", __FILE__, __LINE__,
                              MakePlaceHolderMDSpanType());
    ReportErrorWhenViolateODR(n.LOC(), n.future + ".data", __FILE__, __LINE__,
                              MakePlaceHolderSpannedType());
    return true;
  }

  SpannedType* sty = nullptr;
  if (auto fty = dyn_cast<FutureType>(NodeType(*n.from)))
    sty = fty->GetSpannedType().get();
  else
    sty = cast<SpannedType>(NodeType(*n.from));

  SpannedType* tty = nullptr;
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
      SSTab().ModifySymbolType(n.future + ".span", MakeRankedMDSpanType(rank));
      auto spanned_ty = MakeRankedSpannedType(rank, sty->ElementType(), sto);
      SSTab().ModifySymbolType(n.future + ".data", spanned_ty);
      SSTab().ModifySymbolType(n.future, MakeFutureType(spanned_ty, n.async));
    } else {
      ReportErrorWhenViolateODR(n.LOC(), n.future + ".span", __FILE__, __LINE__,
                                MakeRankedMDSpanType(rank));
      auto spanned_ty = MakeRankedSpannedType(rank, sty->ElementType(), sto);
      ReportErrorWhenViolateODR(n.LOC(), n.future + ".data", __FILE__, __LINE__,
                                spanned_ty);
      ReportErrorWhenViolateODR(n.LOC(), n.future, __FILE__, __LINE__,
                                MakeFutureType(spanned_ty, n.async));
    }
  } else {
    if (n.async) {
      Error(n.LOC(), "forbid to associated async dma without a named future.");
      error_count++;
    }
  }

  if (!isa<AST::Memory>(n.to)) {
    if (sty->Dims() != tty->Dims()) {
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
            "padding settings");
      error_count++;
    }
  }
  return true;
}

bool EarlySemantics::Visit(AST::ChunkAt& n) {
  __TRACE_EACH_VISIT__(n)

  n.data->accept(*this);
  auto nty = NodeType(*n.data);
  if (n.sa) nty = NodeType(*n.sa);

  if (!isa<SpannedType>(nty) && !isa<FutureType>(nty)) {
    Error(n.LOC(),
          "expecting '" + n.data->name + "` of a spanned data or future type.");
    error_count++;
  }

  SpannedType* sty = nullptr;
  if (auto fty = dyn_cast<FutureType>(nty))
    sty = fty->GetSpannedType().get();
  else
    sty = cast<SpannedType>(nty);

  if (n.positions) {
    n.positions->accept(*this);
    size_t rank = sty->Dims();
    size_t r_count = 0;
    for (auto& v : n.positions->AllValues()) {
      auto ty = NodeType(*v);
      if (!IsBoundedType(ty)) {
        Error(n.LOC(),
              "expecting '" + v->TypeNameString() + "` be a bounded type.");
        error_count++;
      }
      r_count += ty->Dims();
      v->SetType(ty);
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
  __TRACE_EACH_VISIT__(n)

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
  __TRACE_EACH_VISIT__(n)

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
    if (!isa<SpannedType>(ty) && !isa<IntegerType>(ty) &&
        !isa<BooleanType>(ty)) {
      Error(n.LOC(), "(" + std::to_string(count) + "th) argument of type '" +
                         PSTR(ty) +
                         "` can not be passed to the kernel function.");
      error_count++;
    }
  }
  return true;
}

bool EarlySemantics::Visit(AST::Swap& n) {
  __TRACE_EACH_VISIT__(n)
  auto lty = NodeType(*n.lhs);
  auto rty = NodeType(*n.rhs);

  if (!isa<FutureType>(lty)) {
    Error(n.LOC(), "only support swapping of 'future'. (" + n.lhs->name + ": " +
                       PSTR(lty) + ").");
    error_count++;
    return false;
  }

  if (!lty->ApprxEqual(*rty)) {
    Error(n.LOC(), "swapping data of different types (" + PSTR(lty) + " vs. " +
                       PSTR(rty));
    error_count++;
    return false;
  }

  return true;
}

bool EarlySemantics::Visit(AST::Select& n) {
  __TRACE_EACH_VISIT__(n)

  // TODO(wsj) isa<IntegerType>(rty)?
  if (!isa<BoundedIntegerType>(NodeType(*n.select_factor))) {
    Error(n.LOC(), "expecting `" + PSTR(n.select_factor) +
                       "` be a bounded integer type.");
    error_count++;
  }

  // TODO(wsj) assert bound <= span_val_list.count ?

  // check value types in val_list are the same
  assert(n.span_expr_list->Count() > 0);
  const auto& v0 = n.span_expr_list->AllValues()[0];
  auto v0ty = NodeType(*v0);
  assert(isa<SpannedType>(v0ty) &&
         "For now, select only support spanned type variables!");
  for (auto& v : n.span_expr_list->AllValues()) {
    assert(isa<SpannedType>(NodeType(*v)));
    // TODO: need shape checking at typecheck
    if (auto sty = dyn_cast<SpannedType>(NodeType(*v))) {
      if (!sty->ApprxEqual(*v0ty)) {
        Error(v->LOC(), "expecting `" + PSTR(v) + "` is the same type as `" +
                            PSTR(v0) + "`.");
        error_count++;
      }
    } else {
      Error(v->LOC(), "expecting `" + PSTR(v) + "` to be spanned type.");
      error_count++;
    }
  }

  SetNodeType(n, NodeType(*v0));

  return true;
}

bool EarlySemantics::Visit(AST::Return& n) {
  __TRACE_EACH_VISIT__(n)
  found_return = true;
  if (parallel_level != 0) {
    Error(n.LOC(), "unable to return inside the parallel-by block(s).");
    error_count++;
    return false;
  }
  return true;
}

bool EarlySemantics::Visit(AST::LoopRange& n) {
  __TRACE_EACH_VISIT__(n)

  return true;
}

bool EarlySemantics::Visit(AST::ForeachBlock& n) {
  __TRACE_EACH_VISIT__(n)
  for (auto& i : n.getRanges()) {
    if (auto id = dyn_cast<AST::LoopRange>(i)->iv) {
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

bool EarlySemantics::Visit(AST::FunctionDecl& n) {
  __TRACE_EACH_VISIT__(n)

  if (n.ret_type->IsVoid())
    requires_return = false;
  else if (n.ret_type->IsUnknown())
    return_deduction = true;
  else
    requires_return = true;

  return true;
}

bool EarlySemantics::Visit(AST::ChoreoFunction& n) {
  __TRACE_EACH_VISIT__(n)
  return true;
}
bool EarlySemantics::Visit(AST::CppSourceCode& n) {
  __TRACE_EACH_VISIT__(n)
  return true;
}
bool EarlySemantics::Visit(AST::Program& n) {
  __TRACE_EACH_VISIT__(n)
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
    if (trace_visit) os << "Error in " << file << ", line: " << line << ".\n";
    return false;
  }
  SSTab().DefineSymbol(name, type);  // TODO: improve the type
  return true;
}

bool EarlySemantics::HasError() {
  if (error_count > 0) {
    os << "Totally " << error_count << " errors have been detected.\n";
    return true;
  }
  return false;
}
