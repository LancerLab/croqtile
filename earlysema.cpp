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
  }

  if (isa<AST::Parameter>(&n)) {
    in_decl = false;
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

bool EarlySemantics::Visit(AST::Expr& n) {
  __TRACE_EACH_VISIT__(n)
  if (auto ref = n.GetReference()) {
    auto rty = NodeType(*ref);
    assert(!isa<UnknownType>(rty) && "reference type is unknown.");
    SetNodeType(n, rty);
  } else if (n.op == "dataof") {
    auto ty = NodeType(*n.value_r);
    if (!isa<FutureType>(ty)) {
      Error(n.LOC(), "in operation \"" + n.op +
                         "\": expecting a future type but got `" + PSTR(ty) +
                         "'.");
      error_count++;
      return false;
    }
    SetNodeType(n, MakeUninitSpannedType());
  } else if (n.op == "sizeof") {
    auto ty = NodeType(*n.value_r);
    if (!isa<MDSpanType>(ty)) {
      Error(n.LOC(), "in operation \"" + n.op +
                         "\": expecting a mdspan type but got `" + PSTR(ty) +
                         "'.");
      error_count++;
      return false;
    }
    SetNodeType(n, MakeIntegerType());
  } else if (n.op == "dimof") {
    auto lty = NodeType(*n.value_l);
    auto rty = NodeType(*n.value_r);
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
    auto ty = NodeType(*n.value_r);
    if (!IsBoundedType(ty)) {
      Error(n.LOC(), "in operation \"" + n.op +
                         "\": expecting a bounded type but got `" + PSTR(ty) +
                         "'.");
      error_count++;
      return false;
    }
    SetNodeType(n, MakeIntegerType());
  } else if ((n.op == "+") || (n.op == "-") || (n.op == "*") || (n.op == "/") ||
             (n.op == "%")) {
    auto lty = NodeType(*n.value_l);
    auto rty = NodeType(*n.value_r);
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
      SetNodeType(n, MakeDimedMDSpanType(lty->Dims()));
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
               (isa<BoundedIntegerType>(lty) && isa<BoundedIntegerType>(rty))){
      // allow only * operator for catesian products on two bounded-vars
      // currently, only support boundedituple * boundedint or boundedint * boundedint
      // os << STR(n.value_l) << "lty = " << PSTR(lty) << "; rty = " << PSTR(rty);
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
      if ((n.op != "/") && (n.op != "%")) {
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
    auto lty = NodeType(*n.value_l);
    auto rty = NodeType(*n.value_r);
    assert(false);
    if (!(lty->ApprxEqual(*rty))) {
      Error(n.LOC(), "in operation \"" + n.op +
                         "\": unable to apply to the types (" + PSTR(lty) +
                         " vs. " + PSTR(rty) + ").");
      error_count++;
      return false;
    }
    SetNodeType(n, MakeBooleanType());
  } else if ((n.op == "&&") || (n.op == "||")) {
    auto lty = NodeType(*n.value_l);
    auto rty = NodeType(*n.value_r);
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
    auto rty = NodeType(*n.value_r);
    if (!isa<BooleanType>(rty)) {  // TODO: will we allow integer?
      Error(n.LOC(), "in operation \"" + n.op +
                         "\": unable to apply to the type (" + PSTR(rty) +
                         ").");
      error_count++;
      return false;
    }
    SetNodeType(n, MakeBooleanType());
  } else if (n.op == "?") {
    auto cty = NodeType(*n.value_c);
    auto lty = NodeType(*n.value_l);
    auto rty = NodeType(*n.value_r);
    if (!isa<BooleanType>(cty) || (!lty->ApprxEqual(*rty))) {
      Error(n.LOC(), "in operation \"" + n.op +
                         "\": unable to apply to the types (" + PSTR(cty) +
                         ") " + PSTR(lty) + " : " + PSTR(rty) + ").");
      error_count++;
      return false;
    }
    SetNodeType(n, lty);
  } else
    choreo_unreachable("operation in expression is not supported yet.");
  return true;
}

bool EarlySemantics::Visit(AST::MultiDimSpans& n) {
  __TRACE_EACH_VISIT__(n)
  size_t rank = InvalidRank();

  // try to figure out the dimensions
  if (auto mvals = dyn_cast<AST::MultiValues>(n.list))
    rank = mvals->Count();
  else if (isa<AST::Expr>(n.list))
    rank = n.list->GetType()->Dims();

  if (n.Rank() == InvalidRank())
    n.SetRank(rank);
  else if (rank == InvalidRank())
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

  SetNodeType(n, MakeDimedMDSpanType(rank));
  return true;
}

bool EarlySemantics::Visit(AST::NamedTypeDecl& n) {
  __TRACE_EACH_VISIT__(n)
  assert(n.init_expr && "missing init expr.");
  auto ety = NodeType(*n.init_expr);
  auto nty = (n.rank != InvalidRank()) ? MakeDimedMDSpanType(n.rank)
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
                              MakeDimedMDSpanType(ty->Dims()));
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
                                MakeDimedMDSpanType(ty->Dims()));
      if (trace_visit)
        os << "Error in " << __FILE__ << ", line: " << __LINE__ << ".\n";
    }
    return true;
  }

  auto vty = SSTab().LookupSymbol(n.name);  // variable type
  auto ety = NodeType(*n.value);            // assignment expression type
  // ituples/mdspan/spanned can not be assigned after initialization
  if (isa<ITupleType>(ety) || isa<MDSpanType>(ety) || isa<SpannedType>(ety)) {
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
                       ety->Name() + "'.");
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
  // sema type has been generated at construction ast. refine with dims
  if (isa<SpannedType>(n.GetType())) {
    if (auto sty = dyn_cast<MDSpanType>(n.mdspan_type->GetType())) {
      SetNodeType(n, MakeDimedSpannedType(sty->Dims(), n.base_type));
    }
    return true;
  }
  return true;
}

bool EarlySemantics::Visit(AST::Identifier& n) {
  __TRACE_EACH_VISIT__(n)
  if (in_decl)
    ReportErrorWhenViolateODR(n.LOC(), n.name, __FILE__, __LINE__);
  else
    ReportErrorWhenUseBeforeDefine(n.LOC(), n.name);
  return true;
}

bool EarlySemantics::Visit(AST::Parameter& n) {
  __TRACE_EACH_VISIT__(n)
  if (n.sym) {
    SSTab().ModifySymbolType(n.sym->name, n.type->GetType());
    if (auto ty = dyn_cast<SpannedType>(n.type->GetType()))
      SSTab().DefineSymbol(n.sym->name + ".span",
                           MakeDimedMDSpanType(ty->Dims()));
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
                            MakeUninitBoundedITupleType());
  return true;
}

bool EarlySemantics::Visit(AST::WhereBind& n) {
  __TRACE_EACH_VISIT__(n)
  return true;
}

bool EarlySemantics::Visit(AST::WithIn& n) {
  __TRACE_EACH_VISIT__(n)
  in_decl = true;
  if (n.with) {
    n.with->accept(*this);
    if (n.with_matchers) {
      // simple infer the rank from matchers
      SSTab().ModifySymbolType(
          n.with->name, MakeBoundedITupleType(Shape(n.with_matchers->Count())));
    } else {
      // can not figure out the dimensions at this time
      SSTab().ModifySymbolType(n.with->name, MakeUninitBoundedITupleType());
    }
  }
  if (n.with_matchers) {
    n.with_matchers->accept(*this);
    for (auto v : n.with_matchers->AllValues()) {
      if (!isa<AST::Identifier>(v)) {
        Error(v->LOC(), "expecting an identifier.");
        continue;
      }
      auto sname = cast<AST::Identifier>(v)->name;
      SSTab().ModifySymbolType(sname, MakeBoundedIntegerType(sname));
    }
  }
  in_decl = false;

  auto ity = NodeType(*n.in);
  if (!isa<MDSpanType>(ity)) {
    Error(n.in->LOC(),
          "expecting a span type but got the " + PSTR(ity) + " type.");
    error_count++;
  }

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

bool EarlySemantics::Visit(AST::DMA& n) {
  __TRACE_EACH_VISIT__(n)
  ReportErrorWhenViolateODR(n.LOC(), n.future, __FILE__, __LINE__,
                            MakeFutureType(n.async));
  size_t rank = cast<SpannedType>(NodeType(*n.from))->Dims();
  ReportErrorWhenViolateODR(n.LOC(), n.future + ".span", __FILE__, __LINE__,
                            MakeDimedMDSpanType(rank));
  ReportErrorWhenViolateODR(n.LOC(), n.future + ".data", __FILE__, __LINE__,
                            MakeDimedSpannedType(rank));
  return true;
}

bool EarlySemantics::Visit(AST::ChunkAt& n) {
  __TRACE_EACH_VISIT__(n)

  n.data->accept(*this);
  if (!isa<SpannedType>(NodeType(*n.data))) {
    Error(n.LOC(), "expecting '" + n.data->name + "` of a spanned data.");
    error_count++;
  }
  if (n.positions) {
    n.positions->accept(*this);
    for (auto& v : n.positions->AllValues()) {
      // if this is a expr, do expr check, otherwise to normal bounded var check
      // if (auto node = cast<AST::Expr>(v)) {
      //   this->Visit(*node);
      // } else {
        auto ty = NodeType(*v);
        if (!isa<BoundedIntegerType>(ty) &&
            !isa<BoundedIntegerType>(ty) &&
            !isa<BoundedITupleType>(ty)) {
          Error(n.LOC(), "expecting '" + v->TypeNameString() +
                             "` be a bounded type.");
          error_count++;
        }
      // }
    }
  }
  size_t rank = cast<SpannedType>(NodeType(*n.data))->Dims();
  SetNodeType(n, MakeDimedSpannedType(rank));
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

bool EarlySemantics::Visit(AST::ForeachBlock& n) {
  __TRACE_EACH_VISIT__(n)
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
