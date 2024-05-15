#include "earlysema.hpp"

using namespace Choreo;

#define __TRACE_EACH_VISIT__(n)       \
  if (trace_visit) {                  \
    os << n.NodeTypeString() << ": "; \
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
  n.SetType(MakeIntegerType());
  return true;
}

bool EarlySemantics::Visit(AST::Expr& n) {
  __TRACE_EACH_VISIT__(n)
  if (auto ref = n.GetReference()) {
    assert(!isa<UnknownType>(ref) && "reference type is unknown.");
    n.SetType(ref->GetType());
  } else if (n.op == "dataof") {
    auto ty = NodeType(*n.value_r);
    if (!isa<FutureType>(ty)) {
      Error(n.LOC(), "in operation \"" + n.op +
                         "\": expecting a future type but got `" + PSTR(ty) +
                         "'.");
      error_count++;
      return false;
    }
    n.SetType(MakeUninitSpannedType());
  } else if (n.op == "sizeof") {
    auto ty = NodeType(*n.value_r);
    if (!isa<MDSpanType>(ty)) {
      Error(n.LOC(), "in operation \"" + n.op +
                         "\": expecting a mdspan type but got `" + PSTR(ty) +
                         "'.");
      error_count++;
      return false;
    }
    n.SetType(MakeIntegerType());
  } else if (n.op == "dimof") {
    auto lty = NodeType(*n.value_l);
    auto rty = NodeType(*n.value_r);
    if (!isa<MDSpanType>(lty)) {
      Error(n.LOC(), "in operation \"" + n.op +
                         "\": expecting a mdspan type but got `" + PSTR(lty) +
                         "'.");
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
    n.SetType(MakeIntegerType());
  } else if (n.op == "ubound") {
    auto ty = NodeType(*n.value_r);
    if (!IsBoundedType(ty)) {
      Error(n.LOC(), "in operation \"" + n.op +
                         "\": expecting a bounded type but got `" + PSTR(ty) +
                         "'.");
      error_count++;
      return false;
    }
    n.SetType(MakeIntegerType());
  } else if ((n.op == "+") || (n.op == "-") || (n.op == "*") || (n.op == "/") ||
             (n.op == "%")) {
    auto lty = NodeType(*n.value_l);
    auto rty = NodeType(*n.value_r);
    if ((isa<MDSpanType>(lty) && isa<ITupleType>(rty)) ||
        (isa<MDSpanType>(rty) && isa<ITupleType>(rty))) {
      if (lty->Dims() != rty->Dims()) {
        Error(n.LOC(), "in operation \"" + n.op +
                           "\": dimension inconsistent (" +
                           std::to_string(lty->Dims()) + " vs. " +
                           std::to_string(rty->Dims()) + ").");
        error_count++;
        return false;
      }
      n.SetType(MakeDimedMDSpanType(lty->Dims()));
    } else if ((isa<MDSpanType>(lty) && isa<MDSpanType>(rty)) ||
               (isa<ITupleType>(lty) && isa<ITupleType>(rty)) ||
               (isa<BooleanType>(lty) && isa<BooleanType>(rty))) {
      Error(n.LOC(), "in operation \"" + n.op +
                         "\": unable to apply to the types (" + PSTR(lty) +
                         " vs. " + PSTR(rty) + ").");
      error_count++;
      return false;
    } else if (isa<IndexType>(lty) && isa<IndexType>(rty)) {
      Error(n.LOC(), "in operation \"" + n.op +
                         "\": unable to apply to the types (" + PSTR(lty) +
                         " vs. " + PSTR(rty) + ").");
      error_count++;
      return false;
    } else if (!(isa<IntegerType>(lty) && isa<IndexType>(rty)) &&
               !(isa<IntegerType>(rty) && isa<IndexType>(lty)) &&
               (*lty != *rty)) {
      Error(n.LOC(), "in operation \"" + n.op +
                         "\": unable to apply to the types (" + PSTR(lty) +
                         " vs. " + PSTR(rty) + ").");
      error_count++;
      return false;
    }
    n.SetType(lty);
  } else if ((n.op == "<") || (n.op == ">") || (n.op == "==") ||
             (n.op == "!=") || (n.op == "<=") || (n.op == ">=")) {
    auto lty = NodeType(*n.value_l);
    auto rty = NodeType(*n.value_r);
    if (*lty != *rty) {
      Error(n.LOC(), "in operation \"" + n.op +
                         "\": unable to apply to the types (" + PSTR(lty) +
                         " vs. " + PSTR(rty) + ").");
      error_count++;
      return false;
    }
    n.SetType(MakeBooleanType());
  } else if ((n.op == "&&") || (n.op == "||")) {
    auto lty = NodeType(*n.value_l);
    auto rty = NodeType(*n.value_r);
    if (!isa<BooleanType>(lty) || !isa<BooleanType>(rty)) {
      Error(n.LOC(), "in operation \"" + n.op +
                         "\": unable to apply to the types (" + PSTR(lty) +
                         " vs. " + PSTR(rty) + ").");
      error_count++;
      return false;
    }
    n.SetType(MakeBooleanType());
  } else if (n.op == "!") {
    auto rty = NodeType(*n.value_r);
    if (!isa<BooleanType>(rty)) {  // TODO: will we allow integer?
      Error(n.LOC(), "in operation \"" + n.op +
                         "\": unable to apply to the type (" + PSTR(rty) +
                         ").");
      error_count++;
      return false;
    }
    n.SetType(MakeBooleanType());
  } else if (n.op == "?") {
    auto cty = NodeType(*n.value_c);
    auto lty = NodeType(*n.value_l);
    auto rty = NodeType(*n.value_r);
    if (!isa<BooleanType>(cty) || (*lty != *rty)) {
      Error(n.LOC(), "in operation \"" + n.op +
                         "\": unable to apply to the types (" + PSTR(cty) +
                         ") " + PSTR(lty) + " : " + PSTR(rty) + ").");
      error_count++;
      return false;
    }
    n.SetType(lty);
  } else
    choreo_unreachable("operation in expression is not supported yet.");
  return true;
}

bool EarlySemantics::Visit(AST::MultiDimSpans& n) {
  __TRACE_EACH_VISIT__(n)
  n.SetType(MakeDimedMDSpanType(n.Dims()));
  return true;
}

bool EarlySemantics::Visit(AST::NamedTypeDecl& n) {
  __TRACE_EACH_VISIT__(n)
  ReportErrorWhenViolateODR(n.LOC(), n.name_str, __FILE__, __LINE__,
                            NodeType(*n.init_expr));
  return true;
}

bool EarlySemantics::Visit(AST::NamedVariableDecl& n) {
  __TRACE_EACH_VISIT__(n)
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
  return true;
}

bool EarlySemantics::Visit(AST::Assignment& n) {
  __TRACE_EACH_VISIT__(n)
  return true;
}

bool EarlySemantics::Visit(AST::IntIndex& n) {
  __TRACE_EACH_VISIT__(n)
  n.SetType(MakeIndexType());
  return true;
}

bool EarlySemantics::Visit(AST::DataType& n) {
  __TRACE_EACH_VISIT__(n)
  // sema type has been generated at construction ast
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
    SSTab().ModifySymbolType(n.with->name, MakeUninitBoundedITupleType());
  }
  if (n.with_matchers) {
    n.with_matchers->accept(*this);
    for (auto v : n.with_matchers->GetValues()) {
      if (!isa<AST::Identifier>(v)) {
        Error(v->LOC(), "expecting an identifier.");
        continue;
      }
      SSTab().ModifySymbolType(cast<AST::Identifier>(v)->name,
                               MakeUninitBoundedITupleType());
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
    for (auto& v : n.positions->GetValues()) {
      auto ty = NodeType(*v);
      if (!isa<BoundedIntegerType>(ty) && !isa<BoundedITupleType>(ty)) {
        Error(n.LOC(), "expecting '" + cast<AST::Identifier>(v)->name +
                           "` be a bounded type.");
        error_count++;
      }
    }
  }
  size_t rank = cast<SpannedType>(NodeType(*n.data))->Dims();
  n.SetType(MakeDimedSpannedType(rank));
  return true;
}

bool EarlySemantics::Visit(AST::Wait& n) {
  __TRACE_EACH_VISIT__(n)

  for (auto& v : n.targets->GetValues()) {
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
          "Unable to call kernel function outside the parallel-by block(s).");
    error_count++;
    return false;
  }

  size_t count = 0;
  for (auto& v : n.arguments->GetValues()) {
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
    Error(n.LOC(), "Unable to return inside the parallel-by block(s).");
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
