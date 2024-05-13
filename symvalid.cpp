#include "symvalid.hpp"

using namespace Choreo;

#define __TRACE_EACH_VISIT__(n)       \
  if (trace_visit) {                  \
    os << n.NodeTypeString() << ": "; \
    os << "\n";                       \
  }

bool SymbolValidator::BeforeVisit(AST::Node& n) {
  if (isa<AST::Program>(&n)) {
    SSTab().EnterScope("");  // global scope
  } else if (auto f = dyn_cast<AST::ChoreoFunction>(&n)) {
    SSTab().EnterScope(f->name);
    requires_return = false;
    found_return = false;
  } else if (isa<AST::ParallelBy>(&n)) {
    static size_t count = 0;
    SSTab().EnterScope("paraby_" + std::to_string(count++));
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

bool SymbolValidator::AfterVisit(AST::Node& n) {
  if (isa<AST::Program>(&n) || isa<AST::ChoreoFunction>(&n) ||
      isa<AST::ParallelBy>(&n) || isa<AST::WithBlock>(&n) ||
      isa<AST::ForeachBlock>(&n)) {
    SSTab().LeaveScope();
  }

  if (auto f = dyn_cast<AST::ChoreoFunction>(&n)) {
    if (requires_return && !found_return) {
      Error(n.LOC(), "non-void function '" + f->name + "` does not contain a return statement.");
      error_count++;
    } else if (!requires_return && found_return) {
      Error(n.LOC(), "return statement found in void function '" + f->name + "`.");
      error_count++;
    }
  }

  if (isa<AST::Parameter>(&n)) {
    in_decl = false;
  }
  return true;
}

bool SymbolValidator::Visit(AST::MultiNodes& n) {
  __TRACE_EACH_VISIT__(n)
  return true;
}

bool SymbolValidator::Visit(AST::MultiValues& n) {
  __TRACE_EACH_VISIT__(n)
  return true;
}

bool SymbolValidator::Visit(AST::IntLiteral& n) {
  __TRACE_EACH_VISIT__(n)
  return true;
}

bool SymbolValidator::Visit(AST::Expr& n) {
  __TRACE_EACH_VISIT__(n)
  return true;
}

bool SymbolValidator::Visit(AST::MultiDimSpans& n) {
  __TRACE_EACH_VISIT__(n)
  return true;
}

bool SymbolValidator::Visit(AST::NamedTypeDecl& n) {
  __TRACE_EACH_VISIT__(n)
  ReportErrorWhenViolateODR(n.LOC(), n.name_str, __FILE__, __LINE__);
  ReportErrorWhenViolateODR(n.LOC(), n.name_str + ".span", __FILE__, __LINE__);
  return true;
}

bool SymbolValidator::Visit(AST::NamedVariableDecl& n) {
  __TRACE_EACH_VISIT__(n)
  ReportErrorWhenViolateODR(n.LOC(), n.name_str, __FILE__, __LINE__);
  if (n.type->isSpanned())
    ReportErrorWhenViolateODR(n.LOC(), n.name_str + ".span", __FILE__,
                              __LINE__);
  return true;
}

bool SymbolValidator::Visit(AST::IntTuple& n) {
  __TRACE_EACH_VISIT__(n)
  return true;
}

bool SymbolValidator::Visit(AST::Assignment& n) {
  __TRACE_EACH_VISIT__(n)
  return true;
}

bool SymbolValidator::Visit(AST::IntIndex& n) {
  __TRACE_EACH_VISIT__(n)
  return true;
}

bool SymbolValidator::Visit(AST::DataType& n) {
  __TRACE_EACH_VISIT__(n)
  return true;
}

bool SymbolValidator::Visit(AST::Identifier& n) {
  __TRACE_EACH_VISIT__(n)
  if (in_decl)
    ReportErrorWhenViolateODR(n.LOC(), n.name, __FILE__, __LINE__);
  else
    ReportErrorWhenUseBeforeDefine(n.LOC(), n.name);
  return true;
}

bool SymbolValidator::Visit(AST::Parameter& n) {
  __TRACE_EACH_VISIT__(n)
  if (n.sym && n.type->isSpanned())
    ReportErrorWhenViolateODR(n.LOC(), n.sym->name + ".span", __FILE__,
                              __LINE__);

  return true;
}

bool SymbolValidator::Visit(AST::ParamList& n) {
  __TRACE_EACH_VISIT__(n)
  return true;
}

bool SymbolValidator::Visit(AST::ParallelBy& n) {
  __TRACE_EACH_VISIT__(n)
  ReportErrorWhenViolateODR(n.LOC(), n.biv, __FILE__, __LINE__);
  return true;
}

bool SymbolValidator::Visit(AST::WhereBind& n) {
  __TRACE_EACH_VISIT__(n)
  return true;
}

bool SymbolValidator::Visit(AST::WithIn& n) {
  __TRACE_EACH_VISIT__(n)
  in_decl = true;
  if (n.with) n.with->accept(*this);
  if (n.with_matchers) n.with_matchers->accept(*this);
  in_decl = false;
  return true;
}
bool SymbolValidator::Visit(AST::WithBlock& n) {
  __TRACE_EACH_VISIT__(n)
  return true;
}
bool SymbolValidator::Visit(AST::Memory& n) {
  __TRACE_EACH_VISIT__(n)
  return true;
}
bool SymbolValidator::Visit(AST::DMA& n) {
  ReportErrorWhenViolateODR(n.LOC(), n.future, __FILE__, __LINE__);
  ReportErrorWhenViolateODR(n.LOC(), n.future + ".span", __FILE__, __LINE__);
  ReportErrorWhenViolateODR(n.LOC(), n.future + ".data", __FILE__, __LINE__);
  __TRACE_EACH_VISIT__(n)
  return true;
}

bool SymbolValidator::Visit(AST::ChunkAt& n) {
  __TRACE_EACH_VISIT__(n)
  n.data->accept(*this);
  if (n.positions) n.positions->accept(*this);
  return true;
}

bool SymbolValidator::Visit(AST::Wait& n) {
  __TRACE_EACH_VISIT__(n)
  return true;
}
bool SymbolValidator::Visit(AST::Call& n) {
  __TRACE_EACH_VISIT__(n)
  return true;
}
bool SymbolValidator::Visit(AST::Return& n) {
  __TRACE_EACH_VISIT__(n)
  found_return = true;
  return true;
}
bool SymbolValidator::Visit(AST::ForeachBlock& n) {
  __TRACE_EACH_VISIT__(n)
  return true;
}

bool SymbolValidator::Visit(AST::FunctionDecl& n) {
  __TRACE_EACH_VISIT__(n)

  if (n.ret_type->IsVoid())
    requires_return = false;
  else
    requires_return = true;

  return true;
}

bool SymbolValidator::Visit(AST::ChoreoFunction& n) {
  __TRACE_EACH_VISIT__(n)
  return true;
}
bool SymbolValidator::Visit(AST::CppSourceCode& n) {
  __TRACE_EACH_VISIT__(n)
  return true;
}
bool SymbolValidator::Visit(AST::Program& n) {
  __TRACE_EACH_VISIT__(n)
  return true;
}

bool SymbolValidator::ReportErrorWhenUseBeforeDefine(const location& loc,
                                                     const std::string& name) {
  if (!SSTab().IsDeclared(name)) {
    Error(loc, "symbol `" + name + "' is used before declaration.");
    ++error_count;
    return false;
  }
  return true;
}

bool SymbolValidator::ReportErrorWhenViolateODR(const location& loc,
                                                const std::string& name,
                                                const char* file, int line) {
  if (SSTab().DeclaredInScope(name)) {
    Error(loc, "symbol `" + name + "' has been declared already.");
    ++error_count;
    if (trace_visit) os << "Error in " << file << ", line: " << line << ".\n";
    return false;
  }
  SSTab().DefineSymbol(name, MakeUnknownType());  // TODO: improve the type
  return true;
}

bool SymbolValidator::HasError() {
  if (error_count > 0) {
    os << "Totally " << error_count << " errors have been detected.\n";
    return true;
  }
  return false;
}
