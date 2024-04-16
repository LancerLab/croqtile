#include "symvalid.hpp"

using namespace Choreo;

#define __TRACE_EACH_VISIT__(n)       \
  if (trace_visit) {                  \
    os << n.NodeTypeString() << ": "; \
    os << "\n";                       \
  }

bool SymbolValidator::BeforeVisit(AST::Node& n) {
  if (auto f = dyn_cast<AST::ChoreoFunction>(&n)) {
    SSTab().EnterScope(f->name);
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

  if (isa<AST::Parameter>(&n) || isa<AST::WithIn>(&n)) {
    in_decl = true;
  }

  return true;
}

bool SymbolValidator::AfterVisit(AST::Node& n) {
  if (isa<AST::ChoreoFunction>(&n) || isa<AST::ParallelBy>(&n) ||
      isa<AST::WithBlock>(&n) || isa<AST::ForeachBlock>(&n)) {
    SSTab().LeaveScope();
  }

  if (isa<AST::Parameter>(&n) || isa<AST::WithIn>(&n)) {
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
  ReportErrorWhenViolateODR(n.LOC(), n.name_str);
  ReportErrorWhenViolateODR(n.LOC(), n.name_str + ".span");
  return true;
}

bool SymbolValidator::Visit(AST::NamedVariableDecl& n) {
  __TRACE_EACH_VISIT__(n)
  ReportErrorWhenViolateODR(n.LOC(), n.name_str);
  if (n.type->isSpanned())
    ReportErrorWhenViolateODR(n.LOC(), n.name_str + ".span");
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
    ReportErrorWhenViolateODR(n.LOC(), n.name);
  else
    ReportErrorWhenUseBeforeDefine(n.LOC(), n.name);
  return true;
}

bool SymbolValidator::Visit(AST::Parameter& n) {
  __TRACE_EACH_VISIT__(n)
  if (n.sym && n.type->isSpanned())
    ReportErrorWhenViolateODR(n.LOC(), n.sym->name + ".span");
    
  return true;
}

bool SymbolValidator::Visit(AST::ParamList& n) {
  __TRACE_EACH_VISIT__(n)
  return true;
}

bool SymbolValidator::Visit(AST::ParallelBy& n) {
  __TRACE_EACH_VISIT__(n)
  ReportErrorWhenViolateODR(n.LOC(), n.biv);
  return true;
}

bool SymbolValidator::Visit(AST::RequireBind& n) {
  __TRACE_EACH_VISIT__(n)
  return true;
}

bool SymbolValidator::Visit(AST::WithIn& n) {
  __TRACE_EACH_VISIT__(n)
  ReportErrorWhenViolateODR(n.LOC(), n.with->name);
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
  ReportErrorWhenViolateODR(n.LOC(), n.future);
  ReportErrorWhenViolateODR(n.LOC(), n.future + ".span");
  ReportErrorWhenViolateODR(n.LOC(), n.future + ".data");
  __TRACE_EACH_VISIT__(n)
  return true;
}
bool SymbolValidator::Visit(AST::ChunkAt& n) {
  __TRACE_EACH_VISIT__(n)
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
  return true;
}
bool SymbolValidator::Visit(AST::ForeachBlock& n) {
  __TRACE_EACH_VISIT__(n)
  return true;
}
bool SymbolValidator::Visit(AST::FunctionDecl& n) {
  __TRACE_EACH_VISIT__(n)
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
                                                const std::string& name) {
  if (SSTab().DeclaredInScope(name)) {
    Error(loc, "symbol `" + name + "' has been declared already.");
    ++error_count;
    return false;
  }
  SSTab().DefineSymbol(name, MakeUnknownType());  // TODO: improve the type
  return true;
}

bool SymbolValidator::HasError() {
  if (error_count > 0) {
    os << "Totally " << error_count << " errors are detected.\n";
    return true;
  }
  return false;
}
