#include "typecheck.hpp"

using namespace Choreo;

#define __TRACE_EACH_VISIT__(n)       \
  if (trace_visit) {                  \
    os << n.NodeTypeString() << ": "; \
    os << "\n";                       \
  }

bool TypeChecker::BeforeVisit(AST::Node& n) {
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
  return true;
}

bool TypeChecker::AfterVisit(AST::Node& n) {
  if (isa<AST::ChoreoFunction>(&n) || isa<AST::ParallelBy>(&n) ||
      isa<AST::WithBlock>(&n) || isa<AST::ForeachBlock>(&n)) {
    SSTab().LeaveScope();
  }
  return true;
}

bool TypeChecker::Visit(AST::MultiNodes& n) {
  __TRACE_EACH_VISIT__(n)
  return true;
}
bool TypeChecker::Visit(AST::MultiValues& n) {
  __TRACE_EACH_VISIT__(n)
  return true;
}
bool TypeChecker::Visit(AST::IntLiteral& n) {
  __TRACE_EACH_VISIT__(n)
  return true;
}
bool TypeChecker::Visit(AST::Expr& n) {
  __TRACE_EACH_VISIT__(n)
  return true;
}
bool TypeChecker::Visit(AST::MultiDimSpans& n) {
  __TRACE_EACH_VISIT__(n)
  return true;
}
bool TypeChecker::Visit(AST::NamedTypeDecl& n) {
  __TRACE_EACH_VISIT__(n)
  return true;
}
bool TypeChecker::Visit(AST::NamedVariableDecl& n) {
  __TRACE_EACH_VISIT__(n)
  return true;
}
bool TypeChecker::Visit(AST::IntTuple& n) {
  __TRACE_EACH_VISIT__(n)
  return true;
}
bool TypeChecker::Visit(AST::Assignment& n) {
  __TRACE_EACH_VISIT__(n)
  return true;
}
bool TypeChecker::Visit(AST::IntIndex& n) {
  __TRACE_EACH_VISIT__(n)
  return true;
}
bool TypeChecker::Visit(AST::DataType& n) {
  __TRACE_EACH_VISIT__(n)
  return true;
}
bool TypeChecker::Visit(AST::Identifier& n) {
  __TRACE_EACH_VISIT__(n)
  return true;
}
bool TypeChecker::Visit(AST::Parameter& n) {
  __TRACE_EACH_VISIT__(n)
  return true;
}
bool TypeChecker::Visit(AST::ParamList& n) {
  __TRACE_EACH_VISIT__(n)
  return true;
}
bool TypeChecker::Visit(AST::ParallelBy& n) {
  __TRACE_EACH_VISIT__(n)
  return true;
}
bool TypeChecker::Visit(AST::RequireBind& n) {
  __TRACE_EACH_VISIT__(n)
  return true;
}
bool TypeChecker::Visit(AST::WithIn& n) {
  __TRACE_EACH_VISIT__(n)
  return true;
}
bool TypeChecker::Visit(AST::WithBlock& n) {
  __TRACE_EACH_VISIT__(n)
  return true;
}
bool TypeChecker::Visit(AST::Memory& n) {
  __TRACE_EACH_VISIT__(n)
  return true;
}
bool TypeChecker::Visit(AST::DMA& n) {
  __TRACE_EACH_VISIT__(n)
  return true;
}
bool TypeChecker::Visit(AST::ChunkAt& n) {
  __TRACE_EACH_VISIT__(n)
  return true;
}
bool TypeChecker::Visit(AST::Wait& n) {
  __TRACE_EACH_VISIT__(n)
  return true;
}
bool TypeChecker::Visit(AST::Call& n) {
  __TRACE_EACH_VISIT__(n)
  return true;
}
bool TypeChecker::Visit(AST::Return& n) {
  __TRACE_EACH_VISIT__(n)
  return true;
}
bool TypeChecker::Visit(AST::ForeachBlock& n) {
  __TRACE_EACH_VISIT__(n)
  return true;
}
bool TypeChecker::Visit(AST::FunctionDecl& n) {
  __TRACE_EACH_VISIT__(n)
  return true;
}
bool TypeChecker::Visit(AST::ChoreoFunction& n) {
  __TRACE_EACH_VISIT__(n)
  return true;
}
bool TypeChecker::Visit(AST::CppSourceCode& n) {
  __TRACE_EACH_VISIT__(n)
  return true;
}
bool TypeChecker::Visit(AST::Program& n) {
  __TRACE_EACH_VISIT__(n)
  return true;
}
