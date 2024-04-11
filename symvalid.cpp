#include "symvalid.hpp"

using namespace Choreo;

#define __TRACE_EACH_VISIT__(n)       \
  if (trace_visit) {                  \
    os << n.NodeTypeString() << ": "; \
    os << "\n";                       \
  }

bool SymbolValidator::BeforeVisit(AST::Node &) {
  return true;
}

bool SymbolValidator::AfterVisit(AST::Node &) {
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
  return true;
}
bool SymbolValidator::Visit(AST::NamedVariableDecl& n) {
  __TRACE_EACH_VISIT__(n)
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
  return true;
}
bool SymbolValidator::Visit(AST::Parameter& n) {
  __TRACE_EACH_VISIT__(n)
  return true;
}
bool SymbolValidator::Visit(AST::ParamList& n) {
  __TRACE_EACH_VISIT__(n)
  return true;
}
bool SymbolValidator::Visit(AST::ParallelBy& n) {
  __TRACE_EACH_VISIT__(n)
  return true;
}
bool SymbolValidator::Visit(AST::RequireBind& n) {
  __TRACE_EACH_VISIT__(n)
  return true;
}
bool SymbolValidator::Visit(AST::WithIn& n) {
  __TRACE_EACH_VISIT__(n)
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
