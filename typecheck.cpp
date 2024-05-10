#include "typecheck.hpp"

#include "aux.hpp"

using namespace Choreo;

#define __TRACE_EACH_VISIT__(n)       \
  if (trace_visit) {                  \
    os << n.NodeTypeString() << ": "; \
    os << "\n";                       \
  }

bool TypeChecker::BeforeVisitImpl(AST::Node&) { return true; }

bool TypeChecker::AfterVisitImpl(AST::Node&) { return true; }

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
  if (!ReportUnknown(n, __FILE__, __LINE__)) return false;
  return true;
}
bool TypeChecker::Visit(AST::Expr& n) {
  __TRACE_EACH_VISIT__(n)
  if (!ReportUnknown(n, __FILE__, __LINE__)) return false;
  return true;
}
bool TypeChecker::Visit(AST::MultiDimSpans& n) {
  __TRACE_EACH_VISIT__(n)
  if (!ReportUnknown(n, __FILE__, __LINE__)) return false;
  return true;
}
bool TypeChecker::Visit(AST::NamedTypeDecl& n) {
  __TRACE_EACH_VISIT__(n)
  if (!ReportUnknown(n, __FILE__, __LINE__)) return false;
  return true;
}
bool TypeChecker::Visit(AST::NamedVariableDecl& n) {
  __TRACE_EACH_VISIT__(n)
  if (!ReportUnknown(n, __FILE__, __LINE__)) return false;
  return true;
}
bool TypeChecker::Visit(AST::IntTuple& n) {
  __TRACE_EACH_VISIT__(n)
  if (!ReportUnknown(n, __FILE__, __LINE__)) return false;
  return true;
}
bool TypeChecker::Visit(AST::Assignment& n) {
  __TRACE_EACH_VISIT__(n)
  if (!ReportUnknown(n, __FILE__, __LINE__)) return false;
  return true;
}
bool TypeChecker::Visit(AST::IntIndex& n) {
  __TRACE_EACH_VISIT__(n)
  if (!ReportUnknown(n, __FILE__, __LINE__)) return false;
  return true;
}
bool TypeChecker::Visit(AST::DataType& n) {
  __TRACE_EACH_VISIT__(n)
  if (!ReportUnknown(n, __FILE__, __LINE__)) return false;
  return true;
}
bool TypeChecker::Visit(AST::Identifier& n) {
  __TRACE_EACH_VISIT__(n)
  if (PrefixedWith(n.name, "$")) return true;  // do not check internal symbols
  if (isa<UnknownType>(GetSymbolType(n.name))) {
    ++error_count;
    Error(n.LOC(), "failed to get/infer the type of " + n.name + ".");
    return false;
  }
  return true;
}
bool TypeChecker::Visit(AST::Parameter& n) {
  __TRACE_EACH_VISIT__(n)
  if (!ReportUnknown(n, __FILE__, __LINE__)) return false;
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
  if (!ReportUnknown(n, __FILE__, __LINE__)) return false;
  return true;
}
bool TypeChecker::Visit(AST::ChunkAt& n) {
  __TRACE_EACH_VISIT__(n)
  if (!ReportUnknown(n, __FILE__, __LINE__)) return false;
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

bool TypeChecker::ReportUnknown(AST::Node& n, const char* file, int line) {
  if (AST::typeof<UnknownType>(&n)) {
    ++error_count;
    Error(n.LOC(), "failed to get/infer the type.");
    if (trace_visit) os << file << ":" << line << ", " << AST::STR(n) << "\n";
    return false;
  }
  return true;
}

bool TypeChecker::HasError() {
  if (error_count) {
    os << "Totally " << error_count << " errors are detected in type check.\n";
    return true;
  }
  return false;
}
