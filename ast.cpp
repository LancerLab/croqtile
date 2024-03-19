#include "visitor.hpp"

namespace AST {

void MultiNodes::accept(Choreo::Visitor& v) {
  for (auto& sub : values) sub->accept(v);
}

void Boolean::accept(Choreo::Visitor& v) { (void)v; }
void IntLiteral::accept(Choreo::Visitor& v) { (void)v; }
void IntList::accept(Choreo::Visitor& v) { (void)v; }
void SValList::accept(Choreo::Visitor& v) { (void)v; }
void Expr::accept(Choreo::Visitor& v) { (void)v; }
void MultiDimSpans::accept(Choreo::Visitor& v) { (void)v; }
void NamedTypeDecl::accept(Choreo::Visitor& v) { (void)v; }
void NamedVariableDecl::accept(Choreo::Visitor& v) { (void)v; }
void IntTuple::accept(Choreo::Visitor& v) { (void)v; }
void Assignment::accept(Choreo::Visitor& v) { (void)v; }
void IntIndex::accept(Choreo::Visitor& v) { (void)v; }
void NthBound::accept(Choreo::Visitor& v) { (void)v; }
void IntIndexList::accept(Choreo::Visitor& v) { (void)v; }
void DataType::accept(Choreo::Visitor& v) { (void)v; }

void Identifier::accept(Choreo::Visitor& v) { v.Visit(*this); }

void ParamList::accept(Choreo::Visitor& v) { v.Visit(*this); }

void IfElse::accept(Choreo::Visitor& v) { (void)v; }

void ParallelBy::accept(Choreo::Visitor& v) {
  v.Visit(*this);

  statms->accept(v);

  v.AfterVisit(*this);
}

void RequireBind::accept(Choreo::Visitor& v) { (void)v; }

void WithIn::accept(Choreo::Visitor& v) { v.Visit(*this); }

void WithBlock::accept(Choreo::Visitor& v) {
  withins->accept(v);
  if (reqs) reqs->accept(v);
  v.Visit(*this);
  statms->accept(v);
}

void Memory::accept(Choreo::Visitor& v) { v.Visit(*this); }

void DMA::accept(Choreo::Visitor& v) { v.Visit(*this); }

void ChunkAt::accept(Choreo::Visitor& v) { (void)v; }
void Wait::accept(Choreo::Visitor& v) { (void)v; }
void Return::accept(Choreo::Visitor& v) { (void)v; }
void Call::accept(Choreo::Visitor& v) { (void)v; }
void ForeachBlock::accept(Choreo::Visitor& v) {
  v.Visit(*this);
  statms->accept(v);
}

void FunctionDecl::accept(Choreo::Visitor& v) {
  params->accept(v);
  ret_type->accept(v);

  v.Visit(*this);
}

void ChoreoFunction::accept(Choreo::Visitor& v) {
  v.BeforeVisit(*this);
  v.Visit(*this);

  f_decl.accept(v);
  statms->accept(v);

  v.AfterVisit(*this);
}

void CppSourceCode::accept(Choreo::Visitor& v) { v.Visit(*this); }

void Program::accept(Choreo::Visitor& v) {
  v.BeforeVisit(*this);

  v.Visit(*this);
  for (auto ptr : nodes) ptr->accept(v);

  v.AfterVisit(*this);
}

}  // end of namespace AST
