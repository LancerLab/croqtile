#include "visitor.hpp"

namespace AST {

void MultiNodes::accept(Choreo::Visitor& v) {
  for (auto& sub : values) sub->accept(v);
}

void Boolean::accept(Choreo::Visitor& v) {}
void IntLiteral::accept(Choreo::Visitor& v) {}
void IntList::accept(Choreo::Visitor& v) {}
void SValList::accept(Choreo::Visitor& v) {}
void Expr::accept(Choreo::Visitor& v) {}
void MultiSpans::accept(Choreo::Visitor& v) {}
void NamedDecl::accept(Choreo::Visitor& v) {}
void IntTuple::accept(Choreo::Visitor& v) {}
void Assignment::accept(Choreo::Visitor& v) {}
void IntIndex::accept(Choreo::Visitor& v) {}
void NthBound::accept(Choreo::Visitor& v) {}
void IntIndexList::accept(Choreo::Visitor& v) {}
void DataType::accept(Choreo::Visitor& v) {}
void Identifier::accept(Choreo::Visitor& v) {}

void ParamList::accept(Choreo::Visitor& v) {
   v.Visit(*this);
}

void IfElse::accept(Choreo::Visitor& v) {}
void ParallelBy::accept(Choreo::Visitor& v) {}
void RequireBind::accept(Choreo::Visitor& v) {}
void WithIn::accept(Choreo::Visitor& v) {}
void WithBlock::accept(Choreo::Visitor& v) {}
void Memory::accept(Choreo::Visitor& v) {}
void DMA::accept(Choreo::Visitor& v) {}
void ChunkAt::accept(Choreo::Visitor& v) {}
void Wait::accept(Choreo::Visitor& v) {}
void Call::accept(Choreo::Visitor& v) {}
void ForeachBlock::accept(Choreo::Visitor& v) {}

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

void CppSourceCode::accept(Choreo::Visitor& v) {
  v.Visit(*this);
}

void Program::accept(Choreo::Visitor& v) {
  v.BeforeVisit(*this);

  v.Visit(*this);
  for (auto ptr : nodes) ptr->accept(v);

  v.AfterVisit(*this);
}


} // end of namespace AST
