#include "visitor.hpp"

namespace Choreo {

namespace AST {

void MultiNodes::accept(Choreo::Visitor& v) {
  for (auto& sub : values) sub->accept(v);
  v.Visit(*this);
}

void MultiValues::accept(Choreo::Visitor& v) {
  for (auto& sub : values) sub->accept(v);
  v.Visit(*this);
}

void Boolean::accept(Choreo::Visitor& v) { v.Visit(*this); }
void IntLiteral::accept(Choreo::Visitor& v) { v.Visit(*this); }

void Expr::accept(Choreo::Visitor& v) {
  if (value_c) value_c->accept(v);
  if (value_l) value_l->accept(v);

  assert(value_r && "invalid expression found.");
  value_r->accept(v);

  v.Visit(*this);
}

void MultiDimSpans::accept(Choreo::Visitor& v) {
  v.BeforeVisit(*this);

  if (list) list->accept(v);
  v.Visit(*this);

  v.AfterVisit(*this);
}

void NamedTypeDecl::accept(Choreo::Visitor& v) {
  if (init_expr) init_expr->accept(v);
  v.Visit(*this);
}

void NamedVariableDecl::accept(Choreo::Visitor& v) {
  if (mem) mem->accept(v);
  if (type) type->accept(v);
  if (init_expr) init_expr->accept(v);

  v.Visit(*this);
}

void IntTuple::accept(Choreo::Visitor& v) {
  vlist->accept(v);
  v.Visit(*this);
}

void Assignment::accept(Choreo::Visitor& v) {
  value->accept(v);
  v.Visit(*this);
}

void IntIndex::accept(Choreo::Visitor& v) {
  v.BeforeVisit(*this);
  value->accept(v);
  v.Visit(*this);
  v.AfterVisit(*this);
}

void DataType::accept(Choreo::Visitor& v) {
  if (mdspan_type) mdspan_type->accept(v);
  v.Visit(*this);
}

void Identifier::accept(Choreo::Visitor& v) { v.Visit(*this); }

void Parameter::accept(Choreo::Visitor& v) {
  v.BeforeVisit(*this);
  type->accept(v);
  if (sym) sym->accept(v);
  v.Visit(*this);
  v.AfterVisit(*this);
}

void ParamList::accept(Choreo::Visitor& v) {
  for (auto p : values) p->accept(v);

  v.Visit(*this);
}

void IfElse::accept(Choreo::Visitor& v) { (void)v; }

void ParallelBy::accept(Choreo::Visitor& v) {
  v.BeforeVisit(*this);
  v.Visit(*this);

  stmts->accept(v);

  v.AfterVisit(*this);
}

void WhereBind::accept(Choreo::Visitor& v) {
  v.BeforeVisit(*this);
  lhs->accept(v);
  rhs->accept(v);
  v.Visit(*this);
  v.AfterVisit(*this);
}

void WithIn::accept(Choreo::Visitor& v) {
  v.BeforeVisit(*this);
  in->accept(v);
  // have to handle identifier/matcher inside 'within'
  v.Visit(*this);
  v.AfterVisit(*this);
}

void WithBlock::accept(Choreo::Visitor& v) {
  v.BeforeVisit(*this);
  withins->accept(v);
  if (reqs) reqs->accept(v);
  v.Visit(*this);
  stmts->accept(v);
  v.AfterVisit(*this);
}

void Memory::accept(Choreo::Visitor& v) { v.Visit(*this); }

void DMA::accept(Choreo::Visitor& v) {
  from->accept(v);
  to->accept(v);
  v.Visit(*this);
}

void ChunkAt::accept(Choreo::Visitor& v) {
  // note: visit the positions inside
  v.Visit(*this);
}

void Select::accept(Choreo::Visitor& v) {
  select_factor->accept(v);
  span_expr_list->accept(v);
  v.Visit(*this);
}

void Wait::accept(Choreo::Visitor& v) {
  v.BeforeVisit(*this);
  targets->accept(v);
  v.Visit(*this);
  v.AfterVisit(*this);
}

void Return::accept(Choreo::Visitor& v) {
  if (value) value->accept(v);
  v.Visit(*this);
}

void Call::accept(Choreo::Visitor& v) {
  v.BeforeVisit(*this);
  arguments->accept(v);
  v.Visit(*this);
  v.AfterVisit(*this);
}

void LoopRange::accept(Choreo::Visitor& v) {
  iv->accept(v);
  v.Visit(*this);
}

void ForeachBlock::accept(Choreo::Visitor& v) {
  v.BeforeVisit(*this);
  ranges->accept(v);
  v.Visit(*this);
  stmts->accept(v);
  v.AfterVisit(*this);
}

void FunctionDecl::accept(Choreo::Visitor& v) {
  params->accept(v);
  ret_type->accept(v);

  v.Visit(*this);
}

void ChoreoFunction::accept(Choreo::Visitor& v) {
  v.BeforeVisit(*this);

  f_decl.accept(v);
  stmts->accept(v);

  v.Visit(*this);

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

}  // end of namespace Choreo
