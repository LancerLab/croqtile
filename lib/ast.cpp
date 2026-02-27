#include "visitor.hpp"

namespace Choreo {

namespace AST {

void MultiNodes::accept(Choreo::Visitor& v) {
  v.BeforeVisit(*this);

  for (auto& sub : values) sub->accept(v);
  v.Visit(*this);

  v.AfterVisit(*this);
}

void MultiValues::accept(Choreo::Visitor& v) {
  for (auto& sub : values) sub->accept(v);
  v.Visit(*this);
}

void NoValue::accept(Choreo::Visitor& v) { v.Visit(*this); }
void BoolLiteral::accept(Choreo::Visitor& v) { v.Visit(*this); }
void IntLiteral::accept(Choreo::Visitor& v) { v.Visit(*this); }
void FloatLiteral::accept(Choreo::Visitor& v) { v.Visit(*this); }
void StringLiteral::accept(Choreo::Visitor& v) { v.Visit(*this); }

void Expr::accept(Choreo::Visitor& v) {
  v.BeforeVisit(*this);

  if (value_c) value_c->accept(v);
  if (value_l) value_l->accept(v);

  assert(value_r && "invalid expression found.");
  value_r->accept(v);

  v.Visit(*this);
}

void AttributeExpr::accept(Choreo::Visitor& v) {
  v.BeforeVisit(*this);
  if (attr_values) attr_values->accept(v);
  v.Visit(*this);
}

void CastExpr::accept(Choreo::Visitor& v) {
  v.BeforeVisit(*this);
  assert(!GetC());
  assert(!GetL());

  assert(GetR() && "invalid expression found.");
  GetR()->accept(v);
  // visit the inner expr first

  v.Visit(*this);
}

void MultiDimSpans::accept(Choreo::Visitor& v) {
  v.BeforeVisit(*this);

  if (list) list->accept(v);
  v.Visit(*this);

  v.AfterVisit(*this);
}

void NamedTypeDecl::accept(Choreo::Visitor& v) {
  v.BeforeVisit(*this);

  if (init_expr) init_expr->accept(v);
  v.Visit(*this);

  v.AfterVisit(*this);
}

void NamedVariableDecl::accept(Choreo::Visitor& v) {
  v.BeforeVisit(*this);

  if (mem) mem->accept(v);
  if (array_dims) array_dims->accept(v);
  // be careful of the accpeting orders
  if (init_value) init_value->accept(v);
  if (type) type->accept(v);
  if (init_expr) init_expr->accept(v);

  v.Visit(*this);

  v.AfterVisit(*this);
}

void IntTuple::accept(Choreo::Visitor& v) {
  v.BeforeVisit(*this);
  vlist->accept(v);
  v.Visit(*this);
  v.AfterVisit(*this);
}

void SpanAs::accept(Choreo::Visitor& v) {
  v.BeforeVisit(*this);

  id->accept(v);
  //  nid->accept(v);
  list->accept(v);
  v.Visit(*this);

  v.AfterVisit(*this);
}

void DataAccess::accept(Choreo::Visitor& v) {
  v.BeforeVisit(*this);

  data->accept(v);
  if (indices) indices->accept(v);

  v.Visit(*this);

  v.AfterVisit(*this);
}

void Assignment::accept(Choreo::Visitor& v) {
  v.BeforeVisit(*this);

  da->accept(v);
  value->accept(v);
  v.Visit(*this);

  v.AfterVisit(*this);
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

void IfElseBlock::accept(Choreo::Visitor& v) {
  v.BeforeVisit(*this);
  pred->accept(v);
  v.Visit(*this);
  if (if_stmts) if_stmts->accept(v);
  v.InMidVisit(*this);
  if (else_stmts) else_stmts->accept(v);
  v.AfterVisit(*this);
}

void ParallelBy::accept(Choreo::Visitor& v) {
  v.BeforeVisit(*this);

  if (bound_expr) bound_expr->accept(v);
  if (cmpt_bounds) cmpt_bounds->accept(v);
  v.Visit(*this);

  // handle identifier/matcher inside 'parallelby'

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
  v.BeforeVisit(*this);

  if (operation == ".pad") {
    auto pc = cast<PadConfig>(config);
    pc->pad_high->accept(v);
    pc->pad_low->accept(v);
    pc->pad_mid->accept(v);
    pc->GetPadValue()->accept(v);
  }

  if (operation != ".any") {
    from->accept(v);
    to->accept(v);
  }

  v.Visit(*this);
  v.AfterVisit(*this);
}

void MMA::accept(Choreo::Visitor& v) {
  v.BeforeVisit(*this);

  if (operation->IsKind(MMAOperation::Fill)) {
    operation->FillingValue()->accept(v);
    if (operation->FillingArrayDims()) operation->FillingArrayDims()->accept(v);
  } else if (operation->IsKind(MMAOperation::Load))
    operation->LoadFrom()->accept(v);
  else if (operation->IsKind(MMAOperation::Exec)) {
    if (operation->HasScale()) {
      if (operation->ScaleA()) operation->ScaleA()->accept(v);
      if (operation->ScaleB()) operation->ScaleB()->accept(v);
    }
  } else if (operation->IsKind(MMAOperation::Store))
    operation->StoreTo()->accept(v);

  v.Visit(*this);
  v.AfterVisit(*this);
}

Choreo::AST::SpannedOperation::~SpannedOperation() = default;

// this is not a visitor type that must be invoked manually
void SOP::Tiling::accept(Visitor& v) { tfactor->accept(v); }

void SOP::TileAt::accept(Visitor& v) {
  tfactor->accept(v);
  indices->accept(v);
}

void SOP::SubSpan::accept(Visitor& v) {
  subspan->accept(v);
  if (indices) indices->accept(v);
  if (steps) steps->accept(v);
}

void SOP::ModSpan::accept(Visitor& v) {
  subspan->accept(v);
  if (indices) indices->accept(v);
  if (steps) steps->accept(v);
}

void SOP::View::accept(Visitor& v) {
  subspan->accept(v);
  if (offsets) offsets->accept(v);
  if (strides) strides->accept(v);
}

void SOP::Reshape::accept(Visitor& v) { newspan->accept(v); }

void ChunkAt::accept(Choreo::Visitor& v) {
  // handle span_as
  if (sa) sa->accept(v);
  // indices seems to be always present, check the size.
  if (indices && indices->Count() > 0) indices->accept(v);
  // note: visit the spanned_operations inside
  v.Visit(*this);
}

void Select::accept(Choreo::Visitor& v) {
  v.BeforeVisit(*this);
  select_factor->accept(v);
  expr_list->accept(v);
  v.Visit(*this);
  v.AfterVisit(*this);
}

void Wait::accept(Choreo::Visitor& v) {
  v.BeforeVisit(*this);
  targets->accept(v);
  v.Visit(*this);
  v.AfterVisit(*this);
}

void Trigger::accept(Choreo::Visitor& v) {
  v.BeforeVisit(*this);
  targets->accept(v);
  v.Visit(*this);
  v.AfterVisit(*this);
}

void Break::accept(Choreo::Visitor& v) {
  v.BeforeVisit(*this);
  v.Visit(*this);
  v.AfterVisit(*this);
}

void Continue::accept(Choreo::Visitor& v) {
  v.BeforeVisit(*this);
  v.Visit(*this);
  v.AfterVisit(*this);
}

void Return::accept(Choreo::Visitor& v) {
  v.BeforeVisit(*this);
  if (value) value->accept(v);
  v.Visit(*this);
  v.AfterVisit(*this);
}

void Call::accept(Choreo::Visitor& v) {
  v.BeforeVisit(*this);
  if (template_args) template_args->accept(v);
  arguments->accept(v);
  v.Visit(*this);
  v.AfterVisit(*this);
}

void Rotate::accept(Choreo::Visitor& v) {
  v.BeforeVisit(*this);
  ids->accept(v);
  v.Visit(*this);
  v.AfterVisit(*this);
}

void Synchronize::accept(Choreo::Visitor& v) {
  v.BeforeVisit(*this);
  v.Visit(*this);
  v.AfterVisit(*this);
}

void LoopRange::accept(Choreo::Visitor& v) {
  if (lbound) lbound->accept(v);
  if (ubound) ubound->accept(v);
  iv->accept(v);
  v.Visit(*this);
}

void ForeachBlock::accept(Choreo::Visitor& v) {
  v.BeforeVisit(*this);
  ranges->accept(v);
  if (suffixs) suffixs->accept(v);
  v.Visit(*this);
  if (stmts) stmts->accept(v);
  v.AfterVisit(*this);
}

void InThreadsBlock::accept(Choreo::Visitor& v) {
  v.BeforeVisit(*this);
  pred->accept(v);
  v.Visit(*this);
  if (stmts) stmts->accept(v);
  v.AfterVisit(*this);
}

void WhileBlock::accept(Choreo::Visitor& v) {
  v.BeforeVisit(*this);
  pred->accept(v);
  v.Visit(*this);
  if (stmts) stmts->accept(v);
  v.AfterVisit(*this);
}

void IncrementBlock::accept(Choreo::Visitor& v) {
  v.BeforeVisit(*this);
  bvs->accept(v);
  pred->accept(v);
  v.Visit(*this);
  if (stmts) stmts->accept(v);
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

void DeviceFunctionDecl::accept(Choreo::Visitor& v) {
  v.BeforeVisit(*this);
  v.Visit(*this);
}

void Program::accept(Choreo::Visitor& v) {
  v.BeforeVisit(*this);

  v.Visit(*this);
  nodes->accept(v);

  v.AfterVisit(*this);
}

} // end of namespace AST

} // end of namespace Choreo
