#include "loop_vectorize.hpp"
#include "vector_typeinfer.hpp"

namespace Choreo {

// LoopChecker
LoopChecker::LoopChecker() : VisitorWithScope("loop-checker") {}

bool LoopChecker::BeforeVisitImpl(AST::Node&) { return true; }

bool LoopChecker::AfterVisitImpl(AST::Node&) { return true; }

bool LoopChecker::Visit(AST::ForeachBlock& n) {
  if (!n.IsNorm()) AllNormLoop = false; // not all loops are normalized
  if (AST::NeedVectorize(n)) NeedVectorize = true;
  return true;
}

bool LoopChecker::HasVectorization() const { return NeedVectorize; }

bool LoopChecker::IsAllLoopNorm() const { return AllNormLoop; }

// LoopAnalysis
LoopAnalysis::LoopAnalysis(const ptr<SymbolTable> s_tab)
    : LoopVisitor(s_tab, "loopanalysis"), li(AST::Make<LoopInfo>()) {}
bool LoopAnalysis::Visit(AST::ForeachBlock& n) {
  auto iv = n.GetIV();
  li->iv2loop[InScopeName(iv->name)] = lname;

  ptr<Loop> loop = AST::Make<Loop>(lname, iv->name, iv->GetType());
  ptr<AST::Call> vectorize = nullptr;
  int vector_width = 1;
  if (AST::NeedVectorize(n, vectorize)) {
    vector_width = AST::GetIntLiteral(vectorize->GetArguments()[1])->ValS32();
    loop->need_vectorize = true;
    loop->can_vectorize =
        true; // assume it can be vectorized, will be checked later
  }

  loop->vector_width = vector_width;
  // update loop info
  li->loops.emplace(lname, loop);
  auto parent_loop_name = li->GetParentLoopName(lname);
  if (!parent_loop_name.empty()) {
    auto it = li->loops.find(parent_loop_name);
    if (it != li->loops.end()) { it->second->sub_loops.push_back(loop); }
  }

  n.loop = loop;
  return true;
}
ptr<LoopInfo> LoopAnalysis::GetLoopInfo() const { return li; }

// LoopVectorizeLegalityChecker
LoopVectorizeLegalityChecker::LoopVectorizeLegalityChecker(
    const ptr<SymbolTable> s_tab, ptr<LoopInfo> l, ptr<ScopedSCEVTable> s)
    : LoopVisitor(s_tab, "loop_vectorize_legality"), li(l), scev_table(s) {}

bool LoopVectorizeLegalityChecker::HasVectorize() const { return !all_illegal; }

bool LoopVectorizeLegalityChecker::NeedCheck() {
  auto loop = li->GetLoop(lname);
  if (!loop) return false;
  if (loop->NeedVectorize()) return true;
  return false;
}

bool LoopVectorizeLegalityChecker::CheckDataAccessAlignment(
    AST::DataAccess& n) {
  if (!n.AccessElement()) return true;

  auto span_ty = dyn_cast<SpannedType>(n.data->GetType());
  assert(span_ty && "data access should be on spanned type.");
  auto e_ty = span_ty->ElementType();
  auto n_ty = n.GetType();
  auto nds = n.GetDiversityShape();

  if (!nds.Varying()) return true;
  // In GCU target 3.0 the alignment requirement is equal to the vector size of
  // simd operands.
  auto alignment = SizeOf(e_ty) * cur_loop->GetVectorWidth();

  auto dim = span_ty->GetShape().DimCount();
  auto accmulate_size = sbe::nu(1);
  for (int i = dim - 1; i >= 0; --i) {
    auto indice = n.GetIndices()[i];
    auto indice_expr = dyn_cast<AST::Expr>(indice);
    assert(indice_expr && "index should be an expression.");
    auto iscev = dyn_cast<SCEVAddRecExpr>(indice_expr->GetSCEV());
    // step is 1 if we cannot determine the step
    auto step = sbe::nu(1);
    if (iscev)
      step = iscev->step->GetValue();
    else if (auto ival = dyn_cast<SCEVVal>(indice_expr->GetSCEV())) {
      if (ival->GetValue()->IsNumeric()) step = ival->GetValue();
    }

    if (i < int(dim - 1))
      accmulate_size = accmulate_size * span_ty->GetShape().ValueAt(i + 1);
    auto stride = step * accmulate_size * sbe::nu(SizeOf(e_ty));

    bool IsAligned = false;
    if (auto num_stride = dyn_cast<sbe::NumericValue>(stride)) {
      if (num_stride->Value() % alignment == 0) { IsAligned = true; }
    }

    if (!IsAligned) {
      if (debug_visit)
        dbgs() << indent << "unaligned data access: " << STR(n)
               << ", stride at the " << i + 1 << "th indice: " << STR(stride)
               << "{" << alignment << "}.\n";
      cur_loop->can_vectorize = false;
      return false;
    }
  }

  return true;
}

/* One foreachblock is legal to be vectorized when all the following conditions
are met:
  0. it is marked to be vectorized.
  1. it is a normalized loop.
  2. it is an innermost loop.
  3. it does not contain unsupported statements, e.g., DMA, wait, trigger,
  return, select, etc.
  4. it does not contain unsupported jump statements, e.g., break, continue,
  etc(currently).
  5. it does not contain call statments(currently).
  5. all data accesses in the loop are aligned if target arch does not support
  unaligned access
  6. all assignments inside loop should not assign to a outside-defined
  variable(currently), this will prevent any reduction computation.
  7. all data accesses inside one same loop should has same element type.
  8. the vector width of vectorized data access should be consistent with SIMD
  width of target arch.
  9. it ignores data dependence analysis for now. */
bool LoopVectorizeLegalityChecker::Visit(AST::ForeachBlock& n) {
  TraceEachVisit(n);
  cur_loop = n.loop;
  if (debug_visit) {
    indent = "";
    dbgs() << indent << "entering loop " << cur_loop->IVName() << "\n";
    indent += "  ";
  }
  // reset data type of data accesses for each loop
  data_type = BaseType::UNKNOWN;

  if (!n.IsNorm() || !cur_loop) {
    Error1(n.LOC(), "cannot vectorize non-normalized loop.");
    cur_loop->can_vectorize = false;
    return false;
  }
  // skip if the loop does not need to be vectorized
  if (!cur_loop->NeedVectorize()) {
    if (debug_visit)
      dbgs() << indent << "skip loop " << cur_loop->IVName()
             << " without vectorization hint.\n";
    cur_loop->can_vectorize = false;
    return true;
  }

  if (!li->IsInnermostLoop(cur_loop->loop_name)) {
    Error1(n.LOC(), "only innermost loop can be vectorized.");
    cur_loop->can_vectorize = false;
    return false;
  }

  auto vector_width = cur_loop->GetVectorWidth();
  auto IsPowerOf2 = [](int n) { return (n > 0) && ((n & (n - 1)) == 0); };
  ptr<AST::Call> vectorize = nullptr;
  AST::NeedVectorize(n, vectorize);
  assert(vectorize && "vectorize hint should exist.");

  if (vector_width <= 1 || !IsPowerOf2(vector_width)) {
    Error1(vectorize->GetArguments()[1]->LOC(),
           "vector width should be a power of 2 greater than 1.");
    cur_loop->can_vectorize = false;
    return false;
  }

  return true;
}

bool LoopVectorizeLegalityChecker::Visit(AST::WhileBlock& n) {
  TraceEachVisit(n);
  Error1(n.LOC(), "while loop is not supported for vectorization.");
  cur_loop->can_vectorize = false;
  return false;
}

bool LoopVectorizeLegalityChecker::Visit(AST::DMA& n) {
  TraceEachVisit(n);
  if (!NeedCheck()) return true;
  Error1(n.LOC(), "DMA is not supported for vectorization.");
  cur_loop->can_vectorize = false;
  return false;
}

bool LoopVectorizeLegalityChecker::Visit(AST::Wait& n) {
  TraceEachVisit(n);
  if (!NeedCheck()) return true;
  Error1(n.LOC(), "data access is not supported for vectorization.");
  cur_loop->can_vectorize = false;
  return false;
}

bool LoopVectorizeLegalityChecker::Visit(AST::Trigger& n) {
  TraceEachVisit(n);
  if (!NeedCheck()) return true;
  Error1(n.LOC(), "trigger is not supported for vectorization.");
  cur_loop->can_vectorize = false;
  return false;
}

bool LoopVectorizeLegalityChecker::Visit(AST::Rotate& n) {
  TraceEachVisit(n);
  if (!NeedCheck()) return true;
  Error1(n.LOC(), "rotate is not supported for vectorization.");
  cur_loop->can_vectorize = false;
  return false;
}

bool LoopVectorizeLegalityChecker::Visit(AST::Return& n) {
  TraceEachVisit(n);
  if (!NeedCheck()) return true;
  Error1(n.LOC(), "return is not supported for vectorization.");
  cur_loop->can_vectorize = false;
  return false;
}

bool LoopVectorizeLegalityChecker::Visit(AST::Select& n) {
  TraceEachVisit(n);
  if (!NeedCheck()) return true;
  Error1(n.LOC(), "select is not supported for vectorization.");
  cur_loop->can_vectorize = false;
  return false;
}

bool LoopVectorizeLegalityChecker::Visit(AST::IncrementBlock& n) {
  TraceEachVisit(n);
  if (!NeedCheck()) return true;
  Error1(n.LOC(), "increment block is not supported for vectorization.");
  cur_loop->can_vectorize = false;
  return false;
}

bool LoopVectorizeLegalityChecker::Visit(AST::Synchronize& n) {
  TraceEachVisit(n);
  if (!NeedCheck()) return true;
  Error1(n.LOC(), "synchronize is not supported for vectorization.");
  cur_loop->can_vectorize = false;
  return false;
}

bool LoopVectorizeLegalityChecker::Visit(AST::InThreadsBlock& n) {
  TraceEachVisit(n);
  if (!NeedCheck()) return true;
  Error1(n.LOC(), "in-threads block is not supported for vectorization.");
  cur_loop->can_vectorize = false;
  return false;
}

bool LoopVectorizeLegalityChecker::Visit(AST::ParallelBy& n) {
  TraceEachVisit(n);
  if (!NeedCheck()) return true;
  Error1(n.LOC(), "parallel-by is not supported for vectorization.");
  cur_loop->can_vectorize = false;
  return false;
}

void LoopVectorizeLegalityChecker::AddLoopUse(std::string sym, location loc) {
  if (loop_uses.find(sym) == loop_uses.end()) {
    loop_uses.emplace(sym, std::vector<location>{loc});
  } else {
    loop_uses[sym].push_back(loc);
  }
}

void LoopVectorizeLegalityChecker::FindLoopUses(ptr<AST::Node> n) {
  if (!n) return;
  if (auto id = AST::GetIdentifier(n)) {
    auto id_sym = InScopeName(id->name);
    AddLoopUse(id_sym, id->LOC());
  } else if (auto call = AST::GetCall(n)) {
    for (auto arg : call->GetArguments()) FindLoopUses(arg);
  } else if (auto da = dyn_cast<AST::DataAccess>(n)) {
    if (!da->AccessElement()) FindLoopUses(da->data);
  } else if (auto e = dyn_cast<AST::Expr>(n)) {
    FindLoopUses(e->GetC());
    FindLoopUses(e->GetL());
    FindLoopUses(e->GetR());
  }
}

void LoopVectorizeLegalityChecker::AddLoopDef(std::string sym, location loc) {
  if (loop_defs.find(sym) == loop_defs.end()) {
    loop_defs.emplace(sym, std::vector<location>{loc});
  } else {
    loop_defs[sym].push_back(loc);
  }
}

bool LoopVectorizeLegalityChecker::Visit(AST::NamedVariableDecl& n) {
  TraceEachVisit(n);
  if (!NeedCheck()) return true;
  if (n.IsArray()) {
    if (debug_visit)
      dbgs() << indent
             << "Array definition is not supported in vectorization.\n";
    cur_loop->can_vectorize = false;
    return true;
  } else {
    auto name_sym = InScopeName(n.name_str);
    FindLoopUses(n.init_expr);
    if (loop_uses.find(name_sym) != loop_uses.end()) {
      if (debug_visit)
        dbgs() << indent
               << "variable " + name_sym +
                      " is used before its declaration in the loop.\n";
      cur_loop->can_vectorize = false;
      return true;
    }
    AddLoopDef(name_sym, n.LOC());
  }

  return true;
}

bool LoopVectorizeLegalityChecker::Visit(AST::Assignment& n) {
  TraceEachVisit(n);
  if (!NeedCheck()) return true;

  if (!n.AssignToDataElement()) {
    auto name_sym = InScopeName(n.GetName());
    FindLoopUses(n.value);
    if (loop_uses.find(name_sym) != loop_uses.end()) {
      if (debug_visit)
        dbgs() << indent
               << "variable " + name_sym +
                      " is used before its assignment in the loop.\n";
      cur_loop->can_vectorize = false;
      return true;
    }
    AddLoopDef(name_sym, n.LOC());
  } else {
    auto value_ds = n.value->GetDiversityShape();
    auto lhs_ds = n.da->GetDiversityShape();
    // if there is a varying value assigned to a non-varying data access,
    // it may need a reduction operation, which is not supported currently.
    if (value_ds.Varying() && !lhs_ds.Varying()) {
      if (debug_visit)
        dbgs() << indent
               << "cannot assign a varying value to a non-varying data access: "
               << STR(n) << "\n";
      cur_loop->can_vectorize = false;
      return true;
    }
  }

  return true;
}

bool LoopVectorizeLegalityChecker::Visit(AST::Call& n) {
  TraceEachVisit(n);
  if (!NeedCheck()) return true;
  if (n.IsAnno()) return true;
  if (debug_visit)
    dbgs() << indent
           << "call is currently not supported in loop vectorization\n";

  cur_loop->can_vectorize = false;
  return true;
}

bool LoopVectorizeLegalityChecker::Visit(AST::DataAccess& n) {
  TraceEachVisit(n);
  // check mixed type
  if (!n.AccessElement()) return true;
  auto elem_ty = GetBaseType(*n.GetType());
  if (data_type == BaseType::UNKNOWN)
    data_type = elem_ty;
  else if (elem_ty != data_type) {
    if (debug_visit)
      dbgs() << indent
             << "vectorization of mixed type is currently not supported\n";
    cur_loop->can_vectorize = false;
    return true;
  }
  auto vector_width = cur_loop->GetVectorWidth();
  auto vector_size = SizeOf(elem_ty) * vector_width;
  auto single_vector_size = CCtx().GetSingleVectorByteSize();

  if (vector_size != single_vector_size &&
      vector_size != 2 * single_vector_size &&
      vector_size != 4 * single_vector_size) {
    if (debug_visit)
      dbgs() << indent
             << "vectorization with unsupported vector width: " << vector_width
             << "\n";

    cur_loop->can_vectorize = false;
    return true;
  }

  // alignment check, GCU4 supports unaligned simd memory access
  if (CCtx().GetArch() != TargetArch::GCU4) {
    if (!CheckDataAccessAlignment(n)) {
      cur_loop->can_vectorize = false;
      return true;
    }
  }

  return true;
}

bool LoopVectorizeLegalityChecker::BeforeAfterVisitImpl(AST::Node& n) {
  if (auto fb = dyn_cast<AST::ForeachBlock>(&n)) {
    auto loop = fb->loop;
    if (loop && loop->CanVectorize()) {
      if (debug_visit)
        dbgs() << indent << "loop " << loop->IVName()
               << " is legal to be vectorized\n";
      all_illegal = false;
    }
  }
  return true;
}

// BranchSimplicition
BranchSimplicition::BranchSimplicition(const ptr<SymbolTable> s_tab)
    : LoopVisitor(s_tab, "branch-simplicition") {}

bool BranchSimplicition::Visit(AST::IfElseBlock& n) {
  TraceEachVisit(n);
  if (n.HasElse()) return true;

  auto if_stmts = n.if_stmts;

  if (if_stmts && if_stmts->Count() == 1) {
    if (auto single_IF = dyn_cast<AST::IfElseBlock>(if_stmts->values[0])) {
      if (!single_IF->HasElse()) {
        auto pred_a = n.GetPred();
        auto pred_b = single_IF->GetPred();
        auto new_pred = AST::Make<AST::Expr>(n.LOC(), "&&", pred_a, pred_b);
        new_pred->SetType(pred_a->GetType());
        new_pred->SetDiversityShape(ComputeDiversityShape(
            pred_a->GetDiversityShape(), pred_b->GetDiversityShape()));
        n.pred = new_pred;
        n.if_stmts = single_IF->if_stmts;
      }
    }
  }

  return true;
}

// Linearizer
bool Linearizer::NeedLinearize() {
  auto loop = li->GetLoop(lname);
  if (!loop) return false;
  if (loop->CanVectorize()) return true;
  return false;
}

Linearizer::Linearizer(const ptr<SymbolTable> s_tab, ptr<LoopInfo> l,
                       ptr<DiversityInfo> d)
    : LoopVisitor(s_tab, "linearizer"), li(l), di(d) {}

// linearize branch inside vectorized loops, all divergent branches will be
// linearized, its else branch will be removed and then inserted after the
// if-else block with a negated predicate
bool Linearizer::Visit(AST::MultiNodes& n) {
  TraceEachVisit(n);
  if (!NeedLinearize()) return true;

  for (size_t stmt_index = 0; stmt_index < n.Count(); ++stmt_index) {
    if (auto if_block = dyn_cast<AST::IfElseBlock>(n.SubAt(stmt_index))) {
      auto pred = if_block->GetPred();
      auto pred_ds = pred->GetDiversityShape();
      // if this is a uniform branch, keep it
      if (pred_ds.Uniform()) continue;
      assert(!pred_ds.Unknown());
      // if this is a divergent branch, we need to linearize it
      // Firstly, we need to normalize the if_else block to ensure it does not
      // have else branch. Secondly, we need to insert a negated if-else block
      // consecutively after the original if-else block.

      auto else_stmts = if_block->else_stmts;
      if (!else_stmts) continue;
      auto neg_pred = AST::Make<AST::Expr>(if_block->LOC(), "!", pred->Clone());
      neg_pred->SetType(pred->GetType());
      neg_pred->SetDiversityShape(pred_ds);
      auto neg_if_block =
          AST::Make<AST::IfElseBlock>(if_block->LOC(), neg_pred, else_stmts);
      n.Insert(neg_if_block, ++stmt_index);
      if_block->else_stmts = nullptr; // remove the else stmts
      if (debug_visit) {
        dbgs() << "[linearize] Inserted negated if-else block: "
               << STR(neg_if_block->pred)
               << " after original if-else block: " << STR(if_block->pred)
               << "\n";
      }
    }
  }

  return true;
}

// MaskGen
// it generates mask variables for divergent branches in vectorized loops.
// Mask variables are boolean vectors with the same width as SIMD width.
// The mask variable is true for lanes where the branch predicate is true,
// and false otherwise. The mask variable is used to control the execution of
// instructions inside the branch.
// For simplicity, we use a mask variable "exec" to represent the
// execution mask for all statements inside the loop. The "exec" mask is
// updated at the beginning of each divergent branch and restored at the end
// of the branch. We maintain a stack of masks to handle nested branches.
// The "exec" mask is initialized by comparsion between the loop IV and the loop
// bound.
MaskGen::MaskGen(const ptr<SymbolTable> s_tab, ptr<LoopInfo> l,
                 ptr<DiversityInfo> d)
    : LoopVisitor(s_tab, "mask"), li(l), di(d) {}

std::string MaskGen::MaskName() {
  static int mask_count = 0;
  return "mask" + std::to_string(mask_count++);
}

bool MaskGen::NeedTransform() {
  auto loop = li->GetLoop(lname);
  if (!loop) return false;
  if (loop->CanVectorize()) return true;
  return false;
}

bool MaskGen::HasDivergentBranch(AST::MultiNodes& n) {
  for (size_t stmt_index = 0; stmt_index < n.Count(); ++stmt_index) {
    if (auto if_block = dyn_cast<AST::IfElseBlock>(n.SubAt(stmt_index))) {
      auto pred_ds = if_block->GetPred()->GetDiversityShape();
      if (pred_ds.Divergent()) return true;
      auto then_true = HasDivergentBranch(*if_block->if_stmts);
      if (then_true) return true;
      if (if_block->else_stmts) {
        auto else_true = HasDivergentBranch(*if_block->else_stmts);
        if (else_true) return true;
      }
    }
  }
  return false;
}

ptr<AST::Expr> MaskGen::MakeMaskExpr(const location& loc,
                                     const ptr<AST::Node>& lhs,
                                     const ptr<AST::Node>& rhs,
                                     std::string op) {
  ptr<AST::Expr> mask = nullptr;
  if (op.empty()) {
    mask = AST::Make<AST::Expr>(loc, lhs);
  } else {
    mask = AST::Make<AST::Expr>(loc, op, lhs, rhs);
  }
  mask->SetType(MakeVectorType(BaseType::BOOL, cur_loop->GetVectorWidth()));
  mask->SetDiversityShape(DiversityShapeKind::DIVERGENT);
  return mask;
}

ptr<AST::Expr> MaskGen::MakeMaskIdExpr(const location& loc,
                                       const std::string& name) {
  auto mask = AST::MakeIdExpr(loc, name);
  mask->SetType(MakeVectorType(BaseType::BOOL, cur_loop->GetVectorWidth()));
  mask->SetDiversityShape(DiversityShapeKind::DIVERGENT);
  return mask;
}

ptr<AST::NamedVariableDecl> MaskGen::MakeMaskDecl(const location& loc,
                                                  const std::string& name,
                                                  const ptr<AST::Expr>& rhs) {
  auto mask_ty = MakeVectorType(BaseType::BOOL, cur_loop->GetVectorWidth());
  auto data_type = AST::Make<AST::DataType>(loc, BaseType::BOOL);
  data_type->SetType(mask_ty);
  auto mask =
      AST::Make<AST::NamedVariableDecl>(loc, name, data_type, nullptr, rhs);
  mask->SetType(mask_ty);
  mask->SetDiversityShape(DiversityShapeKind::DIVERGENT);
  mask->AddNote("masking");
  return mask;
}

ptr<AST::Assignment> MaskGen::MakeMaskAssign(const location& loc,
                                             const std::string& name,
                                             const ptr<AST::Expr>& rhs) {
  auto mask = AST::Make<AST::Assignment>(loc, name, rhs);
  auto mask_ty = MakeVectorType(BaseType::BOOL, cur_loop->GetVectorWidth());
  mask->SetDecl(false);
  mask->SetType(mask_ty);
  mask->SetDiversityShape(DiversityShapeKind::DIVERGENT);
  mask->da->SetType(mask_ty);
  mask->da->SetDiversityShape(DiversityShapeKind::DIVERGENT);
  mask->AddNote("masking");
  return mask;
}

// restore the execution mask after divergent branch
bool MaskGen::Visit(AST::MultiNodes& n) {
  TraceEachVisit(n);
  if (!NeedTransform()) return true;
  if (mask_stack.empty()) { return true; }

  auto mask = mask_stack.top();
  for (size_t stmt_index = 0; stmt_index < n.Count(); ++stmt_index) {
    if (auto if_block = dyn_cast<AST::IfElseBlock>(n.SubAt(stmt_index))) {
      if (!if_block->IsDivergent()) continue;

      auto exec = MakeMaskAssign(if_block->LOC(), "exec",
                                 MakeMaskIdExpr(if_block->LOC(), mask));
      n.Insert(exec, ++stmt_index);
      if (debug_visit)
        dbgs() << "[mask] Inserted exec assignment: " << STR(exec)
               << " after divergenet branch: " << STR(if_block->GetPred())
               << "\n";
    }
  }
  return true;
}

// initialize the execution mask at the beginning of vectorized loop
bool MaskGen::Visit(AST::ForeachBlock& n) {
  TraceEachVisit(n);
  cur_loop = n.loop;
  auto smi = cur_loop->smi;

  auto loc = n.stmts->LOC();
  assert(n.IsNorm() && "Loop should be normalized before MaskGen.");
  ptr<AST::Call> vectorize = nullptr;
  if (!AST::NeedVectorize(n, vectorize)) return true;
  assert(li->IsInnermostLoop(lname));

  auto mask_ty = MakeVectorType(BaseType::BOOL, cur_loop->GetVectorWidth());

  auto iv = n.GetIV();
  auto iv_ty = iv->GetType();
  auto upper_bound = GetSingleUpperBound(iv_ty);
  ptr<AST::NamedVariableDecl> loop_cond = nullptr;
  ptr<AST::Expr> mask_expr = nullptr;
  auto cur_mask_name = MaskName();
  // if the upper bound is a constant and is divisible by vector width
  if (auto ub_nu = dyn_cast<sbe::NumericValue>(upper_bound);
      ub_nu && ub_nu->Value() % cur_loop->GetVectorWidth() == 0) {
    // if there is no divergent branch inside the loop, masking is not needed
    if (!HasDivergentBranch(*n.stmts)) {
      if (debug_visit)
        dbgs() << "[mask] No divergent branch inside vectorized loop: "
               << n.GetIV()->name << ", skip mask generation.\n";
      return true;
    }
    auto bool_literal = AST::Make<AST::BoolLiteral>(loc, true);
    bool_literal->SetType(MakeBooleanType(true));
    bool_literal->SetDiversityShape(DiversityShapeKind::UNIFORM);
    mask_expr = MakeMaskExpr(n.LOC(), bool_literal);
    mask_expr->Note().emplace("broadcast",
                              std::to_string(cur_loop->GetVectorWidth()));
    loop_cond = MakeMaskDecl(loc, cur_mask_name, mask_expr);
    loop_cond->init_expr->SetType(MakeBooleanType(true));
    smi->all_true_masks.insert(cur_mask_name);
  } else {
    // otherwise create mask by comparing iv and upper bound
    auto ub_expr = AST::Make<AST::Expr>(n.LOC(), "ubound", iv);
    ub_expr->SetType(MakeIntegerType());
    ub_expr->SetDiversityShape(DiversityShapeKind::UNIFORM);
    mask_expr = MakeMaskExpr(n.LOC(), iv, ub_expr, "<=");
    loop_cond = MakeMaskDecl(loc, cur_mask_name, mask_expr);
  }

  SSTab().DefineSymbol(loop_cond->name_str, mask_ty);
  di->AssignSymbolShape(InScopeName(cur_mask_name),
                        loop_cond->GetDiversityShape());

  auto cur_mask =
      MakeMaskDecl(loc, "exec", MakeMaskIdExpr(loc, loop_cond->name_str));
  smi->SetMaskInScope(SSTab().ScopeName(), cur_mask_name);
  smi->SetVectorWidth(cur_loop->GetVectorWidth());

  SSTab().DefineSymbol(cur_mask->name_str, mask_ty);
  di->AssignSymbolShape(InScopeName(cur_mask->name_str),
                        cur_mask->GetDiversityShape());

  mask_stack.push(cur_mask_name);

  n.stmts->Insert(loop_cond, 0);
  n.stmts->Insert(cur_mask, 1);
  if (debug_visit) {
    dbgs() << "[mask] Inserted loop condition(exec): " << STR(loop_cond)
           << " at the beginning of loop: " << n.GetIV()->name << "\n";
    dbgs() << "[mask] Inserted scoped mask: " << STR(cur_mask)
           << " at the beginning of loop: " << n.GetIV()->name << "\n";
  }

  return true;
}

bool MaskGen::Visit(AST::IfElseBlock& n) {
  TraceEachVisit(n);
  if (!NeedTransform()) return true;
  auto smi = cur_loop->smi;
  auto loc = n.if_stmts->LOC();
  auto pred = n.GetPred();
  auto pred_ds = pred->GetDiversityShape();
  if (pred_ds.Uniform()) return true;
  assert(pred_ds.Divergent() && "predicate should be divergent in MaskGen.");
  assert(!n.HasElse() &&
         "if-else block should not have else branch in MaskGen.");

  // divergent branch
  auto top_mask = mask_stack.top();
  auto mask_expr =
      MakeMaskExpr(n.LOC(), MakeMaskIdExpr(loc, top_mask), pred, "&&");
  auto cur_mask = MakeMaskDecl(n.LOC(), MaskName(), mask_expr);

  SSTab().DefineSymbol(
      cur_mask->name_str,
      MakeVectorType(BaseType::BOOL, cur_loop->GetVectorWidth()));
  di->AssignSymbolShape(InScopeName(cur_mask->name_str),
                        cur_mask->GetDiversityShape());
  // push the current mask to the stack
  mask_stack.push(cur_mask->name_str);

  auto cur_exec =
      MakeMaskAssign(n.LOC(), "exec", MakeMaskIdExpr(loc, cur_mask->name_str));

  auto stmts = n.if_stmts;
  stmts->Insert(cur_mask, 0);
  stmts->Insert(cur_exec, 1);

  smi->SetMaskInScope(SSTab().ScopeName(), cur_mask->name_str);

  if (debug_visit) {
    dbgs() << "[mask] Inserted scoped mask: " << STR(cur_mask)
           << " at the beginning of divergent branch: " << STR(pred) << "\n";
    dbgs() << "[mask] Inserted exec assignment: " << STR(cur_exec)
           << " after divergenet branch: " << STR(pred) << "\n";
  }
  return true;
}

bool MaskGen::Visit(AST::DataAccess& n) {
  TraceEachVisit(n);
  if (!NeedTransform()) return true;
  if (!n.AccessElement()) return true;
  auto smi = cur_loop->smi;
  auto span_ty = dyn_cast<SpannedType>(n.data->GetType());
  assert(span_ty && "data access should be on spanned type.");
  auto e_ty = span_ty->ElementType();
  smi->SetMaskEType(e_ty);
  return true;
}

bool MaskGen::BeforeAfterVisitImpl(AST::Node& n) {
  if (auto if_block = dyn_cast<AST::IfElseBlock>(&n)) {
    if (if_block->IsDivergent() && mask_stack.size() > 1) {
      // pop the mask stack
      mask_stack.pop();
    }
  }

  return true;
}

// LoopVectorizer
LoopVectorizer::LoopVectorizer()
    : VisitorWithSymTab("loop", CCtx().GetGlobalSymbolTable()) {}

bool LoopVectorizer::BeforeVisitImpl(AST::Node&) { return true; }
bool LoopVectorizer::AfterVisitImpl(AST::Node&) { return true; }

bool LoopVectorizer::RunOnProgram(AST::Node& root) {
  if (!isa<AST::Program>(&root)) {
    Error1(root.LOC(), "Not running a choreo program.");
    return false;
  }
  if (CCtx().GetTarget() != CompileTarget::Topscc) {
    Error1(root.LOC(),
           "Loop vectorization transformations are only for topscc target.");
    return true;
  }
  if (prt_visitor) dbgs() << "|- " << GetName() << NewL;
  debug_visit |= CCtx().TraceVectorize();
  LoopChecker lc;
  lc.SetDebugVisit(debug_visit);
  lc.SetTraceVisit(trace_visit);
  root.accept(lc);
  if (HasError() || abend_after) return false;
  if (!lc.IsAllLoopNorm() || !lc.HasVectorization()) return true;
  if (prt_visitor) dbgs() << " |- " << lc.GetName() << NewL;

  LoopAnalysis la(SymTab());
  la.SetDebugVisit(debug_visit);
  la.SetTraceVisit(trace_visit);
  root.accept(la);
  auto li = la.GetLoopInfo();
  if (debug_visit) {
    dbgs() << "LoopInfo:\n";
    li->dump(dbgs());
    dbgs() << "\n";
  }
  if (HasError() || abend_after) return false;
  if (prt_visitor) dbgs() << " |- " << la.GetName() << NewL;

  if (lc.HasVectorization()) {
    if (debug_visit) dbgs() << "\n[diversity] start diversity analysis.\n";
    DiversityAnalysisHandler da(SymTab(), li);
    da.SetDebugVisit(debug_visit);
    da.SetTraceVisit(trace_visit);
    da.RunOnProgram(root);
    if (prt_visitor) dbgs() << " |- " << da.GetName() << NewL;
    if (HasError() || abend_after) return false;
    auto di = da.GetDiversityAnalysis();

    if (debug_visit) dbgs() << "\n[scev] start scalar evolution analysis.\n";
    ScalarEvolutionAnalysis sba(SymTab(), li);
    sba.SetDebugVisit(debug_visit);
    sba.SetTraceVisit(trace_visit);
    root.accept(sba);
    if (prt_visitor) dbgs() << " |- " << sba.GetName() << NewL;
    if (HasError() || abend_after) return false;
    auto scev_tab = sba.GetScevTab();

    if (debug_visit)
      dbgs() << "\n[legality] start loop vectorization legality check.\n";
    LoopVectorizeLegalityChecker lvlc(SymTab(), li, scev_tab);
    lvlc.SetDebugVisit(debug_visit);
    lvlc.SetTraceVisit(trace_visit);
    root.accept(lvlc);
    if (HasError() || abend_after) return false;
    if (prt_visitor) dbgs() << " |- " << lvlc.GetName() << NewL;
    if (!lvlc.HasVectorize()) {
      if (debug_visit)
        dbgs() << "[loop vectorization] legality check failed. Skip "
                  "vectorization.\n";
      return true;
    }

    if (debug_visit) dbgs() << "\n[vinfer] start inferring vector types.\n";
    VectorTypeInfer vti(SymTab(), li, di);
    vti.SetDebugVisit(debug_visit);
    vti.SetTraceVisit(trace_visit);
    root.accept(vti);
    if (prt_visitor) dbgs() << " |- " << vti.GetName() << NewL;
    if (HasError() || abend_after) return false;

    if (debug_visit) dbgs() << "\n[linearize] start linearization.\n";
    Linearizer ln(SymTab(), li, di);
    ln.SetDebugVisit(debug_visit);
    ln.SetTraceVisit(trace_visit);
    root.accept(ln);
    if (prt_visitor) dbgs() << " |- " << ln.GetName() << NewL;
    if (HasError() || abend_after) return false;

    BranchSimplicition bs(SymTab());
    bs.SetDebugVisit(debug_visit);
    bs.SetTraceVisit(trace_visit);
    root.accept(bs);
    if (prt_visitor) dbgs() << " |- " << bs.GetName() << NewL;
    if (HasError() || abend_after) return false;

    if (debug_visit) dbgs() << "\n[mask] start geneating masks.\n";
    MaskGen mg(SymTab(), li, di);
    mg.SetDebugVisit(debug_visit);
    mg.SetTraceVisit(trace_visit);
    root.accept(mg);
    if (prt_visitor) dbgs() << " |- " << mg.GetName() << NewL;
    if (HasError() || abend_after) return false;
  }

  return true;
}

} // namespace Choreo
