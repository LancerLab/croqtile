#include "loop_vectorize.hpp"
#include "vector_typeinfer.hpp"

namespace Choreo {

// LoopChecker
VectorizationHintChecker::VectorizationHintChecker()
    : VisitorWithScope("hint-checker") {}

bool VectorizationHintChecker::BeforeVisitImpl(AST::Node&) { return true; }

bool VectorizationHintChecker::AfterVisitImpl(AST::Node&) { return true; }

bool VectorizationHintChecker::Visit(AST::ForeachBlock& n) {
  if (!n.IsNorm()) has_hint = false; // not all loops are normalized
  if (AST::HasVectorizationHint(n)) has_hint = true;
  return true;
}

bool VectorizationHintChecker::HasVectorizationHint() const { return has_hint; }

// LoopAnalysis
LoopAnalysis::LoopAnalysis(const ptr<SymbolTable>& s_tab)
    : VisitorWithSymTab("loop-analysis", s_tab), li(AST::Make<LoopInfo>()) {}

bool LoopAnalysis::BeforeVisitImpl(AST::Node&) { return true; }

bool LoopAnalysis::AfterVisitImpl(AST::Node& n) {
  if (isa<AST::ForeachBlock>(&n)) {
    parent_loop_name = li->GetParentLoop(parent_loop_name)
                           ? li->GetParentLoop(parent_loop_name)->LoopName()
                           : "";
  }
  return true;
}

bool LoopAnalysis::Visit(AST::ForeachBlock& n) {
  auto iv = n.GetIV();

  ptr<Loop> loop = n.loop;
  auto loop_name = loop->LoopName();
  loop->SetIVSym(InScopeName(iv->name));
  loop->SetIVType(iv->GetType());

  ptr<AST::AttributeExpr> vectorization_hint = nullptr;
  if (AST::HasVectorizationHint(n, vectorization_hint)) {
    loop->SetHasVectorizationHint(true);
    loop->SetNeedVectorize(true);
    loop->SetVectorFactor(
        AST::GetIntLiteral(vectorization_hint->AttrValueAt(1))->ValS32());
  } else {
    // it needs trying to vectorize innermost loops later, but currently
    // loop info is not ready, it cannot used to judge whether a loop is
    // innermost or not. In later simple loop checker, we will set
    // need_vectorize for innermost loops without vectorization hint.
  }

  // update loop info
  li->AddLoop(loop);
  if (auto parent_loop = li->GetLoop(parent_loop_name)) {
    parent_loop->AddSubLoop(loop);
    loop->SetParentLoop(parent_loop);
  }
  n.loop = loop;
  parent_loop_name = loop_name;
  return true;
}
ptr<LoopInfo> LoopAnalysis::GetLoopInfo() const { return li; }

// LoopVectorizeSimpleChecker
LoopVectorizeSimpleChecker::LoopVectorizeSimpleChecker(
    const ptr<SymbolTable> s_tab, ptr<LoopInfo> l)
    : LoopVisitor(s_tab, "loop-vectorize-simple-checker"), li(l) {
  for (auto& [loop_name, loop] : li->GetAllLoops()) {
    can_vectorizes[loop_name] = true;
  }
}

bool LoopVectorizeSimpleChecker::ExistLoopVectorizationLegal() {
  for (auto& [loop_name, can_vectorize] : can_vectorizes) {
    auto loop = li->GetLoop(loop_name);
    if (loop->NeedVectorize() && can_vectorize) return true;
  }
  return false;
}

bool LoopVectorizeSimpleChecker::NeedCheck() {
  if (!InLoop()) return false;
  if (cur_loop->NeedVectorize()) return true;
  if (CCtx().Vectorize() && li->IsInnermostLoop(cur_loop->LoopName()))
    return true;
  return false;
}

void LoopVectorizeSimpleChecker::SetLoopVectorizationFailed() {
  can_vectorizes[cur_loop->LoopName()] = false;
}

bool LoopVectorizeSimpleChecker::Visit(AST::DataAccess& n) {
  TraceEachVisit(n);
  if (!NeedCheck()) return true;
  auto elem_ty = GetBaseType(*n.GetType());
  if (cur_loop->GetDataType() == BaseType::UNKNOWN) {
    cur_loop->SetDataType(elem_ty);
    if (CCtx().TargetVectorizeTypes().count(elem_ty) == 0) {
      if (debug_visit)
        dbgs() << indent << "data type " << STR(elem_ty)
               << " is not legal for vectorization on target architecture "
               << ToUpper(CCtx().GetArch()) << "\n";
      SetLoopVectorizationFailed();
      return true;
    }
  } else if (elem_ty != cur_loop->GetDataType()) {
    if (debug_visit)
      dbgs() << indent
             << "vectorization of mixed type is currently not supported\n";
    SetLoopVectorizationFailed();
    return true;
  }
  return true;
}

bool LoopVectorizeSimpleChecker::Visit(AST::ForeachBlock& n) {
  TraceEachVisit(n);
  if (AST::HasVectorizationHint(n) &&
      !li->IsInnermostLoop(cur_loop->LoopName())) {
    Error1(n.LOC(), "only innermost loop can be vectorized.");
    SetLoopVectorizationFailed();
    return false;
  }
  // only loop with vectorization hint or innermost loop needs to be checked
  if (!NeedCheck()) return true;
  cur_loop->SetNeedVectorize(true);

  if (!n.IsNorm() || !cur_loop) {
    Error1(n.LOC(), "cannot vectorize non-normalized loop.");
    SetLoopVectorizationFailed();
    return false;
  }

  if (Pb_level == 0) {
    Error1(n.LOC(), "only loop inside parallel-by can be vectorized.");
    SetLoopVectorizationFailed();
    return false;
  }
  return true;
}

bool LoopVectorizeSimpleChecker::Visit(AST::WhileBlock& n) {
  TraceEachVisit(n);
  Error1(n.LOC(), "while loop is not supported for vectorization.");
  SetLoopVectorizationFailed();
  return false;
}

bool LoopVectorizeSimpleChecker::Visit(AST::DMA& n) {
  TraceEachVisit(n);
  if (!NeedCheck()) return true;
  Error1(n.LOC(), "DMA is not supported for vectorization.");
  SetLoopVectorizationFailed();
  return false;
}

bool LoopVectorizeSimpleChecker::Visit(AST::Wait& n) {
  TraceEachVisit(n);
  if (!NeedCheck()) return true;
  Error1(n.LOC(), "data access is not supported for vectorization.");
  SetLoopVectorizationFailed();
  return false;
}

bool LoopVectorizeSimpleChecker::Visit(AST::Trigger& n) {
  TraceEachVisit(n);
  if (!NeedCheck()) return true;
  Error1(n.LOC(), "trigger is not supported for vectorization.");
  SetLoopVectorizationFailed();
  return false;
}

bool LoopVectorizeSimpleChecker::Visit(AST::Rotate& n) {
  TraceEachVisit(n);
  if (!NeedCheck()) return true;
  Error1(n.LOC(), "rotate is not supported for vectorization.");
  SetLoopVectorizationFailed();
  return false;
}

bool LoopVectorizeSimpleChecker::Visit(AST::Return& n) {
  TraceEachVisit(n);
  if (!NeedCheck()) return true;
  Error1(n.LOC(), "return is not supported for vectorization.");
  SetLoopVectorizationFailed();
  return false;
}

bool LoopVectorizeSimpleChecker::Visit(AST::Select& n) {
  TraceEachVisit(n);
  if (!NeedCheck()) return true;
  Error1(n.LOC(), "select is not supported for vectorization.");
  SetLoopVectorizationFailed();
  return false;
}

bool LoopVectorizeSimpleChecker::Visit(AST::IncrementBlock& n) {
  TraceEachVisit(n);
  if (!NeedCheck()) return true;
  Error1(n.LOC(), "increment block is not supported for vectorization.");
  SetLoopVectorizationFailed();
  return false;
}

bool LoopVectorizeSimpleChecker::Visit(AST::Synchronize& n) {
  TraceEachVisit(n);
  if (!NeedCheck()) return true;
  Error1(n.LOC(), "synchronize is not supported for vectorization.");
  SetLoopVectorizationFailed();
  return false;
}

bool LoopVectorizeSimpleChecker::Visit(AST::InThreadsBlock& n) {
  TraceEachVisit(n);
  if (!NeedCheck()) return true;
  Error1(n.LOC(), "in-threads block is not supported for vectorization.");
  SetLoopVectorizationFailed();
  return false;
}

bool LoopVectorizeSimpleChecker::Visit(AST::ParallelBy& n) {
  TraceEachVisit(n);
  if (!NeedCheck()) return true;
  Error1(n.LOC(), "parallel-by is not supported for vectorization.");
  SetLoopVectorizationFailed();
  return false;
}

bool LoopVectorizeSimpleChecker::Visit(AST::Call& n) {
  TraceEachVisit(n);
  if (!NeedCheck()) return true;
  if (n.IsAnno()) return true;
  if (n.IsArith()) return true;
  if (debug_visit)
    dbgs() << indent << "found call stmt in loop `" << cur_loop->LoopName()
           << "`, which currently not supported in loop "
              "vectorization\n";

  SetLoopVectorizationFailed();
  return true;
}

void LoopVectorizeSimpleChecker::AddLoopUse(std::string sym, location loc) {
  if (loop_uses.find(sym) == loop_uses.end()) {
    loop_uses.emplace(sym, std::vector<location>{loc});
  } else {
    loop_uses[sym].push_back(loc);
  }
}

void LoopVectorizeSimpleChecker::FindLoopUses(ptr<AST::Node> n) {
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

void LoopVectorizeSimpleChecker::AddLoopDef(std::string sym, location loc) {
  if (loop_defs.find(sym) == loop_defs.end()) {
    loop_defs.emplace(sym, std::vector<location>{loc});
  } else {
    loop_defs[sym].push_back(loc);
  }
}

bool LoopVectorizeSimpleChecker::Visit(AST::NamedVariableDecl& n) {
  TraceEachVisit(n);
  if (!NeedCheck()) return true;
  if (n.IsArray()) {
    if (debug_visit)
      dbgs() << indent
             << "Array definition is not supported in vectorization.\n";
    SetLoopVectorizationFailed();
    return true;
  } else {
    auto name_sym = InScopeName(n.name_str);
    FindLoopUses(n.init_expr);
    auto nds = n.GetDiversityShape();
    if (nds.Varying() && loop_uses.find(name_sym) != loop_uses.end() &&
        loop_defs.find(name_sym) == loop_defs.end()) {
      if (debug_visit)
        dbgs() << indent
               << "variable " + name_sym +
                      " is used before its declaration in the loop.\n";
      SetLoopVectorizationFailed();
      return true;
    }
    AddLoopDef(name_sym, n.LOC());
  }

  return true;
}

bool LoopVectorizeSimpleChecker::Visit(AST::Assignment& n) {
  TraceEachVisit(n);
  if (!NeedCheck()) return true;

  if (!n.AssignToDataElement()) {
    auto name_sym = InScopeName(n.GetName());
    FindLoopUses(n.value);
    auto nds = n.da->GetDiversityShape();
    if (nds.Varying() && loop_uses.find(name_sym) != loop_uses.end() &&
        loop_defs.find(name_sym) == loop_defs.end()) {
      if (debug_visit)
        dbgs() << indent
               << "variable " + name_sym +
                      " is used before its assignment in the loop.\n";
      SetLoopVectorizationFailed();
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
      SetLoopVectorizationFailed();
      return true;
    }
  }

  return true;
}

bool LoopVectorizeSimpleChecker::AfterBeforeVisitImpl(AST::Node& n) {
  if (isa<AST::ParallelBy>(&n)) { Pb_level++; }
  return true;
}

bool LoopVectorizeSimpleChecker::BeforeAfterVisitImpl(AST::Node& n) {
  if (auto fb = dyn_cast<AST::ForeachBlock>(&n)) {
    auto loop = fb->loop;
    assert(loop);
    auto loop_name = loop->LoopName();
    if (li->IsInnermostLoop(loop_name) && can_vectorizes[loop_name]) {
      loop->SetNeedVectorize(true);
      if (debug_visit)
        dbgs() << "[SimpleCheck] loop `" << loop->LoopName()
               << "` is promised to be vectorized.\n";
    } else {
      loop->SetNeedVectorize(false);
      if (debug_visit)
        dbgs() << "[SimpleCheck] loop `" << loop->LoopName()
               << "` cannot be vectorized.\n";
    }
  } else if (isa<AST::ParallelBy>(&n)) {
    Pb_level--;
    assert(Pb_level >= 0);
  }
  return true;
}

// LoopVectorizeLegalityChecker
LoopVectorizeLegalityChecker::LoopVectorizeLegalityChecker(
    const ptr<SymbolTable> s_tab, ptr<LoopInfo> l, ptr<ScopedSCEVTable> s)
    : LoopVisitor(s_tab, "loop_vectorize_legality"), li(l), scev_table(s) {
  for (auto& [loop_name, loop] : li->GetAllLoops()) {
    can_vectorizes[loop_name] = loop->NeedVectorize() ? true : false;
  }
}

bool LoopVectorizeLegalityChecker::NeedCheck() {
  if (!InLoop()) return false;
  if (cur_loop->LoopName() == appointed_loop) return true;
  return false;
}

void LoopVectorizeLegalityChecker::SetLoopVectorizationFailed() {
  can_vectorizes[cur_loop->LoopName()] = false;
}

void LoopVectorizeLegalityChecker::CheckLegalityInLoop(
    AST::ForeachBlock& loop_node) {
  cur_loop = loop_node.loop;
  if (debug_visit)
    dbgs() << "[Check] Check legality of loop `" << cur_loop->LoopName()
           << "` for vectorization.\n";
  loop_node.accept(*this);
}

bool LoopVectorizeLegalityChecker::IsLegalToVectorize(
    const std::string& lname) {
  if (can_vectorizes.find(lname) != can_vectorizes.end())
    return can_vectorizes[lname];
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
  // The alignment requirement is equal to the vector size of
  // simd operands.
  auto alignment = SizeOf(e_ty) * cur_loop->GetVectorFactor();

  auto da_scev = n.GetSCEV();
  if (auto da_ar = dyn_cast<SCEVAddRecExpr>(da_scev)) {
    auto base = da_ar->GetBase();
    while (auto inner_ar = dyn_cast<SCEVAddRecExpr>(base)) {
      auto step = inner_ar->GetStep();
      auto step_val = step->GetValue();
      if (auto num_step = dyn_cast<sbe::NumericValue>(step_val)) {
        auto stride = num_step->Value() * SizeOf(e_ty);
        bool IsAligned = false;
        if (stride % alignment == 0) { IsAligned = true; }
        if (!IsAligned) {
          if (debug_visit)
            dbgs() << indent << "unaligned data access: " << STR(n)
                   << ", stride: " << STR(stride) << "{" << alignment << "}.\n";
          return false;
        }
      }
      base = inner_ar->GetBase();
    }
  }

  return true;
}

bool LoopVectorizeLegalityChecker::Visit(AST::ForeachBlock& n) {
  TraceEachVisit(n);

  auto vector_factor = cur_loop->GetVectorFactor();
  if (static_cast<size_t>(vector_factor) > CCtx().TargetVectorizeLimit()) {
    if (debug_visit)
      dbgs() << "[plan] vector factor " << cur_loop->GetVectorFactor()
             << " exceeds architecture limit " << CCtx().TargetVectorizeLimit()
             << ", skip this plan.\n";
    SetLoopVectorizationFailed();
    return false;
  }

  auto IsPowerOf2 = [](int n) { return (n > 0) && ((n & (n - 1)) == 0); };
  ptr<AST::AttributeExpr> vectorization_hint = nullptr;
  if (AST::HasVectorizationHint(n, vectorization_hint)) {
    if (vector_factor <= 1 || !IsPowerOf2(vector_factor)) {
      Error1(vectorization_hint->AttrValueAt(1)->LOC(),
             "vector width should be a power of 2 greater than 1.");
      SetLoopVectorizationFailed();
      return false;
    }
    // simple check about loop count
    auto loop_count = cur_loop->GetLoopCount();
  }

  return true;
}

bool LoopVectorizeLegalityChecker::Visit(AST::DataAccess& n) {
  TraceEachVisit(n);
  // check mixed type
  if (!n.AccessElement()) return true;
  auto elem_ty = GetBaseType(*n.GetType());
  auto vector_factor = cur_loop->GetVectorFactor();
  auto vector_size = SizeOf(elem_ty) * vector_factor;
  auto single_vector_size = CCtx().GetVectorLength();

  if (vector_size != single_vector_size &&
      vector_size != 2 * single_vector_size &&
      vector_size != 4 * single_vector_size) {
    if (debug_visit)
      dbgs() << indent
             << "vectorization with unsupported vector width: " << vector_factor
             << "\n";
    SetLoopVectorizationFailed();
    return true;
  }

  // It applies the alignment check unless the target supports unaligned simd
  // memory access
  if (CCtx().GetTarget().EnforceVectorAlignment(CCtx().GetArch()))
    if (!CheckDataAccessAlignment(n)) {
      SetLoopVectorizationFailed();
      return true;
    }

  DiversityShape offset_shape = DiversityShape(UNIFORM, sbe::nu(0));
  auto indices = n.GetIndices();
  for (int idx = indices.size() - 1; idx >= 0; --idx) {
    // process from last index to first index
    auto index = indices[idx];
    offset_shape =
        ComputeDiversityShape(offset_shape, index->GetDiversityShape(), "+");
  }

  if (offset_shape.Varying() && !offset_shape.Stride(1)) {
    // need gather/scatter for vectorized access
    if (CCtx().GetTarget().EnforceVectorAlignment(CCtx().GetArch()))
      SetLoopVectorizationFailed();
    if (debug_visit) {
      dbgs() << indent << "Gather/Scatter is not legal for vectorization"
             << " in " << ToUpper(CCtx().GetArch()) << " target \n";
    }
  } else {
    if (debug_visit) {
      dbgs() << indent
             << "Gather/Scatter is needed for vectorized access: " << STR(n)
             << "\n";
    }
  }

  return true;
}

bool LoopVectorizeLegalityChecker::BeforeAfterVisitImpl(AST::Node& n) {
  if (auto fb = dyn_cast<AST::ForeachBlock>(&n)) {
    auto loop = fb->loop;
    if (loop->NeedVectorize() && can_vectorizes[loop->LoopName()]) {
      if (debug_visit)
        dbgs() << "[check] vector factor = " << loop->GetVectorFactor()
               << " is legal for loop `" << loop->LoopName() << "`.\n";
    } else if (loop->NeedVectorize() && !can_vectorizes[loop->LoopName()]) {
      if (debug_visit)
        dbgs() << "[check] vector factor = " << loop->GetVectorFactor()
               << " is illegal for loop `" << loop->LoopName()
               << "`, cannot vectorize.\n";
    }
  }
  return true;
}

bool LoopVectorizeLegalityChecker::ExistLoopVectorizationLegal() {
  for (auto& [loop_name, can_vectorize] : can_vectorizes) {
    auto loop = li->GetLoop(loop_name);
    if (loop->CanVectorize()) return true;
  }
  return false;
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
  if (!InLoop()) return false;
  if (cur_loop->CanVectorize()) return true;
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
  if (!InLoop()) return false;
  if (cur_loop->CanVectorize()) return true;
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
  mask->SetType(MakeVectorType(BaseType::BOOL, cur_loop->GetVectorFactor()));
  mask->SetDiversityShape(DiversityShapeKind::DIVERGENT);
  return mask;
}

ptr<AST::Expr> MaskGen::MakeMaskIdExpr(const location& loc,
                                       const std::string& name) {
  auto mask = AST::MakeIdExpr(loc, name);
  mask->SetType(MakeVectorType(BaseType::BOOL, cur_loop->GetVectorFactor()));
  mask->SetDiversityShape(DiversityShapeKind::DIVERGENT);
  return mask;
}

ptr<AST::NamedVariableDecl> MaskGen::MakeMaskDecl(const location& loc,
                                                  const std::string& name,
                                                  const ptr<AST::Expr>& rhs) {
  auto mask_ty = MakeVectorType(BaseType::BOOL, cur_loop->GetVectorFactor());
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
  auto mask_ty = MakeVectorType(BaseType::BOOL, cur_loop->GetVectorFactor());
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
  if (!NeedTransform()) return true;
  assert(n.IsNorm() && "Loop should be normalized before MaskGen.");
  assert(li->IsInnermostLoop(cur_loop->LoopName()) &&
         "Only innermost loop can be vectorized.");

  auto smi = cur_loop->GetScopedMaskInfo();
  auto loc = n.stmts->LOC();
  auto mask_ty = MakeVectorType(BaseType::BOOL, cur_loop->GetVectorFactor());
  auto iv = n.GetIV();
  auto iv_ty = iv->GetType();
  auto upper_bound = GetSingleUpperBound(iv_ty);
  ptr<AST::NamedVariableDecl> loop_cond = nullptr;
  ptr<AST::Expr> mask_expr = nullptr;
  auto cur_mask_name = MaskName();
  // if the upper bound is a constant and is divisible by vector width
  if (auto ub_nu = dyn_cast<sbe::NumericValue>(upper_bound);
      ub_nu && ub_nu->Value() % cur_loop->GetVectorFactor() == 0) {
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
    mask_expr->AddNote("broadcast",
                       std::to_string(cur_loop->GetVectorFactor()));
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
  smi->SetVectorWidth(cur_loop->GetVectorFactor());

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
  auto smi = cur_loop->GetScopedMaskInfo();
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
      MakeVectorType(BaseType::BOOL, cur_loop->GetVectorFactor()));
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
  auto smi = cur_loop->GetScopedMaskInfo();
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
LoopVectorizer::LoopVectorizer() : VisitorWithSymTab("loop") {}

bool LoopVectorizer::BeforeVisitImpl(AST::Node&) { return true; }
bool LoopVectorizer::AfterVisitImpl(AST::Node&) { return true; }

bool LoopVectorizer::CheckVectorizationHint(AST::Node& root) {
  VectorizationHintChecker vhc;
  vhc.SetDebugVisit(debug_visit);
  vhc.SetTraceVisit(trace_visit);
  root.accept(vhc);
  if (HasError() || abend_after) return false;
  if (!vhc.HasVectorizationHint() && !CCtx().Vectorize()) {
    if (debug_visit)
      dbgs() << "[loop vectorization] no loop is marked to be vectorized or "
                "option '--vectorize' is not set. Skip vectorization.\n";
    return false;
  }
  if (prt_visitor) dbgs() << " |- " << vhc.GetName() << NewL;
  return true;
}

bool LoopVectorizer::AnalyzeLoops(AST::Node& root) {
  LoopAnalysis la(SymTab());
  la.SetDebugVisit(debug_visit);
  la.SetTraceVisit(trace_visit);
  la.RunOnProgram(root);
  li = la.GetLoopInfo();
  if (debug_visit) li->dump(dbgs());
  if (HasError() || abend_after) return false;
  if (prt_visitor) dbgs() << " |- " << la.GetName() << NewL;

  return true;
}

bool LoopVectorizer::CheckSimply(AST::Node& root) {
  if (debug_visit)
    dbgs() << "\n[simplecheck] start loop vectorization simple check.\n";
  LoopVectorizeSimpleChecker lvc(SymTab(), li);
  lvc.SetDebugVisit(debug_visit);
  lvc.SetTraceVisit(trace_visit);
  lvc.RunOnProgram(root);
  if (HasError() || abend_after) return false;
  if (!lvc.ExistLoopVectorizationLegal()) {
    if (debug_visit)
      dbgs() << "[SimpleCheck] No loop is legal to be vectorized, "
                "auto-vectorization exits.";
    return false;
  }
  if (prt_visitor) dbgs() << " |- " << lvc.GetName() << NewL;
  return true;
}

bool LoopVectorizer::AnalyzeDiversityShape(AST::Node& root) {
  if (debug_visit) dbgs() << "\n[diversity] start diversity analysis.\n";
  DiversityAnalysisHandler da(SymTab(), li);
  da.SetDebugVisit(debug_visit);
  da.SetTraceVisit(trace_visit);
  da.RunOnProgram(root);
  if (prt_visitor) dbgs() << " |- " << da.GetName() << NewL;
  if (HasError() || abend_after) return false;
  di = da.GetDiversityAnalysis();

  return true;
}

// struct CostEngine final : public LoopVisitor {
//   CostEngine()
//       : LoopVisitor(nullptr, "loop_vectorization_cost_model") {}
//   bool Visit(AST::ForeachBlock& n) {
//     TraceEachVisit(n);
//     return true;
//   }
// };

// struct LoopVectorizationCostModel {
// private:
//   ptr<LoopInfo> li;
//   CostEngine ce;
// public:
//   LoopVectorizationCostModel(ptr<LoopInfo> l)
//       : li(l) {}

//   size_t EstimateVectorizationCost(ptr<AST::ForeachBlock> loop) {
//     loop->accept(ce);
//     return 0;
//   }
// };

struct LoopVectorizationPlanner final : public LoopVisitor {
private:
  ptr<LoopInfo> li;
  ptr<ScalarEvolutionAnalysis> sea;
  ptr<LoopVectorizeLegalityChecker> lvc;
  // store the most profitable vectorization factor for each loop
  std::unordered_map<std::string, size_t> vector_factors;

  std::vector<size_t> vector_register_widths;
  void UpdatePlan(const std::string& lname, size_t plan_id) {
    auto loop = li->GetLoop(lname);
    if (!loop->NeedVectorize()) return;
    if (plan_id >= vector_register_widths.size()) return;
    if (loop->HasVectorizationHint()) return;
    size_t element_size = loop->GetDataType() == BaseType::UNKNOWN
                              ? 4
                              : SizeOf(loop->GetDataType());
    loop->SetVectorFactor(vector_register_widths[plan_id] / element_size);
    return;
  }

  void SettlePlan(const std::string& lname) {
    auto loop = li->GetLoop(lname);
    if (!loop->CanVectorize()) {
      loop->SetVectorFactor(1);
      return;
    }
    if (loop->HasVectorizationHint()) return;
    loop->SetVectorFactor(vector_factors[lname]);
    if (debug_visit)
      dbgs() << "\n[plan] settled vectorization factor for loop `"
             << loop->LoopName() << "`: " << loop->GetVectorFactor() << "\n";
    return;
  }

public:
  LoopVectorizationPlanner(const ptr<SymbolTable> s_tab, ptr<LoopInfo> l,
                           ptr<ScalarEvolutionAnalysis> s,
                           ptr<LoopVectorizeLegalityChecker> lc)
      : LoopVisitor(s_tab, "loop_vectorization_planner"), li(l), sea(s),
        lvc(lc) {
    vector_register_widths = {};
    auto vector1 = CCtx().GetVectorLength();
    if (CCtx().TargetSupportVectorize()) {
      vector_register_widths.push_back(vector1);
      vector_register_widths.push_back(vector1 * 2);
      vector_register_widths.push_back(vector1 * 4);
    } else {
      choreo_unreachable(
          "unsupported target arch in loop vectorization planner.");
    }
  }

  bool Visit(AST::Program& n) {
    n.accept(*sea);
    return true;
  }

  bool Visit(AST::ForeachBlock& n) {
    // for each loop, do legality check and update vectorization plan
    if (!cur_loop->NeedVectorize()) return true;
    if (debug_visit)
      dbgs() << "\n[plan] Planning vectorization for loop `"
             << cur_loop->LoopName() << "`\n";
    auto loop_name = cur_loop->LoopName();
    size_t plan_id = 0;
    while (plan_id < vector_register_widths.size()) {
      UpdatePlan(loop_name, plan_id++);
      if (debug_visit) {
        if (cur_loop->HasVectorizationHint())
          dbgs() << "\n[plan] found vectorization hint, set vector factor to: "
                 << cur_loop->GetVectorFactor() << "\n";
        else
          dbgs() << "\n[plan] try vector factor: "
                 << cur_loop->GetVectorFactor() << "\n";
      }
      sea->ReComputeInLoop(loop_name, false);
      lvc->CheckLegalityInLoop(n);
      if (lvc->IsLegalToVectorize(loop_name)) {
        auto loop_count = cur_loop->GetLoopCount();
        // if loop count is known and less than vector factor and current plan
        // is not 1, skip the new plan
        if (loop_count->IsNumeric()) {
          if (sbe::clt(loop_count, sbe::nu(cur_loop->GetVectorFactor())) &&
              vector_factors[loop_name] > 1) {
            if (debug_visit)
              dbgs() << "[plan] loop count " << STR(loop_count)
                     << " is less than vector factor "
                     << cur_loop->GetVectorFactor() << ", skip this plan.\n";
            continue;
          }
        }
        vector_factors[loop_name] = cur_loop->GetVectorFactor(); // cost model
        cur_loop->SetCanVectorize(true);
      }
      // if there is vectorization hint, do not try other plans
      if (cur_loop->HasVectorizationHint()) break;
    }

    SettlePlan(loop_name);
    sea->ReComputeInLoop(loop_name);

    return true;
  }
};

// do loop vectorization legality check until all loops are checked
bool LoopVectorizer::ComputeVectorizationPlan(AST::Node& root) {
  ptr<ScalarEvolutionAnalysis> sea =
      std::make_shared<ScalarEvolutionAnalysis>(SymTab(), li);
  sea->SetDebugVisit(debug_visit);
  sea->SetTraceVisit(trace_visit);

  ptr<LoopVectorizeLegalityChecker> lvc =
      std::make_shared<LoopVectorizeLegalityChecker>(SymTab(), li,
                                                     sea->GetScevTab());
  lvc->SetDebugVisit(debug_visit);
  lvc->SetTraceVisit(trace_visit);

  LoopVectorizationPlanner planner(SymTab(), li, sea, lvc);
  planner.SetDebugVisit(debug_visit);
  planner.SetTraceVisit(trace_visit);
  planner.RunOnProgram(root);

  if (!lvc->ExistLoopVectorizationLegal()) {
    if (debug_visit)
      dbgs() << "[plan] No loop is legal to be vectorized, "
                "auto-vectorization exits.\n";
    return false;
  } else {
    if (debug_visit) {
      dbgs() << "\n[plan] Final vectorization plan:\n";
      for (auto& [loop_name, loop] : li->GetAllLoops()) {
        if (loop->CanVectorize()) {
          dbgs() << "  - Loop `" << loop_name
                 << "` will be vectorized with factor: "
                 << loop->GetVectorFactor() << "\n";
        } else {
          dbgs() << "  - Loop `" << loop_name << "` cannot be vectorized.\n";
        }
      }
    }
  }

  if (prt_visitor) dbgs() << " |- " << planner.GetName() << NewL;
  if (HasError() || abend_after) return false;

  return true;
}

bool LoopVectorizer::InferenceType(AST::Node& root) {
  if (debug_visit) dbgs() << "\n[vinfer] start inferring vector types.\n";
  VectorTypeInfer vti(SymTab(), li, di);
  vti.SetDebugVisit(debug_visit);
  vti.SetTraceVisit(trace_visit);
  vti.RunOnProgram(root);
  if (prt_visitor) dbgs() << " |- " << vti.GetName() << NewL;
  if (HasError() || abend_after) return false;
  return true;
}

bool LoopVectorizer::LinearizeBranch(AST::Node& root) {
  if (debug_visit) dbgs() << "\n[linearize] start linearization.\n";
  Linearizer ln(SymTab(), li, di);
  ln.SetDebugVisit(debug_visit);
  ln.SetTraceVisit(trace_visit);
  ln.RunOnProgram(root);
  if (prt_visitor) dbgs() << " |- " << ln.GetName() << NewL;
  if (HasError() || abend_after) return false;

  BranchSimplicition bs(SymTab());
  bs.SetDebugVisit(debug_visit);
  bs.SetTraceVisit(trace_visit);
  root.accept(bs);
  if (prt_visitor) dbgs() << " |- " << bs.GetName() << NewL;
  if (HasError() || abend_after) return false;

  return true;
}

bool LoopVectorizer::GenerateMask(AST::Node& root) {
  if (debug_visit) dbgs() << "\n[mask] start geneating masks.\n";
  MaskGen mg(SymTab(), li, di);
  mg.SetDebugVisit(debug_visit);
  mg.SetTraceVisit(trace_visit);
  mg.RunOnProgram(root);
  if (prt_visitor) dbgs() << " |- " << mg.GetName() << NewL;
  if (HasError() || abend_after) return false;
  return true;
}

// main function to run loop auto-vectorization
bool LoopVectorizer::RunOnProgramImpl(AST::Node& root) {
  // pre-checks
  if (!isa<AST::Program>(&root)) {
    Error1(root.LOC(), "Not running a choreo program.");
    return false;
  }
  if (!CCtx().TargetSupportVectorize()) {
    Error1(root.LOC(),
           "Loop vectorization transformations is not available for " +
               std::string(CCtx().TargetName()) + ".");
    return true;
  }
  if (prt_visitor) dbgs() << "|- " << GetName() << NewL;
  debug_visit |= CCtx().TraceVectorize();

  // main steps
  if (!CheckVectorizationHint(root)) {
    if (HasError() || abend_after) return false;
    return true;
  }

  if (!AnalyzeLoops(root)) {
    if (HasError() || abend_after) return false;
    return true;
  }

  if (!AnalyzeDiversityShape(root)) {
    if (HasError() || abend_after) return false;
    return true;
  }

  if (!CheckSimply(root)) {
    if (HasError() || abend_after) return false;
    return true;
  }

  if (!ComputeVectorizationPlan(root)) {
    if (HasError() || abend_after) return false;
    return true;
  }

  if (!InferenceType(root)) {
    if (HasError() || abend_after) return false;
    return true;
  }

  if (!LinearizeBranch(root)) {
    if (HasError() || abend_after) return false;
    return true;
  }

  if (!GenerateMask(root)) {
    if (HasError() || abend_after) return false;
    return true;
  }

  return true;
}

} // namespace Choreo
