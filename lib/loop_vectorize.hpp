#ifndef __CHOREO_DIVERGENT_ANALYSIS_HPP__
#define __CHOREO_DIVERGENT_ANALYSIS_HPP__
#include "ast.hpp"
#include "context.hpp"
#include "diversity_analysis.hpp" // Ensure this header defines DiversityAnalysis
#include "io.hpp"
#include "loop_utils.hpp"
#include "visitor.hpp"

namespace Choreo {
struct LoopVectorizeLegalityChecker final : public LoopVisitor {
private:
  ptr<LoopInfo> li;

  bool NeedCheck() {
    auto loop = li->GetLoop(lname);
    if (!loop) return false;
    if (AST::NeedVectorize(*loop->loop)) return true;
    return false;
  }

public:
  LoopVectorizeLegalityChecker(ptr<LoopInfo> l)
      : LoopVisitor(nullptr, "loop_vectorize_legality"), li(l) {}

  bool Visit(AST::ForeachBlock& n) override {
    TraceEachVisit(n);
    if (!n.IsNorm()) {
      Error(n.LOC(), "cannot vectorize non-normalized loop.");
      return false;
    }
    if (AST::NeedVectorize(n) && !li->IsInnermostLoop(lname)) {
      Error(n.LOC(), "cannot vectorize non-innermost loop.");
      return false;
    }
    // todo: more check
    return true;
  }

  bool Visit(AST::WhileBlock& n) override {
    TraceEachVisit(n);
    Error(n.LOC(), "while loop is not supported for vectorization.");
    return false;
  }

  bool Visit(AST::DMA& n) override {
    TraceEachVisit(n);
    if (!NeedCheck()) return true;
    Error(n.LOC(), "DMA is not supported for vectorization.");
    return false;
  }

  bool Visit(AST::Wait& n) override {
    TraceEachVisit(n);
    if (!NeedCheck()) return true;
    Error(n.LOC(), "data access is not supported for vectorization.");
    return false;
  }

  bool Visit(AST::Trigger& n) override {
    TraceEachVisit(n);
    if (!NeedCheck()) return true;
    Error(n.LOC(), "trigger is not supported for vectorization.");
    return false;
  }

  bool Visit(AST::Rotate& n) override {
    TraceEachVisit(n);
    if (!NeedCheck()) return true;
    Error(n.LOC(), "rotate is not supported for vectorization.");
    return false;
  }

  bool Visit(AST::Return& n) override {
    TraceEachVisit(n);
    if (!NeedCheck()) return true;
    Error(n.LOC(), "return is not supported for vectorization.");
    return false;
  }

  bool Visit(AST::Select& n) override {
    TraceEachVisit(n);
    if (!NeedCheck()) return true;
    Error(n.LOC(), "select is not supported for vectorization.");
    return false;
  }

  bool Visit(AST::IncrementBlock& n) override {
    TraceEachVisit(n);
    if (!NeedCheck()) return true;
    Error(n.LOC(), "increment block is not supported for vectorization.");
    return false;
  }

  bool Visit(AST::Synchronize& n) override {
    TraceEachVisit(n);
    if (!NeedCheck()) return true;
    Error(n.LOC(), "synchronize is not supported for vectorization.");
    return false;
  }

  bool Visit(AST::InThreadsBlock& n) override {
    TraceEachVisit(n);
    if (!NeedCheck()) return true;
    Error(n.LOC(), "in-threads block is not supported for vectorization.");
    return false;
  }

  bool Visit(AST::ParallelBy& n) override {
    TraceEachVisit(n);
    if (!NeedCheck()) return true;
    Error(n.LOC(), "parallel-by is not supported for vectorization.");
    return false;
  }

  bool Visit(AST::NamedVariableDecl& n) override {
    TraceEachVisit(n);
    if (!NeedCheck()) return true;
    if (n.IsArray()) {
      Error(n.LOC(), "array variable declaration in vectorization check: " +
                         STR(n) + ".");
      return false;
    }
    return true;
  }
};

struct BranchSimplicition final : public LoopVisitor {
public:
  BranchSimplicition(const ptr<SymbolTable> s_tab)
      : LoopVisitor(s_tab, "branch-simplicition") {}

  bool Visit(AST::IfElseBlock& n) override {
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

    // todo: handle the case of multiple if-else blocks
    return true;
  }
};

// linearize branch inside vectorized loops
struct Linearizer final : public LoopVisitor {
private:
  ptr<LoopInfo> li;
  ptr<DiversityInfo> di;

  bool NeedTransform() {
    auto loop = li->GetLoop(lname);
    if (!loop) return false;
    if (AST::NeedVectorize(*loop->loop)) return true;
    return false;
  }

public:
  Linearizer(const ptr<SymbolTable> s_tab, ptr<LoopInfo> l,
             ptr<DiversityInfo> d)
      : LoopVisitor(s_tab, "linearizer"), li(l), di(d) {}

  bool Visit(AST::MultiNodes& n) override {
    TraceEachVisit(n);
    if (!NeedTransform()) return true;

    for (size_t stmt_index = 0; stmt_index < n.Count(); ++stmt_index) {
      if (auto if_block = dyn_cast<AST::IfElseBlock>(n.SubAt(stmt_index))) {

        auto pred = if_block->GetPred();
        auto pred_ds = pred->GetDiversityShape();
        // if this is a uniform branch, keep it
        if (pred_ds.Uniform()) continue;
        // if this is a divergent branch, we need to linearize it
        // Firstly, we need to normalize the if_else block to ensure it does not have
        // else branch. Secondly, we need to insert a negated if-else block consecutively after
        // the original if-else block.

        auto else_stmts = if_block->else_stmts;
        if (!else_stmts) continue;
        auto neg_pred =
            AST::Make<AST::Expr>(if_block->LOC(), "!", pred->Clone());
        neg_pred->SetType(pred->GetType());
        neg_pred->SetDiversityShape(pred_ds);
        auto neg_if_block =
            AST::Make<AST::IfElseBlock>(if_block->LOC(), neg_pred, else_stmts);
        n.Insert(neg_if_block, ++stmt_index);
        if_block->else_stmts = nullptr; // remove the else stmts
        if (debug_visit) {
          dbgs() << "[linearize] Inserted negated if-else block: "
                 << STR(neg_if_block->pred) << " after original if-else block: "
                 << STR(if_block->pred) << "\n";
        }
      }
    }

    return true;
  }
};

struct MaskGen final : public LoopVisitor {
private:
  ptr<LoopInfo> li;
  ptr<DiversityInfo> di;
  std::stack<std::string> mask_stack;
  ptr<AST::Node> exec;
  int mask_count = 0;
  int vector_width = 0;
  ptr<Type> vbool_ty = nullptr;

  inline std::string MaskName() {
    return "mask" + std::to_string(mask_count++);
  }

  bool NeedTransform() {
    auto loop = li->GetLoop(lname);
    if (!loop) return false;
    if (AST::NeedVectorize(*loop->loop)) return true;
    return false;
  }

public:
  MaskGen(const ptr<SymbolTable> s_tab, ptr<LoopInfo> l, ptr<DiversityInfo> d)
      : LoopVisitor(s_tab, "mask"), li(l), di(d) {}

  bool Visit(AST::MultiNodes& n) override {
    TraceEachVisit(n);
    if (!NeedTransform()) return true;
    if (mask_stack.empty()) {
      return true;
    }

    auto mask = mask_stack.top();
    for (size_t stmt_index = 0; stmt_index < n.Count(); ++stmt_index) {
      if (auto if_block = dyn_cast<AST::IfElseBlock>(n.SubAt(stmt_index))) {
        if (!if_block->IsDivergent()) continue;
        auto exec = AST::Make<AST::Assignment>(
            if_block->LOC(), "exec", AST::MakeIdExpr(if_block->LOC(), mask));
        exec->SetType(vbool_ty);
        exec->SetDiversityShape(DiversityShape(DiversityShapeKind::DIVERGENT));
        exec->da->SetType(vbool_ty);
        n.Insert(exec, ++stmt_index);
        if (debug_visit)
          dbgs() << "[mask] Inserted exec assignment: "
                 << STR(exec) << " after divergenet branch: "
                 << STR(if_block->GetPred()) << "\n";
      }
    }
    return true;
  }

  bool Visit(AST::ForeachBlock& n) override {
    TraceEachVisit(n);
    auto loc = n.stmts->LOC();
    assert(n.IsNorm() && "Loop should be normalized before MaskGen.");
    ptr<AST::Call> vectorize = nullptr;
    if (!AST::NeedVectorize(n, vectorize)) return true;
    assert(li->IsInnermostLoop(lname));
    vector_width = AST::GetIntLiteral(vectorize->GetArguments()[1])->ValS32();
    vbool_ty = MakeVectorType(BaseType::BOOL, vector_width);

    auto iv = n.GetIV();
    auto iv_ty = iv->GetType();
    auto upper_bound = AST::Make<AST::Expr>(n.LOC(), "ubound", iv);
    upper_bound->SetType(MakeIntegerType());
    auto mask_expr = AST::Make<AST::Expr>(n.LOC(), "<=", iv, upper_bound);
    mask_expr->SetType(vbool_ty);
    auto data_type = AST::Make<AST::DataType>(loc, BaseType::BOOL);
    data_type->SetType(vbool_ty);
    auto loop_cond = AST::Make<AST::NamedVariableDecl>(loc, "exec", data_type,
                                                       nullptr, mask_expr);
    loop_cond->SetType(vbool_ty);
    loop_cond->SetDiversityShape(DiversityShape(DiversityShapeKind::DIVERGENT));
    SSTab().DefineSymbol(loop_cond->name_str, vbool_ty);
    di->AssignSymbolShape(InScopeName("exec"), loop_cond->GetDiversityShape());

    auto cur_mask = AST::Make<AST::NamedVariableDecl>(
        loc, MaskName(), data_type, nullptr,
        AST::MakeIdExpr(loc, loop_cond->name_str));

    cur_mask->SetType(vbool_ty);
    cur_mask->SetDiversityShape(DiversityShape(DiversityShapeKind::DIVERGENT));
    SSTab().DefineSymbol(cur_mask->name_str, vbool_ty);
    di->AssignSymbolShape(InScopeName(cur_mask->name_str),
                          cur_mask->GetDiversityShape());

    exec = loop_cond;
    mask_stack.push(cur_mask->name_str);

    n.stmts->Insert(loop_cond, 0);
    n.stmts->Insert(cur_mask, 1);
    if (debug_visit) {
      dbgs() << "[mask] Inserted loop condition(exec): " << STR(loop_cond)
              << " at the beginning of loop: " << n.GetIV()->name << "\n";
      dbgs() << "[mask] Inserted scoped mask: " << STR(cur_mask)
             << " at the beginning of loop: " <<  n.GetIV()->name << "\n";
    }

    return true;
  }

  bool Visit(AST::IfElseBlock& n) override {
    TraceEachVisit(n);
    if (!NeedTransform()) return true;
    auto loc = n.if_stmts->LOC();
    auto pred = n.GetPred();
    auto pred_ds = pred->GetDiversityShape();
    if (pred_ds.Uniform()) return true;
    assert(pred_ds.Divergent() && "predicate should be divergent in MaskGen.");
    assert(!n.HasElse() &&
           "if-else block should not have else branch in MaskGen.");
    auto data_type = AST::Make<AST::DataType>(loc, BaseType::BOOL);
    data_type->SetType(vbool_ty);

    // divergent branch
    auto top_mask = mask_stack.top();
    // current mask is the conjunction of the top mask and the predicate of the
    // if block
    auto mask_expr = AST::Make<AST::Expr>(n.LOC(), "&&",
                                          AST::MakeIdExpr(loc, top_mask), pred);
    mask_expr->SetType(vbool_ty);
    mask_expr->SetDiversityShape(DiversityShape(DiversityShapeKind::DIVERGENT));

    auto cur_mask = AST::Make<AST::NamedVariableDecl>(
        n.LOC(), MaskName(), data_type, nullptr, mask_expr);
    cur_mask->SetType(vbool_ty);
    cur_mask->SetDiversityShape(DiversityShape(DiversityShapeKind::DIVERGENT));
    SSTab().DefineSymbol(cur_mask->name_str, vbool_ty);
    di->AssignSymbolShape(InScopeName(cur_mask->name_str),
                          cur_mask->GetDiversityShape());
    // push the current mask to the stack
    mask_stack.push(cur_mask->name_str);
    auto cur_exec = AST::Make<AST::Assignment>(
        n.LOC(), "exec", AST::MakeIdExpr(loc, cur_mask->name_str));
    cur_exec->SetType(cur_mask->GetType());
    cur_exec->SetDiversityShape(DiversityShape(DiversityShapeKind::DIVERGENT));
    cur_exec->da->SetType(vbool_ty);

    auto stmts = n.if_stmts;
    stmts->Insert(cur_mask, 0);
    stmts->Insert(cur_exec, 1);

    if (debug_visit) {
      dbgs() << "[mask] Inserted scoped mask: " << STR(cur_mask)
             << " at the beginning of divergent branch: " << STR(pred) << "\n";
      dbgs() << "[mask] Inserted exec assignment: "
             << STR(cur_exec) << " after divergenet branch: "
             << STR(pred) << "\n";
    }
    return true;
  }

  bool BeforeAfterVisitImpl(AST::Node& n) override {
    if (auto if_block = dyn_cast<AST::IfElseBlock>(&n)) {
      if (if_block->IsDivergent() && mask_stack.size() > 1) {
        // pop the mask stack
        mask_stack.pop();
      }
    }

    return true;
  }
};

struct LoopHandler final : public VisitorWithSymTab {
  LoopHandler(const ptr<SymbolTable> s_tab)
      : VisitorWithSymTab("loop", s_tab) {}
  bool BeforeVisitImpl(AST::Node&) override { return true; }
  bool AfterVisitImpl(AST::Node&) override { return true; }
  bool RunOnProgram(AST::Node& root) override {
    if (!isa<AST::Program>(&root)) {
      Error(root.LOC(), "Not running a choreo program.");
      return false;
    }
    if (prt_visitor) dbgs() << "|- " << GetName() << NewL;

    LoopChecker lc;
    lc.SetDebugVisit(debug_visit);
    lc.SetTraceVisit(trace_visit);
    root.accept(lc);
    if (HasError() || abend_after) return false;
    if (!lc.IsAllLoopNorm()) return true;
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

    LoopVectorizeLegalityChecker lvlc(li);
    lvlc.SetDebugVisit(debug_visit);
    lvlc.SetTraceVisit(trace_visit);
    root.accept(lvlc);
    if (HasError() || abend_after) return false;
    if (prt_visitor) dbgs() << " |- " << lvlc.GetName() << NewL;

    if (lc.HasVectorization()) {
      if (debug_visit)
        dbgs() << "\n[diversity] start diversity analysis.\n";
      DiversityAnalysisHandler da(SymTab(), li);
      da.SetDebugVisit(debug_visit);
      da.SetTraceVisit(trace_visit);
      da.RunOnProgram(root); 
      if (prt_visitor) dbgs() << " |- " << da.GetName() << NewL;
      if (HasError() || abend_after) return false;
      auto di = da.GetDiversityAnalysis();

      if (debug_visit)
        dbgs() << "\n[linearize] start linearization.\n";
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

      if (debug_visit)
        dbgs() << "\n[mask] start geneating masks.\n";
      MaskGen mg(SymTab(), li, di);
      mg.SetDebugVisit(debug_visit);
      mg.SetTraceVisit(trace_visit);
      root.accept(mg);
      if (prt_visitor) dbgs() << " |- " << mg.GetName() << NewL;
      if (HasError() || abend_after) return false;
    }

    return true;
  }
};

} // namespace Choreo
#endif // __CHOREO_DIVERGENT_ANALYSIS_HPP__