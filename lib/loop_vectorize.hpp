#ifndef __CHOREO_DIVERGENT_ANALYSIS_HPP__
#define __CHOREO_DIVERGENT_ANALYSIS_HPP__
#include "ast.hpp"
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

  bool Visit(AST::IfElseBlock& n) override {
    TraceEachVisit(n);
    if (!NeedCheck()) return true;
    if (!n.IsNorm()) {
      Error(n.LOC(), "cannot vectorize non-normalized if-else block.");
      return false;
    }
    return true;
  }
};

struct MaskGen final : public LoopVisitor {
private:
  ptr<LoopInfo> li;
  std::stack<ptr<AST::Node>> mask_stack;
  ptr<AST::Node> exec;
  ptr<AST::MultiNodes> inserted_stmts;
  int mask_count = 0;


  inline std::string MaskName() {
    return "mask" + std::to_string(mask_count++);
  }

public:
  MaskGen(const ptr<SymbolTable> s_tab, ptr<LoopInfo> l)
      : LoopVisitor(s_tab, "mask"), li(l) {}

  bool Visit(AST::MultiNodes& n) override {
    TraceEachVisit(n);
    inserted_stmts = dyn_cast<AST::MultiNodes>(n.CloneImpl());
    return true;
  }

  bool Visit(AST::ForeachBlock& n) override {
    TraceEachVisit(n);

    auto loc = n.LOC();
    assert(n.IsNorm() && "Loop should be normalized before MaskGen.");
    ptr<AST::Call> vectorize = nullptr;
    if (!AST::NeedVectorize(n, vectorize)) return true;
    assert(li->IsInnermostLoop(lname));
    auto vector_width = AST::GetIntLiteral(vectorize->GetArguments()[1]);
    auto iv = n.GetIV();
    auto iv_ty = iv->GetType();
    auto upper_bound = AST::Make<AST::Expr>(n.LOC(), "ubound", iv);
    upper_bound->SetType(MakeIntegerType());
    auto mask_expr = AST::Make<AST::Expr>(n.LOC(), "<=", iv, upper_bound);
    auto vbool_ty = MakeVectorType(BaseType::BOOL, vector_width->ValS32());
    mask_expr->SetType(vbool_ty);
    auto data_type = AST::Make<AST::DataType>(loc, BaseType::BOOL);
    data_type->SetType(vbool_ty);
    auto loop_cond = AST::Make<AST::NamedVariableDecl>(loc, "exec", data_type,
                                                       nullptr, mask_expr);
    loop_cond->SetType(vbool_ty);
    SSTab().DefineSymbol(loop_cond->name_str, loop_cond->GetType());

    auto cur_mask = AST::Make<AST::NamedVariableDecl>(
        loc, MaskName(), data_type, nullptr, loop_cond);
    cur_mask->SetType(vbool_ty);
    SSTab().DefineSymbol(cur_mask->name_str, cur_mask->GetType());
    exec = loop_cond;
    mask_stack.push(exec);

    n.stmts->Insert(loop_cond, 0);
    return true;
  }

  bool Visit(AST::IfElseBlock& n) override {
    TraceEachVisit(n);
    auto Loop = li->loops.find(lname);
    if (Loop == li->loops.end()) return true;

    assert(n.IsNorm());

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
      DiversityAnalysisHandler da(SymTab(), li);
      da.SetDebugVisit(debug_visit);
      da.SetTraceVisit(trace_visit);
      da.RunOnProgram(root);
      if (prt_visitor) dbgs() << " |- " << da.GetName() << NewL;
      if (HasError() || abend_after) return false;
      auto di = da.GetDiversityAnalysis();

      MaskGen mg(SymTab(), li);
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