#ifndef __CHOREO_DIVERGENT_ANALYSIS_HPP__
#define __CHOREO_DIVERGENT_ANALYSIS_HPP__

#include "ast.hpp"
#include "context.hpp"
#include "diversity_analysis.hpp"
#include "loop_utils.hpp"
#include "scalar_evolution.hpp"
#include "visitor.hpp"
#include <unordered_map>

namespace Choreo {

// LoopChecker is used to find one foreachblock if has vectorization hint and if
// all loops are normalized
struct VectorizationHintChecker final : public VisitorWithScope {
  VectorizationHintChecker();

  bool has_hint = false;
  bool BeforeVisitImpl(AST::Node&) override;
  bool AfterVisitImpl(AST::Node&) override;
  bool Visit(AST::ForeachBlock& n) override;

  bool HasVectorizationHint() const;
  bool IsAllLoopNorm() const;
};

// LoopAnalysis
struct LoopAnalysis final : public VisitorWithSymTab {
  LoopAnalysis(const ptr<SymbolTable>& s_tab);

  ptr<LoopInfo> li;
  std::string parent_loop_name = "";
  bool BeforeVisitImpl(AST::Node&) override;
  bool AfterVisitImpl(AST::Node&) override;

  bool Visit(AST::ForeachBlock& n) override;
  ptr<LoopInfo> GetLoopInfo() const;
};

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
  9. it ignores data dependence analysis for now.
  10. it must be inside Parallelby.
  we use two-pass checking:
  1st pass: LoopVectorizeSimpleChecker, check simple conditions unrelated to
vector factor. 2nd pass: LoopVectorizeLegalityChecker, check other conditions
related to vector factor, like data access alignment.
*/

// LoopVectorizeSimpleChecker
struct LoopVectorizeSimpleChecker final : public LoopVisitor {
private:
  ptr<LoopInfo> li;
  std::unordered_map<std::string, bool> can_vectorizes;
  std::unordered_map<std::string, std::vector<location>> loop_defs;
  std::unordered_map<std::string, std::vector<location>> loop_uses;
  int Pb_level = 0;
  std::string indent = "    ";

  bool NeedCheck();
  void SetLoopVectorizationFailed();
  void AddLoopUse(std::string sym, location);
  void FindLoopUses(ptr<AST::Node> n);
  void AddLoopDef(std::string sym, location);

public:
  LoopVectorizeSimpleChecker(const ptr<SymbolTable> s_tab, ptr<LoopInfo> l);
  bool AfterBeforeVisitImpl(AST::Node& n) override;
  bool BeforeAfterVisitImpl(AST::Node& n) override;

  bool Visit(AST::ForeachBlock& n) override;
  bool Visit(AST::WhileBlock& n) override;
  bool Visit(AST::DMA& n) override;
  bool Visit(AST::Wait& n) override;
  bool Visit(AST::Trigger& n) override;
  bool Visit(AST::Rotate& n) override;
  bool Visit(AST::Return& n) override;
  bool Visit(AST::Select& n) override;
  bool Visit(AST::IncrementBlock& n) override;
  bool Visit(AST::Synchronize& n) override;
  bool Visit(AST::InThreadsBlock& n) override;
  bool Visit(AST::ParallelBy& n) override;
  bool Visit(AST::Call& n) override;

  // stmts
  bool Visit(AST::NamedVariableDecl& n) override;
  bool Visit(AST::Assignment& n) override;
  bool Visit(AST::DataAccess& n) override;

  bool ExistLoopVectorizationLegal();
};

// LoopVectorizeLegalityChecker
struct LoopVectorizeLegalityChecker final : public LoopVisitor {
private:
  ptr<LoopInfo> li;
  ptr<ScopedSCEVTable> scev_table;
  std::string appointed_loop;
  ptr<AST::Program> root_ptr;
  std::string indent = "    ";
  // target architecture limits
  std::map<ArchId, size_t> max_limits; // arch limits

  bool NeedCheck();
  void SetLoopVectorizationFailed();
  bool CheckDataAccessAlignment(AST::DataAccess& n);
  bool BeforeAfterVisitImpl(AST::Node& n) override;

public:
  std::unordered_map<std::string, bool> can_vectorizes;
  LoopVectorizeLegalityChecker(const ptr<SymbolTable> s_tab, ptr<LoopInfo> l,
                               ptr<ScopedSCEVTable> s);
  void CheckLegalityInLoop(AST::ForeachBlock& loop_node);
  bool IsLegalToVectorize(const std::string& lname);
  bool Visit(AST::ForeachBlock& n) override;
  bool Visit(AST::DataAccess& n) override;

  bool ExistLoopVectorizationLegal();
};

struct BranchSimplicition final : public LoopVisitor {
public:
  BranchSimplicition(const ptr<SymbolTable> s_tab);
  bool Visit(AST::IfElseBlock& n) override;
};

// linearize branch inside vectorized loops
struct Linearizer final : public LoopVisitor {
private:
  ptr<LoopInfo> li;
  ptr<DiversityInfo> di;

  bool NeedLinearize();

public:
  Linearizer(const ptr<SymbolTable> s_tab, ptr<LoopInfo> l,
             ptr<DiversityInfo> d);
  bool Visit(AST::MultiNodes& n) override;
};

// Masking generation for vectorized loops
// todo: break/continue statement
struct MaskGen final : public LoopVisitor {
private:
  ptr<LoopInfo> li;
  ptr<DiversityInfo> di;
  std::stack<std::string> mask_stack;

  std::string MaskName();
  bool NeedTransform();
  bool HasDivergentBranch(AST::MultiNodes& n);
  ptr<AST::Expr> MakeMaskExpr(const location& loc, const ptr<AST::Node>& lhs,
                              const ptr<AST::Node>& rhs = nullptr,
                              std::string op = "");
  ptr<AST::Expr> MakeMaskIdExpr(const location& loc, const std::string& name);
  ptr<AST::NamedVariableDecl> MakeMaskDecl(const location& loc,
                                           const std::string& name,
                                           const ptr<AST::Expr>& rhs);
  ptr<AST::Assignment> MakeMaskAssign(const location& loc,
                                      const std::string& name,
                                      const ptr<AST::Expr>& rhs);

public:
  MaskGen(const ptr<SymbolTable> s_tab, ptr<LoopInfo> l, ptr<DiversityInfo> d);
  bool Visit(AST::MultiNodes& n) override;
  bool Visit(AST::ForeachBlock& n) override;
  bool Visit(AST::IfElseBlock& n) override;
  bool Visit(AST::DataAccess& n) override;
  bool BeforeAfterVisitImpl(AST::Node& n) override;
};

struct LoopVectorizer final : public VisitorWithSymTab {
private:
  ptr<LoopInfo> li = nullptr;
  ptr<DiversityInfo> di = nullptr;
  ptr<ScopedSCEVTable> scev_tab = nullptr;

public:
  LoopVectorizer();
  bool BeforeVisitImpl(AST::Node&) override;
  bool AfterVisitImpl(AST::Node&) override;
  bool RunOnProgramImpl(AST::Node& root) override;

  bool CheckVectorizationHint(AST::Node& root);
  bool AnalyzeLoops(AST::Node& root);
  bool CheckSimply(AST::Node& root);
  bool AnalyzeDiversityShape(AST::Node& root);
  bool ComputeVectorizationPlan(AST::Node& root);
  bool InferenceType(AST::Node& root);
  bool LinearizeBranch(AST::Node& root);
  bool GenerateMask(AST::Node& root);
};

} // namespace Choreo
#endif // __CHOREO_DIVERGENT_ANALYSIS_HPP__
