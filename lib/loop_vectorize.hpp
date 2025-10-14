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
struct LoopChecker final : public VisitorWithScope {
  LoopChecker();

  bool NeedVectorize = false;
  bool AllNormLoop = true;
  bool BeforeVisitImpl(AST::Node&) override;
  bool AfterVisitImpl(AST::Node&) override;
  bool Visit(AST::ForeachBlock& n) override;

  bool HasVectorization() const;
  bool IsAllLoopNorm() const;
};

// LoopAnalysis
struct LoopAnalysis final : public VisitorWithSymTab {
  LoopAnalysis();

  ptr<LoopInfo> li;
  std::string parent_loop_name = "";
  static int loop_count;
  static std::string GenerateLoopName();
  bool BeforeVisitImpl(AST::Node&) override;
  bool AfterVisitImpl(AST::Node&) override;

  bool Visit(AST::ForeachBlock& n) override;
  ptr<LoopInfo> GetLoopInfo() const;
};

struct LoopVectorizeLegalityChecker final : public LoopVisitor {
private:
  ptr<LoopInfo> li;
  ptr<ScopedSCEVTable> scev_table;
  BaseType data_type = BaseType::UNKNOWN;
  std::unordered_map<std::string, std::vector<location>> loop_defs;
  std::unordered_map<std::string, std::vector<location>> loop_uses;
  int Pb_level = 0;
  bool all_illegal = true;
  std::string indent = "";

  bool NeedCheck();
  bool CheckDataAccessAlignment(AST::DataAccess& n);

  void AddLoopUse(std::string sym, location);
  void FindLoopUses(ptr<AST::Node> n);
  void AddLoopDef(std::string sym, location);
  bool AfterBeforeVisitImpl(AST::Node& n) override;
  bool BeforeAfterVisitImpl(AST::Node& n) override;

public:
  LoopVectorizeLegalityChecker(const ptr<SymbolTable> s_tab, ptr<LoopInfo> l,
                               ptr<ScopedSCEVTable> s);

  bool HasVectorize() const;

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
  bool Visit(AST::DataAccess& n) override;
  // stmts
  bool Visit(AST::NamedVariableDecl& n) override;
  bool Visit(AST::Assignment& n) override;
  bool Visit(AST::Call& n) override;
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
  LoopVectorizer();
  bool BeforeVisitImpl(AST::Node&) override;
  bool AfterVisitImpl(AST::Node&) override;
  bool RunOnProgram(AST::Node& root) override;
};

} // namespace Choreo
#endif // __CHOREO_DIVERGENT_ANALYSIS_HPP__