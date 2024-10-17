#ifndef __CHOREO_FACTOR_TRANS_HPP__
#define __CHOREO_FACTOR_TRANS_HPP__

// This is very specific to factor, since factor can only use select expressions

#include "visitor.hpp"

namespace Choreo {

struct FactorTrans : public VisitorWithSymTab {
 private:
  std::ostream &os;
  size_t error_count = 0;

  ptr<FutureBufferMap> fut_buf; // map a future to its associated buffer

  std::vector<AST::Swap *> cur_swaps;
  std::unordered_map<AST::Swap *, std::unordered_map<std::string, std::string>>
      swap_pre;
  std::unordered_map<AST::Swap *, std::unordered_map<std::string, std::string>>
      swap_post;

  const std::string NameToReplace(const std::string &name) const {
    for (auto &item : swap_pre) {
      if (item.second.count(name)) return item.second.at(name);
    }
    for (auto &item : swap_post) {
      if (item.second.count(name)) return item.second.at(name);
    }
    return name;  // no replacement
  }

 private:
  bool BeforeVisitImpl(AST::Node &n) {
    if (auto f = dyn_cast<AST::ForeachBlock>(&n)) {
      for (auto &stmt : f->stmts->AllSubs()) {
        if (auto swap = dyn_cast<AST::Swap>(stmt)) cur_swaps.push_back(swap);
      }
    }
    return true;
  }

  bool AfterVisitImpl(AST::Node &n) {
    if (auto f = dyn_cast<AST::ForeachBlock>(&n)) {
      cur_swaps.clear();
      swap_pre.clear();
      swap_post.clear();
    }
    return true;
  }

 public:
  FactorTrans(const ptr<SymbolTable> s_tab, const ptr<FutureBufferMap> & fb, std::ostream &o = std::cout)
      : VisitorWithSymTab("dynshape", s_tab), os(o), fut_buf(fb) {}
  ~FactorTrans() {}

  bool Visit(AST::MultiNodes &) { return true; }
  bool Visit(AST::MultiValues &) { return true; }
  bool Visit(AST::IntLiteral &) { return true; }
  bool Visit(AST::Boolean &) { return true; }
  bool Visit(AST::Expr &) { return true; }
  bool Visit(AST::MultiDimSpans &) { return true; }
  bool Visit(AST::NamedTypeDecl &) { return true; }
  bool Visit(AST::NamedVariableDecl &) { return true; }
  bool Visit(AST::IntTuple &) { return true; }
  bool Visit(AST::Assignment &) { return true; }
  bool Visit(AST::IntIndex &) { return true; }
  bool Visit(AST::DataType &) { return true; }
  bool Visit(AST::Identifier &n) {
    n.name = NameToReplace(n.name);
    return true;
  }
  bool Visit(AST::Parameter &) { return true; }
  bool Visit(AST::ParamList &) { return true; }
  bool Visit(AST::ParallelBy &) { return true; }
  bool Visit(AST::WhereBind &) { return true; }
  bool Visit(AST::WithIn &) { return true; }
  bool Visit(AST::WithBlock &) { return true; }
  bool Visit(AST::Memory &) { return true; }
  bool Visit(AST::SpanAs &) { return true; }
  bool Visit(AST::DMA &) { return true; }
  bool Visit(AST::ChunkAt &) { return true; }
  bool Visit(AST::Wait &) { return true; }
  bool Visit(AST::Call &) { return true; }
  bool Visit(AST::Swap &n) {
    if (swap_pre.count(&n)) swap_pre.erase(&n);
    return true;
  }
  bool Visit(AST::Select &) { return true; }
  bool Visit(AST::Return &) { return true; }
  bool Visit(AST::LoopRange &) { return true; }

  bool Visit(AST::ForeachBlock &n) {
    if (cur_swaps.empty()) return true;

    if (n.ranges->Count() > 1)
      choreo_unreachable("swapping inside multi-bounds is yet to support.");

    auto &ranges = n.getRanges();
    auto iv_name = cast<AST::LoopRange>(ranges[0])->iv->name;
    auto lbound = cast<AST::LoopRange>(ranges[0])->lbound;
    if (!IsValidBound(lbound)) lbound = 0;

    auto Condition = AST::Make<AST::Expr>(
        n.LOC(), "%",
        AST::Make<AST::Expr>(n.LOC(),
                             AST::Make<AST::Identifier>(n.LOC(), iv_name)),
        AST::Make<AST::Expr>(n.LOC(), AST::Make<AST::IntLiteral>(n.LOC(), 2)));
    Condition->SetType(MakeIntegerType());

    std::vector<ptr<AST::Node>> new_stmts;

    // NOTE: must take care of the symbols and associated types
    for (auto swap : cur_swaps) {
      auto sty = swap->lhs->GetType();
      auto lname = swap->lhs->name;
      auto rname = swap->rhs->name;
      auto lr_list =
          AST::Make<AST::MultiValues>(n.LOC(), ", ", swap->lhs, swap->rhs);
      auto rl_list =
          AST::Make<AST::MultiValues>(n.LOC(), ", ", swap->rhs, swap->lhs);
      auto true_on_lhs = AST::Make<AST::Select>(n.LOC(), Condition, lr_list);
      auto true_on_rhs = AST::Make<AST::Select>(n.LOC(), Condition, rl_list);
      true_on_lhs->SetType(sty);
      true_on_rhs->SetType(sty);
      auto lbs =
          AST::Make<AST::Assignment>(n.LOC(), lname + "_pre_swap", true_on_lhs);
      auto las = AST::Make<AST::Assignment>(n.LOC(), lname + "_post_swap",
                                            true_on_rhs);
      auto rbs =
          AST::Make<AST::Assignment>(n.LOC(), rname + "_pre_swap", true_on_rhs);
      auto ras = AST::Make<AST::Assignment>(n.LOC(), rname + "_post_swap",
                                            true_on_lhs);
      lbs->SetType(sty);
      las->SetType(sty);
      rbs->SetType(sty);
      ras->SetType(sty);

      // record the name mapping
      swap_pre[swap].emplace(lname, lname + "_pre_swap");
      swap_pre[swap].emplace(rname, rname + "_pre_swap");
      swap_post[swap].emplace(lname, lname + "_post_swap");
      swap_post[swap].emplace(rname, rname + "_post_swap");

      // Add it into the new stmts
      new_stmts.push_back(lbs);
      new_stmts.push_back(las);
      new_stmts.push_back(rbs);
      new_stmts.push_back(ras);
    }

    n.stmts->values.insert(n.stmts->values.begin(), new_stmts.begin(),
                           new_stmts.end());

    std::cout << PSTR(n.stmts);
    return true;
  }

  bool Visit(AST::FunctionDecl &) { return true; }
  bool Visit(AST::ChoreoFunction &) { return true; }
  bool Visit(AST::CppSourceCode &) { return true; }
  bool Visit(AST::Program &) { return true; }

  bool HasError() { return false; }
};

}  // end namespace Choreo

#endif  // __CHOREO_FACTOR_TRANS_HPP__
