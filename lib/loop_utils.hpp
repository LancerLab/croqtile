#ifndef __CHOREO_MASK_GEN_HPP__
#define __CHOREO_MASK_GEN_HPP__

#include "ast.hpp"
#include "context.hpp"
#include "io.hpp"
#include "types.hpp"
#include "utils.hpp"
#include "visitor.hpp"
#include <ostream>
#include <string>
#include <unordered_map>
namespace Choreo {

struct LoopChecker final : public VisitorWithScope {
  LoopChecker() : VisitorWithScope("loop-checker") {}

  bool NeedVectorize = false;
  bool AllNormLoop = true;
  bool BeforeVisitImpl(AST::Node&) override { return true; }
  bool AfterVisitImpl(AST::Node&) override { return true; }
  bool Visit(AST::ForeachBlock& n) override {
    if (!n.IsNorm()) AllNormLoop = false; // not all loops are normalized
    if (n.suffixs) {
      for (auto suffix : n.suffixs->values) {
        if (auto suffix_call = AST::GetCall(suffix);
            suffix_call->IsAnno() &&
            suffix_call->function->name == "vectorize") {
          NeedVectorize = true;
          return true;
        }
      }
    }
    return true;
  }

  bool HasVectorization() const { return NeedVectorize; }
  bool IsAllLoopNorm() const { return AllNormLoop; }
};

struct LoopVisitor : public VisitorWithSymTab {
protected:
  std::string lname;
  int loop_count = 0;

  void TraceEachVisit(const AST::Node& n) {
    if (trace_visit) { dbgs() << n.TypeNameString() << "\n"; }
  }

  void EnterLoopScope(const std::string& loop_name) {
    lname = lname + "::loop_" + loop_name + std::to_string(loop_count++);
    if (debug_visit && !lname.empty()) dbgs() << "Entering : " << lname << "\n";
  }

  void LeaveLoopScope() {
    if (debug_visit && !lname.empty()) dbgs() << "Leaving :  " << lname << "\n";
    size_t pos = lname.rfind("::loop_");
    if (pos != std::string::npos) {
      lname = lname.substr(0, pos);
    } else {
      lname.clear();
    }
  }
  virtual bool AfterBeforeVisitImpl(AST::Node&) { return true; }

  virtual bool BeforeAfterVisitImpl(AST::Node&) { return true; }

  bool BeforeVisitImpl(AST::Node& n) override {
    if (trace_visit) dbgs() << "before visiting " << n.TypeNameString() << "\n";
    if (auto loop = dyn_cast<AST::ForeachBlock>(&n)) {
      assert(loop->IsNorm() && "Loop should be normalized before LoopVisitor.");
      auto iv_name = loop->GetIV()->name;
      EnterLoopScope(iv_name);
    }
    AfterBeforeVisitImpl(n);
    return true;
  }

  bool AfterVisitImpl(AST::Node& n) override {
    if (trace_visit) dbgs() << "after visiting " << n.TypeNameString() << "\n";
    BeforeAfterVisitImpl(n);
    if (auto loop = dyn_cast<AST::ForeachBlock>(&n)) {
      assert(loop->IsNorm() && "Loop should be normalized before LoopVisitor.");
      LeaveLoopScope();
    }
    return true;
  }

public:
  LoopVisitor(const ptr<SymbolTable> s_tab, const std::string& pn)
      : VisitorWithSymTab(pn, s_tab) {}
};

struct Loop {
  std::string lname;
  ptr<AST::ForeachBlock> loop;
  std::vector<ptr<Loop>> sub_loops;

  ptr<AST::MultiNodes> GetStmts() const {
    if (loop) return loop->stmts;
    return nullptr;
  }

  explicit Loop(std::string n, const ptr<AST::ForeachBlock> l)
      : lname(n), loop(l), sub_loops() {
    assert(l && "Loop cannot be null.");
  }

  void dump(std::ostream& os, const std::string& prefix = "") const {
    if (!loop) return;
    os << prefix << lname << " //" << loop->LOC() << "\n";
    for (const auto& sub_loop : sub_loops) {
      sub_loop->dump(os, prefix + "  ");
    }
  }

  ptr<BoundedIntegerType> GetIVType() const {
    if (loop) {
      auto iv = loop->GetIV();
      if (auto iv_ty = dyn_cast<BoundedIntegerType>(iv->GetType()))
        return iv_ty;
      else if (auto iv_ty = dyn_cast<BoundedITupleType>(iv->GetType())) {
        assert(iv_ty->Dims() == 1 &&
               "IV type should be a single dimension tuple.");
        auto res = MakeBoundedIntegerType(iv_ty->GetUpperBounds().ValueAt(0));
        res->stride = iv_ty->GetStride(0);
        res->width = iv_ty->GetWidth(0);
        return res;
      }
    }
    return nullptr;
  }

  std::string IVName() { return loop->GetIV()->name; }
};

struct LoopInfo {
  std::unordered_map<std::string, std::string> iv2loop;
  std::unordered_map<std::string, Loop> loops;
  std::unordered_map<std::string, Loop> top_loops;

  std::string GetParentLoopName(const std::string& lname) const {
    auto removeLastLoop = [](const std::string& input) -> std::string {
      size_t lastPos = input.rfind("::");
      if (lastPos == std::string::npos) {
        return input; // No "::" found, return the original string
      }

      if (lastPos == 0) return "";
      return input.substr(0, lastPos);
    };

    std::string parent_loop_name = removeLastLoop(lname);
    while (!parent_loop_name.empty()) {
      if (loops.find(parent_loop_name) != loops.end()) {
        return parent_loop_name;
      }
      parent_loop_name = removeLastLoop(parent_loop_name);
    }
    return "";
  }

  std::string GetOutermostLoopName(const std::string& lname) const {
    auto parent_lname = lname;
    while (GetParentLoopName(parent_lname) != "") {
      parent_lname = GetParentLoopName(parent_lname);
    }

    return parent_lname;
  }

  bool IsInnermostLoop(const std::string& lname) const {
    return loops.find(lname) != loops.end() &&
           loops.at(lname).sub_loops.empty();
  }

  ptr<Loop> GetLoop(const std::string& lname) const {
    auto it = loops.find(lname);
    if (it != loops.end()) {
      return AST::Make<Loop>(it->second.lname, it->second.loop);
    }
    return nullptr;
  }

  void dump(std::ostream& os) {
    for (const auto& [loop_name, loop] : loops) { loop.dump(os); }
  }
};

struct LoopAnalysis final : public LoopVisitor {
  ptr<LoopInfo> li;

public:
  LoopAnalysis(const ptr<SymbolTable> s_tab)
      : LoopVisitor(s_tab, "loopanalysis"), li(AST::Make<LoopInfo>()) {}

  bool Visit(AST::ForeachBlock& n) override {
    auto iv = n.GetIV();
    li->iv2loop[InScopeName(iv->name)] = lname;
    Loop loop(lname, dyn_cast<AST::ForeachBlock>(n.CloneImpl()));
    li->loops.emplace(lname, loop);
    auto parent_loop_name = li->GetParentLoopName(lname);
    if (!parent_loop_name.empty()) {
      auto it = li->loops.find(parent_loop_name);
      if (it != li->loops.end()) {
        it->second.sub_loops.push_back(AST::Make<Loop>(lname, loop.loop));
      }
    }

    return true;
  }

  ptr<LoopInfo> GetLoopInfo() const { return li; }
};

} // end namespace Choreo
#endif // __CHOREO_MASK_GEN_HPP__