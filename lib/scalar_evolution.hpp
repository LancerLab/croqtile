#ifndef __CHOREO_SCALAR_EVOLUTION_HPP__
#define __CHOREO_SCALAR_EVOLUTION_HPP__

#include "ast.hpp"
#include "context.hpp"
#include "loop_utils.hpp"
#include "symvals.hpp"
#include "types.hpp"
#include <sstream>
#include <string>
#include <unordered_map>

namespace Choreo {

struct SCEVVal : public SCEV {
  ValueItem value;
  ptr<Loop> loop = nullptr;
  SCEVVal(ValueItem v, ptr<Loop> l = nullptr) : value(v), loop(l) {}
  SCEVType GetType() const override { return Val; }
  std::string ToString() const override { return STR(value); }
  bool IsLoopInVariant(ptr<Loop> l) const override {
    assert(l && "loop cannot be null.");
    if (!loop) return true;
    return loop->HasLoop(l->lname);
  }
  ValueItem GetValue() const override { return value; }
  __UDT_TYPE_INFO__(SCEV, SCEVVal)
};

struct SCEVAddRecExpr : public SCEV {
  ptr<SCEV> base;
  ptr<SCEV> step;
  ptr<Loop> loop;
  ValueItem times = UncomputableValueItem(); // optional, for step * n
  SCEVAddRecExpr(ptr<SCEV> b, ptr<SCEV> s, ptr<Loop> l)
      : base(b), step(s), loop(l) {}
  SCEVType GetType() const override { return AddRecExpr; }
  std::string ToString() const override {
    std::ostringstream ss;
    ss << "{" << base->ToString() << ", +, " << step->ToString() << "} <"
       << loop->lname << ">";
    return ss.str();
  }
  bool IsLoopInVariant(ptr<Loop> l) const override {
    assert(l && loop && "loop cannot be null.");
    return loop->HasLoop(l->lname);
  }
  ValueItem GetValue() const override { return UncomputableValueItem(); }
  __UDT_TYPE_INFO__(SCEV, SCEVAddRecExpr)
};

inline ptr<SCEVAddRecExpr> MakeSCEVAddRecExpr(ptr<SCEV> base, ptr<SCEV> step,
                                              ptr<Loop> loop) {
  return std::make_shared<SCEVAddRecExpr>(base, step, loop);
}
inline ptr<SCEVVal> MakeSCEVVal(ValueItem v, ptr<Loop> loop = nullptr) {
  return std::make_shared<SCEVVal>(v, loop);
}
inline ptr<SCEVAddRecExpr> MakeSCEVAddRecExpr(ValueItem base, ValueItem step,
                                              ptr<Loop> loop) {
  return std::make_shared<SCEVAddRecExpr>(MakeSCEVVal(base), MakeSCEVVal(step),
                                          loop);
}
inline ptr<SCEVAddRecExpr> MakeSCEVAddRecExpr(ptr<SCEV> base, ValueItem step,
                                              ptr<Loop> loop) {
  return std::make_shared<SCEVAddRecExpr>(base, MakeSCEVVal(step), loop);
}

inline std::string STR(const ptr<SCEV>& scev) {
  if (!scev) return "UNKNOWN";
  return scev->ToString();
}

struct ScopedSCEVTable {
  ptr<LoopInfo> li;
  std::unordered_map<std::string, std::unordered_map<std::string, ptr<SCEV>>>
      scoped_scev; // k: loop-name, v: (k: iv-name, v: scev)

  void dump(std::ostream& os) {
    for (auto& item : scoped_scev) {
      os << "Loop: " << item.first << "\n";
      for (auto& iv_scev : item.second) {
        os << "  " << iv_scev.first << " -> " << STR(iv_scev.second) << "\n";
      }
    }
  }

  ptr<SCEV> GetSCEV(const std::string& sym_name, const std::string& loop_name) {
    auto it = scoped_scev.find(loop_name);
    if (it != scoped_scev.end()) {
      auto& scev_map = it->second;
      auto scev_it = scev_map.find(sym_name);
      if (scev_it != scev_map.end()) { return scev_it->second; }
    }
    return nullptr;
  }

  bool IsAssignedInLoop(const std::string& sym_name,
                        const std::string& loop_name) {
    auto it = scoped_scev.find(loop_name);
    if (it != scoped_scev.end()) {
      auto& scev_map = it->second;
      return scev_map.find(sym_name) != scev_map.end();
    }
    return false;
  }

  bool AssignSCEV(const std::string& sym_name, const std::string& loop_name,
                  ptr<SCEV> scev) {
    if (IsAssignedInLoop(sym_name, loop_name)) {
      choreo_unreachable("SCEV for IV `" + sym_name + "` in loop `" +
                         loop_name + "` has been assigned.");
      return false;
    }
    scoped_scev[loop_name][sym_name] = scev;
    return true;
  }
};

// only analyze scalar evolution for scalars including immutables/mutables and
// bounded variables, do not analyze arrays, spanns, references, dmas, etc.
class ScalarEvolutionAnalysis final : public LoopVisitor {
private:
  ptr<LoopInfo> li;
  ptr<ScopedSCEVTable> ssetab;

private:
  bool InLoop();
  bool InVectorizedLoop();
  bool NeedAnalyze(ptr<Type> ty) {
    if (InAnno) return false; // do not analyze scalar evolution in annotation
    return IsActualBoundedIntegerType(ty) || isa<ScalarIntegerType>(ty);
  }

  std::string SymName(std::string name) {
    if (!PrefixedWith(name, "::"))
      return InScopeName(name);
    else
      return name;
  }

  bool IsAssignedSym(const std::string& sym_name) {
    auto loop_name = InLoop() ? lname : NoLoopName();
    while (true) {
      if (ssetab->IsAssignedInLoop(sym_name, loop_name)) return true;
      auto parent_loop = li->GetParentLoopName(loop_name);
      if (parent_loop.empty()) break;
      loop_name = parent_loop;
    }
    return false;
  }

  void AssignSCEVToSym(const std::string& sym_name, ptr<SCEV> scev,
                       std::string loop_name) {
    ssetab->scoped_scev[loop_name][sym_name] = scev;
  }

  ptr<SCEV> GetSCEVOfSym(const std::string& sym_name) {
    auto loop_name = InLoop() ? lname : NoLoopName();
    while (true) {
      auto scev = ssetab->GetSCEV(sym_name, loop_name);
      if (scev) return scev;
      auto parent_loop = li->GetParentLoopName(loop_name);
      if (parent_loop.empty()) break;
      loop_name = parent_loop;
    }
    return nullptr;
  }

  ptr<SCEV> ComputeARSCEV(ptr<SCEV> lhs, ptr<SCEV> rhs, std::string op);

public:
  ScalarEvolutionAnalysis(const ptr<SymbolTable> s_tab, ptr<LoopInfo> l)
      : LoopVisitor(s_tab, "scalar_evolution"), li(l),
        ssetab(AST::Make<ScopedSCEVTable>()) {
    assert(s_tab != nullptr);
  }

  ptr<ScopedSCEVTable> GetScevTab() { return ssetab; }

public:
  bool Visit(AST::Expr& n) override;
  bool Visit(AST::NamedVariableDecl& n) override;
  bool Visit(AST::Identifier& n) override;
  bool Visit(AST::Assignment& n) override;
  bool Visit(AST::Call& n) override;
  bool Visit(AST::ForeachBlock& n) override;
  bool Visit(AST::ParallelBy& n) override;
  bool Visit(AST::Parameter& n) override;
}; // end class ScalarEvolution

} // end namespace Choreo

#endif