#ifndef __CHOREO_SCALAR_EVOLUTION_HPP__
#define __CHOREO_SCALAR_EVOLUTION_HPP__

#include "ast.hpp"
#include "context.hpp"
#include "loop_utils.hpp"
#include "types.hpp"
#include "visitor.hpp"
#include <string>
#include <unordered_map>

namespace Choreo {

// scev(scalar evolution) is a way to represent how a variable changes over
// iterations in a loop. The design of scev refers to scalar evolution in LLVM.
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
  std::set<std::string> with_syms; // with symbols defined in with-in blocks
  ptr<AST::Program> root_ptr;
  std::string appointed_loop;
  std::string indent = "    ";

private:
  bool InLoop();

  bool InAppointedLoop();

  bool NeedAnalyze(ptr<Type> ty) {
    return IsActualBoundedIntegerType(ty) || isa<ScalarIntegerType>(ty);
  }

  std::string SymName(std::string name) {
    if (!PrefixedWith(name, "::"))
      return InScopeName(name);
    else
      return name;
  }

  bool IsAssignedSym(const std::string& sym_name) {
    ptr<Loop> loop = cur_loop;
    while (true) {
      auto loop_name = loop ? loop->LoopName() : NoLoopName();
      if (ssetab->IsAssignedInLoop(sym_name, loop_name)) return true;
      if (!loop) break;
      auto parent_loop = loop->GetParentLoop();
      loop = parent_loop;
    }
    return false;
  }

  void AssignSCEVToSym(const std::string& sym_name, ptr<SCEV> scev,
                       std::string loop_name) {
    ssetab->scoped_scev[loop_name][sym_name] = scev;
  }

  ptr<SCEV> GetSCEVOfSym(const std::string& sym_name) {
    auto loop = cur_loop;
    while (true) {
      auto loop_name = loop ? loop->LoopName() : NoLoopName();
      auto scev = ssetab->GetSCEV(sym_name, loop_name);
      if (scev) return scev;
      if (!loop) break;
      auto parent_loop = loop->GetParentLoop();
      loop = parent_loop;
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
  void ReComputeInLoop(const std::string& lname, bool debug = true);

public:
  bool Visit(AST::Program& n) override;
  bool Visit(AST::Expr& n) override;
  bool Visit(AST::CastExpr& n) override;
  bool Visit(AST::NamedVariableDecl& n) override;
  bool Visit(AST::Identifier& n) override;
  bool Visit(AST::Assignment& n) override;
  bool Visit(AST::Call& n) override;
  bool Visit(AST::DataAccess& n) override;
  bool Visit(AST::WithIn& n) override;
  bool Visit(AST::ForeachBlock& n) override;
  bool Visit(AST::ParallelBy& n) override;
  bool Visit(AST::Parameter& n) override;
}; // end class ScalarEvolution

} // end namespace Choreo

#endif