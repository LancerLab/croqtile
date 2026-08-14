#ifndef __CHOREO_MAP_HOIST_HPP__
#define __CHOREO_MAP_HOIST_HPP__

#include <algorithm>
#include <map>
#include <set>
#include <stack>
#include <vector>

#include "ast.hpp"
#include "context.hpp"
#include "visitor.hpp"

namespace Choreo {

// Optimization: hoist loop-invariant `buffer.map` statements out of `foreach`
// loops so the mapping happens once before the loop instead of once per
// iteration. The matching unmap is emitted by the code generator at the map's
// enclosing scope exit, so relocating the map out of the loop also sinks the
// unmap out of the loop automatically.
//
// This pass runs after semantic analysis (type inference), so it must also
// re-scope the map result symbol (and its `.data`/`.span` companions) from the
// loop scope into the enclosing scope. Otherwise code generation, which
// resolves symbols by walking up from the current scope, would fail to find
// the relocated result.
struct MapHoist final : public VisitorWithSymTab {
  struct LoopFrame {
    AST::ForeachBlock* loop = nullptr; // the loop being exited
    AST::MultiNodes* parent = nullptr; // MultiNodes that holds the loop
    int index = -1;                    // loop index within its parent
    std::set<std::string> iv_names;    // iteration variables of this loop
  };

  std::stack<AST::MultiNodes*> multi_nodes;
  std::vector<LoopFrame> loops;
  // Deferred insertions keyed by the parent MultiNodes that will receive the
  // hoisted nodes. Insertions are applied in Visit(MultiNodes) after all of
  // its children have been traversed, so we never mutate a node list while it
  // is still being iterated.
  std::map<AST::MultiNodes*, std::map<int, std::vector<ptr<AST::Node>>>>
      mnodes_insertions;

  MapHoist() : VisitorWithSymTab("map-hoist") {}

  bool BeforeVisitImpl(AST::Node& n) override {
    if (auto mn = dyn_cast<AST::MultiNodes>(&n)) {
      multi_nodes.push(mn);
    } else if (auto fb = dyn_cast<AST::ForeachBlock>(&n)) {
      LoopFrame frame;
      frame.loop = fb;
      frame.parent = multi_nodes.empty() ? nullptr : multi_nodes.top();
      if (frame.parent) frame.index = frame.parent->GetIndex(fb);
      for (const auto& rng : fb->GetRanges())
        frame.iv_names.insert(cast<AST::LoopRange>(rng)->GetIVName());
      loops.push_back(frame);
    }
    return true;
  }

  bool Visit(AST::MultiNodes& n) override {
    auto it = mnodes_insertions.find(&n);
    if (it != mnodes_insertions.end()) {
      // Insert from the highest index down so lower indices stay valid while
      // we shift the list.
      for (auto rit = it->second.rbegin(); rit != it->second.rend(); ++rit) {
        const auto& nodes = rit->second;
        auto pos = n.values.begin() + rit->first;
        n.values.insert(pos, nodes.begin(), nodes.end());
      }
      mnodes_insertions.erase(it);
    }
    multi_nodes.pop();
    return true;
  }

  bool AfterVisitImpl(AST::Node& n) override {
    if (isa<AST::ForeachBlock>(&n)) {
      HoistInvariantMaps(loops.back());
      loops.pop_back();
    }
    return true;
  }

  void HoistInvariantMaps(const LoopFrame& frame) {
    auto* fb = frame.loop;
    if (!frame.parent || frame.index < 0 || !fb->stmts) return;

    std::vector<ptr<AST::Node>> hoisted;
    for (const auto& sub : fb->stmts->values) {
      auto bm = dyn_cast<AST::BufferMap>(sub);
      if (bm && bm->IsMap() && IsLoopInvariant(*bm, frame.iv_names))
        hoisted.push_back(sub);
    }
    if (hoisted.empty()) return;

    std::set<AST::Node*> removed;
    for (const auto& h : hoisted) removed.insert(h.get());
    auto& body = fb->stmts->values;
    body.erase(std::remove_if(body.begin(), body.end(),
                              [&removed](const ptr<AST::Node>& p) {
                                return removed.count(p.get()) > 0;
                              }),
               body.end());

    // Re-scope each hoisted result before the node is relocated so later
    // passes (notably code generation) resolve it from the enclosing scope.
    for (const auto& h : hoisted)
      if (auto bm = dyn_cast<AST::BufferMap>(h); bm && !bm->result.empty())
        RescopeResult(bm->result);

    auto& pending = mnodes_insertions[frame.parent][frame.index];
    for (const auto& h : hoisted) pending.push_back(h);

    VST_DEBUG(for (const auto& h : hoisted) dbgs()
              << "[MapHoist] Hoisted buffer.map out of loop: " << PSTR(h)
              << "\n");
  }

  // Move a buffer.map result symbol from the current (loop) scope into its
  // parent scope. Early semantics registers the result plus `.data` and
  // `.span` companion symbols, all of which must move together.
  void RescopeResult(const std::string& name) {
    auto* tab = SymTab().get();
    const std::string old_scope = scoped_symtab.ScopeName();
    const std::string new_scope = scoped_symtab.getParentScope();
    if (new_scope.empty() || new_scope == old_scope) return;

    static const char* kSuffixes[] = {"", ".data", ".span"};
    for (const char* suffix : kSuffixes) {
      const std::string old_scoped = old_scope + name + suffix;
      auto* sym = tab->GetSymbol(old_scoped);
      if (!sym) continue;
      auto ty = sym->GetType();
      tab->RemoveSymbol(old_scoped);
      tab->AddSymbol(new_scope + name + suffix, ty);
    }
  }

  bool IsLoopInvariant(const AST::BufferMap& bm,
                       const std::set<std::string>& iv_names) const {
    return !RefersTo(bm.source, iv_names) && !RefersTo(bm.offset, iv_names) &&
           !RefersTo(bm.size, iv_names);
  }

  // True if the node references any symbol in `names`. Reuses the shared
  // ReferredSymbols collector (unscoped), so `names` must be unscoped too.
  static bool RefersTo(const ptr<AST::Node>& n,
                       const std::set<std::string>& names) {
    if (!n) return false;
    for (const auto& sym : ReferredSymbols(n.get())) {
      if (names.count(sym)) return true;
    }
    return false;
  }
};

} // namespace Choreo

#endif // __CHOREO_MAP_HOIST_HPP__
