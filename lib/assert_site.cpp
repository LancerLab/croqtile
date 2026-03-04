#include "assert_site.hpp"
#include "context.hpp"
#include "symvals.hpp"

namespace Choreo {

void AssertSite::ResetFunction() {
  def_map.clear();
  node_order.clear();
  walk_order = 0;
}

void AssertSite::RecordDef(const std::string& scoped_name, AST::Node* n) {
  def_map[scoped_name] = n;
  VST_DEBUG(dbgs() << "[assertsite] def: " << scoped_name << " at " << n->LOC()
                   << "\n");
}

bool AssertSite::BeforeVisitImpl(AST::Node& n) {
  node_order[&n] = walk_order++;

  if (isa<AST::ChoreoFunction>(&n)) { ResetFunction(); }

  return true;
}

bool AssertSite::AfterVisitImpl(AST::Node& n) {
  if (isa<AST::ChoreoFunction>(&n)) { ResolveDefSiteAssertions(); }
  return true;
}

bool AssertSite::Visit(AST::Parameter& n) {
  // Function parameters are treated as already defined at the function entry.
  // Use the ChoreoFunction node as the defining node.
  if (!n.HasSymbol()) return true;
  auto scoped = InScopeName(n.sym->name);
  // The defining "node" for a parameter is the ChoreoFunction itself; however,
  // we do not have direct access to it here.  Instead, record the parameter
  // node with walk_order 0 — it is guaranteed to precede every statement.
  RecordDef(scoped, &n);
  return true;
}

bool AssertSite::Visit(AST::NamedVariableDecl& n) {
  auto scoped = InScopeName(n.name_str);
  RecordDef(scoped, &n);
  return true;
}

bool AssertSite::Visit(AST::Assignment& n) {
  if (n.IsDecl()) {
    // `let x = ...` is a declaration assignment — record definition.
    auto scoped = InScopeName(n.GetName());
    RecordDef(scoped, &n);
  }
  return true;
}

void AssertSite::ResolveDefSiteAssertions() {
  if (fname.empty()) return;

  auto& assessor = FCtx(fname).GetAssessor();
  // We operate by mutating the assertion vector in-place.  The assessor
  // exposes a const ref, so we const_cast here — the pass owns the mutation
  // semantics and runs in a single-threaded pipeline.
  auto& assertions =
      const_cast<std::vector<Assertion>&>(assessor.GetAssertions());

  for (auto& ar : assertions) {
    if (ar.type != AssessType::DEF_SITE) continue;

    // Collect all symbols referenced by the assertion expression.
    auto syms = GetSymbols(ar.expr);
    if (syms.empty()) {
      // No symbolic references — cannot determine a def site.
      // Promote to ENTRY so the check still runs at function entry.
      ar.type = AssessType::ENTRY;
      continue;
    }

    // Find the latest (in walk-order) defining node among all referenced
    // symbols.  This guarantees every referenced variable is defined when
    // the assertion is emitted.
    AST::Node* latest_def = nullptr;
    size_t latest_order = 0;
    bool all_resolved = true;

    for (const auto& vi : syms) {
      auto sym_name = VISym(vi);
      if (!sym_name) continue;

      auto it = def_map.find(*sym_name);
      if (it == def_map.end()) {
        // Symbol not found in the def map — it might be a global symbol
        // (defined outside this function).  Fall back to ENTRY.
        all_resolved = false;
        VST_DEBUG(dbgs() << "[assertsite] symbol not found in def_map: "
                         << *sym_name << ", promoting to ENTRY.\n");
        break;
      }

      auto order_it = node_order.find(it->second);
      size_t ord = (order_it != node_order.end()) ? order_it->second : 0;
      if (latest_def == nullptr || ord > latest_order) {
        latest_def = it->second;
        latest_order = ord;
      }
    }

    if (!all_resolved || latest_def == nullptr) {
      ar.type = AssessType::ENTRY;
      continue;
    }

    // If every referenced symbol is defined by a Parameter node, the
    // assertion should be emitted at function entry — same as ENTRY.
    // Keeping it as DEF_SITE would cause it to be emitted during the
    // Parameter visit, which happens *before* the host function declaration
    // is written to the output stream.
    if (isa<AST::Parameter>(latest_def)) {
      ar.type = AssessType::ENTRY;
      VST_DEBUG(dbgs() << "[assertsite] DEF_SITE assertion \"" << ar.message
                       << "\" promoted to ENTRY (parameter-only def).\n");
      continue;
    }

    ar.node = latest_def;
    ar.emit_node = nullptr; // clear so EmitTarget() uses the new node
    VST_DEBUG(dbgs() << "[assertsite] DEF_SITE assertion \"" << ar.message
                     << "\" placed at " << latest_def->LOC() << "\n");
  }
}

} // end namespace Choreo
