#include "assert_site.hpp"
#include "context.hpp"
#include "symvals.hpp"

namespace Choreo {

namespace {

const char* EmitPositionStr(AssertionEmitPosition pos) {
  switch (pos) {
  case AssertionEmitPosition::BEFORE_NODE: return "before";
  case AssertionEmitPosition::AFTER_NODE: return "after";
  }
  return "unknown";
}

bool IsEnabledAtThreshold(AssertionCost cost, AssertionCost threshold) {
  return static_cast<int>(cost) <= static_cast<int>(threshold);
}

} // namespace

void AssertSite::ResetFunction() {
  def_map.clear();
  parent_map.clear();
  node_order.clear();
  walk_order = 0;
}

void AssertSite::RecordDef(const std::string& scoped_name, AST::Node* n) {
  def_map[scoped_name] = n;
  VST_DEBUG(dbgs() << "[assertsite] def: " << scoped_name << " at " << n->LOC()
                   << "\n");
}

AST::Node* AssertSite::FirstStmtOf(const ptr<AST::MultiNodes>& stmts) const {
  if (!stmts || stmts->None()) return nullptr;
  return stmts->SubAt(0).get();
}

AssertionEmitPosition AssertSite::SiteEmitPosition(AST::Node* n) const {
  if (!n) return AssertionEmitPosition::AFTER_NODE;
  if (isa<AST::ParallelBy>(n) || isa<AST::WithBlock>(n) ||
      isa<AST::ForeachBlock>(n) || isa<AST::InThreadsBlock>(n) ||
      isa<AST::IfElseBlock>(n) || isa<AST::WhileBlock>(n))
    return AssertionEmitPosition::BEFORE_NODE;
  return AssertionEmitPosition::AFTER_NODE;
}

AST::Node* AssertSite::NextStatementInBlock(AST::Node* n) const {
  if (!n) return nullptr;

  auto pit = parent_map.find(n);
  auto parent = (pit == parent_map.end()) ? nullptr : pit->second;
  auto block = dyn_cast<AST::Block>(parent);
  if (!block || !block->stmts) return nullptr;

  auto idx = block->stmts->GetIndex(n);
  if (idx < 0) return nullptr;
  if (static_cast<size_t>(idx + 1) >= block->stmts->Count()) return nullptr;
  return block->stmts->SubAt(static_cast<size_t>(idx + 1)).get();
}

AST::MultiNodes* AssertSite::FindStatementContainer(AST::Node* use) const {
  AST::Node* cur = use;
  while (cur) {
    auto pit = parent_map.find(cur);
    if (pit == parent_map.end()) break;
    if (auto mn = dyn_cast<AST::MultiNodes>(pit->second)) return mn;
    cur = pit->second;
  }
  return nullptr;
}

AST::Node* AssertSite::BubbleToSiblingScope(AST::Node* def,
                                            AST::MultiNodes* container) const {
  if (!def || !container) return def;

  auto def_it = node_order.find(def);
  if (def_it == node_order.end()) return def;
  size_t def_ord = def_it->second;

  // Find the direct child of `container` whose subtree contains `def`.
  // A child's subtree spans node_order values from the child's own order up to
  // (but not including) the next sibling's order.  We pick the last child
  // whose order is <= def_ord.
  AST::Node* enclosing_child = nullptr;
  size_t enclosing_ord = 0;
  for (size_t i = 0; i < container->Count(); ++i) {
    auto child = container->SubAt(i).get();
    auto child_it = node_order.find(child);
    if (child_it == node_order.end()) continue;
    size_t child_ord = child_it->second;
    if (child_ord <= def_ord) {
      if (enclosing_child == nullptr || child_ord > enclosing_ord) {
        enclosing_child = child;
        enclosing_ord = child_ord;
      }
    }
  }
  return enclosing_child ? enclosing_child : def;
}

AST::Node* AssertSite::LaterNode(AST::Node* lhs, AST::Node* rhs) const {
  if (!lhs) return rhs;
  if (!rhs) return lhs;

  auto lit = node_order.find(lhs);
  auto rit = node_order.find(rhs);
  auto lo = (lit == node_order.end()) ? 0 : lit->second;
  auto ro = (rit == node_order.end()) ? 0 : rit->second;
  return (lo >= ro) ? lhs : rhs;
}

size_t AssertSite::EstimateLoopTripCount(AST::Node* n) const {
  if (auto fb = dyn_cast<AST::ForeachBlock>(n)) {
    size_t trip = 1;
    for (auto range_node : fb->GetRanges()) {
      auto range = cast<AST::LoopRange>(range_node);
      if (!range->ubound || !range->lbound) {
        trip *= 100;
        continue;
      }
      auto ub_expr = dyn_cast<AST::Expr>(range->ubound);
      auto lb_expr = dyn_cast<AST::Expr>(range->lbound);
      if (!ub_expr || !lb_expr || !ub_expr->Opts().HasVal() ||
          !lb_expr->Opts().HasVal()) {
        trip *= 100;
        continue;
      }
      auto ub = VIInt(ub_expr->Opts().GetVal());
      auto lb = VIInt(lb_expr->Opts().GetVal());
      if (!ub || !lb) {
        trip *= 100;
        continue;
      }
      auto step = range->step == GetInvalidStep() ? 1 : std::abs(range->step);
      auto span = std::max<int64_t>(0, ub.value() - lb.value());
      trip *=
          static_cast<size_t>(std::max<int64_t>(1, (span + step - 1) / step));
    }
    return trip;
  }

  if (isa<AST::ParallelBy>(n)) return 100;
  if (isa<AST::WhileBlock>(n)) return 100;
  return 1;
}

uint64_t AssertSite::EstimateAssertionCost(AST::Node* site) const {
  uint64_t cost = 1;
  for (auto cur = site; cur;) {
    if (isa<AST::ForeachBlock>(cur) || isa<AST::ParallelBy>(cur) ||
        isa<AST::WhileBlock>(cur))
      cost *= EstimateLoopTripCount(cur);
    auto pit = parent_map.find(cur);
    cur = (pit == parent_map.end()) ? nullptr : pit->second;
  }
  return cost;
}

AssertionCost AssertSite::CategorizeCost(uint64_t cost) const {
  if (cost >= 10000) return AssertionCost::HIGH;
  if (cost >= 100) return AssertionCost::MEDIUM;
  return AssertionCost::LOW;
}

bool AssertSite::BeforeVisitImpl(AST::Node& n) {
  parent_map[&n] = parent_stack.empty() ? nullptr : parent_stack.back();
  parent_stack.push_back(&n);
  node_order[&n] = walk_order++;

  if (isa<AST::ChoreoFunction>(&n)) { ResetFunction(); }

  return true;
}

bool AssertSite::AfterVisitImpl(AST::Node& n) {
  if (isa<AST::ChoreoFunction>(&n)) { HoistAssertions(); }
  if (!parent_stack.empty()) parent_stack.pop_back();
  return true;
}

bool AssertSite::Visit(AST::Parameter& n) {
  // Function parameters are treated as already defined at the function entry.
  // Use the ChoreoFunction node as the defining node.
  if (!n.HasSymbol()) return true;
  auto scoped = InScopeName(n.sym->name);
  // The defining "node" for a parameter is the ChoreoFunction itself; however,
  // we do not have direct access to it here.  Instead, record the parameter
  // node with walk_order 0 -- it is guaranteed to precede every statement.
  RecordDef(scoped, &n);
  return true;
}

bool AssertSite::Visit(AST::NamedVariableDecl& n) {
  auto scoped = InScopeName(n.name_str);
  RecordDef(scoped, &n);
  return true;
}

bool AssertSite::Visit(AST::Assignment& n) {
  // Record both declarations and mutations as hoist barriers (cases 1 & 2).
  //
  // Case 1 -- declaration:  `mutable int j = 0`  introduces the variable.
  // Case 2 -- mutation:     `j = j + 1`  changes the value after definition;
  //   any reference to j after this point may observe the mutated value, so
  //   the assertion cannot be hoisted above this assignment either.
  //
  // `AssignToDataElement()` is true for subscript writes (arr[i] = ...); those
  // do not change the scalar index variable itself and are therefore skipped.
  if (!n.AssignToDataElement()) {
    auto scoped = InScopeName(n.GetName());
    RecordDef(scoped, &n);
  }
  return true;
}

bool AssertSite::Visit(AST::ParallelBy& n) {
  auto def_site = FirstStmtOf(n.stmts);
  if (!def_site) def_site = &n;
  RecordDef(InScopeName(n.BPV()->name), def_site);
  for (auto sym : n.AllSubPVs()) {
    auto id = cast<AST::Identifier>(sym);
    RecordDef(InScopeName(id->name), def_site);
  }
  return true;
}

bool AssertSite::Visit(AST::WithBlock& n) {
  auto def_site = FirstStmtOf(n.stmts);
  if (!def_site) def_site = &n;

  if (n.withins) {
    for (auto wi : n.withins->AllSubs()) {
      auto within = dyn_cast<AST::WithIn>(wi);
      if (!within) continue;

      if (within->with) RecordDef(InScopeName(within->with->name), def_site);

      if (within->with_matchers) {
        for (auto v : within->GetMatchers()) {
          auto id = cast<AST::Identifier>(v);
          RecordDef(InScopeName(id->name), def_site);
        }
      }
    }
  }

  return true;
}

void AssertSite::HoistAssertions() {
  if (fname.empty()) return;

  auto& assessor = FCtx(fname).GetAssessor();
  // We operate by mutating the assertion vector in-place.  The assessor
  // exposes a const ref, so we const_cast here -- the pass owns the mutation
  // semantics and runs in a single-threaded pipeline.
  auto& assertions =
      const_cast<std::vector<Assertion>&>(assessor.GetAssertions());

  for (auto& ar : assertions) {
    if (ar.type == AssessType::ENTRY) {
      ar.estimated_cost = 1;
      ar.cost = AssertionCost::LOW;
      ar.enabled = true;
      continue;
    }

    // Collect all symbols referenced by the assertion expression.
    auto syms = GetSymbols(ar.expr);
    if (syms.empty()) {
      // No symbolic references -- safe to place at entry.
      ar.type = AssessType::ENTRY;
      ar.node = nullptr;
      ar.emit_node = nullptr;
      ar.emit_position = AssertionEmitPosition::AFTER_NODE;
      ar.estimated_cost = 1;
      ar.cost = AssertionCost::LOW;
      ar.enabled = true;
      continue;
    }

    // Find the latest defining node among all referenced symbols. This keeps
    // the assertion after every required definition.
    AST::Node* latest_def = nullptr;
    size_t latest_order = 0;
    bool all_resolved = true;

    for (const auto& vi : syms) {
      auto sym_name = VISym(vi);
      if (!sym_name) continue;

      auto it = def_map.find(*sym_name);
      if (it == def_map.end()) {
        // Symbol not found in the def map -- it may be global or otherwise not
        // hoistable inside this function.
        all_resolved = false;
        VST_DEBUG(dbgs() << "[assertsite] symbol not found in def_map: "
                         << *sym_name << ", keeping use-site.\n");
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
      ar.type = AssessType::USE_SITE;
      if (!ar.node) ar.node = ar.emit_node;
      ar.emit_position = SiteEmitPosition(ar.EmitTarget());
      ar.estimated_cost = EstimateAssertionCost(ar.EmitTarget());
      ar.cost = CategorizeCost(ar.estimated_cost);
      ar.enabled =
          IsEnabledAtThreshold(ar.cost, CCtx().RuntimeCheckCostThreshold());
      continue;
    }

    // Case 3.iii -- block-level barrier: if `latest_def` is nested inside a
    // block statement (while, foreach, if, ...) that lives in the same scope as
    // the access, bubble `latest_def` up to that block statement so the
    // assertion is placed AFTER the block exits, not deep inside it.
    auto use_container = FindStatementContainer(ar.EmitTarget());
    auto bubbled_def = BubbleToSiblingScope(latest_def, use_container);
    bool was_bubbled = (bubbled_def != latest_def);

    auto hoist_site = bubbled_def;

    // If the def-site is a PredBlock (foreach, while, if, ...) that contains
    // the original emit node, keep the assertion at the original emit node.
    // This ensures loop-body assertions (e.g. BoundedType iteration variables)
    // fire inside the loop rather than before it.
    if (!was_bubbled && isa<AST::PredBlock>(latest_def) && ar.emit_node) {
      auto em_it = node_order.find(ar.emit_node);
      if (em_it != node_order.end() && em_it->second > latest_order)
        hoist_site = ar.emit_node;
    }

    // Conservative hoisting: when the only defining node is a function
    // parameter and the assertion was not bubbled, it was created inside a
    // conditional scope (guard escalated its type from ENTRY to HOIST_SITE).
    // Hoist to the very first statement of the enclosing block -- there is no
    // local-definition barrier, so the assertion can safely be moved ahead of
    // any other statements (like `mutable int pre = ...`) in the branch body.
    if (isa<AST::Parameter>(latest_def) && !was_bubbled) {
      if (hoist_site == latest_def) {
        AST::Node* first_stmt = (use_container && use_container->Count() > 0)
                                    ? use_container->SubAt(0).get()
                                    : nullptr;
        if (first_stmt) {
          ar.type = AssessType::HOIST_SITE;
          ar.node = first_stmt;
          ar.emit_node = first_stmt;
          ar.emit_position = AssertionEmitPosition::BEFORE_NODE;
          ar.estimated_cost = EstimateAssertionCost(first_stmt);
          ar.cost = CategorizeCost(ar.estimated_cost);
          ar.enabled =
              IsEnabledAtThreshold(ar.cost, CCtx().RuntimeCheckCostThreshold());
          VST_DEBUG(dbgs() << "[assertsite] assertion \"" << ar.message
                           << "\" hoisted to block start (parameter, no "
                              "barrier)\n");
          continue;
        }
        // No enclosing block found -- fall through to normal HOIST placement.
      }
    }

    // When the def was bubbled out of a nested scope, emit AFTER the scope
    // so the assertion fires after the scope exits, not inside it.
    auto emit_pos = was_bubbled ? AssertionEmitPosition::AFTER_NODE
                                : SiteEmitPosition(hoist_site);

    ar.type = AssessType::HOIST_SITE;
    ar.node = hoist_site;
    ar.emit_node = hoist_site;
    ar.emit_position = emit_pos;
    ar.estimated_cost = EstimateAssertionCost(hoist_site);
    ar.cost = CategorizeCost(ar.estimated_cost);
    ar.enabled =
        IsEnabledAtThreshold(ar.cost, CCtx().RuntimeCheckCostThreshold());
    VST_DEBUG(dbgs() << "[assertsite] assertion \"" << ar.message
                     << "\" placed at " << hoist_site->LOC() << " ("
                     << EmitPositionStr(ar.emit_position)
                     << (was_bubbled ? ", bubbled" : "")
                     << ", cost=" << ar.estimated_cost
                     << ", enabled=" << ar.enabled << ")\n");
  }

  if (CCtx().ShowAssess()) PrintAssertionReport();

  // Always accumulate aggregate statistics (used by --stats).
  {
    auto& stats = CCtx().GetAssessmentStats();
    const auto& log = assessor.GetAssessmentLog();
    const auto& all = assessor.GetAssertions();
    for (const auto& ae : log) {
      ++stats.total;
      // Per-usage-type total
      switch (ae.usage_type) {
      case UsageType::UnClassified: ++stats.unclassified_total; break;
      case UsageType::ShapeCompatibility: ++stats.shape_compat_total; break;
      case UsageType::ElementAccess: ++stats.elem_access_total; break;
      case UsageType::LoopBound: ++stats.loop_bound_total; break;
      case UsageType::HardwareConstraint: ++stats.hw_constraint_total; break;
      }
      switch (ae.outcome) {
      case AssessOutcome::STATIC_TRUE: ++stats.static_true; break;
      case AssessOutcome::STATIC_FALSE: ++stats.static_false; break;
      case AssessOutcome::RUNTIME: {
        ++stats.runtime_total;
        // Per-usage-type runtime
        switch (ae.usage_type) {
        case UsageType::UnClassified: ++stats.unclassified_runtime; break;
        case UsageType::ShapeCompatibility: ++stats.shape_compat_runtime; break;
        case UsageType::ElementAccess: ++stats.elem_access_runtime; break;
        case UsageType::LoopBound: ++stats.loop_bound_runtime; break;
        case UsageType::HardwareConstraint:
          ++stats.hw_constraint_runtime;
          break;
        }
        if (ae.assertion_idx < all.size()) {
          const auto& ar = all[ae.assertion_idx];
          switch (ar.cost) {
          case AssertionCost::LOW: ++stats.runtime_low; break;
          case AssertionCost::MEDIUM: ++stats.runtime_medium; break;
          case AssertionCost::HIGH: ++stats.runtime_high; break;
          }
          if (ar.enabled)
            ++stats.runtime_enabled;
          else
            ++stats.runtime_disabled;
        }
        break;
      }
      }
    }
  }
}

void AssertSite::PrintAssertionReport() const {
  auto& assessor = FCtx(fname).GetAssessor();
  const auto& log = assessor.GetAssessmentLog();
  const auto& all = assessor.GetAssertions();
  if (log.empty()) return;

  // Count outcomes for the header.
  size_t n_strue = 0, n_sfalse = 0, n_runtime = 0;
  for (const auto& ae : log) {
    if (ae.outcome == AssessOutcome::STATIC_TRUE)
      ++n_strue;
    else if (ae.outcome == AssessOutcome::STATIC_FALSE)
      ++n_sfalse;
    else
      ++n_runtime;
  }

  auto type_str = [](AssessType t) -> const char* {
    switch (t) {
    case AssessType::ENTRY: return "ENTRY    ";
    case AssessType::HOIST_SITE: return "HOIST    ";
    case AssessType::USE_SITE: return "USE_SITE ";
    }
    return "?        ";
  };
  auto cost_str = [](AssertionCost c) -> const char* {
    switch (c) {
    case AssertionCost::LOW: return "low   ";
    case AssertionCost::MEDIUM: return "medium";
    case AssertionCost::HIGH: return "high  ";
    }
    return "?     ";
  };
  auto pos_str = [](AssertionEmitPosition p) -> const char* {
    return p == AssertionEmitPosition::BEFORE_NODE ? "before" : "after ";
  };

  errs() << "\n[assertions] function: " << fname << "  (" << log.size()
         << " assessed:"
         << "  " << n_strue << " static-true,"
         << "  " << n_sfalse << " static-false,"
         << "  " << n_runtime << " runtime)\n";
  errs() << "  " << std::string(75, '-') << "\n";

  size_t idx = 0;
  for (const auto& ae : log) {
    if (ae.outcome == AssessOutcome::STATIC_TRUE) {
      // Provably safe at compile time -- no code generated.
      errs()
          << "  [" << idx++ << "] static-true   "
          << "(compile-time: always passes \xe2\x80\x94 no code generated)\n";
      errs() << "       message : " << ae.message << "\n";
      errs() << "       loc     : " << ae.loc << "\n";
    } else if (ae.outcome == AssessOutcome::STATIC_FALSE) {
      // Proven unsafe -- compile error/warning already emitted.
      errs() << "  [" << idx++ << "] static-false  "
             << "(compile-time: always fails \xe2\x80\x94 compile "
                "error/warning)\n";
      errs() << "       message : " << ae.message << "\n";
      errs() << "       loc     : " << ae.loc << "\n";
    } else {
      // RUNTIME: look up the matching Assertion for hoist/cost info.
      const Assertion* ar =
          (ae.assertion_idx < all.size()) ? &all[ae.assertion_idx] : nullptr;

      if (ar) {
        auto* site = ar->EmitTarget();
        errs() << "  [" << idx++ << "] " << type_str(ar->type)
               << "  enabled=" << (ar->enabled ? "yes" : "no ")
               << "  cost=" << cost_str(ar->cost)
               << "  estimated=" << ar->estimated_cost << "\n";
        errs()
            << "       assess  : runtime (cannot evaluate at compile time)\n";
        errs() << "       message : " << ae.message << "\n";
        errs() << "       loc     : " << ae.loc << "\n";

        if (ar->type == AssessType::ENTRY) {
          errs() << "       site    : function entry (host runtime_check)\n";
        } else if (site) {
          errs() << "       site    : " << site->LOC() << "  ("
                 << pos_str(ar->emit_position) << ")"
                 << "  [" << site->TypeNameString() << "]\n";
        } else {
          errs() << "       site    : (none)\n";
        }
      } else {
        // Assertion record not yet available (should not happen after
        // hoisting).
        errs() << "  [" << idx++
               << "] runtime      (assertion record unavailable)\n";
        errs() << "       message : " << ae.message << "\n";
        errs() << "       loc     : " << ae.loc << "\n";
      }
    }
  }
  errs() << "  " << std::string(75, '-') << "\n";
}

} // end namespace Choreo
