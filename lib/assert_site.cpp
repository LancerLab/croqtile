#include "assert_site.hpp"
#include "context.hpp"
#include "symvals.hpp"

namespace Choreo {

namespace {

const char* EmitPositionStr(AssertionEmitPosition pos) {
  switch (pos) {
  case AssertionEmitPosition::BEFORE_NODE: return "before";
  case AssertionEmitPosition::AFTER_NODE: return "after";
  case AssertionEmitPosition::IN_BLOCK: return "in-block";
  }
  return "unknown";
}

bool IsEnabledAtThreshold(AssertionCost cost, AssertionCost threshold) {
  return static_cast<int>(cost) <= static_cast<int>(threshold);
}

} // namespace

void AssertSite::ResetFunction() {
  def_map.clear();
  scope_map.clear();
  barrier_map.clear();
  node_order.clear();
  scope_level.clear();
  block_cost.clear();
  scopes.clear();
  walk_order = 0;
  body_order = 0;
}

void AssertSite::RecordBarrier(const std::string& sym, AST::Node* n) {
  assert(PrefixedWith(sym, "::") && "requires scoped name.");
  if (!barrier_map.count(sym))
    barrier_map.emplace(sym, std::set<AST::Node*>{n});
  else
    barrier_map[sym].insert(n);
  VST_DEBUG(dbgs() << "[assert-site] barrier: " << sym << " at ";
            if (auto pred = dyn_cast<AST::PredBlock>(n);
                pred && pred->HasPredicate()) dbgs()
            << PSTR(pred->GetPredicate());
            else dbgs() << n->LOC(); dbgs() << "\n");
}

bool AssertSite::RecordBarrierFromPred(const ptr<sbe::SymbolicExpression>& pred,
                                       AST::Node* node) {
  // Conservative: constrained symbols are barriers
  for (auto& csym : GetSymbols(pred)) {
    auto sym_name = VISym(csym);
    if (sym_name) RecordBarrier(*sym_name, node);
  }
  return true;
}

void AssertSite::RecordDef(const std::string& sym, AST::Node* n) {
  assert(PrefixedWith(sym, "::") && "requires scoped name.");
  def_map[sym] = n;
  VST_DEBUG(dbgs() << "[assert-site] def: " << sym << " at " << n->LOC()
                   << "\n");
}

AST::Node* AssertSite::FirstStmtOf(AST::Block* blk) const {
  assert(blk->GetBody() && !blk->GetBody()->None());
  return blk->GetBody()->SubAt(0).get();
}

size_t AssertSite::EstimateLoopTripCount(AST::Node* n) const {
  if (auto fb = dyn_cast<AST::ForeachBlock>(n)) {
    size_t trip = 1;
    for (auto range_node : fb->GetRanges()) {
      auto range = cast<AST::LoopRange>(range_node);
      auto sty = GetSymbolType(range->IVName());
      int64_t span = 100; // default trip count
      if (IsActualBoundedIntegerType(sty)) {
        auto ub = GetSingleUpperBound(sty);
        auto lb = sbe::nu(0);
        auto ub_addend_expr = dyn_cast<AST::Expr>(range->ubound);
        auto lb_addend_expr = dyn_cast<AST::Expr>(range->lbound);
        auto ub_addend = sbe::nu(0);
        auto lb_addend = sbe::nu(0);
        if (ub_addend_expr && ub_addend_expr->Opts().HasVal())
          ub_addend = ub_addend_expr->Opts().GetVal();
        if (lb_addend_expr && lb_addend_expr->Opts().HasVal())
          lb_addend = lb_addend_expr->Opts().GetVal();
        if (auto cval = VIInt(ub + ub_addend - (lb + lb_addend))) span = *cval;
      }
      auto step = range->step == GetInvalidStep() ? 1 : std::abs(range->step);
      trip *=
          static_cast<size_t>(std::max<int64_t>(1, (span + step - 1) / step));
    }
    return trip;
  }

  if (auto pb = dyn_cast<AST::ParallelBy>(n)) {
    switch (pb->GetLevel()) {
    case ParallelLevel::BLOCK: return 4;
    default: return 1;
    }
  }
  if (isa<AST::WhileBlock>(n)) return 100;
  return 1;
}

uint64_t AssertSite::EstimateAssertionCost(AST::Node* site) const {
  uint64_t cost = 1;
  if (scope_map.count(site)) {
    auto cur_node = scope_map.at(site);
    while (cur_node) {
      assert(cur_node->IsBlock());
      cost *= block_cost.at(cur_node);
      if (!scope_map.count(cur_node)) break;
      cur_node = scope_map.at(cur_node);
    }
  }
  return cost;
}

AssertionCost AssertSite::CategorizeCost(uint64_t cost) const {
  if (cost >= 500)
    return AssertionCost::HIGH;
  else if (cost >= 50)
    return AssertionCost::MEDIUM;
  else if (cost == 1)
    return AssertionCost::ENTRY;
  else
    return AssertionCost::LOW;
}

bool AssertSite::BeforeVisitImpl(AST::Node& n) {
  if (isa<AST::ChoreoFunction>(&n)) { ResetFunction(); }

  scope_level[&n] = scopes.size();
  node_order[&n] = walk_order++;
  if (auto pb = dyn_cast<AST::ParallelBy>(&n)) {
    node_order[pb->BPV().get()] = walk_order++;
    for (auto spv : pb->AllSubPVs()) node_order[spv.get()] = walk_order++;
  } else if (isa<AST::Parameter>(&n)) {
    allow_named_dim = true;
  } else if (auto dma = dyn_cast<AST::DMA>(&n)) {
    node_order[dma->GetFrom().get()] = walk_order++;
    node_order[dma->GetTo().get()] = walk_order++;
  }

  if (n.IsBlock()) {
    // Record the enclosing block for this block node too, so that
    // EstimateAssertionCost can traverse the full scope chain through
    // nested blocks (e.g. if-else inside parallel-by).
    if (!scopes.empty()) scope_map[&n] = scopes.back();
    scopes.push_back(&n);
    block_cost[&n] = EstimateLoopTripCount(&n);
  } else if (!scopes.empty())
    scope_map[&n] = scopes.back();

  return true;
}

bool AssertSite::AfterVisitImpl(AST::Node& n) {
  if (n.IsBlock()) scopes.pop_back();

  if (isa<AST::FunctionDecl>(&n)) {
    body_order = walk_order + 1;
  } else if (auto fn = dyn_cast<AST::ChoreoFunction>(&n)) {
    HoistAssertions(fn);
    EstimateAssertions();
  } else if (isa<AST::Parameter>(&n))
    allow_named_dim = false;
  return true;
}

bool AssertSite::Visit(AST::DataType&) {
  allow_named_dim = false; // no duplicated symbol is allowed except for mdspan
  return true;
}

bool AssertSite::Visit(AST::Parameter& n) {
  // Function parameters are treated as already defined at the function entry.
  // Use the ChoreoFunction node as the defining node.
  if (!n.HasSymbol()) return true;
  // The defining "node" for a parameter is the ChoreoFunction itself; however,
  // we do not have direct access to it here.  Instead, record the parameter
  // node with walk_order 0 -- it is guaranteed to precede every statement.
  RecordDef(InScopeName(n.sym->name), &n);
  return true;
}

bool AssertSite::Visit(AST::NamedVariableDecl& n) {
  RecordDef(InScopeName(n.GetName()), &n);
  return true;
}

bool AssertSite::Visit(AST::Assignment& n) {
  if (!n.AssignToDataElement()) RecordDef(InScopeName(n.GetName()), &n);
  return true;
}

bool AssertSite::Visit(AST::ParallelBy& n) {
  auto def_site = &n;
  RecordDef(InScopeName(n.BPV()->name), def_site);
  for (auto sym : n.AllSubPVs()) {
    auto id = cast<AST::Identifier>(sym);
    RecordDef(InScopeName(id->name), def_site);
  }
  return true;
}

bool AssertSite::Visit(AST::WithBlock& n) {
  auto def_site = &n;

  if (n.withins) {
    for (auto wi : n.withins->AllSubs()) {
      auto within = cast<AST::WithIn>(wi);

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

bool AssertSite::Visit(AST::LoopRange& n) {
  if (n.BoundIsMutated() && scope_map.count(&n)) {
    // Record a barrier only if the IV is findable in the symbol table.
    // Synthetic iteration variables (e.g. `foreach i in [K]` where `i`
    // is a fresh name) are not registered and InScopeName would abort.
    auto iv = n.IVName();
    std::string scoped_iv;
    std::string scope_name = scoped_symtab.ScopeName();
    while (true) {
      std::string candidate = scope_name + iv;
      if (SymTab() && SymTab()->Exists(candidate)) {
        scoped_iv = candidate;
        break;
      }
      size_t last = scope_name.rfind("::");
      if (last == std::string::npos) break;
      size_t prev = scope_name.rfind("::", last - 1);
      if (prev == std::string::npos) break;
      scope_name = scope_name.substr(0, prev + 2);
    }
    if (!scoped_iv.empty()) RecordBarrier(scoped_iv, scope_map[&n]);
  }
  return true;
}

bool AssertSite::Visit(AST::IfElseBlock& n) {
  if (IsValidValueItem(n.GetIfScopePredicate()))
    return RecordBarrierFromPred(n.GetIfScopePredicate(), &n);
  return true;
}

bool AssertSite::Visit(AST::WhileBlock& n) {
  if (IsValidValueItem(n.GetScopePredicate()))
    return RecordBarrierFromPred(n.GetScopePredicate(), &n);
  return true;
}

bool AssertSite::Visit(AST::InThreadsBlock& n) {
  if (IsValidValueItem(n.GetScopePredicate()))
    return RecordBarrierFromPred(n.GetScopePredicate(), &n);
  return true;
}

bool AssertSite::Visit(AST::Identifier& n) {
  if (allow_named_dim) { RecordDef(InScopeName(n.name), scopes.back()); }
  return true;
}

void AssertSite::HoistAssertions(AST::ChoreoFunction* fnode) {
  assert(fnode);

  auto& assessor = FCtx(fname).GetAssessor();
  // We operate by mutating the assertion vector in-place.  The assessor
  // exposes a const ref, so we const_cast here -- the pass owns the mutation
  // semantics and runs in a single-threaded pipeline.
  auto& assertions =
      const_cast<std::vector<Assertion>&>(assessor.GetAssertions());

  for (auto& ar : assertions) {
    assert(ar.node != nullptr);
    assert(node_order.count(ar.node));
    auto n_order = node_order[ar.node];

    // Collect all symbols referenced by the assertion expression.
    auto syms = GetSymbols(ar.expr);

    struct NodeSite {
      AST::Node* site = nullptr;
      size_t order = 0;
      AssertionEmitPosition ae_pos = AssertionEmitPosition::BEFORE_NODE;
      NodeSite(AST::Node* s, size_t o,
               AssertionEmitPosition aep = AssertionEmitPosition::BEFORE_NODE)
          : site(s), order(o), ae_pos(aep) {}
      bool operator<(const NodeSite& ns) const { return order < ns.order; }
      bool operator>(const NodeSite& ns) const { return order > ns.order; }
    };

    // By default, no hoisting
    NodeSite hoisted{nullptr, 0};

    // find the latest define sites for all referenced symbol
    for (const auto& vi : syms) {
      auto sym_name = VISym(vi);
      if (!sym_name) continue;

      // inspect the def-site
      assert(def_map.count(*sym_name) && "symbol is not defined.");
      auto earliest_site = def_map[*sym_name];
      assert(node_order.count(earliest_site) && "node is not ordered");
      auto earliest_order = node_order[earliest_site];

      // some mutables can be altered after this check:
      //   mutable int a = ...;
      //   while (a > 0) {
      //     ... = buf.at(a);
      //     a = ...;
      //   }
      // conservative choice: do not hoist
      if (earliest_order > n_order) {
        hoisted.site = ar.node;
        hoisted.order = n_order;
        break;
      }
      NodeSite earliest(earliest_site, earliest_order,
                        AssertionEmitPosition::AFTER_NODE);

      // inspect the barriers between def and current
      if (auto barrier_sites = barrier_map.find(*sym_name);
          barrier_sites != barrier_map.end()) {
        for (auto barrier_site : barrier_sites->second) {
          assert(node_order.count(barrier_site) && "node is not ordered");
          auto barrier_order = node_order[barrier_site];
          // ignore any later barriers
          if (barrier_order >= n_order) continue;

          NodeSite barrier{barrier_site, barrier_order,
                           AssertionEmitPosition::AFTER_NODE};

          if (barrier_site->IsBlock() &&
              (scope_level[ar.node] > scope_level[barrier_site]))
            barrier.ae_pos = AssertionEmitPosition::IN_BLOCK;

          if (barrier > earliest) earliest = barrier;
        }
      }

      if (earliest > hoisted) hoisted = earliest;
    }

    // Bubble up: when the hoisted site is nested more deeply than ar.node
    // (e.g. a mutated variable inside a foreach body while the assertion is
    // after the loop), pull the hoist point out to the enclosing block.
    if (hoisted.site && hoisted.ae_pos != AssertionEmitPosition::IN_BLOCK) {
      auto* target = hoisted.site;
      while (scope_level.count(target) && scope_level.count(ar.node) &&
             scope_level[target] > scope_level[ar.node]) {
        auto it = scope_map.find(target);
        if (it == scope_map.end()) break;
        target = it->second;
      }
      if (target && target != hoisted.site && scope_level.count(target) &&
          scope_level[target] >= scope_level[ar.node]) {
        hoisted.site = target;
        hoisted.ae_pos = AssertionEmitPosition::AFTER_NODE;
        hoisted.order =
            node_order.count(target) ? node_order[target] : hoisted.order;
      }
    }

    // ENTRY check: if every referenced symbol is a function parameter,
    // elevate to ENTRY so the assertion is emitted as a host runtime_check.
    bool is_entry =
        (hoisted.site == nullptr || isa<AST::Parameter>(hoisted.site));

    if (hoisted.site == nullptr) {
      hoisted.site = ar.node;
      hoisted.order = n_order;
    }

    // --disable-assert-hoist: force non-entry assertions to remain at use site
    if (!is_entry && CCtx().DisableAssertHoist()) {
      hoisted.site = ar.node;
      hoisted.order = n_order;
    }

    bool is_hoisted = false;
    if (is_entry) {
      // hoisted as parameter reference
      ar.type = AssessType::ENTRY;
      ar.emit_node = fnode;
      ar.emit_position = AssertionEmitPosition::AFTER_NODE;
      is_hoisted = n_order >= body_order;
    } else if (hoisted.site == ar.node) {
      ar.type = AssessType::USE_SITE;
      ar.emit_node = ar.node;
      ar.emit_position = AssertionEmitPosition::BEFORE_NODE;
      is_hoisted = false;
    } else {
      // hoisted to non-entry
      ar.type = AssessType::HOIST_SITE;
      ar.node = hoisted.site;
      if (hoisted.ae_pos == AssertionEmitPosition::IN_BLOCK) {
        if (auto ieblk = dyn_cast<AST::IfElseBlock>(hoisted.site)) {
          ar.emit_node = FirstStmtOf(ieblk);
          if (ieblk->HasElse()) {
            auto else_stmts = ieblk->GetElseBody()->AllSubs();
            auto efirst = else_stmts[0].get();
            auto else_order = node_order[efirst];
            if (n_order >= else_order) ar.emit_node = efirst;
          }
          ar.emit_position = AssertionEmitPosition::BEFORE_NODE;
        } else {
          auto* hblk = cast<AST::Block>(hoisted.site);
          ar.emit_node = FirstStmtOf(hblk);
          ar.emit_position = AssertionEmitPosition::BEFORE_NODE;
        }
      } else {
        ar.emit_node = hoisted.site;
        ar.emit_position = hoisted.ae_pos;
      }
      is_hoisted = true;
    }

    VST_DEBUG(dbgs() << "[assert-site] " << ((is_hoisted) ? "(hoisted) " : "")
                     << "assertion \"" << ar.message << "\" placed at "
                     << ar.emit_node->LOC() << " ("
                     << EmitPositionStr(ar.emit_position) << ")\n");
  }
}

void AssertSite::EstimateAssertions() {
  auto& assessor = FCtx(fname).GetAssessor();
  auto& assertions =
      const_cast<std::vector<Assertion>&>(assessor.GetAssertions());

  for (auto& ar : assertions) {
    ar.estimated_cost = EstimateAssertionCost(ar.emit_node);
    ar.cost = CategorizeCost(ar.estimated_cost);
    ar.enabled =
        IsEnabledAtThreshold(ar.cost, CCtx().RuntimeCheckCostThreshold());
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
          case AssertionCost::ENTRY: ++stats.runtime_entry; break;
          case AssertionCost::LOW: ++stats.runtime_low; break;
          case AssertionCost::MEDIUM: ++stats.runtime_medium; break;
          case AssertionCost::HIGH: ++stats.runtime_high; break;
          case AssertionCost::NONE:
          default: choreo_unreachable("unsupported assertion cost."); break;
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
    case AssertionCost::ENTRY: return "entry   ";
    case AssertionCost::LOW: return "low   ";
    case AssertionCost::MEDIUM: return "medium";
    case AssertionCost::HIGH: return "high  ";
    case AssertionCost::NONE:
    default: choreo_unreachable("unsupported assertion cost."); break;
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
        errs() << "  [" << idx++ << "] " << type_str(ar->type)
               << "  enabled=" << (ar->enabled ? "yes" : "no ")
               << "  cost=" << cost_str(ar->cost)
               << "  estimated=" << ar->estimated_cost << "\n";
        errs() << "       assess  : runtime " << STR(ar->expr) << "\n";
        errs() << "       message : " << ae.message << "\n";
        errs() << "       loc     : " << ae.loc << "\n";

        auto* site = ar->EmitTarget();
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
