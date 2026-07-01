// Liveness Analysis Pass
//
// Two-phase algorithm:
//
//   Phase 1 (AST traversal via Visitor):
//     - Walk the AST in visitor order (BeforeVisit -> children -> Visit ->
//       AfterVisit).
//     - For each statement node, record def/use sets and alias/binding
//       relationships.  Also build a CFG of BasicBlocks with if/else and
//       inthreads diamonds.
//     - Extra uses are inserted at loop scope boundaries when a variable
//       defined in an outer scope is used inside a loop.
//
//   Phase 2 (ComputeLiveRange):
//     - Traverse each function's CFG in reverse topological order.
//     - Compute live_in / live_out per statement using the standard equation:
//         live_in(s) = use(s) U (live_out(s) - def(s))
//       extended with transitive closure over alias and binding maps.
//     - Build per-variable live ranges and merge adjacent intervals.
//     - Verify postconditions on the merged ranges.
//
// The result (var_ranges) is consumed by the MemReuse pass.

#include "liveness_analysis.hpp"
#include "ast.hpp"
#include "aux.hpp"
#include "types.hpp"
#include "visitor.hpp"
#include <tuple>

using namespace Choreo;

#define DUMP_EACH_STMT true
#define DUMP_STMT_WITH_TYPE_INFO false
#define ONLY_SHOW_BUFFER false

// ==========================================================================
// Liveness node coverage guard.
// When adding a new AST node type that should be treated as a statement:
//   1. Add it to HasStmt()
//   2. Add a Visit() override in the header and .cpp
//   3. Add a case in DumpStmtBriefly()
//   4. Update kNumStmtTypes below
// The runtime assert in HasStmt() will fire if the count is wrong.
// ==========================================================================
static constexpr size_t kNumStmtTypes = 18;
static_assert(kNumStmtTypes == LivenessAnalyzer::NumVisitOverrides(),
              "HasStmt type count and Visit override count are out of sync. "
              "When adding a new statement node type, update both HasStmt() "
              "and add a Visit() override.");

// ---------- Set operations ----------

LivenessAnalyzer::VarSet& LivenessAnalyzer::SetUnionInPlace(VarSet& a,
                                                            const VarSet& b) {
  a.insert(b.begin(), b.end());
  return a;
}

LivenessAnalyzer::VarSet& LivenessAnalyzer::SetDiffInPlace(VarSet& a,
                                                           const VarSet& b) {
  for (const auto& item : b) a.erase(item);
  return a;
}

// ---------- Operand and type queries ----------

LivenessAnalyzer::VarSet
LivenessAnalyzer::GetAllSymbolicOperands(const AST::Node* n) const {
  if (auto id = dyn_cast<AST::Identifier>(n)) {
    if (id->name == "__choreo_no_tiling__") return {};
    return {InScopeNameForRef(id->name)};
  }
  if (auto expr = dyn_cast<AST::Expr>(n)) {
    VarSet res;
    if (auto c = expr->GetC())
      SetUnionInPlace(res, GetAllSymbolicOperands(c.get()));
    for (const auto& e : {expr->GetL(), expr->GetR()})
      if (e) SetUnionInPlace(res, GetAllSymbolicOperands(e.get()));
    return res;
  }
  if (isa<AST::IntLiteral>(n) || isa<AST::FloatLiteral>(n) ||
      isa<AST::StringLiteral>(n) || isa<AST::BoolLiteral>(n))
    return {};
  if (auto ii = dyn_cast<AST::IntIndex>(n))
    return GetAllSymbolicOperands(ii->value.get());
  if (auto mds = dyn_cast<AST::MultiDimSpans>(n)) {
    if (!mds->ref_name.empty()) return {InScopeName(mds->ref_name)};
    auto mv = dyn_cast<AST::MultiValues>(mds->list);
    if (!mv) {
      VST_DEBUG(dbgs() << "the list of mds is not multivalues: " << PSTR(n)
                       << "\n");
      return {};
    }
    VarSet res;
    for (const auto& v : mv->AllValues())
      SetUnionInPlace(res, GetAllSymbolicOperands(v.get()));
    return res;
  }
  if (auto da = dyn_cast<AST::DataAccess>(n)) return {da->GetDataName()};
  if (auto it = dyn_cast<AST::IntTuple>(n)) {
    VarSet res;
    for (const auto& v : it->GetValues()->AllValues())
      SetUnionInPlace(res, GetAllSymbolicOperands(v.get()));
    return res;
  }
  if (isa<AST::Call>(n)) return {};
  if (auto chunkat = dyn_cast<AST::ChunkAt>(n)) {
    VarSet res;
    res.insert(InScopeName(chunkat->RefSymbol()));
    for (auto tsi : chunkat->AllOperations())
      for (const auto& rfn : tsi->ReferredNodes())
        SetUnionInPlace(res, GetAllSymbolicOperands(rfn.get()));
    if (chunkat->indices)
      for (const auto& idx : chunkat->indices->AllValues())
        SetUnionInPlace(res, GetAllSymbolicOperands(idx.get()));
    return res;
  }
  if (isa<AST::Nullptr>(n)) return {};
  choreo_unreachable("node type: " + n->TypeNameString() +
                     " is not handled yet.");
  return {};
}

inline bool LivenessAnalyzer::IsRef(const AST::Node& n) {
  return n.HasNote("ref");
}

bool LivenessAnalyzer::HasStmt(const AST::Node& n) const {
  // Each type here must have a Visit() override and DumpStmtBriefly() case.
  // Count must match kNumStmtTypes (currently 18).
  bool is_stmt = isa<AST::NamedTypeDecl>(&n) ||     // 1
                 isa<AST::NamedVariableDecl>(&n) || // 2
                 isa<AST::Assignment>(&n) ||        // 3
                 isa<AST::DMA>(&n) ||               // 4
                 isa<AST::MMA>(&n) ||               // 5
                 isa<AST::Wait>(&n) ||              // 6
                 isa<AST::Call>(&n) ||              // 7
                 isa<AST::Rotate>(&n) ||            // 8
                 isa<AST::Synchronize>(&n) ||       // 9
                 isa<AST::Trigger>(&n) ||           // 10
                 isa<AST::Return>(&n) ||            // 11
                 isa<AST::ParallelBy>(&n) ||        // 12
                 isa<AST::WithBlock>(&n) ||         // 13
                 isa<AST::ForeachBlock>(&n) ||      // 14
                 isa<AST::WhileBlock>(&n) ||        // 15
                 isa<AST::InThreadsBlock>(&n) ||    // 16
                 isa<AST::IfElseBlock>(&n) ||       // 17
                 isa<AST::ChoreoFunction>(&n);      // 18
  return is_stmt;
}

// ---------- Scope and name utilities ----------

inline std::string
LivenessAnalyzer::GetScopedName(const std::string& name) const {
  assert(!name.empty() && "expecting a valid name.");
  return PrefixedWith(name, "::") ? name : InScopeNameForRef(name);
}

std::string LivenessAnalyzer::RemoveWithin(const std::string& s) {
  auto scopes = SplitStringByDelimiter(s, "::");
  std::string ret = "::";
  for (const auto& scope : scopes) {
    if (PrefixedWith(scope, "within")) continue;
    ret += scope + "::";
  }
  return ret;
}

int LivenessAnalyzer::ScopeCompare(const std::string& s1,
                                   const std::string& s2) {
  std::string s1_no_within = RemoveWithin(s1);
  std::string s2_no_within = RemoveWithin(s2);
  // ::foo::A == ::foo::A
  if (s1_no_within == s2_no_within) return 0;
  // ::foo::A < ::foo::A::B
  if (PrefixedWith(s2_no_within, s1_no_within)) return -1;
  // ::foo::A::B > ::foo::A
  return 1;
}

// target: find the first loop between outer_scope and inner_scope
// `::foo::pb::`, `::foo::pb::within0::foreach0::within1::foreach1::`
// => `::foo::pb::within0::foreach0::`
// suppose variable `x` in defined in `::foo::pb::`, and there is a use
// of `x` in `::foo::pb::within0::foreach0::within1::foreach1::`, then
// the liveness of `x` should be extended to the end of
// `::foo::pb::within0::foreach0::` explicitly.
std::string
LivenessAnalyzer::ExactFirstLoopScope(const std::string& outer_scope,
                                      const std::string& inner_scope) {
  assert(PrefixedWith(inner_scope, outer_scope) &&
         "expecting the inner scope to be prefixed with the outer scope.");
  auto scopes_outer = SplitStringByDelimiter(outer_scope, "::");
  auto scopes_inner = SplitStringByDelimiter(inner_scope, "::");

  size_t so_size = scopes_outer.size();
  size_t si_size = scopes_inner.size();
  size_t idx;
  for (idx = so_size; idx < si_size; idx++)
    if (PrefixedWith(scopes_inner[idx], "foreach_") ||
        PrefixedWith(scopes_inner[idx], "while_"))
      break;

  if (idx == si_size) return "";

  auto scopes_ret = std::vector<std::string>(scopes_inner.begin(),
                                             scopes_inner.begin() + idx + 1);
  return "::" + DelimitedString(scopes_ret, "::") + "::";
}

// ---------- Def/Use/Alias/Binding bookkeeping ----------

void LivenessAnalyzer::AddUse(const Stmt* s, const std::string& var,
                              bool add_extra_use) {
  std::string svar = InScopeNameForRef(RemoveSuffix(var, ".data"));
  VST_DEBUG(dbgs() << "USE: " << svar << "\n");
  stmt_linfo[s].use.insert(svar);
  var_events[svar].push_back({EventKind::Use, SSTab().ScopeName()});
  if (inthreads_async_level && !visiting_synchronize)
    AddAsyncInthreadsVar(SSTab().ScopeName(), svar);
  // when using a var, its binding should also be treated as used.
  // important when dealing with `AST::Rotate` or future
  if (bindings_.count(svar)) {
    auto binding_vars = TransitiveClosure({svar}, bindings_);
    for (const auto& binding_var : binding_vars) {
      if (stmt_linfo[s].use.count(binding_var)) continue;
      AddUse(s, binding_var, add_extra_use);
    }
  }
  if (future_buffers.count(svar))
    for (const auto& [src, dst] : future_buffers[svar])
      AddUse(s, dst, add_extra_use);
  if (!add_extra_use) return;
  // for bounded vars defined in paraby, we add extra uses in other place.
  if (paraby_bounded_vars.count(svar)) return;
  // add the extra uses in `scope_end`
  for (const auto& event : var_events.at(svar)) {
    // only consider the def event.
    if (event.first != EventKind::Def) continue;
    // If there is no loop in the current scope, no need to add extra use.
    if (SSTab().ScopeName().find("foreach_") == std::string::npos &&
        SSTab().ScopeName().find("while_") == std::string::npos)
      continue;
    int res = ScopeCompare(event.second, SSTab().ScopeName());
    if (res < 0) {
      /*
      {
        def x                              (event.second)
        loop {                             (exact_scope)
          {
            use x                          (SSTab().ScopeName())
            ...
          }
        } // add extra use in '}' for x    (end of exact_scope)
      }
      */
      std::string exact_scope =
          ExactFirstLoopScope(event.second, SSTab().ScopeName());
      if (exact_scope.empty()) continue;
      if (!events_to_add[scope2stmt.at(exact_scope)].count(
              {EventKind::Use, svar})) {
        VST_DEBUG({
          dbgs() << "RECORD extra use " << svar << "\n\tin the end of "
                 << stmt2id[scope2stmt.at(exact_scope)] << "\n";
        });
        events_to_add[scope2stmt.at(exact_scope)].insert(
            {EventKind::Use, svar});
      }
    } else if (res == 0) {
      /*
      {
        def x                             (event.second)
        use x;                            (SSTab().ScopeName())
      }
      no need to add extra use
      */
      break;
    } else {
      // TODO: the case is invalid?
      assert(false);
      /*
      {
        def x
        loop {
          def x;                          (event.second)
          {
            use x;
            ...
          }
        }
        use x;                            (SSTab().ScopeName())
      }
      then res is 1, which means that
      the current scope !>= the scope of the definition point
      just ignore it.
      */
      continue;
    }
  }
}

void LivenessAnalyzer::AddUse(const Stmt* s, const VarSet& vars,
                              bool add_extra_use) {
  for (const auto& var : vars) AddUse(s, var, add_extra_use);
}

void LivenessAnalyzer::AddDef(const Stmt* s, const std::string& var,
                              bool is_buffer_or_future) {
  std::string svar = PrefixedWith(var, "::") ? var : SSTab().ScopeName() + var;
  VST_DEBUG(dbgs() << "DEF: " << svar << "\n");
  stmt_linfo[s].def.insert(svar);
  var_events[svar].push_back({EventKind::Def, SSTab().ScopeName()});
  if (is_buffer_or_future) {
    std::string svar_span = svar + ".span";
    VST_DEBUG(dbgs() << "\tand .span: " << svar_span << "\n");
    stmt_linfo[s].def.insert(svar_span);
    var_events[svar_span].push_back({EventKind::Def, SSTab().ScopeName()});
  }
}

void LivenessAnalyzer::AddIsAlias(const Stmt* s, const std::string& alias_var) {
  std::string salias = GetScopedName(alias_var);
  VST_DEBUG(dbgs() << "AddIsAlias: " << salias << "\n");
  assert(stmt_linfo[s].name_if_alias.empty() && "expecting no alias before.");
  stmt_linfo[s].name_if_alias = salias;
}

// y = x.spanas(...), then y is alias to x
void LivenessAnalyzer::AddAlias(const std::string& alias_var,
                                const std::string& original_var) {
  std::string salias = GetScopedName(alias_var);
  std::string soriginal = GetScopedName(original_var);
  VST_DEBUG(dbgs() << "Add alias: " << salias << " <-> " << soriginal << "\n");
  alias_[salias] = soriginal;
}

void LivenessAnalyzer::AddIsBinding(const Stmt* s,
                                    const std::string& bind_res) {
  std::string sres = GetScopedName(bind_res);
  VST_DEBUG(dbgs() << "AddIsBinding: " << sres << "\n");
  assert(stmt_linfo[s].name_if_binding.empty() &&
         "expecting no binding before.");
  stmt_linfo[s].name_if_binding = sres;
}

// `x = dma.copy y => z`
// then `y` and `z` are bound with `x`
void LivenessAnalyzer::AddBinding(const std::string& bind_res,
                                  const std::string& bind_src) {
  std::string sres = GetScopedName(bind_res);
  std::string ssrc = GetScopedName(bind_src);
  VST_DEBUG(dbgs() << "AddBinding: " << sres << " <- " << ssrc << "\n");
  bindings_[sres].insert(ssrc);
}

void LivenessAnalyzer::RemoveBinding(const std::string& bind_res,
                                     const std::string& bind_src) {
  std::string sres = GetScopedName(bind_res);
  std::string ssrc = GetScopedName(bind_src);
  VST_DEBUG(dbgs() << "RemoveBinding: " << sres << " <- " << ssrc << "\n");
  bindings_[sres].erase(ssrc);
}

void LivenessAnalyzer::AddFut2Buffers(const std::string& fut,
                                      const DMABufInfo& buf_info) {
  std::string sfut = GetScopedName(fut);
  std::string ssrc = GetScopedName(buf_info.first);
  std::string sdst = GetScopedName(buf_info.second);
  VST_DEBUG(dbgs() << "AddFut2Buffers: " << sfut << " -> " << ssrc << ", "
                   << sdst << "\n");
  future_buffers[sfut].insert({ssrc, sdst});
}

inline void
LivenessAnalyzer::AddAsyncInthreadsVar(const std::string& scope_name,
                                       const std::string& var) {
  async_inthreads_vars[scope_name].insert(GetScopedName(var));
}

namespace {

template <typename MapType>
void ProcessMap(const std::string& current, const MapType& mp,
                LivenessAnalyzer::VarSet& result,
                LivenessAnalyzer::VarSet& processed,
                std::queue<std::string>& queue) {
  if (!mp.count(current)) return;
  const auto& next = mp.at(current);

  if constexpr (std::is_same_v<std::string,
                               typename std::decay<decltype(next)>::type>) {
    if (!processed.count(next)) {
      result.insert(next);
      queue.push(next);
    }
  } else if constexpr (std::is_same_v<
                           LivenessAnalyzer::VarSet,
                           typename std::decay<decltype(next)>::type>) {
    for (const auto& next_var : next)
      if (!processed.count(next_var)) {
        result.insert(next_var);
        queue.push(next_var);
      }
  } else {
    assert(false && "expecting the value of mp is std::string or "
                    "std::unordered_set<std::string> or "
                    "std::set<std::string>.");
  }
}

} // anonymous namespace

template <typename... MapTypes>
LivenessAnalyzer::VarSet
LivenessAnalyzer::TransitiveClosure(const VarSet& vars,
                                    const MapTypes&... maps) {
  VarSet result = vars;
  VarSet processed;
  std::queue<std::string> queue;
  for (const auto& item : vars) queue.push(item);

  while (!queue.empty()) {
    std::string current = queue.front();
    queue.pop();
    if (processed.count(current)) continue;
    processed.insert(current);

    (ProcessMap(current, maps, result, processed, queue), ...);
  }
  return result;
}

namespace {

void TopoSortBB(
    std::shared_ptr<LivenessAnalyzer::BasicBlock> root,
    std::unordered_set<size_t>& visited,
    std::vector<std::shared_ptr<LivenessAnalyzer::BasicBlock>>& order) {
  if (!root || visited.count(root->id)) return;
  visited.insert(root->id);
  for (auto succ : root->succs) TopoSortBB(succ, visited, order);
  order.push_back(root);
}

} // anonymous namespace

// ---------- Dataflow computation ----------

void LivenessAnalyzer::UpdateVarRange(const std::string& var, size_t id) {
  auto& range = var_ranges[var];
  if (range.Values().empty()) {
    range.PushBack(Range{id, id});
  } else {
    auto& last = range.Values().back();
    if (id + 1 == last.start)
      last.start = id;
    else
      range.PushBack(Range{id, id});
  }
}

void LivenessAnalyzer::DumpLivenessResults(
    const std::vector<std::pair<std::string, Ranges>>& var_live_ranges,
    const std::map<std::string, std::map<size_t, BlockInfo>>& bb_infos_map) {
  VST_DEBUG({
    for (auto& [var, ranges] : var_live_ranges) {
      if (ONLY_SHOW_BUFFER && !buffers.count(var)) continue;
      if (buffers.count(var)) {
        dbgs() << "BUFFER " << var << "\n";
        dbgs() << "\trange:";
        for (const auto& range : ranges.Values()) {
          dbgs() << " [" << range.start << ", " << range.end;
          dbgs() << "]";
        }
        dbgs() << "\n";
      } else {
        dbgs() << "VAR    " << var << "\n";
        dbgs() << "\trange:";
        for (const auto& range : ranges.Values())
          dbgs() << " [" << range.start << ", " << range.end << "] ";
        dbgs() << "\n";
      }
    }
    auto PrintSet = [](std::ostream& os, const std::string& label,
                       const LivenessAnalyzer::VarSet& vars) {
      os << "\t" << label << ": " << vars.size() << "\n";
      for (const auto& item : vars) os << "\t\t" << item << "\n";
    };
    for (const Stmt* s : preorder_stmts) {
      const auto& sl = stmt_linfo[s];
      if (ONLY_SHOW_BUFFER && !sl.buffer_related) continue;
      dbgs() << "stmt: " << StmtStr(s);
      PrintSet(dbgs(), "use", sl.use);
      PrintSet(dbgs(), "def", sl.def);
      PrintSet(dbgs(), "live_in", sl.live_in);
      PrintSet(dbgs(), "live_out", sl.live_out);
      dbgs() << "\n";
    }
    PrintSet(dbgs(), "buffers", buffers);
    dbgs() << "\n";
    for (const auto& [fname, bb_infos] : bb_infos_map) {
      dbgs() << "For " << fname << ":\n";
      for (const auto& [id, bbi] : bb_infos) {
        dbgs() << "bb " << id << ":\n";
        PrintSet(dbgs(), "live_in", bbi.in);
        PrintSet(dbgs(), "live_out", bbi.out);
      }
    }
    dbgs() << "\n";
  });

  auto IsGlobalOrBuiltIn = [&](const std::string& var_name) {
    if (var_name == "::__choreo_no_tiling__") return true;
    if (var_name == "::__choreo_parent_dim__") return true;
    return false;
  };
  VST_DEBUG(if (!stmt_linfo[preorder_stmts[0]].live_in.empty()) {
    std::set<std::string> li_set;
    for (const auto& item : stmt_linfo[preorder_stmts[0]].live_in)
      if (!IsGlobalOrBuiltIn(item)) li_set.insert(item);
    if (!li_set.empty()) {
      errs() << StmtStr(preorder_stmts[0]);
      errs() << "live_in of the first stmt is not empty, including:\n";
      for (const auto& item : li_set) errs() << "\t" << item << "\n";
      choreo_unreachable("expecting the live_in of the first stmt is empty.");
    }
  });
}

void LivenessAnalyzer::ComputeLiveRange() {
  // fname -> bb_infos
  std::map<std::string, std::map<size_t, BlockInfo>> bb_infos_map;

  for (const auto& [fname, bbl] : bb_lists) {
    auto& bb_infos = bb_infos_map[fname];
    std::unordered_set<size_t> visited;
    std::vector<std::shared_ptr<BasicBlock>> order;
    TopoSortBB(bbl.front(), visited, order);
    for (auto& bb : order) {
      VST_DEBUG(dbgs() << "visiting bb " << bb->id << "\n");
      auto& bb_info = bb_infos[bb->id];
      for (auto succ : bb->succs) {
        auto& succ_info = bb_infos[succ->id];
        SetUnionInPlace(bb_info.out, succ_info.in);
      }
      if (bb->stmt_ids.empty()) bb_info.in = bb_info.out;
      for (int i = bb->stmt_ids.size() - 1; i >= 0; --i) {
        const Stmt* s = preorder_stmts[bb->stmt_ids[i]];
        auto& sl = stmt_linfo[s];
        if (i == static_cast<int>(bb->stmt_ids.size()) - 1)
          sl.live_out = bb_info.out;
        else
          sl.live_out = stmt_linfo[preorder_stmts[bb->stmt_ids[i + 1]]].live_in;
        // live_in = use U (live_out - def), computed in-place
        sl.live_in = sl.live_out;
        SetDiffInPlace(sl.live_in, sl.def);
        SetUnionInPlace(sl.live_in, sl.use);
        // If x is in use, then alias_[x] and bindings_[x] should also
        // be in use.
        for (const auto& item : sl.use) {
          VarSet alias_tc, binding_tc;
          if (alias_.count(item)) {
            alias_tc = TransitiveClosure({alias_[item]}, alias_, bindings_);
            SetUnionInPlace(sl.live_in, alias_tc);
          }
          if (bindings_.count(item)) {
            binding_tc = TransitiveClosure(bindings_[item], bindings_, alias_);
            SetUnionInPlace(sl.live_in, binding_tc);
          }
          VST_DEBUG({
            if (!alias_tc.empty() || !binding_tc.empty())
              dbgs() << "The tc of use " << item << " in " << StmtStr(s);
            if (!alias_tc.empty())
              for (const auto& i : alias_tc) dbgs() << "\t" << i << "\n";
            if (!binding_tc.empty())
              for (const auto& i : binding_tc) dbgs() << "\t" << i << "\n";
          });
        }
        if (i == 0) bb_info.in = sl.live_in;
        // Restore binding relationship deleted by AST::Wait
        if (stmt2binding_restore.count(s)) {
          assert(isa<AST::Wait>(s));
          for (const auto& fut_name : stmt2binding_restore[s])
            for (const auto& [src, dst] : future_buffers[fut_name])
              AddBinding(fut_name, src);
        }
        size_t id = stmt2id[s];
        for (const auto& var : sl.live_in) UpdateVarRange(var, id);
        for (const auto& var : sl.def) {
          if (sl.live_out.count(var)) {
            assert(!var_ranges[var].Values().empty());
            UpdateVarRange(var, id);
          } else {
            var_ranges[var].PushBack(Range{id, id});
          }
        }
      }
    }
  }
  std::vector<std::pair<std::string, Ranges>> var_live_ranges;
  for (auto& [var, ranges] : var_ranges) {
    ranges.Merge();
    var_live_ranges.push_back({var, ranges});
  }
  std::sort(var_live_ranges.begin(), var_live_ranges.end(),
            [&](const std::pair<std::string, Ranges>& a,
                const std::pair<std::string, Ranges>& b) {
              if (a.second.Empty()) return !b.second.Empty();
              if (b.second.Empty()) return false;
              return std::tie(a.second.front().start, a.first) <
                     std::tie(b.second.front().start, b.first);
            });

  DumpLivenessResults(var_live_ranges, bb_infos_map);
  VerifyLiveRanges();
}

void LivenessAnalyzer::VerifyLiveRanges() const {
  for (const auto& [var, ranges] : var_ranges) {
    const auto& vals = ranges.Values();
    for (size_t i = 0; i < vals.size(); ++i) {
      assert(vals[i].start <= vals[i].end && "live range start exceeds end");
      if (i + 1 < vals.size())
        assert(vals[i].end < vals[i + 1].start &&
               "overlapping live ranges after merge");
    }
  }
}

// ---------- Visit helpers and dump ----------

void LivenessAnalyzer::HandleSelect(AST::Node& n, ptr<AST::Select> sel) {
  std::string name;
  if (auto nvd = dyn_cast<AST::NamedVariableDecl>(&n))
    name = nvd->name_str;
  else if (auto assign = dyn_cast<AST::Assignment>(&n))
    name = assign->GetName();
  else
    choreo_unreachable("expecting a NamedVariableDecl or Assignment node.");
  assert(isa<FutureType>(NodeType(*sel)) || isa<SpannedType>(NodeType(*sel)));
  stmt_linfo[current_stmt].buffer_related = true;
  AddDef(current_stmt, name);
  AddIsBinding(current_stmt, name);
  for (const auto& item : sel->expr_list->AllValues()) {
    auto id = AST::GetIdentifier(*item);
    if (id) {
      AddUse(current_stmt, id->name);
      // Bind name with id->name
      // because if the sym in select is future
      // id->name has been bound with the src and dst.
      // And in co code, the future is used directly later.
      AddBinding(name, id->name);
      for (const DMABufInfo& buf_info : future_buffers[InScopeName(id->name)])
        AddFut2Buffers(name, buf_info);
    } else if (auto expr = dyn_cast<AST::Expr>(item)) {
      // TODO: actually, there are many situations that the node can be either
      // an identifier or an expression. Need to handle them in other Nodes!
      VarSet ops = GetAllSymbolicOperands(item.get());
      AddUse(current_stmt, ops);
      for (const auto& op : ops) {
        AddBinding(name, op);
        if (future_buffers.count(op))
          for (const DMABufInfo& buf_info : future_buffers[op])
            AddFut2Buffers(name, buf_info);
      }
    } else {
      choreo_unreachable("only expecting an identifier in Select, but got " +
                         PSTR(item));
    }
  }
}

inline std::string LivenessAnalyzer::StmtStr(const Stmt* stmt) const {
  return stmt2str.at(stmt);
}

ptr<LivenessAnalyzer::BB> LivenessAnalyzer::StartChildBB() {
  bb_list.push_back(cur_bb);
  auto child = AST::Make<BasicBlock>();
  child->id = cur_bb->id + 1;
  ConnectBB(cur_bb, child);
  cur_bb = child;
  return child;
}

std::string LivenessAnalyzer::GetEventName(const AST::Node& node) {
  auto expr = dyn_cast<AST::Expr>(&node);
  assert(IsSymbolOrArrayRef(node) &&
         "expect either symbol or array reference.");
  if (expr->op == Op::ElemOf) return AST::GetArrayBaseSymbol(*expr)->name;
  return AST::GetIdentifier(node)->name;
}

void LivenessAnalyzer::DumpStmtBriefly(const Stmt& n, std::ostream& os,
                                       bool dump_brace, bool only_else) {
#if DUMP_STMT_WITH_TYPE_INFO
  os << std::left << std::setw(25) << n.TypeNameString();
#endif
  auto id = std::to_string(stmt2id.at(&n));
  if (id.size() < 3) id = std::string(3 - id.size(), ' ') + id;
  os << "(" << id << ") ";
  os << dump_indent_;

  // special case for the else scope of if-else block
  if (only_else) {
    os << "else" << (dump_brace ? " {" : "") << "\n";
    return;
  }

  if (const auto ntd = dyn_cast<AST::NamedTypeDecl>(&n)) {
    os << ntd->name_str << " : " << PSTR(ntd->init_expr);
  } else if (const auto nvd = dyn_cast<AST::NamedVariableDecl>(&n)) {
    DumpNVD(*nvd, os);
  } else if (const auto assign = dyn_cast<AST::Assignment>(&n)) {
    assign->da->Print(os);
    os << " = " << PSTR(assign->value);
  } else if (const auto dma = dyn_cast<AST::DMA>(&n)) {
    DumpDMA(*dma, os);
  } else if (const auto mma = dyn_cast<AST::MMA>(&n)) {
    DumpMMA(*mma, os);
  } else if (const auto w = dyn_cast<AST::Wait>(&n)) {
    os << "wait ";
    w->targets->Print(os);
  } else if (const auto c = dyn_cast<AST::Call>(&n)) {
    os << "call " << PSTR(c->function);
    if (c->template_args) {
      os << "<";
      c->template_args->InlinePrint(os);
      os << ">";
    }
    os << "(";
    c->arguments->InlinePrint(os);
    os << ")";
  } else if (const auto r = dyn_cast<AST::Rotate>(&n)) {
    os << ((r->ids->Count() == 2) ? "swap" : "rotate") << "(";
    r->ids->InlinePrint(os);
    os << ")";
  } else if (const auto sync = dyn_cast<AST::Synchronize>(&n)) {
    os << "sync." << STR(sync->Resource());
  } else if (const auto tr = dyn_cast<AST::Trigger>(&n)) {
    os << "trigger ";
    tr->targets->Print(os);
  } else if (const auto ret = dyn_cast<AST::Return>(&n)) {
    os << "return";
    if (ret->value) os << " " << PSTR(ret->value);
  } else if (const auto pb = dyn_cast<AST::ParallelBy>(&n)) {
    os << "parallel ";
    os << pb->BPV()->name << " = {";
    os << DelimitedSTR(pb->AllSubPVs());
    os << "} by " << "[";
    pb->PrintBounds(os);
    os << "]";
    if (pb->GetLevel() != ParallelLevel::NONE)
      os << " : " << STR(pb->GetLevel());
  } else if (const auto wb = dyn_cast<AST::WithBlock>(&n)) {
    os << "with ";
    for (size_t i = 0; i < wb->withins->Count(); ++i) {
      if (i > 0) os << ", ";
      auto w = cast<AST::WithIn>(wb->withins->values[i]);
      if (w->with) os << w->with->name;
      if (w->with_matchers) {
        if (w->with) os << " = ";
        os << "{";
        w->with_matchers->InlinePrint(os);
        os << "}";
      }
      os << " in " << PSTR(w->in);
    }
    if (wb->reqs) {
      os << " where ";
      wb->reqs->Print(os);
    }
  } else if (const auto fb = dyn_cast<AST::ForeachBlock>(&n)) {
    os << "foreach ";
    for (size_t i = 0; i < fb->ranges->Count(); ++i) {
      if (i > 0) os << ", ";
      auto lr = cast<AST::LoopRange>(fb->ranges->values[i]);
      os << lr->GetRV()->name << "(";
      os << (lr->lbound ? PSTR(lr->lbound) : "") << ":";
      os << (lr->ubound ? PSTR(lr->ubound) : "") << ":";
      os << (IsValidStep(lr->step) ? std::to_string(lr->step) : "") << ")";
    }
  } else if (const auto wb = dyn_cast<AST::WhileBlock>(&n)) {
    os << "while ";
    os << PSTR(wb->pred);
  } else if (const auto itb = dyn_cast<AST::InThreadsBlock>(&n)) {
    os << "inthreads" << (itb->async ? ".async " : " ") << PSTR(itb->pred);
  } else if (const auto ie = dyn_cast<AST::IfElseBlock>(&n)) {
    os << "if ";
    if (!HasStmt(*ie->pred)) {
      ie->pred->Print(os, " ");
    } else {
      os << "(the condition is the next line)";
    }
  } else if (const auto cf = dyn_cast<AST::ChoreoFunction>(&n)) {
    os << cf->f_decl.name << "(";
    for (size_t i = 0; i < cf->f_decl.params->values.size(); ++i) {
      if (i > 0) os << ", ";
      const auto& p = cf->f_decl.params->values[i];
      if (p->HasSymbol()) {
        auto id = cast<AST::Identifier>(p->sym);
        GetSymbolType(id->name)->Print(os);
        os << " " << id->name;
      } else {
        choreo_unreachable("expecting the parameter has a symbol.");
      }
    }
    os << ")";
  } else if (const auto dummy = dyn_cast<ScopeEnd>(&n)) {
    os << "}";
  } else {
    choreo_unreachable(
        "unhandled stmt type in DumpStmtBriefly: " + n.TypeNameString() +
        ". Did you add it to HasStmt() but not here?");
  }

  os << (dump_brace ? " {" : "") << "\n";
}

void LivenessAnalyzer::DumpNVD(const AST::NamedVariableDecl& nvd,
                               std::ostream& os) {
  if (nvd.type && nvd.type->IsUnknown()) {
    GetSymbolType(nvd.name_str)->Print(os);
    os << " " << nvd.name_str;
  } else {
    if (nvd.mem) {
      nvd.mem->Print(os);
      os << " ";
    }
    if (nvd.type)
      nvd.type->Print(os);
    else
      GetSymbolType(nvd.name_str)->Print(os);
    os << " " << nvd.name_str;
    if (nvd.IsArray())
      for (const auto& d : nvd.ArrayDimensions()->AllValues())
        os << "[" << STR(d) << "]";
  }
  if (nvd.init_expr)
    os << " " << nvd.init_str << " " << PSTR(nvd.init_expr);
  else if (nvd.init_value)
    os << " " << nvd.init_str << " {" << PSTR(nvd.init_value) << "}";
}

void LivenessAnalyzer::DumpDMA(const AST::DMA& dma, std::ostream& os) {
  if (dma.operation == ".any") {
    os << (dma.future.empty() ? "?" : dma.future);
    os << " = dma.any";
  } else {
    if (dma.future.empty() && !dma.HasEvent())
      assert(!dma.IsAsync() &&
             "async dma should have a future or event to wait on.");
    else
      os << dma.future << " = ";
    os << "dma" << dma.operation;
    if (dma.IsAsync()) os << ".async";
    if (dma.HasEvent()) os << "<" << STR(dma.Event()) << ">";
    os << (dma.config ? " " + PSTR(dma.config) : "") << " ";
    os << STR(dma.from) << " => " << STR(dma.to);
  }
  if (dma.chained) {
    if (!dma.chain_to.empty()) os << ", chain_to " << dma.chain_to;
    if (!dma.chain_from.empty()) os << ", chain_from " << dma.chain_from;
  }
}

void LivenessAnalyzer::DumpMMA(const AST::MMA& mma, std::ostream& os) {
  // TODO: handel swizzle, scale; use values after typeinfer.
  auto op = mma.GetOperation();
  if (auto frag = op->GetFrag()) {
    switch (op->Tag()) {
    case AST::MMAOperation::Fill: {
      if (op->FillingIsDecl()) {
        os << PSTR(frag) << " = mma.fill." << STR(op->FillingType()) << " "
           << PSTR(op->FillingValue());
      } else {
        os << " = mma.fill." << STR(op->FillingType()) << " " << PSTR(frag)
           << ", " << PSTR(op->FillingValue());
      }
    } break;
    case AST::MMAOperation::Load: {
      os << "mma.load " << (op->IsAsync() ? ".async" : "") << " "
         << PSTR(op->LoadFrom());
    } break;
    case AST::MMAOperation::LoadR: {
      os << "mma.load " << PSTR(op->LoadFrom());
      if (op->LoadTo()) os << ", " << PSTR(op->LoadTo());
    } break;
    case AST::MMAOperation::Exec: {
      os << "mma.exec";
      switch (op->GetMethod()) {
      case AST::MMAOperation::ROW_ROW: os << ".ROW.ROW"; break;
      case AST::MMAOperation::ROW_COL: os << ".ROW.COL"; break;
      case AST::MMAOperation::COL_COL: os << ".COL.COL"; break;
      case AST::MMAOperation::COL_ROW: os << ".COL.ROW"; break;
      default: choreo_unreachable("unsupported dma execution mode."); break;
      }
      if (op->IsSparse()) os << ".SP";
      if (op->HasScale()) os << ".SCALE";
      // TODO: missing scale
      os << " " << op->ExecOperand(0) << ", " << op->ExecOperand(1) << ", "
         << op->ExecOperand(2);
    } break;
    case AST::MMAOperation::Store: {
      os << "mma.store" << (op->StoreIsTranspose() ? ".transp" : "") << " "
         << op->StoreFrom() << ", " << PSTR(op->StoreTo());
    } break;
    case AST::MMAOperation::Commit: {
      os << "mma.commit";
    } break;
    case AST::MMAOperation::Wait: {
      os << "mma.wait<" << op->WaitDepth() << ">";
    } break;
    case AST::MMAOperation::Scale: {
      os << "mma.scale " << PSTR(op->ScaleAccumulator()) << ", "
         << PSTR(op->ScaleA()) << ", " << PSTR(op->ScaleB());
    } break;
    default: choreo_unreachable("unexpect MMA operation.");
    }
  }
}

// ---------- CFG construction ----------

bool LivenessAnalyzer::IsLoopBlock(const AST::Node& n) {
  return isa<AST::ForeachBlock>(&n) || isa<AST::WhileBlock>(&n);
}

// TODO: should wait dma be treated as holding point?
bool LivenessAnalyzer::IsSyncPoint(const AST::Node& n) {
  return isa<AST::Synchronize>(&n);
}

bool LivenessAnalyzer::ShouldIndent(const AST::Node& n) {
  return isa<AST::ParallelBy>(&n) || isa<AST::WithBlock>(&n) ||
         isa<AST::ForeachBlock>(&n) || isa<AST::WhileBlock>(&n) ||
         isa<AST::InThreadsBlock>(&n) || isa<AST::IfElseBlock>(&n) ||
         isa<AST::ChoreoFunction>(&n);
}

void LivenessAnalyzer::RecordStmt(const Stmt& s, bool dump_brace,
                                  bool only_else) {
  std::stringstream ss;
  DumpStmtBriefly(s, ss, dump_brace, only_else);
  stmt2str.emplace(&s, ss.str());
#if DUMP_EACH_STMT
  VST_DEBUG(dbgs() << StmtStr(&s));
#endif
  DumpStmtBriefly(s, stmts_with_indent, dump_brace, only_else);
}

ptr<LivenessAnalyzer::ScopeEnd>
LivenessAnalyzer::RegisterScopeEnd(AST::Node& origin, ptr<BasicBlock> bb,
                                   bool dump_brace, bool only_else) {
  auto se = AST::Make<ScopeEnd>(origin.LOC(), &origin);
  scope_ends.push_back(se);
  preorder_stmts.push_back(se.get());
  stmt2id.emplace(se.get(), stmt_id);
  bb->stmt_ids.push_back(stmt_id);
  stmt2bb.emplace(se.get(), bb);
  ++stmt_id;
  RecordStmt(*se, dump_brace, only_else);
  return se;
}

void LivenessAnalyzer::HandleStmtInBefore(AST::Node& n) {
  if (!HasStmt(n)) return;

  preorder_stmts.push_back(&n);

  current_stmt = &n;
  stmt2id.emplace(&n, stmt_id);
  RecordStmt(n, ShouldIndent(n));
  if (ShouldIndent(n)) {
    IncrDumpIndent();
    scope2stmt.emplace(SSTab().ScopeName(), &n);
  }

  if (ShouldNewBB(n)) {
    if (cur_bb) bb_list.push_back(cur_bb);
    cur_bb = AST::Make<BasicBlock>();
    if (!bb_list.empty()) {
      cur_bb->id = bb_list.back()->id + 1;
      ConnectBB(bb_list.back(), cur_bb);
    }
  }
  cur_bb->stmt_ids.push_back(stmt_id);
  stmt2bb.emplace(current_stmt, cur_bb);
  ++stmt_id;

  if (isa<AST::IfElseBlock>(&n)) {
    cur_bb->is_condition = true;
    ie_bb_list.push(IfElseBBs{._if = cur_bb});
  } else if (isa<AST::InThreadsBlock>(&n)) {
    cur_bb->is_inthreads = true;
    it_bb_list.push(InThreadsBBs{._it = cur_bb});
  }

  if (IsSyncPoint(n)) cur_bb->is_sync_point = true;
}

void LivenessAnalyzer::HandleStmtInMid(AST::Node& n) {
  if (!HasStmt(n)) return;

  auto ie = dyn_cast<AST::IfElseBlock>(&n);
  if (!ie) return;

  bb_list.push_back(cur_bb);
  if (ie->GetThenBody()) ie_bb_list.top()._then = bb_list.back();
  auto _else = AST::Make<BasicBlock>();
  _else->id = bb_list.back()->id + 1;
  ie_bb_list.top()._else = _else;
  assert(!ie_bb_list.empty());
  ConnectBB(ie_bb_list.top()._if, _else);
  cur_bb = _else;

  // if there is no else block,
  // do the rbrace job in HandleStmtInAfter as usual.
  if (!ie->HasElse()) return;

  // complete the end of `if scope`     (rbrace)
  // process the start of `else scope`  (lbrace)
  DecrDumpIndent();
  RegisterScopeEnd(n, ie_bb_list.top()._then, false);

  auto else_start = RegisterScopeEnd(n, ie_bb_list.top()._else, true, true);
  scope2stmt.emplace(SSTab().ScopeName(), else_start.get());
  IncrDumpIndent();
}

void LivenessAnalyzer::HandleStmtInAfter(AST::Node& n) {
  if (!HasStmt(n)) return;

  // only after-handle stmts which need indent
  if (!ShouldIndent(n)) return;

  DecrDumpIndent();

  // generate the rbrace as scope end
  auto rbrace = AST::Make<ScopeEnd>(n.LOC(), &n);
  scope_ends.push_back(rbrace);
  preorder_stmts.push_back(rbrace.get());
  stmt2id.emplace(rbrace.get(), stmt_id);

  // Assign rbrace to the correct basic block
  ptr<BB> rbrace_bb = cur_bb;
  if (auto ie = dyn_cast<AST::IfElseBlock>(&n); ie && !ie->HasElse())
    rbrace_bb = ie_bb_list.top()._then;
  rbrace_bb->stmt_ids.push_back(stmt_id);
  stmt2bb.emplace(rbrace.get(), rbrace_bb);

  // Wire up the end of control flow constructs
  if (isa<AST::InThreadsBlock>(&n)) {
    bb_list.push_back(cur_bb);
    it_bb_list.top()._then = cur_bb;
    auto _end = AST::Make<BasicBlock>();
    _end->is_end = true;
    _end->id = bb_list.back()->id + 1;
    ConnectBB(it_bb_list.top()._then, _end);
    end2cond.emplace(_end, it_bb_list.top()._it);
    it_bb_list.pop();
    cur_bb = _end;
  } else if (auto ie = dyn_cast<AST::IfElseBlock>(&n)) {
    if (ie->HasElse()) ie_bb_list.top()._else = cur_bb;
    bb_list.push_back(cur_bb);
    auto _end = AST::Make<BasicBlock>();
    _end->is_end = true;
    _end->id = bb_list.back()->id + 1;
    ConnectBB(ie_bb_list.top()._then, _end);
    ConnectBB(ie_bb_list.top()._else, _end);
    end2cond.emplace(_end, ie_bb_list.top()._if);
    ie_bb_list.pop();
    cur_bb = _end;
  } else if (auto cf = dyn_cast<AST::ChoreoFunction>(&n)) {
    bb_list.push_back(cur_bb);
    bb_lists.emplace(cf->name, bb_list);
    bb_list.clear();
    cur_bb = nullptr;
  }

  ++stmt_id;
  RecordStmt(*rbrace, false);

  if (IsLoopBlock(n) && events_to_add.count(&n)) {
    for (const auto& [event_type, var] : events_to_add[&n]) {
      assert(event_type == EventKind::Use && "unexpected event type.");
      VST_DEBUG(dbgs() << "EXTRA use: " << var << " in "
                       << stmt2id[rbrace.get()] << "\n";);
      AddUse(rbrace.get(), var, false);
    }
  }
}

// ---------- Visitor framework callbacks ----------

bool LivenessAnalyzer::BeforeVisitImpl(AST::Node& n) {
  // skip the BeforeVisitImpl of Call node in pattern: v = Call x(...)
  if (auto c = dyn_cast<AST::Call>(&n); c && c->IsExpr()) return true;

  HandleStmtInBefore(n);

  if (auto cf = dyn_cast<AST::ChoreoFunction>(&n)) {
    for (const auto& param : cf->f_decl.params->values) {
      if (!param->HasSymbol()) continue;
      std::string sname = InScopeName(param->sym->name);
      if (auto sty = dyn_cast<SpannedType>(param->GetType())) {
        stmt_linfo[current_stmt].buffer_related = true;
        AddBuffer(sname, sty->m_type);
        if (sty->RuntimeShaped()) {
          auto shape = sty->GetShape();
          for (const auto& v : shape.GetDynamicSymbols())
            AddDef(current_stmt, STR(v));
        }
      }
      AddDef(current_stmt, sname, true);
    }
  } else if (auto ib = dyn_cast<AST::InThreadsBlock>(&n)) {
    // For vars which are defined inside inthreads.async block,
    // an extra use should be added to the sync point statement.
    // In other words, their liveness is extended to the sync point.
    if (ib->async) ++inthreads_async_level;
  }

  return true;
}

bool LivenessAnalyzer::InMidVisitImpl(AST::Node& n) {
  HandleStmtInMid(n);
  return true;
}

namespace {

__attribute__((unused)) void DumpCfgToDot(
    const std::map<std::string, std::vector<ptr<LivenessAnalyzer::BB>>>&
        blocks_list) {
  for (const auto& [fname, blocks] : blocks_list) {
    assert(!blocks.empty());
    std::string base_name = fname + ".dot";
    std::string filename = base_name;
    const auto& debug_dir = CCtx().GetDebugFileDir();
    if (!debug_dir.empty()) {
      filename = debug_dir;
      if (!filename.empty()) {
        const char last = filename.back();
        if (last != '/' && last != '\\') filename += '/';
      }
      filename += base_name;
    }
    std::ofstream ofs(filename);
    ofs << "digraph CFG {\n";
    ofs << "  node [shape=box, style=filled, fillcolor=lightgray];\n";
    for (auto& bb : blocks) {
      std::string label = "BB" + std::to_string(bb->id);
      if (!bb->stmt_ids.empty()) {
        label += "\\nstmt: ";
        bool roll = true;
        std::string temp = "";
        for (size_t i = 0; i < bb->stmt_ids.size(); i++) {
          temp += std::to_string(bb->stmt_ids[i]);
          if (i > 0 && bb->stmt_ids[i] != bb->stmt_ids[i - 1] + 1) roll = false;
          if (i + 1 < bb->stmt_ids.size()) temp += ",";
        }
        if (roll && bb->stmt_ids.size() > 1)
          label += std::to_string(bb->stmt_ids.front()) + " ~ " +
                   std::to_string(bb->stmt_ids.back());
        else
          label += temp;
      }
      if (bb->is_sync_point) label += "\\n(sync)";
      if (bb->is_condition) label += "\\n(cond)";
      if (bb->is_inthreads) label += "\\n(inthreads)";
      if (bb->is_end) label += "\\n(end)";

      ofs << "  " << bb->id << " [label=\"" << label << "\"];\n";

      for (auto& succ : bb->succs)
        ofs << "  " << bb->id << " -> " << succ->id << ";\n";
    }

    ofs << "}\n";
    ofs.close();
    dbgs() << "CFG written to " << filename << std::endl;
  }
}

} // anonymous namespace

bool LivenessAnalyzer::AfterVisitImpl(AST::Node& n) {
  HandleStmtInAfter(n);

  if (isa<AST::Program>(&n)) {
    VST_DEBUG(dbgs() << "\n" << stmts_with_indent.str() << "\n");
    ComputeLiveRange();
  } else if (auto w = dyn_cast<AST::Wait>(&n)) {
    // after wait node, the async dma is done, remove the binding
    for (const auto& f : w->GetTargets()) {
      if (!isa<FutureType>(NodeType(*f))) continue;
      auto id = AST::GetIdentifier(*f);
      assert(id && "expecting an identifier in Wait.");
      auto fut_name = InScopeName(id->name);
      assert(future_buffers.count(fut_name) &&
             "expecting the future to be in future_buffers.");
      for (const auto& [src, dst] : future_buffers[fut_name])
        RemoveBinding(fut_name, src);
      // since we will calculate live_in and live_out after visiting all the
      // nodes, we should record the binding info to do restoration in
      // ComputeLiveRange().
      stmt2binding_restore[&n].push_back(fut_name);
    }
  } else if (auto ib = dyn_cast<AST::InThreadsBlock>(&n)) {
    if (ib->async) --inthreads_async_level;
  }

  return true;
}

// ---------- Per-node Visit handlers ----------

bool LivenessAnalyzer::Visit(AST::NamedTypeDecl& n) {
  TraceEachVisit(n);
  AddDef(&n, n.name_str);
  AddUse(&n, GetAllSymbolicOperands(n.init_expr.get()));
  return true;
}

bool LivenessAnalyzer::Visit(AST::NamedVariableDecl& n) {
  TraceEachVisit(n);
  // mem buffer can only be defined here
  auto ty = GetSymbolType(n.name_str);
  if (auto sel = dyn_cast<AST::Select>(n.init_expr)) {
    // TODO: could be a select with non-span vars?
    HandleSelect(n, sel);
    return true;
  }
  if (isa<ScalarType>(ty) || isa<VectorType>(ty)) {
    if (n.init_expr)
      AddUse(current_stmt, GetAllSymbolicOperands(n.init_expr.get()));
    AddDef(current_stmt, n.name_str);
    return true;
  }
  if (isa<StringType>(ty))
    choreo_unreachable("string variables are not supported yet.");
  if (isa<ITupleType>(ty) || isa<BoundedType>(ty)) {
    AddDef(current_stmt, n.name_str);
    AddUse(current_stmt, GetAllSymbolicOperands(n.init_expr.get()));
    return true;
  }
  if (auto sty = dyn_cast<SpannedType>(ty)) {
    stmt_linfo[current_stmt].buffer_related = true;
    if (!IsRef(n)) {
      AddDef(current_stmt, n.name_str, true);
      AddBuffer(InScopeName(n.name_str), sty->m_type);
      return true;
    }
    VST_DEBUG(dbgs() << "The nvd is a reference: " << STR(n) << ".\n");
    assert(n.init_expr && "expecting the init_expr is not nullptr.");
    auto e = dyn_cast<AST::Expr>(n.init_expr);
    assert(e && "expecting the init_expr is an expr.");
    if (auto sa = dyn_cast<AST::SpanAs>(e->GetR())) {
      AddDef(current_stmt, n.name_str, true);
      AddUse(current_stmt, sa->id->name);
      AddIsAlias(current_stmt, n.name_str);
      AddAlias(n.name_str, sa->id->name);
    } else if (e->GetOp() == Op::ElemOf) {
      // `s0 = s[xxx]`, where s is of spannedarray type. Alias conservatively.
      AddDef(current_stmt, n.name_str, true);
      std::string base_array = AST::GetArrayBaseSymbol(*e)->name;
      AddUse(current_stmt, base_array);
      AddIsAlias(current_stmt, n.name_str);
      AddAlias(n.name_str, base_array);
    } else {
      choreo_unreachable("expecting the init_expr is a span_as.");
    }
    return true;
  }
  if (isa<EventType>(ty)) {
    AddDef(current_stmt, n.name_str);
    return true;
  }
  choreo_unreachable("unexpect type of variable decl: " + PSTR(ty) + ".");
  return true;
}

bool LivenessAnalyzer::Visit(AST::Assignment& n) {
  TraceEachVisit(n);
  if (auto sel = dyn_cast<AST::Select>(n.value)) {
    HandleSelect(n, sel);
    return true;
  }
  if (auto sa = dyn_cast<AST::SpanAs>(n.value)) {
    assert(IsRef(n) && "expecting the spanas assignment is a reference.");
    stmt_linfo[current_stmt].buffer_related = true;
    AddDef(current_stmt, n.GetName(), true);
    AddUse(current_stmt, sa->id->name);
    AddIsAlias(current_stmt, n.GetName());
    AddAlias(n.GetName(), sa->id->name);
    return true;
  }
  VST_DEBUG(dbgs() << "The assignment is not sel or sa: " << STR(n) << ".\n");
  if (n.AssignToDataElement())
    AddUse(current_stmt, n.GetDataArrayName());
  else if (PrefixedWith(n.GetName(), "anon_"))
    AddDef(current_stmt, n.GetName());
  else {
    if (n.IsDecl())
      AddDef(current_stmt, n.GetName());
    else
      AddUse(current_stmt, n.GetName());
  }

  if (isa<AST::Expr>(n.value))
    AddUse(current_stmt, GetAllSymbolicOperands(n.value.get()));
  else if (!isa<AST::Call>(n.value)) // Call is handled in Visit(AST::Call)
    choreo_unreachable("expecting the assignment value is an expr or call.");
  return true;
}

bool LivenessAnalyzer::Visit(AST::ParallelBy& n) {
  TraceEachVisit(n);
  assert(n.HasSubPVs() && "expecting the parallel-by has bpv and cmpt_bpvs.");
  AddDef(current_stmt, n.BPV()->name);
  events_to_add[current_stmt].insert({EventKind::Use, n.BPV()->name});
  paraby_bounded_vars.insert(n.BPV()->name);
  for (const auto& iv_symbol : n.AllSubPVs()) {
    std::string iv_symbol_name = cast<AST::Identifier>(iv_symbol)->name;
    AddDef(current_stmt, iv_symbol_name);
    events_to_add[current_stmt].insert({EventKind::Use, iv_symbol_name});
    paraby_bounded_vars.insert(iv_symbol_name);
    AddBinding(n.BPV()->name, iv_symbol_name);
  }
  return true;
}

bool LivenessAnalyzer::Visit(AST::WithBlock& n) {
  TraceEachVisit(n);
  VarSet def_set;
  for (const auto& item : n.withins->AllSubs()) {
    auto w = cast<AST::WithIn>(item);
    if (w->with) {
      AddDef(current_stmt, w->with->name);
      def_set.insert(w->with->name);
    }
    if (w->with_matchers) {
      for (const auto& item : w->with_matchers->AllValues()) {
        auto id = cast<AST::Identifier>(item);
        AddDef(current_stmt, id->name);
        def_set.insert(id->name);
        if (w->with) AddBinding(w->with->name, id->name);
      }
    }
    AddUse(current_stmt, GetAllSymbolicOperands(w->in.get()));
  }
  if (n.reqs) {
    for (const auto& req : n.reqs->AllSubs()) {
      auto wb = cast<AST::WhereBind>(req);
      // special case: if a symbol in where is defined in with or with_matchers,
      // ignore it. Because it is a use that just after define.
      if (!def_set.count(AST::GetIdentifier(*wb->lhs)->name))
        AddUse(current_stmt, AST::GetIdentifier(*wb->lhs)->name);
      if (!def_set.count(AST::GetIdentifier(*wb->rhs)->name))
        AddUse(current_stmt, AST::GetIdentifier(*wb->rhs)->name);
    }
  }
  return true;
}

bool LivenessAnalyzer::Visit(AST::DMA& n) {
  TraceEachVisit(n);
  stmt_linfo[current_stmt].buffer_related = true;
  /*
  If the dma is of sync, then only the dst buffer is alias to the future.
  The src buffer can be reused immediately after the dma stmt.

  If the dma is of async, then the future is alias to both the src/dst buffers.
  The corresponding src buffer can be reused only after the future has been
  waited.
  */
  if (n.future.empty()) {
    assert((!n.IsAsync() || n.HasEvent()) &&
           "async dma should have a future or a event.");
    AddUse(current_stmt, n.FromSymbol());
    AddUse(current_stmt, n.ToSymbol());
  } else if (n.operation == ".any") {
    AddDef(current_stmt, n.future, true);
    dma_any_futures.insert(InScopeName(n.future));
    return true;
  } else {
    if (dma_any_futures.count(InScopeName(n.future)))
      AddUse(current_stmt, n.future);
    else
      AddDef(current_stmt, n.future, true);
    AddIsBinding(current_stmt, n.future);
    if (n.IsAsync()) AddBinding(n.future, n.FromSymbol());
    AddBinding(n.future, n.ToSymbol());
    AddFut2Buffers(n.future, DMABufInfo{n.FromSymbol(), n.ToSymbol()});
  }

  if (n.chained && !n.chain_from.empty()) AddUse(current_stmt, n.chain_from);

  if (auto pc = dyn_cast<PadConfig>(n.GetConfig())) {
    for (const auto& mv : {pc->pad_low, pc->pad_high, pc->pad_mid})
      for (const auto& v : mv->AllValues())
        AddUse(current_stmt, GetAllSymbolicOperands(v.get()));
    AddUse(current_stmt, GetAllSymbolicOperands(pc->GetPadValue().get()));
  }

  return true;
}

bool LivenessAnalyzer::Visit(AST::MMA& n) {
  TraceEachVisit(n);
  stmt_linfo[current_stmt].buffer_related = true;
  auto op = n.GetOperation();
  switch (op->Tag()) {
  case AST::MMAOperation::Scale: {
    AddUse(current_stmt, GetAllSymbolicOperands(op->ScaleA().get()));
    AddUse(current_stmt, GetAllSymbolicOperands(op->ScaleB().get()));
  } break;
  case AST::MMAOperation::Exec: {
    AddUse(current_stmt, GetAllSymbolicOperands(op->ExecOperand(1).get()));
    AddUse(current_stmt, GetAllSymbolicOperands(op->ExecOperand(2).get()));
  } break;
  case AST::MMAOperation::Store: {
    AddUse(current_stmt, GetAllSymbolicOperands(op->StoreTo().get()));
  } break;
  case AST::MMAOperation::Load:
  case AST::MMAOperation::LoadR: {
    AddUse(current_stmt, GetAllSymbolicOperands(op->LoadFrom().get()));
  } break;
  default: break;
  }
  return true;
}

bool LivenessAnalyzer::Visit(AST::ChunkAt& n) {
  TraceEachVisit(n);
  assert(n.sa == nullptr && "after norm, there should be no span_as.");

  AddUse(current_stmt, n.RefSymbol());

  for (auto tsi : n.AllOperations())
    for (const auto& rfn : tsi->ReferredNodes()) {
      VST_DEBUG(dbgs() << "chunkat referred node: " << PSTR(rfn) << ".\n");
      VarSet operands = GetAllSymbolicOperands(rfn.get());
      AddUse(current_stmt, operands);
    }
  return true;
}

bool LivenessAnalyzer::Visit(AST::Wait& n) {
  TraceEachVisit(n);
  for (const auto& item : n.GetTargets()) {
    if (isa<FutureType>(NodeType(*item))) {
      // future
      stmt_linfo[current_stmt].buffer_related = true;
      auto id = AST::GetIdentifier(*item);
      assert(id && "expecting an identifier in Wait.");
      AddUse(current_stmt, id->name);
      const std::string sname = InScopeName(id->name);
      if (!future_buffers.count(sname))
        assert(
            dma_any_futures.count(sname) &&
            "expecting the future to be in dma_any if not in future_buffers.");
      for (const auto& [src, dst] : future_buffers[sname]) {
        AddUse(current_stmt, src);
        AddUse(current_stmt, dst);
      }
    } else {
      AddUse(current_stmt, GetEventName(*item));
    }
  }
  return true;
}

bool LivenessAnalyzer::Visit(AST::Call& n) {
  TraceEachVisit(n);
  for (const auto& arg : n.GetArguments()) {
    auto sty = GetSpannedType(NodeType(*arg));
    if (!sty) {
      AddUse(current_stmt, GetAllSymbolicOperands(arg.get()));
      continue;
    }
    stmt_linfo[current_stmt].buffer_related = true;
    if (auto id = AST::GetIdentifier(*arg)) {
      AddUse(current_stmt, id->name);
      continue;
    }
    auto expr = dyn_cast<AST::Expr>(arg);
    if (!expr) continue;
    if (expr->op == Op::DataOf || expr->op == Op::MDataOf) {
      // TODO: will only the dims of future be used?
      assert(isa<FutureType>(expr->GetR()->GetType()) &&
             "expect a future operand.");
      if (auto id = cast<AST::Expr>(expr->GetR())->GetSymbol())
        AddUse(current_stmt, id->name);
      else
        choreo_unreachable("Can not retrieve name of the future.");
    } else if (expr->op == Op::AddrOf) {
      if (auto id = AST::GetIdentifier(expr->GetR()))
        AddUse(current_stmt, id->name);
      else if (!isa<AST::DataAccess>(expr->GetR()))
        choreo_unreachable("Can not retrieve name of the future.");
    } else if (expr->op == Op::ElemOf) {
      if (auto base = AST::GetArrayBaseSymbol(*expr))
        AddUse(current_stmt, base->name);
    } else {
      assert(isa<AST::ChunkAt>(expr->GetR()) && "expect a chunkat operand.");
    }
  }
  return true;
}

bool LivenessAnalyzer::Visit(AST::Rotate& n) {
  TraceEachVisit(n);
  stmt_linfo[current_stmt].buffer_related = true;
  VarSet uses;
  for (const auto& item : n.GetIds()) {
    assert(isa<AST::Identifier>(item));
    auto id = cast<AST::Identifier>(item);
    auto sname = InScopeName(id->name);
    uses.insert(sname);
    for (const auto& [src, dst] : future_buffers[sname]) {
      uses.insert(src);
      uses.insert(dst);
    }
    // Each future in a Rotate is bound to every other future in the Rotate
    for (const auto& other : n.GetIds()) {
      auto other_id = cast<AST::Identifier>(other);
      auto other_sname = InScopeName(other_id->name);
      if (other_sname == sname) continue;
      for (const DMABufInfo& buf_info : future_buffers[other_sname])
        AddFut2Buffers(id->name, buf_info);
      AddBinding(id->name, other_id->name);
    }
  }
  AddUse(current_stmt, uses);
  return true;
}

bool LivenessAnalyzer::Visit(AST::Synchronize& n) {
  TraceEachVisit(n);
  std::string cur_scope = SSTab().ScopeName();
  visiting_synchronize = true;
  for (const auto& [scope, vars] : async_inthreads_vars) {
    if (!PrefixedWith(scope, cur_scope)) continue;
    // the scope of vars is inside cur_scope.
    /*
    {
      inthreads.async() { def x; ...}
      inthreads.async() { def y; ...}
      sync.shared; // add extra use of both x and y here manually!
    }
    */
    for (const auto& var : vars) AddUse(&n, var, false);
  }
  visiting_synchronize = false;
  return true;
}

bool LivenessAnalyzer::Visit(AST::Trigger& n) {
  TraceEachVisit(n);
  for (const auto& e : n.GetEvents()) AddUse(&n, GetEventName(*e));
  return true;
}

bool LivenessAnalyzer::Visit(AST::Select& n) {
  TraceEachVisit(n);
  // already handled in NamedVariableDecl or Assignment
  return true;
}

bool LivenessAnalyzer::Visit(AST::Return& n) {
  TraceEachVisit(n);
  auto vty = NodeType(*n.value);
  if (isa<SpannedType>(vty)) {
    stmt_linfo[current_stmt].buffer_related = true;
    if (auto id = AST::GetIdentifier(*n.value)) {
      AddUse(current_stmt, id->name);
    } else if (auto expr = dyn_cast<AST::Expr>(n.value);
               expr && expr->op == Op::DataOf) {
      auto id = cast<AST::Expr>(expr->GetR())->GetSymbol();
      assert(id && "expect a symbol");
      AddUse(current_stmt, id->name);
    } else {
      choreo_unreachable(
          "expecting the return value is an identifier or future.data.");
    }
  } else {
    auto expr = dyn_cast<AST::Expr>(n.value);
    assert(expr && "expecting the return value is a select.");
    VarSet operands = GetAllSymbolicOperands(expr.get());
    AddUse(current_stmt, operands);
  }
  return true;
}

bool LivenessAnalyzer::Visit(AST::ForeachBlock& n) {
  TraceEachVisit(n);
  for (const auto& item : n.GetRanges()) {
    auto range = cast<AST::LoopRange>(item);
    // Although the range var is reset to zero, still treat it as a use.
    AddUse(current_stmt, range->GetRVName());
    for (const auto& offset : {range->lbound, range->ubound}) {
      if (!offset) continue;
      if (auto id = AST::GetIdentifier(*offset))
        AddUse(current_stmt, id->name);
      else
        choreo_unreachable(
            "expecting the bound offset in LoopRange is an Identifier.");
    }
  }
  return true;
}

bool LivenessAnalyzer::Visit(AST::WhileBlock& n) {
  TraceEachVisit(n);
  AddUse(current_stmt, GetAllSymbolicOperands(n.pred.get()));
  return true;
}

bool LivenessAnalyzer::Visit(AST::InThreadsBlock& n) {
  TraceEachVisit(n);
  AddUse(current_stmt, GetAllSymbolicOperands(n.pred.get()));
  it_bb_list.top()._then = StartChildBB();
  return true;
}

bool LivenessAnalyzer::Visit(AST::IfElseBlock& n) {
  TraceEachVisit(n);
  AddUse(current_stmt, GetAllSymbolicOperands(n.pred.get()));
  assert(cur_bb && cur_bb == ie_bb_list.top()._if);
  ie_bb_list.top()._then = StartChildBB();
  return true;
}

bool LivenessAnalyzer::Visit(AST::FunctionDecl& n) {
  // deal with the parameters in `AST::ChoreoFunction`.
  TraceEachVisit(n);
  return true;
}

bool LivenessAnalyzer::Visit(AST::ChoreoFunction& n) {
  TraceEachVisit(n);
  // deal with ChoreoFunction in BeforeVisitImpl due to the orders in
  // accept().
  return true;
}
