
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

LivenessAnalyzer::VarSet LivenessAnalyzer::SetUnion(const VarSet& a,
                                                    const VarSet& b) {
  VarSet ret = a;
  ret.insert(b.begin(), b.end());
  return ret;
}

LivenessAnalyzer::VarSet LivenessAnalyzer::SetDiff(const VarSet& a,
                                                   const VarSet& b) {
  VarSet ret = a;
  for (const auto& item : b) ret.erase(item);
  return ret;
}

LivenessAnalyzer::VarSet
LivenessAnalyzer::GetAllSymbolicOperands(const AST::Node* n) const {
  if (auto id = dyn_cast<AST::Identifier>(n)) {
    return {InScopeName(id->name)};
  } else if (auto expr = dyn_cast<AST::Expr>(n)) {
    VarSet res;
    if (auto c = expr->GetC())
      res = SetUnion(res, GetAllSymbolicOperands(c.get()));
    for (const auto& e : {expr->GetL(), expr->GetR()})
      if (e) res = SetUnion(res, GetAllSymbolicOperands(e.get()));
    return res;
  } else if (isa<AST::IntLiteral>(n) || isa<AST::FloatLiteral>(n) ||
             isa<AST::StringLiteral>(n)) {
    return {};
  } else if (auto ii = dyn_cast<AST::IntIndex>(n)) {
    return GetAllSymbolicOperands(ii->value.get());
  } else if (auto mds = dyn_cast<AST::MultiDimSpans>(n)) {
    if (mds->ref_name != "") {
      return {InScopeName(mds->ref_name)};
    } else {
      auto mv = dyn_cast<AST::MultiValues>(mds->list);
      if (!mv) {
        VST_DEBUG(dbgs() << "the list of mds is not multivalues: " << PSTR(n)
                         << "\n");
        return {};
      }
      VarSet res;
      for (const auto& v : mv->AllValues())
        res = SetUnion(res, GetAllSymbolicOperands(v.get()));
      return res;
    }
  } else if (auto da = dyn_cast<AST::DataAccess>(n)) {
    return {da->GetDataName()};
  } else if (auto it = dyn_cast<AST::IntTuple>(n)) {
    VarSet res;
    for (const auto& v : it->GetValues()->AllValues())
      res = SetUnion(res, GetAllSymbolicOperands(v.get()));
    return res;
  } else if (auto call = dyn_cast<AST::Call>(n)) {
    (void)call;
    return {};
  } else {
    choreo_unreachable("expecting the node to be an Expr or Identifier.");
    return {};
  }
}

inline bool LivenessAnalyzer::IsRef(const AST::Node& n) {
  return n.GetNote().find("ref") != std::string::npos;
}

bool LivenessAnalyzer::HasStmt(const AST::Node& n) const {
  return isa<AST::NamedTypeDecl>(&n) || isa<AST::NamedVariableDecl>(&n) ||
         isa<AST::Assignment>(&n) || isa<AST::DMA>(&n) || isa<AST::Wait>(&n) ||
         isa<AST::Call>(&n) || isa<AST::Rotate>(&n) ||
         isa<AST::Synchronize>(&n) || isa<AST::Trigger>(&n) ||
         isa<AST::Return>(&n) || isa<AST::ParallelBy>(&n) ||
         isa<AST::WithBlock>(&n) || isa<AST::ForeachBlock>(&n) ||
         isa<AST::InThreadsBlock>(&n) || isa<AST::IfElseBlock>(&n) ||
         isa<AST::ChoreoFunction>(&n);
}

inline std::string
LivenessAnalyzer::GetScopedName(const std::string& name) const {
  assert(name != "" && "expecting a valid name.");
  return PrefixedWith(name, "::") ? name : InScopeName(name);
}

// TODO: need to be tested.
inline std::string RemoveWithin(const std::string& s) {
  auto scopes = SplitStringByDelimiter(s, "::");
  std::string ret = "::";
  for (const auto& scope : scopes) {
    if (PrefixedWith(scope, "within")) continue;
    ret += scope + "::";
  }
  return ret;
}

// TODO: `within` is useless when insert extra uses?
int ScopeCompare(const std::string& s1, const std::string& s2) {
  // TODO: is the functionality of `within` correct?
  std::string s1_no_within = RemoveWithin(s1);
  std::string s2_no_within = RemoveWithin(s2);
  // ::foo::A <-> ::foo::A
  if (s1_no_within == s2_no_within) return 0;
  // ::foo::A <-> ::foo::A::B
  if (PrefixedWith(s2_no_within, s1_no_within)) return -1;
  // ::foo::A::B <-> ::foo::A
  return 1;
}

// target: find the first loop between outer_scope and inner_scope
// `::foo::pb::`, `::foo::pb::within0::foreach0::within1::foreach1::`
// => `::foo::pb::within0::foreach0::`
// suppose variable `x` in defined in `::foo::pb::`, and there is a use
// of `x` in `::foo::pb::within0::foreach0::within1::foreach1::`, then
// the liveness of `x` should be extended to the end of
// `::foo::pb::within0::foreach0::` explicitly.
std::string ExactFirstLoopScope(const std::string& outer_scope,
                                const std::string& inner_scope) {
  assert(PrefixedWith(inner_scope, outer_scope) &&
         "expecting the inner scope to be prefixed with the outer scope.");
  auto scopes_outer = SplitStringByDelimiter(outer_scope, "::");
  auto scopes_inner = SplitStringByDelimiter(inner_scope, "::");

  size_t offset = 1;
  size_t so_size = scopes_outer.size();
  size_t si_size = scopes_inner.size();
  while (so_size - 1 + offset < si_size &&
         PrefixedWith(scopes_inner[so_size - 1 + offset], "within_"))
    ++offset;
  auto scopes_ret = std::vector<std::string>(scopes_inner.begin(),
                                             scopes_inner.begin() +
                                                 scopes_outer.size() + offset);
  return "::" + DelimitedString(scopes_ret, "::") + "::";
}

void LivenessAnalyzer::AddUse(const Stmt* s, const std::string& var,
                              bool add_extra_use) {
  std::string svar = GetScopedName(RemoveSuffix(var, ".data"));
  VST_DEBUG(dbgs() << "USE: " << svar << "\n");
  linfo[s].use.insert(svar);
  var_events[svar].push_back({"use", SSTab().ScopeName()});
  if (inthreads_async_level && !visiting_synchronize)
    AddAsyncInthreadsVar(SSTab().ScopeName(), svar);
  // when using a var, its binding should also be treated as used.
  // important when dealing with `AST::Rotate`
  if (Bindings.count(svar)) {
    auto binding_vars = TransitiveClosure({svar}, Bindings);
    for (const auto& binding_var : binding_vars) {
      if (linfo[s].use.count(binding_var)) continue;
      AddUse(s, binding_var, add_extra_use);
    }
  }
  if (fut2buffers.count(svar))
    for (const auto& [src, dst] : fut2buffers[svar])
      AddUse(s, dst, add_extra_use);
  if (!add_extra_use) return;
  // for bounded vars defined in paraby, we add extra uses in other place.
  if (paraby_bounded_vars.count(var)) return;
  // add the extra uses in `scope_end`
  for (const auto& event : var_events.at(svar)) {
    // only consider the def event.
    if (event.first != "def") continue;
    // errs() << "event for var: " << svar << "\n";
    // errs() << "event.second: " << event.second << "\n";
    // errs() << "current scope: " << SSTab().ScopeName() << "\n";
    int res = ScopeCompare(event.second, SSTab().ScopeName());
    // errs() << "res: " << res << "\n";
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
      // errs() << "exact_scope of (" << event.second << ", "
      //           << SSTab().ScopeName() << "): " << exact_scope << "\n";
      if (!events_to_add[scope2stmt.at(exact_scope)].count({"use", svar})) {
        VST_DEBUG({
          dbgs() << "RECORD extra use " << svar << "\n\tin the end of "
                 << stmt2number[scope2stmt.at(exact_scope)] << "\n";
        });
        events_to_add[scope2stmt.at(exact_scope)].insert({"use", svar});
      }
    } else if (res == 0) {
      // no need to add extra use
      break;
    } else {
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
  std::string svar = GetScopedName(var);
  VST_DEBUG(dbgs() << "DEF: " << svar << "\n");
  linfo[s].def.insert(svar);
  var_events[svar].push_back({"def", SSTab().ScopeName()});
  if (is_buffer_or_future) {
    std::string svar_span = svar + ".span";
    VST_DEBUG(dbgs() << "\tand the .span: " << svar_span << "\n");
    linfo[s].def.insert(svar_span);
    var_events[svar_span].push_back({"def", SSTab().ScopeName()});
  }
  VST_DEBUG(dbgs() << "\n");
}

void LivenessAnalyzer::AddIsAlias(const Stmt* s, const std::string& alias_var) {
  std::string salias = GetScopedName(alias_var);
  VST_DEBUG(dbgs() << "AddIsAlias: " << salias << "\n\n");
  assert(linfo[s].name_if_alias == "" && "expecting no alias before.");
  linfo[s].name_if_alias = salias;
}

// y = x.spanas(...), then y is alias to x
void LivenessAnalyzer::AddAlias(const std::string& alias_var,
                                const std::string& original_var) {
  std::string salias = GetScopedName(alias_var);
  std::string soriginal = GetScopedName(original_var);
  VST_DEBUG(dbgs() << "Add alias: " << salias << " <-> " << soriginal
                   << "\n\n");
  Alias[salias] = soriginal;
}

void LivenessAnalyzer::RemoveAlias(const std::string& alias_var) {
  std::string salias = GetScopedName(alias_var);
  assert(Alias.count(salias) &&
         "expecting the alias to be in the alias of current.");
  VST_DEBUG(dbgs() << "Remove alias: " << salias << " <-> " << Alias[salias]
                   << "\n\n");
  Alias.erase(salias);
}

void LivenessAnalyzer::AddIsBinding(const Stmt* s,
                                    const std::string& bind_res) {
  std::string sres = GetScopedName(bind_res);
  VST_DEBUG(dbgs() << "AddIsBinding: " << sres << "\n\n");
  assert(linfo[s].name_if_binding == "" && "expecting no binding before.");
  linfo[s].name_if_binding = sres;
}

// `x = dma.copy y => z`
// then `y` and `z` are bound with `x`
void LivenessAnalyzer::AddBinding(const std::string& bind_res,
                                  const std::string& bind_src) {
  std::string sres = GetScopedName(bind_res);
  std::string ssrc = GetScopedName(bind_src);
  VST_DEBUG(dbgs() << "AddBinding: " << sres << " <- " << ssrc << "\n\n");
  Bindings[sres].insert(ssrc);
}

void LivenessAnalyzer::RemoveBinding(const std::string& bind_res,
                                     const std::string& bind_src) {
  std::string sres = GetScopedName(bind_res);
  std::string ssrc = GetScopedName(bind_src);
  VST_DEBUG(dbgs() << "RemoveBinding: " << sres << " <- " << ssrc << "\n\n");
  Bindings[sres].erase(ssrc);
}

void LivenessAnalyzer::AddFut2Buffers(const std::string& fut,
                                      const BufInfo& buf_info) {
  std::string sfut = GetScopedName(fut);
  std::string ssrc = GetScopedName(buf_info.first);
  std::string sdst = GetScopedName(buf_info.second);
  VST_DEBUG(dbgs() << "AddFut2Buffers: " << sfut << " -> " << ssrc << ", "
                   << sdst << "\n");
  if (fut2buffers[sfut].count({ssrc, sdst})) {
    VST_DEBUG(dbgs() << "\talready exists!\n");
  } else {
    VST_DEBUG(dbgs() << "\tinserted!\n");
    fut2buffers[sfut].insert({ssrc, sdst});
  }
}

inline void
LivenessAnalyzer::AddAsyncInthreadsVar(const std::string& scope_name,
                                       const std::string& var) {
  // VST_DEBUG(dbgs() << "Add async inthreads var: " << var << "\n");
  async_inthreads_vars[scope_name].insert(GetScopedName(var));
}

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

void LivenessAnalyzer::ComputeLiveInOut() {
  for (int i = preorder_stmts.size() - 1; i >= 0; --i) {
    const Stmt* s = preorder_stmts[i];

    if (i < (int)preorder_stmts.size() - 1)
      linfo[s].live_out = linfo[preorder_stmts[i + 1]].live_in;

    VarSet all_use = linfo[s].use;

    linfo[s].live_in =
        SetUnion(linfo[s].use, SetDiff(linfo[s].live_out, linfo[s].def));

    // If x is in use, then Alias[x] and Bindings[x] should also be in use.
    for (const auto& item : linfo[s].use) {
      VarSet alias_tc, binding_tc;
      if (Alias.count(item)) {
        alias_tc = TransitiveClosure({Alias[item]}, Alias, Bindings);
        linfo[s].live_in = SetUnion(linfo[s].live_in, alias_tc);
      }
      if (Bindings.count(item)) {
        binding_tc = TransitiveClosure(Bindings[item], Bindings, Alias);
        linfo[s].live_in = SetUnion(linfo[s].live_in, binding_tc);
      }
      VST_DEBUG({
        if (!alias_tc.empty() || !binding_tc.empty())
          dbgs() << "The tc of use " << item << " in " << SSTR(s);
        if (!alias_tc.empty())
          for (const auto& i : alias_tc) dbgs() << "\t" << i << "\n";
        if (!binding_tc.empty())
          for (const auto& i : binding_tc) dbgs() << "\t" << i << "\n";
      });
    }

    // Restore binding relationship deleted by AST::Wait
    if (stmt2binding_restore.count(s)) {
      assert(isa<AST::Wait>(s));
      std::string fut_name = stmt2binding_restore[s];
      for (const auto& [src, dst] : fut2buffers[fut_name])
        AddBinding(fut_name, src);
    }
  }
  VST_DEBUG({
    auto PrintSet = [](std::ostream& os, const std::string& label,
                       const LivenessAnalyzer::VarSet& vars) {
      os << "\t" << label << ": " << vars.size() << "\n";
      for (const auto& item : vars) os << "\t\t" << item << "\n";
    };
    for (const Stmt* s : preorder_stmts) {
      if (ONLY_SHOW_BUFFER && !linfo[s].buffer_related) continue;
      dbgs() << "stmt: " << SSTR(s);
      PrintSet(dbgs(), "use", linfo[s].use);
      PrintSet(dbgs(), "def", linfo[s].def);
      PrintSet(dbgs(), "live_in", linfo[s].live_in);
      PrintSet(dbgs(), "live_out", linfo[s].live_out);
      dbgs() << "\n";
    }
    PrintSet(dbgs(), "buffers", buffers);
    dbgs() << "\n";
  });
#if 1
  VST_DEBUG({
    if (!linfo[preorder_stmts[0]].live_in.empty()) {
      errs() << SSTR(preorder_stmts[0]);
      errs() << "live_in of the first stmt is not empty, including:\n";
      for (const auto& item : linfo[preorder_stmts[0]].live_in)
        errs() << "\t" << item << "\n";
      choreo_unreachable("expecting the live_in of the first stmt is empty.");
    }
  });
#else
  assert(linfo[preorder_stmts[0]].live_in.empty() &&
         "expecting the live_in of the first stmt is empty.");
#endif
}

// the start and end of the live range are both inclusive.
void LivenessAnalyzer::ComputeLiveRange() {
  // record the def points of each variable.
  std::map<std::string, std::vector<size_t>> var_def_points;

  // collect all the def points of each variable.
  for (const auto* stmt : preorder_stmts)
    for (const auto& var : linfo[stmt].def)
      var_def_points[var].push_back(stmt2number.at(stmt));

  std::vector<std::pair<std::string, Ranges>> var_live_ranges;

  // calculate all the live ranges of each variable.
  for (const auto& [var, def_points] : var_def_points) {
    Ranges ranges;

    // for each def point, find the corresponding live range.
    for (size_t def_point : def_points) {
      // find the last use of the variable after the def point.
      size_t end_point = def_point;

      // traverse from the def point to the end.
      for (size_t i = def_point + 1; i < preorder_stmts.size(); ++i) {
        auto current_stmt = preorder_stmts[i];
        // if the variable is in the live_in or use of the current stmt,
        // update the end_point.
        if (linfo[current_stmt].live_in.count(var) ||
            linfo[current_stmt].use.count(var))
          end_point = i;

        // if the variable is defined in the current stmt, stop the traverse.
        if (linfo[current_stmt].def.count(var) && i != def_point) break;
      }

      // only add the live range if the variable is actually used.
      if (end_point > def_point) ranges.PushBack(Range{def_point, end_point});
    }

    // merge the overlapping ranges.
    ranges.Merge();

    var_live_ranges.push_back(std::make_pair(var, ranges));
    var_ranges.emplace(var, ranges);
  }
  std::sort(var_live_ranges.begin(), var_live_ranges.end(),
            [&](const std::pair<std::string, Ranges>& a,
                const std::pair<std::string, Ranges>& b) {
              return std::tie(var_def_points[a.first].front(), a.first) <
                     std::tie(var_def_points[b.first].front(), b.first);
            });
  VST_DEBUG({
    for (const auto& [var, ranges] : var_live_ranges) {
      dbgs() << (buffers.count(var) ? "BUFFER " : "VAR    ") << var << "\n";
      if (ranges.Values().empty()) {
        dbgs() << "\trange: [" << var_def_points[var].front() << ", x] ";
      } else {
        dbgs() << "\trange:";
        for (const auto& range : ranges.Values())
          dbgs() << " [" << range.start << ", " << range.end << "] ";
      }
      dbgs() << "\n";
    }
  });
}

// n is the NamedVariableDecl or Assignment node.
void LivenessAnalyzer::HandleSelect(AST::Node& n, ptr<AST::Select> sel) {
  std::string name;
  if (auto nvd = dyn_cast<AST::NamedVariableDecl>(&n))
    name = nvd->name_str;
  else if (auto assign = dyn_cast<AST::Assignment>(&n))
    name = assign->GetName();
  else
    assert(false && "expecting a NamedVariableDecl or Assignment node!");
  // if (!isa<FutureType>(NodeType(*sel))) {
  //   dbgs() << "type of " << PSTR(sel) << " is\n\t" << AST::TYPE_STR(*sel)
  //          << "\n";
  // }
  assert(isa<FutureType>(NodeType(*sel)) || isa<SpannedType>(NodeType(*sel)));
  linfo[current_stmt].buffer_related = true;
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
      for (const BufInfo& buf_info : fut2buffers[InScopeName(id->name)])
        AddFut2Buffers(name, buf_info);
    } else if (auto expr = dyn_cast<AST::Expr>(item)) {
      // TODO: actually, there are many situations that the node can be either
      // an identifier or an expression. Need to handle them in other Nodes!
      VarSet ops = GetAllSymbolicOperands(item.get());
      AddUse(current_stmt, ops);
      for (const auto& op : ops) {
        AddBinding(name, op);
        if (fut2buffers.count(op))
          for (const BufInfo& buf_info : fut2buffers[op])
            AddFut2Buffers(name, buf_info);
      }
    } else {
      choreo_unreachable("only expecting an identifier in Select, but got " +
                         PSTR(item));
    }
  }
}

inline std::string LivenessAnalyzer::SSTR(const Stmt* stmt) const {
  return stmt2str.at(stmt);
}

std::string dump_indent = "";

inline void IncrDumpIndent() { dump_indent += "  "; }

inline void DecrDumpIndent() {
  if (dump_indent.size() < 2)
    choreo_unreachable("the indent can not be decreased.");
  dump_indent = dump_indent.substr(2);
}

void LivenessAnalyzer::DumpStmtBriefly(const Stmt& n, std::ostream& os,
                                       bool dump_brace, bool only_else) {
#if DUMP_STMT_WITH_TYPE_INFO
  os << std::left << std::setw(25) << n.TypeNameString();
#endif
  auto num = std::to_string(stmt2number.at(&n));
  if (num.size() < 3) num = std::string(3 - num.size(), ' ') + num;
  os << "(" << num << ") ";
  os << dump_indent;

  // special case for the else scope of if-else block
  if (only_else) {
    os << "else" << (dump_brace ? " {" : "") << "\n";
    return;
  }

  if (const auto ntd = dyn_cast<AST::NamedTypeDecl>(&n)) {
    os << ntd->name_str << " : " << PSTR(ntd->init_expr);
  } else if (const auto nvd = dyn_cast<AST::NamedVariableDecl>(&n)) {
    if (nvd->type && nvd->type->IsUnknown()) {
      GetSymbolType(nvd->name_str)->Print(os);
      os << " " << nvd->name_str;
    } else {
      if (nvd->mem) {
        nvd->mem->Print(os);
        os << " ";
      }
      if (nvd->type)
        nvd->type->Print(os);
      else
        GetSymbolType(nvd->name_str)->Print(os);
      os << " " << nvd->name_str;
      if (nvd->IsArray())
        for (auto d : nvd->array_dims) os << "[" << d << "]";
    }
    if (nvd->init_expr)
      os << " " << nvd->init_str << " " << PSTR(nvd->init_expr);
    else if (nvd->init_value)
      os << " " << nvd->init_str << " {" << PSTR(nvd->init_value) << "}";
  } else if (const auto assign = dyn_cast<AST::Assignment>(&n)) {
    assign->da->Print(os);
    os << " = " << PSTR(assign->value);
  } else if (const auto dma = dyn_cast<AST::DMA>(&n)) {
    if (dma->operation == ".any") {
      os << (dma->future.empty() ? "?" : dma->future);
      os << " = dma.any";
    } else {
      if (dma->future.empty())
        assert(!dma->async && "expecting the dma is not async.");
      else
        os << dma->future << " = ";
      os << "dma" << dma->operation << (dma->async ? ".async" : "");
      os << (dma->config ? " " + PSTR(dma->config) : "") << " ";
      os << STR(dma->from) << " => " << STR(dma->to);
    }
    if (dma->chained) {
      if (dma->chain_to != "") os << ", chain_to " << dma->chain_to;
      if (dma->chain_from != "") os << ", chain_from " << dma->chain_from;
    }
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
    os << "sync." << PSTR(sync->scope);
  } else if (const auto tr = dyn_cast<AST::Trigger>(&n)) {
    os << "trigger ";
    tr->targets->Print(os);
  } else if (const auto ret = dyn_cast<AST::Return>(&n)) {
    os << "return";
    if (ret->value) os << " " << PSTR(ret->value);
  } else if (const auto pb = dyn_cast<AST::ParallelBy>(&n)) {
    os << "parallel ";
    os << pb->bpv->name << " = {";
    pb->cmpt_bpvs->InlinePrint(os);
    os << "} by " << "[";
    pb->cmpt_bounds->InlinePrint(os);
    os << "]";
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
      os << lr->iv->name << "(";
      os << (lr->lbound ? PSTR(lr->lbound) : "") << ":";
      os << (lr->ubound ? PSTR(lr->ubound) : "") << ":";
      os << (IsValidStride(lr->stride) ? std::to_string(lr->stride) : "")
         << ")";
    }
  } else if (const auto itb = dyn_cast<AST::InThreadsBlock>(&n)) {
    os << "inthreads" << (itb->async ? ".async " : " ") << PSTR(itb->pred);
  } else if (const auto ie = dyn_cast<AST::IfElseBlock>(&n)) {
    os << "if ";
    if (!HasStmt(*ie->pred)) {
      ie->pred->Print(os, " ");
    } else {
      os << "(the condition is the next line)";
    }
    // TODO: support else stmts
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
        assert(false && "expecting the parameter has a symbol.");
      }
    }
    os << ")";
  } else if (const auto dummy = dyn_cast<ScopeEnd>(&n)) {
    os << "}";
  } else {
    assert(false && "unexpected stmt type.");
  }

  os << (dump_brace ? " {" : "") << "\n";
}

inline bool IsLoopBlock(const AST::Node& n) {
  return isa<AST::ParallelBy>(&n) || isa<AST::ForeachBlock>(&n);
}

inline bool ShouldIndent(const AST::Node& n) {
  return isa<AST::ParallelBy>(&n) || isa<AST::WithBlock>(&n) ||
         isa<AST::ForeachBlock>(&n) || isa<AST::InThreadsBlock>(&n) ||
         isa<AST::IfElseBlock>(&n) || isa<AST::ChoreoFunction>(&n);
}

void LivenessAnalyzer::HandleStmtInBefore(AST::Node& n) {
  if (!HasStmt(n)) return;

  preorder_stmts.push_back(&n);

  current_stmt = &n;
  stmt2number.emplace(&n, stmt_number);
  ++stmt_number;

  DumpStmtBriefly(n, stmts_with_indent, ShouldIndent(n));
  if (ShouldIndent(n)) {
    IncrDumpIndent();
    scope2stmt.emplace(SSTab().ScopeName(), &n);
  }
  std::stringstream ss;
  DumpStmtBriefly(n, ss, false);
  stmt2str.emplace(&n, ss.str());
#if DUMP_EACH_STMT
  VST_DEBUG(dbgs() << SSTR(&n));
#endif
}

void LivenessAnalyzer::HandleStmtInMid(AST::Node& n) {
  if (!HasStmt(n)) return;

  auto ie = dyn_cast<AST::IfElseBlock>(&n);
  if (!ie) return;

  // if there is no else block,
  // do the rbrace job in HandleStmtInAfter as usual.
  if (!ie->HasElse()) return;

  // complete the end of `if scope`     (rbrace)
  // process the start of `end scope`   (lbrace)

  DecrDumpIndent();

  auto if_end = AST::Make<ScopeEnd>(n.LOC(), &n);
  scope_ends.push_back(if_end);
  preorder_stmts.push_back(if_end.get());
  stmt2number.emplace(if_end.get(), stmt_number);
  ++stmt_number;
  DumpStmtBriefly(*if_end, stmts_with_indent, false);

  std::stringstream ss;
  DumpStmtBriefly(*if_end, ss, false);
  stmt2str.emplace(if_end.get(), ss.str());
#if DUMP_EACH_STMT
  VST_DEBUG(dbgs() << SSTR(if_end.get()));
#endif

  auto else_start = AST::Make<ScopeEnd>(n.LOC(), &n);
  scope_ends.push_back(else_start);
  preorder_stmts.push_back(else_start.get());
  stmt2number.emplace(else_start.get(), stmt_number);
  ++stmt_number;
  DumpStmtBriefly(*else_start, stmts_with_indent, true, true);

  IncrDumpIndent();
  scope2stmt.emplace(SSTab().ScopeName(), else_start.get());

  ss.str("");
  DumpStmtBriefly(*else_start, ss, false);
  stmt2str.emplace(else_start.get(), ss.str());
#if DUMP_EACH_STMT
  VST_DEBUG(dbgs() << SSTR(else_start.get()));
#endif
}

void LivenessAnalyzer::HandleStmtInAfter(AST::Node& n) {
  if (!HasStmt(n)) return;

  // only after-handle stmts which need indent
  if (!ShouldIndent(n)) return;

  DecrDumpIndent();

  auto rbrace = AST::Make<ScopeEnd>(n.LOC(), &n);

  scope_ends.push_back(rbrace);
  preorder_stmts.push_back(rbrace.get());
  stmt2number.emplace(rbrace.get(), stmt_number);
  ++stmt_number;

  DumpStmtBriefly(*rbrace, stmts_with_indent, false);

  if (IsLoopBlock(n)) {
    if (events_to_add.count(&n)) {
      for (const auto& [event_type, var] : events_to_add[&n]) {
        if (event_type != "use") assert(false && "unexpected event type.");
        VST_DEBUG(dbgs() << "EXTRA use: " << var << " in "
                         << stmt2number[rbrace.get()] << "\n";);
        AddUse(rbrace.get(), var, false);
      }
    }
  }
  std::stringstream ss;
  DumpStmtBriefly(*rbrace, ss, false);
  stmt2str.emplace(rbrace.get(), ss.str());
#if DUMP_EACH_STMT
  VST_DEBUG(dbgs() << SSTR(rbrace.get()));
#endif
}

bool LivenessAnalyzer::BeforeVisitImpl(AST::Node& n) {
  HandleStmtInBefore(n);

  if (auto cf = dyn_cast<AST::ChoreoFunction>(&n)) {
    for (const auto& param : cf->f_decl.params->values) {
      if (!param->HasSymbol()) continue;
      std::string sname = InScopeName(param->sym->name);
      if (auto sty = dyn_cast<SpannedType>(param->GetType())) {
        linfo[current_stmt].buffer_related = true;
        buffers.insert(sname);
        if (sty->RuntimeShaped()) {
          auto shape = sty->GetShape();
          for (const auto& [_, v] : shape.GetDynamicDims())
            AddDef(current_stmt, v);
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

bool LivenessAnalyzer::AfterVisitImpl(AST::Node& n) {
  HandleStmtInAfter(n);

  if (isa<AST::Program>(&n)) {
    VST_DEBUG(dbgs() << "\n" << stmts_with_indent.str() << "\n");
    ComputeLiveInOut();
    ComputeLiveRange();
  } else if (auto w = dyn_cast<AST::Wait>(&n)) {
    auto FuturesOf = [&](const AST::Wait& n) {
      std::vector<ptr<AST::Node>> ret;
      for (auto item : n.GetTargets())
        if (isa<FutureType>(NodeType(*item))) ret.push_back(item);
      return ret;
    };
    // after wait node, the async dma is done, remove the binding
    for (const auto& f : FuturesOf(*w)) {
      auto id = AST::GetIdentifier(*f);
      assert(id && "expecting an identifier in Wait.");
      auto fut_name = InScopeName(id->name);
      assert(fut2buffers.count(fut_name) &&
             "expecting the future to be in fut2buffers.");
      for (const auto& [src, dst] : fut2buffers[fut_name])
        RemoveBinding(fut_name, src);
      // since we will calculate live_in and live_out after visiting all the
      // nodes, we should record the binding info to do restoration in
      // ComputeLiveInOut().
      stmt2binding_restore[&n] = fut_name;
    }
  } else if (isa<AST::InThreadsBlock>(&n)) {
    --inthreads_async_level;
  }

  return true;
}

bool LivenessAnalyzer::Visit(AST::NamedTypeDecl& n) {
  TraceEachVisit(n);
  // TODO: handle the case of named type decl
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
  if (isa<ScalarType>(ty)) {
    AddDef(current_stmt, n.name_str);
  } else if (isa<StringType>(ty)) {
    // TODO: handle the case of string type
    assert(false && "not implemented yet.");
  } else if (isa<IndexType>(ty)) {
    // TODO: handle the case of index type
    assert(false && "not implemented yet.");
  } else if (isa<ITupleType>(ty)) {
    AddDef(current_stmt, n.name_str);
    // TODO: arith between ituple?
    AddUse(current_stmt, GetAllSymbolicOperands(n.init_expr.get()));
  } else if (isa<MDSpanType>(ty)) {
    // TODO: handle the case of mdspan type
    assert(false && "not implemented yet.");
  } else if (auto sty = dyn_cast<SpannedType>(ty)) {
    linfo[current_stmt].buffer_related = true;
    if (!IsRef(n)) {
      AddDef(current_stmt, n.name_str, true);
      buffers.insert(InScopeName(n.name_str));
    } else {
      VST_DEBUG(dbgs() << "The nvd is a reference: " << STR(n) << ".\n\n");
      assert(n.init_expr && "expecting the init_expr is not nullptr.");
      if (auto e = dyn_cast<AST::Expr>(n.init_expr)) {
        if (auto sa = dyn_cast<AST::SpanAs>(e->GetR())) {
          AddDef(current_stmt, n.name_str, true);
          AddUse(current_stmt, sa->id->name);
          AddAlias(n.name_str, sa->id->name);
        } else {
          assert(false && "expecting the init_expr is a span_as.");
        }
      } else {
        assert(false && "expecting the init_expr is an expr.");
      }
    }
  } else if (isa<BoundedType>(ty)) {
    AddDef(current_stmt, n.name_str);
    AddUse(current_stmt, GetAllSymbolicOperands(n.init_expr.get()));
  } else if (isa<FutureType>(ty)) {
    // TODO: handle the case of future type
    assert(false && "not implemented yet.");
  } else if (isa<EventType>(ty)) {
    AddDef(current_stmt, n.name_str);
  } else {
    assert(false && "expecting the type is spanned, scalar, string, index, "
                    "ituple, or mdspan.");
  }
  return true;
}

bool LivenessAnalyzer::Visit(AST::Assignment& n) {
  TraceEachVisit(n);

  if (auto sel = dyn_cast<AST::Select>(n.value)) {
    // TODO: could be a select with non-span vars?
    HandleSelect(n, sel);
  } else if (auto sa = dyn_cast<AST::SpanAs>(n.value)) {
    assert(IsRef(n) && "expecting the spanas assignment is a reference.");
    linfo[current_stmt].buffer_related = true;
    AddDef(current_stmt, n.GetName(), true);
    AddUse(current_stmt, sa->id->name);
    AddIsAlias(current_stmt, n.GetName());
    AddAlias(n.GetName(), sa->id->name);
  } else {
    assert(!IsRef(n) && "expecting the assignment is not a reference.");
    VST_DEBUG(dbgs() << "The assignment is not sel or sa: " << STR(n)
                     << ".\n\n");
    if (n.AssignToDataElement())
      AddUse(current_stmt, n.GetDataArrayName());
    else
      AddDef(current_stmt, n.GetName());
    if (auto expr = dyn_cast<AST::Expr>(n.value)) {
      VarSet operands = GetAllSymbolicOperands(expr.get());
      AddUse(current_stmt, operands);
    } else {
      assert(false && "expecting the assignment value is an expr.");
    }
  }
  return true;
}

bool LivenessAnalyzer::Visit(AST::ParallelBy& n) {
  TraceEachVisit(n);
  assert(n.bpv && n.cmpt_bpvs &&
         "expecting the parallel-by has bpv and cmpt_bpvs.");
  AddDef(current_stmt, n.bpv->name);
  events_to_add[current_stmt].insert({"use", n.bpv->name});
  paraby_bounded_vars.insert(n.bpv->name);
  for (const auto& iv_symbol : n.cmpt_bpvs->AllValues()) {
    std::string iv_symbol_name = cast<AST::Identifier>(iv_symbol)->name;
    AddDef(current_stmt, iv_symbol_name);
    events_to_add[current_stmt].insert({"use", iv_symbol_name});
    paraby_bounded_vars.insert(iv_symbol_name);
    AddBinding(n.bpv->name, iv_symbol_name);
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
      // special case: if a symbol in where is def in with or with_matchers,
      // ignore it. Because it is a use that after define.
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
  linfo[current_stmt].buffer_related = true;
  /*
  If the dma is sync, then only the dst buffer is alias to the future.
  The src buffer can be reused immediately after the dma done.

  If the dma is async, then the future is alias to the src buffer.
  The corresponding src buffer can be reused only after the future has been
  waited.
  */
  if (n.future.empty()) {
    assert(!n.async && "async dma should have a future.");
    AddUse(current_stmt, n.FromSymbol());
    AddUse(current_stmt, n.ToSymbol());
  } else {
    if (n.operation == ".any") {
      AddDef(current_stmt, n.future, true);
      dma_any.insert(InScopeName(n.future));
      return true;
    }

    if (dma_any.count(InScopeName(n.future)))
      AddUse(current_stmt, n.future);
    else
      AddDef(current_stmt, n.future, true);

    AddUse(current_stmt, n.FromSymbol());
    AddUse(current_stmt, n.ToSymbol());

    AddIsBinding(current_stmt, n.future);
    if (n.async) AddBinding(n.future, n.FromSymbol());
    AddBinding(n.future, n.ToSymbol());
    AddFut2Buffers(n.future, BufInfo{n.FromSymbol(), n.ToSymbol()});
  }

  if (n.chained && n.chain_from != "") AddUse(current_stmt, n.chain_from);
  return true;
}

bool LivenessAnalyzer::Visit(AST::ChunkAt& n) {
  TraceEachVisit(n);
  assert(n.sa == nullptr && "after norm, there should be no span_as.");
  // `n.data` is already handled in Visit(AST::DMA& n)
  for (auto tsi : n.AllTSInfo())
    for (const auto& pos : tsi->GetIndices()) {
      VST_DEBUG(dbgs() << "chunkat position: " << PSTR(pos) << ".\n");
      if (auto expr = dyn_cast<AST::Expr>(pos)) {
        VarSet operands = GetAllSymbolicOperands(expr.get());
        AddUse(current_stmt, operands);
      } else if (auto id = dyn_cast<AST::Identifier>(pos)) {
        // ignore the __choreo_no_tiling__
        if (id->name == "__choreo_no_tiling__") continue;
        AddUse(current_stmt, id->name);
      } else {
        assert(false && "expecting the chunkat position is an expr.");
      }
    }
  return true;
}

bool LivenessAnalyzer::Visit(AST::Wait& n) {
  TraceEachVisit(n);
  for (const auto& item : n.GetTargets()) {
    if (isa<FutureType>(NodeType(*item))) {
      // future
      linfo[current_stmt].buffer_related = true;
      auto id = AST::GetIdentifier(*item);
      assert(id && "expecting an identifier in Wait.");
      AddUse(current_stmt, id->name);
      const std::string sname = InScopeName(id->name);
      if (!fut2buffers.count(sname))
        assert(dma_any.count(sname) &&
               "expecting the future to be dma_any if not in fut2buffers.");
      for (const auto& [src, dst] : fut2buffers[sname]) {
        AddUse(current_stmt, src);
        AddUse(current_stmt, dst);
      }
    } else {
      // event
      auto expr = dyn_cast<AST::Expr>(item);
      assert(IsSymbolOrArrayRef(*item) &&
             "expect either symbol or array reference.");
      bool is_array_ref = (expr->op == "elemof");
      std::string name;
      if (is_array_ref)
        name = AST::GetArrayBaseSymbol(*expr)->name;
      else
        name = AST::GetIdentifier(*item)->name;
      AddUse(&n, name);
    }
  }
  return true;
}

bool LivenessAnalyzer::Visit(AST::Call& n) {
  TraceEachVisit(n);
  linfo[current_stmt].buffer_related = true;
  for (const auto& arg : n.GetArguments()) {
    auto sty = GetSpannedType(NodeType(*arg));
    if (sty) {
      if (auto id = AST::GetIdentifier(*arg)) {
        AddUse(current_stmt, id->name);
      } else if (auto expr = dyn_cast<AST::Expr>(arg)) {
        if (expr->op == "dataof") {
          // TODO: will only the dims of future be used?
          assert(isa<FutureType>(expr->GetR()->GetType()) &&
                 "expect a future operand.");
          if (auto id = cast<AST::Expr>(expr->GetR())->GetSymbol())
            AddUse(current_stmt, id->name);
          else
            choreo_unreachable("Can not retrieve name of the future.");
        } else if (expr->op == "addrof") {
          if (auto id = AST::GetIdentifier(expr->GetR()))
            AddUse(current_stmt, id->name);
          else if (!isa<AST::DataAccess>(expr->GetR()))
            choreo_unreachable("Can not retrieve name of the future.");
        } else {
          assert(isa<AST::ChunkAt>(expr->GetR()) &&
                 "expect a chunkat operand.");
        }
      }
    } else {
      if (auto expr = dyn_cast<AST::Expr>(arg)) {
        VarSet operands = GetAllSymbolicOperands(expr.get());
        AddUse(current_stmt, operands);
      } else if (isa<AST::StringLiteral>(arg)) {
      } else {
        VST_DEBUG({
          dbgs() << "the argument is neither expr nor string literal: "
                 << STR(*arg) << ".\n";
          dbgs() << "its type is: " << AST::TYPE_STR(*arg) << ".\n";
        });
        assert(false && "expecting the argument is an expr.");
      }
    }
  }
  return true;
}

bool LivenessAnalyzer::Visit(AST::Rotate& n) {
  TraceEachVisit(n);
  linfo[current_stmt].buffer_related = true;
  VarSet uses;
  for (const auto& item : n.GetIds()) {
    assert(isa<AST::Identifier>(item));
    auto id = cast<AST::Identifier>(item);
    auto sname = InScopeName(id->name);
    uses.insert(sname);
    for (const auto& [src, dst] : fut2buffers[sname]) {
      uses.insert(src);
      uses.insert(dst);
    }
    // Each future in a Rotate is bound to every other future in the Rotate
    for (const auto& other : n.GetIds()) {
      auto other_id = cast<AST::Identifier>(other);
      auto other_sname = InScopeName(other_id->name);
      if (other_sname == sname) continue;
      for (const BufInfo& buf_info : fut2buffers[other_sname])
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
  for (const auto& e : n.GetEvents()) {
    auto expr = dyn_cast<AST::Expr>(e);
    assert(IsSymbolOrArrayRef(*e) &&
           "expect either symbol or array reference.");
    bool is_array_ref = (expr->op == "elemof");
    std::string name;
    if (is_array_ref)
      name = AST::GetArrayBaseSymbol(*expr)->name;
    else
      name = AST::GetIdentifier(*e)->name;
    AddUse(&n, name);
  }
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
    if (auto id = AST::GetIdentifier(*n.value)) {
      linfo[current_stmt].buffer_related = true;
      AddUse(current_stmt, id->name);
    } else if (auto expr = dyn_cast<AST::Expr>(n.value);
               expr && expr->op == "dataof") {
      linfo[current_stmt].buffer_related = true;
      auto id = cast<AST::Expr>(expr->GetR())->GetSymbol();
      assert(id && "expect a symbol");
      AddUse(current_stmt, id->name);
    } else {
      assert(false &&
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
    // Although `iv` is reset to zero, still treat it as a use.
    AddUse(current_stmt, range->IVName());
    for (const auto& offset : {range->lbound, range->ubound}) {
      if (!offset) continue;
      if (auto id = AST::GetIdentifier(*offset))
        AddUse(current_stmt, id->name);
      else
        assert(false &&
               "expecting the bound offset in LoopRange is an Identifier.");
    }
  }
  return true;
}

bool LivenessAnalyzer::Visit(AST::InThreadsBlock& n) {
  TraceEachVisit(n);
  AddUse(current_stmt, GetAllSymbolicOperands(n.pred.get()));
  return true;
}

bool LivenessAnalyzer::Visit(AST::IfElseBlock& n) {
  TraceEachVisit(n);
  VarSet operands = GetAllSymbolicOperands(n.pred.get());
  AddUse(current_stmt, operands);
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

bool LivenessAnalyzer::HasError() {
  if (error_count > 0) {
    dbgs() << "Totally " << error_count << " errors have been detected.\n";
    return true;
  }
  return false;
}
