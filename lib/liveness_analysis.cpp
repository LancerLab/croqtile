
#include "liveness_analysis.hpp"
#include "ast.hpp"
#include "aux.hpp"
#include "types.hpp"
#include "visitor.hpp"

using namespace Choreo;

#define DUMP_EACH_STMT true
#define DUMP_STMT_WITH_TYPE_INFO false
#define IGNORE_GLOBAL_BUFFER true
#define ONLY_SHOW_BUFFER false

LivenessAnalyzer::VarSet LivenessAnalyzer::SetUnion(const VarSet& a,
                                                    const VarSet& b) const {
  VarSet ret = a;
  ret.insert(b.begin(), b.end());
  return ret;
}

LivenessAnalyzer::VarSet LivenessAnalyzer::SetDiff(const VarSet& a,
                                                   const VarSet& b) const {
  VarSet ret = a;
  for (const auto& item : b) { ret.erase(item); }
  return ret;
}

LivenessAnalyzer::VarSet
LivenessAnalyzer::GetAllSymbolicOperands(AST::Node* n) const {
  auto expr = dyn_cast<AST::Expr>(n);
  if (!expr) { return {}; }
  if (auto ref = expr->GetReference()) {
    if (auto id = expr->GetSymbol()) {
      // TODO(opt): actually, using xxx.span will only care about the span, not
      // the data, so the data of the future is reusable if feasible.
      return {InScopeName(RemoveSuffix(id->name, ".span"))};
    } else {
      if (isa<AST::IntLiteral>(ref)) {
      } else {
        VST_DEBUG(std::cerr << PSTR(ref) << "\n";
                  std::cerr << AST::TYPE_STR(*ref) << "\n";);
        assert(false &&
               "expecting the symbol of the reference is an identifier.");
      }
    }
  }
  VarSet ret;
  if (auto c = expr->GetC()) {
    ret = SetUnion(ret, GetAllSymbolicOperands(c.get()));
  }
  if (auto l = expr->GetL()) {
    ret = SetUnion(ret, GetAllSymbolicOperands(l.get()));
  }
  if (auto r = expr->GetR()) {
    ret = SetUnion(ret, GetAllSymbolicOperands(r.get()));
  }
  return ret;
}

inline bool LivenessAnalyzer::IsRef(const AST::Node& n) const {
  return n.GetNote().find("ref") != std::string::npos;
}

bool LivenessAnalyzer::IsStmt(const AST::Node& n) const {
  return isa<AST::NamedTypeDecl>(&n) || isa<AST::NamedVariableDecl>(&n) ||
         isa<AST::Assignment>(&n) || isa<AST::DMA>(&n) || isa<AST::Wait>(&n) ||
         isa<AST::Call>(&n) || isa<AST::Rotate>(&n) || isa<AST::Return>(&n) ||
         isa<AST::ParallelBy>(&n) || isa<AST::WithBlock>(&n) ||
         isa<AST::ForeachBlock>(&n) || isa<AST::ChoreoFunction>(&n);
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
    if (PrefixedWith(scope, "within")) { continue; }
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
std::string ExactScope(const std::string& outer_scope,
                       const std::string& inner_scope) {
  assert(PrefixedWith(inner_scope, outer_scope) &&
         "expecting the inner scope to be prefixed with the outer scope.");
  auto scopes_outer = SplitStringByDelimiter(outer_scope, "::");
  auto scopes_inner = SplitStringByDelimiter(inner_scope, "::");

  size_t offset = 1;
  size_t so_size = scopes_outer.size();
  size_t si_size = scopes_inner.size();
  while (so_size - 1 + offset < si_size &&
         PrefixedWith(scopes_inner[so_size - 1 + offset], "within_")) {
    ++offset;
  }
  auto scopes_ret = std::vector<std::string>(scopes_inner.begin(),
                                             scopes_inner.begin() +
                                                 scopes_outer.size() + offset);
  // std::cerr << "outer scope: " << outer_scope << "\n";
  // std::cerr << "inner scope: " << inner_scope << "\n";
  // std::cerr << "scopes_ret: " << DelimitedString(scopes_ret, "::") << "\n";
  return "::" + DelimitedString(scopes_ret, "::") + "::";
}

void LivenessAnalyzer::AddUse(const Stmt* s, const std::string& var,
                              bool is_future, bool add_extra_use) {
  std::string svar = GetScopedName(var);
  VST_DEBUG(dbgs() << "use: " << svar << "\n\n");
  linfo[s].use.insert(svar);
  var_events[svar].push_back({"use", SSTab().ScopeName()});
  // when using a var, its binding should also be treated as used.
  // important when dealing with `AST::Rotate`
  if (Bindings.count(svar)) {
    auto binding_vars = TransitiveClosure({svar}, Bindings);
    for (const auto& binding_var : binding_vars) {
      if (linfo[s].use.count(binding_var)) { continue; }
      AddUse(s, binding_var);
    }
  }
  // TODO: Using a future means that the corresponding dst buffer is being used.
  if (is_future) {
    for (const auto& [src, dst] : fut2buffers[svar]) { AddUse(s, dst); }
  }
  if (!add_extra_use) return;
  // for bounded vars which are defined in paraby, we add extra uses in other
  // place.
  if (paraby_bounded_vars.count(var)) return;
  // add the extra uses in `scope_end`
  for (const auto& event : var_events[svar]) {
    // only consider the def event.
    if (event.first != "def") { continue; }
    int res = ScopeCompare(event.second, SSTab().ScopeName());
    if (res < 0) {
      std::string exact_scope = ExactScope(event.second, SSTab().ScopeName());
      if (!events_to_add[scope2stmt.at(exact_scope)].count({"use", svar})) {
        VST_DEBUG(std::cerr << "outer scope: " << event.second << "\n";
                  std::cerr << "inner scope: " << SSTab().ScopeName() << "\n";
                  std::cerr << "exact scope: " << exact_scope << "\n";
                  dbgs() << "record extra use: " << svar << "\n\tin "
                         << exact_scope << "\n";);
        events_to_add[scope2stmt.at(exact_scope)].insert({"use", svar});
      }
    } else if (res == 0) {
      break;
    } else {
      assert(
          false &&
          "expecting the scope of the event is less than the current scope.");
    }
  }
}

void LivenessAnalyzer::AddUse(const Stmt* s, const VarSet& vars, bool is_future,
                              bool add_extra_use) {
  for (const auto& var : vars) { AddUse(s, var, is_future, add_extra_use); }
}

void LivenessAnalyzer::AddDef(const Stmt* s, const std::string& var,
                              bool is_buffer) {
  std::string svar = GetScopedName(var);
  VST_DEBUG(dbgs() << "def: " << svar << "\n");
  linfo[s].def.insert(svar);
  var_events[svar].push_back({"def", SSTab().ScopeName()});
  if (is_buffer) {
    VST_DEBUG(dbgs() << "\tis buffer, size: "
                     << (buf_sizes.count(svar)
                             ? std::to_string(buf_sizes.at(svar))
                             : "runtime shaped")
                     << "\n\n");
  } else {
    VST_DEBUG(dbgs() << "\n");
  }
}

void LivenessAnalyzer::AddBufStmt(const Stmt* s, Storage sto) {
  auto nvd = dyn_cast<AST::NamedVariableDecl>(s);
  assert(nvd && "expecting NamedVariableDecl node as stmt!");
  std::string sbuf = GetScopedName(nvd->name_str);
  buffers.insert(sbuf);
  buf_nodes[sto].insert(nvd);
  if (sto == Storage::GLOBAL) {
    global_buffers.insert(sbuf);
  } else if (sto == Storage::SHARED) {
    shared_buffers.insert(sbuf);
  } else if (sto == Storage::LOCAL) {
    local_buffers.insert(sbuf);
  } else {
    assert(false && "expecting the storage is SHARED, LOCAL or GLOBAL!");
  }
}

// y = x.spanas(...), then y is alias to x
void LivenessAnalyzer::AddAlias(const std::string& alias_var,
                                const std::string& original_var) {
  std::string salias = GetScopedName(alias_var);
  std::string soriginal = GetScopedName(original_var);
  VST_DEBUG(dbgs() << "Add alias: " << salias << " <-> " << soriginal
                   << "\n\n");
// TODO: should we record the first var as alias or the last one?
#if 0
  if (Alias.count(salias)) {
    // alias to the first var
    Alias[salias] = Alias[soriginal];
  } else {
    Alias[salias] = soriginal;
  }
#else
  Alias[salias] = soriginal;
#endif
}

void LivenessAnalyzer::RemoveAlias(const std::string& alias_var) {
  std::string salias = GetScopedName(alias_var);
  assert(Alias.count(salias) &&
         "expecting the alias to be in the alias of current.");
  VST_DEBUG(dbgs() << "Remove alias: " << salias << " <-> " << Alias[salias]
                   << "\n\n");
  Alias.erase(salias);
}

void LivenessAnalyzer::AddIsAlias(const Stmt* s, const std::string& alias_var) {
  std::string salias = GetScopedName(alias_var);
  VST_DEBUG(dbgs() << "AddIsAlias: " << salias << "\n\n");
  assert(linfo[s].name_if_alias == "" && "expecting no alias before.");
  linfo[s].name_if_alias = salias;
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
  // TODO:
  // assert(Bindings[sres].count(ssrc) &&
  //  "expecting the bind src to be in the bindings of res.");
  Bindings[sres].erase(ssrc);
}

void LivenessAnalyzer::AddFut2Buffers(const std::string& fut,
                                      const std::string& src,
                                      const std::string& dst) {
  std::string sfut = GetScopedName(fut);
  std::string ssrc = GetScopedName(src);
  std::string sdst = GetScopedName(dst);
  VST_DEBUG(dbgs() << "AddFut2Buffers: " << sfut << " -> " << ssrc << ", "
                   << sdst << "\n");
  if (fut2buffers[sfut].count({ssrc, sdst})) {
    VST_DEBUG(dbgs() << "\talready exists!\n");
  } else {
    VST_DEBUG(dbgs() << "\tinserted!\n");
    fut2buffers[sfut].insert({ssrc, sdst});
  }
}

template <typename MapType>
void ProcessMap(const std::string& current, const MapType& mp,
                LivenessAnalyzer::VarSet& result,
                LivenessAnalyzer::VarSet& processed,
                std::queue<std::string>& queue) {
  if (!mp.count(current)) { return; }
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
    for (const auto& next_var : next) {
      if (!processed.count(next_var)) {
        result.insert(next_var);
        queue.push(next_var);
      }
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
    if (processed.count(current)) { continue; }
    processed.insert(current);

    // 使用折叠表达式处理每个map
    (ProcessMap(current, maps, result, processed, queue), ...);
  }
  return result;
}

void LivenessAnalyzer::ComputeLiveInOut() {
  for (int i = stmts_preordered.size() - 1; i >= 0; --i) {
    const Stmt* s = stmts_preordered[i];

    if (i < (int)stmts_preordered.size() - 1)
      linfo[s].live_out = linfo[stmts_preordered[i + 1]].live_in;

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
      // clang-format off
      VST_DEBUG(
        if (!alias_tc.empty() || !binding_tc.empty())
          dbgs() << "The tc of use " << item << " in " << SSTR(s);
        if (!alias_tc.empty()) {
          for (const auto& i : alias_tc) dbgs() << "\t" << i << "\n";
        }
        if (!binding_tc.empty()) {
          for (const auto& i : binding_tc) dbgs() << "\t" << i << "\n";
        }
      );
      // clang-format on
    }

    // Restore binding relationship deleted by AST::Wait
    if (stmt2binding_restore.count(s)) {
      assert(isa<AST::Wait>(s));
      std::string fut_name = stmt2binding_restore[s];
      for (const auto& [src, dst] : fut2buffers[fut_name]) {
        AddBinding(fut_name, src);
      }
    }
  }
#if 1
  VST_DEBUG(if (!linfo[stmts_preordered[0]].live_in.empty()) {
    std::cerr << "live_in of the first stmt is not empty.\n";
    for (const auto& item : linfo[stmts_preordered[0]].live_in) {
      std::cerr << "\t" << item << "\n";
    }
  });
#else
  assert(linfo[stmts_preordered[0]].live_in.empty() &&
         "expecting the live_in of the first stmt is empty.");
#endif
  // clang-format off
  VST_DEBUG(
    for (const Stmt* s : stmts_preordered) {
      if (ONLY_SHOW_BUFFER && !linfo[s].buffer_related) continue;
      dbgs() << "stmt: " << SSTR(s);
      dbgs() << "\tuse: " << linfo[s].use.size() << "\n";
      for (const auto& item : linfo[s].use)
        dbgs() << "\t\t" << item << "\n";
      dbgs() << "\tdef: " << linfo[s].def.size() << "\n";
      for (const auto& item : linfo[s].def)
        dbgs() << "\t\t" << item << "\n";
      dbgs() << "\tlive_in: " << linfo[s].live_in.size() << "\n";
      for (const auto& item : linfo[s].live_in)
        dbgs() << "\t\t" << item << "\n";
      dbgs() << "\tlive_out: " << linfo[s].live_out.size() << "\n";
      for (const auto& item : linfo[s].live_out)
        dbgs() << "\t\t" << item << "\n";
      dbgs() << "\n";
    }
    dbgs() << "buffers: " << buffers.size() << "\n"; 
    for (const auto& item : buffers)
      dbgs() << "\t" << item << "\n";
    dbgs() << "\n";
  );
  // clang-format on
}

// TODO: need to be tested.
// the start and end of the live range are both inclusive.
void LivenessAnalyzer::ComputeLiveRange() {
  // record the def points of each variable.
  std::map<std::string, std::vector<size_t>> var_def_points;

  // collect all the def points of each variable.
  for (const auto* stmt : stmts_preordered) {
    for (const auto& var : linfo[stmt].def) {
      var_def_points[var].push_back(stmt2number.at(stmt));
    }
  }

  // calculate all the live ranges of each variable.
  for (const auto& [var, def_points] : var_def_points) {
    std::vector<Range> ranges;

    // for each def point, find the corresponding live range.
    for (size_t def_point : def_points) {
      // find the last use of the variable after the def point.
      size_t end_point = def_point;

      // traverse from the def point to the end.
      for (size_t i = def_point + 1; i < stmts_preordered.size(); ++i) {
        auto current_stmt = stmts_preordered[i];
        // if the variable is in the live_in or use of the current stmt,
        // update the end_point.
        if (linfo[current_stmt].live_in.count(var) ||
            linfo[current_stmt].use.count(var)) {
          end_point = i;
        }

        // if the variable is defined in the current stmt, stop the traverse.
        if (linfo[current_stmt].def.count(var) && i != def_point) { break; }
      }

      // only add the live range if the variable is actually used.
      if (end_point > def_point) { ranges.push_back({def_point, end_point}); }
    }

    // merge the overlapping ranges.
    if (!ranges.empty()) {
      std::sort(
          ranges.begin(), ranges.end(),
          [](const Range& a, const Range& b) { return a.start < b.start; });

      std::vector<Range> merged_ranges;
      merged_ranges.push_back(ranges[0]);

      for (size_t i = 1; i < ranges.size(); ++i) {
        Range& last = merged_ranges.back();

        // if the current range overlaps with the last range, merge them.
        if (ranges[i].start <= last.end) {
          last.end = std::max(last.end, ranges[i].end);
        } else {
          // otherwise, add a new range.
          merged_ranges.push_back(ranges[i]);
        }
      }

      // store the merged live ranges.
      var_ranges[var] = merged_ranges;

      // clang-format off
      VST_DEBUG(
        #if IGNORE_GLOBAL_BUFFER
        if (global_buffers.count(var)) {
          dbgs() << "IGNORE global buffer: " << var << "\n";
          continue;
        }
        if (Alias.count(var) && global_buffers.count(Alias[var])) {
          dbgs() << "IGNORE var: " << var << "\n\twhich is an alias of "
                 << Alias[var] << "\n";
          continue;
        }
        #endif
        dbgs() << (buffers.count(var) ? "BUFFER " : "VAR    ") << var << "\n";
        for (const auto& range : merged_ranges) {
          dbgs() << "\trange: [" << range.start << ", " << range.end << "]\n";
        }
      );
      // clang-format on
    }
  }
}

// n is the NamedVariableDecl or Assignment node.
void LivenessAnalyzer::HandleSelect(AST::Node& n, ptr<AST::Select> sel) {
  std::string name;
  if (auto nvd = dyn_cast<AST::NamedVariableDecl>(&n)) {
    name = nvd->name_str;
  } else if (auto assign = dyn_cast<AST::Assignment>(&n)) {
    name = assign->name;
  } else {
    assert(false && "expecting a NamedVariableDecl or Assignment node!");
  }
  assert(isa<FutureType>(NodeType(*sel)) &&
         "expecting a future type select node!");
  linfo[current_stmt].buffer_related = true;
  AddDef(current_stmt, name);
  AddIsBinding(current_stmt, name);
  for (const auto& item : sel->expr_list->AllValues()) {
    auto id = AST::GetIdentifier(*item);
    if (!id)
      choreo_unreachable("only expecting an identifier in Select, but got " +
                         PSTR(item));
    auto fut_name = InScopeName(id->name);
    assert(fut2buffers.count(fut_name) &&
           "expecting the future to be in fut2buffers.");
    assert(FBInfo().count(fut_name) &&
           "expecting the future to be in FBInfo! Currently only support "
           "future in select op(thus, liveness analysis should be after the "
           "check or do the check!).");
    AddUse(current_stmt, fut_name, true);
    // Bind name with fut_name
    // because fut_name has been bound with the src and dst.
    // And in co code, the future is used directly later.
    AddBinding(name, fut_name);
  }
}

std::string LivenessAnalyzer::SSTR(const Stmt* stmt) const {
  return stmt2str.at(stmt);
}

std::string DUMP_INDENT = "";

void LivenessAnalyzer::DumpStmtBriefly(const Stmt& n, std::ostream& os,
                                       bool dump_brace) {
#if DUMP_STMT_WITH_TYPE_INFO
  os << std::left << std::setw(20) << n.TypeNameString();
#endif
  auto num = std::to_string(stmt2number.at(&n));
  if (num.size() < 3) num = std::string(3 - num.size(), ' ') + num;
  os << "(" << num << ") ";
  os << DUMP_INDENT;
  if (const auto ntd = dyn_cast<AST::NamedTypeDecl>(&n)) {
    os << ntd->name_str << " " << ntd->init_str << " " << PSTR(ntd->init_expr);
  } else if (const auto nvd = dyn_cast<AST::NamedVariableDecl>(&n)) {
    if (nvd->mem) {
      nvd->mem->Print(os);
      os << " ";
    }
    if (nvd->type) {
      nvd->type->Print(os);
    } else {
      GetSymbolType(nvd->name_str)->Print(os);
    }
    os << " " << nvd->name_str;
    if (nvd->init_expr)
      os << " " << nvd->init_str << " " << PSTR(nvd->init_expr);
    else if (nvd->init_value)
      os << " " << nvd->init_str << " {" << PSTR(nvd->init_value) << "}";
  } else if (const auto assign = dyn_cast<AST::Assignment>(&n)) {
    os << assign->name << " = " << PSTR(assign->value);
  } else if (const auto dma = dyn_cast<AST::DMA>(&n)) {
    if (dma->operation == ".any") {
      os << (dma->future.empty() ? "?" : dma->future);
      os << " = dma.any";
    } else {
      if (dma->future.empty()) {
        assert(!dma->async && "expecting the dma is not async.");
      } else {
        os << dma->future << " = ";
      }
      os << "dma" << dma->operation << (dma->async ? ".async" : "");
      os << (dma->config ? " " + PSTR(dma->config) : "") << " ";
      os << STR(dma->from) << " => " << STR(dma->to);
      assert(!dma->chained && "expecting the dma is not chained.");
    }
    // TODO:
    // if (chained) {
    //   if (chain_to != "")
    //     os << "\n" << prefix << "  `- chained to: " << chain_to;
    //   if (chain_from != "")
    //     os << "\n" << prefix << "  `- chained from: " << chain_from;
    // }
  } else if (const auto w = dyn_cast<AST::Wait>(&n)) {
    os << "wait ";
    for (size_t i = 0; i < w->GetFutures().size(); ++i) {
      if (i > 0) os << ", ";
      auto id = AST::GetIdentifier(*w->GetFutures()[i]);
      os << id->name;
    }
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
  } else if (const auto ret = dyn_cast<AST::Return>(&n)) {
    os << "return";
    if (ret->value) { os << " " << PSTR(ret->value); }
  } else if (const auto pb = dyn_cast<AST::ParallelBy>(&n)) {
    os << "parallel ";
    os << pb->biv->name << " = {";
    pb->iv_symbols->InlinePrint(os);
    os << "} by " << "[";
    pb->bounds->InlinePrint(os);
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
  } else if (const auto fb = dyn_cast<AST::ForeachBlock>(&n)) {
    assert(!fb->pred && "expecting the foreach block has no pred.");
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

bool IsLoopBlock(AST::Node& n) {
  return isa<AST::ParallelBy>(&n) || isa<AST::ForeachBlock>(&n);
}

bool ShouldIndent(AST::Node& n) {
  return isa<AST::ParallelBy>(&n) || isa<AST::WithBlock>(&n) ||
         isa<AST::ForeachBlock>(&n) || isa<AST::ChoreoFunction>(&n);
}

bool LivenessAnalyzer::BeforeVisitImpl(AST::Node& n) {
  if (IsStmt(n)) {
    stmts_preordered.push_back(&n);

    current_stmt = &n;
    stmt2number.emplace(&n, stmt_number);
    ++stmt_number;

    stmt2visit_order[&n].visit_begin = stmt_visit_order;
    ++stmt_visit_order;

    DumpStmtBriefly(n, stmts_with_indent, ShouldIndent(n));
    if (ShouldIndent(n)) {
      DUMP_INDENT += "  ";
      scope2stmt[SSTab().ScopeName()] = &n;
    }
    std::stringstream ss;
    DumpStmtBriefly(n, ss, false);
    stmt2str.emplace(&n, ss.str());
#if DUMP_EACH_STMT
    VST_DEBUG(dbgs() << SSTR(&n));
#endif
  }

  if (auto cf = dyn_cast<AST::ChoreoFunction>(&n)) {
    // TODO: should we def the function name?
    for (const auto& param : cf->f_decl.params->values) {
      if (param->HasSymbol()) {
        if (auto sty = dyn_cast<SpannedType>(param->GetType())) {
          // TODO: AddBufStmt here got error.
          linfo[current_stmt].buffer_related = true;
          std::string sbuf = InScopeName(param->sym->name);
          buffers.insert(sbuf);
          global_buffers.insert(sbuf);
          if (!sty->RuntimeShaped()) {
            // TODO: should we align the size to 512?!
            buf_sizes.emplace(sbuf, sty->ByteSize());
          } else {
            // TODO: handle the runtime shaped buffer.
            // the buffers which are runtime shaped is not in `buf_sizes`.
          }
          AddDef(current_stmt, sbuf, true);
          // TODO: how to show the "stmt"?
        } else {
          assert(false);
        }
      }
    }
  }

  return true;
}

bool LivenessAnalyzer::AfterVisitImpl(AST::Node& n) {
  if (IsStmt(n)) {
    stmt2visit_order[&n].visit_end = stmt_visit_order;
    ++stmt_visit_order;

    if (ShouldIndent(n)) {
      assert(DUMP_INDENT.size() >= 2);
      DUMP_INDENT.pop_back();
      DUMP_INDENT.pop_back();

      auto rbrace = AST::Make<ScopeEnd>(n.LOC(), &n);
      stmt2visit_order[rbrace.get()].visit_begin = stmt_visit_order;
      ++stmt_visit_order;
      stmt2visit_order[rbrace.get()].visit_end = stmt_visit_order;
      ++stmt_visit_order;

      scope_ends.push_back(rbrace);
      stmts_preordered.push_back(rbrace.get());
      stmt2number.emplace(rbrace.get(), stmt_number);
      ++stmt_number;

      DumpStmtBriefly(*rbrace, stmts_with_indent, false);

      if (IsLoopBlock(n)) {
        if (events_to_add.count(&n)) {
          for (const auto& [event_type, var] : events_to_add[&n]) {
            if (event_type != "use") assert(false && "unexpected event type.");
            VST_DEBUG(dbgs() << "insert extra uses: " << var << " in "
                             << stmt2number[rbrace.get()] << "\n";);
            AddUse(rbrace.get(), var, false, false);
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
  }

  if (isa<AST::Program>(&n)) {
    // TODO: do clearing of all the maps per cofunc?
    // TODO: Where is it more appropriate to perform the calculation?
    VST_DEBUG(dbgs() << "\n" << stmts_with_indent.str() << "\n");
    ComputeLiveInOut();
    ComputeLiveRange();
  }

  if (auto w = dyn_cast<AST::Wait>(&n)) {
    for (const auto& f : w->GetFutures()) {
      auto id = AST::GetIdentifier(*f);
      assert(id && "expecting an identifier in Wait.");
      auto fut_name = InScopeName(id->name);
      assert(fut2buffers.count(fut_name) &&
             "expecting the future to be in fut2buffers.");
      // TODO: may have bugs here!
      // may remove unwaited future's buffer
      // but the process will be checked in choreo::future
      for (const auto& [src, dst] : fut2buffers[fut_name]) {
        RemoveBinding(fut_name, src);
      }
      // since we will calculate live_in and live_out after visiting all the
      // nodes, we should record the binding info to do restoration in
      // ComputeLiveInOut().
      stmt2binding_restore[&n] = fut_name;
    }
  }

  return true;
}

bool LivenessAnalyzer::Visit(AST::MultiNodes& n) {
  TraceEachVisit(n);
  return true;
}
bool LivenessAnalyzer::Visit(AST::MultiValues& n) {
  TraceEachVisit(n);
  return true;
}
bool LivenessAnalyzer::Visit(AST::IntLiteral& n) {
  TraceEachVisit(n);
  return true;
}
bool LivenessAnalyzer::Visit(AST::FloatLiteral& n) {
  TraceEachVisit(n);
  return true;
}
bool LivenessAnalyzer::Visit(AST::StringLiteral& n) {
  TraceEachVisit(n);
  return true;
}
bool LivenessAnalyzer::Visit(AST::Boolean& n) {
  TraceEachVisit(n);
  return true;
}
bool LivenessAnalyzer::Visit(AST::Expr& n) {
  TraceEachVisit(n);
  return true;
}
bool LivenessAnalyzer::Visit(AST::MultiDimSpans& n) {
  TraceEachVisit(n);
  return true;
}
bool LivenessAnalyzer::Visit(AST::NamedTypeDecl& n) {
  TraceEachVisit(n);
  // TODO: handle the case of named type decl
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
  } else if (isa<MDSpanType>(ty)) {
    // TODO: handle the case of mdspan type
    assert(false && "not implemented yet.");
  } else if (auto sty = dyn_cast<SpannedType>(ty)) {
    linfo[current_stmt].buffer_related = true;
    if (!IsRef(n)) {
      AddBufStmt(current_stmt, sty->GetStorage());
      if (!sty->RuntimeShaped())
        buf_sizes.emplace(InScopeName(n.name_str), sty->ByteSize());
      AddDef(current_stmt, n.name_str, true);
    } else {
      VST_DEBUG(dbgs() << "The nvd is a reference: " << STR(n) << ".\n\n");
      assert(n.init_expr && "expecting the init_expr is not nullptr.");
      if (auto e = dyn_cast<AST::Expr>(n.init_expr)) {
        if (auto sa = dyn_cast<AST::SpanAs>(e->GetR())) {
          AddDef(current_stmt, n.name_str);
          AddUse(current_stmt, sa->id->name);
          AddAlias(n.name_str, sa->id->name);
        } else {
          assert(false && "expecting the init_expr is a span_as.");
        }
      } else {
        assert(false && "expecting the init_expr is an expr.");
      }
      // TODO: how to handle the case of ref: a = b where b is a buffer?
      // init_expr or init_val?
    }
  } else if (isa<BoundedType>(ty)) {
    // TODO: handle the case of bounded type
    assert(false && "not implemented yet.");
  } else if (isa<FutureType>(ty)) {
    // TODO: handle the case of future type
    assert(false && "not implemented yet.");
  } else {
    assert(false && "expecting the type is spanned, scalar, string, index, "
                    "ituple, or mdspan.");
  }
  return true;
}
bool LivenessAnalyzer::Visit(AST::IntTuple& n) {
  TraceEachVisit(n);
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
    AddDef(current_stmt, n.name);
    AddUse(current_stmt, sa->id->name);
    AddIsAlias(current_stmt, n.name);
    AddAlias(n.name, sa->id->name);
  } else {
    assert(!IsRef(n) && "expecting the assignment is not a reference.");
    VST_DEBUG(dbgs() << "The assignment is not sel or sa: " << STR(n)
                     << ".\n\n");
    AddDef(current_stmt, n.name);
    if (auto expr = dyn_cast<AST::Expr>(n.value)) {
      VarSet operands = GetAllSymbolicOperands(expr.get());
      AddUse(current_stmt, operands);
    } else {
      assert(false && "expecting the assignment value is an expr.");
    }
  }
  return true;
}
bool LivenessAnalyzer::Visit(AST::IntIndex& n) {
  TraceEachVisit(n);
  return true;
}
bool LivenessAnalyzer::Visit(AST::DataType& n) {
  TraceEachVisit(n);
  return true;
}
bool LivenessAnalyzer::Visit(AST::Identifier& n) {
  TraceEachVisit(n);
  return true;
}
bool LivenessAnalyzer::Visit(AST::Parameter& n) {
  TraceEachVisit(n);
  return true;
}
bool LivenessAnalyzer::Visit(AST::ParamList& n) {
  TraceEachVisit(n);
  return true;
}
bool LivenessAnalyzer::Visit(AST::ParallelBy& n) {
  TraceEachVisit(n);
  assert(n.biv && n.iv_symbols &&
         "expecting the parallelby has biv and iv_symbols.");
  AddDef(current_stmt, n.biv->name);
  events_to_add[current_stmt].insert({"use", n.biv->name});
  paraby_bounded_vars.insert(n.biv->name);
  for (const auto& iv_symbol : n.iv_symbols->AllValues()) {
    std::string iv_symbol_name = cast<AST::Identifier>(iv_symbol)->name;
    AddDef(current_stmt, iv_symbol_name);
    events_to_add[current_stmt].insert({"use", iv_symbol_name});
    paraby_bounded_vars.insert(iv_symbol_name);
    AddBinding(n.biv->name, iv_symbol_name);
  }
  return true;
}
bool LivenessAnalyzer::Visit(AST::WhereBind& n) {
  TraceEachVisit(n);
  return true;
}
bool LivenessAnalyzer::Visit(AST::WithIn& n) {
  TraceEachVisit(n);

  return true;
}
bool LivenessAnalyzer::Visit(AST::WithBlock& n) {
  TraceEachVisit(n);
  if (n.reqs) { assert(false && "not implemented yet."); }
  for (const auto& item : n.withins->AllSubs()) {
    auto w = cast<AST::WithIn>(item);
    if (w->with) { AddDef(current_stmt, w->with->name); }
    if (w->with_matchers) {
      for (const auto& item : w->with_matchers->AllValues()) {
        auto id = cast<AST::Identifier>(item);
        AddDef(current_stmt, id->name);
        // TODO: should we add binding here?
        // also in paraby
        if (w->with) AddBinding(w->with->name, id->name);
      }
    }
  }
  return true;
}
bool LivenessAnalyzer::Visit(AST::Memory& n) {
  TraceEachVisit(n);
  return true;
}
bool LivenessAnalyzer::Visit(AST::SpanAs& n) {
  TraceEachVisit(n);
  return true;
}
bool LivenessAnalyzer::Visit(AST::DMA& n) {
  TraceEachVisit(n);
  // TODO: should future be bound with the position in chunkat?
  // TODO: chain DMA?
  // TODO: release the src buffer after the DMA done.
  linfo[current_stmt].buffer_related = true;
  if (n.future.empty()) {
    assert(!n.async && "async dma should have a future.");
    AddUse(current_stmt, n.FromSymbol());
    AddUse(current_stmt, n.ToSymbol());
    // If the dma is sync, then only the dst buffer is alias to the future.
    // The src buffer can be reused immediately after the dma done.
  } else {
    // TODO: handle the case of .any
    // if xxx = dma again, is it a use or def?
    if (n.operation == ".any") {
      AddDef(current_stmt, n.future);
      dma_any.insert(InScopeName(n.future));
      return true;
    }

    // TODO: currently, treat it as use!
    if (dma_any.count(InScopeName(n.future))) {
      AddUse(current_stmt, n.future);
    } else {
      AddDef(current_stmt, n.future);
    }

    AddUse(current_stmt, n.FromSymbol());
    AddUse(current_stmt, n.ToSymbol());
    AddIsBinding(current_stmt, n.future);
    // If the dma is async, then the future is alias to the src buffer.
    // The corresponding src buffer can be reused only after the future has been
    // waited.
    if (n.async) { AddBinding(n.future, n.FromSymbol()); }
    AddBinding(n.future, n.ToSymbol());
    AddFut2Buffers(n.future, n.FromSymbol(), n.ToSymbol());
  }
  return true;
}
bool LivenessAnalyzer::Visit(AST::ChunkAt& n) {
  TraceEachVisit(n);
  assert(n.sa == nullptr && "after norm, there should be no span_as.");
  // `n.data` is already handled in Visit(AST::DMA& n)
  if (!n.positions) return true;
  for (const auto& pos : n.positions->AllValues()) {
    VST_DEBUG(dbgs() << "chunkat position: " << PSTR(pos) << ".\n");
    if (auto expr = dyn_cast<AST::Expr>(pos)) {
      VarSet operands = GetAllSymbolicOperands(expr.get());
      AddUse(current_stmt, operands);
    } else if (auto id = dyn_cast<AST::Identifier>(pos)) {
      AddUse(current_stmt, id->name);
    } else {
      assert(false && "expecting the chunkat position is an expr.");
    }
  }
  return true;
}
bool LivenessAnalyzer::Visit(AST::Wait& n) {
  TraceEachVisit(n);
  linfo[current_stmt].buffer_related = true;
  for (const auto& f : n.GetFutures()) {
    auto id = AST::GetIdentifier(*f);
    assert(id && "expecting an identifier in Wait.");
    AddUse(current_stmt, id->name);
    const std::string sname = InScopeName(id->name);
    assert(fut2buffers.count(sname) &&
           "expecting the future to be in fut2buffers.");
    for (const auto& [src, dst] : fut2buffers[sname]) {
      AddUse(current_stmt, src);
      AddUse(current_stmt, dst);
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
        if (expr->op != "dataof") {
          choreo_unreachable(
              "can only use future.data or buffer as parameters in call!");
        }
        // TODO: will only the dims of future be used?
        assert(isa<FutureType>(expr->GetR()->GetType()) &&
               "expect a future operand.");
        if (auto id = cast<AST::Expr>(expr->GetR())->GetSymbol())
          AddUse(current_stmt, id->name, true);
        else
          choreo_unreachable("Can not retrieve name of the future.");
      }
      // VST_DEBUG(dbgs() << "the node is in call args, and is span:\n\t"
      //                  << PSTR(arg) << "\n");
    } else {
      if (auto expr = dyn_cast<AST::Expr>(arg)) {
        VarSet operands = GetAllSymbolicOperands(expr.get());
        AddUse(current_stmt, operands);
      } else {
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
      for (const auto& [src, dst] : fut2buffers[other_sname]) {
        AddFut2Buffers(id->name, src, dst);
      }
      AddBinding(id->name, other_id->name);
    }
  }
  AddUse(current_stmt, uses);
  return true;
}
bool LivenessAnalyzer::Visit(AST::Select& n) {
  TraceEachVisit(n);
  // already handled in NamedVariableDecl or Assignment
  return true;
}
bool LivenessAnalyzer::Visit(AST::Return& n) {
  TraceEachVisit(n);
  // TODO: can we return future.data?
  auto vty = NodeType(*n.value);
  if (isa<SpannedType>(vty)) {
    if (auto id = AST::GetIdentifier(*n.value)) {
      linfo[current_stmt].buffer_related = true;
      AddUse(current_stmt, id->name);
    } else {
      assert(false && "expecting the return value is an identifier.");
    }
  } else {
    auto expr = dyn_cast<AST::Expr>(n.value);
    assert(expr && "expecting the return value is a select.");
    VarSet operands = GetAllSymbolicOperands(expr.get());
    AddUse(current_stmt, operands);
  }
  return true;
}
bool LivenessAnalyzer::Visit(AST::LoopRange& n) {
  TraceEachVisit(n);
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
      if (auto id = dyn_cast<AST::Identifier>(offset)) {
        AddUse(current_stmt, id->name);
      } else {
        assert(false &&
               "expecting the bound offset in LoopRange is an Identifier.");
      }
    }
  }
  // TODO: test with n.pred!
  if (n.pred) {
    VarSet operands = GetAllSymbolicOperands(n.pred.get());
    AddUse(current_stmt, operands);
  }
  return true;
}
bool LivenessAnalyzer::Visit(AST::IncrementBlock& n) {
  TraceEachVisit(n);
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
bool LivenessAnalyzer::Visit(AST::CppSourceCode& n) {
  TraceEachVisit(n);
  return true;
}
bool LivenessAnalyzer::Visit(AST::Program& n) {
  TraceEachVisit(n);
  return true;
}

bool LivenessAnalyzer::HasError() {
  if (error_count > 0) {
    dbgs() << "Totally " << error_count << " errors have been detected.\n";
    return true;
  }
  return false;
}