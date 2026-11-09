#include "sync_recovery.hpp"

#include <algorithm>
#include <cstdlib>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include <map>
#include <set>

#include "aux.hpp"
#include "context.hpp"
#include "liveness_analysis.hpp"
#include "mem_reuse.hpp"

using namespace Choreo;

namespace {
// Elimination helpers (defined with the analysis utilities below).
const char* CoveringOp(const AST::Node* s, const AST::MultiNodes& n, size_t j,
                       const AST::Node& ins, const std::string& buffer);
bool SelfDischarging(const AST::Node* s, const AST::Node& ins,
                     const std::string& buffer);
} // namespace

bool SyncRecovery::RunOnProgramImpl(AST::Node& root) {
  // TODO: proper CCtx flag; env var keeps the scaffold unintrusive.
  if (!std::getenv("CHOREO_SYNC_RECOVERY")) return true;
  if (mr == nullptr) return true;
  keep_insertions_ = std::getenv("CHOREO_SYNC_RECOVERY_KEEP") != nullptr;
  root_ = &root;
  RunRecovery();
  // C6: drive the traversal that splices the pending insertions.
  if (!pending_inserts_.empty()) root.accept(*this);
  dbgs() << "[sync-recovery] summary: " << total_slots_ << " shared slots, "
         << total_handovers_ << " handovers (" << hb_merged_handovers_
         << " hb-merged), " << inserted_fences_ << " fences, "
         << inserted_waits_ << " completion waits inserted, " << elided_fences_
         << " fences, " << elided_waits_ << " completion waits elided, "
         << self_waits_ << " completion waits self-discharged, "
         << discharged_waits_ << " completion waits discharged in place\n";
  return true;
}

bool SyncRecovery::Visit(AST::MultiNodes& n) {
  if (pending_inserts_.empty()) return true;
  for (size_t i = 0; i < n.Count(); ++i) {
    auto it = pending_inserts_.find(n.SubAt(i).get());
    if (it == pending_inserts_.end()) continue;
    std::vector<PendingInsert>& inserts = it->second;
    // Elimination first: the containing list is what makes domination
    // decidable, so coverage is judged here rather than at synthesis time.
    Eliminate(inserts, n, i);
    // `idx` tracks the anchor as earlier insertions shift it to the right.
    size_t idx = i;
    for (const auto& p : inserts)
      if (!p.after) {
        if (isa<AST::Fence>(p.node.get()))
          ++inserted_fences_;
        else
          ++inserted_waits_;
        dbgs() << "[sync-recovery] insert " << p.node->TypeNameString()
               << " before stmt\n";
        n.Insert(p.node, idx++);
      }
    size_t after_idx = idx + 1;
    for (const auto& p : inserts)
      if (p.after) {
        if (isa<AST::Fence>(p.node.get()))
          ++inserted_fences_;
        else
          ++inserted_waits_;
        dbgs() << "[sync-recovery] insert " << p.node->TypeNameString()
               << " after stmt\n";
        n.Insert(p.node, after_idx++);
      }
    i = after_idx - 1;
    pending_inserts_.erase(it);
  }
  return true;
}

void SyncRecovery::AddInsert(const AST::Node* anchor,
                             const ptr<AST::Node>& node, bool after,
                             int not_before, const std::string& buffer,
                             bool self_discharge) {
  auto& list = pending_inserts_[anchor];
  for (auto& p : list) {
    if (p.after != after) continue;
    if (auto* f = dyn_cast<AST::Fence>(p.node.get())) {
      if (auto* g = dyn_cast<AST::Fence>(node.get()))
        if (f->GetEntity() == g->GetEntity()) {
          // Merge the preconditions: elimination must fold against the last
          // access of every obligation the collapsed op discharges.
          p.not_before = std::max(p.not_before, not_before);
          return;
        }
    } else if (auto* w = dyn_cast<AST::CompletionWait>(p.node.get())) {
      if (auto* v = dyn_cast<AST::CompletionWait>(node.get()))
        if (w->GetEngine() == v->GetEngine() && w->GetKind() == v->GetKind()) {
          // A smaller pending count drains more: keep the stronger node.
          if (v->GetPending() < w->GetPending()) p.node = node;
          p.not_before = std::max(p.not_before, not_before);
          p.self_discharge = p.self_discharge && self_discharge;
          return;
        }
    }
  }
  list.push_back(
      PendingInsert{node, after, not_before, buffer, self_discharge});
}

void SyncRecovery::Eliminate(std::vector<PendingInsert>& inserts,
                             const AST::MultiNodes& n, size_t anchor_idx) {
  if (keep_insertions_ || inserts.empty()) return;
  assert(!visit_stack_.empty() && visit_stack_.back() == &n &&
         "elimination must run at the list currently being visited");
  std::vector<PendingInsert> keep;
  keep.reserve(inserts.size());
  for (const auto& ins : inserts) {
    const char* reason = nullptr;
    // A statement covers the insertion iff it executes on every path that
    // reaches the insertion site and follows the producer's last access. Both
    // the enclosing-list prefix and the prefixes of its ancestor lists have
    // that property (must direction).
    auto probe = [&](const AST::Node* s, const AST::MultiNodes& lst,
                     size_t j) -> const char* {
      auto pit = preorder_index_.find(s);
      if (pit == preorder_index_.end()) return nullptr; // our own insertion
      int sidx = pit->second;
      if (ins.not_before >= 0 && sidx <= ins.not_before) {
        if (sidx == ins.not_before && ins.self_discharge &&
            SelfDischarging(s, *ins.node, ins.buffer))
          return "synchronous engine access";
        return nullptr;
      }
      return CoveringOp(s, lst, j, *ins.node, ins.buffer);
    };

    // Site-level slice: the prefix of the enclosing statement list.
    long last = ins.after ? (long)anchor_idx : (long)anchor_idx - 1;
    for (long j = 0; j <= last && reason == nullptr; ++j)
      reason = probe(n.SubAt((size_t)j).get(), n, (size_t)j);

    // Cross-block slice: every list on the ancestry stack contributes the
    // siblings before the child that contains the anchor. Those execute
    // whenever the anchor does, including per iteration when a loop header is
    // crossed, so they are must-available cover sites as well.
    for (size_t k = 0; k + 1 < visit_stack_.size() && reason == nullptr; ++k) {
      auto* up = dyn_cast<AST::MultiNodes>(visit_stack_[k]);
      if (up == nullptr) continue;
      const AST::Node* child = visit_stack_[k + 1];
      size_t stop = up->Count();
      bool found = false;
      for (size_t j = 0; j < up->Count(); ++j)
        if (up->SubAt(j).get() == child) {
          stop = j;
          found = true;
          break;
        }
      if (!found) continue; // child not tracked; stay conservative
      for (size_t j = 0; j < stop && reason == nullptr; ++j)
        reason = probe(up->SubAt(j).get(), *up, j);
    }
    if (reason == nullptr) {
      keep.push_back(ins);
      continue;
    }
    if (isa<AST::Fence>(ins.node.get()))
      ++elided_fences_;
    else if (std::string(reason) == "synchronous engine access")
      ++self_waits_;
    else
      ++elided_waits_;
    dbgs() << "[sync-recovery] elide " << ins.node->TypeNameString() << " ("
           << reason << ")\n";
  }
  inserts.swap(keep);
}

namespace {

using HBGraph = LivenessAnalyzer::HBGraph;

// Same scope lookup as MemReuse::ProtoType: exact match, else longest
// scope that starts with the device-function name.
const HBGraph* FindHBGraph(const std::map<std::string, HBGraph>& hb_graphs,
                           const std::string& df_name) {
  auto it = hb_graphs.find(df_name);
  if (it != hb_graphs.end()) return &it->second;
  const HBGraph* best = nullptr;
  size_t best_len = 0;
  for (const auto& [scope, graph] : hb_graphs) {
    if (scope.find(df_name) == 0 && scope.size() > best_len) {
      best_len = scope.size();
      best = &graph;
    }
  }
  return best;
}

struct Tenant {
  std::string buffer_id;
  size_t offset;
  size_t size;
};

// Phase indices whose access set contains the buffer, in phase order.
std::vector<int> PhasesOf(const HBGraph& g, const std::string& buf) {
  std::vector<int> out;
  for (size_t i = 0; i < g.phases.size(); ++i)
    if (g.phases[i].buffers_accessed.count(buf)) out.push_back((int)i);
  return out;
}

// --- C3: coarse access-domain typing ---
// Canonical domains: GR generic-read, GW generic-write, ER engine-read,
// EW engine-write. Tags recorded by the liveness analyzer: "AddUse" (GR),
// "def" (GW), "DMA-from" (ER), "DMA-to" (EW). MMA operands currently land
// as generic (fine for sm_120: no tcgen05/wgmma; TODO for sm_90a/sm_100).
std::set<std::string> TenantDomains(const HBGraph& g,
                                    const std::vector<int>& phases,
                                    const std::string& buf) {
  std::set<std::string> doms;
  for (int p : phases) {
    auto it = g.phases[p].buffer_access_kinds.find(buf);
    if (it == g.phases[p].buffer_access_kinds.end()) continue;
    for (const auto& tag : it->second) {
      if (tag == "AddUse")
        doms.insert("GR");
      else if (tag == "def")
        doms.insert("GW");
      else if (tag == "DMA-from")
        doms.insert("ER");
      else if (tag == "DMA-to")
        doms.insert("EW");
      else if (tag == "observe")
        continue; // completion observation, not a data access
      else
        doms.insert(tag);
    }
  }
  return doms;
}

// --- C4: obligation derivation (Table 2, NVIDIA instantiation) ---
// Per handover A -> B, each (domain-of-A, domain-of-B) combination yields a
// dependence class (RAW/WAR/WAW) and the obligations it carries, with the
// selected primitive (P-numbers as in the paper's Table 1). Phase-granular
// for now; per-site refinement is the precision ladder's next rung.
struct Obligation {
  std::string dep;       // RAW | WAR | WAW
  std::string pair;      // e.g. EW->GR
  std::string cls;       // completion | visibility | ordering
  std::string primitive; // P-tag with the concrete discharge op
};

std::vector<Obligation> DeriveObligations(const std::set<std::string>& a,
                                          const std::set<std::string>& b) {
  std::vector<Obligation> out;
  auto has = [](const std::set<std::string>& s, const char* k) {
    return s.count(k) != 0;
  };
  auto emit = [&](const char* dep, const char* pair, const char* cls,
                  const char* prim) {
    out.push_back(Obligation{dep, pair, cls, prim});
  };
  for (const char* ad : {"GR", "GW", "ER", "EW"}) {
    if (!has(a, ad)) continue;
    for (const char* bd : {"GR", "GW", "ER", "EW"}) {
      if (!has(b, bd)) continue;
      bool a_writes = ad[1] == 'W', a_reads = !a_writes;
      bool b_writes = bd[1] == 'W', b_reads = !b_writes;
      bool a_eng = ad[0] == 'E', b_eng = bd[0] == 'E';
      std::string pair = std::string(ad) + "->" + bd;
      if (a_writes && b_reads) { // RAW
        if (!a_eng && !b_eng)
          emit("RAW", pair.c_str(), "ordering", "P1:sync-edge");
        else if (!a_eng && b_eng)
          emit("RAW", pair.c_str(), "visibility", "P6:fence.proxy.async");
        else if (a_eng && !b_eng)
          emit("RAW", pair.c_str(), "completion", "engine-wait(P8/P9)");
        else
          emit("RAW", pair.c_str(), "completion", "P9:wait_group");
      } else if (a_reads && b_writes) { // WAR
        if (!a_eng && !b_eng)
          emit("WAR", pair.c_str(), "ordering", "P1:sync-edge");
        else if (!a_eng && b_eng)
          emit("WAR", pair.c_str(), "completion", "P1:before-reissue");
        else if (a_eng && !b_eng)
          emit("WAR", pair.c_str(), "completion", "read-drain(P9.read/P8)");
        else
          emit("WAR", pair.c_str(), "completion", "read-drain(P9.read)");
      } else if (a_writes && b_writes) { // WAW
        if (!a_eng && !b_eng)
          emit("WAW", pair.c_str(), "ordering", "P1:sync-edge");
        else if (!a_eng && b_eng) {
          emit("WAW", pair.c_str(), "ordering", "P1:sync-edge");
          emit("WAW", pair.c_str(), "visibility", "P6:publish-before-write");
        } else if (a_eng && !b_eng)
          emit("WAW", pair.c_str(), "completion", "engine-wait(P8/P9)");
        else
          emit("WAW", pair.c_str(), "completion", "P9:wait_group(full)");
      }
      // GR->GR / ER->GR / GR->ER: no writer, no obligation (RAR control).
    }
  }
  return out;
}

// Signal names on the phase boundaries strictly between phase `from`
// (exclusive) and phase `to` (inclusive) that separate the two tenants.
// Same-workgroup boundaries come from intermediate phases; the cross-
// workgroup case surfaces as the signal_out of `from` / signal_in of `to`.
std::vector<std::string> WitnessSignals(const HBGraph& g, int from, int to) {
  std::vector<std::string> sigs;
  auto add = [&](const std::string& s) {
    if (!s.empty() && std::find(sigs.begin(), sigs.end(), s) == sigs.end())
      sigs.push_back(s);
  };
  add(g.phases[from].signal_out);
  if (g.phases[from].wg_id == g.phases[to].wg_id)
    for (int i = from + 1; i <= to; ++i) {
      add(g.phases[i].signal_in);
      add(g.phases[i].signal_out);
    }
  else
    add(g.phases[to].signal_in);
  return sigs;
}

// Producer-side frontier: the statement where the producer's phase hands the
// slot over. Proxy fences and bulk-group drains are per-thread state, so they
// must execute on the threads that issued the async work: after the last
// access, before the handover sync. When the phase closes on a sync statement
// the insertion goes before it; when it closes on an access, right after it.
struct FrontierSite {
  const AST::Node* stmt = nullptr;
  bool after = false;
};

FrontierSite
ProducerFrontier(const std::vector<const LivenessAnalyzer::Stmt*>& po,
                 const HBGraph& g, int phase) {
  const auto& ph = g.phases[phase];
  if (po.empty() || ph.last_stmt_id >= po.size()) return {};
  size_t sid = ph.last_stmt_id;
  // The boundary may be a synthetic ScopeEnd (not in the AST); walk back to
  // the last real statement (typically the handover trigger).
  while (sid > ph.first_stmt_id && isa<LivenessAnalyzer::ScopeEnd>(po[sid]))
    --sid;
  const AST::Node* s = po[sid];
  if (s == nullptr || isa<LivenessAnalyzer::ScopeEnd>(s)) return {};
  if (isa<AST::Trigger>(s) || isa<AST::Barrier>(s) || isa<AST::Synchronize>(s))
    return {s, false};
  return {s, true};
}

// --- elimination: attribution and availability ---------------------------

// Does `n`'s subtree refer to `buffer`? Names are compared unscoped so that a
// phase's scoped buffer id matches the identifiers in the access statement.
bool MentionsBuffer(const AST::Node* n, const std::string& buffer) {
  if (n == nullptr || buffer.empty()) return false;
  // Composite nodes whose operands ReferredSymbols does not walk.
  if (auto* d = dyn_cast<AST::DMA>(n))
    return MentionsBuffer(d->from.get(), buffer) ||
           MentionsBuffer(d->to.get(), buffer);
  if (auto* a = dyn_cast<AST::Assignment>(n))
    return MentionsBuffer(a->da.get(), buffer) ||
           MentionsBuffer(a->value.get(), buffer);
  const std::string buf = UnScopedExpr(buffer);
  for (const auto& sym : ReferredSymbols(const_cast<AST::Node*>(n), nullptr))
    if (UnScopedExpr(sym) == buf) return true;
  return false;
}

bool DmaReadsSlot(const AST::DMA* dma, const std::string& buffer) {
  return dma->from != nullptr && MentionsBuffer(dma->from.get(), buffer);
}

bool DmaWritesSlot(const AST::DMA* dma, const std::string& buffer) {
  return dma->to != nullptr && MentionsBuffer(dma->to.get(), buffer);
}

// Does the phase leave an async engine read of the slot outstanding before
// `upto`? A synchronous read is self-discharging only when it is not hiding
// earlier async work: a blocking copy retires its own read, not the groups
// issued by a previous async copy.
bool HasEarlierAsyncEngineRead(
    const std::vector<const LivenessAnalyzer::Stmt*>& po, const HBGraph& g,
    int phase, int upto, const std::string& buffer) {
  if (upto < 0) return false;
  const auto& ph = g.phases[phase];
  for (size_t i = ph.first_stmt_id;
       i < static_cast<size_t>(upto) && i < po.size(); ++i) {
    auto* d = dyn_cast<AST::DMA>(po[i]);
    if (d != nullptr && d->IsAsync() && DmaReadsSlot(d, buffer)) return true;
  }
  return false;
}

// Does the handover producer guarantee its engine writes have retired before
// the receiver's phase-entry event wait? The receiver's first access is
// ordered after its `wait <sig>` only if the producer's closing trigger is
// ordered after the writes: either the trigger carries the copies' native
// completion (`trigger <sig> after f`) or it is immediately preceded by a
// source `wait <f>` on every async copy that wrote the slot. The latter is
// the common "wait the copy, then signal" idiom; both let the receiver's
// source event wait discharge the engine-wait obligation in place.
bool SourceEventWaitDischarges(
    const std::vector<const LivenessAnalyzer::Stmt*>& po, const HBGraph& g,
    const std::vector<int>& prod_phases, int prod_phase, int recv_phase,
    const std::string& buffer, const AST::Node* closing_stmt, int trig_idx) {
  if (closing_stmt == nullptr) return false;
  const auto& pp = g.phases[prod_phase];
  const auto& rp = g.phases[recv_phase];
  // The receiver's phase must start at the wait on the same event the
  // producer triggers; a chained or barrier-ordered handover is not covered.
  if (pp.signal_out.empty() || rp.signal_in != pp.signal_out) return false;
  auto* trig = dyn_cast<AST::Trigger>(const_cast<AST::Node*>(closing_stmt));
  if (trig == nullptr || trig->GetEvents().size() != 1) return false;

  // Futures whose completion the trigger itself binds (native expect_tx).
  std::set<std::string> trigger_bound;
  if (trig->HasDependencies())
    for (const auto& dependency : trig->GetDependencies()) {
      auto id = AST::GetIdentifier(*dependency);
      if (!id) return false;
      trigger_bound.insert(UnScopedExpr(id->name));
    }
  // Futures awaited by the statement immediately before the trigger. Requiring
  // an immediate predecessor in the same statement list (not merely the
  // previous preorder statement across a synthetic scope end) keeps a wait
  // inside a conditional branch from being mistaken for an unconditional one.
  std::set<std::string> awaited;
  if (trig_idx > 0 && trig_idx <= (int)po.size()) {
    if (auto* w = dyn_cast<AST::Wait>(po[trig_idx - 1])) {
      for (const auto& t : w->GetTargets()) {
        auto id = AST::GetIdentifier(*t);
        if (!id) continue;
        awaited.insert(UnScopedExpr(id->name));
      }
    }
  }

  // Every async engine write to the slot must be ordered before the trigger.
  // A write in an earlier producer phase is not covered by a wait placed just
  // before the closing trigger, so it needs the trigger's own binding.
  bool saw_async_write = false;
  for (int p : prod_phases) {
    const auto& ph = g.phases[p];
    for (size_t i = ph.first_stmt_id; i <= ph.last_stmt_id && i < po.size();
         ++i) {
      auto* d = dyn_cast<AST::DMA>(po[i]);
      if (d == nullptr || !d->IsAsync() || !DmaWritesSlot(d, buffer)) continue;
      saw_async_write = true;
      const std::string f = UnScopedExpr(d->future);
      if (trigger_bound.count(f)) {
        // The native expect_tx lowering is a TMA-load path. SemaChecker
        // already rejects trigger-after on a non-shared event.
        if (!d->IsTMA()) return false;
        continue;
      }
      if (p == prod_phase && awaited.count(f)) continue;
      return false;
    }
  }
  return saw_async_write;
}

// Direction filter for drain preconditions: a read drain folds against the
// last engine read, a full drain against the last engine write.
enum class AccessDir { Any, Read, Write };

// Preorder index of the last statement in the phase that touches the slot in
// the requested direction. Read/Write restrict to engine (DMA) statements.
int LastSlotAccessIndex(const std::vector<const LivenessAnalyzer::Stmt*>& po,
                        const HBGraph& g, int phase, const std::string& buffer,
                        AccessDir dir) {
  const auto& ph = g.phases[phase];
  int last = -1;
  for (size_t i = ph.first_stmt_id; i <= ph.last_stmt_id && i < po.size();
       ++i) {
    const AST::Node* s = po[i];
    if (s == nullptr || isa<LivenessAnalyzer::ScopeEnd>(s)) continue;
    if (dir == AccessDir::Any) {
      if (!MentionsBuffer(s, buffer)) continue;
    } else {
      auto* d = dyn_cast<AST::DMA>(s);
      if (d == nullptr) continue;
      if (dir == AccessDir::Read && !DmaReadsSlot(d, buffer)) continue;
      if (dir == AccessDir::Write && !DmaWritesSlot(d, buffer)) continue;
    }
    last = (int)i;
  }
  return last;
}

// A synchronous engine access has no outstanding work once it retires, so it
// discharges its own read-drain obligation (a blocking copy has finished
// reading before the next statement runs). Full drains are not discharged
// this way: a thread-executed copy does not retire a group's writes.
bool SelfDischarging(const AST::Node* s, const AST::Node& ins,
                     const std::string& buffer) {
  auto* cw = dyn_cast<AST::CompletionWait>(&ins);
  if (cw == nullptr || buffer.empty()) return false;
  if (cw->GetKind() != CompletionWaitKind::READ_DRAIN) return false;
  auto* d = dyn_cast<AST::DMA>(s);
  return d != nullptr && !d->IsAsync() && DmaReadsSlot(d, buffer);
}

// Covering order (Table 1 instantiation, must direction): an operation
// available at the insertion site that discharges everything the insertion
// would. Returns a short attribution string, or nullptr when it does not
// cover.
const char* CoveringOp(const AST::Node* s, const AST::MultiNodes& n, size_t j,
                       const AST::Node& ins, const std::string& buffer) {
  if (auto* w = dyn_cast<AST::CompletionWait>(&ins)) {
    if (auto* cw = dyn_cast<AST::CompletionWait>(s)) {
      if (cw->GetEngine() != FenceEntity::TMA &&
          cw->GetEngine() != FenceEntity::DMA)
        return nullptr;
      // A stronger (smaller) pending count covers a weaker one. A full drain
      // retires both writes and reads; a read drain retires only reads.
      if (cw->GetKind() == CompletionWaitKind::FULL_DRAIN &&
          cw->GetPending() <= w->GetPending())
        return "available complete drain";
      if (cw->GetKind() == CompletionWaitKind::READ_DRAIN &&
          w->GetKind() == CompletionWaitKind::READ_DRAIN &&
          cw->GetPending() <= w->GetPending())
        return "available read drain";
      return nullptr;
    }
    if (auto* wt = dyn_cast<AST::Wait>(s)) {
      // A wait on a future covers the drain only if that future belongs to an
      // async copy that touched this slot in the covered direction (the wait
      // retires that copy's outstanding work).
      for (const auto& t : wt->GetTargets()) {
        auto id = AST::GetIdentifier(*t);
        if (!id) continue;
        const std::string fname = UnScopedExpr(id->name);
        for (size_t k = 0; k < j; ++k) {
          auto* d = dyn_cast<AST::DMA>(n.SubAt(k).get());
          if (d == nullptr || !d->IsAsync()) continue;
          if (UnScopedExpr(d->future) != fname) continue;
          if (w->GetKind() == CompletionWaitKind::READ_DRAIN &&
              DmaReadsSlot(d, buffer))
            return "source wait on the draining copy";
          if (w->GetKind() == CompletionWaitKind::FULL_DRAIN &&
              DmaWritesSlot(d, buffer))
            return "source wait on the draining copy";
        }
      }
      return nullptr;
    }
    return nullptr;
  }
  if (isa<AST::Fence>(&ins)) {
    if (auto* f = dyn_cast<AST::Fence>(s)) {
      const FenceEntity e = f->GetEntity();
      // Only engine-path fences publish to the async proxy. The CUTE backend
      // lowers ALL/THREADS to a generic threadfence, which does not establish
      // `fence.proxy.async`, so they cannot cover the inserted proxy fence.
      if (e == FenceEntity::TMA || e == FenceEntity::DMA)
        return "available proxy fence";
    }
  }
  return nullptr;
}

} // end anonymous namespace

void SyncRecovery::RunRecovery() {
  const auto& df_ctxs = mr->RecoveryDFCtxs();
  const auto& hb_graphs = mr->LA().HBGraphs();
  const auto& po = mr->LA().PreorderStmts();
  for (size_t i = 0; i < po.size(); ++i) preorder_index_[po[i]] = (int)i;

  for (const auto& [df_name, ctx] : df_ctxs) {
    // --- C1: tenant reconstruction from post-coloring offsets ---
    std::vector<Tenant> tenants;
    for (const auto& b : ctx.buffers) {
      auto it = ctx.mem_offset.find(b.buffer_id);
      if (it == ctx.mem_offset.end()) continue;
      tenants.push_back(Tenant{b.buffer_id, it->second, b.size});
    }
    if (tenants.size() < 2) continue;
    std::sort(tenants.begin(), tenants.end(),
              [](const Tenant& a, const Tenant& b) {
                if (a.offset != b.offset) return a.offset < b.offset;
                return a.buffer_id < b.buffer_id;
              });

    // Sweep into slots: tenants whose [offset, offset+size) intervals
    // overlap share a physical slot.
    std::vector<std::vector<Tenant>> slots;
    for (const auto& t : tenants) {
      if (!slots.empty()) {
        size_t cur_end = 0;
        for (const auto& m : slots.back())
          cur_end = std::max(cur_end, m.offset + m.size);
        if (t.offset < cur_end) {
          slots.back().push_back(t);
          continue;
        }
      }
      slots.push_back({t});
    }

    bool printed_df = false;
    const HBGraph* g = FindHBGraph(hb_graphs, df_name);

    for (const auto& slot : slots) {
      if (slot.size() < 2) continue;
      ++total_slots_;
      if (!printed_df) {
        dbgs() << "[sync-recovery] device function: " << df_name << "\n";
        printed_df = true;
      }
      size_t slot_off = slot.front().offset;
      size_t slot_end = 0;
      for (const auto& t : slot)
        slot_end = std::max(slot_end, t.offset + t.size);
      dbgs() << "  slot @" << slot_off << " bytes [" << slot_off << ", "
             << slot_end << "): " << slot.size() << " tenants\n";

      // --- C2: handover enumeration over the HB order ---
      std::vector<std::vector<int>> phase_lists;
      for (const auto& t : slot) {
        auto ps = g ? PhasesOf(*g, t.buffer_id) : std::vector<int>{};
        phase_lists.push_back(ps);
        dbgs() << "    tenant " << UnScopedExpr(t.buffer_id) << " offset "
               << t.offset << " size " << t.size;
        if (ps.empty())
          dbgs() << " (no HB phases)\n";
        else
          dbgs() << " phases [" << ps.front() << ".." << ps.back() << "]";
        auto doms =
            g ? TenantDomains(*g, ps, t.buffer_id) : std::set<std::string>{};
        if (!doms.empty()) {
          dbgs() << " domains {";
          bool first = true;
          for (const auto& d : doms) {
            if (!first) dbgs() << ",";
            dbgs() << d;
            first = false;
          }
          dbgs() << "}";
        }
        dbgs() << "\n";
      }
      if (!g) {
        dbgs() << "    (no HB graph for scope; handovers unordered)\n";
        continue;
      }

      std::vector<std::set<std::string>> tenant_doms;
      for (size_t i = 0; i < slot.size(); ++i)
        tenant_doms.push_back(
            TenantDomains(*g, phase_lists[i], slot[i].buffer_id));

      for (size_t i = 0; i < slot.size(); ++i)
        for (size_t j = 0; j < slot.size(); ++j) {
          if (i == j || phase_lists[i].empty() || phase_lists[j].empty())
            continue;
          int a_last = phase_lists[i].back();
          int b_first = phase_lists[j].front();
          if (!g->Reaches(a_last, b_first)) continue;
          if (g->Reaches(phase_lists[j].back(), phase_lists[i].front()))
            continue; // concurrent in both directions; not a handover

          ++total_handovers_;
          bool hb_merged = false;
          for (const auto& [x, y] : ctx.hb_overlapped_pairs)
            if ((x == slot[i].buffer_id && y == slot[j].buffer_id) ||
                (x == slot[j].buffer_id && y == slot[i].buffer_id)) {
              hb_merged = true;
              break;
            }
          if (hb_merged) ++hb_merged_handovers_;

          dbgs() << "    handover " << UnScopedExpr(slot[i].buffer_id) << " -> "
                 << UnScopedExpr(slot[j].buffer_id)
                 << (hb_merged ? " [hb-merged]" : " [liveness-free]")
                 << " stmts [" << g->phases[a_last].last_stmt_id << " -> "
                 << g->phases[b_first].first_stmt_id << "]";
          auto sigs = WitnessSignals(*g, a_last, b_first);
          if (!sigs.empty()) {
            dbgs() << " witness:";
            for (const auto& s : sigs) dbgs() << " " << s;
          }
          auto obligations = DeriveObligations(tenant_doms[i], tenant_doms[j]);
          if (!obligations.empty()) {
            dbgs() << " obligations:";
            for (const auto& o : obligations)
              dbgs() << " " << o.dep << "[" << o.pair << "](" << o.cls << ":"
                     << o.primitive << ")";
          }
          // C6: frontier insertion targets (always-insert; the elimination
          // pass removes the redundant ones later). Both a visibility fence
          // and a bulk-group drain are per-thread: they must execute on the
          // threads that produced the slot, after the last access and before
          // the sync that hands the slot over --- i.e. inside the producer
          // tenant, at the phase boundary (never on the receiver).
          auto site = ProducerFrontier(po, *g, a_last);
          bool event_discharge = false;
          if (site.stmt != nullptr) {
            const int last_access = LastSlotAccessIndex(
                po, *g, a_last, slot[i].buffer_id, AccessDir::Any);
            const int last_engine_read = LastSlotAccessIndex(
                po, *g, a_last, slot[i].buffer_id, AccessDir::Read);
            const int last_engine_write = LastSlotAccessIndex(
                po, *g, a_last, slot[i].buffer_id, AccessDir::Write);
            for (const auto& o : obligations) {
              if (o.cls == "visibility")
                AddInsert(site.stmt,
                          AST::Make<AST::Fence>(
                              site.stmt->LOC(), ParallelLevel::GROUP,
                              Storage::SHARED, FenceEntity::TMA),
                          site.after, last_access);
              if (o.cls != "completion") continue;
              // Group-bound completion classes drain on the producing
              // threads: a WAR read-drain when the engine read the slot, a
              // full drain when the engine wrote it and the next tenant
              // reuses it. `engine-wait` classes are bound to the engine's
              // own event on the receiver and stay dump-only.
              if (o.primitive.rfind("read-drain", 0) == 0) {
                bool can_self = false;
                if (last_engine_read >= 0 &&
                    static_cast<size_t>(last_engine_read) < po.size()) {
                  auto* d = dyn_cast<AST::DMA>(po[last_engine_read]);
                  can_self = d != nullptr && !d->IsAsync() &&
                             DmaReadsSlot(d, slot[i].buffer_id) &&
                             !HasEarlierAsyncEngineRead(po, *g, a_last,
                                                        last_engine_read,
                                                        slot[i].buffer_id);
                }
                AddInsert(site.stmt,
                          AST::Make<AST::CompletionWait>(
                              site.stmt->LOC(), FenceEntity::TMA,
                              CompletionWaitKind::READ_DRAIN, 0),
                          site.after, last_engine_read, slot[i].buffer_id,
                          can_self);
              } else if (o.primitive.rfind("P9:wait_group", 0) == 0)
                AddInsert(site.stmt,
                          AST::Make<AST::CompletionWait>(
                              site.stmt->LOC(), FenceEntity::TMA,
                              CompletionWaitKind::FULL_DRAIN, 0),
                          site.after, last_engine_write, slot[i].buffer_id);
              // `engine-wait` classes are bound to the engine's own event on
              // the receiver. When the receiver's phase starts at a wait on
              // the producer's trigger event and the producer has already
              // ordered the engine writes before that trigger, the source
              // event wait discharges the obligation in place; no primitive
              // is synthesized. KEEP shows the raw synthesis (nothing), so
              // the attribution is only recorded on the normal path.
              else if (o.primitive.rfind("engine-wait", 0) == 0 &&
                       !keep_insertions_ && !event_discharge) {
                auto idx_it = preorder_index_.find(site.stmt);
                const int trig_idx =
                    idx_it == preorder_index_.end() ? -1 : idx_it->second;
                if (SourceEventWaitDischarges(po, *g, phase_lists[i], a_last,
                                              b_first, slot[i].buffer_id,
                                              site.stmt, trig_idx)) {
                  event_discharge = true;
                  ++discharged_waits_;
                }
              }
            }
          }
          dbgs() << "\n";
          if (event_discharge)
            dbgs() << "[sync-recovery] discharge AST::CompletionWait"
                   << " (source event wait)\n";
        }
    }
  }
}
