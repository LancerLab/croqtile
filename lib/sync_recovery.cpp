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

bool SyncRecovery::RunOnProgramImpl(AST::Node& root) {
  // TODO: proper CCtx flag; env var keeps the scaffold unintrusive.
  if (!std::getenv("CHOREO_SYNC_RECOVERY")) return true;
  if (mr == nullptr) return true;
  root_ = &root;
  RunRecovery();
  // C6: drive the traversal that splices the pending insertions.
  if (!pending_inserts_.empty()) root.accept(*this);
  dbgs() << "[sync-recovery] summary: " << total_slots_ << " shared slots, "
         << total_handovers_ << " handovers (" << hb_merged_handovers_
         << " hb-merged), " << inserted_fences_ << " fences, "
         << inserted_waits_ << " completion waits inserted\n";
  return true;
}

bool SyncRecovery::Visit(AST::MultiNodes& n) {
  if (pending_inserts_.empty()) return true;
  for (size_t i = 0; i < n.Count(); ++i) {
    auto it = pending_inserts_.find(n.SubAt(i).get());
    if (it == pending_inserts_.end()) continue;
    std::vector<PendingInsert>& inserts = it->second;
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
                             const ptr<AST::Node>& node, bool after) {
  auto& list = pending_inserts_[anchor];
  for (const auto& p : list) {
    if (p.after != after) continue;
    if (auto* f = dyn_cast<AST::Fence>(p.node.get())) {
      if (auto* g = dyn_cast<AST::Fence>(node.get()))
        if (f->GetEntity() == g->GetEntity()) return;
    } else if (auto* w = dyn_cast<AST::CompletionWait>(p.node.get())) {
      if (auto* v = dyn_cast<AST::CompletionWait>(node.get()))
        if (w->GetEngine() == v->GetEngine() && w->GetKind() == v->GetKind() &&
            w->GetPending() == v->GetPending())
          return;
    }
  }
  list.push_back(PendingInsert{node, after});
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

} // end anonymous namespace

void SyncRecovery::RunRecovery() {
  const auto& df_ctxs = mr->RecoveryDFCtxs();
  const auto& hb_graphs = mr->LA().HBGraphs();

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
          const auto& po = mr->LA().PreorderStmts();
          auto site = ProducerFrontier(po, *g, a_last);
          if (site.stmt != nullptr) {
            for (const auto& o : obligations) {
              if (o.cls == "visibility")
                AddInsert(site.stmt,
                          AST::Make<AST::Fence>(
                              site.stmt->LOC(), ParallelLevel::GROUP,
                              Storage::SHARED, FenceEntity::TMA),
                          site.after);
              // WAR with the engine as reader: the receiving tenant may
              // overwrite the slot only once the engine has drained the
              // reads it issued. Other completion classes lower to waits
              // bound to the engine's own event, not to a copy group.
              if (o.cls == "completion" &&
                  o.primitive.rfind("read-drain", 0) == 0)
                AddInsert(site.stmt,
                          AST::Make<AST::CompletionWait>(
                              site.stmt->LOC(), FenceEntity::TMA,
                              CompletionWaitKind::READ_DRAIN, 0),
                          site.after);
            }
          }
          dbgs() << "\n";
        }
    }
  }
}
