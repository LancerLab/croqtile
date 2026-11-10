#ifndef __CHOREO_SYNC_RECOVERY_HPP__
#define __CHOREO_SYNC_RECOVERY_HPP__

#include <map>
#include <set>
#include <vector>

#include "visitor.hpp"

namespace Choreo {

struct MemReuse;

// Synchronization recovery for compiler-planned buffer reuse (MARA).
//
// Dump-only milestone (plan: mara/cc27/plan/implementation-plan.md, C1/C2):
// reconstructs physical slots from MemReuse's post-coloring offsets, orders
// each slot's tenants with the HB graph, and reports handover candidates
// with the separating signal chain (the witness). No AST mutation yet;
// obligation derivation and insertion land behind the same entry point.
//
// Enable with the CHOREO_SYNC_RECOVERY env var (proper CCtx flag is a TODO).
struct SyncRecovery : public VisitorWithSymTab {
private:
  const MemReuse* mr = nullptr;
  AST::Node* root_ = nullptr;
  // Nodes to insert at a handover frontier, keyed by their anchor statement:
  // `before` for the producer-side sync that closes the writing tenant (the
  // fence and the bulk-group drain must run on the issuing threads), `after`
  // when the producer phase ends on an access rather than a sync.
  struct PendingInsert {
    ptr<AST::Node> node;
    bool after = false;
    // Preorder index of the producer's last access to the slot: a covering
    // operation must follow it, otherwise it does not publish / drain the
    // access the obligation is about.
    int not_before = -1;
    // The slot the insertion guards (empty for fences, which carry their own
    // frontier).
    std::string buffer;
    // A read-drain may be discharged by a synchronous engine read only when
    // no earlier async read of the same slot is outstanding.
    bool self_discharge = false;
  };
  std::map<const AST::Node*, std::vector<PendingInsert>> pending_inserts_;
  // Preorder index of every statement, from the liveness analyzer.
  std::map<const AST::Node*, int> preorder_index_;
  // Node ancestry during the splice traversal. At Visit(MultiNodes) time the
  // top of the stack is the list being spliced and the entries below it are
  // its ancestors, which makes the ancestor-list prefixes available as
  // must-dominating cover sites above the enclosing statement.
  std::vector<AST::Node*> visit_stack_;
  size_t inserted_fences_ = 0;
  size_t inserted_waits_ = 0;
  size_t elided_fences_ = 0;
  size_t elided_waits_ = 0;
  size_t self_waits_ = 0;
  size_t discharged_waits_ = 0;
  size_t total_slots_ = 0, total_handovers_ = 0, hb_merged_handovers_ = 0;
  // Debug aid: keep the raw insertion set (skip elimination) so the dump and
  // the emitted code can be attributed to synthesis alone.
  bool keep_insertions_ = false;

public:
  SyncRecovery() : VisitorWithSymTab("SyncRecovery") {}
  void Bind(const MemReuse* m) { mr = m; }

  bool RunOnProgramImpl(AST::Node& root) override;
  void RunRecovery();

  // Splices the pending insertions into the statement list that contains
  // their anchor statement.
  bool Visit(AST::MultiNodes& n) override;

private:
  // Queues `node` at the frontier anchored on `anchor`, collapsing duplicates
  // (the same fence entity or the same wait primitive at the same site).
  void AddInsert(const AST::Node* anchor, const ptr<AST::Node>& node,
                 bool after, int not_before = -1,
                 const std::string& buffer = {}, bool self_discharge = false);

  // Structured must-cover fold at splice time: list prefixes, ancestor-list
  // prefixes, loop summaries, and trigger-bound event waits. Only operations
  // that execute on every structured path to the anchor are accepted.
  void Eliminate(std::vector<PendingInsert>& inserts, const AST::MultiNodes& n,
                 size_t anchor_idx);

  // Maintain the ancestry stack used by the elimination fold. A few accept()
  // implementations emit BeforeVisit without AfterVisit (Expr and its
  // subclasses, AttributeExpr, DeviceFunctionDecl); keeping those off the
  // stack preserves its balance.
  static bool TracksAncestry(AST::Node& n) {
    return !isa<AST::Expr>(&n) && !isa<AST::AttributeExpr>(&n) &&
           !isa<AST::DeviceFunctionDecl>(&n);
  }
  bool BeforeVisitImpl(AST::Node& n) override {
    if (TracksAncestry(n)) visit_stack_.push_back(&n);
    return true;
  }
  bool AfterVisitImpl(AST::Node& n) override {
    if (!TracksAncestry(n)) return true;
    assert(!visit_stack_.empty() && visit_stack_.back() == &n &&
           "unbalanced ancestry stack");
    visit_stack_.pop_back();
    return true;
  }
};

} // end namespace Choreo

#endif // __CHOREO_SYNC_RECOVERY_HPP__
