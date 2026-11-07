#ifndef __CHOREO_SYNC_RECOVERY_HPP__
#define __CHOREO_SYNC_RECOVERY_HPP__

#include <map>
#include <set>

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
  };
  std::map<const AST::Node*, std::vector<PendingInsert>> pending_inserts_;
  // Preorder index of every statement, from the liveness analyzer.
  std::map<const AST::Node*, int> preorder_index_;
  size_t inserted_fences_ = 0;
  size_t inserted_waits_ = 0;
  size_t elided_fences_ = 0;
  size_t elided_waits_ = 0;
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
                 const std::string& buffer = {});

  // Availability fold over the enclosing statement list: drops insertions
  // covered by an operation that dominates their anchor (evaluate at splice
  // time, when the containing list is known).
  void Eliminate(std::vector<PendingInsert>& inserts, const AST::MultiNodes& n,
                 size_t anchor_idx);

  // RunOnProgramImpl skips the traversal entirely; these exist only to
  // satisfy the abstract base.
  bool BeforeVisitImpl(AST::Node&) override { return true; }
  bool AfterVisitImpl(AST::Node&) override { return true; }
};

} // end namespace Choreo

#endif // __CHOREO_SYNC_RECOVERY_HPP__
