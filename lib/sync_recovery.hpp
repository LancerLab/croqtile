#ifndef __CHOREO_SYNC_RECOVERY_HPP__
#define __CHOREO_SYNC_RECOVERY_HPP__

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

public:
  SyncRecovery() : VisitorWithSymTab("SyncRecovery") {}
  void Bind(const MemReuse* m) { mr = m; }

  bool RunOnProgramImpl(AST::Node& root) override;
  void RunRecovery();

private:
  // RunOnProgramImpl skips the traversal entirely; these exist only to
  // satisfy the abstract base.
  bool BeforeVisitImpl(AST::Node&) override { return true; }
  bool AfterVisitImpl(AST::Node&) override { return true; }
};

} // end namespace Choreo

#endif // __CHOREO_SYNC_RECOVERY_HPP__
