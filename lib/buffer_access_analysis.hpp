#ifndef __CHOREO_BUFFER_ACCESS_ANALYSIS_HPP__
#define __CHOREO_BUFFER_ACCESS_ANALYSIS_HPP__

#include "access_tracking.hpp"
#include "buffer_access.hpp"
#include "visitor.hpp"

namespace Choreo {

// Records, in program order, every (READ | WRITE) access of a storage-bearing
// buffer over the normalized AST. Unlike LivenessAnalyzer (which treats a
// write to an existing buffer as a USE so it survives to the end of its live
// range), this pass records faithful producer -> consumer direction: a write
// is a WRITE, a read is a READ. The log is what a fence-insertion decision
// queries to find each DMA edge's producer (previous writer of the source
// buffer) and consumer (next reader of the destination buffer).
//
// Buffer identity is delegated to the shared AccessTracker so this pass and
// LivenessAnalyzer agree on what an alias / binding / no-storage-alias /
// future-buffer resolves to.
struct BufferAccessAnalyzer : public VisitorWithSymTab {
  using VarSet = std::set<std::string>;

  BufferAccessAnalyzer()
      : VisitorWithSymTab("bufferaccess"),
        tracker_([this](const std::string& n) { return InScopeNameForRef(n); },
                 [this](const std::string& n) { return InScopeName(n); }) {
    auto_declare_symbols = true;
    if (trace_visit) debug_visit = true;
    tracker_.debug = debug_visit;
  }
  ~BufferAccessAnalyzer() {}

  bool BeforeVisitImpl(AST::Node& n) override;
  bool AfterVisitImpl(AST::Node& n) override;

  bool Visit(AST::NamedVariableDecl& n) override;
  bool Visit(AST::Assignment& n) override;
  bool Visit(AST::DMA& n) override;
  bool Visit(AST::BufferMap& n) override;
  bool Visit(AST::MMA& n) override;
  bool Visit(AST::Call& n) override;
  bool Visit(AST::Rotate& n) override;

private:
  using DMABufInfo = AccessTracker::DMABufInfo;

  AccessTracker tracker_;
  std::vector<BufferAccessEvent> events_;
  std::unordered_map<std::string, AST::Node*> future_producers_;
  std::vector<ParallelLevel> pl_stack_;
  size_t order_ = 0;

  bool IsRef(const AST::Node& n) const { return n.HasNote("ref"); }

  ParallelLevel CurrentLevel() const {
    return pl_stack_.empty() ? ParallelLevel::NONE : pl_stack_.back();
  }

  std::string Scoped(const std::string& name) const {
    assert(!name.empty() && "expecting a valid name.");
    return PrefixedWith(name, "::") ? name : InScopeNameForRef(name);
  }

  // Resolve a (possibly unscoped / alias / future) name to the concrete
  // storage-bearing buffer(s) it denotes.
  VarSet ResolveBuffers(const std::string& name) const;

  // Storage of a canonical scoped buffer name, or Storage::NONE if the name
  // is not a storage-bearing buffer.
  Storage StorageOf(const std::string& scoped_name) const;

  // Record one access, resolving the name to its underlying buffer(s).
  void Record(AccessKind kind, AST::Node* stmt, const std::string& name,
              AccessEntity entity);
  void RecordAll(AccessKind kind, AST::Node* stmt, const VarSet& names,
                 AccessEntity entity);

  VarSet GetAllSymbolicOperands(const AST::Node* n) const {
    return tracker_.GetAllSymbolicOperands(n);
  }
};

} // namespace Choreo

#endif // __CHOREO_BUFFER_ACCESS_ANALYSIS_HPP__
