#ifndef __CHOREO_FENCE_INSERTION_HPP__
#define __CHOREO_FENCE_INSERTION_HPP__

#include "buffer_access.hpp"
#include "visitor.hpp"

#include <string>
#include <unordered_map>
#include <vector>

namespace Choreo {

// Stage 3 of automatic DMA fence insertion. Consumes the BufferAccessLog
// recorded by BufferAccessAnalyzer (whose buffer identity comes from the
// shared Stage-2 AccessTracker) and, for every DMA edge, computes the fence
// requirements from the target's SelectDMAFences table, elides fences already
// covered by same-engine in-order execution, and attaches the producer /
// consumer fence annotation to the AST::DMA (producer site) and AST::Wait
// (consumer site) nodes.
//
// Nothing emits a fence yet. The annotation is observable via
// --dump-fence-insertion and is consumed by the codegen sink in a later stage.
struct FenceInsertion : public VisitorWithSymTab {
  FenceInsertion();
  ~FenceInsertion() {}

  bool BeforeVisitImpl(AST::Node& n) override;
  bool AfterVisitImpl(AST::Node&) override { return true; }

  bool Visit(AST::DMA& n) override;
  bool Visit(AST::Wait& n) override;

private:
  bool enabled_;
  bool dump_;

  // Built lazily on the first Program visit (the log is only populated after
  // BufferAccessAnalyzer runs, which is later than this object is built).
  bool indexed_ = false;
  // Scoped buffer name -> ordered access events on that buffer.
  std::unordered_map<std::string, std::vector<const BufferAccessEvent*>>
      by_buffer_;
  // DMA statement node -> its own source (READ) / destination (WRITE) event.
  std::unordered_map<const AST::Node*, const BufferAccessEvent*> dma_src_;
  std::unordered_map<const AST::Node*, const BufferAccessEvent*> dma_dst_;
  // Scoped future name -> consumer fences to attach at the matching wait.
  std::unordered_map<std::string, std::vector<FenceKind>> consumer_by_future_;

  void BuildIndex();
  const BufferAccessEvent* PrevWrite(const std::string& buffer,
                                     size_t order) const;
  const BufferAccessEvent* NextRead(const std::string& buffer,
                                    size_t order) const;
  static std::string JoinKinds(const std::vector<FenceKind>& kinds);
  void Count(const std::vector<FenceKind>& producer,
             const std::vector<FenceKind>& consumer);
};

} // namespace Choreo

#endif // __CHOREO_FENCE_INSERTION_HPP__
