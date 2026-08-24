#ifndef __CHOREO_BUFFER_ACCESS_HPP__
#define __CHOREO_BUFFER_ACCESS_HPP__

#include "parallel_level.hpp"

#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace Choreo {

namespace AST {
struct Node;
}

// Direction of a buffer access.
enum class AccessKind { READ, WRITE };

// The agent that performs a buffer access. THREADS is the executing thread
// stream, DMA/TMA are data-movement engines, MMA is the matrix accelerator.
// This is kept independent of FenceEntity so the header only depends on the
// early Storage / ParallelLevel part of types.hpp and can be stored directly
// on CompilationContext without pulling in the visitor / AST headers.
enum class AccessEntity { THREADS, DMA, TMA, MMA };

inline static const std::string STR(AccessEntity e) {
  switch (e) {
  case AccessEntity::THREADS: return "THREADS";
  case AccessEntity::DMA: return "DMA";
  case AccessEntity::TMA: return "TMA";
  case AccessEntity::MMA: return "MMA";
  }
  return "UNKNOWN";
}

// One (READ | WRITE) access of a storage-bearing buffer, in program order.
struct BufferAccessEvent {
  AccessKind kind = AccessKind::READ;
  AST::Node* stmt = nullptr; // the statement node that performs the access
  std::string buffer;        // canonical scoped buffer name
  Storage storage = Storage::NONE;
  AccessEntity entity = AccessEntity::THREADS;
  ParallelLevel level = ParallelLevel::NONE; // enclosing parallel level
  size_t order = 0;                          // global program order
};

// The buffer access log produced by BufferAccessAnalyzer and consumed by
// FenceInsertion. It is the shared statement-keyed (READ | WRITE) record that
// a fence-insertion decision queries to find each DMA edge's producer (the
// previous writer of the source buffer) and consumer (the next reader of the
// destination buffer).
struct BufferAccessLog {
  // Ordered list of every buffer access in the program.
  std::vector<BufferAccessEvent> events;
  // scoped future name -> set of (src, dst) scoped buffer pairs it transfers.
  std::unordered_map<std::string, std::set<std::pair<std::string, std::string>>>
      future_buffers;
  // scoped future name -> the DMA node that produces it.
  std::unordered_map<std::string, AST::Node*> future_producers;
};

} // namespace Choreo

#endif // __CHOREO_BUFFER_ACCESS_HPP__
