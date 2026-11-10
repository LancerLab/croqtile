#ifndef __CHOREO_PARALLEL_LEVEL_HPP__
#define __CHOREO_PARALLEL_LEVEL_HPP__

// The two memory-hierarchy enums shared by the buffer-access log and the rest
// of the compiler. Kept in a leaf header (no includes) so headers such as
// buffer_access.hpp can depend on them without pulling in the full types.hpp
// (which, via context.hpp, would create an include cycle).

namespace Choreo {
enum class Storage {
  REG, /* register, normally not explicit */
  LOCAL,
  SHARED,
  GROUP_SHARED, /* shared<group>: visible within a GROUP */
  GLOBAL /*device global*/,
  NODE /*cluster node*/,
  DEFAULT,
  NONE
};
enum class ParallelLevel {
  THREAD,
  GROUP,
  GROUPx4,
  BLOCK,
  CLUSTER, /* thread block cluster (TBC), above block */
  DEVICE,
  TERM /* terminal machine in cluster*/,
  SEQ,
  NONE /*bottom*/,
  UNKNOWN /*top*/
};
// The data-movement engine class of a DMA statement. DMA is the generic
// (sync/async) copy engine; TMA is the tensor-memory bulk-copy engine whose
// reachable storage tiers differ (it only moves global <-> shared). Targets
// use this to declare per-type storage-pair legality via
// Target::IsDMAStorageSupported.
enum class DMAKind {
  DMA,
  TMA,
};
} // namespace Choreo

#endif // __CHOREO_PARALLEL_LEVEL_HPP__
