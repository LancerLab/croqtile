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
// Runtime query function for a parallel level, e.g. `max_block_parallel_by`.
// `GROUPx4` maps to `group4` because `group-4` is not a valid C++ identifier.
// Levels without a capacity query map to `none`.
inline const char* MaxParallelByFunc(ParallelLevel lvl) {
  switch (lvl) {
  case ParallelLevel::BLOCK: return "max_block_parallel_by";
  case ParallelLevel::THREAD: return "max_thread_parallel_by";
  case ParallelLevel::GROUP: return "max_group_parallel_by";
  case ParallelLevel::GROUPx4: return "max_group4_parallel_by";
  case ParallelLevel::CLUSTER: return "max_cluster_parallel_by";
  case ParallelLevel::DEVICE: return "max_device_parallel_by";
  default: return "max_none_parallel_by";
  }
}
} // namespace Choreo

#endif // __CHOREO_PARALLEL_LEVEL_HPP__
