#ifndef __CHOREO_GCU_FENCE_HPP__
#define __CHOREO_GCU_FENCE_HPP__

#include "target.hpp"
#include "types.hpp"

namespace Choreo {

/// GCU DMA fence-insertion table (plan section 4), keyed on the storage
/// direction of a DMA edge. Returns the fence requirements for the producer
/// (source) and consumer (destination) sites as lists of (space, entity, order)
/// triples; empty lists mean no fence. Multi-hop moves are decomposed before
/// this table is consulted.
///
/// Only RELEASE fences are needed on GCU: the reader side is strongly ordered
/// (local VDMEM and shared L2 are not per-thread caches, and the VLIW pipeline
/// issues loads in static order), so the ACQUIRE side is always elided.
inline FenceSelection SelectGCUDMAFences(Storage src, Storage dst) {
  using K = FenceKind;
  if (src == Storage::GLOBAL && dst == Storage::SHARED)
    return {{}, {K{Storage::SHARED, FenceEntity::DMA, FenceOrder::RELEASE}}};
  if (src == Storage::SHARED && dst == Storage::LOCAL)
    return {{K{Storage::LOCAL, FenceEntity::DMA, FenceOrder::RELEASE}},
            {K{Storage::LOCAL, FenceEntity::DMA, FenceOrder::RELEASE}}};
  if (src == Storage::LOCAL && dst == Storage::GLOBAL)
    return {{K{Storage::LOCAL, FenceEntity::THREADS, FenceOrder::RELEASE}}, {}};
  return {};
}

} // namespace Choreo

#endif // __CHOREO_GCU_FENCE_HPP__
