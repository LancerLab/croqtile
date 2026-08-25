#ifndef __CHOREO_HEAP_SIMULATOR_HPP__
#define __CHOREO_HEAP_SIMULATOR_HPP__

// Interval-coloring allocator shared by MemReuse (scratch-buffer coloring)
// and the DMA resource allocator (finite completion-slot coloring).
//
// A "chunk" is anything whose lifetime is a set of statement-id intervals and
// whose footprint is a fixed size (bytes for buffers, 1 for DMA slots).  Two
// chunks interfere when their intervals overlap; the allocator assigns each
// chunk an offset (slot) so interfering chunks never share one, while
// provably non-overlapping chunks may.

#include "liveness_analysis.hpp"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <vector>

namespace Choreo {

// A memory chunk (scratch buffer or DMA handle) whose lifetime is described
// by a set of statement-id intervals.
struct Buffer {
  using Range = LivenessAnalyzer::Range;

  size_t size;
  std::vector<Range> ranges;
  std::string buffer_id;

  bool Interfere(const Buffer& other) const {
    size_t i = 0, j = 0;
    while (i < ranges.size() && j < other.ranges.size()) {
      if (ranges[i].Overlaps(other.ranges[j])) return true;
      if (ranges[i].end < other.ranges[j].end)
        ++i;
      else
        ++j;
    }
    return false;
  }

  void Sort() { std::sort(ranges.begin(), ranges.end()); }
};

// Interval-coloring allocator.  Assigns each chunk an offset (slot) such that
// interfering chunks never overlap, while provably non-overlapping chunks may
// share memory.  Used by MemReuse for scratch memory and by the DMA resource
// allocator for finite completion slots.
struct HeapSimulator {
  using Chunk = Buffer;
  using Chunks = std::vector<Chunk>;

  // memory allocation result
  struct Result {
    // must use std::map for string keys to keep the order buffer_id!
    std::map<std::string, size_t> chunk_offsets; // offset of each buffer
    size_t heap_size;                            // total memory size
  };

  using HBOverride =
      std::function<bool(const std::string&, const std::string&)>;
  using HBMustInterfere =
      std::function<bool(const std::string&, const std::string&)>;

  // global decreasing size best fit allocate algorithm
  // (support arbitrary alignment)
  // hb_override: if set, called for interfering pairs. Returns true if the
  // HB analysis proves the pair is non-interfering (safe to overlap).
  // hb_must_interfere: if set, called for non-interfering pairs. Returns
  // true if multi-instance concurrency makes overlap unsafe.
  Result GlobalDecreasingSizeBestFitAllocate(
      const std::vector<Chunk>& chunks, size_t alignment = 0,
      HBOverride hb_override = nullptr,
      HBMustInterfere hb_must_interfere = nullptr) {
    Result result;
    result.heap_size = 0;

    size_t length = chunks.size();

    // sort by size descending, then by buffer_id ascending for stability
    std::vector<Chunk> sorted_chunks = chunks;
    std::sort(sorted_chunks.begin(), sorted_chunks.end(),
              [](const Chunk& a, const Chunk& b) {
                if (a.size != b.size) return a.size > b.size;
                return a.buffer_id < b.buffer_id;
              });

    // build interference graph - represent which buffers' lifetime overlap
    // TODO: O(n^2) maybe can be optimized
    std::vector<std::vector<bool>> interference_graph(
        length, std::vector<bool>(length, false));

    for (size_t i = 0; i < length; ++i)
      for (size_t j = i + 1; j < length; ++j) {
        if (sorted_chunks[i].Interfere(sorted_chunks[j])) {
          if (hb_override && hb_override(sorted_chunks[i].buffer_id,
                                         sorted_chunks[j].buffer_id))
            continue;
          interference_graph[i][j] = true;
          interference_graph[j][i] = true;
        } else if (hb_must_interfere &&
                   hb_must_interfere(sorted_chunks[i].buffer_id,
                                     sorted_chunks[j].buffer_id)) {
          interference_graph[i][j] = true;
          interference_graph[j][i] = true;
        }
      }

    // assign space for each buffer
    std::map<size_t, size_t> assigned_offsets;

    using Range = std::pair<size_t, size_t>;

    for (size_t i = 0; i < length; ++i) {
      const Chunk& chunk = sorted_chunks[i];

      // collect the allocated regions that overlap with the current buffer
      std::vector<Range> forbidden_ranges;
      for (size_t j = 0; j < i; ++j) {
        if (interference_graph[i][j] && assigned_offsets.count(j)) {
          // the current buffer and the buffer in j-th position overlap in
          // lifetime, so they can't be allocated to the same position
          forbidden_ranges.push_back(
              {assigned_offsets[j],
               assigned_offsets[j] + sorted_chunks[j].size});
        }
      }

      // sort the forbidden ranges by the start position
      std::sort(forbidden_ranges.begin(), forbidden_ranges.end());

      // merge the overlapping forbidden ranges
      if (!forbidden_ranges.empty()) {
        std::vector<Range> merged_ranges;
        merged_ranges.push_back(forbidden_ranges[0]);

        for (size_t j = 1; j < forbidden_ranges.size(); ++j) {
          auto& last = merged_ranges.back();
          const auto& current = forbidden_ranges[j];

          if (current.first <= last.second)
            last.second = std::max(last.second, current.second);
          else
            merged_ranges.push_back(current);
        }

        forbidden_ranges = std::move(merged_ranges);
      }

      // find the first valid position that satisfies the alignment
      // requirement
      size_t pos = 0;
      pos = AlignUp(pos, alignment);

      bool found_valid_position = false;
      for (size_t j = 0; j <= forbidden_ranges.size(); ++j) {
        // check if the current position is valid
        if (j == forbidden_ranges.size() ||
            pos + chunk.size <= forbidden_ranges[j].first) {
          found_valid_position = true;
          break;
        }

        // update the position to the current forbidden range
        pos = forbidden_ranges[j].second;
        // ensure the new position satisfies the alignment requirement
        pos = AlignUp(pos, alignment);
      }

      if (!found_valid_position) {
        // this should not happen in normal cases, because we always can find
        // a position after all forbidden ranges but just in case, we should
        // handle this situation
        errs() << "Error: Could not find valid position for buffer "
               << chunk.buffer_id << std::endl;
        // indicate allocation failed
        result.chunk_offsets[chunk.buffer_id] = -1;
        continue;
      }

      // assign the aligned offset to the current buffer
      size_t aligned_offset = pos;
      assigned_offsets.emplace(i, aligned_offset);

      // update the result
      result.chunk_offsets[chunk.buffer_id] = aligned_offset;
      result.heap_size =
          std::max(result.heap_size, aligned_offset + chunk.size);
    }

    // ensure the final heap size also satisfies the alignment requirement
    result.heap_size = AlignUp(result.heap_size, alignment);

    return result;
  }

  Result Allocate(const std::vector<Chunk>& chunks, int64_t alignment = 0,
                  HBOverride hb_override = nullptr,
                  HBMustInterfere hb_must_interfere = nullptr) {
    return GlobalDecreasingSizeBestFitAllocate(chunks, alignment, hb_override,
                                               hb_must_interfere);
  }

private:
  static size_t AlignUp(size_t x, size_t alignment) {
    if (alignment == 0) return x;
    return (x + alignment - 1) / alignment * alignment;
  }
};

} // namespace Choreo

#endif // __CHOREO_HEAP_SIMULATOR_HPP__
