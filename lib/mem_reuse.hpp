#ifndef __CHOREO_MEM_REUSE_HPP__
#define __CHOREO_MEM_REUSE_HPP__

#include "ast.hpp"
#include "codegen.hpp"
#include "context.hpp"
#include "liveness_analysis.hpp"
#include "symvals.hpp"
#include "typeresolve.hpp"
#include "types.hpp"
#include "visitor.hpp"

#include <cstdint>
#include <numeric>

namespace Choreo {

// Analyze memory: storage, shape(size)
struct MemAnalyzer : public VisitorWithSymTab {
  int parallel_level;
  // NOTE: use paraby scope to distinguish different device functions.
  // If equal to co func name, indicate that not in device scope.
  std::string cur_dev_func_name;

  // whether JIT memory reuse is needed.
  // the key is dec func name, the val is false by default.
  std::map<std::string, bool> have_dynamic_shape;

  std::unordered_map<std::string, ValueItem> buf_size;
  std::unordered_map<std::string, Storage> buf_sto;
  std::unordered_map<std::string, std::string> buf_dev_func_name;
  std::set<std::string> event_vars;

  MemAnalyzer() : VisitorWithSymTab("memanlz") {}
  ~MemAnalyzer() {}

private:
  bool BeforeVisitImpl(AST::Node& n) override;
  bool AfterVisitImpl(AST::Node&) override;
  bool Visit(AST::NamedVariableDecl& n) override;

  static bool IsRef(const AST::Node& n) { return n.HasNote("ref"); }
};

struct MemReuse : public VisitorWithSymTab {
private:
  LivenessAnalyzer la;
  MemAnalyzer ma;

  int parallel_level = 0;
  int max_parallel_level = 0;
  // NOTE: use paraby scope to distinguish different device functions.
  // If equal to co func name, indicate that not in device scope.
  std::string cur_dev_func_name;
  using Range = LivenessAnalyzer::Range;
  struct Buffer {
    size_t size;
    std::vector<Range> ranges;
    std::string buffer_id;
    bool Interfere(const Buffer& other) const {
      for (const auto& a : this->ranges)
        for (const auto& b : other.ranges)
          if (a.Overlaps(b)) return true;
      return false;
    }
    void Sort() { std::sort(ranges.begin(), ranges.end()); }
  };
  struct DBuffer {
    std::string size;
    std::vector<Range> ranges;
    std::string buffer_id;
    void Sort() { std::sort(ranges.begin(), ranges.end()); }
  };

  struct DevFuncMemReuseCtx {
    std::string local_spm_name;
    std::string shared_spm_name;
    size_t local_spm_size;
    size_t shared_spm_size;
    std::vector<Buffer> buffers;
    std::vector<DBuffer> dynamic_buffers;
    std::map<std::string, size_t> mem_offset;
    void SortBuffers() {
      for (auto& b : buffers) b.Sort();
      for (auto& b : dynamic_buffers) b.Sort();
    }
  };

  static std::string RangesSTR(std::vector<Range> ranges, char lp = '[',
                               char rp = ']') {
    std::ostringstream oss;
    auto it = ranges.begin();
    if (it != ranges.end()) {
      oss << lp << it->start << "," << it->end << rp;
      ++it;
    }
    for (; it != ranges.end(); ++it)
      oss << ", " << lp << it->start << "," << it->end << rp;
    return oss.str();
  }

  std::map<std::string, DevFuncMemReuseCtx> df_ctxs;

  DevFuncMemReuseCtx& DFCtx(std::string dev_func_name = "") {
    if (dev_func_name == "") assert(cur_dev_func_name != "");
    return df_ctxs[dev_func_name == "" ? cur_dev_func_name : dev_func_name];
  }

  std::map<std::string, DevFuncMemReuseCtx>& DFCtxs() { return df_ctxs; }
  const std::map<std::string, DevFuncMemReuseCtx>& DFCtxs() const {
    return df_ctxs;
  }

  std::string GetFuncNameFromScopedName(const std::string& name) const {
    // indicate that it is a co function name
    if (!PrefixedWith(name, "::")) return name;
    return SplitStringByDelimiter(name, "::", true)[0];
  }

  std::string GetDeclDevFuncOfBuffer(std::string buf_name) const {
    return ma.buf_dev_func_name.at(buf_name);
  }

  struct HeapSimulator {
    using Chunk = Buffer;
    using Chunks = std::vector<Chunk>;

    // memory allocation result
    struct Result {
      // must use std::map for string keys to keep the order buffer_id!
      std::map<std::string, size_t> chunk_offsets; // offset of each buffer
      size_t heap_size;                            // total memory size
    };

    // global decreasing size best fit allocate algorithm
    // (support arbitrary alignment)
    Result GlobalDecreasingSizeBestFitAllocate(const std::vector<Chunk>& chunks,
                                               size_t alignment = 0) {
      Result result;
      result.heap_size = 0;

      size_t length = chunks.size();

      // sort by size in descending order
      // TODO: use idx or pointer rather than Chunk
      std::vector<Chunk> sorted_chunks = chunks;
      std::sort(sorted_chunks.begin(), sorted_chunks.end(),
                [](const Chunk& a, const Chunk& b) { return a.size > b.size; });

      // build interference graph - represent which buffers' lifetime overlap
      // TODO: O(n^2) maybe can be optimized
      std::vector<std::vector<bool>> interference_graph(
          length, std::vector<bool>(length, false));

      for (size_t i = 0; i < length; ++i)
        for (size_t j = i + 1; j < length; ++j)
          if (sorted_chunks[i].Interfere(sorted_chunks[j])) {
            interference_graph[i][j] = true;
            interference_graph[j][i] = true;
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

    Result Allocate(const std::vector<Chunk>& chunks, int64_t alignment = 0) {
      return GlobalDecreasingSizeBestFitAllocate(chunks, alignment);
    }
  };

public:
  MemReuse() : VisitorWithSymTab("memreuse") {
    if (trace_visit) debug_visit = true;
    // TODO: maybe should do the same for other passes.
    if (disabled) CCtx().SetMemReuse(false);
  }
  ~MemReuse() {}

private:
  bool BeforeVisitImpl(AST::Node&) override;
  bool AfterVisitImpl(AST::Node&) override;

  static int Size_t2Int(size_t s) {
    if (s <= (size_t)std::numeric_limits<int>::max())
      return static_cast<int>(s);
    choreo_unreachable("size_t to int conversion failed, val: " +
                       std::to_string(s));
  }

  static size_t AlignUp(size_t x, size_t alignment) {
    if (alignment == 0) return x;
    return (x + alignment - 1) / alignment * alignment;
  }

  bool Visit(AST::NamedVariableDecl&) override;
  void Initialize();
  void AnalyzeMemOffset();
  void ProtoType(const std::string& dev_fname, DevFuncMemReuseCtx& ctx,
                 std::string idx_suffix);
  bool ValidateResult(const HeapSimulator::Result& res,
                      const HeapSimulator::Chunks& chunks);
  void ApplyMemOffset(AST::NamedVariableDecl& n, Storage sto);
  bool RunOnProgramImpl(AST::Node& root) override;
};

class MemoryReuse : public VisitorGroup {
private:
  MemReuse mr;

public:
  MemoryReuse() : VisitorGroup("MemoryReuse", mr) {}
};

} // end namespace Choreo

#endif // __CHOREO_MEM_REUSE_HPP__
