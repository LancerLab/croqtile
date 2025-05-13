#ifndef __CHOREO_MEM_REUSE_HPP__
#define __CHOREO_MEM_REUSE_HPP__

#include "ast.hpp"
#include "codegen.hpp"
#include "context.hpp"
#include "ginac/ginac.h"
#include "liveness_analysis.hpp"
#include "typeresolve.hpp"
#include "types.hpp"
#include "visitor.hpp"

#include <cstdint>
#include <numeric>

namespace Choreo {

// Analyze memory: storage, shape(size)
struct MemAnalyzer : public VisitorWithSymTab {
  using Symbol = GiNaC::symbol;
  using SymExpr = GiNaC::ex;

  // whether JIT memory reuse is needed
  bool have_dynamic_shape = false;

  std::map<std::string, Symbol> symbol_map;
  std::map<std::string, SymExpr> sym_expr_map;

  // using BSize = std::variant<size_t, SymExpr>;
  using BSize = std::variant<size_t, std::string>;
  std::unordered_map<std::string, BSize> buf_size;
  std::unordered_map<std::string, Storage> buf_sto;
  LivenessAnalyzer::VarSet event_vars;

  MemAnalyzer() : VisitorWithSymTab("memanlz", CCtx().GetGlobalSymbolTable()) {}
  ~MemAnalyzer() {}

private:
  bool BeforeVisitImpl(AST::Node& n) override;
  bool AfterVisitImpl(AST::Node&) override { return true; }
  bool Visit(AST::NamedVariableDecl& n) override;

  static inline bool IsRef(const AST::Node& n) {
    return n.GetNote().find("ref") != std::string::npos;
  }
  static inline std::string ExSTR(const SymExpr& sym_expr) {
    std::ostringstream oss;
    oss << sym_expr;
    return oss.str();
  }
  SymExpr StringifyOpFromSymExpr(const SymExpr& sym_expr_l,
                                 const std::string& op,
                                 const SymExpr& sym_expr_r);
  SymExpr GetSymExprFromStr(std::string str);
  SymExpr GetSymExprFromSizeExpr(std::string size_expr);
};

struct MemReuse : public VisitorWithSymTab {
private:
  std::string cur_func_name;
  const LivenessAnalyzer& la;
  const MemAnalyzer& ma;

  int parallel_level = 0;
  int max_parallel_level = 0;

  std::map<std::string, size_t> mem_offset;

  struct SpmSize {
    size_t local_spm_size;
    size_t shared_spm_size;
  };
  std::map<std::string, SpmSize> spm_size_map;

  // update when entering co func.
  std::string local_spm_name;
  std::string shared_spm_name;

  struct Buffer {
    size_t size;
    size_t start_time;
    size_t end_time;
    std::string buffer_id;
  };
  struct DBuffer {
    std::string size;
    size_t start_time;
    size_t end_time;
    std::string buffer_id;
  };
  std::vector<Buffer> buffers;
  std::vector<DBuffer> dynamic_buffers;

  struct HeapSimulator {
  public:
    using Chunk = Buffer;
    using Chunks = std::vector<Chunk>;

    // memory allocation result
    struct Result {
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

      for (size_t i = 0; i < length; ++i) {
        for (size_t j = i + 1; j < length; ++j) {
          if (sorted_chunks[i].start_time <= sorted_chunks[j].end_time &&
              sorted_chunks[j].start_time <= sorted_chunks[i].end_time) {
            interference_graph[i][j] = true;
            interference_graph[j][i] = true;
          }
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
          std::cerr << "Error: Could not find valid position for buffer "
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
  MemReuse(const LivenessAnalyzer& la, const MemAnalyzer& ma)
      : VisitorWithSymTab("memreuse", CCtx().GetGlobalSymbolTable()), la(la),
        ma(ma) {
    if (trace_visit) debug_visit = true;
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
  void ProtoType();
  bool ValidateResult(const HeapSimulator::Result& res,
                      const HeapSimulator::Chunks& chunks);
  void ApplyMemOffset(AST::NamedVariableDecl& n, Storage sto);
};

} // end namespace Choreo

#endif // __CHOREO_MEM_REUSE_HPP__
