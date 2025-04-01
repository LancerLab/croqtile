#ifndef __CHOREO_MEM_REUSE_HPP__
#define __CHOREO_MEM_REUSE_HPP__

#include "ast.hpp"
#include "context.hpp"
#include "typeresolve.hpp"
#include "types.hpp"
#include "visitor.hpp"

#include "liveness_analysis.hpp"
#include <cstdint>

namespace Choreo {

struct MemReuse : public VisitorWithSymTab {
  using StrUintMap = std::unordered_map<std::string, size_t>;

private:
  TypeConstraints type_equals{this};
  const LivenessAnalyzer& la;

  int parallel_level = 0;
  int max_parallel_level = 0;

  StrUintMap local_mem_offsets;
  StrUintMap shared_mem_offsets;

  std::string local_spm_name;
  std::string shared_spm_name;

  const std::unordered_map<std::string, Storage>& buf2sto;
  // TODO: extend this to support multiple co functions
  // cause the spm should be allocated for each co function
  const StrUintMap& buffer_size;
  StrUintMap buffer_alloc_time;
  StrUintMap buffer_release_time;

  struct HeapSimulator {
  public:
    struct Chunk {
      size_t size;
      size_t start_time;
      size_t end_time;
      std::string buffer_id;
    };

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

      assert(alignment >= 0 && "Alignment must be a positive integer");

      size_t size = chunks.size();

      auto AlignUp = [alignment](size_t x) -> size_t {
        if (alignment == 0) return x;
        return (x + alignment - 1) / alignment * alignment;
      };

      // sort by size in descending order
      // TODO: use idx or pointer rather than Chunk
      std::vector<Chunk> sorted_chunks = chunks;
      std::sort(sorted_chunks.begin(), sorted_chunks.end(),
                [](const Chunk& a, const Chunk& b) { return a.size > b.size; });

      // build interference graph - represent which buffers' lifetime overlap
      // TODO: O(n^2) maybe can be optimized
      std::vector<std::vector<bool>> interference_graph(
          size, std::vector<bool>(size, false));

      for (size_t i = 0; i < size; ++i) {
        for (size_t j = i + 1; j < size; ++j) {
          if (sorted_chunks[i].start_time <= sorted_chunks[j].end_time &&
              sorted_chunks[j].start_time <= sorted_chunks[i].end_time) {
            interference_graph[i][j] = true;
            interference_graph[j][i] = true;
          }
        }
      }

      // assign space for each buffer
      std::unordered_map<size_t, size_t> assigned_offsets;

      using Range = std::pair<size_t, size_t>;

      for (size_t i = 0; i < size; ++i) {
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
        pos = AlignUp(pos);

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
          pos = AlignUp(pos);
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
      result.heap_size = AlignUp(result.heap_size);

      return result;
    }

    Result Allocate(const std::vector<Chunk>& chunks, int64_t alignment = 0) {
      return GlobalDecreasingSizeBestFitAllocate(chunks, alignment);
    }
  };

private:
  bool BeforeVisitImpl(AST::Node&) override;
  bool AfterVisitImpl(AST::Node&) override;

  virtual void TraceEachVisit(AST::Node& n, bool detail = false,
                              const std::string& m = "") const {
    if (!trace_visit) return;
    if (detail)
      dbgs() << m << STR(n) << "\n";
    else
      dbgs() << m << n.TypeNameString() << "\n";
  }

public:
  MemReuse(const LivenessAnalyzer& la)
      : VisitorWithSymTab("memreuse", CCtx().GetGlobalSymbolTable()), la(la),
        buf2sto(la.Buf2Sto()), buffer_size(la.BufSizes()) {
    if (trace_visit) debug_visit = true; // force debug when tracing
    if (debug_visit) type_equals.SetDebug(true);

    const auto& var_ranges = la.VarRanges();
    for (const auto& [sname, size] : buffer_size) {
      auto ranges = var_ranges.at(sname);
      // For now, there is no case that a var is used in multiple ranges.
      // Because there is no reassignment.
      if (ranges.Values().size() == 0) {
        dbgs() << "Warning: buffer " << sname << " is never used!\n";
        continue;
      }
      if (ranges.Values().size() > 1) {
        dbgs() << "Warning: buffer " << sname
               << " is used in multiple ranges:\n";
        for (const auto& r : ranges.Values())
          dbgs() << "\t[" << r.start << ", " << r.end << "]\n";
        assert(ranges.Values().size() == 1);
      }
      buffer_alloc_time[sname] = ranges.Values()[0].start;
      buffer_release_time[sname] = ranges.Values()[0].end;
    }
  }
  ~MemReuse() {}

  void AnalyzeMemOffset() {
    ProtoType();
    // TODO: do validation after offset is calculated
  }

  void ProtoType() {
    // define local and shared memory chunks
    std::vector<HeapSimulator::Chunk> local_chunks;
    std::vector<HeapSimulator::Chunk> shared_chunks;

    // convert buffer_size to HeapSimulator::Chunk format
    for (const auto& [sname, size] : buffer_size) {
      if (buffer_release_time.count(sname) == 0) continue;
      HeapSimulator::Chunk chunk{size, buffer_alloc_time.at(sname),
                                 buffer_release_time.at(sname), sname};
      if (buf2sto.at(sname) == Storage::LOCAL) {
        local_chunks.push_back(chunk);
      } else if (buf2sto.at(sname) == Storage::SHARED) {
        shared_chunks.push_back(chunk);
      }
    }

    HeapSimulator simulator;

    // allocate local memory
    if (!local_chunks.empty()) {
      HeapSimulator::Result local_result =
          simulator.Allocate(local_chunks, 512);
      // TODO: should we align the size?
      CCtx().SetLocalSPMSize(local_result.heap_size);
      for (const auto& [buffer_id, offset] : local_result.chunk_offsets) {
        local_mem_offsets.emplace(buffer_id, offset);
      }

      VST_DEBUG(dbgs() << "Local memory usage: " << local_result.heap_size
                       << " bytes\n");
    }

    // allocate shared memory
    if (!shared_chunks.empty()) {
      // TODO: shared mem alignment?
      HeapSimulator::Result shared_result = simulator.Allocate(shared_chunks);
      // TODO: should we align the size?
      CCtx().SetSharedSPMSize(shared_result.heap_size);

      for (const auto& [buffer_id, offset] : shared_result.chunk_offsets) {
        shared_mem_offsets.emplace(buffer_id, offset);
      }

      VST_DEBUG(dbgs() << "Shared memory usage: " << shared_result.heap_size
                       << " bytes\n");
    }
  }

  void ApplyMemOffset(AST::NamedVariableDecl& n, Storage sto) {
    // TODO: before apply, do validation!
    // maybe should discretize the ranges of the buffer both in analysis and
    // validation.
    assert(sto == Storage::LOCAL || sto == Storage::SHARED);
    auto sname = InScopeName(n.name_str);
    auto spm_name = (sto == Storage::LOCAL ? local_spm_name : shared_spm_name);
    auto mem_offsets =
        (sto == Storage::LOCAL ? local_mem_offsets : shared_mem_offsets);
    VST_DEBUG({ dbgs() << STR(sto) << " buffer: " << sname << ".\n\t"; });

    if (!mem_offsets.count(sname)) {
      VST_DEBUG(dbgs() << "has no valid reuse offset!\n");
      return;
    }
    VST_DEBUG({
      dbgs() << "using spm:   " << spm_name
             << "\n\twith offset: " << mem_offsets.at(sname) << "\n";
    });
    n.note.append("reuse, " + spm_name + ", ");
    n.note.append("offset, " + std::to_string(mem_offsets.at(sname)) + ", ");
  }

  bool Visit(AST::MultiNodes&) override;
  bool Visit(AST::MultiValues&) override;
  bool Visit(AST::IntLiteral&) override;
  bool Visit(AST::FloatLiteral&) override;
  bool Visit(AST::StringLiteral&) override;
  bool Visit(AST::Boolean&) override;
  bool Visit(AST::Expr&) override;
  bool Visit(AST::MultiDimSpans&) override;
  bool Visit(AST::NamedTypeDecl&) override;
  bool Visit(AST::NamedVariableDecl&) override;
  bool Visit(AST::IntTuple&) override;
  bool Visit(AST::Assignment&) override;
  bool Visit(AST::IntIndex&) override;
  bool Visit(AST::DataType&) override;
  bool Visit(AST::Identifier&) override;
  bool Visit(AST::Parameter&) override;
  bool Visit(AST::ParamList&) override;
  bool Visit(AST::ParallelBy&) override;
  bool Visit(AST::WhereBind&) override;
  bool Visit(AST::WithIn&) override;
  bool Visit(AST::WithBlock&) override;
  bool Visit(AST::Memory&) override;
  bool Visit(AST::SpanAs&) override;
  bool Visit(AST::DMA&) override;
  bool Visit(AST::ChunkAt&) override;
  bool Visit(AST::Wait&) override;
  bool Visit(AST::Call&) override;
  bool Visit(AST::Rotate&) override;
  bool Visit(AST::Select&) override;
  bool Visit(AST::Return&) override;
  bool Visit(AST::LoopRange&) override;
  bool Visit(AST::ForeachBlock&) override;
  bool Visit(AST::InThreadsBlock&) override;
  bool Visit(AST::IncrementBlock&) override;
  bool Visit(AST::FunctionDecl&) override;
  bool Visit(AST::ChoreoFunction&) override;
  bool Visit(AST::CppSourceCode&) override;
  bool Visit(AST::Program&) override;

  bool HasError() override;
};

} // end namespace Choreo

#endif // __CHOREO_MEM_REUSE_HPP__
