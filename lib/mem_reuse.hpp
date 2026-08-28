#ifndef __CHOREO_MEM_REUSE_HPP__
#define __CHOREO_MEM_REUSE_HPP__

#include "ast.hpp"
#include "codegen.hpp"
#include "context.hpp"
#include "heap_simulator.hpp"
#include "liveness_analysis.hpp"
#include "symvals.hpp"
#include "typeresolve.hpp"
#include "types.hpp"
#include "visitor.hpp"

#include <cstdint>
#include <functional>
#include <numeric>

namespace Choreo {

// Analyze memory: storage, shape(size)
struct MemAnalyzer : public VisitorWithSymTab {
  int parallel_level;
  // NOTE: use paraby scope to distinguish different device functions.
  // If equal to co func name, indicate that not in device scope.
  std::string cur_dev_fname;

  // whether JIT memory reuse is needed.
  // dev_fname -> (sto -> need).
  std::map<std::string, std::map<Storage, bool>> sto_have_dyn;

  std::unordered_map<std::string, ValueItem> buf_size;
  std::unordered_map<std::string, Storage> buf_sto;
  std::unordered_map<std::string, std::string> buf_dev_func_name;
  std::unordered_map<std::string, size_t> buf_alignment;
  std::set<std::string> event_vars;

  MemAnalyzer() : VisitorWithSymTab("memanlz") {}
  ~MemAnalyzer() {}

private:
  bool BeforeVisitImpl(AST::Node& n) override;
  bool AfterVisitImpl(AST::Node&) override;
  bool Visit(AST::NamedVariableDecl& n) override;

  // NOTE: all nvd with ref note will be ignored in memory reuse.
  static bool IsRef(const AST::Node& n) { return n.HasNote("ref"); }
};

struct MemReuse : public VisitorWithSymTab {
private:
  LivenessAnalyzer la;
  MemAnalyzer ma;

  int parallel_level = 0;
  // NOTE: use paraby scope to distinguish different device functions.
  // If equal to co func name, indicate that not in device scope.
  std::string cur_dev_fname;
  using Range = LivenessAnalyzer::Range;
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
      std::sort(buffers.begin(), buffers.end(),
                [](const Buffer& a, const Buffer& b) {
                  return a.buffer_id < b.buffer_id;
                });
      std::sort(dynamic_buffers.begin(), dynamic_buffers.end(),
                [](const DBuffer& a, const DBuffer& b) {
                  return a.buffer_id < b.buffer_id;
                });
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
  std::map<std::string, size_t> shared_alignment_reqs;

  DevFuncMemReuseCtx& DFCtx(std::string dev_func_name = "") {
    if (dev_func_name == "") assert(cur_dev_fname != "");
    return df_ctxs[dev_func_name == "" ? cur_dev_fname : dev_func_name];
  }

  std::map<std::string, DevFuncMemReuseCtx>& DFCtxs() { return df_ctxs; }
  const std::map<std::string, DevFuncMemReuseCtx>& DFCtxs() const {
    return df_ctxs;
  }

  std::string GetFuncNameFromScopedName(const std::string& name) const {
    // indicate that it is a co function name
    if (!PrefixedWith(name, "::")) return name;
    return SplitFirst(name, "::");
  }

  std::string GetDeclDevFuncOfBuffer(std::string buf_name) const {
    return ma.buf_dev_func_name.at(buf_name);
  }

  void CollectSharedAlignmentRequirements(AST::Node& root);
  size_t SharedAlignmentForDevFunc(const std::string& df_name) const;
  size_t RequestedAlignmentForDevFunc(Storage sto,
                                      const std::string& df_name) const;
  size_t AlignmentForDevFunc(Storage sto, const std::string& df_name) const;

public:
  MemReuse() : VisitorWithSymTab("memreuse") {
    if (trace_visit) debug_visit = true;
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
  bool ShouldReuseStorage(Storage sto,
                          const std::string& dev_func_name = "") const;
  bool ShouldReuseBuffer(const std::string& buffer_id, Storage sto,
                         const std::string& dev_func_name) const;
  void Initialize();
  void AnalyzeMemOffset();
  void ProtoType(const std::string& dev_fname, DevFuncMemReuseCtx& ctx,
                 std::string idx_suffix);
  bool
  ValidateResult(const HeapSimulator::Result& res,
                 const HeapSimulator::Chunks& chunks,
                 HeapSimulator::HBOverride hb_override = nullptr,
                 HeapSimulator::HBMustInterfere hb_must_interfere = nullptr);
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
