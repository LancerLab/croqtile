#ifndef __CHOREO_CONTEXT_HPP__
#define __CHOREO_CONTEXT_HPP__

// shared global context for a compilation process

#include "assess.hpp"
#include "fragment_layout.hpp"
#include "io.hpp"
#include "loc.hpp"
#include "symvals.hpp"
#include "target.hpp"
#include "types.hpp"
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <unordered_map>
#include <utility>

extern Choreo::location loc;

namespace Choreo {

namespace AST {
struct Node;
}

class PlDepthMap;

enum class OutputKind {
  PreProcessedCode,
  TargetSourceCode,
  // Hetero offload: emit a standalone translation unit for the offload .o.
  // Includes the device kernel (ds) and the target launch entry (hs) that
  // defines __hetero_<device>_... (alloc, dispatch, sync). Does NOT include
  // hetero orchestration host code or main() - those are in the hetero host .o.
  DeviceSourceOnly,
  TargetModule,
  TargetAssembly,
  TargetExecutable,
  ShellScript,
};

enum class DebugLinePathMode {
  WorkspaceRelative,
  Absolute,
};

inline bool RequiresE2ECompilation(OutputKind ok) {
  switch (ok) {
  case OutputKind::TargetModule:
  case OutputKind::TargetAssembly:
  case OutputKind::TargetExecutable:
  case OutputKind::ShellScript: return true;
  default: break;
  }
  return false;
}

inline static const std::string STR(OutputKind ok) {
  switch (ok) {
  case OutputKind::PreProcessedCode: return "PreProcessedCode";
  case OutputKind::TargetSourceCode: return "TargetSourceCode";
  case OutputKind::DeviceSourceOnly: return "DeviceSourceOnly";
  case OutputKind::TargetModule: return "TargetModule";
  case OutputKind::TargetAssembly: return "TargetAssembly";
  case OutputKind::TargetExecutable: return "TargetExecutable";
  case OutputKind::ShellScript: return "ShellScript";
  default: choreo_unreachable("Unsupported output kind.");
  }
  return "";
}

enum DMABufferKind {
  DOK_UNKNOWN,
  DOK_SYMBOL,
  DOK_CHUNK,
};

inline const char* STR(DMABufferKind dok) {
  switch (dok) {
  case DOK_UNKNOWN: return "UNKNOWN";
  case DOK_SYMBOL: return "SYMBOL";
  case DOK_CHUNK: return "CHUNK";
  default: choreo_unreachable("Unsupported operand kind.");
  }
  return "";
}

struct DMABufferInfo {
  std::string buffer; // buffer name, if explicitly named
  DMABufferKind from_kind = DOK_UNKNOWN;
  DMABufferKind to_kind = DOK_UNKNOWN;
};

// per-function(name) future-buffer info
using FutureBufferInfo = std::map<std::string, DMABufferInfo>;
using FBItemInfo = FutureBufferInfo::value_type;

inline const std::string STR(const FBItemInfo& fbi) {
  std::ostringstream oss;
  oss << "[Future] " << fbi.first << " (buffer: " << fbi.second.buffer << "), "
      << STR(fbi.second.from_kind) << " -> " << STR(fbi.second.to_kind);
  return oss.str();
}

inline const std::string STR(const FutureBufferInfo& fbi) {
  std::ostringstream oss;
  oss << "Future-Buffers:\n";
  for (auto& item : fbi) oss << STR(item) << "\n";
  return oss.str();
}

inline bool FBIContainsBuffer(const FutureBufferInfo& buffer_info,
                              const std::string& name) {
  for (const auto& item : buffer_info) {
    if (item.second.buffer == name) { return true; }
  }
  return false;
}

struct OptimizedValues {
private:
  ValueList val_exprs;
  ValueList ub_exprs;
  ValueItem size_expr = GetInvalidValueItem();

public:
  void SetVal(ValueItem vi) {
    val_exprs.clear();
    val_exprs.push_back(vi);
  }
  void SetVals(const ValueList& vis) {
    val_exprs.clear();
    for (auto vi : vis) {
      if (!IsValidValueItem(vi))
        choreo_unreachable("invalid value item.");
      else
        val_exprs.push_back(vi->Normalize());
    }
  }
  void SetSize(ValueItem vi) {
    if (IsValidValueItem(vi))
      size_expr = vi->Normalize();
    else
      choreo_unreachable("invalid value item.");
  }
  void SetUBound(ValueItem vi) {
    ub_exprs.clear();
    ub_exprs.push_back(vi);
  }
  void SetUBounds(const ValueList& vis) {
    ub_exprs.clear();
    for (auto vi : vis) {
      if (IsValidValueItem(vi))
        ub_exprs.push_back(vi->Normalize());
      else
        choreo_unreachable("invalid value item.");
    }
  }
  bool HasVal() const { return val_exprs.size() == 1; }
  bool HasVals() const { return !val_exprs.empty(); }
  bool HasSize() const { return IsValidValueItem(size_expr); }
  bool HasUBound() const { return ub_exprs.size() == 1; }
  bool HasUBounds() const { return !ub_exprs.empty(); }
  const ValueItem GetVal() const {
    if (val_exprs.size() != 1)
      choreo_unreachable("not single value item.");
    else if (!IsValidValueItem(val_exprs[0]))
      choreo_unreachable("invalid value item.");
    return val_exprs[0];
  }
  const ValueList& GetVals() const {
    if (!HasVals()) choreo_unreachable("have no value.");
    return val_exprs;
  }
  ValueList& GetVals() {
    if (!HasVals()) choreo_unreachable("have no value.");
    return val_exprs;
  }
  ValueItem GetSize() const { return size_expr; }
  const ValueItem GetUBound() const {
    if (ub_exprs.size() != 1)
      choreo_unreachable("not single value item.");
    else if (!IsValidValueItem(ub_exprs[0]))
      choreo_unreachable("invalid value item.");
    return ub_exprs[0];
  }
  ValueList& GetUBounds() { return ub_exprs; }
  const ValueList& GetUBounds() const { return ub_exprs; }
};

struct RuntimeCheckEntry {
  std::string lhs;
  std::string op;
  std::string rhs;

  location loc;
  std::string message;
  std::map<std::string, std::string> notes;
};

enum class MMAType { WMMA, CTMMA, WGMMA, EFMMA, UKERNEL };

struct FragmentLayoutInfo {
  size_t regs_per_thread = 0;
  std::string thread_count_expr;
};

// per-function context
class FunctionContext {
private:
  FutureBufferInfo fbi;
  std::map<std::string, OptimizedValues> sym_values;
  std::vector<RuntimeCheckEntry> rt_checks;
  Assessor assessor;
  // TODO: consider to merge
  std::map<std::string, MMAType> frag_mma_type;
  std::map<std::string, std::string> MMA_policy_of_frag;

  std::map<std::string, FragmentLayoutInfo> fragment_info;
  std::map<std::string, FragmentLayout> fragment_layouts;

  struct DynMemReuseInfo {
    struct InfoEntry {
      // the name of simulator var
      std::string simulator;
      // the name of chunks vector
      std::string chunks_name;
      // the live ranges of each chunk
      std::vector<std::string> chunks;
      // result of HeapSimulator
      std::string result;
      // name of offset array
      std::string offsets_name;
      // offset names in device func
      std::vector<std::string> offset_args;
      // var of spm size
      std::string spm_size;
    };
    std::map<Storage, InfoEntry> infos;
  };
  struct StaticMemReuseInfo {
    struct InfoEntry {
      // the actual size of spm
      size_t spm_size;
    };
    std::map<Storage, InfoEntry> infos;
  };
  // device func name => mri
  std::map<std::string, ptr<DynMemReuseInfo>> dyn_mr_infos;
  std::map<std::string, ptr<StaticMemReuseInfo>> static_mr_infos;

  bool has_device_parallel = false;

public:
  bool HasDeviceParallel() const { return has_device_parallel; }
  void SetHasDeviceParallel(bool v) { has_device_parallel = v; }
  FutureBufferInfo& GetFutureBufferInfo() { return fbi; }
  OptimizedValues& GetSymbolValues(const std::string& sym) {
    return sym_values[sym];
  }
  const OptimizedValues& GetSymbolValues(const std::string& sym) const {
    return sym_values.at(sym);
  }
  bool HasSymbolValues(const std::string& sym) const {
    return sym_values.count(sym);
  }

  void AppendRtCheck(RuntimeCheckEntry rc) { rt_checks.push_back(rc); }
  std::vector<RuntimeCheckEntry>& GetRtChecks() { return rt_checks; }

  const std::vector<Assertion>& GetAssertions() const {
    return assessor.GetAssertions();
  }

  std::vector<Assertion> GetAssertions(AssessType aty) const {
    return assessor.GetAssertions(aty);
  }

  /// Bind a visitor and return the assessor for assessment calls.
  Assessor& GetAssessor(Visitor& v) { return assessor.Bind(v); }
  Assessor& GetAssessor() { return assessor; }
  const Assessor& GetAssessor() const { return assessor; }

  // return `nullptr` if no memory reuse info
  ptr<DynMemReuseInfo> GetDynMemReuseInfo(const std::string& dev_func) const {
    if (!dyn_mr_infos.count(dev_func)) return nullptr;
    return dyn_mr_infos.at(dev_func);
  }
  ptr<DynMemReuseInfo> SetDynMemReuseInfo(const std::string& dev_func) {
    if (dyn_mr_infos.count(dev_func)) return dyn_mr_infos.at(dev_func);
    auto info = std::make_shared<DynMemReuseInfo>();
    dyn_mr_infos.emplace(dev_func, info);
    return info;
  }
  ptr<StaticMemReuseInfo>
  GetStaticMemReuseInfo(const std::string& dev_func) const {
    if (!static_mr_infos.count(dev_func)) return nullptr;
    return static_mr_infos.at(dev_func);
  }
  ptr<StaticMemReuseInfo> SetStaticMemReuseInfo(const std::string& dev_func) {
    if (static_mr_infos.count(dev_func)) return static_mr_infos.at(dev_func);
    auto info = std::make_shared<StaticMemReuseInfo>();
    static_mr_infos.emplace(dev_func, info);
    return info;
  }
  bool HaveDynamicBuffer(const std::string& dev_func, Storage sto) const {
    auto mri = GetDynMemReuseInfo(dev_func);
    if (!mri) return false;
    return mri->infos.count(sto);
  }
  bool FragHasMMAType(const std::string& scoped_frag_name) const {
    if (!PrefixedWith(scoped_frag_name, "::"))
      choreo_unreachable("expect the fragament name is scoped.");
    return frag_mma_type.count(scoped_frag_name);
  }
  bool FragIsWMMA(const std::string& scoped_frag_name) const {
    if (!PrefixedWith(scoped_frag_name, "::"))
      choreo_unreachable("expect the fragament name is scoped.");
    return frag_mma_type.at(scoped_frag_name) == MMAType::WMMA;
  }
  bool FragIsCTMMA(const std::string& scoped_frag_name) const {
    if (!PrefixedWith(scoped_frag_name, "::"))
      choreo_unreachable("expect the fragament name is scoped.");
    return frag_mma_type.at(scoped_frag_name) == MMAType::CTMMA;
  }
  // WGMMA-specific methods
  bool FragIsWGMMA(const std::string& scoped_frag_name) const {
    if (!PrefixedWith(scoped_frag_name, "::"))
      choreo_unreachable("expect the fragament name is scoped.");
    if (!frag_mma_type.count(scoped_frag_name)) return false;
    return frag_mma_type.at(scoped_frag_name) == MMAType::WGMMA;
  }

  bool FragIsUKERNEL(const std::string& scoped_frag_name) const {
    if (!PrefixedWith(scoped_frag_name, "::"))
      choreo_unreachable("expect the fragament name is scoped.");
    if (!frag_mma_type.count(scoped_frag_name)) return false;
    return frag_mma_type.at(scoped_frag_name) == MMAType::UKERNEL;
  }

  void SetFragMMAType(const std::string& scoped_frag_name, MMAType mma_ty) {
    if (!PrefixedWith(scoped_frag_name, "::"))
      choreo_unreachable("expect the fragament name is scoped.");
    if (frag_mma_type.count(scoped_frag_name)) {
      if (frag_mma_type.at(scoped_frag_name) != mma_ty)
        choreo_unreachable(
            "expect the fragment to be always of wmma/mma/wgmma.");
    } else {
      frag_mma_type.emplace(scoped_frag_name, mma_ty);
    }
  }
  std::string MMAPolicyOfFrag(const std::string& scoped_frag_name) const {
    if (!PrefixedWith(scoped_frag_name, "::"))
      choreo_unreachable("expect the fragament name is scoped.");
    return MMA_policy_of_frag.at(scoped_frag_name);
  }
  void SetMMAPolicyOfFrag(const std::string& scoped_frag_name,
                          const std::string& mma_policy) {
    if (!PrefixedWith(scoped_frag_name, "::"))
      choreo_unreachable("expect the fragament name is scoped.");
    MMA_policy_of_frag[scoped_frag_name] = mma_policy;
  }
  const std::map<std::string, MMAType>& GetFragMMATypes() const {
    return frag_mma_type;
  }

  void SetFragmentInfo(const std::string& scoped_frag_name,
                       const FragmentLayoutInfo& info) {
    fragment_info[scoped_frag_name] = info;
  }
  bool HasFragmentInfo(const std::string& scoped_frag_name) const {
    return fragment_info.count(scoped_frag_name);
  }
  const FragmentLayoutInfo&
  GetFragmentInfo(const std::string& scoped_frag_name) const {
    return fragment_info.at(scoped_frag_name);
  }

  void SetFragmentLayout(const std::string& scoped, const FragmentLayout& fl) {
    fragment_layouts[scoped] = fl;
  }
  bool HasFragmentLayout(const std::string& scoped) const {
    return fragment_layouts.count(scoped);
  }
  const FragmentLayout& GetFragmentLayout(const std::string& scoped) const {
    return fragment_layouts.at(scoped);
  }

  void SetFragIsRS(const std::string& scoped_frag_name) {
    wgmma_rs_frags.insert(scoped_frag_name);
  }
  bool FragIsRS(const std::string& scoped_frag_name) const {
    return wgmma_rs_frags.count(scoped_frag_name) > 0;
  }

private:
  std::set<std::string> wgmma_rs_frags;
};

/// Aggregate statistics for the loop auto-vectorizer across all functions.
struct VectorizerStats {
  size_t loops_analyzed = 0;    // total foreach loops seen by the vectorizer
  size_t loops_vectorized = 0;  // loops successfully vectorized
  size_t loops_rejected = 0;    // loops rejected by legality checks
  size_t loops_hinted = 0;      // loops with explicit vectorize() hints
  size_t max_vector_factor = 0; // largest vector factor chosen
  size_t masks_generated = 0;   // masks inserted for divergent control flow
};

/// Aggregate statistics for the memory reuse pass across all functions.
struct MemReuseStats {
  size_t buffers_analyzed = 0;   // total buffers considered for reuse
  size_t static_buffers = 0;     // buffers with compile-time-known sizes
  size_t dynamic_buffers = 0;    // buffers requiring JIT heap simulation
  size_t device_functions = 0;   // device functions with reuse contexts
  size_t total_buffer_bytes = 0; // sum of individual buffer sizes before reuse
  size_t total_static_heap_bytes =
      0; // sum of heap_size from static allocations
};

/// Aggregate statistics for assessments and assertions across all functions.
struct AssessmentStats {
  size_t total = 0;            // total assessments evaluated
  size_t static_true = 0;      // resolved at compile time (always passes)
  size_t static_false = 0;     // proven false at compile time (error/warning)
  size_t runtime_total = 0;    // runtime assertions generated
  size_t runtime_entry = 0;    // runtime assertions with entry estimated cost
  size_t runtime_low = 0;      // runtime assertions with low estimated cost
  size_t runtime_medium = 0;   // runtime assertions with medium estimated cost
  size_t runtime_high = 0;     // runtime assertions with high estimated cost
  size_t runtime_enabled = 0;  // runtime assertions enabled for emission
  size_t runtime_disabled = 0; // runtime assertions suppressed by cost filter
  // Per-usage-type assessment counts (total evaluated, including static)
  size_t unclassified_total = 0;  // UsageType::ShapeCompatibility
  size_t shape_compat_total = 0;  // UsageType::ShapeCompatibility
  size_t elem_access_total = 0;   // UsageType::ElementAccess
  size_t loop_bound_total = 0;    // UsageType::LoopBound
  size_t hw_constraint_total = 0; // UsageType::HardwareConstraint
  // Per-usage-type runtime assertion counts
  size_t unclassified_runtime = 0;
  size_t shape_compat_runtime = 0;
  size_t elem_access_runtime = 0;
  size_t loop_bound_runtime = 0;
  size_t hw_constraint_runtime = 0;
};

class SymbolTable;
// per-compilation context
class CompilationContext {
private:
  std::map<std::string, FunctionContext> function_contexts;
  AssessmentStats assessment_stats; // accumulated across all functions
  VectorizerStats vectorizer_stats; // accumulated across all functions
  MemReuseStats mem_reuse_stats;    // accumulated across all functions
  std::unique_ptr<Target> compile_target = nullptr;
  std::map<std::string, std::unique_ptr<Target>> device_targets;
  std::vector<ArchId> archs;
  std::vector<FeatureToggle> features;
  OutputKind out_kind = OutputKind::TargetExecutable;
  int8_t opt_level = -1; // undecided

  // PlDepthMap is built when the compilation target or arch is configured,
  // and invalidated whenever either changes. PlDepthMap::Get() accesses this.
  friend class PlDepthMap;
  struct PlDepthMapHolder;
  std::shared_ptr<PlDepthMapHolder> pl_depth_map_;
  void InvalidatePlDepthMap() { pl_depth_map_ = nullptr; }

private:
  // compiler configurations
  bool debug_symtab = false;
  bool dump_ast = false;            // dump the AST after parsing
  bool no_codegen = false;          // stop before code generation
  bool print_pass_names = false;    // print pass name before pass run
  bool time_passes = false;         // measure time per compiler pass
  bool no_pre_process = false;      // do not invoke pre-processor
  bool drop_comment = false;        // drop any comments
  bool debug_all = false;           // enable full debug
  bool show_inferred_types = false; // show the inferred types
  bool show_strides = false;        // show the strides
  bool dump_symtab = false;         // dump symbol table after type check
  bool visualize = false;           // visualize the DMAs
  bool cross_compile = false;       // TODO: figure out
  bool trace_vn = false;            // trace the value numbering
  bool trace_vectorize = false;     // trace the masking
  bool show_source_loc = true;    // show source code location when error, etc.
  bool liveness = false;          // analyze the liveness of the program
  bool mem_reuse = false;         // reuse the memory of the program
  bool simplify_fp_valno = false; // simplify the floating point value number
  bool verify = false;            // verify visitors for legality
  bool gen_debug_info = false;    // generate source-level debug information
  bool target_debug_info = false; // pass debug flags to target compilation
  bool diag_dma = false;          // diagnose DMA at runtime
  bool loop_norm = false;         // enable loop normalization
  bool no_vectorize = false;      // do not vectorize any foreach loop
  bool vectorize = false;         // enable loop vectorization
  size_t max_local_mem_capacity =
      0; // max local memory capacity per thread (0: use default)
  size_t shared_mem_alignment = 0; // alignment of shared memory set by user
  bool inhibit_warning = false;    // Inhibit all warning messages.
  bool warning_as_error = false;   // Make all warnings into errors.
  bool show_assess = false;        // Print assessment report after hoisting.
  bool trace_assess = false;       // Trace assessment processing.
  bool print_stats = false;        // Print aggregate assessment statistics.
  AssertionCost rtc_cost_threshold =
      AssertionCost::ENTRY; // Runtime check assertion level
  bool disable_cuda_runtime_env_check =
      false;                 // Do not emit cuda runtime env check.
  bool use_warpspec = false; // Enable warp-specialized synchronization for
                             // shared event/full-empty pipelines.
  bool hoist_wgmma_arrive = false; // Hoist warpgroup_arrive before unrolled
                                   // foreach with WGMMA exec but no commit.
  bool single_thread_producer =
      true; // In warpspec mode, use a single producer thread
            // for producer inthreads; otherwise guard
            // producer TMA/event ops individually.
  // Skip wg_barrier.sync() before shared-to-global TMA copies when set via CLI.
  bool skip_epilogue_group_sync = false;
  bool fast_compile = false; // Use precompiled CuTe runtime for faster nvcc
                             // compilation via separate compilation + linking.
  bool use_fast_math = true; // Pass --use_fast_math to nvcc for faster
                             // transcendentals (exp/div) in generated code.
  bool use_target_lib = false; // Lower __lib_* builtins to target library calls
                               // (e.g. native hardware library calls).
  std::string debug_file_dir;  // directory for compiler debug artifacts
  std::string api_mode = "cffi"; // API mode for generated code
  DebugLinePathMode debug_line_path_mode = DebugLinePathMode::WorkspaceRelative;

private:
  std::shared_ptr<SymbolTable> sym_tab = nullptr; // global symbol table
  ptr<AST::Node> pre_sema_root_;

public:
  bool NeedPreSemaClone() const {
    return compile_target && compile_target->Name() == "hetero";
  }
  void SavePreSemaRoot(AST::Node& r);
  ptr<AST::Node> GetPreSemaRoot() const { return pre_sema_root_; }

  // Captures mutable target-compilation state so that device offload
  // pipelines can run in isolation and then restore the parent state.
  // Grouping these together enables future thread-based target compilation.
  struct TargetCompilationState {
    std::unique_ptr<Target> target;
    std::vector<ArchId> archs;
    OutputKind output_kind = OutputKind::TargetExecutable;
    std::shared_ptr<SymbolTable> sym_tab;
    unsigned anonymous_count = 0;
    unsigned anon_pb_count = 0;
    // CodeGenInfo held opaquely; managed by Save/Restore implementation.
    struct CGIHolder;
    std::unique_ptr<CGIHolder> cgi_holder;
    // PlDepthMap snapshot.
    std::shared_ptr<PlDepthMapHolder> pl_depth_map;
    TargetCompilationState();
    ~TargetCompilationState();
    TargetCompilationState(TargetCompilationState&&) noexcept;
    TargetCompilationState& operator=(TargetCompilationState&&) noexcept;
  };

  TargetCompilationState SaveTargetState();
  void RestoreTargetState(TargetCompilationState&& state);

private:
  std::unordered_map<std::string, std::string> cl_macros; // defined macros
  std::vector<std::string> include_paths;
  std::vector<std::string> library_paths;
  std::vector<std::string> libraries;
  std::vector<std::string> source_lines;

  std::unordered_map<int, std::vector<MacroSub>> line_macro_subs;

public:
  bool DebugSymTab() const { return debug_symtab; }

  void SetGlobalSymbolTable(const std::shared_ptr<SymbolTable>& st) {
    sym_tab = st;
  }

  std::shared_ptr<SymbolTable>& GetGlobalSymbolTable() {
    if (!sym_tab) choreo_unreachable("global symbol table is invalid.");
    return sym_tab;
  }

  FunctionContext& GetFunctionContext(const std::string fname) {
    return function_contexts[fname];
  }
  const FunctionContext& GetFunctionContext(const std::string fname) const {
    return function_contexts.at(fname);
  }

  const Target& GetTarget() const { return *compile_target; }
  Target& GetTarget() { return *compile_target; }
  bool SetTarget(std::unique_ptr<Target>&& ct) {
    if (ct == nullptr) return false;
    compile_target = std::move(ct);
    InvalidatePlDepthMap();
    return true;
  }

  std::unique_ptr<Target> SwapTarget(std::unique_ptr<Target>&& ct) {
    auto old = std::move(compile_target);
    compile_target = std::move(ct);
    InvalidatePlDepthMap();
    return old;
  }

  const std::string TargetName() const { return compile_target->Name(); }

  bool HasDeviceTargets() const { return !device_targets.empty(); }

  void AddDeviceTarget(const std::string& device_name,
                       std::unique_ptr<Target>&& t) {
    device_targets[device_name] = std::move(t);
  }

  const Target* GetDeviceTarget(const std::string& device_name) const {
    auto it = device_targets.find(device_name);
    return it != device_targets.end() ? it->second.get() : nullptr;
  }

  const std::map<std::string, std::unique_ptr<Target>>& DeviceTargets() const {
    return device_targets;
  }
#if 0
  // useful for MPI
  std::string GetSubTarget() const { return compile_sub_target; }
  void SetSubTarget(CompileTarget ct) { compile_sub_target = ct; }
#endif

  bool IsArchSet() const { return !archs.empty(); }
  const std::vector<ArchId> GetArchs() const {
    if (archs.size() == 0) return {GetTarget().DefaultArch()};
    return archs;
  }
  const ArchId GetArch() const {
    if (archs.size() == 0) return GetTarget().DefaultArch();
    if (archs.size() != 1)
      choreo_unreachable("unexpected multiple architecture.");
    return archs[0];
  }
  void ClearArchs() {
    archs.clear();
    InvalidatePlDepthMap();
  }
  void SetArchs(const std::vector<ArchId>& a) {
    archs = a;
    InvalidatePlDepthMap();
  }
  void AddArch(const ArchId& arch) {
    ArchId resolved = arch;
    if (arch == "native") {
      resolved = GetTarget().ResolveNativeArch();
      if (resolved.empty())
        choreo_unreachable("-arch=native is not supported by target '" +
                           GetTarget().Name() + "'.");
      errs() << "note: -arch=native resolved to '" << resolved << "'\n";
    }
    if (!GetTarget().IsArchSupported(resolved))
      choreo_unreachable("-arch=" + resolved + " is not supported by target '" +
                         GetTarget().Name() + "'.");
    archs.push_back(resolved);
    InvalidatePlDepthMap();
  }

  int GetOptimizationLevel() const {
    if (opt_level == -1)
      return TargetDefaultOptLevel();
    else
      return (int)opt_level;
  }
  void SetOptimizationLevel(int8_t lv) { opt_level = lv; }

  OutputKind GetOutputKind() { return out_kind; }
  void SetOutputKind(OutputKind ok) { out_kind = ok; }

  size_t GetMemCapacity(Storage sto) const {
    if (sto == Storage::LOCAL &&
        GetTarget().IsFeatureSupported(STR(ChoreoFeature::SLML)) &&
        MaxLocalMemCapacity() != 0)
      return MaxLocalMemCapacity();
    return GetTarget().GetMemCapacity(sto, GetArch());
  }

  // return memory alignment in byte. Used in memory reuse pass.
  size_t GetMemoryAlignmentByte(const ArchId& arch, Storage sto) const {
    if (sto == Storage::SHARED && SharedMemAlignment() != 0)
      return SharedMemAlignment();
    return GetTarget().GetMemAlignmentByte(sto, arch);
  }
  size_t GetMemoryAlignmentByte(Storage sto) const {
    return GetTarget().GetMemAlignmentByte(sto, GetArch());
  }

  size_t GetMinGroupDim() const {
    return GetTarget().GetMinGroupDim(GetArch());
  }

  size_t GetMaxParallelByCount(ParallelLevel pl) const {
    return GetTarget().GetMaxParallelByCount(pl, GetArch());
  }

  size_t GetMaxThreadsPerBlock() const {
    return GetTarget().GetMaxThreadsPerBlock(GetArch());
  }

  size_t GetVectorLength() const {
    return GetTarget().GetVectorLength(GetArch());
  }

  int TargetDefaultOptLevel() const {
    return GetTarget().DefaultOptLevel(GetArch());
  }

public:
  // Getters of compiler configurations
  bool DumpAst() const { return dump_ast; }
  bool NoCodegen() const { return no_codegen; }
  bool PrintPassNames() const { return print_pass_names; }
  bool TimePasses() const { return time_passes; }
  bool NoPreProcess() const { return no_pre_process; }
  bool DropComments() const { return drop_comment; }
  bool DebugAll() const { return debug_all; }
  bool ShowInferredTypes() const { return show_inferred_types; }
  bool ShowStrides() const { return show_strides; }
  bool DumpSymtab() const { return dump_symtab; }
  bool Visualize() const { return visualize; }
  bool CrossCompile() const { return cross_compile; }
  bool TraceValueNumbers() const { return trace_vn; }
  bool TraceVectorize() const { return trace_vectorize; }
  bool LivenessAnalysis() const { return liveness; }
  bool MemReuse() const { return mem_reuse; }
  bool SimplifyFpValno() const { return simplify_fp_valno; }
  bool VerifyVisitors() const { return verify; }
  bool GenDebugInfo() const { return gen_debug_info; }
  bool TargetDebugInfo() const { return target_debug_info; }
  DebugLinePathMode GetDebugLinePathMode() const {
    return debug_line_path_mode;
  }
  bool DMADiagnosis() const { return diag_dma; }
  bool LoopNorm() const { return loop_norm; }
  bool NoVectorize() const { return no_vectorize; }
  bool Vectorize() const { return vectorize; }
  size_t MaxLocalMemCapacity() const { return max_local_mem_capacity; }
  size_t SharedMemAlignment() const { return shared_mem_alignment; }
  bool InhibitWarning() const { return inhibit_warning; }
  bool WarningAsError() const { return warning_as_error; }
  bool ShowAssess() const { return show_assess; }
  bool TraceAssess() const { return trace_assess; }
  bool PrintStats() const { return print_stats; }
  const AssessmentStats& GetAssessmentStats() const { return assessment_stats; }
  AssessmentStats& GetAssessmentStats() { return assessment_stats; }
  const VectorizerStats& GetVectorizerStats() const { return vectorizer_stats; }
  VectorizerStats& GetVectorizerStats() { return vectorizer_stats; }
  const MemReuseStats& GetMemReuseStats() const { return mem_reuse_stats; }
  MemReuseStats& GetMemReuseStats() { return mem_reuse_stats; }
  AssertionCost RuntimeCheckCostThreshold() const { return rtc_cost_threshold; }
  bool DisableRuntimeCheck() const {
    return rtc_cost_threshold == AssertionCost::NONE;
  }
  bool DisableCudaRuntimeEnvCheck() const {
    return disable_cuda_runtime_env_check;
  }
  bool UseWarpSpec() const { return use_warpspec; }
  bool HoistWGMMAArrive() const { return hoist_wgmma_arrive; }
  bool SingleThreadProducer() const { return single_thread_producer; }
  bool SkipEpilogueGroupSync() const { return skip_epilogue_group_sync; }
  bool FastCompile() const { return fast_compile; }
  bool UseFastMath() const { return use_fast_math; }
  bool UseTargetLib() const { return use_target_lib; }
  const std::string& GetDebugFileDir() const { return debug_file_dir; }
  void SetDebugFileDir(const std::string& dir) { debug_file_dir = dir; }
  const std::string& GetApiMode() const { return api_mode; }
  void SetApiMode(const std::string& mode) { api_mode = mode; }

  // Setters of compiler configurations
  void SetDumpAst(bool value) { dump_ast = value; }
  void SetNoCodegen(bool value) { no_codegen = value; }
  void SetPrintPassNames(bool value) { print_pass_names = value; }
  void SetTimePasses(bool value) { time_passes = value; }
  void SetNoPreProcess(bool value) { no_pre_process = value; }
  void SetDropComments(bool value) { drop_comment = value; }
  void SetDebugAll(bool value) { debug_all = value; }
  void SetShowInferredTypes(bool value) { show_inferred_types = value; }
  void SetShowStrides(bool value) { show_strides = value; }
  void SetDumpSymtab(bool value) { dump_symtab = value; }
  void SetVisualize(bool value) { visualize = value; }
  void SetCrossCompile(bool value) { cross_compile = value; }
  void SetTraceValueNumbers(bool value) { trace_vn = value; }
  void SetTraceVectorize(bool value) { trace_vectorize = value; }
  void SetLivenessAnalysis(bool value) { liveness = value; }
  void SetMemReuse(bool value) { mem_reuse = value; }
  void SetSimplifyFpValno(bool value) { simplify_fp_valno = value; }
  void SetVerifyVisitors(bool value) { verify = value; }
  void SetGenDebugInfo(bool value) { gen_debug_info = value; }
  void SetTargetDebugInfo(bool value) { target_debug_info = value; }
  void SetDebugLinePathMode(DebugLinePathMode mode) {
    debug_line_path_mode = mode;
  }
  void SetDMADiagnosis(bool value) { diag_dma = value; }
  void SetLoopNorm(bool value) { loop_norm = value; }
  void SetNoVectorize(bool value) { no_vectorize = value; }
  void SetVectorize(bool value) { vectorize = value; }
  void SetMaxLocalMemCapacityPerThread(size_t sz) {
    max_local_mem_capacity = sz;
  }
  void SetUseWarpSpec(bool value) { use_warpspec = value; }
  void SetHoistWGMMAArrive(bool value) { hoist_wgmma_arrive = value; }
  void SetSingleThreadProducer(bool value) { single_thread_producer = value; }
  void SetSkipEpilogueGroupSync(bool value) {
    skip_epilogue_group_sync = value;
  }
  void SetFastCompile(bool value) { fast_compile = value; }
  void SetUseFastMath(bool value) { use_fast_math = value; }
  void SetUseTargetLib(bool value) { use_target_lib = value; }
  void SetSharedMemAlignment(size_t value) { shared_mem_alignment = value; }
  void SetInhibitWarning(bool value) { inhibit_warning = value; }
  void SetWarningAsError(bool value) { warning_as_error = value; }
  void SetShowAssess(bool value) { show_assess = value; }
  void SetTraceAssess(bool value) { trace_assess = value; }
  void SetPrintStats(bool value) { print_stats = value; }
  void SetRuntimeCheckCostThreshold(AssertionCost cost) {
    rtc_cost_threshold = cost;
  }
  void SetDisableCudaRuntimeEnvCheck(bool value) {
    disable_cuda_runtime_env_check = value;
  }

  const std::unordered_map<std::string, std::string>& GetCLMacros() const {
    return cl_macros;
  }

  std::unordered_map<std::string, std::string>& GetCLMacros() {
    return cl_macros;
  }

  const std::vector<std::string>& GetIncPaths() const { return include_paths; }
  std::vector<std::string>& GetIncPaths() { return include_paths; }

  const std::vector<std::string>& GetLibPaths() const { return library_paths; }
  std::vector<std::string>& GetLibPaths() { return library_paths; }

  const std::vector<std::string>& GetLibs() const { return libraries; }
  std::vector<std::string>& GetLibs() { return libraries; }

  void SetShowSourceLocation(bool s) { show_source_loc = s; }
  bool ShowSourceLocation() const { return show_source_loc; }

  void ReadSourceLines(std::istream& input) {
    std::string line;
    // TODO: do not read all lines for large source file
    while (std::getline(input, line)) source_lines.push_back(line);
  }

  std::string GetSourceLine(int line_no) const {
    if (line_no > 0 && line_no <= (int)source_lines.size()) {
      return source_lines[line_no - 1];
    }
    return "";
  }

  void SetLineMacroSubs(int line_no, std::vector<Choreo::MacroSub> subs) {
    line_macro_subs[line_no] = std::move(subs);
  }

  int MapExpandedColToOriginal(int line_no, int exp_col) const {
    auto it = line_macro_subs.find(line_no);
    if (it == line_macro_subs.end()) return exp_col;
    int offset = 0;
    for (const auto& s : it->second) {
      int exp_start = s.orig_col + offset;
      if (exp_col < exp_start) break;
      if (exp_col < exp_start + s.repl_len) return s.orig_col;
      offset += s.repl_len - s.orig_len;
    }
    return exp_col - offset;
  }

public:
  int ArchNum() const { return GetTarget().ArchNum(GetArch()); }
  bool HasFeature(ChoreoFeature cf) const {
    return GetTarget().IsFeatureSupported(STR(cf));
  }
  bool HasFeature(ChoreoFeature cf, ArchId arch) const {
    return GetTarget().IsFeatureSupported(arch, STR(cf));
  }
  bool TargetSupportEvent() const {
    return HasFeature(ChoreoFeature::EVENT, GetArch());
  }
  bool TargetSupportMMA() const {
    return HasFeature(ChoreoFeature::MMA, GetArch());
  }
  bool TargetSupportMMAUKernel() const {
    return HasFeature(ChoreoFeature::MMA_UKERNEL, GetArch());
  }
  bool TargetSupportWGMMA() const {
    return HasFeature(ChoreoFeature::WGMMA, GetArch());
  }
  bool TargetSupportTMA() const {
    return HasFeature(ChoreoFeature::TMA, GetArch());
  }
  bool TargetSupportMemAlloc() const {
    return HasFeature(ChoreoFeature::MEMALLOC, GetArch());
  }
  bool TargetSupportVectorize() const {
    return HasFeature(ChoreoFeature::VECTORIZE, GetArch());
  }
  bool TargetSupportAsyncDMA() const {
    return HasFeature(ChoreoFeature::ASYNC_DMA, GetArch());
  }

  size_t TargetVectorizeLimit() const {
    return GetTarget().VectorizeLimit(GetArch());
  }
  auto TargetVectorizeTypes() const {
    return GetTarget().VectorizableTypes(GetArch());
  }
  auto TargetParallelLevels() const {
    return GetTarget().GetParallelLevels(GetArch());
  }
  auto TargetSwizzleModes() const {
    return GetTarget().SupportedSwizzleModes(GetArch());
  }

public:
  static CompilationContext& GetInstance() {
    static CompilationContext instance;
    return instance;
  }
};

inline CompilationContext& CCtx() { return CompilationContext::GetInstance(); }
inline FunctionContext& FCtx(const std::string& fname) {
  return CCtx().GetFunctionContext(fname);
}

} // end namespace Choreo

#endif //__CHOREO_CONTEXT_HPP__
