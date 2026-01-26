#ifndef __CHOREO_CONTEXT_HPP__
#define __CHOREO_CONTEXT_HPP__

// shared global context for a compilation process

#include "loc.hpp"
#include "symvals.hpp"
#include "target.hpp"
#include "types.hpp"
#include <map>
#include <memory>
#include <sstream>

extern Choreo::location loc;

namespace Choreo {

enum class OutputKind {
  PreProcessedCode,
  TargetSourceCode,
  TargetModule,
  TargetAssembly,
  TargetExecutable,
  ShellScript,
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
  std::vector<ValueItem> val_exprs;
  // TODO: distinguish values and mdspans
  // std::vector<ValueItem> mds_exprs;
  std::vector<ValueItem> ub_exprs;
  ValueItem size_expr = GetInvalidValueItem();

public:
  void SetVal(ValueItem vi) {
    val_exprs.clear();
    val_exprs.push_back(vi);
  }
  void SetVals(const std::vector<ValueItem>& vis) {
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
  void SetUBounds(const std::vector<ValueItem>& vis) {
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

// TODO: use assert experssions to replace string like entry
struct Assertion {
  ptr<sbe::SymbolicExpression> expr;

  bool is_host;
  location loc;
  std::string message;
};

enum class MMAType { WMMA, CTMMA, WGMMA, EFMMA };
// per-function context
class FunctionContext {
private:
  FutureBufferInfo fbi;
  std::map<std::string, OptimizedValues> sym_values;
  std::vector<RuntimeCheckEntry> rt_checks;
  std::vector<Assertion> assertions;
  // TODO: consider to merge
  std::map<std::string, MMAType> frag_mma_type;
  std::map<std::string, std::string> MMA_policy_of_frag;

  struct DynMemReuseInfo {
    std::string simulator;
    struct InfoEntry {
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

public:
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

  void InsertAssertion(const ptr<sbe::SymbolicExpression>& ar,
                       const location& l, const std::string& s,
                       bool is_host = true) {
    // the none computable expressions are ignored. verbose?
    assert(IsComputable(ar));
    assertions.push_back({ar, is_host, l, s});
  }
  const std::vector<Assertion>& GetAssertions() const { return assertions; }

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
};

class SymbolTable;
// per-compilation context
class CompilationContext {
private:
  std::map<std::string, FunctionContext> function_contexts;
  std::unique_ptr<Target> compile_target = nullptr;
  std::vector<ArchId> archs;
  std::vector<FeatureToggle> features;
  OutputKind out_kind = OutputKind::TargetExecutable;
  int8_t opt_level = -1; // undecided

private:
  // compiler configurations
  bool debug_symtab = false;
  bool dump_ast = false;            // dump the AST after parsing
  bool no_codegen = false;          // stop before code generation
  bool print_pass_names = false;    // print pass name before pass run
  bool no_pre_process = false;      // do not invoke pre-processor
  bool drop_comment = false;        // drop any comments
  bool debug_all = false;           // enable full debug
  bool show_inferred_types = false; // show the inferred types
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
  bool gen_debug_info = false;    // generate debug information
  bool diag_dma = false;          // diagnose DMA at runtime
  bool loop_norm = false;         // enable loop normalization
  bool no_vectorize = false;      // do not vectorize any foreach loop
  bool vectorize = false;         // enable loop vectorization
  size_t max_local_mem_capacity =
      0; // max local memory capacity per thread (0: use default)
  bool mem_default_aligned = true; // alignment is set by default in mem reuse.
  bool inhibit_warning = false;    // Inhibit all warning messages.
  bool warning_as_error = false;   // Make all warnings into errors.
  bool disable_runtime_check = false; // Disable all runtime checks.
  std::string debug_file_dir;         // directory for compiler debug artifacts
  std::string api_mode = "cffi";      // API mode for generated code

private:
  std::shared_ptr<SymbolTable> sym_tab = nullptr; // global symbol table

private:
  std::unordered_map<std::string, std::string> cl_macros; // defined macros
  std::vector<std::string> include_paths;
  std::vector<std::string> library_paths;
  std::vector<std::string> libraries;
  std::vector<std::string> source_lines;

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
    return true;
  }

  const std::string TargetName() const { return compile_target->Name(); }
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
  void AddArch(const ArchId& arch) {
    if (!GetTarget().IsArchSupported(arch))
      choreo_unreachable("-arch=" + arch + " is not supported by target '" +
                         GetTarget().Name() + "'.");
    archs.push_back(arch);
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
  size_t GetMemoryAlignment(const ArchId& arch, Storage sto) const {
    if (!MemDefaultAligned()) return 1;
    return GetTarget().GetMemAlignment(sto, arch);
  }

  size_t GetMinGroupDim() const {
    return GetTarget().GetMinGroupDim(GetArch());
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
  bool NoPreProcess() const { return no_pre_process; }
  bool DropComments() const { return drop_comment; }
  bool DebugAll() const { return debug_all; }
  bool ShowInferredTypes() const { return show_inferred_types; }
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
  bool DMADiagnosis() const { return diag_dma; }
  bool LoopNorm() const { return loop_norm; }
  bool NoVectorize() const { return no_vectorize; }
  bool Vectorize() const { return vectorize; }
  size_t MaxLocalMemCapacity() const { return max_local_mem_capacity; }
  bool MemDefaultAligned() const { return mem_default_aligned; }
  bool InhibitWarning() const { return inhibit_warning; }
  bool WarningAsError() const { return warning_as_error; }
  bool DisableRuntimeCheck() const { return disable_runtime_check; }
  const std::string& GetDebugFileDir() const { return debug_file_dir; }
  void SetDebugFileDir(const std::string& dir) { debug_file_dir = dir; }
  const std::string& GetApiMode() const { return api_mode; }
  void SetApiMode(const std::string& mode) { api_mode = mode; }

  // Setters of compiler configurations
  void SetDumpAst(bool value) { dump_ast = value; }
  void SetNoCodegen(bool value) { no_codegen = value; }
  void SetPrintPassNames(bool value) { print_pass_names = value; }
  void SetNoPreProcess(bool value) { no_pre_process = value; }
  void SetDropComments(bool value) { drop_comment = value; }
  void SetDebugAll(bool value) { debug_all = value; }
  void SetShowInferredTypes(bool value) { show_inferred_types = value; }
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
  void SetDMADiagnosis(bool value) { diag_dma = value; }
  void SetLoopNorm(bool value) { loop_norm = value; }
  void SetNoVectorize(bool value) { no_vectorize = value; }
  void SetVectorize(bool value) { vectorize = value; }
  void SetMaxLocalMemCapacityPerThread(size_t sz) {
    max_local_mem_capacity = sz;
  }
  void SetMemDefaultAligned(bool value) { mem_default_aligned = value; }
  void SetInhibitWarning(bool value) { inhibit_warning = value; }
  void SetWarningAsError(bool value) { warning_as_error = value; }
  void SetDisableRuntimeCheck(bool value) { disable_runtime_check = value; }

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

  size_t TargetVectorizeLimit() const {
    return GetTarget().VectorizeLimit(GetArch());
  }
  auto TargetVectorizeTypes() const {
    return GetTarget().VectorizableTypes(GetArch());
  }
  auto TargetParallelLevels() const {
    return GetTarget().GetParallelLevels(GetArch());
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
