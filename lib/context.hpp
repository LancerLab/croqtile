#ifndef __CHOREO_CONTEXT_HPP__
#define __CHOREO_CONTEXT_HPP__

// shared global context for a compilation process

#include "loc.hpp"
#include "symvals.hpp"
#include "types.hpp"
#include <map>
#include <memory>
#include <sstream>

extern Choreo::location loc;

namespace Choreo {

// The target languages
enum class CompileTarget {
  Unknown,
  Factor,
  Topscc,
  CUDA,
  Cute,
  MPI,
};

inline static const std::string STR(CompileTarget ct) {
  switch (ct) {
  case CompileTarget::Unknown: return "Unknown";
  case CompileTarget::Factor: return "Factor";
  case CompileTarget::Topscc: return "Topscc";
  case CompileTarget::CUDA: return "CUDA";
  case CompileTarget::Cute: return "Cute";
  case CompileTarget::MPI: return "MPI";
  default: choreo_unreachable("Unsupported operand kind.");
  }
  return "";
}

enum class TargetArch {
  Unknown,
  GCU20,
  GCU21,
  GCU3,
  GCU4,
  GPU, // TODO: unclear
  // Nv series
  SM_70,
  SM_75,
  SM_80,
  SM_86,
  SM_89,
  SM_90,
  SM_100,
  SM_120,
  End
};

inline static const std::string STR(TargetArch ta) {
  switch (ta) {
  case TargetArch::Unknown: return "Unknown";
  case TargetArch::GCU20: return "GCU200";
  case TargetArch::GCU21: return "GCU210";
  case TargetArch::GCU3: return "GCU300";
  case TargetArch::GCU4: return "GCU400";
  case TargetArch::GPU: return "GPU";
  case TargetArch::SM_70: return "SM_70";
  case TargetArch::SM_75: return "SM_75";
  case TargetArch::SM_80: return "SM_80";
  case TargetArch::SM_86: return "SM_86";
  case TargetArch::SM_89: return "SM_89";
  case TargetArch::SM_90: return "SM_90";
  case TargetArch::SM_100: return "SM_100";
  case TargetArch::SM_120: return "SM_120";
  default: choreo_unreachable("Unsupported operand kind.");
  }
  return "";
}

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

// per-function context
class FunctionContext {
private:
  FutureBufferInfo fbi;
  std::map<std::string, OptimizedValues> sym_values;
  std::vector<RuntimeCheckEntry> rt_checks;
  std::vector<Assertion> assertions;

  struct MemReuseInfo {
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
  // device func name => mri
  std::map<std::string, ptr<MemReuseInfo>> mem_reuse_infos;

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
  ptr<MemReuseInfo> GetMemReuseInfo(const std::string& dev_func) const {
    if (!mem_reuse_infos.count(dev_func)) return nullptr;
    return mem_reuse_infos.at(dev_func);
  }
  ptr<MemReuseInfo> SetMemReuseInfo(const std::string& dev_func) {
    if (mem_reuse_infos.count(dev_func)) return mem_reuse_infos.at(dev_func);
    auto info = std::make_shared<MemReuseInfo>();
    mem_reuse_infos.emplace(dev_func, info);
    return info;
  }
  bool HaveDynamicBuffer(const std::string& dev_func, Storage sto) const {
    auto mri = GetMemReuseInfo(dev_func);
    if (!mri) return false;
    return mri->infos.count(sto);
  }
};

class SymbolTable;
// per-compilation context
class CompilationContext {
private:
  std::map<std::string, FunctionContext> function_contexts;
  CompileTarget compile_target = CompileTarget::Unknown;
  CompileTarget compile_sub_target = CompileTarget::Unknown;
  TargetArch arch = TargetArch::Unknown;
  OutputKind out_kind = OutputKind::TargetExecutable;
  uint8_t opt_level = 0;

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

  CompileTarget GetTarget() const { return compile_target; }
  void SetTarget(CompileTarget ct) { compile_target = ct; }

  // useful for MPI
  CompileTarget GetSubTarget() const { return compile_sub_target; }
  void SetSubTarget(CompileTarget ct) { compile_sub_target = ct; }

  TargetArch GetArch() const { return arch; }
  void SetArch(TargetArch ta) { arch = ta; }

  uint8_t GetOptimizationLevel() const { return opt_level; }
  void SetOptimizationLevel(uint8_t lv) { opt_level = lv; }

  OutputKind GetOutputKind() { return out_kind; }
  void SetOutputKind(OutputKind ok) { out_kind = ok; }

  size_t GetMemCapacity(Storage sto) const {
    switch (arch) {
    case TargetArch::GCU20: // TODO
    case TargetArch::GCU21: {
      switch (sto) {
      case Storage::LOCAL: return 1008ull * 1024;             // 1008KB
      case Storage::SHARED: return 24ull * 1024 * 1024;       // 24MB
      case Storage::GLOBAL: return 4ull * 1024 * 1024 * 1024; // 4GB
      default: choreo_unreachable("Unsupported mem level.");
      }
    }

    case TargetArch::GCU3: {
      switch (sto) {
      case Storage::LOCAL:
        switch (GetTarget()) {
        case CompileTarget::Factor: return 1.5 * 1024 * 1024; // 1.5MB
        case CompileTarget::Topscc:
          return 1.5 * 1024 * 1024 - 512; // special case
        default: choreo_unreachable("Unhandled target.");
        }
      case Storage::SHARED:
        switch (GetTarget()) {
        case CompileTarget::Factor: return 24ull * 1024 * 1024; // 24MB
        case CompileTarget::Topscc: return 64ull * 1024 * 1024; // 64MB
        default: choreo_unreachable("Unhandled target.");
        }
      case Storage::GLOBAL: return 40.75 * 1024 * 1024 * 1024; // 40.75GB
      default: choreo_unreachable("Unsupported mem level.");
      }
    }

    case TargetArch::GCU4: {
      switch (sto) {
      case Storage::LOCAL:
        switch (GetTarget()) {
        case CompileTarget::Topscc:
          return 1.5 * 1024 * 1024 - 512; // todo: check this
        default: choreo_unreachable("Unhandled target.");
        }
      case Storage::SHARED:
        switch (GetTarget()) {
        case CompileTarget::Topscc:
          return 64ull * 1024 * 1024; // todo: check this
        default: choreo_unreachable("Unhandled target.");
        }
      case Storage::GLOBAL:
        return 40.75 * 1024 * 1024 * 1024; // todo: check this
      default: choreo_unreachable("Unsupported mem level.");
      }
    }

    case TargetArch::GPU:
    case TargetArch::SM_70:
    case TargetArch::SM_75:
    case TargetArch::SM_80:
    case TargetArch::SM_86:
    case TargetArch::SM_89:
    case TargetArch::SM_90:
    case TargetArch::SM_100:
    case TargetArch::SM_120: {
      switch (sto) {
      case Storage::LOCAL: {
        if (MaxLocalMemCapacity() > 0) { return MaxLocalMemCapacity(); }
        switch (arch) {
        case TargetArch::SM_70:
        case TargetArch::SM_75: return 1024;
        case TargetArch::SM_80:
        case TargetArch::SM_86: return 2048;
        case TargetArch::SM_89: return 3072;
        case TargetArch::SM_90: return 4096;
        case TargetArch::SM_100: return 4096;
        case TargetArch::SM_120: return 4096 /* TO-confirm */;
        case TargetArch::GPU:
        default: return 1024;
        }
      }
      // TODO: Memory capacity of GPU is not only determined by the
      // arch, but also by the specific model?
      case Storage::SHARED: return 48ull * 1024;              // 48KB
      case Storage::GLOBAL: return 8ull * 1024 * 1024 * 1024; // 8GB
      default: choreo_unreachable("Unsupported mem level.");
      }
    }

    default: choreo_unreachable("Unsupported target arch.");
    }
    return 0;
  }

  size_t GetMinGroupDim() const {
    size_t bound = 1;
    if (GetTarget() == CompileTarget::Topscc) {
      switch (GetArch()) {
      case TargetArch::GCU20:
      case TargetArch::GCU21:
      case TargetArch::GCU3: break;
      case TargetArch::GCU4: bound = 8; break;
      default: choreo_unreachable("unsupported target arch.");
      }
    } else if (GetTarget() == CompileTarget::Cute)
      bound = 32;
    else
      choreo_unreachable("unsupported target.");

    return bound;
  }

  size_t GetSingleVectorByteSize() const {
    switch (GetArch()) {
    case TargetArch::GCU3: return 128;
    case TargetArch::GCU4: return 512;
    default: choreo_unreachable("Unsupported target arch.");
    }
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
  bool SupportEvent() const {
    switch (compile_target) {
    case CompileTarget::Topscc:
      return arch == TargetArch::GCU3 || arch == TargetArch::GCU4;
    case CompileTarget::Cute: return true;
    default: break;
    }
    return false;
  }

  bool SupportMMA() const {
    switch (compile_target) {
    case CompileTarget::Cute: return true;
    default: break;
    }
    return false;
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
