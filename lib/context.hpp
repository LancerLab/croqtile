#ifndef __CHOREO_CONTEXT_HPP__
#define __CHOREO_CONTEXT_HPP__

// shared global context for a compilation process

#include "loc.hpp"
#include "symvals.hpp"
#include "types.hpp"
#include <map>
#include <memory>
#include <sstream>

namespace Choreo {

// The target languages
enum class CompileTarget {
  Unknown,
  Factor,
  Topscc,
  CUDA,
};

inline static const std::string STR(CompileTarget ct) {
  switch (ct) {
  case CompileTarget::Unknown: return "Unknown";
  case CompileTarget::Factor: return "Factor";
  case CompileTarget::Topscc: return "Topscc";
  case CompileTarget::CUDA: return "CUDA";
  default: choreo_unreachable("Unsupported operand kind.");
  }
  return "";
}

enum class TargetArch {
  Unknown,
  GCU20,
  GCU21,
  GCU3,
  GPU,
};

inline static const std::string STR(TargetArch ta) {
  switch (ta) {
  case TargetArch::Unknown: return "Unknown";
  case TargetArch::GCU20: return "GCU200";
  case TargetArch::GCU21: return "GCU210";
  case TargetArch::GCU3: return "GCU300";
  case TargetArch::GPU: return "GPU";
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
  ValueItem int_expr = GetInvalidValueItem();
  ValueItem size_expr = GetInvalidValueItem();
};

struct RuntimeCheckEntry {
  std::string lhs;
  std::string op;
  std::string rhs;

  location loc;
  std::string message;
  std::map<std::string, std::string> notes;
};

// per-function context
class FunctionContext {
  FutureBufferInfo fbi;
  std::map<std::string, OptimizedValues> sym_values;
  std::vector<RuntimeCheckEntry> rt_checks;

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
};

class SymbolTable;

// per-compilation context
class CompilationContext {
private:
  std::map<std::string, FunctionContext> function_contexts;
  CompileTarget compile_target = CompileTarget::Unknown;
  TargetArch arch = TargetArch::Unknown;
  OutputKind out_kind = OutputKind::TargetExecutable;
  uint8_t opt_level = 0;

private:
  // compiler configurations
  bool debug_symtab = false;
  bool dump_ast = false;
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

private:
  std::shared_ptr<SymbolTable> sym_tab = nullptr; // global symbol table

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

  TargetArch GetArch() const { return arch; }
  void SetArch(TargetArch ta) { arch = ta; }

  uint8_t GetOptimizationLevel() const { return opt_level; }
  void SetOptimizationLevel(uint8_t lv) { opt_level = lv; }

  OutputKind GetOutputKind() { return out_kind; }
  void SetOutputKind(OutputKind ok) { out_kind = ok; }

  size_t GetMemCapacity(Storage sto) const {
    switch (arch) {
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

    default: choreo_unreachable("Unsupported target arch.");
    }
    return 0;
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
