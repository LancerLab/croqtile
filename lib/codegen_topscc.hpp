#ifndef __CHOREO_CODEGEN_TOPSCC_HPP__
#define __CHOREO_CODEGEN_TOPSCC_HPP__

#include <iostream>
#include <sstream>

#include "ast.hpp"
#include "codegen.hpp"
#include "io.hpp"
#include "types.hpp"

using namespace Choreo;

namespace Choreo {

namespace Topscc {

struct SymbolicDimensionInfo {
  std::string hsd_expr; // (host) expression of the symbolic dimension
  size_t param_index;   // index of the parameter that gives the shape with the
                        // dimension
  size_t dim_index;     // dimension index inside a shape
};

// information about choreo symbolic dimensions
using SDimsInfo = std::map<std::string, SymbolicDimensionInfo>;

enum CodeSegment {
  CS_UNKNOWN,
  CS_USER,
  CS_COK,
  CS_CO,
};

inline const char* NameBaseType(BaseType bt, bool co_only = true) {
  switch (bt) {
  case BaseType::F64: return "double";
  case BaseType::F32: return "float";
  case BaseType::F16: return (co_only) ? "choreo::half" : "choreo::f16";
  case BaseType::BF16: return (co_only) ? "choreo::bfloat16" : "choreo::bf16";
  case BaseType::F8: return "choreo::f8";
  case BaseType::U64: return "unsigned long long";
  case BaseType::U32: return "unsigned int";
  case BaseType::U16: return "unsigned short";
  case BaseType::U8: return "unsigned char";
  case BaseType::S64: return "long long";
  case BaseType::S32: return "int";
  case BaseType::S16: return "short";
  case BaseType::S8: return "char";
  case BaseType::BOOL: return "bool";
  default: choreo_unreachable("unsupported base-type: " + STR(bt) + ".");
  }
  return "";
}

// map choreo symbols to the generated host, device names
class ScopedSymbolMap {
  using SymbolMap = std::unordered_map<std::string, std::string>;
  std::vector<SymbolMap> host_map;
  std::vector<SymbolMap> device_map;
  bool debug;

public:
  ScopedSymbolMap(bool d = false) : debug(d) {}
  void EnterScope() {
    host_map.push_back({});
    device_map.push_back({});
  }
  void LeaveScope() {
    host_map.pop_back();
    device_map.pop_back();
  }
  void MapHostSymbol(const std::string& csym, const std::string& name) {
    assert(!host_map.back().count(csym) && "symbol existed");
    if (debug)
      dbgs() << "[Host] Map symbol: " << csym << " -> " << name << "\n";
    host_map.back()[csym] = name;
  }
  void MapDeviceSymbol(const std::string& csym, const std::string& name) {
    assert(PrefixedWith(csym, "::") && "expect a scoped name.");
    assert(!device_map.back().count(csym) && "symbol existed");
    if (debug)
      dbgs() << "[Device] Map symbol: " << csym << " -> " << name << "\n";
    device_map.back()[csym] = name;
  }
  void MapDeviceSymbolIfNotExist(const std::string& csym,
                                 const std::string& name) {
    assert(PrefixedWith(csym, "::") && "expect a scoped name.");
    if (!device_map.back().count(csym)) {
      if (debug)
        dbgs() << "[Device] Map symbol: " << csym << " -> " << name << "\n";
      MapDeviceSymbol(csym, name);
    }
  }

  void DumpHostMap() const {
    dbgs()
        << "==================== Host Map Information ====================\n";
    // Print a formatted table with columns for symbol and buffer name
    dbgs() << std::setw(30) << std::left << "Symbol" << std::setw(50)
           << std::left << " -> Host Name" << "\n";
    dbgs()
        << "--------------------------------------------------------------\n";

    for (auto& table : host_map) {
      if (table.empty()) continue;
      for (const auto& entry : table) {
        dbgs() << std::setw(30) << std::left << entry.first // Symbol
               << " -> " << entry.second << "\n";           // Buffer Name
      }
    }

    dbgs() << "================================================================"
           << "\n";
  }
  void DumpDeviceMap() const {
    dbgs()
        << "==================== Device Map Information ====================\n";
    // Print a formatted table with columns for symbol and buffer name
    dbgs() << std::setw(30) << std::left << "Symbol" << std::setw(50)
           << std::left << " -> Device Name" << "\n";
    dbgs()
        << "----------------------------------------------------------------\n";

    for (auto& table : device_map) {
      if (table.empty()) continue;

      for (const auto& entry : table) {
        dbgs() << std::setw(30) << std::left << entry.first // Symbol
               << " -> " << entry.second << "\n";           // Buffer Name
      }
    }

    dbgs() << "================================================================"
           << "\n";
  }

  // only for specific purpose
  void RemapDeviceSymbol(const std::string& csym, const std::string& name) {
    assert(PrefixedWith(csym, "::") && "expect a scoped name.");
    device_map.back()[csym] = name;
  }

  void RemapHostSymbol(const std::string& csym, const std::string& name) {
    assert(PrefixedWith(csym, "::") && "expect a scoped name.");
    host_map.back()[csym] = name;
  }

  const std::string HostName(const std::string& csym) const {
    for (auto mapit = host_map.rbegin(); mapit != host_map.rend(); ++mapit)
      if (mapit->count(csym)) return (*mapit).at(csym);
    return csym;
  }

  bool HasHostName(const std::string& csym) const {
    for (auto mapit = host_map.rbegin(); mapit != host_map.rend(); ++mapit)
      if (mapit->count(csym)) return true;
    return false;
  }

  const std::string DeviceName(const std::string& csym) const {
    for (auto mapit = device_map.rbegin(); mapit != device_map.rend(); ++mapit)
      if (mapit->count(csym)) return (*mapit).at(csym);
    return csym;
  }

  bool HasDeviceName(const std::string& csym) const {
    for (auto mapit = device_map.rbegin(); mapit != device_map.rend(); ++mapit)
      if (mapit->count(csym)) return true;
    return false;
  }

  const std::string DeviceNameOrNull(const std::string& csym) const {
    for (auto mapit = device_map.rbegin(); mapit != device_map.rend(); ++mapit)
      if (mapit->count(csym)) return (*mapit).at(csym);
    return "";
  }
};

struct TopsccCodeGen : public CodeGenerator {
private:
  // fixed codegen info, which is not updated during the TopsccCodegen
  ptr<CodeGenInfo> cgi;
  // update when visiting nodes in TopsccCodegen at any time.
  // Only use it for function parameters.
  ptr<CodeGenInfo> updating_cgi;
  ScopedSymbolMap ssm;
  ptr<Loop> cur_loop;

public:
  TopsccCodeGen(const ptr<CodeGenInfo>& ci)
      : CodeGenerator("codegen", CCtx().GetGlobalSymbolTable()), cgi(ci) {
    updating_cgi = AST::Make<CodeGenInfo>();
    cur_loop = nullptr;
  }

  bool BeforeVisitImpl(AST::Node&) override;
  bool InMidVisitImpl(AST::Node&) override;
  bool AfterVisitImpl(AST::Node&) override;

  bool Visit(AST::MultiNodes&) override { return true; };
  bool Visit(AST::MultiValues&) override { return true; };
  bool Visit(AST::IntLiteral&) override { return true; };
  bool Visit(AST::FloatLiteral&) override { return true; };
  bool Visit(AST::Expr&) override { return true; };
  bool Visit(AST::MultiDimSpans&) override { return true; };
  bool Visit(AST::NamedTypeDecl&) override { return true; };
  bool Visit(AST::IntTuple&) override { return true; };
  bool Visit(AST::IntIndex&) override { return true; };
  bool Visit(AST::DataType&) override { return true; };
  bool Visit(AST::Identifier&) override { return true; };
  bool Visit(AST::Parameter&) override { return true; };
  bool Visit(AST::Memory&) override { return true; };
  bool Visit(AST::ChunkAt&) override { return true; };
  bool Visit(AST::Select&) override { return true; };
  bool Visit(AST::LoopRange&) override { return true; };
  bool Visit(AST::Program&) override { return true; };

  bool Visit(AST::ParamList&) override;
  bool Visit(AST::WithIn&) override;
  bool Visit(AST::WhereBind&) override;
  bool Visit(AST::WithBlock&) override;
  bool Visit(AST::ForeachBlock&) override;
  bool Visit(AST::InThreadsBlock&) override;
  bool Visit(AST::IfElseBlock&) override;
  bool Visit(AST::WhileBlock&) override;
  bool Visit(AST::Assignment&) override;
  bool Visit(AST::ParallelBy&) override;
  bool Visit(AST::DMA&) override;
  bool Visit(AST::Wait&) override;
  bool Visit(AST::Trigger&) override;
  bool Visit(AST::Break&) override;
  bool Visit(AST::Continue&) override;
  bool Visit(AST::Rotate&) override;
  bool Visit(AST::Synchronize&) override;
  bool Visit(AST::Call&) override;
  bool Visit(AST::NamedVariableDecl&) override;
  bool Visit(AST::CppSourceCode& n) override;
  bool Visit(AST::ChoreoFunction&) override;
  bool Visit(AST::FunctionDecl&) override;
  bool Visit(AST::Return&) override;

private:
  CodeSegment cs = CS_UNKNOWN;
  std::vector<std::string> code_segments; // multiple code segment

  std::string device_fn; // current device function name

  std::string h_indent; // host indentation
  std::string d_indent; // device indentation

  Storage parallel_level = Storage::NONE;
  Storage max_parallel_level = Storage::NONE;
  std::vector<Storage> pl_stack;
  // idx of the most outer pb
  int parallel_idx = -1;

  size_t host_param_count = 0;     // host parameter count
  ptr<FunctionType> fty = nullptr; // current function type
  bool void_return = false;

  SDimsInfo symbolic_dimensions;

  std::ostringstream ds;            // device stream
  std::ostringstream hs;            // host stream
  std::ostringstream return_stream; // stream for return node

  std::map<std::string, std::string> claimed_dte;
  std::vector<std::string> pld_checklist = {};

  std::set<std::string> global_buffers; // global buffers
  bool emit_call = true;                // emit the call statement

private:
  void EmitFixedHostHead();
  void EmitFixedDeviceHead();

  void EmitHostFuncDecl(std::ostringstream&);
  void EmitDeviceFuncDecl(std::ostringstream&);

  void EmitSource();
  void EmitScript(std::ostream& os, const std::string& exe_fn = "");
  bool CompileWithScript(const std::string&);

  void EmitHostRuntimeCheck();
  // emit mem reuse script for each device function.
  void EmitMemReuse(const std::string& dev_func_name);
  void EmitTopsFree();

private:
  void IncrHostIndent() { h_indent += "  "; }
  void IncrDeviceIndent() { d_indent += "  "; }
  void DecrHostIndent() {
    if (h_indent.size() < 2)
      choreo_unreachable("the indent can not be decreased.");
    h_indent = h_indent.substr(0, h_indent.size() - 2);
  }
  void DecrDeviceIndent() {
    if (d_indent.size() < 2)
      choreo_unreachable("the indent can not be decreased.");
    d_indent = d_indent.substr(0, d_indent.size() - 2);
  }

  std::ostringstream& Stream() { return IsHost() ? hs : ds; }
  std::ostringstream& IndStream() {
    if (IsHost()) {
      hs << h_indent;
      return hs;
    } else {
      ds << d_indent;
      return ds;
    }
  }
  const std::string Indent() { return IsHost() ? h_indent : d_indent; }
  void IncrIndent() { return IsHost() ? IncrHostIndent() : IncrDeviceIndent(); }
  void DecrIndent() { return IsHost() ? DecrHostIndent() : DecrDeviceIndent(); }

private:
  void ResetChoreoFunctionStates() {
    host_param_count = 0; // reset the count of host parameter
    symbolic_dimensions.clear();
    claimed_dte.clear();
    fty = nullptr;
    void_return = false;
    emit_call = true;
    parallel_idx = -1;
  }

  std::string GenHostParamName() {
    return "hp" + std::to_string(host_param_count++);
  }

  // return all the parameters of device function in topscc code.
  FilterRange<SymbolDetail>
  GetDeviceFuncIns(const ptr<CodeGenInfo>& info) const {
    return info->GetDeviceAllIns(fname);
  }

  FilterRange<SymbolDetail>
  GetChoreoFuncIns(const ptr<CodeGenInfo>& info) const {
    return info->GetParameters(fname);
  }

  const FutureBufferInfo& FBInfo() const {
    return FCtx(fname).GetFutureBufferInfo();
  }

  // check if the placeholder buffer exists
  // this check can only be processed when all device symbol
  // has been mapped
  void PLDCheck() {
    VST_DEBUG(ssm.DumpHostMap());
    VST_DEBUG(ssm.DumpDeviceMap());
    for (size_t idx = 0; idx < pld_checklist.size(); ++idx) {
      auto pld_name = pld_checklist[idx];
      assert(ssm.HasDeviceName(pld_name) && "buffer has been defined");
    }
  }

  bool IsChoreoInput(const std::string& sname) {
    assert(PrefixedWith(sname, "::") && "expect a scoped name.");
    for (auto& item : GetChoreoFuncIns(cgi))
      if (sname == item.name) return true;
    return false;
  }

  bool HasChoreoOutput() { return !void_return; }

  bool IsChoreoOutput(const std::string& sname) {
    assert(PrefixedWith(sname, "::") && "expect a scoped name.");
    return cgi->IsReturnSymbol(fname, sname);
  }

  bool IsHostSymbol(const std::string& sym) const {
    assert(PrefixedWith(sym, "::") && "expect a scoped name.");
    // host symbol does not have any paraby
    return sym.find("::paraby") == std::string::npos;
  }

  bool NeedDeviceFunc() const { return cgi->HasParallelBy(fname); }

  bool IsHost() const { return parallel_level == Storage::NONE; }

  bool IsFutureBlockShared(const std::string& n) const {
    assert(PrefixedWith(n, "::") && "requires a scoped name.");
    return cgi->GetFunctionSharedFutures(fname).count(n);
  }
  bool IsFutureWarpLocal(const std::string& n) const {
    assert(PrefixedWith(n, "::") && "requires a scoped name.");
    return cgi->GetFunctionLocalFutures(fname).count(n);
  }

  bool IsDMABlockShared(AST::DMA&) const {
    return (parallel_level == Storage::SHARED) &&
           (max_parallel_level == Storage::LOCAL ||
            max_parallel_level == Storage::SUB);
  }
  bool IsDMAWarpLocal(AST::DMA&) const {
    return (parallel_level == Storage::LOCAL &&
            max_parallel_level == Storage::SUB);
  }

  const std::string ExprCastSTR(AST::ptr<AST::Node> n,
                                std::optional<std::variant<int, float>> val,
                                BaseType to, BaseType from,
                                bool is_host = true) const;

  const std::string ValueSTR(const ValueItem& vi, bool LL_suffix = false) const;
  const std::string ValueListSTR(const ValueList& vl, std::string sep = ", ",
                                 bool LL_suffix = false) const;
  const std::string OpValueSTR(const ValueItem& vi,
                               const std::string& parent_op,
                               const bool is_left_child,
                               bool LL_suffix = false) const;
  const std::string ExprSTR(AST::ptr<AST::Node>, bool is_host = true) const;
  const std::string OpExprSTR(AST::ptr<AST::Node>, const std::string& parent_op,
                              const bool is_left_child, bool is_host) const;
  const std::string CallSTR(AST::Call&) const;
  const std::string DASTR(AST::ptr<AST::DataAccess>&, const std::string& = "",
                          bool is_load = true, bool masking = false) const;
  const std::string BuildTcleLoad(const std::string& addr_str,
                                  const std::string& ty_str) const;
  const std::string BuildTcleStore(const std::string& addr_str,
                                   const std::string& ty_str,
                                   const std::string& val_str) const;

  std::optional<std::string> ThreadIdString(const ptr<AST::Identifier>&) const;
  std::optional<std::string>
  SubThreadIdString(const ptr<AST::Identifier>&) const;
  std::pair<std::string, size_t> GenMdsOffset(const ptr<AST::ChunkAt>,
                                              ptr<DMAConfig> = nullptr) const;
  const std::string TileBaseOffset(const ptr<AST::ChunkAt>&) const;
  const std::string
  GenOffset(const ptr<AST::ChunkAt>&,
            size_t end_idx = std::numeric_limits<size_t>::max()) const;
  const std::string ShapeSTR(const Shape&, const std::string& = ", ",
                             BaseType cast_to = BaseType::UNKNOWN) const;
  const std::string SSMName(const std::string& sname, bool is_host) const {
    return (is_host) ? ssm.HostName(sname) : ssm.DeviceName(sname);
  }
  const std::string AddressOffset(const Shape&, const AST::DataAccess&,
                                  bool) const;
  // if it requires wrapping code in a single thread
  bool RequiresImplPred(Storage) const;
  const std::string VectorTypeSTR(const ptr<Type>& vt) const;
};

} // namespace Topscc

} // end namespace Choreo

#endif // __CHOREO_CODEGEN_TOPSCC_HPP__
