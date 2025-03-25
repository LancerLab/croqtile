#ifndef __CHOREO_CODEGEN_TOPSCC_HPP__
#define __CHOREO_CODEGEN_TOPSCC_HPP__

#include <filesystem>
#include <iostream>
#include <sstream>
#include <thread>

#include "ast.hpp"
#include "codegen.hpp"
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

inline const char* NameBaseType(BaseType ft, bool co_only = true) {
  switch (ft) {
  case BaseType::DOUBLE: return "double";
  case BaseType::FLOAT: return "float";
  case BaseType::F32: return "float";
  case BaseType::HALF:
  case BaseType::F16: return (co_only) ? "choreo::half" : "choreo::f16";
  case BaseType::BFP16:
  case BaseType::BF16: return (co_only) ? "choreo::bfloat16" : "choreo::bf16";
  case BaseType::HALF8:
  case BaseType::F8: return "choreo::f8";
  case BaseType::U32: return "unsigned int";
  case BaseType::U16: return "unsigned short";
  case BaseType::U8: return "unsigned char";
  case BaseType::INT:
  case BaseType::S32: return "int";
  case BaseType::S16: return "short";
  case BaseType::S8: return "char";
  case BaseType::BOOL: return "bool";
  default: choreo_unreachable("unsupported base-type: " + STR(ft) + ".");
  }
  return "";
}

// map choreo symbols to the generated host, device names
class ScopedSymbolMap {
  using SymbolMap = std::unordered_map<std::string, std::string>;
  std::vector<SymbolMap> host_map;
  std::vector<SymbolMap> device_map;

public:
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
    host_map.back()[csym] = name;
  }
  void MapDeviceSymbol(const std::string& csym, const std::string& name) {
    assert(PrefixedWith(csym, "::") && "expect a scoped name.");
    assert(!device_map.back().count(csym) && "symbol existed");
    device_map.back()[csym] = name;
  }
  void MapDeviceSymbolIfNotExist(const std::string& csym,
                                 const std::string& name) {
    assert(PrefixedWith(csym, "::") && "expect a scoped name.");
    if (!device_map.back().count(csym)) MapDeviceSymbol(csym, name);
  }

  void DumpDeviceMap() {
    dbgs() << "==================== Device Map Information ===================="
           << std::endl;
    dbgs() << "Symbol -> Buffer Name Mapping:" << "\n";
    dbgs() << "--------------------------------------------------------------"
           << "\n";

    for (auto& table : device_map) {
      // Print a formatted table with columns for symbol and buffer name
      dbgs() << std::setw(30) << std::left << "Symbol" << std::setw(50)
             << std::left << "Buffer Name" << "\n";
      dbgs() << "--------------------------------------------------------------"
             << "\n";

      for (const auto& entry : table) {
        dbgs() << std::setw(30) << std::left << entry.first // Symbol
               << std::setw(50) << std::left << entry.second
               << "\n"; // Buffer Name
      }
    }

    dbgs() << "--------------------------------------------------------------"
           << "\n";
    dbgs() << "================================================================"
           << "\n";
  }

  // only for specific purpose
  void RemapDeviceSymbol(const std::string& csym, const std::string& name) {
    assert(PrefixedWith(csym, "::") && "expect a scoped name.");
    device_map.back()[csym] = name;
  }

  const std::string HostName(const std::string& csym) const {
    for (auto mapit = host_map.rbegin(); mapit != host_map.rend(); ++mapit)
      if (mapit->count(csym)) return (*mapit).at(csym);
    return csym;
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
  ptr<CodeGenInfo> cgi;
  ScopedSymbolMap ssm;

public:
  TopsccCodeGen(const ptr<CodeGenInfo>& ci)
      : CodeGenerator("codegen", CCtx().GetGlobalSymbolTable()), cgi(ci) {
    cu_name = "__choreo_" + OptionRegistry::GetInstance().GetInputName();
    cmp_dir = CreateUniquePath();
  }

  bool BeforeVisitImpl(AST::Node&) override;
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
  bool Visit(AST::ParamList&) override { return true; };
  bool Visit(AST::Memory&) override { return true; };
  bool Visit(AST::ChunkAt&) override { return true; };
  bool Visit(AST::Select&) override { return true; };
  bool Visit(AST::LoopRange&) override { return true; };
  bool Visit(AST::Program&) override { return true; };

  bool Visit(AST::WithIn&) override;
  bool Visit(AST::WhereBind&) override;
  bool Visit(AST::WithBlock&) override;
  bool Visit(AST::ForeachBlock&) override;
  bool Visit(AST::InThreadsBlock&) override;
  bool Visit(AST::Assignment&) override;
  bool Visit(AST::ParallelBy&) override;
  bool Visit(AST::DMA&) override;
  bool Visit(AST::Wait&) override;
  bool Visit(AST::PrintNode&) override;
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

  std::string cmp_dir; // work directory
  std::string cu_name; // compilation unit name

  std::string device_fn; // current device function name

  std::string h_indent; // host indentation
  std::string d_indent; // device indentation

  int parallel_level = 0;
  int max_parallel_level = 0;
  bool max_parallel_level_valid = false;

  size_t host_param_count = 0; // host parameter count

  ptr<FunctionType> fty = nullptr; // current function type
  bool void_return = false;

  SDimsInfo symbolic_dimensions;

  std::ostringstream ds; // device stream
  std::ostringstream hs; // host stream

  std::map<std::string, std::string> claimed_dte;
  std::vector<std::string> pld_checklist = {};

private:
  void EmitFixedHostHead();
  void EmitFixedDeviceHead();

  void EmitHostFuncDecl(std::ostringstream&);
  void EmitDeviceFuncDecl(std::ostringstream&);

  void EmitSource();
  void EmitScript(std::ostream& os, const std::string& exe_fn = "");
  bool CompileWithScript(const std::string&);
  void EmitHostRuntimeCheck();

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

private:
  void ResetChoreoFunctionStates() {
    host_param_count = 0; // reset the count of host parameter
    symbolic_dimensions.clear();
    claimed_dte.clear();
    fty = nullptr;
    void_return = false;
  }

  std::string GenHostParamName() {
    return "hp" + std::to_string(host_param_count++);
  }

  FilterRange<SymbolDetail> GetDeviceFuncIns() {
    return cgi->GetDeviceAllocatables(fname);
  }

  FilterRange<SymbolDetail> GetChoreoFuncIns() {
    return cgi->GetParameters(fname);
  }

  const FutureBufferInfo& FBInfo() const {
    return FCtx(fname).GetFutureBufferInfo();
  }

  // check if the placeholder buffer exists
  // this check can only be processed when all device symbol
  // has been mapped
  void PLDCheck() {
    VST_DEBUG(ssm.DumpDeviceMap());
    for (size_t idx = 0; idx < pld_checklist.size(); ++idx) {
      auto pld_name = pld_checklist[idx];
      assert(ssm.HasDeviceName(pld_name) && "buffer has been defined");
    }
  }

  bool IsChoreoInput(const std::string& sname) {
    assert(PrefixedWith(sname, "::") && "expect a scoped name.");
    for (auto& item : GetChoreoFuncIns())
      if (sname == item.name) return true;
    return false;
  }

  std::string GetChoreoInputAtLastPos() {
    auto func_ins = GetChoreoFuncIns(); // 获取所有 items
    return func_ins.back()->name;       // 获取最后一个 item 的 name
  }

  bool HasChoreoOutput() { return !void_return; }

  bool IsChoreoOutput(const std::string& sname) {
    assert(PrefixedWith(sname, "::") && "expect a scoped name.");
    return cgi->IsReturnSymbol(fname, sname);
  }

  bool IsHostSide() const {
    // if current stmt not enter parallel-by btw device and host
    // max_parallel_level is not set or set to 0 or has explicit distance to
    // inner-most
    return (max_parallel_level == 0 || parallel_level + 1 < max_parallel_level);
  }

  bool IsHostSymbol(const std::string& sym) const {
    int count = 0;
    size_t pos = 0;
    std::string target = "paraby";
    int host_side_parallel_lv_cnt = std::max(max_parallel_level - 2, 0);

    // find the target substring from the current position
    while ((pos = sym.find(target, pos)) != std::string::npos) {
      count++;
      pos += target.length(); // Move pos to the end of the found target
    }

    return (count <= host_side_parallel_lv_cnt);
  }

  bool NeedDeviceFunc() const { return cgi->HasParallelBy(fname); }

  bool IsFutureBlockShared(const std::string& n) const {
    assert(PrefixedWith(n, "::") && "requires a scoped name.");
    return cgi->GetFunctionSharedFutures(fname).count(n);
  }
  bool IsDMABlockShared(AST::DMA&) const {
    return (parallel_level == 1) && (max_parallel_level == 2);
  }

  const std::string ValueSTR(const ValueItem& vi) const;
  const std::string ExprSTR(AST::ptr<AST::Node>, bool is_host = true) const;
};

} // namespace Topscc

} // end namespace Choreo

#endif // __CHOREO_CODEGEN_TOPSCC_HPP__
