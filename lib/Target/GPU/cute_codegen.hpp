#ifndef __CHOREO_CODEGEN_CUTE_HPP__
#define __CHOREO_CODEGEN_CUTE_HPP__

#include <filesystem>
#include <iostream>
#include <sstream>
#include <thread>
#include <unordered_set>

#include "ast.hpp"
#include "codegen.hpp"
#include "operator_info.hpp"
#include "types.hpp"

using namespace Choreo;

namespace Choreo {

namespace Cute {

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

inline const char* NameBaseType(BaseType bt) {
  switch (bt) {
  case BaseType::F64: return "double";
  case BaseType::TF32: return "tf32";
  case BaseType::F32: return "float";
  case BaseType::F16: return "f16";
  case BaseType::BF16: return "bf16";
  case BaseType::F8_E4M3: return "f8_e4m3";
  case BaseType::F8_E5M2: return "f8_e5m2";
  case BaseType::F8_UE4M3: return "f8_ue4m3";
  case BaseType::F8_UE8M0: return "f8_ue8m0";
  case BaseType::F6_E2M3: return "choreo::f6_e2m3";
  case BaseType::F6_E3M2: return "choreo::f6_e3m2";
  case BaseType::F4_E2M1: return "choreo::f4_e2m1";
  case BaseType::U64: return "unsigned long long";
  case BaseType::U32: return "unsigned int";
  case BaseType::U16: return "unsigned short";
  case BaseType::U8: return "unsigned char";
  case BaseType::S64: return "long long";
  case BaseType::S32: return "int";
  case BaseType::S16: return "short";
  case BaseType::S8: return "signed char";
  case BaseType::U6: return "uint6b_t";
  case BaseType::U4: return "uint4b_t";
  case BaseType::U2: return "uint2b_t";
  case BaseType::U1: return "uint1b_t";
  case BaseType::S6: return "int6b_t";
  case BaseType::S4: return "int4b_t";
  case BaseType::S2: return "int2b_t";
  case BaseType::BIN1: return "bin1_t";
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

  void DumpHostMap() {
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
  void DumpDeviceMap() {
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

struct CuteCodeGen : public CodeGenerator {
private:
  // Only use it for function parameters.
  CodeGenInfo updating_cgi;
  ScopedSymbolMap ssm;

public:
  CuteCodeGen() : CodeGenerator("codegen") {
    cu_name = "__choreo_" + OptionRegistry::GetInstance().GetInputName();
    cmp_dir = CreateUniquePath();
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
  bool Visit(AST::MMA&) override;
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

  std::string cmp_dir; // work directory
  std::string cu_name; // compilation unit name

  std::string device_fn; // current device function name

  std::string h_indent; // host indentation
  std::string d_indent; // device indentation

  std::stack<ParallelLevel> levels;
  ParallelLevel Level() const { return levels.top(); }
  bool IsParallel() const { return levels.size() > 2; }
  bool NeedLevelPred() const {
    return IsParallel() && (Level() != ParallelLevel::THREAD);
  }

  // idx of the most outer pb
  int parallel_idx = -1;
  AST::ParallelBy* cur_pb = nullptr;

  size_t host_param_count = 0; // host parameter count

  ptr<FunctionType> fty = nullptr; // current function type
  bool void_return = false;

  SDimsInfo symbolic_dimensions;

  std::ostringstream ds;            // device stream
  std::ostringstream hs;            // host stream
  std::ostringstream return_stream; // stream for return node

  std::map<std::string, std::string> claimed_futs;
  std::vector<std::string> pld_checklist = {};

  std::set<std::string> global_buffers; // global buffers
  bool emit_call = true;                // emit the call statement

  std::set<std::string> cooperatives; // futures with cooperative-dma
  std::unordered_set<std::string> async_subbyte_futures;

  // mma related
  size_t reg_num_d;
  // once the flag is set, always use dynamic reuse!
  bool set_cuda_func_attribute_max_dynamic_shared_memory_size = false;
  static const std::string vid_pfx;
  // block dim enforcement level, default to thread level
  ParallelLevel bdim_level = ParallelLevel::THREAD;
  // TODO: for now, only support one stream!
  std::string stream_name;

private:
  void EmitFixedHostHead();
  void EmitFixedDeviceHead();

  void EmitHostFuncDecl(std::ostringstream&);
  void EmitDeviceFuncDecl(std::ostringstream&, AST::ParallelBy*,
                          const ValueItem& cur_ring_offset);

  void EmitSource();
  void EmitScript(std::ostream& os, const std::string& exe_fn = "");
  bool CompileWithScript(const std::string&);

  void EmitHostRuntimeCheck();
  void EmitDeviceVirtualIndices(AST::ParallelBy*);
  // emit mem reuse script for each device function.
  void EmitMemReuse(const std::string& dev_func_name);
  void EmitCudaFree();
  void EmitRuntimeEnvironmentChecker(std::ostream&) const;

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
  const std::string Indent() const { return IsHost() ? h_indent : d_indent; }
  void IncrIndent() { return IsHost() ? IncrHostIndent() : IncrDeviceIndent(); }
  void DecrIndent() { return IsHost() ? DecrHostIndent() : DecrDeviceIndent(); }

private:
  void ResetChoreoFunctionStates() {
    host_param_count = 0; // reset the count of host parameter
    symbolic_dimensions.clear();
    claimed_futs.clear();
    async_subbyte_futures.clear();
    fty = nullptr;
    void_return = false;
    emit_call = true;
    parallel_idx = -1;
  }

  std::string GenHostParamName() {
    return "hp" + std::to_string(host_param_count++);
  }

  // return all the parameters of device function in code.
  FilterRange<SymbolDetail> GetDeviceFuncIns(CodeGenInfo& info) const {
    return info.GetDeviceAllIns(fname);
  }

  FilterRange<SymbolDetail> GetChoreoFuncIns(CodeGenInfo& info) const {
    return info.GetParameters(fname);
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
    return cgi.IsReturnSymbol(fname, sname);
  }

  bool IsHostSymbol(const std::string& sym) const {
    assert(PrefixedWith(sym, "::") && "expect a scoped name.");
    // host symbol does not have any paraby
    return sym.find("::paraby") == std::string::npos;
  }

  bool NeedDeviceFunc() const { return cgi.HasParallelBy(fname); }

  bool IsHost() const { return Level() == ParallelLevel::SEQ; }

  bool IsFutureBlockShared(const std::string& n) const {
    assert(PrefixedWith(n, "::") && "requires a scoped name.");
    return cgi.GetFunctionSharedFutures(fname).count(n);
  }
  bool IsFutureWarpLocal(const std::string& n) const {
    assert(PrefixedWith(n, "::") && "requires a scoped name.");
    return cgi.GetFunctionLocalFutures(fname).count(n);
  }

  const std::string ExprCastSTR(AST::ptr<AST::Node> n,
                                std::optional<std::variant<int, float>> val,
                                BaseType to, BaseType from,
                                bool is_host = true) const;

  const std::string ValueSTR(const ValueItem& vi, bool = false,
                             bool = false) const;
  const std::string ValueSTR(const ValueList& vl, bool LL_suffix = false,
                             bool shp_lit = false,
                             const std::string& sep = ", ") const;
  const std::string OpValueSTR(const ValueItem& vi,
                               const std::string& parent_op,
                               const bool is_left_child, bool LL_suffix = false,
                               bool = false) const;
  const std::string ExprSTR(AST::ptr<AST::Node>, bool is_host = true) const;
  const std::string OpExprSTR(AST::ptr<AST::Node>, const std::string& parent_op,
                              const bool is_left_child, bool is_host) const;
  const std::string CallSTR(AST::Call&) const;

  std::optional<std::string> ThreadIdString(const ptr<AST::Identifier>&) const;
  std::pair<std::string, size_t> GenMdsOffset(const ptr<AST::ChunkAt>,
                                              ptr<DMAConfig> = nullptr) const;
  const ValueList GenIndices(const ptr<AST::ChunkAt>&,
                             const ptr<DMAConfig>& = nullptr) const;
  const std::string TileBaseOffset(const ptr<AST::ChunkAt>&) const;
  const ValueItem
  GenOffset(const ptr<AST::ChunkAt>&,
            size_t end_idx = std::numeric_limits<size_t>::max()) const;
  const ValueList GenStrides(const Shape& shape,
                             const std::vector<size_t>& = {}) const;
  const ValueList GenStrides(const ptr<AST::ChunkAt>&,
                             const std::vector<size_t>& = {}) const;
  const std::string ShapeSTR(const Shape&, bool = false,
                             const std::string& = ", ",
                             BaseType cast_to = BaseType::UNKNOWN) const;
  const std::string ReShapeSTR(const Shape&, const std::vector<size_t>&,
                               bool = false, const std::string& = ", ") const;
  const std::string SSMName(const std::string& sname, bool is_host) const {
    return (is_host) ? ssm.HostName(sname) : ssm.DeviceName(sname);
  }

  bool ThreadCooperative(AST::DMA&) const;
  bool HasWGMMAInFunction() const;
  std::pair<std::string, std::string>
  GenTensorDecl(const std::string& name, const std::string& buf_expr,
                const Storage sto, BaseType bty, const Shape& shp,
                bool is_host = false, const std::string& offset = "",
                const std::string& strides = "",
                const std::vector<size_t>& transp = {},
                bool use_wgmma_layout = false, int swizzle_value = 128) const;
  void EmitTMAConfiguration(AST::ParallelBy* pb);
  const std::optional<std::string> GetTMAName(AST::DMA&) const;

  size_t GetRegNumOfFrag(ValueItem m, ValueItem n) {
    auto mi = VIInt(m);
    auto ni = VIInt(n);
    if (!mi || !ni)
      choreo_unreachable("expect m and n of mma to be numeric value!");
    return mi.value() * ni.value() / CCtx().GetMinGroupDim();
  }

  void UseUint32Reg(bool& use_uint32, size_t& reg_num, BaseType bt) {
    if (bt != BaseType::F32 && bt != BaseType::F64) {
      use_uint32 = true;
      reg_num /= 4 / SizeOf(bt);
    }
  }

  const std::string EmitSpannedArith(AST::Expr& e) const;
};

} // namespace Cute

} // end namespace Choreo

#endif // __CHOREO_CODEGEN_CUTE_HPP__
