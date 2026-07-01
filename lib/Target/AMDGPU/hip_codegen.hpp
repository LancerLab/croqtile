#ifndef __CHOREO_CODEGEN_HIP_HPP__
#define __CHOREO_CODEGEN_HIP_HPP__

#include <iostream>
#include <sstream>
#include <stack>

#include "ast.hpp"
#include "codegen.hpp"
#include "codegen_utils.hpp"
#include "hip_dma_plan.hpp"
#include "operator_info.hpp"
#include "types.hpp"

using namespace Choreo;

namespace Choreo {
namespace HIP {

inline const char* HIPNameBaseType(BaseType bt) {
  switch (bt) {
  case BaseType::F64: return "double";
  case BaseType::F32: return "float";
  case BaseType::F16: return "__half";
  case BaseType::BF16: return "hip_bfloat16";
  case BaseType::U64: return "unsigned long long";
  case BaseType::U32: return "unsigned int";
  case BaseType::U16: return "unsigned short";
  case BaseType::U8: return "unsigned char";
  case BaseType::S64: return "long long";
  case BaseType::S32: return "int";
  case BaseType::S16: return "short";
  case BaseType::S8: return "signed char";
  case BaseType::BOOL: return "bool";
  default: choreo_unreachable("unsupported base-type: " + STR(bt) + ".");
  }
  return "";
}

using ScopedSymbolMap = ::Choreo::ScopedSymbolMap;

struct HIPCodeGen : public CodeGenerator {
private:
  ScopedSymbolMap ssm;

public:
  HIPCodeGen() : CodeGenerator("hip-codegen") {
    hip_name = "__choreo_" + OptionRegistry::GetInstance().GetInputName();
    cmp_dir = CreateUniquePath();
  }

  bool BeforeVisitImpl(AST::Node&) override;
  bool InMidVisitImpl(AST::Node&) override;
  bool AfterVisitImpl(AST::Node&) override;

  bool Visit(AST::MultiNodes&) override { return true; }
  bool Visit(AST::MultiValues&) override { return true; }
  bool Visit(AST::IntLiteral&) override { return true; }
  bool Visit(AST::FloatLiteral&) override { return true; }
  bool Visit(AST::Expr&) override { return true; }
  bool Visit(AST::MultiDimSpans&) override { return true; }
  bool Visit(AST::IntTuple&) override { return true; }
  bool Visit(AST::IntIndex&) override { return true; }
  bool Visit(AST::DataType&) override { return true; }
  bool Visit(AST::Identifier&) override { return true; }
  bool Visit(AST::Parameter&) override { return true; }
  bool Visit(AST::Memory&) override { return true; }
  bool Visit(AST::ChunkAt&) override { return true; }
  bool Visit(AST::Select&) override { return true; }
  bool Visit(AST::LoopRange&) override { return true; }
  bool Visit(AST::Program&) override { return true; }

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
  bool Visit(AST::Yield&) override;
  bool Visit(AST::Rotate&) override;
  bool Visit(AST::Synchronize&) override;
  bool Visit(AST::Call&) override;
  bool Visit(AST::NamedVariableDecl&) override;
  bool Visit(AST::NamedTypeDecl&) override;
  bool Visit(AST::CppSourceCode& n) override;
  bool Visit(AST::ChoreoFunction&) override;
  bool Visit(AST::FunctionDecl&) override;
  bool Visit(AST::Return&) override;

private:
  std::string cmp_dir;
  std::string hip_name;
  std::string device_fn;

  std::string h_indent;
  std::string d_indent;

  std::stack<ParallelLevel> levels;
  ParallelLevel Level() const { return levels.top(); }
  bool IsParallel() const { return levels.size() > 2; }
  bool NeedLevelPred() const {
    return IsParallel() && (Level() != ParallelLevel::THREAD);
  }

  std::unordered_map<std::string, int> emitted_device_names_;
  std::string UniqueDeviceName(const std::string& name) {
    auto it = emitted_device_names_.find(name);
    if (it == emitted_device_names_.end()) {
      emitted_device_names_[name] = 0;
      return name;
    }
    return name + "__" + std::to_string(++it->second);
  }

  int parallel_idx = -1;
  AST::ParallelBy* cur_pb = nullptr;
  ParallelLevel bdim_level = ParallelLevel::THREAD;

  size_t host_param_count = 0;
  ptr<FunctionType> fty = nullptr;
  bool void_return = false;

  struct SymDimInfo {
    std::string hsd_expr;
    size_t param_index;
    size_t dim_index;
  };
  std::map<std::string, SymDimInfo> symbolic_dimensions;

  std::ostringstream ds;
  std::ostringstream hs;
  std::ostringstream return_stream;

  std::set<std::string> global_buffers;
  std::vector<std::string> event_global_buffers;
  bool emit_call = true;
  std::vector<std::string> code_segments;

  bool IsHost() const { return Level() == ParallelLevel::SEQ; }

  void IncrHostIndent() { h_indent += "  "; }
  void IncrDeviceIndent() { d_indent += "  "; }
  void DecrHostIndent() {
    if (h_indent.size() >= 2)
      h_indent = h_indent.substr(0, h_indent.size() - 2);
  }
  void DecrDeviceIndent() {
    if (d_indent.size() >= 2)
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
  void IncrIndent() { IsHost() ? IncrHostIndent() : IncrDeviceIndent(); }
  void DecrIndent() { IsHost() ? DecrHostIndent() : DecrDeviceIndent(); }

  std::string GenHostParamName() {
    return "hp" + std::to_string(host_param_count++);
  }

  FilterRange<SymbolDetail> GetDeviceFuncIns(CodeGenInfo& info) const {
    return info.GetDeviceAllIns(fname);
  }
  FilterRange<SymbolDetail> GetChoreoFuncIns(CodeGenInfo& info) const {
    return info.GetParameters(fname);
  }

  bool NeedDeviceFunc() const { return cgi.HasParallelBy(fname); }

  const std::string ExprSTR(AST::ptr<AST::Node>, bool is_host = true) const;
  const std::string OpExprSTR(AST::ptr<AST::Node>, const std::string&, bool,
                              bool) const;
  const std::string CallSTR(AST::Call&) const;
  const std::string ValueSTR(const ValueItem& vi, bool = false,
                             bool = false) const;
  const std::string ValueSTR(const ValueList& vl, bool = false, bool = false,
                             const std::string& sep = ", ") const;
  const std::string ShapeSTR(const Shape&, bool = false,
                             const std::string& = ", ",
                             BaseType = BaseType::UNKNOWN) const;

  std::string GenOffset(const AST::ptr<AST::ChunkAt>& ca) const;

  void EmitFixedHostHead();
  void EmitFixedDeviceHead();
  void EmitSource();
  void EmitScript(std::ostream& os, const std::string& exe_fn = "");
  bool CompileWithScript(const std::string&);

  void EmitHostFuncDecl(std::ostringstream&);
  void EmitDeviceFuncDecl(std::ostringstream&, AST::ParallelBy*,
                          const ValueItem&);
  void EmitDeviceVirtualIndices(AST::ParallelBy*);
  void EmitHipFree();

  void EmitDMACopy(const HIPDMALoweringDecision& dec,
                   const std::string& from_expr, const std::string& to_expr,
                   const ptr<SpannedType>& to_sty);
  void EmitDMAPad(AST::DMA& n, const HIPDMALoweringDecision& dec,
                  const std::string& from_expr, const std::string& to_expr,
                  const ptr<SpannedType>& from_sty,
                  const ptr<SpannedType>& to_sty);
  void EmitDMATranspose(AST::DMA& n, const HIPDMALoweringDecision& dec,
                        const std::string& from_expr,
                        const std::string& to_expr,
                        const ptr<SpannedType>& from_sty,
                        const ptr<SpannedType>& to_sty);

  void ResetChoreoFunctionStates() {
    host_param_count = 0;
    symbolic_dimensions.clear();
    fty = nullptr;
    void_return = false;
    emit_call = true;
    parallel_idx = -1;
    bdim_level = ParallelLevel::THREAD;
    emitted_device_names_.clear();
    event_global_buffers.clear();
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
  std::string SSMName(const std::string& sname, bool is_host) const {
    return is_host ? ssm.HostName(sname) : ssm.DeviceName(sname);
  }

  CodeGenInfo updating_cgi;
};

} // namespace HIP
} // namespace Choreo

#endif // __CHOREO_CODEGEN_HIP_HPP__
