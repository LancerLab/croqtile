#ifndef __CHOREO_CODEGEN_TOPSCC_HPP__
#define __CHOREO_CODEGEN_TOPSCC_HPP__

#include <iostream>
#include <sstream>

#include "ast.hpp"
#include "codegen.hpp"
#include "codegen_utils.hpp"
#include "gcu_mma_codegen.hpp"
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
  case BaseType::U64: return "uint64_t";
  case BaseType::U32: return "unsigned int";
  case BaseType::U16: return "unsigned short";
  case BaseType::U8: return "unsigned char";
  case BaseType::S64: return "int64_t";
  case BaseType::S32: return "int";
  case BaseType::S16: return "short";
  case BaseType::S8: return "char";
  case BaseType::BOOL: return "bool";
  default: choreo_unreachable("unsupported base-type: " + STR(bt) + ".");
  }
  return "";
}

using ScopedSymbolMap = ::Choreo::ScopedSymbolMap;

struct TopsccCodeGen : public CodeGenerator {
private:
  // update when visiting nodes in TopsccCodegen at any time.
  // Only use it for function parameters.
  CodeGenInfo updating_cgi;
  ScopedSymbolMap ssm;
  ptr<Loop> cur_loop;

public:
  TopsccCodeGen() : CodeGenerator("codegen") { cur_loop = nullptr; }

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
  bool Visit(AST::BufferMap&) override;
  bool Visit(AST::Wait&) override;
  bool Visit(AST::Trigger&) override;
  bool Visit(AST::Break&) override;
  bool Visit(AST::AsmStmt&) override;
  bool Visit(AST::Continue&) override;
  bool Visit(AST::Rotate&) override;
  bool Visit(AST::Synchronize&) override;
  bool Visit(AST::Barrier&) override;
  bool Visit(AST::Fence&) override;
  bool Visit(AST::Call&) override;
  bool Visit(AST::NamedVariableDecl&) override;
  bool Visit(AST::CppSourceCode& n) override;
  bool Visit(AST::ChoreoFunction&) override;
  bool Visit(AST::FunctionDecl&) override;
  bool Visit(AST::Return&) override;
  bool Visit(AST::MMA&) override;

private:
  AcoreMMACodeGenState acore_mma;

  std::string ResolveFragAddr(const AST::ptr<AST::Expr>& frag);

  CodeSegment cs = CS_UNKNOWN;
  std::vector<std::string> code_segments; // multiple code segment
  std::vector<CodeSegment> segment_tags;  // per-segment kind for filtering

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

  bool device_defers_launch = false;
  AST::ParallelBy* deferred_device_pb = nullptr;

  size_t host_param_count = 0;     // host parameter count
  ptr<FunctionType> fty = nullptr; // current function type
  bool void_return = false;
  std::string stream_name; // deprecated: kept for ABI, no longer populated
  bool extern_smem; // true: has dynamic smem, decl `extern __shared__ ...`
  ValueItem shared_spm_size;

  SDimsInfo symbolic_dimensions;

  std::ostringstream ds;            // device stream
  std::ostringstream hs;            // host stream
  std::ostringstream return_stream; // stream for return node

  LineDirectiveState host_line_state;
  LineDirectiveState device_line_state;

  std::map<std::string, std::string> claimed_dte;
  int dte_pool_size = 0;
  int anon_dte_slot = -1;
  bool has_nofuture_rotate = false;
  std::map<std::string, int> dte_pool_slots;
  std::map<int, std::string> dte_named_vars;
  std::set<std::string> waited_futures; // futures that have been waited

  struct NoFutureInfo {
    std::string data_ptr;
    std::string event_var;
    int dte_slot;
  };
  std::map<std::string, NoFutureInfo> nofuture_vars;
  std::vector<std::string> pld_checklist = {};

  // Track buffer.map/remap results keyed by source buffer symbol,
  // so that remap can find the existing mapped handle and unmap
  // can invalidate mappings at scope exit.
  // value: {mapped_result_name, bts, size_expr_str}
  std::map<std::string, std::tuple<std::string, std::string, std::string>>
      pending_mapped_buffers_;

  std::set<std::string> global_buffers;   // global buffers
  bool current_pb_is_cooperative = false; // set before EmitDeviceFuncDecl
  bool emit_call = true;                  // emit the call statement
  bool has_acore_call = false;            // track acore:: library usage
  bool has_lib_gemm_general = false;      // track general gemm fallback usage
  bool has_lib_fallback = false;          // track non-gemm lib fallback usage

  std::unordered_map<std::string, int> emitted_device_names_;
  std::vector<std::unordered_map<std::string, int>> emitted_names_stack_;
  std::string UniqueDeviceName(const std::string& name) {
    auto it = emitted_device_names_.find(name);
    if (it == emitted_device_names_.end()) {
      emitted_device_names_[name] = 0;
      return name;
    }
    return name + "__" + std::to_string(++it->second);
  }
  void PushEmittedNames() {
    emitted_names_stack_.push_back(emitted_device_names_);
  }
  void PopEmittedNames() {
    if (!emitted_names_stack_.empty()) {
      emitted_device_names_ = emitted_names_stack_.back();
      emitted_names_stack_.pop_back();
    }
  }

private:
  void EmitFixedHostHead();
  void EmitFixedDeviceHead();

  bool EnableLineDirective() const { return CCtx().GenDebugInfo(); }
  bool ShouldEmitLineDirective(AST::Node& n) const;
  std::string ResolveLineDirectivePath(const location& loc) const;
  static std::string EscapeLineDirectivePath(const std::string& path);
  void EmitLineDirective(AST::Node& n);
  void ResetLineDirectiveState();

  void EmitHostFuncDecl(std::ostringstream&);
  void EmitDeviceFuncDecl(std::ostringstream&);

  void EmitSource();
  void EmitScript(std::ostream& os, const std::string& exe_fn = "");
  bool CompileWithScript(const std::string&);

  void EmitHostRuntimeCheck();
  // emit mem reuse script for each device function.
  void EmitMemReuse(const std::string& dev_func_name);
  void EmitTopsFree();

  // site-level assertion emission
  std::unordered_map<AST::Node*, std::vector<Assertion>> pre_site_assertions;
  std::unordered_map<AST::Node*, std::vector<Assertion>> post_site_assertions;
  void BuildSiteAssertionMap();
  void EmitPreSiteAssertions(AST::Node& n);
  void EmitPostSiteAssertions(AST::Node& n);

  void EmitLibCall(AST::Call& n, const std::string& func_name,
                   std::ostringstream& os, const std::string& indent);

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
    dte_pool_size = 0;
    anon_dte_slot = -1;
    has_nofuture_rotate = false;
    dte_pool_slots.clear();
    dte_named_vars.clear();
    waited_futures.clear();
    nofuture_vars.clear();
    fty = nullptr;
    void_return = false;
    emit_call = true;
    parallel_idx = -1;
    stream_name = "";
    pre_site_assertions.clear();
    post_site_assertions.clear();
    emitted_device_names_.clear();
    acore_mma.Reset();
    ResetLineDirectiveState();
  }

  std::string GenHostParamName() {
    return "hp" + std::to_string(host_param_count++);
  }

  // return all the parameters of device function in topscc code.
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

  Storage FutureStorage(const std::string& n) const {
    assert(PrefixedWith(n, "::") && "requires a scoped name.");
    if (cgi.GetFunctionSharedFutures(fname).count(n))
      return Storage::SHARED;
    else if (cgi.GetFunctionLocalFutures(fname).count(n))
      return Storage::LOCAL;
    assert("illegal future.");
    return Storage::NONE;
  }
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
                                BaseType to, BaseType from, bool is_host = true,
                                size_t element_count = 1,
                                bool is_explicit = false) const;

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
  const std::string BuildTcleLoad(const std::string& addr,
                                  const std::string& ty,
                                  const std::string& mask = "",
                                  const std::string& other = "") const;
  const std::string BuildTcleStore(const std::string& addr,
                                   const std::string& val,
                                   const std::string& mask = "") const;
  const std::string BuildTcleGather(const std::string& base,
                                    const std::string& offset,
                                    const std::string& ty_str,
                                    const std::string& mask = "",
                                    const std::string& other = "") const;
  const std::string BuildTcleScatter(const std::string& value,
                                     const std::string& base,
                                     const std::string& offset,
                                     const std::string& mask = "") const;

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
  const std::string VectorTypeSTR(const ptr<Type>& vt) const;
  const std::string DMATypeSTR(Storage, bool block_level = true) const;
  void EmitDTEDecl(std::ostringstream& os, const std::string& indent,
                   Storage sto, const std::string& varname,
                   bool with_scope = false, bool block_level = true) const;
};

} // namespace Topscc

} // end namespace Choreo

#endif // __CHOREO_CODEGEN_TOPSCC_HPP__
