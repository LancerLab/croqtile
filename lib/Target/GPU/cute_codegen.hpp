#ifndef __CHOREO_CODEGEN_CUTE_HPP__
#define __CHOREO_CODEGEN_CUTE_HPP__

#include <deque>
#include <filesystem>
#include <iostream>
#include <set>
#include <sstream>
#include <thread>
#include <unordered_set>

#include "ast.hpp"
#include "codegen.hpp"
#include "codegen_utils.hpp"
#include "fragment_layout.hpp"
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
  case BaseType::U64: return "uint64_t";
  case BaseType::U32: return "unsigned int";
  case BaseType::U16: return "unsigned short";
  case BaseType::U8: return "unsigned char";
  case BaseType::S64: return "int64_t";
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

using ScopedSymbolMap = ::Choreo::ScopedSymbolMap;

enum class AutomapStrategy {
  REGISTER_DIRECT,
  REGISTER_WITH_BROADCAST,
  FLAT_STRIDE
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
  bool Visit(AST::NamedTypeDecl&) override;
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
  bool Visit(AST::ApplyBlock&) override;
  bool Visit(AST::FragTransfer&) override;
  bool Visit(AST::FragReduce&) override;
  bool Visit(AST::Wait&) override;
  bool Visit(AST::Trigger&) override;
  bool Visit(AST::Break&) override;
  bool Visit(AST::Continue&) override;
  bool Visit(AST::Yield&) override;
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
  bool extended_mma = false;

  SDimsInfo symbolic_dimensions;

  std::ostringstream ds;            // device stream
  std::ostringstream hs;            // host stream
  std::ostringstream return_stream; // stream for return node

  LineDirectiveState host_line_state;
  LineDirectiveState device_line_state;

  std::map<std::string, std::string> claimed_futs;
  std::set<std::string> tma_futures_;
  std::vector<std::string> pld_checklist = {};

  size_t future_count_ = 0;
  size_t dma_count_ = 0;

  std::set<std::string> global_buffers; // global buffers
  bool emit_call = true;                // emit the call statement

  // Tracks emitted device variable names per device function to detect
  // collisions when nested scopes declare variables with the same short name.
  std::unordered_map<std::string, int> emitted_device_names_;
  std::vector<std::unordered_map<std::string, int>> emitted_names_stack_;
  std::vector<std::unordered_map<std::string, std::string>>
      known_val_str_stack_;
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
    known_val_str_stack_.push_back(known_val_str_to_var_);
  }
  void PopEmittedNames() {
    if (!emitted_names_stack_.empty()) {
      emitted_device_names_ = emitted_names_stack_.back();
      emitted_names_stack_.pop_back();
    }
    if (!known_val_str_stack_.empty()) {
      known_val_str_to_var_ = known_val_str_stack_.back();
      known_val_str_stack_.pop_back();
    }
  }

  std::set<std::string> cooperatives; // futures with cooperative-dma
  std::unordered_set<std::string> async_subbyte_futures;

  // mma related
  size_t reg_num_d;
  // once the flag is set, always use dynamic reuse!
  bool set_cuda_func_attribute_max_dynamic_shared_memory_size = false;
  static const std::string vid_pfx;
  // block dim enforcement level, default to thread level
  ParallelLevel bdim_level = ParallelLevel::THREAD;
  size_t current_thread_count = 0;
  std::string current_thread_count_expr;
  std::string stream_name; // deprecated: kept for ABI, no longer populated
  int tma_count = 0;
  int tma_future_count = 0;
  bool cluster_defers_launch = false;
  AST::ParallelBy* deferred_cluster_pb = nullptr;
  std::string deferred_spm_decls;
  std::deque<std::string> recent_tma_tx_bytes;
  bool has_pending_wgmma_finalize = false;
  std::string pending_wgmma_acc_sym;
  bool has_explicit_mma_wait = false;
  bool wgmma_arrive_state_declared = false;
  std::set<std::string> cluster_trigger_events_;
  std::set<std::string> event_arrive_tx_events_;
  std::set<std::string> tma_bound_event_triggers_;
  std::set<std::string> empty_event_names_;
  std::set<std::string> waited_events_;
  std::vector<std::string> pending_barrier_inits_;
  std::vector<std::string> pending_tma_prefetch_names_;
  // Stack of innermost foreach loop induction variable names (__iv_xxx).
  // Used to derive inline phase formulas for mbarrier waits.
  std::vector<std::string> foreach_iv_stack_;
  bool in_named_var_decl_ = false;
  bool has_analyzed_warpspec = false;
  bool warpspec_wgmma_arrived = false;
  AST::InThreadsBlock* current_inthreads = nullptr;

  bool in_register_direct_automap_ = false;
  bool vec4_automap_skip_ = false;
  std::string reg_loop_var_;
  std::map<std::string, std::string> automap_frag_reg_expr_;
  std::map<std::string, std::string> frag_apply_iv_map_;
  std::set<AST::Node*> apply_row_hoisted_stmts_;
  bool apply_has_main_loop_ = true;
  // Maps scoped IV name -> preferred upper-bound C++ expression.
  // Populated in Visit(WithIn) when the range source is a declared variable.
  std::map<std::string, std::string> iv_upper_bound_expr_;

  // Maps normalized ValueSTR output -> declared C++ variable name.
  // Populated when an `auto var = expr;` declaration is emitted.
  // OpValueSTR checks this to reuse the name instead of re-expanding
  // complex subexpressions (e.g. kv_tiles inside kv_bound's ternary).
  std::unordered_map<std::string, std::string> known_val_str_to_var_;

  AutomapStrategy AnalyzeAutomap(const AST::ForeachBlock& n);

  struct BaseScaleAccumInfo {
    std::string frag_sym;
    std::string frag_expr;
    std::string scale_frag_name;
    std::string scale_a_name;
    std::string scale_a_valid_rows_name;
    std::string scale_b_name;
    std::string scale_a_expr;
    std::string scale_a_valid_rows_expr;
    std::string scale_b_expr;
    std::string scale_a_ld;
    std::string acc_ty;
    std::string scale_frag_ty;
    std::string dim_n;
    int scale_a_static_rows = -1;
    size_t reg_num_d = 0;
  };
  struct HoistedScaleAccumInfo : BaseScaleAccumInfo {};
  struct ExplicitScaleAccumInfo : BaseScaleAccumInfo {
    bool consumed = false;
  };
  std::vector<std::vector<std::string>> hoisted_scale_decl_scopes;
  std::unordered_set<std::string> active_hoisted_scale_decls;
  std::vector<std::optional<HoistedScaleAccumInfo>> hoisted_scale_accum_scopes;
  std::vector<std::vector<ExplicitScaleAccumInfo>> explicit_scale_accum_scopes;
  std::unordered_map<std::string, ptr<AST::ChunkAt>> live_chunk_aliases;
  std::unordered_map<std::string, SwizMode> shared_buf_swiz_;

  struct TMAInnerSplit {
    size_t swiz_elems;
  };
  std::unordered_map<std::string, TMAInnerSplit> tma_inner_splits_;

  struct FragChunkRSInfo {
    std::string parent_c_sym;
    std::string offset_var;
    size_t regs_per_step = 0;
  };
  std::unordered_map<std::string, FragChunkRSInfo> frag_chunk_rs_aliases_;

private:
  void FlushBarrierInits();
  std::string InlinePhaseExpr(int stages, bool is_fill) const;
  void EmitFixedHostHead();
  void EmitFixedDeviceHead();
  void EmitFastCompileCache(std::ostream& os, const std::string& precomp_cu);
  static uint32_t ContentFingerprint();

  bool EnableLineDirective() const { return CCtx().GenDebugInfo(); }
  bool EnableDebugTypeRTTI() const {
    return CCtx().GenDebugInfo() || CCtx().TargetDebugInfo();
  }
  bool ShouldEmitLineDirective(AST::Node& n) const;
  std::string ResolveLineDirectivePath(const location& loc) const;

  struct PrepackedU32Info {
    std::string device_name;
    bool use_packed_u32 = false;
  };

  PrepackedU32Info resolvePrepackedU32Meta(const std::string& ref_sym,
                                           bool forceFlag);
  void emitPrepackedU32Snippet(const std::string& metaVar,
                               const std::string& deviceArray,
                               const std::string& rowStride,
                               const std::string& colStride);
  void emitPrepackedU32TileLoadSnippet(const std::string& metaVar,
                                       const std::string& tileAddr,
                                       const std::string& rowStride);
  void emitFp8PrepackedU32TileLoadSnippet(const std::string& metaVar,
                                          const std::string& tileAddr,
                                          const std::string& rowStride,
                                          const std::string& colStride);
  void emitPrepackedV2TileLoadSnippet(const std::string& metaVar,
                                      const std::string& baseName,
                                      const std::string& tileAddr,
                                      const std::string& rowStride,
                                      const std::string& tileOffset = "");
  void emitPrepackedV2Snippet(const std::string& metaVar,
                              const std::string& baseName,
                              const std::string& deviceArray,
                              const std::string& rowStride,
                              const std::string& colStride);
  void emitFp8PrepackedV2TileLoadSnippet(const std::string& metaVar,
                                         const std::string& baseName,
                                         const std::string& tileAddr,
                                         const std::string& rowStride,
                                         const std::string& colStride);
  static std::string EscapeLineDirectivePath(const std::string& path);
  void EmitLineDirective(AST::Node& n);
  void ResetLineDirectiveState();

  void EmitHostFuncDecl(std::ostringstream&);
  void EmitDeviceFuncDecl(std::ostringstream&, AST::ParallelBy*,
                          const ValueItem&);
  std::optional<int64_t> GetSetRegLimit(AST::ParallelBy* pb) const;
  std::optional<int64_t> GetLaunchBoundsMinBlocks(AST::ParallelBy* pb) const;

  void EmitSource();
  void EmitScript(std::ostream& os, const std::string& exe_fn = "");
  bool CompileWithScript(const std::string&);

  void EmitHostRuntimeCheck();
  void EmitDeviceVirtualIndices(AST::ParallelBy*);
  // emit mem reuse script for each device function.
  void EmitMemReuse(const std::string& dev_func_name);
  void EmitCudaFree();
  void EmitDebugSpannedRTTI(std::ostringstream& os, const std::string& indent,
                            const std::string& sym, const ptr<SpannedType>& sty,
                            const std::string& data_expr,
                            const std::vector<std::string>& shape_exprs,
                            const std::vector<std::string>& stride_exprs) const;

  // site-level assertion emission
  std::unordered_map<AST::Node*, std::vector<Assertion>> pre_site_assertions;
  std::unordered_map<AST::Node*, std::vector<Assertion>> post_site_assertions;
  void BuildSiteAssertionMap();
  void EmitPreSiteAssertions(AST::Node& n);
  void EmitPostSiteAssertions(AST::Node& n);

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
    tma_futures_.clear();
    async_subbyte_futures.clear();
    future_count_ = 0;
    dma_count_ = 0;
    fty = nullptr;
    void_return = false;
    emit_call = true;
    parallel_idx = -1;
    pre_site_assertions.clear();
    post_site_assertions.clear();
    recent_tma_tx_bytes.clear();
    has_pending_wgmma_finalize = false;
    has_explicit_mma_wait = false;
    wgmma_arrive_state_declared = false;
    set_cuda_func_attribute_max_dynamic_shared_memory_size = false;
    hoisted_scale_decl_scopes.clear();
    active_hoisted_scale_decls.clear();
    hoisted_scale_accum_scopes.clear();
    live_chunk_aliases.clear();
    shared_buf_swiz_.clear();
    tma_inner_splits_.clear();
    frag_chunk_rs_aliases_.clear();
    cluster_trigger_events_.clear();
    event_arrive_tx_events_.clear();
    tma_bound_event_triggers_.clear();
    pending_barrier_inits_.clear();
    in_named_var_decl_ = false;
    emitted_device_names_.clear();
    ResetLineDirectiveState();
  }

  static void CollectClusterTriggerEvents(AST::Node* node,
                                          std::set<std::string>& out) {
    if (!node) return;
    if (auto* trigger = dyn_cast<AST::Trigger>(node)) {
      if (trigger->IsClusterScope()) {
        for (auto& f : trigger->GetEvents()) {
          auto expr = cast<AST::Expr>(f);
          if (expr->op == Op::ElemOf) {
            auto bid = AST::GetArrayBaseSymbol(*expr);
            out.insert(bid->name);
          } else if (auto sym = expr->GetSymbol()) {
            out.insert(sym->name);
          }
        }
      }
    }
    if (node->HasBody()) {
      if (auto body = node->GetBody()) {
        for (auto& child : body->values)
          CollectClusterTriggerEvents(child.get(), out);
      }
    } else if (auto* mn = dyn_cast<AST::MultiNodes>(node)) {
      for (auto& child : mn->values)
        CollectClusterTriggerEvents(child.get(), out);
    }
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
                                BaseType to, BaseType from, bool is_host = true,
                                bool is_explicit = false) const;

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
  const ValueItem TileAddr(const ptr<AST::ChunkAt>& ca, bool is_host,
                           ValueItem scale = sbe::nu(1)) const;

  std::optional<std::string> ThreadIdString(const ptr<AST::Identifier>&) const;
  std::pair<std::string, size_t> GenMdsOffset(const ptr<AST::ChunkAt>,
                                              ptr<DMAConfig> = nullptr) const;
  const ValueList GenIndices(const ptr<AST::ChunkAt>&,
                             const ptr<DMAConfig>& = nullptr) const;
  const std::string TileBaseOffset(const ptr<AST::ChunkAt>&) const;
  const ValueItem
  GenOffset(const ptr<AST::ChunkAt>&,
            size_t end_idx = std::numeric_limits<size_t>::max()) const;
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
  const AST::MMAOperation*
  FindFirstScaledWGMMAExec(const ptr<AST::Node>& n) const;
  void EmitWGMMAFinalize(std::ostringstream& os, const std::string& indent,
                         bool force_wait = false);
  std::string LinearizeArrayOffset(const std::string& base_expr,
                                   const std::vector<AST::ptr<AST::Node>>& subs,
                                   const ValueList& array_dims,
                                   const ValueItem& elem_count,
                                   bool is_host) const;
  std::pair<std::string, std::string>
  GetDMABufferExpr(const std::string& sym,
                   const ptr<AST::MultiValues> subscription,
                   const ptr<Type>& sym_ty) const;
  void EmitGroupX4Sync(std::ostringstream& os, const std::string& indent,
                       int thread_count = 0) const;
  std::optional<HoistedScaleAccumInfo> AnalyzeHoistableScaledWGMMAAccum(
      const ptr<AST::Node>& n, const std::vector<std::string>& loop_refs) const;
  bool CollectHoistableScaledWGMMAAccum(
      const ptr<AST::Node>& n, const std::vector<std::string>& loop_refs,
      HoistedScaleAccumInfo& info, bool& saw_scaled_exec) const;
  const HoistedScaleAccumInfo* CurrentHoistedScaleAccum() const;
  std::vector<ExplicitScaleAccumInfo>
  AnalyzeExplicitScaleAccumScope(const ptr<AST::MultiNodes>& body) const;
  std::string GenScaleValidRowsExpr(const ptr<AST::ChunkAt>& ca) const;
  int GetScaleStaticRows(const ptr<AST::ChunkAt>& ca) const;
  bool HasPlainWGMMAExecForFrag(const ptr<AST::Node>& n,
                                const std::string& frag_sym) const;
  ExplicitScaleAccumInfo*
  CurrentExplicitScaleAccumForFrag(const std::string& frag_sym);
  void EmitScaleAccumCall(const std::string& acc_ty, const std::string& dim_n,
                          const std::string& d_expr,
                          const std::string& scale_d_expr,
                          const std::string& sa_name, const std::string& sa_ld,
                          const std::string& valid_rows_name,
                          const std::string& sb_name);
  std::pair<std::string, std::string>
  GenTensorDecl(const std::string& name, const std::string& buf_expr,
                const Storage sto, BaseType bty, const Shape& shp,
                bool is_host = false, const std::string& offset = "",
                const std::string& strides = "",
                const std::vector<size_t>& transp = {},
                bool use_wgmma_layout = false, SwizMode = SwizMode::NONE) const;
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
  std::string EmitUniformFragmentAccess(const ptr<AST::ChunkAt>& ca,
                                        bool is_host) const;
  std::string
  ResolveAutomapThreadExpr(const ptr<AST::AttributeExpr>& automap_attr) const;

  bool IsWarpSpecActive() const {
    return CCtx().UseWarpSpec() || has_analyzed_warpspec ||
           cgi.GetFunctionTrait(fname).has_warpspec_pattern;
  }

  bool InSpecWarp() const {
    return IsWarpSpecActive() && current_inthreads &&
           current_inthreads->HasActiveThreads() &&
           current_inthreads->inthreads_level == ParallelLevel::GROUPx4;
  }

  bool ScopeAlreadySingleThreadForLevel(ParallelLevel level) const {
    if (!current_inthreads || !current_inthreads->HasScopeThreadMask())
      return false;
    auto& mask = current_inthreads->GetScopeThreadMask();
    int64_t unit_size;
    switch (level) {
    case ParallelLevel::GROUPx4: unit_size = 128; break;
    case ParallelLevel::GROUP: unit_size = 32; break;
    default: unit_size = (int64_t)mask.size(); break;
    }
    for (int64_t start = 0; start < (int64_t)mask.size(); start += unit_size) {
      int64_t count = 0;
      int64_t end = std::min(start + unit_size, (int64_t)mask.size());
      for (int64_t i = start; i < end; ++i)
        if (mask[i]) ++count;
      if (count > 1) return false;
    }
    return true;
  }

  int64_t CurrentWarpGroupCount() const {
    return CurrentWarpGroupIndices().size();
  }

  std::vector<int64_t> CurrentWarpGroupIndices() const {
    if (!current_inthreads || !current_inthreads->HasActiveThreads() ||
        !current_inthreads->HasScopeThreadMask())
      return {};

    auto& mask = current_inthreads->GetScopeThreadMask();
    std::vector<int64_t> consumer_wgs;
    for (size_t i = 0; i < mask.size(); i += 128)
      for (size_t j = i; j < std::min(i + 128, mask.size()); ++j)
        if (mask[j]) {
          consumer_wgs.push_back(i / 128);
          break;
        }
    return consumer_wgs;
  }

  int64_t CurrentScopeThreadsCount() const {
    return CurrentScopeThreadIndices().size();
  }

  std::vector<int64_t> CurrentScopeThreadIndices() const {
    if (!current_inthreads || !current_inthreads->HasActiveThreads() ||
        !current_inthreads->HasScopeThreadMask())
      return {};

    auto& mask = current_inthreads->GetScopeThreadMask();
    std::vector<int64_t> thread_indices;
    for (size_t i = 0; i < mask.size(); ++i)
      if (mask[i]) thread_indices.push_back(i);
    return thread_indices;
  }

  // In warpspec mode, use wg_barrier.sync() instead of __syncthreads()
  // because __syncthreads() requires all threads in the block, which would
  // deadlock when only a subset (one warpgroup) is in scope.
  bool NeedWarpSpecGroupX4SyncForCurrentScope() {
    return IsWarpSpecActive() && bdim_level == ParallelLevel::GROUPx4;
  }
};

} // namespace Cute

} // end namespace Choreo

#endif // __CHOREO_CODEGEN_CUTE_HPP__
