#include "cute_codegen.hpp"
#include "codegen_utils.hpp"

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <numeric>
#include <sstream>
#include <thread>

#include "ast.hpp"
#include "choreo_cute_header.inc"
#include "choreo_device_api_header.inc"
#include "choreo_header.inc"
#include "choreo_precompiled_cu.inc"
#include "codegen.hpp"
#include "operator_info.hpp"

#ifndef __CHOREO_CUDA_DIR__
  #warning "missing macro definition of __CHOREO_CUDA_DIR__"
#endif // __CHOREO_CUDA_DIR__

#ifndef __CHOREO_CUTE_DIR__
  #warning "missing macro definition of __CHOREO_CUTE_DIR__"
#endif // __CHOREO_CUTE_DIR__

// #define USING_OP_INFO

using namespace Choreo;
using namespace Choreo::Cute;

// TMA_Swizzle enum and cuda_stringify helper for code generation
enum class TMA_Swizzle {
  NONE = 0, // No swizzle
  B32 = 1,  // 32B swizzle
  B64 = 2,  // 64B swizzle
  B128 = 3  // 128B swizzle
};

inline const char* cuda_stringify(TMA_Swizzle swizzle) {
  switch (swizzle) {
  case TMA_Swizzle::NONE:
    return "CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE";
  case TMA_Swizzle::B32: return "CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_32B";
  case TMA_Swizzle::B64: return "CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_64B";
  case TMA_Swizzle::B128:
    return "CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B";
  default: return "CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE";
  }
}

extern Option<bool> native_f16;
extern Option<bool> native_bf16;
extern Option<bool> verbose;
extern Option<bool> use_pic;
extern Option<bool> tma_cluster_aware;
extern Option<bool> ptx_barrier;
extern Option<bool> use_stmatrix;
extern Option<bool> hoist_offset;
extern Option<bool> hoist_scale;

namespace Choreo {
extern Option<bool> sim_sparse;
extern Option<bool> use_prepack;
extern Option<bool> use_prepack_v2;
} // namespace Choreo
Option<bool> use_cuda_type(OptionKind::Hidden, "-use-cuda-type", "", true,
                           "use cuda built-in types.");

namespace cute {

inline const char* CudaDeviceMemory(Storage st) {
  switch (st) {
  case Storage::SHARED: return "__shared__";
  case Storage::LOCAL: return ""; // local = per-thread stack/register storage
  default: choreo_unreachable("device storage type is not supported.");
  }
  return "";
}

inline std::string CudaParamStorage(Storage st) {
  switch (st) {
  case Storage::SHARED: return "__shared__";
  default: return "";
  }
  return "";
}

inline const std::string GetCopyAtomName(bool is_tma, size_t idx) {
  std::string res = "choreo_copy_atom";
  return res + (is_tma ? "_t_" : "_d_") + std::to_string(idx);
}

const std::string TMAMapDataType(BaseType bt) {
  switch (bt) {
  case BaseType::F16:
    return "CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_FLOAT16";
  case BaseType::BF16:
    return "CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16";
  case BaseType::F32:
    return "CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_FLOAT32";
  case BaseType::F64:
    return "CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_FLOAT64";
  case BaseType::F8_E4M3:
  case BaseType::F8_E5M2:
  case BaseType::F6_E2M3:
  case BaseType::F6_E3M2:
  case BaseType::F4_E2M1:
  case BaseType::U8:
    return "CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_UINT8";
  case BaseType::U16:
    return "CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_UINT16";
  case BaseType::U32:
    return "CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_UINT32";
  case BaseType::U64:
    return "CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_UINT64";
  case BaseType::S8: return "CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_INT8";
  case BaseType::S16:
    return "CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_INT16";
  case BaseType::S32:
    return "CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_INT32";
  case BaseType::S64:
    return "CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_INT64";
  case BaseType::BOOL:
    return "CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BOOL";
  default: choreo_unreachable("unsupported type: " + STR(bt) + ".");
  }
  return "";
}

} // namespace cute

using namespace cute;

const std::string CuteCodeGen::vid_pfx = "__choreo_v";

static inline std::string HostParamTypeStringifyForCute(const Choreo::Type& ty,
                                                        bool is_ref);

bool CuteCodeGen::ShouldEmitLineDirective(AST::Node& n) const {
  return isa<AST::WithBlock>(&n) || isa<AST::ForeachBlock>(&n) ||
         isa<AST::InThreadsBlock>(&n) || isa<AST::IfElseBlock>(&n) ||
         isa<AST::WhileBlock>(&n) || isa<AST::Assignment>(&n) ||
         isa<AST::ParallelBy>(&n) || isa<AST::DMA>(&n) || isa<AST::MMA>(&n) ||
         isa<AST::Wait>(&n) || isa<AST::Trigger>(&n) || isa<AST::Break>(&n) ||
         isa<AST::Continue>(&n) || isa<AST::Rotate>(&n) ||
         isa<AST::Synchronize>(&n) || isa<AST::Call>(&n) ||
         isa<AST::NamedVariableDecl>(&n) || isa<AST::Return>(&n);
}

std::string CuteCodeGen::EscapeLineDirectivePath(const std::string& path) {
  return EscapeLinePathForDirective(path);
}

std::string CuteCodeGen::ResolveLineDirectivePath(const location& loc) const {
  return ResolveDebugLinePath(loc, CCtx().GetDebugLinePathMode());
}

void CuteCodeGen::EmitLineDirective(AST::Node& n) {
  if (!EnableLineDirective() || !ShouldEmitLineDirective(n)) return;

  auto loc = n.LOC();
  if (loc.begin.line <= 0) return;

  auto file = ResolveLineDirectivePath(loc);
  if (file.empty()) return;

  auto& line_state = IsHost() ? host_line_state : device_line_state;
  if (line_state.valid && line_state.line == loc.begin.line &&
      line_state.file == file)
    return;

  auto& os = IsHost() ? hs : ds;
  os << "#line " << loc.begin.line << " \"" << EscapeLineDirectivePath(file)
     << "\"\n";

  line_state.line = loc.begin.line;
  line_state.file = file;
  line_state.valid = true;
}

void CuteCodeGen::ResetLineDirectiveState() {
  host_line_state = {};
  device_line_state = {};
}

const std::optional<std::string> CuteCodeGen::GetTMAName(AST::DMA& n) const {
  if (cur_pb == nullptr) return std::nullopt;
  auto& tma_descs = cgi.GetTMADesc(cur_pb);
  for (auto desc : tma_descs) {
    if (n.from == desc.GetFrom()) {
      assert(n.to == desc.GetTo());
      return desc.GetName();
    }
  }
  return std::nullopt;
}

// Check if WGMMA is used in the current function
bool CuteCodeGen::HasWGMMAInFunction() const {
  // Check if any MMA fragment in the current function is WGMMA
  const auto& frag_mma_types = FCtx(fname).GetFragMMATypes();
  for (const auto& [frag_name, mma_type] : frag_mma_types) {
    if (mma_type == MMAType::WGMMA) { return true; }
  }
  return false;
}

namespace {
size_t SharedAlignmentBytes(const Choreo::CompilationContext& ctx,
                            SwizMode swizzle_mode) {
  return std::max(ctx.GetMemoryAlignmentByte(Storage::SHARED),
                  SwizzleAlignmentByte(swizzle_mode));
}

enum class MmaLoadShapeWarningKind { None, Dynamic, Misaligned };

std::pair<ValueItem, ValueItem> GetMmaLoadExpectedDims(const MMAInfo& ssmi) {
  auto mma_m = ssmi.shape.at(0);
  auto mma_n = ssmi.shape.at(1);
  auto mma_k = ssmi.shape.at(2);

  switch (ssmi.frag) {
  case MMAInfo::FRAG_A:
    switch (ssmi.method) {
    case AST::MMAOperation::ROW_ROW:
    case AST::MMAOperation::ROW_COL: return {mma_m, mma_k};
    case AST::MMAOperation::COL_ROW:
    case AST::MMAOperation::COL_COL: return {mma_k, mma_m};
    }
    break;
  case MMAInfo::FRAG_B:
    switch (ssmi.method) {
    case AST::MMAOperation::ROW_ROW:
    case AST::MMAOperation::COL_ROW: return {mma_n, mma_k};
    case AST::MMAOperation::ROW_COL:
    case AST::MMAOperation::COL_COL: return {mma_k, mma_n};
    }
    break;
  default: break;
  }

  return {mma_m, mma_k};
}

MmaLoadShapeWarningKind
GetMmaLoadShapeWarningKind(const MMAInfo& ssmi,
                           const ptr<SpannedType>& parent_ty) {
  if (!parent_ty || parent_ty->GetShape().Rank() < 2)
    return MmaLoadShapeWarningKind::None;

  auto [expected_dim0, expected_dim1] = GetMmaLoadExpectedDims(ssmi);
  if (!VIIsInt(expected_dim0) || !VIIsInt(expected_dim1))
    return MmaLoadShapeWarningKind::None;

  auto src_dim0 = parent_ty->GetShape().ValueAt(0);
  auto src_dim1 =
      parent_ty->GetShape().ValueAt(parent_ty->GetShape().Rank() - 1);
  bool dim0_ok =
      VIIsInt(src_dim0) && (*VIInt(src_dim0) % *VIInt(expected_dim0) == 0);
  bool dim1_ok =
      VIIsInt(src_dim1) && (*VIInt(src_dim1) % *VIInt(expected_dim1) == 0);

  if (!VIIsInt(src_dim0) || !VIIsInt(src_dim1))
    return MmaLoadShapeWarningKind::Dynamic;
  if (!dim0_ok || !dim1_ok) return MmaLoadShapeWarningKind::Misaligned;
  return MmaLoadShapeWarningKind::None;
}
} // namespace

const AST::MMAOperation*
CuteCodeGen::FindFirstScaledWGMMAExec(const ptr<AST::Node>& n) const {
  if (!n) return nullptr;
  if (!hoist_scale) return nullptr;

  if (auto mma = dyn_cast<AST::MMA>(n)) {
    auto op = mma->GetOperation();
    if (!op || op->Tag() != AST::MMAOperation::Exec || !op->HasScale())
      return nullptr;
    auto c_sym = AST::FragName(op->ExecOperand(0));
    auto scoped_c_sym = InScopeName(c_sym);
    if (FCtx(fname).FragHasMMAType(scoped_c_sym) &&
        FCtx(fname).FragIsWGMMA(scoped_c_sym))
      return op.get();
    return nullptr;
  }

  if (auto mn = dyn_cast<AST::MultiNodes>(n)) {
    for (auto& item : mn->values)
      if (auto* op = FindFirstScaledWGMMAExec(item)) return op;
    return nullptr;
  }

  if (auto if_else = dyn_cast<AST::IfElseBlock>(n)) {
    if (if_else->GetThenBody())
      if (auto* op = FindFirstScaledWGMMAExec(if_else->GetThenBody()))
        return op;
    if (if_else->GetElseBody())
      if (auto* op = FindFirstScaledWGMMAExec(if_else->GetElseBody()))
        return op;
    return nullptr;
  }

  if (auto block = dyn_cast<AST::Block>(n)) {
    return block->GetBody() ? FindFirstScaledWGMMAExec(block->GetBody())
                            : nullptr;
  }

  return nullptr;
}

std::string CuteCodeGen::LinearizeArrayOffset(
    const std::string& base_expr, const std::vector<ptr<AST::Node>>& subs,
    const ValueList& array_dims, const ValueItem& elem_count,
    bool is_host) const {
  assert(subs.size() <= array_dims.size());
  std::string idx;
  for (size_t i = 0; i < subs.size(); ++i) {
    if (idx.empty())
      idx = ExprSTR(subs[i], is_host);
    else
      idx = "(" + idx + ")*" + ValueSTR(array_dims[i]) + "+" +
            ExprSTR(subs[i], is_host);
  }
  for (size_t i = subs.size(); i < array_dims.size(); ++i)
    idx = "(" + idx + ")*" + ValueSTR(array_dims[i]);
  return base_expr + " + (" + idx + ")*(" + ValueSTR(elem_count) + ")";
}

std::pair<std::string, std::string>
CuteCodeGen::GetDMABufferExpr(const std::string& sym,
                              const ptr<AST::MultiValues> subscription,
                              const ptr<Type>& sym_ty) const {
  std::string buf_expr = "";
  std::string sname = InScopeName(sym);
  if (isa<FutureType>(sym_ty) && !IsHostSymbol(sname)) {
    std::string buf_name = sname + ".data";
    buf_expr = ssm.DeviceName(buf_name);
  } else if (isa<FutureType>(sym_ty) && IsHostSymbol(sname) &&
             !const_cast<CuteCodeGen*>(this)->IsChoreoInput(sname) &&
             !const_cast<CuteCodeGen*>(this)->IsChoreoOutput(sname)) {
    buf_expr =
        UnScopedName(const_cast<FutureBufferInfo&>(FBInfo())[sname].buffer);
  } else {
    buf_expr = ssm.DeviceName(sname);
  }

  std::string buf_name = buf_expr;
  if (subscription != nullptr) {
    if (auto array_ty = dyn_cast<ArrayType>(sym_ty);
        array_ty && CCtx().MemReuse()) {
      std::string array_idx = "";
      auto subscriptions = subscription->AllValues();
      const ValueList& array_sizes = array_ty->Dimensions();
      for (size_t i = 0; i < subscriptions.size(); ++i) {
        if (array_idx.empty())
          array_idx = ExprSTR(subscriptions[i], IsHost());
        else
          array_idx = "(" + array_idx + ")*" + ValueSTR(array_sizes[i]) + "+" +
                      ExprSTR(subscriptions[i], IsHost());
      }
      std::string elem_count =
          ValueSTR(cast<SpannedType>(sym_ty)->GetShape().ElementCountValue());
      buf_expr += " + (" + array_idx + ")*(" + elem_count + ")";
    } else {
      for (auto expr : subscription->AllValues())
        buf_expr += "[" + ExprSTR(expr, IsHost()) + "]";
    }
  }
  return std::make_pair(buf_name, buf_expr);
}

void CuteCodeGen::EmitGroupX4Sync(std::ostringstream& os,
                                  const std::string& indent) const {
  os << indent << "wg_barrier.sync();\n";
}

void CuteCodeGen::EmitWGMMAFinalize(std::ostringstream& os,
                                    const std::string& indent) {
  os << indent << "// Finalize WGMMA operations\n";
  os << indent << "warpgroup_commit_batch();\n";
  os << indent << "warpgroup_wait<0>();\n";
  has_pending_wgmma_finalize = false;
  warpspec_wgmma_arrived = false;
}

std::optional<CuteCodeGen::HoistedScaleAccumInfo>
CuteCodeGen::AnalyzeHoistableScaledWGMMAAccum(
    const ptr<AST::Node>& n, const std::vector<std::string>& loop_refs) const {
  HoistedScaleAccumInfo info;
  if (!hoist_scale) return std::nullopt;
  bool saw_scaled_exec = false;
  if (!CollectHoistableScaledWGMMAAccum(n, loop_refs, info, saw_scaled_exec) ||
      !saw_scaled_exec)
    return std::nullopt;
  return info;
}

bool CuteCodeGen::CollectHoistableScaledWGMMAAccum(
    const ptr<AST::Node>& n, const std::vector<std::string>& loop_refs,
    HoistedScaleAccumInfo& info, bool& saw_scaled_exec) const {
  if (!n) return true;
  if (!hoist_scale) return true;

  if (auto mma = dyn_cast<AST::MMA>(n)) {
    auto op = mma->GetOperation();
    if (!op) return true;

    if (op->Tag() == AST::MMAOperation::Store) {
      auto store_frag = AST::FragName(op->StoreFrom());
      if (saw_scaled_exec && store_frag == info.frag_sym) return false;
      return true;
    }

    if (op->Tag() != AST::MMAOperation::Exec || !op->HasScale()) return true;

    auto c_sym = AST::FragName(op->ExecOperand(0));
    auto scoped_c_sym = InScopeName(c_sym);
    if (!FCtx(fname).FragHasMMAType(scoped_c_sym) ||
        !FCtx(fname).FragIsWGMMA(scoped_c_sym))
      return true;

    auto scale_a_expr = ExprSTR(op->ScaleA(), false);
    auto scale_b_expr = ExprSTR(op->ScaleB(), false);
    for (const auto& loop_ref : loop_refs) {
      if ((!loop_ref.empty() &&
           scale_a_expr.find(loop_ref) != std::string::npos) ||
          (!loop_ref.empty() &&
           scale_b_expr.find(loop_ref) != std::string::npos))
        return false;
    }

    auto& ssmi_c = cgi.GetSymbolMMA(scoped_c_sym);
    auto acc_ty = NameBaseType(ssmi_c.ty);
    auto scale_a_strides = GenStrides(op->ScaleA());
    auto scale_a_sty = GetSpannedType(NodeType(*op->ScaleA()));
    auto scale_a_shape = scale_a_sty->GetShape();
    bool scale_a_transposed = VIIsInt(scale_a_shape.ValueAt(0)) &&
                              *VIInt(scale_a_shape.ValueAt(0)) == 1;
    std::string scale_a_ld = scale_a_transposed
                                 ? ValueSTR(scale_a_strides.back())
                                 : ValueSTR(scale_a_strides.front());
    auto scale_a_name = c_sym + "_scale_a_ptr";
    auto scale_a_valid_rows_name = c_sym + "_scale_a_valid_rows";
    auto scale_b_name = c_sym + "_scale_b_val";
    auto scale_frag_name = c_sym + "_scale_frag";
    auto frag_expr = ExprSTR(op->ExecOperand(0), false);
    auto scale_a_valid_rows_expr = GenScaleValidRowsExpr(op->ScaleA());
    int scale_a_static_rows = GetScaleStaticRows(op->ScaleA());
    std::string dim_n = STR(ssmi_c.shape.at(1));

    auto acc_dtype = ssmi_c.ty;
    ValueItem frag_len = ssmi_c.shape[1] / sbe::nu(2);
    if (ssmi_c.ty == BaseType::F16) {
      acc_dtype = BaseType::U32;
      frag_len = frag_len / sbe::nu(2);
    }
    auto reg_num = VIInt(frag_len);
    if (!reg_num)
      choreo_unreachable("expect scaled WGMMA frag length to be numeric");

    if (!saw_scaled_exec) {
      info.frag_sym = c_sym;
      info.frag_expr = frag_expr;
      info.scale_frag_name = scale_frag_name;
      info.scale_a_name = scale_a_name;
      info.scale_a_valid_rows_name = scale_a_valid_rows_name;
      info.scale_b_name = scale_b_name;
      info.scale_a_expr = scale_a_expr;
      info.scale_a_valid_rows_expr = scale_a_valid_rows_expr;
      info.scale_b_expr = scale_b_expr;
      info.scale_a_ld = scale_a_ld;
      info.acc_ty = acc_ty;
      info.scale_frag_ty = NameBaseType(acc_dtype);
      info.dim_n = dim_n;
      info.scale_a_static_rows = scale_a_static_rows;
      info.reg_num_d = (size_t)*reg_num;
      saw_scaled_exec = true;
      return true;
    }

    return info.frag_sym == c_sym && info.frag_expr == frag_expr &&
           info.scale_a_expr == scale_a_expr &&
           info.scale_a_valid_rows_expr == scale_a_valid_rows_expr &&
           info.scale_a_static_rows == scale_a_static_rows &&
           info.scale_b_expr == scale_b_expr && info.scale_a_ld == scale_a_ld &&
           info.acc_ty == acc_ty && info.dim_n == dim_n &&
           info.reg_num_d == (size_t)*reg_num;
  }

  if (auto fb = dyn_cast<AST::ForeachBlock>(n)) {
    return !fb->GetBody() ||
           CollectHoistableScaledWGMMAAccum(fb->GetBody(), loop_refs, info,
                                            saw_scaled_exec);
  }

  if (auto mn = dyn_cast<AST::MultiNodes>(n)) {
    for (auto& item : mn->values)
      if (!CollectHoistableScaledWGMMAAccum(item, loop_refs, info,
                                            saw_scaled_exec))
        return false;
    return true;
  }

  if (auto if_else = dyn_cast<AST::IfElseBlock>(n)) {
    if (if_else->GetThenBody() &&
        !CollectHoistableScaledWGMMAAccum(if_else->GetThenBody(), loop_refs,
                                          info, saw_scaled_exec))
      return false;
    if (if_else->GetElseBody() &&
        !CollectHoistableScaledWGMMAAccum(if_else->GetElseBody(), loop_refs,
                                          info, saw_scaled_exec))
      return false;
    return true;
  }

  if (auto block = dyn_cast<AST::Block>(n)) {
    return !block->GetBody() ||
           CollectHoistableScaledWGMMAAccum(block->GetBody(), loop_refs, info,
                                            saw_scaled_exec);
  }

  if (auto with_block = dyn_cast<AST::WithBlock>(n)) {
    return !with_block->GetBody() ||
           CollectHoistableScaledWGMMAAccum(with_block->GetBody(), loop_refs,
                                            info, saw_scaled_exec);
  }

  return true;
}

const CuteCodeGen::HoistedScaleAccumInfo*
CuteCodeGen::CurrentHoistedScaleAccum() const {
  if (!hoist_scale) return nullptr;
  for (auto it = hoisted_scale_accum_scopes.rbegin();
       it != hoisted_scale_accum_scopes.rend(); ++it) {
    if (it->has_value()) return &it->value();
  }
  return nullptr;
}

std::string
CuteCodeGen::GenScaleValidRowsExpr(const ptr<AST::ChunkAt>& ca) const {
  if (!ca) return "0x3fffffff";

  auto scale_tile_ty = GetSpannedType(NodeType(*ca));
  auto scale_tile_shape = scale_tile_ty->GetShape();
  bool transposed = scale_tile_shape.Rank() > 0 &&
                    VIIsInt(scale_tile_shape.ValueAt(0)) &&
                    *VIInt(scale_tile_shape.ValueAt(0)) == 1;
  size_t row_dim = transposed ? scale_tile_shape.Rank() - 1 : 0;

  if (ca->NoOperation()) {
    if (scale_tile_shape.Rank() == 0 || row_dim >= scale_tile_shape.Rank())
      return "0x3fffffff";
    return "static_cast<int>(" + ValueSTR(scale_tile_shape.ValueAt(row_dim)) +
           ")";
  }

  ptr<AST::SpannedOperation> row_op = nullptr;
  for (auto it = ca->AllOperations().rbegin(); it != ca->AllOperations().rend();
       ++it) {
    if (!isa<AST::SOP::Reshape>(*it)) {
      row_op = *it;
      break;
    }
  }
  if (!row_op) return "0x3fffffff";

  auto src_ty = GetSpannedType(GetSymbolType(ca->RefSymbol()));
  auto src_shape = src_ty->GetShape();
  if (src_shape.Rank() == 0 || row_op->GetBlockShape().Rank() == 0)
    return "0x3fffffff";

  row_dim = transposed ? row_op->GetBlockShape().Rank() - 1 : 0;
  if (row_dim >= src_shape.Rank() || row_dim >= row_op->GetBlockShape().Rank())
    return "0x3fffffff";

  auto row_extent = ValueSTR(src_shape.ValueAt(row_dim));
  auto row_block = ValueSTR(row_op->GetBlockShape().ValueAt(row_dim));
  std::string row_base;
  if (auto indices = row_op->GetIndices();
      indices && row_dim < indices->Count()) {
    auto row_index = ExprSTR(indices->ValueAt(row_dim), false);
    row_base = "((" + row_index + ") * (" + row_block + "))";
  } else if (auto offsets = row_op->GetOffsets();
             offsets && row_dim < offsets->Count()) {
    // View(...).From(...) uses absolute offsets rather than tile indices.
    row_base = "(" + ExprSTR(offsets->ValueAt(row_dim), false) + ")";
  } else {
    return "0x3fffffff";
  }

  return "((" + row_base + " < (" + row_extent + ")) ? static_cast<int>((((" +
         row_extent + ") - (" + row_base + ")) < (" + row_block + ")) ? ((" +
         row_extent + ") - (" + row_base + ")) : (" + row_block + ")) : 0)";
}

int CuteCodeGen::GetScaleStaticRows(const ptr<AST::ChunkAt>& ca) const {
  if (!ca) return -1;

  auto scale_tile_ty = GetSpannedType(NodeType(*ca));
  auto scale_tile_shape = scale_tile_ty->GetShape();
  bool transposed = scale_tile_shape.Rank() > 0 &&
                    VIIsInt(scale_tile_shape.ValueAt(0)) &&
                    *VIInt(scale_tile_shape.ValueAt(0)) == 1;
  size_t row_dim = transposed ? scale_tile_shape.Rank() - 1 : 0;
  if (ca->NoOperation()) {
    if (scale_tile_shape.Rank() == 0 || row_dim >= scale_tile_shape.Rank())
      return -1;
    auto row_extent = VIInt(scale_tile_shape.ValueAt(row_dim));
    return row_extent ? static_cast<int>(*row_extent) : -1;
  }

  ptr<AST::SpannedOperation> row_op = nullptr;
  for (auto it = ca->AllOperations().rbegin(); it != ca->AllOperations().rend();
       ++it) {
    if (!isa<AST::SOP::Reshape>(*it)) {
      row_op = *it;
      break;
    }
  }
  if (!row_op) return -1;

  row_dim = transposed ? row_op->GetBlockShape().Rank() - 1 : 0;
  if (row_dim >= row_op->GetBlockShape().Rank()) return -1;

  auto row_block = VIInt(row_op->GetBlockShape().ValueAt(row_dim));
  return row_block ? static_cast<int>(*row_block) : -1;
}

std::vector<CuteCodeGen::ExplicitScaleAccumInfo>
CuteCodeGen::AnalyzeExplicitScaleAccumScope(
    const ptr<AST::MultiNodes>& body) const {
  std::vector<ExplicitScaleAccumInfo> infos;
  if (!body) return infos;

  auto extract_chunk_alias = [](const ptr<AST::Node>& n) -> ptr<AST::ChunkAt> {
    if (!n) return nullptr;
    if (auto ca = dyn_cast<AST::ChunkAt>(n)) return ca;
    if (auto expr = dyn_cast<AST::Expr>(n)) {
      if (auto ref = expr->GetReference()) return dyn_cast<AST::ChunkAt>(ref);
    }
    return nullptr;
  };

  std::unordered_set<std::string> seen_frags;
  std::unordered_map<std::string, ptr<AST::ChunkAt>> chunk_aliases;
  for (size_t idx = 0; idx < body->values.size(); ++idx) {
    if (auto decl = dyn_cast<AST::NamedVariableDecl>(body->values[idx])) {
      auto rhs_ca = extract_chunk_alias(decl->init_expr);
      auto name = decl->GetName();
      auto scoped_name = InScopeName(name);
      if (rhs_ca) {
        chunk_aliases[name] = rhs_ca;
        chunk_aliases[scoped_name] = rhs_ca;
      } else {
        chunk_aliases.erase(name);
        chunk_aliases.erase(scoped_name);
      }
      continue;
    }

    if (auto assign = dyn_cast<AST::Assignment>(body->values[idx])) {
      if (!assign->AssignToDataElement()) {
        auto rhs_ca = extract_chunk_alias(assign->value);
        auto name = assign->GetName();
        auto scoped_name = InScopeName(name);
        if (rhs_ca) {
          chunk_aliases[name] = rhs_ca;
          chunk_aliases[scoped_name] = rhs_ca;
        } else {
          chunk_aliases.erase(name);
          chunk_aliases.erase(scoped_name);
        }
      }
      continue;
    }

    auto mma = dyn_cast<AST::MMA>(body->values[idx]);
    if (!mma) continue;

    auto op = mma->GetOperation();
    if (!op || op->Tag() != AST::MMAOperation::Scale) continue;

    auto c_sym = AST::FragName(op->ScaleAccumulator());
    auto scoped_c_sym = InScopeName(c_sym);
    if (!FCtx(fname).FragHasMMAType(scoped_c_sym) ||
        !FCtx(fname).FragIsWGMMA(scoped_c_sym) || seen_frags.count(c_sym))
      continue;

    bool saw_exec = false;
    for (size_t prior = 0; prior < idx; ++prior) {
      if (HasPlainWGMMAExecForFrag(body->values[prior], c_sym)) {
        saw_exec = true;
        break;
      }
    }
    if (!saw_exec) continue;

    auto& ssmi_c = cgi.GetSymbolMMA(scoped_c_sym);
    auto acc_dtype = ssmi_c.ty;
    ValueItem frag_len = ssmi_c.shape[1] / sbe::nu(2);
    if (ssmi_c.ty == BaseType::F16) {
      acc_dtype = BaseType::U32;
      frag_len = frag_len / sbe::nu(2);
    }
    auto reg_num = VIInt(frag_len);
    if (!reg_num)
      choreo_unreachable("expect explicit mma.scale frag length to be numeric");

    ExplicitScaleAccumInfo info;
    info.frag_sym = c_sym;
    info.frag_expr = ExprSTR(op->ScaleAccumulator(), false);
    info.scale_frag_name = c_sym + "_scale_frag";
    info.scale_a_name = c_sym + "_scale_a_ptr";
    info.scale_a_valid_rows_name = c_sym + "_scale_a_valid_rows";
    info.scale_b_name = c_sym + "_scale_b_val";
    auto resolved_scale_a = op->ScaleA();
    if (resolved_scale_a && resolved_scale_a->NoOperation()) {
      auto ref_sym = resolved_scale_a->RefSymbol();
      auto it = chunk_aliases.find(ref_sym);
      if (it == chunk_aliases.end())
        it = chunk_aliases.find(UnScopedName(ref_sym));
      if (it == chunk_aliases.end()) {
        auto expr_sym = ExprSTR(op->ScaleA(), false);
        it = chunk_aliases.find(expr_sym);
      }
      if (it != chunk_aliases.end()) resolved_scale_a = it->second;
    }

    info.scale_a_expr = ExprSTR(op->ScaleA(), false);
    info.scale_a_valid_rows_expr = GenScaleValidRowsExpr(resolved_scale_a);
    info.scale_b_expr = ExprSTR(op->ScaleB(), false);
    {
      auto sa_strides = GenStrides(resolved_scale_a);
      auto sa_sty = GetSpannedType(NodeType(*resolved_scale_a));
      auto sa_shape = sa_sty->GetShape();
      bool sa_transposed =
          VIIsInt(sa_shape.ValueAt(0)) && *VIInt(sa_shape.ValueAt(0)) == 1;
      info.scale_a_ld = sa_transposed ? ValueSTR(sa_strides.back())
                                      : ValueSTR(sa_strides.front());
    }
    info.acc_ty = NameBaseType(ssmi_c.ty);
    info.scale_frag_ty = NameBaseType(acc_dtype);
    info.dim_n = STR(ssmi_c.shape.at(1));
    info.scale_a_static_rows = GetScaleStaticRows(resolved_scale_a);
    info.reg_num_d = *reg_num;
    infos.push_back(info);
    seen_frags.insert(c_sym);
  }

  return infos;
}

bool CuteCodeGen::HasPlainWGMMAExecForFrag(const ptr<AST::Node>& n,
                                           const std::string& frag_sym) const {
  if (!n) return false;

  if (auto mma = dyn_cast<AST::MMA>(n)) {
    auto op = mma->GetOperation();
    if (!op || op->Tag() != AST::MMAOperation::Exec || op->HasScale())
      return false;
    auto c_sym = AST::FragName(op->ExecOperand(0));
    if (c_sym != frag_sym) return false;
    auto scoped_c_sym = InScopeName(c_sym);
    return FCtx(fname).FragHasMMAType(scoped_c_sym) &&
           FCtx(fname).FragIsWGMMA(scoped_c_sym);
  }

  if (auto mn = dyn_cast<AST::MultiNodes>(n)) {
    for (auto& item : mn->values)
      if (HasPlainWGMMAExecForFrag(item, frag_sym)) return true;
    return false;
  }

  if (auto fb = dyn_cast<AST::ForeachBlock>(n))
    return fb->GetBody() && HasPlainWGMMAExecForFrag(fb->GetBody(), frag_sym);

  if (auto block = dyn_cast<AST::Block>(n))
    return block->GetBody() &&
           HasPlainWGMMAExecForFrag(block->GetBody(), frag_sym);

  if (auto if_else = dyn_cast<AST::IfElseBlock>(n))
    return (if_else->GetThenBody() &&
            HasPlainWGMMAExecForFrag(if_else->GetThenBody(), frag_sym)) ||
           (if_else->GetElseBody() &&
            HasPlainWGMMAExecForFrag(if_else->GetElseBody(), frag_sym));

  return false;
}

CuteCodeGen::ExplicitScaleAccumInfo*
CuteCodeGen::CurrentExplicitScaleAccumForFrag(const std::string& frag_sym) {
  for (auto it = explicit_scale_accum_scopes.rbegin();
       it != explicit_scale_accum_scopes.rend(); ++it) {
    for (auto& info : *it) {
      if (!info.consumed && info.frag_sym == frag_sym) return &info;
    }
  }
  return nullptr;
}

void CuteCodeGen::EmitScaleAccumCall(
    const std::string& acc_ty, const std::string& dim_n,
    const std::string& d_expr, const std::string& scale_d_expr,
    const std::string& sa_name, const std::string& sa_ld,
    const std::string& valid_rows_name, const std::string& sb_name) {
  ds << d_indent << "scale_accumulator<" << acc_ty << ", float, " << dim_n
     << ">(reinterpret_cast<" << acc_ty << "*>(" << d_expr
     << "), reinterpret_cast<" << acc_ty << "*>(" << scale_d_expr << "), "
     << sa_name << ", " << sa_ld << ", " << valid_rows_name << ", " << sb_name
     << ");\n";
}

// return mds name and the declaration string.
// If offset is not empty, means that need to do memory viewing.
//   Just add offset to buf_expr, then utilize new_shape.
std::pair<std::string, std::string> CuteCodeGen::GenTensorDecl(
    const std::string& bname, const std::string& buf_expr, const Storage sto,
    BaseType bty, const Shape& shp, bool is_host, const std::string& offset,
    const std::string& strides, const std::vector<size_t>& transp,
    bool use_wgmma_layout, SwizMode swizzle_mode) const {
  static int shp_cnt = 0;
  shp_cnt++;
  auto shpcnt = std::to_string(shp_cnt);

  auto shp_name = "__shape" + shpcnt + "_" + bname;
  auto lyt_name = "__layout" + shpcnt + "_" + bname;
  auto std_name = "__stride" + shpcnt + "_" + bname;
  auto tsr_name = "__tensor" + shpcnt + "_" + bname;

  std::string mem_ty;
  if (sto == Storage::GLOBAL || sto == Storage::DEFAULT)
    mem_ty = "gmem";
  else if (sto == Storage::SHARED)
    mem_ty = "smem";
  else if (sto == Storage::LOCAL)
    mem_ty = "";
  else
    choreo_unreachable("unsupported storage type: " + STR(sto));

  std::string bts{NameBaseType(bty)};

  std::ostringstream tsr_decl;

  auto indent = (is_host) ? h_indent : d_indent;

  tsr_decl << indent << "auto " << shp_name << " = cute::make_shape("
           << ((transp.empty()) ? ShapeSTR(shp, true)
                                : ReShapeSTR(shp, transp, true))
           << ");\n";

  // For WGMMA with shared memory destination, use swizzled layout
  if (use_wgmma_layout && sto == Storage::SHARED) {
    // Select swizzle layout based on swizzle value
    std::string swizzle_layout;
    switch (swizzle_mode) {
    case SwizMode::B32:
      swizzle_layout = "cute::SM90::GMMA::Layout_K_SW32_Atom";
      break;
    case SwizMode::B64:
      swizzle_layout = "cute::SM90::GMMA::Layout_K_SW64_Atom";
      break;
    case SwizMode::B128:
      swizzle_layout = "cute::SM90::GMMA::Layout_K_SW128_Atom";
      break;
    case SwizMode::NONE:
      swizzle_layout = "cute::SM90::GMMA::Layout_K_INTER_Atom";
      break;
    default: swizzle_layout = "cute::SM90::GMMA::Layout_K_INTER_Atom"; break;
    }
    tsr_decl << indent << "auto " << lyt_name
             << " = "
                "cute::tile_to_shape("
             << swizzle_layout << "<" << STR(bty) << ">{}, " << shp_name
             << ");\n";
  } else {
    if (!strides.empty())
      tsr_decl << indent << "auto " << std_name << " = cute::make_stride("
               << strides << ");\n";
    tsr_decl << indent << "auto " << lyt_name << " = cute::make_layout("
             << shp_name;
    if (!strides.empty()) tsr_decl << ", " << std_name;
    tsr_decl << ");\n";
  }

  tsr_decl << indent << "auto " << tsr_name << " = cute::make_tensor(";
  if (!mem_ty.empty())
    tsr_decl << "cute::make_" << mem_ty << "_ptr<" << bts << ">";
  tsr_decl << "((" << bts << "*)" << buf_expr
           << ((!offset.empty()) ? (" + " + offset) : "") << ")";
  tsr_decl << ", " << lyt_name << ");\n";

  return {tsr_name, tsr_decl.str()};
}

bool CuteCodeGen::ThreadCooperative(AST::DMA&) const {
  return !CCtx().HasFeature(ChoreoFeature::TMA);
}

const std::string CuteCodeGen::ShapeSTR(const Shape& s, bool shp_lit,
                                        const std::string& delimiter,
                                        BaseType cast_to) const {
  auto& vl = s.Value();
  assert(!vl.empty());

  std::ostringstream oss;
  for (unsigned i = 0; i < vl.size(); ++i) {
    if (i > 0) oss << delimiter;
    bool need_static_cast = (cast_to != BaseType::UNKNOWN && !VIIsInt(vl[i]));
    if (need_static_cast)
      oss << "static_cast<" << NameBaseType(cast_to) << ">(";
    oss << ValueSTR(vl[i], false, shp_lit);
    if (need_static_cast) oss << ")";
  }
  return oss.str();
}

const std::string CuteCodeGen::ReShapeSTR(const Shape& s,
                                          const std::vector<size_t>& order,
                                          bool shp_lit,
                                          const std::string& delimiter) const {
  auto& vl = s.Value();
  assert(!vl.empty());
  assert(order.size() == vl.size());
  std::ostringstream oss;
  for (unsigned i = 0; i < vl.size(); ++i) {
    if (i > 0) oss << delimiter;
    oss << ValueSTR(vl[order[i]], ", ", shp_lit);
  }
  return oss.str();
}

bool CuteCodeGen::BeforeVisitImpl(AST::Node& n) {
  if (trace_visit) dbgs() << "Before visiting " << n.TypeNameString() << "\n";

  EmitPreSiteAssertions(n);

  EmitLineDirective(n);

  if (isa<AST::Program>(&n)) {
    VST_DEBUG(dbgs() << STR(FBInfo()) << "\n");
    // emit the fixed headers
    EmitFixedHostHead();
    EmitFixedDeviceHead();
    ssm.EnterScope();
    ssm.MapDeviceSymbolIfNotExist("::__choreo_no_tiling__", "0");
    levels.push(ParallelLevel::NONE);
  } else if (isa<AST::ChoreoFunction>(&n)) {
    ResetChoreoFunctionStates();
    CollectClusterTriggerEvents(&n, cluster_trigger_events_);
    BuildSiteAssertionMap();
    device_fn = "__choreo_device_" + fname;
    fty = cast<FunctionType>(GetSymbolType(fname));
    ssm.EnterScope();
    levels.push(ParallelLevel::SEQ);
  } else if (auto pb = dyn_cast<AST::ParallelBy>(&n)) {
    levels.push(pb->GetLevel());
    // only on device-side
    bool is_deferred_block = !pb->IsOuter() &&
                             pb->GetLevel() == ParallelLevel::BLOCK &&
                             cluster_defers_launch;
    if (pb->IsOuter() || is_deferred_block) {
      if (!is_deferred_block) parallel_idx += 1;
      cur_pb = pb;
      if (cgi.GetFunctionTrait(fname).multiple_parallelby)
        device_fn = "__choreo_device_" + fname + std::to_string(parallel_idx);
      VST_DEBUG(pb->InlinePrint(dbgs());
                dbgs() << " (max-level: " << STR(TargetMaxLevel()) << ")\n");
    }
    if (pb->GetLevel() == ParallelLevel::GROUPx4 ||
        pb->GetLevel() == ParallelLevel::GROUP) {
      // check if the parallelby level is enforce
      if (pb->IsEnforced()) bdim_level = pb->GetLevel();
    }
  } else if (isa<AST::WithBlock>(&n)) {
    IndStream() << "// with-in: " << n.LOC() << "\n";
    IndStream() << "{\n";
    IncrIndent();
  } else if (isa<AST::ForeachBlock>(&n)) {
    IndStream() << "// foreach: " << n.LOC() << "\n";
  }
  if (isa<AST::IfElseBlock>(&n) || isa<AST::NamedVariableDecl>(&n)) {
    emit_call = false;
  }

  return true;
}

bool CuteCodeGen::InMidVisitImpl(AST::Node& n) {
  if (auto ie = dyn_cast<AST::IfElseBlock>(&n)) {
    if (!ie->HasElse()) return true;
    DecrIndent();
    IndStream() << "} else {\n";
    IncrIndent();
  }
  return true;
}

bool CuteCodeGen::AfterVisitImpl(AST::Node& n) {
  if (trace_visit) dbgs() << "After visiting " << n.TypeNameString() << "\n";

  EmitPostSiteAssertions(n);

  if (isa<AST::Program>(&n)) {
    ssm.LeaveScope();

    switch (CCtx().GetOutputKind()) {
    case OutputKind::TargetSourceCode: EmitSource(); break;
    case OutputKind::TargetModule: {
      if (!CompileWithScript("--compile-module")) {
        error_count++;
        return false;
      }
      break;
    }
    case OutputKind::TargetExecutable: {
      if (!CompileWithScript("--compile-link")) {
        error_count++;
        return false;
      }
      break;
    }
    case OutputKind::ShellScript: {
      EmitScript(outs());
      break;
    }
    default:
      choreo_unreachable("outputkind: " + STR(CCtx().GetOutputKind()) +
                         " is not supported.");
    }
  } else if (isa<AST::ChoreoFunction>(&n)) {
    PLDCheck();
    ssm.LeaveScope();
    code_segments.back() += ds.str() + hs.str();
    ds.str(""); // reset the streams
    hs.str("");
    return_stream.str("");
    stream_name = "";
    tma_count = 0;
    tma_future_count = 0;
  } else if (auto pb = dyn_cast<AST::ParallelBy>(&n)) {
    levels.pop();
    // only on device-side
    bool is_deferred_block = !pb->IsOuter() &&
                             pb->GetLevel() == ParallelLevel::BLOCK &&
                             cluster_defers_launch;
    if (pb->IsOuter() && pb->GetLevel() == ParallelLevel::CLUSTER) {
      cluster_defers_launch = false;
      deferred_cluster_pb = nullptr;
    } else if (pb->IsOuter() || is_deferred_block) {
      if (is_deferred_block) cluster_defers_launch = false;
      cur_pb = nullptr;
      ds << d_indent << "} // end parallel-by\n";
      DecrDeviceIndent();
      ds << "}\n\n";
    } else {
      auto& siblings = cgi.GetPBTree(fname).GetSiblings(pb);
      if (!siblings.empty()) {
        DecrDeviceIndent();
        ds << d_indent << "} // end inner parallel-by\n";
      }
    }
    // reset the block dim enforcement level
    if (pb->GetLevel() == ParallelLevel::THREAD) {
      bdim_level = ParallelLevel::THREAD;
    }
  } else if (isa<AST::WithBlock>(&n)) {
    if (!explicit_scale_accum_scopes.empty())
      explicit_scale_accum_scopes.pop_back();
    DecrIndent();
    IndStream() << "}\n";
  } else if (auto fb = dyn_cast<AST::ForeachBlock>(&n)) {
    std::optional<HoistedScaleAccumInfo> hoisted_scale_accum_info =
        (!hoisted_scale_accum_scopes.empty() &&
         hoisted_scale_accum_scopes.back().has_value())
            ? hoisted_scale_accum_scopes.back()
            : std::nullopt;
    const auto& ranges = fb->GetRangeNodes();
    for (int j = ranges->Count() - 1; j >= 0; --j) {
      auto rng = cast<AST::LoopRange>(ranges->ValueAt(j));
      auto cname = rng->IVName();
      auto ivs = within_map.at(InScopeName(cname));
      for (auto iv_itr = ivs.rbegin(); iv_itr != ivs.rend(); ++iv_itr) {
        DecrIndent();
        IndStream() << "} // " << UnScopedName(*iv_itr) << "\n";
        IndStream() << ssm.DeviceName(*iv_itr) << " = 0;\n"; // must reset
      }
    }
    if (hoisted_scale_accum_info.has_value()) {
      const auto& info = hoisted_scale_accum_info.value();
      EmitScaleAccumCall(info.acc_ty, info.dim_n, info.frag_expr,
                         info.scale_frag_name, info.scale_a_name,
                         info.scale_a_ld, info.scale_a_valid_rows_name,
                         info.scale_b_name);
    }
    if (!hoisted_scale_accum_scopes.empty())
      hoisted_scale_accum_scopes.pop_back();
    if (!hoisted_scale_decl_scopes.empty()) {
      for (const auto& name : hoisted_scale_decl_scopes.back())
        active_hoisted_scale_decls.erase(name);
      hoisted_scale_decl_scopes.pop_back();
    }
    if (!explicit_scale_accum_scopes.empty())
      explicit_scale_accum_scopes.pop_back();
  } else if (auto it = dyn_cast<AST::InThreadsBlock>(&n)) {
    DecrDeviceIndent();
    if (!it->stmts->None()) {
      ds << d_indent << "}";
      if (!it->async && it->outer) ds << "\n" << d_indent << "__syncthreads();";
      ds << " // end inthreads\n";
    }
    current_inthreads = nullptr;
  } else if (auto ie = dyn_cast<AST::IfElseBlock>(&n)) {
    DecrIndent();
    IndStream() << "} // end if-else: " << ie->LOC() << "\n";
  } else if (auto ie = dyn_cast<AST::WhileBlock>(&n)) {
    DecrIndent();
    IndStream() << "} // end while: " << ie->LOC() << "\n";
  } else if (isa<AST::NamedVariableDecl>(&n)) {
    emit_call = true;
  }
  return true;
}

// Note: the function is only used for coordinate in TMA
const ValueList CuteCodeGen::GenIndices(const ptr<AST::ChunkAt>& ca,
                                        const ptr<DMAConfig>& config) const {
  ValueList indices;

  auto& sops = ca->AllOperations();

  if (sops.empty()) {
    auto sz = GetSpannedType(ca->GetType())->GetShape().Rank();
    for (size_t i = 0; i < sz; ++i) { indices.push_back(sbe::nu(0)); }
    return indices;
  }

  size_t sop_base = 0;
  if (auto li = ca->IndexOfLastSpanAs()) sop_base = *li + 1;

  if (sop_base == sops.size()) {
    // span_as is the tail spannedoperation
    auto sz = ca->GetBlockShape().DimCount();
    for (size_t i = 0; i < sz; ++i) indices.push_back(sbe::nu(0));
    return indices;
  }

  // Generate the initial indices
  indices = ValxN(sbe::nu(0), ca->GetBlockShape().DimCount());

  // handle each chunkat inside a seqeunce like 'chunkat(a, b).chunkat(c)...'
  for (size_t sop_idx = sop_base; sop_idx < sops.size(); ++sop_idx) {
    // span_as reshape operation would not affect index generation
    assert(!isa<AST::SOP::Reshape>(sops[sop_idx]));

    // For each chunkat expression, The tiled-block's shape is cooked by shape
    // inference. The block shape is different with the result shape of chunkat
    // expression when using 'modspan', where the result shape represents the
    // shape that applied mod (%) operation. Anyway, for offset, we only care
    // about the tiled-block's shape
    auto& shape = sops[sop_idx]->GetBlockShape();
    auto op = sops[sop_idx];
    ValueList coords;
    // For each 'a, b, c, ...' inside 'chunkat(a, b, c, ...)', that 'b' inside
    // 'chunkat(a, b, c, ...)' could be bounded var like b = {b0, b1} Therefore,
    // we collect all the expressions first.
    if (!op->IndexNodes().empty()) {
      for (auto p : op->IndexNodes()) {
        if (const auto& o = dyn_cast<AST::Expr>(p)->Opts(); o.HasVals()) {
          const auto& vals = o.GetVals();
          for (auto& val : vals) {
            if (sbe::ceq(val, sbe::sym("::__choreo_no_tiling__")))
              coords.push_back(sbe::nu(0));
            else
              coords.push_back(val);
          }
        } else
          coords.push_back(sbe::sym(OpExprSTR(p, "*", true, IsHost())));
      }
    } else {
      // View.from contributes the logical origin directly to the TMA
      // coordinates. Unlike chunk/subspan indices, these offsets are already in
      // the original tensor's coordinate space and must not be scaled by the
      // view block shape.
      if (auto offsets = op->GetOffsets()) {
        for (auto p : offsets->AllValues()) {
          if (const auto& o = dyn_cast<AST::Expr>(p)->Opts(); o.HasVals()) {
            const auto& vals = o.GetVals();
            for (auto& val : vals) {
              if (sbe::ceq(val, sbe::sym("::__choreo_no_tiling__")))
                coords.push_back(sbe::nu(0));
              else
                indices[coords.size()] = indices[coords.size()] + val;
            }
          } else {
            indices[coords.size()] =
                indices[coords.size()] +
                sbe::sym(OpExprSTR(p, "*", true, IsHost()));
          }
        }
      } else {
        coords = ValxN(sbe::nu(0), shape.DimCount());
      }
    }

    if (auto tc = dyn_cast<TransposeConfig>(config)) {
      assert(tc->dim_values.size() == coords.size());
      assert(ca->TilingOperationCount() == 1);
    }

    for (size_t i = 0; i < coords.size(); ++i) {
      // combine 'a' and 'c' between expressions like 'chunkat(a, b).chunk(c,
      // d)'
      indices[i] = indices[i] + coords[i] * shape.ValueAt(i);
    }
  }

  VST_DEBUG(dbgs() << "Indices for chunkat (" << PSTR(ca)
                   << "): " << STR(indices) << "\n");

  return indices;
}

std::pair<std::string, size_t>
CuteCodeGen::GenMdsOffset(const ptr<AST::ChunkAt> ca,
                          ptr<DMAConfig> config) const {
  auto& sops = ca->AllOperations();
  assert(!sops.empty());

  std::vector<std::ostringstream> offsets;

  size_t sop_base = 0;
  if (auto li = ca->IndexOfLastSpanAs()) sop_base = *li + 1;

  if (sop_base == sops.size()) {
    // span_as is the tail spannedoperation
    std::ostringstream oss;
    auto sz = ca->GetBlockShape().DimCount();
    for (size_t i = 0; i < sz; ++i) {
      if (i > 0) oss << ", ";
      oss << "0";
    }
    return {oss.str(), sz};
  }

  // handle each chunkat inside a seqeunce like 'chunkat(a, b).chunkat(c)...'
  for (size_t sop_idx = sop_base; sop_idx < sops.size(); ++sop_idx) {
    // span_as reshape operation would not affect index generation
    assert(!isa<AST::SOP::Reshape>(sops[sop_idx]));

    // For each chunkat expression, The tiled-block's shape is cooked by shape
    // inference. The block shape is different with the result shape of chunkat
    // expression when using 'modspan', where the result shape represents the
    // shape that applied mod (%) operation. Anyway, for offset, we only care
    // about the tiled-block's shape
    auto& shape = sops[sop_idx]->GetBlockShape();

    std::vector<std::string> exprs;
    // For each 'a, b, c, ...' inside 'chunkat(a, b, c, ...)', that 'b' inside
    // 'chunkat(a, b, c, ...)' could be bounded var like b = {b0, b1} Therefore,
    // we collect all the expressions first.
    for (size_t pi = 0; pi < sops[sop_idx]->IndexNodes().size(); ++pi) {
      auto p = sops[sop_idx]->IndexNodes()[pi];
      // exprs[x] will perform multiplication operations with other values later
      // thus the parent_op is `*`
      auto idx_exprs =
          SplitStringByDelimiter(OpExprSTR(p, "*", true, IsHost()));
      for (size_t i = 0; i < idx_exprs.size(); ++i)
        exprs.push_back(idx_exprs[i]);
    }

    if (auto tc = dyn_cast<TransposeConfig>(config)) {
      assert(tc->dim_values.size() == exprs.size());
      assert(ca->TilingOperationCount() == 1);
    }

    // Generate the expression for single chunkat
    // Note that we buffer all expressions of different chunkats by dimensions
    offsets.resize(exprs.size());

    for (size_t i = 0; i < exprs.size(); ++i) {
      // combine 'a' and 'c' between expressions like 'chunkat(a, b).chunk(c,
      // d)'
      if (sop_idx > sop_base) offsets[i] << " + ";

      if (exprs[i] == "__choreo_no_tiling__")
        offsets[i] << "0";
      else
        offsets[i] << "(int)(" << exprs[i] << " * "
                   << ValueSTR(shape.ValueAt(i)) << ")";
    }
  }

  std::ostringstream offset;
  for (size_t i = 0; i < offsets.size(); ++i) {
    if (i != 0) offset << ", ";
    offset << offsets[i].str();
  }

  VST_DEBUG(dbgs() << "Offset for chunkat (" << PSTR(ca)
                   << "): " << offset.str() << "\n");

  return {offset.str(), offsets.size()};
}

// Compute the full flat element-offset for a ChunkAt that contains span_as.
// All spanned operations (view.from, chunkat, subspan.at, etc.) contribute
// additively; Reshape (span_as) nodes are boundary markers and are skipped by
// GenOffset, so the offset accumulates across the reshape boundary.
//
// Example 1 - span_as after tiling:
//
//   f32 [10, 9, 8] a;
//   ... a.subspan(2, 9, 8).at(p, _, _).span_as(...);
//
//    offset = p * 2 * (9 * 8)
//
// Example 2 - span_as before view/chunkat:
//
//   bf16 [B, T, H, K] q;
//   ... q.span_as([B*T, H, K]).view(LEN, H, K).from(bos, 0, 0)
//        .chunkat(i_l, i_h, i_k);
//
//    offset = bos * H * K + i_l * BL * H * K + i_h * BH * K + i_k * BK
//
const std::string
CuteCodeGen::TileBaseOffset(const ptr<AST::ChunkAt>& ca) const {
  auto lidx = ca->IndexOfLastSpanAs();
  if (!lidx.has_value()) choreo_unreachable("unexpect");
  return ValueSTR(GenOffset(ca));
}

// given i.sop(...).sop(...)..., generate the offset of the final span in the
// original span.
// end_idx: the offset is computed by sop in range [0, end_idx).
const ValueItem CuteCodeGen::GenOffset(const ptr<AST::ChunkAt>& ca,
                                       size_t end_idx) const {
  if (ca->NoOperation()) return sbe::nu(0);

  end_idx = std::min(end_idx, ca->OpCount());

  sbe::ExprSum offset;

  // assert(ca->OpCount() == 1 &&
  //        "count of spanned operations in CuTe DMA should be 1.");

  for (size_t i = 0; i < end_idx; ++i) {
    const auto& sop = ca->OpAt(i);
    if (isa<AST::SOP::Reshape>(sop)) {
      continue;
    } else if (isa<AST::SOP::Tiling>(sop) || isa<AST::SOP::TileAt>(sop) ||
               isa<AST::SOP::SubSpan>(sop)) {
      auto idx = sop->GetIndices()->Opts();
      auto strd = sop->GetBlockStrides();
      auto blk = sop->GetBlockShape();
      assert(idx.HasVals());
      assert(idx.GetVals().size() == strd.size());
      assert(blk.Rank() == strd.size());

      for (size_t ith = 0; ith < idx.GetVals().size(); ++ith)
        offset += idx.GetVals()[ith] * strd[ith] * blk.ValueAt(ith);
    } else if (isa<AST::SOP::View>(sop)) {
      auto off = sop->GetOffsets()->Opts();
      auto strd = sop->GetBlockStrides();
      for (size_t ith = 0; ith < off.GetVals().size(); ++ith)
        offset += off.GetVals()[ith] * strd[ith];
    } else
      choreo_unreachable("unsupported spanned operation.");
  }

  return offset.Get();
}

const ValueList CuteCodeGen::GenStrides(const ptr<AST::ChunkAt>& ca,
                                        const std::vector<size_t>& tc) const {
  auto sty = GetSpannedType(NodeType(*ca));
  auto strides = sty->GetStrides();
  if (tc.size() != 0) {
    assert(tc.size() == strides.size());
    ValueList t_strds = strides;
    for (size_t i = 0; i < strides.size(); ++i) t_strds[i] = strides[tc[i]];
    return t_strds;
  }
  return strides;
}

void CuteCodeGen::EmitFixedHostHead() {
  std::ostringstream oss;
  oss <<
      R"(
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

#include "cutlass/cutlass.h"
)";

  oss << "// include the choreo header;\n";
  if (native_f16)
    oss << "#define __CHOREO_TARGET_NATIVE_HALF_FLOAT_SUPPORT__\n";
  if (native_bf16) oss << "#define __CHOREO_TARGET_NATIVE_BF16_SUPPORT__\n";
  oss << "#include \"choreo.h\"\n";
  if (EnableDebugTypeRTTI()) {
    oss << R"(

static __device__ __attribute__((noinline)) void __choreo_cuda_debug_point__() {
  asm volatile("" ::: "memory");
}
)";
  }
  if (cgi.HasTMA()) oss << "namespace cde = cuda::device::experimental;\n";
  oss << "#include <cooperative_groups.h>";
  oss << "\nusing namespace choreo;\n";
  if (CCtx().GetApiMode() != "sglang") {
    if (!CCtx().DisableCudaRuntimeEnvCheck()) {
      oss << "\n#define __CHOREO_REQUIRED_GPU_DEVICE_SM__ " << CCtx().ArchNum()
          << "\n";
      EmitRuntimeEnvironmentChecker(oss);
    }
  }
  code_segments.push_back(oss.str()); // reset the host code
}

void CuteCodeGen::EmitRuntimeEnvironmentChecker(std::ostream& os) const {
  os << "\nstatic inline void __choreo_check_cuda_environment__() ";
  os << R"({
  // ----------- ONE-TIME GUARD -----------
  static bool already_checked = false;
  if (already_checked) return;
  already_checked = true;
  // --------------------------------------

  auto decode_cuda_version =
   [](int v, int& major, int& minor, int& patch) {
    major = v / 1000;
    minor = (v % 1000) / 10;
    patch = v % 10;
  };

  // ----------- Runtime version check -----------
  int runtime_ver = 0;
  cudaError_t err = cudaRuntimeGetVersion(&runtime_ver);
  if (err != cudaSuccess) {
    std::fprintf(stderr,
                "[choreo] CUDA runtime not available: %s\n",
                cudaGetErrorString(err));
    std::exit(EXIT_FAILURE);
  }

  int driver_ver = 0;
  err = cudaDriverGetVersion(&driver_ver);
  if (err != cudaSuccess) {
    std::fprintf(stderr,
                "[choreo] CUDA driver not available: %s\n",
                cudaGetErrorString(err));
    std::exit(EXIT_FAILURE);
  }

  int rMaj, rMin, rPat;
  int dMaj, dMin, dPat;
  decode_cuda_version(runtime_ver, rMaj, rMin, rPat);
  decode_cuda_version(driver_ver, dMaj, dMin, dPat);

  int reqMaj, reqMin, reqPat;
  decode_cuda_version(CUDART_VERSION, reqMaj, reqMin, reqPat);

  if (runtime_ver < CUDART_VERSION) {
    std::fprintf(stderr,
       "[choreo] CUDA runtime too old:\n"
       "  found runtime %d.%d.%d (encoded=%d)\n"
       "  required      %d.%d.%d (encoded=%d)\n",
       rMaj, rMin, rPat, runtime_ver,
       reqMaj, reqMin, reqPat, CUDART_VERSION);
    std::exit(EXIT_FAILURE);
  }

  // Optional: check driver vs runtime mismatch
  if (driver_ver < runtime_ver) {
    std::fprintf(stderr,
       "[choreo] Warning: CUDA driver (%d.%d.%d, encoded=%d) is older than "
       "the CUDA runtime (%d.%d.%d, encoded=%d). This may cause issues.\n",
       dMaj, dMin, dPat, driver_ver,
       rMaj, rMin, rPat, runtime_ver);
  }

  // ----------- Device capability check -----------
  int device_count = 0;
  err = cudaGetDeviceCount(&device_count);
  if (err != cudaSuccess || device_count == 0) {
    std::fprintf(stderr,
                "[choreo] No CUDA-capable devices found.\n");
    std::exit(EXIT_FAILURE);
  }

  // ----------- Device capability check (selected device) -----------
  int device_id = 0;
  cudaDeviceProp prop{};
  err = cudaGetDeviceProperties(&prop, device_id);
  if (err != cudaSuccess) {
    std::fprintf(stderr,
                 "[choreo] cudaGetDeviceProperties failed: %s\n",
                 cudaGetErrorString(err));
    std::exit(EXIT_FAILURE);
  }

  int sm = prop.major * 10 + prop.minor;
  if (sm < __CHOREO_REQUIRED_GPU_DEVICE_SM__) {
    std::fprintf(stderr,
        "[choreo] Compute capability too low on device %d (%s):\n"
        "  found SM %d.%d (sm_%d)\n"
        "  required SM >= %d (sm_%d)\n",
        device_id, prop.name,
        prop.major, prop.minor, sm,
        __CHOREO_REQUIRED_GPU_DEVICE_SM__, __CHOREO_REQUIRED_GPU_DEVICE_SM__);
    std::exit(EXIT_FAILURE);
  }

#if 0
  // ----------- Optional success log -----------
  std::fprintf(stderr,
    "[choreo] CUDA environment OK\n"
    "  runtime %d.%d.%d (encoded=%d)\n"
    "  driver  %d.%d.%d (encoded=%d)\n"
    "  device  %d: %s, SM %d.%d (sm_%d)\n",
    rMaj, rMin, rPat, runtime_ver,
    dMaj, dMin, dPat, driver_ver,
    device_id, prop.name, prop.major, prop.minor, sm);
#endif
}
)";
}
void CuteCodeGen::EmitFixedDeviceHead() {}

CuteCodeGen::PrepackedU32Info
CuteCodeGen::resolvePrepackedU32Meta(const std::string& ref_sym,
                                     bool forceFlag) {
  PrepackedU32Info info;
  if (ref_sym.empty()) return info;

  auto tryResolve = [&](const std::string& sym) -> bool {
    if (!SSTab().IsDeclared(sym)) return false;
    if (auto st = dyn_cast<SpannedType>(GetSymbolType(sym))) {
      if (st->ElementType() == BaseType::U32) {
        auto key = InScopeName(sym + ".data");
        if (ssm.HasDeviceName(key)) {
          info.device_name = ssm.DeviceName(key);
        } else if (ssm.HasDeviceName(InScopeName(sym))) {
          info.device_name = ssm.DeviceName(InScopeName(sym));
        }
        if (!info.device_name.empty()) {
          info.use_packed_u32 = true;
          return true;
        }
      }
    }
    return false;
  };

  if (!tryResolve(ref_sym + "_mdata")) { tryResolve(ref_sym); }

  if (!info.use_packed_u32 && forceFlag) {
    if (SSTab().IsDeclared(ref_sym)) {
      if (auto st = dyn_cast<SpannedType>(GetSymbolType(ref_sym))) {
        if (st->ElementType() == BaseType::U32) {
          if (ssm.HasDeviceName(InScopeName(ref_sym))) {
            info.device_name = ssm.DeviceName(InScopeName(ref_sym));
            info.use_packed_u32 = true;
          }
        }
      }
    }
    if (!info.use_packed_u32) {
      info.device_name = ref_sym;
      info.use_packed_u32 = true;
    }
  }

  return info;
}

void CuteCodeGen::emitPrepackedU32Snippet(const std::string& metaVar,
                                          const std::string& deviceArray,
                                          const std::string& rowStride,
                                          const std::string& colStride) {
  ds << d_indent << "  int __sp_lane = __sp_tid & 31;\n";
  ds << d_indent << "  bool __sp_active = ((__sp_lane & 3) < 2);\n";
  ds << d_indent
     << "  int __sp_row = ((__sp_tid >> 5) * 16) + ((__sp_tid >> 2) & 7);\n";
  ds << d_indent
     << "  int __sp_packed_row = blockIdx.x * 64 + ((__sp_tid >> 5) * 16) + "
        "(((__sp_tid >> 2) & 7) << 1) + (__sp_tid & 1);\n";
  ds << d_indent << "  int __sp_k_idx = __iv_iv_k * 2 + __iv_iv_warp;\n";
  ds << d_indent << "  if (__sp_active) {\n";
  ds << d_indent << "    uint32_t __sp_u32_val = " << deviceArray
     << "[__sp_packed_row * (" << rowStride << ") + __sp_k_idx * (" << colStride
     << ")];\n";
  ds << d_indent << "    " << metaVar << " = __sp_u32_val;\n";
  ds << d_indent << "  }\n";
}

void CuteCodeGen::emitPrepackedU32TileLoadSnippet(
    const std::string& metaVar, const std::string& tileAddr,
    const std::string& rowStride) {
  ds << d_indent << "  int __sp_lane = __sp_tid & 31;\n";
  ds << d_indent << "  bool __sp_active = ((__sp_lane & 3) < 2);\n";
  ds << d_indent
     << "  int __sp_local_row = ((__sp_tid >> 5) * 16) + (((__sp_tid >> 2) & "
        "7) << 1) + (__sp_tid & 1);\n";
  ds << d_indent << "  if (__sp_active) {\n";
  ds << d_indent << "    uint32_t __sp_u32_val = (" << tileAddr
     << ")[__sp_local_row * (" << rowStride << ")];\n";
  ds << d_indent << "    " << metaVar << " = __sp_u32_val;\n";
  ds << d_indent << "  }\n";
}

void CuteCodeGen::emitFp8PrepackedU32TileLoadSnippet(
    const std::string& metaVar, const std::string& tileAddr,
    const std::string& rowStride, const std::string& colStride) {
  ds << d_indent << "  auto* __sp_meta_u32_ptr = (uint32_t*)(" << tileAddr
     << ");\n";
  ds << d_indent
     << "  int __sp_row = ((__sp_tid >> 2) & 7) + ((__sp_tid & 1) << 3) + "
        "((__sp_tid >> 5) << 4);\n";
  ds << d_indent << "  int __sp_u32_col = (__sp_tid >> 1) & 1;\n";
  ds << d_indent << "  " << metaVar << " = __sp_meta_u32_ptr[__sp_row * ("
     << rowStride << ") + __sp_u32_col * (" << colStride << ")];\n";
}

void CuteCodeGen::emitPrepackedV2TileLoadSnippet(
    const std::string& metaVar, const std::string& baseName,
    const std::string& tileAddr, const std::string& rowStride,
    const std::string& tileOffset) {
  int sval = 0, log2s = 0;
  try {
    sval = std::stoi(rowStride);
  } catch (...) {}
  bool pow2 = sval > 0 && (sval & (sval - 1)) == 0;
  if (pow2) {
    int t = sval;
    while (t > 1) {
      log2s++;
      t >>= 1;
    }
  }
  ds << d_indent << "  int __sp_lane = __sp_tid & 31;\n";
  ds << d_indent << "  bool __sp_active = ((__sp_lane & 3) < 2);\n";
  ds << d_indent
     << "  int __sp_lr = ((__sp_tid >> 5) * 16) + (((__sp_tid >> 2) "
        "& 7) << 1) + (__sp_tid & 1);\n";
  ds << d_indent << "  if (__sp_active) {\n";
  if (pow2) {
    int smask = sval - 1;
    int block_shift = log2s + 4;
    std::string off_expr;
    if (!tileOffset.empty())
      off_expr = tileOffset;
    else
      off_expr = std::string("(int)((") + tileAddr + ") - (" + baseName + "))";
    ds << d_indent << "    int __sp_row = (" << off_expr << " >> " << log2s
       << ") + __sp_lr;\n";
    ds << d_indent << "    " << metaVar << " = ((const uint32_t*)(" << baseName
       << ") + ((__sp_row >> 4) << " << block_shift << ") + (__sp_row & 15))[("
       << off_expr << " & " << smask << ") << 4];\n";
  } else {
    ds << d_indent << "    int __sp_to = (int)((const uint32_t*)(" << tileAddr
       << ") - (const uint32_t*)(" << baseName << "));\n";
    ds << d_indent << "    int __sp_br = __sp_to / (" << rowStride << ");\n";
    ds << d_indent << "    int __sp_r = __sp_br + __sp_lr;\n";
    ds << d_indent << "    " << metaVar << " = ((const uint32_t*)(" << baseName
       << "))[((__sp_r >> 4) * (" << rowStride << ") + __sp_to - __sp_br * ("
       << rowStride << ")) * 16 + (__sp_r & 15)];\n";
  }
  ds << d_indent << "  }\n";
}

void CuteCodeGen::emitPrepackedV2Snippet(const std::string& metaVar,
                                         const std::string& baseName,
                                         const std::string& /*deviceArray*/,
                                         const std::string& rowStride,
                                         const std::string& /*colStride*/) {
  ds << d_indent << "  int __sp_lane = __sp_tid & 31;\n";
  ds << d_indent << "  bool __sp_active = ((__sp_lane & 3) < 2);\n";
  ds << d_indent
     << "  int __sp_packed_row = blockIdx.x * 64 + ((__sp_tid >> 5) * 16)"
        " + (((__sp_tid >> 2) & 7) << 1) + (__sp_tid & 1);\n";
  ds << d_indent << "  int __sp_k_idx = __iv_iv_k * 2 + __iv_iv_warp;\n";
  ds << d_indent << "  if (__sp_active) {\n";
  ds << d_indent << "    " << metaVar << " = ((const uint32_t*)(" << baseName
     << "))[((__sp_packed_row >> 4) * (" << rowStride
     << ") + __sp_k_idx) * 16 + (__sp_packed_row & 15)];\n";
  ds << d_indent << "  }\n";
}

void CuteCodeGen::emitFp8PrepackedV2TileLoadSnippet(
    const std::string& metaVar, const std::string& baseName,
    const std::string& tileAddr, const std::string& rowStride,
    const std::string& /*colStride*/) {
  int sval = 0, log2s = 0;
  try {
    sval = std::stoi(rowStride);
  } catch (...) {}
  bool pow2 = sval > 0 && (sval & (sval - 1)) == 0;
  if (pow2) {
    int t = sval;
    while (t > 1) {
      log2s++;
      t >>= 1;
    }
  }
  ds << d_indent
     << "  int __sp_lr = ((__sp_tid >> 2) & 7)"
        " + ((__sp_tid & 1) << 3) + ((__sp_tid >> 5) << 4);\n";
  ds << d_indent << "  int __sp_uc = (__sp_tid >> 1) & 1;\n";
  if (pow2) {
    int smask = sval - 1;
    int half_s = sval >> 1;
    ds << d_indent << "  int __sp_v1 = (int)((" << tileAddr << ") - ("
       << baseName << ")) + (__sp_lr << " << log2s << ") + __sp_uc;\n";
    ds << d_indent << "  int __sp_r = __sp_v1 >> " << log2s << ";\n";
    ds << d_indent << "  int __sp_kc = __sp_v1 & " << smask << ";\n";
    ds << d_indent << "  int __sp_rib = __sp_r & 15;\n";
    ds << d_indent
       << "  int __sp_vtid = ((__sp_rib & 7) << 2)"
          " | ((__sp_kc & 1) << 1) | ((__sp_rib >> 3) & 1);\n";
    ds << d_indent << "  " << metaVar << " = ((const uint32_t*)(" << baseName
       << "))[((__sp_r >> 4) * " << half_s
       << " + (__sp_kc >> 1)) * 32 + __sp_vtid];\n";
  } else {
    ds << d_indent << "  int __sp_log2s = 31 - __clz((int)(" << rowStride
       << "));\n";
    ds << d_indent << "  int __sp_v1 = (int)((" << tileAddr << ") - ("
       << baseName << ")) + (__sp_lr << __sp_log2s) + __sp_uc;\n";
    ds << d_indent << "  int __sp_r = __sp_v1 >> __sp_log2s;\n";
    ds << d_indent << "  int __sp_kc = __sp_v1 & ((" << rowStride
       << ") - 1);\n";
    ds << d_indent << "  int __sp_rib = __sp_r & 15;\n";
    ds << d_indent
       << "  int __sp_vtid = ((__sp_rib & 7) << 2)"
          " | ((__sp_kc & 1) << 1) | ((__sp_rib >> 3) & 1);\n";
    ds << d_indent << "  " << metaVar << " = ((const uint32_t*)(" << baseName
       << "))[((__sp_r >> 4) * ((" << rowStride
       << ") >> 1) + (__sp_kc >> 1)) * 32 + __sp_vtid];\n";
  }
}

void CuteCodeGen::EmitDebugSpannedRTTI(
    std::ostringstream& os, const std::string& indent, const std::string& sym,
    const ptr<SpannedType>& sty, const std::string& data_expr,
    const std::vector<std::string>& shape_exprs,
    const std::vector<std::string>& stride_exprs) const {
  auto rank = sty->Dims();
  assert(shape_exprs.size() == rank && "spanned debug shape rank mismatch");
  assert(stride_exprs.size() == rank && "spanned debug stride rank mismatch");
  os << indent << "choreo::rtti::spanned<" << NameBaseType(sty->ElementType())
     << ", " << rank << "> " << sym << " = {{";

  for (size_t i = 0; i < shape_exprs.size(); ++i) {
    if (i > 0) os << ", ";
    os << shape_exprs[i];
  }

  os << "}, {";
  for (size_t i = 0; i < stride_exprs.size(); ++i) {
    if (i > 0) os << ", ";
    os << stride_exprs[i];
  }

  os << "}, " << data_expr << "};\n";
}

bool CuteCodeGen::Visit(AST::FunctionDecl& n) {
  TraceEachVisit(n);

  assert(n.name == fname && "inconsistent in function names.");
  assert(isa<FunctionType>(n.GetType()) && "unexpected type.");

  auto HandleSymbolicDimensions = [this](const ptr<SpannedType>& sty,
                                         const std::string& hp_name,
                                         size_t hp_index) {
    size_t dim_index = 0;
    for (auto vi : sty->GetShape().Value()) {
      if (auto vale = VISym(vi)) { // the dimension is symbolic
        assert(PrefixedWith(*vale, "::" + fname + "::") &&
               "unexpected symbolic dimension name.");

        auto dim_expr = hp_name + ".shape()[" + std::to_string(dim_index) + "]";
        if (symbolic_dimensions.count(*vale) == 0) {
          symbolic_dimensions[*vale] = {dim_expr, hp_index, dim_index};
          ssm.MapDeviceSymbol(*vale, UnScopedName(*vale));
        }
      }
      assert(!VIIsBop(vi) && "unexpected binary operation.");
      dim_index++;
    }
  };

  // Go through all the symbols appeared in cute host function, do:
  //
  //  - decide the host parameter names,
  //  - map the runtime shape dimensions to the real host code expression
  //  - decide the cute-host parameter names,
  //  - decide the cute-host parameter indices,
  //
  size_t host_pindex = 0;
  for (auto& item : GetChoreoFuncIns(cgi)) {
    if (item.IsParameter()) {
      assert((int)host_pindex == item.p_index);
      item.host_name = UnScopedName(item.name);
      if (auto sty = dyn_cast<SpannedType>(item.type)) {
        ssm.MapHostSymbol(item.name, item.host_name + ".data()");
        HandleSymbolicDimensions(sty, item.host_name, host_pindex);
      } else
        ssm.MapHostSymbol(item.name, item.host_name);
    } else
      item.host_name = UnScopedName(item.name);

    item.h_name = "args[" + std::to_string(host_pindex) + "]";
    item.h_index = host_pindex;
    host_pindex++;
  }

  if (isa<VoidType>(fty->out_ty)) {
    void_return = true;
    VST_DEBUG(dbgs() << fname << ": void return\n");
  }

  EmitHostFuncDecl(hs);

  hs << " {\n";
  IncrHostIndent();

  if (CCtx().GetApiMode() != "sglang")
    if (!CCtx().DisableCudaRuntimeEnvCheck())
      hs << h_indent << "__choreo_check_cuda_environment__();\n";

  // name the symbolic dimensions for better readability
  for (auto item : symbolic_dimensions) {
    hs << h_indent << "auto &" << UnScopedName(item.first) << " = "
       << item.second.hsd_expr << ";\n";
    ssm.MapHostSymbol(item.first, UnScopedName(item.first));
  }

  // emit the runtime checks
  EmitHostRuntimeCheck();

  if (EnableDebugTypeRTTI()) {
    for (auto& item : GetChoreoFuncIns(cgi)) {
      auto sty = dyn_cast<SpannedType>(item.type);
      if (!sty || !item.IsParameter()) continue;

      std::vector<std::string> shape_exprs;
      shape_exprs.reserve(sty->Dims());
      for (size_t i = 0; i < sty->Dims(); ++i)
        shape_exprs.push_back(item.host_name + ".shape()[" + std::to_string(i) +
                              "]");

      std::vector<std::string> stride_exprs;
      const auto& strides = sty->GetStrides();
      assert(strides.size() == sty->Dims() && "missing spanned strides");
      stride_exprs.reserve(sty->Dims());
      for (auto& vi : strides)
        stride_exprs.push_back(UnScopedExpr(ValueSTR(vi)));

      EmitDebugSpannedRTTI(hs, h_indent, "__dbg_" + UnScopedName(item.name),
                           sty, item.host_name + ".data()", shape_exprs,
                           stride_exprs);
    }
  }

  // do not generate device function unless parallel-by exists
  if (NeedDeviceFunc()) {
    // map the choreo input to device memory
    for (auto& item : GetChoreoFuncIns(cgi)) {
      if (auto sty = dyn_cast<SpannedType>(item.type)) {
        if (item.attr == ParamAttr::GLOBAL_INPUT) {
          ssm.MapHostSymbol(item.name + "__device",
                            UnScopedName(item.name) + ".data()");
          continue;
        }
        // Only the globals are declared in host.
        // The shareds/locals are declared in device.
        auto sym = UnScopedName(item.name);
        std::string bts = NameBaseType(sty->ElementType());
        auto buf_sym = sym + "__device";
        hs << h_indent << bts << " * " << buf_sym << " = nullptr;\n";
        hs << h_indent << "choreo::abend_true(cudaMalloc(&" << buf_sym << ", "
           << UnScopedSizeExpr(*sty) << "));\n";
        hs << h_indent << "choreo::abend_true(cudaMemcpy(" << buf_sym << ", "
           << ssm.HostName(item.name) << ", " << UnScopedSizeExpr(*sty)
           << ", cudaMemcpyHostToDevice));\n";
        ssm.MapHostSymbol(item.name + "__device", buf_sym);
        global_buffers.insert(buf_sym);
      }
    }
  }

  return true;
}

bool CuteCodeGen::Visit(AST::ChoreoFunction& n) {
  TraceEachVisit(n);

  // If there is no AST::Return
  if (return_stream.str().empty() && NeedDeviceFunc()) EmitCudaFree();

  has_pending_wgmma_finalize = false;
  warpspec_wgmma_arrived = false;

  DecrHostIndent();
  hs << "}\n\n";

  return true;
}

bool CuteCodeGen::Visit(AST::NamedVariableDecl& n) {
  TraceEachVisit(n);

  auto extract_chunk_alias =
      [](const ptr<AST::Node>& node) -> ptr<AST::ChunkAt> {
    if (!node) return nullptr;
    if (auto ca = dyn_cast<AST::ChunkAt>(node)) return ca;
    if (auto expr = dyn_cast<AST::Expr>(node)) {
      if (auto ref = expr->GetReference()) return dyn_cast<AST::ChunkAt>(ref);
    }
    return nullptr;
  };

  auto nty = NodeType(n);
  auto sym = n.name_str;
  const bool enable_debug_rtti = EnableDebugTypeRTTI();

  bool ref = n.HasNote("ref");
  // workaround:
  // if a symbol is declared but have no symbol value(optimized value)
  // pass it to device func even it is unused.
  auto sname = InScopeName(sym);
  if (!FCtx(fname).HasSymbolValues(sname))
    updating_cgi.AddSymbolDetail(fname, {sname, GetSymbolType(sym), true});
  else {
    auto sv = FCtx(fname).GetSymbolValues(sname);
    // workround: for symbolic valno, treat it as ref to do codegen.
    if (n.IsMutable() && sv.HasVals() && sv.GetVals().size() == 1 &&
        VIIsSym(sv.GetVal()))
      ref = true;
    updating_cgi.AddSymbolDetail(fname, {sname, GetSymbolType(sym), ref});
  }

  if (!IsHost()) {
    if (auto rhs_ca = extract_chunk_alias(n.init_expr)) {
      live_chunk_aliases[sym] = rhs_ca;
      live_chunk_aliases[sname] = rhs_ca;
    } else {
      live_chunk_aliases.erase(sym);
      live_chunk_aliases.erase(sname);
    }
  }

  // The type is determined first, and then
  // the device or host side is determined

  if (auto s = dyn_cast<AST::Select>(n.init_expr)) {
    assert(!IsHost() && "select should be on device side.");
    assert(!s->inDMA);
    size_t val_count = s->expr_list->Count();
    assert(val_count >= 2);
    std::string array_sym = sym + "_select_array__";
    if (isa<FutureType>(NodeType(*s))) {
      ds << d_indent << "future * " << array_sym << "[] = {";
      for (size_t i = 0; i < val_count; i++) {
        if (i > 0) ds << ", ";
        ds << "&" << OpExprSTR(s->expr_list->ValueAt(i), "&", false, false);
      }
      ds << "};\n";
      // make symbol a reference
      ds << d_indent << "future & " << sym << " = *" << array_sym << "["
         << ExprSTR(s->select_factor, false) << "];\n";
    } else
      choreo_unreachable("select of " + PSTR(NodeType(*s)) +
                         " is yet to implement.");

    return true;
  }
  ptr<AST::SpanAs> sa = nullptr;
  if (auto e = dyn_cast<AST::Expr>(n.init_expr)) {
    if (sa = dyn_cast<AST::SpanAs>(e->GetReference())) {
      // handle span_as of global buffer in `HandleGlobal`
      if (!IsHost()) {
        auto sty = dyn_cast<SpannedType>(nty);
        bool is_internal_spanned = PrefixedWith(sym, "anon_") ||
                                   PrefixedWith(sym, "__iv_") ||
                                   SuffixedWith(sym, "__buf__");
        bool use_user_visible_debug_sym =
            enable_debug_rtti && !is_internal_spanned;
        std::string raw_sym = use_user_visible_debug_sym ? "__raw_" + sym : sym;

        auto tty = GetSymbolType(sa->id->name);
        auto base_dev = ssm.DeviceName(InScopeName(sa->id->name));
        std::string base_expr =
            isa<FutureType>(tty) ? base_dev + ".data()" : base_dev;
        if (sa->subscriptions && sa->subscriptions->Count() > 0) {
          auto sat = dyn_cast<SpannedArrayType>(tty);
          assert(sat && "span_as with subscriptions requires array type.");
          auto subs = sa->subscriptions->AllValues();
          std::vector<ptr<AST::Node>> sv(subs.begin(), subs.end());
          base_expr =
              LinearizeArrayOffset(base_expr, sv, sat->Dimensions(),
                                   sty->GetShape().ElementCountValue(), false);
        }
        ds << d_indent << "auto* " << raw_sym << " = ";
        ds << "static_cast<"
           << NameBaseType(dyn_cast<SpannedType>(nty)->ElementType()) << "*>("
           << base_expr << ");\n";
        if (use_user_visible_debug_sym) {
          std::vector<std::string> shape_exprs;
          shape_exprs.reserve(sty->Dims());
          for (auto& vi : sty->GetShape().Value())
            shape_exprs.push_back(UnScopedExpr(ValueSTR(vi)));

          std::vector<std::string> stride_exprs;
          assert(sty->GetStrides().size() == sty->Dims() &&
                 "missing spanned strides");
          stride_exprs.reserve(sty->Dims());
          for (auto& vi : sty->GetStrides())
            stride_exprs.push_back(UnScopedExpr(ValueSTR(vi)));

          EmitDebugSpannedRTTI(ds, d_indent, sym, sty, raw_sym, shape_exprs,
                               stride_exprs);
        }
        ssm.MapDeviceSymbol(InScopeName(sym), raw_sym);
        return true;
      }
    }
  }

  if (auto sty = dyn_cast<SpannedType>(nty)) {
    auto buf_sym = sym + "__device";
    // globals are declared in host, while shareds/locals are declared in device
    auto shape = sty->GetShape();
    std::string bts{NameBaseType(sty->ElementType())};
    auto sto = sty->GetStorage();

    bool spmem = false; // allocatable scratchpad memory: share, local

    auto HandleGlobal = [&]() -> void {
      bts = NameBaseType(sty->ElementType()); // use the device type name

      if (IsChoreoOutput(InScopeName(sym))) {
        // the sym is choreo output
        std::string sym_data = sym + ".data()";
        hs << h_indent << "auto " << sym
           << " = choreo::make_spandata<choreo::" << STR(sty->e_type) << ", "
           << shape.Rank() << ">({"
           << ShapeSTR(shape, false, ", ", BaseType::U64) << "});\n";
        if (n.init_value) {
          // support initialization of output
          hs << h_indent << "std::fill(" << sym_data << ", " << sym_data << "+"
             << sym << ".element_count()"
             << ", "
             << ExprCastSTR(n.init_value, std::nullopt, GetBaseType(*sty),
                            GetBaseType(*n.init_value->GetType()), true)
             << ");\n";
        }
        if (sa) {
          // is span_as
          hs << h_indent << bts << " * " << buf_sym << " = " << sa->id->name;
          bool is_global_arg = false;
          for (const auto& item : GetChoreoFuncIns(cgi)) {
            if (UnScopedName(item.name) == sa->id->name) {
              is_global_arg = (item.attr == ParamAttr::GLOBAL_INPUT);
              break;
            }
          }
          if (is_global_arg)
            hs << ".data();\n";
          else
            hs << "__device;\n";

        } else {
          hs << h_indent << bts << " * " << buf_sym << " = nullptr;\n";
          hs << h_indent << "choreo::abend_true(cudaMalloc(&" << buf_sym << ", "
             << UnScopedSizeExpr(*sty) << "));\n";
          if (n.init_value) {
            hs << h_indent << "choreo::abend_true(cudaMemcpy(" << buf_sym
               << ", " << sym_data << ", " << UnScopedSizeExpr(*sty)
               << ", cudaMemcpyHostToDevice));\n";
          }
        }
        global_buffers.insert(buf_sym);
        return;
      }

      // the sym is not choreo output
      if (FBIContainsBuffer(FBInfo(), InScopeName(sym)) &&
          use_hetero_tileflow && IsHost()) {
        // a non-init global var decl tied with future
        // this hint is enough to say a host side dataflow
        VST_DEBUG(dbgs() << "Found " << buf_sym << " in FBInfo - "
                         << STR(FBInfo()) << "\n");
      } else {
        if (sa) {
          // is span_as
          hs << h_indent << bts << " * " << buf_sym << " = " << sa->id->name;
          bool is_global_arg = false;
          for (const auto& item : GetChoreoFuncIns(cgi)) {
            if (UnScopedName(item.name) == sa->id->name) {
              is_global_arg = (item.attr == ParamAttr::GLOBAL_INPUT);
              break;
            }
          }
          if (is_global_arg)
            hs << ".data();\n";
          else
            hs << "__device;\n";
        } else {
          hs << h_indent << bts << " * " << buf_sym << " = nullptr;\n";
          hs << h_indent << "choreo::abend_true(cudaMalloc(&" << buf_sym << ", "
             << UnScopedSizeExpr(*sty) << "));\n";

          if (!n.init_value) return;

          std::string sym_data = sym + ".data()";
          hs << h_indent << "auto " << sym
             << " = choreo::make_spandata<choreo::" << STR(sty->e_type) << ", "
             << shape.Rank() << ">({"
             << ShapeSTR(shape, false, ", ", BaseType::U64) << "});\n";
          hs << h_indent << "std::fill(" << sym_data << ", " << sym_data << "+"
             << sym << ".element_count()"
             << ", "
             << ExprCastSTR(n.init_value, std::nullopt, GetBaseType(*sty),
                            GetBaseType(*n.init_value->GetType()), true)
             << ");\n";
          hs << h_indent << "choreo::abend_true(cudaMemcpy(" << buf_sym << ", "
             << sym_data << ", " << UnScopedSizeExpr(*sty)
             << ", cudaMemcpyHostToDevice));\n";
        }
      }
    };

    auto HandleSharedLocal = [&]() -> void {
      if (IsChoreoOutput(InScopeName(sym)))
        choreo_unreachable(
            "error: shared/local buffer cannot be Choreo output.");

      auto type_modifiers = (sto == Storage::SHARED ? "__shared__ " : "");

      if (!CCtx().MemReuse()) {
        if (n.HasNote("ref"))
          Error1(
              n.LOC(),
              "buffer reference is enabled only when memory reuse is enabled.");
        ds << d_indent << type_modifiers << bts << " " << sym;
        for (const auto& dim : GetArrayDimensions(nty))
          ds << "[" << ValueSTR(dim) << "]";
        ds << "[" << UnScopedExpr(ElemCountExprOf(*sty)) << "];\n";
        return;
      }

      // memory reuse is enabled

      if (n.HasNote("spm")) {
        if (sto == Storage::SHARED &&
            (FCtx(fname).HaveDynamicBuffer(SSTab().ScopeName(), sto) ||
             set_cuda_func_attribute_max_dynamic_shared_memory_size)) {
          std::string decl = d_indent + "auto " + sym + " = (" + bts + "*)" +
                             device_fn + "__runtime_shared_buffer__;\n";
          if (cluster_defers_launch)
            deferred_spm_decls += decl;
          else
            ds << decl;
        } else {
          size_t alignment = std::stoull(n.GetNote("alignment"));
          if (sto == Storage::SHARED && HasWGMMAInFunction())
            alignment = std::max(alignment, static_cast<size_t>(1024));
          ds << d_indent << type_modifiers << "alignas(" << alignment << ") "
             << bts << " " << sym << "[" << UnScopedExpr(ElemCountExprOf(*sty))
             << "];\n";
        }
        return;
      }

      // the buffer is not the declared whole spm.
      if (n.HasNote("reuse")) {
        auto reuse = n.GetNote("reuse");
        auto offset = n.GetNote("offset");
        ds << d_indent << bts << "* " << sym << " = (" << bts << "*)"
           << "(" << reuse << " + " << offset << ");\n";
      } else {
        // the buffer is not reused
        assert(!n.HasNote("offset"));
        if (n.HasNote("ref")) {
          ds << d_indent << bts << "* " << sym << " = (" << bts << "*)("
             << ExprSTR(n.init_expr, false) << ");\n";
        } else {
          // The buffer is declared but never used.
          // TODO: should we DCE the unused buffer?
          ds << d_indent << type_modifiers << bts << " " << sym << "["
             << UnScopedExpr(ElemCountExprOf(*sty)) << "];\n";
        }
      }
    };

    if (sto == Storage::GLOBAL) {
      if (!IsHost()) choreo_unreachable("error: global var decl in device.");
      HandleGlobal();
      ssm.MapHostSymbol(InScopeName(sym) + "__device", buf_sym);
      ssm.MapHostSymbol(InScopeName(sym), buf_sym);
      ssm.MapDeviceSymbolIfNotExist(InScopeName(sym), sym);
      global_buffers.insert(buf_sym);
    } else if (sto == Storage::SHARED || sto == Storage::LOCAL) {
      if (IsHost()) choreo_unreachable("error: shared/local var decl in host.");
      HandleSharedLocal();
      ssm.MapDeviceSymbol(InScopeName(sym), sym);
      spmem = true;
    } else
      choreo_unreachable("unsupported storage type.");

    bool is_internal_spanned = PrefixedWith(sym, "anon_") ||
                               PrefixedWith(sym, "__iv_") ||
                               SuffixedWith(sym, "__buf__");
    if (enable_debug_rtti && !is_internal_spanned) {
      auto& os = IsHost() ? hs : ds;
      auto& ind = IsHost() ? h_indent : d_indent;

      std::vector<std::string> shape_exprs;
      shape_exprs.reserve(sty->Dims());
      for (auto& vi : sty->GetShape().Value())
        shape_exprs.push_back(UnScopedExpr(ValueSTR(vi)));

      std::vector<std::string> stride_exprs;
      assert(sty->GetStrides().size() == sty->Dims() &&
             "missing spanned strides");
      stride_exprs.reserve(sty->Dims());
      for (auto& vi : sty->GetStrides())
        stride_exprs.push_back(UnScopedExpr(ValueSTR(vi)));

      std::string data_expr;
      if (sto == Storage::GLOBAL)
        data_expr = IsHost() ? buf_sym : sym;
      else
        data_expr = sym;

      std::string debug_sym = "__dbg_" + sym;
      if (IsHost() && sto == Storage::GLOBAL && sa &&
          !IsChoreoOutput(InScopeName(sym)) && !n.init_value)
        debug_sym = sym;

      EmitDebugSpannedRTTI(os, ind, debug_sym, sty, data_expr, shape_exprs,
                           stride_exprs);
    }

    // initialize the spm buffer if needed
    if (spmem && n.init_value) {
      if (sto != Storage::SHARED && sto != Storage::LOCAL)
        choreo_unreachable(
            "error: unexpected storage type in spm initialization.");
      if (sto == Storage::SHARED) {
        ds << d_indent << LevelPred() << " {\n";
        IncrDeviceIndent();
      }
      auto ec = sty->GetShape().ElementCountValue();
      if (ec == sbe::nu(1))
        ds << d_indent << sym << " = " << ExprSTR(n.init_value) << ";\n";
      else {
        ds << d_indent << "for (int i = 0; i < " << ValueSTR(ec) << "; ++i) ";
        ds << sym << "[i] = " << ExprSTR(n.init_value) << ";\n";
      }
      if (sto == Storage::SHARED) {
        DecrDeviceIndent();
        ds << d_indent << "} // single instance\n";
        ds << d_indent << EmitSync(Storage::SHARED) << ";\n";
      }
    }
    return true;
  }

  if (enable_debug_rtti) {
    if (auto bty = dyn_cast<BoundedType>(nty)) {
      IndStream() << "choreo::rtti::bounded_ituple<" << bty->Dims() << "> "
                  << sym;
      if (n.init_expr) Stream() << " = " << ExprSTR(n.init_expr, false) << "";
      Stream() << ";\n";
      return true;
    }

    if (auto mty = dyn_cast<MDSpanType>(nty)) {
      IndStream() << "choreo::rtti::mdspan<" << mty->Dims() << "> " << sym
                  << " = ";
      mty->GetShape().PrintAsList(Stream());
      Stream() << ";\n";
      return true;
    }

    if (auto ity = dyn_cast<ITupleType>(nty)) {
      IndStream() << "choreo::rtti::ituple<" << ity->Dims() << "> " << sym;
      if (n.init_expr) Stream() << " = " << ExprSTR(n.init_expr, false);
      Stream() << ";\n";
      return true;
    }
  } else {
    if (auto bty = dyn_cast<BoundedType>(nty)) {
      // bounded variable is not with a fixed value
      if (!IsActualBoundedIntegerType(bty))
        choreo_unreachable(
            "yet to support: bounded ituple variable code generation.");
      IndStream() << "int " << sym << " = " << ExprSTR(n.init_expr, false)
                  << ";\n";
      return true;
    }
  }

  // when symbol is not valued
  if (isa<ScalarType>(nty) &&
      (IsMutable(*nty) || !FCtx(fname).HasSymbolValues(InScopeName(sym)))) {
    auto mem = n.GetMemory();
    IndStream();
    if (mem != nullptr) {
      auto st = mem->Get();
      Stream() << CudaDeviceMemory(st) << " ";
    }
    Stream() << NameBaseType(GetBaseType(*nty)) << " " << sym;
    if (n.init_expr) Stream() << " = " << ExprSTR(n.init_expr, false);
    Stream() << ";\n";

    // mutables have references
    if (IsMutable(*nty))
      if (!IsHost()) ssm.MapDeviceSymbol(InScopeName(sym), sym);

    return true;
  }

  // handle events
  if (auto ety = dyn_cast<EventArrayType>(nty)) {
    switch (ety->GetStorage()) {
    case Storage::GLOBAL: {
      assert(IsHost());
      auto sym = InScopeName(n.name_str);
      auto buf_sym = n.name_str + "__device";
      hs << h_indent << "bool * " << buf_sym << " = nullptr; // global event\n";
      hs << h_indent << "choreo::abend_true(cudaMalloc(&" << buf_sym << ", "
         << ety->ElemCount() << "));\n";
      hs << h_indent << "choreo::abend_true(cudaMemset(&" << buf_sym << ", 0, "
         << ety->ElemCount() << "));\n";
      ssm.MapHostSymbol(sym, buf_sym);
      ssm.MapDeviceSymbol(sym, n.name_str);
      global_buffers.insert(buf_sym);
    } break;
    case Storage::SHARED: {
      assert(!IsHost());
      bool is_cluster_event = cluster_trigger_events_.count(n.name_str) > 0;

      if (is_cluster_event) {
        ds << d_indent << "__shared__ __align__(8) uint64_t " << n.name_str;
        ety->PrintAsCArray(ds);
        ds << "; // shared event mbarrier (cluster-scope)\n";
        ds << d_indent << "int " << n.name_str << "__phase";
        ety->PrintAsCArray(ds);
        ds << ";\n";
      } else {
        ds << d_indent << "__shared__ cuda::barrier<cuda::thread_scope_block> "
           << n.name_str;
        ety->PrintAsCArray(ds);
        ds << "; // shared event barrier\n";
      }

      if (IsWarpSpecActive()) {
        ds << d_indent << "// initialize the event barrier\n";
        ds << d_indent << "if (__CHOREO_BLOCK_SINGLE__) {\n";

        auto event_tc = ety->event->GetThreadCount();
        auto& ft = cgi.GetFunctionTrait(fname);
        auto auto_init = [&](int64_t explicit_tc,
                             const std::string& ev) -> std::string {
          if (explicit_tc > 0) return std::to_string(explicit_tc);
          auto ep = ft.GetEventParticipation(ev);
          if (ep > 0) return std::to_string(ep);
          return "(blockDim.x - 128) + 1";
        };
        std::string event_init_count = auto_init(event_tc, n.name_str);

        if (is_cluster_event) {
          GenerateSubscriptions(
              ds, "  " + d_indent + "choreo::tma_mbarrier_init(&" + n.name_str,
              ", (blockDim.x / 128 - 1) * choreo::tma_cluster_dim());\n",
              ety->Dimensions());
        } else {
          GenerateSubscriptions(ds, "  " + d_indent + "init(&" + n.name_str,
                                ", " + event_init_count + ");\n",
                                ety->Dimensions());
        }

        ds << d_indent << "  cde::fence_proxy_async_shared_cta();\n";
        ds << d_indent << "}\n";
        ds << d_indent << "__syncthreads();\n";
        if (is_cluster_event) {
          GenerateSubscriptions(ds, d_indent + n.name_str + "__phase",
                                " = 0;\n", ety->Dimensions());
        }
        break;
      }

      ds << d_indent << "// initialize the event barrier\n";
      ds << d_indent << "if (__CHOREO_BLOCK_SINGLE__) {\n";
      auto event_tc = ety->event->GetThreadCount();
      std::string init_count;
      if (event_tc > 0) {
        init_count = std::to_string(event_tc);
      } else {
        auto ep = cgi.GetFunctionTrait(fname).GetEventParticipation(n.name_str);
        init_count = ep > 0 ? std::to_string(ep) : "(blockDim.x - 128) + 1";
      }
      if (is_cluster_event) {
        GenerateSubscriptions(
            ds, "  " + d_indent + "choreo::tma_mbarrier_init(&" + n.name_str,
            ", (blockDim.x / 128 - 1) * choreo::tma_cluster_dim());\n",
            ety->Dimensions());
      } else {
        GenerateSubscriptions(ds, "  " + d_indent + "init(&" + n.name_str,
                              ", " + init_count + ");\n", ety->Dimensions());
      }
      ds << d_indent << "  cde::fence_proxy_async_shared_cta();\n";
      ds << d_indent << "}\n";
      ds << d_indent << "__syncthreads();\n";
      if (is_cluster_event) {
        GenerateSubscriptions(ds, d_indent + n.name_str + "__phase", " = 0;\n",
                              ety->Dimensions());
      }
    } break;
    case Storage::LOCAL: {
      assert(!IsHost());
      ds << d_indent << CudaDeviceMemory(ety->GetStorage())
         << " __volatile__ bool " << n.name_str;
      ety->PrintAsCArray(ds);
      ds << "; // " << STR(ety->GetStorage()) << " event\n";
      ds << d_indent << "// initialize the event\n";
      GenerateSubscriptions(ds, d_indent + n.name_str, " = false;\n",
                            ety->Dimensions());
      ds << d_indent << EmitSync(ety->GetStorage()) << ";\n";
    } break;
    default: break;
    }
  } else if (auto ety = dyn_cast<EventType>(nty)) {
    switch (ety->GetStorage()) {
    case Storage::GLOBAL: {
      assert(IsHost());
      auto sym = InScopeName(n.name_str);
      auto buf_sym = n.name_str + "__device";
      hs << h_indent << "bool * " << buf_sym << " = nullptr; // global event\n";
      hs << h_indent << "choreo::abend_true(cudaMalloc(&" << buf_sym
         << ", 1));\n";
      hs << h_indent << "choreo::abend_true(cudaMemset(&" << buf_sym
         << ", 0, 1));\n";
      ssm.MapHostSymbol(sym, buf_sym);
      ssm.MapDeviceSymbol(sym, n.name_str);
      global_buffers.insert(buf_sym);
    } break;
    case Storage::SHARED: {
      assert(!IsHost());
      ds << d_indent << "__shared__ cuda::barrier<cuda::thread_scope_block> "
         << n.name_str << "; // shared event barrier\n";
      ds << d_indent << "if (__CHOREO_BLOCK_SINGLE__) {\n";
      {
        auto etc = ety->GetThreadCount();
        std::string ic;
        if (etc > 0)
          ic = std::to_string(etc);
        else {
          auto ep =
              cgi.GetFunctionTrait(fname).GetEventParticipation(n.name_str);
          ic = ep > 0 ? std::to_string(ep) : "(blockDim.x - 128) + 1";
        }
        ds << d_indent << "  init(&" << n.name_str << ", " << ic << ");\n";
      }
      ds << d_indent << "  cde::fence_proxy_async_shared_cta();\n";
      ds << d_indent << "}\n";
      ds << d_indent << "__syncthreads();\n";
    } break;
    case Storage::LOCAL: {
      assert(!IsHost());
      ds << d_indent << CudaDeviceMemory(ety->GetStorage())
         << " __volatile__ bool " << n.name_str << "; // "
         << STR(ety->GetStorage()) << " event\n";
      ds << d_indent << n.name_str << " = false;\n";
      ds << d_indent << EmitSync(ety->GetStorage()) << ";\n";
    } break;
    default: break;
    }
  }

  return true;
}

bool CuteCodeGen::Visit(AST::NamedTypeDecl& n) {
  TraceEachVisit(n);

  auto nty = NodeType(n);
  auto sym = n.name_str;
  const bool enable_debug_rtti = EnableDebugTypeRTTI();

  auto sname = InScopeName(sym);
  if (!FCtx(fname).HasSymbolValues(sname))
    updating_cgi.AddSymbolDetail(fname, {sname, GetSymbolType(sym), false});

  if (auto mty = dyn_cast<MDSpanType>(nty)) {
    if (mty->Dims() <= 1) {
      if (mty->HasSufficientInfo()) {
        IndStream() << "int " << sym << " = "
                    << ValueSTR(mty->GetShape().ValueAt(0)) << ";\n";
      }
      return true;
    }

    if (!enable_debug_rtti) return true;

    IndStream() << "choreo::rtti::mdspan<" << mty->Dims() << "> " << sym
                << " = ";
    mty->GetShape().PrintAsList(Stream());
    Stream() << ";\n";
    return true;
  }

  return true;
}

bool CuteCodeGen::Visit(AST::Assignment& n) {
  TraceEachVisit(n);

  auto extract_chunk_alias =
      [](const ptr<AST::Node>& node) -> ptr<AST::ChunkAt> {
    if (!node) return nullptr;
    if (auto ca = dyn_cast<AST::ChunkAt>(node)) return ca;
    if (auto expr = dyn_cast<AST::Expr>(node)) {
      if (auto ref = expr->GetReference()) return dyn_cast<AST::ChunkAt>(ref);
    }
    return nullptr;
  };

  auto nty = NodeType(n);
  const bool enable_debug_rtti = EnableDebugTypeRTTI();

  // self-updating operation has been generated already
  auto sty = GetSpannedType(nty);
  // if (sty && sty->GetStorage() == Storage::REG && n.HasNote("update"))
  //   return true;

  if (!n.AssignToDataElement()) {
    auto name = n.GetName();
    bool ref = n.HasNote("ref");
    if (!SSTab().IsDeclared(name) && !isa<AST::SpanAs>(n.value))
      updating_cgi.AddSymbolDetail(
          fname, {InScopeName(name), GetSymbolType(name), ref});

    auto scoped_name = InScopeName(name);
    if (auto rhs_ca = extract_chunk_alias(n.value)) {
      live_chunk_aliases[name] = rhs_ca;
      live_chunk_aliases[scoped_name] = rhs_ca;
    } else {
      live_chunk_aliases.erase(name);
      live_chunk_aliases.erase(scoped_name);
    }
  }

  if (auto s = dyn_cast<AST::Select>(n.value)) {
    assert(!IsHost() && "select should be on device side.");
    assert(!s->inDMA);
    size_t val_count = s->expr_list->Count();
    assert(val_count >= 2);
    std::string array_sym = n.GetName() + "_select_array__";
    if (isa<FutureType>(nty)) {
      ds << d_indent << "future * " << array_sym << "[] = {";
      for (size_t i = 0; i < val_count; i++) {
        if (i > 0) ds << ", ";
        ds << "&" << OpExprSTR(s->expr_list->ValueAt(i), "&", false, false);
      }
      ds << "};\n";
      // make symbol a reference
      ds << d_indent << "future & " << n.GetName() << " = *" << array_sym << "["
         << ExprSTR(s->select_factor, false) << "];\n";
    } else if (auto sty = dyn_cast<SpannedType>(nty)) {
      auto bts = NameBaseType(sty->ElementType());
      ds << d_indent << bts << " * " << array_sym << "[] = {";
      for (size_t i = 0; i < val_count; i++) {
        if (i > 0) ds << ", ";
        ds << ExprSTR(s->expr_list->ValueAt(i), false);
      }
      ds << "};\n";
      // make symbol a reference
      ds << d_indent << bts << " & " << n.GetName() << " = *" << array_sym
         << "[" << ExprSTR(s->select_factor, false) << "];\n";
    } else
      choreo_unreachable("select of " + PSTR(NodeType(*s)) +
                         " is yet to implement.");

    return true;
  }

  if (auto sa = dyn_cast<AST::SpanAs>(n.value)) {
    assert(!IsHost() && "span-as should be on device side.");
    bool is_internal_spanned = PrefixedWith(n.GetName(), "anon_") ||
                               PrefixedWith(n.GetName(), "__iv_") ||
                               SuffixedWith(n.GetName(), "__buf__");
    bool use_user_visible_debug_sym = enable_debug_rtti && !is_internal_spanned;
    std::string raw_sym =
        use_user_visible_debug_sym ? "__raw_" + n.GetName() : n.GetName();
    ds << d_indent << "auto * " << raw_sym << " = ";
    auto tty = GetSymbolType(sa->id->name);
    auto sty = dyn_cast<SpannedType>(nty);
    ds << "static_cast<"
       << NameBaseType(dyn_cast<SpannedType>(nty)->ElementType()) << "*>(";
    if (isa<FutureType>(tty))
      ds << sa->id->name << ".data());\n";
    else
      ds << sa->id->name << ");\n";
    if (use_user_visible_debug_sym && sty) {
      std::vector<std::string> shape_exprs;
      shape_exprs.reserve(sty->Dims());
      for (auto& vi : sty->GetShape().Value())
        shape_exprs.push_back(UnScopedExpr(ValueSTR(vi)));

      std::vector<std::string> stride_exprs;
      assert(sty->GetStrides().size() == sty->Dims() &&
             "missing spanned strides");
      stride_exprs.reserve(sty->Dims());
      for (auto& vi : sty->GetStrides())
        stride_exprs.push_back(UnScopedExpr(ValueSTR(vi)));

      EmitDebugSpannedRTTI(ds, d_indent, n.GetName(), sty, raw_sym, shape_exprs,
                           stride_exprs);
    }
    ssm.MapDeviceSymbol(InScopeName(n.GetName()), raw_sym);
    return true;
  }

  if (n.AssignToDataElement()) {
    if (!IsHost())
      ds << d_indent << ExprSTR(n.da, false) << " = " << ExprSTR(n.value, false)
         << ";\n";

    if (IsHost()) {
      // TODO: test the case!
      choreo_unreachable(
          "error: assignment to data element should be on device side.");
    }

    return true;
  }

  if (isa<BoundedType>(nty) || isa<SpannedType>(nty) || isa<FutureType>(nty)) {
    assert(!IsHost() && "bounded/spanned/future should be on device side.");
    if (auto bit = dyn_cast<BoundedITupleType>(nty);
        enable_debug_rtti && bit && n.IsDecl() && bit->Dims() > 1) {
      ds << d_indent << "choreo::rtti::bounded_ituple<" << bit->Dims() << "> "
         << n.GetName() << " = " << ExprSTR(n.value, false) << ";\n";
      return true;
    }
    if (isa<SpannedType>(nty)) {
      ssm.MapDeviceSymbolIfNotExist(InScopeName(n.GetName()), n.GetName());
    }

    if (!(sty && sty->GetStorage() == Storage::REG && n.HasNote("update")))
      ds << d_indent
         << ((!n.IsDecl() || (IsMutable(*nty) && !isa<SpannedType>(nty)))
                 ? ""
                 : "auto ")
         << n.GetName() << " = ";
    else
      ds << d_indent;
    ds << ExprSTR(n.value, false) << ";\n";
    return true;
  }

  if (isa<ScalarType>(nty)) {
    if (IsHost())
      hs << h_indent << ((!n.IsDecl()) ? "" : "auto ") << n.GetName() << " = "
         << ExprSTR(n.value, true) << ";\n";
    else
      ds << d_indent << ((!n.IsDecl()) ? "" : "auto ") << n.GetName() << " = "
         << ExprSTR(n.value, false) << ";\n";
    return true;
  }

  errs() << "Assignment " << STR(n) << " unprocessed, not supported "
         << PSTR(nty) << "\n";
  return false;
}

bool CuteCodeGen::Visit(AST::ParallelBy& n) {
  TraceEachVisit(n);

  auto& lcs = cgi.GetFunctionLaunches(fname);
  if (parallel_idx < 0) parallel_idx = 0;
  if (lcs.size() <= static_cast<size_t>(parallel_idx)) {
    lcs.resize(static_cast<size_t>(parallel_idx + 1));
  }
  auto& lconfig = lcs[parallel_idx];
  // Ensure launch config is up-to-date even when earlier passes skipped it.
  switch (n.GetLevel()) {
  case ParallelLevel::CLUSTER: lconfig.SetClusterCount(n.BoundValues()); break;
  case ParallelLevel::BLOCK: lconfig.SetBlockCount(n.BoundValues()); break;
  case ParallelLevel::GROUP: lconfig.SetGroupCount(n.BoundValues()); break;
  case ParallelLevel::GROUPx4: lconfig.SetGroupx4Count(n.BoundValues()); break;
  case ParallelLevel::THREAD: lconfig.SetThreadCount(n.BoundValues()); break;
  default: break;
  }

  // add the device name map
  std::string dname[] = {"x", "y", "z"};
  switch (n.GetLevel()) {
  case ParallelLevel::CLUSTER: {
    std::string crank[] = {
        "choreo::tma_cluster_rank()",
        "(choreo::tma_cluster_rank() / " +
            ValueSTR(n.BoundValues().size() > 0 ? n.BoundValues()[0]
                                                : sbe::nu(1)) +
            ")",
        "(choreo::tma_cluster_rank() / (" +
            ValueSTR(n.BoundValues().size() > 0 ? n.BoundValues()[0]
                                                : sbe::nu(1)) +
            " * " +
            ValueSTR(n.BoundValues().size() > 1 ? n.BoundValues()[1]
                                                : sbe::nu(1)) +
            "))"};
    if (n.AllSubPVs().size() == 1)
      ssm.MapDeviceSymbol(InScopeName(n.BPV()->name), crank[0]);
    for (size_t i = 0; i < n.AllSubPVs().size(); ++i)
      ssm.MapDeviceSymbol(InScopeName(n.GetSubPV(i)->name), crank[i]);
  } break;
  case ParallelLevel::BLOCK:
    for (size_t i = 0; i < n.AllSubPVs().size(); ++i)
      ssm.MapDeviceSymbol(InScopeName(n.GetSubPV(i)->name),
                          "blockIdx." + dname[i]);
    if (n.AllSubPVs().size() == 1)
      ssm.MapDeviceSymbol(InScopeName(n.BPV()->name), "blockIdx.x");
    break;
  case ParallelLevel::GROUPx4: {
    if (n.AllSubPVs().size() == 1)
      ssm.MapDeviceSymbol(InScopeName(n.BPV()->name), vid_pfx + "g4id_x");
    if (n.AllSubPVs().size() > 0)
      ssm.MapDeviceSymbol(InScopeName(n.GetSubPV(0)->name), vid_pfx + "g4id_x");
    if (n.AllSubPVs().size() > 1)
      ssm.MapDeviceSymbol(InScopeName(n.GetSubPV(1)->name), vid_pfx + "g4id_y");
    if (n.AllSubPVs().size() > 2)
      ssm.MapDeviceSymbol(InScopeName(n.GetSubPV(2)->name), vid_pfx + "g4id_z");
  } break;
  case ParallelLevel::GROUP: {
    if (n.AllSubPVs().size() == 1)
      ssm.MapDeviceSymbol(InScopeName(n.BPV()->name), vid_pfx + "gid_x");

    if (n.AllSubPVs().size() > 0)
      ssm.MapDeviceSymbol(InScopeName(n.GetSubPV(0)->name), vid_pfx + "gid_x");
    if (n.AllSubPVs().size() > 1)
      ssm.MapDeviceSymbol(InScopeName(n.GetSubPV(1)->name), vid_pfx + "gid_y");
    if (n.AllSubPVs().size() > 2)
      ssm.MapDeviceSymbol(InScopeName(n.GetSubPV(2)->name), vid_pfx + "gid_z");
  } break;
  case ParallelLevel::THREAD: {
    if (n.AllSubPVs().size() == 1)
      ssm.MapDeviceSymbol(InScopeName(n.BPV()->name), vid_pfx + "tid_x");

    if (n.AllSubPVs().size() > 0)
      ssm.MapDeviceSymbol(InScopeName(n.GetSubPV(0)->name), vid_pfx + "tid_x");
    if (n.AllSubPVs().size() > 1)
      ssm.MapDeviceSymbol(InScopeName(n.GetSubPV(1)->name), vid_pfx + "tid_y");
    if (n.AllSubPVs().size() > 2)
      ssm.MapDeviceSymbol(InScopeName(n.GetSubPV(2)->name), vid_pfx + "tid_z");
  } break;
  default:
    choreo_unreachable("unsupported parallel-by level: " + STR(n.GetLevel()) +
                       ".");
  }

  // When CLUSTER is the outer parallel, defer launch codegen to the inner
  // BLOCK. CLUSTER sets cluster dims and maps cluster rank variables; the
  // actual kernel launch / device-function declaration is emitted by the BLOCK
  // visit.
  if (n.IsOuter() && n.GetLevel() == ParallelLevel::CLUSTER) {
    cluster_defers_launch = true;
    deferred_cluster_pb = &n;
    auto cluster_scope = SSTab().ScopeName();
    auto mri = FCtx(fname).GetStaticMemReuseInfo(cluster_scope);
    if (mri && mri->infos.count(Storage::SHARED) &&
        mri->infos.at(Storage::SHARED).spm_size > 48 * 1024)
      set_cuda_func_attribute_max_dynamic_shared_memory_size = true;
    ds << d_indent << "// cluster parallel-by: " << n.LOC() << "\n";
    return true;
  }

  bool emit_launch =
      n.IsOuter() || (!n.IsOuter() && n.GetLevel() == ParallelLevel::BLOCK &&
                      cluster_defers_launch);

  // only do the whole codegen when accessing the outer parallel-by
  if (emit_launch) {
    tma_future_count = 0;

    auto required_shared_align = [&]() -> size_t {
      size_t alignment = CCtx().GetMemoryAlignmentByte(Storage::SHARED);
      auto collect = [&](auto&& self, const ptr<AST::Node>& node) -> void {
        if (!node) return;
        if (auto mn = dyn_cast<AST::MultiNodes>(node)) {
          for (auto& item : mn->values) self(self, item);
          return;
        }
        if (auto mma = dyn_cast<AST::MMA>(node)) {
          auto op = mma->GetOperation();
          if (!op || op->Tag() != AST::MMAOperation::Load) return;
          alignment = std::max(
              alignment, SharedAlignmentBytes(CCtx(), op->GetSwizzleMode()));
          return;
        }
        if (auto block = dyn_cast<AST::Block>(node)) {
          self(self, block->GetBody());
          return;
        }
        if (auto if_else = dyn_cast<AST::IfElseBlock>(node)) {
          self(self, if_else->GetThenBody());
          self(self, if_else->GetElseBody());
          return;
        }
      };
      collect(collect, n.GetBody());
      return alignment;
    }();

    ValueItem cur_spm_size = sbe::nu(0);
    ValueItem ring_start = sbe::nu(0);
    ValueItem ring_size = sbe::nu(0);

    EmitMemReuse(SSTab().ScopeName());

    EmitTMAConfiguration(&n);

    hs << h_indent << "dim3 __" << fname << "_gdims" << parallel_idx << "("
       << ValueSTR(lconfig.block_count.x) << ", "
       << ValueSTR(lconfig.block_count.y) << ", "
       << ValueSTR(lconfig.block_count.z) << ");\n";
    // GPU groups are virtual
    // we binds all choreo threads to blockDim.x, and all choreo groups to
    // blockDim.y this aligns choreo row-major convention to cuda's col-major
    // oriented convention. users can still keep binding left-most parallel
    // variable to left-most tensor dim, and right to right. without mindset to
    // CUDA's thread majority that left-most are leading dim (thread x)
    auto inner_thr_count = lconfig.thread_count.x * lconfig.thread_count.y *
                           lconfig.thread_count.z;
    auto group_count = lconfig.group_count.x * lconfig.group4_count.x *
                       lconfig.group_count.y * lconfig.group_count.z;
    auto thr_count = inner_thr_count * group_count;

    hs << h_indent << "dim3 __" << fname << "_bdims" << parallel_idx << "("
       << ValueSTR(thr_count) << ", 1, 1"
       << ");\n";

    // plan the shared memory that is decided at runtime
    if (cgi.HasAsyncDMA(fname)) {
      ring_size = group_count * sbe::nu(8);
      // add the size of the future ring (see choreo.h)
      cur_spm_size += ring_size;
    }

    /*
    | static shared | dynamic shared |
    ^                 ^
    shared_base       shared_base + static_size
    must satisfy the constraints:
    - static shared <= 48KB
    - dynamic shared <= MaxDynamicSharedMemorySize
    - the sum <= the capacity of arch
    */
    auto EmitCudaFuncAttributeMaxDynamicSharedMemorySize = [&]() -> void {
      hs << h_indent << "cudaFuncSetAttribute(" << device_fn
         << ", cudaFuncAttributeMaxDynamicSharedMemorySize, "
         << ValueSTR(cur_spm_size) << " + (" << required_shared_align
         << " - 1));\n";
      set_cuda_func_attribute_max_dynamic_shared_memory_size = true;
    };

    if (auto dev_name = SSTab().ScopeName();
        FCtx(fname).HaveDynamicBuffer(dev_name, Storage::SHARED)) {
      // add the size of dynamic shared
      auto mri = FCtx(fname).GetDynMemReuseInfo(dev_name);
      assert(mri);
      auto code_spm_end =
          sbe::sym(mri->infos[Storage::SHARED].spm_size)->Normalize();
      cur_spm_size += code_spm_end;
      ring_start = code_spm_end;

      EmitCudaFuncAttributeMaxDynamicSharedMemorySize();
      Note(n.LOC(),
           "In the current kernel `" + device_fn +
               "`, cudaFuncAttributeMaxDynamicSharedMemorySize is set, cause "
               "shared memory usage has exceeded the default limit 48KB.");
    } else {
      // add the size of static shared
      auto mri = FCtx(fname).GetStaticMemReuseInfo(dev_name);
      if (!mri && deferred_cluster_pb) {
        auto parent =
            dev_name.substr(0, dev_name.rfind("::", dev_name.size() - 3) + 2);
        mri = FCtx(fname).GetStaticMemReuseInfo(parent);
      }
      if (mri) {
        // 48KB is the largest capacity that static shared memory supports.
        if (mri->infos[Storage::SHARED].spm_size > 48 * 1024) {
          auto code_spm_end = sbe::nu(mri->infos[Storage::SHARED].spm_size);
          cur_spm_size += code_spm_end;
          ring_start = code_spm_end;
          EmitCudaFuncAttributeMaxDynamicSharedMemorySize();
          Note(
              n.LOC(),
              "In the current kernel `" + device_fn +
                  "`, cudaFuncAttributeMaxDynamicSharedMemorySize is set to `" +
                  ValueSTR(cur_spm_size) + "` bytes, " +
                  "cause shared memory usage" +
                  " has exceeded the default limit 48KB.");
        }
      }
    }

    hs << h_indent << device_fn << "<<<__" << fname << "_gdims" << parallel_idx
       << ", __" << fname << "_bdims" << parallel_idx;

    bool explicit_smem = false;
    if (!sbe::ceq(cur_spm_size, sbe::nu(0))) {
      // TODO: conservative padding. To be optimized.
      hs << ", " << ValueSTR(cur_spm_size) << " + (" << required_shared_align
         << " - 1)";
      explicit_smem = true;
    }
    if (stream_name != "") {
      if (!explicit_smem) hs << ", 0";
      hs << ", " << stream_name;
    }
    hs << ">>>(";

    size_t i = 0;
    for (auto& item : GetDeviceFuncIns(updating_cgi)) {
      auto sname = item.name;
      if (isa<SpannedType>(item.type)) sname += "__device";
      if (!PrefixedWith(scoped_symtab.ScopeName(), GetScope(sname))) continue;
      hs << ((i++ == 0) ? "" : ", ");
      if (ssm.HasHostName(sname))
        hs << ssm.HostName(sname);
      else
        hs << UnScopedName(ssm.DeviceName(sname));
    }
    for (auto item : symbolic_dimensions) {
      hs << ((i++ > 0) ? ", " : "");
      hs << UnScopedName(item.first);
    }
    if (const auto& mri = FCtx(fname).GetDynMemReuseInfo(SSTab().ScopeName()))
      for (const auto& [sto, ie] : mri->infos)
        for (size_t idx = 0; idx < ie.offset_args.size(); ++idx)
          hs << ((i++ > 0) ? ", " : "") << ie.offsets_name << "[" << idx << "]";

    // tma configurations
    for (auto desc : cgi.GetTMADesc(&n))
      hs << ", " << desc.GetName() + "_tensor_map";

    if (!ring_start->IsNumeric()) hs << ", " << ValueSTR(ring_start);

    hs << ");\n";

    if (!n.IsAsync()) {
      if (stream_name != "")
        hs << h_indent << "choreo::abend_true(cudaStreamSynchronize("
           << stream_name << "));\n";
      else
        hs << h_indent << "choreo::abend_true(cudaDeviceSynchronize());\n";
    }

    // copy the span passed by ref back to host
    for (const auto& item : GetChoreoFuncIns(updating_cgi)) {
      if (isa<SpannedType>(item.type)) {
        auto oname = UnScopedName(item.name);
        if (item.attr != ParamAttr::GLOBAL_INPUT && item.IsReference())
          hs << h_indent << "choreo::abend_true(cudaMemcpy(" << oname
             << ".data(), " << oname + "__device" << ", "
             << UnScopedSizeExpr(*item.type) << ", cudaMemcpyDeviceToHost));\n";
      }
    }

    // handle device function
    EmitDeviceFuncDecl(ds, &n, ring_start);
    ds << " {\n";
    IncrDeviceIndent();
    if (!(sbe::ceq(cur_spm_size, sbe::nu(0)) &&
          sbe::ceq(ring_start, sbe::nu(0)))) {
      ds << d_indent << "extern __shared__ char " << device_fn
         << "__runtime_shared_buffer__raw[];\n";
      // NOTE: If extern shared mem is enabled, then its address will be
      // immediately followed by the preceding static shared area.
      // Therefore, we need to manually perform the alignment.
      ds << d_indent << "auto " << device_fn
         << "__runtime_shared_buffer__ = "
            "reinterpret_cast<char*>(aligned_up_ptr<"
         << required_shared_align << " * 8>(" << device_fn
         << "__runtime_shared_buffer__raw));\n";
      if (!sbe::ceq(cur_spm_size, sbe::nu(0)) && cgi.HasAsyncDMA(fname)) {
        ds << d_indent << "auto " << device_fn
           << "__ring__ = reinterpret_cast<choreo::future_ring<6>*>("
           << device_fn << "__runtime_shared_buffer__ + " + ValueSTR(ring_start)
           << ");\n";
        ds << d_indent << "if (threadIdx.x <= " << ValueSTR(group_count)
           << " && threadIdx.y == 0 && threadIdx.z == 0)";
        ds << d_indent << "  " << device_fn
           << "__ring__[threadIdx.x].init();\n";
        ds << d_indent << "__syncthreads();  // must sync\n";
      }

    } else
      ds << d_indent << "auto " << device_fn << "__ring__ = nullptr;\n";
    if (EnableDebugTypeRTTI()) {
      if (EnableLineDirective()) {
        auto loc = n.LOC();
        auto file = ResolveLineDirectivePath(loc);
        if (!file.empty() && loc.begin.line > 0)
          ds << d_indent << "#line " << loc.begin.line << " \"" << file
             << "\"\n";
      }
      ds << d_indent
         << "if (__CHOREO_BLOCK_SINGLE__) __choreo_cuda_debug_point__();\n";
    }
    if (lconfig.HasCluster()) {
      ds << d_indent << "choreo::tma_cluster_sync();\n";
    }
    if (!deferred_spm_decls.empty()) {
      ds << deferred_spm_decls;
      deferred_spm_decls.clear();
    }
    ds << d_indent << "{ // parallel-by: " << n.LOC() << "\n";
  } else {
    auto& siblings = cgi.GetPBTree(fname).GetSiblings(&n);
    if (!siblings.empty()) {
      ds << d_indent << "{ // inner parallel-by: " << n.LOC() << "\n";
      IncrDeviceIndent();
    }
  }

  if (n.GetLevel() == ParallelLevel::BLOCK &&
      (cgi.HasTMA() || IsWarpSpecActive())) {
    ds << d_indent
       << "auto wg_barrier = "
          "cooperative_groups::tiled_partition<128>(cooperative_groups::this_"
          "thread_block());\n";
  }
  auto& tma_descs = cgi.GetTMADescs()[&n];
  auto* cluster_pb = deferred_cluster_pb;
  if (tma_descs.empty() && cluster_pb) {
    auto& cluster_tma = cgi.GetTMADescs()[cluster_pb];
    if (!cluster_tma.empty()) { tma_descs = cluster_tma; }
  }
  if (!tma_descs.empty()) {
    assert(n.GetLevel() == ParallelLevel::BLOCK);
    int emitted_tma_init_idx = 0;
    for (TMADesc& desc : tma_descs) {
      auto cp_atom = GetCopyAtomName(true, emitted_tma_init_idx);
      auto tma_barrier_name = cp_atom + "_barrier";
      auto f_sty = GetSpannedType(desc.GetFrom()->GetType());
      auto t_sty = GetSpannedType(desc.GetTo()->GetType());
      auto io_sty = (t_sty->GetStorage() == Storage::SHARED) ? t_sty : f_sty;
      bool rank2_tma = io_sty->GetShape().Rank() == 2;
      bool has_cluster = !cgi.GetFunctionLaunches(fname).empty() &&
                         cgi.GetFunctionLaunches(fname)[0].HasCluster();
      bool use_ptx_barrier_for_desc =
          (tma_cluster_aware || ptx_barrier || has_cluster) && rank2_tma;
      auto in_thr_block = desc.GetInThreadsBlock();
      auto inner_pb_level = desc.GetPBLevel();

      bool skip_load_tma_init_block = IsWarpSpecActive() && desc.IsLoad() &&
                                      in_thr_block &&
                                      inner_pb_level == ParallelLevel::GROUPx4;

      if (skip_load_tma_init_block) { continue; }

      emitted_tma_init_idx++;

      if (use_ptx_barrier_for_desc) {
        ds << d_indent << "__shared__ __align__(8) uint64_t "
           << tma_barrier_name << ";\n";
      } else {
        ds << d_indent << "__shared__ cuda::barrier<cuda::thread_scope_block> "
           << tma_barrier_name << ";\n";
      }
      ds << d_indent << "if (__CHOREO_BLOCK_SINGLE__) {\n";
      // if in_thr_block is specified, the number of threads is compitable with
      // inner parallel-by the barrier. otherwise, all threads in the CTA will
      // wait.

      std::string threads_waited;
      if (!in_thr_block) {
        threads_waited = "blockDim.x";
      } else if (inner_pb_level == ParallelLevel::GROUP) {
        threads_waited = IsWarpSpecActive() ? "1" : "32";
      } else if (inner_pb_level == ParallelLevel::GROUPx4) {
        threads_waited = IsWarpSpecActive() ? "1" : "128";
      }

      if (use_ptx_barrier_for_desc) {
        ds << d_indent << "  choreo::tma_mbarrier_init(&" << tma_barrier_name
           << ", 1);\n";
      } else {
        ds << d_indent << "  init(&" << tma_barrier_name << ", "
           << threads_waited << ");\n";
        ds << d_indent << "  cde::fence_proxy_async_shared_cta();\n";
      }
      ds << d_indent << "}\n";
      ds << d_indent << "__syncthreads();\n";
      if (use_ptx_barrier_for_desc) {
        ds << d_indent << "TMAAtom " << cp_atom << "{};\n";
        ds << d_indent << cp_atom << ".EnablePTXMBarrier(&" << tma_barrier_name
           << ");\n\n";
      } else {
        ds << d_indent << "TMAAtom " << cp_atom << "{&" << cp_atom
           << "_barrier};\n\n";
      }
    }
  }
  EmitDeviceVirtualIndices(&n);

  return true;
}

bool CuteCodeGen::Visit(AST::DMA& n) {
  // Currently, DMA in host-side:
  // - will not generate any future.
  // - are performed directly by manipulating pointers.
  // - not support tiling.
  // - not support async.

  // Generate DMA transfer and choreo::future in device-side
  auto claimFuture = [this,
                      &n](const std::string& buf_expr, bool is_async,
                          bool is_tma = false,
                          const std::string& mdata_expr = "") -> std::string {
    if (!n.future.empty() && claimed_futs.count(InScopeName(n.future)))
      return n.future;

    auto future_name = n.future;

    auto cp_atom =
        GetCopyAtomName(is_tma, (is_tma ? tma_future_count : dma_count_));
    if (!is_tma && is_async) {
      ds << d_indent << "AsyncCopyAtom " << cp_atom << "{};\n";
    }

    if (future_name.empty()) {
      future_name = "__choreo_anon_fut__" + std::to_string(future_count_);
    } else {
      claimed_futs.emplace(InScopeName(n.future), cp_atom);
      auto fsty = GetSpannedType(GetSymbolType(n.future));
      ssm.MapDeviceSymbol(InScopeName(n.future), n.future);
      ssm.MapDeviceSymbol(InScopeName(n.future) + ".data",
                          n.future + ".data()");
      if (n.IsSparse())
        ssm.MapDeviceSymbol(InScopeName(n.future) + ".mdata",
                            n.future + ".mdata()");
    }
    future_count_++;
    if (is_tma)
      ++tma_future_count;
    else
      ++dma_count_;

    ds << d_indent << "future " << future_name;
    ds << "(\"" << n.future << "\", " << n.LOC().begin.line << ", "
       << n.LOC().begin.column;
    if (!buf_expr.empty()) { ds << ", " << buf_expr; }
    if (!mdata_expr.empty()) { ds << ", " << mdata_expr; }
    ds << ");\n";
    if (is_tma) {
      ds << d_indent << future_name << ".is_tma = true;\n";
      ds << d_indent << future_name << ".set_atom(&" << cp_atom << ");\n";
    } else if (is_async) {
      ds << d_indent << future_name << ".set_atom(&" << cp_atom << ");\n";
      ds << d_indent << future_name << ".set_ring(" << device_fn
         << "__ring__);\n";
      ds << d_indent << future_name << ".id = " << future_count_ << ";\n";
    }

    return future_name;
  };

  auto nty = NodeType(n);
  if (auto ph = dyn_cast<PlaceHolderType>(nty)) {
    assert(ph->GetBaseType() == BaseType::FUTURE);
    // must set the buffer
    auto buf_name = FBInfo().at(InScopeName(n.future)).buffer;

    // Handle placeholder checks that need to postpone after all lv processed
    // Currently, the only case is the plder tied to global buffer

    // assert(ssm.HasDeviceName(buf_name) && "buffer has been defined");
    if (!ssm.HasDeviceName(buf_name)) pld_checklist.push_back(buf_name);

    // dma.any in host-side is of no practical use.
    // It should not be claimed. And there is no future to remap to.
    if (IsHost()) return true;

    claimFuture(UnScopedName(buf_name), true, n.IsTMA());
    // make following buffer reference all be indirect
    // TODO: any better idea than this
    auto fsty = GetSpannedType(GetSymbolType(n.future));
    ssm.RemapDeviceSymbol(buf_name, n.future + ".data()");
    return true;
  }

  assert(isa<AST::ChunkAt>(n.from) && "Unexpected type for DMA's source.");
  assert(isa<AST::ChunkAt>(n.to) && "Unexpected type for DMA's destination.");

  auto fty = dyn_cast<FutureType>(nty);
  assert(fty && "Invalid type of DMA statement!");
  if (fty->IsAsync()) assert(!n.future.empty() || n.HasEvent());

  auto f_ca = cast<AST::ChunkAt>(n.from);
  auto t_ca = cast<AST::ChunkAt>(n.to);
  auto f_sym = f_ca->data->name;
  auto t_sym = t_ca->data->name;
  auto f_idx = f_ca->indices;
  auto t_idx = t_ca->indices;
  auto f_ty = GetSymbolType(f_sym);
  auto t_ty = GetSymbolType(t_sym);
  // the spanned type of sym in from chunkat
  auto f_sty = GetSpannedType(f_ty);
  // the spanned type of sym in to chunkat
  auto t_sty = GetSpannedType(t_ty);

  assert(f_sty && "can not retrieve data from 'from'.");
  assert(t_sty && "can not retrieve data from 'to'.");

  auto SymbolToSymbol = [f_ca, t_ca]() -> bool {
    return f_ca->NoTilingOperation() && t_ca->NoTilingOperation();
  };
  auto SymbolToTile = [f_ca, t_ca]() -> bool {
    return f_ca->NoTilingOperation() && t_ca->HasTilingOperation();
  };
  auto TileToSymbol = [f_ca, t_ca]() -> bool {
    return f_ca->HasTilingOperation() && t_ca->NoTilingOperation();
  };
  auto TileToTile = [f_ca, t_ca]() -> bool {
    return f_ca->HasTilingOperation() && t_ca->HasTilingOperation();
  };
  auto HasReshape = [f_ca, t_ca]() -> bool {
    return f_ca->HasReshape() || t_ca->HasReshape();
  };

  if (t_sty->GetStorage() == Storage::GLOBAL && use_hetero_tileflow &&
      IsHost()) {
    std::string bts = NameBaseType(t_sty->ElementType());
    auto buf_sym = t_sym + "__device";
    auto buf_sym_from = f_sym + "__device";
    static int s_cnt = 0;
    if (n.operation == ".copy") {
      if (SymbolToSymbol()) {
        // direct copy
        hs << h_indent << bts << " * " << buf_sym << " = " << buf_sym_from
           << ";\n";
      } else if (SymbolToTile()) {
        auto off_name = "__slice_offset" + std::to_string(s_cnt++) + "__" +
                        f_sym + "_2_" + t_sym;
        auto [offset, offcnt] = GenMdsOffset(t_ca);
        hs << h_indent << "int " << off_name << " = " << offset << ";\n";
        hs << h_indent << bts << " * " << buf_sym << " + " << off_name << " = "
           << buf_sym_from << ";\n";
      } else if (TileToSymbol()) {
        auto off_name = "__slice_offset" + std::to_string(s_cnt++) + "__" +
                        f_sym + "_2_" + t_sym;
        auto [offset, offcnt] = GenMdsOffset(f_ca);
        hs << h_indent << "int " << off_name << " = " << offset << ";\n";
        hs << h_indent << bts << " * " << buf_sym << " = " << buf_sym_from
           << " + " << off_name << ""
           << ";\n";
      } else
        choreo_unreachable("not support dual-side chunkat in dma copy");
    } else
      choreo_unreachable("not support host-side dma other than copy");

    return true;
  }

  // TODO: how to do tiling in host-side?
  if ((t_sty->GetStorage() == Storage::GLOBAL ||
       IsChoreoInput(InScopeName(t_sym))) &&
      IsHost()) {
    if (n.IsAsync()) choreo_unreachable("not support host-side async dma yet");
    std::string bts = NameBaseType(t_sty->ElementType());
    std::string buf_sym_from;
    std::string buf_sym;
    std::string cuda_dma_kind = "cudaMemcpy";
    if (global_buffers.count(f_sym + "__device")) {
      buf_sym_from = f_sym + "__device";
      cuda_dma_kind.append("Device");
    } else {
      buf_sym_from = f_sym + ".data()";
      if (f_sty->GetStorage() == Storage::GLOBAL)
        cuda_dma_kind.append("Device");
      else
        cuda_dma_kind.append("Host");
    }

    if (global_buffers.count(t_sym + "__device")) {
      buf_sym = t_sym + "__device";
      cuda_dma_kind.append("ToDevice");
    } else {
      buf_sym = t_sym + ".data()";
      if (f_sty->GetStorage() == Storage::GLOBAL)
        cuda_dma_kind.append("ToDevice");
      else
        cuda_dma_kind.append("ToHost");
    }

    if (n.operation == ".copy") {
      if (SymbolToSymbol()) {
        // direct copy
        hs << h_indent << "choreo::abend_true(cudaMemcpy(" << buf_sym << ", "
           << buf_sym_from << ", " << UnScopedSizeExpr(*f_sty) << ", "
           << cuda_dma_kind << "));\n";
      } else
        choreo_unreachable(
            "not support tiling chunkat in dma copy at host side for now");
    } else
      choreo_unreachable("not support host-side dma other than copy");

    return true;
  }

  // TODO: correct?
  if (IsHost()) choreo_unreachable("the dma is not supported in host side!");

  auto GetBufferExpr = [this](const std::string& sym,
                              const ptr<AST::MultiValues> subscription,
                              const ptr<Type>& sym_ty) {
    std::string buf_expr = "";
    std::string sname = InScopeName(sym);
    if (isa<FutureType>(sym_ty) && !IsHostSymbol(sname)) {
      std::string buf_name = sname + ".data";
      buf_expr = ssm.DeviceName(buf_name);
    } else if (isa<FutureType>(sym_ty)
               // This only matches the host-side buffer that is defined in
               // choreo DMA and tied to future but host-side data copy does
               // not really do device-level DMA, and the future is basically
               // a phantom handle do not emit any concrete code at host-side.
               && IsHostSymbol(sname) && !IsChoreoInput(sname) &&
               !IsChoreoOutput(sname)) {
      buf_expr =
          UnScopedName(const_cast<FutureBufferInfo&>(FBInfo())[sname].buffer);
    } else
      buf_expr = ssm.DeviceName(sname);

    std::string buf_name = buf_expr;
    if (subscription != nullptr) {
      if (auto array_ty = dyn_cast<ArrayType>(sym_ty);
          array_ty && CCtx().MemReuse()) {
        // Suppose we declared `shared s32[3,4] i[2]`
        // For `i[1]`, if memory reuse is enabled, we need to generate pointer
        // expr `i + 1 * (3*4)` rather than array subscript expr `i[1]`.
        // Because if memory reuse is enabled, `i` is declared as point not
        // array!
        std::string array_idx = "";
        auto subscriptions = subscription->AllValues();
        const ValueList& array_sizes = array_ty->Dimensions();
        for (size_t i = 0; i < subscriptions.size(); ++i) {
          if (array_idx.empty())
            array_idx = ExprSTR(subscriptions[i], IsHost());
          else
            array_idx = "(" + array_idx + ")*" + ValueSTR(array_sizes[i]) +
                        "+" + ExprSTR(subscriptions[i], IsHost());
        }
        std::string elem_count =
            ValueSTR(cast<SpannedType>(sym_ty)->GetShape().ElementCountValue());
        buf_expr += " + (" + array_idx + ")*(" + elem_count + ")";
      } else {
        for (auto expr : subscription->AllValues())
          buf_expr += "[" + ExprSTR(expr, IsHost()) + "]";
      }
    }
    return std::make_pair(buf_name, buf_expr);
  };

  // if the value is dst, means MAY do optimization on dst
  enum DMA_OP : uint8_t {
    none = 0,
    dst = 1 << 0,
    src = 1 << 1,
    both = src | dst
  };

  const auto f_buf = GetBufferExpr(f_sym, f_idx, f_ty);
  const auto t_buf = GetBufferExpr(t_sym, t_idx, t_ty);

  std::string mdata_expr = "";
  if (n.IsSparse() && !n.future.empty()) {
    static size_t mdata_count = 0;
    auto mdata_sym = "__choreo_mdata__" + std::to_string(mdata_count++);
    auto m = ValueSTR(f_sty->GetShape().ValueAt(0));
    auto nval = ValueSTR(f_sty->GetShape().ValueAt(1));
    auto k = ValueSTR(f_sty->GetShape().ValueAt(2));
    ds << d_indent << "constexpr int " << mdata_sym << "_M = " << m << ";\n";
    ds << d_indent << "constexpr int " << mdata_sym << "_N = " << nval << ";\n";
    ds << d_indent << "constexpr int " << mdata_sym << "_K = " << k << ";\n";
    if (t_sty->GetStorage() == Storage::SHARED)
      ds << d_indent << "__shared__ uint8_t " << mdata_sym << "[" << mdata_sym
         << "_M * " << mdata_sym << "_N * (" << mdata_sym << "_K / 4)];\n";
    else
      ds << d_indent << "uint8_t " << mdata_sym << "[" << mdata_sym << "_M * "
         << mdata_sym << "_N * (" << mdata_sym << "_K / 4)];\n";
    mdata_expr = mdata_sym;
  }

  auto future_name = n.future;
  bool bind_data = SymbolToSymbol() || TileToSymbol() || TileToTile();
  std::string bound_mdata_expr = (n.IsSparse() && bind_data) ? mdata_expr : "";
  bool use_tma = n.IsTMA();
  bool suppress_tma_future = false;
  bool suppress_dma_future = false;

  if (use_tma && IsWarpSpecActive() && bdim_level == ParallelLevel::GROUPx4 &&
      n.IsAsync()) {
    suppress_tma_future = true;
  }
  if (!use_tma && IsWarpSpecActive() && bdim_level == ParallelLevel::GROUPx4 &&
      n.IsAsync()) {
    suppress_dma_future = true;
  }
  // bind the data to the future
  if (!suppress_tma_future && !suppress_dma_future) {
    if (bind_data)
      future_name =
          claimFuture(t_buf.second, fty->IsAsync(), use_tma, bound_mdata_expr);
    else
      future_name = claimFuture("", fty->IsAsync(), use_tma, "");
  }

  // vars to do make_tiled_copy
  struct TiledCopyEntry {
    std::pair<size_t, size_t> thr_layout;
    std::pair<size_t, size_t> val_layout;
    // if set, use AutoVectorizingCopyWithAssumedAlignment
    size_t vectorized_align_bits;
    bool one_row_per_warp;
  };
  enum class TailCopyKind : uint8_t {
    None,
    G2S,
    S2G,
  };
  struct TiledCopyPlan {
    std::optional<TiledCopyEntry> entry;
    TailCopyKind tail_copy_kind = TailCopyKind::None;
    std::optional<TiledCopyEntry> tail_copy_helper_entry;
  };
  TiledCopyPlan tiled_copy_plan;

  auto PrepareTiledCopyEntry =
      [&](const location& loc, bool one_row_per_warp, size_t mem_align_byte,
          const Shape& tile_shape, const ValueList& tile_strd,
          BaseType bty) -> std::optional<TiledCopyEntry> {
    if (tile_shape.Rank() != 2) {
      Note(loc, "the span (maybe after span_as) is not of rank 2.");
      return std::nullopt;
    }
    if (tile_shape.IsDynamic()) {
      Note(loc, "the tile is of dynamic shape (" + STR(tile_shape) +
                    "), unable to do make_tiled_copy.");
      return std::nullopt;
    }

    const auto& lcs = cgi.GetFunctionLaunches(CurrentFunctionName());
    assert(parallel_idx != -1 && parallel_idx < static_cast<int>(lcs.size()));
    const auto& lconfig = lcs[parallel_idx];
    auto inner_thr_count = lconfig.thread_count.x * lconfig.thread_count.y *
                           lconfig.thread_count.z;
    auto group_count = lconfig.group_count.x * lconfig.group4_count.x *
                       lconfig.group_count.y * lconfig.group_count.z;
    auto thr_count_vi =
        IsWarpSpecActive() ? inner_thr_count : inner_thr_count * group_count;
    if (!VIIsInt(inner_thr_count)) {
      Note(loc, "#thread (" + STR(inner_thr_count) +
                    ") is dynamic, unable to do make_tiled_copy.");
      return std::nullopt;
    }
    size_t thr_count = *VIInt(thr_count_vi);

    if (!VIIsInt(tile_strd[1])) {
      Note(loc, "the col stride of span is dynamic (" + STR(tile_strd[1]) +
                    "). make_tiled_copy can be implemented only when it is "
                    "integer 1 (row-major).");
      return std::nullopt;
    }
    if (*VIInt(tile_strd[1]) != 1) {
      Note(loc, "the col stride of span is " + STR(tile_strd[1]) +
                    ". make_tiled_copy can be implemented only when it is "
                    "integer 1 (row-major).");
      return std::nullopt;
    }

    TiledCopyEntry ret;
    size_t tile_m = *VIInt(tile_shape.ValueAt(0));
    size_t tile_n = *VIInt(tile_shape.ValueAt(1));
    size_t elem_byte = SizeOf(bty);
    size_t row_byte = tile_n * elem_byte;

    auto XBitsPerThread = [&](size_t bits_per_thread) -> bool {
      size_t bytes_per_thread = bits_per_thread / 8;
      if (row_byte % bytes_per_thread != 0) {
        Note(loc, "row_byte (" + std::to_string(row_byte) +
                      ") is not divisible by " +
                      std::to_string(bytes_per_thread) + ", unable to do " +
                      std::to_string(bits_per_thread) +
                      " bits vectorized copy.");
        return false;
      }
      size_t thr_per_row = row_byte / bytes_per_thread;
      if (thr_count % thr_per_row != 0) {
        Note(loc, "#thread (" + std::to_string(thr_count) +
                      ") is not divisible by thr_per_row (" +
                      std::to_string(thr_per_row) + "), unable to do " +
                      std::to_string(bits_per_thread) +
                      " bits vectorized copy.");
        return false;
      }
      size_t thr_per_col = thr_count / thr_per_row;
      if (tile_m % thr_per_col != 0) {
        Note(loc, "#row of tile (" + std::to_string(tile_m) +
                      ") is not divisible by thr_per_col (" +
                      std::to_string(thr_per_col) + "), unable to do " +
                      std::to_string(bits_per_thread) +
                      " bits vectorized copy.");
        return false;
      }
      ret.thr_layout.first = thr_per_col;
      ret.thr_layout.second = thr_per_row;
      ret.val_layout.first = tile_m / thr_per_col;
      ret.val_layout.second = tile_n / thr_per_row;
      return true;
    };

    if (!one_row_per_warp && VIIsInt(tile_strd[0])) {
      int tile_strd_0 = *VIInt(tile_strd[0]);
      auto AlignWith = [&](size_t _alignment_bit) -> bool {
        if ((mem_align_byte * 8) % _alignment_bit != 0) return false;
        if ((tile_strd_0 * elem_byte * 8) % _alignment_bit != 0) return false;
        return true;
      };
      for (size_t expect_alignment_bit : {128, 64, 32, 16}) {
        if (expect_alignment_bit < elem_byte * 8) break;
        if (!AlignWith(expect_alignment_bit)) {
          Note(loc,
               "alignment is not satisfied, unable to utilize vectorized " +
                   std::to_string(expect_alignment_bit) +
                   " bit load inst: tensor alignment is " +
                   std::to_string(mem_align_byte) +
                   " bytes, stride of first dim of tile is " +
                   std::to_string(tile_strd_0 * elem_byte) + " bytes.");
        } else {
          if (XBitsPerThread(expect_alignment_bit)) {
            ret.vectorized_align_bits = expect_alignment_bit;
            ret.one_row_per_warp = false;
            Note(loc, "do make_tiled_copy with assumed " +
                          std::to_string(expect_alignment_bit) +
                          " bits vectorized copy atom.");
            return ret;
          }
        }
      }
    }

    Note(loc, "alignment is not satisfied, unable to utilize vectorized "
              "load inst. Fallback to UniversalCopy.");
    constexpr size_t thr_per_warp = 32;
    if (thr_count < 32) {
      Note(loc, "#thread in block (" + std::to_string(thr_count) +
                    ") is less than 32, unable to do warp-per-row "
                    "make_tiled_copy. Fallback to normal copy.");
      return std::nullopt;
    }
    if (tile_n % 32 != 0) {
      Note(loc, "#col of tile is not divisible by 32, unable to do "
                "warp-per-row make_tiled_copy. Fallback to normal copy.");
      return std::nullopt;
    }
    constexpr size_t thr_per_row = thr_per_warp;
    if (thr_count % thr_per_row != 0) {
      Note(loc, "#thread in block (" + std::to_string(thr_count) +
                    ") is not divisible by #thread in a warp (" +
                    std::to_string(thr_per_row) +
                    "), unable to do make_tiled_copy.");
      return std::nullopt;
    }
    size_t thr_per_col = thr_count / thr_per_row;
    if (tile_m % thr_per_col != 0) {
      Note(loc, "#row of tile (" + std::to_string(tile_m) +
                    ") is not divisible by #thread per col (" +
                    std::to_string(thr_per_col) +
                    "), unable to do make_tiled_copy.");
      return std::nullopt;
    }
    ret.thr_layout.first = thr_per_col;
    ret.thr_layout.second = thr_per_row;
    ret.val_layout.first = tile_m / thr_per_col;
    ret.val_layout.second = tile_n / thr_per_row;
    ret.one_row_per_warp = true;
    ret.vectorized_align_bits = 0;
    return ret;
  };

  // used in dma only. Utilize all the threads in block to do DMA.
  auto TiledCopyPrepare = [&](bool one_row_per_warp, size_t mem_align_byte,
                              const ptr<AST::ChunkAt>& ca) -> TiledCopyPlan {
    /*
    for now, `make_tiled_copy` here has some constraints:
      only support 2D DMA;
      do not support dynamic #thread;
    2026/3/24:
    1. Dynamic shape is supported now, dynamic shape will be
    lowering into copy_if with predication.
    2. DMA inside inthreads is supported now, but we only enable it
    when WarpSpec is used and the parallel level is GROUPx4,
    which means the DMA is performed by a single warpgroup of 128
    threads.
    */
    TiledCopyPlan plan;
    const auto& loc = ca->LOC();

    bool shape_mismatch_tail_copy =
        f_ca->GetBlockShape().Rank() == 2 &&
        t_ca->GetBlockShape().Rank() == 2 &&
        STR(f_ca->GetBlockShape()) != STR(t_ca->GetBlockShape());

    bool tail_copy_g2s = n.operation == ".copy" && ca == f_ca &&
                         (f_sty->GetStorage() == Storage::GLOBAL ||
                          f_sty->GetStorage() == Storage::DEFAULT) &&
                         t_sty->GetStorage() == Storage::SHARED &&
                         shape_mismatch_tail_copy;
    bool tail_copy_s2g = n.operation == ".copy" && ca == t_ca &&
                         f_sty->GetStorage() == Storage::SHARED &&
                         (t_sty->GetStorage() == Storage::GLOBAL ||
                          t_sty->GetStorage() == Storage::DEFAULT) &&
                         shape_mismatch_tail_copy;

    if (tail_copy_g2s)
      plan.tail_copy_kind = TailCopyKind::G2S;
    else if (tail_copy_s2g)
      plan.tail_copy_kind = TailCopyKind::S2G;

    if (plan.tail_copy_kind != TailCopyKind::None) {
      const auto& lcs = cgi.GetFunctionLaunches(CurrentFunctionName());
      assert(parallel_idx != -1 && parallel_idx < static_cast<int>(lcs.size()));
      const auto& lconfig = lcs[parallel_idx];
      auto inner_thr_count = lconfig.thread_count.x * lconfig.thread_count.y *
                             lconfig.thread_count.z;
      auto group_count = lconfig.group_count.x * lconfig.group4_count.x *
                         lconfig.group_count.y * lconfig.group_count.z;
      auto thr_count_vi =
          (IsWarpSpecActive() && bdim_level == ParallelLevel::GROUPx4)
              ? inner_thr_count
              : inner_thr_count * group_count;
      if (VIIsInt(thr_count_vi) && VIIsInt(fty->GetShape().ValueAt(1))) {
        size_t thr_count = *VIInt(thr_count_vi);
        BaseType elem_type = plan.tail_copy_kind == TailCopyKind::G2S
                                 ? f_sty->ElementType()
                                 : t_sty->ElementType();
        size_t elem_byte = SizeOf(elem_type);
        size_t tile_n = *VIInt(fty->GetShape().ValueAt(1));
        constexpr size_t bytes_per_copy = 16;
        if (bytes_per_copy % elem_byte == 0) {
          size_t elems_per_copy = bytes_per_copy / elem_byte;
          if (tile_n % elems_per_copy == 0) {
            size_t thr_per_row = tile_n / elems_per_copy;
            if (thr_per_row != 0 && thr_count % thr_per_row == 0) {
              TiledCopyEntry entry;
              entry.thr_layout.first = thr_count / thr_per_row;
              entry.thr_layout.second = thr_per_row;
              entry.val_layout.first = 1;
              entry.val_layout.second = elems_per_copy;
              entry.vectorized_align_bits = 128;
              entry.one_row_per_warp = false;
              plan.tail_copy_helper_entry = entry;
              return plan;
            }
          }
        }
      }
    }

    plan.entry = PrepareTiledCopyEntry(loc, one_row_per_warp, mem_align_byte,
                                       ca->GetBlockShape(), GenStrides(ca),
                                       t_sty->ElementType());

    return plan;
  };

  auto DMACodeGen = [&]() {
    std::string f_mds_offset = "";
    std::string t_mds_offset = "";
    Shape f_shape = f_sty->GetShape();
    Shape t_shape = t_sty->GetShape();
    const auto& f_buf_name = f_buf.first;
    const auto& t_buf_name = t_buf.first;

    if (auto idx = f_ca->IndexOfLastSpanAs()) {
      f_mds_offset = TileBaseOffset(f_ca);
      f_shape = f_ca->OpAt(*idx)->GetBlockShape();
    } else {
      f_mds_offset = ValueSTR(GenOffset(f_ca));
    }

    if (auto idx = t_ca->IndexOfLastSpanAs()) {
      t_mds_offset = TileBaseOffset(t_ca);
      t_shape = t_ca->OpAt(*idx)->GetBlockShape();
    } else {
      t_mds_offset = ValueSTR(GenOffset(t_ca));
    }

    std::vector<size_t> transp_config;
    if (n.operation == ".transp")
      transp_config = cast<TransposeConfig>(n.GetConfig())->dim_values;

    // Determine if we should use WGMMA layout for destination tensor
    bool use_wgmma_layout_t = HasWGMMAInFunction() &&
                              t_sty->GetStorage() == Storage::SHARED &&
                              (t_sty->ElementType() == BaseType::F16 ||
                               t_sty->ElementType() == BaseType::BF16 ||
                               t_sty->ElementType() == BaseType::F8_E4M3);

    auto f_stride = GenStrides(f_ca, transp_config);
    auto t_stride = GenStrides(t_ca);

    auto swizzle_mode = n.GetSwizzleMode();

    bool use_tail_copy_g2s =
        tiled_copy_plan.tail_copy_kind == TailCopyKind::G2S;
    bool use_tail_copy = tiled_copy_plan.tail_copy_kind != TailCopyKind::None;

    Shape f_mds_shape =
        (n.operation == ".pad" ? f_ca->GetBlockShape() : fty->GetShape());
    Shape f_ca_shape = f_ca->GetBlockShape();
    if (use_tail_copy) f_mds_shape = t_ca->GetBlockShape();
    Shape t_mds_shape = fty->GetShape();
    if (use_tail_copy) t_mds_shape = t_ca->GetBlockShape();
    std::optional<TiledCopyEntry> tail_copy_entry =
        tiled_copy_plan.tail_copy_helper_entry;
    bool use_tail_copy_helper = use_tail_copy && tail_copy_entry.has_value() &&
                                tail_copy_entry->vectorized_align_bits == 128;

    std::string f_mds_name;
    std::string t_mds_name;
    const auto f_mds = GenTensorDecl(
        RemoveSuffix(f_buf_name, ".data()"), f_buf_name, f_sty->GetStorage(),
        f_sty->ElementType(), f_mds_shape, false, f_mds_offset,
        ValueSTR(f_stride, false, true), {}, false);
    const auto t_mds = GenTensorDecl(
        RemoveSuffix(t_buf_name, ".data()"), t_buf_name, t_sty->GetStorage(),
        t_sty->ElementType(), t_mds_shape, false, t_mds_offset,
        ValueSTR(t_stride, false, true), {}, use_wgmma_layout_t, swizzle_mode);

    f_mds_name = f_mds.first;
    t_mds_name = t_mds.first;
    ds << f_mds.second;
    ds << t_mds.second;

    // handles dma related to shared memory
    // For async cp.async (non-TMA), threads in the warp must participate;
    // avoid the block-single guard so every lane issues copy/trigger.
    bool need_single_instance = !ThreadCooperative(n);
    if (fty->IsAsync() && !n.IsTMA()) need_single_instance = false;
    bool is_subbyte_copy = (n.operation == ".copy") && !n.IsSparse() &&
                           (IsFloatSubByteType(f_sty->ElementType()) ||
                            IsFloatSubByteType(t_sty->ElementType()));
    bool need_subbyte_async_sync = false;

    if (need_single_instance) {
      ds << d_indent << LevelPred() << " {\n";
      IncrDeviceIndent();
    }
    if (!n.future.empty()) cooperatives.insert(InScopeName(n.future));

    if (n.operation == ".copy" || n.operation == ".transp") {
      if (n.IsSparse() && n.operation == ".copy" && SymbolToSymbol() &&
          !fty->IsAsync()) {
        std::string meta_ptr;
        if (isa<FutureType>(f_ty))
          meta_ptr = "((uint8_t*)" + f_sym + ".mdata())";
        else
          meta_ptr = "((uint8_t*)" + mdata_expr + ")";

        auto M = ValueSTR(f_shape.ValueAt(0));
        auto N = ValueSTR(f_shape.ValueAt(1));
        auto K = ValueSTR(f_shape.ValueAt(2));
        ds << d_indent << "constexpr int __sp_M = " << M << ";\n";
        ds << d_indent << "constexpr int __sp_N = " << N << ";\n";
        ds << d_indent << "constexpr int __sp_K = " << K << ";\n";
        ds << d_indent
           << "static_assert(__sp_K % 4 == 0, \"sparse K must be multiple of "
              "4\");\n";

        auto f_ptr = std::string("((") + NameBaseType(f_sty->ElementType()) +
                     "*)" + f_buf.second + ")";
        auto t_ptr = std::string("((") + NameBaseType(t_sty->ElementType()) +
                     "*)" + t_buf.second + ")";
        if (t_sty->GetStorage() == Storage::SHARED) {
          ds << d_indent << "// sparse encode (2:4)\n";
          ds << d_indent << "for (int i = 0; i < __sp_M; ++i) {\n";
          ds << d_indent << "  for (int j = 0; j < __sp_N; ++j) {\n";
          ds << d_indent << "    for (int k4 = 0; k4 < __sp_K / 4; ++k4) {\n";
          ds << d_indent
             << "      int base = (i * __sp_N + j) * __sp_K + k4 * 4;\n";
          ds << d_indent
             << "      int out_base = (i * __sp_N + j) * (__sp_K / 2) + k4 * "
                "2;\n";
          ds << d_indent << "      auto a0 = " << f_ptr << "[base + 0];\n";
          ds << d_indent << "      auto a1 = " << f_ptr << "[base + 1];\n";
          ds << d_indent << "      auto a2 = " << f_ptr << "[base + 2];\n";
          ds << d_indent << "      auto a3 = " << f_ptr << "[base + 3];\n";
          ds << d_indent << "      uint8_t mask = 0;\n";
          ds << d_indent << "      int count = 0;\n";
          ds << d_indent << "      if (a0 != 0 && count < 2) { " << t_ptr
             << "[out_base + count] = a0; mask |= 1; ++count; }\n";
          ds << d_indent << "      if (a1 != 0 && count < 2) { " << t_ptr
             << "[out_base + count] = a1; mask |= 2; ++count; }\n";
          ds << d_indent << "      if (a2 != 0 && count < 2) { " << t_ptr
             << "[out_base + count] = a2; mask |= 4; ++count; }\n";
          ds << d_indent << "      if (a3 != 0 && count < 2) { " << t_ptr
             << "[out_base + count] = a3; mask |= 8; ++count; }\n";
          ds << d_indent
             << "      if (count < 2) { for (int t = count; t < 2; ++t) "
             << t_ptr << "[out_base + t] = 0; }\n";
          ds << d_indent << "      " << meta_ptr
             << "[(i * __sp_N + j) * (__sp_K / 4) + k4] = mask;\n";
          ds << d_indent << "    }\n";
          ds << d_indent << "  }\n";
          ds << d_indent << "}\n";
        } else {
          ds << d_indent << "// sparse decode (2:4)\n";
          ds << d_indent << "for (int i = 0; i < __sp_M; ++i) {\n";
          ds << d_indent << "  for (int j = 0; j < __sp_N; ++j) {\n";
          ds << d_indent << "    for (int k4 = 0; k4 < __sp_K / 4; ++k4) {\n";
          ds << d_indent
             << "      int out_base = (i * __sp_N + j) * __sp_K + k4 * 4;\n";
          ds << d_indent
             << "      int in_base = (i * __sp_N + j) * (__sp_K / 2) + k4 * "
                "2;\n";
          ds << d_indent << "      uint8_t mask = " << meta_ptr
             << "[(i * __sp_N + j) * (__sp_K / 4) + k4];\n";
          ds << d_indent << "      int idx = 0;\n";
          ds << d_indent << "      " << t_ptr
             << "[out_base + 0] = (mask & 1) ? " << f_ptr
             << "[in_base + idx++] : 0;\n";
          ds << d_indent << "      " << t_ptr
             << "[out_base + 1] = (mask & 2) ? " << f_ptr
             << "[in_base + idx++] : 0;\n";
          ds << d_indent << "      " << t_ptr
             << "[out_base + 2] = (mask & 4) ? " << f_ptr
             << "[in_base + idx++] : 0;\n";
          ds << d_indent << "      " << t_ptr
             << "[out_base + 3] = (mask & 8) ? " << f_ptr
             << "[in_base + idx++] : 0;\n";
          ds << d_indent << "    }\n";
          ds << d_indent << "  }\n";
          ds << d_indent << "}\n";
        }
      } else if (use_tail_copy_helper) {
        const auto& entry = tail_copy_entry.value();
        // Check 128-bit alignment for the async copy atom. Both
        // the global stride AND the chunk column offset must be
        // 16-byte aligned (otherwise cp.async 128b will fault).
        const auto& tc_global_stride = use_tail_copy_g2s ? f_stride : t_stride;
        BaseType tc_elem_type =
            use_tail_copy_g2s ? t_sty->ElementType() : f_sty->ElementType();
        size_t tc_elem_byte = SizeOf(tc_elem_type);
        // Chunk column dim for the global side.
        auto tc_chunk_col = use_tail_copy_g2s
                                ? f_ca_shape.ValueAt(1)
                                : t_ca->GetBlockShape().ValueAt(1);
        bool tc_stride_static_aligned = false;
        bool tc_need_runtime_check = false;
        if (tc_global_stride.size() >= 1) {
          if (VIIsInt(tc_global_stride[0]) && VIIsInt(tc_chunk_col)) {
            int64_t sb = *VIInt(tc_global_stride[0]) *
                         static_cast<int64_t>(tc_elem_byte);
            int64_t cb =
                *VIInt(tc_chunk_col) * static_cast<int64_t>(tc_elem_byte);
            tc_stride_static_aligned = (sb % 16 == 0) && (cb % 16 == 0);
          } else {
            tc_need_runtime_check = true;
          }
        }

        auto EmitCopyIfG2S128 = [&]() {
          IndStream() << "choreo::copy_if_g2s<"
                      << (use_wgmma_layout_t ? "true" : "false") << ", "
                      << NameBaseType(tc_elem_type) << ", "
                      << entry.thr_layout.first << ", "
                      << entry.thr_layout.second << ", "
                      << entry.val_layout.first << ", "
                      << entry.val_layout.second << ">(" << f_mds_name << ", "
                      << t_mds_name << ", [&](const auto& __coord) { return "
                      << "cute::elem_less(__coord, cute::make_shape("
                      << ShapeSTR(f_ca_shape, true) << ")); });\n";
        };
        auto EmitCopyIfG2S32 = [&]() {
          size_t epc32 = 4 / tc_elem_byte;
          size_t tile_n = entry.thr_layout.second * entry.val_layout.second;
          size_t thr_count = entry.thr_layout.first * entry.thr_layout.second;
          size_t tpr32 = tile_n / epc32;
          size_t tr32 = thr_count / tpr32;
          IndStream() << "choreo::copy_if_g2s<"
                      << (use_wgmma_layout_t ? "true" : "false") << ", "
                      << NameBaseType(tc_elem_type) << ", " << tr32 << ", "
                      << tpr32 << ", " << 1 << ", " << epc32 << ", 32>("
                      << f_mds_name << ", " << t_mds_name
                      << ", [&](const auto& __coord) { return "
                      << "cute::elem_less(__coord, cute::make_shape("
                      << ShapeSTR(f_ca_shape, true) << ")); });\n";
        };
        auto EmitCopyIfS2G = [&]() {
          IndStream() << "choreo::copy_if_s2g<" << NameBaseType(tc_elem_type)
                      << ", " << entry.thr_layout.first << ", "
                      << entry.thr_layout.second << ", "
                      << entry.val_layout.first << ", "
                      << entry.val_layout.second << ">(" << f_mds_name << ", "
                      << t_mds_name << ", [&](const auto& __coord) { return "
                      << "cute::elem_less(__coord, cute::make_shape("
                      << ShapeSTR(f_ca_shape, true) << ")); });\n";
        };

        if (use_tail_copy_g2s) {
          if (tc_stride_static_aligned) {
            EmitCopyIfG2S128();
          } else if (tc_need_runtime_check) {
            IndStream() << "if ((" << ValueSTR(tc_global_stride[0]) << " * "
                        << tc_elem_byte << ") % 16 == 0 && ("
                        << ValueSTR(tc_chunk_col) << " * " << tc_elem_byte
                        << ") % 16 == 0) {\n";
            IncrIndent();
            EmitCopyIfG2S128();
            DecrIndent();
            IndStream() << "} else {\n";
            IncrIndent();
            EmitCopyIfG2S32();
            DecrIndent();
            IndStream() << "}\n";
          } else {
            EmitCopyIfG2S32();
          }
        } else {
          EmitCopyIfS2G();
        }
      } else if (is_subbyte_copy) {
        const auto f_byte = GenTensorDecl(
            RemoveSuffix(f_buf_name, ".data()") + "_byte", f_buf_name,
            f_sty->GetStorage(), BaseType::U8, fty->GetShape(), false,
            f_mds_offset, ValueSTR(f_stride, false, true));
        const auto t_byte = GenTensorDecl(
            RemoveSuffix(t_buf_name, ".data()") + "_byte", t_buf_name,
            t_sty->GetStorage(), BaseType::U8, fty->GetShape(), false,
            t_mds_offset, ValueSTR(t_stride, false, true));
        ds << f_byte.second;
        ds << t_byte.second;
        if (fty->IsAsync()) {
          if (!n.future.empty())
            async_subbyte_futures.insert(InScopeName(n.future));
          ds << d_indent << "cute::copy(*(AsyncCopyAtom*)" << future_name
             << ".get_atom(), " << f_byte.first << ", " << t_byte.first
             << ");\n";
          ds << d_indent << "cute::cp_async_fence();\n";
          ds << d_indent << future_name << ".trigger();\n";
          if (need_single_instance) need_subbyte_async_sync = true;
        } else {
          ds << d_indent << "cute::copy(" << f_byte.first << ", "
             << t_byte.first << ");\n";
        }
      } else if (fty->IsAsync()) {
        ds << d_indent << "cute::copy(*(AsyncCopyAtom*)" << future_name
           << ".get_atom(), " << f_mds_name << ", " << t_mds_name << ");\n";
        ds << d_indent << "cute::cp_async_fence();\n";
        ds << d_indent << future_name << ".trigger();\n";
      } else {
        if (tiled_copy_plan.entry.has_value() &&
            n.GetSwizzleMode() == SwizMode::NONE) {
          const TiledCopyEntry& entry = tiled_copy_plan.entry.value();
          IndStream() << "{\n";
          IncrIndent();
          IndStream() << "auto tiled_copy = cute::make_tiled_copy(\n";
          IncrIndent();
          IndStream() << "cute::Copy_Atom<cute::";
          if (entry.vectorized_align_bits == 0)
            Stream() << "UniversalCopy<" << t_sty->ElementType() << ">, ";
          else
            Stream() << "AutoVectorizingCopyWithAssumedAlignment<"
                     << entry.vectorized_align_bits << ">, ";
          Stream() << t_sty->ElementType() << ">{},\n";
          IndStream() << "cute::make_layout(cute::make_shape(cute::Int<"
                      << entry.thr_layout.first << ">{}, cute::Int<"
                      << entry.thr_layout.second
                      << ">{}), cute::make_stride(cute::Int<"
                      << entry.thr_layout.second << ">{}, cute::Int<1>{})),\n";
          IndStream() << "cute::make_layout(cute::make_shape(cute::Int<"
                      << entry.val_layout.first << ">{}, cute::Int<"
                      << entry.val_layout.second << ">{}))\n";
          DecrIndent();
          IndStream() << ");\n";
          IndStream()
              << "auto thr_copy = tiled_copy.get_thread_slice(threadIdx.x);\n";
          IndStream() << "auto src_thr = thr_copy.partition_S(" << f_mds_name
                      << ");\n";
          IndStream() << "auto dst_thr = thr_copy.partition_D(" << t_mds_name
                      << ");\n";
          IndStream() << "cute::copy(tiled_copy, src_thr, dst_thr);\n";
          DecrIndent();
          IndStream() << "}\n";
        } else {
          // Try copy_if_g2s/copy_if_s2g as a general fallback before
          // resorting to the low-performance opt_copy.
          bool is_dma_g2s = (f_sty->GetStorage() == Storage::GLOBAL ||
                             f_sty->GetStorage() == Storage::DEFAULT) &&
                            t_sty->GetStorage() == Storage::SHARED;
          bool is_dma_s2g = f_sty->GetStorage() == Storage::SHARED &&
                            (t_sty->GetStorage() == Storage::GLOBAL ||
                             t_sty->GetStorage() == Storage::DEFAULT);

          bool used_copy_if = false;

          if (is_dma_g2s || is_dma_s2g) {
            // Determine a static box shape from either side (prefer the
            // shared side, which is almost always statically shaped).
            Shape box_shape;
            bool has_static_box = false;

            Shape shared_block =
                is_dma_g2s ? t_ca->GetBlockShape() : f_ca->GetBlockShape();
            Shape other_block =
                is_dma_g2s ? f_ca->GetBlockShape() : t_ca->GetBlockShape();

            if (shared_block.Rank() == 2 && !shared_block.IsDynamic()) {
              box_shape = shared_block;
              has_static_box = true;
            } else if (other_block.Rank() == 2 && !other_block.IsDynamic()) {
              box_shape = other_block;
              has_static_box = true;
            }

            if (has_static_box) {
              size_t box_m = *VIInt(box_shape.ValueAt(0));
              size_t box_n = *VIInt(box_shape.ValueAt(1));
              BaseType elem_type =
                  is_dma_g2s ? t_sty->ElementType() : f_sty->ElementType();
              size_t elem_byte = SizeOf(elem_type);
              constexpr size_t kBytesPerCopy = 16; // 128 bits

              // copy_if_g2s uses SM80_CP_ASYNC_CACHEGLOBAL_ZFILL
              // <uint128_t> which requires 16-byte aligned source
              // addresses. Both the row stride AND the chunk column
              // offset must be 16-byte aligned.
              const auto& global_stride = is_dma_g2s ? f_stride : t_stride;
              auto global_chunk_col = is_dma_g2s
                                          ? f_ca->GetBlockShape().ValueAt(1)
                                          : t_ca->GetBlockShape().ValueAt(1);
              bool stride_static_128 = false;
              bool stride_need_rtcheck = false;
              if (global_stride.size() >= 1) {
                if (VIIsInt(global_stride[0]) && VIIsInt(global_chunk_col)) {
                  int64_t sb = *VIInt(global_stride[0]) *
                               static_cast<int64_t>(elem_byte);
                  int64_t cb = *VIInt(global_chunk_col) *
                               static_cast<int64_t>(elem_byte);
                  stride_static_128 = (sb % 16 == 0) && (cb % 16 == 0);
                } else {
                  stride_need_rtcheck = true;
                }
              }

              const auto& lcs = cgi.GetFunctionLaunches(CurrentFunctionName());
              bool have_launch = parallel_idx != -1 &&
                                 parallel_idx < static_cast<int>(lcs.size());

              if (have_launch) {
                const auto& lconfig = lcs[parallel_idx];
                auto inner_thr = lconfig.thread_count.x *
                                 lconfig.thread_count.y *
                                 lconfig.thread_count.z;
                auto group_cnt = lconfig.group_count.x *
                                 lconfig.group4_count.x *
                                 lconfig.group_count.y * lconfig.group_count.z;
                auto thr_count_vi =
                    (IsWarpSpecActive() && bdim_level == ParallelLevel::GROUPx4)
                        ? inner_thr
                        : inner_thr * group_cnt;

                if (VIIsInt(thr_count_vi)) {
                  size_t thr_count = *VIInt(thr_count_vi);

                  // Compute 128-bit thread layout.
                  bool has_128 = false;
                  size_t epc128 = 0, tpr128 = 0, tr128 = 0;
                  if (kBytesPerCopy % elem_byte == 0) {
                    epc128 = kBytesPerCopy / elem_byte;
                    if (box_n % epc128 == 0) {
                      tpr128 = box_n / epc128;
                      if (tpr128 && thr_count % tpr128 == 0) {
                        tr128 = thr_count / tpr128;
                        has_128 = tr128 && box_m % tr128 == 0;
                      }
                    }
                  }

                  // Compute 32-bit thread layout.
                  constexpr size_t kBytesPerCopy32 = 4;
                  bool has_32 = false;
                  size_t epc32 = 0, tpr32 = 0, tr32 = 0;
                  if (kBytesPerCopy32 % elem_byte == 0) {
                    epc32 = kBytesPerCopy32 / elem_byte;
                    if (box_n % epc32 == 0) {
                      tpr32 = box_n / epc32;
                      if (tpr32 && thr_count % tpr32 == 0) {
                        tr32 = thr_count / tpr32;
                        has_32 = tr32 && box_m % tr32 == 0;
                      }
                    }
                  }

                  bool can_emit =
                      (stride_static_128 && has_128) ||
                      (stride_need_rtcheck && has_128 && has_32) ||
                      (!stride_static_128 && !stride_need_rtcheck && has_32);

                  if (can_emit) {
                    std::string src_name, dst_name;
                    bool shapes_match = STR(f_ca->GetBlockShape()) ==
                                        STR(t_ca->GetBlockShape());

                    if (shapes_match && !f_mds_shape.IsDynamic()) {
                      src_name = f_mds_name;
                      dst_name = t_mds_name;
                    } else {
                      const auto f_box = GenTensorDecl(
                          RemoveSuffix(f_buf_name, ".data()") + "_cif",
                          f_buf_name, f_sty->GetStorage(), f_sty->ElementType(),
                          box_shape, false, f_mds_offset,
                          ValueSTR(f_stride, false, true));
                      const auto t_box = GenTensorDecl(
                          RemoveSuffix(t_buf_name, ".data()") + "_cif",
                          t_buf_name, t_sty->GetStorage(), t_sty->ElementType(),
                          box_shape, false, t_mds_offset,
                          ValueSTR(t_stride, false, true), {},
                          use_wgmma_layout_t, swizzle_mode);
                      ds << f_box.second;
                      ds << t_box.second;
                      src_name = f_box.first;
                      dst_name = t_box.first;
                    }

                    Shape pred_shape = is_dma_g2s ? f_ca->GetBlockShape()
                                                  : t_ca->GetBlockShape();
                    std::string pred_str;
                    if (!pred_shape.IsDynamic() &&
                        STR(pred_shape) == STR(box_shape)) {
                      pred_str = "[](const auto&) { return true; }";
                    } else {
                      pred_str = "[&](const auto& __coord) { return "
                                 "cute::elem_less(__coord, "
                                 "cute::make_shape(" +
                                 ShapeSTR(pred_shape, true) + ")); }";
                    }

                    auto EmitG2S = [&](size_t tr, size_t tpr, size_t epc,
                                       int bits) {
                      IndStream()
                          << "choreo::copy_if_g2s<"
                          << (use_wgmma_layout_t ? "true" : "false") << ", "
                          << NameBaseType(t_sty->ElementType()) << ", " << tr
                          << ", " << tpr << ", " << 1 << ", " << epc;
                      if (bits != 128) IndStream() << ", " << bits;
                      IndStream() << ">(" << src_name << ", " << dst_name
                                  << ", " << pred_str << ");\n";
                    };

                    if (is_dma_g2s) {
                      if (stride_static_128) {
                        EmitG2S(tr128, tpr128, epc128, 128);
                      } else if (stride_need_rtcheck) {
                        IndStream() << "if ((" << ValueSTR(global_stride[0])
                                    << " * " << elem_byte << ") % 16 == 0 && ("
                                    << ValueSTR(global_chunk_col) << " * "
                                    << elem_byte << ") % 16 == 0) {\n";
                        IncrIndent();
                        EmitG2S(tr128, tpr128, epc128, 128);
                        DecrIndent();
                        IndStream() << "} else {\n";
                        IncrIndent();
                        EmitG2S(tr32, tpr32, epc32, 32);
                        DecrIndent();
                        IndStream() << "}\n";
                      } else {
                        EmitG2S(tr32, tpr32, epc32, 32);
                      }
                    } else {
                      IndStream() << "choreo::copy_if_s2g<"
                                  << NameBaseType(f_sty->ElementType()) << ", "
                                  << tr128 << ", " << tpr128 << ", " << 1
                                  << ", " << epc128 << ">(" << src_name << ", "
                                  << dst_name << ", " << pred_str << ");\n";
                    }
                    used_copy_if = true;
                  }
                }
              }
            }
          }

          if (!used_copy_if) {
            bool both_dynamic = (is_dma_g2s || is_dma_s2g) &&
                                f_ca->GetBlockShape().IsDynamic() &&
                                t_ca->GetBlockShape().IsDynamic();
            if (both_dynamic) {
              Warning(n.LOC(), "DMA: both source and destination have "
                               "dynamic shapes; falling back to basic "
                               "copy (performance loss expected).");
            }
            Note(n.LOC(), "DMA is lowered to low-performance opt_copy API.");
            IndStream() << "opt_copy(" << f_mds_name << ", " << t_mds_name
                        << ");\n";
          }
        }
      }
      VerboseDMA(ds, d_indent, t_sym, f_sym, n.operation.substr(1), "", 1,
                 ", line " + std::to_string(n.LOC().begin.line));
    } else if (n.operation == ".pad") {
      static int pad_cnt = 0;
      auto pad_config = cast<PadConfig>(n.GetConfig());

      auto pcmvSTR = [&](ptr<AST::MultiValues> mv) -> std::string {
        std::string res;
        for (const auto& v : mv->AllValues()) {
          if (!res.empty()) res += ", ";
          res += ExprSTR(v, IsHost());
        }
        return res;
      };

      ds << d_indent << "cute::fill(" << t_mds_name << ", "
         << ExprSTR(pad_config->GetPadValue(), IsHost()) << ");\n";
      std::string pad_offset = "__pad_offset" + std::to_string(pad_cnt);
      ds << d_indent << "auto " << pad_offset << " = " << t_mds_name
         << ".layout()("
         << "cute::make_coord(" << pcmvSTR(pad_config->pad_low) << "));\n";
      const auto t_pad_mds = GenTensorDecl(
          RemoveSuffix(t_buf_name, ".data()"), t_buf_name, t_sty->GetStorage(),
          t_sty->ElementType(), f_ca->GetBlockShape(), false, pad_offset,
          ValueSTR(t_stride, false, true), {}, use_wgmma_layout_t,
          swizzle_mode);
      std::string t_pad_mds_name{t_pad_mds.first};
      std::string t_pad_mds_decl{t_pad_mds.second};
      ds << t_pad_mds_decl;

      if (fty->IsAsync()) {
        ds << d_indent << "cute::copy(*(AsyncCopyAtom*)" << future_name
           << ".get_atom(), " << f_mds_name << ", " << t_pad_mds_name << ");\n";
        ds << d_indent << "cute::cp_async_fence();\n";
        ds << d_indent << future_name << ".trigger();\n";
      } else {
        ds << d_indent << "opt_copy(" << f_mds_name << ", " << t_pad_mds_name
           << ");\n";
      }

      ++pad_cnt;

      VerboseDMA(ds, d_indent, t_sym, f_sym, n.operation.substr(1), "", 1,
                 ", line " + std::to_string(n.LOC().begin.line));
    }

    if (need_single_instance) {
      DecrDeviceIndent();
      ds << d_indent << "} // single instance\n";
    }

    if (need_subbyte_async_sync) { ds << d_indent << "__syncthreads();\n"; }

    if (!fty->IsAsync()) {
      // not async, must syncthreads immediately
      // else, defer the sync till the wait time
      if (NeedWarpSpecGroupX4SyncForCurrentScope())
        EmitGroupX4Sync(ds, d_indent);
      else
        ds << d_indent << "__syncthreads();\n";
    }
  };

  auto TMACodeGen = [&]() {
    if (n.operation != ".copy")
      choreo_unreachable("unsupported tma operation: " + n.operation + ".");

    const auto& tma_descs = cgi.GetTMADescs()[cur_pb];
    ParallelLevel tma_sync_level = ParallelLevel::BLOCK;
    if (n.IsTMA() && !isa<PlaceHolderType>(NodeType(n))) {
      int tma_idx = tma_count++;
      assert(tma_idx < static_cast<int>(tma_descs.size()));
      const TMADesc& tma_desc = tma_descs[tma_idx];
      auto in_thr_block = tma_desc.GetInThreadsBlock();
      if (in_thr_block) {
        tma_sync_level = tma_desc.GetPBLevel();
      } else if (IsWarpSpecActive() &&
                 (t_sty->GetStorage() == Storage::GLOBAL ||
                  t_sty->GetStorage() == Storage::DEFAULT) &&
                 f_sty->GetStorage() == Storage::SHARED &&
                 bdim_level == ParallelLevel::GROUPx4) {
        tma_sync_level = ParallelLevel::GROUPx4;
      }
    }

    auto fsto = f_sty->GetStorage();
    auto tsto = t_sty->GetStorage();
    std::string f_mds_offset = "";
    std::string t_mds_offset = "";
    Shape f_shape = f_sty->GetShape();
    Shape t_shape = t_sty->GetShape();
    const auto& f_buf_expr = f_buf.second;
    const auto& t_buf_expr = t_buf.second;

    if (auto idx = f_ca->IndexOfLastSpanAs()) {
      f_mds_offset = TileBaseOffset(f_ca);
      f_shape = f_ca->OpAt(*idx)->GetBlockShape();
    } else
      f_mds_offset = ValueSTR(GenOffset(f_ca));

    if (auto idx = t_ca->IndexOfLastSpanAs()) {
      t_mds_offset = TileBaseOffset(t_ca);
      t_shape = t_ca->OpAt(*idx)->GetBlockShape();
    } else
      t_mds_offset = ValueSTR(GenOffset(t_ca));

    auto tname = GetTMAName(n);
    assert(tname.has_value());
    if ((fsto == Storage::GLOBAL || fsto == Storage::DEFAULT) &&
        tsto == Storage::SHARED) {
      std::string tma_tx_bytes_expr = std::to_string(
          t_ca->GetBlockShape().ElementCount() * SizeOf(t_sty->ElementType()));
      std::string t_buf_expr_with_offset = t_buf_expr;
      if (!t_mds_offset.empty() && t_mds_offset != "0") {
        t_buf_expr_with_offset =
            "(" + t_buf_expr + " + (" + t_mds_offset + "))";
      }
      // async tma copy can be:
      // 1. event only and non-warpspec
      // 2. future only and non-warpspec
      // 3. event only and warpspec
      // 4. future only and warpspec
      bool event_only = n.HasEvent();
      bool warpspec_only = InSpecWarp();
      bool future_only = !future_name.empty();

      assert(!(event_only && future_only) &&
             "event and future cannot be both present");
      if (warpspec_only && event_only) {
        assert(n.IsAsync() && "warpspec event-only tma copy must be async");
      }

      auto rev_indices = Reverse(GenIndices(f_ca));
      std::vector<std::string> hoisted_rev_indices;
      if (hoist_offset && t_shape.Rank() == 2 && rev_indices.size() == 2) {
        auto make_dim_name = [&](size_t dim_index) {
          auto dim = f_sty->GetShape().ValueAt(f_sty->Dims() - 1 - dim_index);
          auto dim_name = ToLower(UnScopedExpr(ValueSTR(dim)));
          if (dim_name.empty())
            return std::string("d") + std::to_string(dim_index);
          bool valid = std::all_of(
              dim_name.begin(), dim_name.end(),
              [](unsigned char ch) { return std::isalnum(ch) || ch == '_'; });
          if (!valid) return std::string("d") + std::to_string(dim_index);
          return dim_name;
        };

        auto base_name = f_ca->data ? f_ca->data->name : std::string("src");
        for (size_t idx = 0; idx < rev_indices.size(); ++idx) {
          auto offset_name = base_name + "_" + make_dim_name(idx) + "_offset";
          ds << d_indent << "const unsigned " << offset_name << " = "
             << ValueSTR(rev_indices.at(idx)) << ";\n";
          hoisted_rev_indices.push_back(offset_name);
        }
      }
      bool is_multicast_tma = n.IsMulticast() && n.IsTMA();
      bool has_cluster = !cgi.GetFunctionLaunches(fname).empty() &&
                         cgi.GetFunctionLaunches(fname)[0].HasCluster();
      bool use_ptx_tma_sync = (tma_cluster_aware || ptx_barrier ||
                               is_multicast_tma || has_cluster) &&
                              t_shape.Rank() == 2;
      bool emit_tma_single_guard =
          !ScopeAlreadySingleThreadForLevel(tma_sync_level);
      if (!warpspec_only)
        assert(emit_tma_single_guard &&
               "non-warpspec tma copy must be single-threaded");

      std::string tma_issue_prefix = emit_tma_single_guard ? "  " : "";

      if (emit_tma_single_guard) {
        if (tma_sync_level == ParallelLevel::GROUP)
          ds << d_indent << "if (__CHOREO_GROUP_SINGLE__) {\n";
        else if (tma_sync_level == ParallelLevel::GROUPx4)
          ds << d_indent << "if (__CHOREO_GROUPX4_SINGLE__) {\n";
        else
          ds << d_indent << "if (__CHOREO_BLOCK_SINGLE__) {\n";
      }

      if (use_ptx_tma_sync) {
        std::string ptx_bar_expr;
        if (event_only) {
          ptx_bar_expr = "(uint64_t*)&(" + ExprSTR(n.Event(), IsHost()) + ")";
        } else {
          ptx_bar_expr =
              "((TMAAtom*)" + future_name + ".get_atom())->ptx_barrier()";
          std::string expect_fn = "choreo::tma_mbarrier_expect_tx";
          ds << d_indent << tma_issue_prefix << expect_fn << "(" << ptx_bar_expr
             << ", " << tma_tx_bytes_expr << ");\n";
        }

        auto coord0_expr = hoisted_rev_indices.empty()
                               ? ValueSTR(rev_indices.at(0))
                               : hoisted_rev_indices.at(0);
        auto coord1_expr = hoisted_rev_indices.empty()
                               ? ValueSTR(rev_indices.at(1))
                               : hoisted_rev_indices.at(1);
        if (is_multicast_tma) {
          const auto& lcfg = cgi.GetFunctionLaunches(fname);
          auto cluster_total = lcfg.empty() ? sbe::nu(1)
                                            : lcfg[0].cluster_count.x *
                                                  lcfg[0].cluster_count.y *
                                                  lcfg[0].cluster_count.z;
          ds << d_indent << tma_issue_prefix
             << "if (choreo::tma_cluster_rank() == 0) {\n";
          ds << d_indent << tma_issue_prefix << "  "
             << "choreo::tma_load_2d_shared_cluster_global_mbarrier_multicast("
                "(void*)"
             << t_buf_expr_with_offset << ", (const void*)&" << *tname
             << "_tensor_map, " << ptx_bar_expr << ", " << coord0_expr << ", "
             << coord1_expr << ", "
             << "static_cast<uint16_t>((1u << " << ValueSTR(cluster_total)
             << ") - 1u)"
             << ");\n";
          ds << d_indent << tma_issue_prefix << "}\n";
        } else if (tma_cluster_aware) {
          ds << d_indent << tma_issue_prefix
             << "choreo::tma_load_2d_shared_cluster_global_mbarrier((void*)"
             << t_buf_expr_with_offset << ", (const void*)&" << *tname
             << "_tensor_map, " << ptx_bar_expr << ", " << coord0_expr << ", "
             << coord1_expr << ");\n";
        } else {
          ds << d_indent << tma_issue_prefix
             << "choreo::tma_load_2d_shared_cta_global_mbarrier((void*)"
             << t_buf_expr_with_offset << ", (const void*)&" << *tname
             << "_tensor_map, " << ptx_bar_expr << ", " << coord0_expr << ", "
             << coord1_expr << ");\n";
        }
      } else {
        std::string tma_barrier_arg =
            future_only
                ? "((TMAAtom*)" + future_name + ".get_atom())->barrier()"
                : ExprSTR(n.Event(), IsHost());

        ds << d_indent << tma_issue_prefix << "cde::cp_async_bulk_tensor_"
           << t_shape.Rank() << "d_global_to_shared(" << t_buf_expr_with_offset
           << ", &" << *tname << "_tensor_map, "
           << (hoisted_rev_indices.empty() ? ValueSTR(rev_indices)
                                           : hoisted_rev_indices.at(0) + ", " +
                                                 hoisted_rev_indices.at(1))
           << ", " << tma_barrier_arg << ");\n";
        if (future_only) {
          ds << d_indent << tma_issue_prefix << "((TMAAtom*)" << future_name
             << ".get_atom())->token() = "
                "cuda::device::barrier_arrive_tx(((TMAAtom*)"
             << future_name << ".get_atom())->barrier(), 1, "
             << tma_tx_bytes_expr << ");\n";
        }
      }
      if (emit_tma_single_guard) {
        ds << d_indent << "}";
        if (future_only && !use_ptx_tma_sync) {
          ds << " else {\n";
          ds << d_indent << "  ((TMAAtom*)" << future_name
             << ".get_atom())->token() = ((TMAAtom*)" << future_name
             << ".get_atom())->barrier().arrive();\n";
          ds << d_indent << "}\n";
        } else
          ds << "\n";
      }
      recent_tma_tx_bytes.push_back(tma_tx_bytes_expr);
      if (recent_tma_tx_bytes.size() > 8) recent_tma_tx_bytes.pop_front();

      // For async tma.copy.async, trigger the future
      // For sync tma.copy, default behavior is immediate wait.
      // Under --use-warpspec, defer this wait to event barrier protocol
      // so producer-consumer pipelining remains asynchronous like ref kernels.
      if (fty->IsAsync()) {
        if (future_only) ds << d_indent << future_name << ".trigger();\n";
      } else {
        // Synchronous tma.copy: wait immediately
        // Make sure the future is marked initialized before marking it nowait
        // to avoid runtime diagnostics when the state is still ST_NONE.
        if (future_only) {
          if (use_ptx_tma_sync) {
            ds << d_indent << "choreo::tma_mbarrier_wait_parity(((TMAAtom*)"
               << future_name << ".get_atom())->ptx_barrier(), ((TMAAtom*)"
               << future_name << ".get_atom())->ptx_phase_bit());\n";
            ds << d_indent << "((TMAAtom*)" << future_name
               << ".get_atom())->toggle_ptx_phase();\n";
          } else {
            ds << d_indent << "((TMAAtom*)" << future_name
               << ".get_atom())->barrier().wait(std::move(((TMAAtom*)"
               << future_name << ".get_atom())->token()));\n";
          }
          ds << d_indent << future_name << ".set_nowait();\n\n";
        }
      }
    } else if ((tsto == Storage::GLOBAL || tsto == Storage::DEFAULT) &&
               fsto == Storage::SHARED) {
      ds << d_indent << "cde::fence_proxy_async_shared_cta();\n";

      if (tma_sync_level == ParallelLevel::GROUP)
        ds << d_indent << "__syncwarp();\n";
      else if (tma_sync_level == ParallelLevel::GROUPx4)
        EmitGroupX4Sync(ds, d_indent);
      else if (NeedWarpSpecGroupX4SyncForCurrentScope())
        EmitGroupX4Sync(ds, d_indent);
      else
        ds << d_indent << "__syncthreads();\n";

      if (tma_sync_level == ParallelLevel::GROUP)
        ds << d_indent << "if (__CHOREO_GROUP_SINGLE__) {\n";
      else if (tma_sync_level == ParallelLevel::GROUPx4)
        ds << d_indent << "if (__CHOREO_GROUPX4_SINGLE__) {\n";
      else
        ds << d_indent << "if (__CHOREO_BLOCK_SINGLE__) {\n";

      ds << d_indent << "  cde::cp_async_bulk_tensor_" << t_shape.Rank()
         << "d_shared_to_global(&" << *tname << "_tensor_map, "
         << ValueSTR(Reverse(GenIndices(t_ca))) << ", " << f_buf_expr << ");\n";
      ds << d_indent << "  cde::cp_async_bulk_commit_group();\n";
      ds << d_indent << "}\n";
      // DO not check or wait. TMA share=>global is special
    }
  };

  if (use_tma)
    TMACodeGen();
  else {
    bool g2s = false, s2g = false;
    if ((f_sty->GetStorage() == Storage::GLOBAL ||
         f_sty->GetStorage() == Storage::DEFAULT) &&
        t_sty->GetStorage() == Storage::SHARED)
      g2s = true;
    if (f_sty->GetStorage() == Storage::SHARED &&
        (t_sty->GetStorage() == Storage::GLOBAL ||
         t_sty->GetStorage() == Storage::DEFAULT))
      s2g = true;
    // AutoVectorizingCopyWithAssumedAlignment check the alignment of both src
    // and dst. Cuda global buffer is all aligned with 256 bytes
    size_t mem_align_byte =
        std::min(static_cast<size_t>(256),
                 CCtx().GetMemoryAlignmentByte(Storage::SHARED));
    // TODO: how to determin `one_row_per_warp`? maybe add option for user.
    if (g2s) {
      bool one_row_per_warp = false;
      tiled_copy_plan =
          TiledCopyPrepare(one_row_per_warp, mem_align_byte, f_ca);
    } else if (s2g) {
      bool one_row_per_warp = false;
      tiled_copy_plan =
          TiledCopyPrepare(one_row_per_warp, mem_align_byte, t_ca);
    }
    DMACodeGen();
  }

  return true;
}

bool CuteCodeGen::Visit(AST::MMA& n) {
  auto& op = *n.GetOperation();

  if (op.Tag() == AST::MMAOperation::Commit) {
    EmitWGMMAFinalize(ds, d_indent);
    return true;
  }

  const ptr<AST::Expr>& frag = op.GetFrag(); // the primary frag node
  std::string sym = AST::FragName(frag);     // the primary symbol
  std::string scoped_frag_sym = InScopeName(sym);
  if (!FCtx(fname).FragHasMMAType(scoped_frag_sym)) {
    Error1(n.LOC(), "the MMA operation of `" + scoped_frag_sym +
                        "` cannot be executed.");
    choreo_unreachable(
        "MMA information is incomplete (maybe lack of mma exec).");
  }

  auto TraverseWholeArrayBegin = [&](const ValueList& array_dims,
                                     std::string& indices) {
    for (size_t i = 0; i < array_dims.size(); ++i) {
      std::string idx = "idx" + std::to_string(i);
      indices += "[" + idx + "]";
      ds << d_indent << "for (int " << idx << " = 0; " << idx << " < "
         << ValueSTR(array_dims[i]) << "; ++" << idx << ")\n";
      IncrDeviceIndent();
    }
  };
  auto TraverseWholeArrayEnd = [&](const ValueList& array_dims) {
    for (size_t i = 0; i < array_dims.size(); ++i) DecrDeviceIndent();
  };

  static int fill_cnt = 0;

  if (FCtx(fname).FragIsWGMMA(scoped_frag_sym)) {
    // WGMMA codegen path (128-thread warp group) using PTX inline assembly
    auto& ssmi = cgi.GetSymbolMMA(scoped_frag_sym);
    if (!sbe::is_pow2(ssmi.shape[1])) extended_mma = true;
    // Determine accumulator type: f32 for f16->f32, f16 for f16->f16
    std::string accum_type = (ssmi.ty == BaseType::F16) ? "f16" : "f32";
    switch (op.Tag()) {
    case AST::MMAOperation::Fill: {
      // dtype of accu: s32, f16, f32 (f16 => u32)
      auto acc_dtype = ssmi.ty;
      ValueItem frag_len = ssmi.shape[1] / sbe::nu(2); // N / 2
      bool use_uint32 = false;
      if (ssmi.ty == BaseType::F16) {
        use_uint32 = true;
        acc_dtype = BaseType::U32;
        frag_len = frag_len / sbe::nu(2);
      }
      if (!VIIsInt(frag_len))
        choreo_unreachable(
            "expect the length of wgmma fragment to be integer but not symbol");
      // the register number in a single mc frag.
      reg_num_d = *VIInt(frag_len);

      ptr<ArrayType> aty = nullptr;
      if (op.FillingArrayDims())
        aty = cast<ArrayType>(NodeType(*op.FillingArrayDims()));

      if (op.FillingIsDecl()) {
        ds << d_indent << NameBaseType(acc_dtype) << " " << sym;
        if (aty)
          for (const auto& dim : aty->Dimensions())
            ds << "[" << ValueSTR(dim) << "]";
        ds << "[" << reg_num_d << "];\n";
        ssm.MapDeviceSymbol(InScopeName(sym), sym);
      }
      // TODO: #pragma unroll
      // if ubound is large, may lead to low performance
      std::string scalar_init_val =
          ExprCastSTR(op.FillingValue(), std::nullopt, ssmi.ty,
                      GetBaseType(*op.FillingValue()->GetType()), false);
      std::string frag_iv_str = "__frag_init_val" + std::to_string(fill_cnt);
      if (use_uint32) {
        std::string temp = "__fiv_temp" + std::to_string(fill_cnt);
        ds << d_indent << "uint32_t " << frag_iv_str << " = broadcast_to_u32("
           << scalar_init_val << ");\n";
      } else {
        ds << d_indent << NameBaseType(ssmi.ty) << " " << frag_iv_str << " = "
           << scalar_init_val << ";\n";
      }
      if (aty && !AST::FragIsArrayElem(frag)) {
        // need to fill the whole fragment array
        std::string indices;
        TraverseWholeArrayBegin(aty->Dimensions(), indices);
        // the loop body
        ds << d_indent << "for (int idx = 0; idx < " << reg_num_d
           << "; ++idx) {\n";
        IncrDeviceIndent();
        ds << d_indent << sym << indices << "[idx] = " << frag_iv_str << ";\n";
        DecrDeviceIndent();
        ds << d_indent << "}\n";
        TraverseWholeArrayEnd(aty->Dimensions());
      } else {
        ds << d_indent << "for (int idx = 0; idx < " << reg_num_d
           << "; ++idx)\n";
        IncrDeviceIndent();
        ds << d_indent << ExprSTR(frag, false) << "[idx] = " << frag_iv_str
           << ";\n";
        DecrDeviceIndent();
      }
      ++fill_cnt;
    } break;
    case AST::MMAOperation::Load: {
      // Check if loading from a buffer whose shape may not evenly divide
      // the MMA fragment. Static shared buffers are always tile-sized, but
      // dynamic-shaped buffers (e.g., global or runtime-sized) may have
      // dimensions that don't divide evenly by the MMA atom shape.
      {
        auto ca = op.LoadFrom();
        auto parent_sym = ca->RefSymbol();
        auto parent_ty = GetSpannedType(GetSymbolType(parent_sym));
        switch (GetMmaLoadShapeWarningKind(ssmi, parent_ty)) {
        case MmaLoadShapeWarningKind::Dynamic:
          Warning(n.LOC(), "mma.load source buffer '" + parent_sym +
                               "' has dynamic shape; ensure its dimensions are "
                               "divisible by the MMA atom shape to avoid "
                               "out-of-bounds access.");
          break;
        case MmaLoadShapeWarningKind::Misaligned:
          Warning(n.LOC(),
                  "mma.load source buffer '" + parent_sym +
                      "' has shape that does not evenly divide the MMA "
                      "atom; this may cause out-of-bounds shared memory "
                      "access.");
          break;
        case MmaLoadShapeWarningKind::None: break;
        }
      }
      std::string elem_ty = NameBaseType(ssmi.ty);
      std::string mma_policy = FCtx(fname).MMAPolicyOfFrag(InScopeName(sym));
      bool policy_is_sparse = mma_policy.find("SPARSE::") != std::string::npos;
      if (policy_is_sparse && ssmi.frag == MMAInfo::FRAG_E) {
        auto tile_addr = TileAddr(op.LoadFrom(), false);
        auto strides = GenStrides(op.LoadFrom());
        auto load_from_sty = dyn_cast<SpannedType>(op.LoadFrom()->GetType());
        auto k_val = VIInt(ssmi.shape.at(2));
        bool policy_is_fp8 = mma_policy.find("E4M3") != std::string::npos ||
                             mma_policy.find("E5M2") != std::string::npos;
        bool fp8_sparse_k64 = policy_is_fp8 && k_val && *k_val == 64;
        bool sparse_k32_16bit = !policy_is_fp8 && k_val && *k_val == 32;
        bool sparse_k64_16bit = !policy_is_fp8 && k_val && *k_val == 64;
        bool prepack_single_col = false;
        if (load_from_sty && load_from_sty->Dims() >= 2) {
          if (auto meta_cols = VIInt(load_from_sty->GetShape().ValueAt(1)))
            prepack_single_col = (*meta_cols == 1);
        }
        bool meta_64 = k_val && *k_val > 32 && !fp8_sparse_k64;
        std::string meta_ty = meta_64 ? "uint64_t" : "uint32_t";
        std::string row_stride = ValueSTR(strides.at(0));
        std::string col_stride = ValueSTR(strides.at(1));
        ds << d_indent << meta_ty << " " << sym << " = 0;\n";
        // Detect host prepacked-u32 metadata and emit device-side indexing
        // that reads the host-provided prepacked u32 array directly. Fallback
        // to the existing byte-by-byte assembly when detection fails.
        // The --use-prepack / --use-prepack-v2 flags force this path.
        bool v2_mode = use_prepack_v2.GetValue();
        std::string ref_sym = op.LoadFrom()->RefSymbol();
        auto prepackInfo =
            resolvePrepackedU32Meta(ref_sym, use_prepack.GetValue() || v2_mode);

        ds << d_indent << "{\n";
        ds << d_indent << "  int __sp_tid = threadIdx.x % 128;\n";
        if (!prepackInfo.use_packed_u32)
          ds << d_indent << "  auto* __sp_meta_ptr = (uint8_t*)("
             << ValueSTR(tile_addr) << ");\n";
        if (fp8_sparse_k64) {
          if (prepackInfo.use_packed_u32 && v2_mode) {
            emitFp8PrepackedV2TileLoadSnippet(sym, prepackInfo.device_name,
                                              ValueSTR(tile_addr), row_stride,
                                              col_stride);
          } else if (prepackInfo.use_packed_u32) {
            emitFp8PrepackedU32TileLoadSnippet(sym, ValueSTR(tile_addr),
                                               row_stride, col_stride);
          } else {
            ds << d_indent
               << "  int __sp_row = ((__sp_tid >> 2) & 7) + ((__sp_tid & 1) << "
                  "3) + ((__sp_tid >> 5) << 4);\n";
            ds << d_indent
               << "  int __sp_byte_col = ((__sp_tid >> 1) & 1) << 2;\n";
            ds << d_indent << "  #pragma unroll\n";
            ds << d_indent
               << "  for (int byte_idx = 0; byte_idx < 4; ++byte_idx) {\n";
            ds << d_indent << "    uint8_t packed = __sp_meta_ptr[__sp_row * ("
               << row_stride << ") + (__sp_byte_col + byte_idx) * ("
               << col_stride << ")];\n";
            ds << d_indent << "    " << sym << " |= (static_cast<" << meta_ty
               << ">(packed) << (8 * byte_idx));\n";
            ds << d_indent << "  }\n";
          }
        } else if (prepackInfo.use_packed_u32 && v2_mode) {
          if (prepack_single_col) {
            auto tile_off = GenOffset(op.LoadFrom());
            emitPrepackedV2TileLoadSnippet(sym, prepackInfo.device_name,
                                           ValueSTR(tile_addr), row_stride,
                                           ValueSTR(tile_off));
          } else
            emitPrepackedV2Snippet(sym, prepackInfo.device_name,
                                   prepackInfo.device_name, row_stride,
                                   col_stride);
        } else if (prepackInfo.use_packed_u32) {
          if (prepack_single_col)
            emitPrepackedU32TileLoadSnippet(sym, ValueSTR(tile_addr),
                                            row_stride);
          else
            emitPrepackedU32Snippet(sym, prepackInfo.device_name, row_stride,
                                    col_stride);
        } else if (sparse_k32_16bit || sparse_k64_16bit) {
          ds << d_indent
             << "  int __sp_row = ((__sp_tid >> 5) * 16) + ((__sp_tid >> 2) & "
                "7);\n";
          ds << d_indent << "  int __sp_byte_col = ((__sp_tid & "
             << (sparse_k32_16bit ? "1" : "3") << ") << 1);\n";
          ds << d_indent << "  uint8_t __sp_b0 = __sp_meta_ptr[__sp_row * ("
             << row_stride << ") + __sp_byte_col * (" << col_stride << ")];\n";
          ds << d_indent << "  uint8_t __sp_b1 = __sp_meta_ptr[__sp_row * ("
             << row_stride << ") + (__sp_byte_col + 1) * (" << col_stride
             << ")];\n";
          ds << d_indent
             << "  uint8_t __sp_b2 = __sp_meta_ptr[(__sp_row + 8) * ("
             << row_stride << ") + __sp_byte_col * (" << col_stride << ")];\n";
          ds << d_indent
             << "  uint8_t __sp_b3 = __sp_meta_ptr[(__sp_row + 8) * ("
             << row_stride << ") + (__sp_byte_col + 1) * (" << col_stride
             << ")];\n";
          ds << d_indent << "  " << sym << " |= (static_cast<" << meta_ty
             << ">(__sp_b0) << 0);\n";
          ds << d_indent << "  " << sym << " |= (static_cast<" << meta_ty
             << ">(__sp_b1) << 8);\n";
          ds << d_indent << "  " << sym << " |= (static_cast<" << meta_ty
             << ">(__sp_b2) << 16);\n";
          ds << d_indent << "  " << sym << " |= (static_cast<" << meta_ty
             << ">(__sp_b3) << 24);\n";
        } else {
          ds << d_indent << "  int __sp_lane = __sp_tid % 32;\n";
          ds << d_indent << "  int __sp_warp = __sp_tid / 32;\n";
          ds << d_indent
             << "  int __sp_row = __sp_warp * 16 + (__sp_lane / 4);\n";
          ds << d_indent
             << "  constexpr int __sp_meta_bytes = " << STR(ssmi.shape.at(2))
             << " / 8;\n";
          ds << d_indent << "  #pragma unroll\n";
          ds << d_indent
             << "  for (int byte_idx = 0; byte_idx < __sp_meta_bytes; "
                "++byte_idx) {\n";
          ds << d_indent << "    uint8_t packed = __sp_meta_ptr[__sp_row * ("
             << row_stride << ") + byte_idx * (" << col_stride << ")];\n";
          ds << d_indent << "    " << sym << " |= (static_cast<" << meta_ty
             << ">(packed) << (8 * byte_idx));\n";
          ds << d_indent << "  }\n";
        }
        ds << d_indent << "}\n";
        ssm.MapDeviceSymbol(InScopeName(sym), sym);
        break;
      }
      auto tile_addr = TileAddr(op.LoadFrom(), false);
      ds << d_indent << elem_ty << "* " << sym << "_smem_ptr = (" << elem_ty
         << "*)(" << ValueSTR(tile_addr) << ");\n";
      [[maybe_unused]] bool frag_is_fp8 =
          ssmi.ty == BaseType::F8_E4M3 || ssmi.ty == BaseType::F8_E5M2 ||
          ssmi.ty == BaseType::F8_UE4M3 || ssmi.ty == BaseType::F8_UE8M0;
      std::string major_order = "WGMMA_MajorOrder::MN_MAJOR";
      std::string cute_major_order = "cute::SM90::GMMA::Major::MN";
      if (ssmi.frag == MMAInfo::FRAG_A) {
        if (ssmi.method == AST::MMAOperation::ROW_ROW ||
            ssmi.method == AST::MMAOperation::ROW_COL) {
          major_order = "WGMMA_MajorOrder::K_MAJOR";
          cute_major_order = "cute::SM90::GMMA::Major::K";
        }
      } else if (ssmi.frag == MMAInfo::FRAG_B) {
        if (ssmi.method == AST::MMAOperation::ROW_ROW ||
            ssmi.method == AST::MMAOperation::COL_ROW) {
          major_order = "WGMMA_MajorOrder::K_MAJOR";
          cute_major_order = "cute::SM90::GMMA::Major::K";
        }
      }
      // Get swizzle value from MMA operation (default NONE / NS)
      auto swizzle_val = op.GetSwizzleMode();
      std::string swizzle_enum;
      std::string sparse_layout_suffix;
      switch (swizzle_val) {
      case SwizMode::NONE: swizzle_enum = "WGMMA_Swizzle::NS"; break;
      case SwizMode::B32: swizzle_enum = "WGMMA_Swizzle::B32"; break;
      case SwizMode::B64: swizzle_enum = "WGMMA_Swizzle::B64"; break;
      case SwizMode::B128: swizzle_enum = "WGMMA_Swizzle::B128"; break;
      default: swizzle_enum = "WGMMA_Swizzle::NS"; break;
      }
      switch (swizzle_val) {
      case SwizMode::NONE: sparse_layout_suffix = "INTER"; break;
      case SwizMode::B32: sparse_layout_suffix = "SW32"; break;
      case SwizMode::B64: sparse_layout_suffix = "SW64"; break;
      case SwizMode::B128: sparse_layout_suffix = "SW128"; break;
      default: sparse_layout_suffix = "INTER"; break;
      }
      // For fp8 sparse FRAG_A, prefer the CUTE SpAtom descriptor path which
      // handles sparse metadata encoding. However, for swizzle modes where
      // the SpAtom's minimum K dimension exceeds the MMA K (e.g., B64/B128
      // with fp8 sparse: SpAtom K=128/256 > MMA K=64), the tile_to_shape
      // would fail. Fall back to wgmma_make_smem_desc in those cases.
      bool sparse_a_needs_cute_desc =
          policy_is_sparse && ssmi.frag == MMAInfo::FRAG_A && frag_is_fp8;
      if (sparse_a_needs_cute_desc &&
          (swizzle_val == SwizMode::B64 || swizzle_val == SwizMode::B128)) {
        sparse_a_needs_cute_desc = false;
      }
      if (sparse_a_needs_cute_desc) {
        auto m_val = STR(ssmi.shape.at(0));
        auto k_val = STR(ssmi.shape.at(2));
        std::string sparse_layout_atom =
            (major_order == "WGMMA_MajorOrder::K_MAJOR")
                ? ("cute::SM90::GMMA::Layout_K_" + sparse_layout_suffix +
                   "_SpAtom")
                : ("cute::SM90::GMMA::Layout_MN_" + sparse_layout_suffix +
                   "_SpAtom");
        ds << d_indent << "auto desc_" << sym
           << "_tensor = cute::make_tensor("
              "cute::make_smem_ptr(cute::recast_ptr<cute::sparse_elem<2, "
           << elem_ty << ">>(" << sym << "_smem_ptr)), "
           << "cute::tile_to_shape(" << sparse_layout_atom << "<" << elem_ty
           << ", 2>{}, cute::make_shape(cute::Int<" << m_val
           << ">{}, cute::Int<" << k_val << ">{})));\n";
        ds << d_indent << "auto desc_" << sym
           << "_obj = cute::SM90::GMMA::make_gmma_desc<" << cute_major_order
           << ">(desc_" << sym << "_tensor);\n";
        ds << d_indent << "uint64_t desc_" << sym << " = desc_" << sym
           << "_obj.desc_;\n";
      } else {
        ds << d_indent << "uint64_t desc_" << sym << " = wgmma_make_smem_desc<"
           << major_order << ", " << swizzle_enum << ">(" << sym
           << "_smem_ptr);\n";
      }
      if (policy_is_sparse && ssmi.frag == MMAInfo::FRAG_A) {
        std::string ref_sym = op.LoadFrom()->RefSymbol();
        if (!ref_sym.empty()) {
          auto mdata_sym_name = ref_sym + "_mdata";
          if (SSTab().IsDeclared(mdata_sym_name)) {
            auto mdata_key = InScopeName(mdata_sym_name + ".data");
            if (ssm.HasDeviceName(mdata_key)) {
              ds << d_indent << "uint8_t* " << sym << "_mdata_ptr = (uint8_t*)"
                 << ssm.DeviceName(mdata_key) << ";\n";
            } else if (ssm.HasDeviceName(InScopeName(mdata_sym_name))) {
              ds << d_indent << "uint8_t* " << sym << "_mdata_ptr = (uint8_t*)"
                 << ssm.DeviceName(InScopeName(mdata_sym_name)) << ";\n";
            }
          } else if (isa<FutureType>(GetSymbolType(ref_sym))) {
            ds << d_indent << "uint8_t* " << sym << "_mdata_ptr = (uint8_t*)"
               << ref_sym << ".mdata();\n";
          }
        }
      }
      ssm.MapDeviceSymbol(InScopeName(sym), sym);
    } break;
    case AST::MMAOperation::Exec: {
      // Detect memory layout based on MMA execution method
      // mma.row.row: both A and B are K_MAJOR (left operand K-major, right
      // operand K-major) mma.row.col: A is K_MAJOR, B is MN_MAJOR (left operand
      // K-major, right operand MN-major)
      int trans_a = 0;
      int trans_b = 0;
      if (op.GetMethod() == AST::MMAOperation::ROW_ROW) {
        trans_a = 0;
        trans_b = 0;
      } else if (op.GetMethod() == AST::MMAOperation::ROW_COL) {
        trans_a = 0;
        trans_b = 1;
      } else if (op.GetMethod() == AST::MMAOperation::COL_ROW) {
        trans_a = 1;
        trans_b = 0;
      } else if (op.GetMethod() == AST::MMAOperation::COL_COL) {
        trans_a = 1;
        trans_b = 1;
      } else {
        choreo_unreachable("Unsupported MMA execution method");
      }
      auto c_sym = AST::FragName(op.ExecOperand(0));
      auto a_sym = AST::FragName(op.ExecOperand(1));
      auto b_sym = AST::FragName(op.ExecOperand(2));
      has_pending_wgmma_finalize = true;
      if (!(IsWarpSpecActive() && bdim_level == ParallelLevel::GROUPx4 &&
            warpspec_wgmma_arrived)) {
        ds << d_indent << "warpgroup_arrive();\n";
        if (IsWarpSpecActive() && bdim_level == ParallelLevel::GROUPx4)
          warpspec_wgmma_arrived = true;
      }
      std::string mma_policy = FCtx(fname).MMAPolicyOfFrag(InScopeName(c_sym));
      std::string cc = SplitStringByDelimiter(mma_policy, "::")[0];
      auto cute_gmma_major_cast = "static_cast<cute::" + cc + "::GMMA::Major>";
      // WGMMA execution using unified template with automatic descriptor
      // selection Operands: C (accum), A, B - result stored in C
      ds << d_indent
         << "// Note: warpgroup_arrive() should be called once before first "
            "WGMMA\n";
      ds << d_indent
         << "// and warpgroup_wait() should be called once after all WGMMAs\n";
      bool policy_is_tn = mma_policy.rfind("_TN") != std::string::npos;
      bool policy_is_sparse = mma_policy.find("SPARSE::") != std::string::npos;
      std::string meta_var;
      if (policy_is_sparse) {
        if (op.ExecOperand(3)) {
          meta_var = AST::FragName(op.ExecOperand(3));
        } else {
          auto& ssmi_a = cgi.GetSymbolMMA(InScopeName(a_sym));
          meta_var = a_sym + "_meta";
          std::string meta_ptr = a_sym + "_mdata_ptr";
          auto k_val = VIInt(ssmi_a.shape.at(2));
          bool policy_is_fp8 = mma_policy.find("E4M3") != std::string::npos ||
                               mma_policy.find("E5M2") != std::string::npos;
          bool fp8_sparse_k64 = policy_is_fp8 && k_val && *k_val == 64;
          bool sparse_k32_16bit = !policy_is_fp8 && k_val && *k_val == 32;
          bool sparse_k64_16bit = !policy_is_fp8 && k_val && *k_val == 64;
          bool meta_64 = true;
          if (k_val) meta_64 = (*k_val > 32) && !fp8_sparse_k64;
          std::string meta_ty = meta_64 ? "uint64_t" : "uint32_t";
          ds << d_indent << meta_ty << " " << meta_var << " = 0;\n";
          ds << d_indent << "{\n";
          ds << d_indent << "  int __sp_tid = threadIdx.x % 128;\n";
          // Detect host prepacked-u32 metadata for the exec site
          bool v2_mode_exec = use_prepack_v2.GetValue();
          auto prepackInfo = resolvePrepackedU32Meta(
              a_sym, use_prepack.GetValue() || v2_mode_exec);
          if (!prepackInfo.use_packed_u32) {
            ds << d_indent
               << "  constexpr int __sp_K = " << STR(ssmi_a.shape.at(2))
               << ";\n";
            ds << d_indent << "  " << meta_ty << " __sp_meta = 0;\n";
          } else if (v2_mode_exec) {
            emitPrepackedV2Snippet(meta_var, prepackInfo.device_name,
                                   prepackInfo.device_name, "128", "1");
          } else {
            emitPrepackedU32Snippet(meta_var, prepackInfo.device_name, "128",
                                    "1");
          }
          if (fp8_sparse_k64) {
            ds << d_indent
               << "  int __sp_row = ((__sp_tid >> 2) & 7) + ((__sp_tid & 1) << "
                  "3) + ((__sp_tid >> 5) << 4);\n";
            ds << d_indent
               << "  int __sp_byte_col = ((__sp_tid >> 1) & 1) << 2;\n";
            ds << d_indent << "  #pragma unroll\n";
            ds << d_indent
               << "  for (int byte_idx = 0; byte_idx < 4; ++byte_idx) {\n";
            ds << d_indent << "    uint8_t packed = " << meta_ptr
               << "[__sp_row * (__sp_K / 8) + (__sp_byte_col + byte_idx)];\n";
            ds << d_indent << "    __sp_meta |= (static_cast<" << meta_ty
               << ">(packed) << (8 * byte_idx));\n";
            ds << d_indent << "  }\n";
          } else if (sparse_k32_16bit || sparse_k64_16bit) {
            ds << d_indent
               << "  int __sp_row = ((__sp_tid >> 5) * 16) + ((__sp_tid >> 2) "
                  "& 7);\n";
            ds << d_indent << "  int __sp_byte_col = ((__sp_tid & "
               << (sparse_k32_16bit ? "1" : "3") << ") << 1);\n";
            ds << d_indent << "  uint8_t __sp_b0 = " << meta_ptr
               << "[__sp_row * (__sp_K / 8) + __sp_byte_col];\n";
            ds << d_indent << "  uint8_t __sp_b1 = " << meta_ptr
               << "[__sp_row * (__sp_K / 8) + (__sp_byte_col + 1)];\n";
            ds << d_indent << "  uint8_t __sp_b2 = " << meta_ptr
               << "[(__sp_row + 8) * (__sp_K / 8) + __sp_byte_col];\n";
            ds << d_indent << "  uint8_t __sp_b3 = " << meta_ptr
               << "[(__sp_row + 8) * (__sp_K / 8) + (__sp_byte_col + 1)];\n";
            ds << d_indent << "  __sp_meta |= (static_cast<" << meta_ty
               << ">(__sp_b0) << 0);\n";
            ds << d_indent << "  __sp_meta |= (static_cast<" << meta_ty
               << ">(__sp_b1) << 8);\n";
            ds << d_indent << "  __sp_meta |= (static_cast<" << meta_ty
               << ">(__sp_b2) << 16);\n";
            ds << d_indent << "  __sp_meta |= (static_cast<" << meta_ty
               << ">(__sp_b3) << 24);\n";
          } else {
            ds << d_indent << "  int __sp_lane = __sp_tid % 32;\n";
            ds << d_indent << "  int __sp_warp = __sp_tid / 32;\n";
            ds << d_indent
               << "  int __sp_row = __sp_warp * 16 + (__sp_lane / 4);\n";
            ds << d_indent << "  constexpr int __sp_meta_bytes = __sp_K / 8;\n";
            ds << d_indent << "  #pragma unroll\n";
            ds << d_indent
               << "  for (int byte_idx = 0; byte_idx < __sp_meta_bytes; "
                  "++byte_idx) {\n";
            ds << d_indent << "    uint8_t packed = " << meta_ptr
               << "[__sp_row * __sp_meta_bytes + byte_idx];\n";
            ds << d_indent << "    __sp_meta |= (static_cast<" << meta_ty
               << ">(packed) << (8 * byte_idx));\n";
            ds << d_indent << "  }\n";
          }
          ds << d_indent << "  " << meta_var << " = __sp_meta;\n";
          ds << d_indent << "}\n";
        }
      }
      auto& ssmi_c = cgi.GetSymbolMMA(InScopeName(c_sym));
      std::string acc_ty = NameBaseType(ssmi_c.ty);
      auto scale_a_expr = op.HasScale() ? ExprSTR(op.ScaleA(), false) : "";
      auto scale_b_expr = op.HasScale() ? ExprSTR(op.ScaleB(), false) : "";
      auto* explicit_scale_info =
          op.HasScale() ? nullptr : CurrentExplicitScaleAccumForFrag(c_sym);
      const auto* hoisted_scale_info = CurrentHoistedScaleAccum();
      bool use_hoisted_scale_accum = op.HasScale() && hoisted_scale_info;
      bool use_explicit_scale_accum = explicit_scale_info != nullptr;

      if (op.HasScale() || use_explicit_scale_accum) {
        // dtype of accu: s32, f16, f32 (f16 => u32)
        auto acc_dtype = ssmi.ty;
        ValueItem frag_len = ssmi.shape[1] / sbe::nu(2);
        if (ssmi.ty == BaseType::F16) {
          acc_dtype = BaseType::U32;
          frag_len = frag_len / sbe::nu(2);
        }
        reg_num_d = *VIInt(frag_len);

        if (op.HasScale() && !use_hoisted_scale_accum) {
          ds << d_indent << NameBaseType(acc_dtype) << " " << c_sym
             << "_scale_frag[" << reg_num_d << "];\n";
          ds << d_indent << "memset(" << c_sym << "_scale_frag, 0, sizeof("
             << c_sym << "_scale_frag));\n";
        }
      }
      ds << d_indent << "cute::" << mma_policy << "<";
      if (!policy_is_tn) {
        ds << cute_gmma_major_cast << "(" << trans_a << "), "
           << cute_gmma_major_cast << "(" << trans_b << ")";
      }
      ds << ">::fma("
         << "desc_" << a_sym << ", desc_" << b_sym;
      for (size_t i = 0; i < reg_num_d; ++i) {
        if (use_explicit_scale_accum)
          ds << ", " << explicit_scale_info->scale_frag_name << "[" << i << "]";
        else if (op.HasScale())
          ds << ", "
             << (use_hoisted_scale_accum ? hoisted_scale_info->scale_frag_name
                                         : c_sym + "_scale_frag")
             << "[" << i << "]";
        else
          ds << ", " << ExprSTR(frag, false) << "[" << i << "]";
      }
      if (policy_is_sparse) ds << ", " << meta_var;
      ds << ");\n";

      if (op.HasScale() && !use_hoisted_scale_accum) {
        std::string dim_n = STR(ssmi_c.shape.at(1));
        auto scale_a_strides = GenStrides(op.ScaleA());
        auto sa_sty3 = GetSpannedType(NodeType(*op.ScaleA()));
        auto sa_shape3 = sa_sty3->GetShape();
        bool sa_transposed3 =
            VIIsInt(sa_shape3.ValueAt(0)) && *VIInt(sa_shape3.ValueAt(0)) == 1;
        std::string scale_a_ld = sa_transposed3
                                     ? ValueSTR(scale_a_strides.back())
                                     : ValueSTR(scale_a_strides.front());
        auto scale_a_name = c_sym + "_scale_a_ptr";
        auto scale_a_valid_rows_name = c_sym + "_scale_a_valid_rows";
        auto scale_b_name = c_sym + "_scale_b_val";
        auto scale_a_valid_rows_expr = GenScaleValidRowsExpr(op.ScaleA());
        if (!hoist_offset || !active_hoisted_scale_decls.count(scale_a_name)) {
          ds << d_indent << "float* " << scale_a_name << " = (float*)("
             << scale_a_expr << ");\n";
          ds << d_indent << "int " << scale_a_valid_rows_name << " = "
             << scale_a_valid_rows_expr << ";\n";
          ds << d_indent << "float " << scale_b_name << " = "
             << "static_cast<float>(" << scale_b_expr << ");\n";
        }
        EmitScaleAccumCall(acc_ty, dim_n, ExprSTR(frag, false),
                           c_sym + "_scale_frag", scale_a_name, scale_a_ld,
                           scale_a_valid_rows_name, scale_b_name);
      }
    } break;
    case AST::MMAOperation::Scale: {
      auto c_sym = AST::FragName(op.ScaleAccumulator());
      if (auto* info = CurrentExplicitScaleAccumForFrag(c_sym)) {
        auto resolved_scale_a = op.ScaleA();
        if (resolved_scale_a && resolved_scale_a->NoOperation()) {
          auto ref_sym = resolved_scale_a->RefSymbol();
          auto it = live_chunk_aliases.find(ref_sym);
          if (it == live_chunk_aliases.end())
            it = live_chunk_aliases.find(UnScopedName(ref_sym));
          if (it == live_chunk_aliases.end()) {
            auto expr_sym = ExprSTR(op.ScaleA(), false);
            it = live_chunk_aliases.find(expr_sym);
          }
          if (it != live_chunk_aliases.end()) resolved_scale_a = it->second;
        }
        auto scale_a_valid_rows_expr = GenScaleValidRowsExpr(resolved_scale_a);
        if (scale_a_valid_rows_expr == "0x3fffffff") {
          Warning(n.LOC(), "mma.scale could not determine valid row count for "
                           "scale_a; assuming all rows are valid. If the scale "
                           "buffer has fewer rows than the MMA fragment, this "
                           "may cause out-of-bounds access.");
        }
        ds << d_indent << "float* " << info->scale_a_name << " = (float*)("
           << info->scale_a_expr << ");\n";
        ds << d_indent << "int " << info->scale_a_valid_rows_name << " = "
           << scale_a_valid_rows_expr << ";\n";
        ds << d_indent << "float " << info->scale_b_name
           << " = static_cast<float>(" << info->scale_b_expr << ");\n";
        EmitScaleAccumCall(
            info->acc_ty, info->dim_n, ExprSTR(op.ScaleAccumulator(), false),
            info->scale_frag_name, info->scale_a_name, info->scale_a_ld,
            info->scale_a_valid_rows_name, info->scale_b_name);
        info->consumed = true;
        break;
      }
      Error1(
          n.LOC(),
          "mma.scale requires a preceding plain WGMMA exec in the same scope.");
      return false;
    } break;
    case AST::MMAOperation::Store: {
      if (has_pending_wgmma_finalize) EmitWGMMAFinalize(ds, d_indent);
      auto ca = op.StoreTo();
      auto t_sym = ca->data->name;
      auto ty = GetSymbolType(t_sym);
      auto f_sty = GetSpannedType(ty);
      auto accum_type = ssmi.ty;
      std::string buf_expr = isa<FutureType>(ty) ? t_sym + ".data()" : t_sym;
      if (ca->indices != nullptr) {
        if (auto array_ty = dyn_cast<ArrayType>(ty);
            array_ty && CCtx().MemReuse()) {
          std::string array_idx;
          auto subscriptions = ca->indices->AllValues();
          const ValueList& array_sizes = array_ty->Dimensions();
          for (size_t i = 0; i < subscriptions.size(); ++i) {
            if (array_idx.empty())
              array_idx = ExprSTR(subscriptions[i], IsHost());
            else
              array_idx = "(" + array_idx + ")*" + ValueSTR(array_sizes[i]) +
                          "+" + ExprSTR(subscriptions[i], IsHost());
          }
          std::string elem_count =
              ValueSTR(f_sty->GetShape().ElementCountValue());
          buf_expr += " + (" + array_idx + ")*(" + elem_count + ")";
        } else {
          for (auto expr : ca->indices->AllValues())
            buf_expr += "[" + ExprSTR(expr, IsHost()) + "]";
        }
      }
      const auto f_mds = GenTensorDecl(
          RemoveSuffix(t_sym, ".data()"), buf_expr, f_sty->GetStorage(),
          f_sty->ElementType(), ca->GetBlockShape(), false,
          ValueSTR(GenOffset(ca)), ValueSTR(GenStrides(ca), false, true));
      ds << f_mds.second;
      std::string DIM_N_STR = STR(ssmi.shape.at(1));
      std::string CUTE_WGMMA_ATOM =
          "CUTE_WGMMA_M" + STR(ssmi.shape.at(0)) + "K" + STR(ssmi.shape.at(2));
      const bool store_trans = op.StoreIsTranspose();
      auto to_shape = ca->GetBlockShape();
      auto mc_dim_m = ssmi.shape.at(0);
      auto ca_dim_m = to_shape.ValueAt(0);

      // --- Mask determination ---
      bool need_m_mask = false;
      bool need_n_mask = false;
      std::string row_guard_expr;
      std::string col_guard_expr;

      if (op.StoreHasExplicitMask()) {
        need_m_mask = true;
        row_guard_expr = ExprSTR(op.StoreRowMask(), IsHost());
        if (op.StoreColMask()) {
          need_n_mask = true;
          col_guard_expr = ExprSTR(op.StoreColMask(), IsHost());
        }
      } else {
        // Auto-detection: infer mask from tensor shapes and access patterns

        // Row (M) mask - Source 1: block shape < MMA shape or runtime
        need_m_mask = !VIIsInt(ca_dim_m) || *VIInt(ca_dim_m) < *VIInt(mc_dim_m);
        if (need_m_mask) {
          row_guard_expr = ValueSTR(ca_dim_m);
          Warning(
              n.LOC(),
              "mma.store destination block shape (row=" + ValueSTR(ca_dim_m) +
                  ") is smaller than the MMA fragment; implicit row masking "
                  "is applied. Consider using 'mma.store.mask' with an "
                  "explicit guard instead.");
        }

        // Row (M) mask - Source 2: parent M not aligned with tile M
        if (!need_m_mask && f_sty->GetStorage() == Storage::GLOBAL &&
            !store_trans && !use_stmatrix) {
          auto parent_sym = ca->RefSymbol();
          auto parent_ty = GetSpannedType(GetSymbolType(parent_sym));
          if (parent_ty && parent_ty->GetShape().Rank() >= 2) {
            auto parent_m = parent_ty->GetShape().ValueAt(0);
            auto tile_m = to_shape.ValueAt(0);
            if (VIIsInt(tile_m)) {
              auto tile_m_val = *VIInt(tile_m);
              bool m_aligned =
                  VIIsInt(parent_m) && (*VIInt(parent_m) % tile_m_val == 0);
              if (!m_aligned) {
                std::string row_idx_str;
                for (size_t i = 0; i < ca->OpCount(); ++i) {
                  const auto& sop = ca->OpAt(i);
                  if (isa<AST::SOP::Tiling>(sop) ||
                      isa<AST::SOP::TileAt>(sop) ||
                      isa<AST::SOP::SubSpan>(sop)) {
                    auto idx = sop->GetIndices();
                    if (idx && idx->Count() >= 2) {
                      row_idx_str = ExprSTR(idx->ValueAt(0), IsHost());
                    }
                  }
                }
                if (!row_idx_str.empty()) {
                  need_m_mask = true;
                  row_guard_expr = "((int)" + ValueSTR(parent_m) + " - (int)(" +
                                   row_idx_str + ") * " + STR(tile_m_val) + ")";
                  VST_DEBUG(dbgs() << n.LOC()
                                   << ": auto-detected row masking for "
                                      "mma.store: parent M dimension ("
                                   << ValueSTR(parent_m)
                                   << ") may not be aligned with tile M ("
                                   << STR(tile_m_val) << ").\n");
                }
              }
            }
          }
        }

        // Column (N) mask: parent N not aligned with tile N
        if (f_sty->GetStorage() == Storage::GLOBAL && !store_trans &&
            !use_stmatrix) {
          auto parent_sym = ca->RefSymbol();
          auto parent_ty = GetSpannedType(GetSymbolType(parent_sym));
          if (parent_ty && parent_ty->GetShape().Rank() >= 2) {
            auto parent_n =
                parent_ty->GetShape().ValueAt(parent_ty->GetShape().Rank() - 1);
            auto tile_n = to_shape.ValueAt(to_shape.Rank() - 1);
            auto tile_n_val = *VIInt(tile_n);
            bool n_aligned =
                VIIsInt(parent_n) && (*VIInt(parent_n) % tile_n_val == 0);
            if (!n_aligned) {
              need_n_mask = true;
              std::string col_idx_str;
              for (size_t i = 0; i < ca->OpCount(); ++i) {
                const auto& sop = ca->OpAt(i);
                if (isa<AST::SOP::Tiling>(sop) || isa<AST::SOP::TileAt>(sop) ||
                    isa<AST::SOP::SubSpan>(sop)) {
                  auto idx = sop->GetIndices();
                  if (idx && idx->Count() >= 2) {
                    size_t col_dim = idx->Count() - 1;
                    col_idx_str = ExprSTR(idx->ValueAt(col_dim), IsHost());
                  }
                }
              }
              if (!col_idx_str.empty()) {
                col_guard_expr = "((int)" + ValueSTR(parent_n) + " - (int)(" +
                                 col_idx_str + ") * " + STR(tile_n_val) + ")";
                VST_DEBUG(dbgs() << n.LOC()
                                 << ": auto-detected column masking for "
                                    "mma.store: parent N dimension ("
                                 << ValueSTR(parent_n)
                                 << ") may not be aligned with tile N ("
                                 << STR(tile_n_val) << ").\n");
              } else {
                need_n_mask = false;
              }
            }
          }
        }
      }

      if (need_m_mask)
        assert(!use_stmatrix && !store_trans &&
               "masked store only supports non-transpose store to "
               "global memory. Support for more cases can be added if needed.");

      // --- Emit store ---
      if (use_stmatrix) {
        bool use_stmatrix_f32_bf16 = accum_type == BaseType::F32 &&
                                     f_sty->ElementType() == BaseType::BF16 &&
                                     !store_trans;
        bool use_stmatrix_f32_bf16_trans =
            accum_type == BaseType::F32 &&
            f_sty->ElementType() == BaseType::BF16 && store_trans;
        ds << d_indent
           << (use_stmatrix_f32_bf16
                   ? "store_fragment_d_stmatrix_f32_bf16<"
                   : (use_stmatrix_f32_bf16_trans
                          ? "store_fragment_d_stmatrix_trans_f32_bf16<"
                          : (store_trans ? "store_fragment_d_stmatrix_trans<"
                                         : "store_fragment_d_stmatrix<")))
           << CUTE_WGMMA_ATOM << ", " << DIM_N_STR << ">(" << f_mds.first
           << ", "
           << "reinterpret_cast<" << NameBaseType(accum_type) << "*>("
           << ExprSTR(frag, false) << "));\n";
      } else {
        std::string frag_cast = std::string("reinterpret_cast<") +
                                NameBaseType(accum_type) + "*>(" +
                                ExprSTR(frag, false) + ")";
        std::string full_call =
            (store_trans ? "store_fragment_d_trans<" : "store_fragment_d<") +
            CUTE_WGMMA_ATOM + ", " + DIM_N_STR + ">(" + f_mds.first + ", " +
            frag_cast + ");\n";
        std::string mma_m_str = ValueSTR(mc_dim_m);

        if (need_m_mask && need_n_mask) {
          ds << d_indent << "{ int __rg = " << row_guard_expr
             << "; int __cg = " << col_guard_expr << ";\n";
          ds << d_indent << "  if (__rg >= " << mma_m_str
             << " && __cg >= " << DIM_N_STR << ")\n";
          ds << d_indent << "    " << full_call;
          ds << d_indent << "  else\n";
          ds << d_indent << "    store_fragment_d_mask_row_col<"
             << CUTE_WGMMA_ATOM << ", " << DIM_N_STR << ">(" << f_mds.first
             << ", " << frag_cast << ", __rg, __cg);\n";
          ds << d_indent << "}\n";
        } else if (need_m_mask) {
          ds << d_indent << "{ int __rg = " << row_guard_expr << ";\n";
          ds << d_indent << "  if (__rg >= " << mma_m_str << ")\n";
          ds << d_indent << "    " << full_call;
          ds << d_indent << "  else\n";
          ds << d_indent << "    store_fragment_d_mask_row<" << CUTE_WGMMA_ATOM
             << ", " << DIM_N_STR << ">(" << f_mds.first << ", " << frag_cast
             << ", __rg);\n";
          ds << d_indent << "}\n";
        } else if (need_n_mask) {
          ds << d_indent << "{ int __cg = " << col_guard_expr << ";\n";
          ds << d_indent << "  if (__cg >= " << DIM_N_STR << ")\n";
          ds << d_indent << "    " << full_call;
          ds << d_indent << "  else\n";
          ds << d_indent << "    store_fragment_d_mask_col<" << CUTE_WGMMA_ATOM
             << ", " << DIM_N_STR << ">(" << f_mds.first << ", " << frag_cast
             << ", __cg);\n";
          ds << d_indent << "}\n";
        } else {
          ds << d_indent << full_call;
        }
      }
    } break;
    default: break;
    }
  } else if (FCtx(fname).FragIsWMMA(scoped_frag_sym)) {
    auto FragSTR = [](MMAInfo::Fragment frag) {
      switch (frag) {
      case MMAInfo::FRAG_A: return "matrix_a";
      case MMAInfo::FRAG_B: return "matrix_b";
      case MMAInfo::FRAG_C: return "accumulator";
      default: choreo_unreachable("unsupported frag."); break;
      }
      return "";
    };
    switch (op.Tag()) {
    case AST::MMAOperation::Fill: {
      auto& ssmi = cgi.GetSymbolMMA(InScopeName(sym));
      auto sty = GetSpannedType(GetSymbolType(sym));
      assert(sty);
      ptr<ArrayType> aty = nullptr;
      if (op.FillingArrayDims())
        aty = cast<ArrayType>(NodeType(*op.FillingArrayDims()));
      if (op.FillingIsDecl()) {
        ds << d_indent
           << "nvcuda::wmma::fragment<nvcuda::wmma::" << FragSTR(ssmi.frag)
           << ", ";
        ds << ValueSTR(ssmi.shape) << ", " << NameBaseType(ssmi.ty) << "> "
           << sym;
        if (aty)
          for (const auto& dim : aty->Dimensions())
            ds << "[" << ValueSTR(dim) << "]";
        ds << ";\n";
        ssm.MapDeviceSymbol(InScopeName(sym), sym);
      }
      if (aty && !AST::FragIsArrayElem(frag)) {
        std::string indices;
        TraverseWholeArrayBegin(aty->Dimensions(), indices);
        ds << d_indent << "nvcuda::wmma::fill_fragment(" << sym << indices
           << ", (" << NameBaseType(ssmi.ty) << ")"
           << ExprSTR(op.FillingValue(), false) << ");\n";
        TraverseWholeArrayEnd(aty->Dimensions());
      } else {
        ds << d_indent << "nvcuda::wmma::fill_fragment(" << ExprSTR(frag, false)
           << ", (" << NameBaseType(ssmi.ty) << ")"
           << ExprSTR(op.FillingValue(), false) << ");\n";
      }
      ++fill_cnt;
    } break;
    case AST::MMAOperation::Load: {
      auto& ssmi = cgi.GetSymbolMMA(InScopeName(sym));
      auto sty = GetSpannedType(GetSymbolType(sym));
      auto fty = GetSpannedType(GetSymbolType(op.LoadFrom()->RefSymbol()));
      if (ssmi.frag == MMAInfo::FRAG_A || ssmi.frag == MMAInfo::FRAG_B) {
        ds << d_indent
           << "nvcuda::wmma::fragment<nvcuda::wmma::" << FragSTR(ssmi.frag)
           << ", ";
        std::string wmma_major;
        if (ssmi.frag == MMAInfo::FRAG_A) {
          switch (ssmi.method) {
          case AST::MMAOperation::ExecMethod::ROW_COL:
          case AST::MMAOperation::ExecMethod::ROW_ROW:
            wmma_major = "nvcuda::wmma::row_major";
            break;
          case AST::MMAOperation::ExecMethod::COL_COL:
          case AST::MMAOperation::ExecMethod::COL_ROW:
            wmma_major = "nvcuda::wmma::col_major";
            break;
          default: choreo_unreachable("invalid MMA execution method");
          }
        } else if (ssmi.frag == MMAInfo::FRAG_B) {
          switch (ssmi.method) {
          case AST::MMAOperation::ExecMethod::ROW_COL:
          case AST::MMAOperation::ExecMethod::COL_COL:
            wmma_major = "nvcuda::wmma::row_major";
            break;
          case AST::MMAOperation::ExecMethod::ROW_ROW:
          case AST::MMAOperation::ExecMethod::COL_ROW:
            wmma_major = "nvcuda::wmma::col_major";
            break;
          default: choreo_unreachable("invalid MMA execution method");
          }
        }
        ds << ValueSTR(ssmi.shape) << ", " << NameBaseType(ssmi.ty) << ", "
           << wmma_major << "> " << sym << ";\n";

        ds << d_indent << "nvcuda::wmma::load_matrix_sync(" << sym << ", "
           << ValueSTR(TileAddr(op.LoadFrom(), false)) << ", "
           << ValueSTR(fty->GetShape().ValueAt(fty->GetShape().Rank() - 1))
           << ");\n";
        ssm.MapDeviceSymbol(InScopeName(sym), sym);
      } else if (ssmi.frag == MMAInfo::FRAG_C) {
        ds << d_indent
           << "nvcuda::wmma::fragment<nvcuda::wmma::" << FragSTR(ssmi.frag)
           << ", ";
        ds << ValueSTR(ssmi.shape) << ", " << NameBaseType(ssmi.ty) << "> "
           << sym << ";\n";
        ds << d_indent << "nvcuda::wmma::load_matrix_sync(" << sym << ", "
           << ExprSTR(op.LoadFrom(), false) << ", "
           << ValueSTR(fty->GetShape().ValueAt(fty->GetShape().Rank() - 1))
           << ", nvcuda::wmma::mem_row_major);\n";
      } else {
        choreo_unreachable("unexpect MMA frag");
      }
    } break;
    case AST::MMAOperation::Exec: {
      ds << d_indent << "nvcuda::wmma::mma_sync("
         << ExprSTR(op.ExecOperand(0), false) << ", "
         << ExprSTR(op.ExecOperand(1), false) << ", "
         << ExprSTR(op.ExecOperand(2), false) << ", "
         << ExprSTR(op.ExecOperand(0), false) << ");\n";
    } break;
    case AST::MMAOperation::Scale:
      Error1(n.LOC(),
             "mma.scale is only supported for WGMMA on the cute target.");
      return false;
    case AST::MMAOperation::Store: {
      auto tty = GetSpannedType(GetSymbolType(op.StoreTo()->RefSymbol()));
      ds << d_indent << "nvcuda::wmma::store_matrix_sync("
         << ValueSTR(TileAddr(op.StoreTo(), false)) << ", "
         << ExprSTR(op.StoreFrom(), false) << ", "
         << ValueSTR(tty->GetShape().ValueAt(1)) << ", "
         << (op.StoreIsTranspose() ? "nvcuda::wmma::mem_col_major"
                                   : "nvcuda::wmma::mem_row_major")
         << ");\n";
    } break;
    default: break;
    }
  } else if (FCtx(fname).FragIsCTMMA(scoped_frag_sym)) {
    // CUTE MMA api name in choreo
    auto GetMMAAtomName = [](MMAInfo& ssmi) -> std::string {
      std::string CUTE_MMA_ATOM = "CUTE_MMA_M" + STR(ssmi.shape.at(0)) + "N" +
                                  STR(ssmi.shape.at(1)) + "K" +
                                  STR(ssmi.shape.at(2));
      return CUTE_MMA_ATOM;
    };
    // special case of reg num
    auto RegNumOf8x8x4 = [](const ValueList& shape, BaseType bt,
                            MMAInfo::Fragment f, size_t& reg_num) {
      if (sbe::ceq(shape[0], sbe::nu(8)) && sbe::ceq(shape[1], sbe::nu(8)) &&
          sbe::ceq(shape[2], sbe::nu(4))) {
        if (f == MMAInfo::FRAG_C) {
          if (bt == BaseType::F16)
            reg_num = 4;
          else if (bt == BaseType::F32)
            reg_num = 8;
          else
            assert(false && "unexpect mma config!");
        } else if (f == MMAInfo::FRAG_A || f == MMAInfo::FRAG_B) {
          assert(bt == BaseType::F16);
          reg_num = 2;
        }
      }
    };
    // n is not wmma. Inline PTX.
    switch (op.Tag()) {
    case AST::MMAOperation::Fill: {
      auto& ssmi = cgi.GetSymbolMMA(InScopeName(sym));
      assert(ssmi.ty != BaseType::UNKNOWN);
      auto sty = GetSpannedType(GetSymbolType(sym));
      assert(sty);
      reg_num_d = GetRegNumOfFrag(sty->GetShape().ValueAt(0),
                                  sty->GetShape().ValueAt(1));
      bool use_uint32 = false;
      UseUint32Reg(use_uint32, reg_num_d, ssmi.ty);
      RegNumOf8x8x4(ssmi.shape, ssmi.ty, MMAInfo::FRAG_C, reg_num_d);

      ptr<ArrayType> aty = nullptr;
      if (op.FillingArrayDims())
        aty = cast<ArrayType>(NodeType(*op.FillingArrayDims()));

      if (op.FillingIsDecl()) {
        ds << d_indent << (use_uint32 ? "uint32_t" : NameBaseType(ssmi.ty))
           << " " << sym;
        if (aty)
          for (const auto& dim : aty->Dimensions())
            ds << "[" << ValueSTR(dim) << "]";
        ds << "[" << reg_num_d << "];\n";
        ssm.MapDeviceSymbol(InScopeName(sym), sym);
      }
      // TODO: #pragma unroll
      // if ubound is large, may lead to low performance
      std::string scalar_init_val =
          ExprCastSTR(op.FillingValue(), std::nullopt, ssmi.ty,
                      GetBaseType(*op.FillingValue()->GetType()), false);
      std::string frag_iv_str = "__frag_init_val" + std::to_string(fill_cnt);
      if (use_uint32) {
        std::string temp = "__fiv_temp" + std::to_string(fill_cnt);
        ds << d_indent << "uint32_t " << frag_iv_str << " = broadcast_to_u32("
           << scalar_init_val << ");\n";
      } else {
        ds << d_indent << NameBaseType(ssmi.ty) << " " << frag_iv_str << " = "
           << scalar_init_val << ";\n";
      }

      if (aty && !AST::FragIsArrayElem(frag)) {
        // need to fill the whole fragment array
        std::string indices;
        TraverseWholeArrayBegin(aty->Dimensions(), indices);
        // the loop body
        ds << d_indent << "for (int idx = 0; idx < " << reg_num_d
           << "; ++idx) {\n";
        IncrDeviceIndent();
        ds << d_indent << sym << indices << "[idx] = " << frag_iv_str << ";\n";
        DecrDeviceIndent();
        ds << d_indent << "}\n";
        TraverseWholeArrayEnd(aty->Dimensions());
      } else {
        ds << d_indent << "for (int idx = 0; idx < " << reg_num_d
           << "; ++idx)\n";
        IncrDeviceIndent();
        ds << d_indent << ExprSTR(frag, false) << "[idx] = " << frag_iv_str
           << ";\n";
        DecrDeviceIndent();
      }
      ++fill_cnt;
    } break;
    case AST::MMAOperation::Load: {
      auto ca = op.LoadFrom();
      auto f_sym = ca->data->name;
      auto ty = GetSymbolType(f_sym);
      auto f_sty = GetSpannedType(ty);
      std::string buf_expr = isa<FutureType>(ty) ? f_sym + ".data()" : f_sym;
      if (ca->indices != nullptr) {
        if (auto array_ty = dyn_cast<ArrayType>(ty);
            array_ty && CCtx().MemReuse()) {
          std::string array_idx;
          auto subscriptions = ca->indices->AllValues();
          const ValueList& array_sizes = array_ty->Dimensions();
          for (size_t i = 0; i < subscriptions.size(); ++i) {
            if (array_idx.empty())
              array_idx = ExprSTR(subscriptions[i], IsHost());
            else
              array_idx = "(" + array_idx + ")*" + ValueSTR(array_sizes[i]) +
                          "+" + ExprSTR(subscriptions[i], IsHost());
          }
          std::string elem_count =
              ValueSTR(f_sty->GetShape().ElementCountValue());
          buf_expr += " + (" + array_idx + ")*(" + elem_count + ")";
        } else {
          for (auto expr : ca->indices->AllValues())
            buf_expr += "[" + ExprSTR(expr, IsHost()) + "]";
        }
      }
      const auto f_mds = GenTensorDecl(
          RemoveSuffix(f_sym, ".data()"), buf_expr, f_sty->GetStorage(),
          f_sty->ElementType(), ca->GetBlockShape(), false,
          ValueSTR(GenOffset(ca)), ValueSTR(GenStrides(ca), false, true));
      ds << f_mds.second;
      auto ssmi = cgi.GetSymbolMMA(InScopeName(sym));
      // Check for dynamic-shaped or misaligned source buffer
      {
        auto parent_sym = ca->RefSymbol();
        auto parent_ty = GetSpannedType(GetSymbolType(parent_sym));
        switch (GetMmaLoadShapeWarningKind(ssmi, parent_ty)) {
        case MmaLoadShapeWarningKind::Dynamic:
          Warning(n.LOC(), "mma.load source buffer '" + parent_sym +
                               "' has dynamic shape; ensure its dimensions are "
                               "divisible by the MMA atom shape to avoid "
                               "out-of-bounds access.");
          break;
        case MmaLoadShapeWarningKind::Misaligned:
          Warning(n.LOC(),
                  "mma.load source buffer '" + parent_sym +
                      "' has shape that does not evenly divide the MMA "
                      "atom; this may cause out-of-bounds shared memory "
                      "access.");
          break;
        case MmaLoadShapeWarningKind::None: break;
        }
      }
      std::string mma_policy = FCtx(fname).MMAPolicyOfFrag(InScopeName(sym));
      bool policy_is_sparse = mma_policy.find("SPARSE") != std::string::npos;
      auto m_val = VIInt(ssmi.shape.at(0));
      auto n_val = VIInt(ssmi.shape.at(1));
      auto k_val = VIInt(ssmi.shape.at(2));
      bool shape_is_m16n8k16 = m_val && n_val && k_val && *m_val == 16 &&
                               *n_val == 8 && *k_val == 16;
      bool shape_is_m16n8k32 = m_val && n_val && k_val && *m_val == 16 &&
                               *n_val == 8 && *k_val == 32;
      bool shape_is_m16n8k64 = m_val && n_val && k_val && *m_val == 16 &&
                               *n_val == 8 && *k_val == 64;
      auto mma_atom_name = [&]() {
        if (policy_is_sparse && shape_is_m16n8k32)
          return std::string("CUTE_MMA_SPARSE_M16N8K32");
        if (policy_is_sparse && shape_is_m16n8k16)
          return std::string("CUTE_MMA_SPARSE_M16N8K16");
        if (policy_is_sparse && shape_is_m16n8k64)
          return std::string("CUTE_MMA_SPARSE_M16N8K64");
        return GetMMAAtomName(ssmi);
      };
      if (ssmi.frag == MMAInfo::FRAG_A || ssmi.frag == MMAInfo::FRAG_B ||
          ssmi.frag == MMAInfo::FRAG_E) {
        std::string frag_suffix;
        if (ssmi.frag == MMAInfo::FRAG_A)
          frag_suffix = "a";
        else if (ssmi.frag == MMAInfo::FRAG_B)
          frag_suffix = "b";
        else
          frag_suffix = "e";

        std::string CUTE_MMA_ATOM = mma_atom_name();
        ds << d_indent << "auto " << sym << " = load_fragment_" << frag_suffix
           << "<" << CUTE_MMA_ATOM << ">(" << f_mds.first << ");\n";

        if (policy_is_sparse && ssmi.frag == MMAInfo::FRAG_A) {
          std::string ref_sym = op.LoadFrom()->RefSymbol();
          if (!ref_sym.empty()) {
            auto mdata_sym_name = ref_sym + "_mdata";
            if (SSTab().IsDeclared(mdata_sym_name)) {
              auto mdata_key = InScopeName(mdata_sym_name + ".data");
              if (ssm.HasDeviceName(mdata_key)) {
                ds << d_indent << "uint8_t* " << sym
                   << "_mdata_ptr = (uint8_t*)" << ssm.DeviceName(mdata_key)
                   << ";\n";
              } else if (ssm.HasDeviceName(InScopeName(mdata_sym_name))) {
                ds << d_indent << "uint8_t* " << sym
                   << "_mdata_ptr = (uint8_t*)"
                   << ssm.DeviceName(InScopeName(mdata_sym_name)) << ";\n";
              }
            } else if (isa<FutureType>(GetSymbolType(ref_sym))) {
              ds << d_indent << "uint8_t* " << sym << "_mdata_ptr = (uint8_t*)"
                 << ref_sym << ".mdata();\n";
            }
          }
        }
      } else if (ssmi.frag == MMAInfo::FRAG_C) {
        auto sty = GetSpannedType(GetSymbolType(sym));
        assert(sty);
        reg_num_d = GetRegNumOfFrag(sty->GetShape().ValueAt(0),
                                    sty->GetShape().ValueAt(1));
        bool use_uint32 = false;
        UseUint32Reg(use_uint32, reg_num_d, ssmi.ty);
        RegNumOf8x8x4(ssmi.shape, ssmi.ty, MMAInfo::FRAG_C, reg_num_d);
        std::string CUTE_MMA_ATOM = mma_atom_name();
        // TODO: add more testcases about load mc & fill mc.
        // TODO: is load mc supported in other xx_mma?
        if (isa<ArrayType>(GetSymbolType(sym))) {
          // only load to a single frag. Need `fill` to decl the frag array!
          ds << d_indent << "load_fragment_d<" << CUTE_MMA_ATOM << ">("
             << f_mds.first << ", "
             << "reinterpret_cast<" << NameBaseType(ssmi.ty) << "*> ("
             << ExprSTR(frag) << "));\n";
        } else {
          // decl + load
          ds << d_indent << (use_uint32 ? "uint32_t" : NameBaseType(ssmi.ty))
             << " " << sym << "[" << reg_num_d << "] ;\n";
          ds << d_indent << "load_fragment_d<" << CUTE_MMA_ATOM << ">("
             << f_mds.first << ", "
             << "reinterpret_cast<" << NameBaseType(ssmi.ty) << "*> (" << sym
             << "));\n";
        }
      } else {
        choreo_unreachable("unexpect MMA frag");
      }
    } break;
    case AST::MMAOperation::Exec: {
      auto c_sym = AST::FragName(op.ExecOperand(0));
      auto a_sym = AST::FragName(op.ExecOperand(1));
      auto b_sym = AST::FragName(op.ExecOperand(2));
      auto e_sym_provided = op.ExecOperand(3);

      std::string mma_policy = FCtx(fname).MMAPolicyOfFrag(InScopeName(c_sym));
      bool policy_is_sparse = mma_policy.find("SPARSE") != std::string::npos;
      std::string meta_var;
      if (policy_is_sparse) {
        if (e_sym_provided) {
          meta_var = AST::FragName(e_sym_provided);
        } else {
          auto& ssmi_a = cgi.GetSymbolMMA(InScopeName(a_sym));
          meta_var = a_sym + "_meta";
          std::string meta_ptr = a_sym + "_mdata_ptr";
          auto k_val_ptr = VIInt(ssmi_a.shape.at(2));
          int k_val = k_val_ptr ? *k_val_ptr : 0;
          std::string meta_ty = (k_val > 64) ? "uint64_t" : "uint32_t";
          ds << d_indent << meta_ty << " " << meta_var << " = 0;\n";
          ds << d_indent << "{\n";
          ds << d_indent << "  int __sp_lane = threadIdx.x & 31;\n";
          ds << d_indent << "  int __sp_group = __sp_lane >> 2;\n";
          ds << d_indent << "  int __sp_tid = __sp_lane & 0x3;\n";
          ds << d_indent
             << "  auto get_nibble = [&](int r, int k4) -> uint32_t {\n";
          ds << d_indent << "    return (uint32_t)" << meta_ptr << "[r * ("
             << (k_val / 4) << ") + k4];\n";
          ds << d_indent << "  };\n";
          if (k_val == 32) {
            ds << d_indent << "  " << meta_var
               << " = (get_nibble(__sp_group, __sp_tid) << 0) | "
                  "(get_nibble(__sp_group + 8, __sp_tid) << 4) | "
                  "(get_nibble(__sp_group, __sp_tid + 4) << 16) | "
                  "(get_nibble(__sp_group + 8, __sp_tid + 4) << 20);\n";
          } else if (k_val == 16) {
            ds << d_indent << "  " << meta_var
               << " = (get_nibble(__sp_group, __sp_tid) << 0) | "
                  "(get_nibble(__sp_group + 8, __sp_tid) << 4);\n";
          } else {
            // Fallback or other shapes
            ds << d_indent << "  for (int k4 = 0; k4 < " << (k_val / 4)
               << "; ++k4) {\n";
            ds << d_indent << "    " << meta_var
               << " |= (get_nibble(__sp_group, k4) << (4 * k4));\n";
            ds << d_indent << "  }\n";
          }
          ds << d_indent << "}\n";
        }
      }

      ds << d_indent << "cute::" << mma_policy << "::fma(";
      for (size_t i = 0; i < reg_num_d; ++i)
        ds << ExprSTR(frag, false) << "[" << i << "], ";
      // TODO: test with mma config except mma.row.col
      auto shape = cgi.GetSymbolMMA(InScopeName(c_sym)).shape;
      auto m = shape[0], n = shape[1], k = shape[2];
      auto a_type = cgi.GetSymbolMMA(InScopeName(a_sym)).ty;
      auto b_type = cgi.GetSymbolMMA(InScopeName(b_sym)).ty;
      size_t reg_num_a = GetRegNumOfFrag(m, k);
      size_t reg_num_b = GetRegNumOfFrag(k, n);
      bool use_uint32 = false;
      UseUint32Reg(use_uint32, reg_num_a, a_type);
      UseUint32Reg(use_uint32, reg_num_b, b_type);
      RegNumOf8x8x4(shape, a_type, MMAInfo::FRAG_A, reg_num_a);
      RegNumOf8x8x4(shape, b_type, MMAInfo::FRAG_B, reg_num_b);

      // Handle sparse A fragment size (reg_num_a is logically for full K)
      if (policy_is_sparse) { reg_num_a /= 2; }

      for (size_t i = 0; i < reg_num_a; ++i) ds << a_sym << "[" << i << "], ";
      for (size_t i = 0; i < reg_num_b; ++i) ds << b_sym << "[" << i << "], ";
      for (size_t i = 0; i < reg_num_d; ++i) {
        ds << ExprSTR(frag, false) << "[" << i << "]";
        if (i != reg_num_d - 1) ds << ", ";
      }
      if (policy_is_sparse) ds << ", " << meta_var << ", 0";
      ds << ");\n";
    } break;
    case AST::MMAOperation::Scale:
      Error1(n.LOC(),
             "mma.scale is only supported for WGMMA on the cute target.");
      return false;
    case AST::MMAOperation::Store: {
      auto ca = op.StoreTo();
      auto f_sym = ca->data->name;
      auto ty = GetSymbolType(f_sym);
      auto f_sty = GetSpannedType(ty);
      auto fca_sty = GetSpannedType(NodeType(*ca));
      std::string buf_expr = isa<FutureType>(ty) ? f_sym + ".data()" : f_sym;
      if (ca->indices != nullptr) {
        if (auto array_ty = dyn_cast<ArrayType>(ty);
            array_ty && CCtx().MemReuse()) {
          std::string array_idx;
          auto subscriptions = ca->indices->AllValues();
          const ValueList& array_sizes = array_ty->Dimensions();
          for (size_t i = 0; i < subscriptions.size(); ++i) {
            if (array_idx.empty())
              array_idx = ExprSTR(subscriptions[i], IsHost());
            else
              array_idx = "(" + array_idx + ")*" + ValueSTR(array_sizes[i]) +
                          "+" + ExprSTR(subscriptions[i], IsHost());
          }
          std::string elem_count =
              ValueSTR(f_sty->GetShape().ElementCountValue());
          buf_expr += " + (" + array_idx + ")*(" + elem_count + ")";
        } else {
          for (auto expr : ca->indices->AllValues())
            buf_expr += "[" + ExprSTR(expr, IsHost()) + "]";
        }
      }
      const auto f_mds =
          GenTensorDecl(RemoveSuffix(f_sym, ".data()"), buf_expr,
                        f_sty->GetStorage(), f_sty->ElementType(),
                        ca->GetBlockShape(), false, ValueSTR(GenOffset(ca)),
                        ValueSTR(fca_sty->GetStrides(), false, true));
      ds << f_mds.second;
      auto ssmi = cgi.GetSymbolMMA(InScopeName(sym));
      std::string mma_policy = FCtx(fname).MMAPolicyOfFrag(InScopeName(sym));
      bool policy_is_sparse = mma_policy.find("SPARSE") != std::string::npos;
      auto m_val = VIInt(ssmi.shape.at(0));
      auto n_val = VIInt(ssmi.shape.at(1));
      auto k_val = VIInt(ssmi.shape.at(2));
      bool shape_is_m16n8k32 = m_val && n_val && k_val && *m_val == 16 &&
                               *n_val == 8 && *k_val == 32;
      bool shape_is_m16n8k16 = m_val && n_val && k_val && *m_val == 16 &&
                               *n_val == 8 && *k_val == 16;
      bool shape_is_m16n8k64 = m_val && n_val && k_val && *m_val == 16 &&
                               *n_val == 8 && *k_val == 64;
      std::string CUTE_MMA_ATOM =
          (policy_is_sparse && shape_is_m16n8k32)   ? "CUTE_MMA_SPARSE_M16N8K32"
          : (policy_is_sparse && shape_is_m16n8k16) ? "CUTE_MMA_SPARSE_M16N8K16"
          : (policy_is_sparse && shape_is_m16n8k64) ? "CUTE_MMA_SPARSE_M16N8K64"
                                                    : GetMMAAtomName(ssmi);
      const bool store_trans_sync = op.StoreIsTranspose();
      std::string frag_cast_sync = std::string("reinterpret_cast<") +
                                   NameBaseType(ssmi.ty) + "*> (" +
                                   ExprSTR(frag, false) + ")";
      std::string full_call_sync =
          (store_trans_sync ? "store_fragment_d_trans<" : "store_fragment_d<") +
          CUTE_MMA_ATOM + ">(" + f_mds.first + ", " + frag_cast_sync + ");\n";

      // Mask determination for mma.sync stores
      bool sync_need_m = false, sync_need_n = false;
      std::string sync_rg, sync_cg;

      if (op.StoreHasExplicitMask()) {
        sync_need_m = true;
        sync_rg = ExprSTR(op.StoreRowMask(), IsHost());
        if (op.StoreColMask()) {
          sync_need_n = true;
          sync_cg = ExprSTR(op.StoreColMask(), IsHost());
        }
        Note(n.LOC(), "mma.store uses explicit mask guard for boundary store.");
      } else if (f_sty->GetStorage() == Storage::GLOBAL && !store_trans_sync) {
        auto parent_sym = ca->RefSymbol();
        auto parent_ty = GetSpannedType(GetSymbolType(parent_sym));
        auto to_shape_sync = ca->GetBlockShape();
        if (parent_ty && parent_ty->GetShape().Rank() >= 2) {
          auto parent_m = parent_ty->GetShape().ValueAt(0);
          auto tile_m = to_shape_sync.ValueAt(0);
          auto parent_n =
              parent_ty->GetShape().ValueAt(parent_ty->GetShape().Rank() - 1);
          auto tile_n = to_shape_sync.ValueAt(to_shape_sync.Rank() - 1);

          auto extract_idx = [&](size_t dim) -> std::string {
            for (size_t i = 0; i < ca->OpCount(); ++i) {
              const auto& sop = ca->OpAt(i);
              if (isa<AST::SOP::Tiling>(sop) || isa<AST::SOP::TileAt>(sop) ||
                  isa<AST::SOP::SubSpan>(sop)) {
                auto idx = sop->GetIndices();
                if (idx && idx->Count() >= 2)
                  return ExprSTR(idx->ValueAt(dim), IsHost());
              }
            }
            return "";
          };

          if (VIIsInt(tile_m)) {
            auto tm = *VIInt(tile_m);
            bool ma = VIIsInt(parent_m) && (*VIInt(parent_m) % tm == 0);
            if (!ma) {
              auto ri = extract_idx(0);
              if (!ri.empty()) {
                sync_need_m = true;
                sync_rg = "((int)" + ValueSTR(parent_m) + " - (int)(" + ri +
                          ") * " + STR(tm) + ")";
                VST_DEBUG(dbgs() << n.LOC()
                                 << ": auto-detected row masking for "
                                    "mma.store: parent M dimension ("
                                 << ValueSTR(parent_m)
                                 << ") may not be aligned with tile M ("
                                 << STR(tm) << ").\n");
              }
            }
          }
          if (VIIsInt(tile_n)) {
            auto tn = *VIInt(tile_n);
            bool na = VIIsInt(parent_n) && (*VIInt(parent_n) % tn == 0);
            if (!na) {
              auto ci = extract_idx(to_shape_sync.Rank() - 1);
              if (!ci.empty()) {
                sync_need_n = true;
                sync_cg = "((int)" + ValueSTR(parent_n) + " - (int)(" + ci +
                          ") * " + STR(tn) + ")";
                VST_DEBUG(dbgs() << n.LOC()
                                 << ": auto-detected column masking for "
                                    "mma.store: parent N dimension ("
                                 << ValueSTR(parent_n)
                                 << ") may not be aligned with tile N ("
                                 << STR(tn) << ").\n");
              }
            }
          }
        }
      }

      auto mma_m_sync = ssmi.shape.at(0);
      if (sync_need_m && sync_need_n) {
        ds << d_indent << "{ int __rg = " << sync_rg
           << "; int __cg = " << sync_cg << ";\n";
        ds << d_indent << "  if (__rg >= " << ValueSTR(mma_m_sync)
           << " && __cg >= 8)\n";
        ds << d_indent << "    " << full_call_sync;
        ds << d_indent << "  else\n";
        ds << d_indent << "    store_fragment_d_mask_row_col<" << CUTE_MMA_ATOM
           << ">(" << f_mds.first << ", " << frag_cast_sync
           << ", __rg, __cg);\n";
        ds << d_indent << "}\n";
      } else if (sync_need_m) {
        ds << d_indent << "{ int __rg = " << sync_rg << ";\n";
        ds << d_indent << "  if (__rg >= " << ValueSTR(mma_m_sync) << ")\n";
        ds << d_indent << "    " << full_call_sync;
        ds << d_indent << "  else\n";
        ds << d_indent << "    store_fragment_d_mask_row<" << CUTE_MMA_ATOM
           << ">(" << f_mds.first << ", " << frag_cast_sync << ", __rg);\n";
        ds << d_indent << "}\n";
      } else if (sync_need_n) {
        ds << d_indent << "{ int __cg = " << sync_cg << ";\n";
        ds << d_indent << "  if (__cg >= 8)\n";
        ds << d_indent << "    " << full_call_sync;
        ds << d_indent << "  else\n";
        ds << d_indent << "    store_fragment_d_mask_col<" << CUTE_MMA_ATOM
           << ">(" << f_mds.first << ", " << frag_cast_sync << ", __cg);\n";
        ds << d_indent << "}\n";
      } else {
        ds << d_indent << full_call_sync;
      }
    } break;
    default: break;
    }
    return "";
  } else {
    choreo_unreachable("unexpect mma type!");
  }
  return true;
}

bool CuteCodeGen::Visit(AST::Rotate& n) {
  if (IsHost())
    choreo_unreachable(
        "rotate is only support in device side(inside parallel-by)!");

  for (auto& id : n.GetIds()) {
    assert(isa<FutureType>(NodeType(*id)) &&
           "only rotating futures are supported.");
    auto ident = cast<AST::Identifier>(id);
    auto scoped_name = InScopeName(ident->name);
    if (!claimed_futs.count(scoped_name)) {
      auto cp_atom = GetCopyAtomName(false, dma_count_);
      ds << d_indent << "AsyncCopyAtom " << cp_atom << "{};\n";
      ds << d_indent << "future " << ident->name << "(\"" << ident->name
         << "\", " << id->LOC().begin.line << ", " << id->LOC().begin.column
         << ");\n";
      ds << d_indent << ident->name << ".set_atom(&" << cp_atom << ");\n";
      ds << d_indent << ident->name << ".set_ring(" << device_fn
         << "__ring__);\n";
      future_count_++;
      ds << d_indent << ident->name << ".id = " << future_count_ << ";\n";
      ++dma_count_;
      claimed_futs.emplace(scoped_name, cp_atom);
      ssm.MapDeviceSymbol(scoped_name, ident->name);
      ssm.MapDeviceSymbol(scoped_name + ".data", ident->name + ".data()");
    }
  }

  ds << d_indent << "choreo::rotate(";
  int i = 0;
  for (auto& id : n.GetIds()) {
    if (i++ > 0) ds << ", ";
    ds << ExprSTR(id, false);
  }
  ds << ");\n";

  return true;
}

bool CuteCodeGen::Visit(AST::Synchronize& n) {
  TraceEachVisit(n);

  switch (n.Resource()) {
  case Storage::GLOBAL:
    hs << h_indent << "cudaDeviceSynchronize();\n";
    hs << h_indent << "verify_device_status();\n";
    break;
  case Storage::SHARED: ds << d_indent << "__syncthreads();\n"; break;
  default:
    choreo_unreachable(
        "unsupported synchronization type: " + STR(n.Resource()) + ".");
  }

  return true;
}

bool CuteCodeGen::Visit(AST::Wait& n) {
  TraceEachVisit(n);

  auto BeginEventCritical = [&]() -> bool {
    if (IsHost()) return false;
    if (IsWarpSpecActive()) return false;
    switch (bdim_level) {
    case ParallelLevel::GROUPx4:
      ds << d_indent << "if (__CHOREO_GROUPX4_SINGLE__) {\n";
      IncrDeviceIndent();
      return true;
    case ParallelLevel::GROUP:
      ds << d_indent << "if (__CHOREO_GROUP_SINGLE__) {\n";
      IncrDeviceIndent();
      return true;
    case ParallelLevel::BLOCK:
      ds << d_indent << "if (__CHOREO_BLOCK_SINGLE__) {\n";
      IncrDeviceIndent();
      return true;
    default: return false;
    }
  };

  auto EndEventCritical = [&](bool guarded) {
    if (!guarded || IsHost()) return;
    DecrDeviceIndent();
    ds << d_indent << "}\n";
    if (IsWarpSpecActive()) return;

    switch (bdim_level) {
    case ParallelLevel::GROUPx4:
      ds << d_indent
         << "cooperative_groups::tiled_partition<128>(cooperative_groups::this_"
            "thread_block()).sync();\n";
      break;
    case ParallelLevel::GROUP: ds << d_indent << "__syncwarp();\n"; break;
    case ParallelLevel::BLOCK: ds << d_indent << "__syncthreads();\n"; break;
    default: break;
    }
  };

  for (auto& t : n.GetTargets()) {
    auto expr = cast<AST::Expr>(t);
    bool is_array_ref = (expr->op == Op::ElemOf);
    auto tty = is_array_ref
                   ? GetSymbolType(AST::GetArrayBaseSymbol(*expr)->name)
                   : NodeType(*t);

    if (isa<FutureType>(tty)) {
      assert(expr->GetSymbol());
      auto name = expr->GetSymbol()->name;
      bool shared_in_block = (IsFutureBlockShared(InScopeName(name)) &&
                              !cooperatives.count(InScopeName(name)));
      if (shared_in_block) {
        ds << d_indent << LevelPred() << " {\n";
        IncrDeviceIndent();
      }
      assert(!IsHost());
      ds << d_indent << ExprSTR(t, false) << ".wait();\n";
      if (shared_in_block) {
        DecrDeviceIndent();
        ds << d_indent << "}\n";
        ds << d_indent << "__syncthreads();\n";
      }
    } else if (auto ety = dyn_cast<EventArrayType>(tty)) {
      if (IsHost())
        choreo_unreachable("yet to support: wait global event in host.");
      switch (ety->GetStorage()) {
      case Storage::GLOBAL:
      case Storage::LOCAL: {
        bool guarded = BeginEventCritical();
        ds << d_indent << "// wait event " << PSTR(t) << "\n";
        ds << d_indent << "while (";
        if (is_array_ref) {
          size_t lvl = GetSubScriptLevel(*expr);
          auto bid = AST::GetArrayBaseSymbol(*expr);
          auto bty =
              cast<EventArrayType>(GetSymbolType(UnScopedName(bid->name)));
          // TODO: "!" same here
          GenerateSubscriptions(ds, "!" + ExprSTR(t, false), " || ",
                                bty->RemainderDimensions(lvl));
        } else
          GenerateSubscriptions(ds, "!" + ExprSTR(t, false), " || ",
                                ety->RemainderDimensions(0));
        ds << "false) continue;\n";
        ds << d_indent << "// reset event " << PSTR(t) << "\n";
        if (is_array_ref) {
          size_t lvl = GetSubScriptLevel(*expr);
          auto bid = AST::GetArrayBaseSymbol(*expr);
          auto bty =
              cast<EventArrayType>(GetSymbolType(UnScopedName(bid->name)));
          GenerateSubscriptions(ds, d_indent + ExprSTR(t, false), " = false;\n",
                                bty->RemainderDimensions(lvl));
        } else
          GenerateSubscriptions(ds, d_indent + ExprSTR(t, false), " = false;\n",
                                ety->RemainderDimensions(0));
        EndEventCritical(guarded);
      } break;
      case Storage::SHARED: {
        std::string base_name;
        if (is_array_ref) {
          auto bid = AST::GetArrayBaseSymbol(*expr);
          base_name = UnScopedName(bid->name);
        } else {
          base_name = UnScopedName(expr->GetSymbol()->name);
        }
        bool is_cluster_event = cluster_trigger_events_.count(base_name) > 0;

        if (is_cluster_event) {
          ds << d_indent << "// wait event(mbarrier) " << PSTR(t)
             << " [cluster-scope]\n";
          // Cluster mbarrier wait_parity + phase tracking must be
          // single-threaded: the phase variable is thread-local and the
          // PTX mbarrier wait is a per-thread spin.  Guard with SINGLE
          // unless the scope already restricts to a single thread.
          bool cluster_wait_guarded =
              IsWarpSpecActive() && bdim_level == ParallelLevel::GROUPx4 &&
              !ScopeAlreadySingleThreadForLevel(ParallelLevel::GROUPx4);
          if (cluster_wait_guarded) {
            ds << d_indent << "if (__CHOREO_GROUPX4_SINGLE__) {\n";
            IncrDeviceIndent();
          }
          if (is_array_ref) {
            std::string bar_expr = ExprSTR(t, false);
            std::string phase_expr =
                base_name + "__phase" + bar_expr.substr(base_name.size());
            ds << d_indent << "choreo::tma_mbarrier_wait_parity(&" << bar_expr
               << ", " << phase_expr << ");\n";
            ds << d_indent << phase_expr << " ^= 1;\n";
          } else {
            GenerateSubscriptions(
                ds,
                d_indent + "choreo::tma_mbarrier_wait_parity(&" +
                    ExprSTR(t, false),
                ", " + base_name + "__phase);\n", ety->RemainderDimensions(0));
            GenerateSubscriptions(ds, d_indent + base_name + "__phase",
                                  " ^= 1;\n", ety->RemainderDimensions(0));
          }
          if (cluster_wait_guarded) {
            DecrDeviceIndent();
            ds << d_indent << "}\n";
          }
        } else {
          ds << d_indent << "// wait event(barrier) " << PSTR(t) << "\n";
          if (is_array_ref) {
            ds << d_indent << ExprSTR(t, false) << ".wait(" << ExprSTR(t, false)
               << ".arrive());\n";
          } else {
            GenerateSubscriptions(ds,
                                  d_indent + ExprSTR(t, false) + ".wait(" +
                                      ExprSTR(t, false) + ".arrive())",
                                  ";\n", ety->RemainderDimensions(0));
          }
        }
      } break;
      default:
        choreo_unreachable("unsupported event array storage '" +
                           STR(ety->GetStorage()) + "'.");
      }
    } else if (auto ety = dyn_cast<EventType>(tty)) {
      if (IsHost())
        choreo_unreachable("yet to support: wait global event in host.");
      switch (ety->GetStorage()) {
      case Storage::GLOBAL:
      case Storage::LOCAL: {
        bool guarded = BeginEventCritical();
        ds << d_indent << "while (" << ExprSTR(t, false)
           << " == false) continue; // spinlock\n";
        if (is_array_ref) {
          ds << d_indent << "// reset event " << PSTR(t) << "\n";
          size_t lvl = GetSubScriptLevel(*expr);
          auto bid = AST::GetArrayBaseSymbol(*expr);
          auto bty =
              cast<EventArrayType>(GetSymbolType(UnScopedName(bid->name)));
          GenerateSubscriptions(ds, d_indent + ExprSTR(t, false), " = false;\n",
                                bty->RemainderDimensions(lvl));
        } else
          ds << d_indent << ExprSTR(t, false) << " = false; // reset event\n";
        EndEventCritical(guarded);
      } break;
      case Storage::SHARED: {
        std::string base_name;
        if (is_array_ref) {
          auto bid = AST::GetArrayBaseSymbol(*expr);
          base_name = UnScopedName(bid->name);
        } else {
          base_name = UnScopedName(expr->GetSymbol()->name);
        }
        bool is_cluster_event = cluster_trigger_events_.count(base_name) > 0;

        if (is_cluster_event) {
          bool cluster_wait_guarded =
              IsWarpSpecActive() && bdim_level == ParallelLevel::GROUPx4 &&
              !ScopeAlreadySingleThreadForLevel(ParallelLevel::GROUPx4);
          if (cluster_wait_guarded) {
            ds << d_indent << "if (__CHOREO_GROUPX4_SINGLE__) {\n";
            IncrDeviceIndent();
          }
          if (is_array_ref) {
            std::string bar_expr = ExprSTR(t, false);
            std::string phase_expr =
                base_name + "__phase" + bar_expr.substr(base_name.size());
            ds << d_indent << "choreo::tma_mbarrier_wait_parity(&" << bar_expr
               << ", " << phase_expr << ");\n";
            ds << d_indent << phase_expr << " ^= 1;\n";
          } else {
            ds << d_indent << "choreo::tma_mbarrier_wait_parity(&"
               << ExprSTR(t, false) << ", " << base_name << "__phase);\n";
            ds << d_indent << base_name << "__phase ^= 1;\n";
          }
          if (cluster_wait_guarded) {
            DecrDeviceIndent();
            ds << d_indent << "}\n";
          }
        } else {
          bool guarded = BeginEventCritical();
          ds << d_indent << ExprSTR(t, false) << ".wait(" << ExprSTR(t, false)
             << ".arrive()); // wait event(barrier)\n";
          EndEventCritical(guarded);
        }
      } break;
      default:
        choreo_unreachable("unsupported event storage '" +
                           STR(ety->GetStorage()) + "'.");
      }
    } else
      choreo_unreachable("unsupported wait target.");
  }

  return true;
}

bool CuteCodeGen::Visit(AST::Break& n) {
  TraceEachVisit(n);
  IndStream() << "break;\n";
  return true;
}

bool CuteCodeGen::Visit(AST::Continue& n) {
  TraceEachVisit(n);
  IndStream() << "continue;\n";
  return true;
}

bool CuteCodeGen::Visit(AST::Yield& n) {
  TraceEachVisit(n);
  IndStream() << "return;\n";
  return true;
}

bool CuteCodeGen::Visit(AST::Trigger& n) {
  TraceEachVisit(n);

  auto SumRecentTMATxBytesExpr = [&]() -> std::string {
    if (recent_tma_tx_bytes.empty()) return "1";
    std::ostringstream oss;
    bool first = true;
    for (const auto& expr : recent_tma_tx_bytes) {
      if (!first) oss << " + ";
      first = false;
      oss << "(" << expr << ")";
    }
    return oss.str();
  };

  auto BeginEventCritical = [&]() -> bool {
    if (IsHost()) return false;
    if (IsWarpSpecActive()) return false;
    switch (bdim_level) {
    case ParallelLevel::GROUPx4:
      ds << d_indent << "if (__CHOREO_GROUPX4_SINGLE__) {\n";
      IncrDeviceIndent();
      return true;
    case ParallelLevel::GROUP:
      ds << d_indent << "if (__CHOREO_GROUP_SINGLE__) {\n";
      IncrDeviceIndent();
      return true;
    case ParallelLevel::BLOCK:
      ds << d_indent << "if (__CHOREO_BLOCK_SINGLE__) {\n";
      IncrDeviceIndent();
      return true;
    default: return false;
    }
  };

  auto EndEventCritical = [&](bool guarded) {
    if (!guarded || IsHost()) return;
    DecrDeviceIndent();
    ds << d_indent << "}\n";
    if (IsWarpSpecActive()) return;
    switch (bdim_level) {
    case ParallelLevel::GROUPx4:
      ds << d_indent
         << "cooperative_groups::tiled_partition<128>(cooperative_groups::this_"
            "thread_block()).sync();\n";
      break;
    case ParallelLevel::GROUP: ds << d_indent << "__syncwarp();\n"; break;
    case ParallelLevel::BLOCK: ds << d_indent << "__syncthreads();\n"; break;
    default: break;
    }
  };

  for (auto& f : n.GetEvents()) {
    auto expr = cast<AST::Expr>(f);
    bool is_array_ref = (expr->op == Op::ElemOf);
    assert(IsSymbolOrArrayRef(*f) &&
           "expect either symbol or array reference.");
    auto fty = is_array_ref
                   ? GetSymbolType(AST::GetArrayBaseSymbol(*expr)->name)
                   : NodeType(*f);
    if (auto ety = dyn_cast<EventArrayType>(fty)) {
      if (IsHost()) {
        assert(ety->GetStorage() == Storage::GLOBAL);
        // TODO: make & into OpExprSTR?
        hs << h_indent << "choreo::abend_true(cudaMemset(&" << ExprSTR(f, true)
           << ", 1, " << ety->ElemCount() << ")); // trigger event\n";
        // TODO: support array reference
      } else {
        switch (ety->GetStorage()) {
        case Storage::GLOBAL:
        case Storage::LOCAL: {
          bool guarded = BeginEventCritical();
          ds << d_indent << "// trigger event " << PSTR(f) << "\n";
          if (is_array_ref) {
            size_t lvl = GetSubScriptLevel(*expr);
            auto bid = AST::GetArrayBaseSymbol(*expr);
            auto bty =
                cast<EventArrayType>(GetSymbolType(UnScopedName(bid->name)));
            GenerateSubscriptions(ds, d_indent + ExprSTR(f, false),
                                  " = true;\n", bty->RemainderDimensions(lvl));
          } else
            GenerateSubscriptions(ds, d_indent + ExprSTR(f, false),
                                  " = true;\n", ety->RemainderDimensions(0));
          EndEventCritical(guarded);
          break;
        }
        case Storage::SHARED: {
          bool is_cluster_trigger = n.IsClusterScope();

          ds << d_indent << "// trigger event(barrier) " << PSTR(f);
          if (is_cluster_trigger) ds << " [cluster-scope]";
          ds << "\n";

          if (is_cluster_trigger) {
            ds << d_indent << "if (__CHOREO_GROUPX4_SINGLE__) {\n";
            IncrDeviceIndent();
            ds << d_indent
               << "for (uint32_t __cta = 0; __cta < "
                  "choreo::tma_cluster_dim(); ++__cta) {\n";
            IncrDeviceIndent();
            if (is_array_ref) {
              ds << d_indent << "choreo::tma_mbarrier_arrive_cluster(&"
                 << ExprSTR(f, false) << ", __cta);\n";
            } else {
              GenerateSubscriptions(
                  ds,
                  d_indent + "choreo::tma_mbarrier_arrive_cluster(&" +
                      ExprSTR(f, false),
                  ", __cta);\n", ety->RemainderDimensions(0));
            }
            DecrDeviceIndent();
            ds << d_indent << "}\n";
            DecrDeviceIndent();
            ds << d_indent << "}\n";
          } else if (!recent_tma_tx_bytes.empty()) {
            auto tx_bytes_expr = SumRecentTMATxBytesExpr();
            // In a multi-threaded warpspec scope where TMA is
            // single-thread guarded, only the TMA-issuing thread
            // reports expected bytes -- others arrive with 0 bytes.
            bool conditional_tx =
                IsWarpSpecActive() &&
                !ScopeAlreadySingleThreadForLevel(ParallelLevel::GROUPx4);
            if (is_array_ref) {
              if (conditional_tx) {
                ds << d_indent << "(void)cuda::device::barrier_arrive_tx("
                   << ExprSTR(f, false) << ", 1, __CHOREO_GROUPX4_SINGLE__ ? "
                   << tx_bytes_expr << " : 0);\n";
              } else if (IsWarpSpecActive()) {
                ds << d_indent << "(void)cuda::device::barrier_arrive_tx("
                   << ExprSTR(f, false) << ", 1, " << tx_bytes_expr << ");\n";
              } else {
                ds << d_indent << "(void)" << ExprSTR(f, false)
                   << ".arrive();\n";
              }
            } else {
              if (conditional_tx) {
                GenerateSubscriptions(
                    ds,
                    d_indent + "(void)cuda::device::barrier_arrive_tx(" +
                        ExprSTR(f, false) +
                        ", 1, __CHOREO_GROUPX4_SINGLE__ ? " + tx_bytes_expr +
                        " : 0)",
                    ";\n", ety->RemainderDimensions(0));
              } else if (IsWarpSpecActive()) {
                GenerateSubscriptions(
                    ds,
                    d_indent + "(void)cuda::device::barrier_arrive_tx(" +
                        ExprSTR(f, false) + ", 1, " + tx_bytes_expr + ")",
                    ";\n", ety->RemainderDimensions(0));
              } else {
                GenerateSubscriptions(
                    ds, d_indent + "(void)" + ExprSTR(f, false), ".arrive();\n",
                    ety->RemainderDimensions(0));
              }
            }
            recent_tma_tx_bytes.clear();
          } else {
            if (is_array_ref) {
              ds << d_indent << "(void)" << ExprSTR(f, false) << ".arrive();\n";
            } else {
              GenerateSubscriptions(ds, d_indent + "(void)" + ExprSTR(f, false),
                                    ".arrive();\n",
                                    ety->RemainderDimensions(0));
            }
          }
          break;
        }
        default:
          choreo_unreachable("unsupported event array storage '" +
                             STR(ety->GetStorage()) + "' to trigger.");
          break;
        }
      }
    } else if (auto ety = dyn_cast<EventType>(fty)) {
      if (IsHost()) {
        assert(ety->GetStorage() == Storage::GLOBAL);
        hs << h_indent << "choreo::abend_true(cudaMemset(&" << ExprSTR(f, true)
           << ", 1, 1)); // trigger event\n";
        // TODO: support array reference
      } else {
        switch (ety->GetStorage()) {
        case Storage::GLOBAL:
        case Storage::LOCAL: {
          bool guarded = BeginEventCritical();
          if (is_array_ref) {
            ds << d_indent << "// trigger event " << PSTR(f) << "\n";
            size_t lvl = GetSubScriptLevel(*expr);
            auto bid = AST::GetArrayBaseSymbol(*expr);
            auto bty =
                cast<EventArrayType>(GetSymbolType(UnScopedName(bid->name)));
            GenerateSubscriptions(ds, d_indent + ExprSTR(f, false),
                                  " = true; // trigger event\n",
                                  bty->RemainderDimensions(lvl));
          } else
            ds << d_indent << ExprSTR(f, false)
               << " = true; // trigger event\n";
          EndEventCritical(guarded);
          break;
        }
        case Storage::SHARED: {
          bool is_cluster_trigger = n.IsClusterScope();

          if (is_cluster_trigger) {
            ds << d_indent << "if (__CHOREO_GROUPX4_SINGLE__) {\n";
            IncrDeviceIndent();
            ds << d_indent
               << "for (uint32_t __cta = 0; __cta < "
                  "choreo::tma_cluster_dim(); ++__cta) {\n";
            IncrDeviceIndent();
            ds << d_indent << "choreo::tma_mbarrier_arrive_cluster(&"
               << ExprSTR(f, false) << ", __cta);\n";
            DecrDeviceIndent();
            ds << d_indent << "}\n";
            DecrDeviceIndent();
            ds << d_indent << "}\n";
          } else if (!recent_tma_tx_bytes.empty() && IsWarpSpecActive()) {
            auto tx_bytes_expr = SumRecentTMATxBytesExpr();
            bool conditional_tx =
                !ScopeAlreadySingleThreadForLevel(ParallelLevel::GROUPx4);
            if (conditional_tx) {
              ds << d_indent << "(void)cuda::device::barrier_arrive_tx("
                 << ExprSTR(f, false) << ", 1, __CHOREO_GROUPX4_SINGLE__ ? "
                 << tx_bytes_expr << " : 0); // trigger event(barrier)\n";
            } else {
              ds << d_indent << "(void)cuda::device::barrier_arrive_tx("
                 << ExprSTR(f, false) << ", 1, " << tx_bytes_expr
                 << "); // trigger event(barrier)\n";
            }
            recent_tma_tx_bytes.clear();
          } else {
            ds << d_indent << "(void)" << ExprSTR(f, false)
               << ".arrive(); // trigger event(barrier)\n";
          }
          break;
        }
        default:
          choreo_unreachable("unsupported event storage '" +
                             STR(ety->GetStorage()) + "' to trigger.");
          break;
        }
      }
    }
  }
  return true;
}

bool CuteCodeGen::Visit(AST::Call& n) {
  TraceEachVisit(n);

  if (!emit_call) return true;

  auto& os = (IsHost()) ? hs : ds;
  auto& indent = (IsHost()) ? h_indent : d_indent;

  // generate the built-in functions
  if (n.IsBIF()) {
    const auto func_name = n.function->name;
    if (func_name == "assert") {
      if (IsHost()) {
        os << indent << "choreo_assert(" << ExprSTR(n.GetArguments().at(0))
           << ", \"" << ExprSTR(n.GetArguments().at(1)) << "\", \""
           << n.LOC().begin.get_filename() << "\", " << n.LOC().begin.get_line()
           << ");\n";
      } else {
        os << indent << "if (!(" << ExprSTR(n.GetArguments().at(0), false)
           << ")) {\n";
        os << indent << "  printf(\"" << n.LOC() << ": choreo assertion abort: "
           << ExprSTR(n.GetArguments().at(1), false) << "\");\n";
        os << indent << "  __co_abort__();\n";
        os << indent << "}\n";
      }
      return true;
    } else if (func_name == "print" || func_name == "println") {
      std::string print_format;
      print_format += "\"";
      std::string print_args;
      auto GenFormatAndArgsFromValueList = [&](const ValueList& vl) {
        std::string format;
        std::ostringstream oss;
        for (size_t i = 0; i < vl.size(); ++i) {
          if (i != 0) {
            format += ", ";
            oss << ", ";
          }
          format += "%lld";
          oss << "static_cast<long long>(" << ValueSTR(vl[i]) << ")";
        }
        std::string args = UnScopedExpr(oss.str());
        return std::make_pair(format, args);
      };
      auto GenFormatAndArgsFromShape = [&](const Shape& shape) {
        return GenFormatAndArgsFromValueList(shape.Value());
      };
      for (const auto& arg : n.GetArguments()) {
        const auto type = NodeType(*arg);
        auto e = cast<AST::Expr>(arg);
        if (isa<StringType>(type)) {
          print_format += ExprSTR(arg, IsHost());
        } else if (isa<ScalarIntegerType>(type)) {
          print_format += "%lld";
          print_args += ExprCastSTR(arg, std::nullopt, BaseType::S64,
                                    type->GetBaseType(), IsHost());
          print_args += ", ";
        } else if (isa<BooleanType>(type) || isa<EventType>(type)) {
          print_format += "%s";
          print_args +=
              "(" + ExprSTR(arg, IsHost()) + " ? \"true\" : \"false\"), ";
        } else if (BaseType bt = type->GetBaseType(); IsFloatType(bt)) {
          print_format += "%f";
          print_args +=
              ExprCastSTR(arg, std::nullopt,
                          bt == BaseType::F64 ? BaseType::F64 : BaseType::F32,
                          bt, IsHost());
          print_args += ", ";
        } else if (isa<ITupleType>(type)) {
          print_format += "{";
          auto [format, args] =
              GenFormatAndArgsFromValueList(e->Opts().GetVals());
          print_format += format;
          print_format += "}";
          print_args += args + ", ";
        } else if (isa<MDSpanType>(type)) {
          print_format += "[";
          auto [format, args] = GenFormatAndArgsFromShape(e->s);
          print_format += format;
          print_format += "]";
          print_args += args + ", ";
        } else if (isa<BoundedIntegerType>(type)) {
          choreo_unreachable("All the BoundedIntegerType vars should have been "
                             "normed to BoundedITupleType vars.");
        } else if (auto bit = dyn_cast<BoundedITupleType>(type)) {
          print_format += "{";
          for (size_t i = 0; i < bit->Dims(); ++i) {
            if (i != 0) print_format += ", ";
            print_format += "%lld";
          }
          print_format += "}";
          std::string args_str = ExprSTR(arg, IsHost());
          auto arg_items = SplitStringByDelimiter(args_str, ", ");
          if (arg_items.size() == bit->Dims()) {
            for (const auto& arg_str : arg_items)
              print_args += "static_cast<long long>(" + arg_str + "), ";
          } else {
            for (size_t i = 0; i < bit->Dims(); ++i)
              print_args += "static_cast<long long>(" + args_str + "[" +
                            std::to_string(i) + "]), ";
          }
        } else if (isa<AddrType>(type)) {
          print_format += "%p";
          print_args += "static_cast<void*>(" + ExprSTR(arg, IsHost()) + "), ";
        } else
          choreo_unreachable(
              "unsupported type for print: " + AST::TYPE_STR(*arg) +
              "\n\targ: " + ExprSTR(arg, IsHost()));
      }
      if (func_name == "println") print_format += "\\n";
      print_format += "\"";
      os << indent << "printf(" << print_format;
      if (auto len = print_args.length(); len > 2) {
        assert(print_args[len - 2] == ',');
        print_args = print_args.substr(0, len - 2); // remove last ", "
        os << ", " << print_args;
      }
      os << ");\n";
      return true;
    } else if (n.IsArith()) {
    } else
      choreo_unreachable("the bif '" + n.function->name +
                         "' is not supported by this target.");
  }

  if (!n.IsExpr()) os << indent << CallSTR(n) << ";\n";

  return true;
}

bool CuteCodeGen::Visit(AST::ParamList& n) {
  int index = 0;
  for (auto param : n.values) {
    auto ty = GetSymbolType(param->sym->name);
    if (isa<StreamType>(ty)) {
      if (stream_name != "")
        choreo_unreachable("Unexpect: only one stream supported now!");
      stream_name = param->sym->name;
      continue;
    }
    updating_cgi.AddSymbolDetail(fname, {InScopeName(param->sym->name),
                                         param->GetType(), param->pass_by_ref,
                                         index++, param->GetAttr()});
  }
  return true;
}

bool CuteCodeGen::Visit(AST::WithIn& n) {
  TraceEachVisit(n);

  if (n.with)
    ssm.MapDeviceSymbol(InScopeName(n.with->name), "__iv_" + n.with->name);

  assert(n.with_matchers && "expected matchers exist.");

  for (auto& v : n.GetMatchers()) {
    auto id = cast<AST::Identifier>(v);
    ssm.RemapDeviceSymbol(InScopeName(id->name), "__iv_" + id->name);
    ssm.RemapHostSymbol(InScopeName(id->name), "__iv_" + id->name);
    // Keep the device side decl, even for host side iv.
    // for visibility of shapes
    if (IsHost()) {
      hs << h_indent << "int __iv_" << id->name << " = 0;\n";
      updating_cgi.AddSymbolDetail(fname,
                                   {InScopeName(id->name), id->GetType(), true,
                                    -1, ParamAttr::NONE, "", true});
    } else
      ds << d_indent << "int __iv_" << id->name << " = 0;\n";
  }

  if (EnableDebugTypeRTTI() && n.with && (n.GetMatchers().size() > 1)) {
    auto& os = Stream();
    os << Indent() << "choreo::rtti::bounded_ituple<" << n.GetMatchers().size()
       << "> __iv_" << n.with->name << " = {{";
    for (size_t i = 0; i < n.GetMatchers().size(); ++i) {
      auto id = cast<AST::Identifier>(n.GetMatchers()[i]);
      os << "__iv_" << id->name;
      if (i + 1 < n.GetMatchers().size()) os << ", ";
    }
    os << "}, {";
    for (size_t i = 0; i < n.GetMatchers().size(); ++i) {
      auto id = cast<AST::Identifier>(n.GetMatchers()[i]);
      auto id_ty = id->GetType();
      if (auto bty = dyn_cast<BoundedType>(id_ty))
        os << ValueSTR(bty->GetUpperBound());
      else
        os << "0";
      if (i + 1 < n.GetMatchers().size()) os << ", ";
    }
    os << "}};\n";
    os << Indent() << "auto " << n.with->name << " = __iv_" << n.with->name
       << ";\n";
  }

  if (n.with && (n.GetMatchers().size() == 1)) {
    auto m1 = cast<AST::Identifier>(n.GetMatchers()[0]);
    ssm.RemapDeviceSymbol(InScopeName(n.with->name), "__iv_" + m1->name);
    ssm.RemapHostSymbol(InScopeName(n.with->name), "__iv_" + m1->name);
  }

  return true;
}

bool CuteCodeGen::Visit(AST::WhereBind& n) {
  TraceEachVisit(n);

  // TODO
  choreo_unreachable("where bind is yet to support.");

  return true;
}

bool CuteCodeGen::Visit(AST::WithBlock& n) {
  TraceEachVisit(n);

  explicit_scale_accum_scopes.push_back(
      !IsHost() ? AnalyzeExplicitScaleAccumScope(n.GetBody())
                : std::vector<ExplicitScaleAccumInfo>{});
  if (!IsHost()) {
    for (const auto& info : explicit_scale_accum_scopes.back()) {
      ds << d_indent << info.scale_frag_ty << " " << info.scale_frag_name << "["
         << info.reg_num_d << "];\n";
      ds << d_indent << "memset(" << info.scale_frag_name << ", 0, sizeof("
         << info.scale_frag_name << "));\n";
    }
  }
  return true;
}

bool CuteCodeGen::Visit(AST::ForeachBlock& n) {
  TraceEachVisit(n);

  explicit_scale_accum_scopes.push_back(
      !IsHost() ? AnalyzeExplicitScaleAccumScope(n.GetBody())
                : std::vector<ExplicitScaleAccumInfo>{});
  hoisted_scale_decl_scopes.push_back({});
  hoisted_scale_accum_scopes.push_back(std::nullopt);

  std::vector<std::string> loop_refs;
  for (auto& rn : n.GetRanges()) {
    auto rng = cast<AST::LoopRange>(rn);
    auto cname = rng->IVName();
    loop_refs.push_back(cname);
    loop_refs.push_back(std::string("__iv_") + cname);
    for (auto iv_name : within_map.at(InScopeName(cname)))
      loop_refs.push_back(SSMName(iv_name, false));
  }

  if (!IsHost() && n.GetBody()) {
    hoisted_scale_accum_scopes.back() =
        AnalyzeHoistableScaledWGMMAAccum(n.GetBody(), loop_refs);

    if (auto* op = FindFirstScaledWGMMAExec(n.GetBody())) {
      auto c_sym = AST::FragName(op->ExecOperand(0));
      auto scale_a_name = c_sym + "_scale_a_ptr";
      auto scale_a_valid_rows_name = c_sym + "_scale_a_valid_rows";
      auto scale_b_name = c_sym + "_scale_b_val";
      auto scale_a_expr = ExprSTR(op->ScaleA(), false);
      auto scale_a_valid_rows_expr = GenScaleValidRowsExpr(op->ScaleA());
      auto scale_b_expr = ExprSTR(op->ScaleB(), false);
      bool invariant_to_loop = true;
      for (const auto& loop_ref : loop_refs) {
        if (!loop_ref.empty() &&
            (scale_a_expr.find(loop_ref) != std::string::npos ||
             scale_b_expr.find(loop_ref) != std::string::npos)) {
          invariant_to_loop = false;
          break;
        }
      }

      if ((hoist_offset || hoisted_scale_accum_scopes.back().has_value()) &&
          invariant_to_loop &&
          !active_hoisted_scale_decls.count(scale_a_name)) {
        ds << d_indent << "float* " << scale_a_name << " = (float*)("
           << scale_a_expr << ");\n";
        ds << d_indent << "int " << scale_a_valid_rows_name << " = "
           << scale_a_valid_rows_expr << ";\n";
        ds << d_indent << "float " << scale_b_name << " = static_cast<float>("
           << scale_b_expr << ");\n";
        active_hoisted_scale_decls.insert(scale_a_name);
        active_hoisted_scale_decls.insert(scale_a_valid_rows_name);
        active_hoisted_scale_decls.insert(scale_b_name);
        hoisted_scale_decl_scopes.back().push_back(scale_a_name);
        hoisted_scale_decl_scopes.back().push_back(scale_a_valid_rows_name);
        hoisted_scale_decl_scopes.back().push_back(scale_b_name);
      }
    }

    if (hoisted_scale_accum_scopes.back().has_value()) {
      const auto& info = hoisted_scale_accum_scopes.back().value();
      ds << d_indent << info.scale_frag_ty << " " << info.scale_frag_name << "["
         << info.reg_num_d << "];\n";
      ds << d_indent << "memset(" << info.scale_frag_name << ", 0, sizeof("
         << info.scale_frag_name << "));\n";
    }
  }

  for (auto& rn : n.GetRanges()) {
    auto rng = cast<AST::LoopRange>(rn);
    auto cname = rng->IVName();
    for (auto iv_name : within_map.at(InScopeName(cname))) {
      auto iv_ty = GetSymbolType(UnScopedName(iv_name));
      assert(IsActualBoundedIntegerType(iv_ty));
      auto iv_bty = cast<BoundedType>(iv_ty);
      IndStream() << "for (" << SSMName(iv_name, IsHost()) << " = "
                  << (rng->lbound ? ("(" + ExprSTR(rng->lbound, IsHost()) + ")")
                                  : "0")
                  << "; " << SSMName(iv_name, IsHost()) << " < "
                  << UnScopedExpr(ValueSTR(iv_bty->GetUpperBound()))
                  << (rng->ubound ? (" + " + ExprSTR(rng->ubound, IsHost()))
                                  : "")
                  << "; ++" << SSMName(iv_name, IsHost()) << ") {\n";
      IncrIndent();
    }
  }

  if (!IsHost()) {
    for (const auto& info : explicit_scale_accum_scopes.back()) {
      ds << d_indent << info.scale_frag_ty << " " << info.scale_frag_name << "["
         << info.reg_num_d << "];\n";
      ds << d_indent << "memset(" << info.scale_frag_name << ", 0, sizeof("
         << info.scale_frag_name << "));\n";
    }
  }

  return true;
}

bool CuteCodeGen::Visit(AST::InThreadsBlock& n) {
  TraceEachVisit(n);
  assert(!IsHost());
  ds << d_indent << "// inthreads: " << n.LOC() << "\n";
  current_inthreads = &n;
  warpspec_wgmma_arrived = false;
  if (!n.stmts->None()) {
    auto pred_str = ExprSTR(n.pred, false);

    // Detect warpspec pattern from active threads analysis.
    if (n.HasScopeThreadMask() && n.inthreads_level == ParallelLevel::GROUPx4 &&
        bdim_level == ParallelLevel::GROUPx4) {
      if (n.ActiveWarpGroup() >= 0) has_analyzed_warpspec = true;
    }

    ds << d_indent << "if (" << pred_str << ") {\n";
  }
  IncrDeviceIndent();
  return true;
}

bool CuteCodeGen::Visit(AST::IfElseBlock& n) {
  TraceEachVisit(n);

  IndStream() << "// if-else: " << n.LOC() << "\n";
  if (auto c = dyn_cast<AST::Call>(n.pred->GetReference()))
    IndStream() << "if (" << CallSTR(*c) << ") {\n";
  else
    IndStream() << "if (" << ExprSTR(n.pred, IsHost()) << ") {\n";
  IncrIndent();
  emit_call = true;
  return true;
}

bool CuteCodeGen::Visit(AST::WhileBlock& n) {
  TraceEachVisit(n);

  IndStream() << "// while: " << n.LOC() << "\n";
  IndStream() << "while (" << ExprSTR(n.pred, IsHost()) << ") {\n";
  IncrIndent();

  return true;
}

bool CuteCodeGen::Visit(AST::Return& n) {
  TraceEachVisit(n);

  return_stream.str("");

  auto vty = NodeType(*n.value);

  if (isa<ScalarType>(vty)) {
    return_stream << "return " << ExprSTR(n.value, true) << ";\n";
  } else if (auto sty = dyn_cast<SpannedType>(vty)) {
    if (auto id = AST::GetIdentifier(*n.value)) {
      auto sym = id->name;
      if (IsChoreoInput(InScopeName(sym))) {
        // return the global storage, must map back
        hs << h_indent << "choreo::abend_true(cudaMemcpy(" << sym << ".data(), "
           << sym << "__device, " << UnScopedSizeExpr(*sty)
           << ", cudaMemcpyDeviceToHost));\n";
        return_stream << "return choreo::copy_as_spanned(" << sym << ".data(), "
                      << sym << ".shape());\n";
      } else if (IsChoreoOutput(InScopeName(sym))) {
        // return the global storage, must map back
        hs << h_indent << "choreo::abend_true(cudaMemcpy(" << sym << ".data(), "
           << sym << "__device, " << UnScopedSizeExpr(*sty)
           << ", cudaMemcpyDeviceToHost));\n";
        return_stream << "return " << sym << ";\n";
      } else {
        choreo_unreachable("unexpected situation");
      }
    } else if (auto expr = cast<AST::Expr>(n.value);
               expr && (expr->op == Op::DataOf || expr->op == Op::MDataOf)) {
      // return future.data/mdata, must map back
      auto id = cast<AST::Expr>(expr->GetR())->GetSymbol();
      assert(id && "expect a symbol");
      auto sym = id->name + "__buf__";
      hs << h_indent << "choreo::abend_true(cudaMemcpy(" << sym << ".data(), "
         << sym << "__device, " << UnScopedSizeExpr(*sty)
         << ", cudaMemcpyDeviceToHost));\n";
      return_stream << "return " << ExprSTR(n.value, true) << ";\n";
    } else {
      choreo_unreachable("not support return value of type: " + PSTR(vty));
    }
  } else {
    choreo_unreachable("not support return value of type: " + PSTR(vty));
  }

  EmitCudaFree();

  hs << h_indent << return_stream.str();

  return true;
}

bool CuteCodeGen::Visit(AST::CppSourceCode& n) {
  TraceEachVisit(n);

  if (n.kind == AST::CppSourceCode::Inline) {
    Stream() << n.GetCode();
  } else {
    CodeSegment cur_cs =
        (n.kind == AST::CppSourceCode::Host) ? CS_USER : CS_COK;
    if (cur_cs != cs) { code_segments.push_back(""); }

    // append the content
    code_segments.back() += n.GetCode();
  }

  return true;
}

void CuteCodeGen::EmitHostFuncDecl(std::ostringstream& oss) {
  // handle the return type
  if (!void_return) {
    if (cgi.HasReturnSymbol(fname)) {
      auto& item = cgi.GetReturnDetail(fname);
      if (item.rty_str != "$")
        oss << item.rty_str;
      else
        oss << HostTypeStringify(*fty->out_ty, true);
    } else {
      oss << HostTypeStringify(*fty->out_ty, true);
    }
  } else
    oss << "void";
  oss << " " << fname << "(";

  // emit the parameters
  size_t host_pindex = 0;
  for (auto& item : GetChoreoFuncIns(cgi)) {
    if (item.IsParameter()) assert((int)host_pindex == item.p_index);
    oss << ((host_pindex == 0) ? "" : ", ")
        << HostParamTypeStringifyForCute(*item.type, item.IsReference()) << " "
        << item.host_name;
    ++host_pindex;
  }
  oss << ")";

  VST_DEBUG(dbgs() << "Host function prototype:\n" << oss.str() << "\n");
}

/*inner parallel-by need different threads-mapping strategy, but launch
config is only set for the outermost parallel-by.
We should allow users to write inner parallel-by blocks like the following
ways:
parallel by 1 : block {
  (1) parallel by 4 : group
        parallel by 32 : thread //  must be 32 if explicit
          ...
  (2) parallel by 1 : group-4
        parallel by 128 : thread // must be 128 if explicit
          ...
  (3) parallel by 128 : thread
        ...
}
The above three cases should all be supported, and can be used within a same
block level parallel-by at same time.
To support this, we flatten the within-block parallel-by levels, and
generate the virtual indices according to the level settings. All indices are
mapped to threadIdx.x, and we compute the virtual indices based on the level
settings. For case (1), we compute the virtual indices based on group and
thread levels. For case (2), we compute the virtual indices based on group-4
and thread levels. For case (3), we only compute the virtual indices based on
thread level.
Note: (1) we must ensure that the total number of threads of all inner
parallel-by be same, because they share the same launch configuration. (2) we
should allow three-dimension parallel-by indices generation.
*/
void CuteCodeGen::EmitDeviceVirtualIndices(AST::ParallelBy* pb) {
  // no need to generate virtual indices for non-enforced parallel-by
  // generated by normalization
  if (!pb->IsEnforced()) return;

  const auto& bvs = pb->BoundValues();
  auto sub_pvs = pb->SubPVs();
  sbe::Operand pv_x = bvs.size() > 0 ? bvs.at(0) : sbe::nu(1);
  sbe::Operand pv_y = bvs.size() > 1 ? bvs.at(1) : sbe::nu(1);
  sbe::Operand pv_z = bvs.size() > 2 ? bvs.at(2) : sbe::nu(1);

  switch (pb->GetLevel()) {
  case ParallelLevel::GROUPx4: {
    assert(pb->AllSubPVs().size() > 0);
    std::string g4id = vid_pfx + "g4id";
    std::string vid_x = vid_pfx + "g4id_x";
    std::string vid_y = vid_pfx + "g4id_y";
    std::string vid_z = vid_pfx + "g4id_z";
    if (pb->AllSubPVs().size() == 1) {
      ds << d_indent << "auto " << vid_x << " = threadIdx.x / 128;\n";
    } else if (pb->AllSubPVs().size() == 2) {
      ds << d_indent << "auto " << g4id << " = threadIdx.x / 128;\n";
      ds << d_indent << "auto " << vid_x << " = " << g4id << " / "
         << ValueSTR(pv_y);
      ds << d_indent << "auto " << vid_y << " = " << g4id << " % "
         << ValueSTR(pv_y) << ";\n";
    } else if (pb->AllSubPVs().size() == 3) {
      ds << d_indent << "auto " << g4id << "g4id = threadIdx.x / 128;\n";
      ds << d_indent << "auto " << vid_x << " = " << g4id << " / "
         << ValueSTR(pv_y) << " / " << ValueSTR(pv_z) << ";\n";
      ds << d_indent << "auto " << vid_y << " = " << g4id << " / "
         << ValueSTR(pv_z) << " % " << ValueSTR(pv_y) << ";\n";
      ds << d_indent << "auto " << vid_z << " = " << g4id << " % "
         << ValueSTR(pv_z) << ";\n";
    }
  } break;
  case ParallelLevel::GROUP: {
    assert(pb->AllSubPVs().size() > 0);
    if (pb->AllSubPVs().size() > 3)
      choreo_unreachable("group parallelism with more than 3 dimensions is "
                         "not supported.");

    std::string gid = vid_pfx + "gid";
    std::string vid_x = vid_pfx + "gid_x";
    std::string vid_y = vid_pfx + "gid_y";
    std::string vid_z = vid_pfx + "gid_z";
    if (pb->AllSubPVs().size() == 1) {
      ds << d_indent << "auto " << vid_x << " = threadIdx.x / 32;\n";
    } else if (pb->AllSubPVs().size() == 2) {
      ds << d_indent << "auto " << gid << " = threadIdx.x / 32;\n";
      ds << d_indent << "auto " << vid_x << " = " << gid << " / "
         << ValueSTR(pv_y) << ";\n";
      ds << d_indent << "auto " << vid_y << " = " << gid << " % "
         << ValueSTR(pv_y) << ";\n";
    } else if (pb->AllSubPVs().size() == 3) {
      ds << d_indent << "auto " << gid << " = threadIdx.x / 32;\n";
      ds << d_indent << "auto " << vid_x << " = " << gid << " / "
         << ValueSTR(pv_y) << " / " << ValueSTR(pv_z) << ";\n";
      ds << d_indent << "auto " << vid_y << " = " << gid << " / "
         << ValueSTR(pv_z) << " % " << ValueSTR(pv_y) << ";\n";
      ds << d_indent << "auto " << vid_z << " = " << gid << " % "
         << ValueSTR(pv_z) << ";\n";
    }
  } break;
  case ParallelLevel::THREAD: {
    assert(pb->AllSubPVs().size() > 0);
    if (pb->AllSubPVs().size() > 3)
      choreo_unreachable("thread parallelism with more than 3 dimensions is "
                         "not supported.");

    std::string tid = vid_pfx + "tid";
    std::string vid_x = vid_pfx + "tid_x";
    std::string vid_y = vid_pfx + "tid_y";
    std::string vid_z = vid_pfx + "tid_z";
    if (pb->AllSubPVs().size() == 1) {
      if (bdim_level == ParallelLevel::GROUPx4)
        ds << d_indent << "auto " << vid_x << " = threadIdx.x % 128;\n";
      else if (bdim_level == ParallelLevel::GROUP)
        ds << d_indent << "auto " << vid_x << " = threadIdx.x % 32;\n";
      else if (bdim_level == ParallelLevel::THREAD)
        ds << d_indent << "auto " << vid_x << " = threadIdx.x;\n";
      else
        choreo_unreachable("invalid bdim level.");
    } else if (pb->AllSubPVs().size() == 2) {
      if (bdim_level == ParallelLevel::GROUPx4)
        ds << d_indent << "auto " << tid << " = threadIdx.x % 128;\n";
      else if (bdim_level == ParallelLevel::GROUP)
        ds << d_indent << "auto " << tid << " = threadIdx.x % 32;\n";
      else if (bdim_level == ParallelLevel::THREAD)
        ds << d_indent << "auto " << tid << " = threadIdx.x;\n";
      else
        choreo_unreachable("invalid bdim level.");

      ds << d_indent << "auto " << vid_x << " = " << tid << " / "
         << ValueSTR(pv_y) << ";\n";
      ds << d_indent << "auto " << vid_y << " = " << tid << " % "
         << ValueSTR(pv_y) << ";\n";
    } else if (pb->AllSubPVs().size() == 3) {
      if (bdim_level == ParallelLevel::GROUPx4)
        ds << d_indent << "auto " << tid << " = threadIdx.x % 128;\n";
      else if (bdim_level == ParallelLevel::GROUP)
        ds << d_indent << "auto " << tid << " = threadIdx.x % 32;\n";
      else if (bdim_level == ParallelLevel::THREAD)
        ds << d_indent << "auto " << tid << " = threadIdx.x;\n";
      else
        choreo_unreachable("invalid bdim level.");

      ds << d_indent << "auto " << vid_x << " = " << tid << " / "
         << ValueSTR(pv_y) << " / " << ValueSTR(pv_z) << ";\n";
      ds << d_indent << "auto " << vid_y << " = " << tid << " / "
         << ValueSTR(pv_z) << " % " << ValueSTR(pv_y) << ";\n";
      ds << d_indent << "auto " << vid_z << " = " << tid << " % "
         << ValueSTR(pv_z) << ";\n";
    }
  } break;
  default: break;
  }
}

void CuteCodeGen::EmitHostRuntimeCheck() {
  if (CCtx().DisableRuntimeCheck()) return;
  // check if the input shape is as declared in choreo
  if (cgi.ParameterCount(fname) == 0) return;

  struct Entry {
    size_t para_ordinal;
    size_t dim;
    std::string elem_name;
  };
  std::map<std::string, std::vector<Entry>> ve_entries_map;

  size_t host_pindex = 0;
  for (const auto& item : GetChoreoFuncIns(cgi)) {
    assert((int)host_pindex == item.p_index);
    auto name = UnScopedName(item.name);
    if (auto sty = dyn_cast<SpannedType>(item.type)) {
      size_t dim_count = 0;
      for (auto vi : sty->GetShape().Value()) {
        auto elem_name = name + ".shape()[" + std::to_string(dim_count) + "]";
        if (auto vale = VIInt(vi)) {
          hs << h_indent << "choreo::runtime_check(" << elem_name
             << " == " << *vale;
          hs << ", \"shape inconsistent on the " << Ordinal(host_pindex + 1)
             << " parameter (\'" << name << "\', dim: " << dim_count
             << "): expect: " << *vale << ", but got \" + std::to_string("
             << elem_name << ") + \".\");\n";
        } else if (VIIsNil(vi)) {
          hs << h_indent << "choreo::runtime_check(" << elem_name
             << " == choreo::__inf__, \"must set 'choreo::__inf__' to the "
                "unbounded dimension on the "
             << Ordinal(host_pindex + 1) << " parameter (\'" << name
             << "\', dim: " << dim_count << "): got \" + std::to_string("
             << elem_name << ") + \".\");\n";
        } else if (auto vale = VISym(vi))
          ve_entries_map[*vale].push_back(
              {host_pindex + 1, dim_count, elem_name});
        dim_count++;
      }
    }
    host_pindex++;
  }

  // Check if the named dims meet the constraint. Eg.
  //
  //   __co__ void foo(f32 [M, N] a, f32 [N, K] b)
  //
  // then a.shape()[1] should be equal to b.shape()[0]
  auto& stats = CCtx().GetAssessmentStats();
  for (const auto& [_, entries] : ve_entries_map) {
    for (size_t i = 1; i < entries.size(); ++i) {
      auto& entry0 = entries[i - 1];
      auto& entry1 = entries[i];
      hs << h_indent << "choreo::runtime_check(" << entry0.elem_name
         << " == " << entry1.elem_name;
      hs << ", \"The shapes of the " << Ordinal(entry0.para_ordinal)
         << " parameter (dim: " << entry0.dim << ") and the "
         << Ordinal(entry1.para_ordinal) << " parameter (dim: " << entry1.dim
         << ") are inconsistent.\");\n";
      ++stats.total;
      ++stats.shape_compat_total;
      ++stats.runtime_total;
      ++stats.shape_compat_runtime;
      ++stats.runtime_entry;
    }
  }

  hs << "\n";

  for (const auto& rc : FCtx(fname).GetRtChecks()) {
    hs << h_indent << "choreo::runtime_check(" << ValueSTR(sbe::sym(rc.lhs))
       << " " << rc.op << " " << ValueSTR(sbe::sym(rc.rhs)) << ", \""
       << rc.message << ", " << rc.loc << "\");\n";
  }

  // ENTRY assertions reference only function parameters / host-visible values
  // -- the assertion-hoisting pass guarantees this. Safe to emit in the host
  // wrapper before the kernel launch.
  for (const auto& ar : FCtx(fname).GetAssertions(AssessType::ENTRY)) {
    if (!ar.enabled) continue;
    hs << h_indent << "choreo::runtime_check(" << ValueSTR(ar.expr, true)
       << ", \"" << ar.message << ", " << ar.loc << "\");\n";
  }

  // USE_SITE and DEF_SITE assertions are emitted in device code (inside the
  // kernel) via EmitSiteAssertions, which is called from AfterVisitImpl during
  // the AST traversal.  They depend on iteration-varying or locally-redefined
  // values that only exist on the device side.
}

void CuteCodeGen::EmitMemReuse(const std::string& df_name) {
  const auto& mri = FCtx(fname).GetDynMemReuseInfo(df_name);
  if (!mri) return;
  hs << h_indent << R"(// JIT memory reuse begin)"
     << "\n";
  for (const auto& [sto, ie] : mri->infos) {
    hs << h_indent << "HeapSimulator::Chunks " << ie.chunks_name << ";\n";
    for (const auto& c : ie.chunks)
      hs << h_indent << ie.chunks_name << ".push_back(" << c << ");\n";
  }
  for (const auto& [sto, ie] : mri->infos) {
    hs << h_indent << "HeapSimulator " << ie.simulator << ";\n";
    hs << h_indent << "HeapSimulator::Result " << ie.result << " = "
       << ie.simulator << ".Allocate(" << ie.chunks_name << ", 512);\n";
    hs << h_indent << "unsigned " << ie.spm_size << " = " << ie.result
       << ".heap_size;\n";
    // special host runtime check
    std::string mem_capacity = std::to_string(CCtx().GetMemCapacity(sto));
    if (!CCtx().DisableRuntimeCheck())
      hs << h_indent << "choreo::runtime_check(" << ie.spm_size
         << " <= (size_t)" << mem_capacity
         << ", \"In the memory reuse of dynamic shapes"
         << ", the size of the initial " << STR(sto)
         << " spm should not exceed the memory usage limit " << mem_capacity
         << " bytes.\");\n";
    hs << h_indent << "unsigned long " << ie.offsets_name << "["
       << mri->infos[sto].offset_args.size() << "];"
       << "\n";
    std::string idx = ie.chunks_name + "_idx";
    hs << h_indent << "size_t " << idx << " = 0;\n";
    hs << h_indent << "for (const auto& [buffer_id, offset] : " << ie.result
       << ".chunk_offsets)\n";
    hs << h_indent << "  " << ie.offsets_name << "[" << idx
       << "++] = offset;\n";
  }
  hs << h_indent << R"(// JIT memory reuse end)"
     << "\n";
}

void CuteCodeGen::EmitTMAConfiguration(AST::ParallelBy* pb) {
  for (auto desc : cgi.GetTMADesc(pb)) {
    ptr<AST::ChunkAt> g_ca = nullptr;
    ptr<AST::ChunkAt> s_ca = nullptr;
    std::string g_sym, s_sym;
    if (desc.IsLoad()) {
      g_ca = desc.GetFrom();
      s_ca = desc.GetTo();
      g_sym = desc.GetFromSymbol();
      s_sym = desc.GetToSymbol();
    } else {
      s_ca = desc.GetFrom();
      g_ca = desc.GetTo();
      s_sym = desc.GetFromSymbol();
      g_sym = desc.GetToSymbol();
    }
    auto gmem_ty = GetSpannedType(GetScopedSymbolType(g_sym));
    auto smem_ty = GetSpannedType(GetScopedSymbolType(s_sym));
#if 0
    auto gtsr = GenTensorDecl(g_ca->RefSymbol(), g_ca->RefSymbol(), gmem_ty->GetStorage(), gmem_ty->ElementType(), gmem_ty->GetShape(), true);
    auto stsr = GenTensorDecl(s_ca->RefSymbol(), s_ca->RefSymbol(), smem_ty->GetStorage(), gmem_ty->ElementType(), smem_ty->GetShape(), true);
    hs << gtsr.second;
    hs << stsr.second;
    if (desc.IsLoad())
    hs << h_indent << "cute::make_tma_copy(TMALoadAtom{}, " << gtsr.first
       << ", " << stsr.first << ");\n";
    else
    hs << h_indent << "cute::make_tma_copy(TMAStoreAtom{}, " << stsr.first
       << ", " << gtsr.first << ");\n";
#endif
    auto g_shape = gmem_ty->GetShape();
    auto g_stride = gmem_ty->GetStrides();
    if (auto idx = g_ca->IndexOfLastSpanAs()) {
      g_shape = g_ca->OpAt(*idx)->GetBlockShape();
      g_stride = g_ca->OpAt(*idx)->GetBlockStrides();
    }
    auto t_shape = g_ca->GetBlockShape();
    auto map_name = desc.GetName() + "_tensor_map";

    // Convert swizzle value to TMA_Swizzle enum and get CUtensorMapSwizzle
    // string
    auto swizzle_mode = desc.GetSwizzleMode();
    TMA_Swizzle tma_swizzle;
    switch (swizzle_mode) {
    case SwizMode::B32: tma_swizzle = TMA_Swizzle::B32; break;
    case SwizMode::B64: tma_swizzle = TMA_Swizzle::B64; break;
    case SwizMode::B128: tma_swizzle = TMA_Swizzle::B128; break;
    default: tma_swizzle = TMA_Swizzle::NONE; break;
    }
    std::string cu_swizzle_str = cuda_stringify(tma_swizzle);

    hs << h_indent << "uint64_t " << desc.GetName() << "_shape[] = {"
       << ValueSTR(Reverse(g_shape.Value())) << "};\n"; // shape of buffer
    // For TMA, strides should be in the same order as shape (not reversed)
    hs << h_indent << "uint64_t " << desc.GetName() << "_strides[] = {"
       << ValueSTR(Trim(Reverse(g_stride * gmem_ty->ElementSizeValue())))
       << "};\n"; // strides of shape
    hs << h_indent << "uint32_t " << desc.GetName() << "_box_shape[] = {"
       << ValueSTR(Reverse(t_shape.Value())) << "};\n"; // shape of tile block
    hs << h_indent << "uint32_t " << desc.GetName() << "_elem_strides[] = {"
       << ValueSTR(ValxN(sbe::nu(1), t_shape.Rank()))
       << "};\n"; // elements' strides
    std::string g_unscoped = UnScopedName(g_sym);

    // Determine whether this tensor is a true GLOBAL argument.
    // Correct behavior is to consult the parameter attribute
    // (ParamAttr::GLOBAL_INPUT) when the symbol corresponds to a
    // function parameter. Do NOT treat choreo output as global by
    // default, because output may be shadowed to device memory.
    bool is_global_arg = false;
    [[maybe_unused]] bool found_param = false;
    for (const auto& item : GetChoreoFuncIns(cgi)) {
      if (UnScopedName(item.name) == g_unscoped) {
        found_param = true;
        is_global_arg = (item.attr == ParamAttr::GLOBAL_INPUT);
        break;
      }
    }

    std::string base_expr = is_global_arg
                                ? (g_unscoped + ".data()")
                                : SSMName((g_unscoped + "__device"), true);

    // errs() << "[choreo][tma] g_sym=" << g_sym
    //        << " g_scoped=" << g_scoped
    //        << " storage=" << STR(gmem_ty->GetStorage())
    //        << " is_global_arg=" << (is_global_arg ? "1" : "0")
    //        << " base_expr=" << base_expr << "\n";
    // errs().flush();

    hs << h_indent << "alignas(64) CUtensorMap " << map_name << "{};\n";
    hs << h_indent << "CUresult " << map_name
       << "_res = cuTensorMapEncodeTiled(\n";
    hs << h_indent << "        &" << map_name << ",\n"; // tensor_map
    hs << h_indent << "        " << TMAMapDataType(gmem_ty->ElementType())
       << ",\n"; // tma element type
    hs << h_indent << "        " << g_shape.Rank() << ",\n";
    hs << h_indent << "        " << base_expr << ",\n"; // base symbol
    hs << h_indent << "        " << desc.GetName() << "_shape,\n";
    hs << h_indent << "        " << desc.GetName() << "_strides,\n";
    hs << h_indent << "        " << desc.GetName() << "_box_shape,\n";
    hs << h_indent << "        " << desc.GetName() << "_elem_strides,\n";
    hs << h_indent
       << "        CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,\n";
    hs << h_indent << "        " << cu_swizzle_str << ",\n";
    hs << h_indent
       << "        CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,\n";
    hs << h_indent
       << "        "
          "CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);\n";
    hs << h_indent << "choreo::abend_true(" << map_name
       << "_res != CUDA_SUCCESS);\n";
  }
}

static inline const std::string
DeviceParamTypeStringify(const Choreo::Type& ty) {
  if (isa<VoidType>(&ty))
    return "void";
  else if (isa<S8Type>(&ty))
    return "signed char";
  else if (isa<U8Type>(&ty))
    return "unsigned char";
  else if (isa<S16Type>(&ty))
    return "short";
  else if (isa<U16Type>(&ty))
    return "unsigned short";
  else if (isa<S32Type>(&ty))
    return "int";
  else if (isa<U32Type>(&ty))
    return "unsigned int";
  else if (isa<S64Type>(&ty))
    return "long long";
  else if (isa<U64Type>(&ty))
    return "unsigned long long";
  else if (isa<BooleanType>(&ty))
    return "bool";
  else if (isa<FloatE4M3Type>(&ty))
    return "choreo::float_e4m3_t";
  else if (isa<FloatE5M2Type>(&ty))
    return "choreo::float_e5m2_t";
  else if (isa<FloatE2M3Type>(&ty))
    return "choreo::float_e2m3_t";
  else if (isa<FloatE3M2Type>(&ty))
    return "choreo::float_e3m2_t";
  else if (isa<FloatE2M1Type>(&ty))
    return "choreo::float_e2m1_t";
  else if (isa<F16Type>(&ty))
    return "choreo::half";
  else if (isa<BF16Type>(&ty))
    return "choreo::bfp16";
  else if (isa<F32Type>(&ty))
    return "float";
  else if (isa<F64Type>(&ty))
    return "double";
  else if (isa<EventArrayType>(&ty))
    return "bool *";
  else if (isa<EventType>(&ty))
    return "bool"; // use bool for event
  else if (auto sty = dyn_cast<SpannedType>(&ty))
    return std::string(NameBaseType(sty->ElementType())) + " *";
  else
    choreo_unreachable(
        "unexpected compile-time type in device parameter: " + STR(ty) + ".");

  return "";
}

static inline std::string HostParamTypeStringifyForCute(const Choreo::Type& ty,
                                                        bool is_ref) {
  if (isa<MDSpanType>(&ty) || isa<ITupleType>(&ty) ||
      isa<BoundedITupleType>(&ty))
    choreo_unreachable(
        "unexpected compile-time type in host parameter: " + STR(ty) + ".");
  return HostTypeStringify(ty, false, is_ref);
}

void CuteCodeGen::EmitCudaFree() {
  assert(IsHost());
  for (const auto& item : GetDeviceFuncIns(updating_cgi)) {
    if (!PrefixedWith(scoped_symtab.ScopeName(), GetScope(item.name))) continue;
    if (!isa<SpannedType>(item.type)) continue;
    if (item.attr == ParamAttr::GLOBAL_INPUT) continue;
    if (!NeedDeviceFunc() && !IsChoreoOutput(item.name)) continue;
    if (item.IsReference()) continue;
    hs << h_indent << "choreo::abend_true(cudaFree(" << UnScopedName(item.name)
       << "__device));\n";
  }
}

void CuteCodeGen::EmitDeviceFuncDecl(std::ostringstream& oss,
                                     AST::ParallelBy* pb,
                                     const ValueItem& ring_start) {
  const auto& lcs = cgi.GetFunctionLaunches(fname);
  if (!lcs.empty() && lcs[0].HasCluster()) {
    auto& cc = lcs[0].cluster_count;
    auto cluster_total = cc.x * cc.y * cc.z;
    oss << "__cluster_dims__(" << ValueSTR(cluster_total)
        << ", 1, 1) __global__ void " << device_fn << "(";
  } else {
    oss << "__global__ void " << device_fn << "(";
  }

  size_t index = 0;
  for (auto& item : GetDeviceFuncIns(updating_cgi)) {
    if (!PrefixedWith(scoped_symtab.ScopeName(), GetScope(item.name))) continue;
    auto dname = (item.need_iv_prefix ? "__iv_" : "") + UnScopedName(item.name);
    if (index++ > 0) oss << ", ";
    oss << DeviceParamTypeStringify(*item.type) << " " << dname;
    ssm.MapDeviceSymbolIfNotExist(item.name, dname);
  }

  for (auto item : symbolic_dimensions) {
    oss << ((index++ > 0) ? ", unsigned " : "unsigned ");
    oss << UnScopedName(item.first);
    ssm.MapDeviceSymbolIfNotExist(item.first, UnScopedName(item.first));
  }

  if (const auto& mri = FCtx(fname).GetDynMemReuseInfo(SSTab().ScopeName()))
    for (const auto& [sto, ie] : mri->infos)
      for (size_t idx = 0; idx < ie.offset_args.size(); ++idx) {
        auto dname = RegexReplaceAll(ie.offset_args[idx], "::", "_");
        oss << ((index++ > 0) ? ", " : "") << "unsigned long " << dname;
      }

  for (auto desc : cgi.GetTMADesc(pb))
    oss << ", const __grid_constant__ CUtensorMap "
        << desc.GetName() + "_tensor_map";

  if (!ring_start->IsNumeric()) oss << ", unsigned " << ValueSTR(ring_start);

  oss << ")";

  VST_DEBUG(dbgs() << "Device function prototype:\n" << oss.str() << "\n");
}

void CuteCodeGen::EmitSource() {
  for (auto& code : code_segments) {
    if (EnableLineDirective())
      outs() << PinLineDirectivePerGeneratedLine(code) << "\n";
    else
      outs() << code << "\n";
  }
}

uint32_t CuteCodeGen::ContentFingerprint() {
  // FNV-1a hash of embedded precompiled source + runtime headers.
  // Any change to the precompiled runtime (or the headers it includes)
  // produces a different fingerprint, invalidating the cache.
  uint32_t h = 2166136261u;
  auto feed = [&](const char* s) {
    for (; *s; ++s) {
      h ^= static_cast<uint8_t>(*s);
      h *= 16777619u;
    }
  };
  feed(__choreo_precompiled_cu_as_string);
  feed(__choreo_header_as_string);
  feed(__choreo_cute_header_as_string);
  return h;
}

void CuteCodeGen::EmitFastCompileCache(std::ostream& os,
                                       const std::string& precomp_cu) {
  char fp_hex[12];
  std::snprintf(fp_hex, sizeof(fp_hex), "%08x", ContentFingerprint());

  // XDG-compliant cache directory
  os << R"(CHOREO_CACHE_DIR="${CHOREO_CACHE_DIR:-${XDG_CACHE_HOME:-${HOME}/.cache}/choreo}")"
     << "\n";

  // Verify we can create/write the cache directory
  os << R"(if ! mkdir -p "${CHOREO_CACHE_DIR}" 2>/dev/null || [ ! -w "${CHOREO_CACHE_DIR}" ]; then)"
     << "\n";
  os << R"(  echo "[choreo-fc] Warning: cannot write to ${CHOREO_CACHE_DIR}, using temp dir" >&2)"
     << "\n";
  os << R"(  CHOREO_CACHE_DIR=$(mktemp -d))" << "\n";
  os << "fi\n\n";

  // Cache key includes arch, content fingerprint, and CUDA toolkit version.
  // This ensures different choreo builds, CUDA versions, or architectures
  // never share a precompiled object.
  os << "CUDA_VER=$(${NVCC} --version | grep -oP 'release "
        "\\K[0-9]+\\.[0-9]+')\n";
  os << "PRECOMP_CACHED=${CHOREO_CACHE_DIR}/"
        "choreo_precompiled_${nv_arch}_cuda${CUDA_VER}_"
     << fp_hex << ".o\n\n";

  // Build with flock to prevent concurrent builds from colliding.
  os << R"(if [ ! -f "${PRECOMP_CACHED}" ]; then)" << "\n";
  os << R"(  LOCK_FILE="${CHOREO_CACHE_DIR}/.choreo_precompile.lock")" << "\n";
  os << R"(  ()" << "\n";
  os << R"(    flock -x 200)" << "\n";
  os << R"(    if [ ! -f "${PRECOMP_CACHED}" ]; then)" << "\n";
  os << R"(      echo "[choreo-fc] Building precompiled CuTe runtime for ${nv_arch} (one-time)..." >&2)"
     << "\n";
  os << "      ${NVCC} -dc ${DCFLAGS} " << precomp_cu
     << " -o \"${PRECOMP_CACHED}.tmp\" && \\\n";
  os << R"(      mv "${PRECOMP_CACHED}.tmp" "${PRECOMP_CACHED}")"
     << "\n";
  os << R"(      echo "[choreo-fc] Cached at ${PRECOMP_CACHED}" >&2)"
     << "\n";
  os << R"(    fi)" << "\n";
  os << R"(  ) 200>"${LOCK_FILE}")" << "\n";
  os << "fi\n\n";

  // Final check: the precompiled object must exist
  os << R"(if [ ! -f "${PRECOMP_CACHED}" ]; then)" << "\n";
  os << R"(  echo "[choreo-fc] Error: failed to build precompiled runtime at ${PRECOMP_CACHED}" >&2)"
     << "\n";
  os << "  exit 1\nfi\n\n";
}

void CuteCodeGen::EmitScript(std::ostream& os, const std::string& exe_fn) {
  auto filename = RemoveDirectoryPrefix(
      RemoveSuffix(OptionRegistry::GetInstance().GetInputFileName(), ".co"));
  os << R"script(#!/usr/bin/env bash

# This is the choreo generated bash script to compile cute code
)script";

  // we must use the built compilation tools
  if (RequiresE2ECompilation(CCtx().GetOutputKind())) {
#ifdef __CHOREO_CUDA_DIR__
    os << "\nif [ -z \"${CUDA_HOME}\" ]; then";
    os << "\n  export CUDA_HOME=" << STRINGIZE(__CHOREO_CUDA_DIR__);
    os << "\nfi\n";
#endif // __CHOREO_CUDA_DIR__
#ifdef __CHOREO_CUTE_DIR__
    os << "\nif [ -z \"${CUTE_HOME}\" ]; then";
    os << "\n  export CUTE_HOME=" << STRINGIZE(__CHOREO_CUTE_DIR__);
    os << "\nfi\n";
#endif // __CHOREO_CUTE_DIR__
  }

  os << R"script(
if [ ! -n "${CUDA_HOME}" ] || [ ! -f ${CUDA_HOME}/bin/nvcc ]; then
  echo "failed to find the CUDA installation."
  echo "install cuda or set CUDA_HOME to cuda installation directory."
  exit 1
fi

if [ ! -n "${CUTE_HOME}" ] || [ ! -f ${CUTE_HOME}/include/cutlass/cutlass.h ]; then
  echo "failed to find the CUTE installation."
  echo "install cuda or set CUTE_HOME to cute installation directory."
  exit 1
fi

NVCC=${CUDA_HOME}/bin/nvcc
NVCC_LIB=${CUDA_LIB}/lib

)script";

  std::string build_path = CreateUniquePath();
  auto cc_file = build_path + "/__choreo_cute_" + filename + ".cu";
  auto exe_file = exe_fn;
  if (exe_file.empty())
    exe_file = build_path + "/__choreo_cute_" + filename + ".exe";
  os << "rm -fr " << build_path << "\n";
  os << "mkdir -p " << build_path << "\n\n";

  // place the choreo header
  if (CCtx().FastCompile()) {
    os << "cat <<'EOF' > " << build_path << "/choreo_device_api.h\n";
    os << __choreo_device_api_header_as_string << "\nEOF\n\n";
  }
  os << "cat <<'EOF' > " << build_path << "/choreo.h\n";
  os << __choreo_header_as_string << "\nEOF\n\n";
  os << "cat <<'EOF' > " << build_path << "/choreo_cute.h\n";
  os << __choreo_cute_header_as_string << "\nEOF\n\n";

  os << "cat <<'EOF' > " << cc_file << "\n";
  for (auto& code : code_segments) {
    if (EnableLineDirective())
      os << PinLineDirectivePerGeneratedLine(code) << "\n";
    else
      os << code << "\n";
  }
  os << "\nEOF\n\n";

  // the arch type
  auto arch_str = ToLower(CCtx().GetArch());
  os << "nv_arch=" << arch_str << "\n";

  os << R"script(
show_usage() {
  echo "  Usage: $0 <actions>"
  echo ""
  echo "  Options:"
  echo "   --execute,           Compile and execute"
  echo "   --compile-link,      Compile and link"
  echo "   --compile-module,    Compile and generate the module"
  echo "   --gen-fatbin,        Compile and generate the fatbin"
  echo "   --lib,               Compile and generate the lib"
  echo ""
  echo "  Environment Variables:"
  echo "   CUDA_HOME:           (Must) Cuda compiler installation path"
  echo "   CUTE_HOME:           (Must) Cute header library path"
  echo "   EXTRA_TARGET_CFLAGS: Extra target compilation flags"
  exit 1
}

# compile, execute
)script";

  bool arch_has_a_suffix = arch_str.size() > 1 && arch_str.back() == 'a';
  if (arch_has_a_suffix) {
    auto compute_str = arch_str;
    compute_str.replace(0, 3, "compute_");
    os << R"(export CFLAGS="-gencode arch=)" << compute_str
       << ",code=" << arch_str
       << R"( -std=c++17 -DCUTLASS_ENABLE_TENSOR_CORE_MMA=1 -D__CHOREO_TARGET_CUTE__ -Xcompiler -static-libstdc++ -lcuda)";
  } else {
    os << R"(export CFLAGS="-arch ${nv_arch} -std=c++17 -DCUTLASS_ENABLE_TENSOR_CORE_MMA=1 -D__CHOREO_TARGET_CUTE__ -Xcompiler -static-libstdc++ -lcuda)";
  }
  if (extended_mma) os << " -DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED ";
  if (CCtx().TargetDebugInfo())
    os << " -O0";
  else
    os << " -O" << CCtx().GetOptimizationLevel();
  if (use_cuda_type)
    os << " -D__USE_CUDA_TYPE__";
  else
    os << " -D__USE_CUTE_TYPE__";

  if (CCtx().TargetDebugInfo()) os << " -g -G";
  if (CCtx().DMADiagnosis()) os << " -D__CHOREO_DMA_DIAGNOSIS__";
  if (!target_options.GetValue().empty())
    os << " " << target_options.GetValue();
  if (use_pic) os << " -fPIC";
  if (verbose) os << " -v"; // if it requires to be verbose
  os << " --expt-relaxed-constexpr";
  // always enclose
  os << " ${EXTRA_TARGET_CFLAGS}";
  std::filesystem::path cwd = std::filesystem::current_path();
  auto input_file = OptionRegistry::GetInstance().GetInputFileName();
  auto input_abs_path = GetAbsPath(cwd.string(), input_file);
  os << " -I" << input_abs_path;
  os << " -I${CUTE_HOME}/include";
  for (auto inc_path : CCtx().GetIncPaths()) os << " -I" << inc_path;
  for (auto lib_path : CCtx().GetLibPaths()) os << " -L" << lib_path;
  for (auto lib : CCtx().GetLibs()) os << " -l" << lib;
  for (auto macro : CCtx().GetCLMacros())
    os << " -D" << macro.first
       << (macro.second.empty() ? "" : ("=" + macro.second));

  os << " -L${CUDA_HOME}/lib64 -lcuda\"";
  os << "\nexport LD_LIBRARY_PATH=${CUDA_LIB}:${LD_LIBRARY_PATH}\n\n";

  if (CCtx().FastCompile()) {
    // --- Fast-compile mode: separate compilation + precompiled runtime ---
    // Compiles kernel with nvcc -dc and links with a cached precompiled
    // CuTe runtime object. The cache is keyed by arch, CUDA version, and
    // a content fingerprint so different choreo versions never collide.
    auto precomp_cu = build_path + "/choreo_precompiled.cu";
    auto kernel_obj = build_path + "/kernel.o";

    // Compile-only flags: strip -l/-L from CFLAGS for nvcc -dc
    os << R"(DCFLAGS=$(echo "${CFLAGS}" | sed 's/ -l[^ ]*//g; s/ -L[^ ]*//g'))"
       << "\n\n";

    // Write the precompiled runtime source
    os << "cat <<'PRECOMP_EOF' > " << precomp_cu << "\n";
    os << __choreo_precompiled_cu_as_string << "\nPRECOMP_EOF\n\n";

    // Content fingerprint (computed at choreo build time)
    EmitFastCompileCache(os, precomp_cu);

    os << R"(if [ "$1" == "--execute" ] || [ "$#" -eq 0 ]; then)" << "\n";
    os << "  ${NVCC} -dc ${DCFLAGS} " << cc_file << " -o " << kernel_obj
       << "\n";
    os << "  ${NVCC} ${CFLAGS} " << kernel_obj << " ${PRECOMP_CACHED} -o "
       << exe_file << "\n";
    os << "  " << exe_file << "\n";
    os << R"(elif [ "$1" == "--compile-module" ]; then)" << "\n";
    os << "  ${NVCC} -dc ${DCFLAGS} " << cc_file << " -o " << exe_file << "\n";
    os << R"(elif [ "$1" == "--compile-link" ]; then)" << "\n";
    os << "  ${NVCC} -dc ${DCFLAGS} " << cc_file << " -o " << kernel_obj
       << "\n";
    os << "  ${NVCC} ${CFLAGS} " << kernel_obj << " ${PRECOMP_CACHED} -o "
       << exe_file << "\n";
    os << R"(elif [ "$1" == "--lib" ]; then)" << "\n";
    os << "  ${NVCC} --lib -Xcompiler -fPIC ${CFLAGS} " << cc_file << " -o "
       << exe_file << "\n";
    os << "\nelse show_usage";
    os << "\nfi";
  } else {
    // --- Standard monolithic compilation ---
    os << R"(if [ "$1" == "--execute" ] || [ "$#" -eq 0 ]; then)";
    if (verbose)
      os << "\n  echo ${NVCC} ${CFLAGS} " << cc_file << " -o " << exe_file;
    os << "\n  ${NVCC} ${CFLAGS} " << cc_file << " -o " << exe_file;
    if (verbose) os << "\n  echo " << exe_file << "\n";
    os << "\n  " << exe_file << "\n";
    os << R"(elif [ "$1" == "--compile-module" ]; then)";
    if (verbose)
      os << "\n  echo ${NVCC} -c ${CFLAGS} " << cc_file << " -o " << exe_file
         << "\n";
    os << "\n  ${NVCC} -c ${CFLAGS} " << cc_file << " -o " << exe_file << "\n";
    os << R"(elif [ "$1" == "--compile-link" ]; then)";
    if (verbose)
      os << "\n  echo ${NVCC} ${CFLAGS} " << cc_file << " -o " << exe_file
         << "\n";
    os << "\n  ${NVCC} ${CFLAGS} " << cc_file << " -o " << exe_file << "\n";
    os << R"(elif [ "$1" == "--lib" ]; then)";
    if (verbose)
      os << "\n  echo ${NVCC} --lib -Xcompiler -fPIC ${CFLAGS} " << cc_file
         << " -o " << exe_file << "\n";
    os << "\n  ${NVCC} --lib -Xcompiler -fPIC ${CFLAGS} " << cc_file << " -o "
       << exe_file << "\n";
    os << "\nelse show_usage";
    os << "\nfi";
  }
}

bool CuteCodeGen::CompileWithScript(const std::string& action) {
  assert(!action.empty() && "no action is specified.");

  char tempFileName[] = "/tmp/choreo_cute_script_XXXXXX";
  int fd = mkstemp(tempFileName);
  if (fd == -1) {
    errs() << "Failed to create temporary file.\n";
    return false;
  }
  close(fd);

  // Open the file for writing
  std::ofstream tempFile(tempFileName);
  if (!tempFile) {
    errs() << "Failed to open temporary file for writing.\n";
    return false;
  }

  auto outfile = OptionRegistry::GetInstance().GetOutputFileName();
  EmitScript(tempFile, outfile);
  tempFile.close(); // important: make sure the temp file is closed

  // Execute the file
  std::string command = "bash " + std::string(tempFileName) + " " + action;
  VST_DEBUG(dbgs() << "Compile " << outfile << ": " << command << "\n");
  int result = system(command.c_str());
  if (result == -1) {
    errs() << "Failed to execute the file.\n";
    return false;
  }

  // Remove the temporary file
  if (remove(tempFileName) != 0) {
    errs() << "Failed to remove the temporary file.\n";
    return false;
  }

  return true;
}

// TODO: eliminate the need of the value replacement?
// Currently, it is guaranteed that ValueSTR can be used safely and directly.
const std::string CuteCodeGen::ValueSTR(const ValueItem& vi, bool LL_suffix,
                                        bool shp_lit) const {
  return OpValueSTR(vi, "", true, LL_suffix, shp_lit);
}

const std::string CuteCodeGen::ValueSTR(const ValueList& vl, bool LL_suffix,
                                        bool shp_lit,
                                        const std::string& sep) const {
  std::ostringstream oss;
  if (!vl.empty()) {
    oss << ValueSTR(vl[0], LL_suffix, shp_lit);
    for (unsigned i = 1; i < vl.size(); ++i)
      oss << sep << ValueSTR(vl[i], LL_suffix, shp_lit);
  }
  return oss.str();
}

const std::string CuteCodeGen::OpValueSTR(const ValueItem& vi,
                                          const std::string& parent_op,
                                          const bool is_left_child,
                                          bool LL_suffix, bool shp_lit) const {
  auto WrapParen = [&](const std::string& s, const std::string& cur_op) {
    if (Operator::NeedParen(cur_op, parent_op, is_left_child))
      return "(" + s + ")";
    // Used to ensure the above guarantee.
    if (parent_op == "") return "(" + s + ")";
    return s;
  };

  if (!IsValidValueItem(vi)) choreo_unreachable("invalid value item.");
  if (VIIsNil(vi)) {
    if (IsHost())
      return "choreo::__inf__";
    else
      return "-1"; // it looks the API requires -1
  } else if (auto iv = VIInt(vi)) {
    if (iv >= (int64_t)std::numeric_limits<int32_t>::max() ||
        iv <= (int64_t)std::numeric_limits<int32_t>::min()) {
      if (shp_lit)
        choreo_unreachable("unable to represent a LLONG shape dimension.");
      else
        return PSTR(vi) + "LL";
    } else if (shp_lit)
      return "cute::Int<" + PSTR(vi) + ">{}";
    else if (LL_suffix)
      return PSTR(vi) + "LL";
    else
      return PSTR(vi);
  } else if (auto bv = VIBool(vi))
    return PSTR(vi);
  else if (auto sv = VISym(vi)) {
    auto res = UnScopedExpr(SSMName(sv.value(), IsHost()));
    if (LL_suffix) return "static_cast<long long>(" + res + ")";
    return res;
  } else if (auto uo = VIUop(vi)) {
    std::string op = STR(uo->GetOpCode());
    std::string res =
        op + OpValueSTR(uo->GetOperand(), op, false, LL_suffix, shp_lit);
    return WrapParen(res, op);
  } else if (auto bo = VIBop(vi)) {
    if (bo->GetOpCode() == OpCode::ADD) {
      if (auto rv = VIInt(bo->GetRight()); rv && rv.value() < 0) {
        std::string res =
            OpValueSTR(bo->GetLeft(), "-", true, LL_suffix, shp_lit) + " - " +
            std::to_string(-rv.value());
        if (rv.value() >= (int64_t)std::numeric_limits<int32_t>::max() ||
            rv.value() <= (int64_t)std::numeric_limits<int32_t>::min())
          res += "LL";
        return WrapParen(res, "-");
      }
    }
    std::string op = STR(bo->GetOpCode());
    std::string res = OpValueSTR(bo->GetLeft(), op, true, LL_suffix, shp_lit) +
                      " " + op + " " +
                      OpValueSTR(bo->GetRight(), op, false, LL_suffix, shp_lit);
    return WrapParen(res, op);
  } else if (auto to = VITop(vi)) {
    std::string op = "?";
    std::string res = OpValueSTR(to->GetPred(), op, true) + " ? " +
                      OpValueSTR(to->GetLeft(), op, true, LL_suffix, shp_lit) +
                      " : " +
                      OpValueSTR(to->GetRight(), op, false, LL_suffix, shp_lit);
    return WrapParen(res, op);
  } else
    choreo_unreachable("unsupported value.");
  return "";
}

// input is a `node` or `std::variant<int, float>`.
// If `val` is existed, use it first.
const std::string CuteCodeGen::ExprCastSTR(
    AST::ptr<AST::Node> n, std::optional<std::variant<int, float>> val,
    BaseType t, BaseType f, bool is_host, bool is_explicit) const {
  std::ostringstream res;
  std::string value;

  if (val.has_value()) {
    auto v = val.value();
    if (std::holds_alternative<int>(v))
      value = std::to_string(std::get<int>(v));
    else if (std::holds_alternative<float>(v))
      value = std::to_string(std::get<float>(v)) + "f";
    else
      choreo_unreachable("unexpect type of v");
  } else {
    assert(n);
    // using "" as op, cause `value` is always used inside `()`
    value = OpExprSTR(n, "", true, is_host);
  }

  if (f == t) return value;

  using BT = BaseType;

  // need to do casting or converting.
  if (!IsValuePreservingCast(f, t) && !is_explicit) {
    if (IsReinterpretiveCast(f, t))
      Warning(n->LOC(), "The implicit type conversion may lead to semantic "
                        "error(without data loss): '" +
                            STR(f) + "' to '" + STR(t) + "'");
    else if (IsLossyCast(f, t))
      Warning(n->LOC(), "The implicit type conversion may lose precision: '" +
                            STR(f) + "' to '" + STR(t) + "'");
    else
      choreo_unreachable("unexpect cast");
  }

  switch (t) {
  case BT::S64: [[fallthrough]];
  case BT::U64: [[fallthrough]];
  case BT::S32: [[fallthrough]];
  case BT::U32: [[fallthrough]];
  case BT::S16: [[fallthrough]];
  case BT::U16: [[fallthrough]];
  case BT::S8: [[fallthrough]];
  case BT::U8: {
    auto nbt = NameBaseType(t);
    if (IsIntegralType(f))
      res << "static_cast<" << nbt << ">(" << value << ")";
    else if (IsFloatType(f)) {
      if (f != BT::F32 && f != BT::F64)
        res << "static_cast<" << nbt << ">("
            << ExprCastSTR(n, val, BT::F32, f, is_host) << ")";
      else
        res << "static_cast<" << nbt << ">(" << value << ")";
    }
    break;
  }
  case BT::F64: {
    if (IsIntegralType(f))
      res << "static_cast<double>(" << value << ")";
    else
      res << "static_cast<double>(" << ExprCastSTR(n, val, BT::F32, f, is_host)
          << ")";
    break;
  }
  case BT::F32: {
    res << (is_host ? "choreo::to_f32(" : "static_cast<float>(") << value
        << ")";
    break;
  }
  case BT::F16:
    res << "choreo::f32_to_f16(" << ExprCastSTR(n, val, BT::F32, f, is_host)
        << ")";
    break;
  case BT::BF16:
    res << "choreo::bf16(" << ExprCastSTR(n, val, BT::F32, f, is_host) << ")";
    break;
  case BT::F8_E4M3:
  case BT::F8_E5M2:
    res << "choreo::utils::from_f32<"
        << ((t == BT::F8_E4M3) ? "f8_e4m3" : "f8_e5m2") << ">("
        << ExprCastSTR(n, val, BT::F32, f, is_host) << ")";
    break;
  default:
    choreo_unreachable("unsupport cast: '" + STR(f) + "' to '" + STR(t) + "'");
  }

  return res.str();
}

const std::string CuteCodeGen::ExprSTR(AST::ptr<AST::Node> e,
                                       bool is_host) const {
  // start with the lowest precedence op ""
  return OpExprSTR(e, "", true, is_host);
}

const ValueItem CuteCodeGen::TileAddr(const ptr<AST::ChunkAt>& ca, bool is_host,
                                      ValueItem scale) const {
  auto caty = cast<SpannedType>(ca->GetType());

  auto offset = GenOffset(ca) * scale;
  ValueItem base;
  if (auto fty = dyn_cast<FutureType>(NodeType(*ca->data))) {
    base = sbe::sym(std::string("(") + NameBaseType(fty->ElementType()) + "*)" +
                    ExprSTR(ca->data, is_host) + ".data()");
  } else
    base = sbe::sym(ExprSTR(ca->data, is_host));

  if (ca->indices != nullptr) {
    auto sym_ty = GetSymbolType(ca->data->name);
    if (auto array_ty = dyn_cast<ArrayType>(sym_ty);
        array_ty && CCtx().MemReuse()) {
      std::string array_idx;
      auto subscriptions = ca->indices->AllValues();
      const ValueList& array_sizes = array_ty->Dimensions();
      for (size_t i = 0; i < subscriptions.size(); ++i) {
        if (array_idx.empty())
          array_idx = ExprSTR(subscriptions[i], is_host);
        else
          array_idx = "(" + array_idx + ")*" + ValueSTR(array_sizes[i]) + "+" +
                      ExprSTR(subscriptions[i], is_host);
      }
      auto f_sty = GetSpannedType(sym_ty);
      std::string elem_count = ValueSTR(f_sty->GetShape().ElementCountValue());
      offset = offset + sbe::sym("(" + array_idx + ")*(" + elem_count + ")");
    }
  }

  return base + offset;
}

const std::string CuteCodeGen::OpExprSTR(AST::ptr<AST::Node> e,
                                         const std::string& parent_op,
                                         bool is_left_child,
                                         bool is_host) const {
  std::ostringstream oss;

  // If output a expression with op to `oss`, then the expr maybe should
  // be wrapped with parentheses, e.g. `oss << a << "+" << b`,
  // should result in `WrapParen("a + b", "+")`
  // If `parent_op` is "*", then parentheses is necessary. If `parent_op` is
  // "-", for `is_left` == right, need parentheses: `xxx - (a + b)`
  auto WrapParen = [&](const std::string& s, const std::string& cur_op) {
    if (Operator::NeedParen(cur_op, parent_op, is_left_child))
      return "(" + s + ")";
    return s;
  };

  if (auto id = dyn_cast<AST::Identifier>(e)) {
    if (id->name == "__choreo_no_tiling__") {
      assert(!is_host);
      return id->name;
    }
    if (within_map.count(InScopeName(id->name)) && !is_host) {
      size_t i = 0;
      for (auto iv_name : within_map.at(InScopeName(id->name)))
        oss << ((i++ == 0) ? "" : ", ")
            << UnScopedName(ssm.DeviceName(iv_name));
    } else {
      oss << UnScopedName(SSMName(InScopeName(id->name), is_host));
    }
  } else if (auto il = dyn_cast<AST::IntLiteral>(e)) {
    oss << il->ValAsString();
  } else if (auto fl = dyn_cast<AST::FloatLiteral>(e)) {
    std::ostringstream fp_val;
    // std::fixed: the value should be in fixed-point notation
    // otherwise, 1.0f => 1f (error)
    if (fl->IsFloat32())
      fp_val << std::fixed << fl->Val_f32() << "f";
    else if (fl->IsFloat64())
      fp_val << std::fixed << fl->Val_f64();
    else
      choreo_unreachable("unsupported float literal.");
    oss << fp_val.str();
  } else if (auto sl = dyn_cast<AST::StringLiteral>(e)) {
    oss << sl->EscapedVal();
  } else if (auto b = dyn_cast<AST::BoolLiteral>(e)) {
    oss << b->value;
  } else if (auto ii = dyn_cast<AST::IntIndex>(e)) {
    // currently, value of IntIndex is always IntLiteral or Identifier
    return OpExprSTR(ii->value, parent_op, true, is_host);
  } else if (auto it = dyn_cast<AST::IntTuple>(e)) {
    if (EnableDebugTypeRTTI()) {
      int i = 0;
      oss << "{";
      for (auto& v : it->GetValues()->AllValues()) {
        if (i++ > 0) oss << ", ";
        oss << ExprSTR(v, is_host);
      }
      oss << "}";
    }
  } else if (auto da = dyn_cast<AST::DataAccess>(e)) {
    if (da->AccessElement()) {
      auto sty = GetSpannedType(GetSymbolType(da->data->name));
      assert(sty && "can only access the element of a spanned type.");
      oss << "*((" << NameBaseType(sty->ElementType()) << "*)"
          << OpExprSTR(da->data, "+", true, is_host);
      size_t idx = 0;
      auto shape = sty->GetShape();
      auto AppendOffset = [this, &oss, &shape, &idx](const ValueItem& op) {
        auto offset = op;
        assert(shape.Rank() >= idx + 1);
        if (shape.Rank() > idx + 1)
          offset = offset * shape.TrimDims(idx + 1).ElementCountValue();
        SimplifyExpression(offset);
        if (!sbe::ceq(offset, sbe::nu(0))) oss << " + " << ValueSTR(offset);
        ++idx;
      };
      for (auto item : da->GetIndices()) {
        if (auto id = AST::GetIdentifier(item)) {
          if (within_map.count(InScopeName(id->name))) {
            auto ivs = within_map.at(InScopeName(id->name));
            for (auto iv_itr = ivs.begin(); iv_itr != ivs.end(); ++iv_itr)
              AppendOffset(sbe::sym(*iv_itr));
          } else
            AppendOffset(sbe::sym(InScopeName(id->name)));
        } else if (auto il = AST::GetIntLiteral(*item)) {
          AppendOffset(sbe::nu(il->Val()));
        } else {
          oss << " + ";
          assert(shape.Rank() >= idx + 1);
          if (shape.Rank() > idx + 1)
            oss << OpExprSTR(item, "*", true, is_host) << "*"
                << ValueSTR(shape.TrimDims(idx + 1).ElementCountValue());
          else
            oss << OpExprSTR(item, "+", false, is_host);
          ++idx;
        }
      }
      oss << ")";
    } else {
      auto sym = da->data->name;
      if (auto sty = GetSpannedType(GetSymbolType(sym))) {
        oss << UnScopedName(SSMName(InScopeName(sym), is_host));
      } else {
        assert(!within_map.count(InScopeName(da->data->name)));
        oss << UnScopedName(SSMName(InScopeName(sym), is_host));
      }
    }
  } else if (auto ce = dyn_cast<AST::CastExpr>(e)) {
    // codegen for scalar type cast
    assert(ce->GetOp() == Op::Cast);
    return ExprCastSTR(ce->GetR(), std::nullopt, ce->ToType(), ce->FromType(),
                       is_host, ce->IsExplicit());
  } else if (auto expr = dyn_cast<AST::Expr>(e)) {
    // utilize the optimize value whenever possible
    if (auto sym = expr->GetSymbol()) {
      auto sname = InScopeName(sym->name);
      if (FCtx(fname).HasSymbolValues(sname)) {
        auto svs = FCtx(fname).GetSymbolValues(sname);
        if (svs.HasVal()) return ValueSTR(svs.GetVal());
      }
    }

    if (ConvertibleToInt(NodeType(*e)))
      if (expr->Opts().HasVal()) return ValueSTR(expr->Opts().GetVal());

    if (expr->IsReference()) {
      if (PSTR(expr) == "_") return "0";
      return OpExprSTR(expr->GetReference(), parent_op, is_left_child, is_host);
    } else if (expr->IsUnary()) {
      if (expr->GetOp() == Op::LogicNot) {
        oss << "!"
            << WrapParen(OpExprSTR(expr->GetR(), "!", false, is_host), "!");
      } else if (expr->GetOp() == Op::GetUBound) {
        auto rty = cast<BoundedType>(NodeType(*expr->GetR()));
        if (rty->Dims() == 1) oss << ValueSTR(rty->GetUpperBound());
      } else if (expr->GetOp() == Op::DataOf || expr->GetOp() == Op::MDataOf) {
        assert(isa<FutureType>(expr->GetR()->GetType()) &&
               "expect a future operand.");
        if (auto id = cast<AST::Expr>(expr->GetR())->GetSymbol()) {
          if (is_host)
            oss << id->name << "__buf__";
          else
            oss << id->name
                << (expr->GetOp() == Op::MDataOf ? ".mdata()" : ".data()");
        } else
          choreo_unreachable("Can not retrieve name of the future.");
      } else if (expr->GetOp() == Op::SizeOf) {
        auto se = expr->Opts().GetSize();
        if (IsValidValueItem(se))
          oss << ValueSTR(se);
        else {
          // TODO: deprecate this implementation
          auto var = RemoveSuffix(*AST::GetName(*expr->GetR()), ".span");
          auto shape = GetShape(GetSymbolType(var));
          assert(shape.IsValid() && "Invalid shape is found");
          oss << ValueSTR(shape.ElementCountValue());
        }
      } else if (expr->GetOp() == Op::PreInc) {
        oss << "++"
            << WrapParen(OpExprSTR(expr->GetR(), "++", false, is_host), "++");
      } else if (expr->GetOp() == Op::PreDec) {
        oss << "--"
            << WrapParen(OpExprSTR(expr->GetR(), "--", false, is_host), "--");
      } else if (expr->GetOp() == Op::AddrOf) {
        if (auto id = AST::GetIdentifier(expr->GetR()))
          oss << OpExprSTR(id, parent_op, is_left_child, is_host);
        else if (isa<AST::DataAccess>(expr->GetR()))
          oss << "&"
              << WrapParen(OpExprSTR(expr->GetR(), "&", false, is_host), "&");
        else
          choreo_unreachable("Can not retrieve name of the spanned data.");
      } else if (expr->GetOp() == Op::BitNot) {
        oss << "~"
            << WrapParen(OpExprSTR(expr->GetR(), "~", false, is_host), "~");
      } else
        choreo_unreachable("unsupported expression op: '" + STR(expr->GetOp()) +
                           "', expr: " + PSTR(expr) + ".");
    } else if (expr->IsBinary()) {
      if (expr->GetOp() == Op::CeilDiv) {
        std::string one = "1";
        auto L = OpExprSTR(expr->GetL(), "+", true, is_host);
        auto R0 = OpExprSTR(expr->GetR(), "+", false, is_host);
        auto R1 = OpExprSTR(expr->GetR(), "/", false, is_host);
        // (L + R0 - 1) / R1
        std::ostringstream res;
        res << "(" << L << " + " << R0 << " - " << one << ") / " << R1;
        oss << WrapParen(res.str(), "/");
      } else if (expr->GetOp() == Op::GetIth) {
        auto lty = cast<BoundedType>(NodeType(*expr->GetL()));
        auto r = expr->GetR();
        if (cast<AST::IntIndex>(r)->IsNegative()) {
          std::ostringstream res;
          // special case: str of R is always a negative integer.
          res << ValueSTR(lty->GetUpperBound()) << " + "
              << OpExprSTR(r, "+", false, is_host);
          oss << WrapParen(res.str(), "+");
        } else
          oss << OpExprSTR(r, parent_op, is_left_child, is_host);
      } else if (expr->GetOp() == Op::ElemOf) {
        auto base_id = AST::GetArrayBaseSymbol(*expr);
        auto base_ty = GetSymbolType(base_id->name);
        if (auto sat = dyn_cast<SpannedArrayType>(base_ty);
            sat && CCtx().MemReuse() && !e->HasNote("mma_frag")) {
          std::vector<ptr<AST::Node>> subscripts;
          ptr<AST::Expr> cur = expr;
          while (cur->GetOp() == Op::ElemOf) {
            subscripts.push_back(cur->GetR());
            if (isa<AST::Identifier>(cur->GetL())) break;
            cur = cast<AST::Expr>(cur->GetL());
          }
          std::reverse(subscripts.begin(), subscripts.end());
          auto base_dev = ssm.DeviceName(InScopeName(base_id->name));
          oss << LinearizeArrayOffset(base_dev, subscripts, sat->Dimensions(),
                                      sat->spty->GetShape().ElementCountValue(),
                                      is_host);
        } else {
          oss << OpExprSTR(expr->GetL(), "[]", true, is_host) << "["
              << OpExprSTR(expr->GetR(), "", true, is_host) << "]";
        }
      } else if (expr->IsArith() || expr->IsLogical() || expr->IsCompare() ||
                 expr->isBitwise()) {
        auto& l = expr->GetL();
        auto& r = expr->GetR();
        auto lty = NodeType(*l);
        auto rty = NodeType(*r);
        auto& op = expr->GetOp();
        auto op_str = STR(op);

        auto IsFp8Scalar = [](const ptr<Type>& ty) -> bool {
          if (auto sty = dyn_cast<ScalarType>(ty)) {
            auto bt = sty->GetBaseType();
            return bt == BaseType::F8_E4M3 || bt == BaseType::F8_E5M2 ||
                   bt == BaseType::F8_UE8M0 || bt == BaseType::F8_UE4M3;
          }
          return false;
        };

        auto ToF32 = [is_host](const std::string& s) -> std::string {
          if (is_host) return "choreo::to_f32(" + s + ")";
          return "static_cast<float>(" + s + ")";
        };
        if (isa<SpannedType>(lty) || isa<SpannedType>(rty)) {
          oss << EmitSpannedArith(*expr);
        } else if (op == Op::UBound && IsActualBoundedIntegerType(lty) &&
                   IsActualBoundedIntegerType(rty)) {
          auto rty = cast<BoundedType>(NodeType(*r));
          assert(rty->Dims() == 1);
          std::string r_upper_bound;
          if (PSTR(r) == "_")
            r_upper_bound = "1";
          else
            r_upper_bound = ValueSTR(rty->GetUpperBound());
          auto L = OpExprSTR(l, "*", true, is_host);
          auto R = OpExprSTR(r, "+", false, is_host);
          std::ostringstream res;
          res << L << " * " << r_upper_bound << " + " << R;
          oss << WrapParen(res.str(), "+");
        } else if ((op == Op::UBoundAdd || op == Op::UBoundSub) &&
                   IsActualBoundedIntegerType(lty) &&
                   isa<ScalarIntegerType>(rty)) {
          oss << OpExprSTR(l, parent_op, is_left_child, is_host);
        } else if (op == Op::UBoundDiv || op == Op::UBoundScale ||
                   op == Op::UBoundMod) {
          choreo_unreachable("unsupported expression op: '" +
                             STR(expr->GetOp()) + "', expr: " + PSTR(expr) +
                             ".");
        } else {
          std::ostringstream res;
          // FP8 scalar arithmetic: upcast operands to FP32 first.
          // This avoids relying on FP8 operator overloads which may not exist.
          if (expr->IsArith() && (IsFp8Scalar(lty) || IsFp8Scalar(rty))) {
            res << ToF32(OpExprSTR(l, op_str, true, is_host)) << " " << op_str
                << " " << ToF32(OpExprSTR(r, op_str, false, is_host));
          } else {
            res << OpExprSTR(l, op_str, true, is_host) << " " << op_str << " "
                << OpExprSTR(r, op_str, false, is_host);
          }
          oss << WrapParen(res.str(), op_str);
        }
      }
    } else if (expr->IsTernary()) {
      std::ostringstream res;
      res << OpExprSTR(expr->GetC(), "?", true, is_host) << " ? "
          << OpExprSTR(expr->GetL(), "?", true, is_host) << " : "
          << OpExprSTR(expr->GetR(), "?", false, is_host);
      oss << WrapParen(res.str(), "?");
    } else
      choreo_unreachable("unsupported expression op: '" + STR(expr->GetOp()) +
                         "', expr: " + PSTR(expr) + ".");
  } else if (auto ca = dyn_cast<AST::ChunkAt>(e)) {
    return ValueSTR(TileAddr(ca, is_host));
  } else if (auto c = dyn_cast<AST::Call>(e)) {
    assert(!is_host);
    return CallSTR(*c);
  } else if (isa<AST::DataType>(e)) {
    return NameBaseType(e->GetType()->GetBaseType());
  } else
    choreo_unreachable("unsupported node type: " + e->TypeNameString() + ".");

  return oss.str();
}

const std::string CuteCodeGen::CallSTR(AST::Call& n) const {
  std::ostringstream oss;
  auto func_name = [&n](const std::string& name) -> std::string {
    if (!n.IsArith()) return name;
    const std::string prefix = "__";
    std::string func_name = name;
    if (auto res = RemovePrefixOrNull(prefix, name)) func_name = *res;
    return "choreo::nv_cute::numerics::" + func_name;
  };

  oss << func_name(n.function->name);

  // emit template arguments
  if (n.template_args) {
    oss << "<";
    size_t i = 0;
    for (auto& ta : n.template_args->AllValues())
      oss << ((i++ == 0) ? "" : ", ") << OpExprSTR(ta, "", true, IsHost());
    oss << ">";
  }

  oss << "(";
  size_t i = 0;
  for (auto& a : n.GetArguments()) {
    oss << ((i++ == 0) ? "" : ", ");
    if (auto sty = GetSpannedType(NodeType(*a))) {
      std::string bts{NameBaseType(sty->ElementType())};
      auto m_ty = sty->GetStorage();
      auto mem_attr = CudaParamStorage(m_ty);
      if (a->HasNote("annotate_as") && !mem_attr.empty())
        bts = mem_attr + " " + bts;
      if (!no_decay_spanview || IsHost())
        oss << "(" << bts << "*)" << OpExprSTR(a, "", true, IsHost());
      else
        oss << "choreo::make_spanview<" << sty->Dims() << ">((" << bts << "*)"
            << OpExprSTR(a, "", true, IsHost()) << ", " << LSTR(sty->GetShape())
            << ")";
    } else if (n.IsArith())
      oss << OpExprSTR(a, "", true, IsHost());
    else
      oss << UnScopedExpr(OpExprSTR(a, "", true, IsHost()));
  }
  oss << ")";

  return oss.str();
}

const std::string CuteCodeGen::EmitSpannedArith(AST::Expr& e) const {
  std::ostringstream oss;
  oss << "choreo::nv_cute::warp_cooperative::"; // namespace
  bool emitted = false;
  if (e.IsBinary()) {
    auto& l = e.GetL();
    auto& r = e.GetR();
    auto lty = NodeType(*l);
    auto rty = NodeType(*r);
    auto lsty = GetSpannedType(lty);
    auto rsty = GetSpannedType(rty);
    if (lsty && isa<ScalarType>(rty)) {
      if (lsty->GetStorage() == Storage::REG) {
        assert(l->HasNote("update"));
        oss << "fragment_scalar_elementwise(" << ExprSTR(l, false) << ", "
            << ExprSTR(r, false) << ", [](" << NameBaseType(lsty->ElementType())
            << " a, " << NameBaseType(lsty->ElementType())
            << " b) { return a + b; })";
        emitted = true;
      }
    }
  }

  if (!emitted) {
    choreo_unreachable("unsupported spanned arithmetic operation");
    return "";
  }
  return oss.str();
}

void CuteCodeGen::BuildSiteAssertionMap() {
  if (CCtx().DisableRuntimeCheck()) return;
  if (fname.empty()) return;

  for (const auto& ar : FCtx(fname).GetAssertions(AssessType::USE_SITE)) {
    if (!ar.enabled || !ar.EmitTarget()) continue;
    if (ar.emit_position == AssertionEmitPosition::BEFORE_NODE)
      pre_site_assertions[ar.EmitTarget()].push_back(ar);
    else
      post_site_assertions[ar.EmitTarget()].push_back(ar);
  }
  for (const auto& ar : FCtx(fname).GetAssertions(AssessType::HOIST_SITE)) {
    if (!ar.enabled || !ar.EmitTarget()) continue;
    if (ar.emit_position == AssertionEmitPosition::BEFORE_NODE)
      pre_site_assertions[ar.EmitTarget()].push_back(ar);
    else
      post_site_assertions[ar.EmitTarget()].push_back(ar);
  }
}

void CuteCodeGen::EmitPreSiteAssertions(AST::Node& n) {
  if (CCtx().DisableRuntimeCheck()) return;
  auto it = pre_site_assertions.find(&n);
  if (it == pre_site_assertions.end()) return;

  for (const auto& ar : it->second) {
    IndStream() << "choreo::choreo_assert(" << ValueSTR(ar.expr, true) << ", \""
                << ar.message << ", " << ar.loc << "\");\n";
  }
}

void CuteCodeGen::EmitPostSiteAssertions(AST::Node& n) {
  if (CCtx().DisableRuntimeCheck()) return;
  auto it = post_site_assertions.find(&n);
  if (it == post_site_assertions.end()) return;

  // HOIST_SITE and USE_SITE assertions are emitted in device code using
  // choreo_assert (printf-based) because std::cerr is not available on device.
  for (const auto& ar : it->second) {
    IndStream() << "choreo::choreo_assert(" << ValueSTR(ar.expr, true) << ", \""
                << ar.message << ", " << ar.loc << "\");\n";
  }
}
