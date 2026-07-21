#include "cute_codegen.hpp"
#include "codegen_utils.hpp"

#include <algorithm>
#include <cmath>
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
#include "choreo_types_cute_header.inc"
#include "choreo_types_header.inc"
#include "codegen.hpp"
#include "dma_plan.hpp"
#include "operator_info.hpp"

// __CHOREO_CUDA_DIR__ and __CHOREO_CUTE_DIR__ are set by CMake when
// ENABLE_CUDA / ENABLE_CUTE are ON.  Code below uses #ifdef guards
// to degrade gracefully when they are absent.

// #define USING_OP_INFO

using namespace Choreo;
using namespace Choreo::Cute;

// Strip the outermost matching parentheses from a string if the opening
// paren at position 0 matches the closing paren at the end.
static std::string StripOuterParens(const std::string& s) {
  if (s.size() < 2 || s.front() != '(' || s.back() != ')') return s;
  int depth = 0;
  for (size_t i = 0; i < s.size() - 1; ++i) {
    if (s[i] == '(')
      ++depth;
    else if (s[i] == ')')
      --depth;
    if (depth == 0) return s;
  }
  return s.substr(1, s.size() - 2);
}

// Zero is exactly representable by every MMA accumulator type. Treat a
// literal zero fill as an intentional conversion so the generic narrowing
// diagnostics do not report a precision loss that cannot occur.
static bool IsNumericZeroLiteral(const AST::ptr<AST::Node>& n) {
  auto value = AST::Ref(n);
  if (auto integer = dyn_cast<AST::IntLiteral>(value))
    return integer->Val() == 0;
  if (auto floating = dyn_cast<AST::FloatLiteral>(value))
    return std::visit([](auto v) { return v == 0; }, floating->value);
  return false;
}

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
extern Option<bool> use_fast_math;
extern Option<bool> tma_cluster_aware;
extern Option<bool> use_stmatrix;

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

static inline std::string GetCopyAtomType(CUDA_COPY_ATOM atom,
                                          BaseType elem_type) {
  std::string elem_name = NameBaseType(elem_type);
  switch (atom) {
  case CUDA_COPY_ATOM::TMA_ATOM: return std::string("choreo::TMAAtom");
  case CUDA_COPY_ATOM::CP_ASYNC_128B:
    return std::string("cute::Copy_Atom<cute::SM80_CP_ASYNC_CACHEGLOBAL_ZFILL<"
                       "cute::uint128_t>, ") +
           elem_name + ">";
  case CUDA_COPY_ATOM::CP_ASYNC_64B:
    return std::string("cute::Copy_Atom<cute::SM80_CP_ASYNC_CACHEALWAYS_ZFILL<"
                       "cute::uint64_t>, ") +
           elem_name + ">";
  case CUDA_COPY_ATOM::CP_ASYNC_32B:
    return std::string("cute::Copy_Atom<cute::SM80_CP_ASYNC_CACHEALWAYS_ZFILL<"
                       "cute::uint32_t>, ") +
           elem_name + ">";
  case CUDA_COPY_ATOM::ASYNC_COPY: {
    std::string copy_elem =
        IsFloatSubByteType(elem_type) ? "uint8_t" : elem_name;
    return std::string("cute::Copy_Atom<cute::UniversalCopy<") + copy_elem +
           ">, " + copy_elem + ">";
  }
  case CUDA_COPY_ATOM::VEC_128B:
    return std::string(
               "cute::Copy_Atom<cute::AutoVectorizingCopyWithAssumedAlignment<"
               "128>, ") +
           elem_name + ">";
  case CUDA_COPY_ATOM::VEC_64B:
    return std::string(
               "cute::Copy_Atom<cute::AutoVectorizingCopyWithAssumedAlignment<"
               "64>, ") +
           elem_name + ">";
  case CUDA_COPY_ATOM::VEC_32B:
    return std::string(
               "cute::Copy_Atom<cute::AutoVectorizingCopyWithAssumedAlignment<"
               "32>, ") +
           elem_name + ">";
  case CUDA_COPY_ATOM::VEC_16B:
    return std::string(
               "cute::Copy_Atom<cute::AutoVectorizingCopyWithAssumedAlignment<"
               "16>, ") +
           elem_name + ">";
  case CUDA_COPY_ATOM::UNIVERSAL_COPY: {
    std::string copy_elem =
        IsFloatSubByteType(elem_type) ? "uint8_t" : elem_name;
    return std::string("cute::Copy_Atom<cute::UniversalCopy<") + copy_elem +
           ">, " + copy_elem + ">";
  }
  case CUDA_COPY_ATOM::UNKNOWN: return "unknown";
  }
  return "unknown";
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

size_t GetWGMMAKAxis(const MMAInfo& ssmi) {
  switch (ssmi.frag) {
  case MMAInfo::FRAG_A:
    switch (ssmi.method) {
    case AST::MMAOperation::ROW_ROW:
    case AST::MMAOperation::ROW_COL: return 1;
    case AST::MMAOperation::COL_ROW:
    case AST::MMAOperation::COL_COL: return 0;
    }
    break;
  case MMAInfo::FRAG_B:
    switch (ssmi.method) {
    case AST::MMAOperation::ROW_ROW:
    case AST::MMAOperation::COL_ROW: return 1;
    case AST::MMAOperation::ROW_COL:
    case AST::MMAOperation::COL_COL: return 0;
    }
    break;
  default: break;
  }
  return 1;
}
} // namespace

const AST::MMAOperation*
CuteCodeGen::FindFirstScaledWGMMAExec(const ptr<AST::Node>& n) const {
  (void)n;
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
                                  const std::string& indent,
                                  int thread_count) const {
  int tc =
      thread_count > 0
          ? thread_count
          : (CurrentScopeThreadsCount() > 0 ? CurrentScopeThreadsCount() : 128);
  os << indent << "choreo::named_barrier_sync(" << std::to_string(tc)
     << ", 15);\n";
}

void CuteCodeGen::EmitWGMMAFinalize(std::ostringstream& os,
                                    const std::string& indent,
                                    bool force_wait) {
  os << indent << "warpgroup_commit_batch();\n";
  if (!has_explicit_mma_wait || force_wait)
    os << indent << "warpgroup_wait<0>();\n";
  if (!pending_wgmma_acc_sym.empty())
    os << indent << "warpgroup_fence_operand(" << pending_wgmma_acc_sym
       << ");\n";
  has_pending_wgmma_finalize = false;
  pending_wgmma_acc_sym.clear();
  if (!has_explicit_mma_wait || force_wait) warpspec_wgmma_arrived = false;
}

std::optional<CuteCodeGen::HoistedScaleAccumInfo>
CuteCodeGen::AnalyzeHoistableScaledWGMMAAccum(
    const ptr<AST::Node>& n, const std::vector<std::string>& loop_refs) const {
  (void)n;
  (void)loop_refs;
  return std::nullopt;
}

bool CuteCodeGen::CollectHoistableScaledWGMMAAccum(
    const ptr<AST::Node>& n, const std::vector<std::string>& loop_refs,
    HoistedScaleAccumInfo& info, bool& saw_scaled_exec) const {
  (void)n;
  (void)loop_refs;
  (void)info;
  (void)saw_scaled_exec;
  return true;
}

const CuteCodeGen::HoistedScaleAccumInfo*
CuteCodeGen::CurrentHoistedScaleAccum() const {
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

  bool use_byte_ptr = (sto == Storage::GLOBAL || sto == Storage::DEFAULT ||
                       sto == Storage::SHARED) &&
                      IsSubByteType(bty);
  std::string ptr_elem = use_byte_ptr ? "uint8_t" : bts;
  tsr_decl << indent << "auto " << tsr_name << " = cute::make_tensor(";
  if (!mem_ty.empty())
    tsr_decl << "cute::make_" << mem_ty << "_ptr<" << ptr_elem << ">";
  tsr_decl << "((" << ptr_elem << "*)" << buf_expr
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
    has_explicit_mma_wait = AST::ContainsMMAWait(n);
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
      emitted_device_names_.clear();
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
  } else if (auto ab = dyn_cast<AST::ApplyBlock>(&n)) {
    if (!IsHost()) {
      if (has_pending_wgmma_finalize) EmitWGMMAFinalize(ds, d_indent, true);

      auto target_name = ab->SpanFragmentName();
      auto scoped = InScopeNameForRef(target_name);

      if (FCtx(fname).HasFragmentLayout(scoped)) {
        auto& fl = FCtx(fname).GetFragmentLayout(scoped);
        auto sym = UnScopedName(SSMName(scoped, false));
        size_t regs = fl.regs_per_thread;

        std::string tid_expr;
        if (bdim_level == ParallelLevel::GROUPx4)
          tid_expr = "(threadIdx.x % 128)";
        else if (!current_thread_count_expr.empty())
          tid_expr = "__choreo_vtid_x";
        else
          tid_expr = "threadIdx.x";

        reg_loop_var_ = "__r";
        in_register_direct_automap_ = true;
        automap_frag_reg_expr_.clear();
        automap_frag_reg_expr_[scoped] = sym + "[" + reg_loop_var_ + "]";

        // Build overrides for all fragment accesses in the body.
        auto BuildOverrides = [&](auto&& self,
                                  const ptr<AST::Node>& node) -> void {
          if (!node) return;
          if (auto nvd = dyn_cast<AST::NamedVariableDecl>(node)) {
            if (nvd->init_expr) self(self, nvd->init_expr);
            return;
          }
          if (auto da = dyn_cast<AST::DataAccess>(node)) {
            if (da->AccessElement()) {
              auto sc = InScopeNameForRef(da->data->name);
              if (FCtx(fname).HasFragmentLayout(sc)) {
                auto s = UnScopedName(SSMName(sc, false));
                auto& other_fl = FCtx(fname).GetFragmentLayout(sc);
                if (other_fl.IsCompatible(fl)) {
                  automap_frag_reg_expr_[sc] = s + "[" + reg_loop_var_ + "]";
                } else if (other_fl.logical_cols <= 1 && fl.logical_cols > 1 &&
                           fl.IsMMAAnchored()) {
                  automap_frag_reg_expr_[sc] =
                      s + "[" + fl.RowFromRegIndex(reg_loop_var_) + "]";
                }
              }
            }
            return;
          }
          if (auto mn = dyn_cast<AST::MultiNodes>(node)) {
            for (auto& child : mn->values) self(self, child);
            return;
          }
          if (auto asgn = dyn_cast<AST::Assignment>(node)) {
            if (asgn->da) self(self, asgn->da);
            if (asgn->value) self(self, asgn->value);
            return;
          }
          if (auto ie = dyn_cast<AST::IfElseBlock>(node)) {
            if (ie->pred) self(self, ie->pred);
            if (ie->stmts) self(self, ie->stmts);
            if (ie->else_stmts) self(self, ie->else_stmts);
            return;
          }
          if (auto expr = dyn_cast<AST::Expr>(node)) {
            if (expr->GetL()) self(self, expr->GetL());
            if (expr->GetR()) self(self, expr->GetR());
            if (expr->GetC()) self(self, expr->GetC());
            return;
          }
          if (auto call = dyn_cast<AST::Call>(node)) {
            for (auto& arg : call->GetArguments()) self(self, arg);
            return;
          }
          if (auto cast_expr = dyn_cast<AST::CastExpr>(node)) {
            if (cast_expr->GetR()) self(self, cast_expr->GetR());
            return;
          }
        };
        BuildOverrides(BuildOverrides, ab->body);

        // Detect which iterators need __frag_iv_X variables.
        // Iterators in fragment .at() are resolved by register-direct mapping
        // and don't need explicit variables. But iterators in non-fragment
        // .at() (global/shared memory) need __frag_iv_X for coordinate
        // computation.
        std::set<std::string> used_iters;
        auto FindUsedIters = [&](auto&& self, const ptr<AST::Node>& node,
                                 bool in_at_index) -> void {
          if (!node) return;
          if (auto expr = dyn_cast<AST::Expr>(node)) {
            if (auto sym = expr->GetSymbol()) {
              if (!in_at_index)
                for (auto& p : ab->iterators)
                  if (sym->name == p) used_iters.insert(p);
            }
            if (expr->GetL()) self(self, expr->GetL(), in_at_index);
            if (expr->GetR()) self(self, expr->GetR(), in_at_index);
            if (expr->GetC()) self(self, expr->GetC(), in_at_index);
            return;
          }
          if (auto mn = dyn_cast<AST::MultiNodes>(node)) {
            for (auto& child : mn->values) self(self, child, in_at_index);
            return;
          }
          if (auto asgn = dyn_cast<AST::Assignment>(node)) {
            if (asgn->da) self(self, asgn->da, in_at_index);
            if (asgn->value) self(self, asgn->value, in_at_index);
            return;
          }
          if (auto ie = dyn_cast<AST::IfElseBlock>(node)) {
            if (ie->pred) self(self, ie->pred, false);
            if (ie->stmts) self(self, ie->stmts, in_at_index);
            if (ie->else_stmts) self(self, ie->else_stmts, in_at_index);
            return;
          }
          if (auto nvd = dyn_cast<AST::NamedVariableDecl>(node)) {
            if (nvd->init_expr) self(self, nvd->init_expr, in_at_index);
            return;
          }
          if (auto da = dyn_cast<AST::DataAccess>(node)) {
            if (da->AccessElement()) {
              auto sc = InScopeNameForRef(da->data->name);
              bool is_frag = FCtx(fname).HasFragmentLayout(sc);
              for (auto& idx : da->GetIndices()) self(self, idx, is_frag);
            }
            return;
          }
          if (auto call = dyn_cast<AST::Call>(node)) {
            for (auto& arg : call->GetArguments()) self(self, arg, in_at_index);
            return;
          }
          if (auto cast_expr = dyn_cast<AST::CastExpr>(node)) {
            if (cast_expr->GetR()) self(self, cast_expr->GetR(), in_at_index);
            return;
          }
        };
        FindUsedIters(FindUsedIters, ab->body, false);

        // Set up iterator variable mappings (only for used iterators).
        for (auto& p : ab->iterators) {
          frag_apply_iv_map_[p] = "__frag_iv_" + p;
        }

        // Row-hoisting: classify each top-level body statement.
        // A statement is "row-only" if it exclusively accesses 1D fragments.
        apply_row_hoisted_stmts_.clear();
        std::vector<AST::Node*> row_only_stmts;
        auto IsRowOnly = [&](const ptr<AST::Node>& stmt) -> bool {
          bool has_2d = false;
          auto Check = [&](auto&& self, const ptr<AST::Node>& node) -> void {
            if (!node) return;
            if (auto nvd = dyn_cast<AST::NamedVariableDecl>(node)) {
              if (nvd->init_expr) self(self, nvd->init_expr);
              return;
            }
            if (auto da = dyn_cast<AST::DataAccess>(node)) {
              if (da->AccessElement()) {
                auto sc = InScopeNameForRef(da->data->name);
                if (FCtx(fname).HasFragmentLayout(sc)) {
                  auto& ffl = FCtx(fname).GetFragmentLayout(sc);
                  if (ffl.IsCompatible(fl)) has_2d = true;
                }
              }
              return;
            }
            if (auto mn = dyn_cast<AST::MultiNodes>(node)) {
              for (auto& c : mn->values) self(self, c);
              return;
            }
            if (auto a = dyn_cast<AST::Assignment>(node)) {
              if (a->da) self(self, a->da);
              if (a->value) self(self, a->value);
              return;
            }
            if (auto ie = dyn_cast<AST::IfElseBlock>(node)) {
              if (ie->pred) self(self, ie->pred);
              if (ie->stmts) self(self, ie->stmts);
              if (ie->else_stmts) self(self, ie->else_stmts);
              return;
            }
            if (auto e = dyn_cast<AST::Expr>(node)) {
              if (e->GetL()) self(self, e->GetL());
              if (e->GetR()) self(self, e->GetR());
              if (e->GetC()) self(self, e->GetC());
              return;
            }
            if (auto c = dyn_cast<AST::Call>(node)) {
              for (auto& arg : c->GetArguments()) self(self, arg);
              return;
            }
            if (auto ce = dyn_cast<AST::CastExpr>(node)) {
              if (ce->GetR()) self(self, ce->GetR());
              return;
            }
          };
          Check(Check, stmt);
          return !has_2d;
        };

        if (auto body_mn = dyn_cast<AST::MultiNodes>(ab->body)) {
          for (auto& child : body_mn->values) {
            if (IsRowOnly(child)) {
              row_only_stmts.push_back(child.get());
              apply_row_hoisted_stmts_.insert(child.get());
            }
          }
        }

        // Emit the apply block opening.
        IndStream() << "{ // apply " << target_name << "\n";
        IncrIndent();

        // Emit row-hoisted statements in a small row loop.
        if (!row_only_stmts.empty()) {
          size_t rows = fl.rows_per_thread;
          std::string row_var = "__row";

          // Save current register-direct state and set up row-only context.
          auto saved_reg_loop = reg_loop_var_;
          auto saved_overrides = automap_frag_reg_expr_;

          reg_loop_var_ = row_var;
          automap_frag_reg_expr_.clear();
          // Map all 1D fragments to direct row-indexed access: sym[__row].
          for (auto& [sc, expr] : saved_overrides) {
            if (!FCtx(fname).HasFragmentLayout(sc)) continue;
            auto& ffl = FCtx(fname).GetFragmentLayout(sc);
            if (ffl.logical_cols <= 1) {
              auto s = UnScopedName(SSMName(sc, false));
              automap_frag_reg_expr_[sc] = s + "[" + row_var + "]";
            }
          }

          IndStream() << "#pragma unroll\n";
          IndStream() << "for (int " << row_var << " = 0; " << row_var << " < "
                      << rows << "; ++" << row_var << ") {\n";
          IncrIndent();

          // Emit each row-only statement directly.
          for (auto* stmt : row_only_stmts) {
            if (auto asgn = dynamic_cast<AST::Assignment*>(stmt)) {
              if (asgn->AssignToDataElement()) {
                auto lhs = ExprSTR(asgn->da, false);
                auto rhs = ExprSTR(asgn->value, false);
                IndStream() << lhs << " = " << rhs << ";\n";
              }
            }
          }

          DecrIndent();
          IndStream() << "}\n";

          // Now mark them to be skipped during main loop traversal.
          for (auto* stmt : row_only_stmts)
            apply_row_hoisted_stmts_.insert(stmt);

          // Restore register-direct state for main loop.
          reg_loop_var_ = saved_reg_loop;
          automap_frag_reg_expr_ = saved_overrides;
        }

        // Emit the main register loop (for full/2D statements).
        bool has_full_stmts = false;
        if (auto body_mn = dyn_cast<AST::MultiNodes>(ab->body)) {
          for (auto& child : body_mn->values)
            if (!apply_row_hoisted_stmts_.count(child.get())) {
              has_full_stmts = true;
              break;
            }
        } else {
          has_full_stmts = true;
        }

        apply_has_main_loop_ = has_full_stmts;
        if (has_full_stmts) {
          for (size_t pi = 0; pi < ab->iterators.size(); ++pi)
            if (used_iters.count(ab->iterators[pi]))
              ds << d_indent << "int __frag_iv_" << ab->iterators[pi]
                 << " = 0;\n";

          IndStream() << "#pragma unroll\n";
          IndStream() << "for (int " << reg_loop_var_ << " = 0; "
                      << reg_loop_var_ << " < " << regs << "; ++"
                      << reg_loop_var_ << ") {\n";
          IncrIndent();

          for (size_t pi = 0; pi < ab->iterators.size(); ++pi) {
            if (!used_iters.count(ab->iterators[pi])) continue;
            std::string iv_var = "__frag_iv_" + ab->iterators[pi];
            if (ab->iterators.size() == 1 || pi == 0)
              ds << d_indent << iv_var << " = "
                 << fl.LogicalRowFromReg(reg_loop_var_, tid_expr) << ";\n";
            else
              ds << d_indent << iv_var << " = "
                 << fl.LogicalColFromReg(reg_loop_var_, tid_expr) << ";\n";
          }
        }
      }
    }
  }
  if (isa<AST::NamedVariableDecl>(&n)) in_named_var_decl_ = true;

  if (!pending_barrier_inits_.empty() && !in_named_var_decl_)
    FlushBarrierInits();

  if (isa<AST::IfElseBlock>(&n) && !IsHost() && has_pending_wgmma_finalize)
    EmitWGMMAFinalize(ds, d_indent);

  if (isa<AST::IfElseBlock>(&n) || isa<AST::NamedVariableDecl>(&n)) {
    emit_call = false;
  }

  return true;
}

bool CuteCodeGen::InMidVisitImpl(AST::Node& n) {
  if (auto ie = dyn_cast<AST::IfElseBlock>(&n)) {
    if (!ie->HasElse()) return true;
    PopEmittedNames();
    PushEmittedNames();
    DecrIndent();
    IndStream() << "} else {\n";
    IncrIndent();
  }
  return true;
}

bool CuteCodeGen::AfterVisitImpl(AST::Node& n) {
  if (trace_visit) dbgs() << "After visiting " << n.TypeNameString() << "\n";

  if (isa<AST::NamedVariableDecl>(&n)) in_named_var_decl_ = false;

  EmitPostSiteAssertions(n);

  if (isa<AST::Program>(&n)) {
    ssm.LeaveScope();

    switch (CCtx().GetOutputKind()) {
    case OutputKind::TargetSourceCode:
    case OutputKind::DeviceSourceOnly: EmitSource(); break;
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
    case OutputKind::TargetLibrary: {
      if (!CompileWithScript("--lib")) {
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
      current_thread_count = 0;
      current_thread_count_expr.clear();
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
    ptr<AST::AttributeExpr> automap_attr;
    const bool has_automap =
        !IsHost() && AST::HasAutomapHint(*fb, automap_attr);
    if (has_automap) {
      if (vec4_automap_skip_) {
        vec4_automap_skip_ = false;
      } else {
        in_register_direct_automap_ = false;
        automap_frag_reg_expr_.clear();
        reg_loop_var_.clear();
        DecrIndent();
        IndStream() << "} // automap\n";
      }
    } else {
      const auto& ranges = fb->GetRangeNodes();
      for (int j = ranges->Count() - 1; j >= 0; --j) {
        auto rng = cast<AST::LoopRange>(ranges->ValueAt(j));
        auto cname = rng->GetIVName();
        auto ivs = within_map.at(InScopeName(cname));
        for (auto iv_itr = ivs.rbegin(); iv_itr != ivs.rend(); ++iv_itr) {
          DecrIndent();
          IndStream() << "} // " << UnScopedName(*iv_itr) << "\n";
          IndStream() << ssm.DeviceName(*iv_itr) << " = 0;\n"; // must reset
        }
      }
    }
    if (!IsHost() && has_pending_wgmma_finalize && !IsWarpSpecActive())
      EmitWGMMAFinalize(ds, d_indent);
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
    if (!foreach_iv_stack_.empty()) foreach_iv_stack_.pop_back();
  } else if (auto it = dyn_cast<AST::InThreadsBlock>(&n)) {
    DecrDeviceIndent();
    if (!it->stmts->None()) {
      ds << d_indent << "}";
      if (!it->async && it->outer) ds << "\n" << d_indent << "__syncthreads();";
      ds << " // end inthreads\n";
    }
    current_inthreads = nullptr;
    PopEmittedNames();
  } else if (auto ie = dyn_cast<AST::IfElseBlock>(&n)) {
    DecrIndent();
    IndStream() << "} // end if-else: " << ie->LOC() << "\n";
    PopEmittedNames();
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
  auto cur_strd = GetSpannedType(GetSymbolType(ca->RefSymbol()))->GetStrides();

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

      // When SubSpan has explicit steps, block_strides already incorporate
      // the step factor (shapeinfer multiplies cur_strd by step values),
      // so multiplying by block_shape would double-count.
      bool has_step = isa<AST::SOP::SubSpan>(sop) && sop->GetSteps();
      for (size_t ith = 0; ith < idx.GetVals().size(); ++ith) {
        if (has_step)
          offset += idx.GetVals()[ith] * strd[ith];
        else
          offset += idx.GetVals()[ith] * strd[ith] * blk.ValueAt(ith);
      }
      cur_strd = strd;
    } else if (isa<AST::SOP::View>(sop)) {
      auto off = sop->GetOffsets()->Opts();
      auto strd = sop->GetBlockStrides();
      for (size_t ith = 0; ith < off.GetVals().size(); ++ith)
        offset += off.GetVals()[ith] * strd[ith];
      cur_strd = sop->GetBlockStrides();
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

void CuteCodeGen::FlushBarrierInits() {
  if (pending_barrier_inits_.empty()) return;
  if (!pending_tma_prefetch_names_.empty()) {
    this->ds << d_indent << "if (threadIdx.x == 0) {\n";
    for (auto& name : pending_tma_prefetch_names_) {
      this->ds << d_indent << "  cute::prefetch_tma_descriptor(&" << name
               << ");\n";
    }
    this->ds << d_indent << "}\n";
    pending_tma_prefetch_names_.clear();
  }
  this->ds << d_indent << "if (threadIdx.x == 0) {\n";
  for (auto& init_line : pending_barrier_inits_) this->ds << init_line;
  this->ds << d_indent << "}\n";
  this->ds << d_indent << "cde::fence_proxy_async_shared_cta();\n";
  this->ds << d_indent << "__syncthreads();\n";
  pending_barrier_inits_.clear();
}

std::string CuteCodeGen::InlinePhaseExpr(int stages, bool is_fill) const {
  if (foreach_iv_stack_.empty()) return is_fill ? "0" : "1";
  const auto& iv = foreach_iv_stack_.back();
  std::string base;
  if (stages == 2)
    base = "((" + iv + " & 3) >> 1)";
  else if (stages == 1)
    base = "(" + iv + " & 1)";
  else
    base = "((" + iv + " / " + std::to_string(stages) + ") & 1)";
  return is_fill ? base : ("(" + base + " ^ 1)");
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

#include "cutlass/arch/barrier.h"
#include "cutlass/cutlass.h"
#include <cutlass/arch/reg_reconfig.h>
)";

  oss << R"(
#ifdef __CUDACC__
#pragma nv_diag_suppress 20054
#endif
)";
  oss << "// include the choreo header;\n";
  if (native_f16)
    oss << "#define __CHOREO_TARGET_NATIVE_HALF_FLOAT_SUPPORT__\n";
  if (native_bf16) oss << "#define __CHOREO_TARGET_NATIVE_BF16_SUPPORT__\n";
  if (CCtx().GetApiMode() != "sglang" && !CCtx().DisableCudaRuntimeEnvCheck())
    oss << "#define __CHOREO_ENABLE_CUDA_RUNTIME_ENV_CHECK__\n"
           "#define __CHOREO_REQUIRED_GPU_DEVICE_SM__ "
        << CCtx().ArchNum() << "\n";
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
  if (cgi.HasTMA())
    oss << "using Barrier = cutlass::arch::ClusterTransactionBarrier;\n";
  code_segments.push_back(oss.str()); // reset the host code
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
        auto key = InScopeName(sym) + ".data";
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
  pending_tma_prefetch_names_.clear();

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

  if (!pending_barrier_inits_.empty() && !isa<EventArrayType>(nty) &&
      !isa<EventType>(nty))
    FlushBarrierInits();

  // Register in the live scoped table so that InScopeNameForRef correctly
  // resolves subsequent references to this variable (declaration-order aware).
  SSTab().DefineSymbol(sym, nty);

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

    // Propagate swizzle for ElemOf-based declarations (e.g., `ma = arr[idx]`).
    auto decl_sty = GetSpannedType(NodeType(n));
    if (decl_sty && decl_sty->GetStorage() == Storage::SHARED && n.init_expr) {
      auto rhs_expr = dyn_cast<AST::Expr>(n.init_expr.get());
      if (rhs_expr && rhs_expr->GetOp() == Op::ElemOf) {
        auto base_sym_node = AST::GetArrayBaseSymbol(*rhs_expr);
        if (base_sym_node) {
          auto base_name = base_sym_node->name;
          auto base_scoped = InScopeName(base_name);
          auto sit = shared_buf_swiz_.find(base_scoped);
          if (sit == shared_buf_swiz_.end())
            sit = shared_buf_swiz_.find(base_name);
          if (sit != shared_buf_swiz_.end()) {
            shared_buf_swiz_[sym] = sit->second;
            shared_buf_swiz_[sname] = sit->second;
          }
        }
      }
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
      if (IsChoreoOutput(InScopeName(n.name_str)))
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
          std::string total_elem_expr = UnScopedExpr(ElemCountExprOf(*sty));
          for (const auto& dim : GetArrayDimensions(nty))
            total_elem_expr =
                "(" + ValueSTR(dim) + ")*(" + total_elem_expr + ")";
          ds << d_indent << type_modifiers << bts << " " << sym << "["
             << total_elem_expr << "];\n";
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
      sym = UniqueDeviceName(n.name_str);
      HandleSharedLocal();
      ssm.MapDeviceSymbol(InScopeName(n.name_str), sym);
      spmem = true;
    } else if (sto == Storage::REG && n.HasNote("fragment_decl")) {
      if (IsHost())
        choreo_unreachable("error: fragment declaration in host scope.");
      sym = UniqueDeviceName(n.name_str);
      auto scoped = InScopeName(n.name_str);
      size_t regs_per_thread = 1;
      std::string thread_count_expr;

      if (FCtx(fname).HasFragmentLayout(scoped)) {
        const auto& fl = FCtx(fname).GetFragmentLayout(scoped);
        regs_per_thread = fl.regs_per_thread;
        thread_count_expr = fl.thread_count_expr;
      } else if (FCtx(fname).FragIsRS(InScopeName(n.name_str))) {
        size_t elem_bytes = SizeOf(sty->ElementType());
        regs_per_thread = 16 / elem_bytes;
        thread_count_expr = "128";
      } else {
        auto total_elem = sty->GetShape().ElementCountValue();
        thread_count_expr = current_thread_count_expr.empty()
                                ? std::string("blockDim.x")
                                : current_thread_count_expr;
        if (auto total = VIInt(total_elem)) {
          if (current_thread_count > 0)
            regs_per_thread =
                (*total + current_thread_count - 1) / current_thread_count;
          else
            regs_per_thread = *total;
        }
      }
      ds << d_indent << bts << " " << sym << "[" << regs_per_thread << "];\n";
      if (n.init_value) {
        std::string init_val =
            ExprCastSTR(n.init_value, std::nullopt, GetBaseType(*sty),
                        GetBaseType(*n.init_value->GetType()), false);
        ds << d_indent << "for (int __frag_init = 0; __frag_init < "
           << regs_per_thread << "; ++__frag_init)\n";
        IncrDeviceIndent();
        ds << d_indent << sym << "[__frag_init] = " << init_val << ";\n";
        DecrDeviceIndent();
      }
      FragmentLayoutInfo finfo;
      finfo.regs_per_thread = regs_per_thread;
      finfo.thread_count_expr = thread_count_expr;
      FCtx(fname).SetFragmentInfo(scoped, finfo);
      ssm.MapDeviceSymbol(scoped, sym);
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
    auto eaname = UniqueDeviceName(n.name_str);
    switch (ety->GetStorage()) {
    case Storage::GLOBAL: {
      assert(IsHost());
      auto sym = InScopeName(n.name_str);
      auto buf_sym = eaname + "__device";
      hs << h_indent << "bool * " << buf_sym << " = nullptr; // global event\n";
      hs << h_indent << "choreo::abend_true(cudaMalloc(&" << buf_sym << ", "
         << ety->ElemCount() << "));\n";
      hs << h_indent << "choreo::abend_true(cudaMemset(&" << buf_sym << ", 0, "
         << ety->ElemCount() << "));\n";
      ssm.MapHostSymbol(sym, buf_sym);
      ssm.MapDeviceSymbol(sym, eaname);
      global_buffers.insert(buf_sym);
    } break;
    case Storage::SHARED: {
      assert(!IsHost());
      bool is_cluster_event = cluster_trigger_events_.count(n.name_str) > 0;

      // All shared events use Cutlass ClusterTransactionBarrier
      auto& ft = cgi.GetFunctionTrait(fname);
      bool is_fill_event = ft.IsTMAFillEvent(n.name_str);

      ds << d_indent << "__shared__ __align__(16) uint64_t " << eaname
         << "__mem";
      ety->PrintAsCArray(ds);
      ds << "; // raw mbarrier storage\n";
      ds << d_indent << "Barrier* " << eaname
         << " = reinterpret_cast<Barrier*>(" << eaname << "__mem);\n";

      if (!is_fill_event) empty_event_names_.insert(n.name_str);

      auto compute_init_count = [&]() -> std::string {
        auto event_tc = ety->event->GetThreadCount();
        if (event_tc > 0) return std::to_string(event_tc);
        if (is_fill_event) return "1";
        if (is_cluster_event)
          return "(blockDim.x / 128 - 1) * choreo::tma_cluster_dim()";
        auto tp = ft.GetEventTriggerParticipation(n.name_str);
        if (tp > 0) return std::to_string(tp);
        return IsWarpSpecActive() ? "(blockDim.x - 128)" : "blockDim.x";
      };
      std::string init_count = compute_init_count();

      {
        std::ostringstream oss;
        GenerateSubscriptions(oss, "  " + d_indent + eaname,
                              ".init(" + init_count + ");\n",
                              ety->Dimensions());
        pending_barrier_inits_.push_back(oss.str());
      }
      ssm.MapDeviceSymbol(InScopeName(n.name_str), eaname);
    } break;
    case Storage::LOCAL: {
      assert(!IsHost());
      ds << d_indent << CudaDeviceMemory(ety->GetStorage())
         << " __volatile__ bool " << eaname;
      ety->PrintAsCArray(ds);
      ds << "; // " << STR(ety->GetStorage()) << " event\n";
      ds << d_indent << "// initialize the event\n";
      GenerateSubscriptions(ds, d_indent + eaname, " = false;\n",
                            ety->Dimensions());
      ds << d_indent << EmitSync(ety->GetStorage()) << ";\n";
      ssm.MapDeviceSymbol(InScopeName(n.name_str), eaname);
    } break;
    default: break;
    }
  } else if (auto ety = dyn_cast<EventType>(nty)) {
    auto ename = UniqueDeviceName(n.name_str);
    switch (ety->GetStorage()) {
    case Storage::GLOBAL: {
      assert(IsHost());
      auto sym = InScopeName(n.name_str);
      auto buf_sym = ename + "__device";
      hs << h_indent << "bool * " << buf_sym << " = nullptr; // global event\n";
      hs << h_indent << "choreo::abend_true(cudaMalloc(&" << buf_sym
         << ", 1));\n";
      hs << h_indent << "choreo::abend_true(cudaMemset(&" << buf_sym
         << ", 0, 1));\n";
      ssm.MapHostSymbol(sym, buf_sym);
      ssm.MapDeviceSymbol(sym, ename);
      global_buffers.insert(buf_sym);
    } break;
    case Storage::SHARED: {
      assert(!IsHost());
      auto& ft = cgi.GetFunctionTrait(fname);
      bool is_fill = ft.IsTMAFillEvent(n.name_str);

      ds << d_indent << "__shared__ __align__(16) uint64_t " << ename
         << "__mem; // raw mbarrier storage\n";
      ds << d_indent << "Barrier& " << ename
         << " = *reinterpret_cast<Barrier*>(&" << ename << "__mem);\n";

      if (!is_fill) empty_event_names_.insert(n.name_str);

      {
        auto etc = ety->GetThreadCount();
        std::string ic;
        if (etc > 0) {
          ic = std::to_string(etc);
        } else if (is_fill) {
          ic = "1";
        } else {
          auto tp = ft.GetEventTriggerParticipation(n.name_str);
          ic = tp > 0 ? std::to_string(tp) : "(blockDim.x - 128)";
        }
        std::ostringstream oss;
        oss << d_indent << "  " << ename << ".init(" << ic << ");\n";
        pending_barrier_inits_.push_back(oss.str());
      }
      ssm.MapDeviceSymbol(InScopeName(n.name_str), ename);
    } break;
    case Storage::LOCAL: {
      assert(!IsHost());
      ds << d_indent << CudaDeviceMemory(ety->GetStorage())
         << " __volatile__ bool " << ename << "; // " << STR(ety->GetStorage())
         << " event\n";
      ds << d_indent << ename << " = false;\n";
      ds << d_indent << EmitSync(ety->GetStorage()) << ";\n";
      ssm.MapDeviceSymbol(InScopeName(n.name_str), ename);
    } break;
    default: break;
    }
  }

  // Fallback: emit declaration for unresolved types with init expressions.
  // Variables like `kv_bound = is_causal ? ... : kv_tiles` may have type
  // "unknown" but still need a C++ declaration so that later references use the
  // variable name instead of re-expanding the full expression tree.
  if (n.init_expr && !IsHost() && !isa<EventType>(nty) &&
      !isa<EventArrayType>(nty)) {
    auto scoped = InScopeName(sym);
    if (!ssm.HasDeviceName(scoped)) {
      auto init_str = ExprSTR(n.init_expr, false);
      // Guard against shadowing: if the init expression references the same
      // C++ name as the variable being declared, emitting `auto x = f(x)`
      // would be self-referential UB.  Skip the declaration in that case.
      auto is_id_char = [](char c) {
        return std::isalnum(static_cast<unsigned char>(c)) || c == '_';
      };
      bool self_ref = false;
      for (auto pos = init_str.find(sym); pos != std::string::npos;
           pos = init_str.find(sym, pos + 1)) {
        bool left_ok = (pos == 0 || !is_id_char(init_str[pos - 1]));
        bool right_ok = (pos + sym.size() >= init_str.size() ||
                         !is_id_char(init_str[pos + sym.size()]));
        if (left_ok && right_ok) {
          self_ref = true;
          break;
        }
      }
      if (!self_ref) {
        ds << d_indent << "auto " << sym << " = " << init_str << ";\n";
        ssm.MapDeviceSymbol(scoped, sym);
        // Register the string form of the value (without the top-level
        // parens that ValueSTR wraps) so the replacement in ValueSTR
        // can match it as a substring inside larger expressions.
        if (FCtx(fname).HasSymbolValues(scoped)) {
          auto& svs = FCtx(fname).GetSymbolValues(scoped);
          if (svs.HasVal()) {
            auto stripped = StripOuterParens(ValueSTR(svs.GetVal()));
            bool is_simple_literal =
                !stripped.empty() &&
                stripped.find_first_not_of("0123456789-") == std::string::npos;
            if (!is_simple_literal) known_val_str_to_var_[stripped] = sym;
          }
        }
      }
    }
  }

  return true;
}

bool CuteCodeGen::Visit(AST::NamedTypeDecl& n) {
  TraceEachVisit(n);

  auto nty = NodeType(n);
  auto sym = n.name_str;
  const bool enable_debug_rtti = EnableDebugTypeRTTI();

  SSTab().DefineSymbol(sym, nty);

  auto sname = InScopeName(sym);
  if (!FCtx(fname).HasSymbolValues(sname))
    updating_cgi.AddSymbolDetail(fname, {sname, GetSymbolType(sym), false});

  if (auto mty = dyn_cast<MDSpanType>(nty)) {
    if (mty->Dims() <= 1) {
      if (mty->HasSufficientInfo()) {
        IndStream() << "int " << sym << " = "
                    << ValueSTR(mty->GetShape().ValueAt(0)) << ";\n";
        if (!IsHost()) {
          ssm.MapDeviceSymbol(sname, sym);
          auto stripped =
              StripOuterParens(ValueSTR(mty->GetShape().ValueAt(0)));
          bool is_simple_literal =
              !stripped.empty() &&
              stripped.find_first_not_of("0123456789-") == std::string::npos;
          if (!is_simple_literal) known_val_str_to_var_[stripped] = sym;
        }
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
  if (vec4_automap_skip_) return true;
  if (apply_row_hoisted_stmts_.count(&n)) return true;

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

      auto parent_sym = rhs_ca->RefSymbol();
      auto parent_scoped = InScopeNameForRef(parent_sym);
      if (FCtx(fname).HasFragmentLayout(parent_scoped)) {
        auto& fl = FCtx(fname).GetFragmentLayout(parent_scoped);
        if (fl.kind == LayoutKind::WGMMA_ACC) {
          auto parent_c_sym = UnScopedName(SSMName(parent_scoped, false));

          std::string offset_var;
          for (auto& sop : rhs_ca->AllOperations()) {
            if (auto ta = dyn_cast<AST::SOP::TileAt>(sop)) {
              auto indices = ta->GetIndices();
              if (indices) {
                for (auto& idx : indices->AllValues()) {
                  auto val = ExprSTR(idx, false);
                  if (val == "__choreo_no_tiling__" || val == "0") continue;
                  offset_var = val;
                  break;
                }
              }
            }
          }

          size_t warp_k = 16;
          size_t num_chunks = fl.logical_cols / warp_k;
          size_t regs_per_step =
              num_chunks > 0 ? fl.regs_per_thread / num_chunks : 0;

          FragChunkRSInfo info;
          info.parent_c_sym = parent_c_sym;
          info.offset_var = offset_var;
          info.regs_per_step = regs_per_step;
          frag_chunk_rs_aliases_[name] = info;
          frag_chunk_rs_aliases_[scoped_name] = info;
          return true;
        }
      }
    } else {
      live_chunk_aliases.erase(name);
      live_chunk_aliases.erase(scoped_name);
    }

    // Propagate swizzle for ElemOf-based aliases (e.g., `ma =
    // lhs_load_s[stage]`). When the RHS is an array element access into a
    // swizzled shared buffer, record the swizzle for the alias name so direct
    // WGMMA can find it.
    if (sty && sty->GetStorage() == Storage::SHARED) {
      auto rhs_expr = dyn_cast<AST::Expr>(n.value.get());
      if (rhs_expr && rhs_expr->GetOp() == Op::ElemOf) {
        auto base_sym = AST::GetArrayBaseSymbol(*rhs_expr);
        if (base_sym) {
          auto base_name = base_sym->name;
          auto base_scoped = InScopeName(base_name);
          auto sit = shared_buf_swiz_.find(base_scoped);
          if (sit == shared_buf_swiz_.end())
            sit = shared_buf_swiz_.find(base_name);
          if (sit != shared_buf_swiz_.end()) {
            shared_buf_swiz_[name] = sit->second;
            shared_buf_swiz_[scoped_name] = sit->second;
          }
        }
      }
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

    bool anon_bounded = n.IsDecl() && isa<BoundedType>(nty) &&
                        PrefixedWith(n.GetName(), "anon_");
    bool tma_subview = n.IsDecl() && isa<SpannedType>(nty) &&
                       cgi.IsTMASubviewSym(InScopeName(n.GetName()));
    if (!(sty && sty->GetStorage() == Storage::REG && n.HasNote("update"))) {
      ds << d_indent;
      if (!n.IsDecl() || (IsMutable(*nty) && !isa<SpannedType>(nty)))
        ds << n.GetName() << " = ";
      else {
        if (anon_bounded || tma_subview) ds << "[[maybe_unused]] ";
        ds << "auto " << n.GetName() << " = ";
      }
    } else
      ds << d_indent;
    ds << ExprSTR(n.value, false) << ";\n";
    return true;
  }

  if (isa<ScalarType>(nty)) {
    bool anon_scalar = n.IsDecl() && PrefixedWith(n.GetName(), "anon_");
    if (IsHost())
      hs << h_indent << ((!n.IsDecl()) ? "" : "auto ") << n.GetName() << " = "
         << ExprSTR(n.value, true) << ";\n";
    else {
      ds << d_indent;
      if (!n.IsDecl())
        ds << n.GetName() << " = ";
      else {
        if (anon_scalar) ds << "[[maybe_unused]] ";
        ds << "auto " << n.GetName() << " = ";
      }
      std::string val_str = ExprSTR(n.value, false);
      ds << val_str << ";\n";
      if (n.IsDecl() && !IsHost()) {
        auto stripped = StripOuterParens(val_str);
        bool is_simple_literal =
            !stripped.empty() &&
            stripped.find_first_not_of("0123456789-") == std::string::npos;
        if (!is_simple_literal) known_val_str_to_var_[stripped] = n.GetName();
      }
    }
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
    current_thread_count = 0;
    current_thread_count_expr.clear();
    if (!n.BoundValues().empty()) {
      if (auto v0 = VIInt(n.BoundValues()[0])) {
        size_t tc = *v0;
        for (size_t i = 1; i < n.BoundValues().size(); ++i)
          if (auto vi = VIInt(n.BoundValues()[i])) tc *= *vi;
        current_thread_count = tc;
        current_thread_count_expr = std::to_string(tc);
      }
    }
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
        mri->infos.at(Storage::SHARED).spm_size >= 48 * 1024)
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
    set_cuda_func_attribute_max_dynamic_shared_memory_size = false; // reset
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
          if (!op || !op->IsLoad()) return;
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

    shared_spm_size = sbe::nu(0);
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
      shared_spm_size += ring_size;
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
         << ValueSTR(shared_spm_size) << " + (" << required_shared_align
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
      shared_spm_size += code_spm_end;
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
        if (mri->infos[Storage::SHARED].spm_size >= 48 * 1024) {
          auto code_spm_end = sbe::nu(mri->infos[Storage::SHARED].spm_size);
          shared_spm_size += code_spm_end;
          ring_start = code_spm_end;
          EmitCudaFuncAttributeMaxDynamicSharedMemorySize();
          Note(
              n.LOC(),
              "In the current kernel `" + device_fn +
                  "`, cudaFuncAttributeMaxDynamicSharedMemorySize is set to `" +
                  ValueSTR(shared_spm_size) + "` bytes, " +
                  "cause shared memory usage" +
                  " has exceeded the default limit 48KB.");
        }
      }
    }

    hs << h_indent << device_fn << "<<<__" << fname << "_gdims" << parallel_idx
       << ", __" << fname << "_bdims" << parallel_idx;

    bool explicit_smem = false;
    if (!sbe::ceq(shared_spm_size, sbe::nu(0))) {
      // TODO: conservative padding. To be optimized.
      hs << ", " << ValueSTR(shared_spm_size) << " + (" << required_shared_align
         << " - 1)";
      explicit_smem = true;
    }
    std::string effective_stream;
    if (n.HasStream()) effective_stream = STR(n.StreamExpr());

    if (effective_stream != "") {
      if (!explicit_smem) hs << ", 0";
      hs << ", " << effective_stream;
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
      if (effective_stream != "")
        hs << h_indent << "choreo::abend_true(cudaStreamSynchronize("
           << effective_stream << "));\n";
      else
        hs << h_indent << "choreo::abend_true(cudaDeviceSynchronize());\n";
    }

    // copy the span passed by ref back to host
    for (const auto& item : GetChoreoFuncIns(updating_cgi)) {
      if (isa<SpannedType>(item.type)) {
        auto oname = UnScopedName(item.name);
        if (item.attr != ParamAttr::GLOBAL_INPUT && item.IsReference())
          hs << h_indent << "choreo::abend_true(cudaMemcpy(" << oname
             << ".data(), " << oname + "__device"
             << ", " << UnScopedSizeExpr(*item.type)
             << ", cudaMemcpyDeviceToHost));\n";
      }
    }

    // handle device function
    EmitDeviceFuncDecl(ds, &n, ring_start);
    ds << " {\n";
    IncrDeviceIndent();
    if (n.HasMaxnreg() && CCtx().ArchNum() >= 90) {
      auto reg_limit = VIInt(n.GetMaxnregArg()->Opts().GetVal());
      if (reg_limit && reg_limit.value() > 0)
        ds << d_indent << "asm volatile(\"setmaxnreg.dec.sync.aligned.u32 "
           << reg_limit.value() << ";\");\n";
    }
    if (!(sbe::ceq(shared_spm_size, sbe::nu(0)) &&
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
      if (!sbe::ceq(shared_spm_size, sbe::nu(0)) && cgi.HasAsyncDMA(fname)) {
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

    } else {
      ds << d_indent << "[[maybe_unused]] auto " << device_fn
         << "__ring__ = nullptr;\n";
    }
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

  auto& tma_descs = cgi.GetTMADescs()[&n];
  auto* cluster_pb = deferred_cluster_pb;
  if (tma_descs.empty() && cluster_pb) {
    auto& cluster_tma = cgi.GetTMADescs()[cluster_pb];
    if (!cluster_tma.empty()) { tma_descs = cluster_tma; }
  }
  if (!tma_descs.empty()) {
    assert(n.GetLevel() == ParallelLevel::BLOCK);
    for (TMADesc& desc : tma_descs)
      pending_tma_prefetch_names_.push_back(desc.GetName() + "_tensor_map");
    int emitted_tma_init_idx = 0;
    for (TMADesc& desc : tma_descs) {
      auto cp_atom_name = GetCopyAtomName(true, emitted_tma_init_idx);
      auto tma_barrier_name = cp_atom_name + "_barrier";
      auto f_sty = GetSpannedType(desc.GetFrom()->GetType());
      auto t_sty = GetSpannedType(desc.GetTo()->GetType());
      auto io_sty = (t_sty->GetStorage() == Storage::SHARED) ? t_sty : f_sty;
      auto in_thr_block = desc.GetInThreadsBlock();
      auto inner_pb_level = desc.GetPBLevel();

      bool skip_load_tma_init_block = IsWarpSpecActive() && desc.IsLoad() &&
                                      in_thr_block &&
                                      inner_pb_level == ParallelLevel::GROUPx4;
      // In warpspec mode, TMA uses event barriers (G2S) or direct
      // commit/wait groups (S2G) -- no TMAAtom needed at all.
      bool skip_warpspec_tma = IsWarpSpecActive() && !desc.IsLoad();

      if (skip_load_tma_init_block || skip_warpspec_tma) { continue; }

      emitted_tma_init_idx++;

      ds << d_indent << "__shared__ __align__(8) uint64_t " << tma_barrier_name
         << ";\n";
      ds << d_indent << "if (__CHOREO_BLOCK_SINGLE__) {\n";
      ds << d_indent << "  choreo::tma_mbarrier_init(&" << tma_barrier_name
         << ", 1);\n";
      ds << d_indent << "}\n";
      ds << d_indent << "__syncthreads();\n";
      ds << d_indent << "TMAAtom " << cp_atom_name << "{&" << tma_barrier_name
         << "};\n\n";
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
  auto nty = NodeType(n);
  if (auto ph = dyn_cast<PlaceHolderType>(nty)) {
    assert(ph->GetBaseType() == BaseType::FUTURE);
    if (!IsHost() && !n.future.empty()) {
      std::string buf_name =
          const_cast<FutureBufferInfo&>(FBInfo())[InScopeName(n.future)].buffer;
      auto cp_atom_name =
          GetCopyAtomName(n.IsTMA(), n.IsTMA() ? tma_future_count : dma_count_);
      if (!claimed_futs.count(InScopeName(n.future))) {
        claimed_futs.emplace(InScopeName(n.future), cp_atom_name);
        ssm.MapDeviceSymbol(InScopeName(n.future), n.future);
        ssm.MapDeviceSymbol(InScopeName(n.future) + ".data",
                            n.future + ".data()");
        future_count_++;
        if (n.IsTMA())
          ++tma_future_count;
        else
          ++dma_count_;
        std::string buf_expr;
        if (ssm.HasDeviceName(buf_name))
          buf_expr = ssm.DeviceName(buf_name);
        else
          buf_expr = UnScopedName(buf_name);
        ds << d_indent << "future " << n.future << "(\"" << n.future << "\", "
           << n.LOC().begin.line << ", " << n.LOC().begin.column;
        if (!buf_expr.empty()) ds << ", " << buf_expr;
        ds << ");\n";
        if (n.IsTMA()) {
          ds << d_indent << n.future << ".is_tma = true;\n";
          ds << d_indent << n.future << ".set_atom(&" << cp_atom_name << ");\n";
          tma_futures_.insert(InScopeName(n.future));
        }
      }
      ssm.RemapDeviceSymbol(buf_name, n.future + ".data()");
    }
    return true;
  }

  const DMALoweringDecision* dma_plan = DMAPlan::Lookup(&n);
  assert(dma_plan && "DMA plan must exist before CuteCodeGen emission");
  auto future_only = !n.future.empty();
  auto event_only = dma_plan->is_async && n.HasEvent();
  assert(!(event_only && future_only) &&
         "event and future cannot be both present");
  bool is_tma = dma_plan->use_tma;
  bool is_async = dma_plan->is_async;

  assert(isa<AST::ChunkAt>(n.from) && "Unexpected type for DMA's source.");
  assert(isa<AST::ChunkAt>(n.to) && "Unexpected type for DMA's destination.");

  auto fty = dyn_cast<FutureType>(nty);
  assert(fty && "Invalid type of DMA statement!");

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

  auto claimFuture = [&](const std::string& buf_expr,
                         const DMALoweringDecision* plan) -> std::string {
    if (!is_tma && !future_only) return "";
    // Event-only TMAs use event barriers and don't need futures
    if (is_tma && !future_only && event_only) return "";
    // Warpspec TMA never needs futures (events or direct commit/wait)
    if (is_tma && !future_only && IsWarpSpecActive()) return "";
    auto future_name = n.future;

    auto cp_atom_name =
        GetCopyAtomName(is_tma, (is_tma ? tma_future_count : dma_count_));
    if (future_name.empty()) {
      future_name = "__choreo_anon_fut__" + std::to_string(future_count_);
    } else {
      // if the future has already been claimed, return the future name
      if (claimed_futs.count(InScopeName(future_name))) return future_name;
      // claim the future
      claimed_futs.emplace(InScopeName(future_name), cp_atom_name);
      ssm.MapDeviceSymbol(InScopeName(future_name), future_name);
      ssm.MapDeviceSymbol(InScopeName(future_name) + ".data",
                          future_name + ".data()");
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
    ds << ");\n";
    if (is_tma) {
      ds << d_indent << future_name << ".is_tma = true;\n";
      ds << d_indent << future_name << ".set_atom(&" << cp_atom_name << ");\n";
      if (!n.future.empty()) tma_futures_.insert(InScopeName(n.future));
    } else if (is_async) {
      auto cp_atom_type = GetCopyAtomType(plan->atom, t_sty->ElementType());
      ds << d_indent << cp_atom_type << " " << cp_atom_name << "{};\n";
      ds << d_indent << future_name << ".set_atom(&" << cp_atom_name << ");\n";
      ds << d_indent << future_name << ".set_ring(" << device_fn
         << "__ring__);\n";
      ds << d_indent << future_name << ".id = " << future_count_ << ";\n";
    }

    return future_name;
  };

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

  assert(!IsHost() && "DMA is not supported on host-side currently");

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

  const auto f_buf = GetBufferExpr(f_sym, f_idx, f_ty);
  const auto t_buf = GetBufferExpr(t_sym, t_idx, t_ty);

  // Only create a future variable when DMA has an explicit future name,
  // or always for TMA (which creates anonymous futures when unnamed).
  std::string future_name = claimFuture(t_buf.second, dma_plan);
  bool has_future = !future_name.empty();

  auto DMACodeGen = [&]() {
    std::string f_mds_offset = "";
    std::string t_mds_offset = "";
    Shape f_shape = f_sty->GetShape();
    Shape t_shape = t_sty->GetShape();
    const auto& f_buf_name = f_buf.first;
    const auto& t_buf_name = t_buf.first;
    // currently, span_as should not be the last operation
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

    // For .pad with naive copy, the source tensor covers the unpadded
    // input region.  For tiled pad, the source needs the full box shape
    // so the tiled_copy partitioning works (predicate handles OOB).
    auto f_mds_shape = (n.operation == ".pad" && !dma_plan->IsTiledDMA())
                           ? dma_plan->from_ca_shape
                           : dma_plan->GetBoxOfFrom();
    // For .pad the destination tensor is the full padded allocation; the
    // tiled copy only covers the inner data region (box_shape = FROM shape).
    auto t_mds_shape = (n.operation == ".pad" && dma_plan->IsTiledDMA())
                           ? t_ca->GetBlockShape()
                           : dma_plan->GetBoxOfTo();
    auto f_stride = dma_plan->from_strides;
    auto t_stride = dma_plan->to_strides;
    auto swizzle_mode = dma_plan->swizzle_mode;
    auto elem_type = dma_plan->elem_type;

    // Determine if we should use WGMMA layout for destination tensor
    bool use_wgmma_layout_t = HasWGMMAInFunction() &&
                              t_sty->GetStorage() == Storage::SHARED &&
                              (t_sty->ElementType() == BaseType::F16 ||
                               t_sty->ElementType() == BaseType::BF16 ||
                               t_sty->ElementType() == BaseType::F8_E4M3);

    // For .transp the DMA plan already produces box_shape and from_strides
    // in the destination (transposed) coordinate space, so we must NOT
    // re-permute the shape here -- that would double-transpose it.
    //
    // When batch_dims is set the outer tensors are unused -- the batch loop
    // creates its own sub-tensors with per-iteration pointer offsets.
    std::string f_mds_name, t_mds_name;
    if (dma_plan->batch_dims.empty() && !dma_plan->dyn_box) {
      // Skip outer tensor decls when batch_dims or dyn_box is set -- those
      // paths create their own sub-tensors with per-iteration offsets.
      const auto f_mds = GenTensorDecl(
          RemoveSuffix(f_buf_name, ".data()"), f_buf.second,
          f_sty->GetStorage(), f_sty->ElementType(), f_mds_shape, false,
          f_mds_offset, ValueSTR(f_stride, false, true), {}, false);
      const auto t_mds =
          GenTensorDecl(RemoveSuffix(t_buf_name, ".data()"), t_buf.second,
                        t_sty->GetStorage(), t_sty->ElementType(), t_mds_shape,
                        false, t_mds_offset, ValueSTR(t_stride, false, true),
                        {}, use_wgmma_layout_t, swizzle_mode);
      f_mds_name = f_mds.first;
      t_mds_name = t_mds.first;
      ds << f_mds.second;
      ds << t_mds.second;
    }

    if (has_future && swizzle_mode != SwizMode::NONE)
      shared_buf_swiz_[future_name] = swizzle_mode;
    if (t_sty->GetStorage() == Storage::SHARED &&
        swizzle_mode != SwizMode::NONE) {
      shared_buf_swiz_[t_buf.first] = swizzle_mode;
      shared_buf_swiz_[InScopeName(t_sym)] = swizzle_mode;
    }

    if (future_only) cooperatives.insert(InScopeName(n.future));
    if (n.operation != ".copy" && n.operation != ".transp" &&
        n.operation != ".pad")
      choreo_unreachable("unsupported dma operation: " + n.operation + ".");

    // --- Shared emit helpers for .copy, .transp, and .pad ---
    auto emit_naive_copy = [&](const std::string& src, const std::string& dst,
                               const CUDA_COPY_ATOM&) {
      bool to_or_from_shared = dma_plan->direction == DMADirection::G2S ||
                               dma_plan->direction == DMADirection::S2G ||
                               dma_plan->direction == DMADirection::S2S;
      if (to_or_from_shared) {
        ds << d_indent << "if (__CHOREO_BLOCK_SINGLE__) {\n";
        ds << d_indent << "  choreo::naive_copy(" << src << ", " << dst
           << ");\n";
        ds << d_indent << "}\n";
        ds << d_indent << "__syncthreads();\n";
      } else {
        ds << d_indent << "choreo::naive_copy(" << src << ", " << dst << ");\n";
      }
    };

    auto emit_tiled_copy = [&](const TiledCopyParams& params,
                               const std::string& src, const std::string& dst,
                               CUDA_COPY_ATOM copy_atom,
                               const std::string& indent_prefix = "",
                               bool emit_fence = true) {
      assert(params.box_shape.size() == 2 && params.thr_layout.size() == 2 &&
             params.val_layout.size() == 2 &&
             "tiled copy expects 2D static layouts");

      auto ind = indent_prefix + d_indent;
      bool use_swizzle = swizzle_mode != SwizMode::NONE;
      bool need_zfill = params.need_pred && dma_plan->is_zfill;
      ds << ind << "choreo::tiled_copy<"
         << GetCopyAtomType(copy_atom, elem_type) << ", "
         << ValueSTR(params.thr_layout[0]) << ", "
         << ValueSTR(params.thr_layout[1]) << ", "
         << ValueSTR(params.val_layout[0]) << ", "
         << ValueSTR(params.val_layout[1]) << ", "
         << (use_swizzle ? "true" : "false") << ", "
         << (params.need_pred ? "true" : "false") << ", "
         << (need_zfill ? "true" : "false") << ">(" << src << ", " << dst
         << ", ";
      if (params.need_pred) {
        auto pred_shape = Shape(params.prediction);
        ds << "[&](const auto& __coord) { return cute::elem_less(__coord, "
              "cute::make_shape("
           << ShapeSTR(pred_shape, true) << ")); }";
      } else {
        ds << "[](const auto&) { return true; }";
      }
      ds << ");\n";

      if (!emit_fence) return;

      switch (copy_atom) {
      case CUDA_COPY_ATOM::CP_ASYNC_128B:
      case CUDA_COPY_ATOM::CP_ASYNC_64B:
      case CUDA_COPY_ATOM::CP_ASYNC_32B:
        if (fty->IsAsync()) {
          ds << ind << "cute::cp_async_fence();\n";
          if (future_only) ds << ind << future_name << ".trigger();\n";
        } else {
          ds << ind << "cute::cp_async_fence();\n";
          ds << ind << "cute::cp_async_wait<0>();\n";
        }
        break;
      default:
        if (fty->IsAsync() && future_only)
          ds << ind << future_name << ".trigger();\n";
        break;
      }
    };

    // Emit fence/trigger/wait for a copy atom (used after batch loops).
    auto emit_copy_fence = [&](CUDA_COPY_ATOM copy_atom,
                               const std::string& indent_prefix = "") {
      auto ind = indent_prefix + d_indent;
      switch (copy_atom) {
      case CUDA_COPY_ATOM::CP_ASYNC_128B:
      case CUDA_COPY_ATOM::CP_ASYNC_64B:
      case CUDA_COPY_ATOM::CP_ASYNC_32B:
        if (fty->IsAsync()) {
          ds << ind << "cute::cp_async_fence();\n";
          if (future_only) ds << ind << future_name << ".trigger();\n";
        } else {
          ds << ind << "cute::cp_async_fence();\n";
          ds << ind << "cute::cp_async_wait<0>();\n";
        }
        break;
      default:
        if (fty->IsAsync() && future_only)
          ds << ind << future_name << ".trigger();\n";
        break;
      }
    };

    // --- Determine actual copy source and destination ---
    // For .pad, the destination is a sub-tensor at pad_low offset inside
    // the full padded tensor.  For .copy/.transp, it is t_mds directly.
    std::string copy_src = f_mds_name;
    std::string copy_dst = t_mds_name;
    if (n.operation == ".pad") {
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

      std::string pad_val_expr = ExprSTR(pad_config->GetPadValue(), IsHost());

      // Fill the full destination tensor with the pad value.
      if (dma_plan->IsTiledDMA())
        ds << d_indent << "choreo::cooperative_fill(" << t_mds_name << ", "
           << pad_val_expr << ");\n";
      else
        ds << d_indent << "cute::fill(" << t_mds_name << ", " << pad_val_expr
           << ");\n";

      // Compute pad_low offset and declare the inner data sub-tensor.
      std::string pad_offset = "__pad_offset" + std::to_string(pad_cnt);
      ds << d_indent << "auto " << pad_offset << " = " << t_mds_name
         << ".layout()("
         << "cute::make_coord(" << pcmvSTR(pad_config->pad_low) << "));\n";

      auto f_data_shape = dma_plan->IsTiledDMA() ? dma_plan->GetBoxOfFrom()
                                                 : dma_plan->from_ca_shape;
      const auto t_pad_mds =
          GenTensorDecl(RemoveSuffix(t_buf_name, ".data()"), t_buf_name,
                        t_sty->GetStorage(), t_sty->ElementType(), f_data_shape,
                        false, pad_offset, ValueSTR(t_stride, false, true), {},
                        use_wgmma_layout_t, swizzle_mode);
      ds << t_pad_mds.second;
      copy_dst = t_pad_mds.first;
      ++pad_cnt;
    }

    // --- Batch loops for high-rank DMA (outer dims peeled into loops) ---
    bool has_batch_loop = !dma_plan->batch_dims.empty();
    std::string batch_f_off, batch_t_off; // accumulated offsets for dyn_box use
    int batch_loop_depth = 0;
    if (has_batch_loop) {
      static int batch_cnt = 0;

      auto mk_off = [](const std::string& base, const std::string& bv,
                       const std::string& stride) -> std::string {
        auto batch = bv + " * " + stride;
        if (base.empty() || base == "0") return batch;
        return "(" + base + ") + " + batch;
      };

      std::string f_off = f_mds_offset;
      std::string t_off = t_mds_offset;
      for (auto& bd : dma_plan->batch_dims) {
        auto bv = "__batch_i" + std::to_string(batch_cnt++);
        ds << d_indent << "for (int " << bv << " = 0; " << bv << " < "
           << ValueSTR(bd.size) << "; ++" << bv << ") {\n";
        IncrDeviceIndent();
        ++batch_loop_depth;
        f_off = mk_off(f_off, bv, ValueSTR(bd.from_stride));
        t_off = mk_off(t_off, bv, ValueSTR(bd.to_stride));
      }

      if (dma_plan->dyn_box) {
        // When dyn_box is set, skip creating sub-tensors here -- the dyn_box
        // loop below creates its own sub-tensors with per-tile offsets.
        batch_f_off = f_off;
        batch_t_off = t_off;
      } else {
        const auto f_sub = GenTensorDecl(
            RemoveSuffix(f_buf_name, ".data()"), f_buf_name,
            f_sty->GetStorage(), f_sty->ElementType(), dma_plan->GetBoxOfFrom(),
            false, f_off, ValueSTR(f_stride, false, true), {}, false);
        const auto t_sub = GenTensorDecl(
            RemoveSuffix(t_buf_name, ".data()"), t_buf_name,
            t_sty->GetStorage(), t_sty->ElementType(), dma_plan->GetBoxOfTo(),
            false, t_off, ValueSTR(t_stride, false, true), {},
            use_wgmma_layout_t, swizzle_mode);
        ds << f_sub.second;
        ds << t_sub.second;
        copy_src = f_sub.first;
        copy_dst = t_sub.first;
      }
    }

    // --- Emit the actual copy (shared for .copy, .transp, and .pad) ---
    bool suppress_fence = has_batch_loop;

    if (dma_plan->dyn_box && dma_plan->IsTiledDMA()) {
      // Dynamic box: emit explicit tiling loop over runtime extents.
      // The inner 2D shape is [dyn_m, dyn_n].  We tile it with a fixed
      // [1, TILE_N] tile using an explicit double-loop.  Full tiles use the
      // fast (vectorized, no-pred) atom; the last partial tile uses the slow
      // (element-level, predicated) atom.
      suppress_fence = true; // fence emitted after all loops
      auto& db = *dma_plan->dyn_box;
      auto resolve_dyn_extent = [&](const ValueItem& extent) {
        if (auto sym = VISym(extent)) {
          std::array<std::string, 3> candidates = {
              *sym,
              InScopeName(UnScopedExpr(*sym)),
              InScopeNameForRef(UnScopedExpr(*sym)),
          };
          for (auto& candidate : candidates) {
            if (!FCtx(fname).HasSymbolValues(candidate)) continue;
            auto svs = FCtx(fname).GetSymbolValues(candidate);
            if (svs.HasVal()) return ValueSTR(svs.GetVal());
          }
        }
        return ValueSTR(extent);
      };
      std::string dyn_m_str = resolve_dyn_extent(db.dyn_extent_m);
      std::string dyn_n_str = resolve_dyn_extent(db.dyn_extent_n);
      std::string tile_n_str = std::to_string(db.tile_n);

      // Row strides for the inner 2D shape.
      std::string f_row_stride = dma_plan->from_strides.empty()
                                     ? dyn_n_str
                                     : ValueSTR(dma_plan->from_strides[0]);
      std::string t_row_stride = dma_plan->to_strides.empty()
                                     ? dyn_n_str
                                     : ValueSTR(dma_plan->to_strides[0]);

      // Base offsets (may include batch loop offset).
      std::string f_base_off = has_batch_loop ? batch_f_off : f_mds_offset;
      std::string t_base_off = has_batch_loop ? batch_t_off : t_mds_offset;

      static int dyn_cnt = 0;
      std::string rm = "__dyn_rm" + std::to_string(dyn_cnt);
      std::string rn = "__dyn_rn" + std::to_string(dyn_cnt);
      std::string pred_n = "__dyn_pn" + std::to_string(dyn_cnt);
      ++dyn_cnt;

      auto emit_dyn_sub_tensor = [&](const std::string& buf_name, Storage sto,
                                     BaseType bty, const std::string& base_off,
                                     const std::string& row_stride,
                                     const std::string& rm_v,
                                     const std::string& rn_v) {
        static int sub_cnt = 0;
        std::string sn = "__dyn_sub" + std::to_string(sub_cnt++);
        std::string st = NameBaseType(bty);
        auto ptr_kind = (sto == Storage::SHARED) ? "smem_ptr" : "gmem_ptr";
        // offset = base_off + rm * row_stride + rn
        std::string off_expr;
        if (!base_off.empty() && base_off != "0")
          off_expr = "(" + base_off + ") + ";
        off_expr += rm_v + " * " + row_stride + " + " + rn_v;
        ds << d_indent << "auto " << sn << " = cute::make_tensor("
           << "cute::make_" << ptr_kind << "<" << st << ">((" << st << "*)"
           << buf_name << " + " << off_expr << "), "
           << "cute::make_layout(cute::make_shape(cute::Int<1>{}, cute::Int<"
           << tile_n_str << ">{}), "
           << "cute::make_stride(cute::Int<" << tile_n_str
           << ">{}, cute::Int<1>{})));\n";
        return sn;
      };

      auto f_sto = f_sty->GetStorage();
      auto t_sto = t_sty->GetStorage();

      ds << d_indent << "for (unsigned " << rm << " = 0; " << rm << " < "
         << "(unsigned)" << dyn_m_str << "; ++" << rm << ") {\n";
      IncrDeviceIndent();
      ds << d_indent << "for (unsigned " << rn << " = 0; " << rn << " < "
         << "(unsigned)" << dyn_n_str << "; " << rn << " += " << tile_n_str
         << ") {\n";
      IncrDeviceIndent();
      ds << d_indent << "unsigned " << pred_n << " = "
         << "(unsigned)" << dyn_n_str << " - " << rn << " < " << tile_n_str
         << " ? (unsigned)" << dyn_n_str << " - " << rn << " : " << tile_n_str
         << ";\n";

      auto f_sub = emit_dyn_sub_tensor(f_buf_name, f_sto, elem_type, f_base_off,
                                       f_row_stride, rm, rn);
      auto t_sub = emit_dyn_sub_tensor(t_buf_name, t_sto, elem_type, t_base_off,
                                       t_row_stride, rm, rn);

      // Full tile -> fast path (vectorized, Pred=false).
      ds << d_indent << "if (" << pred_n << " == " << tile_n_str << ") {\n";
      if (dma_plan->tiled_params_fast) {
        auto fast_p = *dma_plan->tiled_params_fast;
        fast_p.need_pred = false;
        emit_tiled_copy(fast_p, f_sub, t_sub, dma_plan->atom_fast, "  ", false);
      } else {
        auto full_p = *dma_plan->tiled_params;
        full_p.need_pred = false;
        emit_tiled_copy(full_p, f_sub, t_sub, dma_plan->atom, "  ", false);
      }
      ds << d_indent << "} else {\n";
      // Partial tile -> slow path (element-level, Pred=true with runtime
      // bound).
      {
        auto ind = "  " + d_indent;
        bool need_zfill = dma_plan->is_zfill;
        ds << ind << "choreo::tiled_copy<"
           << GetCopyAtomType(dma_plan->atom, elem_type) << ", "
           << ValueSTR(dma_plan->tiled_params->thr_layout[0]) << ", "
           << ValueSTR(dma_plan->tiled_params->thr_layout[1]) << ", "
           << ValueSTR(dma_plan->tiled_params->val_layout[0]) << ", "
           << ValueSTR(dma_plan->tiled_params->val_layout[1]) << ", "
           << "false, true, " << (need_zfill ? "true" : "false") << ">("
           << f_sub << ", " << t_sub << ", "
           << "[&](const auto& __coord) { return cute::elem_less(__coord, "
           << "cute::make_shape(cute::Int<1>{}, " << pred_n << ")); });\n";
      }
      ds << d_indent << "}\n";
      DecrDeviceIndent();
      ds << d_indent << "}\n"; // rn loop
      DecrDeviceIndent();
      ds << d_indent << "}\n"; // rm loop
      if (!has_batch_loop) {
        // No enclosing batch loop -> emit fence here after all dyn_box loops.
        CUDA_COPY_ATOM fence_atom =
            dma_plan->tiled_params_fast ? dma_plan->atom_fast : dma_plan->atom;
        emit_copy_fence(fence_atom);
      }
    } else if (dma_plan->IsTiledDMA() && dma_plan->tiled_params_fast) {
      size_t fast_align = dma_plan->tiled_params_fast->align_bits;
      size_t elems_per_fast = fast_align / (SizeOf(dma_plan->elem_type) * 8);
      bool f_dyn = !dma_plan->from_strides.empty() &&
                   !VIIsInt(dma_plan->from_strides[0]);
      bool t_dyn =
          !dma_plan->to_strides.empty() && !VIIsInt(dma_plan->to_strides[0]);
      std::string cond;
      if (f_dyn) {
        cond = ValueSTR(dma_plan->from_strides[0]) + " % " +
               std::to_string(elems_per_fast) + " == 0";
        auto inner_dim =
            dma_plan->from_ca_shape.ValueAt(dma_plan->from_ca_shape.Rank() - 1);
        if (!VIIsInt(inner_dim)) {
          cond = cond + " && " + ValueSTR(inner_dim) + " % " +
                 std::to_string(elems_per_fast) + " == 0";
        }
      }
      if (t_dyn) {
        std::string t_cond = ValueSTR(dma_plan->to_strides[0]) + " % " +
                             std::to_string(elems_per_fast) + " == 0";
        cond = cond.empty() ? t_cond : cond + " && " + t_cond;
        auto inner_dim =
            dma_plan->to_ca_shape.ValueAt(dma_plan->to_ca_shape.Rank() - 1);
        if (!VIIsInt(inner_dim)) {
          cond = cond + " && " + ValueSTR(inner_dim) + " % " +
                 std::to_string(elems_per_fast) + " == 0";
        }
      }

      ds << d_indent << "if (" << cond << ") {\n";
      emit_tiled_copy(*dma_plan->tiled_params_fast, copy_src, copy_dst,
                      dma_plan->atom_fast, "  ", !suppress_fence);
      ds << d_indent << "} else {\n";
      emit_tiled_copy(*dma_plan->tiled_params, copy_src, copy_dst,
                      dma_plan->atom, "  ", !suppress_fence);
      ds << d_indent << "}\n";
    } else if (dma_plan->IsTiledDMA()) {
      emit_tiled_copy(*dma_plan->tiled_params, copy_src, copy_dst,
                      dma_plan->atom, "", !suppress_fence);
    } else {
      emit_naive_copy(copy_src, copy_dst, dma_plan->atom);
      if (has_future && fty->IsAsync())
        ds << d_indent << future_name << ".trigger();\n";
    }

    if (has_batch_loop) {
      for (int i = 0; i < batch_loop_depth; ++i) {
        DecrDeviceIndent();
        ds << d_indent << "}\n";
      }
      if (dma_plan->IsTiledDMA()) {
        CUDA_COPY_ATOM fence_atom =
            dma_plan->tiled_params_fast ? dma_plan->atom_fast : dma_plan->atom;
        emit_copy_fence(fence_atom);
      }
    }

    VerboseDMA(ds, d_indent, t_sym, f_sym, n.operation.substr(1), "", 1,
               ", line " + std::to_string(n.LOC().begin.line));

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
    int tma_idx = -1;
    if (n.IsTMA() && !isa<PlaceHolderType>(NodeType(n))) {
      tma_idx = tma_count++;
      assert(tma_idx < static_cast<int>(tma_descs.size()));
      const TMADesc& tma_desc = tma_descs[tma_idx];
      auto in_thr_block = tma_desc.GetInThreadsBlock();
      if (in_thr_block) {
        tma_sync_level = tma_desc.GetPBLevel();
      } else if (InSpecWarp() && dma_plan->direction == DMADirection::S2G) {
        // Inside a warp-spec inthreads block, __syncthreads would deadlock;
        // use GROUPx4-level sync for the S2G store guard.
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
    auto swizzle_mode = dma_plan->swizzle_mode;

    if (t_sty->GetStorage() == Storage::SHARED &&
        swizzle_mode != SwizMode::NONE) {
      if (has_future) shared_buf_swiz_[future_name] = swizzle_mode;
      shared_buf_swiz_[t_buf.first] = swizzle_mode;
      shared_buf_swiz_[InScopeName(t_sym)] = swizzle_mode;
    }

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
      std::string t_buf_void_ptr = "(void*)(" + t_buf_expr_with_offset + ")";

      bool warpspec_only = InSpecWarp();
      if (warpspec_only && event_only) {
        assert(n.IsAsync() && "warpspec event-only tma copy must be async");
      }

      auto rev_indices = Reverse(GenIndices(f_ca));

      // When TMA source is a computed subview of a root parameter,
      // recompute TMA coordinates using the subview's outermost stride
      // as inner_dim.  This matches EmitTMAConfiguration which also uses
      // the subview stride for the tensor-map row stride.
      //
      // GenIndices gives logical tile indices that don't account for
      // non-contiguous strides.  Use GenOffset instead to get the true
      // element offset from both the tile position and the subview
      // definition, then split into the TMA's 2D coordinate space.
      if (tma_idx >= 0 && tma_descs[tma_idx].HasRootParam()) {
        const auto& def_ca = tma_descs[tma_idx].GetRootDefCA();
        size_t end_idx = def_ca->OpCount();
        while (end_idx > 0 && isa<AST::SOP::Reshape>(def_ca->OpAt(end_idx - 1)))
          --end_idx;
        auto def_elem_offset = GenOffset(def_ca, end_idx);
        auto tile_elem_offset = GenOffset(f_ca);
        auto total_elem = tile_elem_offset + def_elem_offset;
        auto inner_dim = f_sty->GetStrides().at(0);

        assert(rev_indices.size() >= 2);
        rev_indices.front() = total_elem % inner_dim;
        rev_indices.back() = total_elem / inner_dim;
      }

      // If host-side TMA was split (inner dim exceeded swizzle width),
      // expand 2D coordinates to 3D.  The groups dimension is outermost
      // so TMA fills all rows of group 0 before group 1 (atom-major
      // layout matching WGMMA B128 expectations).
      // Coordinate order: [inner%swiz, outer, inner/swiz]
      bool tma_has_inner_split = false;
      if (tma_idx >= 0) {
        auto split_it = tma_inner_splits_.find(tma_descs[tma_idx].GetName());
        if (split_it != tma_inner_splits_.end()) {
          tma_has_inner_split = true;
          auto se = sbe::nu((int64_t)split_it->second.swiz_elems);
          auto orig_inner = rev_indices.front();
          auto orig_outer = rev_indices.back();
          rev_indices.clear();
          rev_indices.push_back(orig_inner % se);
          rev_indices.push_back(orig_outer);
          rev_indices.push_back(orig_inner / se);
        }
      }
      size_t effective_tma_rank =
          tma_has_inner_split ? t_shape.Rank() + 1 : t_shape.Rank();

      bool is_multicast_tma = n.IsMulticast() && n.IsTMA();
      bool emit_tma_single_guard =
          !ScopeAlreadySingleThreadForLevel(tma_sync_level);
      if (!warpspec_only)
        assert(emit_tma_single_guard &&
               "non-warpspec tma copy must be single-threaded");

      std::string tma_issue_prefix = emit_tma_single_guard ? "  " : "";

      if (emit_tma_single_guard) {
        if (tma_sync_level == ParallelLevel::GROUP)
          ds << d_indent << "if (__CHOREO_GROUP_SINGLE__) {\n";
        else if (tma_sync_level == ParallelLevel::GROUPx4 &&
                 IsWarpSpecActive()) {
          std::string wg_tid_zero = current_thread_count_expr.empty()
                                        ? "(threadIdx.x % 128) == 0"
                                        : "__choreo_vtid_x == 0";
          ds << d_indent << "if (" << wg_tid_zero << ") {\n";
        } else if (tma_sync_level == ParallelLevel::GROUPx4)
          ds << d_indent << "if (__CHOREO_GROUPX4_SINGLE__) {\n";
        else
          ds << d_indent << "if (__CHOREO_BLOCK_SINGLE__) {\n";
      }

      // PTX mbarrier TMA path (unified for all ranks and cases)
      // arrive_and_expect_tx MUST be emitted before the PTX TMA load.
      std::string ptx_bar_expr;
      if (event_only) {
        ptx_bar_expr = "(uint64_t*)&(" + ExprSTR(n.Event(), IsHost()) + ")";
        auto event_expr = ExprSTR(n.Event(), IsHost());
        // Extract the event base name to detect multiple TMA loads
        // sharing the same barrier event within one iteration.
        std::string evt_base_name;
        {
          auto evt = n.Event();
          auto* expr_evt = cast<AST::Expr>(evt.get());
          if (expr_evt && expr_evt->op == Op::ElemOf) {
            auto sym = AST::GetArrayBaseSymbol(*expr_evt);
            if (sym) evt_base_name = sym->name;
          } else if (expr_evt && expr_evt->GetSymbol()) {
            evt_base_name = expr_evt->GetSymbol()->name;
          }
        }
        bool already_arrived = !evt_base_name.empty() &&
                               event_arrive_tx_events_.count(evt_base_name);
        if (already_arrived) {
          // A prior TMA load already called arrive_and_expect_tx on this
          // barrier; only add expected bytes without an extra arrival.
          ds << d_indent << tma_issue_prefix << event_expr
             << ".expect_transaction(" << tma_tx_bytes_expr << ");\n";
        } else {
          ds << d_indent << tma_issue_prefix << event_expr
             << ".arrive_and_expect_tx(" << tma_tx_bytes_expr << ");\n";
          if (!evt_base_name.empty())
            event_arrive_tx_events_.insert(evt_base_name);
        }
        // Mark this event's trigger as a no-op since arrive_and_expect_tx
        // is already emitted before the PTX TMA load.
        if (!evt_base_name.empty())
          tma_bound_event_triggers_.insert(evt_base_name);
      } else if (!future_name.empty()) {
        ptx_bar_expr =
            "((TMAAtom*)" + future_name + ".get_atom())->ptx_barrier()";
        ds << d_indent << tma_issue_prefix << "choreo::tma_mbarrier_expect_tx("
           << ptx_bar_expr << ", " << tma_tx_bytes_expr << ");\n";
      } else {
        auto tma_atom_ref = GetCopyAtomName(true, tma_idx);
        ptx_bar_expr = tma_atom_ref + ".ptx_barrier()";
        ds << d_indent << tma_issue_prefix << "choreo::tma_mbarrier_expect_tx("
           << ptx_bar_expr << ", " << tma_tx_bytes_expr << ");\n";
      }

      auto coord0_expr = ValueSTR(rev_indices.at(0));
      auto coord1_expr = ValueSTR(rev_indices.at(1));

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
           << t_buf_void_ptr << ", (const void*)&" << *tname << "_tensor_map, "
           << ptx_bar_expr << ", " << coord0_expr << ", " << coord1_expr << ", "
           << "static_cast<uint16_t>((1u << " << ValueSTR(cluster_total)
           << ") - 1u)"
           << ");\n";
        ds << d_indent << tma_issue_prefix << "}\n";
      } else if (tma_cluster_aware) {
        ds << d_indent << tma_issue_prefix
           << "choreo::tma_load_2d_shared_cluster_global_mbarrier("
           << t_buf_void_ptr << ", (const void*)&" << *tname << "_tensor_map, "
           << ptx_bar_expr << ", " << coord0_expr << ", " << coord1_expr
           << ");\n";
      } else if (effective_tma_rank == 3) {
        auto coord2_expr = ValueSTR(rev_indices.at(2));
        ds << d_indent << tma_issue_prefix
           << "choreo::tma_load_3d_shared_cta_global_mbarrier("
           << t_buf_void_ptr << ", (const void*)&" << *tname << "_tensor_map, "
           << ptx_bar_expr << ", " << coord0_expr << ", " << coord1_expr << ", "
           << coord2_expr << ");\n";
      } else {
        ds << d_indent << tma_issue_prefix
           << "choreo::tma_load_2d_shared_cta_global_mbarrier("
           << t_buf_void_ptr << ", (const void*)&" << *tname << "_tensor_map, "
           << ptx_bar_expr << ", " << coord0_expr << ", " << coord1_expr
           << ");\n";
      }
      if (emit_tma_single_guard) { ds << d_indent << "}\n"; }
      recent_tma_tx_bytes.push_back(tma_tx_bytes_expr);
      if (recent_tma_tx_bytes.size() > 8) recent_tma_tx_bytes.pop_front();

      // For async tma.copy.async, trigger the future
      // For sync tma.copy, default behavior is immediate wait.
      // In warpspec mode, defer this wait to event barrier protocol
      // so producer-consumer pipelining remains asynchronous like ref kernels.
      if (fty->IsAsync()) {
        if (has_future) ds << d_indent << future_name << ".trigger();\n";
      } else {
        auto tma_atom_ref = GetCopyAtomName(true, tma_idx);
        if (has_future) {
          ds << d_indent << "choreo::tma_mbarrier_wait_parity(((TMAAtom*)"
             << future_name << ".get_atom())->ptx_barrier(), ((TMAAtom*)"
             << future_name << ".get_atom())->ptx_phase_bit());\n";
          ds << d_indent << "((TMAAtom*)" << future_name
             << ".get_atom())->toggle_ptx_phase();\n";
          ds << d_indent << future_name << ".set_nowait();\n\n";
        } else if (!event_only) {
          ds << d_indent << "choreo::tma_mbarrier_wait_parity(" << tma_atom_ref
             << ".ptx_barrier(), " << tma_atom_ref << ".ptx_phase_bit());\n";
          ds << d_indent << tma_atom_ref << ".toggle_ptx_phase();\n";
        }
      }
    } else if ((tsto == Storage::GLOBAL || tsto == Storage::DEFAULT) &&
               fsto == Storage::SHARED) {

      ds << d_indent << "cde::fence_proxy_async_shared_cta();\n";
      {
        if (tma_sync_level == ParallelLevel::GROUP)
          ds << d_indent << "__syncwarp();\n";
        else if (tma_sync_level == ParallelLevel::GROUPx4 ||
                 NeedWarpSpecGroupX4SyncForCurrentScope()) {
          int groupx4_sync_threads =
              CurrentScopeThreadsCount() > 0 ? CurrentScopeThreadsCount() : 128;
          EmitGroupX4Sync(ds, d_indent, groupx4_sync_threads);
        } else
          ds << d_indent << "__syncthreads();\n";
      }

      if (tma_sync_level == ParallelLevel::GROUP) {
        ds << d_indent << "if (__CHOREO_GROUP_SINGLE__) {\n";
      } else if (tma_sync_level == ParallelLevel::GROUPx4) {
        std::string wg_tid_zero = current_thread_count_expr.empty()
                                      ? "(threadIdx.x % 128) == 0"
                                      : "__choreo_vtid_x == 0";
        if (InSpecWarp() && current_inthreads->ActiveWarpGroup() < 0) {
          auto consumer_wgs = CurrentWarpGroupIndices();
          int64_t first_wg = consumer_wgs.empty() ? 0 : consumer_wgs.front();
          ds << d_indent << "if (" << wg_tid_zero
             << " && __choreo_vg4id_x == " << first_wg << ") {\n";
        } else {
          ds << d_indent << "if (" << wg_tid_zero << ") {\n";
        }
      } else {
        ds << d_indent << "if (__CHOREO_BLOCK_SINGLE__) {\n";
      }

      {
        auto s2g_indices_orig = Reverse(GenIndices(t_ca));

        if (tma_idx >= 0 && tma_descs[tma_idx].HasRootParam()) {
          const auto& def_ca = tma_descs[tma_idx].GetRootDefCA();
          size_t end_idx = def_ca->OpCount();
          while (end_idx > 0 &&
                 isa<AST::SOP::Reshape>(def_ca->OpAt(end_idx - 1)))
            --end_idx;
          auto def_elem_offset = GenOffset(def_ca, end_idx);
          auto tile_elem_offset = GenOffset(t_ca);
          auto total_elem = tile_elem_offset + def_elem_offset;
          auto inner_dim = t_sty->GetStrides().at(0);

          assert(s2g_indices_orig.size() >= 2);
          s2g_indices_orig.front() = total_elem % inner_dim;
          s2g_indices_orig.back() = total_elem / inner_dim;
        }

        size_t s2g_rank = t_shape.Rank();
        bool has_split = false;
        sbe::Operand split_se;
        if (tma_idx >= 0) {
          auto s2g_it = tma_inner_splits_.find(tma_descs[tma_idx].GetName());
          if (s2g_it != tma_inner_splits_.end()) {
            has_split = true;
            split_se = sbe::nu((int64_t)s2g_it->second.swiz_elems);
            s2g_rank += 1;
          }
        }

        auto make_s2g_indices = [&](const ValueList& base) -> ValueList {
          if (!has_split) return base;
          ValueList out;
          out.push_back(base.front() % split_se);
          out.push_back(base.back());
          out.push_back(base.front() / split_se);
          return out;
        };

        bool multi_consumer_s2g = IsWarpSpecActive() && InSpecWarp() &&
                                  current_inthreads &&
                                  current_inthreads->ActiveWarpGroup() < 0;

        if (multi_consumer_s2g) {
          auto consumer_wgs = CurrentWarpGroupIndices();
          int64_t first_wg = consumer_wgs.empty() ? 1 : consumer_wgs.front();
          int num_consumers = static_cast<int>(consumer_wgs.size());

          int64_t box_elems = 1;
          {
            auto box = dma_plan->GetBoxOfFrom();
            for (size_t d = 0; d < box.Rank(); ++d)
              if (auto v = VIInt(box.ValueAt(d))) box_elems *= *v;
          }

          auto final_idx_base = make_s2g_indices(s2g_indices_orig);

          auto strReplaceAll = [](std::string s, const std::string& from,
                                  const std::string& to) {
            for (size_t pos = 0; (pos = s.find(from, pos)) != std::string::npos;
                 pos += to.size())
              s.replace(pos, from.size(), to);
            return s;
          };

          if (num_consumers > 1) {
            std::string idx_str = ValueSTR(final_idx_base);
            std::map<std::string, std::string> var_to_expr;
            for (auto& [expr, var] : known_val_str_to_var_)
              var_to_expr[var] = expr;

            ds << d_indent << "  for (int __s2g_ci = 0; __s2g_ci < "
               << num_consumers << "; ++__s2g_ci) {\n";
            ds << d_indent << "    auto __s2g_vg4id = __s2g_ci + " << first_wg
               << ";\n";

            for (auto& [var, expr] : var_to_expr) {
              if (expr.find("__choreo_vg4id_x") != std::string::npos) {
                std::string local_expr =
                    strReplaceAll(expr, "__choreo_vg4id_x", "__s2g_vg4id");
                idx_str = strReplaceAll(idx_str, var, "(" + local_expr + ")");
              }
            }

            ds << d_indent << "    cde::cp_async_bulk_tensor_" << s2g_rank
               << "d_shared_to_global(&" << *tname << "_tensor_map, " << idx_str
               << ", " << f_buf_expr << " + __s2g_ci * " << box_elems << ");\n";
            ds << d_indent << "  }\n";
          } else {
            ds << d_indent << "  cde::cp_async_bulk_tensor_" << s2g_rank
               << "d_shared_to_global(&" << *tname << "_tensor_map, "
               << ValueSTR(final_idx_base) << ", " << f_buf_expr << ");\n";
          }
        } else {
          auto final_idx = make_s2g_indices(s2g_indices_orig);
          ds << d_indent << "  cde::cp_async_bulk_tensor_" << s2g_rank
             << "d_shared_to_global(&" << *tname << "_tensor_map, "
             << ValueSTR(final_idx) << ", " << f_buf_expr << ");\n";
        }
      }
      ds << d_indent << "  cde::cp_async_bulk_commit_group();\n";
      ds << d_indent << "  cde::cp_async_bulk_wait_group_read<0>();\n";
      ds << d_indent << "}\n";
      // DO not check or wait beyond bulk wait (matches tuned CUDA kernels).
    }
  };

  if (n.IsTMA())
    TMACodeGen();
  else
    DMACodeGen();

  return true;
}

bool CuteCodeGen::Visit(AST::MMA& n) {
  auto& op = *n.GetOperation();

  if (op.Tag() == AST::MMAOperation::Commit) {
    if (has_pending_wgmma_finalize) EmitWGMMAFinalize(ds, d_indent);
    return true;
  }

  if (op.Tag() == AST::MMAOperation::Wait) {
    int depth = op.WaitDepth();
    ds << d_indent << "warpgroup_wait<" << depth << ">();\n";
    if (depth == 0) warpspec_wgmma_arrived = false;
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
                      GetBaseType(*op.FillingValue()->GetType()), false,
                      IsNumericZeroLiteral(op.FillingValue()));
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
    case AST::MMAOperation::Desc: {
      // Check if loading from a buffer whose shape may not evenly divide
      // the MMA fragment. Static shared buffers are always tile-sized, but
      // dynamic-shaped buffers (e.g., global or runtime-sized) may have
      // dimensions that don't divide evenly by the MMA atom shape.
      {
        auto ca = op.DescFrom();
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
      if (ssmi.frag != MMAInfo::FRAG_A && ssmi.frag != MMAInfo::FRAG_B &&
          !(policy_is_sparse && ssmi.frag == MMAInfo::FRAG_E)) {
        Error1(n.LOC(),
               "mma.desc only supports WGMMA A/B and sparse metadata "
               "operands");
        break;
      }
      auto desc_source_ty = GetSpannedType(op.DescFrom()->GetType());
      if (ssmi.frag != MMAInfo::FRAG_E &&
          (!desc_source_ty ||
           desc_source_ty->GetStorage() != Storage::SHARED)) {
        Error1(n.LOC(), "WGMMA A/B descriptors require shared memory");
        break;
      }
      if (policy_is_sparse && ssmi.frag == MMAInfo::FRAG_E) {
        auto tile_addr = TileAddr(op.DescFrom(), false);
        auto strides = GenStrides(op.DescFrom());
        auto load_from_sty = dyn_cast<SpannedType>(op.DescFrom()->GetType());
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
        std::string ref_sym = op.DescFrom()->RefSymbol();
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
            auto tile_off = GenOffset(op.DescFrom());
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
      auto tile_addr = TileAddr(op.DescFrom(), false);
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
      // Descriptor layout follows the shared-memory producer. Swizzle is a
      // memory-layout property and is deliberately not part of mma.load.
      auto swizzle_val = SwizMode::NONE;
      auto src_sym = op.DescFrom()->RefSymbol();
      auto sit = shared_buf_swiz_.find(src_sym);
      if (sit == shared_buf_swiz_.end())
        sit = shared_buf_swiz_.find(InScopeName(src_sym));
      if (sit == shared_buf_swiz_.end())
        sit = shared_buf_swiz_.find(UnScopedName(src_sym));
      if (sit != shared_buf_swiz_.end()) swizzle_val = sit->second;
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
        std::string ref_sym = op.DescFrom()->RefSymbol();
        if (!ref_sym.empty()) {
          auto mdata_sym_name = ref_sym + "_mdata";
          if (SSTab().IsDeclared(mdata_sym_name)) {
            auto mdata_key = InScopeName(mdata_sym_name) + ".data";
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
      explicit_mma_descs_.insert(sym);
      explicit_mma_descs_.insert(InScopeName(sym));
      ssm.MapDeviceSymbol(InScopeName(sym), sym);
    } break;
    case AST::MMAOperation::Load: {
      if (ssmi.frag != MMAInfo::FRAG_A) {
        Error1(n.LOC(),
               "WGMMA mma.load only supports a register A operand; use "
               "mma.desc for shared-memory A/B operands");
        break;
      }
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
      auto source_sty = GetSpannedType(ca->GetType());
      if (!source_sty) {
        Error1(n.LOC(), "failed to infer mma.load source type");
        break;
      }
      auto source_tensor =
          GenTensorDecl(RemoveSuffix(ca->RefSymbol(), ".data()"),
                        ValueSTR(TileAddr(ca, false)), source_sty->GetStorage(),
                        source_sty->ElementType(), ca->GetBlockShape(), false,
                        "0", ValueSTR(GenStrides(ca), false, true));
      ds << source_tensor.second;

      auto scoped = InScopeName(sym);
      if (!FCtx(fname).HasFragmentLayout(scoped)) {
        Error1(n.LOC(), "missing fragment layout for WGMMA mma.load result");
        break;
      }
      auto& layout = FCtx(fname).GetFragmentLayout(scoped);
      if (layout.kind != LayoutKind::WGMMA_RS_A ||
          layout.regs_per_thread == 0) {
        Error1(n.LOC(), "mma.load result does not have a WGMMA RS A layout");
        break;
      }
      SwizMode source_swizzle = SwizMode::NONE;
      for (const auto& name : {ca->RefSymbol(), InScopeName(ca->RefSymbol()),
                               UnScopedName(ca->RefSymbol())}) {
        if (auto it = shared_buf_swiz_.find(name);
            it != shared_buf_swiz_.end()) {
          source_swizzle = it->second;
          break;
        }
      }
      ds << d_indent << NameBaseType(source_sty->ElementType()) << " " << sym
         << "[" << layout.regs_per_thread << "];\n";
      ds << d_indent << "{\n";
      ds << d_indent << "  auto* __mma_load_src = reinterpret_cast<const "
         << NameBaseType(source_sty->ElementType()) << "*>("
         << source_tensor.first << ".data().get());\n";
      ds << d_indent << "  int __mma_load_lane = threadIdx.x % 128;\n";
      ds << d_indent << "  #pragma unroll\n";
      ds << d_indent << "  for (int __mma_load_i = 0; __mma_load_i < "
         << layout.regs_per_thread << "; ++__mma_load_i) {\n";
      ds << d_indent << "    size_t __mma_load_logical = "
         << "((__mma_load_lane / 4) % 8 + (__mma_load_lane / 32) * 16 + "
            "((__mma_load_i / 2) % 2) * 8) * "
         << layout.logical_cols << " + (__mma_load_lane % 4) * 2 + "
            "(__mma_load_i % 2) + (__mma_load_i / 4) * 8;\n";
      if (source_swizzle != SwizMode::NONE) {
        int swizzle_bits = source_swizzle == SwizMode::B32 ? 1
                            : source_swizzle == SwizMode::B64 ? 2
                                                                : 3;
        size_t swizzle_mask = ((size_t{1} << swizzle_bits) - 1) << 7;
        ds << d_indent << "    size_t __mma_load_byte = __mma_load_logical * "
           << SizeOf(source_sty->ElementType()) << ";\n";
        ds << d_indent << "    " << sym
           << "[__mma_load_i] = __mma_load_src[(__mma_load_byte ^ "
           << "((__mma_load_byte & " << swizzle_mask << ") >> 3)) / "
           << SizeOf(source_sty->ElementType()) << "];\n";
      } else {
        ds << d_indent << "    " << sym
           << "[__mma_load_i] = __mma_load_src[__mma_load_logical];\n";
      }
      ds << d_indent << "  }\n";
      ds << d_indent << "}\n";
      ssm.MapDeviceSymbol(scoped, sym);
    } break;
    case AST::MMAOperation::LoadR: {
      auto ca = op.LoadFrom();
      auto f_sym = ca->data->name;
      auto ty = GetSymbolType(f_sym);
      auto f_sty = GetSpannedType(ty);
      std::string buf_expr = isa<FutureType>(ty) ? f_sym + ".data()" : f_sym;
      if (ca->indices != nullptr) {
        for (auto expr : ca->indices->AllValues())
          buf_expr += "[" + ExprSTR(expr, IsHost()) + "]";
      }
      const auto f_mds = GenTensorDecl(
          RemoveSuffix(f_sym, ".data()"), buf_expr, f_sty->GetStorage(),
          f_sty->ElementType(), ca->GetBlockShape(), false,
          ValueSTR(GenOffset(ca)), ValueSTR(GenStrides(ca), false, true));
      ds << f_mds.second;
      std::string dst_frag = AST::FragName(op.LoadTo());
      size_t elem_bytes = SizeOf(f_sty->ElementType());
      size_t regs_per_thread = 16 / elem_bytes;
      ds << d_indent << "{\n";
      ds << d_indent << "  auto* __loadr_src = reinterpret_cast<const "
         << NameBaseType(f_sty->ElementType()) << "*>(" << f_mds.first
         << ".data().get());\n";
      ds << d_indent << "  int __loadr_lane = threadIdx.x % 128;\n";
      ds << d_indent << "  #pragma unroll\n";
      ds << d_indent << "  for (int __loadr_i = 0; __loadr_i < "
         << regs_per_thread << "; ++__loadr_i)\n";
      ds << d_indent << "    " << dst_frag
         << "[__loadr_i] = __loadr_src[__loadr_lane * " << regs_per_thread
         << " + __loadr_i];\n";
      ds << d_indent << "}\n";
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
      pending_wgmma_acc_sym = ssm.DeviceName(c_sym);
      ds << d_indent << "warpgroup_fence_operand(" << pending_wgmma_acc_sym
         << ");\n";
      if (!(bdim_level == ParallelLevel::GROUPx4 && warpspec_wgmma_arrived)) {
        ds << d_indent << "warpgroup_arrive();\n";
        if (IsWarpSpecActive() && bdim_level == ParallelLevel::GROUPx4)
          warpspec_wgmma_arrived = true;
      }
      std::string mma_policy = FCtx(fname).MMAPolicyOfFrag(InScopeName(c_sym));
      std::string cc = SplitStringByDelimiter(mma_policy, "::")[0];
      auto cute_gmma_major_cast = "static_cast<cute::" + cc + "::GMMA::Major>";
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

      {
        ValueItem frag_len_c = ssmi_c.shape[1] / sbe::nu(2);
        if (ssmi_c.ty == BaseType::F16) frag_len_c = frag_len_c / sbe::nu(2);
        reg_num_d = *VIInt(frag_len_c);
      }

      if (op.HasScale() || use_explicit_scale_accum) {
        if (op.HasScale() && !use_hoisted_scale_accum) {
          ds << d_indent
             << NameBaseType(ssmi_c.ty == BaseType::F16 ? BaseType::U32
                                                        : ssmi_c.ty)
             << " " << c_sym << "_scale_frag[" << reg_num_d << "];\n";
          ds << d_indent << "memset(" << c_sym << "_scale_frag, 0, sizeof("
             << c_sym << "_scale_frag));\n";
        }
      }
      bool is_rs = FCtx(fname).FragIsRS(InScopeName(c_sym));
      bool a_is_chunk_rs = frag_chunk_rs_aliases_.count(a_sym) > 0;
      if (is_rs) {
        if (a_is_chunk_rs) {
          auto& ci = frag_chunk_rs_aliases_[a_sym];
          ds << d_indent << "warpgroup_fence_operand(" << ci.parent_c_sym
             << ");\n";
        } else {
          ds << d_indent << "warpgroup_fence_operand(" << a_sym << ");\n";
        }
      }
      struct DirectWGMMAOperandInfo {
        bool is_direct = false;
        bool supported = false;
        bool is_rs_direct = false;
        bool is_shared_direct = false;
        size_t k_iters = 1;
        size_t regs_per_step = 0;
        int lbo_override_bytes = 0;
        std::string desc_expr;
        std::string rs_base_expr;
        std::string shared_elem_ty;
        std::string shared_ptr_var;
        std::string shared_ptr_expr;
        std::string shared_desc_var;
        std::string shared_iter_ptr_var;
        std::string shared_iter_desc_var;
        std::string shared_major_order;
        std::string shared_swizzle_enum;
        std::string shared_iter_elem_offset_expr;
        std::string error;
      };
      auto swizzle_to_enum = [](SwizMode swizzle) {
        switch (swizzle) {
        case SwizMode::NONE: return std::string("WGMMA_Swizzle::NS");
        case SwizMode::B32: return std::string("WGMMA_Swizzle::B32");
        case SwizMode::B64: return std::string("WGMMA_Swizzle::B64");
        case SwizMode::B128: return std::string("WGMMA_Swizzle::B128");
        default: return std::string("WGMMA_Swizzle::NS");
        }
      };
      auto major_order_for = [](const MMAInfo& info) {
        if (info.frag == MMAInfo::FRAG_A) {
          return (info.method == AST::MMAOperation::ROW_ROW ||
                  info.method == AST::MMAOperation::ROW_COL)
                     ? std::string("WGMMA_MajorOrder::K_MAJOR")
                     : std::string("WGMMA_MajorOrder::MN_MAJOR");
        }
        if (info.frag == MMAInfo::FRAG_B) {
          return (info.method == AST::MMAOperation::ROW_ROW ||
                  info.method == AST::MMAOperation::COL_ROW)
                     ? std::string("WGMMA_MajorOrder::K_MAJOR")
                     : std::string("WGMMA_MajorOrder::MN_MAJOR");
        }
        return std::string("WGMMA_MajorOrder::MN_MAJOR");
      };
      static int direct_wgmma_desc_cnt = 0;
      auto build_direct_operand =
          [&](const ptr<AST::Expr>& operand_expr, const MMAInfo& operand_info,
              const std::string& iter_expr) -> DirectWGMMAOperandInfo {
        DirectWGMMAOperandInfo info;
        auto elemof_expr =
            operand_expr ? dyn_cast<AST::Expr>(operand_expr.get()) : nullptr;
        auto ca = operand_expr
                      ? dyn_cast<AST::ChunkAt>(operand_expr->GetReference())
                      : nullptr;
        auto sym = operand_expr ? operand_expr->GetSymbol() : nullptr;
        auto elemof_base = (elemof_expr && elemof_expr->op == Op::ElemOf)
                               ? AST::GetArrayBaseSymbol(*elemof_expr)
                               : nullptr;
        if (!ca && !sym && !elemof_base) return info;

        if (sym) {
          auto scoped = InScopeName(sym->name);
          if (explicit_mma_descs_.count(sym->name) ||
              explicit_mma_descs_.count(scoped))
            return info;
        }

        if (policy_is_sparse) {
          if (sym && !ca && !elemof_base) {
            if (auto sym_sty = GetSpannedType(GetSymbolType(sym->name));
                sym_sty && sym_sty->GetStorage() == Storage::REG)
              return info;
          }
          info.is_direct = true;
          info.error =
              "direct sparse WGMMA operands still require explicit mma.load";
          return info;
        }

        auto resolve_operand_spanned_type = [&]() -> ptr<SpannedType> {
          if (ca)
            if (auto ca_ty = GetSpannedType(ca->GetType())) return ca_ty;
          if (elemof_base)
            if (auto elem_ty = GetSpannedType(GetSymbolType(elemof_base->name)))
              return elem_ty;
          if (sym)
            if (auto sym_ty = GetSpannedType(GetSymbolType(sym->name)))
              return sym_ty;
          return GetSpannedType(operand_expr->GetType());
        };
        auto sty = resolve_operand_spanned_type();
        if (!sty) {
          info.is_direct = true;
          info.error = "failed to derive direct WGMMA operand type";
          return info;
        }

        auto atom_k = VIInt(operand_info.shape.at(2));
        if (!atom_k || *atom_k <= 0) {
          info.is_direct = true;
          info.error = "WGMMA atom K must be a compile-time constant";
          return info;
        }

        size_t k_axis = GetWGMMAKAxis(operand_info);
        auto shape = sty->GetShape();
        if (shape.Rank() <= k_axis) {
          info.is_direct = true;
          info.error = "direct WGMMA operand rank is too small";
          return info;
        }

        auto full_k = VIInt(shape.ValueAt(k_axis));
        if (!full_k || *full_k <= 0 || (*full_k % *atom_k) != 0) {
          info.is_direct = true;
          info.error = "direct WGMMA operand K dimension must be a static "
                       "multiple of the atom K";
          return info;
        }

        auto source_sym =
            ca ? ca->RefSymbol()
               : (elemof_base ? elemof_base->name : (sym ? sym->name : ""));
        auto source_scoped =
            source_sym.empty() ? source_sym : InScopeName(source_sym);

        // If source_sym is an alias (assigned via `=` from a ChunkAt), resolve
        // it through live_chunk_aliases to find the underlying root buffer.
        // This propagates swizzle from TMA-loaded buffers to reference aliases.
        if (!ca && !source_sym.empty() &&
            shared_buf_swiz_.find(source_sym) == shared_buf_swiz_.end() &&
            shared_buf_swiz_.find(source_scoped) == shared_buf_swiz_.end()) {
          auto ait = live_chunk_aliases.find(source_sym);
          if (ait == live_chunk_aliases.end())
            ait = live_chunk_aliases.find(source_scoped);
          if (ait != live_chunk_aliases.end()) {
            ca = ait->second;
            source_sym = ca->RefSymbol();
            source_scoped =
                source_sym.empty() ? source_sym : InScopeName(source_sym);
          }
        }
        bool is_chunk_rs_alias =
            !source_sym.empty() &&
            (frag_chunk_rs_aliases_.count(source_sym) > 0 ||
             frag_chunk_rs_aliases_.count(source_scoped) > 0);
        auto source_ty =
            source_sym.empty() ? nullptr : GetSymbolType(source_sym);

        if (sty->GetStorage() == Storage::REG) {
          if (is_chunk_rs_alias) return info;
          if (isa<FutureType>(source_ty)) return info;
          if (!(is_rs && operand_info.frag == MMAInfo::FRAG_A && sym)) {
            info.is_direct = true;
            info.error = "direct WGMMA register operands are only supported "
                         "for RS A operands";
            return info;
          }
          if (!FCtx(fname).HasFragmentLayout(source_scoped)) {
            info.is_direct = true;
            info.error = "missing fragment layout for direct RS WGMMA operand";
            return info;
          }
          auto& fl = FCtx(fname).GetFragmentLayout(source_scoped);
          if (fl.kind != LayoutKind::WGMMA_RS_A || fl.logical_cols == 0 ||
              fl.regs_per_thread == 0 || (fl.logical_cols % *atom_k) != 0) {
            info.is_direct = true;
            info.error =
                "unsupported fragment layout for direct RS WGMMA operand";
            return info;
          }
          size_t num_chunks = fl.logical_cols / *atom_k;
          if (num_chunks == 0 || (fl.regs_per_thread % num_chunks) != 0) {
            info.is_direct = true;
            info.error = "failed to derive register chunk layout for direct RS "
                         "WGMMA operand";
            return info;
          }
          info.is_direct = true;
          info.is_rs_direct = true;
          info.k_iters = static_cast<size_t>(*full_k / *atom_k);
          info.regs_per_step = fl.regs_per_thread / num_chunks;
          info.rs_base_expr = source_sym;
          info.supported = true;
          return info;
        }

        if (sty->GetStorage() != Storage::SHARED) {
          info.is_direct = true;
          info.error = "direct WGMMA operands must reside in shared memory";
          return info;
        }

        auto strides = ca ? GenStrides(ca) : sty->GetStrides();
        if (k_axis >= strides.size()) {
          info.is_direct = true;
          info.error = "failed to derive direct WGMMA operand stride";
          return info;
        }

        auto base_expr = [&]() {
          if (ca) return ValueSTR(TileAddr(ca, false));
          if (elemof_base) {
            std::vector<ptr<AST::Node>> subscripts;
            const AST::Expr* cur = elemof_expr;
            while (cur && cur->GetOp() == Op::ElemOf) {
              subscripts.push_back(cur->GetR());
              if (isa<AST::Identifier>(cur->GetL())) break;
              cur = dyn_cast<AST::Expr>(cur->GetL().get());
            }
            std::reverse(subscripts.begin(), subscripts.end());
            auto base_ty = GetSymbolType(elemof_base->name);
            if (auto array_ty = dyn_cast<ArrayType>(base_ty)) {
              return LinearizeArrayOffset(
                  ssm.DeviceName(InScopeName(elemof_base->name)), subscripts,
                  array_ty->Dimensions(), sty->GetShape().ElementCountValue(),
                  false);
            }
          }
          if (sym) {
            auto sym_ty = GetSymbolType(sym->name);
            if (isa<FutureType>(sym_ty)) return sym->name + ".data()";
          }
          return ExprSTR(operand_expr, false);
        }();
        auto elem_ty = NameBaseType(sty->ElementType());
        auto stride_expr = ValueSTR(strides.at(k_axis));
        SwizMode swizzle = SwizMode::NONE;
        if (auto it = shared_buf_swiz_.find(source_sym);
            it != shared_buf_swiz_.end())
          swizzle = it->second;

        bool is_mn_major =
            major_order_for(operand_info) == "WGMMA_MajorOrder::MN_MAJOR";
        size_t non_k_axis = (k_axis == 0) ? 1 : 0;

        int swiz_atom_cols = 0;
        if (swizzle != SwizMode::NONE) {
          int swiz_bytes = 0;
          switch (swizzle) {
          case SwizMode::B32: swiz_bytes = 32; break;
          case SwizMode::B64: swiz_bytes = 64; break;
          case SwizMode::B128: swiz_bytes = 128; break;
          default: break;
          }
          if (swiz_bytes > 0)
            swiz_atom_cols = swiz_bytes / (int)SizeOf(sty->ElementType());
        }

        // For swizzled layouts, recompute base_expr with physical offsets.
        // In a tiled-swizzled smem layout the non-K axis stride is atom_cols
        // (one swizzle period width), not the logical stride.
        auto phys_base_expr = [&]() -> std::string {
          if (ca && swiz_atom_cols > 0) {
            std::string raw_base;
            if (auto fty = dyn_cast<FutureType>(NodeType(*ca->data)))
              raw_base = std::string("(") + elem_ty + "*)" +
                         ExprSTR(ca->data, false) + ".data()";
            else
              raw_base = ExprSTR(ca->data, false);

            sbe::ExprSum phys;

            if (ca->indices != nullptr) {
              auto sym_ty = GetSymbolType(ca->data->name);
              if (auto array_ty = dyn_cast<ArrayType>(sym_ty)) {
                std::string array_idx;
                auto subscriptions = ca->indices->AllValues();
                const ValueList& array_sizes = array_ty->Dimensions();
                for (size_t i = 0; i < subscriptions.size(); ++i) {
                  if (array_idx.empty())
                    array_idx = ExprSTR(subscriptions[i], false);
                  else
                    array_idx = "(" + array_idx + ")*" +
                                ValueSTR(array_sizes[i]) + "+" +
                                ExprSTR(subscriptions[i], false);
                }
                auto f_sty = GetSpannedType(sym_ty);
                std::string elem_count =
                    ValueSTR(f_sty->GetShape().ElementCountValue());
                phys += sbe::sym("(" + array_idx + ")*(" + elem_count + ")");
              }
            }

            if (ca->NoOperation() && ValueSTR(phys.Get()) == "0")
              return raw_base;

            for (size_t i = 0; i < ca->OpCount(); ++i) {
              const auto& sop = ca->OpAt(i);
              if (isa<AST::SOP::Reshape>(sop)) continue;
              if (!(isa<AST::SOP::Tiling>(sop) || isa<AST::SOP::TileAt>(sop) ||
                    isa<AST::SOP::SubSpan>(sop)))
                continue;
              auto idx = sop->GetIndices()->Opts();
              auto strd = sop->GetBlockStrides();
              auto blk = sop->GetBlockShape();
              bool has_step = isa<AST::SOP::SubSpan>(sop) && sop->GetSteps();
              for (size_t ith = 0; ith < idx.GetVals().size(); ++ith) {
                ValueItem eff_strd =
                    (ith == non_k_axis) ? sbe::nu(swiz_atom_cols) : strd[ith];
                if (has_step)
                  phys += idx.GetVals()[ith] * eff_strd;
                else
                  phys += idx.GetVals()[ith] * eff_strd * blk.ValueAt(ith);
              }
            }
            auto off = ValueSTR(phys.Get());
            if (off == "0") return raw_base;
            return "(" + raw_base + " + " + off + ")";
          }
          return base_expr;
        }();

        // Build the iter-offset expression.
        // Logical formula:  k_iter * atom_k * stride
        // Physical formulas for swizzled layouts:
        //   K-major:  col = k_iter*atom_k;
        //             (col & (atom_cols-1)) +
        //             (col / atom_cols) * full_nonk * atom_cols
        //   MN-major: k_iter * atom_k * atom_cols
        size_t local_k_iters = static_cast<size_t>(*full_k / *atom_k);
        auto iter_offset = [&]() -> std::string {
          if (swiz_atom_cols > 0 && local_k_iters > 1) {
            if (is_mn_major) {
              int step = *atom_k * swiz_atom_cols;
              return "((" + iter_expr + ") * " + std::to_string(step) + ")";
            }
            // K-major: need the full non-K dimension of the parent tensor
            int full_nonk = 0;
            if (ca) {
              auto parent_sty = GetSpannedType(GetSymbolType(ca->RefSymbol()));
              if (parent_sty) {
                auto pv = VIInt(parent_sty->GetShape().ValueAt(non_k_axis));
                if (pv) full_nonk = *pv;
              }
            }
            if (full_nonk == 0) {
              auto sv = VIInt(shape.ValueAt(non_k_axis));
              if (sv) full_nonk = *sv;
            }
            if (full_nonk > 0) {
              int col_group_elems = full_nonk * swiz_atom_cols;
              return "(((" + iter_expr + ") * " + std::to_string(*atom_k) +
                     " & " + std::to_string(swiz_atom_cols - 1) + ") + ((" +
                     iter_expr + ") * " + std::to_string(*atom_k) + " / " +
                     std::to_string(swiz_atom_cols) + ") * " +
                     std::to_string(col_group_elems) + ")";
            }
          }
          return "((" + iter_expr + ") * " + std::to_string(*atom_k) + " * (" +
                 stride_expr + "))";
        }();

        // LBO override: when MN-major and the non-K dimension exceeds one
        // swizzle period, the leading-byte-offset must encode the stride
        // between column groups instead of the default 16-byte line.
        int lbo_override = 0;
        if (swiz_atom_cols > 0 && is_mn_major) {
          int full_k_dim = *full_k;
          auto non_k_val = VIInt(shape.ValueAt(non_k_axis));
          int non_k_dim = non_k_val ? *non_k_val : 0;
          if (non_k_dim > swiz_atom_cols)
            lbo_override =
                full_k_dim * swiz_atom_cols * (int)SizeOf(sty->ElementType());
        }

        info.is_direct = true;
        info.is_shared_direct = true;
        info.k_iters = static_cast<size_t>(*full_k / *atom_k);
        info.shared_elem_ty = elem_ty;
        info.shared_major_order = major_order_for(operand_info);
        info.shared_swizzle_enum = swizzle_to_enum(swizzle);
        info.shared_ptr_expr =
            std::string("(") + elem_ty + "*)(" + phys_base_expr + ")";
        info.shared_iter_elem_offset_expr = iter_offset;
        info.lbo_override_bytes = lbo_override;
        {
          int desc_id = direct_wgmma_desc_cnt++;
          auto suffix = std::to_string(desc_id);
          info.shared_ptr_var = "__choreo_wgmma_ptr_" + suffix;
          info.shared_desc_var = "__choreo_wgmma_desc_" + suffix;
          info.shared_iter_ptr_var = info.shared_ptr_var + "_iter";
          info.shared_iter_desc_var = info.shared_desc_var + "_iter";
        }
        info.desc_expr = info.shared_desc_var;
        info.supported = true;
        return info;
      };

      auto& ssmi_a = cgi.GetSymbolMMA(InScopeName(a_sym));
      auto& ssmi_b = cgi.GetSymbolMMA(InScopeName(b_sym));
      auto direct_a = build_direct_operand(op.ExecOperand(1), ssmi_a,
                                           "__choreo_wgmma_k_iter");
      auto direct_b = build_direct_operand(op.ExecOperand(2), ssmi_b,
                                           "__choreo_wgmma_k_iter");
      if ((direct_a.is_direct && !direct_a.supported) ||
          (direct_b.is_direct && !direct_b.supported)) {
        Error1(n.LOC(), (direct_a.is_direct && !direct_a.supported)
                            ? direct_a.error
                            : direct_b.error);
        return false;
      }

      size_t wgmma_k_iters = 1;
      if (direct_a.is_direct) wgmma_k_iters = direct_a.k_iters;
      if (direct_b.is_direct)
        wgmma_k_iters = std::max(wgmma_k_iters, direct_b.k_iters);
      if ((direct_a.is_direct && direct_a.k_iters != wgmma_k_iters) ||
          (direct_b.is_direct && direct_b.k_iters != wgmma_k_iters)) {
        Error1(n.LOC(), "direct WGMMA operands must agree on the number of "
                        "auto-split K iterations.");
        return false;
      }
      if (wgmma_k_iters > 1 &&
          !((direct_a.is_rs_direct && direct_b.is_direct) ||
            (direct_a.is_direct && direct_b.is_direct))) {
        Error1(n.LOC(), "auto-split WGMMA K iteration requires either direct "
                        "shared/shared operands or an RS direct-register A "
                        "with a direct shared-memory B operand.");
        return false;
      }

      auto emit_wgmma_desc_call = [&](std::ostream& os,
                                      const DirectWGMMAOperandInfo& info,
                                      const std::string& ptr_var) {
        os << "wgmma_make_smem_desc<" << info.shared_major_order << ", "
           << info.shared_swizzle_enum;
        if (info.lbo_override_bytes > 0) os << ", " << info.lbo_override_bytes;
        os << ">(" << ptr_var << ")";
      };
      auto emit_shared_direct_setup = [&](DirectWGMMAOperandInfo& info) {
        if (!info.is_shared_direct) return;
        if (wgmma_k_iters > 1) return;
        ds << d_indent << "auto* " << info.shared_ptr_var << " = "
           << info.shared_ptr_expr << ";\n";
        ds << d_indent << "uint64_t " << info.shared_desc_var << " = ";
        emit_wgmma_desc_call(ds, info, info.shared_ptr_var);
        ds << ";\n";
      };
      auto emit_shared_direct_iter_update =
          [&](const DirectWGMMAOperandInfo& info) {
            if (!info.is_shared_direct || wgmma_k_iters <= 1) return;
            ds << d_indent << "auto* " << info.shared_iter_ptr_var << " = ("
               << info.shared_elem_ty << "*)(" << info.shared_ptr_expr << " + "
               << info.shared_iter_elem_offset_expr << ");\n";
            ds << d_indent << "uint64_t " << info.shared_iter_desc_var << " = ";
            emit_wgmma_desc_call(ds, info, info.shared_iter_ptr_var);
            ds << ";\n";
          };
      auto desc_expr_for = [&](const DirectWGMMAOperandInfo& info,
                               const std::string& fallback) {
        if (!info.is_direct) return fallback;
        if (info.is_shared_direct && wgmma_k_iters > 1)
          return info.shared_iter_desc_var;
        return info.desc_expr;
      };

      emit_shared_direct_setup(direct_a);
      emit_shared_direct_setup(direct_b);

      auto emit_wgmma_fma = [&](const std::string& a_desc_expr,
                                const std::string& b_desc_expr) {
        ds << d_indent << "cute::" << mma_policy << "<";
        if (!policy_is_tn) {
          if (is_rs)
            ds << cute_gmma_major_cast << "(0), " << cute_gmma_major_cast << "("
               << trans_b << ")";
          else
            ds << cute_gmma_major_cast << "(" << trans_a << "), "
               << cute_gmma_major_cast << "(" << trans_b << ")";
        }
        ds << ">::fma(";
        if (is_rs) {
          std::string a_base;
          if (direct_a.is_rs_direct) {
            a_base = direct_a.rs_base_expr;
            if (direct_a.regs_per_step > 0 && wgmma_k_iters > 1)
              a_base += " + __choreo_wgmma_k_iter * " +
                        std::to_string(direct_a.regs_per_step);
          } else if (a_is_chunk_rs) {
            auto& ci = frag_chunk_rs_aliases_[a_sym];
            a_base = ci.parent_c_sym + " + " + ci.offset_var + " * " +
                     std::to_string(ci.regs_per_step);
          } else {
            a_base = a_sym;
          }
          ds << "reinterpret_cast<const uint32_t*>(" << a_base << ")[0], "
             << "reinterpret_cast<const uint32_t*>(" << a_base << ")[1], "
             << "reinterpret_cast<const uint32_t*>(" << a_base << ")[2], "
             << "reinterpret_cast<const uint32_t*>(" << a_base << ")[3], "
             << b_desc_expr;
        } else {
          ds << a_desc_expr << ", " << b_desc_expr;
        }
        for (size_t i = 0; i < reg_num_d; ++i) {
          if (use_explicit_scale_accum)
            ds << ", " << explicit_scale_info->scale_frag_name << "[" << i
               << "]";
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
      };

      if (wgmma_k_iters > 1) {
        ds << d_indent
           << "for (int __choreo_wgmma_k_iter = 0; __choreo_wgmma_k_iter < "
           << wgmma_k_iters << "; ++__choreo_wgmma_k_iter) {\n";
        IncrDeviceIndent();
        emit_shared_direct_iter_update(direct_a);
        emit_shared_direct_iter_update(direct_b);
      }
      emit_wgmma_fma(desc_expr_for(direct_a, "desc_" + a_sym),
                     desc_expr_for(direct_b, "desc_" + b_sym));
      if (wgmma_k_iters > 1) {
        DecrDeviceIndent();
        ds << d_indent << "}\n";
        if (has_pending_wgmma_finalize) EmitWGMMAFinalize(ds, d_indent);
      }

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
        if (!active_hoisted_scale_decls.count(scale_a_name)) {
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
      if (has_pending_wgmma_finalize) EmitWGMMAFinalize(ds, d_indent, true);
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
      bool explicit_m_mask = false;
      bool explicit_n_mask = false;
      std::string row_guard_expr;
      std::string col_guard_expr;

      if (op.StoreHasExplicitMask()) {
        need_m_mask = true;
        explicit_m_mask = true;
        row_guard_expr = ExprSTR(op.StoreRowMask(), IsHost());
        if (op.StoreColMask()) {
          need_n_mask = true;
          explicit_n_mask = true;
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
        bool target_is_shared = f_sty->GetStorage() == Storage::SHARED;
        if (!need_m_mask && f_sty->GetStorage() == Storage::GLOBAL &&
            !store_trans && !(use_stmatrix && target_is_shared)) {
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
            !(use_stmatrix && target_is_shared)) {
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

      // --- Emit store ---
      bool stmatrix_ok = use_stmatrix && f_sty->GetStorage() == Storage::SHARED;
      if (stmatrix_ok) {
        auto n_val = VIInt(ssmi.shape.at(1));
        if (!n_val || *n_val % 8 != 0) stmatrix_ok = false;
      }
      if (stmatrix_ok && (need_m_mask || need_n_mask)) stmatrix_ok = false;
      if (stmatrix_ok) {
        bool is_f32_acc = (accum_type == BaseType::F32);
        bool is_f32_dest = (f_sty->ElementType() == BaseType::F32);
        if (is_f32_acc && is_f32_dest) stmatrix_ok = false;
      }
      if (stmatrix_ok) {
        bool is_f32_acc = (accum_type == BaseType::F32);
        auto elem = f_sty->ElementType();
        bool is_bf16 = (elem == BaseType::BF16);
        bool is_f16 = (elem == BaseType::F16);
        std::string fn;
        if (is_f32_acc && is_bf16 && !store_trans)
          fn = "store_fragment_d_stmatrix_f32_bf16<";
        else if (is_f32_acc && is_bf16 && store_trans)
          fn = "store_fragment_d_stmatrix_trans_f32_bf16<";
        else if (is_f32_acc && is_f16 && !store_trans)
          fn = "store_fragment_d_stmatrix_f32_f16<";
        else if (store_trans)
          fn = "store_fragment_d_stmatrix_trans<";
        else
          fn = "store_fragment_d_stmatrix<";
        ds << d_indent << fn << CUTE_WGMMA_ATOM << ", " << DIM_N_STR << ">("
           << f_mds.first << ", "
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

        if (explicit_m_mask && explicit_n_mask) {
          ds << d_indent << "store_fragment_d_mask_row_col<" << CUTE_WGMMA_ATOM
             << ", " << DIM_N_STR << ">(" << f_mds.first << ", " << frag_cast
             << ", " << row_guard_expr << ", " << col_guard_expr << ");\n";
        } else if (explicit_m_mask) {
          ds << d_indent << "store_fragment_d_mask_row<" << CUTE_WGMMA_ATOM
             << ", " << DIM_N_STR << ">(" << f_mds.first << ", " << frag_cast
             << ", " << row_guard_expr << ");\n";
        } else if (explicit_n_mask) {
          ds << d_indent << "store_fragment_d_mask_col<" << CUTE_WGMMA_ATOM
             << ", " << DIM_N_STR << ">(" << f_mds.first << ", " << frag_cast
             << ", " << col_guard_expr << ");\n";
        } else if (need_m_mask && need_n_mask) {
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
      if (f_sty->GetStorage() == Storage::SHARED) {
        if (NeedWarpSpecGroupX4SyncForCurrentScope())
          EmitGroupX4Sync(ds, d_indent);
        else
          ds << d_indent << "__syncthreads();\n";
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
                      GetBaseType(*op.FillingValue()->GetType()), false,
                      IsNumericZeroLiteral(op.FillingValue()));
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
              auto mdata_key = InScopeName(mdata_sym_name) + ".data";
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
    case AST::MMAOperation::LoadR:
      Error1(n.LOC(), "mma.loadR is only supported for WGMMA.");
      return false;
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
      bool sync_explicit_m = false, sync_explicit_n = false;
      std::string sync_rg, sync_cg;

      if (op.StoreHasExplicitMask()) {
        sync_need_m = true;
        sync_explicit_m = true;
        sync_rg = ExprSTR(op.StoreRowMask(), IsHost());
        if (op.StoreColMask()) {
          sync_need_n = true;
          sync_explicit_n = true;
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
      if (sync_explicit_m && sync_explicit_n) {
        ds << d_indent << "store_fragment_d_mask_row_col<" << CUTE_MMA_ATOM
           << ">(" << f_mds.first << ", " << frag_cast_sync << ", " << sync_rg
           << ", " << sync_cg << ");\n";
      } else if (sync_explicit_m) {
        ds << d_indent << "store_fragment_d_mask_row<" << CUTE_MMA_ATOM << ">("
           << f_mds.first << ", " << frag_cast_sync << ", " << sync_rg
           << ");\n";
      } else if (sync_explicit_n) {
        ds << d_indent << "store_fragment_d_mask_col<" << CUTE_MMA_ATOM << ">("
           << f_mds.first << ", " << frag_cast_sync << ", " << sync_cg
           << ");\n";
      } else if (sync_need_m && sync_need_n) {
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
      auto cp_atom_name = GetCopyAtomName(false, dma_count_);
      ds << d_indent << "AsyncCopyAtom " << cp_atom_name << "{};\n";
      ds << d_indent << "future " << ident->name << "(\"" << ident->name
         << "\", " << id->LOC().begin.line << ", " << id->LOC().begin.column
         << ");\n";
      ds << d_indent << ident->name << ".set_atom(&" << cp_atom_name << ");\n";
      ds << d_indent << ident->name << ".set_ring(" << device_fn
         << "__ring__);\n";
      future_count_++;
      ds << d_indent << ident->name << ".id = " << future_count_ << ";\n";
      ++dma_count_;
      claimed_futs.emplace(scoped_name, cp_atom_name);
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
      bool is_block_shared = IsFutureBlockShared(InScopeName(name));
      bool is_cooperative = cooperatives.count(InScopeName(name));
      bool is_tma = tma_futures_.count(InScopeName(name)) > 0;
      bool single_thread_wait = is_block_shared && !is_cooperative && !is_tma;
      if (single_thread_wait) {
        ds << d_indent << LevelPred() << " {\n";
        IncrDeviceIndent();
      }
      assert(!IsHost());
      ds << d_indent << ExprSTR(t, false) << ".wait();\n";
      if (single_thread_wait) {
        DecrDeviceIndent();
        ds << d_indent << "}\n";
      }
      if (is_block_shared) {
        if (NeedWarpSpecGroupX4SyncForCurrentScope())
          EmitGroupX4Sync(ds, d_indent);
        else
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
        if (ScopeAlreadySingleThreadForLevel(bdim_level)) {
          Error1(n.LOC(), "shared event wait (" + PSTR(t) +
                              ") is inside a single-thread scope; "
                              "mbarrier wait must not be predicated to a "
                              "single thread.");
        }
        std::string base_name;
        if (is_array_ref) {
          auto bid = AST::GetArrayBaseSymbol(*expr);
          base_name = UnScopedName(bid->name);
        } else {
          base_name = UnScopedName(expr->GetSymbol()->name);
        }
        bool is_cluster_event = cluster_trigger_events_.count(base_name) > 0;
        waited_events_.insert(base_name);

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
          {
            auto dims_cl = ety->Dimensions();
            int stages_cl = 2;
            if (!dims_cl.empty()) {
              if (auto nv = dyn_cast<sbe::NumericValue>(dims_cl[0].get()))
                stages_cl = static_cast<int>(nv->Value());
            }
            // Cluster-scoped events always have priming performed (never
            // suppressed), so the phase after init+priming is already
            // flipped.  Use is_fill=true to avoid the ^1 inversion that
            // non-cluster empty events need (where priming is suppressed).
            std::string phase_cl = InlinePhaseExpr(stages_cl, /*is_fill=*/true);
            if (is_array_ref) {
              std::string bar_expr = ExprSTR(t, false);
              ds << d_indent << "choreo::tma_mbarrier_wait_parity((uint64_t*)&"
                 << bar_expr << ", " << phase_cl << ");\n";
            } else {
              ds << d_indent << "choreo::tma_mbarrier_wait_parity((uint64_t*)&"
                 << ExprSTR(t, false) << ", " << phase_cl << ");\n";
            }
          }
          if (cluster_wait_guarded) {
            DecrDeviceIndent();
            ds << d_indent << "}\n";
          }
        } else {
          ds << d_indent << "// wait event(raw mbarrier) " << PSTR(t) << "\n";
          auto& ft = cgi.GetFunctionTrait(fname);
          bool is_fill = ft.IsTMAFillEvent(base_name);
          auto dims = ety->Dimensions();
          int stages = 2;
          if (!dims.empty()) {
            if (auto nv = dyn_cast<sbe::NumericValue>(dims[0].get()))
              stages = static_cast<int>(nv->Value());
          }
          std::string phase_expr = InlinePhaseExpr(stages, is_fill);
          if (is_array_ref && !foreach_iv_stack_.empty()) {
            auto subscript = cast<AST::Expr>(expr->GetR());
            if (subscript && subscript->IsTernary() &&
                subscript->op == Op::Select) {
              const auto& iv = foreach_iv_stack_.back();
              if (stages == 2) {
                phase_expr = is_fill ? ("((" + iv + " == 0) ? 0 : (((" + iv +
                                        " - 1) & 3) >> 1))")
                                     : ("((" + iv + " == 0) ? 1 : ((((" + iv +
                                        " - 1) & 3) >> 1) ^ 1))");
              } else {
                std::string s = std::to_string(stages);
                std::string prev_base = "((" + iv + " - 1) / " + s + ") & 1";
                phase_expr =
                    is_fill ? ("((" + iv + " == 0) ? 0 : (" + prev_base + "))")
                            : ("((" + iv + " == 0) ? 1 : ((" + prev_base +
                               ") ^ 1))");
              }
            }
          }
          if (is_array_ref) {
            std::string bar_expr = ExprSTR(t, false);
            ds << d_indent << bar_expr << ".wait(" << phase_expr << ");\n";
          } else {
            ds << d_indent << ExprSTR(t, false) << ".wait(" << phase_expr
               << ");\n";
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
        if (ScopeAlreadySingleThreadForLevel(bdim_level)) {
          Error1(n.LOC(), "shared event wait (" + PSTR(t) +
                              ") is inside a single-thread scope; "
                              "mbarrier wait must not be predicated to a "
                              "single thread.");
        }
        std::string base_name;
        if (is_array_ref) {
          auto bid = AST::GetArrayBaseSymbol(*expr);
          base_name = UnScopedName(bid->name);
        } else {
          base_name = UnScopedName(expr->GetSymbol()->name);
        }
        bool is_cluster_event = cluster_trigger_events_.count(base_name) > 0;
        waited_events_.insert(base_name);

        if (is_cluster_event) {
          bool cluster_wait_guarded =
              IsWarpSpecActive() && bdim_level == ParallelLevel::GROUPx4 &&
              !ScopeAlreadySingleThreadForLevel(ParallelLevel::GROUPx4);
          if (cluster_wait_guarded) {
            ds << d_indent << "if (__CHOREO_GROUPX4_SINGLE__) {\n";
            IncrDeviceIndent();
          }
          {
            auto& ft_sc = cgi.GetFunctionTrait(fname);
            bool is_fill_sc = ft_sc.IsTMAFillEvent(base_name);
            std::string phase_sc = is_fill_sc ? InlinePhaseExpr(1, true) : "0";
            if (is_array_ref) {
              std::string bar_expr = ExprSTR(t, false);
              ds << d_indent << "choreo::tma_mbarrier_wait_parity((uint64_t*)&"
                 << bar_expr << ", " << phase_sc << ");\n";
            } else {
              ds << d_indent << "choreo::tma_mbarrier_wait_parity((uint64_t*)&"
                 << ExprSTR(t, false) << ", " << phase_sc << ");\n";
            }
          }
          if (cluster_wait_guarded) {
            DecrDeviceIndent();
            ds << d_indent << "}\n";
          }
        } else {
          auto bar_name = ExprSTR(t, false);
          auto& ft_single = cgi.GetFunctionTrait(fname);
          bool is_fill_single = ft_single.IsTMAFillEvent(base_name);
          std::string phase_single =
              is_fill_single ? InlinePhaseExpr(1, true) : "0";
          ds << d_indent << bar_name << ".wait(" << phase_single
             << "); // wait event(raw mbarrier)\n";
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
          std::string bar_base;
          if (is_array_ref)
            bar_base = AST::GetArrayBaseSymbol(*expr)->name;
          else if (expr->GetSymbol())
            bar_base = expr->GetSymbol()->name;

          ds << d_indent << "// trigger event(barrier) " << PSTR(f);
          if (is_cluster_trigger) ds << " [cluster-scope]";
          ds << "\n";

          if (tma_bound_event_triggers_.count(bar_base)) {
            recent_tma_tx_bytes.clear();
          } else if (is_cluster_trigger) {
            ds << d_indent << "if (__CHOREO_GROUPX4_SINGLE__) {\n";
            IncrDeviceIndent();
            ds << d_indent
               << "for (uint32_t __cta = 0; __cta < "
                  "choreo::tma_cluster_dim(); ++__cta) {\n";
            IncrDeviceIndent();
            if (is_array_ref) {
              ds << d_indent
                 << "choreo::tma_mbarrier_arrive_cluster((uint64_t*)&"
                 << ExprSTR(f, false) << ", __cta);\n";
            } else {
              GenerateSubscriptions(
                  ds,
                  d_indent +
                      "choreo::tma_mbarrier_arrive_cluster((uint64_t*)&" +
                      ExprSTR(f, false),
                  ", __cta);\n", ety->RemainderDimensions(0));
            }
            DecrDeviceIndent();
            ds << d_indent << "}\n";
            DecrDeviceIndent();
            ds << d_indent << "}\n";
          } else if (!recent_tma_tx_bytes.empty()) {
            auto tx_bytes_expr = SumRecentTMATxBytesExpr();
            bool conditional_tx =
                IsWarpSpecActive() &&
                !ScopeAlreadySingleThreadForLevel(ParallelLevel::GROUPx4);
            if (is_array_ref) {
              if (conditional_tx) {
                ds << d_indent << ExprSTR(f, false)
                   << ".arrive_and_expect_tx(__CHOREO_GROUPX4_SINGLE__ ? "
                   << tx_bytes_expr << " : 0);\n";
              } else {
                ds << d_indent << ExprSTR(f, false) << ".arrive_and_expect_tx("
                   << tx_bytes_expr << ");\n";
              }
            } else {
              if (conditional_tx) {
                GenerateSubscriptions(
                    ds, d_indent + ExprSTR(f, false),
                    ".arrive_and_expect_tx(__CHOREO_GROUPX4_SINGLE__ ? " +
                        tx_bytes_expr + " : 0);\n",
                    ety->RemainderDimensions(0));
              } else {
                GenerateSubscriptions(ds, d_indent + ExprSTR(f, false),
                                      ".arrive_and_expect_tx(" + tx_bytes_expr +
                                          ");\n",
                                      ety->RemainderDimensions(0));
              }
            }
            recent_tma_tx_bytes.clear();
          } else {
            bool suppress = IsWarpSpecActive() &&
                            empty_event_names_.count(bar_base) &&
                            waited_events_.empty();
            if (suppress) {
              ds << d_indent << "// (priming suppressed for " << PSTR(f)
                 << " -- phase init=1 handles it)\n";
            } else if (is_array_ref) {
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
          std::string evt_bar_name;
          if (expr->GetSymbol()) evt_bar_name = expr->GetSymbol()->name;

          if (tma_bound_event_triggers_.count(evt_bar_name)) {
            ds << d_indent << "// trigger event(barrier) " << PSTR(f)
               << " (no-op: arrive_and_expect_tx at TMA load)\n";
            recent_tma_tx_bytes.clear();
          } else if (is_cluster_trigger) {
            ds << d_indent << "if (__CHOREO_GROUPX4_SINGLE__) {\n";
            IncrDeviceIndent();
            ds << d_indent
               << "for (uint32_t __cta = 0; __cta < "
                  "choreo::tma_cluster_dim(); ++__cta) {\n";
            IncrDeviceIndent();
            ds << d_indent << "choreo::tma_mbarrier_arrive_cluster((uint64_t*)&"
               << ExprSTR(f, false) << ", __cta);\n";
            DecrDeviceIndent();
            ds << d_indent << "}\n";
            DecrDeviceIndent();
            ds << d_indent << "}\n";
          } else if (!recent_tma_tx_bytes.empty()) {
            auto tx_bytes_expr = SumRecentTMATxBytesExpr();
            bool conditional_tx =
                IsWarpSpecActive() &&
                !ScopeAlreadySingleThreadForLevel(ParallelLevel::GROUPx4);
            if (conditional_tx) {
              ds << d_indent << ExprSTR(f, false)
                 << ".arrive_and_expect_tx(__CHOREO_GROUPX4_SINGLE__ ? "
                 << tx_bytes_expr << " : 0); // trigger event(barrier)\n";
            } else {
              ds << d_indent << ExprSTR(f, false) << ".arrive_and_expect_tx("
                 << tx_bytes_expr << "); // trigger event(barrier)\n";
            }
            recent_tma_tx_bytes.clear();
          } else {
            bool suppress = IsWarpSpecActive() &&
                            empty_event_names_.count(evt_bar_name) &&
                            waited_events_.empty();
            if (suppress) {
              ds << d_indent << "// (priming suppressed for " << PSTR(f)
                 << " -- phase init=1 handles it)\n";
            } else {
              ds << d_indent << "(void)" << ExprSTR(f, false)
                 << ".arrive(); // trigger event(barrier)\n";
            }
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
    } else if (func_name == "croq::cuda::setreg_inc" ||
               func_name == "croq::cuda::setreg_dec") {
      if (IsHost()) return true;
      auto arg = cast<AST::Expr>(n.arguments->ValueAt(0));
      auto reg_limit = VIInt(arg->Opts().GetVal());
      if (!reg_limit || reg_limit.value() <= 0)
        choreo_unreachable(
            "croq::cuda::setreg_inc/dec expects a positive integer constant.");
      if (CCtx().ArchNum() >= 90) {
        const char* dir =
            (func_name == "croq::cuda::setreg_inc") ? "inc" : "dec";
        os << indent << "asm volatile(\"setmaxnreg." << dir
           << ".sync.aligned.u32 " << reg_limit.value() << ";\");\n";
      }
      return true;
    } else if (func_name == "__bar_arrive" || func_name == "__bar_sync") {
      if (IsHost()) return true;
      if (n.arguments->Count() != 2)
        choreo_unreachable(func_name +
                           " expects exactly 2 arguments (barrier_id, count).");
      auto id_str = ExprSTR(n.GetArguments().at(0), false);
      auto count_str = ExprSTR(n.GetArguments().at(1), false);
      if (func_name == "__bar_arrive") {
        os << indent << "asm volatile(\"bar.arrive %0, %1;\" :: \"r\"("
           << id_str << "), \"r\"(" << count_str << ") : \"memory\");\n";
      } else {
        os << indent << "asm volatile(\"bar.sync %0, %1;\" :: \"r\"(" << id_str
           << "), \"r\"(" << count_str << ") : \"memory\");\n";
      }
      return true;
    } else if (func_name == "print" || func_name == "println") {
      if (n.CompileTimeEval()) return true;
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
    } else if (n.IsAtomic()) {
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
    if (isa<StreamType>(ty)) continue;
    SSTab().DefineSymbol(param->sym->name, ty);
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

  // When the range source (n.in) is a 1D mdspan referencing a declared
  // variable (e.g. `{bn} in [kv_bound]`), record the variable name as the
  // preferred loop bound for that IV. ForeachBlock codegen consults this
  // map to emit the variable name instead of inlining the full expression.
  if (!IsHost() && n.with_matchers && n.GetMatchers().size() == 1) {
    std::string var_name;
    if (auto expr = dyn_cast<AST::Expr>(n.in)) {
      if (auto ref = expr->GetReference()) {
        if (auto mds = dyn_cast<AST::MultiDimSpans>(ref)) {
          if (auto mv = dyn_cast<AST::MultiValues>(mds->list)) {
            if (mv->AllValues().size() == 1) {
              if (auto inner = AST::GetIdentifier(mv->AllValues()[0])) {
                auto sn = InScopeNameForRef(inner->name);
                if (ssm.HasDeviceName(sn))
                  var_name = UnScopedName(ssm.DeviceName(sn));
              }
            }
          }
        }
      }
      if (var_name.empty()) {
        if (auto sym = expr->GetSymbol()) {
          auto sn = InScopeNameForRef(sym->name);
          if (ssm.HasDeviceName(sn))
            var_name = UnScopedName(ssm.DeviceName(sn));
        }
      }
    }
    if (!var_name.empty()) {
      auto id = cast<AST::Identifier>(n.GetMatchers()[0]);
      auto iv_scoped = InScopeName(id->name);
      for (auto& iv_name : within_map.at(iv_scoped))
        iv_upper_bound_expr_[iv_name] = var_name;
    }
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

// True when the foreach's DIRECT body (not nested foreachs) has MMA Exec
// but no MMA Commit.  Stops at nested foreach boundaries so hoisting
// targets the innermost foreach containing WGMMA.
// Optionally extracts the accumulator symbol name from the first Exec found.
static bool ForeachHasMMAExecWithoutCommit(const ptr<AST::Node>& node,
                                           std::string* acc_sym = nullptr) {
  if (!node) return false;
  bool has_exec = false;
  bool has_commit = false;
  std::unordered_set<std::string> fill_syms;
  auto walk = [&](auto&& self, const ptr<AST::Node>& n) -> void {
    if (!n) return;
    if (auto mma = dyn_cast<AST::MMA>(n)) {
      if (auto op = mma->GetOperation()) {
        if (op->Tag() == AST::MMAOperation::Exec) {
          has_exec = true;
          if (acc_sym && acc_sym->empty())
            *acc_sym = AST::FragName(op->ExecOperand(0));
        }
        if (op->Tag() == AST::MMAOperation::Fill)
          fill_syms.insert(AST::FragName(op->FillingTo()));
        if (op->Tag() == AST::MMAOperation::Commit) has_commit = true;
      }
      return;
    }
    if (dyn_cast<AST::ForeachBlock>(n)) return;
    if (auto mn = dyn_cast<AST::MultiNodes>(n)) {
      for (auto& item : mn->values) self(self, item);
      return;
    }
    if (auto ie = dyn_cast<AST::IfElseBlock>(n)) {
      self(self, ie->GetBody());
      if (ie->HasElse()) self(self, ie->GetElseBody());
      return;
    }
    if (auto block = dyn_cast<AST::Block>(n)) {
      self(self, block->GetBody());
      return;
    }
  };
  walk(walk, node);
  if (acc_sym && !acc_sym->empty() && fill_syms.count(*acc_sym)) return false;
  return has_exec && !has_commit;
}

// Classify an automap body: can we emit a register-direct unrolled loop?
//
// REGISTER_DIRECT: all fragment accesses share compatible layout and use
//   the same iteration variables in order as the foreach ranges.
// REGISTER_WITH_BROADCAST: 2D fragments share layout, 1D fragments are
//   indexed by only the row variable (cross-fragment broadcast).
// FLAT_STRIDE: fallback -- use existing flat index decomposition loop.
AutomapStrategy CuteCodeGen::AnalyzeAutomap(const AST::ForeachBlock& n) {
  const auto& ranges = n.GetRangeNodes();
  size_t num_ranges = ranges->Count();

  std::vector<std::string> iv_names;
  iv_names.reserve(num_ranges);
  for (size_t ri = 0; ri < num_ranges; ++ri) {
    auto rng = cast<AST::LoopRange>(ranges->ValueAt(ri));
    iv_names.push_back(InScopeName(rng->GetIVName()));
  }

  struct FragAccess {
    std::string scoped_name;
    FragmentLayout layout;
    std::vector<std::string> index_iv_names;
    size_t num_indices = 0;
  };
  std::vector<FragAccess> accesses;

  auto CollectAccesses = [&](auto&& self, const ptr<AST::Node>& node) -> bool {
    if (!node) return true;
    if (auto da = dyn_cast<AST::DataAccess>(node)) {
      if (da->AccessElement()) {
        auto scoped = InScopeNameForRef(da->data->name);
        auto sty = GetSpannedType(GetSymbolType(da->data->name));
        if (sty && sty->GetStorage() == Storage::REG &&
            FCtx(fname).HasFragmentLayout(scoped)) {
          FragAccess fa;
          fa.scoped_name = scoped;
          fa.layout = FCtx(fname).GetFragmentLayout(scoped);
          fa.num_indices = da->GetIndices().size();
          for (auto& idx : da->GetIndices()) {
            if (auto id = AST::GetIdentifier(idx)) {
              auto idx_scoped = InScopeNameForRef(id->name);
              fa.index_iv_names.push_back(idx_scoped);
            } else {
              return false;
            }
          }
          accesses.push_back(std::move(fa));
        }
      }
      return true;
    }
    if (auto assign = dyn_cast<AST::Assignment>(node)) {
      if (!self(self, assign->da)) return false;
      if (!self(self, assign->value)) return false;
      return true;
    }
    if (auto expr = dyn_cast<AST::Expr>(node)) {
      if (expr->GetL() && !self(self, expr->GetL())) return false;
      if (expr->GetR() && !self(self, expr->GetR())) return false;
      if (expr->GetC() && !self(self, expr->GetC())) return false;
      return true;
    }
    if (auto mn = dyn_cast<AST::MultiNodes>(node)) {
      for (auto& item : mn->values)
        if (!self(self, item)) return false;
      return true;
    }
    if (auto call = dyn_cast<AST::Call>(node)) {
      if (call->arguments)
        for (auto& a : call->GetArguments())
          if (!self(self, a)) return false;
      return true;
    }
    return true;
  };

  if (!CollectAccesses(CollectAccesses, n.GetBody()))
    return AutomapStrategy::FLAT_STRIDE;
  if (accesses.empty()) return AutomapStrategy::FLAT_STRIDE;

  bool has_2d = false, has_1d = false;
  const FragmentLayout* primary_2d = nullptr;

  for (auto& fa : accesses) {
    bool is_2d = (fa.layout.logical_cols > 1);
    if (is_2d) {
      has_2d = true;
      if (!primary_2d) primary_2d = &fa.layout;
    } else {
      has_1d = true;
    }
  }

  // Case A: all fragment accesses have compatible layouts, same number of
  // indices as foreach ranges, and indices match the iteration variables
  // in order.
  auto IsDirectMatch = [&](const FragAccess& fa) -> bool {
    if (fa.num_indices != num_ranges) return false;
    for (size_t k = 0; k < num_ranges; ++k)
      if (fa.index_iv_names[k] != iv_names[k]) return false;
    return true;
  };

  auto AllCompatible = [&]() -> bool {
    for (size_t i = 1; i < accesses.size(); ++i)
      if (!accesses[0].layout.IsCompatible(accesses[i].layout)) return false;
    return true;
  };

  bool all_direct = true;
  for (auto& fa : accesses)
    if (!IsDirectMatch(fa)) {
      all_direct = false;
      break;
    }

  if (all_direct && AllCompatible()) return AutomapStrategy::REGISTER_DIRECT;

  // Case B: 2D fragments with matching MMA layout + 1D fragments indexed
  // by only the row variable (first foreach iv). Only MMA layouts have a
  // clean RowFromRegIndex that is independent of tid.
  if (has_2d && has_1d && num_ranges == 2 && primary_2d &&
      primary_2d->IsMMAAnchored()) {
    bool ok = true;
    for (auto& fa : accesses) {
      bool is_2d = (fa.layout.logical_cols > 1);
      if (is_2d) {
        if (!IsDirectMatch(fa)) {
          ok = false;
          break;
        }
        if (primary_2d && !fa.layout.IsCompatible(*primary_2d)) {
          ok = false;
          break;
        }
      } else {
        if (fa.num_indices != 1) {
          ok = false;
          break;
        }
        if (fa.index_iv_names[0] != iv_names[0]) {
          ok = false;
          break;
        }
      }
    }
    if (ok) return AutomapStrategy::REGISTER_WITH_BROADCAST;
  }

  return AutomapStrategy::FLAT_STRIDE;
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
    auto cname = rng->GetIVName();
    loop_refs.push_back(cname);
    loop_refs.push_back(std::string("__iv_") + cname);
    for (auto iv_name : within_map.at(InScopeName(cname)))
      loop_refs.push_back(SSMName(iv_name, false));
  }

  if (!IsHost() && !n.GetRanges().empty()) {
    auto rng0 = cast<AST::LoopRange>(n.GetRanges()[0]);
    auto ivs0 = within_map.at(InScopeName(rng0->GetIVName()));
    if (!ivs0.empty())
      foreach_iv_stack_.push_back(SSMName(ivs0.front(), false));
    else
      foreach_iv_stack_.push_back("__iv_" + rng0->GetIVName());
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

      if (hoisted_scale_accum_scopes.back().has_value() && invariant_to_loop &&
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

  int unroll_factor = 0;
  const bool has_unroll = AST::HasUnrollHint(n, unroll_factor);

  {
    std::string hoisted_acc;
    if (!IsHost() && bdim_level == ParallelLevel::GROUPx4 &&
        !warpspec_wgmma_arrived &&
        ForeachHasMMAExecWithoutCommit(n.GetBody(), &hoisted_acc)) {
      if (!hoisted_acc.empty())
        ds << d_indent << "warpgroup_fence_operand("
           << ssm.DeviceName(hoisted_acc) << ");\n";
      ds << d_indent << "warpgroup_arrive();\n";
      warpspec_wgmma_arrived = true;
    }
  }

  if (has_unroll) {
    if (unroll_factor > 0)
      IndStream() << "#pragma unroll " << unroll_factor << "\n";
    else
      IndStream() << "#pragma unroll\n";
  }

  ptr<AST::AttributeExpr> automap_attr;
  const bool has_automap = !IsHost() && AST::HasAutomapHint(n, automap_attr);
  if (has_automap) {
    AutomapStrategy strategy = AnalyzeAutomap(n);

    if (strategy == AutomapStrategy::REGISTER_DIRECT ||
        strategy == AutomapStrategy::REGISTER_WITH_BROADCAST) {
      // Find the primary 2D (or only) fragment to get regs_per_thread.
      const auto& ranges = n.GetRangeNodes();
      std::string primary_scoped;
      size_t regs = 0;
      const FragmentLayout* primary_layout = nullptr;

      auto FindPrimary = [&](auto&& self, const ptr<AST::Node>& node) -> void {
        if (!node) return;
        if (auto da = dyn_cast<AST::DataAccess>(node)) {
          if (da->AccessElement() && primary_scoped.empty()) {
            auto sc = InScopeNameForRef(da->data->name);
            if (FCtx(fname).HasFragmentLayout(sc)) {
              auto& fl = FCtx(fname).GetFragmentLayout(sc);
              if (fl.logical_cols > 1 || !primary_layout) {
                primary_scoped = sc;
                primary_layout = &fl;
                regs = fl.regs_per_thread;
              }
            }
          }
          return;
        }
        if (auto assign = dyn_cast<AST::Assignment>(node)) {
          self(self, assign->da);
          self(self, assign->value);
          return;
        }
        if (auto expr = dyn_cast<AST::Expr>(node)) {
          if (expr->GetL()) self(self, expr->GetL());
          if (expr->GetR()) self(self, expr->GetR());
          if (expr->GetC()) self(self, expr->GetC());
          return;
        }
        if (auto mn = dyn_cast<AST::MultiNodes>(node)) {
          for (auto& item : mn->values) self(self, item);
          return;
        }
      };
      FindPrimary(FindPrimary, n.GetBody());

      reg_loop_var_ = "__r";
      in_register_direct_automap_ = true;
      automap_frag_reg_expr_.clear();

      // Build register expression overrides for each fragment access.
      auto BuildOverrides = [&](auto&& self,
                                const ptr<AST::Node>& node) -> void {
        if (!node) return;
        if (auto da = dyn_cast<AST::DataAccess>(node)) {
          if (da->AccessElement()) {
            auto sc = InScopeNameForRef(da->data->name);
            if (FCtx(fname).HasFragmentLayout(sc)) {
              auto& fl = FCtx(fname).GetFragmentLayout(sc);
              auto sym = UnScopedName(SSMName(sc, false));
              if (strategy == AutomapStrategy::REGISTER_DIRECT) {
                automap_frag_reg_expr_[sc] = sym + "[" + reg_loop_var_ + "]";
              } else {
                // REGISTER_WITH_BROADCAST
                if (fl.logical_cols > 1) {
                  automap_frag_reg_expr_[sc] = sym + "[" + reg_loop_var_ + "]";
                } else {
                  automap_frag_reg_expr_[sc] =
                      sym + "[" +
                      primary_layout->RowFromRegIndex(reg_loop_var_) + "]";
                }
              }
            }
          }
          return;
        }
        if (auto assign = dyn_cast<AST::Assignment>(node)) {
          self(self, assign->da);
          self(self, assign->value);
          return;
        }
        if (auto expr = dyn_cast<AST::Expr>(node)) {
          if (expr->GetL()) self(self, expr->GetL());
          if (expr->GetR()) self(self, expr->GetR());
          if (expr->GetC()) self(self, expr->GetC());
          return;
        }
        if (auto mn = dyn_cast<AST::MultiNodes>(node)) {
          for (auto& item : mn->values) self(self, item);
          return;
        }
      };
      BuildOverrides(BuildOverrides, n.GetBody());

      // Detect vectorized copy pattern: a single assignment where one
      // side is a fragment and the other is contiguous memory, and the
      // layout supports vec4 grouping.
      bool emit_vec4 = false;
      std::string vec4_dst, vec4_src;
      std::string vec4_mem_base;
      bool vec4_mem_is_lhs = false;
      if (primary_layout && primary_layout->IsVectorizable() &&
          strategy == AutomapStrategy::REGISTER_DIRECT) {
        auto body = n.GetBody();
        ptr<AST::Assignment> single_assign;
        // Unwrap nested MultiNodes to find a single assignment.
        auto unwrap = body;
        while (unwrap && unwrap->Count() == 1) {
          if (auto inner = dyn_cast<AST::MultiNodes>(unwrap->SubAt(0)))
            unwrap = inner;
          else
            break;
        }
        if (unwrap && unwrap->Count() == 1)
          single_assign = dyn_cast<AST::Assignment>(unwrap->SubAt(0));
        if (single_assign && single_assign->da->AccessElement()) {
          auto lhs_scoped = InScopeNameForRef(single_assign->da->data->name);
          bool lhs_is_frag = automap_frag_reg_expr_.count(lhs_scoped);
          auto rhs_da = dyn_cast<AST::DataAccess>(single_assign->value);
          if (!rhs_da) {
            if (auto rhs_expr = dyn_cast<AST::Expr>(single_assign->value))
              rhs_da = dyn_cast<AST::DataAccess>(rhs_expr->GetR());
          }
          bool rhs_is_frag = false;
          std::string rhs_scoped;
          if (rhs_da && rhs_da->AccessElement()) {
            rhs_scoped = InScopeNameForRef(rhs_da->data->name);
            rhs_is_frag = automap_frag_reg_expr_.count(rhs_scoped);
          }
          // frag = mem or mem = frag
          if (lhs_is_frag != rhs_is_frag && rhs_da) {
            std::string frag_sym =
                lhs_is_frag ? UnScopedName(SSMName(lhs_scoped, false))
                            : UnScopedName(SSMName(rhs_scoped, false));
            std::string mem_name = lhs_is_frag ? rhs_da->data->name
                                               : single_assign->da->data->name;
            auto mem_sty = GetSpannedType(GetSymbolType(mem_name));
            if (mem_sty && mem_sty->GetStorage() != Storage::REG) {
              std::string tid_expr = ResolveAutomapThreadExpr(automap_attr);
              std::string TW = std::to_string(primary_layout->thread_count *
                                              primary_layout->vec_width);
              std::string W = std::to_string(primary_layout->vec_width);
              vec4_mem_base = std::string("((") +
                              NameBaseType(mem_sty->ElementType()) + "*)" +
                              OpExprSTR(lhs_is_frag ? rhs_da->data
                                                    : single_assign->da->data,
                                        "+", true, false) +
                              ")";
              emit_vec4 = true;
              if (lhs_is_frag) {
                vec4_dst = frag_sym;
                vec4_src = vec4_mem_base;
                vec4_mem_is_lhs = false;
              } else {
                vec4_dst = vec4_mem_base;
                vec4_src = frag_sym;
                vec4_mem_is_lhs = true;
              }
            }
          }
        }
      }

      if (emit_vec4) {
        std::string tid_expr = ResolveAutomapThreadExpr(automap_attr);
        size_t groups = regs / primary_layout->vec_width;
        std::string W = std::to_string(primary_layout->vec_width);
        std::string TW = std::to_string(primary_layout->thread_count *
                                        primary_layout->vec_width);
        std::string r4 = "__r4";

        in_register_direct_automap_ = false;
        automap_frag_reg_expr_.clear();
        reg_loop_var_.clear();

        IndStream() << "#pragma unroll\n";
        IndStream() << "for (int " << r4 << " = 0; " << r4 << " < " << groups
                    << "; ++" << r4 << ") {\n";
        IncrIndent();

        std::string mem_offset = r4 + " * " + TW + " + " + tid_expr + " * " + W;
        std::string frag_offset = r4 + " * " + W;

        if (vec4_mem_is_lhs) {
          ds << d_indent << "*(float4*)(" << vec4_dst << " + " << mem_offset
             << ") = *(float4*)(" << vec4_src << " + " << frag_offset << ");\n";
        } else {
          ds << d_indent << "*(float4*)(" << vec4_dst << " + " << frag_offset
             << ") = *(float4*)(" << vec4_src << " + " << mem_offset << ");\n";
        }

        DecrIndent();
        IndStream() << "} // vec4 automap\n";
        vec4_automap_skip_ = true;
      } else {
        IndStream() << "#pragma unroll\n";
        IndStream() << "for (int " << reg_loop_var_ << " = 0; " << reg_loop_var_
                    << " < " << regs << "; ++" << reg_loop_var_ << ") {\n";
        IncrIndent();

        // Reconstruct iteration variables from register index + tid so that
        // non-fragment expressions (casts, arithmetic, non-fragment array
        // accesses) can use them.
        if (primary_layout) {
          std::string tid_expr = ResolveAutomapThreadExpr(automap_attr);
          size_t num_ranges = ranges->Count();
          if (num_ranges == 1) {
            auto rng = cast<AST::LoopRange>(ranges->ValueAt(0));
            for (auto iv_name : within_map.at(InScopeName(rng->GetIVName())))
              ds << d_indent << ssm.DeviceName(iv_name) << " = "
                 << primary_layout->LogicalRowFromReg(reg_loop_var_, tid_expr)
                 << ";\n";
          } else if (num_ranges == 2) {
            auto rng0 = cast<AST::LoopRange>(ranges->ValueAt(0));
            auto rng1 = cast<AST::LoopRange>(ranges->ValueAt(1));
            for (auto iv_name : within_map.at(InScopeName(rng0->GetIVName())))
              ds << d_indent << ssm.DeviceName(iv_name) << " = "
                 << primary_layout->LogicalRowFromReg(reg_loop_var_, tid_expr)
                 << ";\n";
            for (auto iv_name : within_map.at(InScopeName(rng1->GetIVName())))
              ds << d_indent << ssm.DeviceName(iv_name) << " = "
                 << primary_layout->LogicalColFromReg(reg_loop_var_, tid_expr)
                 << ";\n";
          }
        }
      }
    } else {
      // FLAT_STRIDE: existing decomposition logic
      const auto& ranges = n.GetRangeNodes();
      std::ostringstream total_expr;
      std::vector<std::string> bound_exprs;
      bound_exprs.reserve(ranges->Count());
      for (size_t ri = 0; ri < ranges->Count(); ++ri) {
        auto rng = cast<AST::LoopRange>(ranges->ValueAt(ri));
        auto iv_ty = GetSymbolType(rng->GetIVName());
        auto bit = cast<BoundedType>(iv_ty);
        std::string bound = UnScopedExpr(ValueSTR(bit->GetUpperBound()));
        bound_exprs.push_back(bound);
        if (ri > 0) total_expr << " * ";
        total_expr << bound;
      }
      std::string thread_expr = ResolveAutomapThreadExpr(automap_attr);
      std::string stride = current_thread_count_expr.empty()
                               ? std::string("blockDim.x")
                               : current_thread_count_expr;
      IndStream() << "for (int __automap_flat = " << thread_expr
                  << "; __automap_flat < " << total_expr.str()
                  << "; __automap_flat += " << stride << ") {\n";
      IncrIndent();
      std::vector<std::string> iv_value_exprs(ranges->Count());
      std::string quotient = "__automap_flat";
      for (int ri = (int)ranges->Count() - 1; ri >= 0; --ri) {
        iv_value_exprs[(size_t)ri] = quotient + " % " + bound_exprs[(size_t)ri];
        if (ri > 0) {
          std::string next_q = "__automap_q" + std::to_string(ri);
          ds << d_indent << "auto " << next_q << " = " << quotient << " / "
             << bound_exprs[(size_t)ri] << ";\n";
          quotient = next_q;
        }
      }
      for (size_t ri = 0; ri < ranges->Count(); ++ri) {
        auto rng = cast<AST::LoopRange>(ranges->ValueAt(ri));
        for (auto iv_name : within_map.at(InScopeName(rng->GetIVName())))
          ds << d_indent << ssm.DeviceName(iv_name) << " = "
             << iv_value_exprs[ri] << ";\n";
      }
    }
  } else {
    {
      for (auto& rn : n.GetRanges()) {
        auto rng = cast<AST::LoopRange>(rn);
        auto cname = rng->GetIVName();
        for (auto iv_name : within_map.at(InScopeName(cname))) {
          auto iv_ty = GetSymbolType(UnScopedName(iv_name));
          assert(IsActualBoundedIntegerType(iv_ty));
          auto iv_bty = cast<BoundedType>(iv_ty);
          auto ub_it = iv_upper_bound_expr_.find(iv_name);
          std::string ub_expr =
              ub_it != iv_upper_bound_expr_.end()
                  ? ub_it->second
                  : UnScopedExpr(ValueSTR(iv_bty->GetUpperBound()));
          IndStream() << "for (" << SSMName(iv_name, IsHost()) << " = "
                      << (rng->lbound
                              ? ("(" + ExprSTR(rng->lbound, IsHost()) + ")")
                              : "0")
                      << "; " << SSMName(iv_name, IsHost()) << " < " << ub_expr
                      << (rng->ubound ? (" + " + ExprSTR(rng->ubound, IsHost()))
                                      : "")
                      << "; ++" << SSMName(iv_name, IsHost()) << ") {\n";
          IncrIndent();
        }
      }
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

bool CuteCodeGen::Visit(AST::ApplyBlock& n) {
  TraceEachVisit(n);
  if (IsHost()) return true;

  // Post-order: body has been emitted; close the register loop.
  if (apply_has_main_loop_) {
    DecrIndent();
    IndStream() << "}\n";
  }
  DecrIndent();
  IndStream() << "} // apply " << n.SpanFragmentName() << "\n";

  in_register_direct_automap_ = false;
  automap_frag_reg_expr_.clear();
  reg_loop_var_.clear();
  frag_apply_iv_map_.clear();
  apply_row_hoisted_stmts_.clear();
  apply_has_main_loop_ = true;
  return true;
}

bool CuteCodeGen::Visit(AST::FragTransfer& n) {
  TraceEachVisit(n);
  if (IsHost()) return true;
  if (has_pending_wgmma_finalize) EmitWGMMAFinalize(ds, d_indent, true);

  auto dst_name = n.DstName();
  auto src_name = n.SrcName();
  auto dst_scoped = InScopeNameForRef(dst_name);
  auto src_scoped = InScopeNameForRef(src_name);

  auto dst_sty = GetSpannedType(GetSymbolType(dst_name));
  auto src_sty = GetSpannedType(GetSymbolType(src_name));

  bool dst_is_frag = dst_sty && dst_sty->GetStorage() == Storage::REG &&
                     FCtx(fname).HasFragmentInfo(dst_scoped);
  bool src_is_frag = src_sty && src_sty->GetStorage() == Storage::REG &&
                     FCtx(fname).HasFragmentInfo(src_scoped);

  std::string dst_sym = UnScopedName(SSMName(dst_scoped, false));
  std::string src_sym = UnScopedName(SSMName(src_scoped, false));

  std::string tid_expr;
  if (bdim_level == ParallelLevel::GROUPx4)
    tid_expr = "(threadIdx.x % 128)";
  else if (!current_thread_count_expr.empty())
    tid_expr = "__choreo_vtid_x";
  else
    tid_expr = "threadIdx.x";

  const std::string frag_name =
      (n.op == AST::FragTransferKind::STORE) ? src_name : dst_name;
  auto frag_scoped = InScopeNameForRef(frag_name);
  if (!FCtx(fname).HasFragmentLayout(frag_scoped)) {
    dbgs() << n.OpName() << ": no layout for " << frag_scoped << "\n";
    return true;
  }
  auto& fl = FCtx(fname).GetFragmentLayout(frag_scoped);
  size_t regs = fl.regs_per_thread;

  IndStream() << "{ // " << n.OpName() << " " << dst_name << ", " << src_name
              << "\n";
  IncrIndent();

  if (n.HasLambda()) {
    reg_loop_var_ = "__r";
    in_register_direct_automap_ = true;
    automap_frag_reg_expr_.clear();

    if (dst_is_frag)
      automap_frag_reg_expr_[dst_scoped] = dst_sym + "[" + reg_loop_var_ + "]";
    if (src_is_frag)
      automap_frag_reg_expr_[src_scoped] = src_sym + "[" + reg_loop_var_ + "]";

    for (auto& p : n.params) {
      auto sp = SSTab().ScopedName(p);
      if (!SSTab().IsDeclared(sp)) SSTab().DefineSymbol(sp, MakeIntegerType());
      if (!SymTab()->Exists(sp)) SymTab()->AddSymbol(sp, MakeIntegerType());
      ds << d_indent << "int __frag_iv_" << p << " = 0;\n";
      frag_apply_iv_map_[p] = "__frag_iv_" + p;
    }

    IndStream() << "#pragma unroll\n";
    IndStream() << "for (int " << reg_loop_var_ << " = 0; " << reg_loop_var_
                << " < " << regs << "; ++" << reg_loop_var_ << ") {\n";
    IncrIndent();

    for (size_t pi = 0; pi < n.params.size(); ++pi) {
      std::string iv_var = "__frag_iv_" + n.params[pi];
      if (n.params.size() == 1)
        ds << d_indent << iv_var << " = "
           << fl.LogicalFlatFromReg(reg_loop_var_, tid_expr) << ";\n";
      else if (pi == 0)
        ds << d_indent << iv_var << " = "
           << fl.LogicalRowFromReg(reg_loop_var_, tid_expr) << ";\n";
      else
        ds << d_indent << iv_var << " = "
           << fl.LogicalColFromReg(reg_loop_var_, tid_expr) << ";\n";
    }

    if (n.op == AST::FragTransferKind::STORE) {
      auto mem_scoped = dst_scoped;
      auto mem_sym = dst_sym;
      std::string idx_expr;
      if (n.params.size() == 1)
        idx_expr = frag_apply_iv_map_[n.params[0]];
      else {
        auto cols_str =
            std::to_string(fl.logical_cols > 0 ? fl.logical_cols : 1);
        idx_expr = frag_apply_iv_map_[n.params[0]] + " * " + cols_str + " + " +
                   frag_apply_iv_map_[n.params[1]];
      }
      ds << d_indent << mem_sym << "[" << idx_expr
         << "] = " << ExprSTR(n.body, false) << ";\n";
    } else if (n.op == AST::FragTransferKind::LOAD) {
      auto frag_sym = dst_sym;
      ds << d_indent << frag_sym << "[" << reg_loop_var_
         << "] = " << ExprSTR(n.body, false) << ";\n";
    } else {
      auto frag_sym = dst_sym;
      ds << d_indent << frag_sym << "[" << reg_loop_var_
         << "] = " << ExprSTR(n.body, false) << ";\n";
    }

    DecrIndent();
    IndStream() << "}\n";

    in_register_direct_automap_ = false;
    automap_frag_reg_expr_.clear();
    reg_loop_var_.clear();
    frag_apply_iv_map_.clear();
  } else {
    // Plain transfer (no lambda)
    if (n.op == AST::FragTransferKind::COPY) {
      IndStream() << "#pragma unroll\n";
      IndStream() << "for (int __r = 0; __r < " << regs << "; ++__r)\n";
      IncrIndent();
      IndStream() << dst_sym << "[__r] = " << src_sym << "[__r];\n";
      DecrIndent();
    } else {
      bool can_vec4 = fl.IsVectorizable() && fl.vec_width >= 4;
      bool is_store = (n.op == AST::FragTransferKind::STORE);
      auto& frag_sym_ref = is_store ? src_sym : dst_sym;
      auto& mem_sym_ref = is_store ? dst_sym : src_sym;
      size_t T = fl.thread_count;

      if (can_vec4) {
        size_t groups = regs / fl.vec_width;
        std::string W = std::to_string(fl.vec_width);
        std::string TW = std::to_string(T * fl.vec_width);

        IndStream() << "#pragma unroll\n";
        IndStream() << "for (int __r4 = 0; __r4 < " << groups
                    << "; ++__r4) {\n";
        IncrIndent();
        if (is_store) {
          IndStream() << "*(float4*)(" << mem_sym_ref << " + __r4 * " << TW
                      << " + " << tid_expr << " * " << W << ") = *(float4*)("
                      << frag_sym_ref << " + __r4 * " << W << ");\n";
        } else {
          IndStream() << "*(float4*)(" << frag_sym_ref << " + __r4 * " << W
                      << ") = *(float4*)(" << mem_sym_ref << " + __r4 * " << TW
                      << " + " << tid_expr << " * " << W << ");\n";
        }
        DecrIndent();
        IndStream() << "}\n";
      } else {
        IndStream() << "#pragma unroll\n";
        IndStream() << "for (int __r = 0; __r < " << regs << "; ++__r) {\n";
        IncrIndent();
        std::string mem_idx = fl.LogicalFlatFromReg("__r", tid_expr);
        if (is_store) {
          IndStream() << mem_sym_ref << "[" << mem_idx << "] = " << frag_sym_ref
                      << "[__r];\n";
        } else {
          IndStream() << frag_sym_ref << "[__r] = " << mem_sym_ref << "["
                      << mem_idx << "];\n";
        }
        DecrIndent();
        IndStream() << "}\n";
      }
    }
  }

  DecrIndent();
  IndStream() << "} // " << n.OpName() << "\n";

  return true;
}

bool CuteCodeGen::Visit(AST::FragReduce& n) {
  TraceEachVisit(n);
  if (IsHost()) return true;
  if (has_pending_wgmma_finalize) EmitWGMMAFinalize(ds, d_indent, true);

  // ---- Prepare: resolve symbols, compute all parameters ----

  auto src_scoped = InScopeNameForRef(n.SrcName());
  auto dst_scoped = InScopeNameForRef(n.DstName());

  if (!FCtx(fname).HasFragmentLayout(src_scoped) ||
      !FCtx(fname).HasFragmentLayout(dst_scoped)) {
    dbgs() << "frag.reduce: missing layout for " << src_scoped << " or "
           << dst_scoped << "\n";
    return true;
  }

  auto& src_fl = FCtx(fname).GetFragmentLayout(src_scoped);
  auto src_sym = UnScopedName(SSMName(src_scoped, false));
  auto dst_sym = UnScopedName(SSMName(dst_scoped, false));

  auto rp = src_fl.GetReduceParams();
  bool is_max = (n.op == AST::FragReduceOp::MAX);

  std::string identity =
      is_max ? "__int_as_float(0xff800000) /*-inf*/" : "0.0f";
  std::string reduce_type = is_max ? "choreo::MaxOp" : "choreo::SumOp";
  std::string idx_expr =
      src_fl.ReduceLocalIndex("__row", "__rv", rp.local_cols);
  std::string ws_name =
      rp.NeedsWorkspace() ? ("__reduce_ws_" + n.DstName()) : "";

  // ---- Emit: write CUDA code from prepared parameters ----

  IndStream() << "{ // " << n.OpName() << " " << n.SrcName() << " -> "
              << n.DstName() << "\n";
  IncrIndent();

  if (rp.NeedsWorkspace()) {
    IndStream() << "__shared__ float " << ws_name << "[" << rp.thread_count
                << "];\n";
  }

  // Phase 1: Thread-local reduction over owned column registers.
  IndStream() << "#pragma unroll\n";
  IndStream() << "for (int __row = 0; __row < " << rp.rows_per_thread
              << "; ++__row) {\n";
  IncrIndent();

  IndStream() << "float __local_reduce = " << identity << ";\n";
  IndStream() << "#pragma unroll\n";
  IndStream() << "for (int __rv = 0; __rv < " << rp.local_cols
              << "; ++__rv) {\n";
  IncrIndent();
  if (is_max) {
    IndStream() << "__local_reduce = fmaxf(__local_reduce, " << src_sym << "["
                << idx_expr << "]);\n";
  } else {
    IndStream() << "__local_reduce += " << src_sym << "[" << idx_expr << "];\n";
  }
  DecrIndent();
  IndStream() << "}\n";

  // Phase 2: Cross-thread butterfly AllReduce.
  if (rp.threads_per_row > 1) {
    if (!rp.NeedsWorkspace()) {
      // Inline the butterfly shuffles directly to avoid function boundaries
      // that can cause ptxas WGMMA serialization (C7510/C7514).
      size_t offset = rp.threads_per_row / 2;
      int round = 0;
      while (offset >= 1) {
        std::string vname = "__shfl_v" + std::to_string(round);
        IndStream() << "{\n";
        IncrIndent();
        IndStream() << "float " << vname
                    << " = __shfl_xor_sync(0xffffffff, __local_reduce, "
                    << offset << ");\n";
        if (is_max) {
          IndStream() << "__local_reduce = fmaxf(__local_reduce, " << vname
                      << ");\n";
        } else {
          IndStream() << "__local_reduce = __local_reduce + " << vname << ";\n";
        }
        DecrIndent();
        IndStream() << "}\n";
        if (offset == 1) break;
        offset /= 2;
        round++;
      }
    } else {
      IndStream() << "__local_reduce = choreo::AllReduce<" << reduce_type
                  << ", " << rp.threads_per_row << ", 1>::run(__local_reduce";
      ds << ", " << ws_name;
      ds << ");\n";
    }
  }

  IndStream() << dst_sym << "[__row] = __local_reduce;\n";
  DecrIndent();
  IndStream() << "}\n";

  DecrIndent();
  IndStream() << "} // " << n.OpName() << "\n";

  return true;
}

bool CuteCodeGen::Visit(AST::InThreadsBlock& n) {
  TraceEachVisit(n);
  assert(!IsHost());
  PushEmittedNames();
  ds << d_indent << "// inthreads: " << n.LOC() << "\n";
  current_inthreads = &n;
  warpspec_wgmma_arrived = false;
  waited_events_.clear();
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

  PushEmittedNames();
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

  // Use [[maybe_unused]] to suppress NVCC warnings for virtual indices
  // that may not be referenced in every kernel.
  const char* mu = "[[maybe_unused]] ";

  switch (pb->GetLevel()) {
  case ParallelLevel::GROUPx4: {
    assert(pb->AllSubPVs().size() > 0);
    std::string g4id = vid_pfx + "g4id";
    std::string vid_x = vid_pfx + "g4id_x";
    std::string vid_y = vid_pfx + "g4id_y";
    std::string vid_z = vid_pfx + "g4id_z";
    if (pb->AllSubPVs().size() == 1) {
      ds << d_indent << mu << "auto " << vid_x << " = threadIdx.x / 128;\n";
    } else if (pb->AllSubPVs().size() == 2) {
      ds << d_indent << mu << "auto " << g4id << " = threadIdx.x / 128;\n";
      ds << d_indent << mu << "auto " << vid_x << " = " << g4id << " / "
         << ValueSTR(pv_y) << ";\n";
      ds << d_indent << mu << "auto " << vid_y << " = " << g4id << " % "
         << ValueSTR(pv_y) << ";\n";
    } else if (pb->AllSubPVs().size() == 3) {
      ds << d_indent << mu << "auto " << g4id << " = threadIdx.x / 128;\n";
      ds << d_indent << mu << "auto " << vid_x << " = " << g4id << " / "
         << ValueSTR(pv_y) << " / " << ValueSTR(pv_z) << ";\n";
      ds << d_indent << mu << "auto " << vid_y << " = " << g4id << " / "
         << ValueSTR(pv_z) << " % " << ValueSTR(pv_y) << ";\n";
      ds << d_indent << mu << "auto " << vid_z << " = " << g4id << " % "
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
      ds << d_indent << mu << "auto " << vid_x << " = threadIdx.x / 32;\n";
    } else if (pb->AllSubPVs().size() == 2) {
      ds << d_indent << mu << "auto " << gid << " = threadIdx.x / 32;\n";
      ds << d_indent << mu << "auto " << vid_x << " = " << gid << " / "
         << ValueSTR(pv_y) << ";\n";
      ds << d_indent << mu << "auto " << vid_y << " = " << gid << " % "
         << ValueSTR(pv_y) << ";\n";
    } else if (pb->AllSubPVs().size() == 3) {
      ds << d_indent << mu << "auto " << gid << " = threadIdx.x / 32;\n";
      ds << d_indent << mu << "auto " << vid_x << " = " << gid << " / "
         << ValueSTR(pv_y) << " / " << ValueSTR(pv_z) << ";\n";
      ds << d_indent << mu << "auto " << vid_y << " = " << gid << " / "
         << ValueSTR(pv_z) << " % " << ValueSTR(pv_y) << ";\n";
      ds << d_indent << mu << "auto " << vid_z << " = " << gid << " % "
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
        ds << d_indent << mu << "auto " << vid_x << " = threadIdx.x % 128;\n";
      else if (bdim_level == ParallelLevel::GROUP)
        ds << d_indent << mu << "auto " << vid_x << " = threadIdx.x % 32;\n";
      else if (bdim_level == ParallelLevel::THREAD)
        ds << d_indent << mu << "auto " << vid_x << " = threadIdx.x;\n";
      else
        choreo_unreachable("invalid bdim level.");
    } else if (pb->AllSubPVs().size() == 2) {
      if (bdim_level == ParallelLevel::GROUPx4)
        ds << d_indent << mu << "auto " << tid << " = threadIdx.x % 128;\n";
      else if (bdim_level == ParallelLevel::GROUP)
        ds << d_indent << mu << "auto " << tid << " = threadIdx.x % 32;\n";
      else if (bdim_level == ParallelLevel::THREAD)
        ds << d_indent << mu << "auto " << tid << " = threadIdx.x;\n";
      else
        choreo_unreachable("invalid bdim level.");

      ds << d_indent << mu << "auto " << vid_x << " = " << tid << " / "
         << ValueSTR(pv_y) << ";\n";
      ds << d_indent << mu << "auto " << vid_y << " = " << tid << " % "
         << ValueSTR(pv_y) << ";\n";
    } else if (pb->AllSubPVs().size() == 3) {
      if (bdim_level == ParallelLevel::GROUPx4)
        ds << d_indent << mu << "auto " << tid << " = threadIdx.x % 128;\n";
      else if (bdim_level == ParallelLevel::GROUP)
        ds << d_indent << mu << "auto " << tid << " = threadIdx.x % 32;\n";
      else if (bdim_level == ParallelLevel::THREAD)
        ds << d_indent << mu << "auto " << tid << " = threadIdx.x;\n";
      else
        choreo_unreachable("invalid bdim level.");

      ds << d_indent << mu << "auto " << vid_x << " = " << tid << " / "
         << ValueSTR(pv_y) << " / " << ValueSTR(pv_z) << ";\n";
      ds << d_indent << mu << "auto " << vid_y << " = " << tid << " / "
         << ValueSTR(pv_z) << " % " << ValueSTR(pv_y) << ";\n";
      ds << d_indent << mu << "auto " << vid_z << " = " << tid << " % "
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
    size_t align = ie.alignment ? ie.alignment : 512;
    if (ie.n_buffers > 0 && !ie.interference.empty()) {
      // Emit pre-computed interference matrix (incorporates SALA HB analysis)
      std::string imat_name = ie.chunks_name + "_imat";
      hs << h_indent << "std::vector<bool> " << imat_name << " = {";
      for (size_t k = 0; k < ie.interference.size(); ++k) {
        if (k) hs << ",";
        hs << (ie.interference[k] ? "true" : "false");
      }
      hs << "};\n";
      hs << h_indent << "HeapSimulator::Result " << ie.result << " = "
         << ie.simulator << ".Allocate(" << ie.chunks_name << ", " << align
         << ", " << imat_name << ");\n";
    } else {
      hs << h_indent << "HeapSimulator::Result " << ie.result << " = "
         << ie.simulator << ".Allocate(" << ie.chunks_name << ", " << align
         << ");\n";
    }
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

static const char* CuTensorMapL2PromotionExpr(int l2_promote_bytes) {
  switch (l2_promote_bytes) {
  case 0: return "CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_128B";
  case 64: return "CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_64B";
  case 128: return "CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_128B";
  case 256: return "CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_256B";
  default: return "CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_128B";
  }
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

    std::string g_unscoped = UnScopedName(g_sym);

    // Determine whether this tensor is a true GLOBAL argument.
    bool is_global_arg = false;
    bool found_param = false;
    for (const auto& item : GetChoreoFuncIns(cgi)) {
      if (UnScopedName(item.name) == g_unscoped) {
        found_param = true;
        is_global_arg = (item.attr == ParamAttr::GLOBAL_INPUT);
        break;
      }
    }

    std::string base_expr;
    Shape host_shape = g_shape;
    Shape host_stride = g_stride;
    size_t host_rank = g_shape.Rank();

    if (desc.HasRootParam()) {
      // TMA source is a computed subview (e.g., Q_head derived from Q).
      // Flatten the root parameter's N-D shape to 2D [outer, inner]
      // so the TMA descriptor covers the entire root tensor.
      //
      // Use the subview's outermost stride as inner_dim rather than the
      // root's last dimension.  When squeezed dimensions sit between the
      // tiled and innermost dimensions (e.g. Q[B,SEQ,H,DIM] -> Q_head[SEQ,DIM]
      // with stride [H*DIM,1]), the row stride is H*DIM, not DIM.
      auto root_name = UnScopedName(desc.GetRootParamName());
      const auto& root_shape = desc.GetRootParamShape();

      ValueItem inner_dim = g_stride[0];
      ValueItem total_elems = sbe::nu(1);
      for (size_t i = 0; i < root_shape.Rank(); ++i)
        total_elems = total_elems * root_shape.ValueAt(i);
      ValueItem outer_dim = total_elems / inner_dim;

      host_shape = Shape({outer_dim, inner_dim});
      host_stride = Shape({inner_dim * gmem_ty->ElementSizeValue(),
                           gmem_ty->ElementSizeValue()});
      host_rank = 2;
      base_expr = root_name + ".data()";
    } else if (found_param && is_global_arg) {
      base_expr = g_unscoped + ".data()";
      // Check if a .view() operation flattens to a lower rank
      bool has_view = false;
      for (auto& sop : g_ca->AllOperations()) {
        if (isa<AST::SOP::View>(sop)) {
          has_view = true;
          break;
        }
      }
      if (has_view && host_rank > 2) {
        ValueItem inner_dim = g_shape.ValueAt(g_shape.Rank() - 1);
        ValueItem outer_dim = sbe::nu(1);
        for (size_t i = 0; i + 1 < g_shape.Rank(); ++i)
          outer_dim = outer_dim * g_shape.ValueAt(i);
        host_shape = Shape({outer_dim, inner_dim});
        host_stride = Shape({inner_dim * gmem_ty->ElementSizeValue(),
                             gmem_ty->ElementSizeValue()});
        host_rank = 2;
      } else {
        std::vector<ValueItem> byte_strides;
        for (size_t i = 0; i < host_stride.Rank(); ++i)
          byte_strides.push_back(host_stride.ValueAt(i) *
                                 gmem_ty->ElementSizeValue());
        host_stride = Shape(byte_strides);
      }
    } else if (found_param) {
      base_expr = SSMName((g_unscoped + "__device"), true);
      std::vector<ValueItem> byte_strides;
      for (size_t i = 0; i < host_stride.Rank(); ++i)
        byte_strides.push_back(host_stride.ValueAt(i) *
                               gmem_ty->ElementSizeValue());
      host_stride = Shape(byte_strides);
    } else if (IsChoreoOutput(g_sym)) {
      base_expr = SSMName(InScopeName(g_unscoped) + "__device", true);
      std::vector<ValueItem> byte_strides;
      for (size_t i = 0; i < host_stride.Rank(); ++i)
        byte_strides.push_back(host_stride.ValueAt(i) *
                               gmem_ty->ElementSizeValue());
      host_stride = Shape(byte_strides);
    } else {
      std::ostringstream oss;
      oss << g_ca->LOC();
      choreo_unreachable(
          "TMA source '" + g_unscoped +
          "' is not a function parameter and could not be traced to one. "
          "TMA descriptors require a host-accessible base pointer. At " +
          oss.str());
    }

    // When TMA swizzle is enabled and the box inner dimension (in bytes)
    // exceeds the swizzle width, CUDA requires boxDim[0]*elemSize <=
    // swizzle_bytes.  Fix by splitting the innermost dimension into groups
    // of swiz_elems and using a higher-rank TMA descriptor (e.g. 2D->3D).
    //
    // Dimension ordering: the groups dimension is placed OUTERMOST so that
    // TMA fills all rows of group 0 before group 1.  This produces an
    // atom-major shared-memory layout (matching WGMMA B128 expectations):
    //   [outer, inner] -> [groups, outer, swiz_elems]
    // After Reverse for CUDA: [swiz_elems, outer, groups]
    int swiz_bytes = 0;
    switch (tma_swizzle) {
    case TMA_Swizzle::B32: swiz_bytes = 32; break;
    case TMA_Swizzle::B64: swiz_bytes = 64; break;
    case TMA_Swizzle::B128: swiz_bytes = 128; break;
    default: break;
    }
    size_t elem_size = SizeOf(gmem_ty->ElementType());
    Shape box_shape = t_shape;
    if (swiz_bytes > 0 && t_shape.Rank() >= 1) {
      auto inner_vi = t_shape.ValueAt(t_shape.Rank() - 1);
      if (auto inner_val = VIInt(inner_vi)) {
        size_t inner_bytes = (size_t)*inner_val * elem_size;
        if (inner_bytes > (size_t)swiz_bytes) {
          size_t swiz_elems = (size_t)swiz_bytes / elem_size;

          // Split box: [outer, inner] -> [inner/swiz, outer, swiz]
          auto box_groups =
              sbe::nu((int64_t)(*inner_val / (int64_t)swiz_elems));
          std::vector<ValueItem> new_box;
          new_box.push_back(box_groups);
          for (size_t i = 0; i + 1 < t_shape.Rank(); ++i)
            new_box.push_back(t_shape.ValueAt(i));
          new_box.push_back(sbe::nu((int64_t)swiz_elems));
          box_shape = Shape(new_box);

          // Split host_shape: [outer, inner] -> [groups, outer, swiz]
          auto orig_inner = host_shape.ValueAt(host_shape.Rank() - 1);
          auto num_groups = orig_inner / sbe::nu((int64_t)swiz_elems);
          auto group_stride_val =
              sbe::nu((int64_t)swiz_elems) * gmem_ty->ElementSizeValue();

          std::vector<ValueItem> new_shape;
          new_shape.push_back(num_groups);
          for (size_t i = 0; i + 1 < host_shape.Rank(); ++i)
            new_shape.push_back(host_shape.ValueAt(i));
          new_shape.push_back(sbe::nu((int64_t)swiz_elems));
          host_shape = Shape(new_shape);

          // Strides: [group_stride, outer_stride..., elem_stride]
          std::vector<ValueItem> new_stride;
          new_stride.push_back(group_stride_val);
          for (size_t i = 0; i + 1 < host_stride.Rank(); ++i)
            new_stride.push_back(host_stride.ValueAt(i));
          new_stride.push_back(host_stride.ValueAt(host_stride.Rank() - 1));
          host_stride = Shape(new_stride);

          host_rank += 1;
          tma_inner_splits_[desc.GetName()] = {swiz_elems};
        }
      }
    }

    hs << h_indent << "uint64_t " << desc.GetName() << "_shape[] = {"
       << ValueSTR(Reverse(host_shape.Value())) << "};\n";
    hs << h_indent << "uint64_t " << desc.GetName() << "_strides[] = {"
       << ValueSTR(Trim(Reverse(host_stride.Value()))) << "};\n";
    {
      auto box_vals = Reverse(box_shape.Value());
      hs << h_indent << "uint32_t " << desc.GetName() << "_box_shape[] = {";
      for (size_t i = 0; i < box_vals.size(); ++i) {
        if (i > 0) hs << ", ";
        hs << "(uint32_t)(" << ValueSTR(box_vals[i]) << ")";
      }
      hs << "};\n";
    }
    hs << h_indent << "uint32_t " << desc.GetName() << "_elem_strides[] = {"
       << ValueSTR(ValxN(sbe::nu(1), box_shape.Rank())) << "};\n";

    hs << h_indent << "alignas(64) CUtensorMap " << map_name << "{};\n";
    hs << h_indent << "CUresult " << map_name
       << "_res = cuTensorMapEncodeTiled(\n";
    hs << h_indent << "        &" << map_name << ",\n";
    hs << h_indent << "        " << TMAMapDataType(gmem_ty->ElementType())
       << ",\n";
    hs << h_indent << "        " << host_rank << ",\n";
    hs << h_indent << "        " << base_expr << ",\n";
    hs << h_indent << "        " << desc.GetName() << "_shape,\n";
    hs << h_indent << "        " << desc.GetName() << "_strides,\n";
    hs << h_indent << "        " << desc.GetName() << "_box_shape,\n";
    hs << h_indent << "        " << desc.GetName() << "_elem_strides,\n";
    hs << h_indent
       << "        CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,\n";
    hs << h_indent << "        " << cu_swizzle_str << ",\n";
    hs << h_indent << "        "
       << CuTensorMapL2PromotionExpr(desc.GetPromoteBytes()) << ",\n";
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
    return "int64_t";
  else if (isa<U64Type>(&ty))
    return "uint64_t";
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

std::optional<int64_t> CuteCodeGen::GetSetRegLimit(AST::ParallelBy* pb) const {
  if (!pb) return std::nullopt;

  if (pb->HasMaxnreg()) {
    auto arg = pb->GetMaxnregArg();
    if (arg && arg->Opts().HasVal() && arg->Opts().GetVal()->IsNumeric()) {
      auto reg_limit = VIInt(arg->Opts().GetVal());
      if (reg_limit && reg_limit.value() > 0) return reg_limit.value();
    }
  }

  return std::nullopt;
}

std::optional<int64_t>
CuteCodeGen::GetLaunchBoundsMinBlocks(AST::ParallelBy* pb) const {
  if (!pb || !pb->HasLaunchBounds()) return std::nullopt;
  auto& lb_args = pb->GetLaunchBoundsArgs();
  if (lb_args->Count() < 2) return std::nullopt;
  auto arg = dyn_cast<AST::Expr>(lb_args->ValueAt(1));
  if (!arg || !arg->Opts().HasVal() || !arg->Opts().GetVal()->IsNumeric())
    return std::nullopt;
  auto min_blocks = VIInt(arg->Opts().GetVal());
  if (!min_blocks || min_blocks.value() <= 0) return std::nullopt;
  return min_blocks.value();
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
  known_val_str_to_var_.clear();

  const auto& lcs = cgi.GetFunctionLaunches(fname);
  auto setreg_limit = GetSetRegLimit(pb);
  auto launch_bounds_min_blocks = GetLaunchBoundsMinBlocks(pb);
  bool use_maxnreg_attr = setreg_limit.has_value() && CCtx().ArchNum() < 90;
  std::optional<std::string> launch_bounds_attr;
  if (!lcs.empty()) {
    auto inner_thr_count =
        lcs[0].thread_count.x * lcs[0].thread_count.y * lcs[0].thread_count.z;
    auto group_count = lcs[0].group_count.x * lcs[0].group4_count.x *
                       lcs[0].group_count.y * lcs[0].group_count.z;
    auto thr_count = inner_thr_count * group_count;

    std::string max_thr_str;
    if (pb && pb->HasLaunchBounds()) {
      auto arg0 = dyn_cast<AST::Expr>(pb->GetLaunchBoundsArgs()->ValueAt(0));
      if (arg0 && arg0->Opts().HasVal()) {
        auto v = VIInt(arg0->Opts().GetVal());
        if (v && v.value() > 0) max_thr_str = ValueSTR(arg0->Opts().GetVal());
      }
    }
    if (max_thr_str.empty() && VIInt(thr_count).has_value())
      max_thr_str = ValueSTR(thr_count);

    if (!max_thr_str.empty()) {
      std::optional<int64_t> max_blocks_per_cluster;
      if (pb && pb->HasLaunchBounds() &&
          pb->GetLaunchBoundsArgs()->Count() >= 3) {
        auto arg2 = dyn_cast<AST::Expr>(pb->GetLaunchBoundsArgs()->ValueAt(2));
        if (arg2 && arg2->Opts().HasVal())
          max_blocks_per_cluster = VIInt(arg2->Opts().GetVal());
      }

      if (launch_bounds_min_blocks.has_value()) {
        launch_bounds_attr = "__launch_bounds__(" + max_thr_str + ", " +
                             std::to_string(*launch_bounds_min_blocks);
        if (max_blocks_per_cluster)
          launch_bounds_attr.value() +=
              ", " + std::to_string(*max_blocks_per_cluster);
        launch_bounds_attr.value() += ") ";
      } else if (pb && pb->HasLaunchBounds() &&
                 pb->GetLaunchBoundsArgs()->Count() == 1) {
        launch_bounds_attr = "__launch_bounds__(" + max_thr_str + ") ";
      } else if (IsWarpSpecActive() ||
                 set_cuda_func_attribute_max_dynamic_shared_memory_size) {
        int64_t min_blocks = 1;
        if (auto inferred =
                cgi.GetFunctionTrait(fname).inferred_launch_bounds_min_blocks;
            inferred > 0)
          min_blocks = inferred;
        launch_bounds_attr = "__launch_bounds__(" + max_thr_str + ", " +
                             std::to_string(min_blocks) + ") ";
      }
    }
  }
  if (!lcs.empty() && lcs[0].HasCluster()) {
    auto& cc = lcs[0].cluster_count;
    auto cluster_total = cc.x * cc.y * cc.z;
    oss << "__cluster_dims__(" << ValueSTR(cluster_total) << ", 1, 1) ";
    oss << "__global__ ";
    if (launch_bounds_attr) oss << *launch_bounds_attr;
    if (use_maxnreg_attr) oss << "__maxnreg__(" << setreg_limit.value() << ") ";
    oss << "void " << device_fn << "(";
  } else {
    oss << "__global__ ";
    if (launch_bounds_attr) oss << *launch_bounds_attr;
    if (use_maxnreg_attr) oss << "__maxnreg__(" << setreg_limit.value() << ") ";
    oss << "void " << device_fn << "(";
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
  feed(__choreo_types_header_as_string);
  feed(__choreo_types_cute_header_as_string);
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
  os << R"(  CHOREO_CACHE_DIR=$(mktemp -d))"
     << "\n";
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
  os << R"(if [ ! -f "${PRECOMP_CACHED}" ]; then)"
     << "\n";
  os << R"(  LOCK_FILE="${CHOREO_CACHE_DIR}/.choreo_precompile.lock")"
     << "\n";
  os << R"(  ()"
     << "\n";
  os << R"(    flock -x 200)"
     << "\n";
  os << R"(    if [ ! -f "${PRECOMP_CACHED}" ]; then)"
     << "\n";
  os << R"(      echo "[choreo-fc] Building precompiled CuTe runtime for ${nv_arch} (one-time)..." >&2)"
     << "\n";
  os << "      ${NVCC} -dc ${DCFLAGS} " << precomp_cu
     << " -o \"${PRECOMP_CACHED}.tmp\" && \\\n";
  os << R"(      mv "${PRECOMP_CACHED}.tmp" "${PRECOMP_CACHED}")"
     << "\n";
  os << R"(      echo "[choreo-fc] Cached at ${PRECOMP_CACHED}" >&2)"
     << "\n";
  os << R"(    fi)"
     << "\n";
  os << R"(  ) 200>"${LOCK_FILE}")"
     << "\n";
  os << "fi\n\n";

  // Final check: the precompiled object must exist
  os << R"(if [ ! -f "${PRECOMP_CACHED}" ]; then)"
     << "\n";
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
  os << "cat <<'EOF' > " << build_path << "/choreo_types.h\n";
  os << __choreo_types_header_as_string << "\nEOF\n\n";
  os << "cat <<'EOF' > " << build_path << "/choreo_types_cute.h\n";
  os << __choreo_types_cute_header_as_string << "\nEOF\n\n";
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
  if (CCtx().UseFastMath()) os << " --use_fast_math";
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

    os << R"(if [ "$1" == "--execute" ] || [ "$#" -eq 0 ]; then)"
       << "\n";
    os << "  ${NVCC} -dc ${DCFLAGS} " << cc_file << " -o " << kernel_obj
       << "\n";
    os << "  ${NVCC} ${CFLAGS} " << kernel_obj << " ${PRECOMP_CACHED} -o "
       << exe_file << "\n";
    os << "  " << exe_file << "\n";
    os << R"(elif [ "$1" == "--compile-module" ]; then)"
       << "\n";
    os << "  ${NVCC} -dc ${DCFLAGS} " << cc_file << " -o " << exe_file << "\n";
    os << R"(elif [ "$1" == "--compile-link" ]; then)"
       << "\n";
    os << "  ${NVCC} -dc ${DCFLAGS} " << cc_file << " -o " << kernel_obj
       << "\n";
    os << "  ${NVCC} ${CFLAGS} " << kernel_obj << " ${PRECOMP_CACHED} -o "
       << exe_file << "\n";
    os << R"(elif [ "$1" == "--lib" ]; then)"
       << "\n";
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
  auto result = OpValueSTR(vi, "", true, LL_suffix, shp_lit);
  // Replace known subexpressions with their declared variable names.
  // This avoids re-expanding expressions like kv_tiles when they appear
  // inside larger expressions like kv_bound's ternary.
  if (!IsHost() && !shp_lit && !LL_suffix && !known_val_str_to_var_.empty()) {
    auto isWordChar = [](char c) {
      return std::isalnum(static_cast<unsigned char>(c)) || c == '_';
    };
    for (auto& [expr, var] : known_val_str_to_var_) {
      for (auto pos = result.find(expr); pos != std::string::npos;
           pos = result.find(expr, pos + 1)) {
        auto end = pos + expr.size();
        if (pos > 0 && isWordChar(result[pos - 1])) continue;
        if (end < result.size() && isWordChar(result[end])) continue;
        result.replace(pos, expr.size(), var);
        pos += var.size() - 1;
      }
    }
  }
  return result;
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
            << ExprCastSTR(n, val, BT::F32, f, is_host, is_explicit) << ")";
      else
        res << "static_cast<" << nbt << ">(" << value << ")";
    }
    break;
  }
  case BT::F64: {
    if (IsIntegralType(f))
      res << "static_cast<double>(" << value << ")";
    else
      res << "static_cast<double>("
          << ExprCastSTR(n, val, BT::F32, f, is_host, is_explicit) << ")";
    break;
  }
  case BT::F32: {
    if (IsIntegralType(f))
      res << "static_cast<float>(" << value << ")";
    else {
      if (f == BT::F16)
        res << "choreo::f16_to_f32(" << value << ")";
      else if (f == BT::BF16)
        res << "choreo::bf16_to_f32(" << value << ")";
      else if (f == BT::F64)
        res << "static_cast<float>(" << value << ")";
      else
        res << "static_cast<float>(" << value << ")";
    }
    break;
  }
  case BT::F16:
    res << "choreo::f32_to_f16("
        << ExprCastSTR(n, val, BT::F32, f, is_host, is_explicit) << ")";
    break;
  case BT::BF16:
    res << "choreo::f32_to_bf16("
        << ExprCastSTR(n, val, BT::F32, f, is_host, is_explicit) << ")";
    break;
  case BT::F8_E4M3:
  case BT::F8_E5M2:
    res << "choreo::utils::from_f32<"
        << ((t == BT::F8_E4M3) ? "f8_e4m3" : "f8_e5m2") << ">("
        << ExprCastSTR(n, val, BT::F32, f, is_host, is_explicit) << ")";
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
    if (frag_apply_iv_map_.count(id->name)) {
      oss << frag_apply_iv_map_.at(id->name);
    } else if (within_map.count(InScopeNameForRef(id->name)) && !is_host) {
      size_t i = 0;
      for (auto iv_name : within_map.at(InScopeNameForRef(id->name)))
        oss << ((i++ == 0) ? "" : ", ")
            << UnScopedName(ssm.DeviceName(iv_name));
    } else {
      oss << UnScopedName(SSMName(InScopeNameForRef(id->name), is_host));
    }
  } else if (auto np = dyn_cast<AST::Nullptr>(e)) {
    oss << "nullptr";
  } else if (auto il = dyn_cast<AST::IntLiteral>(e)) {
    oss << il->ValAsString();
  } else if (auto fl = dyn_cast<AST::FloatLiteral>(e)) {
    std::ostringstream fp_val;
    if (fl->IsFloat32()) {
      auto v = fl->Val_f32();
      if (std::isinf(v))
        fp_val << (v < 0 ? "(-INFINITY)" : "INFINITY");
      else if (std::isnan(v))
        fp_val << "NAN";
      else
        fp_val << std::fixed << v << "f";
    } else if (fl->IsFloat64()) {
      auto v = fl->Val_f64();
      if (std::isinf(v))
        fp_val << (v < 0 ? "(-INFINITY)" : "INFINITY");
      else if (std::isnan(v))
        fp_val << "NAN";
      else
        fp_val << std::fixed << v;
    } else {
      choreo_unreachable("unsupported float literal.");
    }
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
      auto scoped = InScopeNameForRef(da->data->name);
      if (in_register_direct_automap_ && automap_frag_reg_expr_.count(scoped)) {
        oss << automap_frag_reg_expr_.at(scoped);
      } else if (FCtx(fname).HasFragmentLayout(scoped)) {
        const auto& fl = FCtx(fname).GetFragmentLayout(scoped);
        if ((fl.kind == LayoutKind::WGMMA_ACC ||
             fl.kind == LayoutKind::WGMMA_RS_A) &&
            da->GetIndices().size() == 2) {
          auto idx_it = da->GetIndices().begin();
          auto row_expr = OpExprSTR(*idx_it++, "+", true, is_host);
          auto col_expr = OpExprSTR(*idx_it, "+", false, is_host);
          oss << UnScopedName(SSMName(scoped, is_host)) << "["
              << fl.ForwardIndex(row_expr, col_expr) << "]";
        } else if (FCtx(fname).HasFragmentInfo(scoped) &&
                   !FCtx(fname).FragIsWGMMA(scoped)) {
          sbe::ExprSum linear;
          size_t idx = 0;
          auto shape = sty->GetShape();
          auto AppendLinear = [&](const ValueItem& op) {
            auto offset = op;
            assert(shape.Rank() >= idx + 1);
            if (shape.Rank() > idx + 1)
              offset = offset * shape.TrimDims(idx + 1).ElementCountValue();
            SimplifyExpression(offset);
            linear += offset;
            ++idx;
          };
          for (auto item : da->GetIndices()) {
            if (auto id = AST::GetIdentifier(item)) {
              if (frag_apply_iv_map_.count(id->name)) {
                AppendLinear(sbe::sym(frag_apply_iv_map_.at(id->name)));
              } else if (within_map.count(InScopeNameForRef(id->name))) {
                auto ivs = within_map.at(InScopeNameForRef(id->name));
                for (auto iv_itr = ivs.begin(); iv_itr != ivs.end(); ++iv_itr)
                  AppendLinear(sbe::sym(*iv_itr));
              } else
                AppendLinear(sbe::sym(InScopeNameForRef(id->name)));
            } else if (auto il = AST::GetIntLiteral(*item)) {
              AppendLinear(sbe::nu(il->Val()));
            } else {
              assert(shape.Rank() >= idx + 1);
              if (shape.Rank() > idx + 1)
                linear += sbe::sym(
                    "(" + OpExprSTR(item, "*", true, is_host) + ")*" +
                    ValueSTR(shape.TrimDims(idx + 1).ElementCountValue()));
              else
                linear += sbe::sym(OpExprSTR(item, "+", false, is_host));
              ++idx;
            }
          }
          const auto& finfo = FCtx(fname).GetFragmentInfo(scoped);
          auto linear_str = ValueSTR(linear.Get());
          if (FCtx(fname).HasFragmentLayout(scoped) &&
              FCtx(fname).GetFragmentLayout(scoped).vec_width > 1) {
            auto& fl = FCtx(fname).GetFragmentLayout(scoped);
            auto TW = std::to_string(fl.thread_count * fl.vec_width);
            auto W = std::to_string(fl.vec_width);
            oss << UnScopedName(SSMName(scoped, is_host)) << "[(" << linear_str
                << ") / " << TW << " * " << W << " + (" << linear_str << ") % "
                << W << "]";
          } else {
            oss << UnScopedName(SSMName(scoped, is_host)) << "[(" << linear_str
                << ") / (" << finfo.thread_count_expr << ")]";
          }
        } else {
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
              if (frag_apply_iv_map_.count(id->name)) {
                AppendOffset(sbe::sym(frag_apply_iv_map_.at(id->name)));
              } else if (within_map.count(InScopeNameForRef(id->name))) {
                auto ivs = within_map.at(InScopeNameForRef(id->name));
                for (auto iv_itr = ivs.begin(); iv_itr != ivs.end(); ++iv_itr)
                  AppendOffset(sbe::sym(*iv_itr));
              } else
                AppendOffset(sbe::sym(InScopeNameForRef(id->name)));
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
        }
      } else if (sty->GetStorage() == Storage::REG &&
                 FCtx(fname).HasFragmentInfo(scoped) &&
                 !FCtx(fname).FragIsWGMMA(scoped)) {
        sbe::ExprSum linear;
        size_t idx = 0;
        auto shape = sty->GetShape();
        auto AppendLinear = [&](const ValueItem& op) {
          auto offset = op;
          assert(shape.Rank() >= idx + 1);
          if (shape.Rank() > idx + 1)
            offset = offset * shape.TrimDims(idx + 1).ElementCountValue();
          SimplifyExpression(offset);
          linear += offset;
          ++idx;
        };
        for (auto item : da->GetIndices()) {
          if (auto id = AST::GetIdentifier(item)) {
            if (frag_apply_iv_map_.count(id->name)) {
              AppendLinear(sbe::sym(frag_apply_iv_map_.at(id->name)));
            } else if (within_map.count(InScopeNameForRef(id->name))) {
              auto ivs = within_map.at(InScopeNameForRef(id->name));
              for (auto iv_itr = ivs.begin(); iv_itr != ivs.end(); ++iv_itr)
                AppendLinear(sbe::sym(*iv_itr));
            } else
              AppendLinear(sbe::sym(InScopeNameForRef(id->name)));
          } else if (auto il = AST::GetIntLiteral(*item)) {
            AppendLinear(sbe::nu(il->Val()));
          } else {
            assert(shape.Rank() >= idx + 1);
            if (shape.Rank() > idx + 1)
              linear += sbe::sym(
                  "(" + OpExprSTR(item, "*", true, is_host) + ")*" +
                  ValueSTR(shape.TrimDims(idx + 1).ElementCountValue()));
            else
              linear += sbe::sym(OpExprSTR(item, "+", false, is_host));
            ++idx;
          }
        }
        const auto& finfo = FCtx(fname).GetFragmentInfo(scoped);
        auto linear_str = ValueSTR(linear.Get());
        if (FCtx(fname).HasFragmentLayout(scoped) &&
            FCtx(fname).GetFragmentLayout(scoped).vec_width > 1) {
          auto& fl = FCtx(fname).GetFragmentLayout(scoped);
          auto TW = std::to_string(fl.thread_count * fl.vec_width);
          auto W = std::to_string(fl.vec_width);
          oss << UnScopedName(SSMName(scoped, is_host)) << "[(" << linear_str
              << ") / " << TW << " * " << W << " + (" << linear_str << ") % "
              << W << "]";
        } else {
          oss << UnScopedName(SSMName(scoped, is_host)) << "[(" << linear_str
              << ") / (" << finfo.thread_count_expr << ")]";
        }
      } else {
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
            if (frag_apply_iv_map_.count(id->name)) {
              AppendOffset(sbe::sym(frag_apply_iv_map_.at(id->name)));
            } else if (within_map.count(InScopeNameForRef(id->name))) {
              auto ivs = within_map.at(InScopeNameForRef(id->name));
              for (auto iv_itr = ivs.begin(); iv_itr != ivs.end(); ++iv_itr)
                AppendOffset(sbe::sym(*iv_itr));
            } else
              AppendOffset(sbe::sym(InScopeNameForRef(id->name)));
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
      }
    } else {
      auto sym = da->data->name;
      if (auto sty = GetSpannedType(GetSymbolType(sym))) {
        oss << UnScopedName(SSMName(InScopeNameForRef(sym), is_host));
      } else {
        assert(!within_map.count(InScopeNameForRef(da->data->name)));
        oss << UnScopedName(SSMName(InScopeNameForRef(sym), is_host));
      }
    }
  } else if (auto ce = dyn_cast<AST::CastExpr>(e)) {
    assert(ce->GetOp() == Op::Cast);
    if (ce->IsForeignCast())
      return "((" + ce->ForeignType() + ")" +
             OpExprSTR(ce->GetR(), "", true, is_host) + ")";
    auto from_bt = ce->FromType();
    if (in_register_direct_automap_ && from_bt == BaseType::F64) {
      if (auto da = dyn_cast<AST::DataAccess>(ce->GetR())) {
        auto sc = InScopeNameForRef(da->data->name);
        if (FCtx(fname).HasFragmentLayout(sc))
          from_bt = ElementType(GetSymbolType(da->data->name));
      } else if (auto expr = dyn_cast<AST::Expr>(ce->GetR())) {
        if (auto ref = dyn_cast<AST::DataAccess>(expr->GetReference())) {
          auto sc = InScopeNameForRef(ref->data->name);
          if (FCtx(fname).HasFragmentLayout(sc))
            from_bt = ElementType(GetSymbolType(ref->data->name));
        }
      }
    }
    return ExprCastSTR(ce->GetR(), std::nullopt, ce->ToType(), from_bt, is_host,
                       ce->IsExplicit());
  } else if (auto expr = dyn_cast<AST::Expr>(e)) {
    // frag.apply lambda param override (must be checked before scope lookup)
    if (auto sym = expr->GetSymbol()) {
      if (frag_apply_iv_map_.count(sym->name))
        return frag_apply_iv_map_.at(sym->name);
    }
    // Use the declared variable name when available rather than inlining
    // the full optimized expression. This avoids re-computing complex
    // expressions (e.g. ternary loop bounds) on every reference.
    if (auto sym = expr->GetSymbol()) {
      auto sname = InScopeNameForRef(sym->name);
      if (!is_host && ssm.HasDeviceName(sname)) {
        return UnScopedName(ssm.DeviceName(sname));
      }
      if (!frag_apply_iv_map_.count(sym->name) &&
          FCtx(fname).HasSymbolValues(sname)) {
        auto svs = FCtx(fname).GetSymbolValues(sname);
        if (svs.HasVal()) return ValueSTR(svs.GetVal());
      }
    }

    if (ConvertibleToInt(NodeType(*e)) && frag_apply_iv_map_.empty()) {
      if (expr->Opts().HasVal()) return ValueSTR(expr->Opts().GetVal());
    }

    if (expr->IsReference()) {
      if (PSTR(expr) == "_") return "0";
      if (auto sym = expr->GetSymbol();
          sym && frag_apply_iv_map_.count(sym->name))
        return frag_apply_iv_map_.at(sym->name);
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
          auto base_dev = ssm.DeviceName(InScopeNameForRef(base_id->name));
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
    auto caty = GetSpannedType(ca->GetType());
    if (caty && caty->GetStorage() == Storage::REG &&
        FCtx(fname).HasFragmentInfo(InScopeName(ca->data->name)) &&
        !FCtx(fname).FragIsWGMMA(InScopeName(ca->data->name)))
      return EmitUniformFragmentAccess(ca, is_host);
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

  if (n.IsAtomic()) {
    static const std::unordered_map<std::string, std::string> atomic_name_map =
        {{"__atomic_add", "atomicAdd"},   {"__atomic_sub", "atomicSub"},
         {"__atomic_exch", "atomicExch"}, {"__atomic_min", "atomicMin"},
         {"__atomic_max", "atomicMax"},   {"__atomic_and", "atomicAnd"},
         {"__atomic_or", "atomicOr"},     {"__atomic_xor", "atomicXor"},
         {"__atomic_cas", "atomicCAS"}};
    auto it = atomic_name_map.find(n.function->name);
    assert(it != atomic_name_map.end());
    oss << it->second << "(";
    size_t i = 0;
    for (auto& a : n.GetArguments()) {
      if (i > 0) oss << ", ";
      if (i == 0)
        oss << "&(" << OpExprSTR(a, "", true, IsHost()) << ")";
      else
        oss << OpExprSTR(a, "", true, IsHost());
      ++i;
    }
    oss << ")";
    return oss.str();
  }

  static const std::unordered_map<std::string, std::string> cuda_intrinsic_map =
      {{"__fmaf", "fmaf"}, {"__frcp_rn", "__frcp_rn"}};

  auto func_name = [&n](const std::string& name) -> std::string {
    if (!n.IsArith()) return name;
    auto it = cuda_intrinsic_map.find(name);
    if (it != cuda_intrinsic_map.end()) return it->second;
    const std::string prefix = "__";
    std::string func_name = name;
    if (auto res = RemovePrefixOrNull(prefix, name)) func_name = *res;
    if (func_name == "min" || func_name == "max")
      return "(choreo::nv_cute::numerics::" + func_name + ")";
    return "choreo::nv_cute::numerics::" + func_name;
  };

  oss << func_name(n.function->name);

  // emit template arguments
  if (n.template_args) {
    oss << "<";
    size_t i = 0;
    for (auto& ta : n.template_args->AllValues()) {
      // Namespace-qualified C++ name: emit verbatim; VALNO strips the prefix.
      std::string ta_str;
      if (auto id = dyn_cast<AST::Identifier>(ta))
        if (id->name.find("::") != std::string::npos) ta_str = id->name;
      if (ta_str.empty())
        if (auto expr = dyn_cast<AST::Expr>(ta))
          if (auto sym = expr->GetSymbol())
            if (sym->name.find("::") != std::string::npos) ta_str = sym->name;
      oss << ((i++ == 0) ? "" : ", ")
          << (ta_str.empty() ? OpExprSTR(ta, "", true, IsHost()) : ta_str);
    }
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

std::string CuteCodeGen::ResolveAutomapThreadExpr(
    const ptr<AST::AttributeExpr>& automap_attr) const {
  if (automap_attr && automap_attr->AttrValueCount() == 1) {
    if (auto pv = AST::GetIdentifier(automap_attr->AttrValueAt(0)))
      return UnScopedName(SSMName(InScopeNameForRef(pv->name), false));
  }
  return vid_pfx + "tid_x";
}

std::string CuteCodeGen::EmitUniformFragmentAccess(const ptr<AST::ChunkAt>& ca,
                                                   bool is_host) const {
  auto scoped = InScopeName(ca->data->name);
  const auto& finfo = FCtx(fname).GetFragmentInfo(scoped);
  auto sym = UnScopedName(SSMName(scoped, is_host));
  auto linear = ValueSTR(GenOffset(ca));
  return sym + "[(" + linear + ") / (" + finfo.thread_count_expr + ")]";
}
