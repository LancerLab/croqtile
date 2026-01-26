#include "cute_codegen.hpp"
#include "codegen_utils.hpp"

#include <filesystem>
#include <iostream>
#include <numeric>
#include <sstream>
#include <thread>

#include "ast.hpp"
#include "choreo_cute_header.inc"
#include "choreo_header.inc"
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

namespace Choreo {
extern Option<bool> sim_sparse;
} // namespace Choreo
Option<bool> use_cuda_type(OptionKind::Hidden, "-use-cuda-type", "", true,
                           "use cuda built-in types.");

namespace cute {

inline const char* CudaDeviceMemory(Storage st) {
  switch (st) {
  case Storage::SHARED: return "__shared__";
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

inline const std::string GetCopyAtomName() {
  static unsigned i = 0;
  return "choreo_copy_atom" + std::to_string(i++);
}

inline void PrintSubscriptions(std::ostream& os, const std::string prefix,
                               const std::string suffix,
                               const std::vector<size_t>& dims,
                               std::vector<size_t>& indices, size_t depth = 0) {
  if (depth == dims.size()) {
    os << prefix;
    for (size_t i : indices) os << "[" << i << "]";
    os << suffix;
    return;
  }

  for (size_t i = 0; i < dims[depth]; ++i) {
    indices[depth] = i;
    PrintSubscriptions(os, prefix, suffix, dims, indices, depth + 1);
  }
}

std::string GetAbsPath(const std::filesystem::path& cwd,
                       const std::string& relative_path) {
  std::filesystem::path rel_path(relative_path);
  std::filesystem::path abs_path = cwd / rel_path;
  abs_path = std::filesystem::weakly_canonical(abs_path).parent_path();
  return abs_path.string();
}

void GenerateSubscriptions(std::ostream& os, const std::string prefix,
                           const std::string suffix,
                           const std::vector<size_t>& dims) {
  std::vector<size_t> indices(dims.size());
  PrintSubscriptions(os, prefix, suffix, dims, indices);
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

// return mds name and the declaration string.
// If offset is not empty, means that need to do memory viewing.
//   Just add offset to buf_expr, then utilize new_shape.
std::pair<std::string, std::string> CuteCodeGen::GenTensorDecl(
    const std::string& bname, const std::string& buf_expr, const Storage sto,
    BaseType bty, const Shape& shp, bool is_host, const std::string& offset,
    const std::string& strides, const std::vector<size_t>& transp,
    bool use_wgmma_layout, int swizzle_value) const {
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
  if (use_wgmma_layout && sto == Storage::SHARED &&
      (bty == BaseType::F16 || bty == BaseType::BF16)) {
    // Select swizzle layout based on swizzle value
    std::string swizzle_layout;
    switch (swizzle_value) {
    case 32: swizzle_layout = "cute::SM90::GMMA::Layout_K_SW32_Atom"; break;
    case 64: swizzle_layout = "cute::SM90::GMMA::Layout_K_SW64_Atom"; break;
    case 128: swizzle_layout = "cute::SM90::GMMA::Layout_K_SW128_Atom"; break;
    default: swizzle_layout = "cute::SM90::GMMA::Layout_K_SW128_Atom"; break;
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
    device_fn = "__choreo_device_" + fname;
    fty = cast<FunctionType>(GetSymbolType(fname));
    ssm.EnterScope();
    levels.push(ParallelLevel::SEQ);
  } else if (auto pb = dyn_cast<AST::ParallelBy>(&n)) {
    levels.push(pb->GetLevel());
    // only on device-side
    if (pb->IsOuter()) {
      parallel_idx += 1;
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
  } else if (isa<AST::IncrementBlock>(&n)) {
    IndStream() << "// incr: " << n.LOC() << "\n";
    IncrIndent();
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
  } else if (auto pb = dyn_cast<AST::ParallelBy>(&n)) {
    levels.pop();
    // only on device-side
    if (pb->IsOuter()) {
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
    DecrIndent();
    IndStream() << "}\n";
  } else if (auto fb = dyn_cast<AST::ForeachBlock>(&n)) {
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
  } else if (auto it = dyn_cast<AST::InThreadsBlock>(&n)) {
    // only on device-side
    DecrDeviceIndent();
    if (!it->stmts->None()) {
      ds << d_indent << "}";
      if (!it->async && it->outer) ds << "\n" << d_indent << "__syncthreads();";
      ds << " // end inthreads\n";
    }
  } else if (auto ie = dyn_cast<AST::IfElseBlock>(&n)) {
    DecrIndent();
    IndStream() << "} // end if-else: " << ie->LOC() << "\n";
  } else if (auto ie = dyn_cast<AST::WhileBlock>(&n)) {
    DecrIndent();
    IndStream() << "} // end while: " << ie->LOC() << "\n";
  } else if (isa<AST::IncrementBlock>(&n)) {
    DecrIndent();
    IndStream() << "}\n";
  } else if (isa<AST::NamedVariableDecl>(&n)) {
    emit_call = true;
  }

  return true;
}

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

  // handle each chunkat inside a seqeunce like 'chunkat(a, b).chunkat(c)...'
  for (size_t sop_idx = sop_base; sop_idx < sops.size(); ++sop_idx) {
    // span_as reshape operation would not affect index generation
    assert(!sops[sop_idx]->SpecifyReshape());

    // For each chunkat expression, The tiled-block's shape is cooked by shape
    // inference. The block shape is different with the result shape of chunkat
    // expression when using 'modspan', where the result shape represents the
    // shape that applied mod (%) operation. Anyway, for offset, we only care
    // about the tiled-block's shape
    auto& shape = sops[sop_idx]->GetBlockShape();

    ValueList exprs;
    // For each 'a, b, c, ...' inside 'chunkat(a, b, c, ...)', that 'b' inside
    // 'chunkat(a, b, c, ...)' could be bounded var like b = {b0, b1} Therefore,
    // we collect all the expressions first.
    for (auto p : sops[sop_idx]->GetIndices()) {
      if (const auto& o = dyn_cast<AST::Expr>(p)->Opts(); o.HasVals()) {
        const auto& vals = o.GetVals();
        for (auto& val : vals) {
          if (sbe::ceq(val, sbe::sym("::__choreo_no_tiling__")))
            exprs.push_back(sbe::nu(0));
          else
            exprs.push_back(val);
        }
      } else
        exprs.push_back(sbe::sym(OpExprSTR(p, "*", true, IsHost())));
    }

    if (auto tc = dyn_cast<TransposeConfig>(config)) {
      assert(tc->dim_values.size() == exprs.size());
      assert(ca->TilingOperationCount() == 1);
    }

    // Generate the expression for single chunkat
    // Note that we buffer all expressions of different chunkats by dimensions
    for (size_t i = 0; i < exprs.size(); ++i) indices.push_back(sbe::nu(1));

    for (size_t i = 0; i < exprs.size(); ++i) {
      // combine 'a' and 'c' between expressions like 'chunkat(a, b).chunk(c,
      // d)'
      indices[i] = (indices[i] * exprs[i] * shape.ValueAt(i))->Normalize();
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
    assert(!sops[sop_idx]->SpecifyReshape());

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
    for (size_t pi = 0; pi < sops[sop_idx]->GetIndices().size(); ++pi) {
      auto p = sops[sop_idx]->GetIndices()[pi];
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

// Example:
//
//   f32 [10, 9, 8] a;
//   ... a.subspan(2, 9, 8).at(p, _, _).span_as(...);
//
// The "Tile Base Offset" (offset ahead of last span_as) is:
//
//    tbo = p * 2 * (9 * 8)
//
const std::string
CuteCodeGen::TileBaseOffset(const ptr<AST::ChunkAt>& ca) const {
  auto lidx = ca->IndexOfLastSpanAs();
  if (!lidx.has_value()) choreo_unreachable("unexpect");
  return ValueSTR(GenOffset(ca, lidx.value()));
}

// given i.sop(...).sop(...)..., generate the offset of the final span in the
// original span. It is VALID if and only if the final span is
// address-contiguous within the original span.
// end_idx: the offset is computed by sop in range [0, end_idx).
const ValueItem CuteCodeGen::GenOffset(const ptr<AST::ChunkAt>& ca,
                                       size_t end_idx) const {
  if (ca->NoOperation()) return sbe::nu(0);

  end_idx = std::min(end_idx, ca->OpCount());

  Shape outer_shape = GetSpannedType(GetSymbolType(ca->data->name))->GetShape();

  auto offset = sbe::nu(0);

  Shape stride_shape;

  assert(ca->OpCount() == 1 &&
         "count of spanned operations in CuTe DMA should be 1.");
  if (auto s = ca->OpAt(0)->GetStrides()) {
    auto stride_vl = ca->OpAt(0)->StridesAsValueList();
    stride_shape = Shape(stride_vl);
  }

  // outer_shape is the shape of original span
  // new_shape is the shape of tiled span
  Shape new_shape;
  for (size_t i = 0; i < end_idx; ++i) {
    const auto& sop = ca->OpAt(i);
    if (sop->SpecifyReshape()) {
      outer_shape = sop->GetBlockShape();
    } else {
      // if stride is defined, use it as tiled shape.
      if (stride_shape.IsValid())
        new_shape = stride_shape;
      else
        new_shape = sop->GetBlockShape();
      size_t i = 0;
      for (auto p : sop->GetIndices()) {
        if (const auto& o = dyn_cast<AST::Expr>(p)->Opts(); o.HasVals()) {
          const auto& vals = o.GetVals();
          for (auto val : vals) {
            auto outer_factor = sbe::nu(1);
            if (outer_shape.Rank() > i + 1)
              outer_factor = outer_shape.TrimDims(i + 1).ElementCountValue();
            auto factor = new_shape.ValueAt(i) * outer_factor;
            offset = offset + val * factor;
            ++i;
          }
        } else {
          auto idx_exprs =
              SplitStringByDelimiter(OpExprSTR(p, "*", true, IsHost()));
          for (auto i_expr : idx_exprs) {
            ValueItem outer_factor = sbe::nu(1);
            if (outer_shape.Rank() > i + 1)
              outer_factor = outer_shape.TrimDims(i + 1).ElementCountValue();
            auto factor = new_shape.ValueAt(i) * outer_factor;
            offset = offset + sbe::sym(i_expr) * factor;
            ++i;
          }
        }
      }
      outer_shape = new_shape;
    }
  }

  return offset;
}

const ValueList CuteCodeGen::GenStrides(const Shape& outer_shape,
                                        const std::vector<size_t>& tc) const {
  // Note: always generate stride since cute::copy may propagate strides

  ValueList strds;
  for (size_t i = 1; i < outer_shape.Rank(); ++i)
    strds.push_back(outer_shape.TrimDims(i).ElementCountValue());
  strds.push_back(sbe::nu(1));

  if (tc.size() != 0) {
    assert(tc.size() == strds.size());
    ValueList t_strds = strds;
    for (size_t i = 0; i < strds.size(); ++i) t_strds[i] = strds[tc[i]];
    return t_strds;
  }

  return strds;
}

const ValueList CuteCodeGen::GenStrides(const ptr<AST::ChunkAt>& ca,
                                        const std::vector<size_t>& tc) const {
  // Note: always generate stride since cute::copy may propagate strides

  // TODO: handle multiple operations
  Shape outer_shape = GetSpannedType(GetSymbolType(ca->data->name))->GetShape();
  return GenStrides(outer_shape, tc);
}

void CuteCodeGen::EmitFixedHostHead() {
  std::ostringstream oss;
  oss <<
      R"(
#include <fstream>
#include <cstdlib>
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
  if (cgi.HasTMA()) oss << "namespace cde = cuda::device::experimental;\n";
  oss << "\nusing namespace choreo;\n";
  if (CCtx().GetApiMode() != "sglang") {
    oss << "\n#define __CHOREO_REQUIRED_GPU_DEVICE_SM__ " << CCtx().ArchNum()
        << "\n";
    EmitRuntimeEnvironmentChecker(oss);
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
    hs << h_indent << "__choreo_check_cuda_environment__();\n";

  // name the symbolic dimensions for better readability
  for (auto item : symbolic_dimensions) {
    hs << h_indent << "auto &" << UnScopedName(item.first) << " = "
       << item.second.hsd_expr << ";\n";
    ssm.MapHostSymbol(item.first, UnScopedName(item.first));
  }

  // emit the runtime checks
  EmitHostRuntimeCheck();

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

  DecrHostIndent();
  hs << "}\n\n";

  return true;
}

bool CuteCodeGen::Visit(AST::NamedVariableDecl& n) {
  TraceEachVisit(n);

  auto nty = NodeType(n);
  auto sym = n.name_str;

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
    sa = dyn_cast<AST::SpanAs>(e->GetReference());
    if (sa) {
      // handle span_as of global buffer in `HandleGlobal`
      if (!IsHost()) {
        ds << d_indent << "auto* " << sym << " = ";
        ds << "static_cast<"
           << NameBaseType(dyn_cast<SpannedType>(nty)->ElementType()) << "*>(";
        auto tty = GetSymbolType(sa->id->name);
        if (isa<FutureType>(tty))
          ds << sa->id->name << ".data());\n";
        else
          ds << sa->id->name << ");\n";
        ssm.MapDeviceSymbol(InScopeName(sym), sym);
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
        ds << d_indent << type_modifiers << bts << " " << sym;
        for (auto dim : n.ArrayDimensions()) ds << "[" << dim << "]";
        ds << "[" << UnScopedExpr(ElemCountExprOf(*sty)) << "];\n";
        return;
      }

      // memory reuse is enabled

      if (n.HasNote("spm")) {
        if (sto == Storage::SHARED &&
            (FCtx(fname).HaveDynamicBuffer(SSTab().ScopeName(), sto) ||
             set_cuda_func_attribute_max_dynamic_shared_memory_size))
          ds << d_indent << "auto " << sym << " = (" << bts << "*)&"
             << device_fn << "__runtime_shared_buffer__;\n";
        else
          ds << d_indent << type_modifiers << "alignas("
             << n.GetNote("alignment") << ") " << bts << " " << sym << "["
             << UnScopedExpr(ElemCountExprOf(*sty)) << "];\n";
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
        // which means that it is declared but never used.
        // TODO: should we DCE the unused buffer?
        assert(!n.HasNote("offset"));
        ds << d_indent << type_modifiers << bts << " " << sym << "["
           << UnScopedExpr(ElemCountExprOf(*sty)) << "];\n";
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

  if (auto bty = dyn_cast<BoundedType>(nty)) {
    // bounded variable is not with a fixed value
    if (!IsActualBoundedIntegerType(bty))
      choreo_unreachable(
          "yet to support: bounded ituple variable code generation.");
    IndStream() << "int " << sym << " = " << ExprSTR(n.init_expr, false)
                << ";\n";
    return true;
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
    case Storage::SHARED:
    case Storage::LOCAL: {
      assert(!IsHost());
      ds << d_indent << CudaDeviceMemory(ety->GetStorage())
         << " __volatile__ bool " << n.name_str;
      ety->PrintAsCArray(ds);
      ds << "; // " << STR(ety->GetStorage()) << " event\n";
      ds << d_indent << "// initialize the event\n";
      if (ety->GetStorage() == Storage::SHARED) {
        ds << d_indent << LevelPred() << " {\n";
        GenerateSubscriptions(ds, "  " + d_indent + n.name_str, " = false;\n",
                              ety->Dimensions());
        ds << d_indent << "}\n";
      } else
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
    case Storage::SHARED:
    case Storage::LOCAL: {
      assert(!IsHost());
      ds << d_indent << CudaDeviceMemory(ety->GetStorage())
         << " __volatile__ bool " << n.name_str << "; // "
         << STR(ety->GetStorage()) << " event\n";
      if (ety->GetStorage() == Storage::SHARED) {
        ds << d_indent << LevelPred() << " {\n";
        ds << d_indent << "  " << n.name_str
           << " = false;\n"; // inited as untriggered
        ds << d_indent << "}\n";
      } else
        ds << d_indent << n.name_str << " = false;\n"; // inited as untriggered
      ds << d_indent << EmitSync(ety->GetStorage()) << ";\n";
    } break;
    default: break;
    }
  }

  return true;
}

bool CuteCodeGen::Visit(AST::Assignment& n) {
  TraceEachVisit(n);

  auto nty = NodeType(n);

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
    ds << d_indent << "auto * " << n.GetName() << " = ";
    auto tty = GetSymbolType(sa->id->name);
    ds << "static_cast<"
       << NameBaseType(dyn_cast<SpannedType>(nty)->ElementType()) << "*>(";
    if (isa<FutureType>(tty))
      ds << sa->id->name << ".data());\n";
    else
      ds << sa->id->name << ");\n";
    ssm.MapDeviceSymbol(InScopeName(n.GetName()), n.GetName());
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
    if (!(sty && sty->GetStorage() == Storage::REG && n.HasNote("update")))
      ds << d_indent << ((IsMutable(*nty)) ? "" : "auto ") << n.GetName()
         << " = ";
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
  case ParallelLevel::BLOCK: lconfig.SetBlockCount(n.BoundValues()); break;
  case ParallelLevel::GROUP: lconfig.SetGroupCount(n.BoundValues()); break;
  case ParallelLevel::GROUPx4: lconfig.SetGroupx4Count(n.BoundValues()); break;
  case ParallelLevel::THREAD: lconfig.SetThreadCount(n.BoundValues()); break;
  default: break;
  }

  // add the device name map
  std::string dname[] = {"x", "y", "z"};
  switch (n.GetLevel()) {
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

  // only do the whole codegen when accessing the outer parallel-by
  if (n.IsOuter()) {

    ValueItem cur_spm_size = sbe::nu(0);
    ValueItem cur_ring_offset = sbe::nu(0);
    ValueItem cur_ring_size = sbe::nu(0);

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
    auto inner_thr_count = (lconfig.thread_count.x * lconfig.thread_count.y *
                            lconfig.thread_count.z)
                               ->Normalize();
    auto group_count = (lconfig.group_count.x * lconfig.group4_count.x *
                        lconfig.group_count.y * lconfig.group_count.z)
                           ->Normalize();
    auto thr_count = (inner_thr_count * group_count)->Normalize();

    hs << h_indent << "dim3 __" << fname << "_bdims" << parallel_idx << "("
       << ValueSTR(thr_count) << ", 1, 1" << ");\n";

    // plan the shared memory that is decided at runtime
    cur_ring_size = (thr_count + sbe::nu(31)) / sbe::nu(32) /* warp size */;

    // add the size of the future ring (see choreo.h)
    if (cgi.HasAsyncDMA(fname))
      cur_spm_size = cur_spm_size + cur_ring_size * sbe::nu(8);

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
         << ValueSTR(cur_spm_size) << ");\n";
      set_cuda_func_attribute_max_dynamic_shared_memory_size = true;
    };

    if (auto dev_name = SSTab().ScopeName();
        FCtx(fname).HaveDynamicBuffer(dev_name, Storage::SHARED)) {
      // add the size of dynamic shared
      auto mri = FCtx(fname).GetDynMemReuseInfo(dev_name);
      assert(mri);
      cur_ring_offset =
          sbe::sym(mri->infos[Storage::SHARED].spm_size)->Normalize();
      cur_spm_size = cur_spm_size + cur_ring_offset;
      cur_spm_size = cur_spm_size->Normalize();
      EmitCudaFuncAttributeMaxDynamicSharedMemorySize();
      Note(n.LOC(),
           "In the current kernel `" + device_fn +
               "`, cudaFuncAttributeMaxDynamicSharedMemorySize is set, cause "
               "shared memory usage has exceeded the default limit 48KB.");
    } else {
      // add the size of static shared
      auto mri = FCtx(fname).GetStaticMemReuseInfo(dev_name);
      if (mri) {
        // 48KB is the largest capacity that static shared memory supports.
        if (mri->infos[Storage::SHARED].spm_size > 48 * 1024) {
          cur_ring_offset = sbe::nu(mri->infos[Storage::SHARED].spm_size);
          cur_spm_size = cur_spm_size + cur_ring_offset;
          cur_spm_size = cur_spm_size->Normalize();
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
      hs << ", " << ValueSTR(cur_spm_size);
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

    if (!cur_ring_offset->IsNumeric()) hs << ", " << ValueSTR(cur_ring_offset);

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
    EmitDeviceFuncDecl(ds, &n, cur_ring_offset);
    ds << " {\n";
    IncrDeviceIndent();
    if (!(sbe::ceq(cur_spm_size, sbe::nu(0)) &&
          sbe::ceq(cur_ring_offset, sbe::nu(0)))) {
      ds << d_indent << "extern __shared__ char " << device_fn
         << "__runtime_shared_buffer__[];\n";
      if (!sbe::ceq(cur_spm_size, sbe::nu(0))) {
        ds << d_indent << "auto " << device_fn
           << "__ring__ = reinterpret_cast<choreo::future_ring<6>*>(&"
           << device_fn
           << "__runtime_shared_buffer__[" + ValueSTR(cur_ring_offset)
           << "]);\n";
        ds << d_indent << "for (int i = 0; i < " << ValueSTR(cur_ring_size)
           << "; ++i)\n";
        ds << d_indent << "  (" << device_fn << "__ring__ + i)->init();\n";
      }

    } else
      ds << d_indent << "auto " << device_fn << "__ring__ = nullptr;\n";
    ds << d_indent << "{ // parallel-by: " << n.LOC() << "\n";
  } else {
    auto& siblings = cgi.GetPBTree(fname).GetSiblings(&n);
    if (!siblings.empty()) {
      ds << d_indent << "{ // inner parallel-by: " << n.LOC() << "\n";
      IncrDeviceIndent();
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

  auto claimFuture = [this,
                      &n](const std::string& buf_expr, bool is_async,
                          bool is_tma = false,
                          const std::string& mdata_expr = "") -> std::string {
    if (!n.future.empty() && claimed_futs.count(InScopeName(n.future)))
      return n.future;

    auto cp_atom = GetCopyAtomName();
    // claim the date transfer engine
    if (is_tma) {
      ds << d_indent << "__shared__ cuda::barrier<cuda::thread_scope_block> "
         << cp_atom << "_barrier;\n";
      ds << d_indent << "if (__CHOREO_BLOCK_SINGLE__) {\n";
      ds << d_indent << "  init(&" << cp_atom
         << "_barrier, blockDim.x * blockDim.y * blockDim.z);\n";
      ds << d_indent << "  cde::fence_proxy_async_shared_cta();\n";
      ds << d_indent << "}\n";
      ds << d_indent << "__syncthreads();\n";
      ds << d_indent << "TMAAtom " << cp_atom << "{&" << cp_atom
         << "_barrier};\n";
    } else if (is_async) {
      ds << d_indent << "AsyncCopyAtom " << cp_atom << "{};\n";
    }

    auto future_name = n.future;
    static size_t future_count = 0;

    if (future_name.empty()) {
      future_name = "__choreo_anon_fut__" + std::to_string(future_count);
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
    future_count++;
    ds << d_indent << "future " << future_name << "(\"" << n.future << "\", "
       << n.LOC().begin.line << ", " << n.LOC().begin.column;
    if (!buf_expr.empty()) ds << ", " << buf_expr;
    if (!mdata_expr.empty()) ds << ", " << mdata_expr;
    ds << ");\n";
    if (is_tma) {
      ds << d_indent << future_name << ".is_tma = true;\n";
      ds << d_indent << future_name << ".set_atom(&" << cp_atom << ");\n";
    } else if (is_async) {
      ds << d_indent << future_name << ".set_atom(&" << cp_atom << ");\n";
      ds << d_indent << future_name << ".set_ring(" << device_fn
         << "__ring__);\n";
      ds << d_indent << future_name << ".id = " << future_count << ";\n";
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
  if (fty->IsAsync()) assert(!n.future.empty());

  auto f_ca = cast<AST::ChunkAt>(n.from);
  auto t_ca = cast<AST::ChunkAt>(n.to);
  auto f_sym = f_ca->data->name;
  auto t_sym = t_ca->data->name;
  auto f_idx = f_ca->indices;
  auto t_idx = t_ca->indices;
  auto f_ty = GetSymbolType(f_sym);
  auto t_ty = GetSymbolType(t_sym);
  auto f_sty = GetSpannedType(f_ty);
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
    if (n.operation == ".copy") {
      if (SymbolToSymbol()) {
        // direct copy
        hs << h_indent << bts << " * " << buf_sym << " = " << buf_sym_from
           << ";\n";
      } else if (SymbolToTile()) {
        static int s_cnt = 0;
        auto off_name = "__slice_offset" + std::to_string(s_cnt++) + "__" +
                        f_sym + "_2_" + t_sym;
        auto [offset, offcnt] = GenMdsOffset(t_ca);
        hs << h_indent << "int " << off_name << " = " << offset << ";\n";
        hs << h_indent << bts << " * " << buf_sym << " + " << off_name << " = "
           << buf_sym_from << ";\n";
      } else if (TileToSymbol()) {
        static int s_cnt = 0;
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
        auto array_sizes = array_ty->Dimensions();
        for (size_t i = 0; i < subscriptions.size(); ++i) {
          if (array_idx.empty())
            array_idx = ExprSTR(subscriptions[i], IsHost());
          else
            array_idx = "(" + array_idx + ")*" +
                        std::to_string(array_sizes[i]) + "+" +
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

  // bind the data to the future
  if (bind_data)
    future_name =
        claimFuture(t_buf.second, fty->IsAsync(), use_tma, bound_mdata_expr);
  else
    future_name = claimFuture("", fty->IsAsync(), use_tma, "");

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
    } else
      f_mds_offset = ValueSTR(GenOffset(f_ca));

    if (auto idx = t_ca->IndexOfLastSpanAs()) {
      t_mds_offset = TileBaseOffset(t_ca);
      t_shape = t_ca->OpAt(*idx)->GetBlockShape();
    } else
      t_mds_offset = ValueSTR(GenOffset(t_ca));

    std::vector<size_t> transp_config;
    if (n.operation == ".transp")
      transp_config = cast<TransposeConfig>(n.GetConfig())->dim_values;

    auto f_stride = GenStrides(f_ca, transp_config);
    auto t_stride = GenStrides(t_ca);

    // Determine if we should use WGMMA layout for destination tensor
    bool use_wgmma_layout_t = HasWGMMAInFunction() &&
                              t_sty->GetStorage() == Storage::SHARED &&
                              (t_sty->ElementType() == BaseType::F16 ||
                               t_sty->ElementType() == BaseType::BF16);

    // Use swizzle value only if explicitly specified, otherwise use 0 (no
    // swizzle)
    int swizzle_value = n.IsSwizzleExplicit() ? n.GetSwizzleValue() : 0;

    const auto f_mds = GenTensorDecl(
        RemoveSuffix(f_buf_name, ".data()"), f_buf_name, f_sty->GetStorage(),
        f_sty->ElementType(),
        (n.operation == ".pad" ? f_ca->GetBlockShape() : fty->GetShape()),
        false, f_mds_offset, ValueSTR(f_stride, false, true), {}, false);
    const auto t_mds = GenTensorDecl(
        RemoveSuffix(t_buf_name, ".data()"), t_buf_name, t_sty->GetStorage(),
        t_sty->ElementType(), fty->GetShape(), false, t_mds_offset,
        ValueSTR(t_stride, false, true), {}, use_wgmma_layout_t, swizzle_value);

    std::string f_mds_name{f_mds.first};
    std::string f_mds_decl{f_mds.second};
    std::string t_mds_name{t_mds.first};
    std::string t_mds_decl{t_mds.second};

    ds << f_mds_decl;
    ds << t_mds_decl;

    // handles dma related to shared memory
    // For async cp.async (non-TMA), threads in the warp must participate;
    // avoid the block-single guard so every lane issues copy/trigger.
    bool need_single_instance = !ThreadCooperative(n);
    if (fty->IsAsync() && !n.IsTMA()) need_single_instance = false;
    bool is_subbyte_copy = (n.operation == ".copy") && !n.IsSparse() &&
                           (IsFloatSubByteType(f_sty->ElementType()) ||
                            IsFloatSubByteType(t_sty->ElementType()));
    bool need_subbyte_async_sync = false;

    if (need_single_instance) ds << d_indent << LevelPred() << " {\n";
    IncrDeviceIndent();
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
          ds << d_indent << future_name << ".trigger();\n";
          if (need_single_instance) need_subbyte_async_sync = true;
        } else {
          ds << d_indent << "cute::copy(" << f_byte.first << ", "
             << t_byte.first << ");\n";
        }
      } else if (fty->IsAsync()) {
        ds << d_indent << "cute::copy(*(AsyncCopyAtom*)" << future_name
           << ".get_atom(), " << f_mds_name << ", " << t_mds_name << ");\n";
        ds << d_indent << future_name << ".trigger();\n";
      } else {
        ds << d_indent << "opt_copy(" << f_mds_name << ", " << t_mds_name
           << ");\n";
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
         << ".layout()(" << "cute::make_coord(" << pcmvSTR(pad_config->pad_low)
         << "));\n";
      const auto t_pad_mds = GenTensorDecl(
          RemoveSuffix(t_buf_name, ".data()"), t_buf_name, t_sty->GetStorage(),
          t_sty->ElementType(), f_ca->GetBlockShape(), false, pad_offset,
          ValueSTR(t_stride, false, true));
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

    DecrDeviceIndent();
    if (need_single_instance) ds << d_indent << "} // single instance\n";

    if (need_subbyte_async_sync) { ds << d_indent << "__syncthreads();\n"; }

    if (!fty->IsAsync()) {
      // not async, must syncthreads immediately
      // else, defer the sync till the wait time
      ds << d_indent << "__syncthreads();\n";
    }
  };

  auto TMACodeGen = [&]() {
    if (n.operation != ".copy")
      choreo_unreachable("unsupported tma operation: " + n.operation + ".");
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
      ds << d_indent << "if (__CHOREO_BLOCK_SINGLE__) {\n";
      ds << d_indent << "  cde::cp_async_bulk_tensor_" << t_shape.Rank()
         << "d_global_to_shared(" << t_buf_expr << ", &" << *tname
         << "_tensor_map, " << ValueSTR(Reverse(GenIndices(f_ca)))
         << ", ((TMAAtom*)" << future_name << ".get_atom())->barrier());\n";
      ds << d_indent << "  ((TMAAtom*)" << future_name
         << ".get_atom())->token() = "
            "cuda::device::barrier_arrive_tx(((TMAAtom*)"
         << future_name << ".get_atom())->barrier(), 1, "
         << ValueSTR(t_sty->ByteSizeValue()) << ");\n";
      ds << d_indent << "} else {\n";
      ds << d_indent << "  ((TMAAtom*)" << future_name
         << ".get_atom())->token() = ((TMAAtom*)" << future_name
         << ".get_atom())->barrier().arrive();\n";
      ds << d_indent << "}\n";

      // For async tma.copy.async, trigger the future
      // For sync tma.copy, directly wait and set_nowait
      if (fty->IsAsync()) {
        ds << d_indent << future_name << ".trigger();\n";
      } else {
        // Synchronous tma.copy: wait immediately
        // Make sure the future is marked initialized before marking it nowait
        // to avoid runtime diagnostics when the state is still ST_NONE.
        ds << d_indent << "(void)" << future_name << ".get_atom();\n";
        ds << d_indent << "((TMAAtom*)" << future_name
           << ".get_atom())->barrier().wait(std::move(((TMAAtom*)"
           << future_name << ".get_atom())->token()));\n";
        ds << d_indent << future_name << ".set_nowait();\n";
      }
    } else if ((tsto == Storage::GLOBAL || tsto == Storage::DEFAULT) &&
               fsto == Storage::SHARED) {
      ds << d_indent << "cde::fence_proxy_async_shared_cta();\n";
      ds << d_indent << "__syncthreads();\n";
      ds << d_indent << "if (__CHOREO_BLOCK_SINGLE__) {\n";
      ds << d_indent << "  cde::cp_async_bulk_tensor_" << t_shape.Rank()
         << "d_shared_to_global(&" << *tname << "_tensor_map, "
         << ValueSTR(Reverse(GenIndices(t_ca))) << ", " << f_buf_expr << ");\n";
      ds << d_indent << "  cde::cp_async_bulk_commit_group();\n";
      ds << d_indent << "  cde::cp_async_bulk_wait_group_read<0>();\n";
      ds << d_indent << "}\n";
      // DO not check or wait. TMA share=>global is special
    }
  };

  if (use_tma)
    TMACodeGen();
  else
    DMACodeGen();

  return true;
}

bool CuteCodeGen::Visit(AST::MMA& n) {
  auto& op = *n.GetOperation();
  std::string scoped_frag_name = InScopeName(op.GetFragSym());
  if (!FCtx(fname).FragHasMMAType(scoped_frag_name)) {
    Error1(n.LOC(), "the MMA operation of `" + scoped_frag_name +
                        "` cannot be executed.");
    choreo_unreachable(
        "MMA information is incomplete (maybe lack of mma exec).");
  }

  if (FCtx(fname).FragIsWGMMA(scoped_frag_name)) {
    // WGMMA codegen path (128-thread warp group) using PTX inline assembly
    auto& ssmi = cgi.GetSymbolMMA(scoped_frag_name);
    // Determine accumulator type: f32 for f16->f32, f16 for f16->f16
    std::string accum_type = (ssmi.ty == BaseType::F16) ? "f16" : "f32";
    switch (op.Tag()) {
    case AST::MMAOperation::Fill: {
      auto sym = op.FillingSymbol();
      // dtype of accu: s32, f16, f32 (f16 => u32)
      auto acc_dtype = ssmi.ty;
      ValueItem frag_len = sbe::bop(OpCode::DIVIDE, ssmi.shape[1], sbe::nu(2))
                               ->Normalize(); // N / 2
      if (ssmi.ty == BaseType::F16) {
        acc_dtype = BaseType::U32;
        frag_len = sbe::bop(OpCode::DIVIDE, frag_len, sbe::nu(2))->Normalize();
      }
      if (!VIIsInt(frag_len))
        choreo_unreachable(
            "expect the length of wgmma fragment to be integer but not symbol");
      reg_num_d = *VIInt(frag_len);

      ds << d_indent << NameBaseType(acc_dtype) << " " << sym << "_frag["
         << reg_num_d << "];\n";
      // TODO: only support init with 0 for now
      ds << d_indent << "memset(" << sym << "_frag, 0, sizeof(" << sym
         << "_frag));\n";
      // Signal warp group that we're about to start WGMMA operations
      ds << d_indent << "warpgroup_arrive();\n";
      ssm.MapDeviceSymbol(InScopeName(sym), sym + "_frag");
    } break;
    case AST::MMAOperation::Load: {
      auto sym = op.LoadTo();
      // For WGMMA, we store the shared memory pointer for later use in Exec
      // The actual data should already be in shared memory
      std::string elem_ty = NameBaseType(ssmi.ty);
      ds << d_indent << elem_ty << "* " << sym << "_smem_ptr = (" << elem_ty
         << "*)(" << ExprSTR(op.LoadFrom(), false) << ");\n";
      [[maybe_unused]] bool frag_is_fp8 =
          ssmi.ty == BaseType::F8_E4M3 || ssmi.ty == BaseType::F8_E5M2 ||
          ssmi.ty == BaseType::F8_UE4M3 || ssmi.ty == BaseType::F8_UE8M0;
      std::string major_order = "WGMMA_MajorOrder::MN_MAJOR";
      if (ssmi.frag == MMAInfo::FRAG_A) {
        if (ssmi.method == AST::MMAOperation::ROW_ROW ||
            ssmi.method == AST::MMAOperation::ROW_COL)
          major_order = "WGMMA_MajorOrder::K_MAJOR";
      } else if (ssmi.frag == MMAInfo::FRAG_B) {
        if (ssmi.method == AST::MMAOperation::ROW_ROW ||
            ssmi.method == AST::MMAOperation::COL_ROW)
          major_order = "WGMMA_MajorOrder::K_MAJOR";
      }
      // Get swizzle value from MMA operation (default 128)
      int swizzle_val = op.GetSwizzleValue();
      std::string swizzle_enum;
      switch (swizzle_val) {
      case 0: swizzle_enum = "WGMMA_Swizzle::NS"; break;
      case 32: swizzle_enum = "WGMMA_Swizzle::B32"; break;
      case 64: swizzle_enum = "WGMMA_Swizzle::B64"; break;
      case 128: swizzle_enum = "WGMMA_Swizzle::B128"; break;
      default: swizzle_enum = "WGMMA_Swizzle::B128"; break;
      }
      ds << d_indent << "uint64_t desc_" << sym << " = wgmma_make_smem_desc<"
         << major_order << ", " << swizzle_enum << ">(" << sym
         << "_smem_ptr);\n";
      if (ssmi.frag == MMAInfo::FRAG_A) {
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
      ssm.MapDeviceSymbol(InScopeName(sym), sym + "_frag");
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
      auto c_sym = op.ExecOperand(0);
      auto a_sym = op.ExecOperand(1);
      auto b_sym = op.ExecOperand(2);
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
      if (policy_is_sparse) {
        auto& ssmi_a = cgi.GetSymbolMMA(InScopeName(a_sym));
        std::string meta_var = a_sym + "_meta";
        std::string meta_ptr = a_sym + "_mdata_ptr";
        auto k_val = VIInt(ssmi_a.shape.at(2));
        bool meta_64 = true;
        if (k_val) meta_64 = (*k_val > 32);
        std::string meta_ty = meta_64 ? "uint64_t" : "uint32_t";
        ds << d_indent << meta_ty << " " << meta_var << " = 0;\n";
        ds << d_indent << "{\n";
        ds << d_indent << "  int __sp_tid = threadIdx.x % 128;\n";
        ds << d_indent << "  int __sp_lane = __sp_tid % 32;\n";
        ds << d_indent << "  int __sp_warp = __sp_tid / 32;\n";
        ds << d_indent
           << "  int __sp_row = __sp_warp * 16 + (__sp_lane / 4);\n";
        ds << d_indent << "  constexpr int __sp_K = " << STR(ssmi_a.shape.at(2))
           << ";\n";
        ds << d_indent << "  constexpr int __sp_chunks = __sp_K / 4;\n";
        ds << d_indent << "  " << meta_ty << " __sp_meta = 0;\n";
        ds << d_indent << "  #pragma unroll\n";
        ds << d_indent << "  for (int k4 = 0; k4 < __sp_chunks; ++k4) {\n";
        ds << d_indent << "    uint8_t mask = " << meta_ptr
           << "[__sp_row * __sp_chunks + k4];\n";
        ds << d_indent << "    int pos0 = 0, pos1 = 0, found = 0;\n";
        ds << d_indent << "    #pragma unroll\n";
        ds << d_indent << "    for (int p = 0; p < 4; ++p) {\n";
        ds << d_indent << "      if (mask & (1 << p)) {\n";
        ds << d_indent
           << "        if (found == 0) pos0 = p; else if (found == 1) pos1 = "
              "p;\n";
        ds << d_indent << "        ++found;\n";
        ds << d_indent << "      }\n";
        ds << d_indent << "    }\n";
        ds << d_indent << "    if (found < 2) { pos1 = pos0; }\n";
        ds << d_indent
           << "    uint32_t nibble = (pos0 & 0x3) | ((pos1 & 0x3) << 2);\n";
        ds << d_indent << "    __sp_meta |= (static_cast<" << meta_ty
           << ">(nibble) << (4 * k4));\n";
        ds << d_indent << "  }\n";
        ds << d_indent << "  " << meta_var << " = __sp_meta;\n";
        ds << d_indent << "}\n";
      }
      ds << d_indent << "cute::" << mma_policy << "<";
      if (!policy_is_tn) {
        ds << cute_gmma_major_cast << "(" << trans_a << "), "
           << cute_gmma_major_cast << "(" << trans_b << ")";
      }
      ds << ">::fma(" << "desc_" << a_sym << ", desc_" << b_sym;
      for (size_t i = 0; i < reg_num_d; ++i)
        ds << ", " << c_sym << "_frag[" << i << "]";
      if (policy_is_sparse) ds << ", " << a_sym << "_meta";
      ds << ");\n";
    } break;
    case AST::MMAOperation::Store: {
      ds << d_indent << "// Finalize WGMMA operations\n";
      ds << d_indent << "warpgroup_commit_batch();\n";
      ds << d_indent << "warpgroup_wait<0>();\n";
      auto ca = op.StoreTo();
      auto f_sym = ca->data->name;
      auto ty = GetSymbolType(f_sym);
      auto f_sty = GetSpannedType(ty);
      auto accum_type = ssmi.ty;
      const auto f_mds = GenTensorDecl(
          RemoveSuffix(f_sym, ".data()"),
          (isa<FutureType>(ty) ? f_sym + ".data()" : f_sym),
          f_sty->GetStorage(), f_sty->ElementType(), ca->GetBlockShape(), false,
          ValueSTR(GenOffset(ca)), ValueSTR(GenStrides(ca), false, true));
      ds << f_mds.second;
      auto sym = op.StoreFrom();
      std::string DIM_N = STR(ssmi.shape.at(1));
      std::string CUTE_WGMMA_ATOM =
          "CUTE_WGMMA_M" + STR(ssmi.shape.at(0)) + "K" + STR(ssmi.shape.at(2));

      ds << d_indent << "store_fragment_d<" << CUTE_WGMMA_ATOM << ", " << DIM_N
         << ">(" << f_mds.first << ", " << "reinterpret_cast<"
         << NameBaseType(accum_type) << "*>(" << sym << "_frag));\n";
    } break;
    default: break;
    }
  } else if (FCtx(fname).FragIsWMMA(scoped_frag_name)) {
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
      auto sym = op.FillingSymbol();
      auto& ssmi = cgi.GetSymbolMMA(InScopeName(sym));
      auto sty = GetSpannedType(GetSymbolType(sym));
      assert(sty);
      ds << d_indent
         << "nvcuda::wmma::fragment<nvcuda::wmma::" << FragSTR(ssmi.frag)
         << ", ";
      ds << ValueSTR(ssmi.shape) << ", " << NameBaseType(ssmi.ty) << "> " << sym
         << "_frag;\n";
      ds << d_indent << "nvcuda::wmma::fill_fragment(" << sym << "_frag, ("
         << NameBaseType(ssmi.ty) << ")" << ExprSTR(op.FillingValue(), false)
         << ");\n";
      ssm.MapDeviceSymbol(InScopeName(sym), sym + "_frag");
    } break;
    case AST::MMAOperation::Load: {
      auto sym = op.LoadTo();
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
           << wmma_major << "> " << sym << "_frag;\n";

        ds << d_indent << "nvcuda::wmma::load_matrix_sync(" << sym << "_frag, "
           << ExprSTR(op.LoadFrom(), false) << ", "
           << ValueSTR(fty->GetShape().ValueAt(fty->GetShape().Rank() - 1))
           << ");\n";
        ssm.MapDeviceSymbol(InScopeName(sym), sym + "_frag");
      } else if (ssmi.frag == MMAInfo::FRAG_C) {
        ds << d_indent
           << "nvcuda::wmma::fragment<nvcuda::wmma::" << FragSTR(ssmi.frag)
           << ", ";
        ds << ValueSTR(ssmi.shape) << ", " << NameBaseType(ssmi.ty) << "> "
           << sym << "_frag;\n";
        ds << d_indent << "nvcuda::wmma::load_matrix_sync(" << sym << "_frag, "
           << ExprSTR(op.LoadFrom(), false) << ", "
           << ValueSTR(fty->GetShape().ValueAt(fty->GetShape().Rank() - 1))
           << ", nvcuda::wmma::mem_row_major);\n";
      } else {
        choreo_unreachable("unexpect MMA frag");
      }
    } break;
    case AST::MMAOperation::Exec: {
      ds << d_indent << "nvcuda::wmma::mma_sync(" << op.ExecOperand(0)
         << "_frag, " << op.ExecOperand(1) << "_frag, " << op.ExecOperand(2)
         << "_frag, " << op.ExecOperand(0) << "_frag);\n";
    } break;
    case AST::MMAOperation::Store: {
      auto tty = GetSpannedType(GetSymbolType(op.StoreTo()->RefSymbol()));
      ds << d_indent << "nvcuda::wmma::store_matrix_sync("
         << ExprSTR(op.StoreTo(), false) << ", " << op.StoreFrom() << "_frag, "
         << ValueSTR(tty->GetShape().ValueAt(1))
         << ", nvcuda::wmma::mem_row_major);\n";
    } break;
    default: break;
    }
  } else if (FCtx(fname).FragIsCTMMA(scoped_frag_name)) {
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
      // TODO: shall we split the op to two diff ops: fragment decl and fill?
      // for example: decl; fill; use; store; fill; use again;
      auto sym = op.FillingSymbol();
      auto& ssmi = cgi.GetSymbolMMA(InScopeName(sym));
      assert(ssmi.ty != BaseType::UNKNOWN);
      auto sty = GetSpannedType(GetSymbolType(sym));
      assert(sty);
      reg_num_d = GetRegNumOfFrag(sty->GetShape().ValueAt(0),
                                  sty->GetShape().ValueAt(1));
      bool use_uint32 = false;
      UseUint32Reg(use_uint32, reg_num_d, ssmi.ty);
      RegNumOf8x8x4(ssmi.shape, ssmi.ty, MMAInfo::FRAG_C, reg_num_d);
      ds << d_indent << (use_uint32 ? "uint32_t" : NameBaseType(ssmi.ty)) << " "
         << sym << "_frag[" << reg_num_d << "] ;\n";
      ds << d_indent << "memset(" << sym << "_frag, 0, sizeof(" << sym
         << "_frag));\n";

    } break;
    case AST::MMAOperation::Load: {
      auto ca = op.LoadFrom();
      auto f_sym = ca->data->name;
      auto ty = GetSymbolType(f_sym);
      auto f_sty = GetSpannedType(ty);
      const auto f_mds = GenTensorDecl(
          RemoveSuffix(f_sym, ".data()"),
          (isa<FutureType>(ty) ? f_sym + ".data()" : f_sym),
          f_sty->GetStorage(), f_sty->ElementType(), ca->GetBlockShape(), false,
          ValueSTR(GenOffset(ca)), ValueSTR(GenStrides(ca), false, true));
      ds << f_mds.second;
      auto sym = op.LoadTo();
      auto ssmi = cgi.GetSymbolMMA(InScopeName(sym));
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
        ds << d_indent << "auto " << sym << "_frag = load_fragment_"
           << frag_suffix << "<" << CUTE_MMA_ATOM << ">(" << f_mds.first
           << ");\n";

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
        ds << d_indent << (use_uint32 ? "uint32_t" : NameBaseType(ssmi.ty))
           << " " << sym << "_frag[" << reg_num_d << "] ;\n";
        ds << d_indent << "load_fragment_d<" << CUTE_MMA_ATOM << ">("
           << f_mds.first << ", " << "reinterpret_cast<"
           << NameBaseType(ssmi.ty) << "*> (" << sym << "_frag));\n";
      } else {
        choreo_unreachable("unexpect MMA frag");
      }
    } break;
    case AST::MMAOperation::Exec: {
      auto c_sym = op.ExecOperand(0);
      auto a_sym = op.ExecOperand(1);
      auto b_sym = op.ExecOperand(2);
      auto e_sym_provided = op.ExecOperand(3);

      std::string mma_policy = FCtx(fname).MMAPolicyOfFrag(InScopeName(c_sym));
      bool policy_is_sparse = mma_policy.find("SPARSE") != std::string::npos;
      std::string meta_var;
      if (policy_is_sparse) {
        if (!e_sym_provided.empty()) {
          meta_var = e_sym_provided + "_frag";
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
        ds << c_sym << "_frag[" << i << "], ";
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

      for (size_t i = 0; i < reg_num_a; ++i)
        ds << a_sym << "_frag[" << i << "], ";
      for (size_t i = 0; i < reg_num_b; ++i)
        ds << b_sym << "_frag[" << i << "], ";
      for (size_t i = 0; i < reg_num_d; ++i) {
        ds << c_sym << "_frag[" << i << "]";
        if (i != reg_num_d - 1) ds << ", ";
      }
      if (policy_is_sparse) ds << ", " << meta_var << ", 0";
      ds << ");\n";
    } break;
    case AST::MMAOperation::Store: {
      auto ca = op.StoreTo();
      auto f_sym = ca->data->name;
      auto ty = GetSymbolType(f_sym);
      auto f_sty = GetSpannedType(ty);
      const auto f_mds = GenTensorDecl(
          RemoveSuffix(f_sym, ".data()"),
          (isa<FutureType>(ty) ? f_sym + ".data()" : f_sym),
          f_sty->GetStorage(), f_sty->ElementType(), ca->GetBlockShape(), false,
          ValueSTR(GenOffset(ca)), ValueSTR(GenStrides(ca), false, true));
      ds << f_mds.second;
      auto sym = op.StoreFrom();
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
      ds << d_indent << "store_fragment_d<" << CUTE_MMA_ATOM << ">("
         << f_mds.first << ", " << "reinterpret_cast<" << NameBaseType(ssmi.ty)
         << "*> (" << sym << "_frag));\n";
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

  ds << d_indent << "choreo::rotate(";
  int i = 0;
  for (auto& id : n.GetIds()) {
    assert(isa<FutureType>(NodeType(*id)) &&
           "only rotating futures are supported.");
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

  for (auto& t : n.GetTargets()) {
    auto tty = NodeType(*t);
    auto expr = cast<AST::Expr>(t);
    bool is_array_ref = (expr->op == "elemof");

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
      case Storage::SHARED:
      case Storage::LOCAL: {
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
      case Storage::SHARED:
      case Storage::LOCAL: {
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

bool CuteCodeGen::Visit(AST::Trigger& n) {
  TraceEachVisit(n);

  for (auto& f : n.GetEvents()) {
    auto expr = cast<AST::Expr>(f);
    bool is_array_ref = (expr->op == "elemof");
    assert(IsSymbolOrArrayRef(*f) &&
           "expect either symbol or array reference.");
    if (auto ety = dyn_cast<EventArrayType>(NodeType(*f))) {
      if (IsHost()) {
        assert(ety->GetStorage() == Storage::GLOBAL);
        // TODO: make & into OpExprSTR?
        hs << h_indent << "choreo::abend_true(cudaMemset(&" << ExprSTR(f, true)
           << ", 1, " << ety->ElemCount() << ")); // trigger event\n";
        // TODO: support array reference
      } else {
        switch (ety->GetStorage()) {
        case Storage::GLOBAL:
        case Storage::SHARED:
        case Storage::LOCAL:
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
          break;
        default:
          choreo_unreachable("unsupported event array storage '" +
                             STR(ety->GetStorage()) + "' to trigger.");
          break;
        }
      }
    } else if (auto ety = dyn_cast<EventType>(NodeType(*f))) {
      if (IsHost()) {
        assert(ety->GetStorage() == Storage::GLOBAL);
        hs << h_indent << "choreo::abend_true(cudaMemset(&" << ExprSTR(f, true)
           << ", 1, 1)); // trigger event\n";
        // TODO: support array reference
      } else {
        switch (ety->GetStorage()) {
        case Storage::GLOBAL:
        case Storage::SHARED:
        case Storage::LOCAL:
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
          break;
        default:
          choreo_unreachable("unsupported event array storage '" +
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
          for (const auto& arg_str : SplitStringByDelimiter(args_str, ", "))
            print_args += "static_cast<long long>(" + arg_str + "), ";
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
      if (stream_name != "") Error1(n.LOC(), "Only one stream supported now!");
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
  // anything required?
  return true;
}

bool CuteCodeGen::Visit(AST::ForeachBlock& n) {
  TraceEachVisit(n);

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

  return true;
}

bool CuteCodeGen::Visit(AST::InThreadsBlock& n) {
  TraceEachVisit(n);
  assert(!IsHost());
  ds << d_indent << "// inthreads: " << n.LOC() << "\n";
  if (!n.stmts->None())
    ds << d_indent << "if (" << ExprSTR(n.pred, false) << ") {\n";
  IncrDeviceIndent();
  return true;
}

bool CuteCodeGen::Visit(AST::IfElseBlock& n) {
  TraceEachVisit(n);

  IndStream() << "// if-else: " << n.LOC() << "\n";
  if (auto c = dyn_cast<AST::Call>(n.pred))
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
               expr && (expr->op == "dataof" || expr->op == "mdataof")) {
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
        << HostTypeStringify(*item.type, false, item.IsReference()) << " "
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
    }
  }

  hs << "\n";

  for (const auto& rc : FCtx(fname).GetRtChecks()) {
    hs << h_indent << "choreo::runtime_check(" << ValueSTR(sbe::sym(rc.lhs))
       << " " << rc.op << " " << ValueSTR(sbe::sym(rc.rhs)) << ", \""
       << rc.message << ", " << rc.loc << "\");\n";
  }

  for (const auto& ar : FCtx(fname).GetAssertions()) {
    hs << h_indent << "choreo::runtime_check(" << ValueSTR(ar.expr, true)
       << ", \"" << ar.message << ", " << ar.loc << "\");\n";
  }
}

void CuteCodeGen::EmitMemReuse(const std::string& df_name) {
  const auto& mri = FCtx(fname).GetDynMemReuseInfo(df_name);
  if (!mri) return;
  hs << h_indent << R"(// JIT memory reuse begin)" << "\n";
  for (const auto& [sto, ie] : mri->infos) {
    hs << h_indent << "HeapSimulator::Chunks " << ie.chunks_name << ";\n";
    for (const auto& c : ie.chunks)
      hs << h_indent << ie.chunks_name << ".push_back(" << c << ");\n";
  }
  hs << h_indent << "HeapSimulator " << mri->simulator << ";\n";
  for (const auto& [sto, ie] : mri->infos) {
    hs << h_indent << "HeapSimulator::Result " << ie.result << " = "
       << mri->simulator << ".Allocate(" << ie.chunks_name << ", 512);\n";
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
         << "bytes.\");\n";
    hs << h_indent << "unsigned long " << ie.offsets_name << "["
       << mri->infos[sto].offset_args.size() << "];" << "\n";
    std::string idx = ie.chunks_name + "_idx";
    hs << h_indent << "size_t " << idx << " = 0;\n";
    hs << h_indent << "for (const auto& [buffer_id, offset] : " << ie.result
       << ".chunk_offsets)\n";
    hs << h_indent << "  " << ie.offsets_name << "[" << idx
       << "++] = offset;\n";
  }
  hs << h_indent << R"(// JIT memory reuse end)" << "\n";
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
    auto t_shape = g_ca->GetBlockShape();
    auto map_name = desc.GetName() + "_tensor_map";

    // Convert swizzle value to TMA_Swizzle enum and get CUtensorMapSwizzle
    // string
    int swizzle_val = desc.GetSwizzleValue();
    TMA_Swizzle tma_swizzle;
    switch (swizzle_val) {
    case 32: tma_swizzle = TMA_Swizzle::B32; break;
    case 64: tma_swizzle = TMA_Swizzle::B64; break;
    case 128: tma_swizzle = TMA_Swizzle::B128; break;
    default: tma_swizzle = TMA_Swizzle::NONE; break;
    }
    std::string cu_swizzle_str = cuda_stringify(tma_swizzle);

    hs << h_indent << "uint64_t " << desc.GetName() << "_shape[] = {"
       << ValueSTR(Reverse(g_shape.Value())) << "};\n"; // shape of buffer
    // For TMA, strides should be in the same order as shape (not reversed)
    hs << h_indent << "uint64_t " << desc.GetName() << "_strides[] = {"
       << ValueSTR(
              Trim(Reverse(GenStrides(g_shape) * gmem_ty->ElementSizeValue())))
       << "};\n"; // strides of shape
    hs << h_indent << "uint32_t " << desc.GetName() << "_box_shape[] = {"
       << ValueSTR(Reverse(t_shape.Value())) << "};\n"; // shape of tile block
    hs << h_indent << "uint32_t " << desc.GetName() << "_elem_strides[] = {"
       << ValueSTR(ValxN(sbe::nu(1), t_shape.Rank()))
       << "};\n"; // elements' strides
    std::string g_unscoped = UnScopedName(g_sym);
    std::string g_scoped = InScopeName(g_unscoped);

    // Determine whether this tensor is a true GLOBAL argument.
    // Correct behavior is to consult the parameter attribute
    // (ParamAttr::GLOBAL_INPUT) when the symbol corresponds to a
    // function parameter. Do NOT treat choreo output as global by
    // default, because output may be shadowed to device memory.
    bool is_global_arg = false;
    bool found_param = false;
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
  else if (auto bitt = dyn_cast<BoundedITupleType>(&ty)) {
    // There should have no BoundedIntegerType.
    // They have been normalized to BoundedITupleType.
    assert(bitt->Dims() == 1);
    (void)bitt;
    return "int";
  } else
    choreo_unreachable("unsupported host function type: " + STR(ty) + ".");
  return "";
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
                                     const ValueItem& cur_ring_offset) {
  oss << "__global__ void " << device_fn << "(";

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

  if (!cur_ring_offset->IsNumeric())
    oss << ", unsigned " << ValueSTR(cur_ring_offset);

  oss << ")";

  VST_DEBUG(dbgs() << "Device function prototype:\n" << oss.str() << "\n");
}

void CuteCodeGen::EmitSource() {
  for (auto& code : code_segments) outs() << code << "\n";
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

  auto build_path = CreateUniquePath();
  auto cc_file = build_path + "/__choreo_cute_" + filename + ".cu";
  auto exe_file = exe_fn;
  if (exe_file.empty())
    exe_file = build_path + "/__choreo_cute_" + filename + ".exe";
  os << "rm -fr " << build_path << "\n";
  os << "mkdir -p " << build_path << "\n\n";

  // place the choreo header
  os << "cat <<'EOF' > " << build_path << "/choreo.h\n";
  os << __choreo_header_as_string << "\nEOF\n\n";
  os << "cat <<'EOF' > " << build_path << "/choreo_cute.h\n";
  os << __choreo_cute_header_as_string << "\nEOF\n\n";

  os << "cat <<'EOF' > " << cc_file << "\n";
  for (auto& code : code_segments) os << code << "\n";
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
  echo ""
  echo "  Environment Variables:"
  echo "   CUDA_HOME:           (Must) Cuda compiler installation path"
  echo "   CUTE_HOME:           (Must) Cute header library path"
  echo "   EXTRA_TARGET_CFLAGS: Extra target compilation flags"
  exit 1
}

# compile, execute
)script";

  os << R"(export CFLAGS="-arch ${nv_arch} -std=c++17 -DCUTLASS_ENABLE_TENSOR_CORE_MMA=1 -D__CHOREO_TARGET_CUTE__ -Xcompiler -static-libstdc++ -lcuda)";
  if (CCtx().GenDebugInfo())
    os << " -O0";
  else
    os << " -O" << CCtx().GetOptimizationLevel();
  if (use_cuda_type)
    os << " -D__USE_CUDA_TYPE__";
  else
    os << " -D__USE_CUTE_TYPE__";

  if (CCtx().GenDebugInfo()) os << " -g -G";
  if (CCtx().DMADiagnosis()) os << " -D__CHOREO_DMA_DIAGNOSIS__";
  if (!target_options.GetValue().empty())
    os << " " << target_options.GetValue();
  if (use_pic) os << " -fPIC";
  if (verbose) os << " -v"; // if it requires to be verbose
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
  os << "\nelse show_usage";
  os << "\nfi";
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
const std::string
CuteCodeGen::ExprCastSTR(AST::ptr<AST::Node> n,
                         std::optional<std::variant<int, float>> val,
                         BaseType t, BaseType f, bool is_host) const {
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
  if (!IsValuePreservingCast(f, t)) {
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

  auto HandleChunkAt = [this, &WrapParen,
                        &parent_op](const ptr<AST::ChunkAt>& ca, bool is_host) {
    auto caty = cast<SpannedType>(ca->GetType());

    auto offset = GenOffset(ca);
    std::string res;
    if (auto fty = dyn_cast<FutureType>(NodeType(*ca->data))) {
      std::string ets = NameBaseType(fty->ElementType());
      res = "(" + ets + "*)" + OpExprSTR(ca->data, parent_op, true, is_host) +
            ".data()";
    } else
      res = OpExprSTR(ca->data, "+", true, is_host);

    if (!sbe::ceq(offset, sbe::nu(0))) res += " + " + ValueSTR(offset);

    return WrapParen(res, "+");
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
    assert(ce->GetOp() == "cast");
    return ExprCastSTR(ce->GetR(), std::nullopt, ce->ToType(), ce->FromType(),
                       is_host);
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
      if (auto ca = dyn_cast<AST::ChunkAt>(expr->GetR())) {
        return HandleChunkAt(ca, is_host);
      } else {
        return OpExprSTR(expr->GetReference(), parent_op, is_left_child,
                         is_host);
      }
    } else if (expr->IsUnary()) {
      if (expr->GetOp() == "!") {
        oss << "!"
            << WrapParen(OpExprSTR(expr->GetR(), "!", false, is_host), "!");
      } else if (expr->GetOp() == "ubound") {
        auto rty = cast<BoundedType>(NodeType(*expr->GetR()));
        if (rty->Dims() == 1) oss << ValueSTR(rty->GetUpperBound());
      } else if (expr->GetOp() == "dataof" || expr->GetOp() == "mdataof") {
        assert(isa<FutureType>(expr->GetR()->GetType()) &&
               "expect a future operand.");
        if (auto id = cast<AST::Expr>(expr->GetR())->GetSymbol()) {
          if (is_host)
            oss << id->name << "__buf__";
          else
            oss << id->name
                << (expr->GetOp() == "mdataof" ? ".mdata()" : ".data()");
        } else
          choreo_unreachable("Can not retrieve name of the future.");
      } else if (expr->GetOp() == "sizeof") {
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
      } else if (expr->GetOp() == "++") {
        oss << "++"
            << WrapParen(OpExprSTR(expr->GetR(), "++", false, is_host), "++");
      } else if (expr->GetOp() == "--") {
        oss << "--"
            << WrapParen(OpExprSTR(expr->GetR(), "--", false, is_host), "--");
      } else if (expr->GetOp() == "addrof") {
        if (auto id = AST::GetIdentifier(expr->GetR()))
          oss << OpExprSTR(id, parent_op, is_left_child, is_host);
        else if (isa<AST::DataAccess>(expr->GetR()))
          oss << "&"
              << WrapParen(OpExprSTR(expr->GetR(), "&", false, is_host), "&");
        else
          choreo_unreachable("Can not retrieve name of the spanned data.");
      } else if (expr->GetOp() == "~") {
        oss << "~"
            << WrapParen(OpExprSTR(expr->GetR(), "~", false, is_host), "~");
      } else
        choreo_unreachable("unsupported expression op: '" + expr->GetOp() +
                           "', expr: " + PSTR(expr) + ".");
    } else if (expr->IsBinary()) {
      if (expr->GetOp() == "cdiv") {
        std::string one = "1";
        auto L = OpExprSTR(expr->GetL(), "+", true, is_host);
        auto R0 = OpExprSTR(expr->GetR(), "+", false, is_host);
        auto R1 = OpExprSTR(expr->GetR(), "/", false, is_host);
        // (L + R0 - 1) / R1
        std::ostringstream res;
        res << "(" << L << " + " << R0 << " - " << one << ") / " << R1;
        oss << WrapParen(res.str(), "/");
      } else if (expr->GetOp() == "getith") {
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
      } else if (expr->GetOp() == "elemof") {
        oss << OpExprSTR(expr->GetL(), "[]", true, is_host) << "["
            << OpExprSTR(expr->GetR(), "", true, is_host) << "]";
      } else if (expr->IsArith() || expr->IsLogical() || expr->IsCompare() ||
                 expr->isBitwise()) {
        auto& l = expr->GetL();
        auto& r = expr->GetR();
        auto lty = NodeType(*l);
        auto rty = NodeType(*r);
        auto& op = expr->GetOp();

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
        } else if (op == "#" && IsActualBoundedIntegerType(lty) &&
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
        } else if ((op == "#+" || op == "#-") &&
                   IsActualBoundedIntegerType(lty) &&
                   isa<ScalarIntegerType>(rty)) {
          oss << OpExprSTR(l, parent_op, is_left_child, is_host);
        } else if (op == "#/" || op == "#*" || op == "#%") {
          choreo_unreachable("unsupported expression op: '" + expr->GetOp() +
                             "', expr: " + PSTR(expr) + ".");
        } else {
          std::ostringstream res;
          // FP8 scalar arithmetic: upcast operands to FP32 first.
          // This avoids relying on FP8 operator overloads which may not exist.
          if (expr->IsArith() && (IsFp8Scalar(lty) || IsFp8Scalar(rty))) {
            res << ToF32(OpExprSTR(l, op, true, is_host)) << " " << op << " "
                << ToF32(OpExprSTR(r, op, false, is_host));
          } else {
            res << OpExprSTR(l, op, true, is_host) << " " << op << " "
                << OpExprSTR(r, op, false, is_host);
          }
          oss << WrapParen(res.str(), op);
        }
      }
    } else if (expr->IsTernary()) {
      std::ostringstream res;
      res << OpExprSTR(expr->GetC(), "?", true, is_host) << " ? "
          << OpExprSTR(expr->GetL(), "?", true, is_host) << " : "
          << OpExprSTR(expr->GetR(), "?", false, is_host);
      oss << WrapParen(res.str(), "?");
    } else
      choreo_unreachable("unsupported expression op: '" + expr->GetOp() +
                         "', expr: " + PSTR(expr) + ".");
  } else if (auto ca = dyn_cast<AST::ChunkAt>(e)) {
    return HandleChunkAt(ca, is_host);
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
    if (name == "__log")
      return "tcle::ln";
    else if (name == "__pow")
      return "tcle::power";
    else {
      const std::string prefix = "__";
      std::string func_name = name;
      if (auto res = RemovePrefixOrNull(prefix, name)) func_name = *res;
      return "tcle::" + func_name;
    }
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
    auto& op = e.GetOp();
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
