#include "codegen_topscc.hpp"

#include <filesystem>
#include <iostream>
#include <numeric>
#include <sstream>
#include <thread>

#include "ast.hpp"
#include "choreo_header.inc"
#include "codegen.hpp"
#include "operator_info.hpp"
#include "types.hpp"

#ifndef __CHOREO_TOPSCC_DIR__
#error "missing macro definition of __CHOREO_TOPSCC_DIR__"
#endif

// #define USING_OP_INFO

using namespace Choreo;
using namespace Choreo::Topscc;

extern Option<bool> native_f16;
extern Option<bool> native_bf16;
extern Option<bool> verbose;
extern Option<std::string> output;
extern Option<bool> use_hetero_tileflow;
extern Option<bool> use_system_toolchain;
extern Option<bool> use_pic;
extern Option<std::string> arch;
extern Option<std::string> target_options;

Option<bool> emit_fatbin(OptionKind::Hidden, "-fb", "", false,
                         "Emit fatbin file.");
Option<bool> no_decay_spanview(OptionKind::Hidden, "--no-decay-spanview",
                               "-ndecay-spv", false,
                               " decay spanview to be pointers.");
Option<bool>
    dma_verbose(OptionKind::Hidden, "--dma-verbose", "", false,
                " print DMA related informtion at runtime (debug only).");
Option<bool> dma_opt(OptionKind::Hidden, "-fopt-dma", "", true,
                     "optimize dma to linear copy.");
Option<bool> split_8byte_dma_transfer(
    OptionKind::Hidden, "-fsplit-8b-dma", "", true,
    "8-byte DMA transfers will be split into 1-byte chunks when platforms that "
    "do not support native 8-byte DMA.");

namespace {

inline void VerboseDMA(std::ostringstream& os, const std::string& indent,
                       const std::string& from, const std::string& to,
                       const std::string action, const std::string& offset,
                       size_t offcnt, const std::string& suffix = "") {
  if (!dma_verbose) return;

  os << indent << "printf(\"" << from << "->" << to << ", " << action
     << " offset: {";
  for (size_t i = 0; i < offcnt; ++i) {
    if (i > 0) os << ", ";
    os << "%d";
  }
  os << "} " << suffix << "\\n\"";
  if (offcnt > 0) os << ", " << offset;
  os << ");\n";
}

inline const std::string ImplicitPred(Storage cur) {
  switch (cur) {
  case Storage::LOCAL: return "__CHOREO_SINGLE_LOCAL__";
  case Storage::SHARED: return "__CHOREO_SINGLE_SHARED__";
  default: choreo_unreachable("unsupported storage level.");
  }
  return "";
}

const char* SingleInstancePredicate(bool shared_in_block) {
  if (shared_in_block) return "__CHOREO_SINGLE_SHARED__";
  return "__CHOREO_SINGLE_LOCAL__";
}

inline const char* SyncByLevel(Storage s) {
  switch (s) {
  case Storage::SHARED: return "__syncthreads()";
  case Storage::LOCAL: return "__syncsubthreads()";
  default:
    choreo_unreachable("unsupported storage location for the synchronization.");
  }
  return "";
}

inline const char* TopsMdsStorage(Storage st) {
  switch (st) {
  case Storage::DEFAULT:
  case Storage::GLOBAL: return "tops::Global";
  case Storage::SHARED: return "tops::Shared";
  case Storage::LOCAL: return "tops::Private";
  default: choreo_unreachable("storage type is not supported.");
  }
  return "";
}

inline const char* TopsDeviceMemory(Storage st) {
  switch (st) {
  case Storage::SHARED: return "__shared__";
  case Storage::LOCAL: return "__local__";
  default: choreo_unreachable("device storage type is not supported.");
  }
  return "";
}

inline std::string TopsParamStorage(Storage st) {
  switch (st) {
  case Storage::SHARED: return "__shared__";
  case Storage::LOCAL: return "__private__";
  default: return "";
  }
  return "";
}

inline const std::string GetDTEContextName() {
  static unsigned i = 0;
  return "choreo_topscc_ctx" + std::to_string(i++);
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

} // namespace

inline const std::string VectorTypeSTR(const ptr<VectorType>& vt) {
  auto elem_ty = vt->e_type;
  auto ec = vt->ec;
  auto elem_size = SizeOf(elem_ty);
  auto vector_size = elem_size * ec;
  std::string vty_str;
  if (vector_size == CCtx().GetSingleVectorByteSize())
    vty_str = "__vector ";
  else if (vector_size == 2 * CCtx().GetSingleVectorByteSize())
    vty_str = "__vector2 ";
  else if (vector_size == 4 * CCtx().GetSingleVectorByteSize())
    vty_str = "__vector4 ";
  else
    choreo_unreachable(
        "unsupported vector size: " + std::to_string(vector_size) + ".");
  vty_str += NameBaseType(elem_ty);
  return vty_str;
}

bool TopsccCodeGen::RequiresImplPred(Storage cur) const {
  // ignore any host code
  if (IsHost()) return false;
  // ignore any expression without storage
  if (cur == Storage::NONE) return false;

  if (max_parallel_level == Storage::SUB) {
    switch (cur) {
    case Storage::SUB: return false;
    case Storage::LOCAL:
    case Storage::SHARED: return true;
    default: choreo_unreachable("irrational storage level.");
    }
  } else if (max_parallel_level == Storage::LOCAL) {
    switch (cur) {
    case Storage::SUB:
    case Storage::LOCAL: return false;
    case Storage::SHARED: return true;
    default: choreo_unreachable("irrational storage level.");
    }
  } else if (max_parallel_level == Storage::SHARED)
    choreo_unreachable("irrational max_parallel_level storage level.");

  choreo_unreachable("unexpected storage level.");
  return false;
}

const std::string TopsccCodeGen::ShapeSTR(const Shape& s,
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
    oss << ValueSTR(vl[i]);
    if (need_static_cast) oss << ")";
  }
  return oss.str();
}

bool TopsccCodeGen::BeforeVisitImpl(AST::Node& n) {
  if (trace_visit) dbgs() << "Before visiting " << n.TypeNameString() << "\n";

  if (isa<AST::Program>(&n)) {
    VST_DEBUG(dbgs() << STR(FBInfo()) << "\n");
    // emit the fixed headers
    EmitFixedHostHead();
    EmitFixedDeviceHead();
    ssm.EnterScope();
    ssm.MapDeviceSymbolIfNotExist("::__choreo_no_tiling__", "0");
  } else if (isa<AST::ChoreoFunction>(&n)) {
    ResetChoreoFunctionStates();
    device_fn = "__choreo_device_" + fname;
    fty = cast<FunctionType>(GetSymbolType(fname));
    ssm.EnterScope();
    pl_stack.clear();
  } else if (auto pb = dyn_cast<AST::ParallelBy>(&n)) {
    // only on device-side
    if (pb->IsOuter()) {
      parallel_idx += 1;
      if (cgi->GetFunctionTrait(fname).multiple_parallelby)
        device_fn = "__choreo_device_" + fname + std::to_string(parallel_idx);
      EmitDeviceFuncDecl(ds);
      ds << " {\n";
      IncrDeviceIndent();
      ds << d_indent << "{ // parallel-by: " << n.LOC() << "\n";
      max_parallel_level = pb->GetMaxLevel();
      VST_DEBUG(pb->InlinePrint(dbgs());
                dbgs() << " (max-level: " << STR(max_parallel_level) << ")\n");
    }
    parallel_level = pb->GetLevel();
    pl_stack.push_back(parallel_level);
  } else if (isa<AST::WithBlock>(&n)) {
    IndStream() << "// with-in: " << n.LOC() << "\n";
    IndStream() << "{\n";
    IncrIndent();
  } else if (auto fb = dyn_cast<AST::ForeachBlock>(&n)) {
    IndStream() << "// foreach: " << n.LOC() << "\n";

    if (fb->suffixs) {
      for (auto& suffix : fb->suffixs->values) {
        if (auto suffix_call = AST::GetCall(suffix);
            suffix_call->IsAnno() &&
            suffix_call->function->name == "vectorize") {
          auto iv = AST::GetIdentifier(suffix_call->GetArguments()[0]);
          std::string sname = scoped_symtab.ScopeName() + iv->name;
          within_map.emplace(sname, std::vector<std::string>{sname});
          bv_map.emplace(sname, std::vector<std::string>{sname});
          ssm.MapDeviceSymbol(sname, "__iv_" + iv->name);
          ssm.MapHostSymbol(sname, "__iv_" + iv->name);
          return true;
        }
      }
    }
  } else if (auto da = dyn_cast<AST::DataAccess>(&n)) {
    if (auto sty = GetSpannedType(GetSymbolType(da->data->name))) {
      if (auto da_ty = dyn_cast<VectorType>(da->GetType())) {
        // if the data access is a vector, we need to generate simple leaptr for
        // this data access
        auto elem_ty = da_ty->e_type;
        std::string vty_str = VectorTypeSTR(da_ty);

        auto data_name = da->GetDataName();
        auto leaptr_name = data_name.substr(0, data_name.find_last_of('.')) +
                           "_ptr" + da->Id();

        ssm.MapDeviceSymbol(InScopeName(data_name) + da->Id(), leaptr_name);

        ds << d_indent << "auto " << leaptr_name << " = tcle::simple_leaptr<"
           << vty_str << ">(";
        ds << "(" << NameBaseType(elem_ty) << "*)"
           << ssm.DeviceName(InScopeName(data_name));

        auto shape = sty->GetShape();
        ds << AddressSTR(shape, *da, false) << ";\n";
      }
    }
  }

  if (isa<AST::IfElseBlock>(&n) || isa<AST::NamedVariableDecl>(&n)) {
    emit_call = false;
  } else if (isa<AST::IncrementBlock>(&n)) {
    IndStream() << "// incr: " << n.LOC() << "\n";
    IncrIndent();
  }

  if (!n.IsBlock() && RequiresImplPred(n.GetLevel())) {
    ds << d_indent << "if (" << ImplicitPred(n.GetLevel())
       << ") { // implicit inthreads\n";
    IncrDeviceIndent();
  }

  return true;
}

bool TopsccCodeGen::InMidVisitImpl(AST::Node& n) {
  if (auto ie = dyn_cast<AST::IfElseBlock>(&n)) {
    if (!ie->HasElse()) return true;
    DecrIndent();
    IndStream() << "} else {\n";
    IncrIndent();
  }
  return true;
}

bool TopsccCodeGen::AfterVisitImpl(AST::Node& n) {
  if (trace_visit) dbgs() << "After visiting " << n.TypeNameString() << "\n";

  if (isa<AST::Program>(&n)) {
    ssm.LeaveScope();

    // internal functionality: fatbin generation
    if (emit_fatbin) {
      if (!CompileWithScript("--gen-fatbin")) {
        error_count++;
        return false;
      } else
        return true;
    }

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
  } else if (auto pb = dyn_cast<AST::ParallelBy>(&n)) {
    // only on device-side
    if (pb->IsOuter()) {
      max_parallel_level = Storage::NONE;
      parallel_level = Storage::NONE;
      pl_stack.clear();
      ds << d_indent << "} // end parallel-by\n";
      DecrDeviceIndent();
      ds << "}\n\n";
    } else {
      assert(!pl_stack.empty());
      pl_stack.pop_back();
      parallel_level = pl_stack.back();
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

  if (!n.IsBlock() && RequiresImplPred(n.GetLevel())) {
    DecrDeviceIndent();
    ds << d_indent << "} // end implicit inthreads\n";
  }

  return true;
}

// tops::mdspan style offset
std::pair<std::string, size_t>
TopsccCodeGen::GenMdsOffset(const ptr<AST::ChunkAt> ca,
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

#if 0
    // the transpose operation requries an index array ???
    std::vector<size_t> indices;
    for (size_t i = 0; i < exprs.size(); ++i)
      if (auto tc = dyn_cast<TransposeConfig>(config))
        indices.push_back(tc->dim_values[i]);
      else
        indices.push_back(i);
#endif

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

  if (split_8byte_dma_transfer) {
    auto sty = GetSpannedType(GetSymbolType(ca->data->name));
    if (SizeOf(sty->ElementType()) == 8) offset << ", 0";
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
TopsccCodeGen::TileBaseOffset(const ptr<AST::ChunkAt>& ca) const {
  auto lidx = ca->IndexOfLastSpanAs();
  if (!lidx.has_value()) choreo_unreachable("unexpect");
  return GenOffset(ca, lidx.value());
}

// given i.sop(...).sop(...)..., generate the offset of the final span in the
// original span. It is VALID if and only if the final span is
// address-contiguous within the original span.
// end_idx: the offset is computed by sop in range [0, end_idx).
const std::string TopsccCodeGen::GenOffset(const ptr<AST::ChunkAt>& ca,
                                           size_t end_idx) const {
  if (ca->NoOperation()) return "";

  end_idx = std::min(end_idx, ca->OpCount());

  Shape outer_shape = GetSpannedType(GetSymbolType(ca->data->name))->GetShape();

  auto offset = sbe::nu(0);

  // outer_shape is the shape of original span
  // new_shape is the shape of tiled span
  Shape new_shape;
  for (size_t i = 0; i < end_idx; ++i) {
    const auto& sop = ca->OpAt(i);
    if (sop->SpecifyReshape()) {
      outer_shape = sop->GetBlockShape();
    } else {
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

  return ValueSTR(offset);
}

void TopsccCodeGen::EmitFixedHostHead() {
  std::ostringstream oss;
  oss <<
      R"(
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

// dependant on the topsruntime
#include "tops/tops_ext.h"
#include "tops/tops_runtime.h"

#if __GCU_ARCH__ >= 300
#include "tcle.h"
#endif // __GCU_ARCH__ >= 300
)";


  oss << "// include the choreo header;\n";
  if (native_f16)
    oss << "#define __CHOREO_TARGET_NATIVE_HALF_FLOAT_SUPPORT__\n";
  if (native_bf16) oss << "#define __CHOREO_TARGET_NATIVE_BF16_SUPPORT__\n";
  oss << R"(#include "choreo.h"

using namespace choreo;

)";
  code_segments.push_back(oss.str()); // reset the host code
}

void TopsccCodeGen::EmitFixedDeviceHead() {}

bool TopsccCodeGen::Visit(AST::FunctionDecl& n) {
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

  // Go through all the symbols appeared in topscc host function, do:
  //
  //  - decide the host parameter names,
  //  - map the runtime shape dimensions to the real host code expression
  //  - decide the topscc-host parameter names,
  //  - decide the topscc-host parameter indices,
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
        std::string bts = NameBaseType(sty->ElementType(), false);
        auto buf_sym = sym + "__device";
        hs << h_indent << bts << " * " << buf_sym << " = nullptr;\n";
        hs << h_indent << "choreo::abend_true(topsMalloc(&" << buf_sym << ", "
           << UnScopedSizeExpr(*sty) << "));\n";
        hs << h_indent << "choreo::abend_true(topsMemcpy(" << buf_sym << ", "
           << ssm.HostName(item.name) << ", " << UnScopedSizeExpr(*sty)
           << ", topsMemcpyHostToDevice));\n";
        ssm.MapHostSymbol(item.name + "__device", buf_sym);
        global_buffers.insert(buf_sym);
      }
    }
  }

  return true;
}

bool TopsccCodeGen::Visit(AST::ChoreoFunction& n) {
  TraceEachVisit(n);

  // If there is no AST::Return
  if (return_stream.str().empty() && NeedDeviceFunc()) EmitTopsFree();

  DecrHostIndent();
  hs << "}\n\n";

  return true;
}

bool TopsccCodeGen::Visit(AST::NamedVariableDecl& n) {
  TraceEachVisit(n);

  auto nty = NodeType(n);
  auto sym = n.name_str;

  bool ref = n.Note().count("ref");
  // workaround:
  // if a symbol is declared but have no symbol value(optimized value)
  // pass it to device func even it is unused.
  if (!FCtx(fname).HasSymbolValues(InScopeName(sym)))
    updating_cgi->AddSymbolDetail(fname,
                                  {InScopeName(sym), GetSymbolType(sym), true});
  else
    updating_cgi->AddSymbolDetail(fname,
                                  {InScopeName(sym), GetSymbolType(sym), ref});

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

  if (auto sty = dyn_cast<SpannedType>(nty)) {
    auto sym__init = sym + "__init";
    auto buf_sym = sym + "__device";
    // globals are declared in host, while shareds/locals are declared in device
    auto shape = sty->GetShape();
    std::string bts{NameBaseType(sty->ElementType())};
    auto sto = sty->GetStorage();

    bool spmem = false; // allocatable scratchpad memory: share, local

    auto HandleGlobal = [&]() -> void {
      bts = NameBaseType(sty->ElementType(), false); // use the device type name

      if (IsChoreoOutput(InScopeName(sym))) {
        // the sym is choreo output
        std::string sym_data = sym + ".data()";
        hs << h_indent << "auto " << sym
           << " = choreo::make_spandata<choreo::" << STR(sty->e_type) << ", "
           << shape.Rank() << ">({" << ShapeSTR(shape, ", ", BaseType::U64)
           << "});\n";
        if (n.init_value) {
          // support initialization of output
          hs << h_indent << "std::fill(" << sym_data << ", " << sym_data << "+"
             << sym << ".element_count()"
             << ", "
             << ExprCastSTR(n.init_value, std::nullopt, GetBaseType(*sty),
                            GetBaseType(*n.init_value->GetType()), true)
             << ");\n";
        }
        hs << h_indent << bts << " * " << buf_sym << " = nullptr;\n";
        hs << h_indent << "choreo::abend_true(topsMalloc(&" << buf_sym << ", "
           << UnScopedSizeExpr(*sty) << "));\n";
        if (n.init_value) {
          hs << h_indent << "choreo::abend_true(topsMemcpy(" << buf_sym << ", "
             << sym_data << ", " << UnScopedSizeExpr(*sty)
             << ", topsMemcpyHostToDevice));\n";
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
        hs << h_indent << bts << " * " << buf_sym << " = nullptr;\n";
        hs << h_indent << "choreo::abend_true(topsMalloc(&" << buf_sym << ", "
           << UnScopedSizeExpr(*sty) << "));\n";

        if (!n.init_value) return;

        // support int/float-point literal initialization
        std::string sym_init_val = sym + "_init_val";
        hs << h_indent << bts << " " << sym_init_val << " = "
           << ExprSTR(n.init_value) << ";\n";
        std::string sym_init_vptr = sym + "_init_vptr";
        size_t data_len = SizeOf(sty->ElementType()) * 8;
        std::string init_val_type;
        switch (data_len) {
        case 32: init_val_type = "int"; break;
        case 16: init_val_type = "unsigned short"; break;
        case 8: init_val_type = "unsigned char"; break;
        default:
          choreo_unreachable("unsupported data length " +
                             std::to_string(data_len) +
                             " in global span init.");
        }
        hs << h_indent << init_val_type << "* " << sym_init_vptr
           << " = reinterpret_cast<" << init_val_type << "*>(&" << sym_init_val
           << ");\n";
        hs << h_indent << "choreo::abend_true(topsMemsetD" << data_len << "("
           << buf_sym << ", *" << sym_init_vptr << ", "
           << UnScopedExpr(ElemCountExprOf(*sty)) << "));\n";
      }
    };

    auto HandleSharedLocal = [&]() -> void {
      if (IsChoreoOutput(InScopeName(sym)))
        choreo_unreachable(
            "error: shared/local buffer cannot be Choreo output.");

      auto type_modifiers =
          (sto == Storage::SHARED ? "__shared__ " : "__local__ __valigned__ ");

      if (!CCtx().MemReuse()) {
        ds << d_indent << type_modifiers << bts << " " << sym;
        for (auto dim : n.ArrayDimensions()) ds << "[" << dim << "]";
        ds << "[" << UnScopedExpr(ElemCountExprOf(*sty)) << "];\n";
        return;
      }

      // memory reuse is enabled

      if (n.Note().count("spm")) {
        ds << d_indent << type_modifiers << bts << " " << sym << "["
           << UnScopedExpr(ElemCountExprOf(*sty)) << "];\n";
        return;
      }

      // the buffer is not the declared whole spm.
      if (auto reuse = FindOrNull(n.Note(), "reuse")) {
        auto offset = n.Note().at("offset");
        ds << d_indent << bts << "* " << sym << " = (" << bts << "*)"
           << "(" << *reuse << " + " << offset << ");\n";
      } else {
        // the buffer is not reused
        // which means that it is declared but never used.
        // TODO: should we DCE the unused buffer?
        assert(!n.Note().count("offset"));
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
      ds << d_indent << "if ("
         << SingleInstancePredicate(sto == Storage::SHARED) << ") {\n";
      IncrDeviceIndent();
      ds << d_indent << "tops_dte_ctx_t " << sym__init << ";\n";
      ds << d_indent << "tops::dte_scope s_" << sym__init << "(" << sym__init
         << ");\n";
      ds << d_indent << "tops::memset(" << sym__init << ", tops::mdspan("
         << TopsMdsStorage(sto) << ", (" << NameBaseType(sty->ElementType())
         << "*)" << sym << ", " << ShapeSTR(sty->GetShape()) << "), "
         << ExprCastSTR(n.init_value, std::nullopt, GetBaseType(*sty),
                        GetBaseType(*n.init_value->GetType()), false)
         << ");\n";
      DecrDeviceIndent();
      ds << d_indent << "} // single instance\n";
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
      Stream() << TopsDeviceMemory(st) << " ";
    }
    Stream() << NameBaseType(GetBaseType(*nty), false) << " " << sym;
    if (n.init_expr) Stream() << " = " << ExprSTR(n.init_expr, false);
    Stream() << ";\n";

    // mutables have references
    if (IsMutable(*nty))
      if (!IsHost()) ssm.MapDeviceSymbol(InScopeName(sym), sym);

    return true;
  }

  if (auto vty = dyn_cast<VectorType>(nty)) {
    IndStream() << VectorTypeSTR(vty) << " " << sym;
    if (n.init_expr) Stream() << " = " << ExprSTR(n.init_expr, false);
    Stream() << ";\n";
    if (!IsHost()) ssm.MapDeviceSymbol(InScopeName(sym), sym);
  }

  // handle events
  if (auto ety = dyn_cast<EventArrayType>(nty)) {
    switch (ety->GetStorage()) {
    case Storage::GLOBAL: {
      assert(IsHost());
      auto sym = InScopeName(n.name_str);
      auto buf_sym = n.name_str + "__device";
      hs << h_indent << "bool * " << buf_sym << " = nullptr; // global event\n";
      hs << h_indent << "choreo::abend_true(topsMalloc(&" << buf_sym << ", "
         << ety->ElemCount() << "));\n";
      hs << h_indent << "choreo::abend_true(topsMemset(&" << buf_sym << ", 0, "
         << ety->ElemCount() << "));\n";
      ssm.MapHostSymbol(sym, buf_sym);
      ssm.MapDeviceSymbol(sym, n.name_str);
      global_buffers.insert(buf_sym);
    } break;
    case Storage::SHARED:
    case Storage::LOCAL: {
      assert(!IsHost());
      ds << d_indent << TopsDeviceMemory(ety->GetStorage())
         << " __volatile__ bool " << n.name_str;
      ety->PrintAsCArray(ds);
      ds << "; // " << STR(ety->GetStorage()) << " event\n";
      ds << d_indent << "// initialize the event\n";
      if (RequiresImplPred(ety->GetStorage())) {
        ds << d_indent << "if (" << ImplicitPred(ety->GetStorage()) << ") {\n";
        GenerateSubscriptions(ds, "  " + d_indent + n.name_str, " = false;\n",
                              ety->Dimensions());
        ds << d_indent << "}\n";
      } else
        GenerateSubscriptions(ds, d_indent + n.name_str, " = false;\n",
                              ety->Dimensions());
      ds << d_indent << SyncByLevel(ety->GetStorage()) << ";\n";
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
      hs << h_indent << "choreo::abend_true(topsMalloc(&" << buf_sym
         << ", 1));\n";
      hs << h_indent << "choreo::abend_true(topsMemset(&" << buf_sym
         << ", 0, 1));\n";
      ssm.MapHostSymbol(sym, buf_sym);
      ssm.MapDeviceSymbol(sym, n.name_str);
      global_buffers.insert(buf_sym);
    } break;
    case Storage::SHARED:
    case Storage::LOCAL: {
      assert(!IsHost());
      ds << d_indent << TopsDeviceMemory(ety->GetStorage())
         << " __volatile__ bool " << n.name_str << "; // "
         << STR(ety->GetStorage()) << " event\n";
      if (RequiresImplPred(ety->GetStorage())) {
        ds << d_indent << "if (" << ImplicitPred(ety->GetStorage()) << ") {\n";
        ds << d_indent << "  " << n.name_str
           << " = false;\n"; // inited as untriggered
        ds << d_indent << "}\n";
      } else
        ds << d_indent << n.name_str << " = false;\n"; // inited as untriggered
      ds << d_indent << SyncByLevel(ety->GetStorage()) << ";\n";
    } break;
    default: break;
    }
  }

  return true;
}

bool TopsccCodeGen::Visit(AST::Assignment& n) {
  TraceEachVisit(n);

  if (!n.AssignToDataElement()) {
    auto name = n.GetName();
    bool ref = n.Note().count("ref");
    if (!SSTab().IsDeclared(name) && !isa<AST::SpanAs>(n.value))
      updating_cgi->AddSymbolDetail(
          fname, {InScopeName(name), GetSymbolType(name), ref});
  }

  auto nty = NodeType(n);

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
    if (isa<FutureType>(tty))
      ds << sa->id->name << ".data();\n";
    else
      ds << sa->id->name << ";\n";
    ssm.MapDeviceSymbol(InScopeName(n.GetName()), n.GetName());
    return true;
  }

  if (n.AssignToDataElement()) {
    if (!IsHost()) {
      ds << d_indent << DASTR(n.da, ExprSTR(n.value, false), false) << ";\n";
    }

    if (IsHost()) {
      // TODO: test the case!
      choreo_unreachable(
          "error: assignment to data element should be on device side.");
    }

    return true;
  }

  if (isa<BoundedType>(nty) || isa<SpannedType>(nty) || isa<FutureType>(nty)) {
    assert(!IsHost() && "bounded/spanned/future should be on device side.");
    ds << d_indent << ((IsMutable(*nty)) ? "" : "auto ") << n.GetName() << " = "
       << ExprSTR(n.value, false) << ";\n";
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

  if (isa<VectorType>(nty)) {
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

bool TopsccCodeGen::Visit(AST::ParallelBy& n) {
  TraceEachVisit(n);

  // add the device name map
  std::string dname[] = {"x", "y", "z"};
  switch (n.GetLevel()) {
  case Storage::SHARED:
    for (size_t i = 0; i < n.AllSubPVs().size(); ++i)
      ssm.MapDeviceSymbol(InScopeName(n.GetSubPV(i)->name),
                          "__tops_bid_" + dname[i] + "()");
    if (n.AllSubPVs().size() == 1)
      ssm.MapDeviceSymbol(InScopeName(n.BPV()->name), "__tops_bid_x()");
    break;
  case Storage::LOCAL:
    for (size_t i = 0; i < n.AllSubPVs().size(); ++i)
      ssm.MapDeviceSymbol(InScopeName(n.GetSubPV(i)->name),
                          "__tops_tid_" + dname[i] + "()");
    if (n.AllSubPVs().size() == 1)
      ssm.MapDeviceSymbol(InScopeName(n.BPV()->name), "__tops_tid_x()");
    break;
  case Storage::SUB:
    for (size_t i = 0; i < n.AllSubPVs().size(); ++i)
      ssm.MapDeviceSymbol(InScopeName(n.GetSubPV(i)->name),
                          "__tops_stid_" + dname[i] + "()");
    if (n.AllSubPVs().size() == 1)
      ssm.MapDeviceSymbol(InScopeName(n.BPV()->name), "__tops_stid_x()");
    break;
  default:
    choreo_unreachable("unsupported parallel-by level: " + STR(n.GetLevel()) +
                       ".");
  }

  // only do the whole codegen when accessing the outer parallel-by
  if (!n.IsOuter()) return true;

  EmitMemReuse(SSTab().ScopeName());

  // note: `thread_dims` for gcu400 is generated in `EmitDeviceFuncDecl`
  auto& lconfig = cgi->GetFunctionLaunches(fname)[parallel_idx];
  hs << h_indent << "dim3 __" << fname << "_gdims" << parallel_idx << "("
     << ValueSTR(lconfig.grid_dim_x) << ", " << ValueSTR(lconfig.grid_dim_y)
     << ", " << ValueSTR(lconfig.grid_dim_z) << ");\n";
  hs << h_indent << "dim3 __" << fname << "_bdims" << parallel_idx << "("
     << ValueSTR(lconfig.block_dim_x) << ", " << ValueSTR(lconfig.block_dim_y)
     << ", " << ValueSTR(lconfig.block_dim_z) << ");\n";
  hs << h_indent << device_fn << "<<<__" << fname << "_gdims" << parallel_idx
     << ", __" << fname << "_bdims" << parallel_idx << ">>>(";

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
  const auto& offset_args =
      FCtx(fname).GetMemReuseOffsetArgs(SSTab().ScopeName());
  std::string mr_idx_suffix = "";
  if (cgi->GetFunctionTrait(fname).multiple_parallelby)
    mr_idx_suffix = std::to_string(parallel_idx);
  if (offset_args.has_value())
    for (const auto& [sto, offsets] : offset_args.value())
      for (size_t idx = 0; idx < offsets.size(); ++idx)
        hs << ((i++ > 0) ? ", " : "") << "__co__" << STR(sto)
           << "_chunk_offsets" << mr_idx_suffix << "[" << idx << "]";

  hs << ");\n";

  if (!n.IsAsync())
    hs << h_indent << "choreo::abend_true(topsDeviceSynchronize());\n";

  // copy the span passed by ref back to host
  for (const auto& item : GetChoreoFuncIns(updating_cgi)) {
    if (isa<SpannedType>(item.type)) {
      auto oname = UnScopedName(item.name);
      if (item.attr != ParamAttr::GLOBAL_INPUT && item.IsReference())
        hs << h_indent << "choreo::abend_true(topsMemcpy(" << oname
           << ".data(), " << oname + "__device" << ", "
           << UnScopedSizeExpr(*item.type) << ", topsMemcpyDeviceToHost));\n";
    }
  }

  return true;
}

bool TopsccCodeGen::Visit(AST::DMA& n) {
  TraceEachVisit(n);

  // Currently, DMA in host-side:
  // - will not generate any future.
  // - are performed directly by manipulating pointers.
  // - not support tiling.
  // - not support async.

  // Generate tops dte and choreo::future in device-side
  auto claimFuture = [this, &n](const std::string& buf_expr) -> std::string {
    if (!n.future.empty() && claimed_dte.count(InScopeName(n.future)))
      return n.future;

    // claim the date transfer engine
    auto dte_ctx = GetDTEContextName();
    ds << d_indent << "tops_dte_ctx_t " << dte_ctx << ";\n";
    auto future_name = n.future;
    if (future_name.empty()) {
      static size_t future_count = 0;
      future_name = "__choreo_anon_fut__" + std::to_string(future_count++);
    } else {
      claimed_dte.emplace(InScopeName(n.future), dte_ctx);
      ssm.MapDeviceSymbol(InScopeName(n.future), n.future);
      ssm.MapDeviceSymbol(InScopeName(n.future) + ".data",
                          n.future + ".data()");
    }
    ds << d_indent << "choreo::future " << future_name << "(" << dte_ctx
       << ", \"" << n.future << "\", " << n.LOC().begin.line << ", "
       << n.LOC().begin.column;
    if (!buf_expr.empty()) ds << ", " << buf_expr;
    ds << ");\n";

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

    claimFuture(UnScopedName(buf_name));
    // make following buffer reference all be indirect
    // TODO: any better idea than this
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
    std::string bts = NameBaseType(t_sty->ElementType(), false);
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
    if (n.async) choreo_unreachable("not support host-side async dma yet");
    std::string bts = NameBaseType(t_sty->ElementType(), false);
    std::string buf_sym_from;
    std::string buf_sym;
    std::string tops_dma_kind = "topsMemcpy";
    if (global_buffers.count(f_sym + "__device")) {
      buf_sym_from = f_sym + "__device";
      tops_dma_kind.append("Device");
    } else {
      buf_sym_from = f_sym + ".data()";
      if (f_sty->GetStorage() == Storage::GLOBAL)
        tops_dma_kind.append("Device");
      else
        tops_dma_kind.append("Host");
    }

    if (global_buffers.count(t_sym + "__device")) {
      buf_sym = t_sym + "__device";
      tops_dma_kind.append("ToDevice");
    } else {
      buf_sym = t_sym + ".data()";
      if (f_sty->GetStorage() == Storage::GLOBAL)
        tops_dma_kind.append("ToDevice");
      else
        tops_dma_kind.append("ToHost");
    }

    if (n.operation == ".copy") {
      if (SymbolToSymbol()) {
        // direct copy
        hs << h_indent << "choreo::abend_true(topsMemcpy(" << buf_sym << ", "
           << buf_sym_from << ", " << UnScopedSizeExpr(*f_sty) << ", "
           << tops_dma_kind << "));\n";
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

  // return mds name and the declaration string.
  // If offset is not empty, means that need to do memory viewing.
  //   Just add offset to buf_expr, then utilize new_shape.
  auto GenMDSDecl =
      [this](const std::string& buf_name, const std::string& buf_expr,
             const ptr<SpannedType>& sty, const std::string& offset = "",
             const Shape& new_shape =
                 Shape()) -> std::pair<std::string, std::string> {
    static int mds_cnt = 0;
    auto mds_name = "__mds" + std::to_string(mds_cnt++) + "_" +
                    RemoveSuffix(buf_name, ".data()");
    auto bt = sty->ElementType();
    bool split_to_char = false;
    if (split_8byte_dma_transfer && SizeOf(bt) == 8) split_to_char = true;
    std::string bts{split_to_char ? "char" : NameBaseType(bt)};

    std::ostringstream mds_decl;

    mds_decl << d_indent << "tops::mdspan " << mds_name << "("
             << TopsMdsStorage(sty->GetStorage()) << ", (" << bts << "*)"
             << buf_expr;
    if (offset == "")
      mds_decl << ", "
               << ShapeSTR(new_shape.IsValid() ? new_shape : sty->GetShape());
    else {
      mds_decl << " + ";
      if (split_to_char)
        mds_decl << offset << " * 8";
      else
        mds_decl << offset;
      mds_decl << ", " << ShapeSTR(new_shape);
    }
    if (split_to_char) mds_decl << ", 8";
    mds_decl << ");\n";
    return {mds_name, mds_decl.str()};
  };

  ValueList f_opt_condition, t_opt_condition;

  // if the value is dst, means MAY do optimization on dst
  enum DMA_OP : uint8_t {
    none = 0,
    dst = 1 << 0,
    src = 1 << 1,
    both = src | dst
  };

  // Check the current DMA. If the new span is address-contiguous within the
  // original span, then slice or deslice can be optimized to linear copy.
  auto OptToLinearCopy = [&]() -> DMA_OP {
    if (!dma_opt) return DMA_OP::none;
    if (SymbolToSymbol()) return DMA_OP::none;

    // return `bool` if able to ensure true positive or true negative.
    // return `ValueList` as the runtime condition if unable to ensure.
    auto IsOptimizableChunkat =
        [&](const ptr<AST::ChunkAt>& ca) -> std::variant<bool, ValueList> {
      ValueList opt_condition;
      Shape shape, new_shape;
      shape = GetSpannedType(GetSymbolType(ca->RefSymbol()))->GetShape();
      // check the shape transformation of each op inside ca
      for (const auto& sop : ca->AllOperations()) {
        if (sop->SpecifyReshape()) {
          shape = sop->GetBlockShape();
        } else {
          new_shape = sop->GetBlockShape();
          auto is_contiguous = IsContiguousSOp(*sop, shape);
          if (auto val = std::get_if<bool>(&is_contiguous);
              val && *val == false)
            return false;
          else if (!val)
            opt_condition.push_back(std::get<ValueItem>(is_contiguous));

          shape = new_shape;
        }
      }
      if (opt_condition.empty()) return true;
      return opt_condition;
    };

    DMA_OP optimizable = DMA_OP::none;
    if (SymbolToTile()) {
      auto res = IsOptimizableChunkat(t_ca);
      if (auto val = std::get_if<bool>(&res); val && *val == true)
        optimizable = DMA_OP::dst;
      else if (!val) {
        t_opt_condition = std::get<ValueList>(res);
        optimizable = DMA_OP::dst;
      }
    } else if (TileToSymbol()) {
      auto res = IsOptimizableChunkat(f_ca);
      if (auto val = std::get_if<bool>(&res); val && *val == true)
        optimizable = DMA_OP::src;
      else if (!val) {
        f_opt_condition = std::get<ValueList>(res);
        optimizable = DMA_OP::src;
      }
    } else if (TileToTile()) {
      // For tile to tile, there are 3 situtations.
      auto res_f = IsOptimizableChunkat(f_ca);
      if (auto val = std::get_if<bool>(&res_f); val && *val == true)
        optimizable = DMA_OP::src;
      else if (!val) {
        f_opt_condition = std::get<ValueList>(res_f);
        optimizable = DMA_OP::src;
      }
      auto res_t = IsOptimizableChunkat(t_ca);
      if (auto val = std::get_if<bool>(&res_t); val && *val == true)
        optimizable = (optimizable & DMA_OP::src) ? DMA_OP::both : DMA_OP::dst;
      else if (!val) {
        t_opt_condition = std::get<ValueList>(res_t);
        optimizable = (optimizable & DMA_OP::src) ? DMA_OP::both : DMA_OP::dst;
      }
    }

    VST_DEBUG({
      if (optimizable != DMA_OP::none) {
        dbgs() << "Optimize DMA at " << n.LOC() << " to linear copy:\n";
        if (optimizable & DMA_OP::src) {
          dbgs() << "\tSRC";
          if (!f_opt_condition.empty()) dbgs() << " (runtime)";
        }
        if (optimizable & DMA_OP::dst) {
          dbgs() << "\tDST";
          if (!t_opt_condition.empty()) dbgs() << " (runtime)";
        }
        dbgs() << "\n";
      }
    });

    return optimizable;
  };

  const auto f_buf = GetBufferExpr(f_sym, f_idx, f_ty);
  const auto t_buf = GetBufferExpr(t_sym, t_idx, t_ty);

  auto future_name = n.future;
  // bind the data to the future
  if (SymbolToSymbol() || TileToSymbol() || TileToTile())
    future_name = claimFuture(t_buf.second);
  else
    future_name = claimFuture("");

  std::string event_name;
  if (fty->IsAsync()) event_name = future_name + "__event__";

  // indicate which side can be optimized to linear copy
  DMA_OP opt_to_linear_copy = OptToLinearCopy();

  // if `no_linear_opt` is true, will not do linear optimization even
  // `opt_to_linear_copy` is available. Use it to control runtime opt.
  auto DMACodeGen = [&](bool no_linear_opt) {
    std::string f_mds_offset = "";
    std::string t_mds_offset = "";
    Shape f_shape = f_sty->GetShape();
    Shape t_shape = t_sty->GetShape();
    const auto& f_buf_name = f_buf.first;
    const auto& f_buf_expr = f_buf.second;
    const auto& t_buf_name = t_buf.first;
    const auto& t_buf_expr = t_buf.second;
    if (auto idx = f_ca->IndexOfLastSpanAs()) {
      f_mds_offset = TileBaseOffset(f_ca);
      f_shape = f_ca->OpAt(*idx)->GetBlockShape();
    }
    if (auto idx = t_ca->IndexOfLastSpanAs()) {
      t_mds_offset = TileBaseOffset(t_ca);
      t_shape = t_ca->OpAt(*idx)->GetBlockShape();
    }
    if (!no_linear_opt && opt_to_linear_copy != DMA_OP::none) {
      if (TileToSymbol()) {
        assert(opt_to_linear_copy == DMA_OP::src);
        f_mds_offset = GenOffset(f_ca);
        f_shape = f_ca->GetBlockShape();
      } else if (SymbolToTile()) {
        assert(opt_to_linear_copy == DMA_OP::dst);
        t_mds_offset = GenOffset(t_ca);
        t_shape = t_ca->GetBlockShape();
      } else if (TileToTile()) {
        if (opt_to_linear_copy & DMA_OP::src) {
          f_mds_offset = GenOffset(f_ca);
          f_shape = f_ca->GetBlockShape();
        }
        if (opt_to_linear_copy & DMA_OP::dst) {
          t_mds_offset = GenOffset(t_ca);
          t_shape = t_ca->GetBlockShape();
        }
      }
    }
    const auto f_mds =
        GenMDSDecl(f_buf_name, f_buf_expr, f_sty, f_mds_offset, f_shape);
    const auto t_mds =
        GenMDSDecl(t_buf_name, t_buf_expr, t_sty, t_mds_offset, t_shape);

    std::string f_mds_name{f_mds.first};
    std::string f_mds_decl{f_mds.second};
    std::string t_mds_name{t_mds.first};
    std::string t_mds_decl{t_mds.second};

    ds << f_mds_decl;
    ds << t_mds_decl;

    // handles dma related to shared memory, where only single thread can
    // operate
    bool local_in_warp = false, shared_in_block = false;
    if (!n.future.empty()) {
      shared_in_block = IsDMABlockShared(n);
      local_in_warp = IsDMAWarpLocal(n);
    }

    assert(!(shared_in_block && local_in_warp) &&
           "local and shared memory should not be used at the same time");
    if (local_in_warp)
      assert(CCtx().GetArch() == TargetArch::GCU4 &&
             "only gcu400 need handle local synchronization");

    if (shared_in_block || local_in_warp) {
      ds << d_indent << "if (" << SingleInstancePredicate(shared_in_block)
         << ") {\n";
      IncrDeviceIndent();
    }

    if (n.operation == ".copy") {
      auto LinearCopy = [&]() -> void {
        ds << d_indent;
        if (!event_name.empty()) ds << "tops::event " + event_name + " = ";
        ds << "tops::memcpy" << (fty->IsAsync() ? "_async" : "") << "(*"
           << future_name << ".get_ctx(), " << t_mds_name << ", " << f_mds_name
           << ");\n";
        VerboseDMA(ds, d_indent, t_sym, f_sym, "copy", "", 0,
                   ", line " + std::to_string(n.LOC().begin.line));
        if (!event_name.empty())
          ds << d_indent << future_name << ".set_event(" << event_name
             << ");\n";
      };
      auto Deslice = [&]() -> void {
        static int ds_cnt = 0;
        auto off_name = "__deslice_offset" + std::to_string(ds_cnt++) + "__" +
                        t_sym + "_2_" + f_sym;
        auto [offset, offcnt] = GenMdsOffset(t_ca);
        VerboseDMA(ds, d_indent, t_sym, f_sym, "deslice", offset, offcnt,
                   ", line " + std::to_string(n.LOC().begin.line));
        ds << d_indent << "int " << off_name << "[] = {" << offset << "};\n";
        ds << d_indent;
        if (!event_name.empty()) ds << "tops::event " + event_name + " = ";
        ds << "tops::deslice" << (fty->IsAsync() ? "_async" : "") << "(*"
           << future_name << ".get_ctx(), " << t_mds_name << ", " << f_mds_name
           << ", " << off_name << ");\n";
        if (!event_name.empty())
          ds << d_indent << future_name << ".set_event(" << event_name
             << ");\n";
      };
      auto Slice = [&]() -> void {
        static int s_cnt = 0;
        auto off_name = "__slice_offset" + std::to_string(s_cnt++) + "__" +
                        f_sym + "_2_" + t_sym;
        auto [offset, offcnt] = GenMdsOffset(f_ca);
        VerboseDMA(ds, d_indent, f_sym, t_sym, "slice", offset, offcnt,
                   ", line " + std::to_string(n.LOC().begin.line));
        ds << d_indent << "int " << off_name << "[] = {" << offset << "};\n";
        ds << d_indent;
        if (!event_name.empty()) ds << "tops::event " + event_name + " = ";
        ds << "tops::slice" << (fty->IsAsync() ? "_async" : "") << "(*"
           << future_name << ".get_ctx(), " << t_mds_name << ", " << f_mds_name
           << ", " << off_name << ");\n";
        if (!event_name.empty())
          ds << d_indent << future_name << ".set_event(" << event_name
             << ");\n";
      };
      auto SliceDeslice = [&]() -> void {
        static int s_cnt = 0;
        auto s_off_name = "__slice_offset" + std::to_string(s_cnt++) + "__" +
                          f_sym + "_2_" + t_sym;
        auto [s_offset, s_offcnt] = GenMdsOffset(f_ca);

        static int ds_cnt = 0;
        auto ds_off_name = "__deslice_offset" + std::to_string(ds_cnt++) +
                           "__" + t_sym + "_2_" + f_sym;
        auto [ds_offset, ds_offcnt] = GenMdsOffset(t_ca);

        VerboseDMA(ds, d_indent, t_sym, f_sym, "deslice", ds_offset, ds_offcnt,
                   ", line " + std::to_string(n.LOC().begin.line));
        ds << d_indent << "int " << s_off_name << "[] = {" << s_offset
           << "};\n";
        ds << d_indent << "int " << ds_off_name << "[] = {" << ds_offset
           << "};\n";

        auto slice_shape_name = "__slice_shape" + std::to_string(s_cnt) + "__" +
                                f_sym + "_2_" + t_sym;
        ds << d_indent << "unsigned int " << slice_shape_name << "[] = {"
           << ShapeSTR(f_ca->GetBlockShape(), ", ", BaseType::U32) << "};\n";
        ds << d_indent;
        if (!event_name.empty()) ds << "tops::event " + event_name + " = ";
        ds << "tops::slice_deslice" << (fty->IsAsync() ? "_async" : "") << "(*"
           << future_name << ".get_ctx(), " << t_mds_name << ", " << f_mds_name
           << ", " << s_off_name << ", " << slice_shape_name << ", "
           << ds_off_name << ");\n";

        if (!event_name.empty())
          ds << d_indent << future_name << ".set_event(" << event_name
             << ");\n";
      };

      if (SymbolToSymbol()) {
        LinearCopy();
      } else if (SymbolToTile()) {
        if (opt_to_linear_copy == DMA_OP::dst && !no_linear_opt)
          LinearCopy();
        else
          Deslice();
      } else if (TileToSymbol()) {
        if (opt_to_linear_copy == DMA_OP::src && !no_linear_opt)
          LinearCopy();
        else
          Slice();
      } else if (TileToTile()) {
        if (opt_to_linear_copy == DMA_OP::both && !no_linear_opt)
          LinearCopy();
        else if (opt_to_linear_copy == DMA_OP::src && !no_linear_opt)
          Deslice();
        else if (opt_to_linear_copy == DMA_OP::dst && !no_linear_opt)
          Slice();
        else
          SliceDeslice();
      } else {
        choreo_unreachable("unexpected situation.");
      }
    } else if (n.operation == ".pad") {
      static int p_cnt = 0;
      p_cnt++;
      auto pcmvSTR = [&](ptr<AST::MultiValues> mv) -> std::string {
        std::string res;
        for (const auto& v : mv->AllValues()) {
          if (!res.empty()) res += ", ";
          res += ExprSTR(v, IsHost());
        }
        return res;
      };
      auto pad_config = cast<PadConfig>(n.GetConfig());
      auto f_buf_name = RemoveSuffix(f_buf_expr, ".data()");
      std::string pad_low = "__pad_low_" + f_buf_name + std::to_string(p_cnt);
      std::string pad_high = "__pad_high_" + f_buf_name + std::to_string(p_cnt);
      std::string pad_mid = "__pad_mid_" + f_buf_name + std::to_string(p_cnt);
      ds << d_indent << "unsigned int " << pad_low << "[] = {"
         << pcmvSTR(pad_config->pad_low) << "};\n";
      ds << d_indent << "unsigned int " << pad_high << "[] = {"
         << pcmvSTR(pad_config->pad_high) << "};\n";
      ds << d_indent << "unsigned int " << pad_mid << "[] = {"
         << pcmvSTR(pad_config->pad_mid) << "};\n";

      auto Pad = [&]() -> void {
        ds << d_indent;
        if (!event_name.empty()) ds << "tops::event " + event_name + " = ";
        ds << "tops::pad" << (fty->IsAsync() ? "_async" : "") << "(*"
           << future_name << ".get_ctx(), " << t_mds_name << ", " << f_mds_name
           << ", " << pad_low << ", " << pad_high << ", " << pad_mid << ", "
           << ExprSTR(pad_config->value, IsHost()) << ");\n";
        // set the device future
        if (!event_name.empty())
          ds << d_indent << future_name << ".set_event(" << event_name
             << ");\n";
      };

      auto SlicePad = [&]() -> void {
        static int s_cnt = 0;
        auto off_name = "__slice_offset" + std::to_string(s_cnt++) + "__" +
                        f_sym + "_2_" + t_sym;
        auto [offset, offcnt] = GenMdsOffset(f_ca);
        VerboseDMA(ds, d_indent, t_sym, f_sym, "slice+pad", offset, offcnt,
                   ", line " + std::to_string(n.LOC().begin.line));
        ds << d_indent << "int " << off_name << "[] = {" << offset << "};\n";
        auto slice_shape_name = "__slice_shape" + std::to_string(s_cnt) + "__" +
                                f_sym + "_2_" + t_sym;
        ds << d_indent << "unsigned int " << slice_shape_name << "[] = {"
           << ShapeSTR(f_ca->GetBlockShape(), ", ", BaseType::U32) << "};\n";
        ds << d_indent;
        if (!event_name.empty()) ds << "tops::event " + event_name + " = ";
        ds << "tops::slice_pad" << (fty->IsAsync() ? "_async" : "") << "(*"
           << future_name << ".get_ctx(), " << t_mds_name << ", " << f_mds_name
           << ", " << off_name << ", " << slice_shape_name << ", " << pad_low
           << ", " << pad_high << ", " << pad_mid << ", "
           << ExprSTR(pad_config->value, IsHost()) << ");\n";
        // set the device future
        if (!event_name.empty())
          ds << d_indent << future_name << ".set_event(" << event_name
             << ");\n";
      };

      if (SymbolToSymbol()) {
        Pad();
      } else if (TileToSymbol()) {
        if (opt_to_linear_copy == DMA_OP::src && !no_linear_opt)
          Pad();
        else
          SlicePad();
      } else {
        choreo_unreachable(
            "only support dma.pad with (symbol=>symbol), (tile=>symbol).");
      }
    } else if (n.operation == ".transp") {
      static int t_cnt = 0;
      auto transp_config = cast<TransposeConfig>(n.GetConfig());
      auto f_buf_name = RemoveSuffix(f_buf_expr, ".data()");
      auto t_buf_name = RemoveSuffix(t_buf_expr, ".data()");
      auto layout_name =
          "__transpose_layout" + std::to_string(t_cnt++) + "__" + f_buf_name;
      ds << d_indent << "int " << layout_name << "[] = {"
         << DelimitedString(transp_config->dim_values) << "};\n";

      auto Transpose = [&]() -> void {
        ds << d_indent;
        if (!event_name.empty()) ds << "tops::event " + event_name + " = ";
        ds << "tops::transpose" << (fty->IsAsync() ? "_async" : "") << "(*"
           << future_name << ".get_ctx(), " << t_mds_name << ", " << f_mds_name
           << ", " << layout_name << ");\n";
        if (!event_name.empty())
          ds << d_indent << future_name << ".set_event(" << event_name
             << ");\n";
      };
      auto SliceTranspose = [&]() -> void {
        static int s_cnt = 0;
        auto off_name = "__slice_offset" + std::to_string(s_cnt++) + "__" +
                        f_sym + "_2_" + t_sym;
        auto [offset, offcnt] = GenMdsOffset(f_ca, n.GetConfig());
        ds << d_indent << "int " << off_name << "[] = {" << offset << "};\n";
        ds << d_indent;
        if (!event_name.empty()) ds << "tops::event " + event_name + " = ";
        ds << "tops::slice_transpose" << (fty->IsAsync() ? "_async" : "")
           << "(*" << future_name << ".get_ctx(), " << t_mds_name << ", "
           << f_mds_name << ", " << off_name << ", " << layout_name << ");\n";
        if (!event_name.empty())
          ds << d_indent << future_name << ".set_event(" << event_name
             << ");\n";
      };
      auto TransposeDeslice = [&]() -> void {
        static int ds_cnt = 0;
        auto off_name = "__deslice_offset" + std::to_string(ds_cnt++) + "__" +
                        t_sym + "_2_" + f_sym;
        auto [offset, offcnt] = GenMdsOffset(t_ca, n.GetConfig());
        ds << d_indent << "int " << off_name << "[] = {" << offset << "};\n";
        ds << d_indent;
        if (!event_name.empty()) ds << "tops::event " + event_name + " = ";
        ds << "tops::transpose_deslice" << (fty->IsAsync() ? "_async" : "")
           << "(*" << future_name << ".get_ctx(), " << t_mds_name << ", "
           << f_mds_name << ", " << layout_name << ", " << off_name << ");\n";
        if (!event_name.empty())
          ds << d_indent << future_name << ".set_event(" << event_name
             << ");\n";
      };

      if (SymbolToSymbol()) {
        Transpose();
      } else if (SymbolToTile()) {
        if (opt_to_linear_copy == DMA_OP::dst && !no_linear_opt)
          Transpose();
        else
          TransposeDeslice();
      } else if (TileToSymbol()) {
        if (opt_to_linear_copy == DMA_OP::src && !no_linear_opt)
          Transpose();
        else
          SliceTranspose();
      } else if (TileToTile()) {
        choreo_unreachable("slice-transpose-deslice is not supported now.");
      }
    }

    if (local_in_warp || shared_in_block) {
      DecrDeviceIndent();
      ds << d_indent << "} // single instance\n";
      if (!fty->IsAsync()) {
        // not async, must syncthreads immediately
        // else, defer the sync till the wait time
        if (shared_in_block) ds << d_indent << "__syncthreads();\n";
        if (local_in_warp) ds << d_indent << "__syncsubthreads();\n";
      }
    }
  };

  if (!f_opt_condition.empty() || !t_opt_condition.empty()) {
    // need generating runtime conditional optimization
    std::ostringstream condition;
    if (!f_opt_condition.empty())
      condition << ValueListSTR(f_opt_condition, " && ");
    if (!t_opt_condition.empty()) {
      if (!f_opt_condition.empty()) condition << " && ";
      condition << ValueListSTR(t_opt_condition, " && ");
    }
    IndStream() << "if (" << condition.str() << ") {\n";
    IncrDeviceIndent();
    DMACodeGen(false);
    DecrDeviceIndent();
    IndStream() << "} else {\n";
    IncrDeviceIndent();
    DMACodeGen(true);
    DecrDeviceIndent();
    IndStream() << "} // end DMA opt: " << n.LOC() << "\n";
  } else {
    DMACodeGen(false);
  }

  return true;
}

bool TopsccCodeGen::Visit(AST::Rotate& n) {
  TraceEachVisit(n);

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

bool TopsccCodeGen::Visit(AST::Synchronize& n) {
  TraceEachVisit(n);

  switch (n.scope->Get()) {
  case Storage::GLOBAL:
    hs << h_indent << "choreo::abend_true(topsDeviceSynchronize());\n";
    break;
  case Storage::SHARED: ds << d_indent << "__syncthreads();\n"; break;
  case Storage::LOCAL: ds << d_indent << "__syncsubthreads();\n"; break;
  default:
    choreo_unreachable("unsupported synchronization type: " + PSTR(n.scope) +
                       ".");
  }

  return true;
}

bool TopsccCodeGen::Visit(AST::Wait& n) {
  TraceEachVisit(n);

  bool local_in_warp = false, shared_in_block = false;
  for (auto& f : n.GetTargets()) {
    if (!isa<FutureType>(NodeType(*f))) continue;
    assert(cast<AST::Expr>(f)->GetSymbol());
    auto name = cast<AST::Expr>(f)->GetSymbol()->name;
    shared_in_block |= IsFutureBlockShared(InScopeName(name));
    local_in_warp |= IsFutureWarpLocal(InScopeName(name));
  }
  assert(!(local_in_warp && shared_in_block) &&
         "local and shared memory should not be used at the same time");
  if (local_in_warp)
    assert(CCtx().GetArch() == TargetArch::GCU4 &&
           "only gcu400 need handle local synchronization");

  if (shared_in_block || local_in_warp) {
    ds << d_indent << "if (" << SingleInstancePredicate(shared_in_block)
       << ") {\n";
    IncrDeviceIndent();
  }

  for (auto& f : n.GetTargets()) {
    auto expr = cast<AST::Expr>(f);
    bool is_array_ref = (expr->op == "elemof");
    if (isa<FutureType>(NodeType(*f))) {
      assert(!IsHost());
      ds << d_indent << ExprSTR(f, false) << ".wait();\n";
    } else if (auto ety = dyn_cast<EventArrayType>(NodeType(*f))) {
      if (IsHost())
        choreo_unreachable("yet to support: wait global event in host.");
      switch (ety->GetStorage()) {
      case Storage::GLOBAL:
      case Storage::SHARED:
      case Storage::LOCAL: {
        ds << d_indent << "// wait event " << PSTR(f) << "\n";
        ds << d_indent << "while (";
        if (is_array_ref) {
          size_t lvl = GetSubScriptLevel(*expr);
          auto bid = AST::GetArrayBaseSymbol(*expr);
          auto bty =
              cast<EventArrayType>(GetSymbolType(UnScopedName(bid->name)));
          // TODO: "!" same here
          GenerateSubscriptions(ds, "!" + ExprSTR(f, false), " || ",
                                bty->RemainderDimensions(lvl));
        } else
          GenerateSubscriptions(ds, "!" + ExprSTR(f, false), " || ",
                                ety->RemainderDimensions(0));
        ds << "false) continue;\n";
        ds << d_indent << "// reset event " << PSTR(f) << "\n";
        if (is_array_ref) {
          size_t lvl = GetSubScriptLevel(*expr);
          auto bid = AST::GetArrayBaseSymbol(*expr);
          auto bty =
              cast<EventArrayType>(GetSymbolType(UnScopedName(bid->name)));
          GenerateSubscriptions(ds, d_indent + ExprSTR(f, false), " = false;\n",
                                bty->RemainderDimensions(lvl));
        } else
          GenerateSubscriptions(ds, d_indent + ExprSTR(f, false), " = false;\n",
                                ety->RemainderDimensions(0));
      } break;
      default:
        choreo_unreachable("unsupported event array storage '" +
                           STR(ety->GetStorage()) + "'.");
      }
    } else if (auto ety = dyn_cast<EventType>(NodeType(*f))) {
      if (IsHost())
        choreo_unreachable("yet to support: wait global event in host.");
      switch (ety->GetStorage()) {
      case Storage::GLOBAL:
      case Storage::SHARED:
      case Storage::LOCAL: {
        ds << d_indent << "while (" << ExprSTR(f, false)
           << " == false) continue; // spinlock\n";
        if (is_array_ref) {
          ds << d_indent << "// reset event " << PSTR(f) << "\n";
          size_t lvl = GetSubScriptLevel(*expr);
          auto bid = AST::GetArrayBaseSymbol(*expr);
          auto bty =
              cast<EventArrayType>(GetSymbolType(UnScopedName(bid->name)));
          GenerateSubscriptions(ds, d_indent + ExprSTR(f, false), " = false;\n",
                                bty->RemainderDimensions(lvl));
        } else
          ds << d_indent << ExprSTR(f, false) << " = false; // reset event\n";
      } break;
      default:
        choreo_unreachable("unsupported event storage '" +
                           STR(ety->GetStorage()) + "'.");
      }
    }
  }

  if (shared_in_block || local_in_warp) {
    DecrDeviceIndent();
    ds << d_indent << "}\n";
    if (shared_in_block) ds << d_indent << "__syncthreads();\n";
    if (local_in_warp) ds << d_indent << "__syncsubthreads();\n";
  }

  return true;
}

bool TopsccCodeGen::Visit(AST::Break& n) {
  TraceEachVisit(n);
  IndStream() << "break;\n";
  return true;
}

bool TopsccCodeGen::Visit(AST::Continue& n) {
  TraceEachVisit(n);
  IndStream() << "continue;\n";
  return true;
}

bool TopsccCodeGen::Visit(AST::Trigger& n) {
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
        hs << h_indent << "choreo::abend_true(topsMemset(&" << ExprSTR(f, true)
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
        hs << h_indent << "choreo::abend_true(topsMemset(&" << ExprSTR(f, true)
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

bool TopsccCodeGen::Visit(AST::Call& n) {
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
          if (CCtx().GetArch() == TargetArch::GCU20 ||
              CCtx().GetArch() == TargetArch::GCU21) {
            print_format += "%d";
            print_args += "(" + ExprSTR(arg, IsHost()) + " ? 1 : 0), ";
          } else {
            print_format += "%s";
            print_args +=
                "(" + ExprSTR(arg, IsHost()) + " ? \"true\" : \"false\"), ";
          }
        } else if (BaseType bt = type->GetBaseType();
                   IsFloatPointBaseType(bt)) {
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
        } else if (isa<BoundedITupleType>(type)) {
          print_format += "{";
          for (int i = 0; i < (int)e->s.Rank(); ++i) {
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
    } else if (n.IsAnno()) {
    } else
      choreo_unreachable("the bif '" + n.function->name +
                         "' is not supported by this target.");
  }

  if (!n.IsExpr()) os << indent << CallSTR(n) << ";\n";

  return true;
}

bool TopsccCodeGen::Visit(AST::ParamList& n) {
  int index = 0;
  for (auto param : n.values)
    updating_cgi->AddSymbolDetail(fname, {InScopeName(param->sym->name),
                                          param->GetType(), param->pass_by_ref,
                                          index++, param->GetAttr()});
  return true;
}

bool TopsccCodeGen::Visit(AST::WithIn& n) {
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
      updating_cgi->AddSymbolDetail(fname,
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

bool TopsccCodeGen::Visit(AST::WhereBind& n) {
  TraceEachVisit(n);

  // TODO
  choreo_unreachable("where bind is yet to support.");

  return true;
}

bool TopsccCodeGen::Visit(AST::WithBlock& n) {
  TraceEachVisit(n);
  // anything required?
  return true;
}

bool TopsccCodeGen::Visit(AST::ForeachBlock& n) {
  TraceEachVisit(n);

  for (auto& rn : n.GetRanges()) {
    auto rng = cast<AST::LoopRange>(rn);
    auto cname = rng->IVName();
    for (auto iv_name : within_map.at(InScopeName(cname))) {
      auto iv_ty = GetSymbolType(UnScopedName(iv_name));
      assert(IsActualBoundedIntegerType(iv_ty));
      auto iv_bty = cast<BoundedITupleType>(iv_ty);
      assert(iv_bty);
      auto stride = iv_bty->GetStride(0);
      auto width = iv_bty->GetWidth(0);
      int increment = stride * width;
      IndStream() << "for (" << SSMName(iv_name, IsHost()) << " = "
                  << (rng->lbound ? ("(" + ExprSTR(rng->lbound, IsHost()) + ")")
                                  : "0")
                  << "; " << SSMName(iv_name, IsHost()) << " < "
                  << UnScopedExpr(ValueSTR(iv_bty->GetUpperBound()))
                  << (rng->ubound ? (" + " + ExprSTR(rng->ubound, IsHost()))
                                  : "")
                  << "; "
                  << (increment != 1 ? (SSMName(iv_name, IsHost()) +
                                        " += " + std::to_string(increment))
                                     : ("++" + SSMName(iv_name, IsHost())))
                  << ") {\n";

      IncrIndent();
    }
  }

  return true;
}

bool TopsccCodeGen::Visit(AST::InThreadsBlock& n) {
  TraceEachVisit(n);
  assert(!IsHost());
  ds << d_indent << "// inthreads: " << n.LOC() << "\n";
  if (!n.stmts->None())
    ds << d_indent << "if (" << ExprSTR(n.pred, false) << ") {\n";
  IncrDeviceIndent();
  return true;
}

bool TopsccCodeGen::Visit(AST::IfElseBlock& n) {
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

bool TopsccCodeGen::Visit(AST::WhileBlock& n) {
  TraceEachVisit(n);

  IndStream() << "// while: " << n.LOC() << "\n";
  IndStream() << "while (" << ExprSTR(n.pred, IsHost()) << ") {\n";
  IncrIndent();

  return true;
}

bool TopsccCodeGen::Visit(AST::Return& n) {
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
        hs << h_indent << "choreo::abend_true(topsMemcpy(" << sym << ".data(), "
           << sym << "__device, " << UnScopedSizeExpr(*sty)
           << ", topsMemcpyDeviceToHost));\n";
        return_stream << "return choreo::copy_as_spanned(" << sym << ".data(), "
                      << sym << ".shape());\n";
      } else if (IsChoreoOutput(InScopeName(sym))) {
        // return the global storage, must map back
        hs << h_indent << "choreo::abend_true(topsMemcpy(" << sym << ".data(), "
           << sym << "__device, " << UnScopedSizeExpr(*sty)
           << ", topsMemcpyDeviceToHost));\n";
        return_stream << "return " << sym << ";\n";
      } else {
        choreo_unreachable("unexpected situation");
      }
    } else if (auto expr = cast<AST::Expr>(n.value);
               expr && expr->op == "dataof") {
      // return future.data, must map back
      auto id = cast<AST::Expr>(expr->GetR())->GetSymbol();
      assert(id && "expect a symbol");
      auto sym = id->name + "__buf__";
      hs << h_indent << "choreo::abend_true(topsMemcpy(" << sym << ".data(), "
         << sym << "__device, " << UnScopedSizeExpr(*sty)
         << ", topsMemcpyDeviceToHost));\n";
      return_stream << "return " << ExprSTR(n.value, true) << ";\n";
    } else {
      choreo_unreachable("not support return value of type: " + PSTR(vty));
    }
  } else {
    choreo_unreachable("not support return value of type: " + PSTR(vty));
  }

  EmitTopsFree();

  hs << h_indent << return_stream.str();

  return true;
}

bool TopsccCodeGen::Visit(AST::CppSourceCode& n) {
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

void TopsccCodeGen::EmitHostFuncDecl(std::ostringstream& oss) {
  // handle the return type
  if (!void_return) {
    if (cgi->HasReturnSymbol(fname)) {
      auto& item = cgi->GetReturnDetail(fname);
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

void TopsccCodeGen::EmitHostRuntimeCheck() {
  // check if the input shape is as declared in choreo
  if (cgi->ParameterCount(fname) == 0) return;

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

void TopsccCodeGen::EmitMemReuse(const std::string& df_name) {
  const auto& script = FCtx(fname).GetMemReuseScript(df_name);
  if (!script.has_value()) return;
  hs << h_indent << R"(// JIT memory reuse begin)" << "\n";
  for (const auto& s : script.value()) { hs << h_indent << s << "\n"; }
  hs << h_indent << R"(// JIT memory reuse end)" << "\n";
}

static inline const std::string
DeviceParamTypeStringify(const Choreo::Type& ty) {
  if (isa<VoidType>(&ty))
    return "void";
  else if (isa<S8Type>(&ty))
    return "char";
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
  else if (isa<F8Type>(&ty))
    return "choreo::half8";
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
    return std::string(NameBaseType(sty->ElementType(), false)) + " *";
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

void TopsccCodeGen::EmitTopsFree() {
  assert(IsHost());
  for (const auto& item : GetDeviceFuncIns(updating_cgi)) {
    if (!PrefixedWith(scoped_symtab.ScopeName(), GetScope(item.name))) continue;
    if (!isa<SpannedType>(item.type)) continue;
    if (item.attr == ParamAttr::GLOBAL_INPUT) continue;
    if (!NeedDeviceFunc() && !IsChoreoOutput(item.name)) continue;
    hs << h_indent << "choreo::abend_true(topsFree(" << UnScopedName(item.name)
       << "__device));\n";
  }
}

void TopsccCodeGen::EmitDeviceFuncDecl(std::ostringstream& oss) {
  if (CCtx().GetArch() == TargetArch::GCU4) {
    auto& lconfig = cgi->GetFunctionLaunches(fname)[parallel_idx];
    oss << "__thread_dims__(" << lconfig.warp_dim_x << ", "
        << lconfig.warp_dim_y << ", " << lconfig.warp_dim_z << ")\n";
  }

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

  const auto& offset_args =
      FCtx(fname).GetMemReuseOffsetArgs(SSTab().ScopeName());
  if (offset_args.has_value())
    for (const auto& [_, offsets] : offset_args.value())
      for (size_t idx = 0; idx < offsets.size(); ++idx) {
        auto dname = RegexReplaceAll(offsets[idx], "::", "_");
        oss << ((index++ > 0) ? ", " : "") << "unsigned long " << dname;
      }

  oss << ")";

  VST_DEBUG(dbgs() << "Device function prototype:\n" << oss.str() << "\n");
}

void TopsccCodeGen::EmitSource() {
  for (auto& code : code_segments) outs() << code << "\n";
}

void TopsccCodeGen::EmitScript(std::ostream& os, const std::string& exe_fn) {
  auto filename = RemoveDirectoryPrefix(
      RemoveSuffix(OptionRegistry::GetInstance().GetInputFileName(), ".co"));
  os << R"script(#!/usr/bin/env bash

# This is the choreo generated bash script to compile topscc code

if [[ -z ${TOPSCC_INSTALL} ]]; then
  if [[ \"$1\" == \"-st\" ]]; then
    TOPSCC_INSTALL=/opt/tops;
    shift 1;
  fi
)script";

  if (use_system_toolchain) {
    os << R"script(
	# Search for the binary in the PATH
	FOUND_PATH=$(which "topscc" 2>/dev/null)

	if [ -n "$FOUND_PATH" ]; then
		# If the binary is found, extract the installation path
		TOPSCC_INSTALL=$(dirname "$(dirname "$FOUND_PATH")")
	elif [[ -f /opt/tops/bin/topscc ]]; then
		# Search for the default topscc installation directory
		TOPSCC_INSTALL=/opt/tops
  elif [[ -d )script"
       << STRINGIZE(__CHOREO_TOPSCC_DIR__) << " ]]; then\n";
    os << "    TOPSCC_INSTALL=" << STRINGIZE(__CHOREO_TOPSCC_DIR__);
    os << R"script(
  fi
)script";
  } else
    os << "  TOPSCC_INSTALL=" << STRINGIZE(__CHOREO_TOPSCC_DIR__) << "\n";

  os << R"script(
fi

if [[ -z "${TOPSCC_INSTALL}" ]]; then
  echo "failed to find the topscc installation."
  echo "install topscc or set TOPSCC_INSTALL to topscc installation directory."
  exit 1
fi

TOPSCC=${TOPSCC_INSTALL}/bin/topscc
TOPSCC_LIB=${TOPSCC_INSTALL}/lib

)script";
#ifdef __CHOREO_GCU_ACORE_DIR__
  os << R"(if [[ -z "${ACORE_INSTALL}" ]]; then)" << "\n";
  os << "  ACORE_INSTALL=" << STRINGIZE(__CHOREO_GCU_ACORE_DIR__) << "\n";
  os << "fi\n";
#endif
  os << R"script(
if [[ ! -z "${ACORE_INSTALL}" ]]; then
  GCU_ACORE_INCLUDE=${ACORE_INSTALL}/include
  GCU_ACORE_LIB_PATH=${ACORE_INSTALL}/lib
  GCU_ACORE_LIB=libacoreop.bc
fi
)script";

  auto build_path = CreateUniquePath();
  auto cc_file = build_path + "/__choreo_topscc_" + filename + ".cpp";
  auto exe_file = exe_fn;
  if (exe_file.empty())
    exe_file = build_path + "/__choreo_topscc_" + filename + ".exe";
  auto fb_file = "__choreo_topscc_" + filename + ".topsfb";
  os << "rm -fr " << build_path << "\n";
  os << "mkdir -p " << build_path << "\n\n";

  // place the choreo header
  os << "cat <<'EOF' > " << build_path << "/choreo.h\n";
  os << __choreo_header_as_string << "\nEOF\n\n";

  os << "cat <<'EOF' > " << cc_file << "\n";
  for (auto& code : code_segments) os << code << "\n";
  os << "\nEOF\n\n";

  // use simulator at this time
  bool use_sim = (CCtx().GetArch() == TargetArch::GCU4);

  // JIT: detect the environment
  if (use_sim)
    os << "gcu_arch=gcu400\n";
  else if (((CCtx().GetOutputKind() == OutputKind::TargetModule) ||
            (CCtx().GetOutputKind() == OutputKind::TargetExecutable) ||
            (CCtx().GetOutputKind() == OutputKind::ShellScript)) &&
           arch.GetValue() != "") {
    // enforce the arch type
    os << "gcu_arch=" << ToLower(STR(CCtx().GetArch())) << "\n";
  } else
    os << R"script(
  # check the device just-in-time
  # TODO: improve the target check with more solid code
  GCU_DEVICE_STR="$(lspci | grep Enflame | head -1)"
  # echo $GCU_DEVICE_STR
  if [[ "${GCU_DEVICE_STR}" == *"S60G"* ]]; then
    gcu_arch=gcu300
  elif [[ "${GCU_DEVICE_STR}" == *"c035"* ]]; then
    gcu_arch=gcu300
    export TOPS_VISIBLE_DEVICES=1
  elif [[ "${GCU_DEVICE_STR}" == *"S60"* ]]; then
    gcu_arch=gcu300
  elif [[ "${GCU_DEVICE_STR}" == *"I20"* ]]; then
    gcu_arch=gcu210
    export TOPS_VISIBLE_DEVICES=1
  elif [[ "$(lspci | grep Tencent)" != "" ]]; then
    gcu_arch=gcu210
  else
    echo "can not determine the GCU device type."
    exit 1
  fi
  )script";

  os << R"script(
show_usage() {
  echo "  Usage: $0 <-st> <actions>"
  echo ""
  echo "  Options:"
  echo "   -st,                 Use default system path for target compilation"
  echo "   --execute,           Compile and execute"
  echo "   --compile-link,      Compile and link"
  echo "   --compile-module,    Compile and generate the module"
  echo "   --gen-fatbin,        Compile and generate the fatbin"
  echo ""
  echo "  Environment Variables:"
  echo "   EXTRA_TARGET_CFLAGS: Extra target compilation flags"
  echo "   TOPSCC_INSTALL:      Topscc compiler installation path"
  echo "   ACORE_INSTALL:       GCU Acore library installation path"
  exit 1
}

option_detect() {
  local tmpf=/tmp/$(date +"%Y%m%d_%H%M%S.%3N")__nasty_option_detect__.cpp
  echo "#include <krt/builtins.h>" > ${tmpf}
  echo "__device__ void foo() { tops::abort();  }" >> ${tmpf}

  ${TOPSCC} ${CFLAGS} -c ${tmpf} -o /dev/null 2>/dev/null

  if [[ $? -eq 0 ]]; then
    export CFLAGS="${CFLAGS} -D__CHOREO_USE_TOPS_ABORT__";
  fi

  rm -f ${tmpf}
}

# compile, execute
)script";

  os << R"(export CFLAGS="-arch ${gcu_arch} -std=c++17 -ltops -lm -O3)";
  if (CCtx().GenDebugInfo()) os << " -g";
  if (!target_options.GetValue().empty())
    os << " " << target_options.GetValue();
  if (use_pic) os << " -fPIC";
  if (verbose) os << " -v"; // if it requires to be verbose
#ifdef __CHOREO_GCU_ACORE_DIR__
  os << R"( --tops-device-lib-path=${GCU_ACORE_LIB_PATH})";
  os << R"( --tops-device-lib=${GCU_ACORE_LIB})";
  os << R"( -I${GCU_ACORE_INCLUDE})";
  os << R"( -D__ACORE_OP__ -fPIC)";
#endif
  // always enclose
  os << " ${EXTRA_TARGET_CFLAGS}";
  std::filesystem::path cwd = std::filesystem::current_path();
  auto input_file = OptionRegistry::GetInstance().GetInputFileName();
  auto input_abs_path = GetAbsPath(cwd.string(), input_file);
  os << " -I" << input_abs_path;
  for (auto inc_path : CCtx().GetIncPaths()) os << " -I" << inc_path;
  for (auto lib_path : CCtx().GetLibPaths()) os << " -L" << lib_path;
  for (auto lib : CCtx().GetLibs()) os << " -l" << lib;
  for (auto macro : CCtx().GetCLMacros())
    os << " -D" << macro.first
       << (macro.second.empty() ? "" : ("=" + macro.second));

  os << "\"";
  os << "\noption_detect";
  if (use_sim) os << "\nexport INTERNAL_GCU_SIM=LIBRA";
  os << "\nexport LD_LIBRARY_PATH=${TOPSCC_LIB}:${LD_LIBRARY_PATH}\n\n";

  os << R"(if [ "$1" == "--execute" ] || [ "$#" -eq 0 ]; then)";
  if (verbose)
    os << "\n  echo ${TOPSCC} ${CFLAGS} " << cc_file << " -o " << exe_file;
  os << "\n  ${TOPSCC} ${CFLAGS} " << cc_file << " -o " << exe_file;
  if (verbose) os << "\n  echo " << exe_file << "\n";
  os << "\n  " << exe_file << "\n";
  os << R"(elif [ "$1" == "--compile-module" ]; then)";
  if (verbose)
    os << "\n  echo ${TOPSCC} -c ${CFLAGS} " << cc_file << " -o " << exe_file
       << "\n";
  os << "\n  ${TOPSCC} -c ${CFLAGS} " << cc_file << " -o " << exe_file << "\n";
  os << R"(elif [ "$1" == "--compile-link" ]; then)";
  if (verbose)
    os << "\n  echo ${TOPSCC} ${CFLAGS} " << cc_file << " -o " << exe_file
       << "\n";
  os << "\n  ${TOPSCC} ${CFLAGS} " << cc_file << " -o " << exe_file << "\n";
  os << R"(elif [ "$1" == "--gen-fatbin" ]; then)";
  os << "\n  __cur_dir=$(pwd)";
  os << "\n  cd " << build_path;
  if (verbose)
    os << "\n  echo ${TOPSCC} -save-temps -c ${CFLAGS} " << cc_file << " -o "
       << exe_file << "\n";
  os << "\n  ${TOPSCC} -save-temps -c ${CFLAGS} " << cc_file << " -o "
     << exe_file << "\n";
  if (verbose)
    os << "\n  echo cp " << cc_file
       << "-tops-dtu-enflame-tops.topsfb ${__cur_dir}/" << fb_file;
  os << "\n  cp " << cc_file << "-tops-dtu-enflame-tops.topsfb ${__cur_dir}/"
     << fb_file;
  os << "\n  cd ${__cur_dir}";
  os << "\n  echo \"Fatbin file generated: " << fb_file << "\"";
  os << "\nelse show_usage";
  os << "\nfi";
}

bool TopsccCodeGen::CompileWithScript(const std::string& action) {
  assert(!action.empty() && "no action is specified.");

  char tempFileName[] = "/tmp/choreo_topscc_script_XXXXXX";
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
const std::string TopsccCodeGen::ValueSTR(const ValueItem& vi,
                                          bool LL_suffix) const {
  return OpValueSTR(vi, "", true, LL_suffix);
}

const std::string TopsccCodeGen::ValueListSTR(const ValueList& vl,
                                              std::string sep,
                                              bool LL_suffix) const {
  std::ostringstream oss;
  if (!vl.empty()) {
    oss << ValueSTR(vl[0], LL_suffix);
    for (unsigned i = 1; i < vl.size(); ++i)
      oss << sep << ValueSTR(vl[i], LL_suffix);
  }
  return oss.str();
}

const std::string TopsccCodeGen::OpValueSTR(const ValueItem& vi,
                                            const std::string& parent_op,
                                            const bool is_left_child,
                                            bool LL_suffix) const {
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
        iv <= (int64_t)std::numeric_limits<int32_t>::min())
      return PSTR(vi) + "LL";
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
    std::string res = op + OpValueSTR(uo->GetOperand(), op, false, LL_suffix);
    return WrapParen(res, op);
  } else if (auto bo = VIBop(vi)) {
    if (bo->GetOpCode() == OpCode::ADD) {
      if (auto rv = VIInt(bo->GetRight()); rv && rv.value() < 0) {
        std::string res = OpValueSTR(bo->GetLeft(), "-", true, LL_suffix) +
                          " - " + std::to_string(-rv.value());
        if (rv.value() >= (int64_t)std::numeric_limits<int32_t>::max() ||
            rv.value() <= (int64_t)std::numeric_limits<int32_t>::min())
          res += "LL";
        return WrapParen(res, "-");
      }
    }
    std::string op = STR(bo->GetOpCode());
    std::string res = OpValueSTR(bo->GetLeft(), op, true, LL_suffix) + " " +
                      op + " " +
                      OpValueSTR(bo->GetRight(), op, false, LL_suffix);
    return WrapParen(res, op);
  } else if (auto to = VITop(vi)) {
    std::string op = "?";
    std::string res = OpValueSTR(to->GetPred(), op, true) + " ? " +
                      OpValueSTR(to->GetLeft(), op, true, LL_suffix) + " : " +
                      OpValueSTR(to->GetRight(), op, false, LL_suffix);
    return WrapParen(res, op);
  } else
    choreo_unreachable("unsupported value.");
  return "";
}

std::optional<std::string>
TopsccCodeGen::ThreadIdString(const ptr<AST::Identifier>& id) const {
  if (id == nullptr) return std::nullopt;
  auto ty = NodeType(*id);
  if (isa<BoundedType>(ty) &&
      PrefixedWith(cast<BoundedType>(ty)->GetNote(), "pv")) {
    auto l = RemovePrefixOrNull("pv:", cast<BoundedType>(ty)->GetNote());
    assert(l.has_value());
    // is marked as parallel whose level is decided by target check
    if (*l == "local")
      return "__tops_tid_x()";
    else if (*l == "shared")
      return "__tops_bid_x()";
    else if (*l == "sub-local")
      return "__tops_stid_x()";
    else
      choreo_unreachable("invalid bounded type note.");
  }
  return std::nullopt;
}

std::optional<std::string>
TopsccCodeGen::SubThreadIdString(const ptr<AST::Identifier>& id) const {
  auto ty = NodeType(*id);
  std::ostringstream oss;
  if (isa<BoundedType>(ty) &&
      PrefixedWith(cast<BoundedType>(ty)->GetNote(), "pi")) {
    auto l = RemovePrefixOrNull("pi:", cast<BoundedType>(ty)->GetNote());
    assert(l.has_value());
    // l should be (x|y|z):(shared|local)
    if (l->length() <= 3)
      choreo_unreachable("invalid bounded type note: " +
                         cast<BoundedType>(ty)->GetNote() + ".");
    oss << "__tops_";
    if (l->substr(2) == "local")
      oss << "tid_";
    else if (l->substr(2) == "shared")
      oss << "bid_";
    else
      choreo_unreachable("invalid bounded type note.");
    if (l->at(0) > 'z' || l->at(0) < 'x')
      choreo_unreachable("invalid bounded type note: " +
                         cast<BoundedType>(ty)->GetNote() + ".");
    oss << l->at(0) << "()";
    return oss.str();
  }
  return std::nullopt;
}

// input is a `node` or `std::variant<int, float>`.
// If `val` is existed, use it first.
const std::string
TopsccCodeGen::ExprCastSTR(AST::ptr<AST::Node> n,
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

  if (t == BT::F8 || f == BT::F8)
    choreo_unreachable("unsupport cast: '" + STR(f) + "' to '" + STR(t) + "'");

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
    auto nbt = NameBaseType(t, is_host);
    if (IsBoolIntegerBaseType(f))
      res << "static_cast<" << nbt << ">(" << value << ")";
    else if (IsFloatPointBaseType(f)) {
      if (f != BT::F32 && f != BT::F64)
        res << "static_cast<" << nbt << ">("
            << ExprCastSTR(n, val, BT::F32, f, is_host) << ")";
      else
        res << "static_cast<" << nbt << ">(" << value << ")";
    }
    break;
  }
  case BT::F64: {
    if (IsBoolIntegerBaseType(f))
      res << "static_cast<double>(" << value << ")";
    else
      res << "static_cast<double>(" << ExprCastSTR(n, val, BT::F32, f, is_host)
          << ")";
    break;
  }
  case BT::F32: {
    if (IsBoolIntegerBaseType(f))
      res << "static_cast<float>(" << value << ")";
    else {
      if (f == BT::F16)
        res << "f16_to_f32(" << value << ")";
      else if (f == BT::BF16)
        res << "(float)(" << value << ")";
      else if (f == BT::F64)
        res << "static_cast<float>(" << value << ")";
      else
        choreo_unreachable("unsupport cast: '" + STR(f) + "' to '" + STR(t) +
                           "'");
    }
    break;
  }
  case BT::F16:
    res << "f32_to_f16(" << ExprCastSTR(n, val, BT::F32, f, is_host) << ")";
    break;
  case BT::BF16:
    res << "choreo::bf16(" << ExprCastSTR(n, val, BT::F32, f, is_host) << ")";
    break;
  default:
    choreo_unreachable("unsupport cast: '" + STR(f) + "' to '" + STR(t) + "'");
  }

  return res.str();
}

const std::string TopsccCodeGen::AddressSTR(const Shape& shape,
                                            const AST::DataAccess& da,
                                            bool is_host) const {
  size_t idx = 0;
  std::ostringstream oss;
  auto AppendOffset = [this, &oss, &shape, &idx](const ValueItem& op) {
    auto offset = op;
    assert(shape.Rank() >= idx + 1);
    if (shape.Rank() > idx + 1)
      offset = offset * shape.TrimDims(idx + 1).ElementCountValue();
    SimplifyExpression(offset);
    if (!sbe::ceq(offset, sbe::nu(0))) oss << " + " << ValueSTR(offset);
    ++idx;
  };
  for (auto item : da.GetIndices()) {
    if (auto id = AST::GetIdentifier(item)) {
      if (auto ids = ThreadIdString(id))
        AppendOffset(sbe::sym(ids.value()));
      else if (auto sids = SubThreadIdString(id))
        AppendOffset(sbe::sym(sids.value()));
      else if (within_map.count(InScopeName(id->name))) {
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
  return oss.str();
}

const std::string TopsccCodeGen::ExprSTR(AST::ptr<AST::Node> e,
                                         bool is_host) const {
  // start with the lowest precedence op ""
  return OpExprSTR(e, "", true, is_host);
}

const std::string TopsccCodeGen::OpExprSTR(AST::ptr<AST::Node> e,
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
    if (auto ids = ThreadIdString(id))
      oss << ids.value();
    else if (auto sids = SubThreadIdString(id))
      oss << sids.value();
    else if (within_map.count(InScopeName(id->name)) && !is_host) {
      size_t i = 0;
      for (auto iv_name : within_map.at(InScopeName(id->name)))
        oss << ((i++ == 0) ? "" : ", ")
            << UnScopedName(ssm.DeviceName(iv_name));
    } else
      oss << UnScopedName(SSMName(InScopeName(id->name), is_host));
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
    if (auto sty = GetSpannedType(GetSymbolType(da->data->name))) {
      if (auto da_ty = dyn_cast<VectorType>(da->GetType())) {
        oss << DASTR(da);
      } else {
        oss << "*((" << NameBaseType(sty->ElementType()) << "*)"
            << OpExprSTR(da->data, "+", true, is_host);
        auto shape = sty->GetShape();

        oss << AddressSTR(shape, *da, is_host);
      }
    } else {
      assert(!da->AccessElement());
      assert(!within_map.count(InScopeName(da->data->name)));
      oss << UnScopedName(SSMName(InScopeName(da->data->name), is_host));
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
        auto caty = cast<SpannedType>(ca->GetType());
        std::string res;
        if (isa<FutureType>(NodeType(*ca->data)))
          res = OpExprSTR(ca->data, parent_op, true, is_host) + ".data() + " +
                GenOffset(ca);
        else
          res = OpExprSTR(ca->data, "+", true, is_host) + " + " + GenOffset(ca);
        return WrapParen(res, "+");
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
      } else if (expr->GetOp() == "dataof") {
        assert(isa<FutureType>(expr->GetR()->GetType()) &&
               "expect a future operand.");
        if (auto id = cast<AST::Expr>(expr->GetR())->GetSymbol()) {
          if (is_host)
            oss << id->name << "__buf__";
          else
            oss << id->name << ".data()";
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
        auto& op = expr->GetOp();
        if (op == "#" && IsActualBoundedIntegerType(l->GetType()) &&
            IsActualBoundedIntegerType(r->GetType())) {
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
                   IsActualBoundedIntegerType(l->GetType()) &&
                   isa<ScalarIntegerType>(r->GetType())) {
          oss << OpExprSTR(l, parent_op, is_left_child, is_host);
        } else if (op == "#/" || op == "#*" || op == "#%") {
          choreo_unreachable("unsupported expression op: '" + expr->GetOp() +
                             "', expr: " + PSTR(expr) + ".");
        } else {
          std::ostringstream res;
          res << OpExprSTR(l, op, true, is_host) << " " << op << " "
              << OpExprSTR(r, op, false, is_host);
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
  } else if (auto c = dyn_cast<AST::Call>(e)) {
    assert(!is_host);
    return CallSTR(*c);
  } else if (isa<AST::DataType>(e)) {
    return NameBaseType(e->GetType()->GetBaseType());
  } else
    choreo_unreachable("unsupported node type: " + e->TypeNameString() + ".");

  return oss.str();
}

const std::string TopsccCodeGen::CallSTR(AST::Call& n) const {
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

  if (n.IsAnno()) {
    if (n.function->name == "vectorize") {
      auto vty = n.GetType();
      assert(isa<VectorType>(vty) && "vectorize should be a vector type.");
      oss << "tcle::mid<" << VectorTypeSTR(dyn_cast<VectorType>(vty)) << ">("
          << ExprSTR(n.GetArguments()[0], IsHost()) << ")";
    }

    else
      choreo_unreachable(
          "unsupported annotation function: " + n.function->name + ".");
    return oss.str();
  }

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
      std::string bts{NameBaseType(sty->ElementType(), IsHost())};
      auto m_ty = sty->GetStorage();
      auto mem_attr = TopsParamStorage(m_ty);
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

const std::string TopsccCodeGen::DASTR(AST::ptr<AST::DataAccess>& da,
                                       const std::string& val_str,
                                       bool is_load) const {
  std::ostringstream oss;
  if (auto sty = GetSpannedType(GetSymbolType(da->data->name))) {
    auto da_ty = da->GetType();
    if (isa<VectorType>(da_ty)) {
      auto data_name = da->GetDataName();
      auto leaptr_name = ssm.DeviceName(InScopeName(data_name) + da->Id());
      if (is_load) {
        oss << leaptr_name << ".load()";
      } else {
        oss << leaptr_name << ".store(" << val_str << ")";
      }
    } else {
      oss << ExprSTR(da, false) << " = " << val_str;
    }
  }
  return oss.str();
}
