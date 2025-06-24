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

inline const char* LocalSharedPredicate() {
  return "threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0";
}
inline const char* SubSharedPredicate() {
  return "threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 && "
         "subThreadIdx.x == 0 && subThreadIdx.y == 0 && subThreadIdx.z == 0";
}

const char* SingleThreadPredicate() {
  if (arch.GetValue() == "gcu400") return SubSharedPredicate();
  return LocalSharedPredicate();
}

const char* SingleSubThreadPredicate() {
  static const char* pred_subthread =
      "subThreadIdx.x == 0 && subThreadIdx.y == 0 && subThreadIdx.z == 0";
  return pred_subthread;
}

inline std::string ImplicitPred(Storage max, Storage cur) {
  if (max == Storage::SUB) {
    if (cur == Storage::LOCAL)
      return SingleSubThreadPredicate();
    else if (cur == Storage::SHARED)
      return SubSharedPredicate();
  } else if (max == Storage::LOCAL && cur == Storage::SHARED)
    return LocalSharedPredicate();
  return "";
}

const char* SingleInstancePredicate(bool shared_in_block) {
  if (shared_in_block) return SingleThreadPredicate();
  return SingleSubThreadPredicate();
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

void GenerateSubscriptions(std::ostream& os, const std::string prefix,
                           const std::string suffix,
                           const std::vector<size_t>& dims) {
  std::vector<size_t> indices(dims.size());
  PrintSubscriptions(os, prefix, suffix, dims, indices);
}

} // namespace

bool TopsccCodeGen::BeforeVisitImpl(AST::Node& n) {
  if (trace_visit) dbgs() << "Before visiting " << n.TypeNameString() << "\n";

  if (isa<AST::Program>(&n)) {
    VST_DEBUG(dbgs() << STR(FBInfo()) << "\n");
    // emit the fixed headers
    EmitFixedHostHead();
    EmitFixedDeviceHead();
    ssm.EnterScope();
  } else if (isa<AST::ChoreoFunction>(&n)) {
    ResetChoreoFunctionStates();
    device_fn = "__choreo_device_" + fname;
    fty = cast<FunctionType>(GetSymbolType(fname));
    ssm.EnterScope();
  } else if (auto pb = dyn_cast<AST::ParallelBy>(&n)) {
    // only on device-side
    if (parallel_level == 0) {
      parallel_idx += 1;
      if (cgi->GetFunctionTrait(fname).multiple_parallelby)
        device_fn = "__choreo_device_" + fname + std::to_string(parallel_idx);
      EmitDeviceFuncDecl(ds);
      ds << " {\n";
      IncrDeviceIndent();
      ds << d_indent << "// parallel-by: " << n.LOC() << "\n";
    }
    parallel_level++;
    max_parallel_level = GetMaxParallelLevelFromNote(*pb);
    max_parallel_level_valid = true;
  } else if (isa<AST::WithBlock>(&n)) {
    if (IsHost()) {
      hs << h_indent << "// with-in: " << n.LOC() << "\n";
      hs << h_indent << "{\n";
      IncrHostIndent();
    } else {
      ds << d_indent << "// with-in: " << n.LOC() << "\n";
      ds << d_indent << "{\n";
      IncrDeviceIndent();
    }
  } else if (isa<AST::ForeachBlock>(&n)) {
    if (IsHost())
      hs << h_indent << "// foreach: " << n.LOC() << "\n";
    else
      ds << d_indent << "// foreach: " << n.LOC() << "\n";
  } else if (isa<AST::IfElseBlock>(&n)) {
    emit_call = false;
  } else if (isa<AST::IncrementBlock>(&n)) {
    if (IsHost()) {
      hs << h_indent << "// incr: " << n.LOC() << "\n";
      IncrHostIndent();
    } else {
      ds << d_indent << "// incr: " << n.LOC() << "\n";
      IncrDeviceIndent();
    }
  }

  if (!IsHost() && max_parallel_level_valid && !n.IsBlock() &&
      n.GetLevel() != Storage::NONE) {
    auto max_pl = GCUDeviceParallelLevel(max_parallel_level);
    auto pred = ImplicitPred(max_pl, n.GetLevel());
    if (!pred.empty()) {
      ds << d_indent << "if (" << pred << ") { // implicit inthreads\n";
      IncrDeviceIndent();
    }
  }

  return true;
}

bool TopsccCodeGen::InMidVisitImpl(AST::Node& n) {
  if (auto ie = dyn_cast<AST::IfElseBlock>(&n)) {
    if (!ie->HasElse()) return true;
    if (IsHost()) {
      DecrHostIndent();
      hs << h_indent << "} else {\n";
      IncrHostIndent();
    } else {
      DecrDeviceIndent();
      ds << d_indent << "} else {\n";
      IncrDeviceIndent();
    }
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
  } else if (isa<AST::ParallelBy>(&n)) {
    // only on device-side
    parallel_level--;
    if (parallel_level == 0) {
      max_parallel_level = 0;
      DecrDeviceIndent();
      ds << "}\n\n";
    }
  } else if (isa<AST::WithBlock>(&n)) {
    if (IsHost()) {
      DecrHostIndent();
      hs << h_indent << "}\n";
    } else {
      DecrDeviceIndent();
      ds << d_indent << "}\n";
    }
  } else if (auto fb = dyn_cast<AST::ForeachBlock>(&n)) {
    const auto& ranges = fb->GetRangeNodes();
    for (int j = ranges->Count() - 1; j >= 0; --j) {
      auto rng = cast<AST::LoopRange>(ranges->ValueAt(j));
      auto cname = rng->IVName();
      auto ivs = within_map.at(InScopeName(cname));
      for (auto iv_itr = ivs.rbegin(); iv_itr != ivs.rend(); ++iv_itr) {
        if (IsHost()) {
          DecrHostIndent();
          hs << h_indent << "} // " << UnScopedName(*iv_itr) << "\n";
          hs << h_indent << ssm.DeviceName(*iv_itr) << " = 0;\n"; // must reset
        } else {
          DecrDeviceIndent();
          ds << d_indent << "} // " << UnScopedName(*iv_itr) << "\n";
          ds << d_indent << ssm.DeviceName(*iv_itr) << " = 0;\n"; // must reset
        }
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
    if (IsHost()) {
      DecrHostIndent();
      hs << h_indent << "} // end if-else: " << ie->LOC() << "\n";
    } else {
      DecrDeviceIndent();
      ds << d_indent << "} // end if-else " << ie->LOC() << "\n";
    }
  } else if (auto ie = dyn_cast<AST::WhileBlock>(&n)) {
    if (IsHost()) {
      DecrHostIndent();
      hs << h_indent << "} // end while: " << ie->LOC() << "\n";
    } else {
      DecrDeviceIndent();
      ds << d_indent << "} // end while: " << ie->LOC() << "\n";
    }
  } else if (isa<AST::IncrementBlock>(&n)) {
    if (IsHost()) {
      DecrHostIndent();
      hs << h_indent << "}\n";
    } else {
      DecrDeviceIndent();
      ds << d_indent << "}\n";
    }
  }

  if (!IsHost() && max_parallel_level_valid && !n.IsBlock() &&
      n.GetLevel() != Storage::NONE) {
    auto max_pl = GCUDeviceParallelLevel(max_parallel_level);
    auto pred = ImplicitPred(max_pl, n.GetLevel());
    if (!pred.empty()) {
      DecrDeviceIndent();
      ds << d_indent << "} // end implicit inthreads\n";
    }
  }

  return true;
}

// tops::mdspan style offset
std::pair<std::string, size_t>
TopsccCodeGen::GenMdsOffset(const ptr<AST::ChunkAt> ca,
                            ptr<DMAConfig> config) const {
  auto& tsis = ca->AllTSInfo();
  assert(!tsis.empty());

  std::vector<std::ostringstream> offsets;
  // handle each chunkat inside a seqeunce like 'chunkat(a, b).chunat(c)...'
  for (size_t tsi_idx = 0; tsi_idx < tsis.size(); ++tsi_idx) {
    // For each chunkat expression, The tiled-block's shape is cooked by shape
    // inference. The block shape is different with the result shape of chunkat
    // expression when using 'modspan', where the result shape represents the
    // shape that applied mod (%) operation. Anyway, for offset, we only care
    // about the tiled-block's shape
    auto& shape = tsis[tsi_idx]->GetBlockShape();

    std::vector<std::string> exprs;
    // For each 'a, b, c, ...' inside 'chunkat(a, b, c, ...)', that 'b' inside
    // 'chunkat(a, b, c, ...)' could be bounded var like b = {b0, b1} Therefore,
    // we collect all the expressions first.
    for (size_t pi = 0; pi < tsis[tsi_idx]->GetIndices().size(); ++pi) {
      auto p = tsis[tsi_idx]->GetIndices()[pi];
      auto idx_exprs = SplitStringByDelimiter(ExprSTR(p, false));
      for (size_t i = 0; i < idx_exprs.size(); ++i)
        exprs.push_back(idx_exprs[i]);
    }

    auto tc = dyn_cast<TransposeConfig>(config);
    if (tc) {
      assert(tc->dim_values.size() == exprs.size());
      assert(tsis.size() == 1);
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
      if (tsi_idx > 0) offsets[i] << " + ";

      if (exprs[i] == "__choreo_no_tiling__")
        offsets[i] << "0";
      else
#ifndef USING_OP_INFO
        offsets[i] << "(int)(" << exprs[i] << " * ("
                   << UnScopedExpr(STR(shape.ValueAt(i))) << "))";
#else
        offsets[i] << "(int)((" << exprs[i] << ") * "
                   << UnScopedExpr(ValueItemAsString(shape.ValueAt(i))) << ")";
#endif
      // TODO: should consider precedence of `*`
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

const std::string TopsccCodeGen::GenOffset(const ptr<AST::ChunkAt>& ca) const {
  std::ostringstream offset;

  if (ca->NoTile()) return "";

  if (ca->AllTSInfo().size() > 1)
    choreo_unreachable("multiple chunkat is yet to support.");

  for (auto& tsi : ca->AllTSInfo()) {
    size_t i = 0;
    auto& shape = ca->GetShape();
    for (auto p : tsi->GetIndices()) {
      auto idx_exprs = SplitStringByDelimiter(ExprSTR(p, IsHost()));
      std::string factor = "1";
      if (shape.Rank() > i)
        factor = shape.TrimDims(i).GetElementCountExpression();
      for (auto i_expr : idx_exprs) {
        if (i != 0) offset << " + ";
        if (i_expr == "__choreo_no_tiling__")
          offset << "0";
        else
          offset << "(" << i_expr << " * " << factor << ")";
        ++i;
      }
    }
  }
  return offset.str();
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

)";

  if (arch.GetValue() == "gcu400" || arch.GetValue() == "gcu300")
    oss << "#include \"tcle.h\"\n";

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
      if (auto vale = VIStr(vi)) { // the dimension is symbolic
        assert(PrefixedWith(*vale, "::" + fname + "::") &&
               "unexpected symbolic dimension name.");

        auto dim_expr = hp_name + ".shape()[" + std::to_string(dim_index) + "]";
        if (symbolic_dimensions.count(*vale) == 0)
          symbolic_dimensions[*vale] = {dim_expr, hp_index, dim_index};
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
      ssm.MapDeviceSymbol(item.name, UnScopedName(item.name));
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

  // name the symbolic dimensions
  for (auto item : symbolic_dimensions) {
    // type of symbolic dims is deduced from API of span
    hs << h_indent << "auto " << UnScopedName(item.first) << " = "
       << item.second.hsd_expr << ";\n";
    ssm.MapHostSymbol(item.first, UnScopedName(item.first));
  }

  // emit the runtime checks
  EmitHostRuntimeCheck();

  // do not generate device function unless parallel-by exists
  if (NeedDeviceFunc()) {
    for (auto item : symbolic_dimensions)
      ssm.MapDeviceSymbol(item.first, UnScopedName(item.first));

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

  bool ref = (n.GetNote().find("ref") != std::string::npos);
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
        ds << "&" << ExprSTR(s->expr_list->ValueAt(i));
      }
      ds << "};\n";
      // make symbol a reference
      ds << d_indent << "future & " << sym << " = *" << array_sym << "["
         << ExprSTR(s->select_factor) << "];\n";
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
           << " = choreo::make_spandata<choreo::" << STR(sty->f_type) << ", "
           << shape.Rank() << ">({" << UnScopedExpr(RSTR(shape)) << "});\n";
        if (n.init_value) {
          // support initialization of output
          hs << h_indent << "std::fill(" << sym_data << ", " << sym_data << "+"
             << sym << ".element_count()"
             << ", "
             << ExprCastSTR(n.init_value, std::nullopt, GetBaseType(*sty),
                            TC2BT(n.init_value->GetType()->tc))
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
        return;
      }

      // the sym is not choreo output
      if (FBIContainsBuffer(FBInfo(), InScopeName(sym)) &&
          use_hetero_tileflow && IsHostSide()) {
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

      if (n.note.find("spm") != std::string::npos) {
        ds << d_indent << type_modifiers << bts << " " << sym << "["
           << UnScopedExpr(ElemCountExprOf(*sty)) << "];\n";
        return;
      }

      // the buffer is not the declared whole spm.
      auto notes = SplitStringByDelimiter(n.note, ", ");
      auto reuse_idx = std::find(notes.begin(), notes.end(), "reuse");
      auto offset_idx = std::find(notes.begin(), notes.end(), "offset");
      if (reuse_idx == notes.end()) {
        // the buffer is not reused
        // which means that it is declared but never used.
        assert(offset_idx == notes.end());
        // TODO: should we DCE the unused buffer?
        ds << d_indent << type_modifiers << bts << " " << sym << "["
           << UnScopedExpr(ElemCountExprOf(*sty)) << "];\n";
      } else {
        auto reuse_name = *(reuse_idx + 1);
        auto offset = *(offset_idx + 1);
        ds << d_indent << bts << "* " << sym << " = (" << bts << "*)"
           << "(" << reuse_name << " + " << offset << ");\n";
      }
    };

    if (sto == Storage::GLOBAL) {
      if (!IsHost()) choreo_unreachable("error: global var decl in device.");
      HandleGlobal();
      ssm.MapHostSymbol(InScopeName(sym) + "__device", buf_sym);
      ssm.MapHostSymbol(InScopeName(sym), buf_sym);
      ssm.MapDeviceSymbolIfNotExist(InScopeName(sym), sym);
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
         << "*)" << sym << ", " << UnScopedExpr(RSTR(sty->GetShape())) << "), "
         << ExprCastSTR(n.init_value, std::nullopt, GetBaseType(*sty),
                        TC2BT(n.init_value->GetType()->tc), false)
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
    if (IsHost())
      hs << h_indent << "int " << sym << " = " << ExprSTR(n.init_expr, false)
         << ";\n";
    else
      ds << d_indent << "int " << sym << " = " << ExprSTR(n.init_expr, false)
         << ";\n";
    return true;
  }

  // when symbol is not valued
  if (isa<ScalarType>(nty) && !FCtx(fname).HasSymbolValues(InScopeName(sym))) {
    (IsHost() ? hs : ds) << (IsHost() ? h_indent : d_indent)
                         << NameBaseType(GetBaseType(*nty), false) << " " << sym
                         << " = " << ExprSTR(n.init_expr, false) << ";\n";

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
      hs << h_indent << "choreo::abend_true(topsMalloc(&" << buf_sym << ", "
         << ety->ElemCount() << "));\n";
      hs << h_indent << "choreo::abend_true(topsMemset(&" << buf_sym << ", 0, "
         << ety->ElemCount() << "));\n";
      ssm.MapHostSymbol(sym, buf_sym);
      ssm.MapDeviceSymbol(sym, n.name_str);
    } break;
    case Storage::SHARED:
    case Storage::LOCAL: {
      assert(!IsHost());
      ds << d_indent << TopsDeviceMemory(ety->GetStorage())
         << " __volatile__ bool " << n.name_str;
      ety->PrintAsCArray(ds);
      ds << "; // " << STR(ety->GetStorage()) << " event\n";
      ds << d_indent << "// initialize the event\n";
      auto max_pl = GCUDeviceParallelLevel(max_parallel_level);
      auto pred = ImplicitPred(max_pl, ety->GetStorage());
      if (!pred.empty()) {
        ds << d_indent << "if (" << pred << ") {\n";
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
    } break;
    case Storage::SHARED:
    case Storage::LOCAL: {
      assert(!IsHost());
      ds << d_indent << TopsDeviceMemory(ety->GetStorage())
         << " __volatile__ bool " << n.name_str << "; // "
         << STR(ety->GetStorage()) << " event\n";
      auto max_pl = GCUDeviceParallelLevel(max_parallel_level);
      auto pred = ImplicitPred(max_pl, ety->GetStorage());
      if (!pred.empty()) {
        ds << d_indent << "if (" << pred << ") {\n";
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
    bool ref = (n.GetNote().find("ref") != std::string::npos);
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
        ds << "&" << ExprSTR(s->expr_list->ValueAt(i));
      }
      ds << "};\n";
      // make symbol a reference
      ds << d_indent << "future & " << n.GetName() << " = *" << array_sym << "["
         << ExprSTR(s->select_factor) << "];\n";
    } else if (auto sty = dyn_cast<SpannedType>(nty)) {
      auto bts = NameBaseType(sty->ElementType());
      ds << d_indent << bts << " * " << array_sym << "[] = {";
      for (size_t i = 0; i < val_count; i++) {
        if (i > 0) ds << ", ";
        ds << ExprSTR(s->expr_list->ValueAt(i));
      }
      ds << "};\n";
      // make symbol a reference
      ds << d_indent << bts << " & " << n.GetName() << " = *" << array_sym
         << "[" << ExprSTR(s->select_factor) << "];\n";
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
    ds << d_indent << ((IsMutable(*nty)) ? "" : "auto ") << n.GetName() << " = "
       << ExprSTR(n.value, false) << ";\n";
    return true;
  }

  if (isa<IntegerType>(nty)) {
    if (IsHost())
      hs << h_indent << ((IsMutable(*nty)) ? "" : "auto ") << n.GetName()
         << " = " << ExprSTR(n.value, false) << ";\n";
    else
      ds << d_indent << ((IsMutable(*nty)) ? "" : "auto ") << n.GetName()
         << " = " << ExprSTR(n.value, false) << ";\n";
    return true;
  }

  errs() << "Assignment " << STR(n) << " unprocessed, not supported "
         << PSTR(nty) << "\n";
  return false;
}

bool TopsccCodeGen::Visit(AST::ParallelBy& n) {
  TraceEachVisit(n);

  // only do the whole codegen when accessing the outer parallel-by
  if (parallel_level != 1) return true;

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
  if (offset_args.has_value())
    for (const auto& [sto, offsets] : offset_args.value())
      for (size_t idx = 0; idx < offsets.size(); ++idx)
        hs << ((i++ > 0) ? ", " : "") << "__co__" << STR(sto)
           << "_chunk_offsets[" << idx << "]";

  hs << ");\n";

  if (!n.async)
    hs << h_indent << "choreo::abend_true(topsDeviceSynchronize());\n";

  // typically, the last buffer is used as output in destination-passing-style
  // convention
  if (!HasChoreoOutput()) {
    std::string oname = "";
    ptr<Type> otype;
    ParamAttr oattr = ParamAttr::NONE;
    bool has_spanned_arg = false;
    for (const auto& item : GetChoreoFuncIns(updating_cgi)) {
      auto sname = item.name;
      if (isa<SpannedType>(item.type)) {
        oname = UnScopedName(sname);
        otype = item.type;
        oattr = item.attr;
        has_spanned_arg = true;
      }
    }

    if (has_spanned_arg && oattr != ParamAttr::GLOBAL_INPUT)
      hs << h_indent << "choreo::abend_true(topsMemcpy(" << oname << ".data(), "
         << oname + "__device" << ", " << UnScopedSizeExpr(*otype)
         << ", topsMemcpyDeviceToHost));\n";
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
    assert(ph->Category() == TypeCategory::FUTURE);
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
    return f_ca->NoTile() && t_ca->NoTile();
  };
  auto SymbolToTile = [f_ca, t_ca]() -> bool {
    return f_ca->NoTile() && t_ca->HasTile();
  };
  auto TileToSymbol = [f_ca, t_ca]() -> bool {
    return f_ca->HasTile() && t_ca->NoTile();
  };
  // Currently, not support tile to tile.

  if (t_sty->GetStorage() == Storage::GLOBAL && use_hetero_tileflow &&
      IsHostSide()) {
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
  if (t_sty->GetStorage() == Storage::GLOBAL && IsHost()) {
    if (n.async) choreo_unreachable("not support host-side async dma yet");
    std::string bts = NameBaseType(t_sty->ElementType(), false);
    auto buf_sym = t_sym + "__device";
    auto buf_sym_from = f_sym + "__device";
    if (n.operation == ".copy") {
      if (SymbolToSymbol()) {
        // direct copy
        hs << h_indent << "choreo::abend_true(topsMemcpy(" << buf_sym << ", "
           << buf_sym_from << ", " << UnScopedSizeExpr(*f_sty)
           << ", topsMemcpyDeviceToDevice));\n";
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
               // choreo DMA and tied to future but host-side data copy does not
               // really do device-level DMA, and the future is basically a
               // phantom handle do not emit any concrete code at host-side.
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
        // expr `i + 1 * (3*4)` rather than array subscript expr `i[1]`. Because
        // if memory reuse is enabled, `i` is declared as point not array!
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
            cast<SpannedType>(sym_ty)->GetShape().GetElementCountExpression();
        buf_expr += " + (" + array_idx + ")*(" + elem_count + ")";
      } else {
        for (auto expr : subscription->AllValues())
          buf_expr += "[" + ExprSTR(expr, IsHost()) + "]";
      }
    }
    return std::make_pair(buf_name, buf_expr);
  };

  // output mdspan declaration in device side, return mds name.
  auto GetMDSName = [this](const std::string& buf_name,
                           const std::string& buf_expr,
                           const ptr<SpannedType>& sty) {
    static int mds_cnt = 0;
    auto mds_name = "__mds" + std::to_string(mds_cnt++) + "_" +
                    RemoveSuffix(buf_name, ".data()");
    std::string bts{NameBaseType(sty->ElementType())};
    ds << d_indent << "tops::mdspan " << mds_name << "("
       << TopsMdsStorage(sty->GetStorage()) << ", (" << bts << "*)" << buf_expr
       << ", " << UnScopedExpr(RSTR(sty->GetShape())) << ");\n";
    return mds_name;
  };

  auto [f_buf_name, f_buf_expr] = GetBufferExpr(f_sym, f_idx, f_ty);
  auto [t_buf_name, t_buf_expr] = GetBufferExpr(t_sym, t_idx, t_ty);
  auto f_mds_name = GetMDSName(f_buf_name, f_buf_expr, f_sty);
  auto t_mds_name = GetMDSName(t_buf_name, t_buf_expr, t_sty);

  auto future_name = n.future;
  // bind the data to the future
  if (SymbolToSymbol() || TileToSymbol())
    future_name = claimFuture(t_buf_expr);
  else
    future_name = claimFuture("");

  std::string event_name;
  if (fty->IsAsync()) event_name = future_name + "__event__";

  // handles dma related to shared memory, where only single thread can operate
  bool local_in_warp = false, shared_in_block = false;
  if (!n.future.empty()) {
    shared_in_block = IsDMABlockShared(n);
    local_in_warp = IsDMAWarpLocal(n);
  }

  assert(!(shared_in_block && local_in_warp) &&
         "local and shared memory should not be used at the same time");
  if (local_in_warp)
    assert(arch.GetValue() == "gcu400" &&
           "only gcu400 need handle local synchronization");

  if (shared_in_block || local_in_warp) {
    ds << d_indent << "if (" << SingleInstancePredicate(shared_in_block)
       << ") {\n";
    IncrDeviceIndent();
  }

  if (n.operation == ".copy") {
    if (SymbolToSymbol()) {
      // no chunkat
      ds << d_indent;
      if (!event_name.empty()) ds << "tops::event " + event_name + " = ";
      ds << "tops::memcpy" << (fty->IsAsync() ? "_async" : "") << "(*"
         << future_name << ".get_ctx(), " << t_mds_name << ", " << f_mds_name
         << ");\n";

      VerboseDMA(ds, d_indent, t_sym, f_sym, "copy", "", 0,
                 ", line " + std::to_string(n.LOC().begin.line));
      // set the device future
      if (!event_name.empty())
        ds << d_indent << future_name << ".set_event(" << event_name << ");\n";
    } else if (SymbolToTile()) {
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
      // set the device future
      if (!event_name.empty()) {
        ds << d_indent << future_name << ".set_event(" << event_name << ");\n";
#if 0
          std::string bts{NameBaseType(t_sty->ElementType())};
          ds << d_indent << future_name << ".set_data(&" << f_mds_name << ".get<"
             << bts << ">(" << offset.str() << "));\n";
#endif
      }
    } else if (TileToSymbol()) {
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
      // set the device future
      if (!event_name.empty())
        ds << d_indent << future_name << ".set_event(" << event_name << ");\n";
    } else {
      choreo_unreachable("not support dual chunkat in one DMA statement");
    }
  } else if (n.operation == ".pad") {
    auto pad_config = cast<PadConfig>(n.GetConfig());
    ds << d_indent << "unsigned int __pad_low_" << f_buf_name << "[] = {"
       << DelimitedString(pad_config->pad_low) << "};\n";
    ds << d_indent << "unsigned int __pad_high_" << f_buf_name << "[] = {"
       << DelimitedString(pad_config->pad_high) << "};\n";
    ds << d_indent << "unsigned int __pad_mid_" << f_buf_name << "[] = {"
       << DelimitedString(pad_config->pad_mid) << "};\n";

    std::string pad_value_str = ExprCastSTR(
        nullptr, pad_config->value, GetBaseType(*f_sty),
        (std::holds_alternative<int>(pad_config->value) ? BaseType::S32
                                                        : BaseType::F32));

    if (SymbolToSymbol()) {
      ds << d_indent;
      if (!event_name.empty()) ds << "tops::event " + event_name + " = ";
      ds << "tops::pad" << (fty->IsAsync() ? "_async" : "") << "(*"
         << future_name << ".get_ctx(), " << t_mds_name << ", " << f_mds_name
         << ", __pad_low_" << f_buf_name << ", __pad_high_" << f_buf_name
         << ", __pad_mid_" << f_buf_name << ", " << pad_value_str << ");\n";
      // set the device future
      if (!event_name.empty())
        ds << d_indent << future_name << ".set_event(" << event_name << ");\n";
    } else if (TileToSymbol()) {
      static int s_cnt = 0;
      auto off_name = "__slice_offset" + std::to_string(s_cnt++) + "__" +
                      f_sym + "_2_" + t_sym;
      auto [offset, offcnt] = GenMdsOffset(f_ca);
      VerboseDMA(ds, d_indent, t_sym, f_sym, "slice+pad", offset, offcnt,
                 ", line " + std::to_string(n.LOC().begin.line));
      ds << d_indent << "int " << off_name << "[] = {" << offset << "};\n";
      auto slice_shape_name = "__slice_shape" + std::to_string(s_cnt++) + "__" +
                              f_sym + "_2_" + t_sym;
      std::ostringstream oss;
      f_ca->GetShape().PrintAsList(oss);
      ds << d_indent << "unsigned int " << slice_shape_name
         << "[] = " << UnScopedExpr(oss.str()) << ";\n";
      ds << d_indent;
      if (!event_name.empty()) ds << "tops::event " + event_name + " = ";
      ds << "tops::slice_pad" << (fty->IsAsync() ? "_async" : "") << "(*"
         << future_name << ".get_ctx(), " << t_mds_name << ", " << f_mds_name
         << ", " << off_name << ", " << slice_shape_name << ", __pad_low_"
         << f_buf_name << ", __pad_high_" << f_buf_name << ", __pad_mid_"
         << f_buf_name << ", " << pad_value_str << ");\n";
      // set the device future
      if (!event_name.empty())
        ds << d_indent << future_name << ".set_event(" << event_name << ");\n";
    } else {
      assert(false &&
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
    if (SymbolToSymbol()) {
      ds << d_indent;
      if (!event_name.empty()) ds << "tops::event " + event_name + " = ";
      ds << "tops::transpose" << (fty->IsAsync() ? "_async" : "") << "(*"
         << future_name << ".get_ctx(), " << t_mds_name << ", " << f_mds_name
         << ", " << layout_name << ");\n";
      // set the device future
      if (!event_name.empty())
        ds << d_indent << future_name << ".set_event(" << event_name << ");\n";
    } else if (TileToSymbol()) {
      static int s_cnt = 0;
      auto off_name = "__slice_offset" + std::to_string(s_cnt++) + "__" +
                      f_sym + "_2_" + t_sym;
      auto [offset, offcnt] = GenMdsOffset(f_ca, n.GetConfig());
      ds << d_indent << "int " << off_name << "[] = {" << offset << "};\n";
      ds << d_indent;
      if (!event_name.empty()) ds << "tops::event " + event_name + " = ";
      ds << "tops::slice_transpose" << (fty->IsAsync() ? "_async" : "") << "(*"
         << future_name << ".get_ctx(), " << t_mds_name << ", " << f_mds_name
         << ", " << off_name << ", " << layout_name << ");\n";
      // set the device future
      if (!event_name.empty())
        ds << d_indent << future_name << ".set_event(" << event_name << ");\n";
    } else if (SymbolToTile()) {
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
      // set the device future
      if (!event_name.empty())
        ds << d_indent << future_name << ".set_event(" << event_name << ");\n";
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
    assert(arch.GetValue() == "gcu400" &&
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
        os << indent << "if (!(" << ExprSTR(n.GetArguments().at(0)) << ")) {\n";
        os << indent << "  printf(\"" << n.LOC()
           << ": choreo assertion abort: " << ExprSTR(n.GetArguments().at(1))
           << "\");\n";
        os << indent << "  __co_abort__();\n";
        os << indent << "}\n";
      }
      return true;
    } else if (func_name == "print" || func_name == "println") {
      std::string print_format;
      print_format += "\"";
      std::string print_args;
      auto GenFormatAndArgsFromShape = [](const Shape& shape) {
        std::string format;
        std::ostringstream oss;
        for (int i = 0; i < (int)shape.Rank(); ++i) {
          if (i != 0) {
            format += ", ";
            oss << ", ";
          }
          format += "%lld";
          oss << "static_cast<long long>(" << shape.ValueAt(i) << ")";
        }
        std::string args = UnScopedExpr(oss.str());
        return std::make_pair(format, args);
      };
      for (const auto& arg : n.GetArguments()) {
        const auto type = NodeType(*arg);
        auto e = cast<AST::Expr>(arg);
        if (isa<StringType>(type)) {
          print_format += ExprSTR(arg);
        } else if (isa<IntegerType>(type)) {
          print_format += "%lld";
          print_args += "static_cast<long long>(" + ExprSTR(arg, false) + "), ";
        } else if (isa<BooleanType>(type) || isa<EventType>(type)) {
          if (CCtx().GetArch() == TargetArch::GCU20 ||
              CCtx().GetArch() == TargetArch::GCU21) {
            print_format += "%d";
            print_args += "(" + ExprSTR(arg, false) + " ? 1 : 0), ";
          } else {
            print_format += "%s";
            print_args +=
                "(" + ExprSTR(arg, false) + " ? \"true\" : \"false\"), ";
          }
        } else if (isa<HalfType>(type)) {
          print_format += "%f";
          print_args += ExprCastSTR(arg, std::nullopt, BaseType::F32,
                                    BaseType::F16, false) +
                        ", ";
        } else if (isa<BFP16Type>(type)) {
          print_format += "%f";
          // because always use the choreo::bf16 in choreo.h as the type
          print_args += ExprCastSTR(arg, std::nullopt, BaseType::F32,
                                    BaseType::BF16, false) +
                        ", ";
        } else if (isa<FloatType>(type)) {
          print_format += "%f";
          print_args += ExprSTR(arg, false) + ", ";
        } else if (isa<DoubleType>(type)) {
          print_format += "%f";
          print_args += ExprSTR(arg, false) + ", ";
        } else if (isa<ITupleType>(type)) {
          print_format += "{";
          auto [format, args] = GenFormatAndArgsFromShape(e->s);
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
          print_format += "%lld";
          print_args += "static_cast<long long>(" + ExprSTR(arg, false) + "), ";
          choreo_unreachable("should not have bit?");
        } else if (isa<BoundedITupleType>(type)) {
          print_format += "{";
          for (int i = 0; i < (int)e->s.Rank(); ++i) {
            if (i != 0) print_format += ", ";
            print_format += "%lld";
          }
          print_format += "}";
          std::string args_str = ExprSTR(arg, false);
          for (const auto& arg_str : SplitStringByDelimiter(args_str, ", "))
            print_args += "static_cast<long long>(" + arg_str + "), ";
        } else if (isa<AddrType>(type)) {
          print_format += "%p";
          print_args += "static_cast<void*>(" + ExprSTR(arg, IsHost()) + "), ";
        } else
          choreo_unreachable("unsupported type for print: " +
                             AST::TYPE_STR(*arg) + "\n\targ: " + ExprSTR(arg));
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

bool TopsccCodeGen::Visit(AST::ParamList& n) {
  int index = 0;
  for (auto param : n.values)
    updating_cgi->AddSymbolDetail(fname, {InScopeName(param->sym->name),
                                          param->GetType(), false, index++,
                                          param->GetAttr()});
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
      auto iv_bty = cast<BoundedType>(iv_ty);
      if (IsHost()) {
        hs << h_indent << "for (" << ssm.DeviceName(iv_name) << " = "
           << (rng->lbound ? ("(" + ExprSTR(rng->lbound, false) + ")") : "0")
           << "; " << ssm.DeviceName(iv_name) << " < "
           << UnScopedExpr(STR(iv_bty->GetUpperBound()))
           << (rng->ubound ? (" + " + ExprSTR(rng->ubound, false)) : "")
           << "; ++" << ssm.DeviceName(iv_name) << ") {\n";
        IncrHostIndent();
      } else {
        ds << d_indent << "for (" << ssm.DeviceName(iv_name) << " = "
           << (rng->lbound ? ("(" + ExprSTR(rng->lbound, false) + ")") : "0")
           << "; " << ssm.DeviceName(iv_name) << " < "
           << UnScopedExpr(STR(iv_bty->GetUpperBound()))
           << (rng->ubound ? (" + " + ExprSTR(rng->ubound, false)) : "")
           << "; ++" << ssm.DeviceName(iv_name) << ") {\n";
        IncrDeviceIndent();
      }
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

  if (IsHost()) {
    hs << h_indent << "// if-else: " << n.LOC() << "\n";
    if (auto c = dyn_cast<AST::Call>(n.pred))
      hs << h_indent << "if (" << CallSTR(*c) << ") {\n";
    else
      hs << h_indent << "if (" << ExprSTR(n.pred, true) << ") {\n";
    IncrHostIndent();
  } else {
    ds << d_indent << "// if-else: " << n.LOC() << "\n";
    if (auto c = dyn_cast<AST::Call>(n.pred))
      ds << d_indent << "if (" << CallSTR(*c) << ") {\n";
    else
      ds << d_indent << "if (" << ExprSTR(n.pred, false) << ") {\n";
    IncrDeviceIndent();
  }
  emit_call = true;
  return true;
}

bool TopsccCodeGen::Visit(AST::WhileBlock& n) {
  TraceEachVisit(n);

  if (IsHost()) {
    hs << h_indent << "// while: " << n.LOC() << "\n";
    hs << h_indent << "while (" << ExprSTR(n.pred, true) << ") {\n";
    IncrHostIndent();
  } else {
    ds << d_indent << "// while: " << n.LOC() << "\n";
    ds << d_indent << "while (" << ExprSTR(n.pred, false) << ") {\n";
    IncrDeviceIndent();
  }
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
    if (IsHost())
      hs << n.GetCode();
    else
      ds << n.GetCode();
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
    oss << ((host_pindex == 0) ? "" : ", ") << HostTypeStringify(*item.type)
        << " " << item.host_name;
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
  std::map<ValueExpr, std::vector<Entry>> ve_entries_map;

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
        } else if (auto vale = VIStr(vi)) {
          ve_entries_map[*vale].push_back(
              {host_pindex + 1, dim_count, elem_name});
        }
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
  else if (isa<IntegerType>(&ty))
    return "int";
  else if (isa<BooleanType>(&ty))
    return "bool";
  else if (isa<Half8Type>(&ty))
    return "choreo::half8";
  else if (isa<HalfType>(&ty))
    return "choreo::half";
  else if (isa<BFP16Type>(&ty))
    return "choreo::bfp16";
  else if (isa<FloatType>(&ty))
    return "float";
  else if (isa<DoubleType>(&ty))
    return "double";
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
    choreo_unreachable("unsupported host function type.");
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
  if (arch.GetValue() == "gcu400") {
    auto& lconfig = cgi->GetFunctionLaunches(fname)[parallel_idx];
    oss << "__thread_dims__(" << lconfig.warp_dim_x << ", "
        << lconfig.warp_dim_y << ", " << lconfig.warp_dim_z << ")\n";
  }

  oss << "__global__ void " << device_fn << "(";

  size_t index = 0;
  for (auto& item : GetDeviceFuncIns(updating_cgi)) {
    if (!PrefixedWith(scoped_symtab.ScopeName(), GetScope(item.name))) continue;
    oss << ((index++ > 0) ? ", " : "");
    oss << DeviceParamTypeStringify(*item.type) << " ";
    oss << (item.need_iv_prefix ? "__iv_" : "") << UnScopedName(item.name);
  }

  for (auto item : symbolic_dimensions) {
    oss << ((index++ > 0) ? ", unsigned " : "unsigned ");
    oss << UnScopedName(item.first);
  }

  const auto& offset_args =
      FCtx(fname).GetMemReuseOffsetArgs(SSTab().ScopeName());
  if (offset_args.has_value())
    for (const auto& [_, offsets] : offset_args.value())
      for (size_t idx = 0; idx < offsets.size(); ++idx)
        oss << ((index++ > 0) ? ", " : "") << "unsigned long "
            << RegexReplaceAll(offsets[idx], "::", "_");

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
  bool use_sim = arch.GetValue() == "gcu400";

  // JIT: detect the environment
  if (use_sim)
    os << "gcu_arch=gcu400\n";
  else if (((CCtx().GetOutputKind() == OutputKind::TargetModule) ||
            (CCtx().GetOutputKind() == OutputKind::TargetExecutable)) &&
           !arch.GetValue().empty()) {
    // enforce the arch type
    os << "gcu_arch=" << arch.GetValue() << "\n";
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
  // always enclose
  os << " ${EXTRA_TARGET_CFLAGS}\"";
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
const std::string TopsccCodeGen::ValueSTR(const ValueItem& vi) const {
  if (!IsValidValueItem(vi)) choreo_unreachable("invalid value item.");
  if (auto iv = VIInt(vi))
    return PSTR(vi);
  else if (auto bv = VIBool(vi))
    return PSTR(vi);
  else if (auto sv = VIStr(vi)) {
    auto name = PSTR(vi);
    if (within_map.count(name)) {
      if (IsHost())
        return ssm.HostName(name);
      else
        return ssm.DeviceName(name);
    } else
      return UnScopedExpr(PSTR(vi));
  } else if (auto bo = VIBop(vi))
    return "(" + ValueSTR(bo->GetLeft()) + " " + STR(bo->GetOpCode()) + " " +
           ValueSTR(bo->GetRight()) + ")";
  else if (auto to = VITop(vi))
    return "(" + ValueSTR(to->GetPred()) + " ? " + ValueSTR(to->GetLeft()) +
           " : " + ValueSTR(to->GetRight()) + ")";
  else
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

// TODO: maybe generate a warning?
// narrowing conversions may cause data truncation
// but the value is given by user, so we can trust it?

// if from and to are not the same, only support from of `S32, FP32, BFP16`
// and to: all integer type, all floating-point type
const std::string
TopsccCodeGen::ExprCastSTR(AST::ptr<AST::Node> n,
                           std::optional<std::variant<int, float>> val,
                           BaseType t, BaseType f, bool is_host) const {

  std::ostringstream res;
  std::string value;

  if (n != nullptr) {
#ifndef USING_OP_INFO
    value = ExprSTR(n, is_host);
#else
    value = OpExprSTR(n, is_host);
#endif
  } else {
    assert(val.has_value());
    auto v = val.value();
    if (std::holds_alternative<int>(v))
      value = std::to_string(std::get<int>(v));
    else if (std::holds_alternative<float>(v))
      value = std::to_string(std::get<float>(v)) + "f";
    else
      choreo_unreachable("unexpect type of v");
  }

  using BT = BaseType;

  if (f == t) return value;
  // TODO: actually, s32 is equal to int

  if (t == BT::F8 || t == BT::HALF8)
    choreo_unreachable("unsupport cast: '" + STR(f) + "' to " + STR(t));

  // need to do casting or converting.
  /*
  suppose
  f8: E4M3: <= 240; E5M2: <= 57344
  f16: E5M10: <= 65504
  bf16: E8M7: <= 3.38×10^38
  f32: <= 3.4×10^38

  U8: [0, 255]
  S8: [-128, 127]
  U16: [0, 65535]
  S16: [-32768, 32767]
  U32: [0, 2^32-1]
  S32: [-2^16, 2^16-1]
  U64: [0, 2^64-1]
  S64: [-2^32, 2^32-1]
  */
  // here, safe means no
  auto IsSafeCast = [f, t]() -> bool {
    static const std::unordered_map<BT, std::unordered_set<BT>> table = {
        {BT::BOOL,
         {BT::BOOL, BT::U8, BT::S8, BT::U16, BT::S16, BT::U32, BT::S32, BT::INT,
          BT::U64, BT::S64, BT::F8, BT::HALF8, BT::BF16, BT::BFP16, BT::F16,
          BT::HALF, BT::F32, BT::FLOAT, BT::DOUBLE}},
        {BT::U8,
         {BT::U8, BT::U16, BT::S16, BT::U32, BT::S32, BT::INT, BT::U64, BT::S64,
          BT::F8, BT::HALF8, BT::BF16, BT::F16, BT::BFP16, BT::HALF, BT::F32,
          BT::FLOAT, BT::DOUBLE}},
        {BT::S8,
         {BT::S8, BT::S16, BT::S32, BT::INT, BT::U64, BT::S64, BT::F8,
          BT::HALF8, BT::BF16, BT::BFP16, BT::F16, BT::HALF, BT::F32, BT::FLOAT,
          BT::DOUBLE}},
        {BT::U16,
         {BT::U16, BT::U32, BT::S32, BT::INT, BT::U64, BT::S64, BT::BF16,
          BT::BFP16, BT::F32, BT::FLOAT, BT::DOUBLE}},
        {BT::S16,
         {BT::S16, BT::S32, BT::INT, BT::U64, BT::S64, BT::BF16, BT::BFP16,
          BT::F32, BT::FLOAT, BT::DOUBLE}},
        {BT::U32,
         {BT::U32, BT::U64, BT::S64, BT::BF16, BT::BFP16, BT::F32, BT::FLOAT,
          BT::DOUBLE}},
        {BT::S32,
         {BT::S32, BT::S64, BT::BF16, BT::BFP16, BT::F32, BT::FLOAT,
          BT::DOUBLE}},
        {BT::U64,
         {BT::U64, BT::BF16, BT::BFP16, BT::F32, BT::FLOAT, BT::DOUBLE}},
        {BT::S64,
         {BT::S64, BT::BF16, BT::BFP16, BT::F32, BT::FLOAT, BT::DOUBLE}},
        {BT::F8,
         {BT::F8, BT::HALF8, BT::BF16, BT::F16, BT::BFP16, BT::HALF, BT::F32,
          BT::FLOAT, BT::DOUBLE}},
        {BT::HALF8,
         {BT::F8, BT::HALF8, BT::BF16, BT::F16, BT::BFP16, BT::HALF, BT::F32,
          BT::FLOAT, BT::DOUBLE}},
        {BT::BF16, {BT::BF16, BT::BFP16, BT::F32, BT::FLOAT, BT::DOUBLE}},
        {BT::F16,
         {BT::BF16, BT::F16, BT::BFP16, BT::HALF, BT::F32, BT::FLOAT,
          BT::DOUBLE}},
        {BT::HALF,
         {BT::BF16, BT::F16, BT::BFP16, BT::HALF, BT::F32, BT::FLOAT,
          BT::DOUBLE}},
        {BT::F32, {BT::F32, BT::FLOAT, BT::DOUBLE}},
        {BT::FLOAT, {BT::F32, BT::FLOAT, BT::DOUBLE}},
        {BT::DOUBLE, {BT::DOUBLE}},
    };
    auto it = table.find(f);
    if (it != table.end() && it->second.count(t)) return true;
    return false;
  };

  // TODO: n maybe nullptr
  if (!IsSafeCast() && n)
    Warning(n->LOC(),
            "type cast is unsafe: '" + STR(f) + "' to '" + STR(t) + "'");

  switch (t) {
  case BT::S64: [[fallthrough]];
  case BT::U64: [[fallthrough]];
  case BT::S32: [[fallthrough]];
  case BT::INT: [[fallthrough]];
  case BT::U32: [[fallthrough]];
  case BT::S16: [[fallthrough]];
  case BT::U16: [[fallthrough]];
  case BT::S8: [[fallthrough]];
  case BT::U8: {
    // TODO: workaround
    if ((f == BT::INT && t == BT::S32) || (f == BT::S32 && t == BT::INT)) {
      res << value;
      break;
    }
    auto nbt = NameBaseType(t, is_host);
    if (IntegerBaseType(f))
      res << "static_cast<" << nbt << ">(" << value << ")";
    else if (FloatPointBaseType(f)) {
      if (f != BT::FLOAT && f != BT::DOUBLE)
        res << "static_cast<" << nbt << ">("
            << ExprCastSTR(n, val, BT::FLOAT, f) << ")";
      else
        res << "static_cast<" << nbt << ">(" << value << ")";
    }
    break;
  }
  case BT::BOOL: {
    if (IntegerBaseType(f))
      res << "static_cast<bool>(" << ExprCastSTR(n, val, BT::S64, f) << ")";
    else if (FloatPointBaseType(f))
      res << "static_cast<bool>(" << ExprCastSTR(n, val, BT::DOUBLE, f) << ")";
    break;
  }
  case BT::DOUBLE: {
    if (IntegerBaseType(f))
      res << "static_cast<double>(" << value << ")";
    else
      res << "static_cast<double>(" << ExprCastSTR(n, val, BT::FLOAT, f) << ")";
    break;
  }
  case BT::FLOAT:
  case BT::F32: {
    if (IntegerBaseType(f))
      res << "static_cast<float>(" << value << ")";
    else {
      if (f == BT::F16 || f == BT::HALF)
        res << "f16_to_f32(" << value << ")";
      else if (f == BaseType::BF16 || f == BaseType::BFP16)
        res << "(float)(" << value << ")";
      else if (f == BT::DOUBLE)
        res << "static_cast<float>(" << value << ")";
      else if (f == BT::F32 || f == BT::FLOAT)
        res << value; // TODO: workaround
      else
        choreo_unreachable("unsupport cast: '" + STR(f) + "' to '" + STR(t) +
                           "'");
    }
    break;
  }
  case BT::HALF:
  case BT::F16:
    res << "f32_to_f16(" << ExprCastSTR(n, val, BT::FLOAT, f) << ")";
    break;
  case BT::BFP16:
  case BT::BF16: {
    if (f == BT::BFP16 || f == BT::BF16)
      res << value; // TODO: workaround
    else
      res << "choreo::bf16(" << ExprCastSTR(n, val, BT::FLOAT, f) << ")";
    break;
  }
  default:
    choreo_unreachable("unsupport cast: '" + STR(f) + "' to '" + STR(t) + "'");
  }

  return res.str();
}

#ifndef USING_OP_INFO
const std::string TopsccCodeGen::ExprSTR(AST::ptr<AST::Node> e,
                                         bool is_host) const {
  std::ostringstream oss;

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
    } else {
      oss << UnScopedName(((is_host) ? ssm.HostName(InScopeName(id->name))
                                     : ssm.DeviceName(InScopeName(id->name))));
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
    return ExprSTR(ii->value, is_host);
  } else if (auto da = dyn_cast<AST::DataAccess>(e)) {
    if (auto sty = GetSpannedType(GetSymbolType(da->data->name))) {
      oss << "*((" << NameBaseType(sty->ElementType()) << "*)"
          << ExprSTR(da->data, is_host);
      size_t idx = 0;
      auto shape = sty->GetShape();
      auto AppendOffset = [this, &oss, &shape, &idx](const ValueItem& op) {
        auto offset = op;
        assert(shape.Rank() >= idx + 1);
        if (shape.Rank() > idx + 1)
          offset = offset * shape.TrimDims(idx + 1).ElementCountValue();
        offset->Normalize();
        if (!sbe::ceq(offset, sbe::nu(0))) oss << " + " << ValueSTR(offset);
        ++idx;
      };
      for (auto item : da->GetIndices()) {
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
        } else if (auto il = AST::GetIntLiteral(*item))
          AppendOffset(sbe::nu(il->Val()));
        else {
          oss << " + (" << ExprSTR(item, is_host);
          assert(shape.Rank() >= idx + 1);
          if (shape.Rank() > idx + 1)
            oss << " * "
                << ValueSTR(shape.TrimDims(idx + 1).ElementCountValue());
          oss << ")";
          ++idx;
        }
      }
      oss << ")";
    } else {
      assert(!da->AccessElement());
      assert(!within_map.count(InScopeName(da->data->name)));
      oss << UnScopedName(((is_host)
                               ? ssm.HostName(InScopeName(da->data->name))
                               : ssm.DeviceName(InScopeName(da->data->name))));
    }
  } else if (auto expr = dyn_cast<AST::Expr>(e)) {
    // utilize the optimize value whenever possible
    if (auto sym = expr->GetSymbol()) {
      auto sname = InScopeName(sym->name);
      if (FCtx(fname).HasSymbolValues(sname)) {
        auto svs = FCtx(fname).GetSymbolValues(sname);
        if (svs.HasVal()) return "(" + UnScopedExpr(STR(svs.GetVal())) + ")";
      }
    }

    if (ConvertibleToInt(NodeType(*e))) {
      if (expr->Opts().HasVal()) return ValueSTR(expr->Opts().GetVal());
    }

    if (expr->IsReference()) {
      if (PSTR(expr) == "_") return "(0)";
      if (auto ca = dyn_cast<AST::ChunkAt>(expr->GetR())) {
        auto caty = cast<SpannedType>(ca->GetType());
        if (isa<FutureType>(NodeType(*ca->data)))
          return ExprSTR(ca->data, is_host) + ".data() + " + GenOffset(ca);
        else
          return ExprSTR(ca->data, is_host) + " + " + GenOffset(ca);
      } else
        return ExprSTR(expr->GetReference(), is_host);
    } else if (expr->IsUnary()) {
      if (expr->GetOp() == "!") {
        oss << "!(" << ExprSTR(expr->GetR(), is_host) << ")";
      } else if (expr->GetOp() == "ubound") {
        auto rty = cast<BoundedType>(NodeType(*expr->GetR()));
        if (rty->Dims() == 1) { oss << ValueSTR(rty->GetUpperBound()); }
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
          oss << shape.GetElementCountExpression();
        }
      } else if (expr->GetOp() == "++") {
        oss << "++" << ExprSTR(expr->GetR(), is_host);
      } else if (expr->GetOp() == "--") {
        oss << "--" << ExprSTR(expr->GetR(), is_host);
      } else if (expr->GetOp() == "addrof") {
        if (auto id = AST::GetIdentifier(expr->GetR())) {
          oss << ExprSTR(id, is_host);
        } else if (isa<AST::DataAccess>(expr->GetR())) {
          oss << "&" << ExprSTR(expr->GetR(), is_host);
        } else
          choreo_unreachable("Can not retrieve name of the spanned data.");
      } else if (expr->GetOp() == "~") {
        oss << "~" << ExprSTR(expr->GetR(), is_host);
      } else
        choreo_unreachable("Unsupported choreo expression.");
    } else if (expr->IsBinary()) {
      if (expr->GetOp() == "cdiv") {
        std::string one = "1";
        oss << "((" << ExprSTR(expr->GetL(), is_host) << ")+("
            << ExprSTR(expr->GetR(), is_host) << ")-(" << one << "))/("
            << ExprSTR(expr->GetR(), is_host) << ")";
      } else if (expr->GetOp() == "getith") {
        auto lty = cast<BoundedType>(NodeType(*expr->GetL()));
        if (cast<AST::IntIndex>(expr->GetR())->IsNegative()) {
          oss << "(";
          oss << ValueSTR(lty->GetUpperBound());
          oss << "+(" << ExprSTR(expr->GetR(), is_host) << "))";
        } else
          oss << "(" << ExprSTR(expr->GetR(), is_host) << ")";
      } else if (expr->GetOp() == "elemof") {
        oss << ExprSTR(expr->GetL(), is_host) << "["
            << ExprSTR(expr->GetR(), is_host) << "]";
      } else if (expr->IsArith() || expr->IsLogical() || expr->IsCompare() ||
                 expr->isBitwise()) {
        auto& l = expr->GetL();
        auto& r = expr->GetR();
        auto& op = expr->GetOp();
        if (op == "#" && IsActualBoundedIntegerType(l->GetType()) &&
            IsActualBoundedIntegerType(r->GetType())) {
          auto rty = cast<BoundedType>(NodeType(*r));
          assert(rty->Dims() == 1);
          if (PSTR(r) == "_")
            oss << "((" << ExprSTR(l, is_host) << ")*(1)+("
                << ExprSTR(r, is_host) << "))";
          else
            oss << "((" << ExprSTR(l, is_host) << ")*("
                << ValueSTR(rty->GetUpperBound()) << ")+("
                << ExprSTR(r, is_host) << "))";
        } else if ((op == "#+" || op == "#-") &&
                   IsActualBoundedIntegerType(l->GetType()) &&
                   isa<IntegerType>(r->GetType()))
          oss << "(" << ExprSTR(l, is_host) << ")";
        else if (op == "#/" || op == "#*" || op == "#%")
          choreo_unreachable("unsupported expression '" + expr->GetOp() +
                             "': " + PSTR(expr) + ".");
        else
          oss << "(" << ExprSTR(l, is_host) << " " << op << " "
              << ExprSTR(r, is_host) << ")";
      }
    } else if (expr->IsTernary()) {
      oss << "((" << ExprSTR(expr->GetC(), is_host) << ") ? "
          << ExprSTR(expr->GetL(), is_host) << " : "
          << ExprSTR(expr->GetR(), is_host) << ")";
    } else
      choreo_unreachable("unsupported expression '" + expr->GetOp() +
                         "': " + PSTR(expr) + ".");
  } else if (auto c = dyn_cast<AST::Call>(e)) {
    assert(!is_host);
    return CallSTR(*c);
  } else
    choreo_unreachable("unsupported expression '" + expr->GetOp() + "'.");

  return oss.str();
}

#else
const std::string TopsccCodeGen::ExprSTR(AST::ptr<AST::Node> e,
                                         bool is_host) const {
  return OpExprSTR(e, is_host);
}

const std::string TopsccCodeGen::OpExprSTR(AST::ptr<AST::Node> e, bool is_host,
                                           const std::string& parent_op,
                                           bool is_left) const {
  std::ostringstream oss;

  auto WrapWithParen = [&](const std::string& s, const std::string& cur_op) {
    if (Operator::NeedParen(cur_op, parent_op, is_left)) return "(" + s + ")";
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
    } else {
      oss << UnScopedName(((is_host) ? ssm.HostName(InScopeName(id->name))
                                     : ssm.DeviceName(InScopeName(id->name))));
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
    return OpExprSTR(ii->value, is_host, parent_op);
  } else if (auto da = dyn_cast<AST::DataAccess>(e)) {
    if (auto sty = GetSpannedType(GetSymbolType(da->data->name))) {
      oss << "*((" << NameBaseType(sty->ElementType()) << "*)"
          << OpExprSTR(da->data, is_host);
      size_t idx = 0;
      auto shape = sty->GetShape();
      for (auto item : da->GetIndices()) {
        if (auto id = AST::GetIdentifier(item)) {
          if (auto ids = ThreadIdString(id))
            oss << " + " << ids.value();
          else if (auto sids = SubThreadIdString(id))
            oss << " + " << sids.value();
          else if (within_map.count(InScopeName(id->name))) {
            auto ivs = within_map.at(InScopeName(id->name));
            for (auto iv_itr = ivs.begin(); iv_itr != ivs.end(); ++iv_itr) {
              auto shape = sty->GetShape();
              auto name =
                  (is_host ? ssm.HostName(*iv_itr) : ssm.DeviceName(*iv_itr));
              assert(shape.Rank() >= idx + 1);
              oss << " + ";
              if (shape.Rank() > idx + 1)
                oss << "(" << name << " * "
                    << shape.TrimDims(idx + 1).GetElementCountExpression()
                    << ")";
              else
                oss << name;
              ++idx;
            }
          } else {
            auto name =
                UnScopedName(is_host ? ssm.HostName(InScopeName(id->name))
                                     : ssm.DeviceName(InScopeName(id->name)));
            assert(shape.Rank() >= idx + 1);
            oss << " + ";
            if (shape.Rank() > idx + 1)
              oss << "(" << name << " * "
                  << shape.TrimDims(idx + 1).GetElementCountExpression() << ")";
            else
              oss << name;
            ++idx;
          }
        } else
          choreo_unreachable("unsupported data access.");
      }
      oss << ")";
    } else {
      assert(!da->AccessElement());
      assert(!within_map.count(InScopeName(da->data->name)));
      oss << UnScopedName(((is_host)
                               ? ssm.HostName(InScopeName(da->data->name))
                               : ssm.DeviceName(InScopeName(da->data->name))));
    }
  } else if (auto expr = dyn_cast<AST::Expr>(e)) {
    // codegen for scalar type promotion
    if (auto pe = dyn_cast<AST::PromoteExpr>(expr)) {
      assert(expr->GetOp() == "promote");
      return ExprCastSTR(pe->GetR(), std::nullopt, pe->ToType(),
                         pe->FromType());
    }

    // utilize the optimize value whenever possible
    if (auto sym = expr->GetSymbol()) {
      auto sname = InScopeName(sym->name);
      if (FCtx(fname).HasSymbolValues(sname)) {
        auto svs = FCtx(fname).GetSymbolValues(sname);
        if (svs.HasVal()) return ValueSTR(svs.GetVal());
      }
    }

    if (ConvertibleToInt(NodeType(*e)))
      if (expr->Opts().HasVal())
        return UnScopedExpr(ValueSTR(expr->Opts().GetVal()));

    if (expr->IsReference()) {
      if (PSTR(expr) == "_") return "0";
      if (auto ca = dyn_cast<AST::ChunkAt>(expr->GetR())) {
        auto caty = cast<SpannedType>(ca->GetType());
        if (isa<FutureType>(NodeType(*ca->data)))
          return OpExprSTR(ca->data, is_host) + ".data() + " + GenOffset(ca);
        else
          return OpExprSTR(ca->data, is_host) + " + " + GenOffset(ca);
      } else {
        // TODO: ref is always the last order, so no need to add paren?
        return OpExprSTR(expr->GetReference(), is_host, "ref");
      }
    } else if (expr->IsUnary()) {
      if (expr->GetOp() == "!") {
        oss << "!" << WrapWithParen(OpExprSTR(expr->GetR(), is_host, "!"), "!");
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
          oss << shape.GetElementCountExpression();
        }
      } else if (expr->GetOp() == "++") {
        oss << "++"
            << WrapWithParen(OpExprSTR(expr->GetR(), is_host, "++"), "++");
      } else if (expr->GetOp() == "--") {
        oss << "--"
            << WrapWithParen(OpExprSTR(expr->GetR(), is_host, "--"), "--");
      } else if (expr->GetOp() == "addrof") {
        if (auto id = AST::GetIdentifier(expr->GetR())) {
          oss << OpExprSTR(id, is_host);
        } else if (isa<AST::DataAccess>(expr->GetR())) {
          oss << "&"
              << WrapWithParen(OpExprSTR(expr->GetR(), is_host, "&"), "&");
        } else
          choreo_unreachable("Can not retrieve name of the spanned data.");
      } else
        choreo_unreachable("unsupported expression op: '" + expr->GetOp() +
                           "', expr: " + PSTR(expr) + ".");
    } else if (expr->IsBinary()) {
      if (expr->GetOp() == "cdiv") {
        std::string one = "1";
        auto L = OpExprSTR(expr->GetL(), is_host, "+");
        auto R0 = OpExprSTR(expr->GetR(), is_host, "+", false);
        auto R1 = OpExprSTR(expr->GetR(), is_host, "/", false);
        // (L + R0 - 1) / R1
        std::ostringstream res;
        res << "(" << L << " + " << R0 << " - " << one << ") / " << R1;
        oss << WrapWithParen(res.str(), "/");
      } else if (expr->GetOp() == "getith") {
        auto lty = cast<BoundedType>(NodeType(*expr->GetL()));
        if (cast<AST::IntIndex>(expr->GetR())->IsNegative()) {
          std::ostringstream res;
          // special case: str of r is always a negative integer.
          res << ValueSTR(lty->GetUpperBound()) << " + "
              << "(" << OpExprSTR(expr->GetR(), is_host, "+", false) << ")";
          oss << WrapWithParen(res.str(), "+");
        } else
          oss << OpExprSTR(expr->GetR(), is_host, parent_op);
      } else if (expr->GetOp() == "elemof") {
        oss << OpExprSTR(expr->GetL(), is_host) << "["
            << OpExprSTR(expr->GetR(), is_host) << "]";
      } else if (expr->IsArith() || expr->IsLogical()) {
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
          auto L = OpExprSTR(l, is_host, "*");
          auto R = OpExprSTR(r, is_host, "+", false);
          std::ostringstream res;
          res << L << " * " << r_upper_bound << " + " << R;
          oss << WrapWithParen(res.str(), "+");
        } else if ((op == "#+" || op == "#-") &&
                   IsActualBoundedIntegerType(l->GetType()) &&
                   isa<IntegerType>(r->GetType()))
          oss << OpExprSTR(l, is_host, parent_op);
        else if (op == "#/" || op == "#*" || op == "#%")
          choreo_unreachable("unsupported expression op: '" + expr->GetOp() +
                             "', expr: " + PSTR(expr) + ".");
        else {
          std::ostringstream res;
          res << OpExprSTR(l, is_host, op) << " " << op << " "
              << OpExprSTR(r, is_host, op, false);
          oss << WrapWithParen(res.str(), op);
        }
      }
    } else if (expr->IsTernary()) {
      std::ostringstream res;
      res << OpExprSTR(expr->GetC(), is_host, "?") << " ? "
          << OpExprSTR(expr->GetL(), is_host, "?") << " : "
          << OpExprSTR(expr->GetR(), is_host, "?", false);
      oss << WrapWithParen(res.str(), "?");
    } else
      choreo_unreachable("unsupported expression op: '" + expr->GetOp() +
                         "', expr: " + PSTR(expr) + ".");
  } else if (auto c = dyn_cast<AST::Call>(e)) {
    assert(!is_host);
    return CallSTR(*c);
  } else
    choreo_unreachable("unsupported expression op: '" + expr->GetOp() + "'.");

  return oss.str();
}
#endif

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

  oss << func_name(n.function->name);

  // emit template arguments
  if (n.template_args) {
    oss << "<";
    size_t i = 0;
    for (auto& ta : n.template_args->AllValues())
      oss << ((i++ == 0) ? "" : ", ") << ExprSTR(ta, IsHost());
    oss << ">";
  }

  oss << "(";
  size_t i = 0;
  for (auto& a : n.GetArguments()) {
    oss << ((i++ == 0) ? "" : ", ");
    if (auto sty = GetSpannedType(NodeType(*a))) {
      std::string bts{NameBaseType(sty->ElementType(), IsHost())};
      if (!no_decay_spanview || IsHost())
        oss << "(" << bts << "*)" << ExprSTR(a, IsHost());
      else
        oss << "choreo::make_spanview<" << sty->Dims() << ">((" << bts << "*)"
            << ExprSTR(a, IsHost()) << ", " << LSTR(sty->GetShape()) << ")";
    } else if (n.IsArith())
      oss << ExprSTR(a, IsHost());
    else
      oss << UnScopedName(ExprSTR(a, IsHost()));
  }
  oss << ")";

  return oss.str();
}
