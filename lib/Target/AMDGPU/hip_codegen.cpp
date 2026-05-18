#include "hip_codegen.hpp"
#include "hip_dma_plan.hpp"
#include "ast.hpp"
#include "codegen.hpp"
#include "context.hpp"
#include "options.hpp"
#include "types.hpp"

#include "choreo_header.inc"
#include "choreo_types_header.inc"

#include <cassert>
#include <filesystem>
#include <fstream>

using namespace Choreo;
using namespace Choreo::HIP;

// ============================================================================
// BeforeVisitImpl / InMidVisitImpl / AfterVisitImpl
// ============================================================================

bool HIPCodeGen::BeforeVisitImpl(AST::Node& n) {
  if (isa<AST::Program>(&n)) {
    EmitFixedHostHead();
    EmitFixedDeviceHead();
    ssm.EnterScope();
    levels.push(ParallelLevel::NONE);
  } else if (isa<AST::ChoreoFunction>(&n)) {
    ResetChoreoFunctionStates();
    device_fn = "__choreo_device_" + fname;
    fty = cast<FunctionType>(GetSymbolType(fname));
    ssm.EnterScope();
    levels.push(ParallelLevel::SEQ);
  } else if (auto pb = dyn_cast<AST::ParallelBy>(&n)) {
    levels.push(pb->GetLevel());
    bool is_outer = pb->IsOuter();
    if (is_outer) {
      parallel_idx += 1;
      cur_pb = pb;
      if (cgi.GetFunctionTrait(fname).multiple_parallelby)
        device_fn = "__choreo_device_" + fname + std::to_string(parallel_idx);
    }
  } else if (isa<AST::WithBlock>(&n)) {
    IndStream() << "{\n";
    IncrIndent();
  } else if (isa<AST::ForeachBlock>(&n)) {
    // handled in Visit
  }
  if (isa<AST::IfElseBlock>(&n) || isa<AST::NamedVariableDecl>(&n)) {
    emit_call = false;
  }
  return true;
}

bool HIPCodeGen::InMidVisitImpl(AST::Node& n) {
  if (auto ie = dyn_cast<AST::IfElseBlock>(&n)) {
    if (!ie->HasElse()) return true;
    DecrIndent();
    IndStream() << "} else {\n";
    IncrIndent();
  }
  return true;
}

bool HIPCodeGen::AfterVisitImpl(AST::Node& n) {
  if (isa<AST::Program>(&n)) {
    ssm.LeaveScope();
    switch (CCtx().GetOutputKind()) {
    case OutputKind::TargetSourceCode: EmitSource(); break;
    case OutputKind::TargetModule:
    case OutputKind::TargetExecutable:
      if (!CompileWithScript(
              CCtx().GetOutputKind() == OutputKind::TargetModule
                  ? "--compile-module"
                  : "--compile-link")) {
        error_count++;
        return false;
      }
      break;
    case OutputKind::ShellScript: EmitScript(outs()); break;
    default:
      choreo_unreachable("outputkind: " + STR(CCtx().GetOutputKind()) +
                         " is not supported by amdgpu target.");
    }
  } else if (isa<AST::ChoreoFunction>(&n)) {
    ssm.LeaveScope();
    levels.pop();
    code_segments.back() += ds.str() + hs.str();
    ds.str("");
    hs.str("");
    return_stream.str("");
  } else if (auto pb = dyn_cast<AST::ParallelBy>(&n)) {
    levels.pop();
    if (pb->IsOuter()) {
      cur_pb = nullptr;
      DecrDeviceIndent();
      ds << d_indent << "}\n";

      hs << h_indent << "hipDeviceSynchronize();\n";
      hs << h_indent << "choreo::verify_device_status();\n";
    }
  } else if (isa<AST::WithBlock>(&n)) {
    DecrIndent();
    IndStream() << "}\n";
  } else if (isa<AST::IfElseBlock>(&n)) {
    DecrIndent();
    IndStream() << "}\n";
    emit_call = true;
  } else if (isa<AST::NamedVariableDecl>(&n)) {
    emit_call = true;
  } else if (isa<AST::ForeachBlock>(&n)) {
    DecrIndent();
    IndStream() << "}\n";
  } else if (isa<AST::WhileBlock>(&n)) {
    DecrIndent();
    IndStream() << "}\n";
  } else if (isa<AST::InThreadsBlock>(&n)) {
    DecrDeviceIndent();
    ds << d_indent << "}\n";
  }
  return true;
}

// ============================================================================
// Visit methods
// ============================================================================

bool HIPCodeGen::Visit(AST::ParamList&) { return true; }
bool HIPCodeGen::Visit(AST::WithIn&) { return true; }
bool HIPCodeGen::Visit(AST::WhereBind&) { return true; }
bool HIPCodeGen::Visit(AST::WithBlock&) { return true; }

bool HIPCodeGen::Visit(AST::MMA&) {
  choreo_unreachable("MMA is not supported by AMDGPU target.");
  return false;
}

bool HIPCodeGen::Visit(AST::Trigger&) {
  choreo_unreachable("Trigger is not supported by AMDGPU target.");
  return false;
}

bool HIPCodeGen::Visit(AST::Rotate&) {
  choreo_unreachable("Rotate is not supported by AMDGPU target.");
  return false;
}

bool HIPCodeGen::Visit(AST::Yield&) {
  choreo_unreachable("Yield is not supported by AMDGPU target.");
  return false;
}

bool HIPCodeGen::Visit(AST::Break& n) {
  TraceEachVisit(n);
  IndStream() << "break;\n";
  return true;
}

bool HIPCodeGen::Visit(AST::Continue& n) {
  TraceEachVisit(n);
  IndStream() << "continue;\n";
  return true;
}

bool HIPCodeGen::Visit(AST::NamedTypeDecl&) { return true; }

bool HIPCodeGen::Visit(AST::CppSourceCode& n) {
  TraceEachVisit(n);
  if (Level() == ParallelLevel::NONE) {
    code_segments.push_back(n.code + "\n");
  } else {
    auto& os = Stream();
    auto& indent = Indent();
    os << indent << n.code << "\n";
  }
  return true;
}

bool HIPCodeGen::Visit(AST::Synchronize& n) {
  TraceEachVisit(n);
  switch (n.Resource()) {
  case Storage::GLOBAL:
    hs << h_indent << "hipDeviceSynchronize();\n";
    hs << h_indent << "choreo::verify_device_status();\n";
    break;
  case Storage::SHARED:
    ds << d_indent << "__syncthreads();\n";
    break;
  default:
    choreo_unreachable("unsupported synchronization type: " +
                       STR(n.Resource()) + ".");
  }
  return true;
}

bool HIPCodeGen::Visit(AST::Wait& n) {
  TraceEachVisit(n);
  ds << d_indent << "__syncthreads();\n";
  return true;
}

// ============================================================================
// FunctionDecl -- emit host function header, map parameters
// ============================================================================

void HIPCodeGen::EmitHostFuncDecl(std::ostringstream& os) {
  os << HostTypeStringify(*fty->out_ty, true) << " " << fname << "(";
  bool first = true;
  for (auto& item : GetChoreoFuncIns(cgi)) {
    if (!item.IsParameter()) continue;
    if (!first) os << ", ";
    os << HostTypeStringify(*item.type) << " " << UnScopedName(item.name);
    first = false;
  }
  os << ")";
}

bool HIPCodeGen::Visit(AST::FunctionDecl& n) {
  TraceEachVisit(n);
  assert(n.name == fname && "inconsistent function names.");
  assert(isa<FunctionType>(n.GetType()) && "unexpected type.");

  auto HandleSymDims = [this](const ptr<SpannedType>& sty,
                              const std::string& hp_name, size_t hp_index) {
    size_t dim_index = 0;
    for (auto vi : sty->GetShape().Value()) {
      if (auto vale = VISym(vi)) {
        auto dim_expr = hp_name + ".shape()[" + std::to_string(dim_index) + "]";
        if (symbolic_dimensions.count(*vale) == 0) {
          symbolic_dimensions[*vale] = {dim_expr, hp_index, dim_index};
          ssm.MapDeviceSymbolIfNotExist(*vale, UnScopedName(*vale));
        }
      }
      dim_index++;
    }
  };

  size_t host_pindex = 0;
  for (auto& item : GetChoreoFuncIns(cgi)) {
    if (item.IsParameter()) {
      item.host_name = UnScopedName(item.name);
      if (auto sty = dyn_cast<SpannedType>(item.type)) {
        ssm.MapHostSymbol(item.name, item.host_name + ".data()");
        HandleSymDims(sty, item.host_name, host_pindex);
      } else {
        ssm.MapHostSymbol(item.name, item.host_name);
      }
    } else {
      item.host_name = UnScopedName(item.name);
    }
    item.h_name = "args[" + std::to_string(host_pindex) + "]";
    item.h_index = host_pindex;
    host_pindex++;
  }

  if (isa<VoidType>(fty->out_ty)) void_return = true;

  EmitHostFuncDecl(hs);
  hs << " {\n";
  IncrHostIndent();

  for (auto& item : symbolic_dimensions) {
    hs << h_indent << "auto &" << UnScopedName(item.first) << " = "
       << item.second.hsd_expr << ";\n";
    ssm.MapHostSymbol(item.first, UnScopedName(item.first));
  }

  if (NeedDeviceFunc()) {
    for (auto& item : GetChoreoFuncIns(cgi)) {
      if (auto sty = dyn_cast<SpannedType>(item.type)) {
        if (item.attr == ParamAttr::GLOBAL_INPUT) {
          ssm.MapHostSymbol(item.name + "__device",
                            UnScopedName(item.name) + ".data()");
          continue;
        }
        auto sym = UnScopedName(item.name);
        std::string bts = HIPNameBaseType(sty->ElementType());
        auto buf_sym = sym + "__device";
        hs << h_indent << bts << " * " << buf_sym << " = nullptr;\n";
        hs << h_indent << "choreo::abend_true(hipMalloc(&" << buf_sym << ", "
           << UnScopedSizeExpr(*sty) << "));\n";
        hs << h_indent << "choreo::abend_true(hipMemcpy(" << buf_sym << ", "
           << ssm.HostName(item.name) << ", " << UnScopedSizeExpr(*sty)
           << ", hipMemcpyHostToDevice));\n";
        ssm.MapHostSymbol(item.name + "__device", buf_sym);
      }
    }
  }
  return true;
}

bool HIPCodeGen::Visit(AST::ChoreoFunction& n) {
  TraceEachVisit(n);
  DecrHostIndent();
  hs << "}\n\n";
  return true;
}

// ============================================================================
// ParallelBy -- emit device kernel + launch
// ============================================================================

bool HIPCodeGen::Visit(AST::ParallelBy& n) {
  TraceEachVisit(n);

  // Map PV names to device expressions for every ParallelBy node (all levels).
  // Use IfNotExist to handle cases where BPV and SubPV(0) share a name.
  std::string dname[] = {"x", "y", "z"};
  switch (n.GetLevel()) {
  case ParallelLevel::BLOCK:
    for (size_t i = 0; i < n.AllSubPVs().size(); ++i)
      ssm.MapDeviceSymbolIfNotExist(InScopeName(n.GetSubPV(i)->name),
                                    "blockIdx." + dname[i]);
    if (n.AllSubPVs().size() == 1)
      ssm.MapDeviceSymbolIfNotExist(InScopeName(n.BPV()->name), "blockIdx.x");
    break;
  case ParallelLevel::GROUP: {
    std::string vid_pfx = "__vid_";
    if (n.AllSubPVs().size() == 1)
      ssm.MapDeviceSymbolIfNotExist(InScopeName(n.BPV()->name), vid_pfx + "gid_x");
    if (n.AllSubPVs().size() > 0)
      ssm.MapDeviceSymbolIfNotExist(InScopeName(n.GetSubPV(0)->name), vid_pfx + "gid_x");
    if (n.AllSubPVs().size() > 1)
      ssm.MapDeviceSymbolIfNotExist(InScopeName(n.GetSubPV(1)->name), vid_pfx + "gid_y");
    if (n.AllSubPVs().size() > 2)
      ssm.MapDeviceSymbolIfNotExist(InScopeName(n.GetSubPV(2)->name), vid_pfx + "gid_z");
  } break;
  case ParallelLevel::THREAD: {
    std::string vid_pfx = "__vid_";
    if (n.AllSubPVs().size() == 1)
      ssm.MapDeviceSymbolIfNotExist(InScopeName(n.BPV()->name), vid_pfx + "tid_x");
    if (n.AllSubPVs().size() > 0)
      ssm.MapDeviceSymbolIfNotExist(InScopeName(n.GetSubPV(0)->name), vid_pfx + "tid_x");
    if (n.AllSubPVs().size() > 1)
      ssm.MapDeviceSymbolIfNotExist(InScopeName(n.GetSubPV(1)->name), vid_pfx + "tid_y");
    if (n.AllSubPVs().size() > 2)
      ssm.MapDeviceSymbolIfNotExist(InScopeName(n.GetSubPV(2)->name), vid_pfx + "tid_z");
  } break;
  default: break;
  }

  if (!n.IsOuter()) {
    EmitDeviceVirtualIndices(&n);
    return true;
  }

  auto& launch_configs = cgi.GetFunctionLaunches(fname);
  assert(parallel_idx < (int)launch_configs.size());
  auto& lc = launch_configs[parallel_idx];

  auto vi_str = [this](const ValueItem& vi) -> std::string {
    return ValueSTR(vi);
  };
  auto block_str = [&vi_str](const ParallelCounts& pc) -> std::string {
    return "dim3(" + vi_str(pc.x) + ", " + vi_str(pc.y) + ", " + vi_str(pc.z) + ")";
  };

  auto grid_str = block_str(lc.block_count);
  auto bdim_str = [&]() -> std::string {
    auto gc = lc.group_count;
    auto tc = lc.thread_count;
    auto prod_x = gc.x * tc.x;
    auto prod_y = gc.y * tc.y;
    auto prod_z = gc.z * tc.z;
    return "dim3(" + vi_str(prod_x) + ", " + vi_str(prod_y) + ", " + vi_str(prod_z) + ")";
  }();

  ds << "\n__global__ void " << device_fn << "(";
  bool first = true;
  for (auto& item : cgi.GetDeviceAllIns(fname)) {
    if (!first) ds << ", ";
    if (auto sty = dyn_cast<SpannedType>(item.type)) {
      ds << HIPNameBaseType(sty->ElementType()) << "* "
         << UnScopedName(item.name);
      ssm.MapDeviceSymbolIfNotExist(item.name, UnScopedName(item.name));
    } else if (CanYieldAnInteger(item.type)) {
      ds << "int " << UnScopedName(item.name);
      ssm.MapDeviceSymbolIfNotExist(item.name, UnScopedName(item.name));
    } else if (auto ety = dyn_cast<EventType>(item.type)) {
      (void)ety;
      continue;
    } else {
      ds << HostTypeStringify(*item.type) << " " << UnScopedName(item.name);
      ssm.MapDeviceSymbolIfNotExist(item.name, UnScopedName(item.name));
    }
    first = false;
  }
  ds << ") {\n";
  IncrDeviceIndent();

  EmitDeviceVirtualIndices(&n);

  hs << h_indent << grid_str.substr(0, 4) << " __grid(" << grid_str.substr(5)
     << ";\n";
  hs << h_indent << "dim3 __block(" << bdim_str.substr(5) << ";\n";
  hs << h_indent << device_fn << "<<<__grid, __block>>>(";
  first = true;
  for (auto& item : cgi.GetDeviceAllIns(fname)) {
    if (auto ety = dyn_cast<EventType>(item.type)) {
      (void)ety;
      continue;
    }
    if (!first) hs << ", ";
    if (dyn_cast<SpannedType>(item.type)) {
      hs << ssm.HostName(item.name + "__device");
    } else {
      hs << ssm.HostName(item.name);
    }
    first = false;
  }
  hs << ");\n";

  return true;
}

void HIPCodeGen::EmitDeviceVirtualIndices(AST::ParallelBy* pb) {
  auto sub_pvs = pb->AllSubPVs();
  if (sub_pvs.empty()) return;
  const auto& bvs = pb->BoundValues();
  auto pv_y = bvs.size() > 1 ? bvs.at(1) : sbe::nu(1);
  auto pv_z = bvs.size() > 2 ? bvs.at(2) : sbe::nu(1);

  switch (pb->GetLevel()) {
  case ParallelLevel::BLOCK: {
    if (sub_pvs.size() == 1) {
      auto name = cast<AST::Identifier>(sub_pvs[0])->name;
      ds << d_indent << "int __vid_bid_x = blockIdx.x;\n";
      ssm.MapDeviceSymbolIfNotExist(InScopeName(name), "__vid_bid_x");
    } else if (sub_pvs.size() == 2) {
      auto name_x = cast<AST::Identifier>(sub_pvs[0])->name;
      auto name_y = cast<AST::Identifier>(sub_pvs[1])->name;
      ds << d_indent << "int __vid_bid = blockIdx.x;\n";
      ds << d_indent << "int __vid_bid_x = __vid_bid / " << ValueSTR(pv_y) << ";\n";
      ds << d_indent << "int __vid_bid_y = __vid_bid % " << ValueSTR(pv_y) << ";\n";
      ssm.MapDeviceSymbolIfNotExist(InScopeName(name_x), "__vid_bid_x");
      ssm.MapDeviceSymbolIfNotExist(InScopeName(name_y), "__vid_bid_y");
    }
    break;
  }
  case ParallelLevel::GROUP: {
    if (sub_pvs.size() >= 1) {
      auto name = cast<AST::Identifier>(sub_pvs[0])->name;
      ds << d_indent << "int __vid_gid_x = threadIdx.x / 32;\n";
      ssm.MapDeviceSymbolIfNotExist(InScopeName(name), "__vid_gid_x");
    }
    if (sub_pvs.size() >= 2) {
      auto name = cast<AST::Identifier>(sub_pvs[1])->name;
      ds << d_indent << "int __vid_gid_y = (threadIdx.x / 32) % " << ValueSTR(pv_y) << ";\n";
      ssm.MapDeviceSymbolIfNotExist(InScopeName(name), "__vid_gid_y");
    }
    break;
  }
  case ParallelLevel::THREAD: {
    if (sub_pvs.size() == 1) {
      auto name = cast<AST::Identifier>(sub_pvs[0])->name;
      ds << d_indent << "int __vid_tid_x = threadIdx.x;\n";
      ssm.MapDeviceSymbolIfNotExist(InScopeName(name), "__vid_tid_x");
    } else if (sub_pvs.size() == 2) {
      auto name_x = cast<AST::Identifier>(sub_pvs[0])->name;
      auto name_y = cast<AST::Identifier>(sub_pvs[1])->name;
      ds << d_indent << "int __vid_tid = threadIdx.x;\n";
      ds << d_indent << "int __vid_tid_x = __vid_tid / " << ValueSTR(pv_y) << ";\n";
      ds << d_indent << "int __vid_tid_y = __vid_tid % " << ValueSTR(pv_y) << ";\n";
      ssm.MapDeviceSymbolIfNotExist(InScopeName(name_x), "__vid_tid_x");
      ssm.MapDeviceSymbolIfNotExist(InScopeName(name_y), "__vid_tid_y");
    }
    break;
  }
  default: break;
  }
}

// ============================================================================
// DMA -- emit memory copies
// ============================================================================

bool HIPCodeGen::Visit(AST::DMA& n) {
  TraceEachVisit(n);

  const HIPDMALoweringDecision* dec = HIPDMAPlan::Lookup(&n);
  if (!dec) {
    choreo_unreachable("DMA plan must exist before HIPCodeGen emission.");
    return false;
  }

  auto from_ca = dyn_cast<AST::ChunkAt>(n.GetFrom());
  auto to_ca = dyn_cast<AST::ChunkAt>(n.GetTo());

  auto from_expr = ExprSTR(n.GetFrom(), IsHost());
  auto to_expr = ExprSTR(n.GetTo(), IsHost());

  if (dec->IsNaive()) {
    if (IsHost()) {
      auto from_sty = GetSpannedType(NodeType(*n.GetFrom()));
      auto to_sty = GetSpannedType(NodeType(*n.GetTo()));
      if (from_sty && to_sty) {
        hs << h_indent << "hipMemcpy(" << to_expr << ", " << from_expr << ", "
           << UnScopedSizeExpr(*to_sty) << ", hipMemcpyDefault);\n";
      }
    } else {
      auto to_sty = GetSpannedType(NodeType(*n.GetTo()));
      if (to_sty) {
        auto size_expr = UnScopedSizeExpr(*to_sty);
        ds << d_indent << "for (size_t __i = threadIdx.x; __i < "
           << size_expr << " / sizeof(" << HIPNameBaseType(dec->elem_type)
           << "); __i += blockDim.x) {\n";
        ds << d_indent << "  " << to_expr << "[__i] = " << from_expr
           << "[__i];\n";
        ds << d_indent << "}\n";
        ds << d_indent << "__syncthreads();\n";
      }
    }
  } else if (dec->IsTiled()) {
    auto to_sty = GetSpannedType(NodeType(*n.GetTo()));
    if (to_sty) {
      auto size_expr = UnScopedSizeExpr(*to_sty);
      ds << d_indent << "for (size_t __i = threadIdx.x; __i < "
         << size_expr << " / sizeof(" << HIPNameBaseType(dec->elem_type)
         << "); __i += blockDim.x) {\n";
      ds << d_indent << "  " << to_expr << "[__i] = " << from_expr
         << "[__i];\n";
      ds << d_indent << "}\n";
      ds << d_indent << "__syncthreads();\n";
    }
  }

  return true;
}

// ============================================================================
// Assignment
// ============================================================================

bool HIPCodeGen::Visit(AST::Assignment& n) {
  TraceEachVisit(n);

  auto nty = NodeType(n);
  auto sty = GetSpannedType(nty);

  if (!n.AssignToDataElement()) {
    auto name = n.GetName();
    bool ref = n.HasNote("ref");
    if (!SSTab().IsDeclared(name) && !isa<AST::SpanAs>(n.value))
      updating_cgi.AddSymbolDetail(
          fname, {InScopeName(name), GetSymbolType(name), ref});
  }

  auto& os = Stream();
  auto& indent = Indent();

  if (n.AssignToDataElement()) {
    os << indent << ExprSTR(n.da, IsHost()) << " = "
       << ExprSTR(n.value, IsHost()) << ";\n";
    return true;
  }

  if (sty && sty->GetStorage() == Storage::REG && n.HasNote("update")) {
    os << indent << ExprSTR(n.da, IsHost()) << " = "
       << ExprSTR(n.value, IsHost()) << ";\n";
    return true;
  }

  auto name = n.GetName();
  auto scoped_name = InScopeName(name);

  if (CanYieldAnInteger(nty)) {
    os << indent << "int " << name << " = " << ExprSTR(n.value, IsHost())
       << ";\n";
    if (IsHost())
      ssm.MapHostSymbol(scoped_name, name);
    else
      ssm.MapDeviceSymbolIfNotExist(scoped_name, name);
    return true;
  }

  if (auto scalar = dyn_cast<ScalarType>(nty)) {
    os << indent << HIPNameBaseType(scalar->GetBaseType()) << " " << name << " = "
       << ExprSTR(n.value, IsHost()) << ";\n";
    if (IsHost())
      ssm.MapHostSymbol(scoped_name, name);
    else
      ssm.MapDeviceSymbolIfNotExist(scoped_name, name);
    return true;
  }

  os << indent << "auto " << name << " = " << ExprSTR(n.value, IsHost())
     << ";\n";
  if (IsHost())
    ssm.MapHostSymbol(scoped_name, name);
  else
    ssm.MapDeviceSymbolIfNotExist(scoped_name, name);

  return true;
}

// ============================================================================
// NamedVariableDecl
// ============================================================================

bool HIPCodeGen::Visit(AST::NamedVariableDecl& n) {
  TraceEachVisit(n);

  auto nty = NodeType(n);
  auto sym = n.name_str;

  SSTab().DefineSymbol(sym, nty);

  bool ref = n.HasNote("ref");
  auto sname = InScopeName(sym);
  if (!FCtx(fname).HasSymbolValues(sname))
    updating_cgi.AddSymbolDetail(fname, {sname, GetSymbolType(sym), true});
  else
    updating_cgi.AddSymbolDetail(fname, {sname, GetSymbolType(sym), ref});

  if (auto sty = dyn_cast<SpannedType>(nty)) {
    auto sto = sty->GetStorage();

    if (IsHost() && sto == Storage::GLOBAL) {
      auto bts = HIPNameBaseType(sty->ElementType());
      auto buf_sym = sym + "__device";
      auto shape = sty->GetShape();

      if (IsChoreoOutput(sname)) {
        hs << h_indent << "auto " << sym
           << " = choreo::make_spandata<choreo::" << STR(sty->e_type) << ", "
           << shape.Rank() << ">({"
           << ShapeSTR(shape, false, ", ", BaseType::U64) << "});\n";
        hs << h_indent << bts << " * " << buf_sym << " = nullptr;\n";
        hs << h_indent << "choreo::abend_true(hipMalloc(&" << buf_sym << ", "
           << UnScopedSizeExpr(*sty) << "));\n";
      } else {
        hs << h_indent << bts << " * " << buf_sym << " = nullptr;\n";
        hs << h_indent << "choreo::abend_true(hipMalloc(&" << buf_sym << ", "
           << UnScopedSizeExpr(*sty) << "));\n";
      }
      ssm.MapHostSymbol(sname + "__device", buf_sym);
      ssm.MapHostSymbol(sname, buf_sym);
      ssm.MapDeviceSymbolIfNotExist(sname, sym);
      global_buffers.insert(buf_sym);
      return true;
    }

    if (!IsHost()) {
      if (sto == Storage::SHARED) {
        ds << d_indent << "__shared__ " << HIPNameBaseType(sty->ElementType())
           << " " << sym << "[" << UnScopedSizeExpr(*sty) << " / sizeof("
           << HIPNameBaseType(sty->ElementType()) << ")];\n";
        ssm.MapDeviceSymbolIfNotExist(sname, sym);
        return true;
      }
      if (sto == Storage::LOCAL || sto == Storage::REG) {
        ds << d_indent << HIPNameBaseType(sty->ElementType()) << " " << sym
           << "[" << UnScopedSizeExpr(*sty) << " / sizeof("
           << HIPNameBaseType(sty->ElementType()) << ")];\n";
        ssm.MapDeviceSymbolIfNotExist(sname, sym);
        return true;
      }
    }
  }

  if (CanYieldAnInteger(nty)) {
    auto& os = Stream();
    auto& indent = Indent();
    if (n.init_expr) {
      os << indent << "int " << sym << " = " << ExprSTR(n.init_expr, IsHost())
         << ";\n";
    } else {
      os << indent << "int " << sym << " = 0;\n";
    }
    if (IsHost())
      ssm.MapHostSymbol(sname, sym);
    else
      ssm.MapDeviceSymbolIfNotExist(sname, sym);
    return true;
  }

  if (auto scalar = dyn_cast<ScalarType>(nty)) {
    auto& os = Stream();
    auto& indent = Indent();
    if (n.init_expr) {
      os << indent << HIPNameBaseType(scalar->GetBaseType()) << " " << sym
         << " = " << ExprSTR(n.init_expr, IsHost()) << ";\n";
    } else {
      os << indent << HIPNameBaseType(scalar->GetBaseType()) << " " << sym
         << " = 0;\n";
    }
    if (IsHost())
      ssm.MapHostSymbol(sname, sym);
    else
      ssm.MapDeviceSymbolIfNotExist(sname, sym);
    return true;
  }

  return true;
}

// ============================================================================
// Control flow
// ============================================================================

bool HIPCodeGen::Visit(AST::ForeachBlock& n) {
  TraceEachVisit(n);
  auto& os = Stream();
  auto& indent = Indent();

  for (auto& rn : n.GetRanges()) {
    auto rng = cast<AST::LoopRange>(rn);
    auto loop_var = rng->IVName();
    os << indent << "for (int " << loop_var << " = "
       << ExprSTR(rng->lbound, IsHost()) << "; " << loop_var << " < "
       << ExprSTR(rng->ubound, IsHost()) << "; " << loop_var << " += "
       << std::to_string(rng->step) << ") {\n";
    IncrIndent();
    auto sname = InScopeName(loop_var);
    if (IsHost())
      ssm.MapHostSymbol(sname, loop_var);
    else
      ssm.MapDeviceSymbolIfNotExist(sname, loop_var);
  }
  return true;
}

bool HIPCodeGen::Visit(AST::InThreadsBlock& n) {
  TraceEachVisit(n);
  if (n.HasActiveThreads()) {
    ds << d_indent << "if (threadIdx.x < " << ValueSTR(n.active_threads)
       << ") {\n";
    IncrDeviceIndent();
  } else {
    ds << d_indent << "{\n";
    IncrDeviceIndent();
  }
  return true;
}

bool HIPCodeGen::Visit(AST::IfElseBlock& n) {
  TraceEachVisit(n);
  auto& os = Stream();
  auto& indent = Indent();
  os << indent << "if (" << ExprSTR(n.GetPredicate(), IsHost()) << ") {\n";
  IncrIndent();
  return true;
}

bool HIPCodeGen::Visit(AST::WhileBlock& n) {
  TraceEachVisit(n);
  auto& os = Stream();
  auto& indent = Indent();
  os << indent << "while (" << ExprSTR(n.GetPredicate(), IsHost()) << ") {\n";
  IncrIndent();
  return true;
}

// ============================================================================
// Call
// ============================================================================

bool HIPCodeGen::Visit(AST::Call& n) {
  TraceEachVisit(n);
  if (!emit_call) return true;

  auto& os = IsHost() ? hs : ds;
  auto& indent = IsHost() ? h_indent : d_indent;

  if (n.IsBIF()) {
    const auto func_name = n.function->name;
    if (func_name == "assert") {
      auto args = n.arguments;
      os << indent << "choreo::choreo_assert(" << ExprSTR(args->ValueAt(0), IsHost())
         << ", \"assertion failed\");\n";
      return true;
    }
    if (func_name == "print" || func_name == "println") {
      os << indent << "printf(";
      auto args = n.arguments;
      for (size_t i = 0; i < args->Count(); i++) {
        if (i > 0) os << ", ";
        os << ExprSTR(args->ValueAt(i), IsHost());
      }
      os << ");\n";
      if (func_name == "println") os << indent << "printf(\"\\n\");\n";
      return true;
    }
    if (n.IsArith()) { return true; }
    if (n.IsAtomic()) { return true; }
    choreo_unreachable("builtin '" + func_name + "' not supported by amdgpu.");
  }

  if (!n.IsExpr()) os << indent << CallSTR(n) << ";\n";
  return true;
}

// ============================================================================
// Return
// ============================================================================

bool HIPCodeGen::Visit(AST::Return& n) {
  TraceEachVisit(n);

  if (!void_return && NeedDeviceFunc()) {
    auto ret_sym = cgi.GetReturnSymbol(fname);
    if (auto id = AST::GetIdentifier(*n.value)) {
      auto sym = id->name;
      auto vty = NodeType(*n.value);
      auto sty = dyn_cast<SpannedType>(vty);
      if (sty) {
        auto sname = InScopeName(sym);
        hs << h_indent << "choreo::abend_true(hipMemcpy("
           << sym << ".data(), "
           << ssm.HostName(sname + "__device") << ", "
           << UnScopedSizeExpr(*sty) << ", hipMemcpyDeviceToHost));\n";
      }
    }
    EmitHipFree();
    auto ret_name = cgi.GetReturnSymbol(fname);
    hs << h_indent << "return " << UnScopedName(ret_name) << ";\n";
  } else if (!NeedDeviceFunc() && !void_return) {
    hs << h_indent << "return " << ExprSTR(n.value, true) << ";\n";
  } else {
    EmitHipFree();
  }
  return true;
}

void HIPCodeGen::EmitHipFree() {
  for (auto& item : cgi.GetDeviceAllocIns(fname)) {
    if (auto sty = dyn_cast<SpannedType>(item.type)) {
      (void)sty;
      if (item.attr == ParamAttr::GLOBAL_INPUT) continue;
      auto sym = UnScopedName(item.name);
      hs << h_indent << "hipFree(" << sym << "__device);\n";
    }
  }
}

// ============================================================================
// Expression stringification
// ============================================================================

const std::string HIPCodeGen::ExprSTR(AST::ptr<AST::Node> n,
                                         bool is_host) const {
  if (!n) return "";

  if (auto lit = dyn_cast<AST::IntLiteral>(n))
    return std::visit([](auto v) { return std::to_string(v); }, lit->value);
  if (auto lit = dyn_cast<AST::FloatLiteral>(n)) {
    return std::visit([](auto v) { return std::to_string(v); }, lit->value);
  }
  if (auto id = dyn_cast<AST::Identifier>(n)) {
    auto sname = InScopeNameForRef(id->name);
    if (is_host && ssm.HasHostName(sname)) return ssm.HostName(sname);
    if (!is_host && ssm.HasDeviceName(sname)) return ssm.DeviceName(sname);
    auto sname2 = InScopeName(id->name);
    if (is_host && ssm.HasHostName(sname2)) return ssm.HostName(sname2);
    if (!is_host && ssm.HasDeviceName(sname2)) return ssm.DeviceName(sname2);
    return id->name;
  }
  if (auto expr = dyn_cast<AST::Expr>(n)) {
    if (expr->IsUnary()) {
      return Choreo::STR(expr->op) + "(" + ExprSTR(expr->GetR(), is_host) + ")";
    }
    if (expr->IsBinary()) {
      return "(" + ExprSTR(expr->GetL(), is_host) + " " + Choreo::STR(expr->op) +
             " " + ExprSTR(expr->GetR(), is_host) + ")";
    }
    if (expr->op == Op::ElemOf) {
      return ExprSTR(expr->GetL(), is_host) + "[" +
             ExprSTR(expr->GetR(), is_host) + "]";
    }
    if (auto ref = expr->GetReference())
      return ExprSTR(ref, is_host);
  }
  if (auto ca = dyn_cast<AST::ChunkAt>(n)) {
    auto sym_name = ca->RefSymbol();
    auto sname = InScopeName(sym_name);
    if (is_host && ssm.HasHostName(sname)) return ssm.HostName(sname);
    if (!is_host && ssm.HasDeviceName(sname)) return ssm.DeviceName(sname);
    return sym_name;
  }
  if (auto da = dyn_cast<AST::DataAccess>(n)) {
    auto sym = da->GetDataName();
    auto sname = InScopeNameForRef(sym);
    std::string base = UnScopedName(SSMName(sname, is_host));
    if (da->AccessElement()) {
      auto sty = GetSpannedType(GetSymbolType(sym));
      if (sty) {
        std::ostringstream oss;
        oss << "*((" << HIPNameBaseType(sty->ElementType()) << "*)" << base;
        auto shape = sty->GetShape();
        size_t idx = 0;
        for (auto item : da->GetIndices()) {
          oss << " + ";
          if (shape.Rank() > idx + 1)
            oss << ExprSTR(item, is_host) << " * "
                << ValueSTR(shape.TrimDims(idx + 1).ElementCountValue());
          else
            oss << ExprSTR(item, is_host);
          ++idx;
        }
        oss << ")";
        return oss.str();
      }
    }
    return base;
  }
  if (auto call = dyn_cast<AST::Call>(n)) return CallSTR(*call);
  if (auto mv = dyn_cast<AST::MultiValues>(n)) {
    if (mv->Count() == 1) return ExprSTR(mv->ValueAt(0), is_host);
  }

  return "/* unsupported expr */";
}

const std::string HIPCodeGen::OpExprSTR(AST::ptr<AST::Node> n,
                                           const std::string&,
                                           bool, bool is_host) const {
  return ExprSTR(n, is_host);
}

const std::string HIPCodeGen::CallSTR(AST::Call& n) const {
  std::string result = n.function->name + "(";
  if (n.arguments) {
    for (size_t i = 0; i < n.arguments->Count(); i++) {
      if (i > 0) result += ", ";
      result += ExprSTR(n.arguments->ValueAt(i), IsHost());
    }
  }
  result += ")";
  return result;
}

const std::string HIPCodeGen::ValueSTR(const ValueItem& vi, bool,
                                          bool) const {
  if (!IsValidValueItem(vi)) choreo_unreachable("invalid value item.");
  if (auto iv = VIInt(vi))
    return std::to_string(*iv);
  else if (auto sv = VISym(vi)) {
    auto res = UnScopedExpr(SSMName(sv.value(), IsHost()));
    return res;
  } else if (auto bo = VIBop(vi)) {
    std::string op = OpCodeToStr(bo->GetOpCode());
    return "(" + ValueSTR(bo->GetLeft()) + " " + op + " " + ValueSTR(bo->GetRight()) + ")";
  } else if (auto uo = VIUop(vi)) {
    std::string op = OpCodeToStr(uo->GetOpCode());
    return op + "(" + ValueSTR(uo->GetOperand()) + ")";
  }
  return PSTR(vi);
}

const std::string HIPCodeGen::ValueSTR(const ValueList& vl, bool ll,
                                          bool shp,
                                          const std::string& sep) const {
  std::string result;
  for (size_t i = 0; i < vl.size(); i++) {
    if (i > 0) result += sep;
    result += ValueSTR(vl[i], ll, shp);
  }
  return result;
}

const std::string HIPCodeGen::ShapeSTR(const Shape& s, bool ll,
                                          const std::string& sep,
                                          BaseType) const {
  return ValueSTR(s.Value(), ll, false, sep);
}

// ============================================================================
// Emit headers and source
// ============================================================================

void HIPCodeGen::EmitFixedHostHead() {
  std::ostringstream oss;
  oss << R"(
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#define __CHOREO_TARGET_AMDGPU__ 1
#include "choreo.h"

#include <hip/hip_runtime.h>

)";
  code_segments.push_back(oss.str());
}

void HIPCodeGen::EmitFixedDeviceHead() {
  // No separate device head needed for HIP (single-source model)
}

void HIPCodeGen::EmitSource() {
  for (auto& code : code_segments)
    outs() << code << "\n";
}

void HIPCodeGen::EmitScript(std::ostream& os, const std::string& exe_fn) {
  auto filename = RemoveDirectoryPrefix(
      RemoveSuffix(OptionRegistry::GetInstance().GetInputFileName(), ".co"));

  os << R"script(#!/usr/bin/env bash

# Choreo-generated bash script to compile HIP code for AMD GPU
)script";

  if (RequiresE2ECompilation(CCtx().GetOutputKind())) {
#ifdef __CHOREO_ROCM_DIR__
    os << "\nif [ -z \"${ROCM_HOME}\" ]; then";
    os << "\n  export ROCM_HOME=" << STRINGIZE(__CHOREO_ROCM_DIR__);
    os << "\nfi\n";
#endif
  }

  os << R"script(
if [ -z "${ROCM_HOME}" ]; then
  if [ -d "/opt/rocm" ]; then
    export ROCM_HOME=/opt/rocm
  else
    echo "failed to find the ROCm installation."
    echo "install ROCm or set ROCM_HOME to the ROCm installation directory."
    exit 1
  fi
fi

if [ ! -f ${ROCM_HOME}/bin/hipcc ]; then
  echo "failed to find hipcc in ${ROCM_HOME}/bin."
  exit 1
fi

HIPCC=${ROCM_HOME}/bin/hipcc

)script";

  std::string build_path = CreateUniquePath();
  auto cc_file = build_path + "/__choreo_hip_" + filename + ".hip";
  auto exe_file = exe_fn;
  if (exe_file.empty())
    exe_file = build_path + "/__choreo_hip_" + filename + ".exe";
  os << "rm -fr " << build_path << "\n";
  os << "mkdir -p " << build_path << "\n\n";

  os << "cat <<'EOF' > " << build_path << "/choreo.h\n";
  os << __choreo_header_as_string << "\nEOF\n\n";
  os << "cat <<'EOF' > " << build_path << "/choreo_types.h\n";
  os << __choreo_types_header_as_string << "\nEOF\n\n";

  os << "cat <<'EOF' > " << cc_file << "\n";
  for (auto& code : code_segments)
    os << code << "\n";
  os << "\nEOF\n\n";

  auto arch_str = ToLower(CCtx().GetArch());
  os << "amdgpu_arch=" << arch_str << "\n";

  os << R"script(
show_usage() {
  echo "  Usage: $0 <actions>"
  echo ""
  echo "  Options:"
  echo "   --execute,           Compile and execute"
  echo "   --compile-link,      Compile and link"
  echo "   --compile-module,    Compile to object"
  echo ""
}

action_compile_module() {
)script";
  os << "  ${HIPCC} -c --offload-arch=${amdgpu_arch} -std=c++17 -I"
     << build_path << " " << cc_file << " -o " << build_path
     << "/__choreo_hip_" << filename << ".o\n";
  os << "}\n\n";

  os << "action_compile_link() {\n";
  os << "  ${HIPCC} --offload-arch=${amdgpu_arch} -std=c++17 -I" << build_path
     << " " << cc_file << " -o " << exe_file << "\n";
  os << "}\n\n";

  os << "action_execute() {\n";
  os << "  action_compile_link\n";
  os << "  " << exe_file << "\n";
  os << "}\n\n";

  os << R"script(
case "${1:-}" in
  --execute)        action_execute ;;
  --compile-link)   action_compile_link ;;
  --compile-module) action_compile_module ;;
  *)                show_usage ;;
esac
)script";
}

bool HIPCodeGen::CompileWithScript(const std::string& action) {
  std::string script_path = cmp_dir + "/__choreo_hip_compile.sh";
  std::filesystem::create_directories(cmp_dir);

  {
    std::ofstream ofs(script_path);
    if (!ofs) {
      errs() << "Failed to create compile script at " << script_path << "\n";
      return false;
    }
    auto out_fn = OptionRegistry::GetInstance().GetOutputFileName();
    EmitScript(ofs, out_fn.empty() ? "" : out_fn);
  }

  std::string cmd = "bash " + script_path + " " + action + " 2>&1";
  int ret = std::system(cmd.c_str());
  if (ret != 0) {
    errs() << "HIP compilation failed (exit code " << ret << ").\n";
    return false;
  }
  return true;
}
