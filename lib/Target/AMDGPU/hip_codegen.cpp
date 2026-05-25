#include "hip_codegen.hpp"
#include "ast.hpp"
#include "codegen.hpp"
#include "context.hpp"
#include "hip_dma_plan.hpp"
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
      if (!CompileWithScript(CCtx().GetOutputKind() == OutputKind::TargetModule
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
  } else if (auto fb = dyn_cast<AST::ForeachBlock>(&n)) {
    const auto& ranges = fb->GetRangeNodes();
    for (int j = ranges->Count() - 1; j >= 0; --j) {
      auto rng = cast<AST::LoopRange>(ranges->ValueAt(j));
      auto cname = rng->IVName();
      auto ivs = within_map.at(InScopeName(cname));
      for (auto iv_itr = ivs.rbegin(); iv_itr != ivs.rend(); ++iv_itr) {
        DecrIndent();
        IndStream() << "}\n";
        IndStream() << SSMName(*iv_itr, IsHost()) << " = 0;\n";
      }
    }
  } else if (isa<AST::WhileBlock>(&n)) {
    DecrIndent();
    IndStream() << "}\n";
  } else if (auto it = dyn_cast<AST::InThreadsBlock>(&n)) {
    DecrDeviceIndent();
    if (!it->stmts->None()) {
      ds << d_indent << "} // end inthreads";
      if (!it->async && it->outer) ds << "\n" << d_indent << "__syncthreads();";
      ds << "\n";
    }
  }
  return true;
}

// ============================================================================
// Visit methods
// ============================================================================

bool HIPCodeGen::Visit(AST::ParamList&) { return true; }
bool HIPCodeGen::Visit(AST::WithIn& n) {
  TraceEachVisit(n);
  if (n.with)
    ssm.MapDeviceSymbol(InScopeName(n.with->name), "__iv_" + n.with->name);

  for (auto& v : n.GetMatchers()) {
    auto id = cast<AST::Identifier>(v);
    ssm.RemapDeviceSymbol(InScopeName(id->name), "__iv_" + id->name);
    ssm.RemapHostSymbol(InScopeName(id->name), "__iv_" + id->name);
    if (IsHost())
      hs << h_indent << "int __iv_" << id->name << " = 0;\n";
    else
      ds << d_indent << "int __iv_" << id->name << " = 0;\n";
  }
  if (n.with && (n.GetMatchers().size() == 1)) {
    auto id = cast<AST::Identifier>(n.GetMatchers()[0]);
    ssm.RemapDeviceSymbol(InScopeName(n.with->name), "__iv_" + id->name);
    ssm.RemapHostSymbol(InScopeName(n.with->name), "__iv_" + id->name);
  }
  return true;
}
bool HIPCodeGen::Visit(AST::WhereBind&) { return true; }
bool HIPCodeGen::Visit(AST::WithBlock&) { return true; }

bool HIPCodeGen::Visit(AST::MMA&) {
  choreo_unreachable("MMA is not yet supported by the HIP target.");
  return false;
}

bool HIPCodeGen::Visit(AST::Trigger& n) {
  TraceEachVisit(n);

  auto EmitDeviceEventTrigger = [&](auto fty, auto f, auto expr,
                                    bool is_array_ref) {
    auto ety_sto = [&]() {
      if (auto ea = dyn_cast<EventArrayType>(fty)) return ea->GetStorage();
      return cast<EventType>(fty)->GetStorage();
    }();
    switch (ety_sto) {
    case Storage::GLOBAL:
      ds << d_indent << "__threadfence();\n";
      [[fallthrough]];
    case Storage::SHARED:
    case Storage::LOCAL:
      if (auto ea = dyn_cast<EventArrayType>(fty)) {
        if (is_array_ref) {
          size_t lvl = AST::GetSubScriptLevel(*expr);
          auto bid = AST::GetArrayBaseSymbol(*expr);
          auto bty =
              cast<EventArrayType>(GetSymbolType(UnScopedName(bid->name)));
          GenerateSubscriptions(ds, d_indent + ExprSTR(f, false), " = true;\n",
                                bty->RemainderDimensions(lvl));
        } else
          GenerateSubscriptions(ds, d_indent + ExprSTR(f, false), " = true;\n",
                                ea->RemainderDimensions(0));
      } else {
        if (is_array_ref) {
          size_t lvl = AST::GetSubScriptLevel(*expr);
          auto bid = AST::GetArrayBaseSymbol(*expr);
          auto bty =
              cast<EventArrayType>(GetSymbolType(UnScopedName(bid->name)));
          GenerateSubscriptions(ds, d_indent + ExprSTR(f, false), " = true;\n",
                                bty->RemainderDimensions(lvl));
        } else
          ds << d_indent << ExprSTR(f, false) << " = true;\n";
      }
      break;
    default:
      choreo_unreachable("unsupported event storage '" + STR(ety_sto) +
                         "' to trigger.");
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
        hs << h_indent << "choreo::abend_true(hipMemset(&" << ExprSTR(f, true)
           << ", 1, " << ety->ElemCount() << "));\n";
      } else {
        EmitDeviceEventTrigger(fty, f, expr, is_array_ref);
      }
    } else if (auto ety = dyn_cast<EventType>(fty)) {
      if (IsHost()) {
        assert(ety->GetStorage() == Storage::GLOBAL);
        hs << h_indent << "choreo::abend_true(hipMemset(&" << ExprSTR(f, true)
           << ", 1, 1));\n";
      } else {
        EmitDeviceEventTrigger(fty, f, expr, is_array_ref);
      }
    }
  }
  return true;
}

bool HIPCodeGen::Visit(AST::Rotate&) {
  choreo_unreachable("Rotate (future ring) is not yet supported by the HIP "
                     "target.");
  return false;
}

bool HIPCodeGen::Visit(AST::Yield& n) {
  TraceEachVisit(n);
  IndStream() << "return;\n";
  return true;
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
  case Storage::SHARED: ds << d_indent << "__syncthreads();\n"; break;
  case Storage::LOCAL: ds << d_indent << "__threadfence_block();\n"; break;
  default:
    choreo_unreachable(
        "unsupported synchronization type: " + STR(n.Resource()) + ".");
  }
  return true;
}

bool HIPCodeGen::Visit(AST::Wait& n) {
  TraceEachVisit(n);

  if (NeedLevelPred()) {
    ds << d_indent << LevelPred(Level()) << "{\n";
    IncrDeviceIndent();
  }

  for (auto& f : n.GetTargets()) {
    auto expr = cast<AST::Expr>(f);
    bool is_array_ref = (expr->op == Op::ElemOf);
    auto fty = is_array_ref
                   ? GetSymbolType(AST::GetArrayBaseSymbol(*expr)->name)
                   : NodeType(*f);
    if (isa<FutureType>(fty)) {
      assert(!IsHost());
      ds << d_indent << ExprSTR(f, false) << ".wait();\n";
    } else if (auto ety = dyn_cast<EventArrayType>(fty)) {
      if (IsHost())
        choreo_unreachable("yet to support: wait global event in host.");
      switch (ety->GetStorage()) {
      case Storage::GLOBAL:
      case Storage::SHARED:
      case Storage::LOCAL: {
        ds << d_indent << "while (";
        if (is_array_ref) {
          size_t lvl = AST::GetSubScriptLevel(*expr);
          auto bid = AST::GetArrayBaseSymbol(*expr);
          auto bty =
              cast<EventArrayType>(GetSymbolType(UnScopedName(bid->name)));
          GenerateSubscriptions(ds, "!" + ExprSTR(f, false), " || ",
                                bty->RemainderDimensions(lvl));
        } else
          GenerateSubscriptions(ds, "!" + ExprSTR(f, false), " || ",
                                ety->RemainderDimensions(0));
        ds << "false) continue;\n";
        if (is_array_ref) {
          size_t lvl = AST::GetSubScriptLevel(*expr);
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
    } else if (auto ety = dyn_cast<EventType>(fty)) {
      if (IsHost())
        choreo_unreachable("yet to support: wait global event in host.");
      switch (ety->GetStorage()) {
      case Storage::GLOBAL:
      case Storage::SHARED:
      case Storage::LOCAL: {
        ds << d_indent << "while (" << ExprSTR(f, false)
           << " == false) continue; // spinlock\n";
        if (is_array_ref) {
          size_t lvl = AST::GetSubScriptLevel(*expr);
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

  if (NeedLevelPred()) {
    DecrDeviceIndent();
    ds << d_indent << "}\n";
    ds << d_indent << "__syncthreads();\n";
  }

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
  if (void_return) EmitHipFree();
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
      ssm.MapDeviceSymbolIfNotExist(InScopeName(n.BPV()->name),
                                    vid_pfx + "gid_x");
    if (n.AllSubPVs().size() > 0)
      ssm.MapDeviceSymbolIfNotExist(InScopeName(n.GetSubPV(0)->name),
                                    vid_pfx + "gid_x");
    if (n.AllSubPVs().size() > 1)
      ssm.MapDeviceSymbolIfNotExist(InScopeName(n.GetSubPV(1)->name),
                                    vid_pfx + "gid_y");
    if (n.AllSubPVs().size() > 2)
      ssm.MapDeviceSymbolIfNotExist(InScopeName(n.GetSubPV(2)->name),
                                    vid_pfx + "gid_z");
  } break;
  case ParallelLevel::THREAD: {
    std::string vid_pfx = "__vid_";
    if (n.AllSubPVs().size() == 1)
      ssm.MapDeviceSymbolIfNotExist(InScopeName(n.BPV()->name),
                                    vid_pfx + "tid_x");
    if (n.AllSubPVs().size() > 0)
      ssm.MapDeviceSymbolIfNotExist(InScopeName(n.GetSubPV(0)->name),
                                    vid_pfx + "tid_x");
    if (n.AllSubPVs().size() > 1)
      ssm.MapDeviceSymbolIfNotExist(InScopeName(n.GetSubPV(1)->name),
                                    vid_pfx + "tid_y");
    if (n.AllSubPVs().size() > 2)
      ssm.MapDeviceSymbolIfNotExist(InScopeName(n.GetSubPV(2)->name),
                                    vid_pfx + "tid_z");
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
    return "dim3(" + vi_str(pc.x) + ", " + vi_str(pc.y) + ", " + vi_str(pc.z) +
           ")";
  };

  auto grid_str = block_str(lc.block_count);
  auto bdim_str = [&]() -> std::string {
    auto gc = lc.group_count;
    auto tc = lc.thread_count;
    auto prod_x = gc.x * tc.x;
    auto prod_y = gc.y * tc.y;
    auto prod_z = gc.z * tc.z;
    return "dim3(" + vi_str(prod_x) + ", " + vi_str(prod_y) + ", " +
           vi_str(prod_z) + ")";
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
    } else if (isa<EventArrayType>(item.type)) {
      ds << "bool * " << UnScopedName(item.name);
      ssm.MapDeviceSymbolIfNotExist(item.name, UnScopedName(item.name));
    } else if (auto et = dyn_cast<EventType>(item.type)) {
      if (et->GetStorage() == Storage::GLOBAL) {
        ds << "bool * " << UnScopedName(item.name);
      } else {
        ds << "bool " << UnScopedName(item.name);
      }
      ssm.MapDeviceSymbolIfNotExist(item.name, UnScopedName(item.name));
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
    if (!first) hs << ", ";
    if (dyn_cast<SpannedType>(item.type)) {
      hs << ssm.HostName(item.name + "__device");
    } else if (isa<EventArrayType>(item.type) || isa<EventType>(item.type)) {
      hs << ssm.HostName(item.name);
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
      ds << d_indent << "int __vid_bid_x = __vid_bid / " << ValueSTR(pv_y)
         << ";\n";
      ds << d_indent << "int __vid_bid_y = __vid_bid % " << ValueSTR(pv_y)
         << ";\n";
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
      ds << d_indent << "int __vid_gid_y = (threadIdx.x / 32) % "
         << ValueSTR(pv_y) << ";\n";
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
      ds << d_indent << "int __vid_tid_x = __vid_tid / " << ValueSTR(pv_y)
         << ";\n";
      ds << d_indent << "int __vid_tid_y = __vid_tid % " << ValueSTR(pv_y)
         << ";\n";
      ssm.MapDeviceSymbolIfNotExist(InScopeName(name_x), "__vid_tid_x");
      ssm.MapDeviceSymbolIfNotExist(InScopeName(name_y), "__vid_tid_y");
    }
    break;
  }
  default: break;
  }
}

// ============================================================================
// DMA -- emit memory copies (copy / pad / transpose, sync / async)
// ============================================================================

bool HIPCodeGen::Visit(AST::DMA& n) {
  TraceEachVisit(n);

  auto nty = NodeType(n);

  // PlaceHolderType: bare future declarations (e.g. "f = dma;") -- no copy.
  if (isa<PlaceHolderType>(nty)) return true;

  const HIPDMALoweringDecision* dec = HIPDMAPlan::Lookup(&n);
  if (!dec) {
    choreo_unreachable("DMA plan must exist before HIPCodeGen emission.");
    return false;
  }

  auto from_expr = ExprSTR(n.GetFrom(), IsHost());
  auto to_expr = ExprSTR(n.GetTo(), IsHost());

  // If this DMA has a future name, map the future symbol to the destination
  // address (including any ChunkAt subview offset) so later "f.data"
  // references resolve correctly.
  if (!IsHost() && !n.future.empty() && isa<AST::ChunkAt>(n.GetTo())) {
    auto future_name = InScopeName(n.future);
    ssm.MapDeviceSymbolIfNotExist(future_name, to_expr);
    ssm.MapDeviceSymbolIfNotExist(future_name + ".data", to_expr);
  }

  bool is_async = dec->is_async;

  // Host-side DMA: hipMemcpy
  if (IsHost()) {
    auto to_sty = GetSpannedType(NodeType(*n.GetTo()));
    if (to_sty) {
      hs << h_indent << "hipMemcpy(" << to_expr << ", " << from_expr << ", "
         << UnScopedSizeExpr(*to_sty) << ", hipMemcpyDefault);\n";
    }
    return true;
  }

  auto from_sty = GetSpannedType(NodeType(*n.GetFrom()));
  auto to_sty = GetSpannedType(NodeType(*n.GetTo()));
  if (!to_sty) return true;

  ds << d_indent << "// DMA " << dec->operation << " [" << STR(dec->strategy)
     << ", " << STR(dec->direction) << "]\n";
  if (dec->IsPad()) {
    EmitDMAPad(n, *dec, from_expr, to_expr, from_sty, to_sty);
  } else if (dec->IsTranspose()) {
    EmitDMATranspose(n, *dec, from_expr, to_expr, from_sty, to_sty);
  } else {
    EmitDMACopy(*dec, from_expr, to_expr, to_sty);
  }

  if (!is_async) ds << d_indent << "__syncthreads();\n";

  return true;
}

void HIPCodeGen::EmitDMACopy(const HIPDMALoweringDecision& dec,
                             const std::string& from_expr,
                             const std::string& to_expr,
                             const ptr<SpannedType>& to_sty) {
  auto bt_name = HIPNameBaseType(dec.elem_type);
  auto size_expr = UnScopedSizeExpr(*to_sty);
  ds << d_indent << "for (size_t __i = threadIdx.x; __i < " << size_expr
     << " / sizeof(" << bt_name << "); __i += blockDim.x) {\n";
  ds << d_indent << "  ((" << bt_name << "*)" << to_expr << ")[__i] = "
     << "((" << bt_name << "*)" << from_expr << ")[__i];\n";
  ds << d_indent << "}\n";
}

void HIPCodeGen::EmitDMAPad(AST::DMA& n, const HIPDMALoweringDecision& dec,
                            const std::string& from_expr,
                            const std::string& to_expr,
                            const ptr<SpannedType>& from_sty,
                            const ptr<SpannedType>& to_sty) {
  auto bt_name = HIPNameBaseType(dec.elem_type);
  auto pad_cfg = dyn_cast<PadConfig>(n.config);
  assert(pad_cfg && "pad DMA must have PadConfig");

  auto to_size_expr = UnScopedSizeExpr(*to_sty);

  std::string pad_val = "0";
  if (pad_cfg->GetPadValue()) pad_val = ExprSTR(pad_cfg->GetPadValue(), false);

  // Step 1: fill entire destination with pad value
  ds << d_indent << "for (size_t __i = threadIdx.x; __i < " << to_size_expr
     << " / sizeof(" << bt_name << "); __i += blockDim.x) {\n";
  ds << d_indent << "  ((" << bt_name << "*)" << to_expr << ")[__i] = ("
     << bt_name << ")" << pad_val << ";\n";
  ds << d_indent << "}\n";
  ds << d_indent << "__syncthreads();\n";

  // Step 2: copy source data into the padded inner region
  if (!from_sty) return;

  auto from_shape = from_sty->GetShape();
  auto to_shape = to_sty->GetShape();
  int rank = from_shape.Rank();

  std::vector<std::string> pad_low_vals;
  if (pad_cfg->pad_low) {
    for (size_t i = 0; i < pad_cfg->pad_low->Count(); ++i)
      pad_low_vals.push_back(ExprSTR(pad_cfg->pad_low->ValueAt(i), false));
  }
  while ((int)pad_low_vals.size() < rank) pad_low_vals.push_back("0");

  auto from_size_expr = UnScopedSizeExpr(*from_sty);
  auto from_elem_count = from_size_expr + " / sizeof(" + bt_name + ")";

  auto from_vals = from_shape.Value();
  auto to_vals = to_shape.Value();
  std::vector<std::string> from_dim_strs, to_dim_strs;
  for (int i = 0; i < rank; ++i) {
    from_dim_strs.push_back(ValueSTR(from_vals[i]));
    to_dim_strs.push_back(ValueSTR(to_vals[i]));
  }

  ds << d_indent << "for (size_t __i = threadIdx.x; __i < " << from_elem_count
     << "; __i += blockDim.x) {\n";

  // Decompose flat index __i into multi-dim coords using from_shape
  ds << d_indent << "  size_t __rem = __i;\n";
  for (int d = 0; d < rank; ++d) {
    std::string dim_name = "__d" + std::to_string(d);
    if (d < rank - 1) {
      // stride = product of remaining dimensions
      std::string stride_expr = from_dim_strs[d + 1];
      for (int k = d + 2; k < rank; ++k)
        stride_expr = "(" + stride_expr + " * " + from_dim_strs[k] + ")";
      ds << d_indent << "  size_t " << dim_name << " = __rem / (" << stride_expr
         << ");\n";
      ds << d_indent << "  __rem = __rem % (" << stride_expr << ");\n";
    } else {
      ds << d_indent << "  size_t " << dim_name << " = __rem;\n";
    }
  }

  // Compute destination flat index with pad_low offsets
  ds << d_indent << "  size_t __dst_idx = ";
  for (int d = 0; d < rank; ++d) {
    if (d > 0) ds << " + ";
    std::string coord =
        "(__d" + std::to_string(d) + " + " + pad_low_vals[d] + ")";
    if (d < rank - 1) {
      std::string stride_expr = to_dim_strs[d + 1];
      for (int k = d + 2; k < rank; ++k)
        stride_expr = "(" + stride_expr + " * " + to_dim_strs[k] + ")";
      ds << coord << " * (" << stride_expr << ")";
    } else {
      ds << coord;
    }
  }
  ds << ";\n";

  ds << d_indent << "  ((" << bt_name << "*)" << to_expr << ")[__dst_idx] = (("
     << bt_name << "*)" << from_expr << ")[__i];\n";
  ds << d_indent << "}\n";
}

void HIPCodeGen::EmitDMATranspose(AST::DMA&, const HIPDMALoweringDecision& dec,
                                  const std::string& from_expr,
                                  const std::string& to_expr,
                                  const ptr<SpannedType>& from_sty,
                                  const ptr<SpannedType>& to_sty) {
  auto bt_name = HIPNameBaseType(dec.elem_type);
  if (!from_sty) return;

  auto from_shape = from_sty->GetShape();
  auto to_shape = to_sty->GetShape();
  int rank = from_shape.Rank();

  auto perm = dec.transpose_perm;
  if (perm.empty()) {
    for (int i = rank - 1; i >= 0; --i) perm.push_back(i);
  }

  auto from_size_expr = UnScopedSizeExpr(*from_sty);
  auto from_elem_count = from_size_expr + " / sizeof(" + bt_name + ")";

  auto from_vals = from_shape.Value();
  auto to_vals = to_shape.Value();
  std::vector<std::string> from_dim_strs, to_dim_strs;
  for (int i = 0; i < rank; ++i) {
    from_dim_strs.push_back(ValueSTR(from_vals[i]));
    to_dim_strs.push_back(ValueSTR(to_vals[i]));
  }

  ds << d_indent << "for (size_t __i = threadIdx.x; __i < " << from_elem_count
     << "; __i += blockDim.x) {\n";

  // Decompose flat index into source multi-dim coords
  ds << d_indent << "  size_t __rem = __i;\n";
  for (int d = 0; d < rank; ++d) {
    std::string dim_name = "__d" + std::to_string(d);
    if (d < rank - 1) {
      std::string stride_expr = from_dim_strs[d + 1];
      for (int k = d + 2; k < rank; ++k)
        stride_expr = "(" + stride_expr + " * " + from_dim_strs[k] + ")";
      ds << d_indent << "  size_t " << dim_name << " = __rem / (" << stride_expr
         << ");\n";
      ds << d_indent << "  __rem = __rem % (" << stride_expr << ");\n";
    } else {
      ds << d_indent << "  size_t " << dim_name << " = __rem;\n";
    }
  }

  // Compute transposed destination flat index
  ds << d_indent << "  size_t __dst_idx = ";
  for (int d = 0; d < rank; ++d) {
    if (d > 0) ds << " + ";
    std::string coord = "__d" + std::to_string(perm[d]);
    if (d < rank - 1) {
      std::string stride_expr = to_dim_strs[d + 1];
      for (int k = d + 2; k < rank; ++k)
        stride_expr = "(" + stride_expr + " * " + to_dim_strs[k] + ")";
      ds << coord << " * (" << stride_expr << ")";
    } else {
      ds << coord;
    }
  }
  ds << ";\n";

  ds << d_indent << "  ((" << bt_name << "*)" << to_expr << ")[__dst_idx] = (("
     << bt_name << "*)" << from_expr << ")[__i];\n";
  ds << d_indent << "}\n";
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

  if (auto sa = dyn_cast<AST::SpanAs>(n.value)) {
    assert(!IsHost() && "span-as should be on device side.");
    auto name = n.GetName();
    auto scoped_name = InScopeName(name);
    ds << d_indent << "auto* " << name << " = ";
    auto tty = GetSymbolType(sa->id->name);
    if (isa<FutureType>(tty))
      ds << SSMName(InScopeNameForRef(sa->id->name), false) << ".data();\n";
    else
      ds << SSMName(InScopeNameForRef(sa->id->name), false) << ";\n";
    ssm.MapDeviceSymbolIfNotExist(scoped_name, name);
    return true;
  }

  auto name = n.GetName();
  auto scoped_name = InScopeName(name);

  bool already_declared = (IsHost() ? ssm.HasHostName(scoped_name)
                                    : ssm.HasDeviceName(scoped_name));

  if (already_declared) {
    os << indent << SSMName(scoped_name, IsHost()) << " = "
       << ExprSTR(n.value, IsHost()) << ";\n";
    return true;
  }

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
    os << indent << HIPNameBaseType(scalar->GetBaseType()) << " " << name
       << " = " << ExprSTR(n.value, IsHost()) << ";\n";
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

  ptr<AST::SpanAs> sa = nullptr;
  if (auto e = dyn_cast<AST::Expr>(n.init_expr))
    sa = dyn_cast<AST::SpanAs>(e->GetReference());

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

      if (sa) {
        auto src_buf = ssm.HostName(InScopeName(sa->id->name) + "__device");
        hs << h_indent << bts << " * " << buf_sym << " = " << src_buf << ";\n";
      } else if (IsChoreoOutput(sname)) {
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
      if (sa) {
        auto base_dev = ssm.DeviceName(InScopeName(sa->id->name));
        ds << d_indent << "auto* " << sym << " = " << base_dev << ";\n";
        ssm.MapDeviceSymbolIfNotExist(sname, sym);
        return true;
      }
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

  if (auto ety = dyn_cast<EventArrayType>(nty)) {
    auto eaname = UniqueDeviceName(n.name_str);
    switch (ety->GetStorage()) {
    case Storage::GLOBAL: {
      assert(IsHost());
      auto buf_sym = eaname + "__device";
      hs << h_indent << "bool * " << buf_sym << " = nullptr;\n";
      hs << h_indent << "choreo::abend_true(hipMalloc(&" << buf_sym << ", "
         << ety->ElemCount() << "));\n";
      hs << h_indent << "choreo::abend_true(hipMemset(" << buf_sym << ", 0, "
         << ety->ElemCount() << "));\n";
      ssm.MapHostSymbol(sname, buf_sym);
      ssm.MapDeviceSymbol(sname, eaname);
      global_buffers.insert(buf_sym);
      event_global_buffers.push_back(buf_sym);
    } break;
    case Storage::SHARED:
    case Storage::LOCAL: {
      assert(!IsHost());
      auto mem_qual =
          (ety->GetStorage() == Storage::SHARED) ? "__shared__ " : "";
      ds << d_indent << mem_qual << "__volatile__ bool " << eaname;
      ety->PrintAsCArray(ds);
      ds << "; // " << STR(ety->GetStorage()) << " event\n";
      if (ety->GetStorage() == Storage::SHARED)
        ds << d_indent << "if (__CHOREO_BLOCK_SINGLE__) {\n";
      GenerateSubscriptions(ds, "  " + d_indent + eaname, " = false;\n",
                            ety->Dimensions());
      if (ety->GetStorage() == Storage::SHARED) {
        ds << d_indent << "}\n";
        ds << d_indent << "__syncthreads();\n";
      }
      ssm.MapDeviceSymbolIfNotExist(sname, eaname);
    } break;
    default: break;
    }
    return true;
  }

  if (auto ety = dyn_cast<EventType>(nty)) {
    auto ename = UniqueDeviceName(n.name_str);
    switch (ety->GetStorage()) {
    case Storage::GLOBAL: {
      assert(IsHost());
      auto buf_sym = ename + "__device";
      hs << h_indent << "bool * " << buf_sym << " = nullptr;\n";
      hs << h_indent << "choreo::abend_true(hipMalloc(&" << buf_sym
         << ", 1));\n";
      hs << h_indent << "choreo::abend_true(hipMemset(" << buf_sym
         << ", 0, 1));\n";
      ssm.MapHostSymbol(sname, buf_sym);
      ssm.MapDeviceSymbol(sname, "(*" + ename + ")");
      global_buffers.insert(buf_sym);
      event_global_buffers.push_back(buf_sym);
    } break;
    case Storage::SHARED:
    case Storage::LOCAL: {
      assert(!IsHost());
      auto mem_qual =
          (ety->GetStorage() == Storage::SHARED) ? "__shared__ " : "";
      ds << d_indent << mem_qual << "__volatile__ bool " << ename << "; // "
         << STR(ety->GetStorage()) << " event\n";
      if (ety->GetStorage() == Storage::SHARED)
        ds << d_indent << "if (__CHOREO_BLOCK_SINGLE__) {\n";
      ds << d_indent << (ety->GetStorage() == Storage::SHARED ? "  " : "")
         << ename << " = false;\n";
      if (ety->GetStorage() == Storage::SHARED) {
        ds << d_indent << "}\n";
        ds << d_indent << "__syncthreads();\n";
      }
      ssm.MapDeviceSymbolIfNotExist(sname, ename);
    } break;
    default: break;
    }
    return true;
  }

  return true;
}

// ============================================================================
// Control flow
// ============================================================================

bool HIPCodeGen::Visit(AST::ForeachBlock& n) {
  TraceEachVisit(n);

  int unroll_factor = 0;
  const bool has_unroll = AST::HasUnrollHint(n, unroll_factor);
  if (has_unroll) {
    if (unroll_factor > 0)
      IndStream() << "#pragma unroll " << unroll_factor << "\n";
    else
      IndStream() << "#pragma unroll\n";
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
  return true;
}

bool HIPCodeGen::Visit(AST::InThreadsBlock& n) {
  TraceEachVisit(n);
  assert(!IsHost());
  if (!n.stmts->None()) {
    auto pred_str = ExprSTR(n.pred, false);
    ds << d_indent << "if (" << pred_str << ") {"
       << (n.async ? " // inthreads.async" : "") << "\n";
  }
  IncrDeviceIndent();
  return true;
}

bool HIPCodeGen::Visit(AST::IfElseBlock& n) {
  TraceEachVisit(n);
  if (auto c = dyn_cast<AST::Call>(n.pred->GetReference()))
    IndStream() << "if (" << CallSTR(*c) << ") {\n";
  else
    IndStream() << "if (" << ExprSTR(n.pred, IsHost()) << ") {\n";
  IncrIndent();
  emit_call = true;
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
      os << indent << "choreo::choreo_assert("
         << ExprSTR(args->ValueAt(0), IsHost()) << ", \"assertion failed\");\n";
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
    if (func_name == "setreg" || func_name == "launch_bounds") { return true; }
    if (n.IsArith()) {         /* fall through to CallSTR emission below */
    } else if (n.IsAtomic()) { /* fall through to CallSTR emission below */
    } else
      choreo_unreachable("builtin '" + func_name +
                         "' not supported by amdgpu.");
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
    if (auto id = AST::GetIdentifier(*n.value)) {
      auto sym = id->name;
      auto vty = NodeType(*n.value);
      auto sty = dyn_cast<SpannedType>(vty);
      if (sty) {
        auto sname = InScopeName(sym);
        hs << h_indent << "choreo::abend_true(hipMemcpy(" << sym << ".data(), "
           << ssm.HostName(sname + "__device") << ", " << UnScopedSizeExpr(*sty)
           << ", hipMemcpyDeviceToHost));\n";
      }
    } else if (auto expr = dyn_cast<AST::Expr>(n.value);
               expr &&
               (expr->GetOp() == Op::DataOf || expr->GetOp() == Op::MDataOf)) {
      if (auto fid = cast<AST::Expr>(expr->GetR())->GetSymbol()) {
        auto vty = GetSymbolType(fid->name);
        auto sty = GetSpannedType(vty);
        if (sty) {
          auto buf_sym = fid->name + "__buf__";
          hs << h_indent << "choreo::abend_true(hipMemcpy(" << buf_sym
             << ".data(), " << buf_sym << "__device, " << UnScopedSizeExpr(*sty)
             << ", hipMemcpyDeviceToHost));\n";
        }
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
  for (auto& buf : event_global_buffers)
    hs << h_indent << "hipFree(" << buf << ");\n";
}

// ============================================================================
// ChunkAt subview element offset (linear index into base span)
// ============================================================================

std::string HIPCodeGen::GenOffset(const AST::ptr<AST::ChunkAt>& ca) const {
  if (!ca || ca->NoOperation()) return "0";

  sbe::ExprSum offset;

  for (size_t i = 0; i < ca->OpCount(); ++i) {
    const auto& sop = ca->OpAt(i);
    if (isa<AST::SOP::Reshape>(sop)) continue;

    if (isa<AST::SOP::Tiling>(sop) || isa<AST::SOP::TileAt>(sop) ||
        isa<AST::SOP::SubSpan>(sop)) {
      auto idx = sop->GetIndices()->Opts();
      auto strd = sop->GetBlockStrides();
      auto blk = sop->GetBlockShape();
      assert(idx.HasVals());
      assert(idx.GetVals().size() == strd.size());
      assert((size_t)blk.Rank() == strd.size());

      bool has_step = isa<AST::SOP::SubSpan>(sop) && sop->GetSteps();
      for (size_t dim = 0; dim < idx.GetVals().size(); ++dim) {
        if (has_step)
          offset += idx.GetVals()[dim] * strd[dim];
        else
          offset += idx.GetVals()[dim] * strd[dim] * blk.ValueAt(dim);
      }
    } else if (isa<AST::SOP::View>(sop)) {
      auto off = sop->GetOffsets()->Opts();
      auto strd = sop->GetBlockStrides();
      for (size_t dim = 0; dim < off.GetVals().size(); ++dim)
        offset += off.GetVals()[dim] * strd[dim];
    } else {
      choreo_unreachable("unsupported spanned operation in HIP GenOffset.");
    }
  }

  return ValueSTR(offset.Get());
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
    if (expr->IsUnary() && expr->op == Op::Cast) {
      return ExprSTR(expr->GetR(), is_host);
    }
    if (expr->IsUnary()) {
      return Choreo::STR(expr->op) + "(" + ExprSTR(expr->GetR(), is_host) + ")";
    }
    if (expr->op == Op::ElemOf) {
      return ExprSTR(expr->GetL(), is_host) + "[" +
             ExprSTR(expr->GetR(), is_host) + "]";
    }
    if (expr->IsBinary()) {
      return "(" + ExprSTR(expr->GetL(), is_host) + " " +
             Choreo::STR(expr->op) + " " + ExprSTR(expr->GetR(), is_host) + ")";
    }
    if (expr->GetOp() == Op::DataOf || expr->GetOp() == Op::MDataOf) {
      if (auto id = cast<AST::Expr>(expr->GetR())->GetSymbol()) {
        auto sname = InScopeName(id->name) + ".data";
        if (!is_host && ssm.HasDeviceName(sname)) return ssm.DeviceName(sname);
        return id->name;
      }
    }
    if (auto ref = expr->GetReference()) return ExprSTR(ref, is_host);
  }
  if (auto ca = dyn_cast<AST::ChunkAt>(n)) {
    auto sym_name = ca->RefSymbol();
    auto sname = InScopeName(sym_name);
    std::string base_expr;
    if (is_host && ssm.HasHostName(sname))
      base_expr = ssm.HostName(sname);
    else if (!is_host && ssm.HasDeviceName(sname))
      base_expr = ssm.DeviceName(sname);
    else
      base_expr = sym_name;

    if (ca->NoOperation()) return base_expr;

    auto offset_str = GenOffset(ca);
    if (offset_str == "0") return base_expr;

    auto sty = GetSpannedType(NodeType(*ca));
    assert(sty && "ChunkAt with span operations expects a SpannedType.");
    auto bt_name = HIPNameBaseType(sty->ElementType());
    return std::string("((") + bt_name + "*)(" + base_expr + ") + " +
           offset_str + ")";
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
  if (auto np = dyn_cast<AST::Nullptr>(n)) return "nullptr";
  if (auto bl = dyn_cast<AST::BoolLiteral>(n))
    return bl->value ? "true" : "false";
  if (auto sl = dyn_cast<AST::StringLiteral>(n)) return sl->EscapedVal();
  if (auto ii = dyn_cast<AST::IntIndex>(n)) return ExprSTR(ii->value, is_host);
  if (auto it = dyn_cast<AST::IntTuple>(n)) {
    std::string r = "{";
    int i = 0;
    for (auto& v : it->GetValues()->AllValues()) {
      if (i++ > 0) r += ", ";
      r += ExprSTR(v, is_host);
    }
    return r + "}";
  }
  if (auto ce = dyn_cast<AST::CastExpr>(n)) {
    if (ce->IsForeignCast())
      return "((" + ce->ForeignType() + ")" + ExprSTR(ce->GetR(), is_host) +
             ")";
    return std::string("static_cast<") + HIPNameBaseType(ce->ToType()) + ">(" +
           ExprSTR(ce->GetR(), is_host) + ")";
  }
  if (auto call = dyn_cast<AST::Call>(n)) return CallSTR(*call);
  if (auto mv = dyn_cast<AST::MultiValues>(n)) {
    if (mv->Count() == 1) return ExprSTR(mv->ValueAt(0), is_host);
  }

  choreo_unreachable("unsupported expression in HIP codegen: " + PSTR(n));
}

const std::string HIPCodeGen::OpExprSTR(AST::ptr<AST::Node> n,
                                        const std::string&, bool,
                                        bool is_host) const {
  return ExprSTR(n, is_host);
}

const std::string HIPCodeGen::CallSTR(AST::Call& n) const {
  if (n.IsAtomic()) {
    static const std::unordered_map<std::string, std::string> atomic_map = {
        {"__atomic_add", "atomicAdd"},   {"__atomic_sub", "atomicSub"},
        {"__atomic_exch", "atomicExch"}, {"__atomic_min", "atomicMin"},
        {"__atomic_max", "atomicMax"},   {"__atomic_and", "atomicAnd"},
        {"__atomic_or", "atomicOr"},     {"__atomic_xor", "atomicXor"},
        {"__atomic_cas", "atomicCAS"}};
    auto it = atomic_map.find(n.function->name);
    assert(it != atomic_map.end());
    std::string result = it->second + "(";
    size_t i = 0;
    for (auto& a : n.GetArguments()) {
      if (i > 0) result += ", ";
      if (i == 0)
        result += "&(" + OpExprSTR(a, "", true, IsHost()) + ")";
      else
        result += OpExprSTR(a, "", true, IsHost());
      ++i;
    }
    result += ")";
    return result;
  }

  if (n.IsArith()) {
    static const std::unordered_map<std::string, std::string> arith_map = {
        {"__sqrt", "sqrtf"},
        {"__rsqrt", "rsqrtf"},
        {"__exp", "expf"},
        {"__expm1", "expm1f"},
        {"__log", "logf"},
        {"__log1p", "log1pf"},
        {"__pow", "powf"},
        {"__sin", "sinf"},
        {"__cos", "cosf"},
        {"__tan", "tanf"},
        {"__asin", "asinf"},
        {"__acos", "acosf"},
        {"__atan", "atanf"},
        {"__atan2", "atan2f"},
        {"__sinh", "sinhf"},
        {"__cosh", "coshf"},
        {"__tanh", "tanhf"},
        {"__ceil", "ceilf"},
        {"__floor", "floorf"},
        {"__round", "roundf"},
        {"__isfinite", "isfinite"},
        {"__sign", "__fsignbit"},
        {"__gelu", "__gelu"},
        {"__sigmoid", "__sigmoid"},
        {"__softplus", "__softplus"},
    };
    auto it = arith_map.find(n.function->name);
    std::string func = (it != arith_map.end()) ? it->second : n.function->name;
    std::string result = func + "(";
    if (n.arguments) {
      for (size_t i = 0; i < n.arguments->Count(); i++) {
        if (i > 0) result += ", ";
        result += OpExprSTR(n.arguments->ValueAt(i), "", true, IsHost());
      }
    }
    result += ")";
    return result;
  }

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

const std::string HIPCodeGen::ValueSTR(const ValueItem& vi, bool, bool) const {
  if (!IsValidValueItem(vi)) choreo_unreachable("invalid value item.");
  if (auto iv = VIInt(vi))
    return std::to_string(*iv);
  else if (auto sv = VISym(vi)) {
    auto res = UnScopedExpr(SSMName(sv.value(), IsHost()));
    return res;
  } else if (auto bo = VIBop(vi)) {
    std::string op = OpCodeToStr(bo->GetOpCode());
    return "(" + ValueSTR(bo->GetLeft()) + " " + op + " " +
           ValueSTR(bo->GetRight()) + ")";
  } else if (auto uo = VIUop(vi)) {
    std::string op = OpCodeToStr(uo->GetOpCode());
    return op + "(" + ValueSTR(uo->GetOperand()) + ")";
  }
  return PSTR(vi);
}

const std::string HIPCodeGen::ValueSTR(const ValueList& vl, bool ll, bool shp,
                                       const std::string& sep) const {
  std::string result;
  for (size_t i = 0; i < vl.size(); i++) {
    if (i > 0) result += sep;
    result += ValueSTR(vl[i], ll, shp);
  }
  return result;
}

const std::string HIPCodeGen::ShapeSTR(const Shape& s, bool ll,
                                       const std::string& sep, BaseType) const {
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
  for (auto& code : code_segments) outs() << code << "\n";
}

void HIPCodeGen::EmitScript(std::ostream& os, const std::string& exe_fn) {
  auto filename = RemoveDirectoryPrefix(
      RemoveSuffix(OptionRegistry::GetInstance().GetInputFileName(), ".co"));

  os << R"script(#!/usr/bin/env bash

# Choreo-generated bash script to compile HIP code for AMD GPU
# ROCm/HSA requires sufficient locked memory; raise if permitted.
ulimit -l unlimited 2>/dev/null || true
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
  for (auto& code : code_segments) os << code << "\n";
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
     << build_path << " " << cc_file << " -o " << build_path << "/__choreo_hip_"
     << filename << ".o\n";
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
