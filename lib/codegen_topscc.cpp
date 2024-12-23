#include "codegen_topscc.hpp"

#include <filesystem>
#include <iostream>
#include <numeric>
#include <sstream>
#include <thread>

#include "ast.hpp"
#include "choreo_header.inc"
#include "codegen.hpp"
#include "types.hpp"

#ifndef __CHOREO_TOPSCC_DIR__
#error "missing macro definition of __CHOREO_TOPSCC_DIR__"
#endif

using namespace Choreo;
using namespace Choreo::Topscc;

extern Option<bool> native_f16;
extern Option<std::string> output;

namespace {

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

inline const std::string GetDTEContextName() {
  static unsigned i = 0;
  return "ctx" + std::to_string(i++);
}

} // namespace

bool TopsccCodeGen::BeforeVisitImpl(AST::Node& n) {
  if (trace_visit) dbgs() << "Before visiting " << n.TypeNameString() << "\n";

  if (isa<AST::Program>(&n)) {
    // emit the fixed headers
    EmitFixedHostHead();
    EmitFixedDeviceHead();
    ssm.EnterScope();
  } else if (isa<AST::ChoreoFunction>(&n)) {
    ResetChoreoFunctionStates();
    device_fn = "__choreo_device_" + fname;
    fty = cast<FunctionType>(GetSymbolType(fname));
    ssm.EnterScope();
  } else if (isa<AST::ParallelBy>(&n)) {
    if (parallel_level == 0)
      ds << d_indent << "// parallel-by: " << n.LOC() << "\n";
    parallel_level++;
  } else if (isa<AST::WithBlock>(&n)) {
    ds << d_indent << "// with-in: " << n.LOC() << "\n";
    ds << d_indent << "{\n";
    IncrDeviceIndent();
  } else if (isa<AST::ForeachBlock>(&n)) {
    ds << d_indent << "// foreach: " << n.LOC() << "\n";
  } else if (isa<AST::IncrementBlock>(&n)) {
    ds << d_indent << "// incr: " << n.LOC() << "\n";
    IncrDeviceIndent();
  }
  return true;
}

bool TopsccCodeGen::AfterVisitImpl(AST::Node& n) {
  if (trace_visit) dbgs() << "After visiting " << n.TypeNameString() << "\n";

  if (isa<AST::Program>(&n)) {
    ssm.LeaveScope();
    switch (CCtx().GetOutputKind()) {
    case OutputKind::TargetSourceCode: EmitSource(); break;
    case OutputKind::TargetModule:
      choreo_unreachable("topscc target module is yet to support.");
      break;
    case OutputKind::TargetExecutable: {
      choreo_unreachable("topscc target executable is yet to support.");
      break;
    }
    case OutputKind::ShellScript: {
      EmitScript();
      break;
    }
    default:
      choreo_unreachable("outputkind: " + STR(CCtx().GetOutputKind()) +
                         " is not supported.");
    }
  } else if (isa<AST::ChoreoFunction>(&n)) {
    ssm.LeaveScope();
    code_segments.back() += ds.str() + hs.str();
    ds.str(""); // reset the streams
    hs.str("");
  } else if (isa<AST::ParallelBy>(&n)) {
    parallel_level--;
  } else if (isa<AST::WithBlock>(&n)) {
    DecrDeviceIndent();
    ds << d_indent << "}\n";
  } else if (auto fb = dyn_cast<AST::ForeachBlock>(&n)) {
    const auto& ranges = fb->GetRangeNodes();
    for (int j = ranges->Count() - 1; j >= 0; --j) {
      auto rng = cast<AST::LoopRange>(ranges->ValueAt(j));
      auto cname = rng->IVName();
      for (auto iv_name : within_map.at(InScopeName(cname))) {
        DecrDeviceIndent();
        ds << d_indent << "} // " << UnScopedName(iv_name) << "\n";
      }
    }
  } else if (isa<AST::IncrementBlock>(&n)) {
    DecrDeviceIndent();
    ds << d_indent << "}\n";
  }
  return true;
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

// include the choreo header;
)";
  if (native_f16) oss << "#define NATIVE_F16_SUPPORT\n";
  oss << R"(#include "choreo.h"

using namespace choreo;

)";
  code_segments.push_back(oss.str()); // reset the host code
}

void TopsccCodeGen::EmitFixedDeviceHead() {}

bool TopsccCodeGen::Visit(AST::FunctionDecl& n) {
  TraceEachVisit(n);

  assert(n.name == fname && "incosistent in function names.");
  assert(isa<FunctionType>(n.GetType()) && "unexpected type.");

  auto HandleSymbolicDimensions = [this](const ptr<SpannedType>& sty,
                                         const std::string& hp_name,
                                         size_t hp_index) {
    size_t dim_index = 0;
    for (auto vi : sty->GetShape().Value()) {
      if (auto vale = dyn_cast<ValueExpr>(&vi)) { // the dimension is symbolic
        assert(PrefixedWith(*vale, "::" + fname + "::") &&
               "unexpected symbolic dimension name.");

        auto dim_expr = hp_name + ".shape()[" + std::to_string(dim_index) + "]";
        if (symbolic_dimensions.count(*vale) == 0)
          symbolic_dimensions[*vale] = {dim_expr, hp_index, dim_index};
      }
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
  for (auto& item : GetChoreoFuncIns()) {
    if (item.IsParameter()) {
      assert((int)host_pindex == item.p_index);
      item.host_name = GenHostParamName();
      ssm.MapHostSymbol(item.name, item.host_name);
      ssm.MapDeviceSymbol(item.name, UnScopedName(item.name));
      if (auto sty = dyn_cast<SpannedType>(item.type))
        HandleSymbolicDimensions(sty, item.host_name, host_pindex);
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
    hs << h_indent << "unsigned " << UnScopedName(item.first) << " = "
       << item.second.hsd_expr << ";\n";
    ssm.MapHostSymbol(item.first, UnScopedName(item.first));
  }

  // do not generate device function unless parallel-by exists
  if (NeedDeviceFunc()) {
    EmitDeviceFuncDecl(ds);
    ds << " {\n";
    IncrDeviceIndent();

    for (auto item : symbolic_dimensions)
      ssm.MapDeviceSymbol(item.first, UnScopedName(item.first));

    // map the choreo input to device memory
    for (auto& item : GetChoreoFuncIns()) {
      if (auto sty = dyn_cast<SpannedType>(item.type)) {
        // Only the globals are declared in host. The shareds/locals are
        // declared in device
        auto sym = UnScopedName(item.name);
        std::string bts = NameBaseType(sty->ElementType());
        auto buf_sym = sym + "__device";
        hs << h_indent << bts << " * " << buf_sym << " = nullptr;\n";
        hs << h_indent << "topsMalloc(&" << buf_sym << ", "
           << UnScopedSizeExpr(*sty) << ");\n";
        hs << h_indent << "topsMemcpy(" << buf_sym << ", "
           << ssm.HostName(item.name) << ".data(), " << UnScopedSizeExpr(*sty)
           << ", topsMemcpyHostToDevice);\n";
        ssm.MapHostSymbol(item.name + "__device", buf_sym);
      }
    }

    for (auto& item : GetDeviceFuncIns()) {
      // handle inputs mdspan decls
      auto sym = UnScopedName(item.name);
      if (auto sty = dyn_cast<SpannedType>(item.type)) {
        ds << d_indent << "tops::mdspan __mds_" << sym << "("
           << TopsMdsStorage(sty->GetStorage()) << ", " << sym << ", "
           << UnScopedExpr(RSTR(sty->GetShape())) << ");\n";
      }
    }
  }

  return true;
}

bool TopsccCodeGen::Visit(AST::ChoreoFunction& n) {
  TraceEachVisit(n);

  DecrHostIndent();
  hs << "}\n\n";

  if (NeedDeviceFunc()) {
    DecrDeviceIndent();
    ds << "}\n\n";
  }

  return true;
}

bool TopsccCodeGen::Visit(AST::NamedVariableDecl& n) {
  TraceEachVisit(n);

  auto nty = NodeType(n);
  auto sym = n.name_str;

  if (auto sty = dyn_cast<SpannedType>(nty)) {
    // globals are declared in host, while shareds/locals are declared in device
    auto shape = sty->GetShape();
    std::string bts{NameBaseType(sty->ElementType())};

    if (sty->GetStorage() == Storage::GLOBAL) {
      auto buf_sym = sym + "__device";
      if (!IsChoreoOutput(InScopeName(sym))) {
        if (!n.init_value) {
          hs << h_indent << bts << " * " << buf_sym << " = nullptr;\n";
          hs << h_indent << "topsMalloc(&" << buf_sym << ", "
             << UnScopedSizeExpr(*sty) << ");\n";
        } else {
          // support simple int literal initialization
          hs << h_indent << bts << " " << sym << "__init["
             << ElemCountExprOf(*sty) << "];\n";
          hs << h_indent << "memset(" << sym << "__init, " << PSTR(n.init_value)
             << ", sizeof(" << sym << "__init));\n";
          hs << h_indent << bts << " * " << buf_sym << "= nullptr;\n";
          hs << h_indent << "topsMalloc(&" << buf_sym << ", "
             << UnScopedSizeExpr(*sty) << ");\n";
          hs << h_indent << "topsMemcpy(" << sym << "__device, " << sym
             << "__init, " << UnScopedSizeExpr(*sty)
             << ", topsMemcpyHostToDevice);\n";
        }
      } else {
        hs << h_indent << "auto " << sym << " = choreo::make_spandata<" << bts
           << ", " << shape.Rank() << ">({" << UnScopedExpr(RSTR(shape))
           << "});\n";
        hs << h_indent << bts << " * " << buf_sym << " = nullptr;\n";
        hs << h_indent << "topsMalloc(&" << buf_sym << ", "
           << UnScopedSizeExpr(*sty) << ");\n";
      }
      ssm.MapHostSymbol(InScopeName(sym) + "__device", buf_sym);
      ssm.MapHostSymbol(InScopeName(sym), sym);
      ssm.MapDeviceSymbol(InScopeName(sym), sym);
    } else if (sty->GetStorage() == Storage::SHARED) {
      if (!IsChoreoOutput(InScopeName(sym))) {
        ds << d_indent << "__shared__ " << bts << " " << sym << "["
           << ElemCountExprOf(*sty) << "];\n";
        ds << d_indent << "tops::mdspan __mds_" << sym << "("
           << TopsMdsStorage(sty->GetStorage()) << ", " << sym << ", "
           << RSTR(sty->GetShape()) << ");\n";
        ssm.MapDeviceSymbol(InScopeName(sym), sym);
      }
    } else if (sty->GetStorage() == Storage::LOCAL) {
      if (!IsChoreoOutput(InScopeName(sym))) {
        ds << d_indent << "__local__ " << bts << " " << sym << "["
           << ElemCountExprOf(*sty) << "];\n";
        ds << d_indent << "tops::mdspan __mds_" << sym << "("
           << TopsMdsStorage(sty->GetStorage()) << ", " << sym << ", "
           << RSTR(sty->GetShape()) << ");\n";
        ssm.MapDeviceSymbol(InScopeName(sym), sym);
      }
    } else
      choreo_unreachable("unsupported storage type.");
  }

  return true;
}

bool TopsccCodeGen::Visit(AST::ParallelBy& n) {
  TraceEachVisit(n);

  if (parallel_level != 1) return true;

  auto& lconfig = cgi->GetFunctionLaunch(fname);
  hs << h_indent << "dim3 __" << fname << "_gdims(" << lconfig.grid_dim_x
     << ", " << lconfig.grid_dim_y << ", " << lconfig.grid_dim_z << ");\n";
  hs << h_indent << "dim3 __" << fname << "_bdims(" << lconfig.block_dim_x
     << ", " << lconfig.block_dim_y << ", " << lconfig.block_dim_z << ");\n";
  hs << h_indent << device_fn << "<<<__" << fname << "_gdims, __" << fname
     << "_bdims>>>(";

  size_t i = 0;
  for (auto& item : GetDeviceFuncIns()) {
    auto sname = item.name;
    if (isa<SpannedType>(item.type)) sname += "__device";
    hs << ((i++ == 0) ? "" : ", ") << ssm.HostName(sname);
  }
  for (auto item : symbolic_dimensions) {
    hs << ((i++ > 0) ? ", " : "");
    hs << UnScopedName(item.first);
  }

  hs << ");\n";

  return true;
}

bool TopsccCodeGen::Visit(AST::DMA& n) {
  TraceEachVisit(n);

  auto nty = NodeType(n);
  if (auto ph = dyn_cast<PlaceHolderType>(nty)) {
    assert(ph->Category() == TypeCategory::FUTURE);
    // TODO
    choreo_unreachable("placeholder is yet to support.");
    return true;
  }

  assert(isa<AST::ChunkAt>(n.from) && "Unexpected type for DMA's source.");
  assert(isa<AST::ChunkAt>(n.to) && "Unexpected type for DMA's destination.");

  auto fty = dyn_cast<FutureType>(nty);
  assert(fty && "Invalid type of DMA statement!");
  if (fty->IsAsync()) assert(!n.future.empty());

  // claim the date transfer engine
  auto dte_ctx = GetDTEContextName();
  ds << d_indent << "tops_dte_ctx_t " << dte_ctx << ";\n";
  ds << d_indent << "tops::dte_scope s_" << dte_ctx << "(" << dte_ctx << ");\n";

  auto f_ca = cast<AST::ChunkAt>(n.from);
  auto t_ca = cast<AST::ChunkAt>(n.to);
  auto f_sym = f_ca->data->name;
  auto t_sym = t_ca->data->name;
  auto f_sty = GetSpannedType(GetSymbolType(f_sym));
  auto t_sty = GetSpannedType(GetSymbolType(t_sym));

  std::string f_nm = "";
  std::string f_mds_name = "";
  const std::string temp_str = f_sym + "__buf__";
  if (!ssm.DeviceNameOrNull(temp_str).empty())
    f_nm = ssm.DeviceName(temp_str);
  else
    f_nm = ssm.DeviceName(InScopeName(f_sym));
  f_mds_name = GetMdsName(f_nm);

  auto t_nm = ssm.DeviceName(InScopeName(t_sym));
  auto t_mds_name = GetMdsName(t_nm);

  assert(f_sty && "can not retrieve data from 'from'.");
  assert(t_sty && "can not retrieve data from 'to'.");

  ssm.MapDeviceSymbol(t_nm, t_nm);

  if (n.operation == ".copy") {
    if (f_ca->positions == nullptr) {
      if (t_ca->positions == nullptr) {
        // no chunkat
        ds << d_indent
           << (fty->IsAsync() ? ("tops::event " + n.future + " = ") : "")
           << "tops::memcpy" << (fty->IsAsync() ? "_async" : "") << "("
           << dte_ctx << ", " << t_mds_name << ", " << f_mds_name << ");\n";
      } else {
        auto off_name = "__deslice_offset__" + t_nm + "_2_" + f_nm;
        ds << d_indent << "int " << off_name << "[] = {";
        size_t i = 0;
        auto shape = f_sty->GetShape();
        for (auto& p : t_ca->positions->AllValues()) {
          if (i != 0) ds << ", ";
          ds << "(int)(" << ExprSTR(p, false) << " * " << STR(shape.ValueAt(i))
             << ")";
          ++i;
        }
        ds << "};\n";
        ds << d_indent
           << (fty->IsAsync() ? ("tops::event " + n.future + " = ") : "")
           << "tops::deslice" << (fty->IsAsync() ? "_async" : "") << "("
           << dte_ctx << ", " << t_mds_name << ", " << f_mds_name << ", "
           << off_name << ");\n";
      }
    } else {
      auto off_name = "__slice_offset__" + f_nm + "_2_" + t_nm;
      ds << d_indent << "int " << off_name << "[] = {";
      size_t i = 0;
      auto shape = t_sty->GetShape();
      for (auto& p : f_ca->positions->AllValues()) {
        if (i != 0) ds << ", ";
        ds << "(int)(" << ExprSTR(p, false) << " * " << STR(shape.ValueAt(i))
           << ")";
        ++i;
      }
      ds << "};\n";
      ds << d_indent
         << (fty->IsAsync() ? ("tops::event " + n.future + " = ") : "")
         << "tops::slice" << (fty->IsAsync() ? "_async" : "") << "(" << dte_ctx
         << ", " << t_mds_name << ", " << f_mds_name << ", " << off_name
         << ");\n";
    }
  } else if (n.operation == ".pad") {
    auto pad_config = cast<PadConfig>(n.GetConfig());
    ds << d_indent << "int __pad_high_" << f_nm << "[] = {"
       << DelimitedString(pad_config->pad_high) << "};\n";
    ds << d_indent << "int __pad_low_" << f_nm << "[] = {"
       << DelimitedString(pad_config->pad_low) << "};\n";
    ds << d_indent << "int __pad_mid_" << f_nm << "[] = {"
       << DelimitedString(pad_config->pad_mid) << "};\n";
    if (f_ca->positions == nullptr) {
      ds << d_indent
         << (fty->IsAsync() ? ("tops::event " + n.future + " = ") : "")
         << "tops::pad" << (fty->IsAsync() ? "_async" : "") << "(" << dte_ctx
         << ", __mds_" << t_nm << ", __mds_" << f_nm << ", __pad_low_" << f_nm
         << ", __pad_high_" << f_nm << ", __pad_mid_" << f_nm << ", "
         << pad_config->value.v << ");\n";
    } else {
      assert(false && "unsupported");
      // TODO: shall we support slice_pad?
    }
  } else if (n.operation == ".transp") {
    auto transp_config = cast<TransposeConfig>(n.GetConfig());
    ds << d_indent << "int __transpose_layout_" << f_nm << "[] = {"
       << DelimitedString(transp_config->dim_values) << "};\n";
    ds << d_indent
       << (fty->IsAsync() ? ("tops::event " + n.future + " = ") : "")
       << "tops::transpose" << (fty->IsAsync() ? "_async" : "") << "("
       << dte_ctx << ", " << t_mds_name << ", " << f_mds_name
       << ", __transpose_layout_" << f_nm << ");\n";
  }

  return true;
}

bool TopsccCodeGen::Visit(AST::Wait& n) {
  TraceEachVisit(n);

  for (auto& f : n.GetFutures())
    ds << d_indent << "tops::wait(" << ExprSTR(f, false) << ");\n";

  return true;
}

bool TopsccCodeGen::Visit(AST::Call& n) {
  TraceEachVisit(n);

  ds << d_indent << n.function->name;

  // emit template arguments
  if (n.template_args) {
    ds << "<";
    size_t i = 0;
    for (auto& ta : n.template_args->AllValues())
      ds << ((i++ == 0) ? "" : ", ") << ExprSTR(ta, false);
    ds << ">";
  }

  ds << "(";
  size_t i = 0;
  for (auto& a : n.GetArguments())
    ds << ((i++ == 0) ? "" : ", ") << ExprSTR(a, false);
  ds << ");\n";

  return true;
}

bool TopsccCodeGen::Visit(AST::WithIn& n) {
  TraceEachVisit(n);

  ssm.MapDeviceSymbol(InScopeName(n.with->name), "__iv_" + n.with->name);

  for (auto& v : n.GetMatchers()) {
    auto id = cast<AST::Identifier>(v);
    ssm.MapDeviceSymbol(InScopeName(id->name), "__iv_" + id->name);
    ds << d_indent << "int __iv_" << id->name << " = 0;\n";
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

  const auto& ranges = n.GetRangeNodes();
  for (int j = ranges->Count() - 1; j >= 0; --j) {
    auto rng = cast<AST::LoopRange>(ranges->ValueAt(j));
    auto cname = rng->IVName();
    for (auto iv_name : within_map.at(InScopeName(cname))) {
      auto iv_ty = GetSymbolType(UnScopedName(iv_name));
      assert(IsActualBoundedIntegerType(iv_ty));
      auto iv_bty = cast<BoundedType>(iv_ty);
      ds << d_indent << "for (" << ssm.DeviceName(iv_name) << " = "
         << (IsValidBound(rng->lbound) ? ("(" + STR(rng->lbound) + ")") : "0")
         << "; " << ssm.DeviceName(iv_name) << " < "
         << UnScopedExpr(STR(iv_bty->GetUpperBound())) << "; ++"
         << ssm.DeviceName(iv_name) << ") {\n";
      IncrDeviceIndent();
    }
  }
  return true;
}

bool TopsccCodeGen::Visit(AST::Return& n) {
  TraceEachVisit(n);

  auto vty = NodeType(*n.value);
  if (isa<ScalarType>(vty)) {
    hs << h_indent << "return " << ExprSTR(n.value, true) << ";\n";
    return true;
  } else if (auto id = AST::GetIdentifier(*n.value)) {
    auto sym = id->name;
    if (IsChoreoInput(InScopeName(sym))) {
      // return the parameter
      hs << h_indent << "return " << ExprSTR(n.value, true) << ";\n";
      return true;
    } else if (IsChoreoOutput(InScopeName(sym))) {
      if (auto sty = dyn_cast<SpannedType>(GetSymbolType(sym))) {
        // return the global storage, must map back
        hs << h_indent << "topsMemcpy(" << sym << ".data(), " << sym
           << "__device, " << UnScopedSizeExpr(*sty)
           << ", topsMemcpyDeviceToHost);\n";
      }
    }
  }

  assert(isa<SpannedType>(vty) && "expect a spanned data.");
  hs << h_indent << "return " << ExprSTR(n.value, true) << ";\n";
  return true;
}

bool TopsccCodeGen::Visit(AST::CppSourceCode& n) {
  TraceEachVisit(n);

  CodeSegment cur_cs = (n.host) ? CS_USER : CS_COK;
  if (cur_cs != cs) { code_segments.push_back(""); }

  // append the content
  code_segments.back() += n.GetCode();

  return true;
}

void TopsccCodeGen::EmitHostFuncDecl(std::ostringstream& oss) {
  // handle the return type
  if (!void_return && cgi->HasReturnSymbol(fname)) {
    auto& item = cgi->GetReturnDetail(fname);
    if (item.rty_str != "$")
      oss << item.rty_str;
    else
      oss << HostTypeStringify(*fty->out_ty, true);
  } else
    oss << "void";
  oss << " " << fname << "(";

  // emit the parameters
  size_t host_pindex = 0;
  for (auto& item : GetChoreoFuncIns()) {
    if (item.IsParameter()) assert((int)host_pindex == item.p_index);
    oss << ((host_pindex == 0) ? "" : ", ") << HostTypeStringify(*item.type)
        << " " << item.host_name;
    ++host_pindex;
  }
  oss << ")";

  VST_DEBUG(dbgs() << "Host function prototype:\n" << oss.str());
}

static inline const std::string
DeviceParamTypeStringify(const Choreo::Type& ty) {
  if (isa<VoidType>(&ty))
    return "void";
  else if (isa<IntegerType>(&ty))
    return "int";
  else if (isa<BooleanType>(&ty))
    return "bool";
  else if (auto sty = dyn_cast<SpannedType>(&ty)) {
    return std::string(NameBaseType(sty->ElementType())) + " *";
  } else
    choreo_unreachable("unsupported host function type.");
  return "";
}

void TopsccCodeGen::EmitDeviceFuncDecl(std::ostringstream& oss) {
  oss << "__global__ void " << device_fn << "(";

  size_t index = 0;
  for (auto& item : GetDeviceFuncIns()) {
    oss << ((index++ > 0) ? ", " : "");
    oss << DeviceParamTypeStringify(*item.type) << " ";
    oss << UnScopedName(item.name);
  }

  for (auto item : symbolic_dimensions) {
    oss << ((index++ > 0) ? ", unsigned " : "unsigned ");
    oss << UnScopedName(item.first);
  }

  oss << ")";

  VST_DEBUG(dbgs() << "Device function prototype:\n" << oss.str());
}

void TopsccCodeGen::EmitSource() {
  for (auto& code : code_segments) outs() << code << "\n";
}

void TopsccCodeGen::EmitScript() {
  auto filename = RemoveDirectoryPrefix(
      RemoveSuffix(OptionRegistry::GetInstance().GetInputFileName(), ".co"));
  outs() << "#!/usr/bin/env bash\n\n";
  outs() << "# This is the choreo generated bash script to compile factor "
            "code\n\n";

  outs() << "TOPSCC_INSTALL=" << STRINGIZE(__CHOREO_TOPSCC_DIR__) << "\n";
  outs() << "TOPSCC=${TOPSCC_INSTALL}/bin/topscc\n";
  outs() << "TOPSCC_LIB=${TOPSCC_INSTALL}/lib\n\n";

  auto build_path = CreateUniquePath();
  auto cc_file = build_path + "/__choreo_topscc_" + filename + ".cpp";
  auto exe_file = build_path + "/__choreo_topscc_" + filename + ".exe";
  auto fb_file = build_path + "/__choreo_topscc_" + filename + ".fb";
  outs() << "rm -fr " << build_path << "\n";
  outs() << "mkdir -p " << build_path << "\n\n";

  // place the choreo header
  outs() << "cat <<'EOF' > " << build_path << "/choreo.h\n";
  outs() << __choreo_header_as_string << "\nEOF\n\n";

  outs() << "cat <<'EOF' > " << cc_file << "\n";
  for (auto& code : code_segments) outs() << code << "\n";
  outs() << "\nEOF\n\n";

  // JIT: detect the environment
  outs() << R"script(
# check the device just-in-time
# TODO: improve the target check with more solid code
GCU_DEVICE_STR="$(lspci | grep Enflame | head -1)"
echo $GCU_DEVICE_STR
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

  outs() << R"script(
show_usage() {
  echo "  Usage: $0 | --execute           -> compile and execute choreo in factor"
  echo "                | --compile-binary    -> compile and generate the binary code"
  echo "                | --compile-fatbin    -> compile and generate the fatbin"
  exit 1
}

# compile, execute
)script";

  outs() << R"(export CFLAGS="-arch ${gcu_arch} -std=c++17 -ltops -lm")";
  outs() << "\nexport LD_LIBRARY_PATH=${TOPSCC_LIB}:${LD_LIBRARY_PATH}\n\n";
  outs() << R"(if [ "$1" == "--execute" ] || [ "$#" -eq 0 ]; then)";
  outs() << "\n  ${TOPSCC} ${CFLAGS} " << cc_file << " -o " << exe_file;
  outs() << "\n  " << exe_file << "\n";
  outs() << R"(elif [ "$1" == "--compile-binary" ]; then)";
  outs() << "\n  ${TOPSCC} -c ${CFLAGS} " << cc_file << " -o " << exe_file
         << "\n";
  outs() << R"(elif [ "$1" == "--compile-fatbin" ]; then)";
  // TODO: figure out the option
  outs() << "\n  ${TOPSCC} -c ${CFLAGS} " << cc_file << " -o " << fb_file
         << "\n";
  outs() << "else show_usage\n";
  outs() << "fi";
}

// TODO: eliminate the need of the value replacement?
const std::string TopsccCodeGen::ValueSTR(const ValueItem& vi) const {
#if 0
  if (auto i = dyn_cast<int>(&vi)) {
    return std::to_string(*i);
  } else if (factor_value) {
    // not int => this is a dynamic var or var bounded by dynamic var.
    return ReplaceFactorDynDimName(STR(vi));
  } else {
    return ReplaceRuntimeNames(STR(vi), "", false);
  }
#endif
  return STR(vi);
}

const std::string TopsccCodeGen::ExprSTR(AST::ptr<AST::Node> e,
                                         bool is_host) const {
  std::ostringstream oss;

  if (auto id = dyn_cast<AST::Identifier>(e)) {
    auto ty = NodeType(*id);
    if (isa<BoundedType>(ty) &&
        PrefixedWith(cast<BoundedType>(ty)->GetNote(), "pv")) {
      auto l = RemovePrefixOrNull("pv:", cast<BoundedType>(ty)->GetNote());
      assert(l.has_value());
      // is marked as parallel whose level is decided by target check
      if (*l == "0")
        oss << "__tops_tid_x()";
      else if (*l == "1")
        oss << "__tops_bid_x()";
      else
        choreo_unreachable("invalid bounded type note.");
    } else if (within_map.count(InScopeName(id->name)) && !is_host) {
      size_t i = 0;
      for (auto iv_name : within_map.at(InScopeName(id->name)))
        oss << ((i++ == 0) ? "" : ", ")
            << UnScopedName(ssm.DeviceName(iv_name));
    } else
      oss << UnScopedName(((is_host) ? ssm.HostName(InScopeName(id->name))
                                     : ssm.DeviceName(InScopeName(id->name))));
  } else if (auto il = dyn_cast<AST::IntLiteral>(e)) {
    oss << "(" << il->value << ")";
  } else if (auto ii = dyn_cast<AST::IntIndex>(e)) {
    return ExprSTR(ii->value, is_host);
  } else if (auto expr = dyn_cast<AST::Expr>(e)) {
    // utilize the optimize value whenever possible
    if (auto sym = expr->GetSymbol()) {
      auto sname = InScopeName(sym->name);
      if (FCtx(fname).HasSymbolValues(sname)) {
        auto svs = FCtx(fname).GetSymbolValues(sname);
        if (IsValidValueItem(svs.int_expr))
          return "(" + STR(svs.int_expr) + ")";
      }
    }
    if (ConvertibleToInt(NodeType(*e))) {
      if (IsValidValueItem(expr->opt_vals.int_expr)) {
        return "(" + STR(expr->opt_vals.int_expr) + ")";
      }
    }
    if (expr->IsReference()) {
      if (expr->GetInt())
        return ExprSTR(expr->GetReference(), is_host);
      else if (expr->GetSymbol())
        return ExprSTR(expr->GetReference(), is_host);
      else if (isa<AST::Expr>(NodeType(*expr->GetR()))) // should this happen?
        return ExprSTR(expr->GetR(), is_host);
      else
        choreo_unreachable("Unsupported reference: " + PSTR(expr));
    } else if (expr->IsUnary()) {
      if (expr->op == "!") {
        oss << "!(" << ExprSTR(expr->GetR(), is_host) << ")";
      } else if (expr->op == "ubound") {
        auto rty = cast<BoundedType>(NodeType(*expr->GetR()));
        // anchor
        if (rty->Dims() == 1) { oss << ValueSTR(rty->GetUpperBound()); }
      } else if (expr->op == "dataof") {
        assert(isa<FutureType>(expr->GetR()->GetType()) &&
               "expect a future operand.");
        if (auto id = cast<AST::Expr>(expr->GetR())->GetSymbol()) {
          if (FBInfo().count(InScopeName(id->name)))
            oss << UnScopedName(FBInfo().at(InScopeName(id->name)).buffer);
          else
            choreo_unreachable("Future '" + id->name +
                               "' is not associated with a buffer.");
        } else
          choreo_unreachable("Can not retrive name of the future.");
      } else if (expr->op == "sizeof") {
        auto var = RemoveSuffix(*AST::GetName(*expr->GetR()), ".span");
        auto shape = GetShape(GetSymbolType(var));
        assert(shape.IsValid() && "Invalid shape is found");
        oss << shape.GetSizeExpression();
      } else
        choreo_unreachable("Unsupported choreo expression.");
    } else if (expr->IsBinary()) {
      if (expr->op == "cdiv") {
        std::string one = "1";
        oss << "((" << ExprSTR(expr->GetL(), is_host) << ")+("
            << ExprSTR(expr->GetR(), is_host) << "-" << one << ")/("
            << ExprSTR(expr->GetR(), is_host) << ")";
      } else if (expr->op == "getith") {
        auto lty = cast<BoundedType>(NodeType(*expr->GetL()));
        if (cast<AST::IntIndex>(expr->GetR())->IsNegative()) {
          oss << "(";
          oss << ValueSTR(lty->GetUpperBound());
          oss << "+(" << ExprSTR(expr->GetR(), is_host) << "))";
        } else
          oss << "(" << ExprSTR(expr->GetR(), is_host) << ")";
      } else if (expr->IsArith() || expr->IsLogical()) {
        auto& l = expr->GetL();
        auto& r = expr->GetR();
        auto& op = expr->op;
        // handle bounded variable times
        if (op == "#" && IsActualBoundedIntegerType(l->GetType()) &&
            IsActualBoundedIntegerType(r->GetType())) {
          auto rty = cast<BoundedType>(NodeType(*r));
          assert(rty->Dims() == 1);
          oss << "((" << ExprSTR(l, is_host) << ")*("
              << ValueSTR(rty->GetUpperBound()) << ")+(" << ExprSTR(r, is_host)
              << "))";
        } else
          oss << "((" << ExprSTR(l, is_host) << ")" << op << "("
              << ExprSTR(r, is_host) << "))";
      }
    } else if (expr->IsTernary()) {
      oss << "(" << ExprSTR(expr->GetC(), is_host) << ") ? ("
          << ExprSTR(expr->GetL(), is_host) << ") : ("
          << ExprSTR(expr->GetR(), is_host) << ")";
    } else
      choreo_unreachable("unsupported expression '" + expr->op +
                         "': " + PSTR(expr) + ".");
  } else if (auto sl = dyn_cast<AST::Select>(e)) {
    size_t val_count = sl->expr_list->Count();
    // if val_count == 1, pingpong is meaningless?
    // (TODO: maybe assert when earlysema)
    assert(val_count >= 2);
    for (size_t i = 0; i < val_count - 1; i++) {
      oss << "select_(" << ExprSTR(sl->select_factor, is_host) << " == ";
      oss << "(" << i << ")";
      oss << ", " << PSTR(sl->expr_list->ValueAt(i))
          << (i < val_count - 1 ? ", " : "");
    }
    oss << PSTR(sl->expr_list->AllValues().back())
        << std::string(val_count - 1, ')');
  } else
    choreo_unreachable("unsupported expression '" + expr->op + "'.");

  return oss.str();
}
