#include "codegen_topscc.hpp"

#include <filesystem>
#include <iostream>
#include <numeric>
#include <sstream>
#include <thread>

#include "ast.hpp"
// #include "choreo_topscc_header.inc"
#include "codegen.hpp"
// #include "topscc_script.inc"
#include "types.hpp"

#ifndef __CHOREO_TOPSCC_DIR__
#error "missing macro definition of __CHOREO_TOPSCC_DIR__"
#endif

using namespace Choreo;
using namespace Choreo::Topscc;

extern Option<bool> native_f16;
extern Option<std::string> output;

bool TopsccCodeGen::BeforeVisitImpl(AST::Node& n) {
  TraceEachVisit(n);

  if (isa<AST::Program>(&n)) {
    // emit the fixed headers
    EmitFixedHostHead();
    EmitFixedDeviceHead();
  } else if (isa<AST::ChoreoFunction>(&n)) {
    ResetChoreoFunctionStates();
    device_fn = "__choreo_device_" + fname;
    fty = cast<FunctionType>(GetSymbolType(fname));
  } else if (isa<AST::ParallelBy>(&n)) {
    parallel_level++;
  }
  return 0;
}

bool TopsccCodeGen::AfterVisitImpl(AST::Node& n) {
  TraceEachVisit(n);

  if (isa<AST::Program>(&n)) {
    code_segments.back() = ds.str() + "\n" + hs.str();
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
      choreo_unreachable("topscc target script is yet to support.");
      break;
    }
    default:
      choreo_unreachable("outputkind: " + STR(CCtx().GetOutputKind()) +
                         " is not supported.");
    }
  } else if (isa<AST::ChoreoFunction>(&n)) {
  } else if (isa<AST::ParallelBy>(&n)) {
    parallel_level--;
  }
  return 0;
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

// include the choreo header\n";
)";
  if (native_f16) oss << "#define NATIVE_F16_SUPPORT\n";
  oss << R"(#include "choreo.h"

using namespace choreo;

} // end anonymous namespace
)";
  code_segments.push_back(oss.str()); // reset the host code
}

void TopsccCodeGen::EmitFixedDeviceHead() {}

bool TopsccCodeGen::Visit(AST::FunctionDecl& d) {
  TraceEachVisit(d);
  assert(d.name == fname && "incosistent in function names.");
  assert(isa<FunctionType>(d.GetType()) && "unexpected type.");

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

        //        idnm_rts.emplace(FineName(UnScopedName(*vale)), *vale);
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
  EmitDeviceFuncDecl(ds);

  hs << " {\n";
  IncrHostIndent();
  ds << " {\n";
  IncrDeviceIndent();

  return true;
}

bool TopsccCodeGen::Visit(AST::ChoreoFunction& n) {
  DecrHostIndent();
  DecrDeviceIndent();
  hs << "\n}\n";
  ds << "\n}\n";

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

inline const char* NameBaseType(BaseType ft) {
  switch (ft) {
  case BaseType::F32: return "float";
  case BaseType::F16: return "__fp16";
  case BaseType::BF16: return "__bf16";
  case BaseType::U32: return "unsigned int";
  case BaseType::U16: return "unsigned short";
  case BaseType::U8: return "unsigned char";
  case BaseType::S32: return "int";
  case BaseType::S16: return "short";
  case BaseType::S8: return "char";
  default: choreo_unreachable("unsupported base-type.");
  }
  return "";
}

static inline const std::string
DeviceParamTypeStringify(const Choreo::Type& ty) {
  if (isa<VoidType>(&ty))
    return "void";
  else if (isa<IntegerType>(&ty))
    return "int";
  else if (isa<BooleanType>(&ty))
    return "bool";
  else if (auto sty = dyn_cast<SpannedType>(&ty))
    return std::string(NameBaseType(sty->ElementType())) + " *";
  else
    choreo_unreachable("unsupported host function type.");
  return "";
}

void TopsccCodeGen::EmitDeviceFuncDecl(std::ostringstream& oss) {
  oss << "__global__ void " << device_fn << "(";

  size_t index = 0;
  for (auto& item : GetDeviceFuncIns()) {
    //    assert(item.d_index == (int)index);
    oss << ((index++ > 0) ? ", " : "");
    oss << DeviceParamTypeStringify(*item.type) << " ";
    oss << UnScopedName(item.name);
  }
  oss << ")";

  VST_DEBUG(dbgs() << "Device function prototype:\n" << oss.str());
}

void TopsccCodeGen::EmitSource() {
  auto ifname = OptionRegistry::GetInstance().GetInputFileName();

  for (auto& code : code_segments) outs() << code << "\n";
}
