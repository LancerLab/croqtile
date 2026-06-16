#ifndef __CHOREO_GCU_MMA_CODEGEN_HPP__
#define __CHOREO_GCU_MMA_CODEGEN_HPP__

// Acore-based MMA codegen for GCU.
//
// This file contains the software-stack-specific state and codegen helpers
// for lowering Choreo MMA operations to acore::matmul calls. If a future
// GCU architecture provides native MMA instructions, a separate codegen
// path can be added without modifying this file.

#include "ast.hpp"
#include "gcu_mma_limit.hpp"
#include "types.hpp"

#include <map>
#include <set>
#include <sstream>
#include <string>

namespace Choreo {

struct AcorePendingExec {
  std::string a_addr;
  std::string b_addr;
  std::string k_dim;
  std::string n_dim;
  int acc_flag = 0;
  int lt_flag = 0;
  bool valid = false;
};

struct AcoreAccumState {
  bool first_exec = true;
  int vab_off = 0;
  std::string ws_name;
  std::string stub_name;
  int static_M = 0;
  BaseType elem_type = BaseType::UNKNOWN;
  BaseType out_type = BaseType::UNKNOWN;
  AST::MMAOperation::ExecMethod method = AST::MMAOperation::ROW_COL;
  AcorePendingExec pending;
};

struct AcoreMMACodeGenState {
  std::map<std::string, AcoreAccumState> accum_states;
  std::map<std::string, std::string> frag_addr;
  std::set<std::string> emitted_stubs;
  std::string stub_code;
  int next_vab_off = 0;

  void Reset() {
    accum_states.clear();
    frag_addr.clear();
    next_vab_off = 0;
  }

  static std::string PtrTypeStr(BaseType bt) {
    switch (bt) {
    case BaseType::F16: return "__fp16";
    case BaseType::BF16: return "__bf16";
    case BaseType::F32: return "float";
    case BaseType::S8: return "char";
    default:
      choreo_unreachable("unsupported acore MMA element type: " + STR(bt));
    }
  }

  static std::string TypeTag(BaseType bt) {
    switch (bt) {
    case BaseType::F16: return "f16";
    case BaseType::BF16: return "bf16";
    case BaseType::F32: return "f32";
    case BaseType::S8: return "s8";
    default:
      choreo_unreachable("unsupported acore MMA element type: " + STR(bt));
    }
  }

  std::string GetOrEmitStub(int M, BaseType in_bt, BaseType out_bt,
                            AST::MMAOperation::ExecMethod method) {
    std::string in_ptr = PtrTypeStr(in_bt);
    std::string out_ptr = PtrTypeStr(out_bt);
    std::string fmt_str =
        (method == AST::MMAOperation::ROW_COL) ? "MK_KN" : "MK_NK";
    std::string stub_name = "__choreo_mma_" + TypeTag(in_bt) + "_" +
                            TypeTag(out_bt) + "_M" + std::to_string(M) + "_" +
                            fmt_str;

    if (emitted_stubs.count(stub_name)) return stub_name;
    emitted_stubs.insert(stub_name);

    std::string acore_fmt = (method == AST::MMAOperation::ROW_COL)
                                ? "acore::MK_KN"
                                : "acore::MK_NK";

    std::ostringstream s;
    s << "__device__ void " << stub_name << "(\n"
      << "    " << out_ptr << "* __restrict__ out,\n"
      << "    " << in_ptr << "* __restrict__ lhs,\n"
      << "    " << in_ptr << "* __restrict__ rhs,\n"
      << "    int* __restrict__ ws,\n"
      << "    int K, int N, int acc, int store, int vab_off, int lt) {\n"
      << "  acore::matmul<" << M << ", " << acore_fmt << ">(\n"
      << "      out, lhs, rhs, (" << in_ptr << "*)nullptr, ws,\n"
      << "      K, N, acc, store, 0, vab_off, lt);\n"
      << "}\n\n";

    stub_code += s.str();
    return stub_name;
  }
};

} // end namespace Choreo

#endif // __CHOREO_GCU_MMA_CODEGEN_HPP__
