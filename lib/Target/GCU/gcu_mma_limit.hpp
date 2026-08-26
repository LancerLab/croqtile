#ifndef __CHOREO_GCU_MMA_LIMIT_HPP__
#define __CHOREO_GCU_MMA_LIMIT_HPP__

#include "ast.hpp"
#include "context.hpp"
#include "lower_libcall.hpp"
#include "types.hpp"

namespace Choreo {

struct AcoreMMAConfig {
  BaseType lhs_type = BaseType::UNKNOWN;
  BaseType rhs_type = BaseType::UNKNOWN;
  BaseType acc_type = BaseType::UNKNOWN;
  int M = 0;
  int N = 0;
  int K = 0;
  AST::MMAOperation::ExecMethod method = AST::MMAOperation::ROW_COL;

  bool IsMK_KN() const { return method == AST::MMAOperation::ROW_COL; }
  bool IsMK_NK() const { return method == AST::MMAOperation::ROW_ROW; }
  AcoreMMAFormat Format() const {
    return IsMK_KN() ? AcoreMMAFormat::MK_KN : AcoreMMAFormat::MK_NK;
  }
};

inline bool IsValidAcoreMMAConfig(const AcoreMMAConfig& cfg) {
  auto bt = cfg.lhs_type;
  if (bt != BaseType::F16 && bt != BaseType::BF16 && bt != BaseType::F32 &&
      bt != BaseType::S8)
    return false;

  bool is_mk_kn = cfg.IsMK_KN();
  if (!is_mk_kn && !cfg.IsMK_NK()) return false;

  if (cfg.M > 0) {
    if (!IsAcoreSupportedStaticM(cfg.M)) {
      if (cfg.M % 64 != 0 || !IsAcoreSupportedDynamicM(bt)) return false;
    }
    auto [ka, na] = GetAcoreAlignments(cfg.M, bt, cfg.acc_type, cfg.Format());
    if (ka == 0) return false;
  }

  return true;
}

inline std::string AcoreMMAConfigStr(const AcoreMMAConfig& cfg) {
  std::string s = "M=" + std::to_string(cfg.M);
  s += " " + STR(cfg.lhs_type);
  s += cfg.IsMK_KN() ? " MK_KN" : " MK_NK";
  return s;
}

inline std::string AcoreFormatStr(AST::MMAOperation::ExecMethod m) {
  return (m == AST::MMAOperation::ROW_COL) ? "acore::MK_KN" : "acore::MK_NK";
}

inline std::string AcoreElemTypeStr(BaseType bt) {
  switch (bt) {
  case BaseType::F16: return "__fp16";
  case BaseType::BF16: return "__bf16";
  case BaseType::F32: return "float";
  case BaseType::S8: return "char";
  default: return "void";
  }
}

inline int AcoreVACCUsage(int M, int N) { return M * N / 32; }

constexpr int ACORE_VACC_BUDGET = 4096;
constexpr int ACORE_VAB_ALIGN = 256;
constexpr int ACORE_WS_SIZE = 2048;

} // end namespace Choreo

#endif // __CHOREO_GCU_MMA_LIMIT_HPP__
