#ifndef __CHOREO_GCU_LOWER_LIBCALL_HPP__
#define __CHOREO_GCU_LOWER_LIBCALL_HPP__

#include "ast.hpp"
#include "aux.hpp"
#include "context.hpp"
#include "types.hpp"
#include "visitor.hpp"

namespace Choreo {

// ============================================================================
// Acore operation classification (kept from original acore_check)
// ============================================================================
enum class AcoreOpCategory {
  MATMUL,
  ADDMM,
  DOT_BIAS_QUANT,
  BINARY,
  UNARY,
  ACTIVATION,
  REDUCE,
  NORM,
  CONV2D,
  CONVERT,
  POINTWISE,
  UNKNOWN
};

inline AcoreOpCategory ClassifyAcoreOp(const std::string& name) {
  if (name == "matmul" || name == "inline_matmul")
    return AcoreOpCategory::MATMUL;
  if (name == "addmm") return AcoreOpCategory::ADDMM;
  if (name == "dot_bias_quant") return AcoreOpCategory::DOT_BIAS_QUANT;

  if (name == "add" || name == "add_relu" || name == "sub" || name == "rsub" ||
      name == "mul" || name == "div" || name == "rdiv" ||
      name == "floor_divide" || name == "trunc_divide" || name == "max" ||
      name == "min" || name == "clamp" || name == "pow" || name == "atan2" ||
      name == "fmod" || name == "remainder" || name == "gt" || name == "ge" ||
      name == "lt" || name == "le" || name == "eq" || name == "ne" ||
      name == "logical_and" || name == "logical_or" || name == "logical_xor")
    return AcoreOpCategory::BINARY;

  if (name == "abs" || name == "neg" || name == "sign" || name == "ceil" ||
      name == "floor" || name == "trunc" || name == "round_afz" ||
      name == "round_tz" || name == "round_ne" || name == "bitwise_not" ||
      name == "logical_not" || name == "is_nan" || name == "is_inf" ||
      name == "is_pos_inf" || name == "is_neg_inf" || name == "is_finite" ||
      name == "signbit" || name == "sqrt" || name == "rsqrt" ||
      name == "cbrt" || name == "reciprocal" || name == "exp" ||
      name == "log" || name == "erf" || name == "erfinv" || name == "erfc" ||
      name == "gelu_erf" || name == "acos" || name == "acosh" ||
      name == "asin" || name == "asinh" || name == "atan" || name == "atanh" ||
      name == "cos" || name == "cosh" || name == "sin" || name == "sinh" ||
      name == "tan" || name == "frac" || name == "popcnt" || name == "sinc")
    return AcoreOpCategory::UNARY;

  if (name == "relu" || name == "gelu" || name == "selu" ||
      name == "hard_swish" || name == "mish" || name == "quick_gelu" ||
      name == "sigmoid" || name == "softplus" || name == "silu" ||
      name == "swish" || name == "tanh" || name == "log_sigmoid" ||
      name == "clipped_relu" || name == "threshold" || name == "elu" ||
      name == "celu" || name == "hard_shrink" || name == "hard_sigmoid" ||
      name == "hard_tanh" || name == "leaky_relu" || name == "soft_shrink" ||
      name == "logit" || name == "glu" || name == "gtu" || name == "geglu" ||
      name == "reglu" || name == "swiglu" || name == "glu_jvp" ||
      name == "log_softmax" || name == "prelu" || name == "rrelu_with_noise")
    return AcoreOpCategory::ACTIVATION;

  if (name == "reduce_sum" || name == "reduce_prod" || name == "reduce_mean" ||
      name == "reduce_max" || name == "reduce_min" || name == "reduce_argmax" ||
      name == "reduce_argmin" || name == "reduce_aminmax" ||
      name == "reduce_any" || name == "reduce_all" ||
      name == "reduce_sum_clamp" || name == "reduce_prod_clamp" ||
      name == "reduce_mean_clamp" || name == "logsumexp" ||
      name == "nan_mean" || name == "dist")
    return AcoreOpCategory::REDUCE;

  if (name == "layer_norm" || name == "instance_norm" ||
      name == "layer_norm_fp16mix" || name == "layer_norm_bf16mix" ||
      name == "layer_norm_fp32")
    return AcoreOpCategory::NORM;

  if (PrefixedWith(name, "conv2d")) return AcoreOpCategory::CONV2D;
  if (name == "convert") return AcoreOpCategory::CONVERT;

  if (name == "where" || name == "conj" || name == "neg_conj" ||
      name == "angle" || name == "polar" || name == "lerp")
    return AcoreOpCategory::POINTWISE;

  return AcoreOpCategory::UNKNOWN;
}

inline std::string GetAcoreFuncName(const std::string& full_name) {
  const std::string prefix = "acore::";
  if (!PrefixedWith(full_name, prefix)) return "";
  return full_name.substr(prefix.size());
}

// ============================================================================
// Acore matmul supported-M table (from extern/include/common/matmul.h)
// ============================================================================
// MMA memory layout formats accepted by acore::matmul.
enum class AcoreMMAFormat { MK_KN, MK_NK };

struct AcoreMatmulPattern {
  int M;                    // static-M value, or 0 for the dynamic (64-aligned) overload
  AcoreMMAFormat format;    // MK_KN (ROW_COL) or MK_NK (ROW_ROW)
  BaseType lhs_type;        // input (and rhs) element type
  BaseType out_type;        // accumulator/output element type
  int K_align;
  int N_align;
};

// Every (M, format, lhs, out) tuple accepted by the `acore::matmul` entry
// points, transcribed from extern/include/common/matmul.h.  The M=512/1024
// rows in matmul.h belong to `acore::addmm`, NOT `acore::matmul`, so they are
// deliberately absent here; such sizes are instead served by the 64-aligned
// dynamic-M overload.
static const AcoreMatmulPattern kAcoreMatmulPatterns[] = {
    // M=1
    {1, AcoreMMAFormat::MK_NK, BaseType::F16, BaseType::F16, 32, 128},
    {1, AcoreMMAFormat::MK_NK, BaseType::BF16, BaseType::BF16, 32, 128},
    {1, AcoreMMAFormat::MK_KN, BaseType::F16, BaseType::F16, 32, 128},
    {1, AcoreMMAFormat::MK_KN, BaseType::BF16, BaseType::BF16, 32, 128},
    {1, AcoreMMAFormat::MK_KN, BaseType::F16, BaseType::F32, 32, 128},
    {1, AcoreMMAFormat::MK_KN, BaseType::BF16, BaseType::F32, 32, 128},
    {1, AcoreMMAFormat::MK_KN, BaseType::F32, BaseType::F32, 16, 64},
    // M=16
    {16, AcoreMMAFormat::MK_KN, BaseType::F32, BaseType::F32, 16, 64},
    {16, AcoreMMAFormat::MK_NK, BaseType::F32, BaseType::F32, 16, 64},
    // M=32
    {32, AcoreMMAFormat::MK_KN, BaseType::F16, BaseType::F16, 32, 128},
    {32, AcoreMMAFormat::MK_KN, BaseType::F16, BaseType::F16, 64, 64},
    {32, AcoreMMAFormat::MK_KN, BaseType::BF16, BaseType::BF16, 32, 128},
    {32, AcoreMMAFormat::MK_KN, BaseType::BF16, BaseType::BF16, 32, 64},
    {32, AcoreMMAFormat::MK_KN, BaseType::F16, BaseType::F32, 64, 64},
    {32, AcoreMMAFormat::MK_KN, BaseType::F16, BaseType::F32, 32, 128},
    {32, AcoreMMAFormat::MK_KN, BaseType::BF16, BaseType::F32, 32, 64},
    {32, AcoreMMAFormat::MK_KN, BaseType::BF16, BaseType::F32, 32, 128},
    {32, AcoreMMAFormat::MK_KN, BaseType::F32, BaseType::F32, 16, 32},
    {32, AcoreMMAFormat::MK_NK, BaseType::F16, BaseType::F16, 32, 128},
    {32, AcoreMMAFormat::MK_NK, BaseType::F16, BaseType::F16, 64, 64},
    {32, AcoreMMAFormat::MK_NK, BaseType::BF16, BaseType::BF16, 32, 128},
    {32, AcoreMMAFormat::MK_NK, BaseType::BF16, BaseType::BF16, 32, 64},
    {32, AcoreMMAFormat::MK_NK, BaseType::F16, BaseType::F32, 64, 64},
    {32, AcoreMMAFormat::MK_NK, BaseType::BF16, BaseType::F32, 32, 64},
    {32, AcoreMMAFormat::MK_NK, BaseType::F32, BaseType::F32, 16, 32},
    // M=49 (MK_NK only)
    {49, AcoreMMAFormat::MK_NK, BaseType::F16, BaseType::F16, 64, 64},
    // M=64
    {64, AcoreMMAFormat::MK_KN, BaseType::F16, BaseType::F16, 32, 128},
    {64, AcoreMMAFormat::MK_KN, BaseType::BF16, BaseType::BF16, 32, 128},
    {64, AcoreMMAFormat::MK_KN, BaseType::F16, BaseType::F32, 32, 128},
    {64, AcoreMMAFormat::MK_KN, BaseType::BF16, BaseType::F32, 32, 128},
    {64, AcoreMMAFormat::MK_KN, BaseType::F16, BaseType::F32, 128, 64},
    {64, AcoreMMAFormat::MK_KN, BaseType::BF16, BaseType::F32, 128, 64},
    {64, AcoreMMAFormat::MK_KN, BaseType::F32, BaseType::F32, 16, 64},
    {64, AcoreMMAFormat::MK_NK, BaseType::F16, BaseType::F16, 32, 128},
    {64, AcoreMMAFormat::MK_NK, BaseType::F16, BaseType::F16, 64, 64},
    {64, AcoreMMAFormat::MK_NK, BaseType::BF16, BaseType::BF16, 32, 128},
    {64, AcoreMMAFormat::MK_NK, BaseType::F16, BaseType::F32, 32, 128},
    {64, AcoreMMAFormat::MK_NK, BaseType::BF16, BaseType::F32, 32, 128},
    {64, AcoreMMAFormat::MK_NK, BaseType::F16, BaseType::F32, 64, 64},
    {64, AcoreMMAFormat::MK_NK, BaseType::BF16, BaseType::F32, 128, 64},
    {64, AcoreMMAFormat::MK_NK, BaseType::F32, BaseType::F32, 16, 64},
    // M=96 (MK_KN only)
    {96, AcoreMMAFormat::MK_KN, BaseType::F16, BaseType::F16, 32, 128},
    {96, AcoreMMAFormat::MK_KN, BaseType::BF16, BaseType::BF16, 32, 128},
    // M=128
    {128, AcoreMMAFormat::MK_KN, BaseType::F16, BaseType::F16, 32, 64},
    {128, AcoreMMAFormat::MK_KN, BaseType::BF16, BaseType::BF16, 32, 64},
    {128, AcoreMMAFormat::MK_KN, BaseType::F16, BaseType::F32, 32, 64},
    {128, AcoreMMAFormat::MK_KN, BaseType::BF16, BaseType::F32, 32, 64},
    {128, AcoreMMAFormat::MK_NK, BaseType::F16, BaseType::F16, 32, 64},
    {128, AcoreMMAFormat::MK_NK, BaseType::BF16, BaseType::BF16, 32, 64},
    {128, AcoreMMAFormat::MK_NK, BaseType::F16, BaseType::F32, 32, 64},
    {128, AcoreMMAFormat::MK_NK, BaseType::BF16, BaseType::F32, 32, 64},
    {128, AcoreMMAFormat::MK_NK, BaseType::F32, BaseType::F32, 16, 32},
    // M=256 (MK_KN only)
    {256, AcoreMMAFormat::MK_KN, BaseType::F16, BaseType::F16, 32, 64},
    {256, AcoreMMAFormat::MK_KN, BaseType::BF16, BaseType::BF16, 32, 64},
    // Dynamic-M (64-aligned overload), M=0 sentinel
    {0, AcoreMMAFormat::MK_KN, BaseType::F16, BaseType::F16, 32, 128},
    {0, AcoreMMAFormat::MK_KN, BaseType::BF16, BaseType::BF16, 32, 128},
    {0, AcoreMMAFormat::MK_KN, BaseType::F32, BaseType::F32, 16, 64},
    {0, AcoreMMAFormat::MK_NK, BaseType::F16, BaseType::F16, 64, 64},
    {0, AcoreMMAFormat::MK_NK, BaseType::BF16, BaseType::BF16, 32, 128},
    {0, AcoreMMAFormat::MK_NK, BaseType::F32, BaseType::F32, 16, 64},
};

// Supported static-M values for the template overload (from matmul.h).
// M=512/1024 are addmm-only and are instead served by the dynamic-M overload.
inline bool IsAcoreSupportedStaticM(int M) {
  static const int supported[] = {1, 16, 32, 49, 64, 96, 128, 256};
  for (int v : supported)
    if (v == M) return true;
  return false;
}

// Returns true if a 64-aligned runtime-M overload exists for this type.
inline bool IsAcoreSupportedDynamicM(BaseType lhs_type) {
  return lhs_type == BaseType::F16 || lhs_type == BaseType::BF16 ||
         lhs_type == BaseType::F32;
}

// Normalize an output type for lookup: an unknown accumulator type defaults
// to the input type (the codegen falls back to the input element type when
// the accumulator fragment type is not statically known).
inline BaseType NormalizeAcoreOutType(BaseType out_type, BaseType lhs_type) {
  return out_type == BaseType::UNKNOWN ? lhs_type : out_type;
}

// Get K/N alignment requirements for given M, lhs type, output type, and
// format.  Returns {K_align, N_align} or {0,0} if unsupported.
inline std::pair<int, int> GetAcoreAlignments(int M, BaseType lhs_type,
                                              BaseType out_type,
                                              AcoreMMAFormat format) {
  // s8 (char) input quantization rows are keyed on the input type alone;
  // acore::matmul always produces `int` output for char input.
  if (lhs_type == BaseType::S8) {
    if (M == 32 || M == 64) return {64, 128};
    if (M == 128) return {64, 64};
    return {0, 0};
  }

  BaseType out = NormalizeAcoreOutType(out_type, lhs_type);
  for (const auto& p : kAcoreMatmulPatterns) {
    if (p.M == M && p.format == format && p.lhs_type == lhs_type &&
        p.out_type == out)
      return {p.K_align, p.N_align};
  }
  return {0, 0};
}

// ============================================================================
// Validation of raw acore:: calls (replaces acore_check.hpp)
// Uses the Assess API for constraint checking where possible.
// ============================================================================
template <typename ErrorFn>
bool CheckAcoreCallStorage(AST::Call& n, const std::string& func_name,
                           ErrorFn error_fn) {
  for (auto& arg : n.GetArguments()) {
    auto ty = arg->GetType();
    if (!ty) continue;
    if (auto sty = dyn_cast<SpannedType>(ty)) {
      auto storage = sty->GetStorage();
      if (storage != Storage::LOCAL && storage != Storage::NONE) {
        error_fn(n.LOC(), "acore::" + func_name +
                              " requires all buffer arguments "
                              "to be in LOCAL (L1) storage, but '" +
                              STR(arg) + "' is in " + STR(storage) +
                              " storage. "
                              "Use dma.copy to move data to local first.");
        return false;
      }
    }
  }
  return true;
}

template <typename ErrorFn>
bool CheckAcoreMatmulArgs(AST::Call& n, const std::string& func_name,
                          ErrorFn error_fn) {
  auto argc = n.GetArguments().size();
  if (argc < 11) {
    error_fn(n.LOC(), "acore::" + func_name +
                          " requires at least 11 arguments, but got " +
                          std::to_string(argc) + ".");
    return false;
  }
  if (argc > 13) {
    error_fn(n.LOC(), "acore::" + func_name +
                          " accepts at most 13 arguments, but got " +
                          std::to_string(argc) + ".");
    return false;
  }
  return true;
}

template <typename ErrorFn>
bool CheckAcoreAddmmArgs(AST::Call& n, const std::string& func_name,
                         ErrorFn error_fn) {
  auto argc = n.GetArguments().size();
  if (argc < 10) {
    error_fn(n.LOC(), "acore::" + func_name +
                          " requires at least 10 arguments, but got " +
                          std::to_string(argc) + ".");
    return false;
  }
  if (argc > 13) {
    error_fn(n.LOC(), "acore::" + func_name +
                          " accepts at most 13 arguments, but got " +
                          std::to_string(argc) + ".");
    return false;
  }
  return true;
}

template <typename ErrorFn>
bool CheckAcoreBinaryArgs(AST::Call& n, const std::string& func_name,
                          ErrorFn error_fn) {
  auto argc = n.GetArguments().size();
  if (argc < 4 || argc > 6) {
    error_fn(n.LOC(), "acore::" + func_name +
                          " requires 4-6 arguments, but got " +
                          std::to_string(argc) + ".");
    return false;
  }
  return true;
}

template <typename ErrorFn>
bool CheckAcoreUnaryArgs(AST::Call& n, const std::string& func_name,
                         ErrorFn error_fn) {
  auto argc = n.GetArguments().size();
  if (argc != 3) {
    error_fn(n.LOC(), "acore::" + func_name +
                          " requires exactly 3 arguments (dst, src, num), "
                          "but got " +
                          std::to_string(argc) + ".");
    return false;
  }
  return true;
}

template <typename ErrorFn>
bool CheckAcoreActivationArgs(AST::Call& n, const std::string& func_name,
                              ErrorFn error_fn) {
  auto argc = n.GetArguments().size();
  if (func_name == "glu" || func_name == "gtu" || func_name == "geglu" ||
      func_name == "reglu" || func_name == "swiglu") {
    if (argc != 4) {
      error_fn(n.LOC(), "acore::" + func_name +
                            " requires 4 arguments (dst, lhs, rhs, num), "
                            "but got " +
                            std::to_string(argc) + ".");
      return false;
    }
    return true;
  }
  if (func_name == "clipped_relu" || func_name == "threshold" ||
      func_name == "elu" || func_name == "celu" || func_name == "hard_shrink" ||
      func_name == "soft_shrink" || func_name == "leaky_relu" ||
      func_name == "logit") {
    if (argc < 4 || argc > 6) {
      error_fn(n.LOC(), "acore::" + func_name +
                            " requires 4-6 arguments, but got " +
                            std::to_string(argc) + ".");
      return false;
    }
    return true;
  }
  if (func_name == "hard_sigmoid" || func_name == "hard_tanh") {
    if (argc != 5) {
      error_fn(n.LOC(), "acore::" + func_name +
                            " requires 5 arguments, but got " +
                            std::to_string(argc) + ".");
      return false;
    }
    return true;
  }
  if (argc < 3 || argc > 5) {
    error_fn(n.LOC(), "acore::" + func_name +
                          " requires 3-5 arguments, but got " +
                          std::to_string(argc) + ".");
    return false;
  }
  return true;
}

template <typename ErrorFn>
bool CheckAcoreReduceArgs(AST::Call& n, const std::string& func_name,
                          ErrorFn error_fn) {
  auto argc = n.GetArguments().size();
  if (argc < 5) {
    error_fn(n.LOC(), "acore::" + func_name +
                          " requires at least 5 arguments, but got " +
                          std::to_string(argc) + ".");
    return false;
  }
  return true;
}

template <typename ErrorFn>
bool CheckAcoreCall(AST::Call& n, ErrorFn error_fn,
                    const std::string& /* arch */) {
  auto func_name = GetAcoreFuncName(n.function->name);
  if (func_name.empty()) return true;

  auto category = ClassifyAcoreOp(func_name);
  if (category == AcoreOpCategory::UNKNOWN) {
    error_fn(n.LOC(), "unknown acore operation '" + func_name +
                          "'. Check acore_op.h for supported operations.");
    return false;
  }

  if (!CheckAcoreCallStorage(n, func_name, error_fn)) return false;

  switch (category) {
  case AcoreOpCategory::MATMUL:
    return CheckAcoreMatmulArgs(n, func_name, error_fn);
  case AcoreOpCategory::ADDMM:
    return CheckAcoreAddmmArgs(n, func_name, error_fn);
  case AcoreOpCategory::BINARY:
    return CheckAcoreBinaryArgs(n, func_name, error_fn);
  case AcoreOpCategory::UNARY:
    return CheckAcoreUnaryArgs(n, func_name, error_fn);
  case AcoreOpCategory::ACTIVATION:
    return CheckAcoreActivationArgs(n, func_name, error_fn);
  case AcoreOpCategory::REDUCE:
    return CheckAcoreReduceArgs(n, func_name, error_fn);
  default: break;
  }
  return true;
}

// ============================================================================
// __lib_gemm lowering information
// Describes how a __lib_gemm BIF should be lowered to acore::matmul or a
// general fallback at codegen time.
// ============================================================================
struct LibGemmLoweringInfo {
  bool use_acore = false;
  bool use_static_M = false;
  int static_M = 0;
  bool is_MK_KN = true;
  int K_align = 0;
  int N_align = 0;
  bool has_bias = false;
  std::string fallback_reason;
};

// Attempt to determine whether __lib_gemm can be lowered to acore::matmul.
// Requires:
// - M is statically evaluable to a supported acore value, OR fits dynamic-M
// - lhs type is f16/bf16/f32
// - K and N alignment constraints satisfied (checked at runtime via Assess)
//
// assess_fn: callable (message, node) -> void for assessment registration
// warn_fn:   callable (loc, message) -> void for warnings
template <typename AssessFn, typename WarnFn>
LibGemmLoweringInfo AnalyzeLibGemm(AST::Call& n, AssessFn /* assess_fn */,
                                   WarnFn /* warn_fn */) {
  LibGemmLoweringInfo info;
  auto argc = n.arguments->Count();
  info.has_bias = (argc == 6);

  // Determine lhs type from A (arg 1)
  auto a_arg = n.arguments->ValueAt(1);
  auto a_ty = a_arg->GetType();
  BaseType lhs_bt = BaseType::UNKNOWN;
  if (a_ty) {
    if (auto sty = dyn_cast<SpannedType>(a_ty))
      lhs_bt = sty->ElementType();
    else if (auto dt = dyn_cast<DeviceDataType>(a_ty))
      lhs_bt = Choreo::GetBaseType(*dt);
    else
      lhs_bt = a_ty->GetBaseType();
  }

  if (lhs_bt != BaseType::F16 && lhs_bt != BaseType::BF16 &&
      lhs_bt != BaseType::F32 && lhs_bt != BaseType::S8) {
    info.use_acore = false;
    info.fallback_reason =
        "unsupported data type for acore matmul: " + STR(lhs_bt);
    return info;
  }

  // Try to statically evaluate M from the out shape.
  // out is arg 0, shape [M, N] -- we need the first dimension.
  auto out_arg = n.arguments->ValueAt(0);
  auto out_ty = out_arg->GetType();
  BaseType out_bt = BaseType::UNKNOWN;
  int static_M = -1;
  if (out_ty) {
    if (auto sty = dyn_cast<SpannedType>(out_ty)) {
      out_bt = sty->ElementType();
      auto shape = sty->GetShape();
      if (shape.Rank() >= 1) {
        auto m_vi = shape.Value()[0];
        if (auto mv = VIInt(m_vi)) static_M = (int)mv.value();
      }
    } else if (auto dt = dyn_cast<DeviceDataType>(out_ty)) {
      out_bt = Choreo::GetBaseType(*dt);
    } else {
      out_bt = out_ty->GetBaseType();
    }
  }

  info.is_MK_KN = true; // default to MK_KN (most commonly supported)

  if (static_M > 0 && IsAcoreSupportedStaticM(static_M)) {
    info.use_acore = true;
    info.use_static_M = true;
    info.static_M = static_M;
    auto [ka, na] = GetAcoreAlignments(static_M, lhs_bt, out_bt,
                                       AcoreMMAFormat::MK_KN);
    info.K_align = ka;
    info.N_align = na;
    if (ka == 0) {
      info.use_acore = false;
      info.fallback_reason = "M=" + std::to_string(static_M) +
                             " is not supported for type " + STR(lhs_bt) +
                             " with MK_KN format in acore matmul";
    }
  } else if (static_M > 0 && (static_M % 64 == 0) &&
             IsAcoreSupportedDynamicM(lhs_bt)) {
    info.use_acore = true;
    info.use_static_M = false;
    info.static_M = static_M;
    auto [ka, na] = GetAcoreAlignments(0, lhs_bt, out_bt, AcoreMMAFormat::MK_KN);
    info.K_align = ka;
    info.N_align = na;
  } else if (static_M > 0) {
    info.use_acore = false;
    info.fallback_reason =
        "M=" + std::to_string(static_M) +
        " is not a supported acore matmul tile size (supported: "
        "1,16,32,49,64,96,128,256 or 64-aligned for "
        "dynamic-M overload)";
  } else {
    info.use_acore = false;
    info.fallback_reason =
        "M dimension cannot be statically evaluated; acore matmul requires "
        "compile-time known M";
  }

  return info;
}

// ============================================================================
// __lib_conv2d lowering information (exploratory -- not yet enabled)
//
// The raw acore::conv2d interface requires:
//   - Template params: ho_step, wo_step, (optionally ci_step, co_step)
//   - Runtime params: out, input, weight, bias,
//                     n, hi, wi, ci, r, s, co, ho, wo,
//                     stride_h, stride_w, dilation_h, dilation_w,
//                     init_vacc_flag, wb_vacc_flag, co_offset
//
// Proposed __lib_conv2d builtin signature:
//   __lib_conv2d(out, input, weight, bias,
//                n, hi, wi, ci, r, s, co, ho, wo,
//                stride_h, stride_w)
//   (15 args, dilation defaults to 1, tile steps auto-selected)
//
// The compiler would:
//   1. Infer ho_step/wo_step from output tile shape
//   2. Auto-select ci_step/co_step from weight shape
//   3. Default dilation to {1,1} (can extend to 17 args for explicit dilation)
//   4. Handle init_vacc/wb_vacc for single-shot and chained modes
//
// Key challenge: unlike matmul where M is the main decision variable, conv2d
// tile steps depend on the relationship between input/output/weight shapes
// AND the hardware constraints on ho_step * wo_step combinations.
// ============================================================================

// ============================================================================
// Unified __lib_* BIF validation for the GCU CHECK pass.
//
// Centralises every target-specific constraint so the check visitor
// never hard-codes individual __lib_ names.
//
// ErrorFn : (loc, message) -> void
// WarnFn  : (loc, message) -> void
// AssessFn: (sbe_expr, message, node) -> void
// ============================================================================

// Return the number of leading buffer (pointer) arguments for a __lib_ call.
inline size_t LibCallBufferArgCount(const std::string& func_name, size_t argc) {
  if (func_name == "__lib_gemm") return (argc == 6) ? 4 : 3;
  if (func_name == "__lib_addmm") return 4;      // out, bias, A, B
  if (func_name == "__lib_layer_norm") return 4; // dst, src, weight, bias
  if (func_name == "__lib_conv2d") return 4;     // out, input, weight, bias
  if (func_name == "__lib_where") return 4;      // dst, cond, x, y
  if (func_name == "__lib_lerp") return 4;       // dst, start, end, weight

  auto base = func_name.substr(6);
  auto cat = ClassifyAcoreOp(base);
  switch (cat) {
  case AcoreOpCategory::BINARY: return 3;  // dst, lhs, rhs
  case AcoreOpCategory::CONVERT: return 2; // dst, src
  case AcoreOpCategory::UNARY: return 2;   // dst, src
  case AcoreOpCategory::REDUCE: return 2;  // dst, src
  case AcoreOpCategory::ACTIVATION: {
    // GLU-family are binary-style: dst, lhs, rhs
    if (base == "glu" || base == "gtu" || base == "geglu" || base == "reglu" ||
        base == "swiglu")
      return 3;
    return 2; // dst, src (parameterised extras are scalars)
  }
  default: return 2;
  }
}

template <typename ErrorFn, typename WarnFn, typename AssessFn>
void ValidateLibCall(AST::Call& n, ErrorFn error_fn, WarnFn warn_fn,
                     AssessFn assess_fn, bool use_target_lib) {
  auto func_name = n.function->name;
  auto argc = n.arguments->Count();

  // --- Storage check (applies to ALL __lib_ buffer calls) ---
  auto n_buf = LibCallBufferArgCount(func_name, argc);
  for (size_t i = 0; i < n_buf && i < argc; ++i) {
    auto arg = n.arguments->ValueAt(i);
    auto ty = arg->GetType();
    if (!ty) continue;
    if (auto sty = dyn_cast<SpannedType>(ty)) {
      auto storage = sty->GetStorage();
      if (storage != Storage::LOCAL && storage != Storage::NONE)
        error_fn(n.LOC(), func_name +
                              " requires all buffer arguments to be in LOCAL "
                              "(L1) storage, but '" +
                              STR(arg) + "' is in " + STR(storage) +
                              " storage.");
    }
  }

  // --- __lib_gemm dimension constraints ---
  if (func_name == "__lib_gemm") {
    bool has_bias = (argc == 6);
    size_t k_idx = has_bias ? 4 : 3;
    size_t n_idx = has_bias ? 5 : 4;

    if (k_idx < argc) {
      if (auto ke = dyn_cast<AST::Expr>(n.arguments->ValueAt(k_idx)))
        if (ke->Opts().HasVal())
          assess_fn(sbe::cmp(">", ke->Opts().GetVal(), sbe::nu(0)),
                    "__lib_gemm: K dimension must be positive.", n);
    }
    if (n_idx < argc) {
      if (auto ne = dyn_cast<AST::Expr>(n.arguments->ValueAt(n_idx)))
        if (ne->Opts().HasVal())
          assess_fn(sbe::cmp(">", ne->Opts().GetVal(), sbe::nu(0)),
                    "__lib_gemm: N dimension must be positive.", n);
    }

    if (use_target_lib) {
      auto info =
          AnalyzeLibGemm(n, [](const std::string&, AST::Node*) {}, warn_fn);
      if (!info.use_acore && !info.fallback_reason.empty())
        warn_fn(n.LOC(), "__lib_gemm will use general (non-acore) fallback: " +
                             info.fallback_reason);
    }
  }

  // --- __lib_addmm dimension constraints ---
  if (func_name == "__lib_addmm") {
    for (size_t idx : {(size_t)4, (size_t)5, (size_t)6}) { // M, K, N
      if (idx >= argc) continue;
      if (auto e = dyn_cast<AST::Expr>(n.arguments->ValueAt(idx)))
        if (e->Opts().HasVal())
          assess_fn(sbe::cmp(">", e->Opts().GetVal(), sbe::nu(0)),
                    func_name + ": dimension args must be positive.", n);
    }
  }

  // --- __lib_layer_norm dimension constraints ---
  if (func_name == "__lib_layer_norm") {
    for (size_t idx : {(size_t)4, (size_t)5}) { // batch, norm_size
      if (idx >= argc) continue;
      if (auto e = dyn_cast<AST::Expr>(n.arguments->ValueAt(idx)))
        if (e->Opts().HasVal())
          assess_fn(sbe::cmp(">", e->Opts().GetVal(), sbe::nu(0)),
                    func_name + ": batch and norm_size must be positive.", n);
    }
  }

  // --- __lib_conv2d dimension constraints ---
  if (func_name == "__lib_conv2d") {
    for (size_t idx = 4; idx < 11 && idx < argc; ++idx) {
      if (auto e = dyn_cast<AST::Expr>(n.arguments->ValueAt(idx)))
        if (e->Opts().HasVal())
          assess_fn(sbe::cmp(">", e->Opts().GetVal(), sbe::nu(0)),
                    func_name + ": spatial/channel dims must be positive.", n);
    }
  }
}

// ============================================================================
// LibCall codegen classification
// ============================================================================
enum class LibCallKind {
  GEMM,
  ADDMM,
  UNARY,
  BINARY,
  ACTIVATION_PARAM,
  ACTIVATION_BIN,
  REDUCE,
  NORM,
  CONV2D,
  CONVERT,
  POINTWISE,
  UNKNOWN
};

inline LibCallKind ClassifyLibCall(const std::string& func_name) {
  if (func_name == "__lib_gemm") return LibCallKind::GEMM;
  if (func_name == "__lib_addmm") return LibCallKind::ADDMM;

  static const std::set<std::string> binary = {
      "__lib_add",  "__lib_sub",       "__lib_mul", "__lib_div",
      "__lib_max",  "__lib_min",       "__lib_pow", "__lib_atan2",
      "__lib_fmod", "__lib_remainder", "__lib_gt",  "__lib_ge",
      "__lib_lt",   "__lib_le",        "__lib_eq",  "__lib_ne",
  };
  static const std::set<std::string> unary = {
      "__lib_abs",        "__lib_neg",         "__lib_sign",
      "__lib_sqrt",       "__lib_rsqrt",       "__lib_cbrt",
      "__lib_reciprocal", "__lib_exp",         "__lib_log",
      "__lib_erf",        "__lib_erfc",        "__lib_ceil",
      "__lib_floor",      "__lib_trunc",       "__lib_round",
      "__lib_sin",        "__lib_cos",         "__lib_tan",
      "__lib_asin",       "__lib_acos",        "__lib_atan",
      "__lib_sinh",       "__lib_cosh",        "__lib_tanh",
      "__lib_relu",       "__lib_gelu",        "__lib_selu",
      "__lib_sigmoid",    "__lib_silu",        "__lib_swish",
      "__lib_softplus",   "__lib_hard_swish",  "__lib_mish",
      "__lib_quick_gelu", "__lib_log_sigmoid",
  };
  static const std::set<std::string> activation_param = {
      "__lib_leaky_relu",   "__lib_elu",          "__lib_celu",
      "__lib_hard_shrink",  "__lib_soft_shrink",  "__lib_logit",
      "__lib_threshold",    "__lib_hard_sigmoid", "__lib_hard_tanh",
      "__lib_clipped_relu",
  };
  static const std::set<std::string> activation_bin = {
      "__lib_glu",
      "__lib_swiglu",
      "__lib_geglu",
      "__lib_reglu",
  };
  static const std::set<std::string> reduce = {
      "__lib_reduce_sum",  "__lib_reduce_max",  "__lib_reduce_min",
      "__lib_reduce_prod", "__lib_reduce_mean",
  };

  if (binary.count(func_name)) return LibCallKind::BINARY;
  if (unary.count(func_name)) return LibCallKind::UNARY;
  if (activation_param.count(func_name)) return LibCallKind::ACTIVATION_PARAM;
  if (activation_bin.count(func_name)) return LibCallKind::ACTIVATION_BIN;
  if (reduce.count(func_name)) return LibCallKind::REDUCE;
  if (func_name == "__lib_layer_norm") return LibCallKind::NORM;
  if (func_name == "__lib_conv2d") return LibCallKind::CONV2D;
  if (func_name == "__lib_convert") return LibCallKind::CONVERT;
  if (func_name == "__lib_where" || func_name == "__lib_lerp")
    return LibCallKind::POINTWISE;
  return LibCallKind::UNKNOWN;
}

// Maps __lib_<name> to the acore:: function name.
inline std::string LibCallToAcoreName(const std::string& func_name) {
  assert(PrefixedWith(func_name, "__lib_"));
  auto base = func_name.substr(6);

  if (base == "round") return "round_ne";
  if (base == "gelu") return "gelu_erf";

  return base;
}

} // end namespace Choreo

#endif // __CHOREO_GCU_LOWER_LIBCALL_HPP__
