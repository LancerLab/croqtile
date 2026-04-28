#ifndef __CHOREO_GCU_TARGET_HPP__
#define __CHOREO_GCU_TARGET_HPP__

#include "context.hpp"
#include "target.hpp"
#include "target_registry.hpp"

namespace Choreo {

class GCUTarget : public Target {
public:
  size_t GetMemCapacity(const Storage& sto, const ArchId& arch) const override {
    if (!IsArchSupported(arch))
      choreo_unreachable("unsupported architecture '" + arch + "'.");

    if (arch == "gcu200" || arch == "gcu210") {
      switch (sto) {
      case Storage::LOCAL: return 1008ull * 1024;             // 1008KB
      case Storage::SHARED: return 24ull * 1024 * 1024;       // 24MB
      case Storage::GLOBAL: return 4ull * 1024 * 1024 * 1024; // 4GB
      default: choreo_unreachable("Unsupported mem level.");
      }
    } else if (arch == "gcu300") {
      /*
      TODO:
      For GCU3, all is different with Scorpio (1 Die) in the link below
      Is S60G same with c035?
      L3 (global) is different with Dorado (3VG per Cluster) in
      http://wiki.enflame.cn/display/~james.zhu/Enflame+GCU+Programming+Model#EnflameGCUProgrammingModel-get_memory_space
      */
      switch (sto) {
      case Storage::LOCAL: {
        return 1.5 * 1024 * 1024 - 512; // 1.5MB minus reserved
      } break;
      case Storage::SHARED: {
        return 64ull * 1024 * 1024; // 64MB
      } break;
      case Storage::GLOBAL: return 40.75 * 1024 * 1024 * 1024; // 40.75GB
      default: choreo_unreachable("unsupported storage level.");
      }
    } else if (arch == "gcu400") {
      assert(Name() == "topscc");
      switch (sto) {
      case Storage::LOCAL: return 1.5 * 1024 * 1024 - 512; // todo: check this
      case Storage::SHARED: return 64ull * 1024 * 1024;    // todo: check this
      case Storage::GLOBAL:
        return 40.75 * 1024 * 1024 * 1024; // TODO: check this
      default: choreo_unreachable("unsupported storage level.");
      }
    } else if (arch == "gcu500") {
      assert(Name() == "topscc");
      switch (sto) {
      case Storage::LOCAL: return 4 * 1024 * 1024 - 512; // todo: check this
      case Storage::SHARED: return 256ull * 1024 * 1024; // todo: check this
      case Storage::GLOBAL:
        return 128ull * 1024 * 1024 * 1024; // todo: check this
      default: choreo_unreachable("unsupported storage level.");
      }
    }
    choreo_unreachable("unsupported target.");
    return 0;
  }
  const ArchId DefaultArch() const override { return "gcu300"; }
  size_t GetMemAlignmentByte(const Storage& sto, const ArchId&) const override {
    switch (sto) {
    case Storage::LOCAL:
    case Storage::SHARED: return 512;
    default: choreo_unreachable("Unsupported mem level.");
    }
    return 0;
  }

  size_t GetMinGroupDim(const ArchId& arch) const override {
    auto arch_num = ArchNum(arch);
    if (arch_num < 400)
      choreo_unreachable("unsupported architecture '" + arch + "'.");
    return 1;
  }

  // Per-architecture limits for parallel-by levels.
  size_t GetMaxParallelByCount(ParallelLevel pl,
                               const ArchId& arch) const override {
    int arch_num = ArchNum(arch);
    // gcu200/210: block <= 4, thread <= 8
    if (arch_num >= 200 && arch_num < 300) {
      if (pl == ParallelLevel::BLOCK) return 4;
      if (pl == ParallelLevel::THREAD) return 8;
    }

    // gcu300: block <= 2, thread <= 12
    if (arch_num >= 300 && arch_num < 400) {
      if (pl == ParallelLevel::BLOCK) return 2;
      if (pl == ParallelLevel::THREAD) return 12;
    }

    // gcu400+: unconstrained for now.
    return 0;
  }

  const std::vector<ParallelLevel>
  GetParallelLevels(const ArchId& arch) const override {
    auto arch_num = ArchNum(arch);
    if (arch_num < 400)
      return {ParallelLevel::SEQ, ParallelLevel::BLOCK, ParallelLevel::THREAD};
    else
      return {ParallelLevel::SEQ, ParallelLevel::BLOCK, ParallelLevel::GROUP,
              ParallelLevel::THREAD};
  }

  bool IsLibCallSupported(const std::string& name) const override {
    static const std::set<std::string> supported = {
        // GEMM: (out, A, B, K, N) or (out, A, B, bias, K, N)
        "__lib_gemm",
        // ADDMM: (out, bias, A, B, M, K, N, alpha, beta)
        "__lib_addmm",

        // Binary elementwise: (dst, lhs, rhs, num)
        "__lib_add", "__lib_sub", "__lib_mul", "__lib_div", "__lib_max",
        "__lib_min", "__lib_pow", "__lib_atan2", "__lib_fmod",
        "__lib_remainder", "__lib_gt", "__lib_ge", "__lib_lt", "__lib_le",
        "__lib_eq", "__lib_ne",

        // Unary elementwise: (dst, src, num)
        "__lib_abs", "__lib_neg", "__lib_sign", "__lib_sqrt", "__lib_rsqrt",
        "__lib_cbrt", "__lib_reciprocal", "__lib_exp", "__lib_log", "__lib_erf",
        "__lib_erfc", "__lib_ceil", "__lib_floor", "__lib_trunc", "__lib_round",
        "__lib_sin", "__lib_cos", "__lib_tan", "__lib_asin", "__lib_acos",
        "__lib_atan", "__lib_sinh", "__lib_cosh", "__lib_tanh",

        // Unary activations (no extra params): (dst, src, num)
        "__lib_relu", "__lib_gelu", "__lib_selu", "__lib_sigmoid", "__lib_silu",
        "__lib_swish", "__lib_softplus", "__lib_hard_swish", "__lib_mish",
        "__lib_quick_gelu", "__lib_log_sigmoid",

        // Parameterised activations: (dst, src, num, ...)
        "__lib_leaky_relu",   // +alpha
        "__lib_elu",          // +alpha
        "__lib_celu",         // +alpha
        "__lib_hard_shrink",  // +lambda
        "__lib_soft_shrink",  // +lambda
        "__lib_logit",        // +eps
        "__lib_threshold",    // +threshold, value
        "__lib_hard_sigmoid", // +alpha, beta
        "__lib_hard_tanh",    // +min_val, max_val
        "__lib_clipped_relu", // +min_val, max_val

        // Binary activations (GLU family): (dst, lhs, rhs, num)
        "__lib_glu", "__lib_swiglu", "__lib_geglu", "__lib_reglu",

        // Reduce: (dst, src, num, reduce_dim, num_reduce)
        "__lib_reduce_sum", "__lib_reduce_max", "__lib_reduce_min",
        "__lib_reduce_prod", "__lib_reduce_mean",

        // Layer norm: (dst, src, weight, bias, batch, norm_size, eps)
        "__lib_layer_norm",

        // Conv2d: (out, input, weight, bias,
        //          batch, hi, wi, ci, co, kh, kw,
        //          stride_h, stride_w, pad_h, pad_w)
        "__lib_conv2d",

        // Convert: (dst, src, num) -- types deduced from pointers
        "__lib_convert",

        // Pointwise:
        "__lib_where", // (dst, cond, x, y, num)
        "__lib_lerp",  // (dst, start, end, weight, num)
    };
    return supported.count(name) > 0;
  }

  std::pair<int, int> LibCallArgRange(const std::string& name) const override {
    if (name == "__lib_gemm") return {5, 6};
    if (name == "__lib_addmm") return {9, 9};

    // Binary: (dst, lhs, rhs, num)
    if (name == "__lib_add" || name == "__lib_sub" || name == "__lib_mul" ||
        name == "__lib_div" || name == "__lib_max" || name == "__lib_min" ||
        name == "__lib_pow" || name == "__lib_atan2" || name == "__lib_fmod" ||
        name == "__lib_remainder" || name == "__lib_gt" || name == "__lib_ge" ||
        name == "__lib_lt" || name == "__lib_le" || name == "__lib_eq" ||
        name == "__lib_ne")
      return {4, 4};

    // Binary activations (GLU family): (dst, lhs, rhs, num)
    if (name == "__lib_glu" || name == "__lib_swiglu" ||
        name == "__lib_geglu" || name == "__lib_reglu")
      return {4, 4};

    // Unary / unary-activation: (dst, src, num)
    if (name == "__lib_abs" || name == "__lib_neg" || name == "__lib_sign" ||
        name == "__lib_sqrt" || name == "__lib_rsqrt" || name == "__lib_cbrt" ||
        name == "__lib_reciprocal" || name == "__lib_exp" ||
        name == "__lib_log" || name == "__lib_erf" || name == "__lib_erfc" ||
        name == "__lib_ceil" || name == "__lib_floor" ||
        name == "__lib_trunc" || name == "__lib_round" || name == "__lib_sin" ||
        name == "__lib_cos" || name == "__lib_tan" || name == "__lib_asin" ||
        name == "__lib_acos" || name == "__lib_atan" || name == "__lib_sinh" ||
        name == "__lib_cosh" || name == "__lib_tanh" || name == "__lib_relu" ||
        name == "__lib_gelu" || name == "__lib_selu" ||
        name == "__lib_sigmoid" || name == "__lib_silu" ||
        name == "__lib_swish" || name == "__lib_softplus" ||
        name == "__lib_hard_swish" || name == "__lib_mish" ||
        name == "__lib_quick_gelu" || name == "__lib_log_sigmoid")
      return {3, 3};

    // Parameterised activations with 1 extra param: (dst, src, num, param)
    if (name == "__lib_leaky_relu" || name == "__lib_elu" ||
        name == "__lib_celu" || name == "__lib_hard_shrink" ||
        name == "__lib_soft_shrink" || name == "__lib_logit")
      return {4, 4};
    // Parameterised activations with 2 extra params: (dst, src, num, p1, p2)
    if (name == "__lib_threshold" || name == "__lib_hard_sigmoid" ||
        name == "__lib_hard_tanh" || name == "__lib_clipped_relu")
      return {5, 5};

    // Reduce: (dst, src, num, reduce_dim, num_reduce)
    if (name == "__lib_reduce_sum" || name == "__lib_reduce_max" ||
        name == "__lib_reduce_min" || name == "__lib_reduce_prod" ||
        name == "__lib_reduce_mean")
      return {5, 5};

    // Layer norm: (dst, src, weight, bias, batch, norm_size, eps)
    if (name == "__lib_layer_norm") return {7, 7};

    // Conv2d: 15 args
    if (name == "__lib_conv2d") return {15, 15};

    // Convert: (dst, src, num)
    if (name == "__lib_convert") return {3, 3};

    // Pointwise: (dst, cond/start, x/end, y/weight, num)
    if (name == "__lib_where" || name == "__lib_lerp") return {5, 5};

    return {-1, -1};
  }

  bool DefaultUseTargetLib() const override { return true; }
};

} // end namespace Choreo

#endif // __CHOREO_GCU_TARGET_HPP__
