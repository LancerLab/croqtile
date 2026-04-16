#ifndef __CHOREO_GCU_LIB_FALLBACK_H__
#define __CHOREO_GCU_LIB_FALLBACK_H__

// General loop-based fallback implementations for __lib_* builtins.
// These are used when the target library path is unavailable
// or when the operation doesn't satisfy hardware constraints.
//
// WARNING: These are simple loop-based implementations intended as
// correctness references, NOT for high-performance use. The compiler
// emits a warning whenever these fallbacks are selected.

#include <cmath>
#include <limits>

namespace choreo {

// =====================================================================
// GEMM fallback: out[M,N] = A[M,K] * B[K,N] + bias[N]
// =====================================================================

template <typename T>
inline void lib_gemm_general(T* out, const T* A, const T* B,
                             int M, int K, int N) {
  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      T acc = 0;
      for (int k = 0; k < K; ++k)
        acc += (T)(A[m * K + k] * B[k * N + n]);
      out[m * N + n] = acc;
    }
  }
}

template <typename T>
inline void lib_gemm_bias_general(T* out, const T* A, const T* B,
                                  const T* bias, int M, int K, int N) {
  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      T acc = 0;
      for (int k = 0; k < K; ++k)
        acc += (T)(A[m * K + k] * B[k * N + n]);
      out[m * N + n] = acc + bias[n];
    }
  }
}

// =====================================================================
// Unary elementwise: lib_<op><T>(dst, src, num)
// =====================================================================

template <typename T>
inline void lib_abs(T* dst, const T* src, int num) {
  for (int i = 0; i < num; ++i)
    dst[i] = (src[i] >= 0) ? src[i] : -src[i];
}

template <typename T>
inline void lib_neg(T* dst, const T* src, int num) {
  for (int i = 0; i < num; ++i)
    dst[i] = -src[i];
}

template <typename T>
inline void lib_sign(T* dst, const T* src, int num) {
  for (int i = 0; i < num; ++i)
    dst[i] = (src[i] > (T)0) ? (T)1 : ((src[i] < (T)0) ? (T)(-1) : (T)0);
}

template <typename T>
inline void lib_sqrt(T* dst, const T* src, int num) {
  for (int i = 0; i < num; ++i)
    dst[i] = (T)std::sqrt((float)src[i]);
}

template <typename T>
inline void lib_rsqrt(T* dst, const T* src, int num) {
  for (int i = 0; i < num; ++i)
    dst[i] = (T)(1.0f / std::sqrt((float)src[i]));
}

template <typename T>
inline void lib_cbrt(T* dst, const T* src, int num) {
  for (int i = 0; i < num; ++i)
    dst[i] = (T)std::cbrt((float)src[i]);
}

template <typename T>
inline void lib_reciprocal(T* dst, const T* src, int num) {
  for (int i = 0; i < num; ++i)
    dst[i] = (T)(1.0f / (float)src[i]);
}

template <typename T>
inline void lib_exp(T* dst, const T* src, int num) {
  for (int i = 0; i < num; ++i)
    dst[i] = (T)std::exp((float)src[i]);
}

template <typename T>
inline void lib_log(T* dst, const T* src, int num) {
  for (int i = 0; i < num; ++i)
    dst[i] = (T)std::log((float)src[i]);
}

template <typename T>
inline void lib_erf(T* dst, const T* src, int num) {
  for (int i = 0; i < num; ++i)
    dst[i] = (T)std::erf((float)src[i]);
}

template <typename T>
inline void lib_erfc(T* dst, const T* src, int num) {
  for (int i = 0; i < num; ++i)
    dst[i] = (T)std::erfc((float)src[i]);
}

template <typename T>
inline void lib_ceil(T* dst, const T* src, int num) {
  for (int i = 0; i < num; ++i)
    dst[i] = (T)std::ceil((float)src[i]);
}

template <typename T>
inline void lib_floor(T* dst, const T* src, int num) {
  for (int i = 0; i < num; ++i)
    dst[i] = (T)std::floor((float)src[i]);
}

template <typename T>
inline void lib_trunc(T* dst, const T* src, int num) {
  for (int i = 0; i < num; ++i)
    dst[i] = (T)std::trunc((float)src[i]);
}

template <typename T>
inline void lib_round_ne(T* dst, const T* src, int num) {
  for (int i = 0; i < num; ++i)
    dst[i] = (T)std::nearbyint((float)src[i]);
}

template <typename T>
inline void lib_sin(T* dst, const T* src, int num) {
  for (int i = 0; i < num; ++i)
    dst[i] = (T)std::sin((float)src[i]);
}

template <typename T>
inline void lib_cos(T* dst, const T* src, int num) {
  for (int i = 0; i < num; ++i)
    dst[i] = (T)std::cos((float)src[i]);
}

template <typename T>
inline void lib_tan(T* dst, const T* src, int num) {
  for (int i = 0; i < num; ++i)
    dst[i] = (T)std::tan((float)src[i]);
}

template <typename T>
inline void lib_asin(T* dst, const T* src, int num) {
  for (int i = 0; i < num; ++i)
    dst[i] = (T)std::asin((float)src[i]);
}

template <typename T>
inline void lib_acos(T* dst, const T* src, int num) {
  for (int i = 0; i < num; ++i)
    dst[i] = (T)std::acos((float)src[i]);
}

template <typename T>
inline void lib_atan(T* dst, const T* src, int num) {
  for (int i = 0; i < num; ++i)
    dst[i] = (T)std::atan((float)src[i]);
}

template <typename T>
inline void lib_sinh(T* dst, const T* src, int num) {
  for (int i = 0; i < num; ++i)
    dst[i] = (T)std::sinh((float)src[i]);
}

template <typename T>
inline void lib_cosh(T* dst, const T* src, int num) {
  for (int i = 0; i < num; ++i)
    dst[i] = (T)std::cosh((float)src[i]);
}

template <typename T>
inline void lib_tanh(T* dst, const T* src, int num) {
  for (int i = 0; i < num; ++i)
    dst[i] = (T)std::tanh((float)src[i]);
}

// =====================================================================
// Unary activations (no extra params): lib_<op><T>(dst, src, num)
// =====================================================================

template <typename T>
inline void lib_relu(T* dst, const T* src, int num) {
  for (int i = 0; i < num; ++i)
    dst[i] = (src[i] > (T)0) ? src[i] : (T)0;
}

template <typename T>
inline void lib_gelu_erf(T* dst, const T* src, int num) {
  for (int i = 0; i < num; ++i) {
    float x = (float)src[i];
    dst[i] = (T)(0.5f * x * (1.0f + std::erf(x * 0.7071067811865476f)));
  }
}

template <typename T>
inline void lib_selu(T* dst, const T* src, int num) {
  constexpr float alpha = 1.6732632423543772f;
  constexpr float scale = 1.0507009873554805f;
  for (int i = 0; i < num; ++i) {
    float x = (float)src[i];
    dst[i] = (T)(scale * (x > 0.0f ? x : alpha * (std::exp(x) - 1.0f)));
  }
}

template <typename T>
inline void lib_sigmoid(T* dst, const T* src, int num) {
  for (int i = 0; i < num; ++i)
    dst[i] = (T)(1.0f / (1.0f + std::exp(-(float)src[i])));
}

template <typename T>
inline void lib_silu(T* dst, const T* src, int num) {
  for (int i = 0; i < num; ++i) {
    float x = (float)src[i];
    dst[i] = (T)(x / (1.0f + std::exp(-x)));
  }
}

template <typename T>
inline void lib_swish(T* dst, const T* src, int num) {
  lib_silu(dst, src, num);
}

template <typename T>
inline void lib_softplus(T* dst, const T* src, int num) {
  for (int i = 0; i < num; ++i)
    dst[i] = (T)std::log(1.0f + std::exp((float)src[i]));
}

template <typename T>
inline void lib_hard_swish(T* dst, const T* src, int num) {
  for (int i = 0; i < num; ++i) {
    float x = (float)src[i];
    float r = x + 3.0f;
    r = (r < 0.0f) ? 0.0f : ((r > 6.0f) ? 6.0f : r);
    dst[i] = (T)(x * r / 6.0f);
  }
}

template <typename T>
inline void lib_mish(T* dst, const T* src, int num) {
  for (int i = 0; i < num; ++i) {
    float x = (float)src[i];
    dst[i] = (T)(x * std::tanh(std::log(1.0f + std::exp(x))));
  }
}

template <typename T>
inline void lib_quick_gelu(T* dst, const T* src, int num) {
  for (int i = 0; i < num; ++i) {
    float x = (float)src[i];
    dst[i] = (T)(x / (1.0f + std::exp(-1.702f * x)));
  }
}

template <typename T>
inline void lib_log_sigmoid(T* dst, const T* src, int num) {
  for (int i = 0; i < num; ++i) {
    float x = (float)src[i];
    dst[i] = (T)(-std::log(1.0f + std::exp(-x)));
  }
}

// =====================================================================
// Parameterised activations: lib_<op><T>(dst, src, num, ...)
// =====================================================================

template <typename T>
inline void lib_leaky_relu(T* dst, const T* src, int num, float alpha) {
  for (int i = 0; i < num; ++i) {
    float x = (float)src[i];
    dst[i] = (T)(x >= 0.0f ? x : alpha * x);
  }
}

template <typename T>
inline void lib_elu(T* dst, const T* src, int num, float alpha) {
  for (int i = 0; i < num; ++i) {
    float x = (float)src[i];
    dst[i] = (T)(x >= 0.0f ? x : alpha * (std::exp(x) - 1.0f));
  }
}

template <typename T>
inline void lib_celu(T* dst, const T* src, int num, float alpha) {
  for (int i = 0; i < num; ++i) {
    float x = (float)src[i];
    float neg_part = alpha * (std::exp(x / alpha) - 1.0f);
    dst[i] = (T)((x > 0.0f) ? x : neg_part);
  }
}

template <typename T>
inline void lib_hard_shrink(T* dst, const T* src, int num, float lambda) {
  for (int i = 0; i < num; ++i) {
    float x = (float)src[i];
    dst[i] = (T)((x < -lambda || x > lambda) ? x : 0.0f);
  }
}

template <typename T>
inline void lib_soft_shrink(T* dst, const T* src, int num, float lambda) {
  for (int i = 0; i < num; ++i) {
    float x = (float)src[i];
    if (x > lambda)
      dst[i] = (T)(x - lambda);
    else if (x < -lambda)
      dst[i] = (T)(x + lambda);
    else
      dst[i] = (T)0;
  }
}

template <typename T>
inline void lib_logit(T* dst, const T* src, int num, float eps) {
  for (int i = 0; i < num; ++i) {
    float x = (float)src[i];
    x = (x < eps) ? eps : ((x > 1.0f - eps) ? 1.0f - eps : x);
    dst[i] = (T)std::log(x / (1.0f - x));
  }
}

template <typename T>
inline void lib_threshold(T* dst, const T* src, int num, float threshold,
                          float value) {
  for (int i = 0; i < num; ++i)
    dst[i] = (T)(((float)src[i] > threshold) ? (float)src[i] : value);
}

template <typename T>
inline void lib_hard_sigmoid(T* dst, const T* src, int num, float alpha,
                             float beta) {
  for (int i = 0; i < num; ++i) {
    float r = alpha * (float)src[i] + beta;
    dst[i] = (T)((r < 0.0f) ? 0.0f : ((r > 1.0f) ? 1.0f : r));
  }
}

template <typename T>
inline void lib_hard_tanh(T* dst, const T* src, int num, float min_val,
                          float max_val) {
  for (int i = 0; i < num; ++i) {
    float x = (float)src[i];
    dst[i] = (T)((x < min_val) ? min_val : ((x > max_val) ? max_val : x));
  }
}

template <typename T>
inline void lib_clipped_relu(T* dst, const T* src, int num, float min_val,
                             float max_val) {
  for (int i = 0; i < num; ++i) {
    float x = (float)src[i];
    x = (x < min_val) ? min_val : x;
    dst[i] = (T)((x > max_val) ? max_val : x);
  }
}

// =====================================================================
// Binary activations (GLU family): lib_<op><T>(dst, lhs, rhs, num)
//
// Gate Linear Unit variants. In each, `lhs` is the value half and
// `rhs` is the gate half. The gate function differs per variant.
// =====================================================================

template <typename T>
inline void lib_glu(T* dst, const T* lhs, const T* rhs, int num) {
  for (int i = 0; i < num; ++i) {
    float gate = 1.0f / (1.0f + std::exp(-(float)rhs[i]));
    dst[i] = (T)((float)lhs[i] * gate);
  }
}

template <typename T>
inline void lib_swiglu(T* dst, const T* lhs, const T* rhs, int num) {
  for (int i = 0; i < num; ++i) {
    float g = (float)rhs[i];
    float gate = g / (1.0f + std::exp(-g));
    dst[i] = (T)((float)lhs[i] * gate);
  }
}

template <typename T>
inline void lib_geglu(T* dst, const T* lhs, const T* rhs, int num) {
  for (int i = 0; i < num; ++i) {
    float g = (float)rhs[i];
    float gate = 0.5f * g * (1.0f + std::erf(g * 0.7071067811865476f));
    dst[i] = (T)((float)lhs[i] * gate);
  }
}

template <typename T>
inline void lib_reglu(T* dst, const T* lhs, const T* rhs, int num) {
  for (int i = 0; i < num; ++i) {
    float gate = ((float)rhs[i] > 0.0f) ? (float)rhs[i] : 0.0f;
    dst[i] = (T)((float)lhs[i] * gate);
  }
}

// =====================================================================
// Binary elementwise: lib_<op><T>(dst, lhs, rhs, num)
// =====================================================================

template <typename T>
inline void lib_add(T* dst, const T* lhs, const T* rhs, int num) {
  for (int i = 0; i < num; ++i)
    dst[i] = lhs[i] + rhs[i];
}

template <typename T>
inline void lib_sub(T* dst, const T* lhs, const T* rhs, int num) {
  for (int i = 0; i < num; ++i)
    dst[i] = lhs[i] - rhs[i];
}

template <typename T>
inline void lib_mul(T* dst, const T* lhs, const T* rhs, int num) {
  for (int i = 0; i < num; ++i)
    dst[i] = lhs[i] * rhs[i];
}

template <typename T>
inline void lib_div(T* dst, const T* lhs, const T* rhs, int num) {
  for (int i = 0; i < num; ++i)
    dst[i] = lhs[i] / rhs[i];
}

template <typename T>
inline void lib_max(T* dst, const T* lhs, const T* rhs, int num) {
  for (int i = 0; i < num; ++i)
    dst[i] = (lhs[i] > rhs[i]) ? lhs[i] : rhs[i];
}

template <typename T>
inline void lib_min(T* dst, const T* lhs, const T* rhs, int num) {
  for (int i = 0; i < num; ++i)
    dst[i] = (lhs[i] < rhs[i]) ? lhs[i] : rhs[i];
}

template <typename T>
inline void lib_pow(T* dst, const T* lhs, const T* rhs, int num) {
  for (int i = 0; i < num; ++i)
    dst[i] = (T)std::pow((float)lhs[i], (float)rhs[i]);
}

template <typename T>
inline void lib_atan2(T* dst, const T* lhs, const T* rhs, int num) {
  for (int i = 0; i < num; ++i)
    dst[i] = (T)std::atan2((float)lhs[i], (float)rhs[i]);
}

template <typename T>
inline void lib_fmod(T* dst, const T* lhs, const T* rhs, int num) {
  for (int i = 0; i < num; ++i)
    dst[i] = (T)std::fmod((float)lhs[i], (float)rhs[i]);
}

template <typename T>
inline void lib_remainder(T* dst, const T* lhs, const T* rhs, int num) {
  for (int i = 0; i < num; ++i)
    dst[i] = (T)std::remainder((float)lhs[i], (float)rhs[i]);
}

template <typename T>
inline void lib_gt(T* dst, const T* lhs, const T* rhs, int num) {
  for (int i = 0; i < num; ++i)
    dst[i] = (T)(lhs[i] > rhs[i] ? 1 : 0);
}

template <typename T>
inline void lib_ge(T* dst, const T* lhs, const T* rhs, int num) {
  for (int i = 0; i < num; ++i)
    dst[i] = (T)(lhs[i] >= rhs[i] ? 1 : 0);
}

template <typename T>
inline void lib_lt(T* dst, const T* lhs, const T* rhs, int num) {
  for (int i = 0; i < num; ++i)
    dst[i] = (T)(lhs[i] < rhs[i] ? 1 : 0);
}

template <typename T>
inline void lib_le(T* dst, const T* lhs, const T* rhs, int num) {
  for (int i = 0; i < num; ++i)
    dst[i] = (T)(lhs[i] <= rhs[i] ? 1 : 0);
}

template <typename T>
inline void lib_eq(T* dst, const T* lhs, const T* rhs, int num) {
  for (int i = 0; i < num; ++i)
    dst[i] = (T)(lhs[i] == rhs[i] ? 1 : 0);
}

template <typename T>
inline void lib_ne(T* dst, const T* lhs, const T* rhs, int num) {
  for (int i = 0; i < num; ++i)
    dst[i] = (T)(lhs[i] != rhs[i] ? 1 : 0);
}

// =====================================================================
// Reduce operations: lib_reduce_<op><T>(dst, src, num, rdim, nred)
// =====================================================================

template <typename T>
inline void lib_reduce_sum(T* dst, const T* src, int num, int rdim,
                           int nred) {
  int out_count = num / nred;
  for (int i = 0; i < out_count; ++i) {
    T acc = (T)0;
    for (int j = 0; j < nred; ++j)
      acc += src[i * nred + j];
    dst[i] = acc;
  }
}

template <typename T>
inline void lib_reduce_max(T* dst, const T* src, int num, int rdim,
                           int nred) {
  int out_count = num / nred;
  for (int i = 0; i < out_count; ++i) {
    T val = src[i * nred];
    for (int j = 1; j < nred; ++j)
      if (src[i * nred + j] > val) val = src[i * nred + j];
    dst[i] = val;
  }
}

template <typename T>
inline void lib_reduce_min(T* dst, const T* src, int num, int rdim,
                           int nred) {
  int out_count = num / nred;
  for (int i = 0; i < out_count; ++i) {
    T val = src[i * nred];
    for (int j = 1; j < nred; ++j)
      if (src[i * nred + j] < val) val = src[i * nred + j];
    dst[i] = val;
  }
}

template <typename T>
inline void lib_reduce_prod(T* dst, const T* src, int num, int rdim,
                            int nred) {
  int out_count = num / nred;
  for (int i = 0; i < out_count; ++i) {
    T acc = (T)1;
    for (int j = 0; j < nred; ++j)
      acc *= src[i * nred + j];
    dst[i] = acc;
  }
}

template <typename T>
inline void lib_reduce_mean(T* dst, const T* src, int num, int rdim,
                            int nred) {
  int out_count = num / nred;
  for (int i = 0; i < out_count; ++i) {
    T acc = (T)0;
    for (int j = 0; j < nred; ++j)
      acc += src[i * nred + j];
    dst[i] = (T)((float)acc / nred);
  }
}

// =====================================================================
// ADDMM: out = alpha * (A @ B) + beta * bias
// lib_addmm<T>(out, bias, A, B, M, K, N, alpha, beta)
// =====================================================================

template <typename T>
inline void lib_addmm(T* out, const T* bias, const T* A, const T* B,
                       int M, int K, int N, float alpha, float beta) {
  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      float acc = 0.0f;
      for (int k = 0; k < K; ++k)
        acc += (float)A[m * K + k] * (float)B[k * N + n];
      out[m * N + n] = (T)(alpha * acc + beta * (float)bias[n]);
    }
  }
}

// =====================================================================
// Layer Norm: lib_layer_norm<T>(dst, src, weight, bias,
//                               batch, norm_size, eps)
// =====================================================================

template <typename T>
inline void lib_layer_norm(T* dst, const T* src, const T* weight,
                           const T* bias, int batch, int norm_size,
                           float eps) {
  for (int b = 0; b < batch; ++b) {
    const T* in = src + b * norm_size;
    T* out = dst + b * norm_size;

    float mean = 0.0f;
    for (int i = 0; i < norm_size; ++i)
      mean += (float)in[i];
    mean /= norm_size;

    float var = 0.0f;
    for (int i = 0; i < norm_size; ++i) {
      float d = (float)in[i] - mean;
      var += d * d;
    }
    var /= norm_size;
    float inv_std = 1.0f / std::sqrt(var + eps);

    for (int i = 0; i < norm_size; ++i)
      out[i] = (T)(((float)in[i] - mean) * inv_std * (float)weight[i] +
                    (float)bias[i]);
  }
}

// =====================================================================
// Conv2d: lib_conv2d<T>(out, input, weight, bias,
//                       batch, hi, wi, ci, co, kh, kw,
//                       stride_h, stride_w, pad_h, pad_w)
// =====================================================================

template <typename T>
inline void lib_conv2d(T* out, const T* input, const T* weight,
                       const T* bias, int batch, int hi, int wi, int ci,
                       int co, int kh, int kw, int stride_h, int stride_w,
                       int pad_h, int pad_w) {
  int ho = (hi + 2 * pad_h - kh) / stride_h + 1;
  int wo = (wi + 2 * pad_w - kw) / stride_w + 1;
  for (int n = 0; n < batch; ++n) {
    for (int oc = 0; oc < co; ++oc) {
      for (int oh = 0; oh < ho; ++oh) {
        for (int ow = 0; ow < wo; ++ow) {
          float acc = (bias != nullptr) ? (float)bias[oc] : 0.0f;
          for (int ic = 0; ic < ci; ++ic) {
            for (int r = 0; r < kh; ++r) {
              for (int s = 0; s < kw; ++s) {
                int ih = oh * stride_h - pad_h + r;
                int iw = ow * stride_w - pad_w + s;
                if (ih >= 0 && ih < hi && iw >= 0 && iw < wi) {
                  int in_idx = ((n * ci + ic) * hi + ih) * wi + iw;
                  int wt_idx = ((oc * ci + ic) * kh + r) * kw + s;
                  acc += (float)input[in_idx] * (float)weight[wt_idx];
                }
              }
            }
          }
          out[((n * co + oc) * ho + oh) * wo + ow] = (T)acc;
        }
      }
    }
  }
}

// =====================================================================
// Convert: lib_convert<DstT, SrcT>(dst, src, num)
// =====================================================================

template <typename DstT, typename SrcT>
inline void lib_convert(DstT* dst, const SrcT* src, int num) {
  for (int i = 0; i < num; ++i)
    dst[i] = (DstT)src[i];
}

// =====================================================================
// Pointwise:
//   lib_where<T>(dst, cond, x, y, num) -- dst[i] = cond[i] ? x[i] : y[i]
//   lib_lerp<T>(dst, start, end, weight, num)
// =====================================================================

template <typename T>
inline void lib_where(T* dst, const T* cond, const T* x, const T* y,
                      int num) {
  for (int i = 0; i < num; ++i)
    dst[i] = ((float)cond[i] != 0.0f) ? x[i] : y[i];
}

template <typename T>
inline void lib_lerp(T* dst, const T* start, const T* end, const T* w,
                     int num) {
  for (int i = 0; i < num; ++i) {
    float s = (float)start[i], e = (float)end[i], wt = (float)w[i];
    dst[i] = (T)(s + wt * (e - s));
  }
}

} // namespace choreo

#endif // __CHOREO_GCU_LIB_FALLBACK_H__
