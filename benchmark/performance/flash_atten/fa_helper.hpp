#pragma once

// Unified host-side benchmark helper for Choreo flash attention MHA kernels.
// Provides data prep, verification, timing, and TFLOPS reporting.
//
// Compile-time configuration:
//   MHA_DIM           head dimension (default 64)
//   MHA_DTYPE         choreo dtype, e.g. choreo::f16 or choreo::bf16 (default f16)
//   MHA_LAYOUT_BHSD   define to enable BHSD device layout with host transpose

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <functional>
#include <iostream>
#include <random>
#include <vector>

#ifndef MHA_DIM
#define MHA_DIM 64
#endif

#ifndef MHA_DTYPE
#define MHA_DTYPE choreo::f16
#endif

#ifndef MHA_ENABLE_VERIFY
#define MHA_ENABLE_VERIFY 1
#endif

#ifndef MHA_ENABLE_TIMING
#define MHA_ENABLE_TIMING 1
#endif

#ifndef H800_PCIE_PEAK_F16_TFLOPS
#define H800_PCIE_PEAK_F16_TFLOPS 1513
#endif

namespace mha_helper {

using scalar_t = MHA_DTYPE;

struct BenchConfig {
  int batch;
  int heads;
  int q_seq;
  int kv_seq;
  bool is_causal;
  const char* label;
};

struct TensorViews {
  choreo::spanned_view<scalar_t, 4> Q;
  choreo::spanned_view<scalar_t, 4> K;
  choreo::spanned_view<scalar_t, 4> V;
  choreo::spanned_view<scalar_t, 4> O;
};

namespace detail {

inline bool env_flag_enabled(const char* key) {
  const char* val = std::getenv(key);
  return val && val[0] == '1' && val[1] == '\0';
}

inline bool enable_verify() {
  if (env_flag_enabled("CHOREO_SKIP_VERIFY")) return false;
#if !MHA_ENABLE_VERIFY
  return false;
#else
  return true;
#endif
}

inline bool enable_timing() {
  if (env_flag_enabled("CHOREO_DISABLE_TIMING")) return false;
#if !MHA_ENABLE_TIMING
  return false;
#else
  return true;
#endif
}

inline choreo::TimerOption read_timer_options() {
  choreo::TimerOption opt;
  opt.warmup = 100;
  opt.repeat = 500;
  if (const char* warmup_env = std::getenv("CHOREO_TIMING_WARMUP")) {
    int value = std::atoi(warmup_env);
    if (value >= 0) opt.warmup = value;
  }
  if (const char* repeat_env = std::getenv("CHOREO_TIMING_REPEAT")) {
    int value = std::atoi(repeat_env);
    if (value > 0) opt.repeat = value;
  }
  return opt;
}

inline size_t verify_num_samples(size_t total_elems) {
  size_t num_samples = 4096;
  if (const char* env = std::getenv("MHA_VERIFY_SAMPLES")) {
    int value = std::atoi(env);
    if (value > 0) num_samples = static_cast<size_t>(value);
  }
  return total_elems < num_samples ? total_elems : num_samples;
}

inline size_t bshd_index(size_t b, size_t seq, size_t h, size_t d,
                         size_t seq_len, size_t H, size_t head_dim) {
  return (((b * seq_len + seq) * H + h) * head_dim + d);
}

inline float read_bshd(const scalar_t* data, size_t b, size_t seq, size_t h,
                       size_t d, size_t seq_len, size_t H, size_t head_dim) {
  return choreo::to_f32(
      data[bshd_index(b, seq, h, d, seq_len, H, head_dim)]);
}

inline void fill_random_bshd(choreo::spanned_data<scalar_t, 4>& tensor,
                             float lo, float hi, std::mt19937& gen) {
  std::uniform_real_distribution<float> dist(lo, hi);
  const size_t n = tensor.shape()[0] * tensor.shape()[1] * tensor.shape()[2] *
                   tensor.shape()[3];
  for (size_t i = 0; i < n; ++i) {
    tensor.data()[i] = scalar_t(dist(gen));
  }
}

inline double attention_flops(const BenchConfig& cfg) {
  const double batch = static_cast<double>(cfg.batch);
  const double heads = static_cast<double>(cfg.heads);
  const double q_seq = static_cast<double>(cfg.q_seq);
  const double kv_seq = static_cast<double>(cfg.kv_seq);
  const double dim = static_cast<double>(MHA_DIM);
  double flops = 2.0 * 2.0 * batch * heads * q_seq * kv_seq * dim;
  if (cfg.is_causal) flops *= 0.5;
  return flops;
}

inline float reference_output(const scalar_t* Q, const scalar_t* K,
                              const scalar_t* V, int b, int h, int qi,
                              int di, bool is_causal, int past_len, size_t H,
                              size_t q_seq, size_t kv_seq) {
  const float scale =
      (1.0f / std::sqrt(static_cast<float>(MHA_DIM))) * 1.44269504f;
  std::vector<float> scores(static_cast<size_t>(kv_seq), 0.0f);
  float row_max = -1e30f;

  for (int kj = 0; kj < static_cast<int>(kv_seq); ++kj) {
    if (is_causal && kj > qi + past_len) continue;
    float dot = 0.0f;
    for (int d = 0; d < MHA_DIM; ++d) {
      dot += read_bshd(Q, static_cast<size_t>(b), static_cast<size_t>(qi),
                       static_cast<size_t>(h), static_cast<size_t>(d), q_seq,
                       H, static_cast<size_t>(MHA_DIM)) *
             read_bshd(K, static_cast<size_t>(b), static_cast<size_t>(kj),
                       static_cast<size_t>(h), static_cast<size_t>(d), kv_seq,
                       H, static_cast<size_t>(MHA_DIM));
    }
    scores[static_cast<size_t>(kj)] = dot;
    row_max = std::max(row_max, dot);
  }

  float lsum = 0.0f;
  for (int kj = 0; kj < static_cast<int>(kv_seq); ++kj) {
    if (is_causal && kj > qi + past_len) continue;
    float prob = exp2f(scores[static_cast<size_t>(kj)] * scale -
                        row_max * scale);
    scores[static_cast<size_t>(kj)] = prob;
    lsum += prob;
  }

  float acc = 0.0f;
  for (int kj = 0; kj < static_cast<int>(kv_seq); ++kj) {
    if (is_causal && kj > qi + past_len) continue;
    acc += scores[static_cast<size_t>(kj)] *
           read_bshd(V, static_cast<size_t>(b), static_cast<size_t>(kj),
                     static_cast<size_t>(h), static_cast<size_t>(di), kv_seq,
                     H, static_cast<size_t>(MHA_DIM));
  }
  return acc / lsum;
}

inline bool verify_output(const BenchConfig& cfg, const scalar_t* Q_h,
                          const scalar_t* K_h, const scalar_t* V_h,
                          const scalar_t* O_h, size_t B, size_t H,
                          size_t q_seq, size_t head_dim) {
  const int past_len = cfg.kv_seq - cfg.q_seq;
  const size_t kv_seq = static_cast<size_t>(cfg.kv_seq);
  const size_t total = B * q_seq * H * head_dim;
  const size_t num_samples = verify_num_samples(total);
  const size_t stride =
      choreo::pick_coprime_stride(total, head_dim, num_samples);

  float max_abs_err = 0.0f;
  double sum_abs_err = 0.0;
  size_t checked = 0;
  size_t failed = 0;

  auto check_point = [&](int b, int h, int qi, int di) {
    const float ref = reference_output(Q_h, K_h, V_h, b, h, qi, di,
                                       cfg.is_causal, past_len, H, q_seq,
                                       kv_seq);
    const float got =
        read_bshd(O_h, static_cast<size_t>(b), static_cast<size_t>(qi),
                  static_cast<size_t>(h), static_cast<size_t>(di), q_seq, H,
                  head_dim);
    if (std::isnan(got) || std::isinf(got)) {
      ++failed;
      ++checked;
      max_abs_err = std::numeric_limits<float>::infinity();
      return;
    }
    const float err = std::abs(got - ref);
    const float tol = 0.05f + 0.1f * std::abs(ref);
    if (err > max_abs_err) max_abs_err = err;
    sum_abs_err += err;
    if (err > tol) ++failed;
    ++checked;
  };

  // Targeted check: last few query rows see the most KV tiles and are most
  // sensitive to online softmax rescaling correctness.
  {
    size_t tail_rows = std::min(q_seq, static_cast<size_t>(8));
    size_t tail_dims = std::min(head_dim, static_cast<size_t>(16));
    for (size_t b = 0; b < B; ++b)
      for (size_t h_idx = 0; h_idx < std::min(H, static_cast<size_t>(2));
           ++h_idx)
        for (size_t r = 0; r < tail_rows; ++r)
          for (size_t d = 0; d < tail_dims; ++d)
            check_point(static_cast<int>(b), static_cast<int>(h_idx),
                        static_cast<int>(q_seq - 1 - r), static_cast<int>(d));
  }

  // Strided random sampling.
  for (size_t idx = 0; idx < total && checked < num_samples; idx += stride) {
    const size_t flat = idx;
    const int di = static_cast<int>(flat % head_dim);
    const int qi = static_cast<int>((flat / head_dim) % q_seq);
    const int h = static_cast<int>((flat / (q_seq * head_dim)) % H);
    const int b = static_cast<int>(flat / (H * q_seq * head_dim));
    check_point(b, h, qi, di);
  }

  const float avg_abs_err =
      checked > 0 ? static_cast<float>(sum_abs_err / checked) : 0.0f;
  const float fail_rate =
      checked > 0 ? static_cast<float>(failed) / static_cast<float>(checked)
                  : 0.0f;
  const bool passed = (failed == 0);

  std::cout << "[VERIFY " << cfg.label << "] max_abs_err=" << max_abs_err
            << " avg_abs_err=" << avg_abs_err << " fail_rate=" << fail_rate
            << " (" << failed << "/" << checked << ") "
            << (passed ? "PASS" : "FAIL") << "\n";
  return passed;
}

#ifdef MHA_LAYOUT_BHSD
inline void transpose_bshd_to_bhsd(const scalar_t* src, scalar_t* dst,
                                   size_t B, size_t seq_len, size_t H,
                                   size_t head_dim) {
  for (size_t b = 0; b < B; ++b) {
    for (size_t h = 0; h < H; ++h) {
      for (size_t s = 0; s < seq_len; ++s) {
        for (size_t d = 0; d < head_dim; ++d) {
          dst[(((b * H + h) * seq_len + s) * head_dim + d)] =
              src[(((b * seq_len + s) * H + h) * head_dim + d)];
        }
      }
    }
  }
}

inline void transpose_bhsd_to_bshd(const scalar_t* src, scalar_t* dst,
                                   size_t B, size_t seq_len, size_t H,
                                   size_t head_dim) {
  for (size_t b = 0; b < B; ++b) {
    for (size_t h = 0; h < H; ++h) {
      for (size_t s = 0; s < seq_len; ++s) {
        for (size_t d = 0; d < head_dim; ++d) {
          dst[(((b * seq_len + s) * H + h) * head_dim + d)] =
              src[(((b * H + h) * seq_len + s) * head_dim + d)];
        }
      }
    }
  }
}

struct HostBhsdBuffers {
  std::vector<scalar_t> q;
  std::vector<scalar_t> k;
  std::vector<scalar_t> v;
  std::vector<scalar_t> o;
};

inline HostBhsdBuffers make_bhsd_host(
    const choreo::spanned_data<scalar_t, 4>& Q_h,
    const choreo::spanned_data<scalar_t, 4>& K_h,
    const choreo::spanned_data<scalar_t, 4>& V_h) {
  const size_t B = Q_h.shape()[0];
  const size_t Q_SEQ = Q_h.shape()[1];
  const size_t H = Q_h.shape()[2];
  const size_t KV_SEQ = K_h.shape()[1];
  const size_t head_dim = Q_h.shape()[3];

  HostBhsdBuffers buf;
  buf.q.resize(B * H * Q_SEQ * head_dim);
  buf.k.resize(B * H * KV_SEQ * head_dim);
  buf.v.resize(B * H * KV_SEQ * head_dim);
  buf.o.resize(B * H * Q_SEQ * head_dim);

  transpose_bshd_to_bhsd(Q_h.data(), buf.q.data(), B, Q_SEQ, H, head_dim);
  transpose_bshd_to_bhsd(K_h.data(), buf.k.data(), B, KV_SEQ, H, head_dim);
  transpose_bshd_to_bhsd(V_h.data(), buf.v.data(), B, KV_SEQ, H, head_dim);
  return buf;
}
#endif // MHA_LAYOUT_BHSD

struct DeviceBuffers {
  scalar_t* q_d = nullptr;
  scalar_t* k_d = nullptr;
  scalar_t* v_d = nullptr;
  scalar_t* o_d = nullptr;

  ~DeviceBuffers() {
    if (q_d) cudaFree(q_d);
    if (k_d) cudaFree(k_d);
    if (v_d) cudaFree(v_d);
    if (o_d) cudaFree(o_d);
  }
};

#ifndef MHA_LAYOUT_BHSD
inline TensorViews upload_tensors(
    const choreo::spanned_data<scalar_t, 4>& Q_h,
    const choreo::spanned_data<scalar_t, 4>& K_h,
    const choreo::spanned_data<scalar_t, 4>& V_h,
    choreo::spanned_data<scalar_t, 4>& O_h, DeviceBuffers& dev) {
  const size_t B = Q_h.shape()[0];
  const size_t Q_SEQ = Q_h.shape()[1];
  const size_t H = Q_h.shape()[2];
  const size_t KV_SEQ = K_h.shape()[1];
  const size_t head_dim = Q_h.shape()[3];

  const size_t q_bytes = B * Q_SEQ * H * head_dim * sizeof(scalar_t);
  const size_t kv_bytes = B * KV_SEQ * H * head_dim * sizeof(scalar_t);

  choreo::abend_true(cudaMalloc(&dev.q_d, q_bytes));
  choreo::abend_true(cudaMalloc(&dev.k_d, kv_bytes));
  choreo::abend_true(cudaMalloc(&dev.v_d, kv_bytes));
  choreo::abend_true(cudaMalloc(&dev.o_d, q_bytes));

  choreo::abend_true(cudaMemcpy(dev.q_d, Q_h.data(), q_bytes,
                                cudaMemcpyHostToDevice));
  choreo::abend_true(cudaMemcpy(dev.k_d, K_h.data(), kv_bytes,
                                cudaMemcpyHostToDevice));
  choreo::abend_true(cudaMemcpy(dev.v_d, V_h.data(), kv_bytes,
                                cudaMemcpyHostToDevice));
  choreo::abend_true(cudaMemcpy(dev.o_d, O_h.data(), q_bytes,
                                cudaMemcpyHostToDevice));
  choreo::abend_true(cudaDeviceSynchronize());

  return TensorViews{
      choreo::make_spanview<scalar_t, 4>(dev.q_d, {B, Q_SEQ, H, head_dim}),
      choreo::make_spanview<scalar_t, 4>(dev.k_d, {B, KV_SEQ, H, head_dim}),
      choreo::make_spanview<scalar_t, 4>(dev.v_d, {B, KV_SEQ, H, head_dim}),
      choreo::make_spanview<scalar_t, 4>(dev.o_d, {B, Q_SEQ, H, head_dim}),
  };
}

inline void download_output(choreo::spanned_data<scalar_t, 4>& O_h,
                            scalar_t* o_d) {
  const size_t bytes = O_h.bytes();
  choreo::abend_true(
      cudaMemcpy(O_h.data(), o_d, bytes, cudaMemcpyDeviceToHost));
  choreo::abend_true(cudaDeviceSynchronize());
}
#else // MHA_LAYOUT_BHSD
inline TensorViews upload_bhsd(
    const HostBhsdBuffers& host, size_t B, size_t Q_SEQ, size_t KV_SEQ,
    size_t H, size_t head_dim, DeviceBuffers& dev) {
  const size_t q_bytes = B * H * Q_SEQ * head_dim * sizeof(scalar_t);
  const size_t kv_bytes = B * H * KV_SEQ * head_dim * sizeof(scalar_t);

  choreo::abend_true(cudaMalloc(&dev.q_d, q_bytes));
  choreo::abend_true(cudaMalloc(&dev.k_d, kv_bytes));
  choreo::abend_true(cudaMalloc(&dev.v_d, kv_bytes));
  choreo::abend_true(cudaMalloc(&dev.o_d, q_bytes));

  choreo::abend_true(cudaMemcpy(dev.q_d, host.q.data(), q_bytes,
                                cudaMemcpyHostToDevice));
  choreo::abend_true(cudaMemcpy(dev.k_d, host.k.data(), kv_bytes,
                                cudaMemcpyHostToDevice));
  choreo::abend_true(cudaMemcpy(dev.v_d, host.v.data(), kv_bytes,
                                cudaMemcpyHostToDevice));
  choreo::abend_true(cudaMemcpy(dev.o_d, host.o.data(), q_bytes,
                                cudaMemcpyHostToDevice));
  choreo::abend_true(cudaDeviceSynchronize());

  return TensorViews{
      choreo::make_spanview<scalar_t, 4>(dev.q_d, {B, H, Q_SEQ, head_dim}),
      choreo::make_spanview<scalar_t, 4>(dev.k_d, {B, H, KV_SEQ, head_dim}),
      choreo::make_spanview<scalar_t, 4>(dev.v_d, {B, H, KV_SEQ, head_dim}),
      choreo::make_spanview<scalar_t, 4>(dev.o_d, {B, H, Q_SEQ, head_dim}),
  };
}
#endif // MHA_LAYOUT_BHSD

} // namespace detail

template <typename KernelFn>
inline int RunBenchmarks(const char* title, const BenchConfig* configs,
                         size_t num_configs, KernelFn kernel_fn) {
  const bool do_verify = detail::enable_verify();
  const bool do_timing = detail::enable_timing();
  const choreo::TimerOption timer_opt = detail::read_timer_options();

#ifdef MHA_LAYOUT_BHSD
  std::cout << title << " (DIM=" << MHA_DIM << ", BSHD host I/O)\n";
  std::cout << "Choreo kernel: BHSD device tiles; host BSHD<->BHSD once per "
               "config\n";
  std::cout << "Timing: device kernel only\n";
#else
  std::cout << title << " (DIM=" << MHA_DIM << ", BSHD host and device)\n";
  std::cout << "Layout: [batch, seqlen, heads, dim] (matches FA3 baseline)\n";
#endif
  if (do_timing) {
    std::cout << "Warmup=" << timer_opt.warmup
              << " Repeat=" << timer_opt.repeat << "\n";
  }
  std::cout << "--------------------------------------------\n";

  std::mt19937 gen(42);
  bool all_passed = true;

  for (size_t ci = 0; ci < num_configs; ++ci) {
    const BenchConfig& cfg = configs[ci];
    std::cout << cfg.label << "\n";

    auto Q_h = choreo::make_spandata<scalar_t>(
        static_cast<size_t>(cfg.batch), static_cast<size_t>(cfg.q_seq),
        static_cast<size_t>(cfg.heads), static_cast<size_t>(MHA_DIM));
    auto K_h = choreo::make_spandata<scalar_t>(
        static_cast<size_t>(cfg.batch), static_cast<size_t>(cfg.kv_seq),
        static_cast<size_t>(cfg.heads), static_cast<size_t>(MHA_DIM));
    auto V_h = choreo::make_spandata<scalar_t>(
        static_cast<size_t>(cfg.batch), static_cast<size_t>(cfg.kv_seq),
        static_cast<size_t>(cfg.heads), static_cast<size_t>(MHA_DIM));
    auto O_h = choreo::make_spandata<scalar_t>(
        static_cast<size_t>(cfg.batch), static_cast<size_t>(cfg.q_seq),
        static_cast<size_t>(cfg.heads), static_cast<size_t>(MHA_DIM));

    detail::fill_random_bshd(Q_h, -2.0f, 2.0f, gen);
    detail::fill_random_bshd(K_h, -2.0f, 2.0f, gen);
    detail::fill_random_bshd(V_h, -1.0f, 1.0f, gen);
    O_h.fill(0.0f);

    detail::DeviceBuffers dev;

#ifdef MHA_LAYOUT_BHSD
    const size_t B = Q_h.shape()[0];
    const size_t Q_SEQ = Q_h.shape()[1];
    const size_t H = Q_h.shape()[2];
    const size_t KV_SEQ = K_h.shape()[1];
    const size_t head_dim = Q_h.shape()[3];

    detail::HostBhsdBuffers bhsd_h =
        detail::make_bhsd_host(Q_h, K_h, V_h);
    TensorViews views = detail::upload_bhsd(bhsd_h, B, Q_SEQ, KV_SEQ, H,
                                            head_dim, dev);
#else
    TensorViews views =
        detail::upload_tensors(Q_h, K_h, V_h, O_h, dev);
#endif

    auto launch_kernel = [&]() { kernel_fn(cfg, views); };

    auto launch_and_sync = [&]() {
      launch_kernel();
      choreo::abend_true(cudaDeviceSynchronize());
    };

#ifdef MHA_LAYOUT_BHSD
    auto fetch_output = [&]() {
      const size_t q_bytes = B * H * Q_SEQ * head_dim * sizeof(scalar_t);
      choreo::abend_true(cudaMemcpy(bhsd_h.o.data(), dev.o_d, q_bytes,
                                    cudaMemcpyDeviceToHost));
      choreo::abend_true(cudaDeviceSynchronize());
      detail::transpose_bhsd_to_bshd(bhsd_h.o.data(), O_h.data(), B, Q_SEQ, H,
                                     head_dim);
    };
#endif

    if (do_timing) {
      const double avg_ms = choreo::timing(launch_kernel, timer_opt);
      const double flops = detail::attention_flops(cfg);
      const double tflops = flops / (avg_ms / 1000.0) / 1e12;
      const double eff =
          (tflops / static_cast<double>(H800_PCIE_PEAK_F16_TFLOPS)) * 100.0;
      std::cout << "Timing avg ms: " << avg_ms << "\n";
      std::cout << "TFLOPS: " << tflops << "\n";
      std::cout << "HW efficiency: " << eff << "%\n";
    } else {
      launch_and_sync();
    }

    if (do_verify) {
      if (!do_timing) launch_and_sync();
#ifdef MHA_LAYOUT_BHSD
      fetch_output();
#else
      detail::download_output(O_h, dev.o_d);
#endif
      const bool passed = detail::verify_output(
          cfg, Q_h.data(), K_h.data(), V_h.data(), O_h.data(), Q_h.shape()[0],
          Q_h.shape()[2], Q_h.shape()[1], Q_h.shape()[3]);
      all_passed = all_passed && passed;
    }

    if (ci + 1 < num_configs)
      std::cout << "--------------------------------------------\n";
  }

  if (!do_verify) {
    std::cout << "Test Passed (verify skipped)\n";
    return 0;
  }

  if (all_passed) {
    std::cout << "Test Passed\n";
    return 0;
  }

  std::cout << "Test FAILED\n";
  return 1;
}

} // namespace mha_helper
