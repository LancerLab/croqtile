
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

#include "cutlass/cutlass.h"

#ifdef __CUDACC__
#pragma nv_diag_suppress 20054
#endif
// include the choreo header;
#define __CHOREO_ENABLE_CUDA_RUNTIME_ENV_CHECK__
#define __CHOREO_REQUIRED_GPU_DEVICE_SM__ 90
#include "choreo.h"
namespace cde = cuda::device::experimental;
#include <cooperative_groups.h>
using namespace choreo;

// Warp-specialized 1p2c with 2-stage pipeline, combined KV events
// + Skip causal mask for tiles that are entirely below the diagonal
// + launch_bounds(1)
// D=128, bf16, causal prefill
//
// Optimization: For tiles where the entire BLOCK_N column range is below
// the causal diagonal for ALL rows in this consumer's WG_M slice, skip
// the expensive per-element mask entirely. Only apply per-element masking
// on the "boundary" tile where the diagonal intersects this consumer's rows.

#define BLOCK_M 128
#define BLOCK_N 128
#define WG_M 64
#define WG_K 16
#define SWIZ 128
#define STAGES 2

#define DIM 128

__global__ __launch_bounds__(384, 1) void __choreo_device_flash_atten(bf16 * Q, bf16 * K, bf16 * V, bf16 * O, float scale, unsigned B, unsigned H, unsigned KV_SEQ, unsigned Q_SEQ, const __grid_constant__ CUtensorMap __choreo_tma_0_tensor_map, const __grid_constant__ CUtensorMap __choreo_tma_1_tensor_map, const __grid_constant__ CUtensorMap __choreo_tma_2_tensor_map) {
  extern __shared__ char __choreo_device_flash_atten__runtime_shared_buffer__raw[];
  auto __choreo_device_flash_atten__runtime_shared_buffer__ = reinterpret_cast<char*>(aligned_up_ptr<128 * 8>(__choreo_device_flash_atten__runtime_shared_buffer__raw));
  { // parallel-by: /home/fem/dev/choreo/benchmark/performance/flash_atten/causal_prefill_d128/choreo/best.co:29.12
  auto anon_2 = (unsigned char*)__choreo_device_flash_atten__runtime_shared_buffer__;
  [[maybe_unused]] auto Q_head = (H * Q_SEQ * (blockIdx.z * 128) + blockIdx.y * 128 + Q);
  [[maybe_unused]] auto K_head = (H * KV_SEQ * (blockIdx.z * 128) + blockIdx.y * 128 + K);
  [[maybe_unused]] auto V_head = (H * KV_SEQ * (blockIdx.z * 128) + blockIdx.y * 128 + V);
  [[maybe_unused]] auto O_head = (H * Q_SEQ * (blockIdx.z * 128) + blockIdx.y * 128 + O);
  bf16* q_shared = (bf16*)(anon_2 + 131072);
  bf16* k_buf = (bf16*)(anon_2 + 0);
  bf16* v_buf = (bf16*)(anon_2 + 65536);
  __shared__ cuda::barrier<cuda::thread_scope_block> qf; // shared event barrier
  __shared__ cuda::barrier<cuda::thread_scope_block> kvf[2]; // shared event barrier
  __shared__ cuda::barrier<cuda::thread_scope_block> kve[2]; // shared event barrier
  if (__CHOREO_BLOCK_SINGLE__) {
    init(&qf, 257);
    init(&kvf[0], 257);
    init(&kvf[1], 257);
    init(&kve[0], 257);
    init(&kve[1], 257);
    cde::fence_proxy_async_shared_cta();
  }
  __syncthreads();
  int kv_bound = (choreo::nv_cute::numerics::min)(((blockIdx.x + 1) * 128 + (KV_SEQ - Q_SEQ) + 128 - 1) / 128, ((KV_SEQ + 127) / 128));
  [[maybe_unused]] auto __choreo_vg4id_x = threadIdx.x / 128;
  [[maybe_unused]] auto __choreo_vtid_x = threadIdx.x % 128;
  // inthreads: /home/fem/dev/choreo/benchmark/performance/flash_atten/causal_prefill_d128/choreo/best.co:44.7
  if ((__choreo_vg4id_x == 0 && __choreo_vtid_x == 0)) {
    cde::cp_async_bulk_tensor_3d_global_to_shared(q_shared, &__choreo_tma_0_tensor_map, ((H * blockIdx.x * 16384 + (H * Q_SEQ * (blockIdx.z * 128) + blockIdx.y * 128)) % (H * 128) % 64), ((H * blockIdx.x * 16384 + (H * Q_SEQ * (blockIdx.z * 128) + blockIdx.y * 128)) / (H * 128)), ((H * blockIdx.x * 16384 + (H * Q_SEQ * (blockIdx.z * 128) + blockIdx.y * 128)) % (H * 128) / 64), qf);
    (void)cuda::device::barrier_arrive_tx(qf, 1, (32768)); // trigger event(barrier)
    // with-in: /home/fem/dev/choreo/benchmark/performance/flash_atten/causal_prefill_d128/choreo/best.co:49.9
    {
      int __iv_bn = 0;
      // foreach: /home/fem/dev/choreo/benchmark/performance/flash_atten/causal_prefill_d128/choreo/best.co:49.9
      for (__iv_bn = 0; __iv_bn < ((KV_SEQ - Q_SEQ + (blockIdx.x + 1) * 128 + 127) / 128 < (KV_SEQ + 127) / 128 ? (KV_SEQ - Q_SEQ + (blockIdx.x + 1) * 128 + 127) / 128 : (KV_SEQ + 127) / 128); ++__iv_bn) {
        int stage = __iv_bn % 2;
        // wait event(barrier)  (kve elemof stage) 
        kve[stage].wait(kve[stage].arrive());
        cde::cp_async_bulk_tensor_3d_global_to_shared(k_buf + (stage)*(16384), &__choreo_tma_1_tensor_map, ((H * KV_SEQ * (blockIdx.z * 128) + blockIdx.y * 128 + H * __iv_bn * 16384) % (H * 128) % 64), ((H * KV_SEQ * (blockIdx.z * 128) + blockIdx.y * 128 + H * __iv_bn * 16384) / (H * 128)), ((H * KV_SEQ * (blockIdx.z * 128) + blockIdx.y * 128 + H * __iv_bn * 16384) % (H * 128) / 64), kvf[stage]);
        cde::cp_async_bulk_tensor_3d_global_to_shared(v_buf + (stage)*(16384), &__choreo_tma_2_tensor_map, ((H * KV_SEQ * (blockIdx.z * 128) + blockIdx.y * 128 + H * __iv_bn * 16384) % (H * 128) % 64), ((H * KV_SEQ * (blockIdx.z * 128) + blockIdx.y * 128 + H * __iv_bn * 16384) / (H * 128)), ((H * KV_SEQ * (blockIdx.z * 128) + blockIdx.y * 128 + H * __iv_bn * 16384) % (H * 128) / 64), kvf[stage]);
        // trigger event(barrier)  (kvf elemof stage) 
        (void)cuda::device::barrier_arrive_tx(kvf[stage], 1, (32768) + (32768));
      } // bn
      __iv_bn = 0;
    }
  } // end inthreads
  // inthreads: /home/fem/dev/choreo/benchmark/performance/flash_atten/causal_prefill_d128/choreo/best.co:60.7
  if ((__choreo_vg4id_x > 0)) {
    int cid = __choreo_vg4id_x - 1;
    // with-in: /home/fem/dev/choreo/benchmark/performance/flash_atten/causal_prefill_d128/choreo/best.co:62.9
    {
      int __iv_s = 0;
      // foreach: /home/fem/dev/choreo/benchmark/performance/flash_atten/causal_prefill_d128/choreo/best.co:62.9
      for (__iv_s = 0; __iv_s < 2; ++__iv_s) {
        // trigger event(barrier)  (kve elemof s) 
        (void)kve[__iv_s].arrive();
      } // s
      __iv_s = 0;
    }
    qf.wait(qf.arrive()); // wait event(barrier)
    float scores_max[2];
    for (int __frag_init = 0; __frag_init < 2; ++__frag_init)
      scores_max[__frag_init] = (-INFINITY);
    float scores_max_prev[2];
    float scores_scale[2];
    float scores_sum[2];
    float logsum[2];
    for (int __frag_init = 0; __frag_init < 2; ++__frag_init)
      logsum[__frag_init] = 0.000000f;
    bf16 acc_s_cast[64];
    float acc_o[64];
    float __frag_init_val0 = 0.000000f;
    for (int idx = 0; idx < 64; ++idx)
      acc_o[idx] = __frag_init_val0;
    // wait event(barrier)  (kvf elemof 0) 
    kvf[0].wait(kvf[0].arrive());
    float acc_s[64];
    float __frag_init_val1 = 0.000000f;
    for (int idx = 0; idx < 64; ++idx)
      acc_s[idx] = __frag_init_val1;
    warpgroup_fence_operand(acc_s);
    warpgroup_arrive();
    // Note: warpgroup_arrive() should be called once before first WGMMA
    // and warpgroup_wait() should be called once after all WGMMAs
    for (int __choreo_wgmma_k_iter = 0; __choreo_wgmma_k_iter < 8; ++__choreo_wgmma_k_iter) {
      auto* __choreo_wgmma_ptr_0_iter = (bf16*)((bf16*)((q_shared + ((__choreo_vg4id_x - 1) * 4096))) + (((__choreo_wgmma_k_iter) * 16 & 63) + ((__choreo_wgmma_k_iter) * 16 / 64) * 8192));
      uint64_t __choreo_wgmma_desc_0_iter = wgmma_make_smem_desc<WGMMA_MajorOrder::K_MAJOR, WGMMA_Swizzle::B128>(__choreo_wgmma_ptr_0_iter);
      auto* __choreo_wgmma_ptr_1_iter = (bf16*)((bf16*)(k_buf + (0)*(16384)) + (((__choreo_wgmma_k_iter) * 16 & 63) + ((__choreo_wgmma_k_iter) * 16 / 64) * 8192));
      uint64_t __choreo_wgmma_desc_1_iter = wgmma_make_smem_desc<WGMMA_MajorOrder::K_MAJOR, WGMMA_Swizzle::B128>(__choreo_wgmma_ptr_1_iter);
      cute::SM90::GMMA::MMA_64x128x16_F32BF16BF16_SS<static_cast<cute::SM90::GMMA::Major>(0), static_cast<cute::SM90::GMMA::Major>(0)>::fma(__choreo_wgmma_desc_0_iter, __choreo_wgmma_desc_1_iter, acc_s[0], acc_s[1], acc_s[2], acc_s[3], acc_s[4], acc_s[5], acc_s[6], acc_s[7], acc_s[8], acc_s[9], acc_s[10], acc_s[11], acc_s[12], acc_s[13], acc_s[14], acc_s[15], acc_s[16], acc_s[17], acc_s[18], acc_s[19], acc_s[20], acc_s[21], acc_s[22], acc_s[23], acc_s[24], acc_s[25], acc_s[26], acc_s[27], acc_s[28], acc_s[29], acc_s[30], acc_s[31], acc_s[32], acc_s[33], acc_s[34], acc_s[35], acc_s[36], acc_s[37], acc_s[38], acc_s[39], acc_s[40], acc_s[41], acc_s[42], acc_s[43], acc_s[44], acc_s[45], acc_s[46], acc_s[47], acc_s[48], acc_s[49], acc_s[50], acc_s[51], acc_s[52], acc_s[53], acc_s[54], acc_s[55], acc_s[56], acc_s[57], acc_s[58], acc_s[59], acc_s[60], acc_s[61], acc_s[62], acc_s[63]);
    }
    warpgroup_commit_batch();
    warpgroup_fence_operand(acc_s);
    warpgroup_wait<0>();
    // with-in: /home/fem/dev/choreo/benchmark/performance/flash_atten/causal_prefill_d128/choreo/best.co:83.9
    {
      int __iv_bn = 0;
      // foreach: /home/fem/dev/choreo/benchmark/performance/flash_atten/causal_prefill_d128/choreo/best.co:83.9
      for (__iv_bn = 0; __iv_bn < ((KV_SEQ - Q_SEQ + (blockIdx.x + 1) * 128 + 127) / 128 < (KV_SEQ + 127) / 128 ? (KV_SEQ - Q_SEQ + (blockIdx.x + 1) * 128 + 127) / 128 : (KV_SEQ + 127) / 128); ++__iv_bn) {
        auto stage = __iv_bn % 2;
        int causal_row_base = (KV_SEQ - Q_SEQ + (blockIdx.x * 128 + (__choreo_vg4id_x - 1) * 64));
        // if-else: /home/fem/dev/choreo/benchmark/performance/flash_atten/causal_prefill_d128/choreo/best.co:88.11
        if (((__iv_bn + 1) * 128 > causal_row_base + 1)) {
          { // frag.apply acc_s
            int __frag_iv_i = 0;
            int __frag_iv_j = 0;
            #pragma unroll
            for (int __r = 0; __r < 64; ++__r) {
              __frag_iv_i = ((threadIdx.x % 128) / 32 * 16 + ((__r) % 4) / 2 * 8 + (threadIdx.x % 128) % 32 / 4);
              __frag_iv_j = ((__r) / 4 * 8 + (threadIdx.x % 128) % 32 % 4 * 2 + (__r) % 2);
              acc_s[__r] = __iv_bn * 128 + __frag_iv_j > blockIdx.x * 128 + cid * 64 + __frag_iv_i + (KV_SEQ - Q_SEQ) ? (-INFINITY) : acc_s[__r];
            }
          } // frag.apply
        } // end if-else: /home/fem/dev/choreo/benchmark/performance/flash_atten/causal_prefill_d128/choreo/best.co:88.11
        { // frag.copy scores_max_prev, scores_max
          #pragma unroll
          for (int __r = 0; __r < 2; ++__r)
            scores_max_prev[__r] = scores_max[__r];
        } // frag.copy
        { // frag.reduce_max acc_s -> scores_max
          #pragma unroll
          for (int __row = 0; __row < 2; ++__row) {
            float __local_reduce = __int_as_float(0xff800000) /*-inf*/;
            #pragma unroll
            for (int __rv = 0; __rv < 32; ++__rv) {
              __local_reduce = fmaxf(__local_reduce, acc_s[(((__rv / 2) * 4) + (__row * 2) + (__rv % 2))]);
            }
            {
              float __shfl_v0 = __shfl_xor_sync(0xffffffff, __local_reduce, 2);
              __local_reduce = fmaxf(__local_reduce, __shfl_v0);
            }
            {
              float __shfl_v1 = __shfl_xor_sync(0xffffffff, __local_reduce, 1);
              __local_reduce = fmaxf(__local_reduce, __shfl_v1);
            }
            scores_max[__row] = __local_reduce;
          }
        } // frag.reduce_max
        { // frag.apply scores_max
          #pragma unroll
          for (int __r = 0; __r < 2; ++__r) {
            scores_max[__r] = (choreo::nv_cute::numerics::max)(scores_max[__r], scores_max_prev[__r]);
          }
        } // frag.apply
        { // frag.apply scores_scale
          #pragma unroll
          for (int __r = 0; __r < 2; ++__r) {
            scores_scale[__r] = choreo::nv_cute::numerics::exp2f(scores_max_prev[__r] * scale - scores_max[__r] * scale);
          }
        } // frag.apply
        { // frag.apply acc_s
          #pragma unroll
          for (int __r = 0; __r < 64; ++__r) {
            acc_s[__r] = choreo::nv_cute::numerics::exp2f(acc_s[__r] * scale - scores_max[(((__r) & 3) >> 1)] * scale);
          }
        } // frag.apply
        { // frag.reduce_sum acc_s -> scores_sum
          #pragma unroll
          for (int __row = 0; __row < 2; ++__row) {
            float __local_reduce = 0.0f;
            #pragma unroll
            for (int __rv = 0; __rv < 32; ++__rv) {
              __local_reduce += acc_s[(((__rv / 2) * 4) + (__row * 2) + (__rv % 2))];
            }
            {
              float __shfl_v0 = __shfl_xor_sync(0xffffffff, __local_reduce, 2);
              __local_reduce = __local_reduce + __shfl_v0;
            }
            {
              float __shfl_v1 = __shfl_xor_sync(0xffffffff, __local_reduce, 1);
              __local_reduce = __local_reduce + __shfl_v1;
            }
            scores_sum[__row] = __local_reduce;
          }
        } // frag.reduce_sum
        { // frag.apply logsum
          #pragma unroll
          for (int __r = 0; __r < 2; ++__r) {
            logsum[__r] = logsum[__r] * scores_scale[__r] + scores_sum[__r];
          }
        } // frag.apply
        { // frag.apply acc_s_cast
          #pragma unroll
          for (int __r = 0; __r < 64; ++__r) {
            acc_s_cast[__r] = choreo::f32_to_bf16(acc_s[__r]);
          }
        } // frag.apply
        { // frag.apply acc_o
          #pragma unroll
          for (int __r = 0; __r < 64; ++__r) {
            acc_o[__r] = acc_o[__r] * scores_scale[(((__r) & 3) >> 1)];
          }
        } // frag.apply
        warpgroup_fence_operand(acc_o);
        warpgroup_arrive();
        // Note: warpgroup_arrive() should be called once before first WGMMA
        // and warpgroup_wait() should be called once after all WGMMAs
        warpgroup_fence_operand(acc_s_cast);
        for (int __choreo_wgmma_k_iter = 0; __choreo_wgmma_k_iter < 8; ++__choreo_wgmma_k_iter) {
          auto* __choreo_wgmma_ptr_2_iter = (bf16*)((bf16*)(v_buf + (stage)*(16384)) + ((__choreo_wgmma_k_iter) * 1024));
          uint64_t __choreo_wgmma_desc_2_iter = wgmma_make_smem_desc<WGMMA_MajorOrder::MN_MAJOR, WGMMA_Swizzle::B128, 16384>(__choreo_wgmma_ptr_2_iter);
          cute::SM90::GMMA::MMA_64x128x16_F32BF16BF16_RS<static_cast<cute::SM90::GMMA::Major>(0), static_cast<cute::SM90::GMMA::Major>(1)>::fma(reinterpret_cast<const uint32_t*>(acc_s_cast + __choreo_wgmma_k_iter * 8)[0], reinterpret_cast<const uint32_t*>(acc_s_cast + __choreo_wgmma_k_iter * 8)[1], reinterpret_cast<const uint32_t*>(acc_s_cast + __choreo_wgmma_k_iter * 8)[2], reinterpret_cast<const uint32_t*>(acc_s_cast + __choreo_wgmma_k_iter * 8)[3], __choreo_wgmma_desc_2_iter, acc_o[0], acc_o[1], acc_o[2], acc_o[3], acc_o[4], acc_o[5], acc_o[6], acc_o[7], acc_o[8], acc_o[9], acc_o[10], acc_o[11], acc_o[12], acc_o[13], acc_o[14], acc_o[15], acc_o[16], acc_o[17], acc_o[18], acc_o[19], acc_o[20], acc_o[21], acc_o[22], acc_o[23], acc_o[24], acc_o[25], acc_o[26], acc_o[27], acc_o[28], acc_o[29], acc_o[30], acc_o[31], acc_o[32], acc_o[33], acc_o[34], acc_o[35], acc_o[36], acc_o[37], acc_o[38], acc_o[39], acc_o[40], acc_o[41], acc_o[42], acc_o[43], acc_o[44], acc_o[45], acc_o[46], acc_o[47], acc_o[48], acc_o[49], acc_o[50], acc_o[51], acc_o[52], acc_o[53], acc_o[54], acc_o[55], acc_o[56], acc_o[57], acc_o[58], acc_o[59], acc_o[60], acc_o[61], acc_o[62], acc_o[63]);
        }
        warpgroup_commit_batch();
        warpgroup_fence_operand(acc_o);
        // trigger event(barrier)  (kve elemof stage) 
        (void)kve[stage].arrive();
        // if-else: /home/fem/dev/choreo/benchmark/performance/flash_atten/causal_prefill_d128/choreo/best.co:112.11
        if ((__iv_bn + 1 < ((KV_SEQ - Q_SEQ + (blockIdx.x + 1) * 128 + 127) / 128 < (KV_SEQ + 127) / 128 ? (KV_SEQ - Q_SEQ + (blockIdx.x + 1) * 128 + 127) / 128 : (KV_SEQ + 127) / 128))) {
          // wait event(barrier)  (kvf elemof  ( (bn + 1)  % 2) ) 
          kvf[(__iv_bn + 1) % 2].wait(kvf[(__iv_bn + 1) % 2].arrive());
          float acc_s[64];
          float __frag_init_val2 = 0.000000f;
          for (int idx = 0; idx < 64; ++idx)
            acc_s[idx] = __frag_init_val2;
          warpgroup_fence_operand(acc_s);
          warpgroup_arrive();
          // Note: warpgroup_arrive() should be called once before first WGMMA
          // and warpgroup_wait() should be called once after all WGMMAs
          for (int __choreo_wgmma_k_iter = 0; __choreo_wgmma_k_iter < 8; ++__choreo_wgmma_k_iter) {
            auto* __choreo_wgmma_ptr_3_iter = (bf16*)((bf16*)((q_shared + ((__choreo_vg4id_x - 1) * 4096))) + (((__choreo_wgmma_k_iter) * 16 & 63) + ((__choreo_wgmma_k_iter) * 16 / 64) * 8192));
            uint64_t __choreo_wgmma_desc_3_iter = wgmma_make_smem_desc<WGMMA_MajorOrder::K_MAJOR, WGMMA_Swizzle::B128>(__choreo_wgmma_ptr_3_iter);
            auto* __choreo_wgmma_ptr_4_iter = (bf16*)((bf16*)(k_buf + ((__iv_bn + 1) % 2)*(16384)) + (((__choreo_wgmma_k_iter) * 16 & 63) + ((__choreo_wgmma_k_iter) * 16 / 64) * 8192));
            uint64_t __choreo_wgmma_desc_4_iter = wgmma_make_smem_desc<WGMMA_MajorOrder::K_MAJOR, WGMMA_Swizzle::B128>(__choreo_wgmma_ptr_4_iter);
            cute::SM90::GMMA::MMA_64x128x16_F32BF16BF16_SS<static_cast<cute::SM90::GMMA::Major>(0), static_cast<cute::SM90::GMMA::Major>(0)>::fma(__choreo_wgmma_desc_3_iter, __choreo_wgmma_desc_4_iter, acc_s[0], acc_s[1], acc_s[2], acc_s[3], acc_s[4], acc_s[5], acc_s[6], acc_s[7], acc_s[8], acc_s[9], acc_s[10], acc_s[11], acc_s[12], acc_s[13], acc_s[14], acc_s[15], acc_s[16], acc_s[17], acc_s[18], acc_s[19], acc_s[20], acc_s[21], acc_s[22], acc_s[23], acc_s[24], acc_s[25], acc_s[26], acc_s[27], acc_s[28], acc_s[29], acc_s[30], acc_s[31], acc_s[32], acc_s[33], acc_s[34], acc_s[35], acc_s[36], acc_s[37], acc_s[38], acc_s[39], acc_s[40], acc_s[41], acc_s[42], acc_s[43], acc_s[44], acc_s[45], acc_s[46], acc_s[47], acc_s[48], acc_s[49], acc_s[50], acc_s[51], acc_s[52], acc_s[53], acc_s[54], acc_s[55], acc_s[56], acc_s[57], acc_s[58], acc_s[59], acc_s[60], acc_s[61], acc_s[62], acc_s[63]);
          }
          warpgroup_commit_batch();
          warpgroup_fence_operand(acc_s);
          warpgroup_wait<0>();
        } else {
          warpgroup_wait<0>();
        } // end if-else: /home/fem/dev/choreo/benchmark/performance/flash_atten/causal_prefill_d128/choreo/best.co:112.11
      } // bn
      __iv_bn = 0;
    }
    { // frag.apply acc_o
      #pragma unroll
      for (int __r = 0; __r < 64; ++__r) {
        acc_o[__r] = acc_o[__r] / logsum[(((__r) & 3) >> 1)];
      }
    } // frag.apply
    [[maybe_unused]] auto anon_1 = (blockIdx.x * 2 + (__choreo_vg4id_x - 1));
    { int __rg = ((int)Q_SEQ - (int)(anon_1) * 64);
      // stmatrix path: write f32 acc_o to shared as bf16, then vectorized global copy
      auto __smem_shape = cute::make_shape(cute::Int<64>{}, cute::Int<128>{});
      auto __smem_stride = cute::make_stride(cute::Int<128>{}, cute::Int<1>{});
      auto __smem_layout = cute::make_layout(__smem_shape, __smem_stride);
      auto __smem_tensor = cute::make_tensor(cute::make_smem_ptr<bf16>(v_buf + cid * 16384), __smem_layout);

      if (__rg >= 64) {
        store_fragment_d_stmatrix_f32_bf16<CUTE_WGMMA_M64K16, 128>(__smem_tensor, reinterpret_cast<float*>(acc_o));
        // v_buf now contains 64x128 bf16 in row-major layout
        // Vectorized copy to global: each of 128 threads handles half a row
        int tid = threadIdx.x % 128;
        int row = tid / 2;
        int half = tid % 2;
        bf16* out_row = (bf16*)O_head + (H * (anon_1 * 64 + row) * 128);
        bf16* smem_row = v_buf + cid * 16384 + row * 128;
        // 64 bf16 = 128 bytes per half = 1 int4 (128-bit) * 8
        int4* src = (int4*)(smem_row + half * 64);
        int4* dst = (int4*)(out_row + half * 64);
        #pragma unroll
        for (int i = 0; i < 8; ++i)
          dst[i] = src[i];
      } else {
        auto __shape1_O_head = cute::make_shape(cute::Int<64>{}, cute::Int<128>{});
        auto __stride1_O_head = cute::make_stride((H * cute::Int<128>{}), cute::Int<1>{});
        auto __layout1_O_head = cute::make_layout(__shape1_O_head, __stride1_O_head);
        auto __tensor1_O_head = cute::make_tensor(cute::make_gmem_ptr<bf16>((bf16*)O_head + (H * anon_1 * 8192)), __layout1_O_head);
        store_fragment_d_mask_row<CUTE_WGMMA_M64K16, 128>(__tensor1_O_head, reinterpret_cast<float*>(acc_o), __rg);
      }
    }
  } // end inthreads
  } // end parallel-by
}

void flash_atten(const choreo::spanned_view<choreo::bf16, 4> & Q, const choreo::spanned_view<choreo::bf16, 4> & K, const choreo::spanned_view<choreo::bf16, 4> & V, const choreo::spanned_view<choreo::bf16, 4> & O) {
  __choreo_check_cuda_environment__();
  auto &B = Q.shape()[0];
  auto &H = Q.shape()[2];
  auto &KV_SEQ = K.shape()[1];
  auto &Q_SEQ = Q.shape()[1];
  float scale = 0.127517f;
  uint64_t __choreo_tma_0_shape[] = {64, (B * H * Q_SEQ * 128 / (H * 128)), (H * 2)};
  uint64_t __choreo_tma_0_strides[] = {(H * 256), 128};
  uint32_t __choreo_tma_0_box_shape[] = {(uint32_t)(64), (uint32_t)(128), (uint32_t)(2)};
  uint32_t __choreo_tma_0_elem_strides[] = {1, 1, 1};
  alignas(64) CUtensorMap __choreo_tma_0_tensor_map{};
  CUresult __choreo_tma_0_tensor_map_res = cuTensorMapEncodeTiled(
          &__choreo_tma_0_tensor_map,
          CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
          3,
          Q.data(),
          __choreo_tma_0_shape,
          __choreo_tma_0_strides,
          __choreo_tma_0_box_shape,
          __choreo_tma_0_elem_strides,
          CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
          CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B,
          CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
          CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  choreo::abend_true(__choreo_tma_0_tensor_map_res != CUDA_SUCCESS);
  uint64_t __choreo_tma_1_shape[] = {64, (B * H * KV_SEQ * 128 / (H * 128)), (H * 2)};
  uint64_t __choreo_tma_1_strides[] = {(H * 256), 128};
  uint32_t __choreo_tma_1_box_shape[] = {(uint32_t)(64), (uint32_t)(128), (uint32_t)(2)};
  uint32_t __choreo_tma_1_elem_strides[] = {1, 1, 1};
  alignas(64) CUtensorMap __choreo_tma_1_tensor_map{};
  CUresult __choreo_tma_1_tensor_map_res = cuTensorMapEncodeTiled(
          &__choreo_tma_1_tensor_map,
          CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
          3,
          K.data(),
          __choreo_tma_1_shape,
          __choreo_tma_1_strides,
          __choreo_tma_1_box_shape,
          __choreo_tma_1_elem_strides,
          CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
          CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B,
          CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
          CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  choreo::abend_true(__choreo_tma_1_tensor_map_res != CUDA_SUCCESS);
  uint64_t __choreo_tma_2_shape[] = {64, (B * H * KV_SEQ * 128 / (H * 128)), (H * 2)};
  uint64_t __choreo_tma_2_strides[] = {(H * 256), 128};
  uint32_t __choreo_tma_2_box_shape[] = {(uint32_t)(64), (uint32_t)(128), (uint32_t)(2)};
  uint32_t __choreo_tma_2_elem_strides[] = {1, 1, 1};
  alignas(64) CUtensorMap __choreo_tma_2_tensor_map{};
  CUresult __choreo_tma_2_tensor_map_res = cuTensorMapEncodeTiled(
          &__choreo_tma_2_tensor_map,
          CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
          3,
          V.data(),
          __choreo_tma_2_shape,
          __choreo_tma_2_strides,
          __choreo_tma_2_box_shape,
          __choreo_tma_2_elem_strides,
          CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
          CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B,
          CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
          CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  choreo::abend_true(__choreo_tma_2_tensor_map_res != CUDA_SUCCESS);
  dim3 __flash_atten_gdims0(((Q_SEQ + 127) / 128), H, B);
  dim3 __flash_atten_bdims0(384, 1, 1);
  cudaFuncSetAttribute(__choreo_device_flash_atten, cudaFuncAttributeMaxDynamicSharedMemorySize, 163840 + (128 - 1));
  __choreo_device_flash_atten<<<__flash_atten_gdims0, __flash_atten_bdims0, 163840 + (128 - 1)>>>(Q.data(), K.data(), V.data(), O.data(), scale, B, H, KV_SEQ, Q_SEQ, __choreo_tma_0_tensor_map, __choreo_tma_1_tensor_map, __choreo_tma_2_tensor_map);
  choreo::abend_true(cudaDeviceSynchronize());
}




#include "mha_helper.hpp"

int main() {
  mha_helper::BenchConfig configs[] = {
    {2, 16, 8192, 8192, true, "B=2 H=16 SEQ=8192 causal prefill"},
    {1, 16, 4096, 4096, true, "B=1 H=16 SEQ=4096 causal prefill"},
  };
  return mha_helper::RunBenchmarks(
      "Choreo Flash Attention iter012 reorder_cast (D=128 bf16 causal)",
      configs, sizeof(configs) / sizeof(configs[0]),
      [&](const mha_helper::BenchConfig& cfg,
          const mha_helper::TensorViews& views) {
        (void)cfg;
        flash_atten(views.Q, views.K, views.V, views.O);
      });
}


