// ============================================================================
// Kernel Configuration
// ============================================================================

// Tiling settigs !!!!!, can be determined no matter what problem size
// Shared memory buffer size (48KB max per block typically)
// for H800, you can request 64KB at most, if explicitly requested, you can get up to 100KB
constexpr int SMEM_SIZE_PER_BLOCK = 32 * 1024;  // 16 KB shared memory buffer
constexpr int SMEM_ELEMENTS_PER_BLOCK = SMEM_SIZE_PER_BLOCK / sizeof(float);  // 4096 elements
constexpr int THREADS_PER_BLOCK = 128;
constexpr int SMEM_ELEMENTS_PER_THREAD = SMEM_ELEMENTS_PER_BLOCK / THREADS_PER_BLOCK;  // 32 elements per thread

/**
 * Variant 1: Native CUDA C - Scalar loads (32B coalesced per warp)
 *
 * Coalesced access pattern (CORRECT):
 * - Each thread loads 1 float (4 bytes) per iteration
 * - Warp of 32 threads: threads 0-7 together load 32 bytes (one cache line)
 * - threadIdx.x provides stride=1 offset (consecutive threads → consecutive addresses)
 * - Same thread's consecutive loads are stride=blockDim.x (256) apart
 *
 * Example for first 2 iterations (256 threads):
 *   Iteration 0: thread 0 loads idx 0, thread 1 loads idx 1, ..., thread 255 loads idx 255
 *   Iteration 1: thread 0 loads idx 256, thread 1 loads idx 257, ..., thread 255 loads idx 511
 */
__global__ void kernel_copy_native(const float* __restrict__ gmem,
                                    float* __restrict__ output,
                                    int n_elements) {
    __shared__ float smem[SMEM_ELEMENTS_PER_BLOCK];

    // Calculate this block's data range
    int total_elements_per_block = (n_elements + gridDim.x - 1) / gridDim.x;
    int block_start = blockIdx.x * total_elements_per_block;
    int block_end = min(block_start + total_elements_per_block, n_elements);

    int n_iters = (total_elements_per_block + SMEM_ELEMENTS_PER_BLOCK - 1) / SMEM_ELEMENTS_PER_BLOCK;

    int tid = threadIdx.x;

    // Loop over SMEM-sized chunks
    for (int iter = 0; iter < n_iters; ++iter) {
        int iter_offset = block_start + iter * SMEM_ELEMENTS_PER_BLOCK;

        // Load GMEM → SMEM
        // Each thread loads ELEMENTS_PER_THREAD (32) floats
        // Access pattern: tid, tid+256, tid+512, ..., tid+31*256
        #pragma unroll
        for (int i = 0; i < SMEM_ELEMENTS_PER_THREAD; ++i) {
            int idx = tid + i * blockDim.x;  // stride = blockDim.x = 256
            int gmem_idx = iter_offset + idx;
            if (gmem_idx < block_end && idx < SMEM_ELEMENTS_PER_BLOCK) {
                // Use __ldg for cache-friendly load
                smem[idx] = __ldg(&gmem[gmem_idx]);
            }
        }
        __syncthreads();

        // Write back SMEM → GMEM
        #pragma unroll
        for (int i = 0; i < SMEM_ELEMENTS_PER_THREAD; ++i) {
            int idx = tid + i * blockDim.x;
            int gmem_idx = iter_offset + idx;
            if (gmem_idx < block_end && idx < SMEM_ELEMENTS_PER_BLOCK) {
                output[gmem_idx] = smem[idx];
            }
        }
        __syncthreads();
    }
}

/**
 * Variant 2: Vectorized load using float4 (128-bit loads, fully coalesced)
 *
 * Each thread loads 8 float4s (32 floats = 128 bytes)
 * Warp loads: 32 threads * 128 bytes = 4KB per iteration
 */
__global__ void kernel_copy_vectorized(const float* __restrict__ gmem,
                                        float* __restrict__ output,
                                        int n_elements) {
    __shared__ float smem[SMEM_ELEMENTS_PER_BLOCK];

    // Calculate this block's data range (aligned to float4 boundaries)
    int total_elements_per_block = (n_elements + gridDim.x - 1) / gridDim.x;
    // Align to 4-element boundary for float4
    total_elements_per_block = ((total_elements_per_block + 3) / 4) * 4;
    int block_start = blockIdx.x * total_elements_per_block;
    int block_end = min(block_start + total_elements_per_block, n_elements);

    // Skip if this block is out of range
    if (block_start >= n_elements) return;

    int n_iters = (total_elements_per_block + SMEM_ELEMENTS_PER_BLOCK - 1) / SMEM_ELEMENTS_PER_BLOCK;

    int tid = threadIdx.x;
    float4* smem4 = reinterpret_cast<float4*>(smem);
    constexpr int FLOAT4_PER_THREAD = SMEM_ELEMENTS_PER_THREAD / 4;  // 32/4 = 8

    for (int iter = 0; iter < n_iters; ++iter) {
        int iter_offset = block_start + iter * SMEM_ELEMENTS_PER_BLOCK;
        if (iter_offset >= block_end) break;

        const float4* gmem4 = reinterpret_cast<const float4*>(gmem + iter_offset);

        // Vectorized load: 8 float4s per thread (128 bytes)
        #pragma unroll
        for (int i = 0; i < FLOAT4_PER_THREAD; ++i) {
            int idx = tid + i * blockDim.x;
            int gmem_idx = iter_offset + idx * 4;
            if (gmem_idx + 3 < n_elements && idx < SMEM_ELEMENTS_PER_BLOCK / 4) {
                // Use __ldg for cache-friendly load
                smem4[idx] = __ldg(&gmem4[idx]);
            }
        }
        __syncthreads();

        // Vectorized store
        float4* output4 = reinterpret_cast<float4*>(output + iter_offset);
        #pragma unroll
        for (int i = 0; i < FLOAT4_PER_THREAD; ++i) {
            int idx = tid + i * blockDim.x;
            int gmem_idx = iter_offset + idx * 4;
            if (gmem_idx + 3 < n_elements && idx < SMEM_ELEMENTS_PER_BLOCK / 4) {
                output4[idx] = smem4[idx];
            }
        }
        __syncthreads();
    }
}

/**
 * Variant 8: PTX version, Vectorized load using float4 (128-bit loads, fully coalesced)
 *
 * Each thread loads 8 float4s (32 floats = 128 bytes)
 * Warp loads: 32 threads * 128 bytes = 4KB per iteration
 */

__global__ void kernel_copy_vectorized_ptx(const float* __restrict__ gmem,
                                        float* __restrict__ output,
                                        int n_elements) {
    __shared__ float smem[SMEM_ELEMENTS_PER_BLOCK];

    // Calculate this block's data range (aligned to float4 boundaries)
    int total_elements_per_block = (n_elements + gridDim.x - 1) / gridDim.x;
    // Align to 4-element boundary for float4
    total_elements_per_block = ((total_elements_per_block + 3) / 4) * 4;
    int block_start = blockIdx.x * total_elements_per_block;
    int block_end = min(block_start + total_elements_per_block, n_elements);

    // Skip if this block is out of range
    if (block_start >= n_elements) return;

    int n_iters = (total_elements_per_block + SMEM_ELEMENTS_PER_BLOCK - 1) / SMEM_ELEMENTS_PER_BLOCK;

    int tid = threadIdx.x;
    float4* smem4 = reinterpret_cast<float4*>(smem);
    constexpr int FLOAT4_PER_THREAD = SMEM_ELEMENTS_PER_THREAD / 4;  // 32/4 = 8

    for (int iter = 0; iter < n_iters; ++iter) {
        int iter_offset = block_start + iter * SMEM_ELEMENTS_PER_BLOCK;
        if (iter_offset >= block_end) break;

        const float4* gmem4 = reinterpret_cast<const float4*>(gmem + iter_offset);

        // Vectorized load: 8 float4s per thread (128 bytes)
        #pragma unroll
        for (int i = 0; i < FLOAT4_PER_THREAD; ++i) {
            int idx = tid + i * blockDim.x;
            int gmem_idx = iter_offset + idx * 4;
            if (gmem_idx + 3 < n_elements && idx < SMEM_ELEMENTS_PER_BLOCK / 4) {
                // Use __ldg for cache-friendly load
                // smem4[idx] = __ldg(&gmem4[idx]);
                float4 val;
                asm volatile(
                    "ld.global.cg.v4.f32 {%0, %1, %2, %3}, [%4]; \n\t"
                    "st.shared.v4.f32 [%5], {%0, %1, %2, %3}; \n\t"
                    : "=f"(val.x), "=f"(val.y), "=f"(val.z), "=f"(val.w)
                    : "l"(gmem4 + idx), "r"((unsigned int)__cvta_generic_to_shared(smem4 + idx))
                    : "memory"
                );
                (void)val;
            }
        }
        __syncthreads();

        // Vectorized store
        float4* output4 = reinterpret_cast<float4*>(output + iter_offset);
        #pragma unroll
        for (int i = 0; i < FLOAT4_PER_THREAD; ++i) {
            int idx = tid + i * blockDim.x;
            int gmem_idx = iter_offset + idx * 4;
            if (gmem_idx + 3 < n_elements && idx < SMEM_ELEMENTS_PER_BLOCK / 4) {
                // output4[idx] = smem4[idx];
                float4 val;
                asm volatile(
                    "ld.shared.v4.f32 {%0, %1, %2, %3}, [%4]; \n\t"
                    "st.global.cg.v4.f32 [%5], {%0, %1, %2, %3}; \n\t"
                    : "=f"(val.x), "=f"(val.y), "=f"(val.z), "=f"(val.w)
                    : "r"((unsigned int)__cvta_generic_to_shared(smem4 + idx)), "l"(output4 + idx)
                    : "memory"
                );
                (void)val;
            }
        }
        __syncthreads();
    }
}

/**
 * Variant 3: Inline PTX with cache hints (ld.global.cg + st.shared)
 *
 * Coalesced access pattern (CORRECT):
 * - Each thread loads 1 float (4 bytes) per iteration using PTX
 * - threadIdx.x provides stride=1 offset
 * - Same thread's consecutive loads are stride=blockDim.x (256) apart
 */
__global__ void kernel_copy_ptx(const float* __restrict__ gmem,
                                 float* __restrict__ output,
                                 int n_elements) {
    __shared__ float smem[SMEM_ELEMENTS_PER_BLOCK];

    // Calculate this block's data range
    int total_elements_per_block = (n_elements + gridDim.x - 1) / gridDim.x;
    int block_start = blockIdx.x * total_elements_per_block;
    int block_end = min(block_start + total_elements_per_block, n_elements);

    int n_iters = (total_elements_per_block + SMEM_ELEMENTS_PER_BLOCK - 1) / SMEM_ELEMENTS_PER_BLOCK;

    int tid = threadIdx.x;

    for (int iter = 0; iter < n_iters; ++iter) {
        int iter_offset = block_start + iter * SMEM_ELEMENTS_PER_BLOCK;

        // PTX load with cache hints
        #pragma unroll
        for (int i = 0; i < SMEM_ELEMENTS_PER_THREAD; ++i) {
            int idx = tid + i * blockDim.x;
            int gmem_idx = iter_offset + idx;
            if (gmem_idx < block_end && idx < SMEM_ELEMENTS_PER_BLOCK) {
                float val;
                const float* gptr = gmem + gmem_idx;
                float* sptr = smem + idx;
                asm volatile(
                    "ld.global.cg.f32 %0, [%1];\n\t"
                    "st.shared.f32 [%2], %0;"
                    : "=f"(val)
                    : "l"(gptr), "r"((unsigned int)__cvta_generic_to_shared(sptr))
                    : "memory"
                );
            }
        }
        __syncthreads();

        // PTX store
        #pragma unroll
        for (int i = 0; i < SMEM_ELEMENTS_PER_THREAD; ++i) {
            int idx = tid + i * blockDim.x;
            int gmem_idx = iter_offset + idx;
            if (gmem_idx < block_end && idx < SMEM_ELEMENTS_PER_BLOCK) {
                float* sptr = smem + idx;
                float* gptr = output + gmem_idx;
                asm volatile(
                    "{\n\t"
                    ".reg .f32 tmp;\n\t"
                    "ld.shared.f32 tmp, [%0];\n\t"
                    "st.global.cg.f32 [%1], tmp;\n\t"
                    "}"
                    :
                    : "r"((unsigned int)__cvta_generic_to_shared(sptr)), "l"(gptr)
                    : "memory"
                );
            }
        }
        __syncthreads();
    }
}

/**
 * Variant 4: CuTe basic copy with Layout partitioning
 *
 * Simple 1D layout:
 * - Data: 8192 elements (SMEM_ELEMENTS)
 * - Threads: 256 (THREADS_PER_BLOCK)
 * - Each thread handles 32 elements (ELEMENTS_PER_THREAD)
 *
 * local_partition divides the 1D tensor among threads.
 * CuTe's copy handles the actual data movement.
 *
 * NOTE: This variant uses UniversalCopy<float> as the copy atom to ensure
 * synchronous (blocking) scalar float copies. Without explicit copy atom specification,
 * CuTe may automatically select async copy instructions (cp.async) which can
 * introduce unexpected latency and different performance characteristics.
 * By explicitly using UniversalCopy, we ensure deterministic synchronous behavior.
 */
__global__ void kernel_copy_cute(const float* __restrict__ gmem,
                                  float* __restrict__ output,
                                  int n_elements) {
    extern __shared__ float smem_cute[];

    // Calculate this block's data range
    int total_elements_per_block = (n_elements + gridDim.x - 1) / gridDim.x;
    int block_start = blockIdx.x * total_elements_per_block;
    int block_end = min(block_start + total_elements_per_block, n_elements);

    // Calculate number of SMEM-sized iterations (same as other kernels)
    int n_iters = (total_elements_per_block + SMEM_ELEMENTS_PER_BLOCK - 1) / SMEM_ELEMENTS_PER_BLOCK;

    // 1D layouts - simple and straightforward
    auto data_layout = make_layout(make_shape(Int<SMEM_ELEMENTS_PER_BLOCK>{}));
    auto thr_layout = make_layout(make_shape(Int<THREADS_PER_BLOCK>{}));

    // Create SMEM tensor (persistent across iterations)
    auto smem_tensor = make_tensor(make_smem_ptr(smem_cute), data_layout);
    // Partition: 8192 / 256 = 32 elements per thread
    auto thr_smem = local_partition(smem_tensor, thr_layout, threadIdx.x);

    int tid = threadIdx.x;

    // Outer loop: iterate over SMEM-sized chunks
    for (int iter = 0; iter < n_iters; ++iter) {
        int iter_offset = block_start + iter * SMEM_ELEMENTS_PER_BLOCK;
        if (iter_offset >= block_end) break;

        // Check remaining elements in this chunk
        int remaining = min(block_end - iter_offset, SMEM_ELEMENTS_PER_BLOCK);

        // For partial chunks, we need to be careful about bounds
        // CuTe copy doesn't have built-in bounds checking, so we use manual loop
        if (remaining < SMEM_ELEMENTS_PER_BLOCK) {
            // Partial chunk: use manual copy with bounds checking
            #pragma unroll
            for (int i = 0; i < SMEM_ELEMENTS_PER_THREAD; ++i) {
                int idx = tid + i * THREADS_PER_BLOCK;
                if (idx < remaining) {
                    smem_cute[idx] = gmem[iter_offset + idx];
                }
            }
            __syncthreads();

            #pragma unroll
            for (int i = 0; i < SMEM_ELEMENTS_PER_THREAD; ++i) {
                int idx = tid + i * THREADS_PER_BLOCK;
                if (idx < remaining) {
                    output[iter_offset + idx] = smem_cute[idx];
                }
            }
            __syncthreads();
        } else {
            // Full chunk: use CuTe copy
            // NOTE: Without explicit copy atom specification, CuTe may automatically
            // select async copy instructions (cp.async) for performance optimization.
            // This is acceptable for this benchmark as we're measuring overall throughput.
            // For deterministic synchronous behavior, use make_tiled_copy with
            // explicit UniversalCopy atom (see Variant 6).
            auto gmem_tensor = make_tensor(make_gmem_ptr(gmem + iter_offset), data_layout);
            auto thr_gmem = local_partition(gmem_tensor, thr_layout, threadIdx.x);

            // Copy GMEM → SMEM using CuTe copy (may use async instructions)
            copy(thr_gmem, thr_smem);
            __syncthreads();

            // Create output GMEM tensor for this iteration
            auto out_tensor = make_tensor(make_gmem_ptr(output + iter_offset), data_layout);
            auto thr_out = local_partition(out_tensor, thr_layout, threadIdx.x);

            // Copy SMEM → GMEM using CuTe copy (may use async instructions)
            copy(thr_smem, thr_out);
            __syncthreads();
        }
    }
}

/**
 * Variant 5: CuTe basic copy with Layout partitioning indexed tail copy
 *
 * Simple 1D layout:
 * - Data: 8192 elements (SMEM_ELEMENTS)
 * - Threads: 256 (THREADS_PER_BLOCK)
 * - Each thread handles 32 elements (ELEMENTS_PER_THREAD)
 *
 * local_partition divides the 1D tensor among threads.
 * CuTe's copy handles the actual data movement.
 *
 * NOTE: This variant does NOT use explicit copy atom specification.
 * Without explicit copy atom, CuTe may automatically select async copy instructions
 * (cp.async) for performance optimization. This is acceptable for throughput measurement,
 * but if deterministic synchronous behavior is required, use explicit copy atoms
 * (see Variant 7 with UniversalCopy<float>).
 */
__global__ void kernel_copy_cute_indexed_tail_copy(const float* __restrict__ gmem,
                                  float* __restrict__ output,
                                  int n_elements) {
    extern __shared__ float smem_cute[];

    // Calculate this block's data range
    int total_elements_per_block = (n_elements + gridDim.x - 1) / gridDim.x;
    int block_start = blockIdx.x * total_elements_per_block;
    int block_end = min(block_start + total_elements_per_block, n_elements);

    // Calculate number of SMEM-sized iterations (same as other kernels)
    int n_iters = (total_elements_per_block + SMEM_ELEMENTS_PER_BLOCK - 1) / SMEM_ELEMENTS_PER_BLOCK;

    // 1D layouts - simple and straightforward
    auto data_layout = make_layout(make_shape(Int<SMEM_ELEMENTS_PER_BLOCK>{}));
    auto thr_layout = make_layout(make_shape(Int<THREADS_PER_BLOCK>{}));

    // Create SMEM tensor (persistent across iterations)
    auto smem_tensor = make_tensor(make_smem_ptr(smem_cute), data_layout);
    // Partition: 8192 / 256 = 32 elements per thread
    auto thr_smem = local_partition(smem_tensor, thr_layout, threadIdx.x);

    int tid = threadIdx.x;

    // Outer loop: iterate over SMEM-sized chunks
    for (int iter = 0; iter < n_iters; ++iter) {
        int iter_offset = block_start + iter * SMEM_ELEMENTS_PER_BLOCK;
        if (iter_offset >= block_end) break;

        // Check remaining elements in this chunk
        int remaining = min(block_end - iter_offset, SMEM_ELEMENTS_PER_BLOCK);

        // For partial chunks, we need to be careful about bounds
        // CuTe copy doesn't have built-in bounds checking, so we use manual loop
        if (remaining < SMEM_ELEMENTS_PER_BLOCK) {
            // Partial chunk: use manual copy with bounds checking
            auto gmem_tensor = make_tensor(make_gmem_ptr(gmem + iter_offset), data_layout);
            auto thr_gmem = local_partition(gmem_tensor, thr_layout, threadIdx.x);

            // Create output GMEM tensor for this iteration
            auto out_tensor = make_tensor(make_gmem_ptr(output + iter_offset), data_layout);
            auto thr_out = local_partition(out_tensor, thr_layout, threadIdx.x);

            #pragma unroll
            for (int i = 0; i < SMEM_ELEMENTS_PER_THREAD; ++i) {
                int idx = tid + i * THREADS_PER_BLOCK;
                if (idx < remaining) {
                    smem_tensor(idx) = gmem_tensor(idx);
                }
            }
            __syncthreads();

            #pragma unroll
            for (int i = 0; i < SMEM_ELEMENTS_PER_THREAD; ++i) {
                int idx = tid + i * THREADS_PER_BLOCK;
                if (idx < remaining) {
                    out_tensor(idx) = smem_tensor(idx);
                }
            }
            __syncthreads();
        } else {
            // Full chunk: use CuTe copy
            // NOTE: Without explicit copy atom specification, CuTe may automatically
            // select async copy instructions (cp.async) for performance optimization.
            // This is acceptable for this benchmark as we're measuring overall throughput.
            // For deterministic synchronous behavior, use explicit copy atoms
            // (see Variant 7 with UniversalCopy<float4>).
            auto gmem_tensor = make_tensor(make_gmem_ptr(gmem + iter_offset), data_layout);
            auto thr_gmem = local_partition(gmem_tensor, thr_layout, threadIdx.x);

            // Copy GMEM → SMEM using CuTe copy (may use async instructions)
            copy(thr_gmem, thr_smem);
            __syncthreads();

            // Create output GMEM tensor for this iteration
            auto out_tensor = make_tensor(make_gmem_ptr(output + iter_offset), data_layout);
            auto thr_out = local_partition(out_tensor, thr_layout, threadIdx.x);

            // Copy SMEM → GMEM using CuTe copy (may use async instructions)
            copy(thr_smem, thr_out);
            __syncthreads();
        }
    }
}

/**
 * Variant 7: CuTe copy with explicit UniversalCopy atom and tail handling
 *
 * Uses Copy_Atom<UniversalCopy<float>> for synchronous scalar copies.
 * Key design:
 * - 2D layout: 32x256 (8192 elements total)
 * - Thread layout: 32x4 (128 threads total) arranged in 2D (m-major)
 * - Value layout: 1x64 (64 float elements per thread)
 * - Copy atom: UniversalCopy<float> for deterministic synchronous behavior
 * - Handles both full chunks and partial chunks (tail) with bounds checking
 *
 * UniversalCopy ensures that CuTe uses synchronous scalar copies instead of
 * automatically selecting async copy instructions (cp.async). This is important
 * for understanding the true synchronous copy performance without async optimizations.
 *
 * Tail handling: For partial chunks, fall back to manual scalar copy to avoid
 * alignment issues and ensure correctness.
 *
 * Performance note: This kernel demonstrates how to use explicit copy atoms
 * to ensure synchronous behavior, avoiding automatic async copy selection by CuTe.
 * The performance is slightly better than v5 (1622.33 vs 1593.65 GB/s) due to
 * better thread layout and partitioning strategy.
 */
__global__ void kernel_copy_cute_universal_copy(const float* __restrict__ gmem,
                                                float* __restrict__ output,
                                                int n_elements) {
    extern __shared__ float smem_cute[];

    // Calculate this block's data range
    int total_elements_per_block = (n_elements + gridDim.x - 1) / gridDim.x;
    int block_start = blockIdx.x * total_elements_per_block;
    int block_end = min(block_start + total_elements_per_block, n_elements);

    // Calculate number of SMEM-sized iterations
    int n_iters = (total_elements_per_block + SMEM_ELEMENTS_PER_BLOCK - 1) / SMEM_ELEMENTS_PER_BLOCK;

    // 2D layout: 32x256 (8192 elements)
    auto data_layout = make_layout(make_shape(Int<32>{}, Int<256>{}));
    auto thr_layout = make_layout(make_shape(Int<32>{}, Int<4>{}));

    // Create SMEM tensor (persistent across iterations)
    auto smem_tensor = make_tensor(make_smem_ptr(smem_cute), data_layout);
    // Partition: 32x256 / (32x4) = 1x64 elements per thread
    auto thr_smem = local_partition(smem_tensor, thr_layout, threadIdx.x);

    // Define copy atom: UniversalCopy<float> ensures synchronous scalar copies
    // This prevents CuTe from automatically selecting async copy instructions
    // NOTE: We use float instead of float4 to avoid alignment issues with partial chunks
    using CopyAtom = Copy_Atom<UniversalCopy<float>, float>;
    auto copy_op = CopyAtom{};

    // Outer loop: iterate over SMEM-sized chunks
    for (int iter = 0; iter < n_iters; ++iter) {
        int iter_offset = block_start + iter * SMEM_ELEMENTS_PER_BLOCK;
        if (iter_offset >= block_end) break;

        // Check remaining elements in this chunk
        int remaining = min(block_end - iter_offset, SMEM_ELEMENTS_PER_BLOCK);

        // Only use float4 copy for full SMEM blocks (8192 elements)
        // For partial blocks, use manual scalar copy to avoid alignment issues
        if (remaining == SMEM_ELEMENTS_PER_BLOCK) {
            // Full chunk: use CuTe copy with explicit float4 copy atom
            // Create float4 view of the data (8192 floats = 2048 float4)
            auto gmem_tensor = make_tensor(make_gmem_ptr(gmem + iter_offset),
                                          make_layout(make_shape(Int<32>{}, Int<256>{})));
            auto thr_gmem = local_partition(gmem_tensor, thr_layout, threadIdx.x);

            // Copy GMEM → SMEM using CuTe copy with float4 copy atom
            copy(copy_op, thr_gmem, thr_smem);
            __syncthreads();

            // Create output GMEM tensor for this iteration
            auto out_tensor = make_tensor(make_gmem_ptr(output + iter_offset),
                                         make_layout(make_shape(Int<32>{}, Int<256>{})));
            auto thr_out = local_partition(out_tensor, thr_layout, threadIdx.x);

            // Copy SMEM → GMEM using CuTe copy with float4 copy atom
            copy(copy_op, thr_smem, thr_out);
            __syncthreads();
        } else {
            // Partial chunk: use manual scalar copy with bounds checking
            #pragma unroll
            for (int i = threadIdx.x; i < remaining; i += blockDim.x) {
                smem_cute[i] = gmem[iter_offset + i];
            }
            __syncthreads();

            #pragma unroll
            for (int i = threadIdx.x; i < remaining; i += blockDim.x) {
                output[iter_offset + i] = smem_cute[i];
            }
            __syncthreads();
        }
    }
}

/**
 * Variant 6: CuTe tiled copy with no tail handling
 *
 * Uses make_tiled_copy to create a copy pattern that avoids tail handling.
 * Key design:
 * - Thread layout: 32x4 (128 threads total) arranged in 2D (m-major)
 * - Value layout: 1x8 (8 elements per thread)
 * - Copy atom: UniversalCopy with float (scalar copy)
 * - All data is processed in complete tiles, no partial chunks
 *
 * The tiled copy pattern ensures:
 * - Coalesced memory access
 * - Efficient thread utilization
 * - No bounds checking needed for full tiles
 * - 2D layout provides better memory access patterns
 *
 * Performance note: This kernel demonstrates how to use make_tiled_copy
 * to avoid tail handling by processing data in complete tiles.
 * The m-major layout (stride<1,32>) ensures coalesced access patterns.
 */
__global__ void kernel_copy_cute_tiled_copy(const float* __restrict__ gmem,
                                            float* __restrict__ output,
                                            int n_elements) {
    extern __shared__ float smem_cute[];

    // Calculate this block's data range
    int total_elements_per_block = (n_elements + gridDim.x - 1) / gridDim.x;
    int block_start = blockIdx.x * total_elements_per_block;
    int block_end = min(block_start + total_elements_per_block, n_elements);

    // Calculate number of SMEM-sized iterations
    int n_iters = (total_elements_per_block + SMEM_ELEMENTS_PER_BLOCK - 1) / SMEM_ELEMENTS_PER_BLOCK;

    // Create tiled copy with UniversalCopy
    // Thread layout: 32x4 = 128 threads (m-major: stride<1,32>)
    // Value layout: 1x8 = 8 elements per thread
    // Total per tile: 32*1 x 4*8 = 32 x 32 = 1024 elements
    // We'll do 8 iterations to fill SMEM (8192 / 1024 = 8 iterations per SMEM chunk)
    using CopyAtom = Copy_Atom<UniversalCopy<float>, float>;
    auto tiled_copy = make_tiled_copy(
        CopyAtom{},
        Layout<Shape<Int<32>, Int<4>>, Stride<Int<1>, Int<32>>>{},  // Thread layout: 32x4 m-major
        Layout<Shape<Int<1>, Int<8>>>{}                              // Value layout: 1x8 m-major
    );

    // Get this thread's copy operation
    auto thr_copy = tiled_copy.get_slice(threadIdx.x);

    // Create SMEM tensor with 2D layout (32 x 256)
    // This matches the tiled copy pattern better
    auto smem_layout = make_layout(make_shape(Int<32>{}, Int<256>{}));
    auto smem_tensor = make_tensor(make_smem_ptr(smem_cute), smem_layout);

    // Outer loop: iterate over SMEM-sized chunks
    for (int iter = 0; iter < n_iters; ++iter) {
        int iter_offset = block_start + iter * SMEM_ELEMENTS_PER_BLOCK;
        if (iter_offset >= block_end) break;

        // Check remaining elements in this chunk
        int remaining = min(block_end - iter_offset, SMEM_ELEMENTS_PER_BLOCK);

        // Full chunk: use tiled copy
        // Create GMEM tensor for this iteration
        auto gmem_layout = make_layout(make_shape(Int<32>{}, Int<256>{}));
        auto gmem_tensor = make_tensor(make_gmem_ptr(gmem + iter_offset), gmem_layout);
        auto thr_gmem = thr_copy.partition_S(gmem_tensor);

        // Partition SMEM tensor as destination
        auto thr_smem = thr_copy.partition_D(smem_tensor);

        // Copy GMEM → SMEM using tiled copy
        copy(tiled_copy, thr_gmem, thr_smem);
        __syncthreads();

        // Create output GMEM tensor for this iteration
        auto out_tensor = make_tensor(make_gmem_ptr(output + iter_offset), gmem_layout);
        auto thr_out = thr_copy.partition_D(out_tensor);

        // Copy SMEM → GMEM using tiled copy
        copy(tiled_copy, thr_smem, thr_out);
        __syncthreads();
    }
}
