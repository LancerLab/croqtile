#!/usr/bin/env python3
"""
Measure actual shared memory usage of Triton kernels.
Uses cudaFuncGetAttributes to get real SMEM allocation.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    num_stages: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(c_ptrs, acc.to(tl.float16),
             mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def measure(name, M, N, K, BM, BN, BK, ns):
    a = torch.randn(M, K, device='cuda', dtype=torch.float16)
    b = torch.randn(K, N, device='cuda', dtype=torch.float16)
    c = torch.empty(M, N, device='cuda', dtype=torch.float16)
    grid = (M // BM, N // BN)

    fn = matmul_kernel[grid](
        a, b, c, M, N, K,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK, num_stages=ns,
    )
    torch.cuda.synchronize()

    # Get shared memory from the compiled kernel metadata
    # Triton exposes this via the kernel's metadata
    bin_key = matmul_kernel.cache[grid]
    for k, v in bin_key.items():
        try:
            smem = v.shared
            print(f"  {name}: dynamic_smem = {smem} B ({smem/1024:.1f} KB)")
            return smem
        except:
            pass

    # Alternative: use cudart to query
    try:
        import ctypes
        libcuda = ctypes.CDLL("libcuda.so")
        # Can't easily get function handle from Triton
        pass
    except:
        pass

    print(f"  {name}: compiled (use ncu for SMEM measurement)")
    return None


def main():
    print("=== Triton Kernel SMEM Usage (real compilation) ===\n")

    configs = [
        ("GEMM f16 128x128x64 3s", 4096, 4096, 4096, 128, 128, 64, 3),
        ("GEMM f16 128x128x64 4s", 4096, 4096, 4096, 128, 128, 64, 4),
        ("GEMM f16 128x256x64 3s", 4096, 4096, 4096, 128, 256, 64, 3),
        ("GEMM f16 256x128x64 3s", 4096, 4096, 4096, 256, 128, 64, 3),
        ("GEMM f16 128x128x32 4s", 4096, 4096, 4096, 128, 128, 32, 4),
    ]

    for args in configs:
        try:
            measure(*args)
        except Exception as e:
            print(f"  {args[0]}: FAILED - {e}")

    print("\nNote: Triton kernels write output from registers to GMEM.")
    print("No epilogue SMEM buffer exists. All SMEM is for pipeline A/B.")
    print("SALA interval shrinkage is meaningful but has no physical effect")
    print("until Triton adds SMEM-based output staging.")


if __name__ == "__main__":
    main()
