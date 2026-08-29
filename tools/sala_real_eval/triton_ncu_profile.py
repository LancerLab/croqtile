#!/usr/bin/env python3
"""Compile and launch a single Triton GEMM so ncu can profile it."""
import torch
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr, num_stages: tl.constexpr):
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

M = N = K = 4096
a = torch.randn(M, K, device='cuda', dtype=torch.float16)
b = torch.randn(K, N, device='cuda', dtype=torch.float16)
c = torch.empty(M, N, device='cuda', dtype=torch.float16)
grid = (M // 128, N // 128)
matmul_kernel[grid](a, b, c, M, N, K,
    a.stride(0), a.stride(1), b.stride(0), b.stride(1),
    c.stride(0), c.stride(1),
    BLOCK_M=128, BLOCK_N=128, BLOCK_K=64, num_stages=3)
torch.cuda.synchronize()
print("Triton GEMM 128x128x64 3s launched successfully")
