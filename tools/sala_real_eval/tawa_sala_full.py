"""
Full SALA evaluation on Tawa NVWS GEMM.
Non-persistent kernel (1 tile per CTA) avoids cross-tile race,
making SALA buffer overlap provably safe.

Must be run with PYTHONPATH=/home/wsj/dev/triton-aref/python
"""

import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor
import sys
import os
import time

print(f"Triton version: {triton.__version__}")
print(f"Triton path: {triton.__file__}")


@triton.jit
def matmul_nvws_single_tile(
    a_desc, b_desc, c_desc,
    M, N, K,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """Non-persistent NVWS GEMM: each CTA processes exactly one tile."""
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = pid_m * BLOCK_SIZE_M
    offs_bn = pid_n * BLOCK_SIZE_N
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for ki in range(k_tiles):
        offs_k = ki * BLOCK_SIZE_K
        a = a_desc.load([offs_am, offs_k])
        b = b_desc.load([offs_bn, offs_k])
        accumulator = tl.dot(a, b.T, accumulator)

    accumulator = accumulator.to(tl.float16)
    c_desc.store([offs_am, offs_bn], accumulator)


def run_correctness(M=4096, N=4096, K=4096, BM=128, BN=128, BK=64):
    """Run correctness check."""
    a = torch.randn((M, K), device="cuda", dtype=torch.float16)
    b = torch.randn((K, N), device="cuda", dtype=torch.float16)
    b = b.T.contiguous()
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    num_tiles = triton.cdiv(M, BM) * triton.cdiv(N, BN)
    grid = (num_tiles,)

    a_desc = TensorDescriptor(a, a.shape, a.stride(), [BM, BK])
    b_desc = TensorDescriptor(b, b.shape, b.stride(), [BN, BK])
    c_desc = TensorDescriptor(c, c.shape, c.stride(), [BM, BN])

    matmul_nvws_single_tile[grid](
        a_desc, b_desc, c_desc,
        M, N, K,
        BLOCK_SIZE_M=BM, BLOCK_SIZE_N=BN, BLOCK_SIZE_K=BK,
        GROUP_SIZE_M=8,
        num_warps=4, num_stages=3,
        enable_warp_specialization=True,
    )
    torch.cuda.synchronize()

    ref = torch.mm(a, b.T)
    err = (c - ref).abs().max().item() / ref.abs().max().item()
    status = "PASS" if err < 0.05 else "FAIL"
    print(f"  Correctness [{M}x{N}x{K}]: rel_err={err:.6f} ({status})")
    return err < 0.05


def run_benchmark(M=4096, N=4096, K=4096, BM=128, BN=128, BK=64,
                  warmup=10, iters=100):
    """Benchmark TFLOPS using CUDA events for accurate timing."""
    a = torch.randn((M, K), device="cuda", dtype=torch.float16)
    b = torch.randn((K, N), device="cuda", dtype=torch.float16)
    b = b.T.contiguous()
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    num_tiles = triton.cdiv(M, BM) * triton.cdiv(N, BN)
    grid = (num_tiles,)

    a_desc = TensorDescriptor(a, a.shape, a.stride(), [BM, BK])
    b_desc = TensorDescriptor(b, b.shape, b.stride(), [BN, BK])
    c_desc = TensorDescriptor(c, c.shape, c.stride(), [BM, BN])

    def launch():
        matmul_nvws_single_tile[grid](
            a_desc, b_desc, c_desc,
            M, N, K,
            BLOCK_SIZE_M=BM, BLOCK_SIZE_N=BN, BLOCK_SIZE_K=BK,
            GROUP_SIZE_M=8,
            num_warps=4, num_stages=3,
            enable_warp_specialization=True,
        )

    for _ in range(warmup):
        launch()
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(iters):
        launch()
    end_event.record()
    torch.cuda.synchronize()
    elapsed_ms = start_event.elapsed_time(end_event)

    tflops = 2.0 * M * N * K * iters / (elapsed_ms / 1000) / 1e12
    ms_per_iter = elapsed_ms / iters
    print(f"  Perf [{M}x{N}x{K}]: {ms_per_iter:.3f} ms, {tflops:.1f} TFLOPS")
    return tflops


if __name__ == "__main__":
    print("\n=== SALA NVWS Evaluation (non-persistent, single-tile-per-CTA) ===\n")

    print("--- Correctness ---")
    all_pass = True
    for sz in [4096, 8192]:
        if not run_correctness(sz, sz, sz):
            all_pass = False
    if not all_pass:
        print("CORRECTNESS FAILED - aborting benchmark")
        sys.exit(1)

    print("\n--- Benchmark ---")
    for sz in [2048, 4096, 8192]:
        run_benchmark(sz, sz, sz)

    print("\nDone!")
