"""
TFLOPS benchmark: Tawa NVWS baseline vs SALA.
Run twice: once without SALA_ENABLE, once with SALA_ENABLE=1.

Usage:
  # Baseline
  rm -rf ~/.triton/cache/
  PYTHONPATH=/home/wsj/dev/triton-aref/python python3 tawa_sala_bench.py

  # SALA
  rm -rf ~/.triton/cache/
  SALA_ENABLE=1 PYTHONPATH=/home/wsj/dev/triton-aref/python python3 tawa_sala_bench.py
"""

import os
import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor
import time

mode = "SALA" if os.environ.get("SALA_ENABLE") == "1" else "Baseline"
print(f"=== Tawa NVWS GEMM Benchmark ({mode}) ===")
print(f"Triton: {triton.__file__}")


@triton.jit
def _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS):
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (tile_id % group_size_m)
    pid_n = (tile_id % num_pid_in_group) // group_size_m
    return pid_m, pid_n


@triton.jit
def matmul_kernel_nested(
    a_desc, b_desc, c_desc,
    M, N, K,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_SMS: tl.constexpr,
):
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS):
        pid_m, pid_n = _compute_pid(
            tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS
        )
        offs_am = pid_m * BLOCK_SIZE_M
        offs_bn = pid_n * BLOCK_SIZE_N

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for ki in range(k_tiles):
            offs_k = ki * BLOCK_SIZE_K
            a = a_desc.load([offs_am, offs_k])
            b = b_desc.load([offs_bn, offs_k])
            accumulator = tl.dot(a, b.T, accumulator)

        accumulator = accumulator.to(tl.float16)
        c_desc.store([offs_am, offs_bn], accumulator)


def benchmark(M, N, K, warmup=10, iters=100):
    BM, BN, BK = 128, 128, 64
    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count

    a = torch.randn((M, K), device="cuda", dtype=torch.float16)
    b = torch.randn((K, N), device="cuda", dtype=torch.float16).T.contiguous()
    c = torch.empty((M, N), device="cuda", dtype=torch.float16)

    a_desc = TensorDescriptor(a, a.shape, a.stride(), [BM, BK])
    b_desc = TensorDescriptor(b, b.shape, b.stride(), [BN, BK])
    c_desc = TensorDescriptor(c, c.shape, c.stride(), [BM, BN])

    grid = (min(NUM_SMS, triton.cdiv(M, BM) * triton.cdiv(N, BN)),)

    def launch():
        matmul_kernel_nested[grid](
            a_desc, b_desc, c_desc,
            M, N, K,
            BLOCK_SIZE_M=BM, BLOCK_SIZE_N=BN, BLOCK_SIZE_K=BK,
            GROUP_SIZE_M=8, NUM_SMS=NUM_SMS,
            num_warps=4, num_stages=3,
            enable_warp_specialization=True,
        )

    # Warmup
    for _ in range(warmup):
        launch()
    torch.cuda.synchronize()

    # Correctness check
    ref = torch.mm(a, b.T)
    err = (c - ref).abs().max().item() / ref.abs().max().item()
    if err > 0.05:
        print(f"  [{M}x{N}x{K}] CORRECTNESS FAIL (rel_err={err:.6f})")
        return None

    # Timed runs with CUDA events
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
    print(f"  [{M}x{N}x{K}]: {ms_per_iter:.3f} ms, {tflops:.1f} TFLOPS")
    return tflops


if __name__ == "__main__":
    print()
    sizes = [
        (2048, 2048, 2048),
        (4096, 4096, 4096),
        (8192, 8192, 8192),
    ]
    results = {}
    for M, N, K in sizes:
        tflops = benchmark(M, N, K)
        results[(M, N, K)] = tflops

    print(f"\n--- Summary ({mode}) ---")
    for (M, N, K), tflops in results.items():
        if tflops:
            print(f"  {M}x{N}x{K}: {tflops:.1f} TFLOPS")
        else:
            print(f"  {M}x{N}x{K}: FAIL")
