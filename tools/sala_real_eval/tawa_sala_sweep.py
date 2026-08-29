"""
Sweep Tawa NVWS GEMM configs to find SMEM-bound cases where SALA improves TFLOPS.

Measures TFLOPS for baseline vs SALA across multiple tile/stage combinations.
Focuses on configs where SMEM (not registers) is the binding occupancy constraint.

Usage:
  rm -rf ~/.triton/cache/
  PYTHONPATH=/home/wsj/dev/triton-aref/python python3.10 tawa_sala_sweep.py [--gpu 1]
"""

import os
import sys
import subprocess
import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

SALA_ON = os.environ.get("SALA_ENABLE") == "1"

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


def benchmark_config(M, N, K, BM, BN, BK, stages, warmup=10, iters=100):
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
            num_warps=4, num_stages=stages,
            enable_warp_specialization=True,
        )

    try:
        for _ in range(warmup):
            launch()
        torch.cuda.synchronize()
    except Exception as e:
        return None, str(e)

    ref = torch.mm(a, b.T)
    err = (c - ref).abs().max().item() / ref.abs().max().item()
    if err > 0.05:
        return None, f"FAIL (rel_err={err:.6f})"

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(iters):
        launch()
    end_event.record()
    torch.cuda.synchronize()
    elapsed_ms = start_event.elapsed_time(end_event)

    tflops = 2.0 * M * N * K * iters / (elapsed_ms / 1000) / 1e12
    return tflops, "OK"


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=1)
    args = parser.parse_args()

    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    mode = "SALA" if SALA_ON else "Baseline"
    print(f"=== Tawa NVWS GEMM Config Sweep ({mode}) ===")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Triton: {triton.__file__}")
    print()

    configs = [
        # (BM, BN, BK, stages) -- exploring register vs SMEM binding
        (128, 128, 64, 3),  # existing: 162 regs, 131 KB -> reg-bound
        (128, 128, 64, 2),  # existing: 162 regs, 98 KB -> reg-bound
        (64, 128, 64, 2),   # existing: 90 regs, 66 KB -> reg-bound
        (64, 64, 64, 3),    # smaller tile: fewer accumulators -> maybe SMEM-bound?
        (64, 64, 64, 2),    # smaller tile, fewer stages
        (64, 64, 64, 4),    # more stages -> higher SMEM
        (64, 128, 64, 3),   # medium tile, 3 stages
        (128, 64, 64, 3),   # asymmetric
        (128, 64, 64, 2),   # asymmetric, 2 stages
    ]

    problem_sizes = [
        (4096, 4096, 4096),
        (8192, 8192, 8192),
    ]

    print(f"{'Config':<20} {'Size':<16} {'TFLOPS':>8}  {'Status'}")
    print("-" * 65)

    for BM, BN, BK, stages in configs:
        label = f"{BM}x{BN} {stages}s"
        for M, N, K in problem_sizes:
            # Clear triton cache for each config to ensure fresh compilation
            tflops, status = benchmark_config(M, N, K, BM, BN, BK, stages)
            size_label = f"{M}x{N}x{K}"
            if tflops is not None:
                print(f"{label:<20} {size_label:<16} {tflops:>8.1f}  {status}")
            else:
                print(f"{label:<20} {size_label:<16} {'N/A':>8}  {status}")
