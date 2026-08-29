"""
Sweep K dimension to find where bar.sync overhead becomes negligible.
Larger K means more compute per tile, reducing relative barrier cost.
"""
import os
import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor
import statistics

mode = "SALA" if os.environ.get("SALA_ENABLE") == "1" else "Baseline"

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


def bench(M, N, K, BM=128, BN=128, BK=64, stages=3, iters=200, trials=3):
    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
    results = []
    for _ in range(trials):
        a = torch.randn((M, K), device="cuda", dtype=torch.float16)
        b = torch.randn((K, N), device="cuda", dtype=torch.float16).T.contiguous()
        c = torch.empty((M, N), device="cuda", dtype=torch.float16)
        a_desc = TensorDescriptor(a, a.shape, a.stride(), [BM, BK])
        b_desc = TensorDescriptor(b, b.shape, b.stride(), [BN, BK])
        c_desc = TensorDescriptor(c, c.shape, c.stride(), [BM, BN])
        grid = (min(NUM_SMS, triton.cdiv(M, BM) * triton.cdiv(N, BN)),)

        def launch():
            matmul_kernel_nested[grid](
                a_desc, b_desc, c_desc, M, N, K,
                BLOCK_SIZE_M=BM, BLOCK_SIZE_N=BN, BLOCK_SIZE_K=BK,
                GROUP_SIZE_M=8, NUM_SMS=NUM_SMS,
                num_warps=4, num_stages=stages,
                enable_warp_specialization=True)

        for _ in range(20):
            launch()
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            launch()
        end.record()
        torch.cuda.synchronize()
        ms = start.elapsed_time(end)
        results.append(2.0 * M * N * K * iters / (ms / 1000) / 1e12)
    return statistics.mean(results)


if __name__ == "__main__":
    print(f"=== K-Sweep Benchmark ({mode}) ===")
    # Fix M=N=4096, sweep K: more K-tiles per output tile = less barrier overhead
    M = N = 4096
    k_values = [1024, 2048, 4096, 8192, 16384]

    print(f"  {'K':<8} {'TFLOPS':>8}  {'K-tiles':>8}")
    print(f"  {'-'*28}")
    for K in k_values:
        k_tiles = K // 64  # BK=64
        t = bench(M, N, K)
        print(f"  {K:<8} {t:>8.1f}  {k_tiles:>8}")
