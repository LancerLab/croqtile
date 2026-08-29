"""
Non-persistent NVWS GEMM: each CTA processes one tile. No cross-tile barrier.
Multiple trials for statistical significance.
"""
import os
import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor
import statistics

mode = "SALA" if os.environ.get("SALA_ENABLE") == "1" else "Baseline"

@triton.jit
def matmul_nvws_single_tile(
    a_desc, b_desc, c_desc,
    M, N, K,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
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


def bench(M, N, K, BM=128, BN=128, BK=64, stages=3, iters=200, trials=5):
    results = []
    for _ in range(trials):
        a = torch.randn((M, K), device="cuda", dtype=torch.float16)
        b = torch.randn((K, N), device="cuda", dtype=torch.float16).T.contiguous()
        c = torch.empty((M, N), device="cuda", dtype=torch.float16)
        num_tiles = triton.cdiv(M, BM) * triton.cdiv(N, BN)
        grid = (num_tiles,)
        a_desc = TensorDescriptor(a, a.shape, a.stride(), [BM, BK])
        b_desc = TensorDescriptor(b, b.shape, b.stride(), [BN, BK])
        c_desc = TensorDescriptor(c, c.shape, c.stride(), [BM, BN])

        def launch():
            matmul_nvws_single_tile[grid](
                a_desc, b_desc, c_desc, M, N, K,
                BLOCK_SIZE_M=BM, BLOCK_SIZE_N=BN, BLOCK_SIZE_K=BK,
                GROUP_SIZE_M=8,
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

    return statistics.mean(results), statistics.stdev(results), results


if __name__ == "__main__":
    print(f"=== Non-Persistent NVWS GEMM Benchmark ({mode}) ===")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    sizes = [(2048, 2048, 2048), (4096, 4096, 4096), (8192, 8192, 8192)]

    for M, N, K in sizes:
        mean, std, raw = bench(M, N, K)
        vals = " ".join(f"{r:.1f}" for r in raw)
        print(f"  {M}^3:  {mean:.1f} +/- {std:.1f} TFLOPS  [{vals}]")
