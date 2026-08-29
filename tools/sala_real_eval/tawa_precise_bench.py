"""
High-precision TFLOPS benchmark for Tawa NVWS GEMM.
Runs multiple independent trials with high iteration counts.
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


def bench_trial(M, N, K, BM, BN, BK, stages, iters=200):
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
            a_desc, b_desc, c_desc, M, N, K,
            BLOCK_SIZE_M=BM, BLOCK_SIZE_N=BN, BLOCK_SIZE_K=BK,
            GROUP_SIZE_M=8, NUM_SMS=NUM_SMS,
            num_warps=4, num_stages=stages,
            enable_warp_specialization=True,
        )

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
    return 2.0 * M * N * K * iters / (ms / 1000) / 1e12


if __name__ == "__main__":
    print(f"=== Precise Tawa NVWS Benchmark ({mode}) ===")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    n_trials = 5

    configs = [
        (128, 128, 64, 3),
        (128, 128, 64, 2),
        (64, 128, 64, 2),
    ]
    sizes = [(4096, 4096, 4096), (8192, 8192, 8192)]

    for BM, BN, BK, S in configs:
        for M, N, K in sizes:
            results = [bench_trial(M, N, K, BM, BN, BK, S) for _ in range(n_trials)]
            mean = statistics.mean(results)
            stdev = statistics.stdev(results) if len(results) > 1 else 0
            vals = " ".join(f"{r:.1f}" for r in results)
            print(f"  {BM}x{BN} {S}s  {M}^3:  {mean:.1f} +/- {stdev:.1f} TFLOPS  [{vals}]")
