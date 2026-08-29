"""
Benchmark configs that SALA enables (would exceed SMEM limit without SALA).
Compare TFLOPS against the best baseline config at each problem size.
"""
import os
import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

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


def benchmark(M, N, K, BM, BN, BK, stages, warmup=10, iters=100):
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

    try:
        for _ in range(warmup):
            launch()
        torch.cuda.synchronize()
    except Exception as e:
        return None, str(e)

    ref = torch.mm(a, b.T)
    err = (c - ref).abs().max().item() / ref.abs().max().item()
    if err > 0.05:
        return None, f"FAIL (err={err:.6f})"

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
    print(f"=== SALA-Enabled Configs Benchmark ({mode}) ===")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    sizes = [
        (2048, 2048, 2048),
        (4096, 4096, 4096),
        (8192, 8192, 8192),
    ]

    configs = [
        # Baseline-capable configs (for comparison)
        (128, 128, 64, 3, "baseline-best"),
        (128, 128, 64, 6, "baseline-max"),
        # SALA-only configs (exceed 228 KB without SALA)
        (128, 128, 64, 7, "SALA-only"),
        (128, 256, 64, 4, "SALA-only"),
        (256, 128, 64, 4, "SALA-only"),
    ]

    print(f"{'Config':<25} {'Tag':<14} {'2048^3':>8} {'4096^3':>8} {'8192^3':>8}")
    print("-" * 70)

    for BM, BN, BK, stages, tag in configs:
        label = f"{BM}x{BN} {stages}s"
        results = []
        for M, N, K in sizes:
            tflops, status = benchmark(M, N, K, BM, BN, BK, stages)
            results.append(tflops)
        r_str = [f"{t:>8.1f}" if t else f"{'N/A':>8}" for t in results]
        print(f"{label:<25} {tag:<14} {r_str[0]} {r_str[1]} {r_str[2]}")
