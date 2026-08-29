"""
Test high-stage GEMM configs to find those that exceed SMEM limits without SALA.
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


if __name__ == "__main__":
    print(f"=== High-Stage GEMM Test ({mode}) ===")
    M, N, K = 4096, 4096, 4096
    configs = [
        (128, 128, 64, 4),
        (128, 128, 64, 5),
        (128, 128, 64, 6),
        (128, 128, 64, 7),
        (128, 256, 64, 3),
        (128, 256, 64, 4),
        (256, 128, 64, 3),
        (256, 128, 64, 4),
    ]

    for BM, BN, BK, stages in configs:
        import gc; gc.collect(); torch.cuda.empty_cache()
        NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
        a = torch.randn((M, K), device="cuda", dtype=torch.float16)
        b = torch.randn((K, N), device="cuda", dtype=torch.float16).T.contiguous()
        c = torch.empty((M, N), device="cuda", dtype=torch.float16)
        a_desc = TensorDescriptor(a, a.shape, a.stride(), [BM, BK])
        b_desc = TensorDescriptor(b, b.shape, b.stride(), [BN, BK])
        c_desc = TensorDescriptor(c, c.shape, c.stride(), [BM, BN])
        grid = (min(NUM_SMS, triton.cdiv(M, BM) * triton.cdiv(N, BN)),)
        try:
            matmul_kernel_nested[grid](
                a_desc, b_desc, c_desc, M, N, K,
                BLOCK_SIZE_M=BM, BLOCK_SIZE_N=BN, BLOCK_SIZE_K=BK,
                GROUP_SIZE_M=8, NUM_SMS=NUM_SMS,
                num_warps=4, num_stages=stages,
                enable_warp_specialization=True,
            )
            torch.cuda.synchronize()
            ref = torch.mm(a, b.T)
            err = (c - ref).abs().max().item() / ref.abs().max().item()
            smem_est = (stages * BM * BK + stages * BN * BK + BM * BN) * 2  # FP16
            print(f"  {BM}x{BN} {stages}s: OK  (err={err:.6f}, est_smem={smem_est/1024:.0f} KB)")
        except Exception as e:
            smem_est = (stages * BM * BK + stages * BN * BK + BM * BN) * 2
            print(f"  {BM}x{BN} {stages}s: FAIL ({e}) (est_smem={smem_est/1024:.0f} KB)")
