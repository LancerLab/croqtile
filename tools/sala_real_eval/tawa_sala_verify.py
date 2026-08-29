"""
Verify SALA correctness on Tawa NVWS GEMM with multiple problem sizes.
Must be run with PYTHONPATH=/home/wsj/dev/triton-aref/python
"""

import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor
import sys

print(f"Triton version: {triton.__version__}")
print(f"Triton path: {triton.__file__}")


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


def run_test(M, N, K):
    a = torch.randn((M, K), device="cuda", dtype=torch.float16)
    b = torch.randn((K, N), device="cuda", dtype=torch.float16)
    b = b.T.contiguous()
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 64
    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count

    a_desc = TensorDescriptor(a, a.shape, a.stride(), [BLOCK_M, BLOCK_K])
    b_desc = TensorDescriptor(b, b.shape, b.stride(), [BLOCK_N, BLOCK_K])
    c_desc = TensorDescriptor(c, c.shape, c.stride(), [BLOCK_M, BLOCK_N])

    grid = (min(NUM_SMS, triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)),)

    matmul_kernel_nested[grid](
        a_desc, b_desc, c_desc,
        M, N, K,
        BLOCK_SIZE_M=BLOCK_M,
        BLOCK_SIZE_N=BLOCK_N,
        BLOCK_SIZE_K=BLOCK_K,
        GROUP_SIZE_M=8,
        NUM_SMS=NUM_SMS,
        num_warps=4,
        num_stages=3,
        enable_warp_specialization=True,
    )
    torch.cuda.synchronize()

    ref = torch.mm(a, b.T)
    err = (c - ref).abs().max().item() / ref.abs().max().item()
    status = "PASS" if err < 0.05 else "FAIL"
    print(f"  [{M}x{N}x{K}] rel_err={err:.6f} ({status})")
    return err < 0.05


if __name__ == "__main__":
    sizes = [
        (2048, 2048, 2048),
        (4096, 4096, 4096),
        (8192, 8192, 8192),
        (4096, 4096, 8192),
        (1024, 4096, 4096),
    ]
    all_pass = True
    for M, N, K in sizes:
        if not run_test(M, N, K):
            all_pass = False
    print(f"\nAll tests: {'PASS' if all_pass else 'FAIL'}")
    sys.exit(0 if all_pass else 1)
