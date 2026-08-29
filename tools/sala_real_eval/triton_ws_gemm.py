"""
Triton 3.7.0 Hopper Warp-Specialized GEMM
Based on 09-persistent-matmul-ws.py from Triton aref_auto_ws branch.
Uses tl.range(..., warp_specialize=True) to enable warp specialization.
"""

import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS):
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (tile_id % group_size_m)
    pid_n = (tile_id % num_pid_in_group) // group_size_m
    return pid_m, pid_n


@triton.jit
def matmul_kernel_ws(
    a_desc,
    b_desc,
    c_desc,
    M,
    N,
    K,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_SMS: tl.constexpr,
    WARP_SPECIALIZE: tl.constexpr,
):
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n

    tile_id_c = start_pid - NUM_SMS
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    for tile_id in tl.range(
        start_pid, num_tiles, NUM_SMS,
        flatten=True,
        warp_specialize=WARP_SPECIALIZE,
    ):
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

        if WARP_SPECIALIZE:
            tile_id_c = tile_id
            offs_am_c = offs_am
            offs_bn_c = offs_bn
        else:
            tile_id_c += NUM_SMS
            pid_m, pid_n = _compute_pid(
                tile_id_c, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS
            )
            offs_am_c = pid_m * BLOCK_SIZE_M
            offs_bn_c = pid_n * BLOCK_SIZE_N

        accumulator = accumulator.to(tl.float16)
        c_desc.store([offs_am_c, offs_bn_c], accumulator)


def matmul_tma_persistent(a, b, ws=False):
    assert a.shape[1] == b.shape[1], "Incompatible dimensions"
    assert a.dtype == b.dtype, "Incompatible dtypes"

    M, K = a.shape
    N, K = b.shape

    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count

    BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 64

    a_desc = TensorDescriptor(a, a.shape, a.stride(), [BLOCK_M, BLOCK_K])
    b_desc = TensorDescriptor(b, b.shape, b.stride(), [BLOCK_N, BLOCK_K])
    c_desc = TensorDescriptor(c, c.shape, c.stride(), [BLOCK_M, BLOCK_N])

    grid = (min(NUM_SMS, triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)),)

    matmul_kernel_ws[grid](
        a_desc, b_desc, c_desc,
        M, N, K,
        BLOCK_SIZE_M=BLOCK_M,
        BLOCK_SIZE_N=BLOCK_N,
        BLOCK_SIZE_K=BLOCK_K,
        GROUP_SIZE_M=8,
        NUM_SMS=NUM_SMS,
        WARP_SPECIALIZE=ws,
        num_warps=4,
        num_stages=3,
    )
    return c


def main():
    torch.manual_seed(0)
    M, N, K = 4096, 4096, 4096
    a = torch.randn((M, K), device="cuda", dtype=torch.float16)
    b = torch.randn((K, N), device="cuda", dtype=torch.float16)
    b = b.T.contiguous()

    # Validate correctness (no WS)
    print("--- No WS (persistent) ---")
    c_nows = matmul_tma_persistent(a, b, ws=False)

    # Validate correctness (WS)
    print("--- With WS (warp_specialize=True) ---")
    c_ws = matmul_tma_persistent(a, b, ws=True)

    # Check results
    ref = torch.matmul(a, b.T)
    nows_ok = torch.allclose(c_nows, ref, atol=1.0, rtol=0.01)
    ws_ok = torch.allclose(c_ws, ref, atol=1.0, rtol=0.01)
    print(f"No-WS correct: {nows_ok}")
    print(f"WS correct:    {ws_ok}")

    # Quick benchmark
    import time
    for label, ws_flag in [("No-WS", False), ("WS", True)]:
        # Warmup
        for _ in range(10):
            matmul_tma_persistent(a, b, ws=ws_flag)
        torch.cuda.synchronize()

        start = time.time()
        iters = 100
        for _ in range(iters):
            matmul_tma_persistent(a, b, ws=ws_flag)
        torch.cuda.synchronize()
        elapsed = time.time() - start

        tflops = 2.0 * M * N * K * iters / elapsed / 1e12
        print(f"{label}: {elapsed/iters*1000:.2f} ms, {tflops:.1f} TFLOPS")


if __name__ == "__main__":
    main()
