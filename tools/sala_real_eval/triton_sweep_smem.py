"""
Sweep Triton tile configurations and measure SMEM via ncu.
Tests multiple tile sizes and pipeline stages.
Outputs results as a table.
"""

import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor
import sys
import subprocess
import re
import os


@triton.jit
def _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS):
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (tile_id % group_size_m)
    pid_n = (tile_id % num_pid_in_group) // group_size_m
    return pid_m, pid_n


@triton.jit
def matmul_kernel(
    a_desc, b_desc, c_desc,
    M, N, K,
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


def run_config(bm, bn, bk, stages, warps, ws):
    M, N, K = 4096, 4096, 4096
    a = torch.randn((M, K), device="cuda", dtype=torch.float16)
    b = torch.randn((K, N), device="cuda", dtype=torch.float16)
    b = b.T.contiguous()
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count

    a_desc = TensorDescriptor(a, a.shape, a.stride(), [bm, bk])
    b_desc = TensorDescriptor(b, b.shape, b.stride(), [bn, bk])
    c_desc = TensorDescriptor(c, c.shape, c.stride(), [bm, bn])

    grid = (min(NUM_SMS, triton.cdiv(M, bm) * triton.cdiv(N, bn)),)

    matmul_kernel[grid](
        a_desc, b_desc, c_desc,
        M, N, K,
        BLOCK_SIZE_M=bm, BLOCK_SIZE_N=bn, BLOCK_SIZE_K=bk,
        GROUP_SIZE_M=8, NUM_SMS=NUM_SMS,
        WARP_SPECIALIZE=ws,
        num_warps=warps, num_stages=stages,
    )
    torch.cuda.synchronize()


if __name__ == "__main__":
    bm = int(sys.argv[1])
    bn = int(sys.argv[2])
    bk = int(sys.argv[3])
    stages = int(sys.argv[4])
    warps = int(sys.argv[5])
    ws = sys.argv[6] == "1"

    ws_label = "WS" if ws else "no_WS"
    print(f"Config: {bm}x{bn} BK={bk} stages={stages} warps={warps} {ws_label}")
    run_config(bm, bn, bk, stages, warps, ws)
    print(f"Done: {bm}x{bn} BK={bk} stages={stages} warps={warps} {ws_label}")
