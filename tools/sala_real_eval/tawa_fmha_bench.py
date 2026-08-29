"""
Benchmark FlexAttention (sliding-window causal) on Tawa with and without SALA.
Non-persistent kernel: uses TMA + warp specialization but no outer tile loop,
so cross-tile bar.sync is not needed.

Usage:
  rm -rf ~/.triton/cache/
  CUDA_VISIBLE_DEVICES=1 PYTHONPATH=/home/wsj/dev/triton-aref/python python3.10 tawa_fmha_bench.py

  rm -rf ~/.triton/cache/
  CUDA_VISIBLE_DEVICES=1 PYTHONPATH=/home/wsj/dev/triton-aref/python SALA_ENABLE=1 python3.10 tawa_fmha_bench.py
"""

import os
import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor
import math

mode = "SALA" if os.environ.get("SALA_ENABLE") == "1" else "Baseline"


def attention_sliding_window_pytorch(Q, K, V, window_size):
    B, H, M, D = Q.shape
    scores = torch.matmul(Q, K.transpose(2, 3)) / (D ** 0.5)
    q_idx = torch.arange(M, device=Q.device)[None, None, :, None]
    k_idx = torch.arange(M, device=Q.device)[None, None, None, :]
    causal_mask = q_idx >= k_idx
    window_mask = (q_idx - k_idx) <= window_size
    mask = causal_mask & window_mask
    scores = scores.masked_fill(~mask, float('-inf'))
    attn = torch.softmax(scores, dim=-1)
    output = torch.matmul(attn, V)
    return output


@triton.jit
def attention_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr,
    q_desc_ptr,
    k_desc_ptr,
    v_desc_ptr,
    stride_qb, stride_qh, stride_qm, stride_qk,
    stride_kb, stride_kh, stride_km, stride_kk,
    stride_vb, stride_vh, stride_vm, stride_vk,
    stride_ob, stride_oh, stride_om, stride_ok,
    B, H, M, D, window_size,
    scale: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    D_HEAD: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_m = tl.program_id(2)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_b = pid_b
    offs_h = pid_h
    offs_bh = offs_b * H + offs_h
    qvk_offset = offs_b.to(tl.int64) * stride_qb + offs_h.to(tl.int64) * stride_qh

    q = q_desc_ptr.load(
        [offs_bh * M + pid_m * BLOCK_SIZE_M, 0]
    )

    m_i = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32) - float('inf')
    l_i = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_SIZE_M, D_HEAD), dtype=tl.float32)

    start_n = tl.maximum(0, (pid_m * BLOCK_SIZE_M) - window_size)
    end_n = (pid_m + 1) * BLOCK_SIZE_M
    start_block = start_n // BLOCK_SIZE_N
    end_block = tl.cdiv(end_n, BLOCK_SIZE_N)

    offs_kv = offs_bh * M + start_block * BLOCK_SIZE_N
    mask_q = offs_m < M

    for pid_n in range(start_block, end_block):
        offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        mask_k = offs_n < M

        k = k_desc_ptr.load([offs_kv, 0])
        scores = tl.dot(q, tl.trans(k)) * scale

        q_idx = offs_m[:, None]
        k_idx = offs_n[None, :]
        causal_mask = q_idx >= k_idx
        window_mask = (q_idx - k_idx) <= window_size
        mask = causal_mask & window_mask
        mask = mask & mask_q[:, None] & mask_k[None, :]
        scores = tl.where(mask, scores, float('-inf'))

        m_ij = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        beta = tl.exp(scores - m_new[:, None])

        l_ij = tl.sum(beta, axis=1)
        l_new = alpha * l_i + l_ij

        v = v_desc_ptr.load([offs_kv, 0])
        beta = beta.to(v.dtype)
        pv = tl.dot(beta, v)

        acc = acc * alpha[:, None] + pv
        m_i = m_new
        l_i = l_new
        offs_kv += BLOCK_SIZE_N

    acc = acc / l_i[:, None]
    o_ptrs = o_ptr + offs_b * stride_ob + offs_h * stride_oh + offs_m[:, None] * stride_om + tl.arange(0, D_HEAD)[None, :] * stride_ok
    tl.store(o_ptrs, acc, mask=mask_q[:, None])


def get_fwd_tma_descriptors(q, k, v, Z, H, N_CTX, HEAD_DIM, BLOCK_M, BLOCK_N):
    desc_q = TensorDescriptor(q, [Z * H * N_CTX, HEAD_DIM], [HEAD_DIM, 1], [BLOCK_M, HEAD_DIM])
    desc_k = TensorDescriptor(k, [Z * H * N_CTX, HEAD_DIM], [HEAD_DIM, 1], [BLOCK_N, HEAD_DIM])
    desc_v = TensorDescriptor(v, [Z * H * N_CTX, HEAD_DIM], [HEAD_DIM, 1], [BLOCK_N, HEAD_DIM])
    return desc_q, desc_k, desc_v


def attention_triton(Q, K, V, window_size, BM=128, BN=128, num_stages=2):
    B, H, M, D = Q.shape
    output = torch.empty_like(Q)
    grid = (B, H, triton.cdiv(M, BM))
    scale = 1.0 / math.sqrt(D)

    desc_q, desc_k, desc_v = get_fwd_tma_descriptors(
        Q, K, V, B, H, M, D, BM, BN,
    )

    attention_kernel[grid](
        Q, K, V, output,
        desc_q, desc_k, desc_v,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        B, H, M, D, window_size,
        scale=scale,
        BLOCK_SIZE_M=BM,
        BLOCK_SIZE_N=BN,
        D_HEAD=D,
        num_warps=8,
        num_stages=num_stages,
        mma_depth=1,
        enable_warp_specialization=True,
        math_wg_pipe=False,
    )
    return output


def bench_attention(B, H, M, D, window_size, BM=128, BN=128, num_stages=2, warmup=10, iters=100):
    Q = torch.randn((B, H, M, D), device='cuda', dtype=torch.float16)
    K = torch.randn((B, H, M, D), device='cuda', dtype=torch.float16)
    V = torch.randn((B, H, M, D), device='cuda', dtype=torch.float16)

    try:
        for _ in range(warmup):
            out = attention_triton(Q, K, V, window_size, BM, BN, num_stages)
        torch.cuda.synchronize()
    except Exception as e:
        return None, str(e)

    # Correctness (quick check on small size)
    if M <= 2048:
        ref = attention_sliding_window_pytorch(Q, K, V, window_size)
        err = (out - ref).abs().max().item()
        if err > 0.1:
            return None, f"FAIL (max_err={err:.4f})"

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(iters):
        attention_triton(Q, K, V, window_size, BM, BN, num_stages)
    end_event.record()
    torch.cuda.synchronize()
    elapsed_ms = start_event.elapsed_time(end_event)

    # FLOPS for causal sliding-window attention (approximate)
    flops_per_matmul = 2.0 * B * H * M * M * D
    total_flops = 2 * flops_per_matmul * 0.5  # causal => ~half
    tflops = total_flops * iters / (elapsed_ms / 1000) / 1e12
    ms_per_iter = elapsed_ms / iters
    return tflops, f"OK ({ms_per_iter:.3f} ms)"


if __name__ == "__main__":
    print(f"=== Tawa FlexAttention Benchmark ({mode}) ===")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Triton: {triton.__file__}")
    print()

    configs = [
        # (B, H, M, D, window, BM, BN, stages)
        (4, 4, 2048, 128, 128, 128, 128, 2),
        (4, 4, 4096, 128, 128, 128, 128, 2),
        (4, 4, 8192, 128, 128, 128, 128, 2),
        (2, 8, 4096, 128, 256, 128, 128, 2),
        (2, 8, 8192, 128, 256, 128, 128, 2),
        (4, 4, 4096, 128, 128, 128, 128, 3),
        (4, 4, 8192, 128, 128, 128, 128, 3),
    ]

    print(f"{'Config':<40} {'TFLOPS':>8}  {'Status'}")
    print("-" * 70)

    for B, H, M, D, W, BM, BN, S in configs:
        label = f"B={B} H={H} M={M} D={D} W={W} {BM}x{BN} {S}s"
        tflops, status = bench_attention(B, H, M, D, W, BM, BN, S)
        if tflops is not None:
            print(f"{label:<40} {tflops:>8.1f}  {status}")
        else:
            print(f"{label:<40} {'N/A':>8}  {status}")
