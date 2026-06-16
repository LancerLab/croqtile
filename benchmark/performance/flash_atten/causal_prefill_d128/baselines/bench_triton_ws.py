#!/usr/bin/env python3
"""Triton fused attention WITH Tawa/aref warp specialization.

This benchmark requires Triton built from the aref_auto_ws branch:
  https://github.com/triton-lang/triton/tree/aref_auto_ws

The aref branch adds automatic warp-specialization compiler passes (NVWS
dialect) that partition the kernel into TMA-load producer and MMA consumer
warp groups, overlapping data movement with computation.

Layout: BHSD internally (tensors are transposed from BSHD).
Activate the triton-aref venv:  source ~/.env/triton-aref/bin/activate
"""

import sys
from pathlib import Path

import torch

VARIANT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VARIANT_DIR.parent / "scripts"))
sys.path.insert(0, str(VARIANT_DIR))

from _fa3_common import BenchCase, apply_gpu_from_cli  # noqa: E402
from _timing import (  # noqa: E402
    cuda_event_timing,
    fa3_attention_flops,
    peak_tflops_for_dtype,
    read_timer_options,
    report_timing,
)
from config import BENCHMARK_CONFIGS, VARIANT_ID  # noqa: E402

import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"


def is_hopper():
    return is_cuda() and torch.cuda.get_device_capability()[0] == 9


def _has_aref_support():
    """Check if this Triton build supports aref warp specialization."""
    try:
        import triton.runtime.jit as jit_mod
        sig = __import__('inspect').signature(jit_mod.JITFunction.run)
        return 'enable_warp_specialization' in sig.parameters or True
    except Exception:
        pass
    return False


@triton.jit
def _attn_fwd_inner_ws(
    acc, l_i, m_i, q,
    desc_k, desc_v,
    offs_hz, start_m, qk_scale,
    BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,
    STAGE: tl.constexpr, offs_m: tl.constexpr, offs_n: tl.constexpr,
    N_CTX: tl.constexpr, fp8_v: tl.constexpr,
):
    if STAGE == 1:
        lo = 0
        hi = start_m * BLOCK_M
    elif STAGE == 2:
        lo = start_m * BLOCK_M
        hi = (start_m + 1) * BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_M)
    else:
        lo = 0
        hi = N_CTX
    lo = lo.to(tl.int32)
    hi = hi.to(tl.int32)
    offs_kv = offs_hz * N_CTX + lo

    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        k = desc_k.load([offs_kv, 0])
        qk = tl.dot(q, k.T)
        if STAGE == 2:
            mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            qk = qk * qk_scale + tl.where(mask, 0, -1.0e6)
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk -= m_ij[:, None]
        else:
            m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
            qk = qk * qk_scale - m_ij[:, None]
        p = tl.math.exp2(qk)
        alpha = tl.math.exp2(m_i - m_ij)
        l_ij = tl.sum(p, 1)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]
        if fp8_v:
            v = desc_v.load([offs_hz * HEAD_DIM, start_n])
            p = p.to(tl.float8e5)
            acc = tl.dot(p, v.T, acc)
        else:
            v = desc_v.load([offs_kv, 0])
            p = p.to(tl.float16)
            acc = tl.dot(p, v, acc)
        m_i = m_ij
        offs_kv += BLOCK_N
    return acc, l_i, m_i


@triton.jit
def _attn_fwd_ws(
    Q, K, V,
    q_desc_ptr, k_desc_ptr, v_desc_ptr, o_desc_ptr,
    sm_scale, M, Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_oz, stride_oh, stride_om, stride_on,
    Z, H, N_CTX,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    STAGE: tl.constexpr,
):
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    qvk_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh

    O_block_ptr = tl.make_block_ptr(
        base=Out + qvk_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_om, stride_on),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    qk_scale = sm_scale
    qk_scale *= 1.44269504  # 1/log(2)
    q = q_desc_ptr.load([off_hz * N_CTX + start_m * BLOCK_M, 0])

    if STAGE & 1:
        acc, l_i, m_i = _attn_fwd_inner_ws(
            acc, l_i, m_i, q,
            k_desc_ptr, v_desc_ptr,
            off_hz, start_m, qk_scale,
            BLOCK_M, HEAD_DIM, BLOCK_N,
            4 - STAGE, offs_m, offs_n,
            N_CTX,
            V.dtype.element_ty == tl.float8e5,
        )
    if STAGE & 2:
        acc, l_i, m_i = _attn_fwd_inner_ws(
            acc, l_i, m_i, q,
            k_desc_ptr, v_desc_ptr,
            off_hz, start_m, qk_scale,
            BLOCK_M, HEAD_DIM, BLOCK_N,
            2, offs_m, offs_n,
            N_CTX,
            V.dtype.element_ty == tl.float8e5,
        )
    m_i += tl.math.log2(l_i)
    acc = acc / l_i[:, None]
    m_ptrs = M + off_hz * N_CTX + offs_m
    tl.store(m_ptrs, m_i)
    tl.store(O_block_ptr, acc.to(Out.type.element_ty))


def triton_ws_attention_fwd(q, k, v, causal, sm_scale):
    """Triton fused attention with aref warp specialization.

    q, k, v: [B, H, S, D] contiguous fp16.
    Returns output [B, H, S, D].
    """
    HEAD_DIM_K = q.shape[-1]
    o = torch.empty_like(q)
    stage = 3 if causal else 1
    Z, H, N_CTX = q.shape[:3]
    BLOCK_M, BLOCK_N = 128, 128
    NUM_WARPS = 8
    NUM_STAGES = 2

    M = torch.empty(
        (q.shape[0], q.shape[1], q.shape[2]),
        device=q.device, dtype=torch.float32,
    )

    desc_q = TensorDescriptor(
        q, [Z * H * N_CTX, HEAD_DIM_K], [HEAD_DIM_K, 1], [BLOCK_M, HEAD_DIM_K],
    )
    desc_k = TensorDescriptor(
        k, [Z * H * N_CTX, HEAD_DIM_K], [HEAD_DIM_K, 1], [BLOCK_N, HEAD_DIM_K],
    )
    desc_v = TensorDescriptor(
        v, [Z * H * N_CTX, HEAD_DIM_K], [HEAD_DIM_K, 1], [BLOCK_N, HEAD_DIM_K],
    )
    desc_o = TensorDescriptor(
        o, [Z * H * N_CTX, HEAD_DIM_K], [HEAD_DIM_K, 1], [BLOCK_M, HEAD_DIM_K],
    )

    WG_SPEC = (("tma_load", NUM_WARPS, 4), ("mma", 0, NUM_WARPS))

    grid = (triton.cdiv(N_CTX, BLOCK_M), Z * H, 1)

    _attn_fwd_ws[grid](
        q, k, v,
        desc_q, desc_k, desc_v, desc_o,
        sm_scale, M, o,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        Z, H,
        N_CTX=N_CTX,
        HEAD_DIM=HEAD_DIM_K,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        STAGE=stage,
        num_stages=NUM_STAGES,
        num_warps=NUM_WARPS,
        mma_depth=1,
        enable_warp_specialization=True,
        math_wg_pipe=True,
        wg_spec_override=WG_SPEC,
    )
    return o


def run_triton_ws_baseline() -> int:
    apply_gpu_from_cli()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    device = torch.device("cuda")
    warmup, repeat = read_timer_options()
    dtype_torch = torch.float16
    peak_ref = peak_tflops_for_dtype(dtype_torch)

    print(f"Variant: {VARIANT_ID}")
    print("Backend: Triton + Tawa/aref warp specialization (aref_auto_ws)")
    idx = torch.cuda.current_device()
    print(f"Device: {torch.cuda.get_device_name(idx)} (current_device={idx})")
    print(f"Dtype: {dtype_torch}")
    print(f"Triton version: {triton.__version__}")
    print(f"HW peak reference: {peak_ref} TFLOPS")
    print(f"Warmup={warmup} Repeat={repeat}")
    print("Layout: BHSD internally (BSHD transposed)")
    print("Config: BM=128 BN=128 stages=2 warps=8 WG_SPEC=mma_first")
    print("--------------------------------------------")

    for i, case in enumerate(BENCHMARK_CONFIGS):
        B, sq, sk = case.batch, case.seqlen_q, case.seqlen_k
        hq, d = case.nheads_q, case.headdim

        assert sq == sk, "Triton FA requires seqlen_q == seqlen_k"

        q_bhsd = torch.randn(
            B, hq, sq, d, device=device, dtype=dtype_torch
        ).contiguous()
        k_bhsd = torch.randn(
            B, hq, sk, d, device=device, dtype=dtype_torch
        ).contiguous()
        v_bhsd = torch.randn(
            B, hq, sk, d, device=device, dtype=dtype_torch
        ).contiguous()

        sm_scale = d ** -0.5

        for _ in range(3):
            triton_ws_attention_fwd(
                q_bhsd, k_bhsd, v_bhsd, case.causal, sm_scale
            )
        torch.cuda.synchronize()

        def launch() -> None:
            triton_ws_attention_fwd(
                q_bhsd, k_bhsd, v_bhsd, case.causal, sm_scale
            )

        avg_ms = cuda_event_timing(launch, warmup, repeat)
        flops = fa3_attention_flops(
            B, hq, sq, sk, d, case.headdim_v, case.causal
        )
        report_timing(
            case.label, avg_ms, flops, backend="triton_ws",
            peak_tflops=peak_ref,
        )

        if i + 1 < len(BENCHMARK_CONFIGS):
            print("--------------------------------------------")

    print("Done")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_triton_ws_baseline())
