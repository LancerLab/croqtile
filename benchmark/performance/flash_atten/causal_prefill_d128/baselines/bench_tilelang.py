#!/usr/bin/env python3
"""TileLang flash attention baseline for causal_prefill_d128."""

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

import tilelang  # noqa: E402
import tilelang.language as T  # noqa: E402


@tilelang.jit(
    out_idx=[3],
    pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
)
def flashattn_kernel(
    batch, heads, seq_len, dim, is_causal,
    block_M=128, block_N=128, num_stages=1, threads=128,
):
    scale = (1.0 / dim) ** 0.5 * 1.44269504  # log2(e)
    shape = [batch, seq_len, heads, dim]
    dtype = T.bfloat16
    accum_dtype = T.float32

    @T.prim_func
    def main(
        Q: T.Tensor(shape, dtype),
        K: T.Tensor(shape, dtype),
        V: T.Tensor(shape, dtype),
        Output: T.Tensor(shape, dtype),
    ):
        with T.Kernel(
            T.ceildiv(seq_len, block_M), heads, batch, threads=threads
        ) as (bx, by, bz):
            Q_shared = T.alloc_shared([block_M, dim], dtype)
            K_shared = T.alloc_shared([block_N, dim], dtype)
            V_shared = T.alloc_shared([block_N, dim], dtype)
            O_shared = T.alloc_shared([block_M, dim], dtype)
            acc_s = T.alloc_fragment([block_M, block_N], accum_dtype)
            acc_s_cast = T.alloc_fragment([block_M, block_N], dtype)
            acc_o = T.alloc_fragment([block_M, dim], accum_dtype)
            scores_max = T.alloc_fragment([block_M], accum_dtype)
            scores_max_prev = T.alloc_fragment([block_M], accum_dtype)
            scores_scale = T.alloc_fragment([block_M], accum_dtype)
            scores_sum = T.alloc_fragment([block_M], accum_dtype)
            logsum = T.alloc_fragment([block_M], accum_dtype)

            T.copy(Q[bz, bx * block_M : (bx + 1) * block_M, by, :], Q_shared)
            T.fill(acc_o, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(accum_dtype))

            loop_range = (
                T.min(
                    T.ceildiv(seq_len, block_N),
                    T.ceildiv((bx + 1) * block_M, block_N),
                )
                if is_causal
                else T.ceildiv(seq_len, block_N)
            )

            for k in T.Pipelined(loop_range, num_stages=num_stages):
                T.copy(
                    K[bz, k * block_N : (k + 1) * block_N, by, :], K_shared
                )
                if is_causal:
                    for i, j in T.Parallel(block_M, block_N):
                        acc_s[i, j] = T.if_then_else(
                            bx * block_M + i >= k * block_N + j,
                            0,
                            -T.infinity(acc_s.dtype),
                        )
                else:
                    for i, j in T.Parallel(block_M, block_N):
                        acc_s[i, j] = T.if_then_else(
                            k * block_N + j >= seq_len,
                            -T.infinity(acc_s.dtype),
                            0,
                        )
                T.gemm(
                    Q_shared,
                    K_shared,
                    acc_s,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullRow,
                )

                T.copy(scores_max, scores_max_prev)
                T.fill(scores_max, -T.infinity(accum_dtype))
                T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                for i in T.Parallel(block_M):
                    scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
                for i in T.Parallel(block_M):
                    scores_scale[i] = T.exp2(
                        scores_max_prev[i] * scale - scores_max[i] * scale
                    )
                for i, j in T.Parallel(block_M, block_N):
                    acc_s[i, j] = T.exp2(
                        acc_s[i, j] * scale - scores_max[i] * scale
                    )
                T.reduce_sum(acc_s, scores_sum, dim=1)
                for i in T.Parallel(block_M):
                    logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                T.copy(acc_s, acc_s_cast)

                for i, j in T.Parallel(block_M, dim):
                    acc_o[i, j] *= scores_scale[i]

                T.copy(
                    V[bz, k * block_N : (k + 1) * block_N, by, :], V_shared
                )
                T.gemm(
                    acc_s_cast,
                    V_shared,
                    acc_o,
                    policy=T.GemmWarpPolicy.FullRow,
                )

            for i, j in T.Parallel(block_M, dim):
                acc_o[i, j] /= logsum[i]
            T.copy(acc_o, O_shared)
            T.copy(
                O_shared,
                Output[bz, bx * block_M : (bx + 1) * block_M, by, :],
            )

    return main


def run_tilelang_baseline() -> int:
    apply_gpu_from_cli()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    device = torch.device("cuda")
    warmup, repeat = read_timer_options()
    dtype_torch = torch.bfloat16
    peak_ref = peak_tflops_for_dtype(dtype_torch)

    print(f"Variant: {VARIANT_ID}")
    print("Backend: TileLang flash attention")
    idx = torch.cuda.current_device()
    print(f"Device: {torch.cuda.get_device_name(idx)} (current_device={idx})")
    print(f"Dtype: {dtype_torch}")
    print(f"HW peak reference: {peak_ref} TFLOPS")
    print(f"Warmup={warmup} Repeat={repeat}")
    print("Layout: BSHD (batch, seqlen, heads, dim)")
    print("--------------------------------------------")

    for i, case in enumerate(BENCHMARK_CONFIGS):
        B, sq, sk = case.batch, case.seqlen_q, case.seqlen_k
        hq, d = case.nheads_q, case.headdim

        assert sq == sk, "TileLang MHA fwd requires seqlen_q == seqlen_k"

        kernel = flashattn_kernel(
            B, hq, sq, d, case.causal,
            block_M=128, block_N=128, num_stages=2, threads=256,
        )

        gen = torch.Generator(device=device)
        gen.manual_seed(42)
        q = torch.empty(B, sq, hq, d, device=device, dtype=dtype_torch)
        k = torch.empty(B, sk, hq, d, device=device, dtype=dtype_torch)
        v = torch.empty(B, sk, hq, d, device=device, dtype=dtype_torch)
        q.uniform_(-0.5, 0.5, generator=gen)
        k.uniform_(-0.5, 0.5, generator=gen)
        v.uniform_(-1.0, 1.0, generator=gen)

        def launch() -> None:
            kernel(q, k, v)

        for _ in range(3):
            launch()
        torch.cuda.synchronize()

        avg_ms = cuda_event_timing(launch, warmup, repeat)
        flops = fa3_attention_flops(
            B, hq, sq, sk, d, case.headdim_v, case.causal
        )
        report_timing(case.label, avg_ms, flops, backend="tilelang",
                      peak_tflops=peak_ref)

        if i + 1 < len(BENCHMARK_CONFIGS):
            print("--------------------------------------------")

    print("Done")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_tilelang_baseline())
