#!/usr/bin/env python3
"""FlashAttention (dao-ailab) baseline for choreo flash_mha_fwd BHSD benchmarks."""

from __future__ import annotations

import argparse

import torch
from flash_attn import flash_attn_func

from common import (
    BENCHMARK_CONFIGS,
    HEAD_DIM,
    cuda_event_timing,
    make_random_bhsd,
    read_timer_options,
    report_timing,
)


def run_flash_attn_baseline(verify: bool = False) -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    device = torch.device("cuda")
    warmup, repeat = read_timer_options()
    softmax_scale = (HEAD_DIM ** -0.5)  # flash_attn default; matches 1/sqrt(d)

    print("FlashAttention (flash_attn) causal decoder (DIM=64)")
    print(f"Warmup={warmup} Repeat={repeat}")
    print(
        "Layout: BSHD (batch,seqlen,heads,dim); BHSD<->BSHD transpose included "
        "in timed region to match a BHSD-first pipeline"
    )
    print("Math: causal=True  scale=1/sqrt(dim)")
    print("--------------------------------------------")

    all_ok = True
    for i, cfg in enumerate(BENCHMARK_CONFIGS):
        q, k, v = make_random_bhsd(cfg, device)
        # flash_attn_func expects (batch, seqlen, heads, dim), not BHSD.
        q_bshd = q.transpose(1, 2).contiguous()
        k_bshd = k.transpose(1, 2).contiguous()
        v_bshd = v.transpose(1, 2).contiguous()

        def launch() -> None:
            flash_attn_func(
                q_bshd,
                k_bshd,
                v_bshd,
                dropout_p=0.0,
                softmax_scale=softmax_scale,
                causal=cfg.is_causal,
            )

        avg_ms = cuda_event_timing(launch, warmup, repeat)
        report_timing(cfg, avg_ms, "flash_attn")

        if verify:
            ref = _reference_standard_attention(q, k, v, cfg.is_causal)
            out_bshd = flash_attn_func(
                q_bshd,
                k_bshd,
                v_bshd,
                softmax_scale=softmax_scale,
                causal=cfg.is_causal,
            )
            out = out_bshd.transpose(1, 2)
            max_err = (out.float() - ref.float()).abs().max().item()
            ok = max_err < 0.05
            print(
                f"[VERIFY flash_attn {cfg.label}] max_abs_err={max_err} "
                f"{'PASS' if ok else 'FAIL'}"
            )
            all_ok = all_ok and ok

        if i + 1 < len(BENCHMARK_CONFIGS):
            print("--------------------------------------------")

    if verify:
        print("Test Passed" if all_ok else "Test FAILED")
        return 0 if all_ok else 1
    print("Done")
    return 0


def _reference_standard_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal: bool
) -> torch.Tensor:
    """Standard attention reference (matches flash_attn numerics)."""
    import torch.nn.functional as F

    dim = q.size(-1)
    scores = torch.einsum("bhqd,bhkd->bhqk", q.float(), k.float())
    scores = scores / (dim ** 0.5)
    if is_causal:
        seq_q = q.size(2)
        seq_kv = k.size(2)
        past_len = seq_kv - seq_q
        mask = torch.tril(
            torch.ones(seq_q, seq_kv, device=scores.device),
            seq_kv - seq_q,
        )
        mask = mask.unsqueeze(0).unsqueeze(0)
        scores = scores.masked_fill(mask == 0, float("-inf"))
    weights = F.softmax(scores, dim=-1)
    return torch.einsum("bhqk,bhkd->bhqd", weights, v.float())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    raise SystemExit(run_flash_attn_baseline(verify=args.verify))


if __name__ == "__main__":
    main()
