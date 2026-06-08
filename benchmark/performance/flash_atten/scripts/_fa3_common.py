"""Shared Hopper FA3 baseline runner for flash_atten variants."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch

VARIANTS_ROOT = Path(__file__).resolve().parent


def apply_gpu_from_cli() -> None:
    """Honor --gpu when bench_fa3.py is invoked directly (optional)."""
    import argparse
    import os

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--gpu", default=None)
    args, _ = parser.parse_known_args()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)


if str(VARIANTS_ROOT) not in sys.path:
    sys.path.insert(0, str(VARIANTS_ROOT))

from _timing import (  # noqa: E402
    cuda_event_timing,
    fa3_attention_flops,
    peak_tflops_for_dtype,
    read_timer_options,
    report_timing,
)


@dataclass(frozen=True)
class BenchCase:
    batch: int
    seqlen_q: int
    seqlen_k: int
    nheads_q: int
    nheads_kv: int
    headdim: int
    headdim_v: int
    causal: bool
    label: str

    @property
    def nheads(self) -> int:
        """FLOPs use query head count."""
        return self.nheads_q


def _require_fa3() -> None:
    try:
        import flash_attn_interface  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "flash_attn_interface not found. Install hopper FA3 in ml-hopper:\n"
            "  cd flash-attention/hopper && pip install . --no-build-isolation"
        ) from exc


def _check_fp8_build() -> None:
    from flash_attn_config import CONFIG

    if CONFIG["build_flags"].get("FLASHATTENTION_DISABLE_FP8", True):
        raise RuntimeError(
            "FP8 kernels not in this FA3 build. Reinstall with "
            "FLASH_ATTENTION_DISABLE_FP8=FALSE"
        )


def make_qkv(
    case: BenchCase,
    device: torch.device,
    dtype: torch.dtype,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float | None, float | None, float | None]:
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    B, sq, sk = case.batch, case.seqlen_q, case.seqlen_k
    hq, hkv, d, dv = case.nheads_q, case.nheads_kv, case.headdim, case.headdim_v

    q_shape = (B, sq, hq, d)
    kv_shape = (B, sk, hkv, dv if dv != d else d)

    if dtype == torch.float8_e4m3fn:
        q_ref = torch.empty(q_shape, device=device, dtype=torch.bfloat16)
        k_ref = torch.empty(kv_shape, device=device, dtype=torch.bfloat16)
        v_ref = torch.empty(kv_shape, device=device, dtype=torch.bfloat16)
        q_ref.uniform_(-0.5, 0.5, generator=gen)
        k_ref.uniform_(-0.5, 0.5, generator=gen)
        v_ref.uniform_(-1.0, 1.0, generator=gen)
        q = q_ref.to(dtype)
        k = k_ref.to(dtype)
        v = v_ref.to(dtype)
        q_descale = torch.rand(B, hkv, device=device, dtype=torch.float32) * 2
        k_descale = torch.rand(B, hkv, device=device, dtype=torch.float32) * 2
        v_descale = torch.rand(B, hkv, device=device, dtype=torch.float32) * 2
        return q, k, v, q_descale, k_descale, v_descale

    q = torch.empty(q_shape, device=device, dtype=dtype)
    k = torch.empty(kv_shape, device=device, dtype=dtype)
    v = torch.empty(kv_shape, device=device, dtype=dtype)
    q.uniform_(-0.5, 0.5, generator=gen)
    k.uniform_(-0.5, 0.5, generator=gen)
    v.uniform_(-1.0, 1.0, generator=gen)
    return q, k, v, None, None, None


def run_fa3_baseline(
    variant_name: str,
    cases: list[BenchCase],
    dtype: torch.dtype,
    *,
    require_fp8: bool = False,
    extra_header: str | None = None,
) -> int:
    apply_gpu_from_cli()
    _require_fa3()
    if require_fp8:
        _check_fp8_build()

    from flash_attn_interface import flash_attn_func

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    device = torch.device("cuda")
    warmup, repeat = read_timer_options()
    print(f"Variant: {variant_name}")
    print("Backend: FlashAttention-3 (flash_attn_interface)")
    idx = torch.cuda.current_device()
    print(f"Device: {torch.cuda.get_device_name(idx)} (current_device={idx})")
    print(f"Dtype: {dtype}")
    peak_ref = peak_tflops_for_dtype(dtype)
    print(f"HW peak reference: {peak_ref} TFLOPS (FP8 uses 2x F16 peak)")
    print(f"Warmup={warmup} Repeat={repeat}")
    print("Layout: BSHD (batch, seqlen, heads, dim)")
    if extra_header:
        print(extra_header)
    print("--------------------------------------------")

    for i, case in enumerate(cases):
        q, k, v, q_descale, k_descale, v_descale = make_qkv(case, device, dtype)
        scale = case.headdim ** -0.5

        def launch() -> None:
            flash_attn_func(
                q,
                k,
                v,
                softmax_scale=scale,
                causal=case.causal,
                q_descale=q_descale,
                k_descale=k_descale,
                v_descale=v_descale,
            )

        for _ in range(3):
            launch()
        torch.cuda.synchronize()

        avg_ms = cuda_event_timing(launch, warmup, repeat)
        flops = fa3_attention_flops(
            case.batch,
            case.nheads,
            case.seqlen_q,
            case.seqlen_k,
            case.headdim,
            case.headdim_v,
            case.causal,
        )
        report_timing(case.label, avg_ms, flops, peak_tflops=peak_ref)

        if i + 1 < len(cases):
            print("--------------------------------------------")

    print("Done")
    return 0
