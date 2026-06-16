"""Shared cudaEvent timing and TFLOPS reporting for FA3 variant benches."""

from __future__ import annotations

import os
from typing import Callable

import torch

# H800 PCIe dense Tensor Core peaks (without structured sparsity).
# NVIDIA spec shows 1513 / 3026 -- those are WITH sparsity (2x).
# Dense: FP16/BF16 = 756.5 TFLOPS, FP8 = 1513 TFLOPS.
H800_PCIE_PEAK_F16_TFLOPS = 756.5
H800_PCIE_PEAK_FP8_TFLOPS = 1513.0


def peak_tflops_for_dtype(dtype) -> float:
    import torch

    if dtype == torch.float8_e4m3fn:
        return H800_PCIE_PEAK_FP8_TFLOPS
    return H800_PCIE_PEAK_F16_TFLOPS


def read_timer_options() -> tuple[int, int]:
    warmup = 50
    repeat = 200
    if env := os.getenv("CHOREO_TIMING_WARMUP"):
        value = int(env)
        if value >= 0:
            warmup = value
    if env := os.getenv("CHOREO_TIMING_REPEAT"):
        value = int(env)
        if value > 0:
            repeat = value
    return warmup, repeat


def fa3_attention_flops(
    batch: int,
    nheads: int,
    seqlen_q: int,
    seqlen_k: int,
    headdim: int,
    headdim_v: int | None,
    causal: bool,
) -> float:
    """Same as hopper/benchmark_attn.py flops()."""
    if causal:
        avg_seqlen = (max(0, seqlen_k - seqlen_q) + seqlen_k) / 2
    else:
        avg_seqlen = seqlen_k
    hd_v = headdim if headdim_v is None else headdim_v
    return batch * nheads * 2 * seqlen_q * avg_seqlen * (headdim + hd_v)


def cuda_event_timing(fn: Callable[[], None], warmup: int, repeat: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeat):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / repeat


def report_timing(
    label: str,
    avg_ms: float,
    flops: float,
    backend: str = "fa3",
    *,
    peak_tflops: float | None = None,
) -> None:
    tflops = flops / (avg_ms / 1000.0) / 1e12
    peak = (
        peak_tflops
        if peak_tflops is not None
        else H800_PCIE_PEAK_F16_TFLOPS
    )
    eff = (tflops / peak) * 100.0
    print(f"[{backend} {label}]")
    print(f"Timing avg ms: {avg_ms}")
    print(f"TFLOPS: {tflops}")
    print(f"HW efficiency: {eff}% (peak_ref={peak} TFLOPS)")
