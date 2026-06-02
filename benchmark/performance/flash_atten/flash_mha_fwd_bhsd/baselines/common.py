"""Shared benchmark utilities for flash attention baselines.

Matches choreo mha_helper.hpp:
  - BHSD layout (batch, heads, seq, dim)
  - DIM=64, decoder causal self-attention (q_seq == kv_seq)
  - FLOPS: 4 * B * H * q_seq * kv_seq * dim, halved when causal
  - Timing: cudaEvent warmup + timed repeat loop (no L2 cache flush)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable

import torch

HEAD_DIM = 64
H800_PCIE_PEAK_F16_TFLOPS = 1513.0


@dataclass(frozen=True)
class BenchConfig:
    batch: int
    heads: int
    q_seq: int
    kv_seq: int
    is_causal: bool
    label: str

    @property
    def dim(self) -> int:
        return HEAD_DIM


# Same cases as choreo flash_mha_fwd_bhsd/choreo/*.co host configs.
BENCHMARK_CONFIGS: list[BenchConfig] = [
    BenchConfig(2, 32, 512, 512, True, "B=2 H=32 SEQ=512 decoder"),
    BenchConfig(2, 32, 1024, 1024, True, "B=2 H=32 SEQ=1024 decoder"),
    BenchConfig(2, 32, 2048, 2048, True, "B=2 H=32 SEQ=2048 decoder"),
    BenchConfig(2, 32, 4096, 4096, True, "B=2 H=32 SEQ=4096 decoder"),
    BenchConfig(1, 32, 8192, 8192, True, "B=1 H=32 SEQ=8192 decoder"),
]


def read_timer_options() -> tuple[int, int]:
    """Mirror mha_helper read_timer_options (warmup=10, repeat=50)."""
    warmup = 10
    repeat = 50
    if env := os.getenv("CHOREO_TIMING_WARMUP"):
        value = int(env)
        if value >= 0:
            warmup = value
    if env := os.getenv("CHOREO_TIMING_REPEAT"):
        value = int(env)
        if value > 0:
            repeat = value
    return warmup, repeat


def attention_flops(cfg: BenchConfig) -> float:
    """Same formula as mha_helper::attention_flops."""
    flops = (
        2.0
        * 2.0
        * cfg.batch
        * cfg.heads
        * cfg.q_seq
        * cfg.kv_seq
        * cfg.dim
    )
    if cfg.is_causal:
        flops *= 0.5
    return flops


def cuda_event_timing(fn: Callable[[], None], warmup: int, repeat: int) -> float:
    """Mirror choreo::timing: warmup, sync, cudaEvent over repeat, return avg ms."""
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


def report_timing(cfg: BenchConfig, avg_ms: float, backend: str) -> None:
    flops = attention_flops(cfg)
    tflops = flops / (avg_ms / 1000.0) / 1e12
    eff = (tflops / H800_PCIE_PEAK_F16_TFLOPS) * 100.0
    print(f"[{backend} {cfg.label}]")
    print(f"Timing avg ms: {avg_ms}")
    print(f"TFLOPS: {tflops}")
    print(f"HW efficiency: {eff}%")


def make_random_bhsd(
    cfg: BenchConfig, device: torch.device, seed: int = 42
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Same value ranges as mha_helper fill_random_bhsd."""
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    shape_q = (cfg.batch, cfg.heads, cfg.q_seq, cfg.dim)
    shape_kv = (cfg.batch, cfg.heads, cfg.kv_seq, cfg.dim)
    q = torch.empty(shape_q, dtype=torch.float16, device=device)
    k = torch.empty(shape_kv, dtype=torch.float16, device=device)
    v = torch.empty(shape_kv, dtype=torch.float16, device=device)
    q.uniform_(-0.5, 0.5, generator=gen)
    k.uniform_(-0.5, 0.5, generator=gen)
    v.uniform_(-1.0, 1.0, generator=gen)
    return q, k, v
