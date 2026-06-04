"""Variant 6: Hopper FP8, D=128, BSHD. Requires FA3 rebuild with FP8 enabled."""

from __future__ import annotations

from _fa3_common import BenchCase

VARIANT_ID = "fp8_d128"
VARIANT_TITLE = "FP8 prefill (D=128)"

DTYPE_NAME = "fp8_e4m3fn"

CANONICAL_LABEL = "B=2 H=16 SEQ=8192 FP8 prefill"

BENCHMARK_CONFIGS: list[BenchCase] = [
    BenchCase(2, 8192, 8192, 16, 16, 128, 128, False, CANONICAL_LABEL),
    BenchCase(2, 8192, 8192, 16, 16, 128, 128, True, "B=2 H=16 SEQ=8192 FP8 causal"),
]
