"""Variant 2: FA3 headline prefill, D=128, non-causal (BSHD). Choreo TBD."""

from __future__ import annotations

from _fa3_common import BenchCase

VARIANT_ID = "prefill_d128"
VARIANT_TITLE = "Prefill MHA (D=128, non-causal)"

DTYPE_NAME = "bf16"

CANONICAL_LABEL = "B=2 H=16 SEQ=8192 prefill"

BENCHMARK_CONFIGS: list[BenchCase] = [
    BenchCase(2, 8192, 8192, 16, 16, 128, 128, False, CANONICAL_LABEL),
    BenchCase(1, 8192, 8192, 16, 16, 128, 128, False, "B=1 H=16 SEQ=8192 prefill"),
]
