"""Variant 5: causal prefill, D=128, BSHD (FA3). Choreo TBD."""

from __future__ import annotations

from _fa3_common import BenchCase

VARIANT_ID = "causal_prefill_d128"
VARIANT_TITLE = "Causal prefill (D=128)"

DTYPE_NAME = "bf16"

CANONICAL_LABEL = "B=2 H=16 SEQ=8192 causal prefill"

BENCHMARK_CONFIGS: list[BenchCase] = [
    BenchCase(2, 8192, 8192, 16, 16, 128, 128, True, CANONICAL_LABEL),
    BenchCase(1, 4096, 4096, 16, 16, 128, 128, True, "B=1 H=16 SEQ=4096 causal prefill"),
]
