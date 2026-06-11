"""Variant 5: causal prefill, D=128, BSHD (FA3). Choreo TBD."""

from __future__ import annotations

from _fa3_common import BenchCase

VARIANT_ID = "causal_prefill_d128"
VARIANT_TITLE = "Causal prefill (D=128)"

DTYPE_NAME = "bf16"

CANONICAL_LABEL = "B=2 H=16 SEQ=8192 causal prefill"

BENCHMARK_CONFIGS: list[BenchCase] = [
    # BenchCase(2, 512, 512, 16, 16, 128, 128, True, "B=2 H=16 SEQ=512 causal prefill"),
    # BenchCase(2, 1024, 1024, 16, 16, 128, 128, True, "B=2 H=16 SEQ=1024 causal prefill"),
    # BenchCase(2, 2048, 2048, 16, 16, 128, 128, True, "B=2 H=16 SEQ=2048 causal prefill"),
    BenchCase(1, 4096, 4096, 16, 16, 128, 128, True, "B=1 H=16 SEQ=4096 causal prefill"),
    BenchCase(2, 8192, 8192, 16, 16, 128, 128, True, CANONICAL_LABEL),
    # BenchCase(1, 16384, 16384, 16, 16, 128, 128, True, "B=1 H=16 SEQ=16384 causal prefill"),
]
