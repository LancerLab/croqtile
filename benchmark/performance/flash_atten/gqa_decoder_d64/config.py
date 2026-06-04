"""Variant 3: GQA decoder causal, D=64 (BSHD for FA3). Choreo GQA kernel TBD."""

from __future__ import annotations

from _fa3_common import BenchCase

VARIANT_ID = "gqa_decoder_d64"
VARIANT_TITLE = "GQA decoder causal (D=64)"

DTYPE_NAME = "fp16"

CANONICAL_LABEL = "B=1 Hq=32 Hkv=4 SEQ=8192 GQA"

BENCHMARK_CONFIGS: list[BenchCase] = [
    BenchCase(1, 8192, 8192, 32, 4, 64, 64, True, CANONICAL_LABEL),
    BenchCase(2, 4096, 4096, 32, 4, 64, 64, True, "B=2 Hq=32 Hkv=4 SEQ=4096 GQA"),
]
