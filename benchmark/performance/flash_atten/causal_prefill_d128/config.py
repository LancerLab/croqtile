"""Variant 5: causal prefill, D=128, BSHD (FA3). Choreo TBD."""

from __future__ import annotations

from _fa3_common import BenchCase

VARIANT_ID = "causal_prefill_d128"
VARIANT_TITLE = "Causal prefill (D=128)"

DTYPE_NAME = "bf16"

CANONICAL_LABEL = "B=4 H=32 SEQ=8192 causal prefill"

BENCHMARK_CONFIGS: list[BenchCase] = [
    BenchCase(4, 4096, 4096, 32, 32, 128, 128, True, "B=4 H=32 SEQ=4096 causal prefill"),
    BenchCase(4, 8192, 8192, 32, 32, 128, 128, True, CANONICAL_LABEL),
    BenchCase(4, 16384, 16384, 32, 32, 128, 128, True, "B=4 H=32 SEQ=16384 causal prefill"),
]

# Which backends to include in compare_all.  Set to False to skip.
BACKENDS: dict[str, bool] = {
    "choreo": True,
    "fa3": True,
    "tilelang": True,
    "triton": True,
    "triton_ws": True,
}

# Timing parameters (overridden by CHOREO_TIMING_WARMUP / _REPEAT env vars)
WARMUP = 50
REPEAT = 200
