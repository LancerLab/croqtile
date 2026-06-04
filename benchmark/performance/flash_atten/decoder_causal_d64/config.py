"""Variant 1: decoder causal self-attention, D=64.

Workload selection (see README.md):
  - q_seq == kv_seq: full-sequence causal pass (decoder chunk / prefill over
    context), not single-token KV-cache decode (variant kv_cache_decode_d64).
  - H=32, D=64: 2048-dim MHA. FA3 and Choreo both BSHD (see README).
  - Seq >= 2048: at least two BLOCK_M=128 tiles along Q; short seq misreports TFLOPS.
  - B=1 S=8192: primary tuning and FA3/Choreo headline compare.
  - B=2 S=4096: batched inference at long context.
  - B=1 S=2048: mid-context regression point.
"""

from __future__ import annotations

from _fa3_common import BenchCase

VARIANT_ID = "decoder_causal_d64"
VARIANT_TITLE = "Decoder causal MHA (D=64)"

DTYPE_NAME = "fp16"

# Primary compare point (tuning target).
CANONICAL_LABEL = "B=1 H=32 SEQ=8192 decoder"

BENCHMARK_CONFIGS: list[BenchCase] = [
    BenchCase(1, 2048, 2048, 32, 32, 64, 64, True, "B=1 H=32 SEQ=2048 decoder"),
    BenchCase(2, 4096, 4096, 32, 32, 64, 64, True, "B=2 H=32 SEQ=4096 decoder"),
    BenchCase(1, 8192, 8192, 32, 32, 64, 64, True, CANONICAL_LABEL),
]
