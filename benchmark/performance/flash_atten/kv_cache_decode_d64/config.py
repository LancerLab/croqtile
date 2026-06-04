"""Variant 4: KV-cache decode, D=64 (q_seq < kv_seq).

Choreo/FA3: BSHD for FA3; Choreo kernel BHSD with past_len = kv_seq - q_seq.
"""

from __future__ import annotations

from _fa3_common import BenchCase

VARIANT_ID = "kv_cache_decode_d64"
VARIANT_TITLE = "KV-cache decode (D=64)"

DTYPE_NAME = "fp16"

CANONICAL_LABEL = "B=1 H=32 q=1 kv=8192 decode"

BENCHMARK_CONFIGS: list[BenchCase] = [
    BenchCase(1, 1, 8192, 32, 32, 64, 64, True, CANONICAL_LABEL),
    BenchCase(1, 128, 8192, 32, 32, 64, 64, True, "B=1 H=32 q=128 kv=8192 chunk"),
]
