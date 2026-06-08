#!/usr/bin/env python3
"""FA3 baseline compare for gqa_decoder_d64 (Choreo N/A)."""

from __future__ import annotations

import sys
from pathlib import Path

VARIANT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(VARIANT_DIR.parent / "scripts"))
sys.path.insert(0, str(VARIANT_DIR))

from _compare_common import run_compare  # noqa: E402
from config import BENCHMARK_CONFIGS, VARIANT_ID  # noqa: E402

if __name__ == "__main__":
    labels = [c.label for c in BENCHMARK_CONFIGS]
    raise SystemExit(
        run_compare(
            VARIANT_DIR,
            VARIANT_ID,
            "D=64 GQA causal decoder, BSHD (FA3 only)",
            labels,
            ("fa3",),
            has_choreo=False,
        )
    )
