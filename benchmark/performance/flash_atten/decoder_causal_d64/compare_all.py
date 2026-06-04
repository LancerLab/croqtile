#!/usr/bin/env python3
"""Compare Choreo vs FA3 for decoder_causal_d64."""

from __future__ import annotations

import sys
from pathlib import Path

VARIANT_DIR = Path(__file__).resolve().parent
FLASH_ATTEN_DIR = VARIANT_DIR.parent
sys.path.insert(0, str(FLASH_ATTEN_DIR))
sys.path.insert(0, str(VARIANT_DIR))

from _compare_common import run_compare  # noqa: E402
from config import BENCHMARK_CONFIGS, VARIANT_ID  # noqa: E402

if __name__ == "__main__":
    labels = [c.label for c in BENCHMARK_CONFIGS]
    raise SystemExit(
        run_compare(
            VARIANT_DIR,
            VARIANT_ID,
            "D=64 causal decoder; FA3 and Choreo BSHD",
            labels,
            ("choreo", "fa3"),
            has_choreo=True,
            default_kernel="v2_manual_s2_1p2c_tma.co",
        )
    )
