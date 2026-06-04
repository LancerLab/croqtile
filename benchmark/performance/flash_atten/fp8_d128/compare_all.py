#!/usr/bin/env python3
"""FA3 baseline compare for fp8_d128."""

from __future__ import annotations

import sys
from pathlib import Path

VARIANT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(VARIANT_DIR.parent))
sys.path.insert(0, str(VARIANT_DIR))

from _compare_common import run_compare  # noqa: E402
from config import BENCHMARK_CONFIGS, VARIANT_ID  # noqa: E402

if __name__ == "__main__":
    labels = [c.label for c in BENCHMARK_CONFIGS]
    raise SystemExit(
        run_compare(
            VARIANT_DIR,
            VARIANT_ID,
            "D=128 FP8 BSHD (FA3 only; rebuild with FP8)",
            labels,
            ("fa3",),
            has_choreo=False,
        )
    )
