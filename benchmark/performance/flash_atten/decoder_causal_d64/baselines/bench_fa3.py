#!/usr/bin/env python3
"""FA3 baseline for variant decoder_causal_d64."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

VARIANT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VARIANT_DIR.parent))
sys.path.insert(0, str(VARIANT_DIR))

from _fa3_common import run_fa3_baseline  # noqa: E402
from config import BENCHMARK_CONFIGS, VARIANT_ID, VARIANT_TITLE  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(
        run_fa3_baseline(
            VARIANT_ID,
            BENCHMARK_CONFIGS,
            torch.float16,
            extra_header="Math: causal=True  scale=1/sqrt(64)",
        )
    )
