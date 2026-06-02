#!/usr/bin/env python3
"""TileLang flash attention baseline (example_mha_fwd_bhsd)."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch

from common import (
    BENCHMARK_CONFIGS,
    cuda_event_timing,
    make_random_bhsd,
    read_timer_options,
    report_timing,
)

# TileLang example: /home/fem/dev/tilelang/examples/flash_attention
TILELANG_FLASH_DIR = Path("/home/fem/dev/tilelang/examples/flash_attention")

# Hopper-tuned static config (from TileLang regression_example).
TILELANG_BLOCK_M = 128
TILELANG_BLOCK_N = 128
TILELANG_NUM_STAGES = 2
TILELANG_THREADS = 256


def _load_flashattn_fn():
    example = TILELANG_FLASH_DIR / "example_mha_fwd_bhsd.py"
    if not example.is_file():
        raise FileNotFoundError(f"TileLang example not found: {example}")
    spec = importlib.util.spec_from_file_location(
        "tilelang_example_mha_fwd_bhsd", example
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load {example}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod.flashattn


def run_tilelang_baseline(verify: bool = False) -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    flashattn = _load_flashattn_fn()
    device = torch.device("cuda")
    warmup, repeat = read_timer_options()

    print("TileLang flash attention BHSD causal decoder (DIM=64)")
    print(
        f"Kernel: block_M={TILELANG_BLOCK_M} block_N={TILELANG_BLOCK_N} "
        f"stages={TILELANG_NUM_STAGES} threads={TILELANG_THREADS}"
    )
    print(f"Warmup={warmup} Repeat={repeat}")
    print("Math: exp2 softmax, past_len=seq_kv-seq_q (same as choreo)")
    print("--------------------------------------------")

    compiled: dict[tuple, object] = {}
    all_ok = True

    for i, cfg in enumerate(BENCHMARK_CONFIGS):
        key = (cfg.batch, cfg.heads, cfg.q_seq, cfg.kv_seq, cfg.dim, cfg.is_causal)
        if key not in compiled:
            compiled[key] = flashattn(
                cfg.batch,
                cfg.heads,
                cfg.q_seq,
                cfg.kv_seq,
                cfg.dim,
                cfg.is_causal,
                block_M=TILELANG_BLOCK_M,
                block_N=TILELANG_BLOCK_N,
                num_stages=TILELANG_NUM_STAGES,
                threads=TILELANG_THREADS,
            )
        kernel = compiled[key]

        q, k, v = make_random_bhsd(cfg, device)

        def launch() -> None:
            kernel(q, k, v)

        avg_ms = cuda_event_timing(launch, warmup, repeat)
        report_timing(cfg, avg_ms, "tilelang")

        if verify:
            spec = importlib.util.spec_from_file_location(
                "tilelang_example_mha_fwd_bhsd_v",
                TILELANG_FLASH_DIR / "example_mha_fwd_bhsd.py",
            )
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            ref = mod.ref_program(q, k, v, cfg.is_causal)
            got = kernel(q, k, v)
            if not isinstance(got, torch.Tensor):
                got = got[0]
            max_err = (got.float() - ref.float()).abs().max().item()
            ok = max_err < 0.05
            print(
                f"[VERIFY tilelang {cfg.label}] max_abs_err={max_err} "
                f"{'PASS' if ok else 'FAIL'}"
            )
            all_ok = all_ok and ok

        if i + 1 < len(BENCHMARK_CONFIGS):
            print("--------------------------------------------")

    if verify:
        print("Test Passed" if all_ok else "Test FAILED")
        return 0 if all_ok else 1
    print("Done")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    raise SystemExit(run_tilelang_baseline(verify=args.verify))


if __name__ == "__main__":
    main()
