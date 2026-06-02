#!/usr/bin/env python3
"""Run TileLang and FlashAttention baselines with choreo-matched configs."""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime, timezone
from pathlib import Path

from common import (
    BENCHMARK_CONFIGS,
    BenchConfig,
    attention_flops,
    cuda_event_timing,
    read_timer_options,
)

BASELINES_DIR = Path(__file__).resolve().parent


def _run_backend(name: str, verify: bool) -> int:
    if name == "tilelang":
        from bench_tilelang import run_tilelang_baseline

        return run_tilelang_baseline(verify=verify)
    if name == "flash_attn":
        from bench_flash_attn import run_flash_attn_baseline

        return run_flash_attn_baseline(verify=verify)
    raise ValueError(f"unknown backend: {name}")


def _collect_rows(backend: str) -> list[dict]:
    """Run one backend and return TSV rows (re-invokes per-config timing)."""
    if backend == "tilelang":
        from bench_tilelang import (
            TILELANG_BLOCK_M,
            TILELANG_BLOCK_N,
            TILELANG_NUM_STAGES,
            TILELANG_THREADS,
            _load_flashattn_fn,
        )

        flashattn = _load_flashattn_fn()
        import torch

        device = torch.device("cuda")
        warmup, repeat = read_timer_options()
        compiled: dict = {}
        rows = []
        for cfg in BENCHMARK_CONFIGS:
            key = (
                cfg.batch,
                cfg.heads,
                cfg.q_seq,
                cfg.kv_seq,
                cfg.dim,
                cfg.is_causal,
            )
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
            q, k, v = _make_tensors(cfg, device)

            def launch(k=kernel, q=q, k_t=k, v_t=v):
                r = k(q, k_t, v_t)
                if isinstance(r, torch.Tensor):
                    pass
                else:
                    _ = r[0]

            avg_ms = cuda_event_timing(launch, warmup, repeat)
            rows.append(_row(cfg, backend, avg_ms))
        return rows

    if backend == "flash_attn":
        from flash_attn import flash_attn_func
        import torch

        device = torch.device("cuda")
        warmup, repeat = read_timer_options()
        scale = (64 ** -0.5)
        rows = []
        for cfg in BENCHMARK_CONFIGS:
            q, k, v = _make_tensors(cfg, device)
            q_bshd = q.transpose(1, 2).contiguous()
            k_bshd = k.transpose(1, 2).contiguous()
            v_bshd = v.transpose(1, 2).contiguous()

            def launch(
                qb=q_bshd,
                kb=k_bshd,
                vb=v_bshd,
                scale=scale,
                causal=cfg.is_causal,
            ):
                flash_attn_func(qb, kb, vb, softmax_scale=scale, causal=causal)

            avg_ms = cuda_event_timing(launch, warmup, repeat)
            rows.append(_row(cfg, backend, avg_ms))
        return rows

    raise ValueError(backend)


def _make_tensors(cfg: BenchConfig, device):
    from common import make_random_bhsd

    return make_random_bhsd(cfg, device)


def _row(cfg: BenchConfig, backend: str, avg_ms: float) -> dict:
    flops = attention_flops(cfg)
    tflops = flops / (avg_ms / 1000.0) / 1e12
    return {
        "backend": backend,
        "label": cfg.label,
        "batch": cfg.batch,
        "heads": cfg.heads,
        "q_seq": cfg.q_seq,
        "kv_seq": cfg.kv_seq,
        "dim": cfg.dim,
        "causal": int(cfg.is_causal),
        "avg_ms": f"{avg_ms:.6f}",
        "tflops": f"{tflops:.4f}",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Flash attention baselines (TileLang + FlashAttention)"
    )
    parser.add_argument(
        "--backend",
        choices=["all", "tilelang", "flash_attn"],
        default="all",
    )
    parser.add_argument("--verify", action="store_true")
    parser.add_argument(
        "--tsv",
        type=Path,
        default=None,
        help="Append results to TSV (default: baselines/results.tsv)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only write TSV, minimal stdout",
    )
    args = parser.parse_args()

    backends = (
        ["tilelang", "flash_attn"]
        if args.backend == "all"
        else [args.backend]
    )

    tsv_path = args.tsv or (BASELINES_DIR / "results.tsv")
    all_rows: list[dict] = []
    rc = 0

    for backend in backends:
        if not args.quiet:
            print(f"\n========== {backend} ==========\n")
            rc |= _run_backend(backend, verify=args.verify)
        else:
            all_rows.extend(_collect_rows(backend))

    if args.quiet:
        _write_tsv(tsv_path, all_rows)
        for row in all_rows:
            print(
                f"{row['backend']}\t{row['label']}\t"
                f"{row['tflops']} TFLOPS\t{row['avg_ms']} ms"
            )
        return

    if args.tsv:
        rows = []
        for backend in backends:
            rows.extend(_collect_rows(backend))
        _write_tsv(tsv_path, rows)

    raise SystemExit(rc)


def _write_tsv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    fieldnames = list(rows[0].keys()) + ["timestamp"]
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        if write_header:
            writer.writeheader()
        for row in rows:
            out = dict(row)
            out["timestamp"] = ts
            writer.writerow(out)


if __name__ == "__main__":
    main()
