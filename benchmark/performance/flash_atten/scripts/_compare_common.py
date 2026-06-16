"""Shared Choreo vs FA3 compare driver for flash_atten variants."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Sequence

TFLOPS_RE = re.compile(r"^TFLOPS:\s*([0-9.eE+-]+)")
FA3_LABEL_RE = re.compile(r"^\[fa3 (.+)\]$")
TILELANG_LABEL_RE = re.compile(r"^\[tilelang (.+)\]$")
TRITON_LABEL_RE = re.compile(r"^\[triton (.+)\]$")
TRITON_WS_LABEL_RE = re.compile(r"^\[triton_ws (.+)\]$")

_NOISE_LINE = re.compile(
    r"^(?:\s*\d{4}-\d{2}-\d{2}.*WARNING:|\[bench\]|ptxas info|Remark:|"
    r"INFO \(|WARNING \()"
)

_COOLDOWN_TARGET_C = 72
_COOLDOWN_TIMEOUT_S = 120
_COOLDOWN_POLL_S = 3


def _gpu_temperature(gpu_id: str | None = None) -> int | None:
    """Read current GPU temperature in Celsius via nvidia-smi."""
    cmd = ["nvidia-smi", "--query-gpu=temperature.gpu",
           "--format=csv,noheader,nounits"]
    if gpu_id is not None:
        cmd.extend(["--id=" + str(gpu_id)])
    try:
        out = subprocess.check_output(cmd, text=True, timeout=5).strip()
        return int(out.splitlines()[0])
    except Exception:
        return None


def _gpu_cooldown(gpu_id: str | None = None) -> None:
    """Wait for GPU to cool below threshold for fair benchmarking."""
    temp = _gpu_temperature(gpu_id)
    if temp is None or temp <= _COOLDOWN_TARGET_C:
        return
    print(
        f"[compare] GPU at {temp}C, cooling to {_COOLDOWN_TARGET_C}C ...",
        file=sys.stderr,
        end="",
    )
    deadline = time.monotonic() + _COOLDOWN_TIMEOUT_S
    while time.monotonic() < deadline:
        time.sleep(_COOLDOWN_POLL_S)
        temp = _gpu_temperature(gpu_id)
        if temp is None or temp <= _COOLDOWN_TARGET_C:
            if temp is not None:
                print(f" {temp}C ok", file=sys.stderr)
            else:
                print(" ok", file=sys.stderr)
            return
        print(f" {temp}C", file=sys.stderr, end="", flush=True)
    temp = _gpu_temperature(gpu_id)
    print(
        f" timeout ({temp}C), proceeding anyway",
        file=sys.stderr,
    )


def _filter_text(text: str) -> str:
    return "\n".join(
        line for line in text.splitlines() if not _NOISE_LINE.search(line)
    )


def collect_choreo(
    choreo_dir: Path,
    labels: list[str],
    kernel: str,
    gpu: str | None,
) -> list[dict]:
    bench_sh = choreo_dir / "bench.sh"
    if not bench_sh.is_file():
        raise FileNotFoundError(f"choreo bench script not found: {bench_sh}")

    cmd = ["bash", str(bench_sh), "--kernel", kernel, "--no-verify"]
    if gpu is not None:
        cmd.extend(["--gpu", gpu])

    env = os.environ.copy()

    print(f"[compare] Running Choreo: {kernel}", file=sys.stderr)
    proc = subprocess.run(
        cmd,
        cwd=str(choreo_dir),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        print(proc.stdout, file=sys.stderr)
        print(proc.stderr, file=sys.stderr)
        raise RuntimeError(f"choreo bench failed (exit {proc.returncode})")

    return _parse_label_tflops(
        _filter_text(proc.stdout + proc.stderr),
        "choreo",
        Path(kernel).name,
        labels,
    )


def collect_fa3(baselines_dir: Path, labels: list[str]) -> list[dict]:
    script = baselines_dir / "bench_fa3.py"
    env = os.environ.copy()

    print("[compare] Running FA3", file=sys.stderr)
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES")
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpu

    proc = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(baselines_dir),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        print(proc.stdout, file=sys.stderr)
        print(proc.stderr, file=sys.stderr)
        raise RuntimeError(f"FA3 bench failed (exit {proc.returncode})")

    rows = _parse_fa3_output(_filter_text(proc.stdout + proc.stderr), labels)
    if len(rows) != len(labels):
        raise RuntimeError(f"expected {len(labels)} FA3 rows, got {len(rows)}")
    return rows


def collect_tilelang(baselines_dir: Path, labels: list[str]) -> list[dict]:
    script = baselines_dir / "bench_tilelang.py"
    if not script.is_file():
        raise FileNotFoundError(f"tilelang bench not found: {script}")

    env = os.environ.copy()

    print("[compare] Running TileLang", file=sys.stderr)
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES")
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpu

    proc = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(baselines_dir),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        print(proc.stdout, file=sys.stderr)
        print(proc.stderr, file=sys.stderr)
        raise RuntimeError(f"TileLang bench failed (exit {proc.returncode})")

    rows = _parse_tilelang_output(
        _filter_text(proc.stdout + proc.stderr), labels
    )
    if len(rows) != len(labels):
        raise RuntimeError(
            f"expected {len(labels)} TileLang rows, got {len(rows)}"
        )
    return rows


def _parse_tilelang_output(text: str, labels: list[str]) -> list[dict]:
    rows: list[dict] = []
    pending: str | None = None
    for line in text.splitlines():
        stripped = line.strip()
        m = TILELANG_LABEL_RE.match(stripped)
        if m:
            inner = m.group(1)
            for lab in labels:
                if inner == lab:
                    pending = lab
                    break
            continue
        if pending is not None:
            match = TFLOPS_RE.match(stripped)
            if match:
                rows.append(
                    {
                        "backend": "tilelang",
                        "kernel": "tilelang_flashattn",
                        "label": pending,
                        "tflops": float(match.group(1)),
                    }
                )
                pending = None
    return rows


def collect_triton(baselines_dir: Path, labels: list[str]) -> list[dict]:
    script = baselines_dir / "bench_triton.py"
    if not script.is_file():
        raise FileNotFoundError(f"triton bench not found: {script}")

    env = os.environ.copy()

    print("[compare] Running Triton", file=sys.stderr)
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES")
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpu

    proc = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(baselines_dir),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        print(proc.stdout, file=sys.stderr)
        print(proc.stderr, file=sys.stderr)
        raise RuntimeError(f"Triton bench failed (exit {proc.returncode})")

    rows = _parse_triton_output(
        _filter_text(proc.stdout + proc.stderr), labels
    )
    if len(rows) != len(labels):
        raise RuntimeError(
            f"expected {len(labels)} Triton rows, got {len(rows)}"
        )
    return rows


def _parse_triton_output(text: str, labels: list[str]) -> list[dict]:
    rows: list[dict] = []
    pending: str | None = None
    for line in text.splitlines():
        stripped = line.strip()
        m = TRITON_LABEL_RE.match(stripped)
        if m:
            inner = m.group(1)
            for lab in labels:
                if inner == lab:
                    pending = lab
                    break
            continue
        if pending is not None:
            match = TFLOPS_RE.match(stripped)
            if match:
                rows.append(
                    {
                        "backend": "triton",
                        "kernel": "triton_fused_attn",
                        "label": pending,
                        "tflops": float(match.group(1)),
                    }
                )
                pending = None
    return rows


def _triton_aref_python() -> str | None:
    """Return the Python interpreter for the triton-aref venv, or None."""
    venv = os.environ.get(
        "TRITON_AREF_ENV", os.path.expanduser("~/.env/triton-aref")
    )
    py = os.path.join(venv, "bin", "python3")
    if os.path.isfile(py):
        return py
    return None


def collect_triton_ws(baselines_dir: Path, labels: list[str]) -> list[dict]:
    script = baselines_dir / "bench_triton_ws.py"
    if not script.is_file():
        raise FileNotFoundError(f"triton_ws bench not found: {script}")

    py = _triton_aref_python()
    if py is None:
        raise RuntimeError(
            "Triton+WS requires the triton-aref venv "
            "(~/.env/triton-aref with aref_auto_ws branch). "
            "Set TRITON_AREF_ENV to override."
        )

    env = os.environ.copy()

    print("[compare] Running Triton+WS (aref venv)", file=sys.stderr)
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES")
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpu

    proc = subprocess.run(
        [py, str(script)],
        cwd=str(baselines_dir),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        print(proc.stdout, file=sys.stderr)
        print(proc.stderr, file=sys.stderr)
        raise RuntimeError(f"Triton+WS bench failed (exit {proc.returncode})")

    rows = _parse_triton_ws_output(
        _filter_text(proc.stdout + proc.stderr), labels
    )
    if len(rows) != len(labels):
        raise RuntimeError(
            f"expected {len(labels)} Triton+WS rows, got {len(rows)}"
        )
    return rows


def _parse_triton_ws_output(text: str, labels: list[str]) -> list[dict]:
    rows: list[dict] = []
    pending: str | None = None
    for line in text.splitlines():
        stripped = line.strip()
        m = TRITON_WS_LABEL_RE.match(stripped)
        if m:
            inner = m.group(1)
            for lab in labels:
                if inner == lab:
                    pending = lab
                    break
            continue
        if pending is not None:
            match = TFLOPS_RE.match(stripped)
            if match:
                rows.append(
                    {
                        "backend": "triton_ws",
                        "kernel": "triton_fused_attn_ws",
                        "label": pending,
                        "tflops": float(match.group(1)),
                    }
                )
                pending = None
    return rows


def _parse_fa3_output(text: str, labels: list[str]) -> list[dict]:
    rows: list[dict] = []
    pending: str | None = None
    for line in text.splitlines():
        stripped = line.strip()
        m = FA3_LABEL_RE.match(stripped)
        if m:
            inner = m.group(1)
            for lab in labels:
                if inner == lab:
                    pending = lab
                    break
            continue
        if pending is not None:
            match = TFLOPS_RE.match(stripped)
            if match:
                rows.append(
                    {
                        "backend": "fa3",
                        "kernel": "flash_attn_3",
                        "label": pending,
                        "tflops": float(match.group(1)),
                    }
                )
                pending = None
    return rows


def _parse_label_tflops(
    text: str, backend: str, kernel: str, labels: list[str]
) -> list[dict]:
    rows: list[dict] = []
    pending: str | None = None
    for line in text.splitlines():
        stripped = line.strip()
        if stripped in labels:
            pending = stripped
            continue
        if pending is not None:
            match = TFLOPS_RE.match(stripped)
            if match:
                rows.append(
                    {
                        "backend": backend,
                        "kernel": kernel,
                        "label": pending,
                        "tflops": float(match.group(1)),
                    }
                )
                pending = None
    if len(rows) != len(labels):
        raise RuntimeError(
            f"expected {len(labels)} {backend} TFLOPS lines, got {len(rows)}"
        )
    return rows


def _short_label(label: str) -> str:
    m = re.search(r"SEQ=(\d+)", label)
    if m:
        return f"SEQ={m.group(1)}"
    m = re.search(r"q=(\d+)", label)
    if m:
        return f"q={m.group(1)}"
    return label[:28]


def print_table(
    variant_id: str,
    subtitle: str,
    backends: Sequence[str],
    labels: list[str],
    by_backend: dict[str, list[dict]],
    choreo_kernel: str | None,
    warmup: int,
    repeat: int,
) -> None:
    col_w = (28,) + (12,) * len(backends)
    headers = ("Config",) + tuple(b.upper() for b in backends)

    def fmt_row(cells: tuple[str, ...]) -> str:
        parts = [f"| {cells[0]:<{col_w[0]}} |"]
        for i, c in enumerate(cells[1:], 1):
            parts.append(f" {c:>{col_w[i]}} |")
        return "".join(parts)

    sep = "+-" + "-+-".join("-" * w for w in col_w) + "-+"

    lookup: dict[str, dict[str, float]] = {b: {} for b in backends}
    for backend, rows in by_backend.items():
        for row in rows:
            lookup[backend][row["label"]] = row["tflops"]

    print()
    print(f"{variant_id}: {subtitle} (warmup={warmup}, repeat={repeat})")
    if choreo_kernel:
        print(f"Choreo kernel: {choreo_kernel}")
    print(sep)
    print(fmt_row(headers))
    print(sep)

    for label in labels:
        cells = [_short_label(label)]
        for backend in backends:
            val = lookup[backend].get(label)
            cells.append(f"{val:.1f}" if val is not None else "n/a")
        print(fmt_row(tuple(cells)))

    print(sep)


def run_compare(
    variant_dir: Path,
    variant_id: str,
    subtitle: str,
    labels: list[str],
    backends: Sequence[str],
    *,
    has_choreo: bool,
    default_kernel: str = "v2_manual_s2_1p1c_tma.co",
    warmup: int | None = None,
    repeat: int | None = None,
) -> int:
    parser = argparse.ArgumentParser(description=f"Compare backends for {variant_id}")
    if has_choreo:
        parser.add_argument(
            "--kernel",
            default=os.environ.get(
                "KERNEL", os.environ.get("CHOREO_KERNEL", default_kernel)
            ),
            help="Choreo .co under choreo/",
        )
    parser.add_argument(
        "--gpu",
        default=os.environ.get("CUDA_VISIBLE_DEVICES", "1"),
        help="CUDA device index",
    )
    if has_choreo:
        parser.add_argument("--skip-choreo", action="store_true")
    parser.add_argument("--skip-fa3", action="store_true")
    parser.add_argument("--skip-tilelang", action="store_true")
    parser.add_argument("--skip-triton", action="store_true")
    parser.add_argument("--skip-triton-ws", action="store_true")
    parser.add_argument(
        "--no-cooldown",
        action="store_true",
        help="Skip GPU cooldown between backends (faster but less fair)",
    )
    args = parser.parse_args()

    _default_warmup = warmup if warmup is not None else 50
    _default_repeat = repeat if repeat is not None else 200
    warmup = int(os.environ.get("CHOREO_TIMING_WARMUP", str(_default_warmup)))
    repeat = int(os.environ.get("CHOREO_TIMING_REPEAT", str(_default_repeat)))
    os.environ["CHOREO_TIMING_WARMUP"] = str(warmup)
    os.environ["CHOREO_TIMING_REPEAT"] = str(repeat)
    do_cooldown = not args.no_cooldown

    choreo_dir = variant_dir / "choreo"
    baselines_dir = variant_dir / "baselines"
    by_backend: dict[str, list[dict]] = {}

    def _maybe_cooldown() -> None:
        if do_cooldown and by_backend:
            _gpu_cooldown(args.gpu)

    if has_choreo and not args.skip_choreo:
        _maybe_cooldown()
        by_backend["choreo"] = collect_choreo(
            choreo_dir, labels, args.kernel, args.gpu
        )
    if not args.skip_fa3:
        _maybe_cooldown()
        by_backend["fa3"] = collect_fa3(baselines_dir, labels)
    if not args.skip_tilelang and "tilelang" in backends:
        tilelang_script = baselines_dir / "bench_tilelang.py"
        if tilelang_script.is_file():
            try:
                _maybe_cooldown()
                by_backend["tilelang"] = collect_tilelang(
                    baselines_dir, labels
                )
            except Exception as exc:
                print(
                    f"[compare] TileLang skipped: {exc}", file=sys.stderr
                )
    if not args.skip_triton and "triton" in backends:
        triton_script = baselines_dir / "bench_triton.py"
        if triton_script.is_file():
            try:
                _maybe_cooldown()
                by_backend["triton"] = collect_triton(
                    baselines_dir, labels
                )
            except Exception as exc:
                print(
                    f"[compare] Triton skipped: {exc}", file=sys.stderr
                )
    if not args.skip_triton_ws and "triton_ws" in backends:
        triton_ws_script = baselines_dir / "bench_triton_ws.py"
        if triton_ws_script.is_file():
            try:
                _maybe_cooldown()
                by_backend["triton_ws"] = collect_triton_ws(
                    baselines_dir, labels
                )
            except Exception as exc:
                print(
                    f"[compare] Triton+WS skipped: {exc}", file=sys.stderr
                )

    if not by_backend:
        print("Nothing to run.", file=sys.stderr)
        return 1

    kernel_name = (
        getattr(args, "kernel", None)
        if has_choreo and not getattr(args, "skip_choreo", True)
        else None
    )
    print_table(
        variant_id,
        subtitle,
        tuple(b for b in backends if b in by_backend),
        labels,
        by_backend,
        kernel_name,
        warmup,
        repeat,
    )
    return 0
