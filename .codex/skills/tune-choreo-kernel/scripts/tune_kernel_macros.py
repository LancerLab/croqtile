#!/usr/bin/env python3
import argparse
import csv
import itertools
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


TFLOPS_RE = re.compile(r"TFLOPS:\s*([0-9]+(?:\.[0-9]+)?)")


def parse_mnk(value: str):
    parts = [p.strip() for p in value.split(",")]
    if len(parts) != 3 or any(not p.isdigit() for p in parts):
        raise argparse.ArgumentTypeError("mnk must be M,N,K (integers)")
    return tuple(int(p) for p in parts)


def parse_macro_kv(spec: str):
    if "=" not in spec:
        raise argparse.ArgumentTypeError(f"invalid macro spec '{spec}', expected NAME=v1,v2")
    key, raw_values = spec.split("=", 1)
    key = key.strip()
    if not key:
        raise argparse.ArgumentTypeError(f"invalid macro name in '{spec}'")
    values = [v.strip() for v in raw_values.split(",") if v.strip()]
    if not values:
        raise argparse.ArgumentTypeError(f"no values provided in '{spec}'")
    return key, values


def update_macro(text: str, macro: str, value: str):
    pattern = re.compile(rf"^(\s*#define\s+{re.escape(macro)}\s+)(\S+)(.*)$", re.MULTILINE)
    if not pattern.search(text):
        return text, False
    text = pattern.sub(rf"\g<1>{value}\g<3>", text, count=1)
    return text, True


def apply_macro_overrides(base_text: str, overrides: dict):
    text = base_text
    missing = []
    for macro, value in overrides.items():
        text, ok = update_macro(text, macro, str(value))
        if not ok:
            missing.append(macro)
    return text, missing


def render_case_name(index: int, params: dict):
    kv = "__".join(f"{k}_{v}" for k, v in sorted(params.items()))
    return f"case_{index:04d}__{kv}" if kv else f"case_{index:04d}"


def find_tflops(output: str):
    values = [float(m.group(1)) for m in TFLOPS_RE.finditer(output)]
    if not values:
        return None
    return values[-1]


def ensure_repo_root():
    root = Path.cwd()
    if not (root / "build" / "choreo").exists():
        raise RuntimeError("must run from repository root containing build/choreo")
    return root


def main():
    parser = argparse.ArgumentParser(
        description="Sweep Choreo kernel macro definitions, verify each candidate, and keep best TFLOPS."
    )
    parser.add_argument("--input-co", required=True, help="Base .co kernel file")
    parser.add_argument("--mnk", required=True, type=parse_mnk, help="Problem size as M,N,K")
    parser.add_argument(
        "--sweep",
        action="append",
        default=[],
        help="Sweep macro values: NAME=v1,v2,v3 (repeatable)",
    )
    parser.add_argument(
        "--set",
        dest="fixed",
        action="append",
        default=[],
        help="Fixed macro override: NAME=value (repeatable)",
    )
    parser.add_argument(
        "--verify-script",
        default=".codex/skills/verify-choreo-kernel/scripts/verify_choreo_kernel.sh",
        help="Path to verify script",
    )
    parser.add_argument("--workdir", default="build/skill-logs/tune-choreo-kernel", help="Workspace for generated files")
    parser.add_argument("--output-dir", default=None, help="Where to place best kernel file (default: input folder)")
    parser.add_argument(
        "--compile-arg",
        action="append",
        default=[],
        help="Extra compile arg forwarded after '--' to verify script (repeatable)",
    )
    parser.add_argument(
        "--run-arg",
        action="append",
        default=[],
        help="Runtime arg forwarded to verify script as --run-arg <arg> (repeatable)",
    )
    parser.add_argument("--timeout-sec", type=int, default=120, help="Verify runtime timeout")
    parser.add_argument("--bench-timeout-sec", type=int, default=240, help="Benchmark timeout")
    parser.add_argument("--topk", type=int, default=5, help="Print top-K results")
    parser.add_argument("--max-cases", type=int, default=0, help="Optional cap of candidate count")
    parser.add_argument("--dry-run", action="store_true", help="Generate cases and commands without running verify")
    args = parser.parse_args()

    repo_root = ensure_repo_root()
    input_co = Path(args.input_co)
    if not input_co.is_absolute():
        input_co = (repo_root / input_co).resolve()
    if not input_co.exists():
        raise FileNotFoundError(f"input .co not found: {input_co}")

    verify_script = Path(args.verify_script)
    if not verify_script.is_absolute():
        verify_script = (repo_root / verify_script).resolve()
    if not verify_script.exists():
        raise FileNotFoundError(f"verify script not found: {verify_script}")

    out_root = Path(args.workdir)
    if not out_root.is_absolute():
        out_root = (repo_root / out_root).resolve()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = out_root / f"run_{stamp}"
    cases_dir = run_root / "cases"
    bins_dir = run_root / "bins"
    logs_dir = run_root / "logs"
    cases_dir.mkdir(parents=True, exist_ok=True)
    bins_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    base_text = input_co.read_text()

    fixed_overrides = {}
    for spec in args.fixed:
        macro, values = parse_macro_kv(spec)
        if len(values) != 1:
            raise ValueError(f"--set expects one value, got '{spec}'")
        fixed_overrides[macro] = values[0]

    sweep_items = [parse_macro_kv(spec) for spec in args.sweep]
    sweep_names = [k for k, _ in sweep_items]
    sweep_values = [vals for _, vals in sweep_items]
    all_products = list(itertools.product(*sweep_values)) if sweep_values else [tuple()]
    if args.max_cases and args.max_cases > 0:
        all_products = all_products[: args.max_cases]

    m, n, k = args.mnk
    for macro, val in (
        ("MATMUL_DEFAULT_M", str(m)),
        ("MATMUL_DEFAULT_N", str(n)),
        ("MATMUL_DEFAULT_K", str(k)),
    ):
        if re.search(rf"^\s*#define\s+{re.escape(macro)}\s+", base_text, re.MULTILINE):
            fixed_overrides[macro] = val

    print(f"Input kernel: {input_co}")
    print(f"Problem size: M={m}, N={n}, K={k}")
    print(f"Candidates: {len(all_products)}")
    print(f"Run root: {run_root}")

    results = []
    for idx, combo in enumerate(all_products, start=1):
        params = dict(zip(sweep_names, combo))
        params.update(fixed_overrides)
        case_name = render_case_name(idx, params)
        case_path = cases_dir / f"{case_name}.co"
        bin_path = bins_dir / case_name
        log_path = logs_dir / f"{case_name}.log"

        case_text, missing = apply_macro_overrides(base_text, params)
        if missing:
            msg = f"missing macros in source: {', '.join(missing)}"
            results.append({"case": case_name, "ok": False, "tflops": None, "msg": msg, "params": params, "file": str(case_path)})
            continue
        case_path.write_text(case_text)

        cmd = [
            str(verify_script),
            str(case_path),
            str(bin_path),
            "--timeout-sec",
            str(args.timeout_sec),
            "--bench-timeout-sec",
            str(args.bench_timeout_sec),
        ]
        for run_arg in args.run_arg:
            cmd.extend(["--run-arg", run_arg])
        if args.compile_arg:
            cmd.append("--")
            cmd.extend(args.compile_arg)

        env = os.environ.copy()
        env["CHOREO_VERIFY_USE_SMALL"] = "0"
        env["CHOREO_VERIFY_BENCH_MNK"] = f"{m},{n},{k}"

        if args.dry_run:
            print(f"[dry-run] {case_name}: {' '.join(cmd)}")
            results.append({"case": case_name, "ok": True, "tflops": None, "msg": "dry-run", "params": params, "file": str(case_path)})
            continue

        proc = subprocess.run(cmd, text=True, capture_output=True, env=env)
        output = proc.stdout + ("\n" + proc.stderr if proc.stderr else "")
        log_path.write_text(output)

        tflops = find_tflops(output)
        ok = proc.returncode == 0 and tflops is not None
        if ok:
            msg = f"TFLOPS={tflops:.4f}"
            print(f"[{idx}/{len(all_products)}] PASS {case_name} {msg}")
        else:
            msg = f"failed (exit={proc.returncode})"
            print(f"[{idx}/{len(all_products)}] FAIL {case_name} {msg}")
        results.append({"case": case_name, "ok": ok, "tflops": tflops, "msg": msg, "params": params, "file": str(case_path)})

    csv_path = run_root / "results.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        header = ["case", "ok", "tflops", "msg", "file"] + sorted(set(list(fixed_overrides.keys()) + sweep_names))
        writer.writerow(header)
        for item in results:
            row = [item["case"], item["ok"], item["tflops"], item["msg"], item["file"]]
            for macro in header[5:]:
                row.append(item["params"].get(macro, ""))
            writer.writerow(row)
    print(f"Result table: {csv_path}")

    ranked = [r for r in results if r["ok"] and r["tflops"] is not None]
    ranked.sort(key=lambda x: x["tflops"], reverse=True)
    if ranked:
        print("Top candidates:")
        for item in ranked[: max(1, args.topk)]:
            print(f"  {item['case']}: {item['tflops']:.4f} TFLOPS")

        best = ranked[0]
        output_dir = Path(args.output_dir).resolve() if args.output_dir else input_co.parent
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = input_co.stem
        best_name = f"{stem}_m{m}_n{n}_k{k}.co"
        best_out = output_dir / best_name
        shutil.copy2(best["file"], best_out)
        print(f"Best kernel: {best_out}")
        return 0

    if args.dry_run:
        print("Dry-run finished. No TFLOPS ranking generated.")
        return 0

    print("No successful candidates with parsed TFLOPS.", file=sys.stderr)
    return 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
