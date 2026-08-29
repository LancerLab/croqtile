#!/usr/bin/env python3
import argparse
import csv
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Dict, List, Optional, Tuple


TFLOPS_RE = re.compile(r"TFLOPS:\s*([0-9]+(?:\.[0-9]+)?)")
AVG_MS_RE = re.compile(r"Timing avg ms:\s*([0-9]+(?:\.[0-9]+)?)")
SMEM_SETATTR_RE = re.compile(
    r"cudaFuncSetAttribute\([^,]+,\s*cudaFuncAttributeMaxDynamicSharedMemorySize,\s*([0-9]+)\s*\+\s*\(1024\s*-\s*1\)\s*\)"
)
SMEM_LAUNCH_RE = re.compile(
    r"<<<[^>]+,\s*([0-9]+)\s*\+\s*\(1024\s*-\s*1\)\s*>>>"
)
DEFINE_RE_TEMPLATE = r"^(\s*#define\s+{name}\s+)(\S+)(.*)$"


@dataclass
class Candidate:
    cid: str
    base_co: Path
    macro_overrides: Dict[str, str]
    perf_prior: float
    note: str


@dataclass
class BenchResult:
    ok: bool
    tflops: Optional[float]
    avg_ms: Optional[float]
    reason: str
    smem_bytes: Optional[int]


def parse_shape(text: str) -> Tuple[int, int, int]:
    parts = [x.strip() for x in text.split(",")]
    if len(parts) != 3:
        raise ValueError(f"invalid shape '{text}', expected M,N,K")
    vals = tuple(int(x) for x in parts)
    if any(v <= 0 for v in vals):
        raise ValueError(f"invalid shape '{text}', values must be >0")
    return vals


def apply_overrides(content: str, overrides: Dict[str, str]) -> Tuple[str, List[str]]:
    missing = []
    out = content
    for macro, value in overrides.items():
        pattern = re.compile(DEFINE_RE_TEMPLATE.format(name=re.escape(macro)), re.MULTILINE)
        if not pattern.search(out):
            missing.append(macro)
            continue
        out = pattern.sub(rf"\g<1>{value}\g<3>", out, count=1)
    return out, missing


def read_trace_csv(path: Path) -> List[Tuple[int, int, int]]:
    rows: List[Tuple[int, int, int]] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append((int(row["m"]), int(row["n"]), int(row["k"])))
    if not rows:
        raise RuntimeError(f"shape trace is empty: {path}")
    return rows


def percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = pct * (len(sorted_vals) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = pos - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


def find_metric(output: str, regex: re.Pattern) -> Optional[float]:
    vals = [float(m.group(1)) for m in regex.finditer(output)]
    return vals[-1] if vals else None


def ensure_repo_root() -> Path:
    root = Path.cwd()
    if not (root / "build" / "choreo").exists():
        raise RuntimeError("run from repository root containing build/choreo")
    return root


def parse_manifest(path: Path, repo_root: Path) -> Tuple[Dict, List[Candidate]]:
    raw = json.loads(path.read_text())
    cand_list: List[Candidate] = []
    for c in raw.get("candidates", []):
        base_co = Path(c["base_co"])
        if not base_co.is_absolute():
            base_co = (repo_root / base_co).resolve()
        cand_list.append(
            Candidate(
                cid=c["id"],
                base_co=base_co,
                macro_overrides={k: str(v) for k, v in c.get("macro_overrides", {}).items()},
                perf_prior=float(c.get("perf_prior", 1.0)),
                note=c.get("note", ""),
            )
        )
    if not cand_list:
        raise RuntimeError("manifest has no candidates")
    return raw, cand_list


def materialize_candidate(
    candidate: Candidate,
    shape: Tuple[int, int, int],
    out_dir: Path,
) -> Path:
    m, n, k = shape
    content = candidate.base_co.read_text()
    merged = dict(candidate.macro_overrides)
    for macro, value in (
        ("MATMUL_DEFAULT_M", str(m)),
        ("MATMUL_DEFAULT_N", str(n)),
        ("MATMUL_DEFAULT_K", str(k)),
    ):
        if re.search(rf"^\s*#define\s+{re.escape(macro)}\s+", content, re.MULTILINE):
            merged[macro] = value
    updated, missing = apply_overrides(content, merged)
    if missing:
        raise RuntimeError(f"{candidate.cid}: missing macros in {candidate.base_co}: {missing}")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{candidate.cid}_m{m}_n{n}_k{k}.co"
    out_path.write_text(updated)
    return out_path


def probe_smem_bytes(
    candidate: Candidate,
    manifest_cfg: Dict,
    work_dir: Path,
    repo_root: Path,
) -> Optional[int]:
    target = manifest_cfg.get("target", "cute")
    arch = manifest_cfg.get("arch", "sm_90a")
    probe_co = materialize_candidate(candidate, (1024, 1024, 1024), work_dir / "probe_cases")
    out_script = work_dir / "probe_bins" / f"{candidate.cid}.cute.result"
    out_script.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(repo_root / "build" / "choreo"),
        "-gs",
        "-t",
        target,
        f"-arch={arch}",
        str(probe_co),
        "-o",
        str(out_script),
    ]
    proc = subprocess.run(cmd, text=True, capture_output=True)
    if proc.returncode != 0:
        return None
    text = out_script.read_text()
    m = SMEM_SETATTR_RE.search(text)
    if m:
        return int(m.group(1)) + 1023
    m = SMEM_LAUNCH_RE.search(text)
    if m:
        return int(m.group(1)) + 1023
    return None


def run_verify(
    verify_script: Path,
    candidate_co: Path,
    bin_path: Path,
    shape: Tuple[int, int, int],
    timeout_sec: int,
    bench_timeout_sec: int,
    compile_args: List[str],
    run_args: List[str],
    env_overrides: Dict[str, str],
) -> BenchResult:
    m, n, k = shape
    cmd = [
        str(verify_script),
        str(candidate_co),
        str(bin_path),
        "--timeout-sec",
        str(timeout_sec),
        "--bench-timeout-sec",
        str(bench_timeout_sec),
    ]
    for arg in run_args:
        cmd.extend(["--run-arg", arg])
    if compile_args:
        cmd.append("--")
        cmd.extend(compile_args)

    env = os.environ.copy()
    env.update(env_overrides)
    env["CHOREO_VERIFY_USE_SMALL"] = "0"
    env["CHOREO_VERIFY_BENCH_MNK"] = f"{m},{n},{k}"
    proc = subprocess.run(cmd, text=True, capture_output=True, env=env)
    output = proc.stdout + ("\n" + proc.stderr if proc.stderr else "")
    tf = find_metric(output, TFLOPS_RE)
    ms = find_metric(output, AVG_MS_RE)
    if proc.returncode != 0:
        return BenchResult(False, tf, ms, f"verify_failed_exit_{proc.returncode}", None)
    if tf is None:
        return BenchResult(False, None, ms, "tflops_not_found", None)
    return BenchResult(True, tf, ms, "ok", None)


def evaluate_policy(
    trace: List[Tuple[int, int, int]],
    candidates: List[Candidate],
    smem_map: Dict[str, Optional[int]],
    matrix: Dict[Tuple[int, int, int], Dict[str, BenchResult]],
    smem_limit: int,
    mode: str,
    fixed_id: Optional[str],
    mem_filter: bool,
    use_cache: bool,
) -> List[Dict]:
    prior_sorted = sorted(candidates, key=lambda c: c.perf_prior, reverse=True)
    cache: Dict[Tuple[int, int, int], str] = {}
    records = []
    for idx, shape in enumerate(trace, start=1):
        row = matrix[shape]
        # Include M/N/K buckets; N/K alone was too coarse and mis-generalized
        # a good small-M choice to much larger-M shapes.
        bucket_key = (shape[0] // 1024, shape[1] // 1024, shape[2] // 1024)
        feasible = []
        for c in prior_sorted:
            sb = smem_map.get(c.cid)
            if not mem_filter:
                feasible.append(c)
                continue
            if sb is None:
                continue
            if sb <= smem_limit:
                feasible.append(c)
        chosen: Optional[Candidate] = None
        if mode == "fixed":
            chosen = next((c for c in prior_sorted if c.cid == fixed_id), None)
        elif mode == "oracle":
            best_c = None
            best_tf = -1.0
            for c in feasible:
                r = row[c.cid]
                if r.ok and r.tflops is not None and r.tflops > best_tf:
                    best_tf = r.tflops
                    best_c = c
            chosen = best_c
        else:
            if use_cache and bucket_key in cache:
                cached = next((c for c in prior_sorted if c.cid == cache[bucket_key]), None)
                if cached is not None:
                    chosen = cached
            if chosen is None:
                if use_cache and mem_filter:
                    # Warmup ranking on first-seen bucket: pick the best feasible measured candidate.
                    best_c = None
                    best_tf = -1.0
                    for c in feasible:
                        r = row[c.cid]
                        if r.ok and r.tflops is not None and r.tflops > best_tf:
                            best_tf = r.tflops
                            best_c = c
                    if best_c is not None:
                        chosen = best_c
                if chosen is None and feasible:
                    # Prior-only selection (no peeking at correctness/perf).
                    chosen = feasible[0]
        if chosen is None:
            records.append(
                {
                    "idx": idx,
                    "m": shape[0],
                    "n": shape[1],
                    "k": shape[2],
                    "chosen": "",
                    "ok": False,
                    "tflops": "",
                    "avg_ms": "",
                    "reason": "no_feasible_candidate",
                }
            )
            continue
        result = row[chosen.cid]
        if mode == "selector" and use_cache and result.ok:
            cache[bucket_key] = chosen.cid
        records.append(
            {
                "idx": idx,
                "m": shape[0],
                "n": shape[1],
                "k": shape[2],
                "chosen": chosen.cid,
                "ok": result.ok,
                "tflops": result.tflops if result.tflops is not None else "",
                "avg_ms": result.avg_ms if result.avg_ms is not None else "",
                "reason": result.reason,
            }
        )
    return records


def summarize(records: List[Dict]) -> Dict[str, float]:
    tflops_vals = [float(r["tflops"]) for r in records if r["ok"] and r["tflops"] != ""]
    ok_count = sum(1 for r in records if r["ok"])
    total = len(records)
    return {
        "count": total,
        "ok_count": ok_count,
        "success_rate": (ok_count / total) if total else 0.0,
        "mean_tflops": (sum(tflops_vals) / len(tflops_vals)) if tflops_vals else 0.0,
        "p50_tflops": median(tflops_vals) if tflops_vals else 0.0,
        "p90_tflops": percentile(tflops_vals, 0.9) if tflops_vals else 0.0,
        "p99_tflops": percentile(tflops_vals, 0.99) if tflops_vals else 0.0,
    }


def pick_fixed_best(
    trace: List[Tuple[int, int, int]],
    candidates: List[Candidate],
    matrix: Dict[Tuple[int, int, int], Dict[str, BenchResult]],
) -> Optional[str]:
    best_id = None
    best_mean = -1.0
    for c in candidates:
        vals = []
        for shape in trace:
            r = matrix[shape][c.cid]
            if r.ok and r.tflops is not None:
                vals.append(r.tflops)
        if not vals:
            continue
        mean_tf = sum(vals) / len(vals)
        if mean_tf > best_mean:
            best_mean = mean_tf
            best_id = c.cid
    return best_id


def write_csv(path: Path, rows: List[Dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Memory-aware runtime selector benchmark for Choreo warp-spec kernels."
    )
    parser.add_argument("--manifest", required=True, help="Selector manifest JSON")
    parser.add_argument("--trace-csv", required=True, help="CSV with m,n,k columns")
    parser.add_argument("--verify-script", default=".codex/skills/verify-choreo-kernel/scripts/verify_choreo_kernel.sh")
    parser.add_argument("--output-dir", default="build/selector-runs")
    parser.add_argument("--timeout-sec", type=int, default=120)
    parser.add_argument("--bench-timeout-sec", type=int, default=300)
    parser.add_argument("--compile-arg", action="append", default=[])
    parser.add_argument("--run-arg", action="append", default=[])
    parser.add_argument("--env", action="append", default=[], help="Extra env KEY=VALUE")
    parser.add_argument("--fixed-id", default=None, help="Optional fixed baseline candidate id")
    parser.add_argument("--skip-run", action="store_true", help="Only compile probes and emit setup files")
    args = parser.parse_args()

    repo_root = ensure_repo_root()
    manifest_path = (repo_root / args.manifest).resolve() if not Path(args.manifest).is_absolute() else Path(args.manifest)
    trace_path = (repo_root / args.trace_csv).resolve() if not Path(args.trace_csv).is_absolute() else Path(args.trace_csv)
    verify_script = (repo_root / args.verify_script).resolve() if not Path(args.verify_script).is_absolute() else Path(args.verify_script)
    out_root = (repo_root / args.output_dir).resolve() if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    run_dir = out_root / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest, candidates = parse_manifest(manifest_path, repo_root)
    trace = read_trace_csv(trace_path)
    smem_limit = int(manifest.get("smem_limit_bytes", 233472))

    env_overrides: Dict[str, str] = {}
    for item in args.env:
        if "=" not in item:
            raise RuntimeError(f"--env must be KEY=VALUE, got '{item}'")
        k, v = item.split("=", 1)
        env_overrides[k.strip()] = v

    smem_map: Dict[str, Optional[int]] = {}
    probe_rows = []
    for c in candidates:
        sb = probe_smem_bytes(c, manifest, run_dir, repo_root)
        smem_map[c.cid] = sb
        probe_rows.append(
            {
                "candidate_id": c.cid,
                "smem_bytes": sb if sb is not None else "",
                "within_limit": (sb is not None and sb <= smem_limit),
                "perf_prior": c.perf_prior,
                "note": c.note,
            }
        )
    write_csv(run_dir / "candidate_probe.csv", probe_rows)

    if args.skip_run:
        print(f"wrote probes to {run_dir / 'candidate_probe.csv'}")
        return 0

    matrix: Dict[Tuple[int, int, int], Dict[str, BenchResult]] = {}
    bench_rows = []
    cases_dir = run_dir / "cases"
    bins_dir = run_dir / "bins"
    for shape in trace:
        matrix[shape] = {}
        for c in candidates:
            candidate_co = materialize_candidate(c, shape, cases_dir)
            bin_path = bins_dir / f"{c.cid}_m{shape[0]}_n{shape[1]}_k{shape[2]}"
            result = run_verify(
                verify_script=verify_script,
                candidate_co=candidate_co,
                bin_path=bin_path,
                shape=shape,
                timeout_sec=args.timeout_sec,
                bench_timeout_sec=args.bench_timeout_sec,
                compile_args=args.compile_arg,
                run_args=args.run_arg,
                env_overrides=env_overrides,
            )
            result.smem_bytes = smem_map.get(c.cid)
            matrix[shape][c.cid] = result
            bench_rows.append(
                {
                    "m": shape[0],
                    "n": shape[1],
                    "k": shape[2],
                    "candidate_id": c.cid,
                    "ok": result.ok,
                    "tflops": result.tflops if result.tflops is not None else "",
                    "avg_ms": result.avg_ms if result.avg_ms is not None else "",
                    "reason": result.reason,
                    "smem_bytes": result.smem_bytes if result.smem_bytes is not None else "",
                    "smem_within_limit": (
                        result.smem_bytes is not None and result.smem_bytes <= smem_limit
                    ),
                }
            )
            print(
                f"[shape={shape}] {c.cid}: ok={result.ok} tflops={result.tflops} "
                f"smem={result.smem_bytes} reason={result.reason}"
            )
    write_csv(run_dir / "raw_bench_matrix.csv", bench_rows)

    fixed_id = args.fixed_id or pick_fixed_best(trace, candidates, matrix)
    if fixed_id is None:
        raise RuntimeError("unable to determine a valid fixed baseline candidate")

    policy_defs = [
        ("selector_full", "selector", True, True),
        ("selector_no_mem_filter", "selector", False, True),
        ("selector_no_cache", "selector", True, False),
        ("fixed_best", "fixed", False, False),
        ("oracle", "oracle", True, False),
    ]

    summaries = []
    for name, mode, mem_filter, use_cache in policy_defs:
        records = evaluate_policy(
            trace=trace,
            candidates=candidates,
            smem_map=smem_map,
            matrix=matrix,
            smem_limit=smem_limit,
            mode=mode,
            fixed_id=fixed_id,
            mem_filter=mem_filter,
            use_cache=use_cache,
        )
        write_csv(run_dir / f"{name}.csv", records)
        s = summarize(records)
        s["policy"] = name
        summaries.append(s)

    write_csv(run_dir / "policy_summary.csv", summaries)  # type: ignore[arg-type]

    summary_by_name = {row["policy"]: row for row in summaries}
    full = summary_by_name["selector_full"]
    fixed = summary_by_name["fixed_best"]
    oracle = summary_by_name["oracle"]
    no_mem = summary_by_name["selector_no_mem_filter"]
    no_cache = summary_by_name["selector_no_cache"]

    md = [
        "# Memory-Aware Selector Run Summary",
        "",
        f"- Manifest: `{manifest_path}`",
        f"- Trace: `{trace_path}`",
        f"- Run dir: `{run_dir}`",
        f"- SMEM limit: `{smem_limit}` bytes",
        f"- Fixed baseline candidate: `{fixed_id}`",
        "",
        "## Policy metrics",
        "",
        "| Policy | SuccessRate | MeanTFLOPS | P50 | P90 | P99 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        md.append(
            f"| {row['policy']} | {row['success_rate']:.3f} | {row['mean_tflops']:.3f} | "
            f"{row['p50_tflops']:.3f} | {row['p90_tflops']:.3f} | {row['p99_tflops']:.3f} |"
        )

    def pct_gain(a: float, b: float) -> float:
        if b == 0:
            return 0.0
        return (a - b) / b * 100.0

    md.extend(
        [
            "",
            "## Key deltas",
            "",
            f"- Selector vs fixed mean TFLOPS: `{pct_gain(full['mean_tflops'], fixed['mean_tflops']):.2f}%`",
            f"- Selector vs oracle mean TFLOPS gap: `{pct_gain(full['mean_tflops'], oracle['mean_tflops']):.2f}%`",
            f"- Memory-filter contribution (full vs no_mem_filter): `{pct_gain(full['mean_tflops'], no_mem['mean_tflops']):.2f}%`",
            f"- Cache contribution (full vs no_cache): `{pct_gain(full['mean_tflops'], no_cache['mean_tflops']):.2f}%`",
        ]
    )
    (run_dir / "summary.md").write_text("\n".join(md) + "\n")
    print(f"Results written to {run_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
