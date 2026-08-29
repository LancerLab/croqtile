#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path
from statistics import median
from typing import Dict, List


def load_policy(path: Path) -> List[Dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    vals = sorted(values)
    if len(vals) == 1:
        return vals[0]
    pos = pct * (len(vals) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(vals) - 1)
    frac = pos - lo
    return vals[lo] * (1.0 - frac) + vals[hi] * frac


def summarize(rows: List[Dict]) -> Dict[str, float]:
    ok_rows = [r for r in rows if str(r["ok"]).lower() == "true"]
    tflops = [float(r["tflops"]) for r in ok_rows if r["tflops"] != ""]
    count = len(rows)
    ok_count = len(ok_rows)
    return {
        "count": count,
        "ok_count": ok_count,
        "success_rate": ok_count / count if count else 0.0,
        "mean_tflops": sum(tflops) / len(tflops) if tflops else 0.0,
        "p50": median(tflops) if tflops else 0.0,
        "p90": percentile(tflops, 0.9) if tflops else 0.0,
        "p99": percentile(tflops, 0.99) if tflops else 0.0,
    }


def write_csv(path: Path, header: List[str], rows: List[List]) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for row in rows:
            w.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build paper-ready asset tables from selector run directory."
    )
    parser.add_argument("--run-dir", required=True, help="Selector run directory")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    selector_full = load_policy(run_dir / "selector_full.csv")
    selector_no_mem = load_policy(run_dir / "selector_no_mem_filter.csv")
    selector_no_cache = load_policy(run_dir / "selector_no_cache.csv")
    fixed = load_policy(run_dir / "fixed_best.csv")
    oracle = load_policy(run_dir / "oracle.csv")

    policies = {
        "selector_full": summarize(selector_full),
        "selector_no_mem_filter": summarize(selector_no_mem),
        "selector_no_cache": summarize(selector_no_cache),
        "fixed_best": summarize(fixed),
        "oracle": summarize(oracle),
    }

    # Effective throughput counts failures as zero TFLOPS.
    def effective_mean(rows: List[Dict]) -> float:
        vals = [float(r["tflops"]) if str(r["ok"]).lower() == "true" and r["tflops"] != "" else 0.0 for r in rows]
        return sum(vals) / len(vals) if vals else 0.0

    effective = {
        "selector_full": effective_mean(selector_full),
        "selector_no_mem_filter": effective_mean(selector_no_mem),
        "selector_no_cache": effective_mean(selector_no_cache),
        "fixed_best": effective_mean(fixed),
        "oracle": effective_mean(oracle),
    }

    def pct_delta_str(a: float, b: float) -> str:
        if b == 0.0:
            if a == 0.0:
                return "0.00%"
            return "inf (baseline is 0)"
        return f"{((a - b) / b * 100.0):.2f}%"

    # Figure data: speedup CDF of selector_full vs fixed_best
    speedups = []
    for s_row, f_row in zip(selector_full, fixed):
        s_ok = str(s_row["ok"]).lower() == "true" and s_row["tflops"] != ""
        f_ok = str(f_row["ok"]).lower() == "true" and f_row["tflops"] != ""
        if not s_ok or not f_ok:
            continue
        speedups.append(float(s_row["tflops"]) / float(f_row["tflops"]))
    speedups_sorted = sorted(speedups)
    cdf_rows = []
    if speedups_sorted:
        n = len(speedups_sorted)
        for i, v in enumerate(speedups_sorted, start=1):
            cdf_rows.append([i / n, v])

    write_csv(
        run_dir / "figure_speedup_cdf_selector_vs_fixed.csv",
        ["cdf", "speedup"],
        cdf_rows,
    )

    write_csv(
        run_dir / "figure_failure_avoidance.csv",
        ["policy", "success_rate", "failure_rate"],
        [
            ["selector_full", policies["selector_full"]["success_rate"], 1.0 - policies["selector_full"]["success_rate"]],
            [
                "selector_no_mem_filter",
                policies["selector_no_mem_filter"]["success_rate"],
                1.0 - policies["selector_no_mem_filter"]["success_rate"],
            ],
            ["selector_no_cache", policies["selector_no_cache"]["success_rate"], 1.0 - policies["selector_no_cache"]["success_rate"]],
        ],
    )

    md = []
    md.append("# CGO Asset Pack: Memory-Aware Configuration Selection")
    md.append("")
    md.append("## Aggregate metrics")
    md.append("")
    md.append("| Policy | SuccessRate | MeanTFLOPS | EffectiveMeanTFLOPS | P50 | P90 | P99 |")
    md.append("|---|---:|---:|---:|---:|---:|---:|")
    for name in ["selector_full", "selector_no_mem_filter", "selector_no_cache", "fixed_best", "oracle"]:
        s = policies[name]
        md.append(
            f"| {name} | {s['success_rate']:.3f} | {s['mean_tflops']:.3f} | {effective[name]:.3f} | "
            f"{s['p50']:.3f} | {s['p90']:.3f} | {s['p99']:.3f} |"
        )

    md.append("")
    md.append("## Ablation deltas")
    md.append("")
    md.append(
        f"- **Memory-fit filter (effective throughput)**: "
        f"{pct_delta_str(effective['selector_full'], effective['selector_no_mem_filter'])} "
        f"(`selector_full` vs `selector_no_mem_filter`)."
    )
    md.append(
        f"- **Cache/warmup effect (effective throughput)**: "
        f"{pct_delta_str(effective['selector_full'], effective['selector_no_cache'])} "
        f"(`selector_full` vs `selector_no_cache`)."
    )
    md.append(
        f"- **Selector vs fixed baseline (effective throughput)**: "
        f"{pct_delta_str(effective['selector_full'], effective['fixed_best'])}."
    )
    md.append(
        f"- **Selector gap to oracle (effective throughput)**: "
        f"{pct_delta_str(effective['selector_full'], effective['oracle'])}."
    )
    md.append("")
    md.append("## Figure inputs")
    md.append("")
    md.append("- `figure_speedup_cdf_selector_vs_fixed.csv`")
    md.append("- `figure_failure_avoidance.csv`")
    md.append("- `raw_bench_matrix.csv`")
    md.append("- `policy_summary.csv`")

    (run_dir / "paper_asset_pack.md").write_text("\n".join(md) + "\n")
    print(f"Wrote paper assets to {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
