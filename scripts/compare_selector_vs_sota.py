#!/usr/bin/env python3
import argparse
import csv
import math
from pathlib import Path
from statistics import mean


def read_manifest_from_summary(summary_path: Path) -> str:
    if not summary_path.exists():
        return "unknown_manifest"
    for line in summary_path.read_text().splitlines():
        if line.startswith("- Manifest:"):
            manifest = line.split("`")[1] if "`" in line else line
            return Path(manifest).name
    return "unknown_manifest"


def workload_label(manifest_name: str) -> str:
    if "1p1c" in manifest_name:
        return "warpspec_1p1c"
    if "1p2c" in manifest_name:
        return "warpspec_1p2c"
    if "selector_manifest.json" in manifest_name:
        return "warpspec_1p2c"
    return manifest_name.replace(".json", "")


def parse_policy_summary(csv_path: Path):
    rows = {}
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows[r["policy"]] = {
                "count": int(float(r["count"])),
                "ok_count": int(float(r["ok_count"])),
                "success_rate": float(r["success_rate"]),
                "mean_tflops": float(r["mean_tflops"]),
                "p50_tflops": float(r["p50_tflops"]),
                "p90_tflops": float(r["p90_tflops"]),
                "p99_tflops": float(r["p99_tflops"]),
            }
    return rows


def pct_delta(a: float, b: float) -> float:
    if b == 0:
        return math.inf if a > 0 else 0.0
    return (a / b - 1.0) * 100.0


def fmt(v: float, nd=2):
    if math.isinf(v):
        return "inf"
    return f"{v:.{nd}f}"


def write_svg(out_svg: Path, workloads, metrics):
    width = 1200
    height = 620
    pad = 50
    chart_w = width - 2 * pad
    chart_h = 220
    colors = {
        "selector_full": "#2563eb",
        "fixed_best": "#16a34a",
        "oracle": "#f59e0b",
        "selector_no_mem_filter": "#dc2626",
    }
    bar_w = 28
    group_gap = 120
    x0 = pad + 80
    y0 = 120
    y1 = 420

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect x="0" y="0" width="100%" height="100%" fill="#f8fafc"/>',
        '<text x="50" y="40" font-family="Arial" font-size="26" font-weight="700" fill="#0f172a">Current Method vs SOTA-Proxy Comparison</text>',
        '<text x="50" y="66" font-family="Arial" font-size="14" fill="#334155">Top: normalized mean TFLOPS (oracle=100). Bottom: success rate (%)</text>',
    ]

    # axes
    lines += [
        f'<line x1="{pad}" y1="{y0 + chart_h}" x2="{width-pad}" y2="{y0 + chart_h}" stroke="#64748b" stroke-width="1.5"/>',
        f'<line x1="{pad}" y1="{y0}" x2="{pad}" y2="{y0 + chart_h}" stroke="#64748b" stroke-width="1.5"/>',
        f'<line x1="{pad}" y1="{y1 + chart_h}" x2="{width-pad}" y2="{y1 + chart_h}" stroke="#64748b" stroke-width="1.5"/>',
        f'<line x1="{pad}" y1="{y1}" x2="{pad}" y2="{y1 + chart_h}" stroke="#64748b" stroke-width="1.5"/>',
    ]

    # grid lines
    for pct in [0, 25, 50, 75, 100]:
        yy_top = y0 + chart_h - chart_h * pct / 100.0
        yy_bot = y1 + chart_h - chart_h * pct / 100.0
        lines.append(f'<line x1="{pad}" y1="{yy_top}" x2="{width-pad}" y2="{yy_top}" stroke="#e2e8f0" stroke-width="1"/>')
        lines.append(f'<line x1="{pad}" y1="{yy_bot}" x2="{width-pad}" y2="{yy_bot}" stroke="#e2e8f0" stroke-width="1"/>')
        lines.append(f'<text x="{pad-34}" y="{yy_top+4}" font-family="Arial" font-size="11" fill="#64748b">{pct}</text>')
        lines.append(f'<text x="{pad-34}" y="{yy_bot+4}" font-family="Arial" font-size="11" fill="#64748b">{pct}</text>')

    policies = ["selector_full", "fixed_best", "oracle", "selector_no_mem_filter"]

    for wi, wl in enumerate(workloads):
        gx = x0 + wi * group_gap
        lines.append(f'<text x="{gx+6}" y="{y0+chart_h+22}" font-family="Arial" font-size="12" fill="#0f172a">{wl}</text>')
        for pi, p in enumerate(policies):
            val_perf = metrics[wl]["norm_perf"].get(p, 0.0)
            val_succ = metrics[wl]["success"].get(p, 0.0) * 100.0
            bx = gx + pi * (bar_w + 8)
            h_perf = chart_h * val_perf / 100.0
            h_succ = chart_h * val_succ / 100.0
            c = colors[p]
            lines.append(f'<rect x="{bx}" y="{y0+chart_h-h_perf}" width="{bar_w}" height="{h_perf}" fill="{c}" opacity="0.9"/>')
            lines.append(f'<rect x="{bx}" y="{y1+chart_h-h_succ}" width="{bar_w}" height="{h_succ}" fill="{c}" opacity="0.9"/>')

    # legend
    lx = 780
    ly = 86
    for i, p in enumerate(policies):
        y = ly + i * 20
        lines.append(f'<rect x="{lx}" y="{y-10}" width="12" height="12" fill="{colors[p]}"/>')
        lines.append(f'<text x="{lx+18}" y="{y}" font-family="Arial" font-size="12" fill="#1e293b">{p}</text>')

    lines += [
        '</svg>',
    ]
    out_svg.write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description="Compare selector_full vs SOTA proxies")
    ap.add_argument("--run", action="append", required=True, help="Run directory containing policy_summary.csv")
    ap.add_argument("--out-dir", required=True, help="Output directory for report and svg")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = []
    for run_dir_s in args.run:
        run_dir = Path(run_dir_s)
        csv_path = run_dir / "policy_summary.csv"
        summary_path = run_dir / "summary.md"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing {csv_path}")
        rows = parse_policy_summary(csv_path)
        manifest = read_manifest_from_summary(summary_path)
        wl = workload_label(manifest)
        data.append((wl, run_dir.name, rows))

    data.sort(key=lambda x: x[0])
    workloads = [d[0] for d in data]

    # Build report metrics
    lines = []
    lines.append("# Current Method vs SOTA-Proxy Experiment")
    lines.append("")
    lines.append("SOTA proxies in this repo:")
    lines.append("- `fixed_best`: strong static baseline")
    lines.append("- `oracle`: upper bound from per-shape best candidate")
    lines.append("- `selector_no_mem_filter`: robustness ablation")
    lines.append("")
    lines.append("| Workload | selector_full | fixed_best | oracle | no_mem_filter | selector vs fixed | selector gap to oracle | success(full/fixed/no_mem) |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")

    agg_gain_vs_fixed = []
    agg_gap_to_oracle = []
    agg_success_full = []
    agg_success_nomem = []

    svg_metrics = {}
    for wl, run_name, rows in data:
        sf = rows["selector_full"]
        fb = rows["fixed_best"]
        oc = rows["oracle"]
        nm = rows.get("selector_no_mem_filter", {"mean_tflops": 0.0, "success_rate": 0.0})
        gain_fixed = pct_delta(sf["mean_tflops"], fb["mean_tflops"])
        gap_oracle = pct_delta(oc["mean_tflops"], sf["mean_tflops"])
        agg_gain_vs_fixed.append(gain_fixed)
        agg_gap_to_oracle.append(gap_oracle)
        agg_success_full.append(sf["success_rate"])
        agg_success_nomem.append(nm["success_rate"])

        lines.append(
            f"| {wl} ({run_name}) | {sf['mean_tflops']:.3f} | {fb['mean_tflops']:.3f} | {oc['mean_tflops']:.3f} | {nm['mean_tflops']:.3f} | "
            f"{fmt(gain_fixed)}% | {fmt(gap_oracle)}% | {sf['success_rate']:.3f}/{fb['success_rate']:.3f}/{nm['success_rate']:.3f} |"
        )

        norm = lambda x: 100.0 * x / oc["mean_tflops"] if oc["mean_tflops"] > 0 else 0.0
        svg_metrics[wl] = {
            "norm_perf": {
                "selector_full": norm(sf["mean_tflops"]),
                "fixed_best": norm(fb["mean_tflops"]),
                "oracle": norm(oc["mean_tflops"]),
                "selector_no_mem_filter": norm(nm["mean_tflops"]),
            },
            "success": {
                "selector_full": sf["success_rate"],
                "fixed_best": fb["success_rate"],
                "oracle": oc["success_rate"],
                "selector_no_mem_filter": nm["success_rate"],
            },
        }

    lines.append("")
    lines.append("## Aggregate view")
    lines.append("")
    lines.append(f"- Mean selector gain vs fixed_best: **{fmt(mean(agg_gain_vs_fixed))}%**")
    lines.append(f"- Mean selector gap to oracle: **{fmt(mean(agg_gap_to_oracle))}%**")
    lines.append(f"- Mean success rate (selector_full): **{fmt(mean(agg_success_full), 3)}**")
    lines.append(f"- Mean success rate (no_mem_filter): **{fmt(mean(agg_success_nomem), 3)}**")
    lines.append("")
    lines.append("## Beyond performance (recommended for paper)")
    lines.append("")
    lines.append("- Robustness: failure rate under dynamic-shape traces (already captured by success_rate).")
    lines.append("- Tail behavior: compare `p90/p99` TFLOPS variance across policies.")
    lines.append("- Selection overhead: add dispatch overhead measurement (`selector decision time`).")
    lines.append("- Generalization: repeat on additional op families and architecture variants.")

    out_md = out_dir / "current_vs_sota_report.md"
    out_md.write_text("\n".join(lines))

    out_svg = out_dir / "current_vs_sota_chart.svg"
    write_svg(out_svg, workloads, svg_metrics)

    print(f"Wrote report: {out_md}")
    print(f"Wrote chart:  {out_svg}")


if __name__ == "__main__":
    main()
