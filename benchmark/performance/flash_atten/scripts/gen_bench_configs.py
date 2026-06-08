#!/usr/bin/env python3
"""Generate a C++ BenchConfig array from a variant's config.py.

Usage:
    python gen_bench_configs.py <variant_dir> [output_path]

Reads <variant_dir>/config.py, emits a C++ header with a static
mha_helper::BenchConfig array that Choreo .co files can #include.
If output_path is omitted, writes to <variant_dir>/choreo/build/bench_configs.inc.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path


def load_config(variant_dir: Path):
    config_path = variant_dir / "config.py"
    if not config_path.exists():
        print(f"ERROR: {config_path} not found", file=sys.stderr)
        sys.exit(1)

    fa_root = variant_dir.parent
    if str(fa_root / "scripts") not in sys.path:
        sys.path.insert(0, str(fa_root / "scripts"))

    spec = importlib.util.spec_from_file_location("config", config_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def generate_inc(configs, variant_id: str) -> str:
    lines = [
        f"// Auto-generated from {variant_id}/config.py -- do not edit",
        "#pragma once",
        "",
        "static const mha_helper::BenchConfig kBenchConfigs[] = {",
    ]
    for c in configs:
        causal = "true" if c.causal else "false"
        label = c.label.replace('"', '\\"')
        lines.append(
            f'    {{{c.batch}, {c.nheads_q}, {c.seqlen_q}, {c.seqlen_k}, '
            f'{causal}, "{label}"}},'
        )
    lines.append("};")
    lines.append(
        f"static constexpr int kBenchConfigCount = "
        f"sizeof(kBenchConfigs) / sizeof(kBenchConfigs[0]);"
    )
    lines.append("")
    return "\n".join(lines)


def main():
    if len(sys.argv) < 2:
        print(__doc__, file=sys.stderr)
        sys.exit(1)

    variant_dir = Path(sys.argv[1]).resolve()
    mod = load_config(variant_dir)

    if len(sys.argv) >= 3:
        out_path = Path(sys.argv[2])
    else:
        out_path = variant_dir / "choreo" / "build" / "bench_configs.inc"

    out_path.parent.mkdir(parents=True, exist_ok=True)

    content = generate_inc(mod.BENCHMARK_CONFIGS, mod.VARIANT_ID)

    if out_path.exists() and out_path.read_text() == content:
        return

    out_path.write_text(content)
    print(f"[gen_bench_configs] wrote {out_path}")


if __name__ == "__main__":
    main()
