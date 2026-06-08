# Choreo: kv_cache_decode_d64

**BHSD** `v2_manual_s2_1p1c_tma.co` kernel (1 producer + 1 consumer warp groups, TMA).
Same `mha_helper.hpp` and `bench.sh` pattern as `decoder_causal_d64`.
Kernel takes runtime `Q_SEQ` / `KV_SEQ` from host tensor shapes.

## Layout

- Device kernel: **BHSD**
- Host I/O: **BSHD** with transpose (not in timed loop)
- FA3 baseline: **BSHD** native

## Workload config

Benchmark workload shapes are defined in `../config.py`. `bench.sh`
auto-generates `build/bench_configs.inc` from it before compiling.
Edit `config.py` to change workloads (q=1/kv=8192 and q=128/kv=8192).

## Run

```bash
bash bench.sh --gpu 1 --kernel v2_manual_s2_1p1c_tma.co --no-verify
bash bench.sh --gpu 1 --kernel v2_manual_s2_1p1c_tma.co --force-compile
```

Repeated runs skip Choreo recompilation when the kernel source hasn't changed.
Use `--force-compile` to override.
