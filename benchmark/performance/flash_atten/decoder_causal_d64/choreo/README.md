# Choreo: decoder_causal_d64

## Layout

- **Kernel** (`*.co`) and **host** (`mha_helper.hpp`): **BSHD**
  `(batch, seqlen, heads, dim)` -- same as FA3 baseline (no layout transpose).
- Device tensors are uploaded with `cudaMemcpy` directly from host BSHD buffers.

## Files

| File | Role |
|------|------|
| `mha_helper.hpp` | BSHD I/O, verify, kernel-only timing |
| `bench.sh` | Compile/run `.co` with compilation caching |
| `v2_manual_s2_1p2c_tma.co` | Default BSHD kernel (TMA + warpspec) |
| `v1_manual_baseline.co` | DMA baseline |

## Workload config

Benchmark workload shapes (batch, heads, seqlen, causal) are defined in
`../config.py`. `bench.sh` auto-generates `build/bench_configs.inc` from it
before compiling. All `.co` files `#include` this header -- edit `config.py`
to change workloads.

## Run

```bash
cd benchmark/performance/flash_atten/decoder_causal_d64/choreo
bash bench.sh --gpu 1 --kernel v2_manual_s2_1p2c_tma.co --no-verify
bash bench.sh --gpu 1 --kernel v2_manual_s2_1p2c_tma.co --force-compile
```

Repeated runs skip Choreo recompilation when the kernel source hasn't changed.
Use `--force-compile` to override.

## Compare with FA3

```bash
cd ..
bash compare_all.sh --gpu 1 --kernel v2_manual_s2_1p2c_tma.co
```
