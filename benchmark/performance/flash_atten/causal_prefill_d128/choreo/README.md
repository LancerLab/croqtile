# Choreo: causal_prefill_d128

## Layout

- **Kernel** (`*.co`) and **host** (`mha_helper.hpp`): **BSHD**
  `(batch, seqlen, heads, dim)` -- same as FA3 baseline (no layout transpose).
- Device tensors are uploaded with `cudaMemcpy` directly from host BSHD buffers.

## Files

| File | Role |
|------|------|
| `mha_helper.hpp` | BSHD I/O, verify, kernel-only timing (bf16, D=128) |
| `bench.sh` | Compile/run `.co` |
| `v1_manual_baseline.co` | DMA baseline (sequential load + compute) |

## Run

```bash
cd benchmark/performance/flash_atten/causal_prefill_d128/choreo
bash bench.sh --gpu 1 --kernel v1_manual_baseline.co
bash bench.sh --gpu 1 --kernel v1_manual_baseline.co --no-verify
```

## Compare with FA3

```bash
cd ..
bash compare_all.sh --gpu 1 --kernel v1_manual_baseline.co
```

Configs in each `.co` match `../config.py` (B=2 S=8192 and B=1 S=4096, causal).
