# Choreo: decoder_causal_d64

## Layout

- **Kernel** (`*.co`) and **host** (`mha_helper.hpp`): **BSHD**
  `(batch, seqlen, heads, dim)` — same as FA3 baseline (no layout transpose).
- Device tensors are uploaded with `cudaMemcpy` directly from host BSHD buffers.

## Files

| File | Role |
|------|------|
| `mha_helper.hpp` | BSHD I/O, verify, kernel-only timing |
| `bench.sh` | Compile/run `.co` |
| `v2_manual_s2_1p2c_tma.co` | Default BSHD kernel (TMA + warpspec) |
| `v1_manual_baseline.co` | DMA baseline |

## Run

```bash
cd benchmark/performance/flash_atten/decoder_causal_d64/choreo
bash bench.sh --gpu 1 --kernel v2_manual_s2_1p2c_tma.co --no-verify
```

## Compare with FA3

```bash
cd ..
bash compare_all.sh --gpu 1 --kernel v2_manual_s2_1p2c_tma.co
```

Configs in each `.co` match `../config.py` (2048 / 4096 / 8192, q_seq=kv_seq).
