# Choreo: kv_cache_decode_d64

**BHSD** `v2_manual_s2_1p1c_tma.co` kernel (1 producer + 1 consumer warp groups, TMA).
Same `mha_helper.hpp` and `bench.sh` pattern as `decoder_causal_d64`.
Kernel takes runtime `Q_SEQ` / `KV_SEQ` from host tensor shapes.

## Layout

- Device kernel: **BHSD**
- Host I/O: **BSHD** with transpose (not in timed loop)
- FA3 baseline: **BSHD** native

## Run

```bash
export CUDA_VISIBLE_DEVICES=1
bash bench.sh --kernel v2_manual_s2_1p1c_tma.co --no-verify
```

Configs embedded in `.co` main: q=1/kv=8192 and q=128/kv=8192.
