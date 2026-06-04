# Flash attention benchmark variants

Six variants under `flash_atten/`. Legacy suites: [`.tmp/`](.tmp/).

Backends: **Choreo** (where available) and **FA3** (`flash_attn_interface`). No TileLang.

## Layout convention

| Component | Layout |
|-----------|--------|
| FA3 baselines | **BSHD** `(batch, seqlen, heads, dim)` |
| Choreo (decoder / kv variants) | **BSHD** on host and device where wired (see variant README) |
| Legacy `.tmp/flash_mha_fwd_bhsd` | **BHSD** kernels + host transpose (not used by `decoder_causal_d64`) |

## Variant index

| ID | Directory | Workload (headline) | Choreo |
|----|-----------|---------------------|--------|
| 1 | `decoder_causal_d64/` | B=1 H=32 S=8192 causal D=64 | Yes (BSHD kernel) |
| 2 | `prefill_d128/` | B=2 H=16 S=8192 non-causal D=128 bf16 | No |
| 3 | `gqa_decoder_d64/` | B=1 Hq=32 Hkv=4 S=8192 causal D=64 | No |
| 4 | `kv_cache_decode_d64/` | B=1 H=32 q=1 kv=8192 D=64 | Yes (BHSD v2 1p1c TMA kernel) |
| 5 | `causal_prefill_d128/` | B=2 H=16 S=8192 causal D=128 bf16 | No |
| 6 | `fp8_d128/` | B=2 H=16 S=8192 FP8 non-causal D=128 | No |

Details and full config tables: each variant `README.md`.

## FA3 install (ml-hopper)

Forward-only, SM90, **D in {64, 128}**, **fp16 + bf16 + FP8 (e4m3)**. Rebuild log:
`/tmp/fa3_hopper_install_fp8.log`.

## Run

```bash
source /home/fem/.env/ml-hopper/bin/activate

# Default GPU 1; override with --gpu
bash decoder_causal_d64/baselines/bench.sh --gpu 1
bash decoder_causal_d64/compare_all.sh --gpu 1
cd decoder_causal_d64/choreo && bash bench.sh --gpu 1 --kernel v2_manual_s2_1p2c_tma.co --no-verify

bash run_all_fa3.sh --gpu 1
```

Shared: `_fa3_common.py`, `_timing.py`, `_compare_common.py`.
