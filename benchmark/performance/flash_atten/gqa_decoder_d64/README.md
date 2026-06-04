# Variant 3: gqa_decoder_d64

Grouped-query attention: **32 query heads**, **4 KV heads** (8:1), causal, D=64.
FA3 uses **BSHD** with `k,v` shaped `(B, S, 4, 64)`.

## Workload

| Label | B | S | Role |
|-------|---|---|------|
| **B=1 Hq=32 Hkv=4 SEQ=8192 GQA** | **1** | **8192** | **Long-context GQA decode (headline)** |
| B=2 Hq=32 Hkv=4 SEQ=4096 GQA | 2 | 4096 | Batched mid-long context |

## Choreo

Not yet — current BHSD kernels use one `H` for Q and KV. FA3 baseline only.

## Run

```bash
bash baselines/bench.sh
bash compare_all.sh
```
