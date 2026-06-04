# Variant 1: decoder_causal_d64

Causal self-attention with **q_seq == kv_seq**: full-context forward (decoder chunk /
prefill), not KV-cache decode (`kv_cache_decode_d64/`).

## Layout

| Backend | Tensor layout |
|---------|----------------|
| **FA3** (`bench_fa3.py`) | **BSHD** `(batch, seqlen, heads, dim)` |
| **Choreo** (`.co` + `mha_helper.hpp`) | **BSHD** `(batch, seqlen, heads, dim)` |

## Workload

| Label | B | H | S | Role |
|-------|---|---|---|------|
| B=1 H=32 SEQ=2048 decoder | 1 | 32 | 2048 | Mid-context |
| B=2 H=32 SEQ=4096 decoder | 2 | 32 | 4096 | Batched long context |
| **B=1 H=32 SEQ=8192 decoder** | **1** | **32** | **8192** | **Headline compare** |

**D=64**, **causal=True**, **fp16**. Seq ≥ 2048 (two+ `BLOCK_M=128` tiles along Q).

## Run

```bash
bash baselines/bench.sh
cd choreo && bash bench.sh --kernel v2_manual_s2_1p2c_tma.co --no-verify
bash compare_all.sh
```
