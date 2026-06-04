# Variant 4: kv_cache_decode_d64

**q_seq < kv_seq**, causal, bottom-right mask: new queries attend to a long KV cache.

## Layout

| Backend | Layout |
|---------|--------|
| FA3 | BSHD |
| Choreo kernel | **BHSD** (reuse `v2_manual_s2_1p1c_tma.co`, `past_len = KV_SEQ - Q_SEQ`) |

## Workload

| Label | q | kv | Role |
|-------|---|-----|------|
| **B=1 H=32 q=1 kv=8192 decode** | **1** | **8192** | **Single-token decode (headline)** |
| B=1 H=32 q=128 kv=8192 chunk | 128 | 8192 | Chunk prefill into cache |

**D=64**, **B=1**, **H=32**, fp16.

## Run

```bash
bash baselines/bench.sh
cd choreo && bash bench.sh --kernel v2_manual_s2_1p1c_tma.co --no-verify
bash compare_all.sh
```
