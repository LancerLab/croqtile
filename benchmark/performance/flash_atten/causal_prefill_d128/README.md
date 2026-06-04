# Variant 5: causal_prefill_d128

Causal full-sequence forward at **D=128** (training / prefill phase). **BSHD**, bf16.
Between decoder D=64 and non-causal `prefill_d128` peak case.

## Workload

| Label | B | S | Role |
|-------|---|---|------|
| **B=2 H=16 SEQ=8192 causal prefill** | **2** | **8192** | **Headline** |
| B=1 H=16 SEQ=4096 causal prefill | 1 | 4096 | Single-request mid-long |

## Choreo

v1 DMA baseline kernel (bf16, BSHD). Same tile pattern as `decoder_causal_d64` v1
with D=128 and bf16 type.

## Run

```bash
bash baselines/bench.sh
cd choreo && bash bench.sh --kernel v1_manual_baseline.co --no-verify
bash compare_all.sh
```
