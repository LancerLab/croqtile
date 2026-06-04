# Variant 2: prefill_d128

Non-causal full attention at **D=128** (FA3 blog / peak-TFLOPS shape). **BSHD**, bf16.

## Workload

| Label | B | H | S | Role |
|-------|---|---|---|------|
| **B=2 H=16 SEQ=8192 prefill** | **2** | **16** | **8192** | **Headline (FA3 blog)** |
| B=1 H=16 SEQ=8192 prefill | 1 | 16 | 8192 | Single-stream long context |

Model width H*D = 2048. Requires FA3 build with **hdim 128**.

## Choreo

Not yet — needs BHSD (or BSHD) kernel at D=128. FA3 baseline only.

## Run

```bash
bash baselines/bench.sh
bash compare_all.sh
```
