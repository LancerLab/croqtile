# Variant 6: fp8_d128

Hopper **FP8** (`float8_e4m3fn`) with descale tensors. **BSHD**, D=128.

## Workload

| Label | causal | Role |
|-------|--------|------|
| **B=2 H=16 SEQ=8192 FP8 prefill** | no | **Peak FP8 headline (blog-class)** |
| B=2 H=16 SEQ=8192 FP8 causal | yes | Causal FP8 prefill |

## FA3 build (ml-hopper)

Installed with FP8 **and** previous paths (forward, SM90, D=64/128, fp16/bf16).

```bash
export FLASH_ATTENTION_DISABLE_FP8=FALSE
# plus same trim flags as d64/d128 build; see /tmp/fa3_hopper_install_fp8.log
```

## Reporting

TFLOPS uses the same FLOP count as FP16/bf16. **HW efficiency** divides by
**2 x H800_PCIE_PEAK_F16** (~3026 TFLOPS) so FP8 util is comparable to FP16 util
(same achieved TFLOPS -> ~half the FP8 efficiency %).

## Choreo

Not implemented. FA3 only.

## Run

```bash
bash baselines/bench.sh
bash compare_all.sh
```
