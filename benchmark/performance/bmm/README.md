# bmm -- Batched Matrix Multiply Benchmarks

Batched matrix multiplication with dynamic shapes. All kernels target SM90a (Hopper).

## Variants

| Variant | Data Type | Tile (MxNxK) | Status |
|---------|-----------|-------------|--------|
| bmm_fp16_dynamic | BF16 | 64x64x64 | PASS |
| bmm_fp8_dynamic | FP8 (E4M3) | 64x64x32 | PASS |

Default problem: B=batch, M=127, N=128, K=128.

## Build & Run

```bash
./choreo -gs -t cute -arch=sm_90a <file>.co -o /tmp/out.cute.result
bash /tmp/out.cute.result --execute
```
