# gemv -- GEMV Benchmarks

General matrix-vector multiply kernels. All kernels target SM90a (Hopper).

## Variants

| Variant | Data Type | Tile (MxK) | Status |
|---------|-----------|-----------|--------|
| gemv_fp8_dynamic | FP8 (E4M3) | 256x128 | PASS |

Default problem: M=4096, K=4096.

## Build & Run

```bash
./choreo -gs -t cute -arch=sm_90a <file>.co -o /tmp/out.cute.result
bash /tmp/out.cute.result --execute
```
