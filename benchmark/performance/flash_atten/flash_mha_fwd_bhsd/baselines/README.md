# Flash attention external baselines

Compare Choreo flash MHA forward kernels against:

| Backend | Source | Layout |
|---------|--------|--------|
| **tilelang** | `/home/fem/dev/tilelang/examples/flash_attention/example_mha_fwd_bhsd.py` | BHSD |
| **flash_attn** | dao-ailab `flash_attn` (ml-hopper venv) | BSHD native; BHSD transpose in timed loop |

## Fairness

Aligned with `choreo/mha_helper.hpp` and `choreo/bench.sh`:

- **Shapes**: decoder causal self-attention, `DIM=64`, `q_seq == kv_seq`
  - B=2 H=32 SEQ in {512, 1024, 2048, 4096}
  - B=1 H=32 SEQ=8192
- **Layout**: Choreo and TileLang use `[batch, heads, seq, dim]` (BHSD).
  FlashAttention uses `[batch, seq, heads, dim]` (BSHD); the baseline times
  `transpose + kernel` so comparisons reflect a BHSD-first integration.
- **Causal**: full lower-triangular mask (`past_len = kv_seq - q_seq = 0`)
- **FLOPS**: `4 * B * H * q_seq * kv_seq * dim`, multiplied by `0.5` when causal
- **Timing**: CUDA events, `warmup=10`, `repeat=50`, `cudaDeviceSynchronize` (no L2 cache flush between reps; unlike TileLang `profiler.do_bench`)
- **Data**: uniform random Q/K in [-0.5, 0.5], V in [-1, 1], seed=42

### Math notes

- **Choreo / TileLang** use `exp2` softmax with `scale = (1/sqrt(dim)) * log2(e)`.
- **FlashAttention** uses standard `exp` softmax with `softmax_scale = 1/sqrt(dim)` (equivalent math, different numerics). Verification uses the appropriate reference per backend.

### TileLang kernel config

Static Hopper config from TileLang regression: `block_M=128`, `block_N=128`, `num_stages=2`, `threads=256`. Choreo kernels may use different tiles (e.g. BM=64); each backend uses a fixed tuned config, not cross-forced tile sizes.

## Requirements

```bash
source /home/fem/.env/ml-hopper/bin/activate
# flash_attn 2.x, tilelang, torch with CUDA
```

TileLang example path (default): `/home/fem/dev/tilelang/examples/flash_attention`

## Usage

```bash
cd benchmark/performance/flash_atten/flash_mha_fwd_bhsd/baselines
chmod +x bench_baselines.sh
./bench_baselines.sh
./bench_baselines.sh --backend flash_attn
./bench_baselines.sh --backend tilelang --verify
./bench_baselines.sh --gpu 0 --tsv results.tsv
```

Compare with Choreo:

```bash
cd ../choreo
./bench.sh --kernel flash_mha_fwd_aitune_2026-06-01_iter038.co --no-verify
```

Optional timing overrides (both choreo and baselines):

```bash
export CHOREO_TIMING_WARMUP=10
export CHOREO_TIMING_REPEAT=50
```
