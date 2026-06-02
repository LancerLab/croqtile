# Flash MHA Forward (BHSD) AI-Tune Results

AI-guided optimization of Choreo flash attention MHA forward on H100 (sm_90a),
DIM=64, causal decoder self-attention. Primary benchmark: B=1 H=32 SEQ=8192.

Source branch: `ai-tune/2026-06-01/flash_mha_fwd` (81 iterations, squashed commit).

## Shipped Winner

| Iter | Kernel | SEQ=8192 TFLOPS | Gain vs baseline | Key optimization |
|------|--------|-----------------|------------------|------------------|
| iter038 + LB2 | `flash_mha_fwd_aitune_2026-06-01_iter038.co` | 258.5 (median) | +39% (186 -> 258.5) | 1p1c BM64, merged events, skip-mask, reverse grid, `--use_fast_math`, `__launch_bounds__(256,2)` |

Progressive KEEP stages on the experiment branch:

| Stage | TFLOPS (8192) | Optimization |
|-------|---------------|--------------|
| baseline | 186 | 1p2c TMA, BM128 BN128 STAGES=2 |
| iter015 | 202 | `--use_fast_math` |
| iter017 | 218 | Skip causal mask on fully-unmasked tiles |
| iter029 | 223 | Reverse grid ordering (load balance) |
| iter030 | 244 | Merge K/V into single barrier pair |
| iter038 | 255 | 1p1c architecture, BLOCK_M=64 |
| +LB2 | 258.5 | `__launch_bounds__(256,2)` scheduling hint |

## Build and Run

```bash
cd benchmark/performance/flash_atten/flash_mha_fwd_bhsd/choreo
bash bench.sh --kernel flash_mha_fwd_aitune_2026-06-01_iter038.co --gpu 1
```

`bench.sh` passes `--use-fast-math` to `./choreo`, which adds `--use_fast_math` to
the generated nvcc command. Expect `Test Passed` after sampled numerical checks.

Profile with Nsight Compute:

```bash
EXTRA_NVCC_FLAGS="--use_fast_math" bash bench.sh \
  --kernel flash_mha_fwd_aitune_2026-06-01_iter038.co --gpu 1 --profile 038
```

## Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `CHOREO_TIMING_WARMUP` | 10 | Timing warmup iterations |
| `CHOREO_TIMING_REPEAT` | 50 | Timing measurement iterations |
| `CHOREO_DISABLE_TIMING` | off | Set to `1` to skip timing |
| `CHOREO_SKIP_VERIFY` | off | Set to `1` to skip verification |
| `MHA_VERIFY_SAMPLES` | 512 | Sampled output elements to verify |
| `EXTRA_NVCC_FLAGS` | (none) | Extra nvcc flags appended to generated script |
| `./choreo --use-fast-math` | off | Pass `--use_fast_math` to nvcc (bench.sh enables this) |

## Key Insights

1. **Macro beats micro**: Architecture changes (1p1c, merged barriers, grid
   ordering, skip-mask) delivered all large gains. 40+ CUDA micro-opts regressed.
2. **WGMMA serialization is structural**: C7510/C7518 from barrier calls,
   AllReduce, and TMA helpers crossing function boundaries. Not fixable in user
   CUDA without compiler/toolchain changes.
3. **3-stage pipelining always loses**: 104KB smem reduces L1 effective capacity;
   tested 6 times, regressed every time.
4. **Causal branch helps scheduling**: Non-causal ceiling is ~253 TFLOPS vs 258
   causal; removing the branch triggers C7514 accumulator-read serialization.
5. **Compiler scheduling is near-optimal**: Manual WGMMA pipelining, softmax
   fusion, and register tricks consistently regressed baseline by 3-18%.

## Compiler Suggestions

1. **Inline TMA/barrier paths**: Emit `mbarrier` PTX inline instead of
   `cuda::barrier` C++ API calls to eliminate C7510 WGMMA serialization.
2. **Fix dma.copy rank check with sqz()**: Pre-declared shared + `.sqz()` subviews
   fail rank check (4 != 2); blocks non-warpspec kernels.
3. **Fix auto-allocated shared aliasing**: Multiple `=> shared` buffers in one
   scope collapse into a single 16KB allocation instead of separate buffers.
4. **Emit `__launch_bounds__(256,2)` by default** for 256-thread 1p1c kernels
   when smem + register analysis shows 2 blocks/SM is achievable.
5. **Respect `--use_fast_math` in softmax**: Avoid injecting `__fmaf_rn` that
   overrides fast exp2f paths (iter063: -26% regression).

## Files

| File | Purpose |
|------|---------|
| `flash_mha_fwd_aitune_2026-06-01_iter038.co` | Winning kernel |
| `mha_helper.hpp` | Host data prep, verify, timing, TFLOPS |
| `../baselines/` | TileLang + FlashAttention baselines (ml-hopper env) |
| `bench.sh` | Compile, run, optional ncu profile |
| `results.tsv` | Full 81-iteration experiment log |
