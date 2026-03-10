# Choreo Matmul Case Guide

## Table of Contents
- 1. Scope and file naming
- 2. Core Choreo matmul syntax skeleton
- 3. Precision options
- 4. Architecture paths and hard constraints
- 5. Optimization options
- 6. Host-side harness patterns
- 7. Fast decision matrix
- 8. Generation checklist

## 1. Scope and file naming

Target directory:
- `benchmark/performance/matmul`

Common naming pattern:
- `matmul_<precision>_<mode>_<arch>.co`

Examples:
- `matmul_f16_dyn_sm90.co`
- `matmul_e4m3_dyn_persis_swizzle_sm90.co`
- `matmul_f16_dyn_sm90_warpspec_1p2c_persis_sta.co`
- `matmul_trans_f16_dyn_sm90.co`

Naming hints:
- `dyn`: non-persistent tiled traversal.
- `persis_sta`: persistent static tile distribution via `NUM_SMS`.
- `swizzle` / `hilbert` / `colmajor`: schedule variants.
- `warpspec_1pNc`: one producer and N consumers.
- `trans`: transposed output layout.

## 2. Core Choreo matmul syntax skeleton

Kernel skeleton (SM90-style WGMMA form):

```co
__co__ void matmul(global f16 [M, K] lhs,
                   global f16 [N, K] rhs,
                   global f16 [M, N] output) {
  parallel {block_m, block_n} by [cdiv(M, MATMUL_WARP_M), cdiv(N, MATMUL_WARP_N)] : block {
    shared f16 [MATMUL_WARP_M, MATMUL_TILE_K] lhs_load_s;
    shared f16 [MATMUL_WARP_N, MATMUL_TILE_K] rhs_load_s;
    mc = mma.fill.f16 0.0f;

    foreach {iv_k} in [cdiv(K, MATMUL_TILE_K)] {
      tma.copy.swiz<MATMUL_SWIZ> lhs.subspan(MATMUL_WARP_M, MATMUL_TILE_K).at(block_m, iv_k) => lhs_load_s;
      tma.copy.swiz<MATMUL_SWIZ> rhs.chunkat(block_n, iv_k) => rhs_load_s;

      foreach {iv_warp} in [cdiv(MATMUL_TILE_K, MATMUL_WARP_K)] {
        parallel p by 1 : group-4 {
          ma = mma.load.swiz<MATMUL_SWIZ> lhs_load_s.chunkat(_, iv_warp);
          mb = mma.load.swiz<MATMUL_SWIZ> rhs_load_s.chunkat(_, iv_warp);
          mma.row.row mc, ma, mb;
        }
      }
    }

    shared f16 [MATMUL_WARP_M, MATMUL_WARP_N] output_s;
    mma.store mc, output_s;
    tma.copy output_s => output.subspan(MATMUL_WARP_M, MATMUL_WARP_N).at(block_m, block_n);
  }
}
```

SM86 baseline syntax deltas:
- Use `dma.copy.async ... => shared` and `wait ...` instead of `tma.copy.swiz`.
- Use per-warp `parallel {warp_m, warp_n} ... : group` and `mma.load` without `.swiz`.

## 3. Precision options

### 3.1 f16 path
- Kernel types: `global f16` input/output.
- Host allocation: `half*` for A/B/C.
- Typical accumulators:
  - `mma.fill.f16` in most dynamic/persistent/warpspec files.
  - `mma.fill.f32` appears in some `mwg_impl` variants.

### 3.2 f8_e4m3 path
- Kernel input types: `global f8_e4m3`.
- Output type: `global f16`.
- Host allocation: `__nv_fp8_e4m3*` for A/B and `half*` for C.
- Usual tolerance is looser than f16 (often around `0.08f`).

## 4. Architecture paths and hard constraints

### 4.1 SM90 (WGMMA + TMA)

Typical fixed constraints in examples:
- `MATMUL_WARP_M == 64`
- `MATMUL_WARP_N % 8 == 0` and `MATMUL_WARP_N <= 256`

f16-specific common constraints:
- `MATMUL_WARP_K == 16`
- `MATMUL_SWIZ == 2 * MATMUL_TILE_K`
- `MATMUL_SWIZ in {32,64,128}`
- Baseline often enforces `MATMUL_TILE_M == MATMUL_WARP_M`

e4m3-specific common constraints:
- `MATMUL_WARP_K == 32`
- `MATMUL_SWIZ == MATMUL_TILE_K`
- `MATMUL_SWIZ in {32,64,128}`
- Baseline often enforces `MATMUL_TILE_M == MATMUL_WARP_M`

### 4.2 SM86 (mma.sync + dma)

Common constraints in examples:
- `MATMUL_MMA_M/N/K = 16`
- `MATMUL_TILE_M % MATMUL_MMA_M == 0`
- `MATMUL_TILE_N % MATMUL_MMA_N == 0`
- `MATMUL_TILE_K % MATMUL_MMA_K == 0`
- Warp-count cap:
  - `(MATMUL_TILE_M / MATMUL_MMA_M) * (MATMUL_TILE_N / MATMUL_MMA_N) <= 8`

## 5. Optimization options

### 5.1 Baseline dynamic tiling
Seed files:
- `matmul_f16_dyn_sm90.co`
- `matmul_e4m3_dynamic_sm90.co`
- `matmul_f16_dyn_sm86.co`

Traits:
- 2D tile launch over M/N.
- No persistent schedule arrays.

### 5.2 Persistent static (`persis_sta`)
Seed files:
- `matmul_f16_dyn_persis_sta_sm90.co`
- `matmul_f16_dyn_persis_sta_sm86.co`
- `matmul_e4m3_dyn_persis_sta_sm90.co`

Traits:
- Define `NUM_SMS`.
- `parallel block_id by NUM_SMS : block`.
- Compute `tile_id = tile_iter # block_id`.

### 5.3 Schedule variants: `swizzle` / `hilbert` / `colmajor`
Seed files:
- `matmul_f16_dyn_persis_swizzle_sm90.co`
- `matmul_f16_dyn_persis_hilbert_sm90.co`
- `matmul_f16_dyn_persis_colmajor_sm90.co`
- `matmul_e4m3_dyn_persis_swizzle_sm90.co`

Traits:
- Host builds tile traversal arrays or mapping logic.
- Kernel consumes schedule (directly or through deterministic mapping).

### 5.4 Warp specialization (`warpspec_1p1c`, `1p2c`, `1p3c`)
Seed files:
- `matmul_f16_dyn_sm90_warpspec_1p1c.co`
- `matmul_f16_dyn_sm90_warpspec_1p2c.co`
- `matmul_f16_dyn_sm90_warpspec_1p3c.co`
- Persistent versions: `*_warpspec_*_persis_sta.co`

Traits:
- `shared event full[STAGES], empty[STAGES]`.
- `parallel p1 by (1 + consumers) : group-4`.
- `inthreads.async (p1 == 0)` producer uses TMA load.
- `inthreads.async (p1 > 0)` consumers do WGMMA compute.
- Typical stage presets:
  - `1p1c`: `MATMUL_STAGES 4`, `MATMUL_TILE_M 64`
  - `1p2c`: `MATMUL_STAGES 3`, `MATMUL_TILE_M 128`
  - `1p3c`: `MATMUL_STAGES 2`, `MATMUL_TILE_M 192`

### 5.5 Multi-workgroup (`mwg`)
Seed files:
- `matmul_f16_dynamic_mwg_sm90.co`
- `matmul_f16_dynamic_mwg_impl_1.co`
- `matmul_e4m3_dynamic_mwg_sm90.co`

Traits:
- Use `parallel p1 by 2 : group-4` split for two sub-tiles/workgroups.
- Often uses larger `MATMUL_TILE_M` with `MATMUL_WARP_N = 64`.

### 5.6 Transposed output
Seed files:
- `matmul_trans_f16_dyn_sm90.co`
- `matmul_trans_f16_dyn_sm86.co`

Traits:
- Output tensor shape changes to `[N, M]`.
- Use `mma.store.transp` in SM90 transposed variant.

### 5.7 Single-tile stmatrix perf microbenchmark
Seed file:
- `matmul_f16_single_tile_stmatrix_perf_sm90.co`

Traits:
- Fixed tiny shape (64x128x16-like setup).
- Targeted microbenchmark style, not general tiling sweep.

## 6. Host-side harness patterns

Keep these patterns stable unless requested:
- CLI flags: `--disable-timing`, `--skip-verify`, optional `--flops=`.
- Env toggles: `CHOREO_DISABLE_TIMING`, `CHOREO_SKIP_VERIFY`, `CHOREO_TIMING_WARMUP`, `CHOREO_TIMING_REPEAT`.
- Timing: `choreo::timing([&]() { ...; cudaDeviceSynchronize(); }, topt)`.
- Verification:
  - f16 usually tolerance `0.05f`.
  - f8_e4m3 usually tolerance `0.08f`.

## 7. Fast decision matrix

Choose seed by request:
- “SM90 + f16 + baseline”: `matmul_f16_dyn_sm90.co`
- “SM90 + f16 + persistent”: `matmul_f16_dyn_persis_sta_sm90.co`
- “SM90 + f16 + warpspec 1p2c”: `matmul_f16_dyn_sm90_warpspec_1p2c.co`
- “SM90 + e4m3 + baseline”: `matmul_e4m3_dynamic_sm90.co`
- “SM90 + e4m3 + swizzle”: `matmul_e4m3_dyn_persis_swizzle_sm90.co`
- “SM86 + f16 + baseline”: `matmul_f16_dyn_sm86.co`
- “Need transposed output”: `matmul_trans_f16_dyn_sm90.co` or `matmul_trans_f16_dyn_sm86.co`

## 8. Generation checklist

1. Pick closest seed file by precision + arch + optimization.
2. Rename file to follow existing naming convention.
3. Update only required macros first (`WARP_*`, `TILE_*`, `SWIZ`, `NUM_SMS`, `STAGES`).
4. Keep or adapt compile-time constraints to match new macros.
5. Ensure kernel tensor types match precision choice.
6. Ensure host allocation/copy types match kernel types.
7. Keep timing + verification flow intact unless user asks otherwise.
8. If schedule-based optimization is selected, include host schedule builder and device schedule buffers.
9. Sanity-check expected output layout (`[M,N]` vs `[N,M]`).
