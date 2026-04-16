# GCU Acore Library Integration Guide

This guide covers how to use the **acore** library from Choreo kernels targeting
the GCU (topscc) backend. Acore provides high-performance implementations of
common operations (matmul, conv2d, elementwise, activations, reductions) that
run on the GCU's VACC hardware.

## Prerequisites

- GCU3 (gcu300) architecture or later
- Acore library installed: run `make setup-gcu-acore` from the repo root
- Build Choreo with `make` (CMake detects acore and defines `__CHOREO_GCU_ACORE_DIR__`)

## Quick Start

### Simple Elementwise (Binary / Unary / Activation)

Binary, unary, and activation acore ops have clean interfaces out of the box.
Use the `call acore::func(...)` syntax directly:

```
// Element-wise add with alpha scaling
__co__ f32 [N] vec_add(f32 [N] a, f32 [N] b) {
  f32[a.span] out;
  parallel p by 4 {
    foreach x in N/256 {
      a_tile = dma.copy.async a.chunkat(p, x) => local;
      b_tile = dma.copy.async b.chunkat(p, x) => local;
      wait a_tile, b_tile;

      local f32[a_tile.span] l1_out;
      call acore::add(l1_out, a_tile.data, b_tile.data, |a_tile.span|);

      out_store = dma.copy.async l1_out => out.chunkat(p, x);
      wait out_store;
    }
  }
  return out;
}
```

Available operations:

| Category | Functions |
|----------|-----------|
| **Binary** | `add`, `sub`, `mul`, `div`, `max`, `min`, `add_relu`, `pow`, `atan2`, `fmod`, `gt`, `ge`, `lt`, `le`, `eq`, `ne`, `logical_and`, `logical_or` |
| **Unary** | `abs`, `neg`, `sqrt`, `rsqrt`, `exp`, `log`, `sin`, `cos`, `tanh`, `ceil`, `floor`, `erf`, `reciprocal` |
| **Activation** | `relu`, `gelu`, `selu`, `silu`, `sigmoid`, `softplus`, `mish`, `hard_swish`, `leaky_relu`, `elu` |
| **Reduce** | `reduce_sum`, `reduce_max`, `reduce_min`, `reduce_mean`, `reduce_argmax`, `reduce_argmin` |

### Matmul with Raw acore Interface

For direct control, use `call acore::matmul<M, format>(...)`:

```
// C[128,128] = A[128,32] * B[32,128] using tiled matmul
__co__ f32 [128, 128] raw_matmul(f16 [128, 32] A, f16 [32, 128] B) {
  f32[128, 128] output;
  parallel p by 4 {
    with tilings = {m_tile, k_tile, n_tile} in [128/32, 32/32, 128/128] {
      foreach m_tile, n_tile {
        local f32[32, 128] l1_out;
        local s32[2048] workspace;
        local f16[128] bias_buf;

        foreach k_tile {
          lhs = dma.copy A.chunkat(m_tile, k_tile) => local;
          rhs = dma.copy B.chunkat(k_tile, n_tile) => local;

          // acore::matmul<M=32, format=MK_KN(0)>
          // Args: dst, lhs, rhs, bias, workspace, K, N,
          //       acc_flag, store_flag, bias_flag, vab_off
          call acore::matmul<32, 0>(
            l1_out, lhs.data, rhs.data, bias_buf,
            workspace, 32, 128, 0, 1, 0, 0);
        }
        dma.copy l1_out => output.chunkat(m_tile, n_tile);
      }
    }
  }
  return output;
}
```

### Matmul with Simplified Wrapper

The raw `acore::matmul` requires many hardware scheduling flags. The
`choreo::acore_gemm` wrappers (from `runtime/gcu/acore_gemm.h`) hide these:

```
#include "gcu/acore.h"

// Same matmul, but using the simplified wrapper
__co__ f32 [128, 128] simple_matmul(f16 [128, 32] A, f16 [32, 128] B) {
  f32[128, 128] output;
  parallel p by 4 {
    with tilings = {m_tile, k_tile, n_tile} in [128/32, 32/32, 128/128] {
      foreach m_tile, n_tile {
        local f32[32, 128] l1_out;
        local s32[2048] workspace;

        foreach k_tile {
          lhs = dma.copy A.chunkat(m_tile, k_tile) => local;
          rhs = dma.copy B.chunkat(k_tile, n_tile) => local;

          call choreo::acore_gemm<32>(
            l1_out, lhs.data, rhs.data, 32, 128, workspace);
        }
        dma.copy l1_out => output.chunkat(m_tile, n_tile);
      }
    }
  }
  return output;
}
```

### Matmul with `__lib_gemm` Builtin (Recommended)

The `__lib_gemm` builtin provides the simplest GEMM interface. It uses standard
arguments and the compiler handles all hardware-specific lowering:

```
__co__ void tiled_gemm(f16 [128, 32] A, f16 [32, 128] B) {
  parallel p by 4 {
    with tilings = {m_tile, k_tile, n_tile} in [128/32, 32/32, 128/128] {
      foreach m_tile, n_tile {
        local f16[32, 128] out;
        foreach k_tile {
          lhs = dma.copy A.chunkat(m_tile, k_tile) => local;
          rhs = dma.copy B.chunkat(k_tile, n_tile) => local;
          __lib_gemm(out, lhs.data, rhs.data, 32, 128);
        }
      }
    }
  }
}
```

**Syntax:**
```
__lib_gemm(out, A, B, K, N);           // without bias
__lib_gemm(out, A, B, bias, K, N);     // with bias
```

On GCU, target-library lowering is **enabled by default**. Use
`--use-target-lib=false` to force the general fallback. Explicit `-utl`
overrides the target's default.

When target-lib is enabled, the compiler:
1. Statically evaluates M from the output buffer shape
2. Checks if M matches a supported acore tile size (1, 16, 32, 49, 64, 96, 128, 256, 512, 1024) or is 64-aligned for dynamic-M
3. Selects the best `acore::matmul` overload and emits all hardware flags automatically
4. If M is unsupported, warns and falls back to a general runtime implementation

With `--use-target-lib=false`, `__lib_gemm` always generates a general
template-based matmul via `choreo::lib_gemm_general` (no acore dependency).

## Acore Interface Reference

### Matmul

#### Raw Interface

```cpp
// Compile-time M (fixed tile height)
acore::matmul<M, format, act>(out, lhs, rhs, bias, workspace,
    K, N, acc_flag, store_flag, bias_flag, vab_off [, launch_times])

// Runtime M (M must be 64-aligned)
acore::matmul<format>(out, lhs, rhs, bias, workspace,
    M, K, N, acc_flag, store_flag, bias_flag, vab_off [, launch_times])
```

**Template parameters:**
- `M` (int): Tile height of LHS and output. Compile-time constant.
- `format` (MatmulFormat): `MK_KN`=0 (both row-major) or `MK_NK`=1 (RHS column-major). Default: `MK_KN`.
- `act` (ActMode): `None`=0, `ReLU`=1, `GeLU`=2. Default: `None`.

**Arguments (user-facing):**
- `out`: Output buffer [M, N]
- `lhs`: Left operand [M, K]
- `rhs`: Right operand [K, N] (or [N, K] if `MK_NK`)
- `bias`: Bias vector [N] (pass nullptr if unused, set `bias_flag=0`)
- `K`, `N`: Matrix dimensions

**Arguments (hardware scheduling):**
- `workspace`: Scratch buffer (int*), ~8KB. Required for TAR register management.
- `acc_flag`: 0 = initialize VACCs; 1 = accumulate into existing VACCs (for K-tiling)
- `store_flag`: 0 = keep in VACCs; 1 = store results to output buffer
- `bias_flag`: 0 = no bias; 1 = add bias
- `vab_off`: VACC bank offset (0 unless sharing VACC space with other ops)
- `launch_times`: 0 = initialize workspace; non-zero = skip workspace init (same K,N)

**VACC usage**: M × N / 32 VACCs per call (4096 total available).

#### Simplified Wrappers (`runtime/gcu/acore_gemm.h`)

| Wrapper | Description |
|---------|-------------|
| `choreo::acore_gemm<M, fmt>(out, A, B, K, N, ws)` | C = A × B |
| `choreo::acore_gemm_bias<M, fmt>(out, A, B, bias, K, N, ws)` | C = A × B + bias |
| `choreo::acore_gemm_act<M, fmt, act>(out, A, B, K, N, ws)` | C = act(A × B) |
| `choreo::acore_addmm<M, fmt, bf>(out, A, B, bias, K, N, ws, α, β)` | C = α(A×B) + β·bias |
| `choreo::acore_gemm_scale<M, fmt>(out, A, B, K, N, ws, α)` | C = α(A × B) |
| `choreo::acore_gemm_chained_first/mid/last<M>` | K-accumulation loop pattern |

Workspace (`ws`): declare as `local s32[2048] workspace;` in Choreo code.
The 2048 words (~8KB) is sufficient for most tile configurations. The workspace
handles TAR register management internally.

### Conv2d

#### Raw Interface

```cpp
// Basic conv2d
acore::conv2d<ho_step, wo_step, bias_flag>(
    out, input, weight, bias,
    n, hi, wi, ci, r, s, co, ho, wo,
    stride_h, stride_w, dilation_h, dilation_w,
    init_vacc_flag, wb_vacc_flag, co_offset)

// With full tile control
acore::conv2d<ho_step, wo_step, ci_step, co_step>(
    out, input, weight, bias,
    n, hi, wi, ci, r, s, co, ho, wo,
    stride_h, stride_w, dilation_h, dilation_w,
    init_vacc_flag, wb_vacc_flag, co_offset [, vab_offset])
```

**Template parameters:**
- `ho_step`, `wo_step`: Output tile height/width steps
- `ci_step`, `co_step`: Channel tile steps (input/output)
- `bias_flag` (on basic overload): 0 = no bias, 1 = add bias

**Arguments (hardware scheduling):**
- `init_vacc_flag`: 1 = initialize accumulators; 0 = accumulate
- `wb_vacc_flag`: 1 = write back results; 0 = keep in VACCs
- `co_offset` / `vab_offset`: Channel offset / VACC bank offset

#### Simplified Wrappers (`runtime/gcu/acore_conv2d.h`)

Use `choreo::Conv2dParams` to bundle geometry:

```cpp
choreo::Conv2dParams p = {
  .n=1, .hi=28, .wi=28, .ci=64,
  .r=3, .s=3, .co=128,
  .ho=28, .wo=28,
  .stride_h=1, .stride_w=1,
  .dilation_h=1, .dilation_w=1
};
```

| Wrapper | Description |
|---------|-------------|
| `choreo::acore_conv2d<ho,wo,bias>(out, in, wt, bias, p)` | Basic conv2d |
| `choreo::acore_conv2d_tiled<ho,wo,ci,co>(out, in, wt, bias, p)` | Full tile control |
| `choreo::acore_conv2d_quant<ho,wo,ci,co>(out, in, wt, scale, bias, p)` | Int8 quantized |
| `choreo::acore_conv2d_group<ho,wo,kci,kco,g>(in, wt, out, bias, p)` | Grouped conv |

## Hardware Constraints

All acore operations have these fundamental requirements:

| Constraint | Value | Notes |
|-----------|-------|-------|
| **Buffer storage** | L1 (local) only | DMA data to local before acore calls |
| **Pointer alignment** | 256 bytes | Choreo's `__valigned__` ensures this |
| **Max SIPs** | 48 (4×12) | `parallel p by 48` or nested |
| **L1 per SIP** | ~1.5 MB | Budget carefully for tiles + workspace |
| **VACC registers** | 4096 per thread | matmul uses M×N/32 per call |

### Matmul M/K/N Alignment

The alignment requirements depend on the data type and format:

| Type | M | K | N |
|------|---|---|---|
| f16/bf16 MK_KN | 32-aligned | 32-aligned | 128-aligned |
| f16/bf16 MK_NK | 32-aligned | 128-aligned | 32-aligned |
| f32 MK_KN | 16-aligned | 16-aligned | 32-aligned |
| int8 | 32-aligned | 64-aligned | 128-aligned |

## Validation

The Choreo compiler performs the following checks on `call acore::` statements:

1. **Known operation**: the function name must be a recognized acore operation
2. **L1 buffer enforcement**: all spanned (buffer) arguments must be `Storage::LOCAL`
3. **Argument count**: per-category validation of argument counts
4. **Auto-include**: `<common/acore_op.h>` is automatically included when any
   `acore::` call is detected (no need to manually `#include` it)

## File Organization

| Path | Purpose |
|------|---------|
| `runtime/gcu/acore.h` | Umbrella header for all Choreo acore wrappers |
| `runtime/gcu/acore_gemm.h` | Simplified GEMM/matmul wrappers |
| `runtime/gcu/acore_conv2d.h` | Simplified conv2d wrappers |
| `lib/Target/GCU/acore_check.hpp` | Compiler validation for acore calls |
| `extern/include/common/acore_op.h` | Raw acore library headers |
| `extern/lib/libacoreop.bc` | Acore device library (linked at compile time) |
