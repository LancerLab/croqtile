# Operator Gallery

A curated index of example `.co` files in the repository, organized by operator pattern and language feature. Use these as references when writing your own Croqtile programs.

## Element-Wise Operations

| File | Description |
|------|-------------|
| `tests/gpu/end2end/add.co` | Basic element-wise addition |
| `tests/gpu/end2end/add-shared.co` | Addition with shared memory staging |
| `tests/gpu/end2end/add-local.co` | Addition with local memory staging |
| `tests/gpu/end2end/copy.co` | Simple data copy |
| `tests/gpu/end2end/pad.co` | Data padding |
| `tests/gpu/end2end/transpose.co` | Data transposition |
| `samples/factor/elementwise_add.co` | Basic elementwise with async DMA |
| `samples/factor/elementwise_multibuffers.co` | Double-buffered elementwise |

## Matrix Multiplication (MMA)

| File | Description |
|------|-------------|
| `tests/gpu/end2end/mma.co` | Basic MMA matmul (f16) |
| `tests/gpu/end2end/mma_v1.co` | MMA variant 1 |
| `tests/gpu/end2end/mma_v2.co` | MMA variant 2 |
| `tests/gpu/end2end/mma_fp8.co` | FP8 MMA |
| `tests/gpu/end2end/mma_fusion_scalar.co` | MMA with scalar fusion |
| `tests/gpu/end2end/matmul.co` | Tiled matmul without MMA |
| `tests/gpu/end2end/matmul-dma.co` | Matmul with explicit DMA |

## WMMA and PTX MMA

| File | Description |
|------|-------------|
| `tests/gpu/end2end/wmma/` | WMMA instruction matrix (16 variants: types x shapes) |
| `tests/gpu/end2end/ptx_mma/` | mma.sync instruction matrix (31 variants) |
| `tests/gpu/end2end/wmma_rr.co` | WMMA row-row |
| `tests/gpu/end2end/wmma_fp8_rr.co` | WMMA FP8 row-row |

## TMA Operations

| File | Description |
|------|-------------|
| `tests/gpu/end2end/tma.co` | Basic TMA copy |
| `tests/gpu/end2end/tma_v1.co` | TMA variant 1 |
| `tests/gpu/end2end/tma_v2.co` | TMA variant 2 |
| `tests/gpu/end2end/tma_fp8.co` | TMA with FP8 data |
| `tests/gpu/end2end/tma_fp6_fp4.co` | TMA with FP6/FP4 data |
| `tests/gpu/end2end/tma_with_global_ref.co` | TMA with global reference |

## Advanced GEMM (SM90 / Hopper)

| File | Description |
|------|-------------|
| `tests/gpu/end2end/warpspec.co` | Warp-specialized GEMM |
| `tests/gpu/end2end/matmul/matmul_f16_dyn_sm90.co` | Dynamic f16 matmul (SM90) |
| `tests/gpu/end2end/matmul/matmul_f16_dyn_persis_*.co` | Persistent scheduling variants |
| `tests/gpu/end2end/matmul/matmul_f16_dyn_sm90_warpspec_*.co` | Warpspec variants (1p1c, 1p3c) |
| `tests/gpu/end2end/matmul/matmul_e4m3_dynamic_sm90.co` | FP8 e4m3 dynamic matmul |
| `tests/gpu/end2end/matmul/blockscale_gemm_*.co` | Block-scaled GEMM |

## Sparse GEMM

| File | Description |
|------|-------------|
| `tests/gpu/end2end/gemm_sp/` | Sparse GEMM variants (6 files) |
| `benchmark/performance/gemm_sp/` | Sparse GEMM benchmarks (70 files) |

## Async Copy Patterns

| File | Description |
|------|-------------|
| `tests/gpu/end2end/async_copy.co` | Basic async copy |
| `tests/gpu/end2end/async_copy_fp8.co` | Async copy with FP8 |

## Numeric Types and Conversions

| File | Description |
|------|-------------|
| `tests/gpu/end2end/float_types.co` | Float type coverage |
| `tests/gpu/end2end/integral_types.co` | Integer type coverage |
| `tests/gpu/end2end/add_fp8_e4m3_to_f16.co` | FP8-to-F16 conversion |
| `tests/gpu/end2end/numerics_core6.co` | Core numeric operations |
| `tests/gpu/end2end/numerics_transcendental.co` | Transcendental functions |

## View Operations

| File | Description |
|------|-------------|
| `tests/infer/view_from.co` | `view(...).from(...)` examples |
| `tests/infer/subspan.co` | `subspan(...).at(...)` examples |
| `tests/infer/span_as.co` | `span_as(...)` reshape examples |
| `tests/parse/stride.co` | `subspan.step.at` strided access |
| `tests/parse/chunkat.co` | `chunkat` parsing examples |

## Samples by Target

### CUDA/CuTe
| File | Description |
|------|-------------|
| `samples/cuda/sgemm/sgemm-basic.co` | Basic SGEMM with shared memory |
| `samples/cuda/sgemm/sgemm-half.co` | Half-precision SGEMM |
| `samples/cuda/layernorm.co` | LayerNorm kernel |
| `samples/cuda/l2loss.co` | L2 loss reduction |

### Factor Target
| File | Description |
|------|-------------|
| `samples/factor/matmul.co` | Basic matmul |
| `samples/factor/matmul-tile-*.co` | Progressive tiling variants (0-7) |
| `samples/factor/dyn_matmul.co` | Dynamic shape matmul |
| `samples/factor/activation-relu-*.co` | ReLU activation (f16/f32/bf16) |

## Memory Management

| File | Description |
|------|-------------|
| `tests/gpu/end2end/memreuse-static.co` | Static memory reuse |
| `tests/gpu/end2end/memreuse-dynamic.co` | Dynamic memory reuse |

## Shape Inference Benchmarks

The `benchmark/shapeinfer/` directory contains 311 `.co` files across 13 operator families (matmul, softmax, relu, conv2d, etc.), each testing shape inference with model-inspired tensor dimensions (BERT, ViT, ResNet, etc.).
