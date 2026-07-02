# CoIR Dialect Gaps -- Known Issues for Future Phases

This document tracks known gaps between Choreo's internal data structures and
the CoIR MLIR dialect, identified during the elementwise prototype work.

## 1. Tensor Strides -- RESOLVED

**Choreo**: `SpannedType` carries both shape and strides (via `GenDenseStrides()`).

**CoIR**: `!coir.tensor<shape x elemType, memspace, strides: [...]>` now supports
optional strides. When strides are absent, dense contiguous (row-major) is assumed.

```
!coir.tensor<128x64xf16, global, strides: [64, 1]>   // explicit strides
!coir.tensor<128x64xf16, global>                      // dense contiguous
```

**ASTCoIRGen** preserves strides from `SpannedType::GetStrides()` and omits them
only when they match dense row-major layout. Both `EmitCUDA` and target emitters
use stride-based linearization for element access: `offset = sum(idx[i] * stride[i])`.

## 2. Parallel Level Mismatch

**Choreo** defines these parallel levels:
- `THREAD`, `GROUP` (warp), `GROUPx4` (warpgroup), `BLOCK`, `CLUSTER`, `DEVICE`
- `TERM`, `SEQ`, `NONE`, `UNKNOWN`

**CoIR** currently supports only:
- `Block`, `Warpgroup`, `Warp`, `Thread`

**Missing levels**:
- `CLUSTER` -- SM90+ thread-block cluster; needed for distributed shared memory
- `DEVICE` -- device-level parallelism (grid of blocks)
- `SEQ` / `NONE` -- sequential regions (could map to `scf.for` instead)

**ASTIRGen mapping** (current):
| Choreo Level | CoIR Level |
|--------------|------------|
| THREAD       | Thread     |
| GROUP        | Warp       |
| GROUPx4      | Warpgroup  |
| BLOCK        | Block      |
| CLUSTER      | Block (fallback) |
| DEVICE       | Block (fallback) |

## 3. DMA Operation Attributes

**Choreo** DMA operations carry rich metadata:
- Source/destination memory spaces and storage qualifiers
- Async semantics (fences, barriers, tokens)
- Transfer size, alignment, vectorization
- Swizzle patterns (for shared memory bank-conflict avoidance)
- Multicast descriptors (SM90 TMA)
- Pipeline stages and buffering depth

**CoIR** `coir.dma.copy` / `coir.tma.copy` currently have:
- Source/dest tensor operands
- Optional `transfer_bytes` attribute
- Token result for async completion

**Missing attributes**:
- `swizzle_mode`: B32/B64/B128 swizzle patterns
- `multicast_mask`: TMA multicast bitmask
- `pipeline_depth`: double/triple buffering stage count
- `vectorize_width`: elements per thread copy
- `fence_scope`: release/acquire semantics

## 4. InThreadsBlock / Warpspec Partitioning

**Choreo** supports `inthreads { ... }` blocks that partition warp/block-level
work among threads with different roles (producer/consumer, 1p1c/1p2c/1p3c patterns).

**CoIR** has no equivalent. Possible future representation:
```mlir
coir.partition {roles = ["producer", "consumer"]} {
  ^producer:
    // ...
  ^consumer:
    // ...
}
```

## 5. ChunkAt / Tiling Operations

**Choreo** `data.chunkat(p, q, _)` creates a sub-tensor view indexed by
parallel variables, with `_` indicating "take the full remaining dimension."

**CoIR** has `coir.tensor.tile` but it currently only takes an index list:
```mlir
%tile = coir.tensor.tile %src[%p, %q] : !coir.tensor<...> -> !coir.tensor<...>
```

**Missing**:
- Wildcard dimension handling (`_` = full slice)
- Integration with parallel variables for automatic bounds inference
- Dynamic tile sizes from parallel decomposition

## 6. Type System: Signed vs Signless Integers

**Decision made**: Use signless `i32` (MLIR convention) instead of signed `si32`
for tensor element types. The `arith` dialect requires signless types.

**Potential issue**: If future operations need to distinguish signed/unsigned
semantics (e.g., for comparison ops, division), we may need explicit conversion
ops or a signed-aware extension.

## 7. Return Value Semantics

**Choreo**: Functions return tensors by value; the code generator materializes
output buffers as kernel arguments.

**CoIR (current)**: `KernelOp` has result types in its function signature.
`KernelReturnOp` carries the returned values. The `EmitCUDA` pass maps return
values to additional pointer parameters in the generated kernel signature.

**Future work**: May need explicit `output` argument annotation to distinguish
input vs output pointers in the generated signature.

## 8. MMA Operations -- GPU Complete, Micro-Kernel Backend Implemented

**Status**: MMA IR is fully supported across the CoIR pipeline.

**GPU (WMMA)**: Complete pipeline from AST to generated code.
- `ASTCoIRGen` lowers `AST::MMA` Fill/Load/Exec/Store to `coir.mma.*` ops
- `LowerMMA` pass annotates ops with `target = "wgmma"` (SM90) or `"mma_sync"` (SM80)
- `EmitCUDA` emits `nvcuda::wmma` API calls
- E2E tested via `tools/coir/tests/gpu/e2e/wmma.co`

**Micro-kernel backend**: Emission implemented for targets that use library-based
MMA via `target = "ukernel"`.
- `LowerMMA` sets `target = "ukernel"` for non-GPU architectures
- Target emitter handles all four MMA ops via micro-kernel stub calls:
  - `mma.fill` -> workspace allocation (`int ws[2048]`)
  - `mma.load` -> address recording (no actual load instruction)
  - `mma.exec` -> deferred exec with acc/store flag state machine
  - `mma.store` -> final micro-kernel call with `store_flag=1`
- K-loop accumulation uses the deferred exec+store pattern:
  iterations 0..N-2 emit with `store=0`, final iteration deferred to `mma.store`
- Out-of-line `__device__` stubs work around linker relocation limitations

**Semantic differences between GPU and micro-kernel MMA**:
| Aspect | GPU (WMMA) | Micro-kernel |
|--------|-----------|-------------|
| Fragment objects | `wmma::fragment` structs | Raw buffer pointers |
| Accumulation | Fragment register state | Hardware accumulator registers |
| Exec model | Immediate `mma_sync` call | Deferred exec+store pattern |
| Store model | Separate `store_matrix_sync` | Combined compute+store via `store_flag` |
| Code generation | Direct API calls | Out-of-line stub wrappers |

**Remaining limitations**:
- `TensorTileOp` in target emitter computes pointer offsets assuming row-major dense layout
- Host wrapper input types inferred from output tensor type (may mismatch for mixed precision)
- `mma.wait`, `mma.scale` are not supported in the micro-kernel backend

## Summary Priority

| Gap | Priority | Needed For |
|-----|----------|-----------|
| Parallel levels (CLUSTER, DEVICE) | Medium | Multi-CTA kernels, grid-level ops |
| Tensor strides | Resolved | -- |
| DMA attributes | High | TMA/DMA-staged kernels (Phase 2) |
| InThreadsBlock | Low | Warpspec persistent kernels |
| ChunkAt tiling | High | DMA-staged elementwise (Phase 2) |
| Signed vs signless | Resolved | -- |
| Return semantics | Resolved | -- |
| MMA (GPU WMMA) | Resolved | -- |
| MMA (micro-kernel) | Resolved | -- |
