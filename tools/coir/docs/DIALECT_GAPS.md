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

**CoIR** `coir.data.copy` / `coir.dma.copy` / `coir.tma.copy` currently have:
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
