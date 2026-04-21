# Atomic Operations

Choreo provides first-class atomic built-in functions for thread-safe read-modify-write operations on shared and global memory. Each atomic operation returns the **old value** at the target address before the modification.

## Available Operations

| Choreo Built-in       | Semantics                           | CUDA Equivalent |
|-----------------------|-------------------------------------|-----------------|
| `__atomic_add(a, v)`  | `*a += v`; returns old `*a`         | `atomicAdd`     |
| `__atomic_sub(a, v)`  | `*a -= v`; returns old `*a`         | `atomicSub`     |
| `__atomic_exch(a, v)` | `*a = v`; returns old `*a`          | `atomicExch`    |
| `__atomic_min(a, v)`  | `*a = min(*a, v)`; returns old `*a` | `atomicMin`     |
| `__atomic_max(a, v)`  | `*a = max(*a, v)`; returns old `*a` | `atomicMax`     |
| `__atomic_and(a, v)`  | `*a &= v`; returns old `*a`         | `atomicAnd`     |
| `__atomic_or(a, v)`   | `*a \|= v`; returns old `*a`        | `atomicOr`      |
| `__atomic_xor(a, v)`  | `*a ^= v`; returns old `*a`         | `atomicXor`     |
| `__atomic_cas(a, cmp, v)` | if `*a == cmp` then `*a = v`; returns old `*a` | `atomicCAS` |

## First Argument: Data Element Access

The first argument must be a **data element access** (`data.at(index)`) referring to either `shared` or `global` storage. Function parameters (which map to global memory on GPU targets) are also accepted.

```
shared s32 [64] counts;
counts.at(tid) = 0;
sync.shared;
__atomic_add(counts.at(bin), 1);  // shared memory atomic
```

Atomics on `local` or `register` storage are not supported and will produce a compile-time error.

## Return Value

All atomic operations return the **old value** before the modification. The return value can be captured via assignment:

```
mutable s32 old = __atomic_add(counts.at(bin), 1);
```

When used as a standalone statement, the return value is discarded:

```
__atomic_add(counts.at(bin), 1);  // old value discarded
```

## Supported Types by Architecture (CUDA/cute target)

Type support varies by GPU architecture. The compiler validates type compatibility and emits an error when the target architecture does not support a given type/operation combination.

### `__atomic_add`

| Type | Minimum Architecture |
|------|---------------------|
| `s32`, `u32`, `u64`, `f32` | All (SM70+) |
| `f64` | SM60+ |
| `f16` | SM70+ |

### `__atomic_sub`

| Type | Minimum Architecture |
|------|---------------------|
| `s32`, `u32` | All |

### `__atomic_exch`

| Type | Minimum Architecture |
|------|---------------------|
| `s32`, `u32`, `u64`, `f32` | All |

### `__atomic_min`, `__atomic_max`

| Type | Minimum Architecture |
|------|---------------------|
| `s32`, `u32` | All |
| `s64`, `u64` | SM35+ |

### `__atomic_and`, `__atomic_or`, `__atomic_xor`

| Type | Minimum Architecture |
|------|---------------------|
| `s32`, `u32` | All |
| `s64`, `u64` | SM35+ |

### `__atomic_cas`

| Type | Minimum Architecture |
|------|---------------------|
| `s32`, `u32`, `u64` | All |
| `u16` | SM70+ |

## Examples

### Shared Memory Histogram

```
__co__ void histogram(s32 [N] input, s32 [BINS] output) {
  parallel p by N : thread {
    shared s32 [BINS] counts;
    if (p < BINS) {
      counts.at(p) = 0;
    }
    sync.shared;
    mutable s32 bin = input.at(p) % BINS;
    __atomic_add(counts.at(bin), 1);
    sync.shared;
    if (p < BINS) {
      output.at(p) = counts.at(p);
    }
  }
}
```

### Global Atomic Reduction

```
__co__ void reduce_sum(f32 [N] input, f32 [1] output) {
  output.at(0) = 0.0f;
  parallel p by N : thread {
    __atomic_add(output.at(0), input.at(p));
  }
}
```

### Compare-and-Swap

```
__co__ void cas_example(s32 [64] data) {
  parallel p by 64 : thread {
    shared s32 [64] buf;
    buf.at(p) = data.at(p);
    sync.shared;
    mutable s32 old = __atomic_cas(buf.at(p), 0, 42);
  }
}
```
