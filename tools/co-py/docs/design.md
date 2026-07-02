# croqtile-python Design Document

## 1. Motivation

Croqtile is a toolchain for programming data-movement entities
(DMAs, TMAs, MMAs) on heterogeneous hardware.  The Choreo front-end
provides a `.co` DSL and compiler.  **croqtile-python** (`import croq`)
provides a Python interface that constructs a Choreo AST directly in
memory, applies the same compiler passes, and emits the generated
CUDA/C++ code -- all without touching `.co` text.

## 2. Architecture

```
+----------------------------------------+
|  Python user code                      |
|  (import croq)                         |
|                                        |
|  +----------------------------------+  |
|  |  croqtile.builder  (pure Python) |  |
|  |  @co decorator, for-loop scopes, |  |
|  |  implicit scope stack, Var ops,  |  |
|  |  dma/tma/mma namespaces          |  |
|  +----------+-----------------------+  |
|             | calls                    |
|  +----------v-----------------------+  |
|  |  croqtile._core  (pybind11 C++) |  |
|  |  Thin wrappers around:           |  |
|  |    - AST node types              |  |
|  |    - ASTPipeline                 |  |
|  |    - CompilationContext          |  |
|  +----------+-----------------------+  |
|             | links                    |
|  +----------v-----------------------+  |
|  |  libchoreo-dev.a  (Croqtile SDK) |  |
|  |  Full compiler: parse, sema,     |  |
|  |  normalization, type inference,  |  |
|  |  codegen, targets                |  |
|  +----------------------------------+  |
+----------------------------------------+
```

### Design principles

1. **Everything is a Var** -- function parameters, declared variables,
   loop indices, MMA accumulators, DMA futures. No string lookups.
2. **For-loop scoping** -- `parallel`, `foreach`, `with_in` all use
   Python's `for` loop protocol. Zero name duplication.
3. **Python's `return`** -- just use `return output` inside `@croq.co`.
4. **`@` operator** -- maps to Choreo's `#` (hierarchical index).

## 3. Language Primitives Mapping

### 3.1 Data Types

```python
croq.s32           # scalar s32
croq.s32[6, 64]    # s32 tensor with shape [6, 64]
croq.f16[128]      # f16 vector of length 128
```

| Choreo `.co` | Python | C++ AST |
|---|---|---|
| `f16` | `croq.f16` | `BaseType::F16` |
| `f32` | `croq.f32` | `BaseType::F32` |
| `bf16` | `croq.bf16` | `BaseType::BF16` |
| `s32` | `croq.s32` | `BaseType::S32` |
| `f32 [M, N]` | `croq.f32[M, N]` | `AST::DataType` with shape |

Storage qualifiers:

| `.co` | Python |
|---|---|
| `global` | `croq.GLOBAL` |
| `shared` | `croq.SHARED` |
| `local`  | `croq.LOCAL`  |

### 3.2 The `@croq.co` Decorator

```python
@croq.co
def ele_add(lhs: croq.s32[2, 3, 64],
            rhs: croq.s32[2, 3, 64]) -> croq.s32[6, 64]:
    output = croq.declare(croq.s32[6, 64], "output")
    for p, q in croq.parallel(p=6, q=64):
        output[p, q] = lhs[p, q] + rhs[p, q]
    return output
```

The decorator:
1. Reads parameter annotations -> `ParamList`.
2. Reads the return annotation -> return type.
3. Passes `Var` proxies as function arguments.
4. Pushes an implicit scope, executes the body, pops the scope.
5. Captures the Python `return` value -> emits AST `Return` node.
6. Returns a `_FunctionBuilder` holding the built AST.

### 3.3 Scoped Blocks (for-loop protocol)

All scoped constructs use Python's `for` loop -- zero name duplication:

```python
# parallel -- maps to Choreo's `parallel p by 6, q by 64 { ... }`
for p, q in croq.parallel(p=6, q=64):
    output[p, q] = lhs[p, q] + rhs[p, q]

# foreach -- maps to `foreach {m, n, k} in [8, 4, 256] { ... }`
for m, n, k in croq.foreach(m=8, n=4, k=256):
    out[m, n] = out[m, n] + a[m, k] * b[k, n]

# with_in -- maps to `with tile_k in 16 { ... }`
for tile_k in croq.with_in(tile_k=16):
    f = croq.dma.copy(lhs.chunkat(px, tile_k), to=croq.LOCAL)
```

### 3.4 Expressions and Operators

All values are `Var` objects:

```python
c = a + b          # binary add
d = a * 2          # auto-coerced literal
e = -a             # negation
lhs[p, q]          # element read (__getitem__)
output[p, q] = e   # element write (__setitem__)
p @ g              # hierarchical index (Choreo's #)
```

### 3.5 DMA / TMA

```python
# sync DMA copy: f = dma.copy input => shared
f = croq.dma.copy(input, to=croq.SHARED, name="f")

# async: f = dma.copy.async input => shared
f = croq.dma.copy_async(input, to=croq.SHARED, name="f")

# wait: wait f
croq.wait(f)

# copy back: dma.copy f.data => output
croq.dma.copy(f.data, to=output)

# TMA (SM90+)
f = croq.tma.copy_async(src, to=lhs_s, swizzle=128)
```

### 3.6 MMA (Matrix Multiply-Accumulate)

```python
mc = croq.mma.fill(0.0)                       # mc = mma.fill 0.0
croq.mma.fill(mc, 0.0)                        # mma.fill mc, 0.0f
ma = croq.mma.load(lhs.chunkat(m @ g0, k))    # ma = mma.load ...
mb = croq.mma.load(rhs.chunkat(k, n @ g1))
croq.mma.exec(mc, ma, mb, method="row.col")   # mma.row.col mc, ma, mb
croq.mma.store(mc, output.chunkat(m @ g0, n @ g1))
```

### 3.7 Data Access Methods

```python
var.chunkat(m, k)              # tiled chunk access
var.subspan(M, K).at(bm, bk)  # subspan with indexing
var.data                       # DMA future data member
```

## 4. Implementation

### Implemented

- `Program`, `ChoreoFunction`, `FunctionDecl`, `ParamList`, `Parameter`
- `DataType`, `Identifier`, `Expr` (arithmetic + element access)
- `MultiNodes`, `MultiValues`, `ChunkAt`
- `NamedVariableDecl`, `Assignment`, `ParallelBy`, `Return`
- `ForeachBlock`, `WithBlock`, `Wait`
- `DMA` (sync/async/tma), `MMA` (fill/load/exec/store/commit)
- `@croq.co` decorator with Python `return` capture
- `for`-loop scoping for `parallel`, `foreach`, `with_in`
- Thread-local implicit scope stack
- `@` operator -> Choreo `#` (hierarchical index)
- `var.chunkat()`, `var.data`, `var.subspan()`

### Future work

- Full `.co` source serializer from AST (for end-to-end compilation)
- Automatic shape inference helpers
- `InThreadsBlock`, `Trigger`, `Synchronize`
- Integration with NumPy for host-side data
- JIT compilation and execution via CUDA driver API

## 5. Build

```bash
make co-py       # build the _core extension
make co-py-test  # run pytest suite
```

## 6. Relationship to Existing Projects

- **Croqtile SDK**: croqtile-python links against the Croqtile SDK
  object libraries. It does not modify the compiler.
- **coir**: Also consumes the Croqtile SDK, for MLIR-based IR work.
- **TVM Relay**: Inspiration for the two-layer design pattern.
