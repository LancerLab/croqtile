# croqtile-python -- Python Bindings for CroqTile

**croqtile-python** (`import croq`) lets you build CroqTile AST programs
from Python, run semantic analysis, and drive the full compiler pipeline
-- no `.co` files required.

## Prerequisites

- **Python 3.8+**
- **CUDA Toolkit** (nvcc) -- required for compiling generated CUDA source
  into GPU executables. Install from
  [developer.nvidia.com/cuda-toolkit](https://developer.nvidia.com/cuda-toolkit).
- **NVIDIA GPU** -- for running compiled kernels. SM80+ recommended (SM86, SM90).
- **C++17 compiler** (GCC 9+ or Clang 5+) -- for building the `_core`
  extension from source.

## Build (in-tree)

co-py is an in-tree subproject of the Choreo repository, built via the
`CROQ_PROJECT` mechanism. From the repository root:

```bash
# Build the _core Python extension (and choreo compiler as prerequisite):
make co-py

# Run the test suite:
make co-py-test

# Clean build artifacts:
make co-py-clean
```

The build automatically downloads pybind11 into `extern/pybind11/` on
first run (controlled by `cmake/PythonBootstrap.cmake`).

After building, use the module directly:

```bash
PYTHONPATH=tools/co-py/src python3 -c "import croq; print(croq.version())"
```

### Requirements

- Python 3.8+ with development headers (`python3-dev`)
- The Choreo compiler prerequisites (GCC 9+, CMake 3.18+, Ninja)

### Python version support

croqtile-python supports **Python 3.8 - 3.12**. The C++ extension
(`_core`) is compiled against the active Python interpreter at build time.
Override with: `CO_PY_PYTHON=python3.11 make co-py`

## Package contents

The `croqtile` package includes:

- **`_core`** -- pybind11 C++ extension for AST construction and
  semantic analysis
- **`builder`** -- Pythonic DSL for defining kernels
- **`runtime`** -- compilation and execution pipeline (`.co` -> CUDA ->
  executable); uses the in-tree `build/choreo` binary

## Quick start

```python
import croq

@croq.co
def ele_add(lhs: croq.s32[6, 64],
            rhs: croq.s32[6, 64]) -> croq.s32[6, 64]:
    output = croq.declare(croq.s32[6, 64], "output")
    for p, q in croq.parallel(p=6, q=64):
        output[p, q] = lhs[p, q] + rhs[p, q]
    return output

prog = croq.Program()
prog.add(ele_add)
print(prog.dump_ast())
```

### End-to-end execution

```python
import croq

prog = croq.Program()
prog.add(my_kernel)
prog.add(host_code)  # C++ main() for verification

# Full pipeline: .co -> CUDA -> compile -> execute
output = croq.runtime.compile_and_run(
    prog, arch="sm_86", check_lines=["Test Passed"])
```

## API overview

### Types

```python
croq.s32              # scalar s32
croq.s32[6, 64]       # s32 tensor with shape [6, 64]
croq.f16[128, 256]    # f16 tensor
```

### Scoped blocks (for-loop protocol)

```python
for p, q in croq.parallel(p=6, q=64):   # parallel-by
for m, k in croq.foreach(m=128, k=256): # foreach loop
for tile_k in croq.with_in(tile_k=16):  # with-in (staging)
```

### DMA / TMA

```python
f = croq.dma.copy(input, to=croq.SHARED, name="f")
f = croq.dma.copy_async(input, to=croq.SHARED, name="f")
croq.wait(f)

f = croq.tma.copy_async(src, to=lhs_s, swizzle=128)  # SM90+
```

### MMA (Matrix Multiply-Accumulate)

```python
mc = croq.mma.fill(0.0)
ma = croq.mma.load(lhs.chunkat(m @ g0, k))
mb = croq.mma.load(rhs.chunkat(k, n @ g1))
croq.mma.exec(mc, ma, mb, method="row.col")
croq.mma.store(mc, output.chunkat(m @ g0, n @ g1))
```

## Development

```bash
CO_PY_PYTHON=python3.11 make co-py-test  # test with specific Python
```

See `docs/design.md` for the full DSL mapping.
