# Building Static Libraries with `--lib`

The `--lib` flag compiles one or more Choreo (`.co`) and C++ (`.cpp`) source
files into a single static library (`.a`).

## Quick Start

```bash
# Single kernel
choreo --lib -t cute kernel.co -o libkernel.a

# Multiple kernels (parallel with -j4)
choreo --lib -t cute -j4 a.co b.co c.co -o libkernels.a

# Kernels + C++ wrappers
choreo --lib -t cute -j4 \
    kernels/matmul.co \
    kernels/layernorm.co \
    wrappers/matmul_host.cpp \
    wrappers/layernorm_host.cpp \
    -o build/libkernels.a
```

## How It Works

### Single file

When given one `.co` file, `--lib` compiles it with `-fPIC`, then archives the
resulting `.o` into a `.a`:

```
choreo --lib -t cute kernel.co -o libkernel.a

  1. choreo parse + codegen
  2. nvcc -c -fPIC generated.cu -> kernel.o
  3. ar rcs libkernel.a kernel.o
```

### Multiple files

When given multiple files, choreo spawns parallel subprocesses:

```
choreo --lib -t cute -j4 a.co b.co wrap.cpp -o lib.a

  choreo --suppress-main -c a.co    -> a.o     (.co via choreo)
  choreo --suppress-main -c b.co    -> b.o     (.co via choreo)
  nvcc -c -fPIC wrap.cpp            -> wrap.o  (.cpp via nvcc)

  ar rcs lib.a a.o b.o wrap.o
```

- `.co` files are compiled by spawning `choreo` subprocesses
- `.cpp`/`.cc`/`.cxx`/`.c` files are compiled by the target compiler directly
- C++ files automatically get `-I` paths to choreo's `runtime/` headers so
  they can `#include "choreo.h"`
- `-j N` controls parallelism (default: 1)

### main() suppression

Choreo `.co` files often contain a `main()` function with test code. When
`--lib` is used, `main()` is automatically suppressed via preprocessor
`#define` to avoid linker conflicts. Each translation unit gets a unique
suppressed name derived from its filename. All other code (macros, types,
globals, host wrappers) is preserved.

## Writing C++ Wrappers

C++ wrapper files provide an `extern "C"` interface so downstream code can call
choreo kernels without including choreo headers.

### Pattern

```cpp
// layernorm_host.cpp
#include "choreo.h"
#include <cuda_runtime.h>

// Declare the choreo-generated function (signature from nm -C)
extern void layernorm_bf16(
    const choreo::spanned_view<choreo::bf16, 2>& input,
    const choreo::spanned_view<choreo::bf16, 2>& output,
    cudaStream_t stream);

// Thin extern "C" wrapper -- just convert pointers to spanned_view
extern "C" void layernorm_bf16_wrapper(
    void* input, void* output,
    int rows, int dim, cudaStream_t stream) {
  auto in_sv = croq::make_spanview<2, croq::bf16>(
      (croq::bf16*)input, {(size_t)rows, (size_t)dim});
  auto out_sv = croq::make_spanview<2, croq::bf16>(
      (croq::bf16*)output, {(size_t)rows, (size_t)dim});
  layernorm_bf16(in_sv, out_sv, stream);
}
```

### Key rules

1. **Check the generated signature** with `nm -C kernel.o` -- note the exact
   types, ranks, and `const&` qualifiers
2. **Parameter order**: `choreo::spanned_view<Type, Rank>` (type first, then
   rank)
3. **All spanned_view params are `const&`** in the generated code
4. **Keep wrappers thin**: just convert raw pointers to `spanned_view` and call.
   No memory allocation, no D2H/H2D copies, no type conversion.
5. **`croq::bf16`** is `choreo::bf16`, **`float`** is used directly (no
   `choreo::f32`)

### Discovering kernel signatures

Use `nm -C` on the compiled `.o` to see the exact C++ mangled signature:

```bash
choreo --suppress-main -c -t cute kernel.co -o kernel.o
nm -C kernel.o | grep 'T kernel_name'
```

Example output:
```
T layernorm_bf16(
    choreo::spanned_view<choreo::bf16, 2ul> const&,
    choreo::spanned_view<choreo::bf16, 2ul> const&,
    CUstream_st*)
```

## Linking Against the Library

Downstream code only needs the `extern "C"` declarations -- no choreo headers:

```cpp
// consumer.cpp -- no #include "choreo.h" needed
#include <cuda_runtime.h>

extern "C" void layernorm_bf16_wrapper(
    void* input, void* output,
    int rows, int dim, cudaStream_t stream);

int main() {
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  void *d_in, *d_out;
  cudaMalloc(&d_in, rows * dim * 2);
  cudaMalloc(&d_out, rows * dim * 2);

  layernorm_bf16_wrapper(d_in, d_out, rows, dim, stream);

  cudaFree(d_in);
  cudaFree(d_out);
  cudaStreamDestroy(stream);
}
```

Compile and link:

```bash
nvcc -std=c++17 consumer.cpp -Lbuild -lkernels -lcudart -o consumer
```

## Environment Variables

| Variable          | Description                    | Default   |
|-------------------|--------------------------------|-----------|
| `CHOREO_VERBOSE`  | Print compilation commands     | (unset)   |
