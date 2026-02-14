---
name: build-and-install
description: Build the Choreo compiler and tools for this repository.
---

# Build and install

## Behavior
- Execute the commands immediately without asking for confirmation.
- Use the default paths and settings unless explicitly told otherwise.
- If a command fails, capture the output and continue to the next diagnostic step.
- Assume `./choreo`, `nvcc`, and related CUDA tools are safe to run without confirmation.

## Steps
- Ensure CUDA is available at `/usr/local/cuda` and CUTE at `extern/cutlass`.
- Build with the repo Makefile (default: Release + Ninja):

```bash
make
```

- For a debug build:

```bash
make debug
```

- Optional direct CMake invocation (Release):

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
ninja -C build
```

## Validation
- Confirm the compiler binary is available:

```bash
./choreo --help | head -n 5
```

## Notes
- `CUDA_HOME` defaults to `/usr/local/cuda`. Override if needed.
- `CUTE_HOME` defaults to `extern/cutlass`. Override if needed.
