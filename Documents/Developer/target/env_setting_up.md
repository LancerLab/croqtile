# Target Environment Setup

To run compiled Croqtile programs on hardware, the target platform's compiler toolchain and runtime drivers must be configured. The Makefile provides setup targets for supported platforms:

```bash
make setup-<target>    # install target compiler and runtime
make <target>-kmd      # configure target hardware driver (if applicable)
```

Refer to the specific target's documentation for exact setup commands, SDK requirements, and hardware configuration.

For the CUDA/CuTe target, ensure:

- CUDA toolkit is installed (default: `/usr/local/cuda`)
- CuTe headers are available (default: `extern/cutlass`)

These paths can be overridden via `CUDA_HOME` and `CUTE_HOME` environment variables.
