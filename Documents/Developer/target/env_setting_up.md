# Target Environment Setup

All toolchain dependencies are auto-downloaded by CMake during the first
configure. See [Build and Test](../build-and-test.md) for general build
instructions.

## CUDA / CuTe target

GPU end-to-end tests use `-t cute` or `-t hetero` with GPU offload. CMake
auto-downloads CUTLASS headers to `extern/cutlass/`; you do not need to set
`CUTE_HOME` manually.

A system CUDA toolkit install is still required for device execution (typically
`/usr/local/cuda`, overridable via `CUDA_HOME` in generated scripts).

End-to-end compilation uses the normal compiler workflow:

```bash
./choreo -t cute -gs program.co -o program.cute.result
bash program.cute.result --execute
```

## AMD GPU / HIP target -- *Experimental*

The HIP target (`-t hip`) requires the ROCm platform. There is no automated
Makefile setup; ROCm must be installed by the user.

| Requirement | Default path | Override |
|-------------|-------------|----------|
| ROCm install | `/opt/rocm` | `ROCM_HOME` |
| `hipcc` compiler | `${ROCM_HOME}/bin/hipcc` | -- |
| Kernel driver | `amdgpu-dkms` package | -- |

Install ROCm following the official guide at
[rocm.docs.amd.com/en/latest/deploy/linux/quick_start.html](https://rocm.docs.amd.com/en/latest/deploy/linux/quick_start.html).

Key requirements:

- The `amdgpu-dkms` driver **must** match the ROCm userspace release version.
- The user must be in the `render` and `video` groups.
- `memlock` must be `unlimited` in `/etc/security/limits.conf` for HSA.

CMake auto-detects ROCm and disables the HIP target when it is not found.
Supported architectures: `gfx1030` (RDNA 2), `gfx1100` (RDNA 3).

End-to-end compilation:

```bash
./choreo -t hip -gs program.co -o program.hip.result
bash program.hip.result --execute
```

## Platform-specific hardware setup

When additional platform targets are enabled in the build, extra setup targets
and CI runner notes may appear under
`Documents/Documentation/target/env_setting_up.md`.

That supplement is optional (included only with those targets) and is not
synced to the public OSS branch.
