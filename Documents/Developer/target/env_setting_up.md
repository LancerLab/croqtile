# Target Environment Setup

Shared environment setup (`make setup-core`) is documented in
[Build and Test](../build-and-test.md). Run that on every fresh clone before
building or testing.

## CUDA / CuTe target

GPU end-to-end tests use `-t cute` or `-t hetero` with GPU offload. After
`setup-core`, the `extern/cutlass` submodule provides CuTe/CUTLASS headers; you
do not need to set `CUTE_HOME` manually.

A system CUDA toolkit install is still required for device execution (typically
`/usr/local/cuda`, overridable via `CUDA_HOME` in generated scripts).

End-to-end compilation uses the normal compiler workflow:

```bash
./choreo -t cute -gs program.co -o program.cute.result
bash program.cute.result --execute
```

## Platform-specific hardware setup

When additional platform targets are enabled in the build, extra setup targets
and CI runner notes may appear under
`Documents/Documentation/target/env_setting_up.md`.

That supplement is optional (included only with those targets) and is not
synced to the public OSS branch.
