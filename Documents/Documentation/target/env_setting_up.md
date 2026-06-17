# Target Environment Setup (GCU)

This supplement is included when the GCU target is enabled in the build. It is
not synced to the public OSS branch.

Shared setup (`make setup-core`) and the OSS-safe baseline are documented in
[Build and Test](../../Developer/build-and-test.md) and
[Developer Target Environment Setup](../../Developer/target/env_setting_up.md).

## GCU-2.x runners

On GCU2 CI machines (`tags: GCU2`):

```bash
make setup-gcu2
```

This runs `setup-core` and installs the GCU-2.x compiler/runtime kit.

`make gcu2-kmd` installs the base kernel-module driver. Without it the device
is not usable. It is optional on CI runners because the driver is normally
already installed and only needs to be set up once:

```bash
make gcu2-kmd
```

CI sequence (from `.gitlab-ci.yml`):

```bash
make setup-gcu2
make build
make ci-test
```

## GCU-3.x runners

On GCU3 CI machines (`tags: GCU3`):

```bash
make setup-gcu3
```

This runs `setup-core`, installs the acore library, and installs the GCU-3.x
platform kit.

`make gcu3-kmd` installs the base kernel-module driver (same one-time /
usually-already-present note as `gcu2-kmd` above):

```bash
make gcu3-kmd
```

CI sequence:

```bash
make setup-gcu3
make build
make ci-test
```

## Other GCU generations

| Platform | Setup target | Driver (kernel module, one-time) |
|----------|--------------|----------------------------------|
| GCU-4.x  | `make setup-gcu4` | `make gcu4-kmd` |
| GCU-5.x  | `make setup-gcu5` | (see extern Makefile) |

Re-install kits after toolchain package updates:

```bash
make resetup-gcu2   # or resetup-gcu3, etc.
```

## AMDGPU runners

On AMDGPU CI machines (`tags: AMDGPU`):

Unlike GCU, ROCm must be pre-installed by the user. There is no
`make setup-amdgpu`.

| Component | Default location | Override env var |
|-----------|-----------------|------------------|
| ROCm install | `/opt/rocm` | `ROCM_HOME` |
| `hipcc` compiler | `${ROCM_HOME}/bin/hipcc` | -- |
| Kernel driver | `amdgpu-dkms` package | -- |

CI sequence (from `.gitlab-ci.yml`):

```bash
make setup-core
make debug
./tests/lit.sh tests/amdgpu/
```

### Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `hipGetDeviceCount` returns 0 | Driver/runtime version mismatch | Ensure `amdgpu-dkms` and ROCm userspace from same release; reboot |
| `HSA_STATUS_ERROR_OUT_OF_RESOURCES` | Locked memory limit too low | Set `memlock unlimited` in `/etc/security/limits.d/99-rocm.conf` |
| `open(/dev/dri/renderD128)` EINVAL | Stale driver state | Reboot or reload amdgpu module |
| SSH submodule clone fails | `/opt/gitlab-runner/ssh/config` permission denied | `sudo chown gitlab-runner:gitlab-runner /opt/gitlab-runner/ssh/config` |
