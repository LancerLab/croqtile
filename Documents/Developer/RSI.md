# RSI — development incident retrospective

This file records development-process incidents and the rules adopted so the
same failure cannot recur. It is about how we work on the compiler, not about
the compiler itself.

## 2026-11-08 — concurrent lit suites wedged the GPU

### What happened

Two `./tests/lit.sh tests/` instances ran at the same time: an orphaned
runner left behind by a stopped background task, and a newly started full
suite. Both executed `tests/gpu/end2end/cooperative_grid_sync.co`, which
launches two blocks with `cudaLaunchCooperativeKernel` and synchronizes them
with `cooperative_groups::this_grid().sync()`.

`grid_group::sync()` lowers to a device-scope arrival
(`ATOM.E.ADD.STRONG.GPU` on the driver-provided grid workspace) followed by an
**unbounded** spin on `LD.E.STRONG.GPU` / `CCTL.IVALL` / `YIELD` and a
`BAR.SYNC`. Under concurrent suite execution the peer block did not make
progress, so the spinning warps stayed resident forever. The CUDA context
became unkillable: `SIGKILL` stayed pending (`/proc/<pid>/status` showed
`SigPnd: 0x100`), `nvidia-smi` hung, and an unrelated benchmark on the same
GPU blocked as well. Recovery required a reboot (a GPU reset needs root).

A second stuck process came from a review subagent that attempted an
end-to-end emulation concurrently with the suites — the same class of
mistake from a different entry point.

### Root causes

1. A suite was started without checking whether another suite was still
   running.
2. `TaskStop` on the background command killed the wrapper, but the
   `lit.sh` process tree (and its compiled GPU test binaries) survived and
   kept executing end-to-end tests.
3. GPU end-to-end tests are not safe under concurrent execution: cooperative
   grid sync has no timeout, so a starved peer turns a test into a permanent
   GPU hang.
4. Subagents also launch test commands; concurrency can come from delegated
   work as much as from the main loop.

### Rules

1. **Check before running.** Before any `./tests/lit.sh` invocation:
   `pgrep -af "tests/lit.sh tests/"`. If anything is running, wait for it or
   stop it; never start a second instance.
2. **Stopping a suite means verifying the process tree.** Use `TaskStop`,
   then `pgrep` again (and check for stray `__choreo_*.exe` test binaries).
   The wrapper exiting is not proof that the tests stopped.
3. **GPU end-to-end tests are serial only.** Do not run `tests/gpu/end2end`
   from two places, and avoid running them while another GPU workload (e.g. a
   piano benchmark) is measuring, unless necessary.
4. **Compiler-only verification does not need the GPU.** When the GPU is
   unavailable, run every non-end-to-end directory with a shim
   `nvidia-smi` that prints the known device (e.g.
   `PATH=/tmp/fakebin:$PATH`, with `nvidia-smi` printing
   `NVIDIA H800 PCIe, 9.0`), covering `tests/parse`, `tests/check`,
   `tests/cli`, `tests/pp`, `tests/cpu`, `tests/norm`, `tests/infer`,
   `tests/others`, `tests/agent`, `tests/lit-self`, `tests/gpu/{check,cli,
   codegen,liveness,norm}` and `tests/amdgpu/{check,codegen}`. End-to-end
   tests must wait for the GPU.
5. **A wedged GPU test cannot be killed.** If a test process is `R` with a
   pending `SIGKILL` (or `D` in a driver wait) and `nvidia-smi` hangs, stop
   issuing kills; reboot or use `sudo nvidia-smi --gpu-reset` (the latter
   needs no active contexts).
6. **Report the state honestly.** If a suite cannot be completed because of
   the environment, say so and record which verification is still pending
   rather than treating a partial run as green.

### What was verified about the kernel

The cooperative-grid-sync kernel itself is correct: 2 blocks × 1 thread, one
global store, one grid sync, one global load/store. Its PTX contains a
`trap` when the driver grid workspace is absent (a non-cooperative launch
would abort instead of spinning), which confirms the launch was cooperative.
The failure is the unbounded spin combined with concurrent execution, not the
kernel logic. The test passed in every serial full-suite run, including
alongside the piano benchmark; it only wedged under two concurrent suites.
