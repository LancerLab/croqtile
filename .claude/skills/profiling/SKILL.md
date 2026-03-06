---
name: profiling
description: Profile kernels and runtime behavior of wip_code.co.
---

# Profiling

## Behavior
- Execute the commands immediately without asking for confirmation.
- Use the default paths and settings unless explicitly told otherwise.
- If a command fails, capture the output and continue to the next diagnostic step.
- Assume `./choreo`, `ncu`, and `nvcc` are safe to run without confirmation.
- If GPU0 fails (OOM or no device), retry once with `CUDA_VISIBLE_DEVICES=1`.

## Steps
- Generate the CUTE script (if needed):

```bash
./choreo -gs -t cute -arch=sm_90a wip_code.co -o wip_code.cute.result
```

- Collect a system timeline with Nsight Systems:

```bash
nsys profile -o wip_nsys_report bash wip_code.cute.result --execute
```

- Collect kernel-level metrics with Nsight Compute:

```bash
/usr/local/cuda/bin/ncu --set full --target-processes all -o wip_ncu_report bash wip_code.cute.result --execute
```

## Notes
- Use `--target-processes all` so child processes from the script are captured.
- If `ncu` is not in PATH, use `/usr/local/cuda/bin/ncu`.
