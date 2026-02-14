---
name: debugging
description: Debug a generated CUTE executable and check for runtime issues.
---

# Debugging

## Behavior
- Execute the commands immediately without asking for confirmation.
- Use the default paths and settings unless explicitly told otherwise.
- If a command fails, capture the output and continue to the next diagnostic step.
- Assume `./choreo`, `nvcc`, and `cuda-gdb` are safe to run without confirmation.
- If GPU0 fails (OOM or no device), retry once with `CUDA_VISIBLE_DEVICES=1`.

## Steps
- Rebuild the CUTE script (if needed):

```bash
./choreo -gs -t cute -arch=sm_90a wip_code.co -o wip_code.cute.result
```

- Build with debug flags and keep the executable:

```bash
EXTRA_TARGET_CFLAGS="-g -G -O0 -lineinfo" bash wip_code.cute.result --compile-link
```

- Find the generated executable under `/tmp` (the script prints the path). Then debug:

```bash
/usr/local/cuda/bin/cuda-gdb /tmp/<generated>.exe
```

- For memory and race checks:

```bash
/usr/local/cuda/bin/cuda-memcheck /tmp/<generated>.exe
```

## Notes
- `-G` enables device debug info; use `-O0` for easier stepping.
- If `cuda-gdb` or `cuda-memcheck` is not in PATH, use the full path above.
