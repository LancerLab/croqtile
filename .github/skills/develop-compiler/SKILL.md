---
name: develop-compiler
description: Modify and rebuild the Choreo compiler itself.
---

# Develop the compiler

## Behavior
- Execute the commands immediately without asking for confirmation.
- Use the default paths and settings unless explicitly told otherwise.
- If a command fails, capture the output and continue to the next diagnostic step.
- Assume `./choreo`, `nvcc`, and `cuda-gdb` are safe to run without confirmation.

## Steps
- Edit compiler sources under `lib/` and `tools/`.
- Rebuild (Release):

```bash
make
```

- Or rebuild (Debug):

```bash
make debug
```

## Validation
- Run a quick compile of `wip_code.co`:

```bash
./choreo -gs -t cute -arch=sm_90a wip_code.co -o wip_code.cute.result
```

- Run tests (optional):

```bash
make test
```
