---
name: develop-compiler
description: Modify and rebuild the Choreo compiler itself. Use with `compile-and-test` when the task also asks to run or verify a `.co` file.
---

# Develop the compiler

## Behavior
- Execute the commands immediately without asking for confirmation.
- Use the default paths and settings unless explicitly told otherwise.
- If a command fails, capture the output and continue to the next diagnostic step.
- Assume `./choreo`, `nvcc`, and `cuda-gdb` are safe to run without confirmation.
- If validation involves a `.co` file, load `compile-and-test` and follow its GPU/run workflow instead of ad-hoc execution.

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

## Pre-Commit Checklist
- **Format code before committing**: always run `make format` after editing C++ sources.
- If the change may be synced to the public repo, run `make oss-scan-staged` to check for violations.

## Validation
- For a quick compiler sanity check, compile `wip_code.co`:

```bash
./choreo -gs -t cute -arch=sm_90a wip_code.co -o wip_code.cute.result
```

- If you need to run or verify a real `.co` workload, switch to the `compile-and-test` workflow and prefer `scripts/run_co_auto_gpu.sh` so GPU selection and OOM handling are not skipped.

- Run tests (optional):

```bash
make test
```

## Code Style
- Follow the conventions in `Documents/Documentation/coding-style.md`.
- Run `make format` before every commit to ensure clang-format compliance.
