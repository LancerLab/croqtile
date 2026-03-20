---
name: choreo-kernel-compiler
description: Compile Choreo kernel source files (`.co`) with `./build/choreo`, including selecting compile flags, setting explicit output artifacts, and verifying successful build results. Use when a request asks to compile, rebuild, or test any Choreo kernel implementation from a `.co` file.
---

# Choreo Kernel Compiler

Compile Choreo kernel `.co` sources into output artifacts with reproducible commands.

## Workflow

1. Validate prerequisites.
- Confirm repository root contains `./build/choreo`.
- Confirm the input file exists and ends with `.co`.
- Prefer absolute paths when the source is outside the current directory.

2. Run standard compile command.
- Prefer this baseline:
  `./build/choreo -c <input.co> -o <output>`
- Add flags only when requested or clearly required by the task:
  `-t <platform>`, `-arch=<processor>`, `-g`, `-tg`, `-es`, `-api=<mode>`, `--use-warpspec`, `--use-prepack`.
- For NVIDIA SM90 benchmarks in this repository, prefer `-t cute -arch=sm_90a`.
- If the input file name contains `warpspec`, append `--use-warpspec` automatically unless already provided.
- If the input file name contains `prepack`, append `--use-prepack` automatically unless already provided.

3. Verify outputs.
- Check command exit code.
- Check that the requested output path exists.
- When compile fails, report the exact failing command and stderr summary.

4. If the task also includes running benchmark executables after compile:
- Run with the program's default timing configuration.
- Never inject `CHOREO_TIMING_WARMUP` or `CHOREO_TIMING_REPEAT` for benchmark runs.

## Command Templates

Compile one kernel:

```bash
./build/choreo -c benchmark/performance/matmul/matmul_f16_dyn_sm90_warpspec_1p1c.co -o matmul_f16_dyn_sm90_warpspec_1p1c
```

Compile with extra target options:

```bash
./build/choreo -c <input.co> -o <output> -t cute -arch=sm_90a -g
```

## Script

Use `scripts/compile_choreo_kernel.sh` for a repeatable wrapper around the baseline flow:

```bash
.codex/skills/choreo-kernel-compiler/scripts/compile_choreo_kernel.sh <input.co> [output] [-- extra choreo args]
```

Examples:

```bash
.codex/skills/choreo-kernel-compiler/scripts/compile_choreo_kernel.sh \
  benchmark/performance/matmul/matmul_f16_dyn_sm90_warpspec_1p1c.co

.codex/skills/choreo-kernel-compiler/scripts/compile_choreo_kernel.sh \
  benchmark/performance/matmul/matmul_f16_dyn_sm90_warpspec_1p1c.co \
  matmul_f16_dyn_sm90_warpspec_1p1c_dbg -- -g -t cute -arch=sm_90a
```

## Notes

- Keep compile commands explicit; do not rely on implicit output names when reproducibility matters.
- Use `./build/choreo --help` to inspect available flags before introducing new options.
