---
name: add-choreo-kernel-cases
description: Generate or update Choreo benchmark matmul kernel case code under benchmark/performance/matmul, including precision selection (f16, f8_e4m3), architecture-specific paths (SM86 mma.sync/dma and SM90 WGMMA/TMA), and optimization variants (dynamic, persistent, warpspec 1p1c/1p2c/1p3c, swizzle, hilbert, colmajor, mwg, trans output, stmatrix perf).
---

# Add Choreo Kernel Cases

Implement new Choreo matmul kernel case files by reusing proven patterns in `benchmark/performance/matmul` and only changing requested dimensions, precision, architecture, and optimization strategy.

## Workflow

1. Confirm case targets from user request.
- Extract: architecture (`sm86` or `sm90`), precision (`f16` or `f8_e4m3`), optimization (`dynamic`, `persistent`, `warpspec`, `mwg`, `swizzle`, `hilbert`, `colmajor`, `trans`, `stmatrix`), and whether this is a new file or update.
- If values are missing, infer from nearest existing naming pattern and state the assumption.

2. Select a seed file first, then edit minimally.
- Pick the closest template from `benchmark/performance/matmul`.
- Keep existing host harness/timing/verification style unchanged unless user explicitly asks to change it.
- Preserve naming conventions in both file name and macros.

3. Apply architecture and precision constraints before writing kernel logic.
- Always keep/adjust compile-time guards (`#if/#error`) so illegal parameter combinations fail early.
- Do not relax constraints unless user explicitly asks.

4. Implement optimization-specific structure.
- For `persistent`: use `NUM_SMS` launch and tile-iterator mapping.
- For `warpspec`: use producer/consumer split with `shared event full/empty` and `inthreads.async`.
- For schedule-based variants (`swizzle`/`hilbert`/`colmajor`): add host-side schedule builder and pass schedule arrays to kernel if needed.

5. Run quality checks.
- Verify naming and macro consistency.
- Verify precision-dependent data types in both kernel and host allocation.
- If toolchain/build is available, run a targeted build or test for the modified case.

## Reference Usage

Read [matmul-case-guide.md](references/matmul-case-guide.md) before code generation. Use sections selectively:
- Syntax skeleton and naming: always.
- SM86 rules: only for `*_sm86.co`.
- SM90 rules: only for `*_sm90.co`.
- Optimization templates: only the requested optimization.

## Execution Rules

- Prefer minimal-diff edits over rewriting whole files.
- Keep macro names and verification style aligned with neighboring cases.
- Keep output deterministic: if multiple seeds match, pick one and state why.
- If user asks to “add a new case”, create one `.co` file plus only the minimum helper code needed in that file.
- Avoid adding unrelated docs or extra files.
