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

## Symbolic (sbe) Infrastructure Conventions

### Use sbe for all value/offset computations in codegen
- `sbe::nu(n)` for integer constants, `sbe::sym("scoped::name")` for symbolic vars.
- Compose with `+`, `*`; use `sbe::ceq(a, b)` for equality tests; `Normalize()` folds constants.
- Materialize to C++ strings via `ValueSTR(operand)` — adds parens for binary sub-expressions.
- Never use `SplitStringByDelimiter` + `OpExprSTR` for offset/index generation; those are legacy patterns predating the sbe library.

### Nodes created in LateNorm must carry sbe vals
- ShapeInference (`INFER` pass) populates `Opts().SetVals()` on all pre-existing `AST::Expr` nodes and records per-symbol values in `FCtx(fname).GetSymbolValues(scoped_name)`.
- Any new `AST::MultiValues` or `AST::Expr` node created in `LateNorm` (e.g. `HostSliceBufferGen`) must have `Opts().SetVals()` populated manually using existing sbe operands — ShapeInference does NOT re-run after LateNorm.
- Preferred pattern in LateNorm for a newly created `MultiValues` wrapping a known bounded variable:
```cpp
if (auto tid = AST::GetIdentifier(tiler_node)) {
  auto sname = InScopeName(tid->name);
  auto& sv = FCtx(fname).GetSymbolValues(sname);
  if (sv.HasVals()) {
    mv_node->Opts().SetVals(sv.GetVals());       // exact sbe operands from ShapeInference
  } else if (bv_map.count(sname)) {
    ValueList mv_vals;
    for (auto& m : bv_map.at(sname))
      mv_vals.push_back(sbe::sym(m));            // reconstruct from bv_map matcher names
    mv_node->Opts().SetVals(mv_vals);
  }
}
```
- `FCtx(fname).GetSymbolValues(sname)` is accessible anywhere `VisitorWithSymTab` (or `LateNormBase`) is in scope.
- `bv_map` (inherited from `VisitorWithSymTab`) maps scoped var names to their per-dimension matcher names.

### GenMdsOffset dispatch contract
- Index-based SOPs (`Tiling`, `SubSpan`, `TileAt`): collect vals per child `Expr` node from `Opts().GetVals()`; if count mismatches block-shape rank, fall back to aggregated `op->GetIndices()->Opts().GetVals()` which **must** be present (assert).
- Offset-based SOPs (`View`): offset child nodes are always `Expr` with vals; assert and use directly.
- Never add `OpExprSTR` fallbacks to these paths — if vals are missing, fix the upstream pass that creates the node.

## Code Style Rules (MANDATORY)

### ASCII-only source files
- ALL source files (`lib/`, `tools/`, `tests/`) must contain only ASCII characters (0x00-0x7F).
- Do NOT use Unicode in comments, strings, or identifiers: no em-dashes (`--` not `--`), no right arrows (`->` not `-->`), no ellipsis (`...` not `...`), no `x` for multiplication.
- After editing, verify with:
```bash
LC_ALL=C grep -Prn '[^\x00-\x7F]' lib/ tools/ tests/check/ --include="*.cpp" --include="*.hpp" --include="*.h" --include="*.co" 2>/dev/null | grep -v '\.swp'
```
  This must return no output.

### Diagnostic output: use errs()/dbgs(), never fprintf
- Use `errs()` (LLVM raw_ostream) for all compiler diagnostic messages, never `std::fprintf(stderr, ...)`.
- Use `dbgs()` for debug-only output guarded by `VST_DEBUG(...)`.
- `errs()` and `dbgs()` are available wherever `ast.hpp` or other Choreo headers are included.
- Example:
```cpp
// WRONG
std::fprintf(stderr, "[choreo] error: %s\n", msg);
// RIGHT
errs() << "[choreo] error: " << msg << "\n";
```
- After editing, verify no fprintf remains:
```bash
grep -rn 'fprintf' lib/ tools/ --include="*.cpp" --include="*.hpp"
```
  This must return no output.
