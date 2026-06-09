# Migration: frag.XX Syntax -> apply/reduce Statements

Date: 2026-06-09

Status: Design (pre-implementation)

## Summary

Replace the `frag.apply`, `frag.reduce_max`, `frag.reduce_sum`, and `frag.copy` syntax
with new first-class statements:

| Old Syntax | New Syntax |
|---|---|
| `frag.apply target, [](i, j) { return expr; };` | `apply {i, j} in target.span { target.at(i, j) = expr; }` |
| `frag.reduce_max src, dst, 1;` | `reduce_max(dst, src, 1);` |
| `frag.reduce_sum src, dst, 1;` | `reduce_sum(dst, src, 1);` |
| `frag.copy dst, src;` | `copy(dst, src);` |
| `frag.store out, frag;` | (removed -- use mma.store or dma.copy) |

The existing implementation (FragmentLayoutPass, WGMMA/REPLICATED_1D layouts, reduce
codegen with cross-thread shuffles) is **fully reusable**. This migration changes only
the parser/AST surface and the codegen entry points.

---

## Motivation

The `frag.apply` syntax has usability issues:

1. **Confusing index semantics** -- `[](i, j)` looks like a C++ lambda, but `i`/`j`
   are not regular variables. They are logical fragment coordinates where each thread
   only processes a subset (determined by fragment layout). Users don't understand
   what `i`/`j` "mean" without understanding thread-to-register mappings.

2. **Functional vs imperative mismatch** -- `frag.apply` uses `return expr` (functional),
   but the surrounding code is imperative (assignments, conditionals). The new `apply`
   uses imperative assignment: `target.at(i, j) = expr`.

3. **Single-expression limitation** -- `frag.apply` body is one return expression.
   Real kernels need multi-statement blocks (conditional masking + scalar updates in
   the same logical iteration, e.g., causal mask + score backup in flash attention).

4. **Reduction placement confusion** -- In the old design, `frag.reduce_max` is a
   statement between `frag.apply` calls. But users tried putting reductions inside
   `frag.apply` (v1 experiments), which is semantically wrong because reductions
   require cross-thread synchronization.

The new design makes the execution model explicit:
- `apply` = per-element, register-local, no communication
- `reduce_max/sum` = collective, requires thread synchronization
- They are always separate statements -- never nested

---

## New Syntax Specification

### apply Statement

```
apply {iv_list} in <span_expr> {
  <statement>*
}
```

Or single-statement form:
```
apply {iv_list} in <span_expr>
  <single_statement>;
```

**Components:**
- `iv_list` -- comma-separated iterator names (e.g., `i` for 1D, `i, j` for 2D)
- `span_expr` -- `fragment_name.span` (provides the logical iteration domain)
- Body -- imperative statements using `.at(iv...)` for fragment access

**Semantic rules:**
1. All statements are purely element-wise (no reductions, no cross-thread ops)
2. Iterators represent logical coordinates into the fragment's shape
3. Each thread processes only its owned portion (determined by fragment layout)
4. Row-hoisting rule: statements referencing only outer iterators execute once per
   outer-index (e.g., `i`-only statements in a 2D apply run once per row)
5. Per-row statements for a given `i` execute before per-element statements for that `i`
6. Source order preserved within each execution tier

### reduce_max / reduce_sum Statement

```
reduce_max(dst_frag, src_frag, dim);
reduce_sum(dst_frag, src_frag, dim);
```

**Operand order:** destination, source, dimension (matches MMA instruction style).

**Semantics:**
- Collective operation: involves thread-local reduction + cross-thread shuffle/barrier
- `dim=1` reduces along columns (2D src -> 1D dst, row-wise reduction)
- `dst_frag` must be pre-declared with compatible shape
- Cannot appear inside an `apply` block

### copy Statement

```
copy(dst_frag, src_frag);
```

**Semantics:**
- Element-wise register copy between same-shape fragments
- Both fragments must have compatible layouts
- Destination inherits source layout during layout propagation

Replaces `frag.copy dst, src;`.

**Both standalone and inline forms are valid:**

```co
// Form 1: standalone statement (clearer intent, self-documenting)
copy(scores_max_prev, scores_max);

// Form 2: inline inside apply (fused with adjacent work, fewer loops)
apply {i, j} in acc_s.span {
  scores_max_prev.at(i) = scores_max.at(i);  // row-hoisted, same cost
  // ... other per-row/per-element work ...
}
```

Both produce identical codegen. Users choose based on readability:
- **Standalone copy**: preferred when the copy is logically separate from adjacent work
- **Inline in apply**: preferred when fusing with other per-row operations (zero extra cost due to row-hoisting)

### frag.store -- Removed

`frag.store out, frag` is removed. Fragment-to-memory transfers use existing
mechanisms:
- `mma.store acc, shared_target` -- for MMA accumulator to shared memory
- `dma.copy` -- for shared-to-global transfers
- Direct `.at()` writes in apply blocks when needed

---

## Implementation Plan: Reusing Existing Infrastructure

The existing implementation handles all the hard parts. The migration is primarily
a parser/AST change with minimal codegen adjustment.

### What Already Works (No Changes Needed)

| Component | File(s) | Capability |
|---|---|---|
| Fragment layout inference | `lib/Target/GPU/fragment_layout_pass.cpp` | WGMMA_ACC, REPLICATED_1D, UNIFORM, CTMMA_ACC layout assignment |
| Layout propagation | `fragment_layout_pass.cpp` lines 297-360 | frag.copy and frag.apply edge propagation (fixpoint) |
| Forward index functions | `lib/fragment_layout.hpp` | `ForwardIndex(i,j)`, `ForwardThread(i,j)` per layout kind |
| Reduce codegen | `cute_codegen.cpp` Visit(FragReduce) | Thread-local loop + cross-thread AllReduce with shuffle |
| Reduce layout derivation | `FragmentLayout::MakeReduceDst()` | Derives REPLICATED_1D from source accumulator layout |
| frag.apply codegen | `cute_codegen.cpp` Visit(FragApply) | Register-direct loop, IV recovery from layout inverse |
| WGMMA-aware indexing | `cute_codegen.cpp` OpExprSTR | Emits `ForwardIndex` for `.at()` on layout-assigned frags |
| Tests | `tests/gpu/end2end/frag_apply.co`, `frag_reduce.co` | End-to-end GPU validation for all layout kinds |

### Changes Required

#### Phase 1: Parser + AST (new nodes alongside old)

| Layer | File | Change |
|---|---|---|
| Scanner | `lib/scanner.l` | Add `reduce_max`, `reduce_sum` as tokens (or reuse `__reduce_max`/`__reduce_sum`) |
| Parser | `lib/parser.yy` | Add `apply_stmt` rule: `APPLY '{' iv_list '}' IN expr '{' stmts '}'` |
| Parser | `lib/parser.yy` | Add `reduce_stmt` rule: `REDUCE_MAX '(' expr ',' expr ',' INT ')'` |
| AST | `lib/ast.hpp` | New `ApplyBlock` node (or reuse `FragApply` with multi-statement body) |
| AST | `lib/ast.hpp` | Optionally rename `FragReduce` to `Reduce` or add alias |

Design decision: we can either (a) introduce new AST nodes and keep old ones for
backward compatibility during transition, or (b) desugar the new syntax into existing
`FragApply`/`FragReduce` AST nodes during parsing so codegen is unchanged.

**Recommended: Option (b) -- desugar during parse.** This avoids duplicating codegen
and gives us the new syntax immediately with zero codegen changes.

#### Phase 2: Desugar apply -> FragApply(s)

The new `apply` body can contain multiple statements at different execution tiers.
The parser/early-sema pass splits them:

```co
apply {i, j} in acc_s.span {
  scores_max_prev.at(i) = scores_max.at(i);   // PER_ROW
  acc_s.at(i, j) = -inf;                       // PER_ELEMENT
}
```

Desugars to two internal FragApply nodes:
1. `FragApply(scores_max_prev, [](i) { return scores_max.at(i); })`
2. `FragApply(acc_s, [](i, j) { return -inf; })`

Or alternatively, the codegen for `ApplyBlock` directly emits the tiered loop
structure (per-row prologue + per-element body).

#### Phase 3: Desugar reduce_max/sum -> FragReduce

```co
reduce_max(scores_max, acc_s, 1);
```

Desugars to:
```
FragReduce(MAX, acc_s, scores_max, 1)
```

Note: operand order flips from new syntax (dst, src, dim) to existing AST (op, src, dst, dim).
This is a trivial parser-level swap.

#### Phase 4: Update FragmentLayoutPass

The layout pass currently visits `FragApply` and `FragReduce` nodes. If we desugar
to these nodes, no changes needed. If we introduce new AST nodes, add Visit overrides:

```cpp
bool FragmentLayoutPass::Visit(AST::ApplyBlock& n) {
  // Same logic as Visit(FragApply): collect body refs for propagation.
  auto target_scoped = InScopeName(n.SpanFragmentName());
  std::vector<std::string> refs;
  CollectBodyRefs(n.body, refs);
  if (!refs.empty()) apply_refs_[target_scoped] = std::move(refs);
  return true;
}
```

#### Phase 5: Codegen for multi-statement apply (row-hoisting)

The key new codegen work: emitting tiered loops for multi-statement apply blocks.

Current `Visit(FragApply)` emits:
```cuda
{ // frag.apply target
#pragma unroll
for (int __r = 0; __r < REGS; ++__r) {
  int __frag_iv_i = LogicalRowFromReg(__r, tid);
  int __frag_iv_j = LogicalColFromReg(__r, tid);
  target[__r] = <body_expr>;
}
} // frag.apply
```

New `Visit(ApplyBlock)` should emit:
```cuda
{ // apply
#pragma unroll
for (int __r_row = 0; __r_row < ROWS_PER_THREAD; ++__r_row) {
  int __frag_iv_i = LogicalRowFromRowReg(__r_row, tid);
  // -- PER_ROW statements --
  scores_max_prev[__r_row] = scores_max[__r_row];

  #pragma unroll
  for (int __r_col = 0; __r_col < COLS_PER_ROW; ++__r_col) {
    int __r = __r_row * COLS_PER_ROW + __r_col;
    int __frag_iv_j = LogicalColFromReg(__r, tid);
    // -- PER_ELEMENT statements --
    if ((bn * BLOCK_N + __frag_iv_j) > (...))
      acc_s[__r] = -inf;
  }
}
} // apply
```

For WGMMA_ACC layout:
- `ROWS_PER_THREAD = rows_per_thread` (typically 2 for 64-row tiles on 128 threads)
- `COLS_PER_ROW = regs_per_thread / rows_per_thread` (typically 16)

For UNIFORM layout:
- Single flat loop (current behavior) with row-change detection for PER_ROW hoisting

#### Phase 6: Deprecate old syntax (gradual)

1. Add compiler warning: "frag.apply is deprecated, use apply {...} in ... syntax"
2. Provide migration script/sed patterns
3. Update all test files
4. Eventually remove `FragApply` parser rule (keep AST node if desugaring uses it)

---

## Syntax Comparison: Complete Flash Attention Softmax

### Old syntax (frag.apply + frag.reduce_max/sum):

```co
frag.apply acc_s, [](i, j) {
  return ((bn * BLOCK_N + j) > (bm * BLOCK_M + wgm * WG_M + i + past_len))
    ? -inf : acc_s.at(i, j);
};
frag.copy scores_max_prev, scores_max;
frag.reduce_max acc_s, scores_max, 1;
frag.apply scores_scale, [](i) {
  return __exp2f(scores_max_prev.at(i) * scale - scores_max.at(i) * scale);
};
frag.apply acc_s, [](i, j) {
  return __exp2f(acc_s.at(i, j) * scale - scores_max.at(i) * scale);
};
frag.reduce_sum acc_s, scores_sum, 1;
frag.apply logsum, [](i) {
  return logsum.at(i) * scores_scale.at(i) + scores_sum.at(i);
};
frag.apply acc_o, [](i, j) {
  return acc_o.at(i, j) * scores_scale.at(i);
};
frag.apply acc_s_cast, [](i, j) {
  return __to<bf16>(acc_s.at(i, j));
};
```

### New syntax (apply + reduce_max/sum):

```co
apply {i, j} in acc_s.span {
  if ((bn * BLOCK_N + j) > (bm * BLOCK_M + wgm * WG_M + i + past_len))
    acc_s.at(i, j) = -inf;
  scores_max_prev.at(i) = scores_max.at(i);
}

reduce_max(scores_max, acc_s, 1);

apply {i, j} in acc_s.span {
  scores_scale.at(i) = __exp2f(scores_max_prev.at(i) * scale - scores_max.at(i) * scale);
  acc_s.at(i, j) = __exp2f(acc_s.at(i, j) * scale - scores_max.at(i) * scale);
}

reduce_sum(scores_sum, acc_s, 1);

apply {i, j} in acc_s.span {
  logsum.at(i) = logsum.at(i) * scores_scale.at(i) + scores_sum.at(i);
  acc_o.at(i, j) = acc_o.at(i, j) * scores_scale.at(i);
  acc_s_cast.at(i, j) = __to<bf16>(acc_s.at(i, j));
}
```

**Improvements:**
- Multi-statement bodies (no need for separate frag.apply per target)
- Imperative assignment (no `return`, no lambda brackets)
- Conditionals as natural `if` statements (not ternary)
- Mixed-rank access in one block (1D `.at(i)` + 2D `.at(i,j)`)
- Explicit separation of element-wise (apply) from collective (reduce)

---

## Row-Hoisting Semantics (Formal)

Inside `apply {i, j} in <span>`:

| Statement accesses | Execution tier | Codegen |
|---|---|---|
| `.at(i, j)` or bare `j` reference | PER_ELEMENT | Inner register loop |
| `.at(i)` only, no `j` dependency | PER_ROW | Outer row loop (hoisted) |
| No iterator dependency | PER_BLOCK | Execute once before loops |

**Ordering guarantee:** For each row `i`, all PER_ROW statements complete before
any PER_ELEMENT statements for that row begin.

**Error cases:**
- Statement writes a 1D fragment and reads a 2D fragment in RHS -> PER_ELEMENT
  (the `j` dependency from the read determines the tier)
- Conflicting tiers writing to the same target -> compile error

---

## Interaction with Fragment Types

The new syntax works uniformly across all fragment declaration styles:

| Declaration | Layout Kind | apply behavior |
|---|---|---|
| `frag f32[M, N] name` | WGMMA_ACC (from MMA usage) | Nested row/col unroll |
| `frag f32[M] name` | REPLICATED_1D (from reduce target) | Row-only unroll |
| `fragment f32[M, N] tile` | UNIFORM (explicit fragment decl) | Flat stride loop |
| `acc = mma.fill.f32 0.0f` | WGMMA_ACC (MMA anchor) | Nested row/col unroll |

The `apply` keyword does not care about the declaration form. It only needs the
fragment to have a layout assigned by FragmentLayoutPass. The codegen reads the layout
and emits the appropriate loop structure.

---

## .span Semantics

`fragment_name.span` returns the logical iteration domain of the fragment:

| Fragment shape | `.span` provides | Iterator count |
|---|---|---|
| `f32[M]` | `[M]` | 1 iterator (`{i}`) |
| `f32[M, N]` | `[M, N]` | 2 iterators (`{i, j}`) |

The span is the **logical shape**, not the physical register count. The compiler
translates logical iteration into physical register traversal using the fragment's
assigned layout.

Iterator count in the apply must match the fragment's rank:
- `apply {i} in scores_max.span` -- OK (1D fragment)
- `apply {i, j} in acc_s.span` -- OK (2D fragment)
- `apply {i, j} in scores_max.span` -- ERROR (rank mismatch)

Accessing other fragments within an apply is allowed if the indices are compatible:
- `scores_max.at(i)` inside `apply {i, j} in acc_s.span` -- OK (i valid for 1D)
- `acc_o.at(i, j)` inside `apply {i, j} in acc_s.span` -- OK (same 2D domain)

---

## Migration Checklist

### Test files to update:
- [ ] `tests/check/frag_apply.co` -- add new-syntax variants
- [ ] `tests/gpu/end2end/frag_apply.co` -- add new-syntax end2end tests
- [ ] `tests/gpu/end2end/frag_reduce.co` -- update to `reduce_max(dst, src, dim)`
- [ ] `tests/gpu/end2end/frag_transfer.co` -- evaluate frag.copy migration

### Benchmark/research files to update:
- [ ] `benchmark/.../v1_manual_baseline.co` -- keep as-is (old syntax reference)
- [ ] `benchmark/.../new_syntax_v2.co` -- already uses new syntax
- [ ] `research/croqtile/flash_atten_v2.co` -- migrate when syntax is implemented
- [ ] `tests/gpu/end2end/flash_atten.co` -- migrate when syntax is implemented

### Compiler files to modify:
- [ ] `lib/scanner.l` -- add tokens for `apply`, `reduce_max`, `reduce_sum`, `span`
- [ ] `lib/parser.yy` -- add grammar rules for new statements
- [ ] `lib/ast.hpp` -- new `ApplyBlock` node (or reuse `FragApply`)
- [ ] `lib/earlysema.cpp` -- validate apply rank match, no reductions inside
- [ ] `lib/Target/GPU/cute_codegen.cpp` -- row-hoisting codegen for multi-stmt apply
- [ ] `lib/Target/GPU/fragment_layout_pass.cpp` -- handle new AST nodes in layout pass

---

## Related Work

- `research/implementation/fragment_and_automap.md` -- original fragment/automap design
- `lib/Target/GPU/fragment_layout_pass.cpp` -- layout inference (fully reusable)
- `lib/fragment_layout.hpp` -- layout kinds and forward index functions
- `tests/check/frag_apply.co` -- existing codegen tests
- `tests/gpu/end2end/frag_reduce.co` -- reduce end2end validation
- `benchmark/.../new_syntax_v2.co` -- target syntax reference file
