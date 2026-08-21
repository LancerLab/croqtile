# Plan: Canonicalize the Legacy `foreach` Range-Source Form into an Explicit Iteration Variable

Issue: https://github.com/LancerLab/croqtile/issues/5

Status: PLAN ONLY (for review, no code changes yet).

---

## 1. Goal

Lower the legacy `foreach` range-source sugar form

```
foreach a(lb:ub[:step]) { ... }
```

into an explicit iteration-variable form

```
foreach __iv_a = a(lb:ub[:step]) { ... }
```

Only the **sugar/legacy form** is canonicalized. The explicit form
`foreach c = a(lb:ub)` already has a named iteration variable and is left
untouched.

`LoopRange` is **renamed** to `RangeExpr` (a single node, not split into two).
The iteration variable and the range source remain two fields of that one node:

- `source` (the `with`-declared bounded variable, e.g. `a`) — the bound.
- `iv` (the iteration variable, e.g. `__iv_a`) — a plain integer that may step
  by more than 1, so it is NOT an induction variable.

This is a normalization (AST-to-AST) change, not a new language feature. It must
be behavior-preserving: existing tests pass unchanged.

---

## 2. Current State (verified)

### 2.1 `LoopRange` (lib/ast.hpp:3838)

- `rv`: range-source variable (a `with`-declared bounded variable; not visible in
  the body). This is the bound.
- `iv`: iteration variable visible in the body. Only set by the parser when an
  explicit local name is written (`foreach local = source(...)`).
- `lbound` / `ubound` / `step` / `scope_predicate`.
- Constructors:
  - `LoopRange(l, i)` -> `rv = i`, `iv = nullptr` (sugar form).
  - `LoopRange(l, i, lb, ub, s=1)` -> sugar with bounds.
  - `LoopRange(l, local, source, lb, ub, s=1)` -> explicit form.
- `GetIV()` = `iv ? iv : rv`; `GetIVName()` = `GetIV()->name`; `GetRVName()` =
  `rv->name`; `GetRV()` = `rv`.
- `HasExplicitIV()` = `iv && iv.get() != rv.get()`.
- `CloneImpl` already copies both `rv` and `iv` (so the rewrite survives clones).

### 2.2 Parser (lib/parser.yy:1950-1990, `range_expr`)

- 3 explicit productions produce `LoopRange(l, id1, id3, lb, ub, step)`.
- 3 sugar productions produce `LoopRange(l, id, lb, ub, step)` with `iv = nullptr`.

### 2.3 Early semantics (lib/earlysema.cpp:3230, `Visit(ForeachBlock)`)

- Only registers the IV via `ReportErrorWhenViolateODR(...)` when
  `HasExplicitIV()` is true. The sugar form never registers an IV, because `iv`
  is null.
- `Visit(LoopRange)` (lib/earlysema.cpp:3217) validates `lbound`/`ubound` are
  scalar integers.

### 2.4 Type inference (lib/typeinfer.cpp)

- `AfterVisitImpl` (lib/typeinfer.cpp:188): for a `ForeachBlock`, does
  `SetNodeType(*range->GetRV(), GetSymbolType(GetRVName()))`. It does NOT define
  an iteration-variable symbol.
- `BuildRangePredicate` (lib/typeinfer.cpp:99): builds the loop predicate with
  `iv = sbe::sym(InScopeName(n.GetRVName()))` and bounds `lb = RV.lb + lbound`,
  `ub = RV.ub + ubound`. Note the predicate's symbolic variable is currently the
  RV name (this matters for codegen unification, section 5).

### 2.5 Symbol replacement in the Normalizer

The symbol table is rebuilt per pass up to `SemaChecker`
(`lib/pipeline.cpp:261-318`); each stage carries its own table. Canonicalization
runs **inside the `Normalizer`** (early normalization). There it must:

1. Create the new iteration-variable symbol `__iv_a` in the foreach scope with a
   `BoundedIntegerType` (the correct *kind*, see section 4). The bound **values**
   are not accurate yet — at this stage any bounded type is just the
   structural `BoundedIntegerType` with placeholder bounds.
2. Replace the old symbol (the sugar form's `a`, which doubled as both the bound
   and the iteration variable) with `__iv_a` wherever it is used as the
   iteration variable inside the body.

Downstream stages (ShapeInference, TypeInference, ...) then see `__iv_a` in the
AST and naturally rebuild it in their own tables with the finalized type.

### 2.6 Bound access `a(-1)`

`a(-1)` is NOT a `DataAccess`. It parses as `Expr(op = "dimof", L = Identifier(a),
R = IntIndex(-1))`, and is disambiguated to `Op::GetIth` in early semantics when
the base is a `BoundedType` (lib/earlysema.cpp:280-293).

In the sugar form, `a` plays a dual role: the range source (bound) and the
iteration variable. Canonicalization splits this: the bound keeps the name `a`,
the iteration variable becomes `__iv_a`. Therefore the body rewrite must
distinguish the two uses of `a`:

- `a` used as the iteration variable (a bare value reference) -> rewrite to
  `__iv_a`.
- `a` used as a bound access (`a(-1)`, `ubound(a)`, `#a`, `a#b`, ...) -> keep
  `a`.

This is the only body rewrite required, and it is driven by the symbol split,
not by a separate bound-tracking feature.

---

## 3. Canonicalization (the core of this change)

### 3.1 Target AST shape

`LoopRange` is **renamed** to `RangeExpr` (same fields, no structural split):

| Source | Before | After |
|--------|--------|-------|
| `foreach a(lb:ub)` | `LoopRange(rv=a, iv=null)` | `RangeExpr(rv=a, iv=__iv_a)` |
| `foreach c = a(lb:ub)` | `LoopRange(rv=a, iv=c)` | `RangeExpr(rv=a, iv=c)` (unchanged) |

Only the legacy form is rewritten: it gains an explicit iteration variable
`__iv_a`, and its body's value references to `a` become `__iv_a`. The explicit
form is left as-is.

### 3.2 AST node

Rename `LoopRange` -> `RangeExpr` in `lib/ast.hpp` (and `lib/ast.cpp`). Keep the
existing fields (`rv`, `iv`, `lbound`, `ubound`, `step`, `scope_predicate`) and
accessors (`GetIV/GetIVName/GetRV/GetRVName/HasExplicitIV/BoundIsMutated`).
Optionally rename `rv` -> `source` for clarity; this is cosmetic and can be a
follow-up. No new node, no let-binding node, no two-node split.

### 3.3 Where the pass lives

As a **sub-pass of the `Normalizer`** (lib/normalize.hpp:1663). Add a
`ForeachCanon : NormBase` member and register it **first** in the group:

```cpp
class Normalizer : public VisitorGroup {
private:
  ForeachCanon fcanon;   // NEW: runs before the others
  CompoundNorm comp;
  ParaByFiller filler;
  LoopNorm ln;
  PredNorm pn;
public:
  Normalizer() : VisitorGroup("norm", fcanon, comp, filler, ln, pn) {}
};
```

It must run **before `LoopNorm`**, because `LoopNorm::Visit(MultiNodes)` reads
`GetIV()`/`GetRV()` and their types; the fresh `__iv_a` must be in place first.

### 3.4 The rewrite algorithm

For each `foreach` range clause that is the legacy form (`!HasExplicitIV()`):

```cpp
bool Visit(AST::ForeachBlock& n) override {
  for (auto& rn : n.GetRanges()) {
    auto rng = cast<AST::RangeExpr>(rn);
    if (rng->HasExplicitIV()) continue;       // explicit form: skip

    std::string name  = rng->GetIVName();     // 'a'
    std::string fresh = "__iv_" + name;

    // 1) set the iteration variable (rename only for the sugar form)
    auto iv = AST::Make<AST::Identifier>(rng->LOC(), fresh);
    rng->iv = iv;

    // 2) create the new symbol in the foreach scope with a BoundedIntegerType
    //    (kind only; bound values are resolved later by valno/shapeinfer)
    DefineIterVar(n, fresh, rng);             // section 4

    // 3) rewrite body: bare 'a' (iteration variable) -> '__iv_a',
    //    skip bound-access forms a(-1) / ubound(a) / #a / a#b / ...
    RewriteBody(n.stmts, name, fresh);
  }
  return true;
}
```

`RewriteBody(node, old, fresh)` is a recursive walk (same pattern as
`CompoundNorm` / `PredNorm::NormExpr`) with this skip-list:

| Context | Action |
|---|---|
| `Expr(op == GetIth, L = Identifier(old), R = IntIndex)` i.e. `a(-1)`, `a(0)` | skip `L` (bound), recurse into `R` |
| `Expr(op == GetUBound, R = Identifier(old))` i.e. `ubound(a)` | skip `R` (bound) |
| `Expr(op == UBound/UBoundAdd/UBoundSub/UBoundScale/UBoundDiv/UBoundMod, operand = Identifier(old))` i.e. `#a`, `a#b` | skip the bounded operand |
| `Identifier(old)` elsewhere (bare value, `DataAccess`/`elemof` index, `Call` arg) | rewrite to `fresh` |

The scope predicate is re-bound to the new iteration variable and its
lbound/ubound (section 4.2): `scope_predicate = lb <= __iv_a < ub`.

### 3.5 Companion fixes

- `LoopNorm::Visit(MultiNodes)` (lib/normalize.hpp:54) reads `GetIV()`/`GetRV()`
  and `GetIV()->GetType()`; after canonicalization `GetIV()` is `__iv_a` (typed
  in section 4), so it continues to work, but confirm it uses the iteration
  variable's (not the range source's) type where appropriate.
- `LoopRange`'s rename to `RangeExpr` propagates to `accept`, `CloneImpl`,
  `Print`, and every visitor/cast site.
- EarlySemantics `Visit(ForeachBlock)` / `Visit(LoopRange)` (lib/earlysema.cpp)
  stay as-is: they validate the pre-canonicalization form.

### 3.6 Acceptance criteria

- `-e` (post-parse) is unchanged.
- `-pa=NORM` shows the legacy form canonicalized:
  `Iteration variables: __iv_a (source: a)`, and the explicit form unchanged:
  `Iteration variables: c (source: a)`.
- Legacy-form body value references to `a` are `__iv_a`; `a(-1)`/`ubound(a)`
  still resolve to the bound.
- The explicit form is byte-for-byte untouched by this pass.

---

## 4. Iteration-variable type derivation

At normalization time the `__iv_a` symbol is created with a `BoundedIntegerType`
— the correct **kind**, but with not-yet-accurate bound values. Before shape
inference, any type that has bounds is of this `BoundedIntegerType` kind; the
accurate symbolic bound values are computed by the valno/shape-inference process,
which then re-sets the type with resolved bounds.

The final bounded type derives from the range source's `BoundedType` plus the
range mutation:

```
lb'   = source.lb + (lbound ? val(lbound) : 0)
ub'   = source.ub + (ubound ? val(ubound) : 0)
step' = range.step
type  = BoundedIntegerType(lb', ub', step')
```

### 4.1 Valno / shape inference resolve the IV's bounds

`ShapeInference` (together with value numbering) must visit the
`foreach`/`RangeExpr` and compute the iteration variable's **lower and upper
bounds** (applying the `lb:ub:step` mutation to the source's `BoundedType`), then
re-set the `BoundedIntegerType` on `__iv_a` with the accurate symbolic bound
values.

Once that bounded type is correct, its `lbound`/`ubound` `ValueItem`s ARE the
symbolic values. There is **no need** to record lbound/ubound separately in
`OptimizedValues` — the type already carries them (this replaces the earlier
`VNK_LBOUND` / `OptimizedValues`-lbound idea; see section 6).

### 4.2 Scope predicate

The scope predicate is always bound to the iteration variable's lbound/ubound:

```
scope_predicate = oc_ge(__iv_a, lb) && oc_lt(__iv_a, ub)
```

where `lb`/`ub` come from `__iv_a`'s `BoundedIntegerType`. This matches
`TypeInference::BuildRangePredicate` (lib/typeinfer.cpp:99), except the symbolic
variable is now the iteration variable `__iv_a` (not the range source `a`).

### 4.3 New factory (lib/types.hpp:3088)

The current factories hard-code `lb = 0` and `step = 1`:

```cpp
inline ptr<BoundedIntegerType> MakeBoundedIntegerType(int ub) {
  return std::make_shared<BoundedIntegerType>(sbe::nu(0), sbe::nu(ub));
}
```

Add:

```cpp
inline ptr<BoundedIntegerType> MakeBoundedIntegerType(const ValueItem& lb,
                                                      const ValueItem& ub,
                                                      int step = 1) {
  return std::make_shared<BoundedIntegerType>(lb, ub, step);
}
```

This is the "derive the exact (lb, ub, step) instead of assuming lb = 0" fix.
`step` already lives on `BoundedIntegerType` (lib/types.hpp:2061); no new type is
required.

---

## 5. Codegen reads loop bounds from the bounded type

After canonicalization + shape inference, the iteration variable `__iv_a` carries
a `BoundedIntegerType` with the correct `lbound`/`ubound`/`step`. Codegen should
therefore derive the loop bounds **from that type**, not from a separate
`OptimizedValues` record.

For each backend (GPU, AMDGPU, CPU, GCU — exact file paths confirmed before
editing):

- Loop-var emission: use the iteration variable's name directly (already `__iv_a`
  for the legacy form; `c` for the explicit form).
- Loop bounds (`lb`, `ub`, `step`): read from the iteration variable's
  `BoundedIntegerType` (`GetLowerBound()`, `GetUpperBound()`, `GetStep()`).
- `within_map` keying: key by the iteration variable name (or range source name)
  consistently; remove the ad-hoc `"__iv_" + name` re-derivation to avoid
  `__iv___iv_a`.
- `BuildRangePredicate` (lib/typeinfer.cpp:99): switch its symbolic variable from
  `InScopeName(GetRVName())` to the iteration variable `__iv_a`, so the emitted
  loop variable matches the predicate.

This phase is behavior-neutral only AFTER sections 3 and 4 land. It can be done
backend-by-backend and is gated by the GCU `__iv_*` CHECK lines (which currently
guard against double-prefix).

---

## 6. No separate lower-bound recording needed

Earlier thinking added `VNK_LBOUND` to `VNKind` and an lbound slot to
`OptimizedValues`. That is **not needed** once the iteration variable's
`BoundedIntegerType` is set correctly (section 4.1): the type's `lbound`/`ubound`
`ValueItem`s are the symbolic values, and codegen reads the bounds from the type
(section 5).

So the shape-inference work is limited to *computing* the IV's lb/ub and setting
the bounded type. No `OptimizedValues` schema change, no `VNKind` change.

### 6.1 Valno (issue point 4)

`ValueNumbering` (lib/valno.cpp) has no `ForeachBlock`/`RangeExpr` visitor; it
simplifies expressions/opcodes only. It does NOT infer iteration-variable bounds,
and needs no change. Bounds flow through the bounded type and the scope predicate.

---

## 7. File change list

Phase 1 (canonicalization):

- `lib/ast.hpp` / `lib/ast.cpp` — rename `LoopRange` -> `RangeExpr` (fields and
  accessors unchanged; optional `rv` -> `source` rename).
- `lib/normalize.hpp` — add `ForeachCanon : NormBase` sub-pass; register it first
  in `Normalizer`; update `LoopNorm` for the renamed node.
- `lib/types.hpp` — add `MakeBoundedIntegerType(lb, ub, step)` factory.
- `lib/shapeinfer.cpp` — compute the iteration variable's lb/ub (with the
  `lb:ub:step` mutation) and set its `BoundedIntegerType`.
- `lib/typeinfer.cpp` — `BuildRangePredicate` uses the iteration variable
  `__iv_a` as the symbolic variable.

Phase 2 (codegen reads bounds from type):

- GPU codegen, AMDGPU codegen, CPU codegen, GCU codegen (paths per tree).

(No Phase 3: the earlier `OptimizedValues`/`VNK_LBOUND` lower-bound recording is
dropped per section 6.)

---

## 8. Test plan

- Parse dump: `choreo -e file.co` unchanged.
- Normalization: `choreo -pa=NORM file.co` asserts
  `Iteration variables: __iv_a (source: a)` for the legacy form, and
  `Iteration variables: c (source: a)` for the explicit form (unchanged).
- Explicit form end-to-end: a codegen test for `foreach c = a(...)` (currently
  untested; exercises the `within_map.at()` fix).
- Bound access: `a(-1)` / `ubound(a)` inside the `with` scope still read the
  bound after the body rewrite.
- Double-prefix regression: GCU `__iv_*` CHECK lines must not become
  `__iv___iv_*`.
- Full suite: `make test-debug`.
- OSS gate: `make oss-scan` (and `make oss-scan-staged` before commit).

---

## 9. Implementation order

1. Phase 1 (sections 3, 4) — rename `LoopRange` -> `RangeExpr`, add the
   `ForeachCanon` sub-pass in the `Normalizer`, compute the IV's bounded type in
   shape inference, and re-point `BuildRangePredicate` at `__iv_a`. Verify with
   `-pa=NORM` and `make test-debug` (behavior-neutral so far).
2. Phase 2 (section 5) — codegen reads loop bounds from the bounded type,
   backend by backend.

Commit message (conventional, <72 chars): e.g.
`feat: canonicalize foreach sugar into explicit iteration variable`.
