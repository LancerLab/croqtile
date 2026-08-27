# Plan: gcu400 `shared<group>` (group-shared) storage tier

This is the implementation plan for the new `shared<group>` storage tier on
gcu400, as designed in `Documents/design/storage-scope.md`. It replaces the
"stale" per-lane DMA fix (the `thread_replicate` note and the
`block_level`-based `local_dte`/`private_dte` switch).

## Overview of changes

The feature threads one new `Storage` value (`GROUP_SHARED`) through the whole
compiler, then removes the per-lane workarounds that are no longer needed.

| Area | File(s) | Change |
|------|---------|--------|
| Keyword / enum | `lib/parallel_level.hpp` | add `Storage::GROUP_SHARED` |
| String | `lib/types.hpp` | `GetStringFrom(Storage)` -> `"groupshared"` |
| Parsing | `lib/parser.yy` | accept `shared<group>` via a scoped-storage rule |
| Sema | `lib/earlysema.cpp` | reject `shared<group>` on targets without GROUP |
| Allocation | `lib/mem_reuse.cpp` | merge `{LOCAL, GROUP_SHARED}` into one VDMEM heap |
| Check | `lib/memcheck.hpp` | add GROUP_SHARED to checked storage + joint budget |
| Cap | `lib/Target/GCU/gcu_target.hpp` | `GetMemCapacity(GROUP_SHARED)` = 0xE0000 |
| DTE | `lib/Target/GCU/topscc_codegen.cpp` | `DMATypeSTR`: GROUP_SHARED -> `local_dte` |
| Codegen | `lib/Target/GCU/topscc_codegen.cpp` | decls, mdspan, sync, fence, remove hack |
| Fence table | `lib/Target/GCU/gcu_fence.hpp` | GROUP_SHARED VDMEM directions |
| Fence wire fmt | `lib/types.hpp` `FenceKindFromName` | parse `"groupshared"` space |
| CoIR (MLIR) | `tools/coir/**` | `TensorMemorySpace::GroupShared` + emit/DTE/fence |
| Tests | `tests/`, `tools/coir/tests/` | parser, type, codegen, fence, memcheck, coir |

---

## Phased delivery

The work is split into seven independently reviewable and buildable phases.
Each phase ends in a review checkpoint: the tree still builds and tests pass,
so a phase can be reviewed and merged without waiting for later phases.

| Phase | Scope | Tests added (see Section 9) |
|-------|-------|-----------------------------|
| 1 | Enum + string form + fence wire-format round-trip | `tests/standalone/gcu/gcu_fence_table_test.cpp` -> `GroupSharedFenceWireRoundTrip` (DONE) |
| 2 | Parser `shared<group>` + sema gating (reject param / non-GROUP targets) | `tests/gcu/check/group_shared_ok.co`, `group_shared_no_group.co`, `group_shared_illegal.co` |
| 3 | Allocation merge + buffer-limit check + `GetMemCapacity` | `tests/gcu/check/mem_usage_check_gcu400_group_shared.co`, `tests/gcu/codegen/topscc/mem_reuse_group_shared.co` |
| 4 | GCU codegen (DTE / decl / mdspan / sync / fence) + remove stale per-lane hack | `tests/gcu/codegen/topscc/group_shared.co`; modify `dma-fence.co`, `sync.co`, `gcu_fence_table_test.cpp` |
| 5 | Other-target `Storage` switch audit (default arms) | none (compile-clean gate: `make build` + `make test`) |
| 6 | CoIR (MLIR) dialect enum, lowering, emit, fence, addr-space | `tools/coir/tests/gcu/{irgen/group-shared.co, emit/group-shared-emit.mlir, transform/group-shared-memreuse.mlir}` |
| 7 | Migration of existing `local`->`shared<group>` buffers + end-to-end | `tests/gcu/end2end/group_shared.co` |

Phases 1-3 are behavior-neutral (no emitted code changes) and can be reviewed
in isolation. Phase 4 is the first phase that changes generated code. Phase 6
is fully independent of Phases 2-5 and can proceed in parallel.

---

## 1. Keyword and enum

### 1.1 `lib/parallel_level.hpp`

Add `GROUP_SHARED` to the `Storage` enum, positioned **after** `SHARED` and
**before** `GLOBAL`:

```cpp
enum class Storage { REG, LOCAL, SHARED, GROUP_SHARED, GLOBAL, NODE,
                     DEFAULT, NONE };
```

Positioning matters: `topscc_codegen.cpp` line ~2821 tests
`dma_sto < Storage::GLOBAL` to mean "on-chip". Placing `GROUP_SHARED` before
`GLOBAL` keeps that predicate correct without edits.

Comment it in scope terms:

```cpp
GROUP_SHARED /* shared<group>: visible within a GROUP (shared by its THREADs) */
```

### 1.2 `lib/types.hpp` `GetStringFrom`

Add the string form with the single-word, underscore-free spelling
`"groupshared"`:

```cpp
{Storage::GROUP_SHARED, "groupshared"},
```

The underscore-free spelling is mandatory, not cosmetic. `FenceKind::Name()`
serializes a fence as `STR(space) + "_" + STR(entity) + "_" + STR(order)`
(e.g. `shared_DMA_RELEASE`), and `FenceKindFromName` (also `lib/types.hpp`)
round-trips it by splitting on `_`. A space named `group_shared` would
resplit to `["group", "shared", "DMA", "RELEASE"]` and deserialize as
`space="group"`, falling through to `Storage::NONE` and silently dropping the
fence. `"groupshared"` keeps the wire format intact with a single new branch
in `FenceKindFromName`:

```cpp
else if (space == "groupshared")
  k.space = Storage::GROUP_SHARED;
```

The user-facing keyword stays `shared<group>`; `GetStringFrom` is only the
internal string form (fence wire format, diagnostics, AST dumps), so the
spelling has no effect on source syntax. `shared<group>` is an input form
produced only by the parser; it is not a round-trip string.

### 1.3 Lexer (`lib/scanner.l`)

No new token required. `shared` already returns `Storage::SHARED`, and
`group` already returns `ParallelLevel::GROUP`. The bracket form is recognized
in the grammar (Section 2), so the lexer is untouched.

---

## 2. Parsing

### 2.1 Scoped-storage non-terminal (`lib/parser.yy`)

Introduce a `scoped_storage` non-terminal that accepts either a bare storage
keyword or the `shared<group>` bracket form, and produces a `Storage` value:

```cpp
scoped_storage
    : STORAGE                     { $$ = $1; }
    | SHARED LT PBLEVEL GT        {
        if ($3 == ParallelLevel::GROUP)
          $$ = Storage::GROUP_SHARED;
        else {
          Parser::error(@3,
              "only 'shared<group>' is a valid scoped storage; "
              "use bare 'local'/'shared'/'global' for other scopes.");
          $$ = Storage::SHARED;
        }
      }
    ;
```

Notes:

- `SHARED` here is the existing `STORAGE` token carrying value
  `Storage::SHARED`; we can keep using `STORAGE` in the alternative and test
  `$1 == Storage::SHARED` instead. Use whichever reads cleanest, but the rule
  must reject `local<...>`, `global<...>`, and `shared<block>`.
- `<` and `>` after a storage keyword only appear in this rule, never in an
  expression context, so there is no grammar ambiguity with the less-than
  operator.
- Register the semantic type: `%nterm <Choreo::Storage> scoped_storage`.

### 2.2 Wire `scoped_storage` into the storage positions

Replace the bare `STORAGE` reference with `scoped_storage` in these rules so
`shared<group>` works everywhere a storage qualifier is expected:

- `named_spanned_decls : STORAGE mdspan_as_type spanned_decls`
  (spanned/array buffers: `shared<group> f32 [M,N] x`).
- `STORAGE named_scalar_decls` (scalar buffers).
- `storage_qual : STORAGE` (used by events and `buffer.map`).
- `chunkat_or_storage_or_select : ... | STORAGE` (DMA destination `=> shared<group>`).
- `sync_stmt : SYNC DOT STORAGE` (optional; see Section 7).
- `fence_stmt : SYNC DOT FENCE COL PBLEVEL LT STORAGE GT` (optional; see Section 7).

`param_storage` (kernel parameters) must keep rejecting the on-chip tiers. Add
`GROUP_SHARED` to its existing rejection of `SHARED`/`LOCAL`.

---

## 3. Semantic validation (`lib/earlysema.cpp`)

After a declaration's storage is propagated (`n.GetMemory()->Get()`), add a
check: `shared<group>` (`Storage::GROUP_SHARED`) is only valid when the target
has a GROUP parallel level. Reuse `TargetHasLevel(ParallelLevel::GROUP)` /
`GetParallelLevels()` from the target interface, mirroring how other
level-dependent features are gated.

- gcu400/450/500: allowed.
- gcu300, GPU (cute): rejected with a clear diagnostic
  ("`shared<group>` requires a GROUP level; this target has no GROUP level").

This keeps gcu300 and GPU code paths free of the new category, so their codegen
switches can leave `GROUP_SHARED` under `default:` (unreachable) without new
code.

---

## 4. Allocation (`lib/mem_reuse.cpp`)

`local` and `shared<group>` must be allocated from one merged VDMEM heap.

### 4.1 Pool mapping helper

Add a small helper used by the allocator:

```cpp
static Storage AllocPoolOf(Storage sto) {
  return sto == Storage::GROUP_SHARED ? Storage::LOCAL : sto;
}
```

### 4.2 `MemReuse::ProtoType`

The iteration `for (auto sto : {Storage::LOCAL, Storage::SHARED})` and the
membership test `ma.buf_sto.at(sname) != sto` currently key everything on the
raw storage. Change them to key on `AllocPoolOf(...)`:

- Iterate pools `{Storage::LOCAL, Storage::SHARED}` (unchanged pool set).
- When deciding whether a buffer belongs to the current pool, compare
  `AllocPoolOf(ma.buf_sto.at(sname)) != sto`.

This routes `GROUP_SHARED` buffers into the LOCAL heap and interference graph
while leaving their `Storage` value untouched for codegen (so they still emit
`__local__` + `local_dte`).

### 4.3 `MemReuse::ApplyMemOffset` / thread-replicate removal

Delete the `thread_replicate` note handling and any per-THREAD offset
multiplication. With the new category, per-lane partitioning is done by
`private_dte` natively (see Section 6), so no manual replication is needed.

### 4.4 `lib/mem_reuse.hpp`

Remove the `thread_replicate` tag if it is declared/used only for the old hack
(confirm no other consumer remains).

---

## 5. Buffer-limit check

### 5.1 `lib/memcheck.hpp` `tocheck_storage`

Add `GROUP_SHARED` to the set of checked storage classes (currently
`{LOCAL, SHARED}` optionally `+GLOBAL`).

### 5.2 Joint budget

Update both the compile-time joint check and the runtime check list so
`GROUP_SHARED` is counted on the replicated (per-SIP) side:

    local_total     = usage[LOCAL]         * replicas_per_pool
    group_shared_total = usage[GROUP_SHARED] * replicas_per_pool
    shared_total    = usage[SHARED]
    local_total + group_shared_total + shared_total <= pool_bytes

- Compile-time: the `ct_tot_mem_usage` joint check (~`memcheck.hpp:141-175`).
- Runtime: `rt_joint_mem_usage_check_list` snapshot (currently fires on
  `LOCAL || SHARED`) must also fire on `GROUP_SHARED`, and the emitted
  runtime check must include the GROUP_SHARED term.

### 5.3 `lib/Target/GCU/gcu_target.hpp` `GetMemCapacity`

Add the case (the current switch has `default: choreo_unreachable`):

```cpp
case Storage::GROUP_SHARED: return 0xE0000; // 896 KB VDMEM, per SIP
```

`GetLocalSharedPool()` needs no change: `replicas_per_pool` and `pool_bytes`
already describe the shared VDMEM pool, which now backs both `local` and
`shared<group>`.

---

## 6. DTE and storage codegen (`lib/Target/GCU/topscc_codegen.cpp`)

### 6.1 `DMATypeSTR`

Replace the `block_level` switch for `Storage::LOCAL` with an unconditional
per-storage mapping:

```cpp
if (CCtx().GetArch() == "gcu400") {
  switch (sto) {
  case Storage::GLOBAL:       return "tops::shared_dte"; // to confirm
  case Storage::SHARED:       return "tops::shared_dte";
  case Storage::GROUP_SHARED: return "tops::local_dte";
  case Storage::LOCAL:        return "tops::private_dte";
  default: choreo_unreachable("unsupported storage for DMA context.");
  }
}
```

The `block_level` parameter stays on `DMATypeSTR` only because gcu300 and the
fallback still use it; the gcu400 branch no longer reads it. Update the gcu300
branch to `choreo_unreachable` on `GROUP_SHARED` (guarded by sema anyway).

### 6.2 Declaration modifiers

`TopsDeviceMemory` / `HandleSharedLocal` `type_modifiers`:

```cpp
auto type_modifiers =
    sto == Storage::SHARED        ? "__shared__ "
  : sto == Storage::GROUP_SHARED  ? "__local__ __valigned__ "
  :                                 "__local__ __private__ __valigned__ ";
```

That is:

- `shared`       -> `__shared__`
- `shared<group>`-> `__local__ __valigned__` (thread-shared VDMEM)
- `local`        -> `__local__ __private__ __valigned__` (per-lane VDMEM)

This matches the topsop per-lane pattern already used in
`samples/cse/v1/op2.cpp` (`__valigned__ __local__ __private__` + `private_dte`).

### 6.3 `TopsMdsStorage`

Map the mdspan address space (`tops::AddrSpace = { Global, Shared, Private,
Local }`):

```cpp
case Storage::GLOBAL:       return "tops::Global";
case Storage::SHARED:       return "tops::Shared";
case Storage::GROUP_SHARED: return "tops::Local";
case Storage::LOCAL:        return "tops::Private";
```

### 6.4 Delete the stale per-lane hack

Remove from `HandleSharedLocal`:

- `ThreadCountExpr`, `ThreadCountConst`, `NeedThreadReplication`;
- `is_local_thread_dma`;
- the `thread_replicate` note branch and the `* ThreadCountExpr()` sizing;
- the `thread_index_expr * local_spm_elem_count` offset arithmetic.

`local` buffers now get `private_dte`-style per-lane partitioning for free; the
`__private__` modifier and `private_dte` context handle the lane slicing, so no
manual scratchpad enlargement or per-THREAD offset is needed.

---

## 7. Synchronization and fences

gcu400 fence support is already proven in-tree:
`tests/gcu/codegen/topscc/sync.co` (`CHECK-GCU4`) emits
`tcle::fence<...::L3_MEM/L2_MEM/L1_VDMEM>()` and
`tests/gcu/codegen/topscc/dma-fence.co` emits the release forms
`L2_MEM_STORE` / `L1_VDMEM_STORE`. So the VDMEM fence domain used by
`shared<group>` is supported on gcu400; the work below is wiring, not new
hardware enablement.

### 7.1 `Visit(AST::Synchronize)`

Add:

```cpp
case Storage::GROUP_SHARED:
  // group barrier: __syncsubthreads() on gcu400+, else __syncthreads()
  ...
```

On gcu400+ emit `__syncsubthreads()` (group/sip barrier), matching how `LOCAL`
currently maps. On gcu300/GPU this is unreachable (sema rejects the category).

### 7.2 `Visit(AST::Fence)` and `EmitAutoFences`

`shared<group>` lives in VDMEM, the same physical memory as `local`, so it
uses the `L1_VDMEM` fence domain. Two emission sites must learn the category:

- `Visit(AST::Fence)` (explicit `sync.fence : ... <storage>`, ~3042):
  `case Storage::GROUP_SHARED: ... tcle::fence<tcle::FenceType::L1_VDMEM>();`
- `EmitAutoFences` (~1858): add
  `case Storage::GROUP_SHARED: ft = "L1_VDMEM";` so auto-inserted release
  fences render as `L1_VDMEM_STORE`.

The bare `L1_VDMEM` (acq-rel) vs `L1_VDMEM_STORE`/`L1_VDMEM_LOAD`
(release/acquire) split is unchanged; `EmitAutoFences` already appends the
suffix from `kind.order`.

### 7.3 `lib/Target/GCU/gcu_fence.hpp` `SelectGCUDMAFences`

Add the VDMEM directions, mirroring the existing `LOCAL` rows:

```cpp
if (src == Storage::GLOBAL && dst == Storage::GROUP_SHARED)
  return {{},
          {K{Storage::GROUP_SHARED, FenceEntity::DMA, FenceOrder::RELEASE}}};
if (src == Storage::GROUP_SHARED && dst == Storage::GLOBAL)
  return {{K{Storage::GROUP_SHARED, FenceEntity::THREADS, FenceOrder::RELEASE}},
          {}};
```

`SHARED -> GROUP_SHARED` and `GROUP_SHARED -> SHARED` decompose through the
LOCAL class (both flush `L1_VDMEM`); confirm `FenceInsertion`'s multi-hop
decomposition produces the expected edges during implementation. Extend
`tests/standalone/gcu/gcu_fence_table_test.cpp` with these directions.

### 7.4 Fence wire format (`lib/types.hpp`)

`FenceKind::Name()` / `FenceKindFromName()` must round-trip the new space. The
underscore-free `"groupshared"` string (Section 1.2) plus the single
`else if` branch keeps `groupshared_DMA_RELEASE` parseable; add a unit test
for the round-trip.

---

## 8. Other targets and switches

Audit all `Storage` switches outside the gcu400 path and add `GROUP_SHARED`
to their `default:` (unreachable) arms so they compile cleanly with the new
enum value:

- `lib/Target/GPU/cute_codegen.cpp` (`CudaDeviceMemory`, sync/fence).
- `lib/Target/GPU/*`, `lib/Target/CPU/*`, `lib/Target/AMDGpu/*`,
  `lib/Target/Hetero/*`.
- Any `for (auto sto : {Storage::LOCAL, Storage::SHARED})` in non-gcu paths.

Since sema rejects `shared<group>` on these targets, the arms are unreachable
and can stay as errors/`default`.

---

## 9. Test plan (concrete, one per phase)

Every phase that changes behavior ships its tests in the same change; a
phase's review gate is "its tests are added and green". Concrete files below.

### Phase 1 (enum + string + wire format) — DONE

- **Modify** `tests/standalone/gcu/gcu_fence_table_test.cpp` — add
  `GroupSharedFenceWireRoundTrip`: asserts
  `__internal__::GetStringFrom(Storage::GROUP_SHARED) == "groupshared"` and
  that `FenceKindFromName("groupshared_DMA_RELEASE")` round-trips
  space/entity/order.

### Phase 2 (parser + sema)

- **Add** `tests/gcu/check/group_shared_ok.co` — positive: `shared<group>` in
  an array decl, a scalar decl, and a `=> shared<group>` DMA destination
  compiles on `-arch=gcu400` (`// CHECK-NOT: error:`).
- **Add** `tests/gcu/check/group_shared_no_group.co` — negative: `shared<group>`
  rejected on `-arch=gcu300` and on the GPU target (no GROUP level).
- **Add** `tests/gcu/check/group_shared_illegal.co` — negative: `shared<block>`,
  `local<...>`, `global<...>` rejected; `shared<group>` as a kernel parameter
  rejected.

### Phase 3 (allocation + budget + capacity)

- **Add** `tests/gcu/check/mem_usage_check_gcu400_group_shared.co` — `local +
  shared<group>` jointly exceed the 7 MiB VDMEM/DSM pool budget (mirrors
  `mem_usage_check_gcu400_fail.co`), asserting the failure message.
- **Add** `tests/gcu/codegen/topscc/mem_reuse_group_shared.co` — a `local` and
  a `shared<group>` buffer reuse one merged `__local__` scratchpool (single
  `__local__ unsigned char __spm_...` pool).

### Phase 4 (GCU codegen + fence table)

- **Add** `tests/gcu/codegen/topscc/group_shared.co` — `FileCheck`:
  `shared<group>` decl -> `__local__ __valigned__`; `local` decl ->
  `__local__ __private__ __valigned__`; group-shared DMA -> `tops::local_dte`;
  local DMA -> `tops::private_dte` (regardless of parallel level); mdspan space
  -> `tops::Local` vs `tops::Private`.
- **Modify** `tests/gcu/codegen/topscc/dma-fence.co` — add `GLOBAL ->
  shared<group>` and `shared<group> -> GLOBAL` cases asserting
  `tcle::fence<tcle::FenceType::L1_VDMEM_STORE>()`.
- **Modify** `tests/standalone/gcu/gcu_fence_table_test.cpp` — add
  `GlobalToGroupShared` and `GroupSharedToGlobal` table tests.
- **Modify** `tests/gcu/codegen/topscc/sync.co` — `sync : shared<group>` emits
  `__syncsubthreads()` on gcu400; a fence over `shared<group>` emits
  `L1_VDMEM`.

### Phase 5 (other-target audit)

- No new lit tests; gate is `make build` (all targets compile) and `make test`
  green, since the new `default:` arms are unreachable and compile-only.

### Phase 6 (CoIR)

- **Add** `tools/coir/tests/gcu/irgen/group-shared.co` —
  `shared<group> f32 [M,N] x` -> `!coir.tensor<MxNxf32, group_shared>`.
- **Add** `tools/coir/tests/gcu/emit/group-shared-emit.mlir` — alloc qualifier
  `__local__ __valigned__`, fence `L1_VDMEM`, group-shared DTE selection.
- **Add** `tools/coir/tests/gcu/transform/group-shared-memreuse.mlir` — memreuse
  merges `local` + `group_shared` into one `__spm_local` pool.

### Phase 7 (migration + end-to-end)

- **Add** `tests/gcu/end2end/group_shared.co` — a gcu400 kernel with a
  `shared<group>` buffer and a `local` buffer coexisting (verifies merged
  allocation + correct DTE + correct results).
- **Update** any existing `.co` under `samples/`, `tests/`, `benchmark/` that
  relied on `local` meaning "per-SIP" to use `shared<group>` explicitly.

---

## 10. Migration of existing code

Existing `local` declarations that relied on group-shared (per-SIP) semantics
must be rewritten to `shared<group>`. Search `samples/`, `tests/`, and
`benchmark/` for `__local__`-intent buffers declared as `local` and update
them. The stale `thread_replicate` note and per-lane offset logic are removed
in the same change, so any test that depended on them must be updated to use
the explicit `shared<group>` / `local` distinction.

---

## 11. CoIR (MLIR) side

The CoIR tooling (`tools/coir/`) is a parallel MLIR pipeline that shares the
Choreo AST frontend but lowers through the CoIR dialect instead of
`TopsccCodeGen`. It has its own `Storage` -> memory-space mapping and its own
emit/fence/DTE selection, all keyed on the `coir::TensorMemorySpace` enum. It
must learn `group_shared` too; the touchpoints are enumerated below.

### 11.1 Dialect enum (`tools/coir/include/Dialect/CoIR/CoIRAttrs.td`)

Add a `GroupShared` case to `TensorMemorySpace`:

```td
def CoIR_TMS_GroupShared : I32EnumAttrCase<"GroupShared", 4, "group_shared">;
```

Append it **after** `Register` (value 4) rather than re-numbering `Local`/
`Register`, to avoid churning every serialized `.mlir` under
`tools/coir/tests/` that carries a `local`/`register` space. The emitted
mnemonic `group_shared` is an MLIR enum-case string (underscore is fine here;
it is not the `lib/types.hpp` fence wire format). Add it to the
`CoIR_TensorMemorySpaceEnum` case list.

### 11.2 AST -> CoIR lowering (`tools/coir/lib/ASTIRGen/ASTCoIRGen.cpp`)

`LowerSpannedType` maps `sty->m_type` to `TensorMemorySpace`. Add:

```cpp
case Storage::GROUP_SHARED:
  memSpace = static_cast<int32_t>(coir::TensorMemorySpace::GroupShared);
  break;
```

### 11.3 GCU emit (`tools/coir/lib/CodeGen/GCU/EmitTopscc.cpp`)

- `getAllocQualifier` (~4222): return `"__local__ __valigned__ "` for
  `GroupShared` (thread-shared VDMEM), versus `__local__` for `Local`.
- `emitReuseInit` (~4238): treat `GroupShared` as a `tops::Local` mdspan space
  (not `tops::Shared`).
- `getDTEType` (~268): the current portable-alias logic keys on
  `srcMS/dstMS <= Shared` ("both global or shared"). `GroupShared` involves
  VDMEM but is **not** per-lane, so it must select the thread-shared DTE
  (`local_dte` on gcu400), not `choreo_sdte` (per-thread) nor the global/shared
  `choreo_cdte_priv`. Add an explicit `GroupShared` branch. Note the coir path
  uses the `choreo_*` portable aliases rather than raw
  `tops::local_dte`/`tops::private_dte`; confirm the correct alias maps to
  `local_dte` on gcu400.
- `emitFence` (~4455): `case coir::TensorMemorySpace::GroupShared:` ->
  `tcle::fence<tcle::FenceType::L1_VDMEM>();`.
- Local-pool carve-out (`isLocal` predicates at ~784 and ~1002, `__spm_local`
  emission at ~1019/~4330): `GroupShared` buffers also live in VDMEM and must
  carve from the same `__spm_local` pool; fold `GroupShared` into the
  `isLocal` predicate (rename to `isVdmem`).

### 11.4 GCU address-space lowering (`tools/coir/lib/CodeGen/GCU/ConvertToGCU.cpp`)

`convertTensorType` (~73) maps `ms==1` (Shared) -> workgroup and everything
else -> global. `GroupShared` must map to the GCU VDMEM (local) address space;
the exact constant is backend-specific (the file header comment lists
`0 = generic/local, 1 = global, 2 = workgroup (shared), 5 = private`). Confirm
which constant `-convert-memref-to-gcu` expects for per-SIP VDMEM and add the
mapping. This is the one item that needs a vendor/backend check during
implementation.

### 11.5 CoIR tests

See Section 9, "Phase 6 (CoIR)": `irgen/group-shared.co`,
`emit/group-shared-emit.mlir`, `transform/group-shared-memreuse.mlir`.
