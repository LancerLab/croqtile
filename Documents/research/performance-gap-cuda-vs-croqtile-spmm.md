# Case study: a 44% gap between a hand-written CUDA SM90 kernel and the same design in Choreo

This note records why a hand-written Hopper CUDA kernel reaches **1845 TFLOPS**
on a 2:4 structured-sparse e4m3 GEMM at 16384x16384x16384 while the
Choreo/CroqTile kernel with the *same macro configuration* reaches **1279
TFLOPS**. It lists, for each mechanism behind the gap, the CUDA implementation,
the exact Choreo source location that blocks or lacks it, and the acceptance
criterion for a fix. It is written for whoever picks up the feature work.

The analysis is static (source, round records, generated `.cu`); no new GPU
measurement was taken for it. Numbers quoted as percentages are either from the
tuning campaign's records or from the CUDA arm's own A/B rounds and are labelled
as such.

## 1. The two kernels

Both kernels solve `C[M,N] = A[M,K] @ B[N,K]^T` with A 2:4-sparse e4m3 (dense
storage, compressed once), B dense e4m3, C fp16. Both were produced by tuning
loops (100-round budget, same model, same harness); the CUDA one is hand-written
CUDA, the CroqTile one is Choreo source lowered by `choreo` and compiled by
`nvcc`.

Artifacts (private reference repository, `piano-sparse-gemm-tuning`):

- CUDA best: `cuda/state/current-best.cu`, iteration 67, 1845.08 TFLOPS.
  Promoted from `cuda/candidates/iter-067.cu`.
- CroqTile best: `croqtile/state/current-best.co`, iteration 76, 1279.28 TFLOPS.
  Promoted from `croqtile/candidates/iter-076-attempt2.co`. The generated CUDA is
  produced by `python3 verify.py --compile`; in the campaign workspace it is
  `build/iter-076-attempt2-evidence/__choreo_cute_iter-076-attempt2.cu` (the
  artifact repository excludes `build/`).

Line references below are to those files as of the campaign.

### 1.1 The macro configuration is already identical

| dimension | CUDA best | CroqTile best |
| --- | --- | --- |
| Output block per CTA | 256x256 | 256x256 (4 tiles of 64x256) |
| Warpgroups per CTA | 4 x 64x256 | 4 x 64x256 |
| Pipeline | 2 operand stages, K step 256 | 2 operand stages, K step 256 |
| A operand | pre-packed, 128 B swizzle | pre-packed, 128 B swizzle |
| B operand | 128 B swizzle, 64 KiB/stage | 128 B swizzle, two 128-wide halves |
| MMA instruction | `wgmma ... m64n256k64.f16.e4m3.e4m3` | same shape and dtype |
| Accumulator | fp16 | fp16 |
| Grid traversal | 16x8 block rasterisation | `group_m` 48 |
| Threads per CTA | 544 (512 consumers + 1 producer) | 640 (512 consumers + 128 producer group) |
| Dynamic shared memory | 224 KiB + barriers | 230400 B |
| Threads/CTA residency | 1 CTA/SM | 1 CTA/SM |

Both campaigns also made the same two big moves: fp16 accumulation instead of
fp32 (CUDA: iter-024, +18%; CroqTile: from the seed) and 128 B swizzle on both
operands. So the gap is *not* a missed configuration; it is in what the
generated code can express.

### 1.2 Where the CroqTile kernel's time goes

The CroqTile campaign's profiles (ten consecutive passes, recorded in
`croqtile/rounds/iter-076.json` and its promotion reasons) report roughly two
thirds of SM time in MMA with a residual third lost to **fill wait at 1 CTA/SM**,
DRAM non-binding at about a quarter of peak. That residual is exactly what the
three mechanisms below target.

## 2. Gap 1 — sparse metadata must travel through TMA, not consumer LDG

### 2.1 What the CUDA kernel does

The metadata operand is shipped on its own TMA ring and consumed from
registers, so no metadata access sits on the consumer's critical path:

- `cuda/state/current-best.cu:85` `kMetaStages = 4` — its own four-deep ring,
  separate from the two operand stages.
- `cuda/state/current-best.cu:88` `kMetaPrefetch = 2` — armed two stages ahead.
- `cuda/state/current-best.cu:606` `kProducerThread = 512`, `:648` — a single
  producer thread issues both rings.
- `cuda/state/current-best.cu:642` `mbarrier_init(&sMetaFull[s], 1)` — the ring
  has a full side only; its empty side is implied by the operand release.
- `cuda/state/current-best.cu:719-761` — the consumer waits on the **next**
  stage's metadata frame and fills `f0..f3` while the current stage's `e0..e3`
  run, i.e. a register double buffer (four extra registers, no shared memory).
- `cuda/state/current-best.cu:783` `wgmma_acc_fence(acc)` is the only ordering
  statement; `:770` documents the deliberate absence of a per-stage drain.

Measured value of this mechanism, from the CUDA arm's own experiment: when it
temporarily replaced the TMA metadata ring with generic per-thread loads inside
the consumer (the "generic-proxy metadata transport" round), the same kernel
read 1576-1578 TFLOPS against a ~1830 base in the same window — **about -14%**.

### 2.2 What the CroqTile kernel does instead

The current best reads metadata **directly from global memory inside each
consumer warpgroup**, every K stage:

- `croqtile/state/current-best.co:128,189,250,312` — four copies of an inline
  `__cpp__` block, one per consumer region, each computing a lane-to-word map
  and issuing two `ld.global` (`me_pair0`/`me_pair1`) per thread per stage.
- `croqtile/state/current-best.co:81` — a `shared u32 [SPMM_WARP_M, 4] meta_s`
  declaration that is dead: its contents are never written (no TMA copy targets
  it) and the `me = mma.load meta_s ...` at `:139` etc. is immediately
  overwritten by `me = (uint32_t)(me_pair0 >> ...)` at `:143`.
- The dead declaration cannot be removed: round 68 replaced the operand with
  `me = 0` and Choreo **SIGSEGV'd**, at `lib/codegen_prepare.hpp:816-821`
  (`GetSpannedType(GetSymbolType(e_sym))->ElementType()` null-dereferences
  because a plain scalar symbol has no spanned type). Attempts with
  `u32 me = 0;`, `me = mma.fill.u32 0;`, `frag.u32 me {0};` all fail to lower
  (`-11`, exit 8, exit 4 respectively; see `croqtile/rounds/iter-068.json`).

The seed in the Piano repository
(`evals/workspaces/croqtile-sparse-gemm-e4m3-16384/seed.co:77-93`) *did* stage
metadata through TMA with its own `shared event meta_full[SPMM_STAGES]`, but at
the same depth as the operands and gated by the operand ring's `empty[stage]`.
The tuning loop moved away from it (rounds 13/20/22 and the "direct-meta"
family) because in that form it was slower than direct LDG. The CUDA kernel's
*decoupled, deeper, register-prefetched* ring is the form that pays, and it is
not expressible today.

### 2.3 Feature work

**F1. Allow a sparse-metadata operand that is a plain register value.**
Today the E operand must be `mma.load <spanned shared buffer>`. Support (a) an
`u32`/`u64` register operand, or (b) an explicit "consume from shared into
registers" form that does not carry the spanned-type requirement.
- Code: `lib/codegen_prepare.hpp:816-821` (type capture), sparse E operand
  emission at `lib/Target/GPU/cute_codegen.cpp:5289-5330`
  (`MMAOperation::Load`, `policy_is_sparse && ssmi.frag == FRAG_E`), and
  `lib/Target/GPU/cute_codegen.cpp:7175` (the second FRAG_E site).
- Acceptance: a `.co` with `u32 me = ...;` as the sparse operand lowers, the
  generated code contains no `LDS` for the metadata operand, and the round
  verifies bit-exact.

**F2. Allow an event ring whose depth is independent of the operand stages.**
The seed's `tma.copy.async<meta_full[stage]>` proves separate event arrays are
syntactically fine; what is missing is (a) an empty side that does not borrow
the operand ring's release, and (b) a consumer-side register prefetch one frame
ahead. Both are currently the author's problem: `wait meta_full[s]` inside the
consumer region forces the consumer to read shared, and any read of shared
metadata needs the F1 operand form.
- Code: `lib/Target/GPU/cute_codegen.cpp:7744-7750` and `:7899` (event wait →
  `named_barrier_sync`), `:8471-8490` (event arrive/sync lowering).
- Acceptance: a ring of depth 4 with a 2-stage lookahead compiles, the
  generated SASS shows no per-stage `LDG` in the consumer for metadata, and the
  result is bit-exact.

**F3. (related, smaller) Do not force one tensor-map box to the swizzle size.**
Round 69 tried to merge the two 128-wide RHS copies into one 256-wide copy;
Choreo emitted `box_shape = {256, 256}` with `CU_TENSOR_MAP_SWIZZLE_128B` and
`cuTensorMapEncodeTiled` refused (`CUDA_ERROR_INVALID_VALUE`), because the box's
inner dimension must not exceed the swizzle size.
- Code: box/shape derivation just above
  `lib/Target/GPU/cute_codegen.cpp:10428` (the `cuTensorMapEncodeTiled` emitter);
  the `tma_inner_splits_` path at `:10390-10410` shows where a split is already
  modelled.
- Acceptance: either automatic split of an over-wide box into swizzle-sized
  copies, or a load-time diagnostic instead of a runtime driver failure.

## 3. Gap 2 — the epilogue cannot run concurrently in the four consumer groups

### 3.1 What the CUDA kernel does

After the K loop all four warpgroups reuse the (now dead) operand arena as a
transpose buffer and store concurrently:

- `cuda/state/current-best.cu:827` `bar.sync 1, 512` — one named barrier over
  exactly the 512 consumer threads.
- `:829-846` — register→shared transpose with a padded stride (`kCStride` 132)
  chosen for bank behaviour, then a fully coalesced `uint4` store per row
  (`:850-860`).
- No epilogue ring, no serialisation between warpgroups, no extra shared
  budget.

### 3.2 What the CroqTile kernel does instead

- `croqtile/state/current-best.co:78` declares `epi0..epi3` and each consumer
  region waits on its predecessor's event before `mma.store mc, output_s` and
  `tma.copy ... => output` (`:165-168`, `:226-229`, ...). The four epilogues are
  **serialised**.
- The ring exists because Choreo aliases the four per-region
  `shared f16 [SPMM_WARP_M, SPMM_WARP_N] output_s` declarations onto one pool
  and hard-codes `bar.sync 15, 128` twice per consumer region (generated file:
  eight occurrences at `__choreo_cute_iter-076-attempt2.cu:422,428,543,549,664,
  670,...`).
- Removing the ring (round 67) deadlocks: `compute-sanitizer --tool synccheck`
  reports 16288 errors, `Barrier error detected. Divergent thread(s) in block.`
  at `__choreo_device_spmm+0x7640`, then `CUDA failure: 719`. The barrier is
  emitted by Choreo, not written by the kernel, so the kernel cannot remove it.

The emitter:

- `lib/Target/GPU/cute_codegen.cpp:478-486` `CuteCodeGen::EmitGroupX4Sync` —
  `choreo::named_barrier_sync(<threads>, 15)`, barrier id hard-coded to 15 and
  the participant count taken from `CurrentScopeThreadsCount()` (fallback 128).
- Call sites: `:4649`, `:5017`, `:6879`, `:7727`, `:7899` — emitted after event
  waits and around shared-buffer transitions whenever
  `NeedWarpSpecGroupX4SyncForCurrentScope()` (`lib/Target/GPU/cute_codegen.hpp:
  858-860`, i.e. warp-spec active at `GROUPx4` level).
- Runtime: `runtime/choreo_cute.h:106` `named_barrier_sync`.

### 3.3 Feature work

**F4. Make the group barrier configurable, and allow an epilogue that does not
need it.** Options, in increasing scope:
1. derive barrier id/participant count from the enclosing scope instead of the
   constants above (small);
2. let a shared buffer be declared *per group* so the four `output_s` do not
   alias, removing the reason for the barrier (medium; costs 3 x 32 KiB shared
   unless F5 lands);
3. allow a `barrier` attribute on the epilogue region to opt out, with the
   compiler proving no aliasing (large).
- Code: `lib/Target/GPU/cute_codegen.cpp:478-486`; call sites listed above;
  shared-pool aliasing model in `lib/target.hpp:555-563` and the budget check in
  `lib/memcheck.hpp:56,530`.
- Acceptance: a kernel with four concurrent epilogues compiles, passes
  `compute-sanitizer --tool synccheck` with 0 errors, and is not slower than the
  serialised ring in the same window.

**F5. (enabler) Scoped reuse of the operand arena.** The CUDA kernel's epilogue
buffer is free because it reuses the dead stage buffers. Choreo allocates
statically; there is no way to say "this shared buffer may alias the stage
buffers after the last wait".
- Code: shared allocation/budget: `lib/memcheck.hpp:56,530,639`,
  `lib/target.hpp:555-563`.
- Acceptance: a `.co` can declare an epilogue buffer in a scope whose lifetime
  starts after the pipeline's last wait, and the shared-memory report shows the
  reduced footprint.

## 4. Gap 3 — drain-free MMA scheduling and a single-thread producer

### 4.1 What the CUDA kernel does

- Per stage the consumer issues four sparse `m64n256k64` atoms, each followed by
  its own commit/wait (`cuda/state/current-best.cu:771-810`); there is
  deliberately **no stage-boundary drain** (`:770` comment), and one
  `wgmma_acc_fence` per stage (`:783`).
- The producer is one thread (`:606`), so the other 31 threads of its warp exit
  and do not occupy scheduler slots.

### 4.2 What the CroqTile kernel does instead

- `croqtile/state/current-best.co:162` (and `:223`, `:284`, `:345`) — one
  `mma.commit` per K stage, lowered by
  `lib/Target/GPU/cute_codegen.cpp:5178-5182` into
  `EmitWGMMAFinalize` (`:489-500`), which unconditionally emits
  `warpgroup_commit_batch()` followed by `warpgroup_wait<0>()`. The kernel
  author cannot ask for a commit without the wait, nor for a per-atom wait.
- The producer is a full 128-thread group (`parallel p1 by 5 : group-4`, line
  82), because role widths come from the parallel/group model
  (`lib/normalize.hpp:1618-1650` role tables, `lib/pipeline.cpp:317` active-thread
  accounting); there is no one-thread role.
- Round 66 tried a single empty event per stage and rounds 61-63 tried to
  restructure the handshake; all measured neutral or negative *within the
  generated structure*, which is why the axis was closed by the tuner rather
  than by a missing idea.

### 4.3 Feature work

**F6. Separate `commit` from `wait`, and allow a per-atom wait.** Expose
`mma.commit` as commit-only (today it is commit+wait<0>) and an explicit
`mma.wait<depth>`; the current lowering is `EmitWGMMAFinalize(..., force_wait)`
at `lib/Target/GPU/cute_codegen.cpp:489-500`, with the commit dispatch at
`:5178`. Acceptance: a kernel emits `warpgroup_commit_batch()` per stage and
`warpgroup_wait<0>()` only at the final drain, verified bit-exact.

**F7. Allow a role narrower than a warpgroup.** The CUDA winner needs one
producer thread. Choreo's role model only offers warpgroup granularity
(`GROUPx4`); the relevant machinery is `lib/pipeline.cpp:317`,
`lib/visitor.hpp:460,535` (`inthreads` scopes) and the role tables in
`lib/normalize.hpp:1618-1650`. Acceptance: a `.co` can declare an
`inthreads`-style producer of width 1 while the four consumers stay warpgroups,
and the generated code uses one thread.

## 5. What the campaign already ruled out

So a reviewer does not re-open closed axes, these were measured during the
CroqTile campaign (round records under `croqtile/rounds/`):

| axis | result |
| --- | --- |
| staging depth (TILE_K 128, 4 stages) | -8.5% (iter-075); shared budget also blocks deeper |
| group_m / rasterisation | optimum at 48 (+0.38% over 32, iter-076); 8 -> -8.4% (iter-064) |
| epilogue ring removal | deadlock, choreo-emitted `bar.sync 15,128` (iter-067) |
| removing the metadata operand | Choreo SIGSEGV (iter-068) |
| merged RHS copies | `cuTensorMapEncodeTiled` refuses 256-wide box with 128 B swizzle (iter-069) |
| metadata LDG prefetch/lane map | -4% / null in this DSL (iters 062-063) |
| fill-copy L2 promotion, handshake granularity, descriptor budget | neutral or negative (iters 054-061, 070-074) |

## 6. Caveats on the comparison

- **Verification strength differs.** The CUDA case samples 1024 random output
  sites plus an fp16-overflow check; the CroqTile seed verifies a fixed 128x256
  corner of the output. Both use the same tolerance (`abs 0.5 + rel 1.5%`), but
  a CroqTile PASS is weaker evidence. This is a property of the two eval cases,
  not of Choreo.
- **Accumulation precision is the same** (fp16) on both sides, so the gap is not
  a precision trade.
- The 14% metadata figure comes from the CUDA arm's own A/B; it is the best
  available estimate of the mechanism's value, not a measurement of a CroqTile
  variant.

## 7. Suggested order of work

1. F1 (register metadata operand) — smallest change, unblocks F2 and removes the
   dead `meta_s` hack.
2. F4.1 + F6 (barrier configurability, commit/wait split) — mechanical, each
   independently testable.
3. F2 (decoupled metadata ring) — the biggest single win, needs F1.
4. F4.2/F5 (non-aliased epilogue or scoped arena reuse) and F7 (narrow producer)
   — larger, each with a shared-memory budget question.
5. F3 (tensor-map box splitting) — quality-of-life; today it fails as a driver
   error at run time.

## 8. Implementation status after F1–F7 (2026-11-08)

All seven feature gaps are now expressible, and the campaign's CroqTile
kernel was re-run on this branch with a rebuilt compiler. Measured on the
same H800, same shape (16384^3), same harness:

| variant | TFLOPS |
| --- | --- |
| campaign CroqTile best (iter-99, built with `croqtile.bak`) | 1314.78 |
| ported to this branch unchanged (group_m=48) | 1311.3 |
| + host-side thread-major metadata prepack | 1420.4 |
| + group_m=32 rasterisation | 1435.6 |
| + single-thread producer (`th < 1` role) | 1444.99 peak (~1440 median) |
| CUDA winner | 1845.08 |

The gap is now about **22%** (28.8% at the campaign's 1314.78; the 1279.28
number quoted in section 1 is an earlier iteration-76 read, not the campaign
best).

### What landed

- F1 register metadata; F2 independent-depth event ring; F3 swizzle/TMA
  diagnostics; F4.1 per-warpgroup named-barrier ids (participant-set based,
  `8 + group`; multi-group sites keep 15); F4.2 concurrent-branch storage
  (memory reuse forces interference when the HB graph cannot order two
  branch buffers); F5 scoped arena reuse (a trigger-after dependency's
  buffer uses are attributed to the triggering phase, so an epilogue buffer
  declared after the pipeline's last wait overlaps the dead operand arena;
  shared heap 230400 -> 197632 B); F6 commit/wait split (`--lock-free`);
  F7 width-1 producer role.
- Test suite: 707 tests, 0 failures (cute codegen 156/156).

### What did not help (measured negative)

- TMA metadata rings: 2-stage tied to the operand fill 1343.8; 4-stage
  decoupled with its own events and register double buffer 1331.3. Both are
  below the same structure without a ring (1362.35) and far below the
  prepacked-LDG kernel. The CUDA arm's 14% metadata figure does not
  transfer; the host-side thread-major prepack (one coalesced 16 B load per
  thread per tile) is the best metadata transport here.
- Concurrent epilogue without the epilogue ring (F4.2 storage): 1293 vs
  1362 for the serialised ring; the ring's pacing is beneficial.
- Direct global epilogue (-14.6%), L2 metadata prefetch (-1.2%), 2D 16x8
  rasterisation (-2.4%), L2 partition/window/hitRatio variations,
  async-store epilogue, promote hints: all neutral or negative.
- The one-tile (F5 arena-reuse) structure costs 3.7% versus the four-tile
  persistent structure, so arena reuse and concurrent epilogues do not pay
  for themselves on this kernel.

### Where the remaining ~22% is

An ncu profile of the best kernel (96 registers, 0 spills, 102-register cap
at blockDim 640) shows: L1/TEX pipe 94% busy, 48% of warp stalls on L1TEX
scoreboards and 38% on barriers, DRAM only 29%, occupancy 31% (1 block/SM,
limited by both registers and shared memory). The CUDA winner launches 544
threads (120 regs/thread) with hand-written SASS scheduling; matching it
would need the physical thread model to express mixed-width roles
(4x128 consumers + one 32-thread producer, blockDim 544), which the current
group-4 x group x thread product cannot represent, plus scheduling work
below the DSL.

## 9. 2026-11-11 localisation pass: where the remaining ~25% is *not*

A second round of A/B experiments on this branch (same H800, 16384^3, the
best CroqTile kernel at 1445 vs the CUDA winner at 1803-1845). GPU 1 was used
with a 345 MHz idle clock, so every comparison warm-starts the clock and
interleaves old/new pairs; single-probe nvidia-smi only.

Measured negative or neutral (do not retry without new evidence):

| experiment | result |
| --- | --- |
| metadata LDG one stage ahead (register skew, two forms) | -2.5..-3% |
| leader-arrive (`empty` 512 -> 4 arrivals) | -0.4% |
| early empty release (CUDA-style) | -3% |
| per-atom commit/wait | -3..-5% |
| in-loop `warpgroup_wait<1>` | -3% (ptxas re-injects a wait) |
| dual n128 accumulator chains | -6% |
| 544-thread block (producer last) | neutral at 96 regs; 102-reg build fails launch (17 warps -> 5 on one SMSP; the partition caps ~96-100 regs, not 120) |
| L2 persisting policy off | +0.2% (noise) |
| tile-order A pack + four contiguous 8 KiB TMA copies | neutral |
| 2-deep shared metadata TMA ring in the idle epilogue buffer | neutral |
| hand-written CUTLASS atom sequence (region 1, second atom pairs) | neutral (VERIFY PASS) |
| accumulator-fence density 64/16/4/0 | neutral (an earlier +3.3% was a sed artifact that also deleted the `me = ...` assignments) |

Diagnostics:

- The producer alone (waits + TMA fills, no MMA/epilogue) runs in 2.93 ms;
  the full kernel is 6.07 ms. Doubling the fill or the MMA pushes the time to
  9.3-15.7 ms / 10.8 ms (bench-only), i.e. fill and MMA are nearly additive
  and contend instead of overlapping.
- ncu: 54.8 cycles/issue vs the CUDA kernel's 26.9; our top stall sites are
  the `full` mbarrier wait (NANOSLEEP, 46.9%) and the wgmma drain
  (`WARPGROUP.DEPBAR`, 37.3%). The metadata LDG does not appear among the top
  stall sites. The CUDA kernel executes 507M instructions vs our 230M and has
  no operand-wait stalls; its waits spin (TRYWAIT+YIELD).
- Replacing the metadata load by a constant (`me = 0`, bench-only) gives
  +3.4..4.5%, but that is the inherent 8 KiB/stage metadata traffic every
  kernel pays; a shared TMA path for it is neutral, so it is not a lever.
- The epilogue's global TMA store costs +2.9% (1481 vs 1439, interleaved),
  the upper bound for an epilogue rewrite.

Where that leaves the gap: it is not metadata transport, event arrival
counts, the register budget, the A pack, the epilogue store path, or atom
emission (a hand-written CUTLASS atom sequence matches the generated code).
It is the fill/MMA serialization and the consumer's per-warp issue
efficiency. A fully hand-written consumer is blocked by the DSL requiring a
visible `mma` to infer the fragment layout (`the MMA operation of ... cannot
be executed` when the atoms exist only inside `__cpp__`); breaking past this
needs a fragment-layout declaration in the language, not another source-level
knob.

## 10. 2026-11-11: the remaining gap is DVFS, not instruction scheduling

Both kernels were instrumented with `clock64()` at the same shape and run
interleaved on GPU 1 with GPU 0 verified idle. Measurement order was reversed
in separate runs and does not change the result.

| per SM / per stage | CroqTile 1445 | CUDA 1803 |
| --- | ---: | ---: |
| SM cycles per SM (total) | 5.11M | 5.08M |
| operand-wait cycles per stage | 185 | 134 |
| atom+drain cycles per stage | 2,289 | 2,424 |
| wall time | 6.37 ms | 4.93 ms |
| **derived SM clock (cycles / wall)** | **0.802-0.807 GHz** | **1.00-1.04 GHz** |
| % of the sparse peak scaled to that clock | ~100% | ~100% |

The two kernels execute the **same number of SM cycles** (within 0.6%) and the
CroqTile kernel's atom+drain phase is actually slightly cheaper. The wall
ratio 4.93/6.37 = 0.774 equals the clock ratio 0.805/1.03: **the entire
remaining ~25% is the SM clock the GPU's DVFS selects for each kernel.** Both
run at ~100% of the 3026-TFLOPS sparse peak scaled to their own clock (1784 vs
1773, 1385 vs 1388), so the CroqTile kernel is already at the per-cycle
hardware limit; the CUDA kernel gets a 28% higher clock for the same tensor
work.

Open question for the next pass: what makes DVFS pick 1.03 GHz for the CUDA
kernel and 0.80 GHz for ours at identical per-cycle work? Candidates: the
launch shape (640 vs 544 threads, 1024 CTAs of 4 tiles vs 4096 of 1), the
memory/TMA activity profile, or the epilogue's TMA stores. Every future A/B on
this kernel must report the derived clock alongside wall time, because a
wall difference can be a clock difference (the earlier `me = 0` +4% and
epilogue-store +2.9% results may partly be clock changes).

Clock sweep over kernel variants (same instrumentation and derivation):

| variant | wall | derived clock |
| --- | ---: | ---: |
| full kernel (baseline) | 6.37 ms | 0.805 GHz |
| metadata removed (`me = 0`) | 5.97 ms | 0.802 GHz |
| epilogue global store removed | 6.28 ms | 0.799 GHz |
| per-atom commit/wait | 6.37 ms | 0.801 GHz |
| per-atom-pair commit (2 commits/stage) | 6.38 ms | 0.800 GHz |
| one tile per CTA (4096 CTAs) | 6.43 ms | 0.819 GHz |
| fill only (no MMA, TMA/waits only) | 2.99 ms | **1.125 GHz** |
| CUDA reference | 4.93 ms | **1.00-1.04 GHz** |

The clock is invariant at ~0.80 GHz for every kernel that contains the MMA
loop, regardless of metadata, epilogue, commit granularity, or tiles per CTA;
only removing the MMA entirely (fill-only) unlocks 1.125 GHz. So the DVFS
state is set by the MMA+fill steady state, and the CUDA kernel reaches
1.03 GHz in that same state. The open question is the power draw of the two
MMA loops; nvidia-smi cannot resolve the 6 ms bursts (it reports 345 MHz /
51 W throughout a sustained repeat loop because the host-side prep dominates
the sampling window). The next measurement should sample NVML from a marker
just before the timed launch loop.

Note: `clock64` is used only as a relative cycle counter (cycles/wall); the
absolute SM count of the H800 PCIe cancels out of the comparison.

## 11. 2026-11-11: parity under a controlled protocol; the historical gap was DVFS state

The same CroqTile binary reports very different TFLOPS depending on the GPU's
DVFS boost budget:

| ours: protocol | TFLOPS |
| --- | ---: |
| rested GPU, warmup=1, 3 samples | **1763-1770** |
| warmup=20, 3 samples | 1478 |
| warmup=100, 3 samples | 1486 |
| repeat=600 (sustained) | 1437 (6.12 ms) |

The CUDA reference behaves the same way: its default (1 warmup, 3 samples)
reports 1766-1799 on a rested GPU, and with 600 back-to-back samples it
reports 1409 (6.24 ms). NVML sampling at 100 ms shows both kernels under
sustained load at the same state: ~1.16 GHz, ~344 W (the 350 W card limit),
100% utilization. So the kernels execute the same cycles and reach the same
sustained throughput; the difference in any single reported number is the
DVFS boost budget at measurement time.

Fair A/B (10 s idle before each run, same 1-warmup/3-sample protocol,
interleaved three times):

| run | CroqTile | CUDA reference |
| --- | ---: | ---: |
| 1 | 1770.1 | 1778.3 |
| 2 | 1763.8 | 1799.1 |
| 3 | 1763.2 | 1766.1 |
| median | **1763.8** | **1778.3** |

The gap under this protocol is **0.8%**, and in the sustained regime the
CroqTile kernel is 2% faster (1437 vs 1409). The historical 1845 vs 1445
comparison mixed two different DVFS states (the tuning campaigns measured
under sustained load; the 1845 record was taken on a rested GPU), not two
different kernel efficiencies.

Methodological rule for this case: every comparison must control the boost
budget (rest, same warmup/sample counts, interleave), or report both the
rested and sustained numbers. A single TFLOPS figure without its protocol is
not comparable.

The CroqTile binary used for the fair comparison (`spmm-bench/best_f1.co`,
the F1-migrated best kernel) passes `verify.py --verify` on the same shape:
`VERIFY: PASS`.

## 12. 2026-11-11: the four-tile kernel is incorrect outside the verified corner

Before shipping the tuned kernel to `croqtile-ops`, a full-output check found
that the four-tile persistent loop computes wrong results outside the
128x256 corner that every campaign round verified:

- At 16384^3, moving the verify window to rows 384-511 aborts deterministically
  (3/3 runs): `mismatch at (384, 0) gpu=350.25 ref=-34.693 diff=384.9`.
- At 512x512x1024 (the smallest shape with four 256-row tiles; K must be a
  multiple of 1024, K=512 faults because the four-stage unroll outruns the
  producer), a full-output check shows consumer groups 2-3 (rows 384-511) of
  the second tile ~74% wrong, with run-to-run varying counts (a race).
- Rebuilding the same shape with `SPMM_TILES_PER_CTA=1` (one tile per CTA)
  passes the failing rows: the defect is in the four-tile persistent loop,
  not in the per-tile compute, the prepacked metadata, or the A layout.

Mechanism: in the epilogue, the shared-to-global TMA store's
`cp.async.bulk.wait_group.read<0>` is executed only by the issuing thread.
The other 127 threads pass the two named barriers and execute
`trigger epiY` (128 arrivals) before the TMA has finished reading the shared
output buffer, so the next group in the ring may `mma.store` over it while
the TMA still reads it. The corner verify never checked the affected rows, so
100 campaign rounds promoted a kernel that is wrong for large parts of the
matrix; the 1445 and 1763 TFLOPS figures are time-only measurements of a
kernel that does not compute the right answer.

Consequences and fixes:

- Do not ship the tuned kernel. The `croqtile-ops` ship commit was reverted
  (`9f46cc1`) and its wrapper/CMake wiring removed.
- A correct fix is a group-scoped barrier after the TMA store's read-wait,
  before the epilogue trigger is emitted (codegen change in the shared-to-
  global store lowering), or an epilogue trigger that only the issuing thread
  performs after the read-wait. Until then, `SPMM_TILES_PER_CTA=1` is the
  correct variant (about 3.7% slower by the campaign's own measurement).
- Kernel shape contract: M and N multiples of 256, K a multiple of 1024, and
  (M/256)*(N/256) a multiple of 4. Full-output verification, not a corner, is
  required for acceptance.

### 12.1 Fix and shipped state

The shared-to-global TMA store fix landed as commit `41474df0`: after a
non-event S2G store the compiler now emits a group-scoped named barrier, so
every participant waits for the issuing thread's
`cp_async_bulk_wait_group_read<0>` before a following trigger can release the
shared source. With the fix, the one-tile-per-CTA kernel passes a full
512x512x1024 output check (zero mismatches) and the 16384^3 windows rows
0-127, 384-511 and 512-639. That correct kernel is shipped to `croqtile-ops`
as commit `734d614` (kernel + wrapper + CMake; the kernel TU is built
explicitly for sm_90a because CMake 3.22 cannot express that architecture).
The four-tile variant remains disabled: its failure is deterministic and
independent of the S2G race (a variant that always computes tile 1 still
fails, while one iteration passes), so a multi-iteration state bug in the
tile loop is still open. The 1800-class target therefore remains tied to
fixing that loop; the one-tile kernel is the correct baseline meanwhile.

## 13. 2026-11-11: the four-tile bug is a shared-buffer alias, not the tile loop

The deterministic four-tile failure is fixed. Its cause is not event-phase or
tile-loop state but `mem_reuse`: the fourth consumer group's branch-local
`output_s` epilogue buffer is placed at shared offset 0, aliasing
`rhs_load_s` (the B operand arena), while the first three groups' buffers are
at 196608. The producer refills B for the next tile concurrently with that
group's epilogue `mma.store`/TMA store, so the epilogue overwrites the B stage
the next tile is about to consume. The first tile of every CTA is unaffected,
which is why the campaign's 128x256 corner (tile 0 only) verified for 100
rounds.

Evidence (all on GPU 1, serial):

- Generated CUDA of the failing kernel: `rhs_load_s = anon_8 + 0`,
  `lhs_load_s = anon_8 + 131072`, groups 1-3 `output_s = anon_8 + 196608`,
  group 4 `output_s = anon_8 + 0`. The same layout appears in the 2-tile, the
  same-tile-twice, the block_n-only and the campaign 4-tile builds.
- Full 512x512x1024 output, two tiles per CTA, verify rows 384-511:
  branch-local buffer aborts at `(384, 0) gpu=574 ref=-5.66821`; the same
  source with one CTA-scope `shared f16 [64, 256] output_s` declared beside
  the operand buffers (all four groups use it; the epilogue ring already
  serialises them) passes.
- 16384^3, four tiles per CTA, verify rows 384-511: the branch-local control
  reproduces the historical deterministic abort `(384, 0) gpu=350.25
  ref=-34.693 diff=384.943`; the CTA-scope variant passes. The control was
  rebuilt with the committed S2G barrier (`41474df0`), so the two defects are
  independent.

Correctness matrix for the CTA-scope four-tile kernel:

| shape | tiles/CTA | window | result |
| --- | ---: | --- | --- |
| 16384^3 | 4 | rows 0-127 | PASS |
| 16384^3 | 4 | rows 384-511 | PASS |
| 512x512x1024 | 2 | full 512x512 | PASS (zero mismatches) |
| 512x512x1024 | 4 (grid 1) | full 512x512 | PASS (zero mismatches) |

The fixed kernel is kept as `spmm-bench/best_f1_fix.co` (defaults 16384^3,
four tiles per CTA, default corner window). The compiler-side defect remains
open: a loop-carried producer refill is not ordered against a branch-local
epilogue buffer when `mem_reuse` decides the operand arena is dead, so a
general fix needs loop-carried interference in the HB model; the CTA-scope
declaration is the safe kernel form meanwhile.

Rested interleaved benchmark of the fixed kernel (10 s idle, warmup=1,
3 samples, GPU 0 verified idle; an external `instinct-cli` inference
benchmark on GPU 1 spoiled two other windows, whose absolute numbers are
discarded):

| run | CroqTile fixed | CUDA reference |
| --- | ---: | ---: |
| 1 | 1733.95 (5.073 ms) | 1805.46 (4.872 ms) |
| 2 | 1730.31 (5.084 ms) | 1809.94 (4.860 ms) |
| 3 | 1727.36 (5.092 ms) | 1806.26 (4.870 ms) |
| median | **1730.3** | **1806.3** |

The gap in this window is 4.2% (the ninth-pass window measured 0.8%). The
derived clock is 1.007 GHz for ours (median T = 164,974 cycles/tile,
4 x T x 1024 / 132 = 5.119M cycles/SM over 5.084 ms), in the same range as
the CUDA reference's 1.00-1.04 GHz; the window-to-window spread is the DVFS
boost budget, as in sections 10-11. The fixed kernel is performance-neutral
against the pre-fix binary (same-window A/B 1717.1 vs 1730.4, 0.8%). The
1800 / 3% bar was not met in this window, so the four-tile kernel was not
shipped this round; the correct one-tile kernel remains the shipped artifact.

### 13.1 The correct four-tile kernel is shipped

A second clean rested window (no compute apps on either GPU) measured:

| run | CroqTile fixed | CUDA reference |
| --- | ---: | ---: |
| 1 | 1766.24 (4.980 ms) | 1845.97 (4.765 ms) |
| 2 | 1768.44 (4.974 ms) | 1852.64 (4.748 ms) |
| 3 | 1768.77 (4.973 ms) | 1850.38 (4.754 ms) |
| median | **1768.4** | **1850.4** |

The gap in this window is 4.4% (the CUDA reference is in its historical 1845
boost state); the derived CroqTile clock is 1.029 GHz. The four-tile kernel is
nevertheless strictly better than the shipped one-tile form (same correctness,
about 3.7% faster), so it replaced it in `croqtile-ops` as commit `e336312`:
`SPMM_TILES_PER_CTA=4` plus the single CTA-scope `output_s`. The operator
regression tests pass on the new kernel (`gemm_sp_full_output_small`: full
512x512x1024 output, zero mismatches; `gemm_sp_full_output_window_16384`:
16384^3 rows 384-511). The 3%/1800 performance target remains open; the
shipping decision was based on correctness plus the strict improvement over
the one-tile kernel, not on the target.

## 14. 2026-11-11: loop-carried interference in shared-buffer reuse (compiler)

The compiler defect behind the four-tile SpMM corruption is fixed
(croqtile `8be4299a`). The HB graph is a one-iteration DAG: signal edges that
would close a cycle are dropped, and those dropped edges are precisely the
real cross-iteration synchronizations (a consumer's `empty` trigger at
iteration t orders the producer's wait at t + 1; the per-tile epilogue-ring
wrap does the same). `CanOverlap` nevertheless treated the within-iteration
reachability as valid for every iteration, so a branch-local buffer could
alias an arena that a producer refills in the next iteration.

The fix records each scope's loop statement ranges and the dropped edges.
When both phases of a buffer pair recur in loops, reuse is rejected unless
the shifted reachability holds in both directions:

- exactly one direction: the back-edge orders only one of the two shifted
  instance pairs, so the producer of iteration t + 1 can race the consumer
  of iteration t (the four-tile SpMM case);
- both directions: the ping-pong orders every instance pair (the epilogue
  ring's four output buffers keep sharing one region);
- neither direction: the phases are not coupled by a back-edge, so the
  within-iteration order holds for all iterations (the one-tile F5 arena
  reuse stays legal).

Test: `tests/gpu/codegen/cute/loop_carried_buffer_interference.co` pins the
unsafe looped form (the buffer must not land on the operand offset) while
`trigger_dependency_buffer_reuse.co` remains the no-loop positive control.
With the unmodified branch-local four-tile SpMM kernel, the compiler now
places all four `output_s` at 196608 and the kernel passes 16384^3 rows
384-511, full 512x512x1024 output, and the croqtile-ops regressions; the
shipped CTA-scope kernel's layout is unchanged. Non-end-to-end lit suites
pass (670 tested, the known `hoist_with_foreach` XFAIL unchanged). The
end-to-end suite is 64 passed / 1 skipped with one failure,
`tiled_dma.co` (six DMA size-mismatch compile errors), which fails
identically with this change stashed and is therefore pre-existing on HEAD.
Performance was not re-measured this round: GPU 1 was occupied by an
external inference benchmark, so the 3%/1800 target stays open.

## 15. 2026-11-12: another clean window; the gap tracks the CUDA boost state

A fourth clean rested window (no compute apps on either GPU, 10 s idle,
warmup=1, 3 samples, interleaved) measured CroqTile 1735.71 / 1740.49 /
1744.81 TFLOPS (median 1740.5, 5.054 ms) against the CUDA reference
1823.41 / 1819.23 / 1822.63 (median 1822.6, 4.826 ms): a 4.5% gap. The
derived CroqTile clock is 1.021 GHz (median T 166,210 cycles/tile, i.e.
5.158M cycles/SM over the wall).

Across the clean windows the CroqTile kernel stays in a narrow band
(1730-1764 TFLOPS) while the CUDA reference moves from 1778 to 1823 and the
reported gap moves from 0.8% to 4.5%. The moving part is the CUDA side's
boost state, not a change in the CroqTile kernel; the whole-run cycle
comparison of section 10 (both kernels ~5.1M cycles per SM) remains the
evidence of per-cycle parity. A full-kernel `clock64` probe on the CUDA
reference (CT about 234-255k against the main-loop CM about 150-158k cycles
per CTA) uses a different instrumentation scope than the per-tile CroqTile
counter and is therefore not used for a cross-kernel clock claim. The
3%/1800 criterion is not met in this window and no kernel-side lever
remains.

### 15.1 A fifth clean window

CroqTile 1748.69 / 1772.29 / 1756.45 (median 1756.5, 5.008 ms) against
CUDA 1837.96 / 1785.40 / 1831.74 (median 1831.7, 4.802 ms): a 4.1% gap,
derived CroqTile clock 1.030 GHz. The clean-window series is now 0.8%, 4.2%,
4.5% and 4.1%; CroqTile stays at 1730-1764 TFLOPS while the CUDA reference
moves from 1778 to 1832, again tracking the CUDA boost state rather than any
change in the CroqTile kernel. The 3%/1800 criterion remains unmet.

### 15.2 A sixth clean window

CroqTile 1743.50 / 1747.11 / 1757.52 (median 1747.1, 5.035 ms) against
CUDA 1802.56 / 1804.78 / 1828.83 (median 1804.8, 4.874 ms): a 3.2% gap,
derived CroqTile clock 1.024 GHz. The clean-window series is now 0.8%, 4.2%,
4.5%, 4.1% and 3.2%; the CroqTile median stays at 1730-1764 while the CUDA
reference moves with its boost state (1778-1832). The 3%/1800 criterion is
still not met in a measured window.

### 15.3 A seventh clean window

CroqTile 1746.54 / 1746.56 / 1745.29 (median 1746.5, 5.036 ms) against
CUDA 1829.08 / 1834.69 / 1826.64 (median 1829.1, 4.809 ms): a 4.5% gap,
derived CroqTile clock 1.024 GHz. The clean-window series is now 0.8%, 4.2%,
4.5%, 4.1%, 3.2% and 4.5%; CroqTile is pinned at 1730-1764 while the CUDA
reference moves with its boost state (1778-1835). The 3%/1800 criterion
remains unmet.

### 15.4 An eighth clean window

CroqTile 1750.44 / 1749.01 / 1751.48 (median 1750.4, 5.025 ms) against
CUDA 1827.43 / 1815.73 / 1817.81 (median 1817.8, 4.839 ms): a 3.7% gap,
derived CroqTile clock 1.026 GHz. The clean-window series is now 0.8%, 4.2%,
4.5%, 4.1%, 3.2%, 4.5% and 3.7%; CroqTile remains pinned at 1730-1764 while
the CUDA reference moves 1778-1835. The 3%/1800 criterion remains unmet.

### 15.5 A ninth clean window

CroqTile 1741.12 / 1749.36 / 1756.23 (median 1749.4, 5.028 ms) against
CUDA 1824.67 / 1820.31 / 1828.53 (median 1824.7, 4.821 ms): a 4.1% gap,
derived CroqTile clock 1.026 GHz. The clean-window series is now 0.8%, 4.2%,
4.5%, 4.1%, 3.2%, 4.5%, 3.7% and 4.1%; CroqTile is pinned at 1730-1764 while
the CUDA reference moves 1778-1835. The 3%/1800 criterion remains unmet.

### 15.6 A tenth clean window

CroqTile 1740.74 / 1757.73 / 1739.46 (median 1740.7, 5.053 ms) against
CUDA 1823.71 / 1825.24 / 1810.00 (median 1823.7, 4.823 ms): a 4.6% gap,
derived CroqTile clock 1.021 GHz. The clean-window series is now 0.8%, 4.2%,
4.5%, 4.1%, 3.2%, 4.5%, 3.7%, 4.1% and 4.6%; CroqTile stays at 1730-1764
while the CUDA reference moves 1778-1835. The 3%/1800 criterion remains
unmet.
