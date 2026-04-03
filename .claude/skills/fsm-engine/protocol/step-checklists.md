# CroqTuner -- Step Checklists

Each FSM state has entry conditions, mandatory actions, and exit conditions. Only read the section matching your current state.

---

## INIT

### Mandatory Actions
1. Create experiment directory tree per `ai-tune-artifacts` skill
2. Copy seed kernel to `iters/iter000_baseline.<ext>`
3. Initialize `results.tsv` with header row
4. Create empty `idea-log.jsonl`
5. Initialize `loop-state.json` via `state-transition.sh INIT`

### Exit -> BASELINE
- Directories exist, seed kernel copied, results.tsv has header

---

## BASELINE

### Mandatory Actions
1. Compile seed kernel for target shape
2. Run verification (MUST pass before timing)
3. Run timing benchmark, capture TFLOPS
4. Save compile output to `logs/iter000_compile.log`
5. Save timing output to `logs/iter000_timing.log`
6. Record baseline in results.tsv (decision = BASELINE)
7. Update metrics: `baseline_tflops`, `current_best_tflops`, `current_best_iter = 0`

### Exit -> PROFILE
- `metrics.baseline_tflops != null`
- Timing log exists on disk

---

## PROFILE

### Mandatory Actions
1. Run ncu on current best kernel (see `ncu-bottleneck` skill)
2. Save `.ncu-rep` to `ncu/` directory
3. Save text summary to `logs/iter<NNN>_ncu.log`
4. Classify bottleneck using decision tree from `ncu-bottleneck` skill
5. Set `guard_flags.ncu_ran_this_iter = true`
6. Set `guard_flags.bottleneck_identified = true`
7. Set `metrics.last_bottleneck = <category>`

### When to skip ncu (acceptable only when ALL true)
- Not the first iteration after baseline
- `consecutive_discards < 3`
- Last ncu was within 5 iterations
- If skipping: still set `ncu_ran_this_iter = true` (waived) and carry forward `last_bottleneck`

### Exit -> IDEATE
- `guard_flags.ncu_ran_this_iter == true`
- `guard_flags.bottleneck_identified == true`
- `metrics.last_bottleneck != null`

---

## IDEATE

### Mandatory Actions
1. Read `idea-log.jsonl`
2. Check diversity requirement (see `idea-diversity-rules.md`)
3. Propose ONE optimization idea grounded in ncu data
4. Verify idea is novel (not in idea-log.jsonl)
5. Set `guard_flags.idea_is_novel = true`
6. Append idea to `idea-log.jsonl`
7. Set `guard_flags.idea_logged = true`

### Exit -> IMPLEMENT
- `guard_flags.idea_is_novel == true`
- `guard_flags.idea_logged == true`

---

## IMPLEMENT

### Mandatory Actions
1. Read current best kernel source
2. Implement the optimization
3. Save as `iters/iter<NNN>_<tag>.<ext>` (may be .co or .cu)
4. Compile. Save output to `logs/iter<NNN>_compile.log`
5. If compile fails: fix and retry up to 3 times
6. Set `guard_flags.compile_succeeded = true`
7. Run verification (unless .cu with no verify harness)
8. Set `guard_flags.correctness_verified = true`

### On Failure (3 retries exhausted)
- Log as DISCARD with `DISCARD_COMPILE_FAIL` or `DISCARD_INCORRECT`
- Skip MEASURE/DECIDE, go directly to STORE

### Exit -> MEASURE
- `guard_flags.compile_succeeded == true`
- `guard_flags.correctness_verified == true`

---

## MEASURE

### Mandatory Actions
1. Run timing benchmark
2. Save output to `logs/iter<NNN>_timing.log`
3. Parse TFLOPS from output
4. Set `metrics.this_iter_tflops = <value>`
5. Set `guard_flags.timing_captured = true`
6. Sanity check: if TFLOPS > 1.5x or < 0.5x current best, re-run

### Exit -> DECIDE
- `guard_flags.timing_captured == true`
- `metrics.this_iter_tflops != null`

---

## DECIDE

### Mandatory Actions
1. Compare `this_iter_tflops` vs `current_best_tflops`
2. If better: KEEP, update best metrics, reset `consecutive_discards = 0`
3. If not better: DISCARD, increment `consecutive_discards`
4. Set `metrics.this_iter_decision`
5. Set `guard_flags.decision_made = true`

### Exit -> STORE
- `guard_flags.decision_made == true`

---

## STORE

### Mandatory Actions (ALL required, in order)
1. Append row to `results.tsv`
2. Set `guard_flags.results_appended = true`
3. Update `idea-log.jsonl` last entry with `result`, `tflops`, `tflops_delta`
4. Update `compaction-summary.md`
5. `git add` experiment directory, `git commit`
6. Set `guard_flags.git_committed = true`
7. Determine next state:
   - If `iteration >= max_iteration` -> `DONE`
   - Else -> `PROFILE` (auto-increments iteration)

### Exit -> PROFILE or DONE
- `guard_flags.results_appended == true`
- `guard_flags.git_committed == true`

---

## DONE

### Mandatory Actions
1. Report final results: best TFLOPS, best iteration, improvement over baseline
2. Final git commit if any pending changes
3. Push branch to remote
