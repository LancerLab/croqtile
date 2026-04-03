# CroqTuner -- Loop Contract (FSM Definition)

## State Machine

```
INIT ──> BASELINE ──> PROFILE ──> IDEATE ──> IMPLEMENT ──> MEASURE ──> DECIDE ──> STORE ──> PROFILE (loop)
                                                                                     │
                                                                                     └──> DONE (iter >= max)
```

## States

| State | Purpose | Next State |
|---|---|---|
| `INIT` | Create dirs, copy seed kernel, init results.tsv | `BASELINE` |
| `BASELINE` | Compile + measure seed kernel (iter 0) | `PROFILE` |
| `PROFILE` | Run ncu (or skip if allowed), identify bottleneck | `IDEATE` |
| `IDEATE` | Propose ONE novel optimization idea | `IMPLEMENT` |
| `IMPLEMENT` | Create new kernel, compile, verify correctness | `MEASURE` |
| `MEASURE` | Run timing benchmark, capture TFLOPS | `DECIDE` |
| `DECIDE` | Compare to current best: KEEP or DISCARD | `STORE` |
| `STORE` | Append results.tsv, git commit, update FSM | `PROFILE` or `DONE` |
| `DONE` | Report final results | (terminal) |

## Iteration Semantics

- After INIT: `iteration = 0` (baseline)
- After STORE: iteration auto-increments on transition to PROFILE
- Files named `iter<NNN>` where NNN = `fsm.iteration`

## Transition Rules

After STORE:
- If `iteration < max_iteration` -> transition to `PROFILE` (auto-increments)
- If `iteration >= max_iteration` -> transition to `DONE`

IMPLEMENT can short-circuit to STORE on compile/verify failure (DISCARD_COMPILE_FAIL / DISCARD_INCORRECT).

## loop-state.json Schema

```json
{
  "schema_version": 1,
  "fsm": {
    "current_state": "<STATE>",
    "iteration": 0,
    "max_iteration": 30,
    "experiment_dir": "benchmark/performance/<family>/aitune/<session>"
  },
  "guard_flags": {
    "ncu_ran_this_iter": false,
    "bottleneck_identified": false,
    "idea_is_novel": false,
    "idea_logged": false,
    "compile_succeeded": false,
    "correctness_verified": false,
    "timing_captured": false,
    "decision_made": false,
    "results_appended": false,
    "git_committed": false
  },
  "metrics": {
    "baseline_tflops": null,
    "current_best_tflops": null,
    "current_best_iter": null,
    "current_best_kernel": null,
    "this_iter_tflops": null,
    "this_iter_decision": null,
    "consecutive_discards": 0,
    "last_bottleneck": null
  },
  "last_updated": "2026-04-03T00:00:00Z"
}
```
