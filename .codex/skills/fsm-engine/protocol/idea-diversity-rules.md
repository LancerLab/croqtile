# CroqTuner -- Idea Diversity & Dedup Rules

## Categories

| Category | Examples |
|---|---|
| `macro` | WARP_N, STAGES, OUTPUT_PAD, L2 promotion, META_TILE_COLS, maxrregcount |
| `structural` | CTA scheduling, SMEM layout, launch_bounds, barrier placement, output epilogue, fence patterns, warp topology changes |
| `choreo` | New .co kernel structure, different choreo flags, tile size in EDSL, copy-shaping patterns |
| `ncu_micro` | Inline PTX, bank conflict fixes, nanosleep, mbarrier tuning, fence_proxy_async, loop unroll |

## Rules

### D1: No Exact Repeats
Read `idea-log.jsonl`. If the EXACT same change has been tried, pick different.

### D2: No Semantic Duplicates
"Set L2 promotion to 128B on descriptor A" and "L2_128B for LHS TMA" are the same idea.

### D3: Category Progression (2+1 Rule)
After 2 consecutive ideas in the same category, the NEXT idea MUST be from a different category.

### D4: Discard-Triggered Escalation

| `consecutive_discards` | Required Action |
|---|---|
| >= 3 | Run ncu (even if recently profiled), switch category |
| >= 5 | Try completely different approach category |
| >= 10 | Radical changes: different WGMMA tile, split-K, .co rewrite |

### D5: Bottleneck-Responsive Ideas
Ideas should address `metrics.last_bottleneck`. See `ncu-bottleneck` skill for the bottleneck-to-idea mapping.

Exception: if bottleneck-responsive ideas are exhausted, try orthogonal ideas that may shift the bottleneck itself.
