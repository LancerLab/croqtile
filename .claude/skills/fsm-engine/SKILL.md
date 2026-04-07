---
name: fsm-engine
description: FSM-driven GPU kernel tuning loop with guard flags, validation scripts, and compaction-safe state. Ensures even weak LLMs follow the ncu-guided optimization protocol mechanically. Use when croq-tuner dispatches you to run the tuning loop.
---

# CroqTuner -- FSM-Driven Tuning Loop

This skill manages the kernel optimization loop through a finite state machine. Instead of relying on LLM memory to follow a long protocol, it uses:
- **`loop-state.json`** -- machine-readable FSM state
- **Validation scripts** -- pre/post checks that block illegal transitions
- **Idea dedup log** -- append-only JSONL preventing repeated ideas
- **Compaction summary** -- structured resume instructions surviving context loss

## FIRST ACTION: Read State

On every invocation, your FIRST action is:

```bash
EXPERIMENT_DIR="benchmark/performance/<kernel_family>/aitune/<session_id>"
STATE_FILE="$EXPERIMENT_DIR/loop-state.json"
cat "$STATE_FILE" 2>/dev/null || echo "NO_STATE"
```

### If NO_STATE -- Fresh Start

1. Read `protocol/identity.md` -- your role and inviolable constraints
2. Read `protocol/loop-contract.md` -- FSM state machine definition
3. Determine the kernel family, mnemonic, and target shape from user request
4. Initialize FSM:
   ```bash
   bash .codex/skills/fsm-engine/scripts/state-transition.sh INIT \
       experiment_dir=$EXPERIMENT_DIR \
       shape_key=<KEY> \
       max_iteration=<N>
   ```
5. Create experiment directories per `ai-tune-artifacts` skill
6. Copy seed/baseline kernel to `iters/iter000_baseline.<ext>`
7. Proceed to execute INIT state actions

### If state file exists -- Resume

1. Read `protocol/identity.md` -- refresh constraints
2. Read `loop-state.json` -- determine exact FSM state
3. Read `compaction-summary.md` if it exists
4. Read `protocol/step-checklists.md` -- ONLY the section for current state
5. If in IDEATE or later: read `idea-log.jsonl`
6. Resume from `fsm.current_state` immediately. Do NOT re-baseline.

## Executing a Step

For EVERY FSM step:

### 1. Pre-Check
```bash
bash .codex/skills/fsm-engine/scripts/pre-step-check.sh <STATE> $EXPERIMENT_DIR
```
If non-zero exit: fix the reported issue before proceeding.

### 2. Execute Step Actions
Follow `protocol/step-checklists.md` for the current state.

### 3. Update Guard Flags
After each mandatory action, update the flag:
```bash
bash .codex/skills/fsm-engine/scripts/state-transition.sh SET $EXPERIMENT_DIR <flag>=true
```

### 4. Post-Check
```bash
bash .codex/skills/fsm-engine/scripts/post-step-check.sh <STATE> $EXPERIMENT_DIR
```
If non-zero exit: complete missing actions.

### 5. Transition
```bash
bash .codex/skills/fsm-engine/scripts/state-transition.sh <NEXT_STATE> $EXPERIMENT_DIR [key=value ...]
```

Then immediately proceed to the next step. Do NOT pause between steps.

## Build Commands

### Choreo-compiled (.co kernels)
```bash
./choreo -gs -t cute -arch=$ARCH [flags] <kernel>.co -o <output>.cute.result
bash <output>.cute.result --execute
```
Common flags: `--use-prepack`
Note: `--use-warpspec` has been removed; the compiler auto-detects warpspec mode.

### NVCC-compiled (.cu kernels)
```bash
nvcc -gencode arch=compute_90a,code=sm_90a -std=c++17 \
  -DCUTLASS_ENABLE_TENSOR_CORE_MMA=1 -D__CHOREO_TARGET_CUTE__ \
  -D__USE_CUDA_TYPE__ -D__CHOREO_DMA_DIAGNOSIS__ \
  -Xcompiler -static-libstdc++ -O3 --use_fast_math \
  -I"$(pwd)/runtime" -I"$(pwd)/extern/cutlass/include" -I"$(pwd)" \
  -L/usr/local/cuda/lib64 -lcuda \
  -o <binary> <kernel>.cu
```

### Timing
```bash
CHOREO_TIMING_WARMUP=5 CHOREO_TIMING_REPEAT=50 <binary_or_bash_command>
```

## Compaction Safety

Before context fills, proactively update `compaction-summary.md` in the experiment dir. After compaction, read `loop-state.json` first, then `compaction-summary.md`, then resume.

## Protocol Files Reference

| File | When to Read | Content |
|---|---|---|
| `protocol/identity.md` | Every invocation | Inviolable constraints |
| `protocol/loop-contract.md` | Fresh start or confused about FSM | State machine definition |
| `protocol/step-checklists.md` | Before each step | Entry/exit conditions per step |
| `protocol/compaction-protocol.md` | After context compaction | How to resume |
| `protocol/idea-diversity-rules.md` | Before IDEATE | Dedup rules, category definitions |

## Related Skills

| Skill | Purpose |
|---|---|
| `ai-tune-artifacts` | Path and naming conventions for all files |
| `ncu-bottleneck` | How to profile and interpret ncu data |
| `choreo-syntax` | DSL reference for editing .co files |
| `compile-and-test` | Build/run workflows |
| `develop-compiler` | When compiler changes are needed |

IMPORTANT: NEVER STOP THE LOOP. Run indefinitely until user interrupts or all iterations are exhausted.
