# CroqTuner -- Compaction & Resume Protocol

## After Context Compaction

Execute IN ORDER:

1. Read `loop-state.json` in the experiment dir -- exact FSM state
2. Read `compaction-summary.md` -- what just happened
3. Read `protocol/identity.md` -- refresh constraints
4. Read `results.tsv` -- iteration history
5. Read `idea-log.jsonl` -- ideas already tried
6. Resume from `fsm.current_state`. Do NOT re-baseline, re-run completed iters, or ask user.

## compaction-summary.md Template

```markdown
# Compaction Summary
## Last Updated
<ISO timestamp>
## IMMEDIATE ACTION
Read `loop-state.json` and resume from state `<STATE>` at iteration <N>.
## Current Task
- Kernel family: <family>
- Session: <session_id>
- Current best: <TFLOPS> TFLOPS at iter <N>
- Baseline: <TFLOPS> TFLOPS
- Last bottleneck: <category>
- Consecutive discards: <N>
## Recent History (last 5)
- iter <N>: <idea> -> <TFLOPS> (<KEEP|DISCARD>)
## DO NOT
- Re-run completed iterations
- Re-baseline
- Ask the user what to do
```

Keep at most 10 recent entries. Accumulate strategy notes across compactions.

## Key Principle

After compaction, spend < 30 seconds reading state files, then immediately resume.
