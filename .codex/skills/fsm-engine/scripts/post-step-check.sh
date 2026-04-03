#!/bin/bash
set -euo pipefail

# post-step-check.sh -- Validate postconditions after executing an FSM step
#
# Usage: post-step-check.sh <STEP_NAME> <experiment_dir>
#
# Exit 0 = OK, non-zero = incomplete (prints what's missing)

if [ $# -lt 2 ]; then
    echo "Usage: post-step-check.sh <STEP_NAME> <experiment_dir>"
    exit 1
fi

STEP="$1"
EXPERIMENT_DIR="$2"
STATE_FILE="$EXPERIMENT_DIR/loop-state.json"
FAIL=0

fail() {
    echo "INCOMPLETE: $1"
    FAIL=1
}

if [ ! -f "$STATE_FILE" ]; then
    echo "INCOMPLETE: No loop-state.json at $STATE_FILE"
    exit 1
fi

ITER=$(jq -r '.fsm.iteration' "$STATE_FILE")
ITER_PAD=$(printf "%03d" "$ITER")

case "$STEP" in
    INIT)
        [ -d "$EXPERIMENT_DIR/iters" ] || fail "iters/ not created"
        [ -d "$EXPERIMENT_DIR/logs" ]  || fail "logs/ not created"
        [ -d "$EXPERIMENT_DIR/ncu" ]   || fail "ncu/ not created"
        [ -f "$EXPERIMENT_DIR/results.tsv" ] || fail "results.tsv not created"
        SEED_FILES=$(ls "$EXPERIMENT_DIR/iters/iter000_"* 2>/dev/null | wc -l)
        [ "$SEED_FILES" -gt 0 ] || fail "No seed kernel at iters/iter000_*"
        ;;
    BASELINE)
        BASELINE_T=$(jq -r '.metrics.baseline_tflops' "$STATE_FILE")
        [ "$BASELINE_T" != "null" ] || fail "baseline_tflops is null"
        [ -f "$EXPERIMENT_DIR/logs/iter000_compile.log" ] || fail "No compile log for baseline"
        [ -f "$EXPERIMENT_DIR/logs/iter000_timing.log" ]  || fail "No timing log for baseline"
        ;;
    PROFILE)
        NCU_RAN=$(jq -r '.guard_flags.ncu_ran_this_iter' "$STATE_FILE")
        [ "$NCU_RAN" = "true" ] || fail "ncu_ran_this_iter not set"
        BOTTLENECK=$(jq -r '.guard_flags.bottleneck_identified' "$STATE_FILE")
        [ "$BOTTLENECK" = "true" ] || fail "bottleneck_identified not set"
        LB=$(jq -r '.metrics.last_bottleneck' "$STATE_FILE")
        [ "$LB" != "null" ] || fail "last_bottleneck is null"
        ;;
    IDEATE)
        NOVEL=$(jq -r '.guard_flags.idea_is_novel' "$STATE_FILE")
        LOGGED=$(jq -r '.guard_flags.idea_logged' "$STATE_FILE")
        [ "$NOVEL" = "true" ] || fail "idea_is_novel not set"
        [ "$LOGGED" = "true" ] || fail "idea_logged not set"
        [ -f "$EXPERIMENT_DIR/idea-log.jsonl" ] || fail "idea-log.jsonl missing"
        ;;
    IMPLEMENT)
        COMPILE_OK=$(jq -r '.guard_flags.compile_succeeded' "$STATE_FILE")
        [ "$COMPILE_OK" = "true" ] || fail "compile_succeeded not set"
        KERNEL_FILES=$(ls "$EXPERIMENT_DIR/iters/iter${ITER_PAD}_"* 2>/dev/null | wc -l)
        [ "$KERNEL_FILES" -gt 0 ] || fail "No kernel file for iter${ITER_PAD}"
        [ -f "$EXPERIMENT_DIR/logs/iter${ITER_PAD}_compile.log" ] || fail "No compile log"
        ;;
    MEASURE)
        TIMING=$(jq -r '.guard_flags.timing_captured' "$STATE_FILE")
        [ "$TIMING" = "true" ] || fail "timing_captured not set"
        TFLOPS=$(jq -r '.metrics.this_iter_tflops' "$STATE_FILE")
        [ "$TFLOPS" != "null" ] || fail "this_iter_tflops is null"
        [ -f "$EXPERIMENT_DIR/logs/iter${ITER_PAD}_timing.log" ] || fail "No timing log"
        ;;
    DECIDE)
        DECISION=$(jq -r '.guard_flags.decision_made' "$STATE_FILE")
        [ "$DECISION" = "true" ] || fail "decision_made not set"
        D=$(jq -r '.metrics.this_iter_decision' "$STATE_FILE")
        [ "$D" != "null" ] || fail "this_iter_decision is null"
        ;;
    STORE)
        APPENDED=$(jq -r '.guard_flags.results_appended' "$STATE_FILE")
        [ "$APPENDED" = "true" ] || fail "results_appended not set"
        COMMITTED=$(jq -r '.guard_flags.git_committed' "$STATE_FILE")
        [ "$COMMITTED" = "true" ] || fail "git_committed not set"
        # Verify results.tsv has iter entry
        if ! grep -q "iter${ITER_PAD}" "$EXPERIMENT_DIR/results.tsv" 2>/dev/null; then
            fail "results.tsv missing entry for iter${ITER_PAD}"
        fi
        ;;
    DONE)
        ;;
    *)
        fail "Unknown step: $STEP"
        ;;
esac

if [ $FAIL -ne 0 ]; then
    echo "---"
    echo "Complete the above actions before transitioning out of $STEP"
    exit 1
fi

echo "OK: Postconditions met for $STEP"
