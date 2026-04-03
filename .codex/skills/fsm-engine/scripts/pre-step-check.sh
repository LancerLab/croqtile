#!/bin/bash
set -euo pipefail

# pre-step-check.sh -- Validate preconditions before executing an FSM step
#
# Usage: pre-step-check.sh <STEP_NAME> <experiment_dir>
#
# Exit 0 = OK, non-zero = blocked (prints what's missing)

if [ $# -lt 2 ]; then
    echo "Usage: pre-step-check.sh <STEP_NAME> <experiment_dir>"
    exit 1
fi

STEP="$1"
EXPERIMENT_DIR="$2"
STATE_FILE="$EXPERIMENT_DIR/loop-state.json"
FAIL=0

fail() {
    echo "BLOCKED: $1"
    FAIL=1
}

if [ ! -f "$STATE_FILE" ]; then
    echo "BLOCKED: No loop-state.json at $STATE_FILE"
    exit 1
fi

CURRENT_STATE=$(jq -r '.fsm.current_state' "$STATE_FILE")

if [ "$CURRENT_STATE" != "$STEP" ]; then
    fail "FSM is in $CURRENT_STATE, not $STEP"
fi

ITER=$(jq -r '.fsm.iteration' "$STATE_FILE")

case "$STEP" in
    INIT)
        ;;
    BASELINE)
        if [ ! -d "$EXPERIMENT_DIR/iters" ]; then
            fail "iters/ directory does not exist"
        fi
        SEED_FILES=$(ls "$EXPERIMENT_DIR/iters/iter000_"* 2>/dev/null | wc -l)
        if [ "$SEED_FILES" -eq 0 ]; then
            fail "No seed kernel at iters/iter000_*"
        fi
        ;;
    PROFILE)
        BASELINE_T=$(jq -r '.metrics.baseline_tflops' "$STATE_FILE")
        if [ "$BASELINE_T" = "null" ] && [ "$ITER" -gt 0 ]; then
            fail "baseline_tflops is null"
        fi
        ;;
    IDEATE)
        NCU_RAN=$(jq -r '.guard_flags.ncu_ran_this_iter' "$STATE_FILE")
        BOTTLENECK_ID=$(jq -r '.guard_flags.bottleneck_identified' "$STATE_FILE")
        BOTTLENECK=$(jq -r '.metrics.last_bottleneck' "$STATE_FILE")
        if [ "$NCU_RAN" != "true" ]; then
            fail "ncu has not been run this iteration (guard_flags.ncu_ran_this_iter != true)"
        fi
        if [ "$BOTTLENECK_ID" != "true" ]; then
            fail "Bottleneck not identified (guard_flags.bottleneck_identified != true)"
        fi
        if [ "$BOTTLENECK" = "null" ]; then
            fail "metrics.last_bottleneck is null"
        fi
        ;;
    IMPLEMENT)
        IDEA_NOVEL=$(jq -r '.guard_flags.idea_is_novel' "$STATE_FILE")
        IDEA_LOGGED=$(jq -r '.guard_flags.idea_logged' "$STATE_FILE")
        if [ "$IDEA_NOVEL" != "true" ]; then
            fail "No novel idea proposed (guard_flags.idea_is_novel != true)"
        fi
        if [ "$IDEA_LOGGED" != "true" ]; then
            fail "Idea not logged (guard_flags.idea_logged != true)"
        fi
        ;;
    MEASURE)
        COMPILE_OK=$(jq -r '.guard_flags.compile_succeeded' "$STATE_FILE")
        VERIFY_OK=$(jq -r '.guard_flags.correctness_verified' "$STATE_FILE")
        if [ "$COMPILE_OK" != "true" ]; then
            fail "Compile did not succeed (guard_flags.compile_succeeded != true)"
        fi
        if [ "$VERIFY_OK" != "true" ]; then
            fail "Correctness not verified (guard_flags.correctness_verified != true)"
        fi
        ;;
    DECIDE)
        TIMING=$(jq -r '.guard_flags.timing_captured' "$STATE_FILE")
        TFLOPS=$(jq -r '.metrics.this_iter_tflops' "$STATE_FILE")
        if [ "$TIMING" != "true" ]; then
            fail "Timing not captured (guard_flags.timing_captured != true)"
        fi
        if [ "$TFLOPS" = "null" ]; then
            fail "metrics.this_iter_tflops is null"
        fi
        ;;
    STORE)
        DECISION=$(jq -r '.guard_flags.decision_made' "$STATE_FILE")
        if [ "$DECISION" != "true" ]; then
            fail "No decision made (guard_flags.decision_made != true)"
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
    echo "Fix the above issues before proceeding with $STEP"
    exit 1
fi

echo "OK: Preconditions met for $STEP"
