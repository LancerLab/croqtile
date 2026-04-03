#!/bin/bash
set -euo pipefail

# state-transition.sh -- Atomically transition FSM state in loop-state.json
#
# Usage:
#   state-transition.sh INIT <experiment_dir> [key=value ...]
#   state-transition.sh <NEXT_STATE> <experiment_dir> [key=value ...]
#   state-transition.sh SET <experiment_dir> <key=value ...>

if [ $# -lt 2 ]; then
    echo "Usage: state-transition.sh <NEXT_STATE|SET> <experiment_dir> [key=value ...]"
    exit 1
fi

ACTION="$1"
EXPERIMENT_DIR="$2"
shift 2

STATE_FILE="$EXPERIMENT_DIR/loop-state.json"

VALID_STATES="INIT BASELINE PROFILE IDEATE IMPLEMENT MEASURE DECIDE STORE DONE"

# --- SET mode: update flags/metrics without transitioning ---
if [ "$ACTION" = "SET" ]; then
    if [ ! -f "$STATE_FILE" ]; then
        echo "ERROR: No loop-state.json at $STATE_FILE"
        exit 1
    fi
    JQ_EXPR=".last_updated = \"$(date -u +"%Y-%m-%dT%H:%M:%SZ")\""
    for kv in "$@"; do
        KEY="${kv%%=*}"
        VAL="${kv#*=}"
        case "$KEY" in
            ncu_ran_this_iter|bottleneck_identified|idea_is_novel|idea_logged|compile_succeeded|correctness_verified|timing_captured|decision_made|results_appended|git_committed)
                JQ_EXPR="$JQ_EXPR | .guard_flags.$KEY = $VAL"
                ;;
            baseline_tflops|current_best_tflops|this_iter_tflops)
                JQ_EXPR="$JQ_EXPR | .metrics.$KEY = $VAL"
                ;;
            current_best_iter|consecutive_discards)
                JQ_EXPR="$JQ_EXPR | .metrics.$KEY = $VAL"
                ;;
            current_best_kernel|this_iter_decision|last_bottleneck)
                JQ_EXPR="$JQ_EXPR | .metrics.$KEY = \"$VAL\""
                ;;
            iteration)
                JQ_EXPR="$JQ_EXPR | .fsm.iteration = $VAL"
                ;;
        esac
    done
    jq "$JQ_EXPR" "$STATE_FILE" > "${STATE_FILE}.tmp"
    mv "${STATE_FILE}.tmp" "$STATE_FILE"
    echo "OK: Updated flags/metrics"
    exit 0
fi

NEXT_STATE="$ACTION"

if ! echo "$VALID_STATES" | grep -qw "$NEXT_STATE"; then
    echo "ERROR: Invalid state '$NEXT_STATE'. Valid: $VALID_STATES"
    exit 1
fi

# Legal transitions
declare -A LEGAL_TRANSITIONS
LEGAL_TRANSITIONS=(
    ["INIT"]="BASELINE"
    ["BASELINE"]="PROFILE"
    ["PROFILE"]="IDEATE"
    ["IDEATE"]="IMPLEMENT"
    ["IMPLEMENT"]="MEASURE STORE"
    ["MEASURE"]="DECIDE STORE"
    ["DECIDE"]="STORE"
    ["STORE"]="PROFILE DONE"
    ["_NEW_"]="INIT"
)

TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

# --- INIT: create fresh state ---
if [ "$NEXT_STATE" = "INIT" ] && [ ! -f "$STATE_FILE" ]; then
    mkdir -p "$EXPERIMENT_DIR"
    cat > "$STATE_FILE" <<INITJSON
{
  "schema_version": 1,
  "fsm": {
    "current_state": "INIT",
    "iteration": 0,
    "max_iteration": 30,
    "experiment_dir": "$EXPERIMENT_DIR"
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
  "last_updated": "$TIMESTAMP"
}
INITJSON
    echo "OK: Created loop-state.json in INIT"

    # Apply overrides
    for kv in "$@"; do
        KEY="${kv%%=*}"
        VAL="${kv#*=}"
        case "$KEY" in
            max_iteration)
                jq --argjson v "$VAL" '.fsm.max_iteration = $v' "$STATE_FILE" > "${STATE_FILE}.tmp"
                mv "${STATE_FILE}.tmp" "$STATE_FILE"
                ;;
            shape_key)
                jq --arg v "$VAL" '.fsm.shape_key = $v' "$STATE_FILE" > "${STATE_FILE}.tmp"
                mv "${STATE_FILE}.tmp" "$STATE_FILE"
                ;;
            experiment_dir)
                jq --arg v "$VAL" '.fsm.experiment_dir = $v' "$STATE_FILE" > "${STATE_FILE}.tmp"
                mv "${STATE_FILE}.tmp" "$STATE_FILE"
                ;;
        esac
    done
    exit 0
fi

# --- Normal transition ---
if [ ! -f "$STATE_FILE" ]; then
    echo "ERROR: No loop-state.json at $STATE_FILE. Run INIT first."
    exit 1
fi

CURRENT_STATE=$(jq -r '.fsm.current_state' "$STATE_FILE")

ALLOWED="${LEGAL_TRANSITIONS[$CURRENT_STATE]:-}"
if [ -z "$ALLOWED" ]; then
    echo "ERROR: Unknown current state '$CURRENT_STATE'"
    exit 1
fi

if ! echo "$ALLOWED" | grep -qw "$NEXT_STATE"; then
    echo "ERROR: Illegal transition $CURRENT_STATE -> $NEXT_STATE"
    echo "       Allowed from $CURRENT_STATE: $ALLOWED"
    exit 1
fi

JQ_EXPR=".fsm.current_state = \"$NEXT_STATE\" | .last_updated = \"$TIMESTAMP\""

# Reset guard flags based on target state
case "$NEXT_STATE" in
    PROFILE)
        # Auto-increment iteration
        if [ "$CURRENT_STATE" = "BASELINE" ] || [ "$CURRENT_STATE" = "STORE" ]; then
            CURRENT_ITER=$(jq -r '.fsm.iteration' "$STATE_FILE")
            NEXT_ITER=$((CURRENT_ITER + 1))
            JQ_EXPR="$JQ_EXPR | .fsm.iteration = $NEXT_ITER"
        fi
        JQ_EXPR="$JQ_EXPR | .guard_flags.ncu_ran_this_iter = false | .guard_flags.bottleneck_identified = false | .guard_flags.idea_is_novel = false | .guard_flags.idea_logged = false | .guard_flags.compile_succeeded = false | .guard_flags.correctness_verified = false | .guard_flags.timing_captured = false | .guard_flags.decision_made = false | .guard_flags.results_appended = false | .guard_flags.git_committed = false | .metrics.this_iter_tflops = null | .metrics.this_iter_decision = null"
        ;;
    IDEATE)
        JQ_EXPR="$JQ_EXPR | .guard_flags.idea_is_novel = false | .guard_flags.idea_logged = false"
        ;;
    IMPLEMENT)
        JQ_EXPR="$JQ_EXPR | .guard_flags.compile_succeeded = false | .guard_flags.correctness_verified = false"
        ;;
    MEASURE)
        JQ_EXPR="$JQ_EXPR | .guard_flags.timing_captured = false"
        ;;
    DECIDE)
        JQ_EXPR="$JQ_EXPR | .guard_flags.decision_made = false"
        ;;
    STORE)
        JQ_EXPR="$JQ_EXPR | .guard_flags.results_appended = false | .guard_flags.git_committed = false"
        ;;
esac

# Apply key=value overrides
for kv in "$@"; do
    KEY="${kv%%=*}"
    VAL="${kv#*=}"
    case "$KEY" in
        ncu_ran_this_iter|bottleneck_identified|idea_is_novel|idea_logged|compile_succeeded|correctness_verified|timing_captured|decision_made|results_appended|git_committed)
            JQ_EXPR="$JQ_EXPR | .guard_flags.$KEY = $VAL"
            ;;
        baseline_tflops|current_best_tflops|this_iter_tflops)
            JQ_EXPR="$JQ_EXPR | .metrics.$KEY = $VAL"
            ;;
        current_best_iter|consecutive_discards)
            JQ_EXPR="$JQ_EXPR | .metrics.$KEY = $VAL"
            ;;
        current_best_kernel|this_iter_decision|last_bottleneck)
            JQ_EXPR="$JQ_EXPR | .metrics.$KEY = \"$VAL\""
            ;;
    esac
done

jq "$JQ_EXPR" "$STATE_FILE" > "${STATE_FILE}.tmp"
mv "${STATE_FILE}.tmp" "$STATE_FILE"

echo "OK: $CURRENT_STATE -> $NEXT_STATE"
