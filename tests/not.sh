#!/bin/bash
"$@"
exit_code=$?
if [ $exit_code -gt 128 ]; then
    # Keep abnormal termination (128+N for signal N) visible so a crashing
    # command cannot satisfy a `not` expectation.
    exit $exit_code
elif [ $exit_code -ne 0 ]; then
    exit 0
else
    exit 1
fi
