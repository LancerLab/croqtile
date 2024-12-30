#!/bin/bash
"$@"
exit_code=$?
if [ $exit_code -ne 0 ]; then
    exit 0
else
    exit 1
fi

