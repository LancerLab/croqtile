#!/usr/bin/env bash
export FLASH_ATTEN_BASELINES_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$(cd "$FLASH_ATTEN_BASELINES_DIR/../.." && pwd)/_bench_fa3.sh" "$@"
