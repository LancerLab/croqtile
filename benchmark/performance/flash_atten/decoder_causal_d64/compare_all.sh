#!/usr/bin/env bash
export FLASH_ATTEN_VARIANT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$(cd "$FLASH_ATTEN_VARIANT_DIR/.." && pwd)/scripts/_compare_all.sh" "$@"
