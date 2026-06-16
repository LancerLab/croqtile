#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CROQTILE_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
EMSDK_DIR="$CROQTILE_DIR/extern/emsdk"
BUILD_DIR="$CROQTILE_DIR/build-wasm"
NATIVE_BUILD_DIR="$CROQTILE_DIR/build"

if [ ! -f "$NATIVE_BUILD_DIR/parser.tab.cc" ]; then
  echo "ERROR: Native build required first (need generated parser files)."
  echo "  cd $CROQTILE_DIR && make build"
  exit 1
fi

if [ ! -d "$EMSDK_DIR/upstream" ]; then
  echo "ERROR: Emscripten SDK not found at $EMSDK_DIR"
  exit 1
fi

source "$EMSDK_DIR/emsdk_env.sh" 2>/dev/null

mkdir -p "$BUILD_DIR"

LIB_DIR="$CROQTILE_DIR/lib"

CORE_SOURCES=(
  "$LIB_DIR/assess.cpp"
  "$LIB_DIR/assert_site.cpp"
  "$LIB_DIR/ast.cpp"
  "$LIB_DIR/earlysema.cpp"
  "$LIB_DIR/semacheck.cpp"
  "$LIB_DIR/shapeinfer.cpp"
  "$LIB_DIR/symtab.cpp"
  "$LIB_DIR/typeinfer.cpp"
  "$LIB_DIR/types.cpp"
  "$LIB_DIR/valno.cpp"
  "$LIB_DIR/visitor.cpp"
  "$LIB_DIR/liveness_analysis.cpp"
  "$LIB_DIR/loop_vectorize.cpp"
  "$LIB_DIR/mem_reuse.cpp"
  "$LIB_DIR/diversity_analysis.cpp"
  "$LIB_DIR/scalar_evolution.cpp"
  "$LIB_DIR/pipeline.cpp"
  "$LIB_DIR/target_registry.cpp"
  "$LIB_DIR/target_utils.cpp"
  "$LIB_DIR/target.cpp"
)

CODEGEN_SOURCES=(
  "$LIB_DIR/codegen.cpp"
  "$LIB_DIR/codegen_utils.cpp"
)

SUPPORT_SOURCES=(
  "$LIB_DIR/command_line.cpp"
)

PP_SOURCES=(
  "$LIB_DIR/preprocess.cpp"
)

PARSER_SOURCES=(
  "$NATIVE_BUILD_DIR/parser.tab.cc"
  "$NATIVE_BUILD_DIR/scanner.yy.cc"
)

TARGET_SOURCES=(
  "$LIB_DIR/Target/GPU/cute_codegen.cpp"
  "$LIB_DIR/Target/GPU/cute_target.cpp"
  "$LIB_DIR/Target/GPU/dma_plan.cpp"
  "$LIB_DIR/Target/GPU/fragment_layout_pass.cpp"
  "$LIB_DIR/sys_utils.cpp"
)

SDK_SOURCES=(
  "$CROQTILE_DIR/tools/sdk/public_api.cpp"
)

COWEB_SOURCES=(
  "$SCRIPT_DIR/co_web.cpp"
  "$SCRIPT_DIR/interpreter.cpp"
)

ALL_SOURCES=(
  "${CORE_SOURCES[@]}"
  "${CODEGEN_SOURCES[@]}"
  "${SUPPORT_SOURCES[@]}"
  "${PP_SOURCES[@]}"
  "${PARSER_SOURCES[@]}"
  "${TARGET_SOURCES[@]}"
  "${SDK_SOURCES[@]}"
  "${COWEB_SOURCES[@]}"
)

VERSION=$(cat "$CROQTILE_DIR/VERSION.txt" | tr -d '\n')

INCLUDE_FLAGS=(
  -I"$LIB_DIR"
  -I"$NATIVE_BUILD_DIR"
  -I"$CROQTILE_DIR/runtime"
  -I"$CROQTILE_DIR/tools/sdk"
  -I"$CROQTILE_DIR"
  -I"$CROQTILE_DIR/extern/include"
)

DEFINES=(
  -D__CHOREO_DEFAULT_TARGET__=\"cute\"
  -DCHOREO_SDK_VERSION=\""$VERSION"\"
)

echo "=== Building co-web.wasm ==="
echo "  Version: $VERSION"
echo "  Sources: ${#ALL_SOURCES[@]} files"
echo ""

em++ \
  "${ALL_SOURCES[@]}" \
  "${INCLUDE_FLAGS[@]}" \
  "${DEFINES[@]}" \
  -std=c++17 \
  -O2 \
  -s MODULARIZE=1 \
  -s EXPORT_NAME="ChoreoModule" \
  -s ALLOW_MEMORY_GROWTH=1 \
  -s INITIAL_MEMORY=33554432 \
  -s ENVIRONMENT=worker \
  -s NO_EXIT_RUNTIME=1 \
  -s DISABLE_EXCEPTION_CATCHING=0 \
  --bind \
  -o "$BUILD_DIR/co-web.js" \
  2>&1

echo ""
echo "=== Build complete ==="
ls -lh "$BUILD_DIR/co-web.js" "$BUILD_DIR/co-web.wasm"
