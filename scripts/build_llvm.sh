#!/bin/bash
# build_llvm.sh [llvm-tag]
#
# Deepest fallback for obtaining extern/llvm-project: clone the LLVM
# monorepo at <llvm-tag> (shallow), build LLVM/MLIR, and install into
# extern/llvm-project.
#
# Normally cmake/LLVMBootstrap.cmake obtains LLVM automatically from the
# internal FTP mirror ($FTP_SERVER) or the public release tarball listed
# in cmake/deps.conf.  This script is used only when no prebuilt package
# can be downloaded.  It is invoked automatically by LLVMBootstrap.cmake
# as the last resort, but can also be run by hand.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHOREO_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$CHOREO_ROOT/extern"

TAG="${1:-}"
if [ -z "$TAG" ]; then
  TAG="$(sed -n 's/^LLVM_SHASH=//p' ../cmake/deps.conf | head -1)"
fi
if [ -z "$TAG" ]; then
  echo "error: LLVM tag not given and LLVM_SHASH missing from cmake/deps.conf" >&2
  exit 1
fi

SRC_REPO="$(pwd)/llvm-project-${TAG}-src"
SRC_DIR="$SRC_REPO/llvm"
BUILD_DIR="$(pwd)/llvm-project-build-${TAG}"
INSTALL_DIR="$(pwd)/llvm-project"

# Clone the requested tag if there is no checkout yet.
if [ ! -d "$SRC_DIR" ]; then
  echo "Cloning LLVM monorepo ($TAG)..."
  git clone --depth 1 --branch "$TAG" \
    https://github.com/llvm/llvm-project.git "$SRC_REPO"
fi

echo "Source: $SRC_REPO ($(git -C "$SRC_REPO" rev-parse HEAD))"

# Clean previous build attempts for this tag.
rm -rf "$BUILD_DIR"
rm -rf "$INSTALL_DIR"

cmake -S "$SRC_DIR" -B "$BUILD_DIR" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
  -DLLVM_ENABLE_PROJECTS=mlir \
  -DLLVM_TARGETS_TO_BUILD="Native;NVPTX" \
  -DLLVM_ENABLE_RTTI=ON \
  -DLLVM_ENABLE_EH=ON \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DLLVM_BUILD_TOOLS=ON \
  -DLLVM_INCLUDE_TOOLS=ON \
  -DLLVM_INCLUDE_TESTS=OFF \
  -DLLVM_INCLUDE_EXAMPLES=OFF \
  -DLLVM_INCLUDE_BENCHMARKS=OFF \
  -DLLVM_BUILD_LLVM_DYLIB=OFF \
  -DLLVM_LINK_LLVM_DYLIB=OFF \
  -DMLIR_LINK_MLIR_DYLIB=OFF \
  -DLLVM_ENABLE_ZLIB=OFF \
  -DLLVM_ENABLE_ZSTD=OFF \
  -DLLVM_ENABLE_TERMINFO=OFF \
  -DLLVM_ENABLE_LIBEDIT=OFF \
  -DLLVM_ENABLE_LIBPFM=OFF \
  -DLLVM_ENABLE_LIBXML2=OFF \
  -DLLVM_ENABLE_FFI=OFF \
  -DLLVM_ENABLE_Z3_SOLVER=OFF \
  -DMLIR_ENABLE_BINDINGS_PYTHON=OFF \
  -DLLVM_OPTIMIZED_TABLEGEN=ON \
  -DLLVM_APPEND_VC_REV=OFF \
  -DLLVM_ENABLE_IDE=OFF

echo "Building LLVM/MLIR ($TAG)..."
ninja -C "$BUILD_DIR"

echo "Installing LLVM/MLIR into $INSTALL_DIR..."
ninja -C "$BUILD_DIR" install

# Optional: remove build tree to reclaim space.
# rm -rf "$BUILD_DIR"

echo "LLVM/MLIR $TAG built and installed successfully."
