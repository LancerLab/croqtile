#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  sync_all.sh <path-to-opensource-repo>
EOF
}

if [[ $# -ne 1 ]]; then
  echo "Error: missing <path-to-choreo-opensource-repo>"
  usage
  exit 2
fi

if [ ! -d "$1" ]; then
  echo "Expect a directory: $1"
  exit 3
fi

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." &> /dev/null && pwd)"
SYNC_DIR=$(mktemp -d /tmp/sync_task.XXXXXX)
OPEN_DIR="$(cd "$1" && pwd)"

#trap 'rm -rf "$SYNC_DIR"; echo "Temp files in ${SYNC_DIR} are cleaned up."' EXIT

if [ ! -d "$SYNC_DIR"  ]; then
  echo "Could not create temp directory."
  exit 4
fi

if [ ! -f "${ROOT_DIR}/README.md" ]; then
  echo "Repository root is not found (${ROOT_DIR})."
  exit 5
fi

if [ ! -f "${OPEN_DIR}/README.md" ]; then
  echo "Target repository root is not found."
  exit 6
fi

echo "Synchronize repository from ${ROOT_DIR} to ${OPEN_DIR}."

git archive HEAD | tar -x -C ${SYNC_DIR}

rm -fr ${SYNC_DIR}/lib/Target/GCU/
rm -fr ${SYNC_DIR}/tests/gcu/
rm -fr ${SYNC_DIR}/benchmark
rm -fr ${SYNC_DIR}/Documents/internal
rm -fr ${SYNC_DIR}/Documents/Documentation/target
rm -fr ${SYNC_DIR}/Documents/GPU-Examples
rm -fr ${SYNC_DIR}/extern/Makefile
rm -fr ${SYNC_DIR}/scripts/*.sh
rm -fr ${SYNC_DIR}/scripts/hooks
rm -fr ${SYNC_DIR}/samples
rm -fr ${SYNC_DIR}/.gitlab-ci.yml
rm -fr ${SYNC_DIR}/runtime/catz
rm -fr ${SYNC_DIR}/.gitignore
rm -fr ${SYNC_DIR}/.gitlab
rm -fr ${SYNC_DIR}/.gitlab*
rm -fr ${SYNC_DIR}/.gitmodules
rm -fr ${SYNC_DIR}/.gitattributes
rm -fr ${SYNC_DIR}/.vscode
rm -fr ${SYNC_DIR}/extern/*
cp ${ROOT_DIR}/extern/not.sh ${SYNC_DIR}/extern/

rsync -av --progress ${SYNC_DIR}/ ${OPEN_DIR}/

echo "Target repository ${OPEN_DIR} is now synchronized."
