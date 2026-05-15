#!/usr/bin/env bash
#set -x

# Pre-commit hook for Choreo.
# Checks: (1) no Chinese comments, (2) C/C++ format via clang-format.
# Install: make install-hooks (auto-runs on first build)

REPO_ROOT="$(git rev-parse --show-toplevel)"

# ---------- Check 1: Chinese comments ----------

CHINESE_COMMENT_PATTERN='(^|[ \t])([#%]|//).*[一-龥]'

files=$(git diff --cached --name-only --diff-filter=ACM | grep -v '^Documents/' | grep -v '^pinocchio/')

echo "[pre-commit] Checking files ..."

has_chinese=0
for file in $files; do
    if [[ -f "$file" ]] && file "$file" | grep -qE 'text|source|script|C source|C\+\+ source|ASCII'; then
        match_lines=$(grep -nE "$CHINESE_COMMENT_PATTERN" "$file" || true)
        if [[ -n "$match_lines" ]]; then
            echo "[pre-commit] Chinese comment found in: $file:line $match_lines"
            has_chinese=1
        fi
    fi
done

if [[ $has_chinese -eq 1 ]]; then
    echo
    echo "[pre-commit] Commit blocked: Please remove all Chinese comments before committing."
    exit 1
fi

# ---------- Check 2: Code format (clang-format) ----------

FORMAT_DIRS="lib runtime tests/standalone"

if [ -x "$REPO_ROOT/extern/clang-format-19-1-2" ]; then
  CLANG_FORMAT="$REPO_ROOT/extern/clang-format-19-1-2"
elif [ -n "${CLANG_FORMAT:-}" ]; then
  CLANG_FORMAT="$CLANG_FORMAT"
else
  CLANG_FORMAT="/usr/bin/clang-format"
fi

if command -v "$CLANG_FORMAT" &>/dev/null; then
  has_format_error=0
  format_checked=0

  for f in $files; do
    in_scope=0
    for d in $FORMAT_DIRS; do
      case "$f" in
        "$d"/*) in_scope=1; break ;;
      esac
    done
    [ "$in_scope" -eq 0 ] && continue

    case "$f" in
      *.c|*.cc|*.cpp|*.cxx|*.h|*.hh|*.hpp|*.hxx|*.cu|*.cuh) ;;
      *) continue ;;
    esac

    [ ! -f "$f" ] && continue

    format_checked=$((format_checked + 1))
    if ! "$CLANG_FORMAT" --dry-run --Werror "$f" 2>/dev/null; then
      echo "[pre-commit] Format error: $f"
      has_format_error=1
    fi
  done

  if [ "$has_format_error" -eq 1 ]; then
    echo
    echo "[pre-commit] Commit blocked: some files are not formatted."
    echo "[pre-commit] Run 'make format' and re-stage the changes."
    exit 1
  fi

  if [ "$format_checked" -gt 0 ]; then
    echo "[pre-commit] Format check passed ($format_checked file(s))."
  fi
fi

exit 0
