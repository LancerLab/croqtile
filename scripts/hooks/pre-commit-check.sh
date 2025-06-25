#!/usr/bin/env bash
#set -x

# Pre-commit hook: Block commit if any code file (excluding Documents and pinocchio) contains Chinese comments
# Usage: Place this script in .git/hooks/pre-commit and make it executable

# Match //, #, or % comments containing Chinese characters
CHINESE_COMMENT_PATTERN='(^|[ \t])([#%]|//).*[一-龥]'

# Find all staged files except those in Documents/ and pinocchio/
files=$(git diff --cached --name-only --diff-filter=ACM | grep -v '^Documents/' | grep -v '^pinocchio/')

echo "[pre-commit] Checking files ..."

has_chinese=0
for file in $files; do
    # echo "[pre-commit] Checking file: $file"
    # Only check text/code files
    if [[ -f "$file" ]] && file "$file" | grep -qE 'text|source|script|C source|C\+\+ source|ASCII'; then
        # Print all lines with Chinese comments
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

exit 0
