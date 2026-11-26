#!/usr/bin/env python3
"""Migrate legacy vector memory syntax to the parenthesized form.

The explicit vector DSL accepts two spellings during the compatibility
window:

    values = vec.load buffer.at(lane);       # legacy
    values = vec.load(buffer.at(lane));      # canonical

    vec.store buffer.at(lane), value;        # legacy
    vec.store(buffer.at(lane), value);       # canonical

This script only rewrites the legacy spelling. It leaves comments untouched
and can process individual files or directories recursively.

Usage:
    python3 scripts/migrate_vec_syntax.py FILE_OR_DIR [...]
    python3 scripts/migrate_vec_syntax.py --write FILE_OR_DIR [...]

Without --write the script only reports the files that would change.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

KEYWORD_RE = re.compile(r"\bvec\.(?:load|store)\b")
FILE_SUFFIXES = {".co"}


def find_stmt_end(text: str, start: int) -> int | None:
    """Return the terminating semicolon position for a legacy statement."""
    depth = 0
    i = start
    in_string = None
    while i < len(text):
        c = text[i]
        if in_string:
            if c == "\\":
                i += 2
                continue
            if c == in_string:
                in_string = None
            i += 1
            continue
        if c in ('"', "'"):
            in_string = c
        elif c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
        elif c == ";" and depth == 0:
            return i
        i += 1
    return None


def migrate_text(text: str) -> tuple[str, int]:
    out = []
    pos = 0
    count = 0
    while True:
        match = KEYWORD_RE.search(text, pos)
        if not match:
            out.append(text[pos:])
            break

        kw_end = match.end()
        line_start = text.rfind("\n", 0, match.start()) + 1
        if "//" in text[line_start : match.start()]:
            out.append(text[pos:kw_end])
            pos = kw_end
            continue

        cursor = kw_end
        while cursor < len(text) and text[cursor] in " \t":
            cursor += 1
        if cursor < len(text) and text[cursor] == "(":
            out.append(text[pos:kw_end])
            pos = kw_end
            continue

        semi = find_stmt_end(text, kw_end)
        if semi is None:
            out.append(text[pos:kw_end])
            pos = kw_end
            continue

        operands = text[kw_end:semi].strip()
        if not operands:
            out.append(text[pos:kw_end])
            pos = kw_end
            continue

        out.append(text[pos:kw_end])
        out.append("(")
        out.append(operands)
        out.append(")")
        pos = semi
        count += 1

    return "".join(out), count


def iter_co_files(paths: list[str]) -> list[Path]:
    files = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            files.extend(p for p in path.rglob("*.co") if p.is_file())
        elif path.is_file():
            files.append(path)
        else:
            print(f"warning: not found: {raw}", file=sys.stderr)
    return sorted(set(files))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Migrate legacy vec.load/vec.store syntax to the parenthesized form."
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="write changes in place (default: dry-run)",
    )
    parser.add_argument("paths", nargs="+", help="files or directories to scan")
    args = parser.parse_args()

    changed_files = 0
    changed_sites = 0
    for path in iter_co_files(args.paths):
        text = path.read_text(encoding="utf-8")
        new_text, count = migrate_text(text)
        if not count:
            continue
        changed_files += 1
        changed_sites += count
        print(f"{'migrated' if args.write else 'would migrate'} {path} ({count})")
        if args.write and new_text != text:
            path.write_text(new_text, encoding="utf-8")

    if not changed_files:
        print("No legacy vector memory syntax found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
