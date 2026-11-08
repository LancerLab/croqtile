#!/usr/bin/env python3
"""Round lock for the CroqTile development loop (Documents/Developer/RSI.md).

A tick that finds an active lock must not start a new task; it resumes the
task the lock names. The lock lives in .git/ so it is never committed and
survives session restarts within the same clone. A lock older than 60 minutes
is considered stale and is cleared by `check`, so an interrupted round cannot
block the loop forever.

Usage:
  scripts/loop_lock.py check    # exit 0: active round printed; exit 1: none
  scripts/loop_lock.py start --focus "<focus>" --task "<one-line task>"
  scripts/loop_lock.py clear
"""

import argparse
import os
import subprocess
import sys
import time

MAX_AGE_SECONDS = 60 * 60


def repo_root() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        sys.exit("loop_lock: not inside a git repository")
    return out.stdout.strip()


def lock_path() -> str:
    return os.path.join(repo_root(), ".git", "croqtile-loop.lock")


def read_lock(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = f.read().splitlines()
    except FileNotFoundError:
        return None
    if not lines or not lines[0].strip().isdigit():
        return None
    entry = lines[1].strip() if len(lines) > 1 else ""
    return int(lines[0]), entry


def cmd_check(_args) -> int:
    path = lock_path()
    lock = read_lock(path)
    if lock is None:
        if os.path.exists(path):
            os.remove(path)
        print("free")
        return 1
    stamp, entry = lock
    age = int(time.time()) - stamp
    if age > MAX_AGE_SECONDS:
        os.remove(path)
        print(f"stale ({age}s): {entry}")
        return 1
    print(f"active ({age}s): {entry}")
    return 0


def cmd_start(args) -> int:
    path = lock_path()
    existing = read_lock(path)
    if existing is not None:
        stamp, entry = existing
        if int(time.time()) - stamp <= MAX_AGE_SECONDS:
            print(f"loop_lock: an active round exists ({entry}); not overwriting")
            return 1
        os.remove(path)
    focus = args.focus or "-"
    task = args.task or "-"
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{int(time.time())}\n{focus} | {task}\n")
    print(f"lock started: {focus} | {task}")
    return 0


def cmd_clear(_args) -> int:
    path = lock_path()
    if os.path.exists(path):
        os.remove(path)
    print("lock cleared")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("check")
    start = sub.add_parser("start")
    start.add_argument("--focus", default="")
    start.add_argument("--task", default="")
    sub.add_parser("clear")
    args = parser.parse_args()
    return {"check": cmd_check, "start": cmd_start, "clear": cmd_clear}[
        args.command
    ](args)


if __name__ == "__main__":
    sys.exit(main())
