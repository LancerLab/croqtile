# Open-Source Sync Developer Guide

This document explains the workflow for synchronizing code between the private
Choreo repository and the public open-source repository (croqtile).

## Architecture

```
  Private repo (origin)         Public repo (oss-shadow)
  ========================      ========================
       main  <-- active dev        main  <-- mirrors oss/main
         |                           ^
         |   cherry-pick (push)      |
         +-----> oss/main -----------+  (git push oss-shadow oss/main:main)
                   ^
                   |   cherry-pick (pull)
                   +---- community contributions
```

- **`main`** -- the private development branch (all code, all targets).
- **`oss/main`** -- a local branch that mirrors the public repository. It
  contains only the GPU path and shared code; all GCU-specific, internal, and
  proprietary content is stripped.
- **`oss-shadow`** -- the git remote pointing to the public GitHub repository.

### What gets excluded

The file `scripts/oss/oss_exclude_paths.txt` defines everything that must NOT
appear on `oss/main`. Key exclusions:

| Category | Paths |
|----------|-------|
| GCU backend | `lib/Target/GCU/`, `tests/gcu/` |
| Benchmarks | `benchmark/` |
| Internal docs | `Documents/internal/`, `Documents/Documentation/target/` |
| AI/Agent configs | `.claude/`, `.codex/`, `.github/skills/`, `.cursor/` |
| Internal scripts | `scripts/*.sh`, `scripts/oss/` |
| Samples | `samples/` |
| CI config | `.gitlab-ci.yml`, `.gitlab/` |
| Performance data | `performance/` |
| External deps | `extern/*` |

### What gets scanned

In addition to path filtering, the keyword scanner (`oss-scan.sh`) blocks any
commit that contains:

1. **Proprietary keywords** -- hardware codenames, internal tool names, company
   identifiers (see `scripts/oss/os_kw.txt` for the full list).
2. **Non-ASCII characters** -- a strict policy. No em-dashes, CJK characters,
   or any byte outside 0x00-0x7F. Use `--` instead of `--`, `->` instead of
   right arrows, etc.

---

## Setup (One-Time)

```bash
# Add the public remote and create the tracking branch
bash scripts/oss/oss-setup.sh
```

This creates the `oss-shadow` remote and the `oss/main` branch tracking
`oss-shadow/main`.

---

## Normal Workflow: Pushing Code to Open Source

### Step 1: Identify commits to sync

```bash
# Find the last sync point on oss/main
git log oss/main --oneline -5

# List commits on main since the sync point
git log --oneline --reverse <last-sync-sha>..main
```

### Step 2: Dry run

Always dry-run first to see what will be included/excluded:

```bash
bash scripts/oss/oss-push.sh -n --range <last-sync-sha>..main
```

For a single commit:

```bash
bash scripts/oss/oss-push.sh -n abc1234
```

### Step 3: Push

```bash
# Single commit
bash scripts/oss/oss-push.sh abc1234

# Range of commits
bash scripts/oss/oss-push.sh --range <from>..<to>

# Multiple specific commits
bash scripts/oss/oss-push.sh abc1234 def5678 ghi9012
```

### Step 4: Review the result

```bash
git log oss/main --oneline -10
git log oss/main -1 --format=fuller   # Verify author is preserved
```

### Step 5: Scan the branch tree (final safety check)

```bash
bash scripts/oss/oss-scan.sh --tree oss/main
```

### Step 6: Push to the public remote

```bash
git push oss-shadow oss/main:main
```

### Author Preservation

The push script preserves the original commit's **author** and **author date**.
The person running the script becomes the **committer**. This means:

```
Author:     Jane <jane@team.com>      <-- who wrote the code
Committer:  You <you@team.com>        <-- who ran oss-push.sh
```

Both are visible with `git log --format=fuller`. GitHub shows the Author in the
web UI.

---

## Normal Workflow: Pulling Community Contributions

When an external contributor submits a fix on the public repo:

### Step 1: Fetch the latest from public

```bash
git fetch oss-shadow
```

### Step 2: Identify the community commit(s)

```bash
git log oss-shadow/main --oneline -10
```

### Step 3: Pull to main

```bash
bash scripts/oss/oss-pull.sh <commit-sha>

# Or a range:
bash scripts/oss/oss-pull.sh --range <from>..<to>
```

The pull script also preserves the original author. No filtering is applied
(public-to-private is always safe).

---

## Handling Violations

When `oss-push.sh` encounters a violation, it:

1. Reports the violation type (keyword / non-ASCII / path).
2. Resets the staged changes (`git reset --hard HEAD`).
3. Skips the commit and marks it as FAILED.
4. Continues to the next commit.

### Common violations and fixes

| Violation | Example | Fix |
|-----------|---------|-----|
| Non-ASCII em-dash | `reports -- others` | Replace `--` with `--` |
| Non-ASCII arrow | `maps -> values` | Replace with `->` or `-->` |
| Non-ASCII CJK | Chinese comments | Rewrite in English |
| Keyword in code | `gcu_arch` in `lit.sh` | Generalize: `target_arch` |
| Keyword in commit msg | `fix gcu ci` | Reword: `fix CI failures` |

### Manual fix workflow

If a commit is blocked by a keyword in shared code:

```bash
# 1. Cherry-pick manually (no-commit so you can edit)
git checkout oss/main
git cherry-pick --no-commit <sha>

# 2. Edit the offending files
vim lib/preprocess.cpp   # fix the keyword
# Replace em-dashes: sed -i 's/\xe2\x80\x94/--/g' file.cpp

# 3. Verify
bash scripts/oss/oss-scan.sh --staged

# 4. Commit with proper author
git commit --author="Original Author <email>" -m "sanitized message"

# 5. Return to main
git checkout main
```

---

## Scanning Modes

The scanner supports multiple modes for different stages:

```bash
# Full branch tree scan (for pre-push verification)
oss-scan.sh --tree oss/main

# Staged changes only (used automatically by oss-push.sh)
oss-scan.sh --staged

# Single commit diff
oss-scan.sh --diff abc1234

# Commit message range
oss-scan.sh --range abc1234..def5678

# Directory on disk
oss-scan.sh --dir /path/to/exported/code
```

---

## Important Manual Steps

These operations require human judgment and cannot be fully automated:

### 1. Commit message sanitization

If the original commit message mentions proprietary terms (e.g., `fix gcu2
regression`), the push will fail. You must:

- Note the original SHA and message.
- Cherry-pick with `--no-commit`.
- Commit with a sanitized message (e.g., `fix device code inclusion`).

### 2. Shared code that references internal targets

Files like `tests/lit.sh`, `lib/preprocess.cpp`, and documentation may
legitimately reference multiple targets by name. Decide whether to:

- **Generalize** the reference (e.g., `gcu_arch` -> `target_arch`).
- **Remove** the GCU-specific section if it is not needed publicly.
- **Keep** it if it is part of a generic multi-target framework (but rename
  the identifiers first).

### 3. Documentation review

Documentation files under `Documents/Documentation/` are included in OSS
(except `Documents/Documentation/target/`). Review them for:

- Proprietary terms
- Internal URLs (enflame.cn, ftp_era)
- Non-ASCII characters

### 4. Pre-push final scan

Always run `oss-scan.sh --tree oss/main` before pushing to the public remote.
This catches anything that slipped through individual commit scans.

### 5. Merge conflicts

When cherry-picking fails with conflicts:

```bash
# On oss/main, resolve conflicts manually
vim <conflicting-file>
git add <conflicting-file>
git cherry-pick --continue

# Or abort and skip the commit
git cherry-pick --abort
```

---

## CI Integration

The `.gitlab-ci.yml` includes:

- **`oss_scan`** job: runs `oss-scan.sh --tree oss/main` on merge requests
  that touch `oss/` branches, preventing accidental keyword leaks.
- **GPU tests** for `oss/` branches: runs the GPU end-to-end test suite
  (no GCU tests, since GCU code is excluded).

---

## Incoming (Pull) Scan

Before pulling public contributions into main, run the pull-scan
to detect file conflicts:

```bash
# Fetch public remote and scan all new commits
make oss-pull-scan
# or directly:
bash scripts/oss/oss-pull-scan.sh --fetch

# Scan specific commits
bash scripts/oss/oss-pull-scan.sh <sha1> <sha2>
```

The pull-scan detects:
- **PRIVATE-PATH**: public commit touches files in our exclude list
- **DIVERGED**: public commit modifies a file that differs between main and oss/main
  (merge conflict risk)
- **CONFIG-ZONE**: public commit touches repo-config files (Makefile, CMakeLists.txt,
  .gitignore, etc.) that are typically managed separately

---

## Periodic Sync Watcher

Use `oss-watch.sh` to automatically monitor the public repo for new commits:

```bash
# Run once (check + notify)
make oss-watch

# Poll every 5 minutes (foreground)
bash scripts/oss/oss-watch.sh --loop

# Poll every 10 minutes (background daemon)
bash scripts/oss/oss-watch.sh --daemon 600

# Stop the daemon
kill $(cat /tmp/oss-watch.pid)
```

On WSL, the watcher sends Windows toast notifications when violations are found.
On Linux, it uses `notify-send`. You can also set a custom hook:

```bash
export OSS_WATCH_HOOK=/path/to/my-notifier.sh
bash scripts/oss/oss-watch.sh --daemon
```

The hook script receives `$1` = title, `$2` = body.

---

## Unified Sync Daemon

For dedicated sync machines (no local development), the `sync_all.sh` script
combines mirror sync (`sync_remotes.sh`) and oss sync into one ordered loop:

```
  GitHub (oss-shadow) <-> local:oss/main | local:main <-> origin <-> mirror
```

Each 2-minute cycle:

1. **sync_remotes**: origin <-> mirror (all branches, bundle-based)
2. **main -> oss/main**: cherry-pick new commits with scan gate
3. **oss/main -> oss-shadow**: push to GitHub if ahead
4. **public -> main**: scan + auto-pull clean incoming commits
5. **main -> origin**: push if ahead from pull

```bash
# Start the daemon (foreground, logs to .git/sync-all/)
make sync-all

# Run one cycle
make sync-all-once

# Or directly:
bash scripts/sync_all.sh --log             # daemon mode
bash scripts/sync_all.sh --once            # single cycle
bash scripts/sync_all.sh --once --dry-run  # preview only
bash scripts/sync_all.sh --skip-mirror     # skip mirror, oss only
```

State is stored in `.git/sync-all/`:
- `last-oss-push-sha` -- last main commit cherry-picked to oss/main
- `sync-all.log` -- cycle log (when `--log` is enabled)

### Bootstrapping

On a fresh sync machine, initialize the state file:

```bash
# Set the last synced SHA to the current tip
echo $(git rev-parse main) > .git/sync-all/last-oss-push-sha
```

---

## AI-Assisted Developer Workflow (/oss-merge)

Internal developers can use the `/oss-merge` AI skill to push their commits
to both `main` and `oss/main` with guided violation fixing. The skill:

1. Previews what will be included/excluded
2. Runs oss-push and catches violations
3. Helps fix keyword/non-ASCII issues interactively
4. Pushes both branches to the developer's fork (never to origin or public)
5. Guides creation of paired merge requests
6. Handles CI failures and commit drops

See `.claude/skills/oss-merge/SKILL.md` for the full workflow.

---

## Quick Reference

| Task | Command |
|------|---------|
| Setup | `bash scripts/oss/oss-setup.sh` |
| Dry-run push | `bash scripts/oss/oss-push.sh -n <commit>` |
| Push commits | `bash scripts/oss/oss-push.sh <commit> ...` |
| Push range | `bash scripts/oss/oss-push.sh --range A..B` |
| Pull from OSS | `bash scripts/oss/oss-pull.sh <commit>` |
| Pull-scan | `bash scripts/oss/oss-pull-scan.sh --fetch` |
| Sync watcher | `bash scripts/oss/oss-watch.sh --daemon` |
| Sync daemon | `bash scripts/sync_all.sh --log` |
| Single sync cycle | `bash scripts/sync_all.sh --once` |
| Scan branch | `bash scripts/oss/oss-scan.sh --tree oss/main` |
| Scan staged | `bash scripts/oss/oss-scan.sh --staged` |
| Scan commit | `bash scripts/oss/oss-scan.sh --diff <sha>` |
| Push to public | `git push oss-shadow oss/main:main` |
| Fetch public | `git fetch oss-shadow` |
