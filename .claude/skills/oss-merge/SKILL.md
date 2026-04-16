---
name: oss-merge
description: Help internal developers push code to both main and oss/main. Use when a developer asks to sync their commits to the open-source branch, fix oss violations, handle CI failures on oss merge requests, or revert/drop an oss commit.
---

# OSS Merge Skill -- Developer Workflow for Open-Source Sync

## MANDATORY RULES (all agents MUST follow)

### Rule 1: NEVER push without explicit instruction

**Do NOT push to `origin/main`, `origin/oss/main`, `oss-shadow`, or `public`
unless the user explicitly asks you to push.** Commits are LOCAL only until
the user says "push", "push to origin", "push to remote", etc.

- `git commit` = OK (local, always allowed when asked to commit)
- `git push` = FORBIDDEN unless the user explicitly requests it
- This applies to ALL remotes, not just oss-shadow/public

When in doubt, show the user what you committed and ask if they want it pushed.

### Rule 2: Every main commit MUST have an oss/main counterpart

When the user asks you to **commit** changes on `main`, you MUST also produce
a corresponding commit on `oss/main`:

1. Commit on `main` first
2. Run `make oss-catchup-dry` (or `oss-push -n <sha>`) to preview
3. If the commit has non-excluded files, cherry-pick to `oss/main` using
   `bash scripts/oss/oss-push.sh <sha>`
4. Verify the oss/main commit passes scan (`oss-scan --tree HEAD`)
5. Verify CMake inputs exist (the script's Gate 3 does this automatically)
6. If ALL files in the commit are excluded, skip the oss/main step (the
   script handles this) -- but still RUN the script so the user sees the skip

**Never leave main ahead of oss/main without the user knowing.**
If oss-push fails or has violations, report them to the user and help fix.

---

## When to Use

- Developer wants to push a commit to both `main` and `oss/main`
- `oss-push` rejects a commit due to violations and the developer needs help fixing it
- CI on an `oss/main` merge request fails and needs fixing
- Developer wants to drop/revert a commit from both branches consistently
- Any request involving `/oss-merge`, "push to oss", "sync to open source"
- **Automatically**: whenever you commit on `main` (Rule 2 above)

## Prerequisites

The developer must have:
- A fork remote (e.g., `my-fork`) pointing to their personal GitLab fork
- The `oss/main` branch and the `oss-shadow` / `public` remotes

If either is missing, Phase 0 below handles creation automatically.

---

## Phase 0: Ensure oss/main Branch Exists

Before any sync work, check that the `oss/main` branch and the required
remotes exist. If they don't, create them on the fly.

```bash
# Check if oss/main exists
if ! git show-ref --verify --quiet refs/heads/oss/main; then
  echo "oss/main branch not found -- creating it now..."

  # Ensure the oss-shadow remote exists
  if ! git remote get-url oss-shadow >/dev/null 2>&1; then
    # Try running oss-setup if available
    if [ -f scripts/oss/oss-setup.sh ]; then
      bash scripts/oss/oss-setup.sh
    else
      # Manual fallback: add oss-shadow remote
      git remote add oss-shadow git@git.enflame.cn:era-dev/choreo-open.git
      git fetch oss-shadow
    fi
  else
    git fetch oss-shadow
  fi

  # Create oss/main from the remote if available, otherwise orphan
  if git show-ref --verify --quiet refs/remotes/oss-shadow/main; then
    git branch oss/main oss-shadow/main
    git branch -u oss-shadow/main oss/main
    echo "Created oss/main tracking oss-shadow/main"
  elif git show-ref --verify --quiet refs/remotes/public/main; then
    git branch oss/main public/main
    echo "Created oss/main from public/main"
  else
    echo "No remote oss branch found -- creating orphan oss/main"
    local_branch="$(git symbolic-ref --short HEAD 2>/dev/null || true)"
    git checkout --orphan oss/main
    git rm -rf . >/dev/null 2>&1 || true
    git commit --allow-empty -m "Initial oss/main branch"
    [ -n "$local_branch" ] && git checkout "$local_branch"
  fi
fi
```

Then verify:

```bash
make oss-status
```

---

## Phase 1: Pre-Flight

Before starting, verify the workspace state:

```bash
# 1. Working tree must be clean
git status --short

# 2. Must be on main
git branch --show-current

# 3. Confirm oss/main exists (Phase 0 handled this)
git log oss/main --oneline -3

# 4. Identify the commit(s) to sync
git log main --oneline -5
```

### Phase 1.1: Rebase Local Branches to Latest (MANDATORY)

Before any oss push/pull operation, make sure both local branches are rebased
to latest remote heads. This prevents stale local state from creating missed
syncs or duplicate cherry-picks.

```bash
# Update remote refs
git fetch --all --prune

# Rebase local main
git checkout main
git rebase origin/main

# Rebase local oss/main (or your configured oss branch)
git checkout oss/main
git rebase origin/oss/main

# Return to main for normal workflow
git checkout main
```

If `origin/oss/main` is missing, create or track it first (see Phase 0).
If either rebase conflicts, resolve conflict first, then continue OSS sync.

Ask the developer which commit(s) they want to push to oss/main. Accept:
- A single SHA or `HEAD`
- A range `<from>..<to>`
- "last N commits"
- "catch up" / "sync everything" -- use `--catchup` mode

### OSS Script Regression Gate (mandatory when editing sync scripts)

If your changes touch any file under `scripts/oss/*.sh`, run:

```bash
bash scripts/oss/test-oss-pull-baseline.sh --quick
```

If changes modify pull/catchup/baseline discovery logic, run full suite too:

```bash
bash scripts/oss/test-oss-pull-baseline.sh
```

Notes:
- The test runs in a temporary sandbox clone and cleans itself up.
- It validates baseline behavior, catchup batching, last-commit discovery,
  marker filtering, and private-path skipping.

---

## Phase 2: Cook the OSS Patch

### 2.0 Catchup Mode (Recommended for "sync everything")

If the developer wants to catch up oss/main with all unsynced main commits:

```bash
# Preview first
make oss-catchup-dry

# Then apply
make oss-catchup
```

This automatically:
1. Scans oss/main cherry-pick trailers to find the last sync point
2. Lists all main commits since that point
3. Skips commits already synced (by trailer) or already reflected (content match)
4. Cherry-picks only genuinely new commits

The output shows "already-synced" (tracked via trailers), "already reflected"
(content matches but no trailer), and "would include" (new changes).

### 2.1 Dry-Run Preview

Always dry-run first to show what will be included/excluded:

```bash
make oss-push-dry COMMIT=<sha>
# or for a range:
make oss-push-dry RANGE=<from>..<to>
```

Show the developer the output. Highlight:
- Included files (will appear on oss/main)
- Excluded files (internal-only, will be stripped)
- "already reflected" means the changes are already on oss/main
- If ALL files are excluded, the commit will be skipped entirely

### 2.2 Apply the Cherry-Pick

```bash
make oss-push COMMIT=<sha>
```

This runs `oss-push.sh` which:
1. Switches to `oss/main`
2. Cherry-picks with path filtering
3. Runs keyword + non-ASCII scan on staged changes
4. Appends cherry-pick provenance tag at commit message end
5. Runs full tree scan after commit
6. Returns to `main`

**The script does NOT push to any remote by default.**

---

## Phase 3: Violation Fix Loop

If the scan rejects the commit, the output will show the violation type and
location. Help the developer fix it.

### Common Violations and Fixes

| Violation | Example | Fix |
|-----------|---------|-----|
| Keyword in code | `gcu_target` in a comment | Generalize: `target_backend` |
| Keyword in commit msg | `fix gcu ci` | Reword: `fix CI failures` |
| Non-ASCII | Unicode arrow `->` | Replace with `->` |
| Non-ASCII | Em-dash `--` | Replace with `--` |
| Ghost include | `#include "Target/GCU/..."` | Remove or guard |

### Fix Workflow

```bash
# 1. Switch to oss/main
git checkout oss/main

# 2. Cherry-pick without committing
git cherry-pick --no-commit <sha>

# 3. Fix the violations in the working tree
#    (agent: edit the offending files here)

# 4. Verify the fix
make oss-scan-staged

# 5. If clean, commit with preserved authorship
clean_msg="$(git log -1 --format=%B <sha> | sed '/^Made-with:/d; /^Generated-by:/d')"
GIT_AUTHOR_DATE="$(git log -1 --format='%ai' <sha>)" \
  git commit --author="$(git log -1 --format='%an <%ae>' <sha>)" \
  -m "${clean_msg}
(cherry picked from <sha> on main)"

# 6. Run full tree scan
make oss-scan

# 7. Return to main
git checkout main
```

### Provenance Tag Requirement (MANDATORY)

Every sync commit created for OSS flow must end with a canonical cherry-pick
provenance line so `oss-pull --catchup` can reliably detect already-synced
commits.

Required footer format:

```text
(cherry picked from <sha> on main)
```

For pulls back to main, required footer is:

```text
(cherry picked from <sha> on oss/main)
```

Do not change this pattern to custom wording; detection logic parses this text.

If the fix requires changes to main as well (e.g., renaming a keyword in
shared code), commit the fix on main FIRST, then redo the oss-push.

### Also fix on main for consistency

When you change code on oss/main to remove violations, make the same change
on main so the branches stay consistent:

```bash
git checkout main
# Apply the same fix
git add <files>
git commit -m "fix: remove proprietary reference in <file>"
```

---

## Phase 4: Push (ONLY when explicitly asked)

**Do NOT push unless the user explicitly asks.** When they do, push BOTH
branches to the remote they specify:

```bash
# Push main branch (original commit)
git push <remote> main

# Push oss/main branch
git push <remote> oss/main
```

**Push target rules:**
- If user says "push" without specifying a remote, ASK which remote
- NEVER push to `oss-shadow`, `public`, or any GitHub URL (the sync
  daemon handles those)
- NEVER push to `origin` unless the user explicitly says "push to origin"
- When pushing to the developer's fork, use their fork remote name

**Note:** The unified sync daemon (`make sync-all`) automatically pushes
`oss/main` to all three remotes (origin, oss-shadow, public). Developers
only need to push to their fork for merge requests.

If the developer does not have a fork remote, guide them:

```bash
git remote add my-fork git@git.enflame.cn:<username>/choreo.git
```

---

## Phase 5: Merge Request Guidance

After pushing, tell the developer to create TWO merge requests:

1. **MR for main**: `<developer-remote>/main` -> `origin/main`
   - Contains the original commit as-is
   - Standard review process

2. **MR for oss/main**: `<developer-remote>/oss/main` -> `origin/oss/main`
   - Contains the filtered/sanitized commit
   - Will be auto-synced to GitHub by the sync daemon

Provide the commands:

```bash
# If using GitLab CLI (glab)
glab mr create --source-branch main --target-branch main \
  --title "<commit title>" --description "Paired with oss/main MR #..."

glab mr create --source-branch oss/main --target-branch oss/main \
  --title "[oss] <commit title>" --description "Paired with main MR #..."
```

Or manual URLs:
```
https://git.enflame.cn/<username>/choreo/-/merge_requests/new?merge_request[source_branch]=main
https://git.enflame.cn/<username>/choreo/-/merge_requests/new?merge_request[source_branch]=oss/main
```

---

## Phase 6: CI Failure Handling

If CI fails on the oss/main merge request:

### 6.1 Diagnose

```bash
# Check CI output (GitLab)
glab ci view

# Common oss/main CI failures:
# - Build failure: missing file that was excluded
# - Test failure: test references excluded code
# - Scan failure: missed keyword/non-ASCII
```

### 6.2 Fix on oss/main

```bash
git checkout oss/main

# Apply the fix
# (agent: make the necessary code changes)

# Verify
make oss-scan-staged

# Amend the last commit (only if it hasn't been merged yet)
git add <fixed-files>
git commit --amend --no-edit

# Or create a new fix commit
git add <fixed-files>
git commit -m "fix: resolve CI failure in oss build"

# Push the fix
git push --force-with-lease <developer-remote> oss/main

git checkout main
```

### 6.3 Keep main consistent

If the fix is also relevant to main:

```bash
git checkout main
# Apply same fix
git add <files>
git commit -m "fix: <same fix description>"
git push <developer-remote> main
```

---

## Phase 7: Drop / Revert a Commit

If the developer decides to abandon the commit from both branches:

### 7.1 Revert on oss/main

```bash
git checkout oss/main

# Find the oss commit to revert
git log --oneline -5

# Revert it
git revert <oss-commit-sha>

git checkout main
```

### 7.2 Revert on main

```bash
git checkout main
git revert <main-commit-sha>
```

### 7.3 Push reverts

```bash
git push <developer-remote> main
git push <developer-remote> oss/main
```

Then create new MRs for the reverts, or update the existing MRs.

### 7.4 If MRs haven't been merged yet

Simpler: just close/abandon both MRs and force-push to remove the commits:

```bash
git checkout oss/main
git reset --hard HEAD~1
git push --force-with-lease <developer-remote> oss/main

git checkout main
git reset --hard HEAD~1
git push --force-with-lease <developer-remote> main
```

---

## Unit Testing OSS Scripts (MANDATORY when editing sync scripts)

The project includes comprehensive unit tests for `oss-push.sh`, `oss-pull.sh`,
and `sync_all.sh` to prevent regressions when script logic is modified.

### Two Test Suites

#### 1. Baseline Regression Tests (test-oss-pull-baseline.sh)

Tests the core pull/catchup/baseline logic:
- `--set-baseline` records sync points correctly
- `--catchup` detects and pulls unpulled commits
- `--last` identifies the newest unpulled commit
- `--max N` limits commits per run
- OSS-push marker filtering works correctly
- Private-only commits are skipped

**When to use:**
- Any change to `oss-pull.sh` baseline/catchup logic
- Any change to `oss-push.sh` marker or filtering logic
- When adding new filtering rules

**Run:**
```bash
# Quick regression (core scenarios): ~47 seconds
bash scripts/oss/test-oss-pull-baseline.sh --quick

# Full suite (all scenarios): ~56 seconds
bash scripts/oss/test-oss-pull-baseline.sh
```

#### 2. Comprehensive Script Tests (test-oss-scripts.sh)

Unit-style tests for individual script options and workflows:
- `oss-push.sh` options: `--catchup`, `--range`, `--dry-run`, `--no-scan`
- `oss-pull.sh` options: `--catchup`, `--last`, `--max N`, `--set-baseline`, `--show-baseline`, `--scan-only`
- `sync_all.sh` modes: `--once`, `--dry-run`, `--skip-mirror`
- Integration scenarios: push->pull roundtrip, baseline->catchup loops

**When to use:**
- Any change to oss-push/pull/sync script options parsing
- When adding new filtering rules or scan checks
- For comprehensive pre-merge validation
- When troubleshooting option combinations

**Run:**
```bash
# Core tests only (~2 minutes):
bash scripts/oss/test-oss-scripts.sh --quick

# Full suite with integration tests (~4 minutes):
bash scripts/oss/test-oss-scripts.sh

# Test one script type:
bash scripts/oss/test-oss-scripts.sh --suite push
bash scripts/oss/test-oss-scripts.sh --suite pull
bash scripts/oss/test-oss-scripts.sh --suite sync

# Verbose output (show git commands):
bash scripts/oss/test-oss-scripts.sh --verbose
```

### Gate: Mandatory Testing When Editing scripts/oss/*.sh

Before committing ANY changes to OSS sync scripts, verify with:

```bash
# During development (quick smoke test):
make oss-test-pull-quick  # or scripts/oss/test-oss-pull-baseline.sh --quick

# Before pushing (comprehensive validation):
make oss-test-pull        # Full baseline suite
bash scripts/oss/test-oss-scripts.sh --quick  # Option validation
```

If you modify:
- `oss-pull.sh` baselinelogic -> required: full `test-oss-pull-baseline.sh`
- `oss-push.sh` filtering -> required: full `test-oss-pull-baseline.sh`
- `oss-push/pull` option parsing -> required: `test-oss-scripts.sh --suite <push|pull>`
- `sync_all.sh` -> required: `test-oss-scripts.sh --suite sync`

All tests run in an isolated sandbox clone and clean up automatically.

---

## Compliance and Scanning

OSS sync scripts enforce strict scanning gates before commits are applied.

### What Gets Scanned

1. **Proprietary keywords** -- hardware names, internal tools, company identifiers
   (defined in `scripts/oss/os_kw.txt`)
2. **Non-ASCII characters** -- strict ASCII-only policy for public code
3. **Ghost references** -- `#include` directives pointing to excluded paths
4. **Coupled changes** -- commits that modify both public and excluded files
   (may indicate structural dependencies)

### Violation Types and Fixes

| Violation | Example | Fix |
|-----------|---------|-----|
| `content` | `gcu_target` in code or string | Remove or generalize: `target_backend` |
| `path` | File path contains keyword | Rename the file/directory |
| `message` | Commit message contains keyword | Reword: `git commit --amend` |
| `non-ascii` | Unicode char like `->` or `--` | Replace with ASCII: `->` or `--` |
| `ghost-include` | `#include "Target/GCU/..."` | Remove or conditionally guard |
| `coupled` | Both public and private files changed | Review: are they interdependent? If yes, refactor to decouple |

### Keyword Examples

Examples of keywords that trigger violations:
- Hardware: `gcu`, `tpu`, `cuda_target`, `sm_90a` (when internal-use only)
- Tools: `enflame`, `internal_ci`, `era_build`
- URLs: `git.enflame.cn`, `era-dev` (when revealing internal infra)
- Company: Internal company names or product codenames

**Policy:** If a term is used internally but NOT a trademark or revealing
internal infrastructure, you can add it to `scripts/oss/oss_kw.txt.`

### Scanning Commands

```bash
# Scan oss/main tree (full check)
make oss-scan

# Scan staged changes only
make oss-scan-staged

# Scan a specific commit
make oss-scan-diff COMMIT=<sha>

# Dry-run a push (preview without committing)
make oss-push-dry COMMIT=<sha>
```

### Handling Manual Commits on oss/main

When you commit directly to `oss/main` and need to fix violations:

```bash
git checkout oss/main
git cherry-pick --no-commit <bad-commit-sha>

# Fix the violations in the working tree
# (edit files, remove keywords, replace non-ASCII)

make oss-scan-staged  # Verify the fix

# Commit with original author preserved
GIT_AUTHOR_DATE="$(git log -1 --format='%ai' <bad-commit-sha>)" \
  git commit --author="$(git log -1 --format='%an <%ae>' <bad-commit-sha>)" \
  -m "$(git log -1 --format=%B <bad-commit-sha> | sed '/^Made-with:/d; /^Generated-by:/d')"

make oss-scan  # Full tree check
```

### Critical: Strip AI Tool Trailers

When committing on `oss/main`, **always remove** AI-tool trailers from the
message (e.g., `Made-with: Cursor`, `Generated-by: Claude`). The project
credits contributors in `Authors:` lines; tool markers are not welcome.

The `oss-push.sh` script strips these automatically, but when committing
manually, use:

```bash
git commit -m "$(git log -1 --format=%B | sed '/^Made-with:/d; /^Generated-by:/d')"
```

---

## Troubleshooting

### Test Failures

#### "FAIL: expected X in output"

**Cause:** A script option or behavior changed; test assertion failed.

**Fix:**
1. Run test with `--verbose`: `bash scripts/oss/test-oss-scripts.sh --verbose`
2. Look for command output that doesn't match expected pattern
3. If the script changed intentionally, update the test assertion
4. Otherwise, revert the recent change that broke the test

#### "Sandbox cleanup failed: Permission denied"

**Cause:** A subprocess or service held an open file in the sandbox.

**Fix:**
```bash
# Manual cleanup:
rm -rf /tmp/tmp.* /tmp/oss_* /tmp/test-* 2>/dev/null || sudo rm -rf ...

# Restart with fresh sandbox:
bash scripts/oss/test-oss-pull-baseline.sh --quick
```

#### "Cannot resolve commit X"

**Cause:** Commit SHA not found; usually a typo or range syntax error.

**Fix:**
```bash
# Verify commit exists
git log --oneline -5
git rev-parse <sha>

# Check range syntax
git rev-list HEAD~5..HEAD  # should list 5 commits
```

### oss-push Failures

#### "Scan failed: keyword 'X' in Y"

**Cause:** Commit contains a proprietary keyword.

**Fix:**
```bash
git checkout oss/main
git cherry-pick --no-commit <sha>
# Edit the offending file(s) to remove or generalize the keyword
make oss-scan-staged
# If clean:
git commit --amend --no-edit
make oss-scan
git checkout main
```

#### "Already synced" (commit skipped)

**Cause:** Commit is already on oss/main (detected via cherry-pick trailer).

**Fix:** No action needed; the script correctly skipped it.

### oss-pull Failures

#### "Conflict file: Makefile modified"

**Cause:** Incoming commit modifies a file managed separately per repo
(Makefile, CMakeLists.txt, README, etc.).

**Fix:**
```bash
# Review the change on oss/main
git log oss/main --oneline -3
git show <oss-commit>:Makefile

# Decide:
# 1. If the change is valid for main too:
#    - Apply it manually on main
#    - Commit separately
# 2. If it's oss-only:
#    - Let the oss/main version stay as-is
#    - The pull scan will continue to reject it (blocks accidental overwrites)
```

#### "Private-only commit pulled (PRIVATE-PATH violation)"

**Cause:** Commit tries to add/modify a file in a private-only path.

**Fix:**
```bash
# Update oss_exclude_paths.txt to include new exclusions
# Then retry oss-pull
bash scripts/oss/oss-pull.sh --catchup
```

#### "Baseline not set"

**Cause:** `--set-baseline` was never run; `--catchup` doesn't know where to start.

**Fix:**
```bash
bash scripts/oss/oss-pull.sh --set-baseline
# Then:
bash scripts/oss/oss-pull.sh --catchup
```

### sync_all Failures

#### "Mirror sync failed"

**Cause:** Bundle mirror is offline or SSH key missing.

**Fix:**
```bash
# Skip mirror phase for this run:
bash scripts/oss/sync_all.sh --skip-mirror --once

# Or configure mirror overrides:
bash scripts/oss/sync_all.sh --mirror-host <new-host> --once
```

#### "Divergence detected on oss/main"

**Cause:** `oss/main` has commits not in public branch; sync daemon needs resolution.

**Fix:**
```bash
# Check what's ahead
git log public/main..oss/main --oneline

# Option 1: Force-push oss/main (if confident)
git push --force-with-lease origin oss/main

# Option 2: Rebase (preserve your commits)
git checkout oss/main
git rebase origin/oss/main
git push origin oss/main

# Then retry
bash scripts/oss/sync_all.sh --once
```

### General Debugging

Enable verbose logging:

```bash
# bash scripts with set -x for tracing
bash -x scripts/oss/oss-pull.sh --catchup

# Or export debug flag if scripts support it
export DEBUG=1
bash scripts/oss/test-oss-pull-baseline.sh --verbose

# Check git operations in detail
git config core.commentChar '#'  # avoid conflicts with special chars
git reflog  # trace branch movements
git log --oneline --graph -10  # visualize branch state
```

---

## Safety Rules

1. **NEVER push to ANY remote** without the user's explicit instruction.
   This includes `origin`, `oss-shadow`, `public`, and GitHub URLs.
   Local commits are always safe; pushes require explicit permission.

2. **NEVER leave main ahead without an oss/main counterpart.**
   Every commit on main must be followed by running oss-push to produce
   a corresponding oss/main commit (or confirm the commit is all-excluded).

3. **ALWAYS strip AI tool trailers** from commit messages on oss/main:
   ```bash
   sed '/^Made-with:/d; /^Generated-by:/d'
   ```

4. **ALWAYS run `make oss-scan`** after committing to oss/main.

5. **ALWAYS keep branches consistent**: if you fix something on oss/main,
   apply the same fix to main (and vice versa).

---

## Quick Reference

| Task | Command |
|------|---------|
| **TESTING** | |
| Test baseline/catchup logic (quick) | `bash scripts/oss/test-oss-pull-baseline.sh --quick` |
| Test baseline/catchup logic (full) | `bash scripts/oss/test-oss-pull-baseline.sh` |
| Test all script options (quick) | `bash scripts/oss/test-oss-scripts.sh --quick` |
| Test all script options (full) | `bash scripts/oss/test-oss-scripts.sh` |
| Test push script only | `bash scripts/oss/test-oss-scripts.sh --suite push` |
| Test pull script only | `bash scripts/oss/test-oss-scripts.sh --suite pull` |
| **SYNC OPERATIONS** | |
| Catch up oss/main | `make oss-catchup` |
| Preview catchup | `make oss-catchup-dry` |
| Preview oss push | `make oss-push-dry COMMIT=<sha>` |
| Push to oss/main | `make oss-push COMMIT=<sha>` |
| **SCANNING** | |
| Scan oss branch | `make oss-scan` |
| Scan staged | `make oss-scan-staged` |
| Scan specific commit | `make oss-scan-diff COMMIT=<sha>` |
| Check sync status | `make oss-status` |
| **HELP** | |
| Full guide | `Documents/internal/oss-sync-developer-guide.md` |

## Related Skills

- **`/compile-and-test`** -- build and test the compiler after changes
- **`/develop-feature`** -- full feature development workflow
- **`/oss-scan`** -- (LEGACY) compliance scanning reference (merged into this skill)
