# OSS Sync Test Suite

This directory holds the automated tests for the Choreo open-source sync
tooling in `scripts/oss/`. The sync pipeline moves code between four refs:

```
public (GitHub)  <->  oss/main  <->  main  <->  origin  <->  bundle-mirror
```

`oss-push.sh` cherry-picks `main` commits to `oss/main` (stripping
private-only paths), `oss-pull.sh` cherry-picks community commits from
`oss/main` back to `main`, and `oss-scan.sh` / `oss-pull-scan.sh` guard the
boundary for keyword, non-ASCII, and path-collision violations. These tests
exist so changes to that machinery can be verified in isolation before they
touch the real branches.

See `Documents/internal/oss-sync-developer-guide.md` for the full architecture.

---

## Design principles

1. **Never mutate the caller's repo.** Every test runs against disposable
   repositories (temporary dirs or `git clone --shared` sandboxes) so a failed
   test cannot corrupt `main`, `oss/main`, or any remote. No test performs a
   real `git push`.

2. **Two layers, two speeds.** Fast unit tests check script existence, option
   parsing, and help text without doing any git work. Slower integration tests
   build throwaway repos and exercise the full push/pull/scan roundtrip. The
   fast layer catches typos and flag regressions; the slow layer catches
   behavioral regressions.

3. **A `--quick` escape hatch.** The heavyweight suites support `--quick` to
   skip non-essential scenarios. Quick mode is the CI/pre-commit gate; full
   mode is the merge gate.

4. **Regression scenarios encode real bugs.** Each time a bug is fixed, a
   scenario is added that would have failed before the fix. The suite is
   therefore a living record of past failure modes, not just happy paths.

5. **Overlay the in-progress tree.** `test-oss-pull-baseline.sh` copies
   uncommitted edits under `scripts/oss/` into its sandbox so it exercises the
   current working tree, not the last commit.

---

## Rationale

- **Sandboxing** is mandatory because the scripts `checkout`, `cherry-pick`,
  `rebase`, and (in production) `push`. A single mistake against a real branch
  is costly to unwind, so tests are structured to make that impossible.

- **Separation of suites** keeps runtime and signal proportional to the change.
  An option-string tweak needs only the unit suite; a change to the catchup
  dedup logic needs the baseline and workflow suites.

- **Patch-id dedup and scan regressions** are documented as first-class
  scenarios because they encode the two most subtle invariants:
  - a commit already present on `main` under a *different* SHA must be skipped
    (fingerprinted by public-file `git patch-id`, ignoring internal-only edits);
  - a public commit touching a repo-config file (`Makefile`, `CMakeLists.txt`)
    must NOT be flagged DIVERGED unless `main` genuinely diverged from the
    commit's parent.

---

## Test scripts and scenarios

### `test-oss-pull-baseline.sh`

Integration tests for `oss-pull.sh --baseline / --catchup / --last`, run in a
shared-clone sandbox. Scenarios (A-I):

| Scenario | What it verifies |
|----------|------------------|
| A | `--show-baseline` falls back to merge-base when no baseline file exists |
| B | `--set-baseline` writes oss HEAD; subsequent `--catchup` is empty, then pulls a new public commit |
| C | `cherry picked from ... on main` marker commits are skipped; only public-native commits land |
| D | `--max` cap limits batch size; re-running gets the next batch |
| E | `--last` returns the newest unpulled commit |
| F | private-only commits (no public files) are skipped |
| G | conflict-zone file (`Makefile`) pulls cleanly when `main` has no private divergence (regression for a false-positive DIVERGED) |
| H | conflict-zone file with genuine private divergence is still flagged DIVERGED and halts |
| I | a duplicate public patch already on `main` under a different SHA (no trailer) is skipped via public-file patch-id dedup |

### `test-oss-scripts.sh`

Unit/option tests for `oss-push.sh`, `oss-pull.sh`, and `sync_all.sh`:
script existence and permissions, help text, option recognition
(`-n`, `--dry-run`, `--catchup`, `--set-baseline`, ...), dry-run safety, and
config-file presence (`oss_exclude_paths.txt`, `os_kw.txt`). Supports
`--suite push|pull|sync` to scope runs.

### `test-oss-workflow.sh`

Self-contained end-to-end integration in temp repos:
- scan: clean tree, keyword in file/path/commit-message, dir mode,
  non-ASCII in staged/dir, ghost include, coupled changes
- push: filters GCU files, skips all-excluded commits, blocks keyword
  content, dry-run, range, author preservation
- pull: cherry-picks back, author preservation
- pull-scan: no false positive when synced, real divergence detected
- sync: diverged cherry-pick conflict is fatal, diverged rebase succeeds,
  origin-ahead pushes to public

---

## Testing method

Run from the repo root:

```bash
# Fast regression gate (run after any scripts/oss/*.sh change)
bash scripts/oss/tests/test-oss-pull-baseline.sh --quick
bash scripts/oss/tests/test-oss-scripts.sh --quick

# Full validation before merge (core logic changes)
bash scripts/oss/tests/test-oss-pull-baseline.sh
bash scripts/oss/tests/test-oss-scripts.sh

# End-to-end workflow integration
bash scripts/oss/tests/test-oss-workflow.sh

# Scope the unit suite to one script
bash scripts/oss/tests/test-oss-scripts.sh --suite pull
```

All suites print a pass/fail summary and exit non-zero on failure. They
create temp sandboxes under `/tmp` (or `mktemp -d`) and remove them on exit,
including on failure.

### When to run what

| Change | Required gate |
|--------|---------------|
| Any `scripts/oss/*.sh` edit | baseline + scripts `--quick` |
| Core catchup/pull/dedup logic | baseline full + workflow |
| Option parsing / help text | scripts suite |
| `scripts/oss/*.sh` merge request | all three full suites |

---

## Style notes

- ASCII only (no em-dashes, arrows, or non-ASCII quotes) per the OSS
  compliance policy.
- Tests are self-cleaning; do not add cases that leave temp state behind.
- Keep quick-mode scenarios focused on the highest-value regressions.
