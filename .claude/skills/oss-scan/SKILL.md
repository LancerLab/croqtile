---
name: oss-scan
description: |
  **LEGACY (merged into oss-merge)** - This skill is now thin reference layer.
  All compliance scanning documentation has been merged into the oss-merge skill
  for unified OSS workflow guidance. Use /oss-merge for comprehensive details.
  
  Retained here for quick command reference only.
---

# OSS Compliance Scan (LEGACY - Use /oss-merge instead)

## DEPRECATION NOTICE

**This skill has been merged into [`/oss-merge`].**

All content, violation types, scanning procedures, and troubleshooting have been
consolidated into the unified OSS merge workflow skill. This page now serves as
a **quick reference only**.

**For comprehensive guidance**, use `/oss-merge` which includes:
- Full scanning documentation
- Violation types and fixes
- Troubleshooting guide
- Unit testing requirements

---

## Quick Reference (Commands Only)

```bash
# Scan the oss/main branch tree (full check)
make oss-scan

# Scan staged changes only
make oss-scan-staged

# Scan a specific commit
make oss-scan-diff COMMIT=<sha>

# Dry-run a push (preview without committing)
make oss-push-dry COMMIT=<sha>

# Push last commit to oss/main
make oss-push-last

# Push a specific commit
make oss-push COMMIT=<sha>

# Push a range
make oss-push-range RANGE=<from>..<to>

# Pull a community contribution from oss/main to main
make oss-pull COMMIT=<sha>

# Scan incoming public commits for conflicts before pulling
make oss-pull-scan

# Check sync status
make oss-status

# Test OSS scripts (new)
bash scripts/oss/test-oss-pull-baseline.sh --quick
bash scripts/oss/test-oss-scripts.sh --quick
```

## What Gets Scanned

1. **Proprietary keywords** (scripts/oss/os_kw.txt)
2. **Non-ASCII characters**
3. **Ghost includes** (references to excluded paths)
4. **Coupled changes** (commits touching both public and private files)

## Violation Quick Fixes

| Violation | Example -> Fix |
|-----------|---------------|
| Keyword in code | `gcu_target` -> `target_backend` |
| Keyword in message | `fix gcu ci` -> `fix CI failures` |
| Non-ASCII | `->` -> `->` or `--` -> `--` |
| Ghost include | Remove `#include "Target/GCU/..."` |
| Coupled file | Review if files are interdependent |

---

## See Also

- **[`/oss-merge`](./SKILL.md)** - Complete OSS workflow (scanning, testing, merging)
- **`/oss-merge` Section: Compliance and Scanning** - Detailed scanning procedures
- **`/oss-merge` Section: Troubleshooting** - Common scan failures and fixes

```

### NEVER push to the public remote

**AI agents must NEVER run `git push` targeting the public remote (`oss-shadow`, `public`, or any URL matching `github.com` / `choreo-open`).**

The `oss/main` branch may only be pushed to `origin` (the internal Enflame GitLab). Pushing to the public remote is a **human-only** action that requires manual review and explicit approval.

Forbidden commands (agent must refuse):
```
git push oss-shadow ...
git push public ...
git push <any-github-url> ...
```

Allowed:
```
git push origin oss/main            # internal mirror only
make oss-push COMMIT=<sha>          # cherry-pick to local oss/main (no remote push)
```

If the user asks you to push to the public remote, **warn them and refuse**. Instead, provide the command they should run manually after their own review.

## Related Skills

- **`/oss-merge`** -- full developer workflow for pushing code to both `main` and
  `oss/main`, including violation fix loops, CI failure handling, and paired MRs.
  Use when a developer asks to sync their commits to the open-source branch.

## Other Reminders

- **Always run `make format` before committing** to ensure consistent code style.
- **Always run `make oss-scan` before pushing to the public remote** as a final safety gate.
- OSS Makefile targets live in `lib/Target/GCU/target.mk` (not the root Makefile).
- The unified sync daemon (`make sync-all`) handles automated syncing on
  dedicated machines. See `scripts/sync_all.sh`.
- Full documentation: `Documents/internal/oss-sync-developer-guide.md`
