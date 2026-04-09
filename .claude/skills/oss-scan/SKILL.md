---
name: oss-scan
description: Scan code for open-source compliance before pushing to the public repository. Use when preparing commits for oss/main, reviewing code for proprietary leaks, or when asked about OSS sync.
---

# OSS Compliance Scan Skill

## When to Use

- Before pushing any commit to `oss/main`
- When reviewing code changes that may be synced publicly
- When asked about OSS sync, keyword violations, or non-ASCII issues
- When a `make oss-push` or `make oss-scan` fails

## Quick Reference

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

# Check sync status
make oss-status
```

## What Gets Scanned

The scanner (`scripts/oss/oss-scan.sh`) checks for:

1. **Proprietary keywords** -- hardware names, internal tools, company identifiers
   (defined in `scripts/oss/os_kw.txt`)
2. **Non-ASCII characters** -- strict ASCII-only policy for public code
3. **Ghost references** -- `#include` directives pointing to excluded paths
4. **Coupled changes** -- commits that modify both public and excluded files
   (may indicate structural dependencies)

## Violation Types and Fixes

| Violation | Meaning | Fix |
|-----------|---------|-----|
| `content` | Keyword found in file content | Remove or generalize the term |
| `path` | File path contains a keyword | Rename the file/directory |
| `message` | Commit message contains a keyword | Reword the commit message |
| `non-ascii` | Non-ASCII bytes in file | Replace: `--` for em-dash, `->` for arrow, English for CJK |
| `ghost-include` | `#include` references excluded path | Remove or guard the include |
| `coupled` | Commit touches both public and excluded files | Review for hidden dependencies |

## Handling Manual Commits

When `oss-push` fails due to violations:

```bash
git checkout oss/main
git cherry-pick --no-commit <sha>
# Fix the violations in staged files
make oss-scan-staged
# Commit with original author
GIT_AUTHOR_DATE='<date>' git commit --author='Name <email>' -m '<sanitized msg>'
git checkout main
```

## Critical Safety Rules

### ALWAYS strip AI tool markers from commit messages

When creating or amending commits on `oss/main`, **always remove** any
`Made-with:`, `Generated-by:`, or similar AI-tool trailer lines (e.g.
`Made-with: Cursor`, `Generated-by: Claude`). The contributors get proper
credit in `Authors:` lines; tool markers are not welcome.

The `oss-push.sh` script strips these automatically, but when committing
manually on `oss/main`, pipe the message through:
```bash
sed '/^Made-with:/d; /^Generated-by:/d'
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

## Other Reminders

- **Always run `make format` before committing** to ensure consistent code style.
- **Always run `make oss-scan` before pushing to the public remote** as a final safety gate.
- OSS Makefile targets live in `lib/Target/GCU/target.mk` (not the root Makefile).
- Full documentation: `Documents/internal/oss-sync-developer-guide.md`
