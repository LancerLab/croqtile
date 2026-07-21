# Croqtile Test Runner (`lit.sh`) -- User and Developer Guide

`tests/lit.sh` is Croqtile's lightweight test runner, named and designed after LLVM's internal `lit` (LLVM Integrated Tester).  Like LLVM lit, it discovers test files, reads embedded `RUN:` directives, substitutes `%`-tokens with real paths and flags, then executes the resulting shell commands.  The output follows the same style: each test is reported as `PASS`, `FAIL`, `XFAIL` (expected failure), or `SKIP`, with a timing column, finishing with a summary line.  Where a `RUN:` command pipes through `FileCheck`, the runner emits the diff on mismatch exactly as LLVM lit does.  The key restriction compared to full LLVM lit is that `lit.sh` is a self-contained Bash script--no Python runtime required--and its hook system is tailored to Croqtile's multi-target requirements.

---

## Quick Start

```bash
# Run a single test
./tests/lit.sh tests/check/if_hoist.co

# Run a directory of tests
./tests/lit.sh tests/check/

# Run the full test suite (parallel, 4 jobs)
./tests/lit.sh -j4 tests/

# Dry run: print which commands would execute without running them
./tests/lit.sh --dry-run tests/

# Run the full suite + write a timestamped log
./tests/lit.sh -l tests/

# Via make
make test          # release build + full suite
make test-debug    # debug build + full suite
```

---

## Test File Format

### `RUN:` directives

A `.co` file may contain one or more `// RUN:` lines.  Each is executed in order; failure of any line marks the test as FAILED.

```co
// RUN: choreo -es %s -o - | FileCheck %s
// RUN: not choreo -DERROR %s 2>&1 | FileCheck --check-prefix=ERR %s
```

A `.cmt` file uses `#` as the comment marker:

```
# RUN: choreo -es %s -o /dev/null 2>&1 | FileCheck %s
```

### Environment-specific `RUN:` variants

| Directive | Executes when |
|-----------|---------------|
| `// RUN: ...` | Always (shell environment) |
| `// RUN-DOCKER: ...` | Inside Docker only |
| `// RUN-<mach>: ...` | Only when `mach` matches (e.g. `// RUN-mt100: ...`) |

### `REQUIRES:` vs `RUN-<mach>`

There are two methods to restrict tests to specific hardware:

* **`REQUIRES:`** — skips the entire test when the condition is not met.
* **`RUN-<mach>:`** — gates individual `RUN:` lines; only the matching
  lines execute on the target hardware.

Do not mix both methods in the same test file. Pick one per file.

### `REQUIRES:` directive

Controls whether a test is skipped entirely:

```co
// REQUIRES: TARGET-MYTARGET    // needs a mytarget device
// REQUIRES: TARGET-MT100       // needs mt100 arch specifically
// REQUIRES: LIBRARY-MYLIB      // needs a specific library
```

### `XFAIL:` directive

Marks a test as an expected failure on specific hardware:

```co
// XFAIL: mt100     // known failure on mt100
// XFAIL: *         // expected failure everywhere
```

### `SKIP:` directive

Unconditionally skips the test (useful for temporarily disabling):

```co
// SKIP: reason
```

---

## `%` Substitutions Reference

`lit.sh` replaces `%` tokens in `RUN:` command lines before execution.

| Token | Replaced With | Source |
|-------|--------------|--------|
| `%s` | Absolute path to the test file | Core (always) |
| `%target` | Target flag (e.g. `-t mytarget`) | Each target's `target_cmd` hook; stripped to empty string when no hook is active |
| `%<name>` | Target-defined value | Target's `target_cmd` hook (e.g. `%mytarget_arch` -> `-arch mt100`) |

### When to use `%target`

Add `%target` to a `RUN:` line when the test's **expected output depends on which codegen backend** is in use.

```co
// RUN: choreo %target -es --runtime-check=all %s -o - | FileCheck %s
```

When this test is run directly from `tests/check/` (no target hook loaded), `%target` becomes empty and `choreo` uses its default target.  When the same test is pulled in via a target's `include_dir` line, `%target` is expanded by that target's `target_cmd` hook.

Tests whose output is entirely target-agnostic do **not** need `%target`.

---

## Per-Directory Configuration (`lit.cfg`)

Each test directory may contain a `lit.cfg` bash file sourced by `lit.sh` when processing files from that directory.  `lit.cfg` registers hooks that fire during test preparation and command execution.

### The `# co-lit` marker

Every `lit.cfg` **must** start with a `# co-lit` first line.  `lit.sh` uses this marker to distinguish Croqtile configs from configs belonging to other tools (e.g. LLVM `lit.cfg` files are Python, not bash).  A file without the marker is silently skipped.

```bash
# co-lit
# rest of the configuration ...
```

### Hook Reference

| Phase | Signature | Purpose |
|-------|-----------|---------|
| `hw_detect` | `my_fn()` | Detect hardware; set `device_type`, `mach`, target-specific arch vars |
| `set_archs` `[TARGET ...]` | `my_fn TARGET-FOO` | Translate `REQUIRES: TARGET-XXX` tokens into entries in `REQ_TARGETS` |
| `all_archs` | `my_fn()` | Populate `REQ_TARGETS` with all known archs (used by `--all-archs`) |
| `target_cmd` `name_of_cmd_var` | `my_fn cmd_ref` | Substitute `%target` and other target-specific tokens in the command string; receives the name of the nameref variable |
| `target_noskip` | `my_fn()` | Return non-zero to skip a test for a target-specific reason (e.g. simulator not present) |
| `target_prepare` | `my_fn()` | Run arbitrary setup before the test executes |

### Registering hooks

```bash
register_hook "hw_detect"  "my_detect_fn"
register_hook "target_cmd" "my_command_fn"
```

Multiple hooks for the same phase are called in registration order.

### Design principle: target-agnostic core

`lit.sh` itself must not contain any target-specific variable names, token substitutions, or detection logic.  The only variables `lit.sh` manages are the framework-level ones: `device_type`, `mach`, and `simulator`.  Everything target-specific -- arch variables, `%`-token expansions, skip conditions -- belongs in the target's own `lit.cfg` and its hooks.  This keeps the core runner extensible without modification.

---

## Adding a New Target

Follow these steps to integrate a new compilation target (e.g. `mytarget`) into `lit.sh`.

### 1. Create the target directory and `lit.cfg`

```
tests/mytarget/
    lit.cfg
```

Minimal `lit.cfg` (must start with `# co-lit`):

```bash
# co-lit
mytarget_arch="none"

# --- Hook functions ---

mytarget_detect() {
  # Use lspci, device files, or vendor CLI to detect hardware.
  # If found, set:
  #   device_type="mytarget"
  #   mach="<arch-string>"   # e.g. "mt100"
  #   mytarget_arch="mt100"
  local _devstr="$(lspci | grep -i 'MyVendor' | head -1)"
  [[ -z "$_devstr" ]] && return
  mytarget_arch="mt100"
  device_type="mytarget"
  mach="${mytarget_arch}"
}

add_mytarget_arch() {
  local tgts="$@"
  for tgt in ${tgts}; do
    if [[ "${tgt}" == "MT100" ]]; then set_add REQ_TARGETS "mt100";
    elif [[ "${tgt}" == "MYTARGETALL" ]]; then
      set_add REQ_TARGETS "mt100" "mt200"
    fi
  done
}

mytarget_command() {
  declare -n cmd_ref="$1"
  cmd_ref="${cmd_ref//%mytarget_arch/-arch ${mytarget_arch}}"
  cmd_ref="${cmd_ref//%target/-t mytarget}"
}

# --- Register hooks ---
register_hook "hw_detect"  "mytarget_detect"
register_hook "set_archs"  "add_mytarget_arch"
register_hook "target_cmd" "mytarget_command"
```

### 2. Add tests

Place `.co` files under `tests/mytarget/` and use standard `RUN:` directives.  For hardware-guarded tests:

```co
// REQUIRES: TARGET-MT100
// RUN: choreo %target -gs %s -o %s.result && bash %s.result --execute | FileCheck %s
```

### 3. Share `tests/check/` tests (optional)

If `tests/check/` tests should also run under the new target's hooks (so `%target` resolves to `-t mytarget`), add an `include_dir` call at the bottom of `tests/mytarget/lit.cfg`:

```bash
# At the bottom of tests/mytarget/lit.cfg:
include_dir "../check"
```

More than one directory can be included; add one call per directory.

### 4. Add `%target` to relevant `tests/check/` tests

If a check test's expected output differs by target, add `%target` to its `RUN:` line:

```co
// RUN: choreo %target -es --runtime-check=all %s -o - | FileCheck %s
```

Tests without `%target` will simply run the same command for every target that includes them -- which is fine if the expected output is target-agnostic.

---

## `include_dir` -- Sharing Tests Across Targets

A `lit.cfg` may call `include_dir("../path")` to declare that another directory's tests should run under **that target's** hooks.  This removes the need to copy tests or maintain per-target symlinks.

**Declaration** -- one call per shared directory, placed at the bottom of `lit.cfg`:
```bash
# in tests/mytarget/lit.cfg
include_dir "../check"   # run tests/check/ under mytarget hooks
include_dir "../norm"    # also run tests/norm/ under mytarget hooks
```

**Behaviour when running `lit.sh tests/` (a parent directory):**
1. `lit.sh` sources every `lit.cfg` found under the given directory in *discovery mode*, collecting all `include_dir` calls.
2. For each declared directory, every `.co` / `.cmt` file is added to the test list with the declaring target's cfg context.
3. Files claimed by at least one `include_dir` are **suppressed** from their own direct (hookless) run -- each file runs exactly once per including target.
4. A file with N `RUN:` lines included by M targets executes NxM times total.

**Behaviour when running a shared directory directly** (e.g. `lit.sh tests/check/`):
- Discovery mode is not active; `include_dir` calls are collected but no files are re-routed.
- `%target` is stripped to empty; tests run with the default target.
- Safe to use for rapid iteration on frontend checks.

---

## Test Count Reference

When running the full suite (`lit.sh tests/`) with targets that use `include_dir`:

| Test category | Run count |
|---------------|-----------|
| `tests/check/*.co` (shared via `include_dir`) | 1x per including target per file |
| Target-specific files (e.g. `tests/mytarget/**/*.co`) | 1x per file |
| Other directories (parse/, norm/, ...) | 1x per file |

A file with 2 `RUN:` lines included by 2 targets executes **4 times** total -- this is expected and correct.

## Dry Run

To see which commands would execute without running them, use the `--dry-run` flag:

```bash
./tests/lit.sh --dry-run tests/ 2>dryrun.txt
```

Each command is printed as `DRYRUN: <file>` to stderr.  The normal PASS/FAIL/SKIP output still appears on stdout with zero-time entries.  Use this to verify test counts after topology changes:

```bash
# Count per-file run frequency:
grep 'DRYRUN:.*check/' dryrun.txt | sed 's|DRYRUN: ||' | sort | uniq -c | sort -rn

# Verify shared files are not running hookless (should print 0):
grep 'DRYRUN: tests/check/' dryrun.txt | wc -l
```

---

## Configuration Chain

When processing a test file, `lit.sh` walks **up** from the file's directory to the directory containing `lit.sh` (or `/`), collecting every `lit.cfg` that carries the `# co-lit` marker.  The collected configs are then sourced in **parent-first** order so that child configs can override parent settings:

```
tests/mytarget/subdir/file.co
  walk up:  subdir/ -> mytarget/ (lit.cfg!) -> tests/ (lit.sh lives here, stop)
  source order:  tests/mytarget/lit.cfg

other/deep/nested/file.co
  walk up:  nested/ -> deep/ (lit.cfg!) -> other/ -> ... -> /
  source order:  other/deep/lit.cfg
```

The walk-up uses the `# co-lit` marker to skip non-Croqtile configs, so there are no hardcoded directory names -- any directory tree works.

When a file is processed via `include_dir` (cfg override), the walk starts from the **declaring** directory, not the file's own directory.  This is what gives files in `tests/check/` access to the hooks defined in the declaring target's `lit.cfg`.

---

## See Also

- [AGENTS.md](../../AGENTS.md) -- Build commands, pass names, compiler options
- [`tests/lit.sh`](../../tests/lit.sh) -- Runner source (v0.33+)
