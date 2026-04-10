# Building and Testing Croqtile

This guide covers how to build the Croqtile compiler from source, run the test suite, and work with the build system.

---

## Prerequisites

- **C++17 compiler**: GCC 9.0+ or Clang 5.0+
- **CMake** 3.18+
- **Ninja** (build system)
- **Flex** and **Bison** (parser generation)

Run the setup target to fetch toolchain dependencies:

```bash
make setup
```

---

## Build Targets

| Command | Description | Output Directory |
|---------|-------------|------------------|
| `make` or `make build` | Default build (Release) | `build/` |
| `make debug` | Debug build with symbols | `build-debug/` |
| `make release` | Explicit release build | `build-release/` |
| `make clean` | Remove all build artifacts | -- |

The build produces two binaries, symlinked to the repo root:

- `./choreo` -- the main compiler
- `./copp` -- the preprocessor

### Build internals

The build uses CMake + Ninja under the hood. The Makefile wraps these commands:

```bash
cmake -G Ninja -B build -DCMAKE_BUILD_TYPE=Release
ninja -C build
```

**Do not invoke `cmake` directly** -- the Makefile configures target-specific flags, submodule paths, and feature toggles automatically.

### SDK build

```bash
make sdk           # build SDK headers + library
make sdk-install   # install to local prefix
make sdk-test      # run SDK test suite
```

### Package build

```bash
make package       # public release package
make package-full  # full package with all targets
```

---

## Running Tests

### Full test suite

```bash
make test              # default build + all tests
make test-debug        # debug build + all tests
make test-release      # release build + all tests
make JOBS=4 test       # parallel test execution (4 jobs)
```

### Single test

Use `lit.sh` directly for individual tests:

```bash
./tests/lit.sh tests/check/if_hoist.co        # single file
./tests/lit.sh tests/check/                    # entire directory
./tests/lit.sh -j4 tests/                      # parallel full suite
```

Target-specific end-to-end tests may require additional hardware or SDK configuration; see [Target Environment Setup](target/env_setting_up.md).

### Standalone tests

```bash
cd tests/standalone/ && make test
```

### Dry run

See what commands would execute without running them:

```bash
./tests/lit.sh --dry-run tests/ 2>dryrun.txt
```

For details on the test runner's directives, hooks, and target integration, see [Lit Test Runner](lit-test-runner.md).

---

## Compiler Usage

### Basic compilation

```bash
./choreo -t <target> program.co        # end-to-end compile for a target
./choreo -t <target> -es program.co    # emit target source only (no target compiler)
./choreo -t <target> -gs program.co    # generate work-script
```

### Debug flags

| Flag | Description |
|------|-------------|
| `-e` | Dump AST after parsing |
| `-i` / `-ii` | Show type inference results |
| `-vn` | Print value numbering trace |
| `-pa=PASS` | Print AST after a specific pass |
| `-sa=PASS` | Stop after a specific pass |
| `-es` | Emit generated source only |
| `-gs` | Generate work-script |
| `-t TARGET` | Set target platform |
| `-arch=ARCH` | Set target architecture |

### Pass names

```
Source -> SEMA -> NORM -> VALNO -> INFER -> LATENORM -> CHECK -> CODEGEN -> Target
```

Use `-pa=PASS` and `-sa=PASS` to inspect intermediate states. See [Compilation Passes](compilation-passes.md) for details.

---

## Code Formatting

The project uses clang-format with LLVM-based rules. Format before every commit:

```bash
make format
```

See [Coding Style](coding-style.md) for the full style guide.

---

## CI and Pre-commit

```bash
make ci-test                                 # CI test target
scripts/oss/oss-scan.sh --staged             # check OSS compliance on staged files
scripts/oss/oss-scan.sh                      # scan entire worktree
```

The OSS scan checks all tracked text files -- including `Documents/Documentation/` and `Documents/Developer/` -- for forbidden keywords, non-ASCII content, and ghost references to excluded paths.

An optional pre-commit hook is available:

```bash
make setup-git-hooks            # install formatting check hook
```

---

## See Also

- [Lit Test Runner](lit-test-runner.md) -- test directives, hooks, `%` substitutions
- [Compilation Passes](compilation-passes.md) -- pass pipeline and debugging
- [Coding Style](coding-style.md) -- formatting, naming, and code patterns
- [Developing a Target](developing-a-target.md) -- adding a new compilation backend
