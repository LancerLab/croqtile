# co-mock -- Choreo Mock Interpreter

`co-mock` is a standalone tool that interprets Choreo (`.co`) programs without
any hardware backend. It is useful for:

- **Testing program logic** -- verify control flow, arithmetic, and data
  movement semantics without a GPU or other accelerator.
- **Interactive debugging** -- step through Choreo programs line by line,
  inspect variables, set breakpoints.
- **CI/regression testing** -- run `.co` test suites quickly on any machine.

## Building

`co-mock` is built automatically with the Choreo project:

```bash
make build
```

The binary is placed at `build/co-mock` and symlinked to the repo root.

To disable building `co-mock`, pass `-DCHOREO_BUILD_CO_MOCK=OFF` to CMake.

## Usage

```
co-mock [options] <input.co>
```

### Options

| Flag               | Description                              |
|--------------------|------------------------------------------|
| `-e, --dump-ast`   | Dump the AST after semantic analysis     |
| `-i, --interactive`| Start in interactive debugger mode       |
| `-s, --script <f>` | Run debugger commands from script file   |
| `-h, --help`       | Show help message                        |

### Run a program

```bash
co-mock test.co
```

### Interactive debugging

```bash
co-mock -i test.co
```

This drops you into a pdb-style debugger that stops before the first
statement:

### Script mode

```bash
co-mock -s commands.txt test.co
```

Script mode feeds debugger commands from a file. When the script is
exhausted, execution continues to completion. This enables automated
debugger testing:

```
co-mock debugger -- type 'h' for help
Debugging: test.co

-> 4    s32[4] a;
(co-mock) _
```

### Debugger commands

| Command             | Description                                      |
|---------------------|--------------------------------------------------|
| `s`, `step`         | Execute one statement, descending into blocks     |
| `n`, `next`         | Execute one statement, stepping over blocks       |
| `c`, `continue`     | Run until next breakpoint or program end          |
| `p <var>`           | Print variable value                              |
| `p <var>[i]`        | Print element `i` of an array variable            |
| `l`, `list`         | Show source code around the current line          |
| `b <line>`          | Set a breakpoint at a source line                 |
| `d <line>`          | Delete a breakpoint (`d` alone clears all)        |
| `info`              | Show all variables in all active scopes           |
| `info futures`      | Show async DMA future status (pending/completed) |
| `info mem`          | Show memory allocations with sizes                |
| `info break`        | Show all breakpoints                              |
| `q`, `quit`         | Exit the debugger                                 |
| `<Enter>`           | Repeat the last step action                       |
| `<varname>`         | Shorthand for `p <varname>`                       |

### Example session

```
$ co-mock -i add.co
co-mock debugger -- type 'h' for help
Debugging: add.co

-> 4    s32[4] a;
(co-mock) b 16
Breakpoint set at line 16
(co-mock) c
B> 16    println(c.at(0));
(co-mock) p c
c = <global s32[4] @ 0x...>
  [11, 22, 33, 44]  (4 elements)
(co-mock) p c[2]
c[2] = 33
(co-mock) info
--- scope 1 ---
  c = <global s32[4] @ 0x...>  [11, 22, 33, 44]
  b = <global s32[4] @ 0x...>  [10, 20, 30, 40]
  a = <global s32[4] @ 0x...>  [1, 2, 3, 4]
(co-mock) c
11
22
33
44
```

## Architecture

`co-mock` uses Choreo as an SDK:

```
.co source --> [Preprocessor] --> [Parser] --> [Semantic Analysis] --> [Interpreter]
                                                                        |
                                                          MockInterpreter + MockMemory
                                                                        |
                                                              Debugger (if -i)
```

- **Frontend**: Reuses Choreo's preprocessor, parser, and semantic pipeline
  (type inference, normalization, etc.) by linking against the Choreo libraries.
- **Backend**: Instead of code generation, the `MockInterpreter` directly
  traverses the analyzed AST and executes it using a virtual memory model
  (`MockMemory`) with LOCAL, SHARED, and GLOBAL storage classes.
- **Debugger**: An interactive command loop that hooks into the interpreter's
  statement dispatch, providing pdb-style stepping and inspection.

The interpreter supports:
- Scalar and array variables with full type system
- `parallel p by N` blocks with threaded execution
- Nested `parallel-by` with correct thread-to-index mapping
- `.at(idx)` element access returning scalar values
- `foreach` loops, `while` loops, `if-else` blocks
- `break`, `continue`, and `return` control flow
- Synchronous and asynchronous DMA with real `std::async` futures
- `wait()` blocking on async DMA futures
- `rotate()` for circular variable swapping
- `InThreadsBlock` predicate-guarded execution
- Built-in functions (`println`, `print`, `assert`, `alignup`, `aligndown`,
  `sqrt`, `sin`, `cos`, `tan`, `exp`, `log`, `pow`, and more)
- 2D array indexing

## Tests

Tests live in `tools/co-mock/tests/` and use `// RUN:` + `FileCheck` directives,
run by the project's shared `lit.sh` test harness.

```bash
# Run all co-mock tests
./tests/lit.sh tools/co-mock/tests/

# Run a single test
./tests/lit.sh tools/co-mock/tests/add.co
```

## Directory Layout

```
tools/co-mock/
  CMakeLists.txt        # Self-contained build configuration
  README.md             # This file
  mock_main.cpp         # Entry point with custom argument parsing
  mock_interp.hpp/cpp   # AST interpreter engine
  mock_memory.hpp/cpp   # Virtual memory model
  debugger.hpp/cpp      # Interactive debugger (pdb-style)
  tests/
    lit.cfg             # Lit test configuration
    add.co              # Arithmetic test
    ifelse.co           # Conditional test
    while.co            # While loop test
    foreach.co          # Foreach loop test
    ...
```
