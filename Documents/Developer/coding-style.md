# Croqtile C++ Coding Style Guide

This document defines the coding conventions for the Croqtile compiler and
runtime. All contributors must follow these rules. Run `make format` before
every commit.

---

## 1. Formatting

The project uses **clang-format** with an LLVM-based configuration. The
`.clang-format` file at the repository root is the authoritative source.

Key settings:

| Setting | Value |
|---------|-------|
| Base style | LLVM |
| Indent width | 2 spaces (no tabs) |
| Column limit | 80 characters |
| Pointer alignment | Left (`int* p`, not `int *p`) |
| Template declarations | Always break before |
| Short if/loop/function | Allowed on single line |

### Apply formatting

```bash
make format
```

This formats all `.cpp`, `.hpp`, `.h`, `.cu`, and `.cuh` files under `lib/`,
`runtime/`, and `tests/standalone/`.

### Pre-commit hook

The repository has an optional pre-commit hook (`scripts/hooks/pre-commit-check.sh`)
that can be installed via `make setup-git-hooks`. It verifies formatting
before each commit.

---

## 2. Naming Conventions

| Element | Convention | Example |
|---------|------------|---------|
| Classes / Structs / Types | PascalCase | `MultiNodes`, `LoopRange`, `ShapeInfer` |
| Functions / Methods | camelCase | `accept()`, `getType()`, `inferShape()` |
| Variables | snake_case | `init_expr`, `num_threads`, `loop_body` |
| Constants | snake_case | `max_buffer_size`, `default_align` |
| Namespaces | PascalCase | `Choreo::AST`, `Choreo::CodeGen` |
| Enum values | PascalCase | `NodeType::Parameter`, `DMAKind::Async` |
| Member variables | snake_case (optional trailing `_`) | `values_`, `parent`, `sym_table` |
| Macros | UPPER_SNAKE_CASE | `__CHOREO_TARGET_CUTE__` |
| File names | snake_case or existing convention | `cute_codegen.cpp`, `shapeinfer.hpp` |

### Getters and setters

Prefer short getter names without `get` prefix when unambiguous:

```cpp
class Expr {
public:
  auto type() const { return type_; }
  void setType(Type t) { type_ = t; }
};
```

For compound getters that compute a value, use `Get` prefix:

```cpp
auto GetR() const { return value_r; }
auto GetAlignedSize() const { return align_up(size_, alignment_); }
```

---

## 3. File Organization

### Directory structure

| Directory | Purpose |
|-----------|---------|
| `lib/` | Compiler source code (passes, AST, codegen) |
| `lib/Target/` | Target-specific backends |
| `runtime/` | Runtime headers shipped with compiled programs |
| `tools/` | Main binary entry points (`choreo`, `copp`) |
| `tests/` | Test files (`.co` files with `RUN:` directives) |
| `samples/` | Example programs |
| `benchmark/` | Performance benchmarks |
| `Documents/` | Documentation |

### Include order

1. Project-local headers
2. External library headers
3. Standard library headers

```cpp
#include "visitor.hpp"
#include "ast.hpp"

#include <extern/cutlass/cute/tensor.hpp>

#include <vector>
#include <string>
#include <cassert>
```

Separate each group with a blank line. Within each group, sort
alphabetically.

---

## 4. Language Standard

- **C++17** is required. Do not use C++20 or later features.
- Compiler support: GCC 9.0+ or Clang 5.0+.

### Preferred C++17 features

- `std::optional`, `std::variant`, `std::string_view`
- Structured bindings: `auto [key, value] = pair;`
- `if constexpr` for compile-time branching
- `[[nodiscard]]`, `[[maybe_unused]]` attributes
- Fold expressions for variadic templates

---

## 5. Code Patterns

### Namespaces

Always use explicit namespace blocks. Do not use `using namespace` in headers.

```cpp
namespace Choreo {
namespace AST {

class MultiNodes {
  // ...
};

} // namespace AST
} // namespace Choreo
```

### Visitor pattern

Croqtile's AST uses the visitor pattern extensively:

```cpp
void MultiNodes::accept(Choreo::Visitor& v) {
  v.BeforeVisit(*this);
  for (auto& sub : values) sub->accept(v);
  v.Visit(*this);
  v.AfterVisit(*this);
}
```

### Error handling

- Use `assert()` for internal invariants and programming errors.
- Use descriptive error messages that include context (pass name, node type).
- The compiler uses pass-based architecture; errors should indicate the
  failing pass.

```cpp
assert(node->type() != nullptr && "type must be resolved before codegen");
```

### RAII and ownership

- Use `std::unique_ptr` for owning pointers.
- Use raw pointers or references for non-owning access.
- Prefer stack allocation when lifetime is bounded by scope.

---

## 6. Comments

### When to comment

- **Do** comment non-obvious algorithms, workarounds, and design decisions.
- **Do** comment public API functions in headers.
- **Do not** write comments that merely restate what the code does.
- **Do not** leave commented-out code in the repository.

### Style

```cpp
// Single-line comments for brief notes.

// Multi-line comments for longer explanations. Wrap at the column
// limit. Start each continuation line with //.
```

Use `// TODO:` for known incomplete work. Include a tracking reference
(issue number or description) when possible:

```cpp
// TODO(#42): handle the edge case for zero-length spans.
```

---

## 7. Public Code Policy (OSS Compliance)

Code that may be synced to the public open-source repository must follow
additional rules:

1. **ASCII only** -- no non-ASCII characters in source files. Use `--` instead
   of em-dashes, `->` instead of Unicode arrows, English instead of CJK.
2. **No proprietary keywords** -- see `scripts/oss/os_kw.txt` for the
   blacklist. Avoid hardware codenames, internal tool names, and company
   identifiers.
3. **No references to excluded paths** -- do not `#include` files from
   excluded directories.

Verify compliance before committing:

```bash
make oss-scan-staged
```

Full OSS workflow documentation: `Documents/internal/oss-sync-developer-guide.md`

---

## 8. Quick Checklist

Before every commit:

- [ ] Run `make format`
- [ ] Run `make build` to verify compilation
- [ ] If touching test files, run `make test` or `./tests/lit.sh <file>`
- [ ] If the change will be synced publicly, run `make oss-scan-staged`
- [ ] No non-ASCII characters in source code
- [ ] No commented-out code blocks
- [ ] No `using namespace` in headers
