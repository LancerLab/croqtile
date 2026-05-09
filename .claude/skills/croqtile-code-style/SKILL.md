---
name: croqtile-code-style
description: |
  C++ code style, engineering practices, and review standards for Croqtile.
  Use when reviewing PRs, checking naming conventions, applying clang-format,
  writing modern C++, running static analysis, or conducting code reviews.
  Covers project conventions, C++17 patterns, clang-tidy, sanitizers, and
  review checklists with severity labels.
---

# Croqtile Code Style

This skill covers C++ conventions, quality tools, and code review for the Croqtile project. For build/test/run/.co workflows, see the `croqtile-developer` skill.

Unified C++ code style, engineering practices, and code review standards for the Croqtile C++ compiler project. Project-specific conventions in this document take precedence over generic C++ guidance below.

## Section 1: Project Conventions (Croqtile-Specific)

### Naming conventions

| Kind | Convention | Examples |
|------|--------------|----------|
| Classes and types | PascalCase | `class MultiNodes` |
| Functions | camelCase | `void accept(Visitor& v)` |
| Variables | snake_case | `auto& sub`, `init_expr` |
| Constants | snake_case | `max_depth`, `default_arity` |
| Namespaces | PascalCase | `Choreo::AST::` |
| Member variables | snake_case, optional `_` suffix | `counter_`, `name` |

### Include order

1. Project headers first
2. External library headers
3. C++ standard library headers

### Error handling and invariants

- Use `assert()` for internal compiler invariants.
- Prefer assert messages that name the failing invariant and, when helpful, which compiler pass or visitor context applies (e.g. which pass produced inconsistent state).
- Destructors should be `noexcept` and must not throw.
- Use RAII for cleanup; write exception-safe code where exceptions are used.

### Architecture patterns

- Pass-based compiler organization.
- Visitor pattern for traversing the AST and similar structures.
- Typical namespace layout includes `Choreo::AST::` for AST-related code.

### Formatting and clang-format

Run `make format` to apply the project formatter (LLVM `clang-format` using the repository `.clang-format`).

Project `.clang-format` settings (do not replace with a different preset):

```
BasedOnStyle: LLVM
IndentWidth: 2
ColumnLimit: 80
PointerAlignment: Left
AlwaysBreakTemplateDeclarations: Yes
AllowShortIfStatementsOnASingleLine: true
AllowShortBlocksOnASingleLine: Always
AllowShortLoopsOnASingleLine: true
AllowShortFunctionsOnASingleLine: true
AllowShortCaseLabelsOnASingleLine: true
DerivePointerAlignment: false
IndentPPDirectives: BeforeHash
```

## Section 2: Modern C++ Guidelines

See `modern-cpp-guidelines.md` for the full reference. Key topics:

- RAII and ownership (unique_ptr, shared_ptr, Rule of 0/3/5)
- Move semantics and const-correctness
- Virtual functions (override, final)
- Template metaprogramming (concepts -- C++20+ informational)
- Memory management (smart pointers, custom allocators)
- Concurrency (atomics, mutex, lock_guard)
- Error handling (exception safety, noexcept)

Note: This project uses C++17. Features marked "C++20+" are informational.

## Section 3: Performance Patterns

### Cache-friendly access

- Prefer Structure of Arrays (SoA) over Array of Structures (AoS) when iterating hot fields.
- Consider data alignment for SIMD-friendly layouts.
- Prefetch when access patterns are predictable.

### Reserve for vectors

```cpp
std::vector<int> build(int n) {
    std::vector<int> out;
    out.reserve(n);  // Avoid reallocations
    for (int i = 0; i < n; ++i) {
        out.push_back(i);
    }
    return out;
}
```

### Algorithmic complexity and hot paths

- Be explicit about expected complexity for passes and visitors on large inputs.
- Avoid redundant work on hot paths: unnecessary allocations, copies, or virtual dispatch where a tighter design helps.
- Measure before micro-optimizing; preserve clear structure unless profiling shows benefit.

## Section 4: Static Analysis and Quality Tools

### clang-tidy

```bash
clang-tidy src/*.cpp -- -std=c++17
clang-tidy src/main.cpp -checks='-*,readability-*,modernize-*'
clang-tidy src/main.cpp -fix
```

Note: `compile_commands.json` is generated automatically by the CMake build. Do NOT run `cmake` directly; just run `make` and the file will be in `build/`.

### cppcheck

```bash
# Basic check
cppcheck --std=c++17 src/

# Enable all checks
cppcheck --enable=all --std=c++17 src/

# Suppress specific warnings
cppcheck --suppress=unusedFunction src/
```

### Include What You Use (IWYU)

```bash
include-what-you-use main.cpp -- -std=c++17 -I./include
```

### clang-format

The authoritative style is the repository `.clang-format`; use `make format` in Croqtile rather than ad hoc style presets.

Optional local checks:

```bash
# Format files directly (when not using make)
clang-format -i src/*.cpp include/*.h

# Check formatting
clang-format --dry-run -Werror src/*.cpp
```

### Sanitizers

#### AddressSanitizer (ASan)

Detects memory leaks, use-after-free, buffer overflows.

```bash
clang++ -fsanitize=address -fno-omit-frame-pointer -g main.cpp -o main
./main
```

#### UndefinedBehaviorSanitizer (UBSan)

Detects null dereference, signed integer overflow, invalid casts, and related undefined behavior.

```bash
clang++ -fsanitize=undefined -fno-omit-frame-pointer -g main.cpp -o main
```

#### ThreadSanitizer (TSan)

Detects data races.

```bash
clang++ -fsanitize=thread -fno-omit-frame-pointer -g main.cpp -o main
```

#### MemorySanitizer (MSan)

Detects use of uninitialized memory.

```bash
clang++ -fsanitize=memory -fno-omit-frame-pointer -g main.cpp -o main
```

## Section 5: Build Reference

For build, test, and run workflows, see the `croqtile-developer` skill. All commands run from the project root via `make`. Do NOT run `cmake`, `ninja`, or enter build directories directly.

## Section 6: Code Review Checklist

Transform code reviews into knowledge sharing through constructive feedback, systematic analysis, and collaborative improvement.

### When to use this guidance

- Reviewing pull requests and code changes
- Conducting architecture reviews
- Mentoring through reviews
- Maintaining code quality for C and C++ code

### Core principles

**Goals of code review:**

- Catch bugs and edge cases
- Ensure maintainability
- Share knowledge across contributors
- Enforce project conventions (including Section 1)
- Improve design where it matters

**Effective feedback is:**

- Specific and actionable
- Educational, not judgmental
- Focused on the code, not the person
- Prioritized (critical versus nice-to-have)

**Severity labels:**

- `[blocking]` -- must fix before merge
- `[important]` -- should fix; discuss if you disagree
- `[nit]` -- optional polish
- `[suggestion]` -- alternative approach to consider
- `[praise]` -- highlight good work

### Review techniques

**Checklist method:** use the checklist below for consistent reviews.

**Question approach:** ask questions instead of only stating problems, for example:

- "What should happen if this container is empty?"
- "How should this behave if the input is invalid?"

**Suggest, do not command:** use collaborative language:

- "Would it be worth extracting this repeated logic?"
- "Could we tighten the lifetime story here so ownership is obvious?"

### C++ review checklist

**Safety and lifetime**

- [ ] Ownership is explicit (RAII; `unique_ptr` by default where heap allocation is needed)
- [ ] No dangling references or views
- [ ] Rule of 0 / 3 / 5 followed for resource-owning types
- [ ] No raw `new` / `delete` in normal business logic
- [ ] Destructors are `noexcept` and do not throw

**API and design**

- [ ] Const-correctness applied consistently
- [ ] Constructors marked `explicit` where implicit conversion would be harmful
- [ ] `override` / `final` used appropriately for virtual functions
- [ ] No object slicing (pass polymorphic objects by reference or pointer)

**Concurrency**

- [ ] Shared mutable state is protected (mutex or atomics)
- [ ] Lock ordering is consistent to avoid deadlock
- [ ] No blocking or heavy work while holding locks inappropriately

**Performance**

- [ ] Unnecessary allocations avoided (`reserve`, moves)
- [ ] Copies avoided on hot paths where profiling or complexity arguments support it
- [ ] Algorithmic complexity is appropriate for the use case

**Tooling and tests**

- [ ] Builds clean with project warning settings
- [ ] Sanitizers exercised on critical or risky changes when practical
- [ ] `clang-tidy` findings addressed when part of the workflow

### Architecture review guide

For larger changes, assess:

- SOLID-style separation of concerns where applicable
- Coupling versus cohesion between modules and passes
- Anti-patterns (god objects, unclear visitor responsibilities, leaky abstractions)
- Fit of patterns (visitors, AST design) to the problem

### Performance review guide

For hot or scalability-sensitive code, check:

- Algorithmic complexity
- Allocation patterns and container growth
- Cache locality of data structures
- Opportunities for hot path improvements grounded in measurement

## Section 7: Performance Profiling

### perf

```bash
perf record -g ./myprogram
perf report
perf stat ./myprogram
```

### gprof

```bash
g++ -pg -g main.cpp -o main
./main
gprof ./main gmon.out > analysis.txt
```

For GPU or CUDA-oriented profiling (for example `nsys`, `ncu`), refer to the `croqtile-developer` skill when working on GPU-related parts of the stack.

## Related skills

- `croqtile-developer` -- building, testing, running .co files, compiler development, debugging
- `modern-cpp-guidelines.md` -- detailed C++ patterns and code examples (supporting file)
