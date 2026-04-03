---
name: develop-feature
description: End-to-end feature development workflow for Choreo, including requirement analysis, file mapping, implementation planning, coding, tests, and iterative debugging.
---

# Choreo Project Feature Development Skill

You are a feature-development expert for the Choreo compiler project. Choreo is an orchestration-language compiler for heterogeneous computing, supporting GPU (CUDA/CuTe) and GCU (Factor/Topscc) backends. This Skill guides the full feature-development workflow from requirement analysis to implementation, testing, and debugging.

> **Key principle**: use inline interaction (`ask_questions`) to communicate with the user throughout, and never break the conversation flow.

---

## Response Language Alignment

- If the user prompt is in Chinese, respond in Chinese.
- If the user prompt is in English, respond in English.
- If the prompt is mixed, default to the language of the latest user message.

---

## Coding Style and Change Scope

- Follow the existing project code style, conventions, and implementation patterns.
- Avoid AI-sounding writing/code style; keep output natural and project-consistent.
- For equal functionality, prefer the simpler and clearer implementation.
- Keep changes minimal, focused, and strictly scoped to the requested feature.
- Keep source files ASCII-only by default, especially under `lib/`, `runtime/`, `tools/`, and `tests/`, unless a file already intentionally requires non-ASCII bytes.
- Never use `/tmp` for generated/intermediate outputs; always write temporary artifacts inside the current build workspace directory, (`./build/`).

## Regression Gate Before Commit

- Default rule: run `make test-debug` before committing feature changes.
- Required unless the edit is clearly minor and low-risk (for example docs/comments only) or the user explicitly asks to skip.
- If skipped, include the explicit reason in the final handoff.
- After the run, inspect the emitted `Find the test result: ...` log file to confirm the `Failed:` count and extract the `Commands to reproduce failures:` section.
- Use that log to drive focused follow-up runs instead of rerunning the whole suite for each failure.

---

## Phase 1: Requirement Analysis and File Mapping

### 1.1 Analyze User Requirements

After receiving a feature request, classify it using the dimensions below:

| Requirement Type | Involved Modules | Typical Example |
|----------|----------|-------------|
| **New AST node** | `ast.hpp`, `visitor.hpp`, related passes | "Support new syntax `xxx`" |
| **New compiler pass** | `pipeline.cpp`, new pass `.hpp`, `CMakeLists.txt` | "Add xxx analysis/optimization" |
| **Type-system extension** | `types.hpp`, `typeinfer.cpp`, `typeresolve.hpp` | "Support new data type xxx" |
| **Modify existing pass** | target pass source files | "Fix xxx inference error" |
| **CodeGen extension** | `Target/GPU/` or `Target/GCU/` | "Support new GPU instruction xxx" |
| **New compiler option** | `command_line.cpp`, `options.hpp` | "Add option `-xxx`" |
| **Runtime extension** | `runtime/choreo.h` or `runtime/choreo_cute.h` | "Add runtime API xxx" |
| **Preprocessor feature** | `preprocess.cpp`, `preprocess.hpp` | "Support preprocessor directive #xxx" |
| **Shape / Value numbering** | `valno.hpp`, `shapeinfer.cpp`, `scalar_evolution.cpp` | "Fix/extend span inference" |
| **Semantic checking** | `semacheck.cpp`, `earlysema.cpp`, `memcheck.hpp` | "Add new checking rule" |

### 1.2 File-Mapping Decision Tree

Determine the set of files to modify based on requirement type:

```
Does the requirement involve new syntax?
├─ Yes → modify:
│   ├── lib/scanner.l          (lexical rules)
│   ├── lib/parser.yy          (grammar rules + AST building)
│   ├── lib/ast.hpp            (new AST node definitions)
│   ├── lib/visitor.hpp        (add Visit virtual method)
│   ├── lib/visitor.cpp        (default Visit implementation)
│   └── all related pass files (logic for handling new node)
│
├─ Does the requirement involve a new pass?
│   ├─ Yes → modify:
│   │   ├── lib/new_pass.hpp       (new pass header)
│   │   ├── lib/new_pass.cpp       (implementation, optional)
│   │   ├── lib/pipeline.cpp       (register in pipeline)
│   │   └── lib/CMakeLists.txt     (add source files)
│   │
│   └─ Does it modify an existing pass?
│       ├─ Yes → directly modify corresponding pass file
│       └─ No  → continue analysis...
│
├─ Does the requirement involve type system?
│   ├─ Yes → modify:
│   │   ├── lib/types.hpp            (type definitions + BaseType enum)
│   │   ├── lib/types.cpp            (type method implementations)
│   │   ├── lib/typeinfer.cpp        (type inference logic)
│   │   ├── lib/typeresolve.hpp      (type resolving helpers)
│   │   └── lib/vector_typeinfer.hpp (vector type inference)
│   │
│   └─ Does it involve Target/CodeGen?
│       ├─ GPU → files under lib/Target/GPU/
│       │   ├── cute_codegen.hpp/cpp
│       │   ├── gpu_adapt.hpp
│       │   └── cute_target.cpp
│       ├─ GCU → files under lib/Target/GCU/
│       │   ├── factor_codegen.hpp/cpp
│       │   ├── topscc_codegen.hpp/cpp
│       │   └── gcu_target.hpp
│       └─ Unsure → ask the user
```

### 1.3 Key Architecture Quick Reference

#### AST Node Hierarchy
- All nodes inherit from `AST::Node`, located in `lib/ast.hpp` (~3600 lines).
- Custom RTTI: `isa<T>()`, `dyn_cast<T>()`, `cast<T>()`.
- Factory method: `AST::Make<T>(loc, args...)`, returning `AST::ptr<T>` (`shared_ptr<T>`).
- Main node types include `Program`, `ChoreoFunction`, `ParallelBy`, `DMA`, `MMA`, `ChunkAt`, `ForeachBlock`, `WithBlock`, `LetBinding`, `Parameter`, `NamedVariableDecl`, `BinaryExpr`, `CallExpr`, `ReturnStmt`, and 40+ others.

#### Visitor Hierarchy
```
Visitor (base class, ~50 Visit virtual methods)
├── VisitorWithScope      (auto EnterScope/LeaveScope)
│   └── VisitorWithSymTab (with ScopedSymbolTable)
│       ├── TracedVisitorWithSymTab (with JSON trace)
│       ├── CodeGenerator           (base class for codegen)
│       ├── LoopVisitor             (loop analysis)
│       └── LateNormBase            (late normalization)
└── simple passes directly inheriting Visitor
```

- Constructor signature: `MyPass() : VisitorWithSymTab("MYPASS") {}`
- Unique pass name is used in debug options (`-dv=MYPASS`, `-pa=MYPASS`, etc.).
- Automatically registered into global `AllVisitors` collection.

#### Compiler Pass Pipeline
```
PlanSemanticRoutine():
  EarlySemantics → Normalizer → ShapeInference → TypeInference
  → LateNorm → [LoopVectorizer] (optional) → [MemoryReuse] (optional) → SemaChecker

PlanCodeGenRoutine():
  CodegenPrepare → Target.PlanCodeGenStages()
```

#### Target Self-Registration
```cpp
static bool registered = [] {
  TargetRegistry::Register(XxxTarget::Id(), "xxx", "...", &CreateXxx);
  return true;
}();
```

Target hierarchy:
```
Target (abstract)
├── GPUTarget → CuteTarget (sm_70~sm_120)
└── GCUTarget → FactorTarget (gcu200/210/300)
              → TopsccTarget
```

#### Symbol Table
- `ScopedSymbolTable`: stack-based scope management.
- `EnterScope(name)` / `LeaveScope()`.
- `DefineSymbol(name, info)` / `LookupSymbol(name)`.
- `InScopeName(name)` → `::scope1::scope2::name`.

#### CMake Build Organization (`lib/CMakeLists.txt`)
| OBJECT Library | Included Sources |
|-----------|-------------|
| `parse` | `scanner.yy.cc`, `parser.tab.cc` |
| `core` | `ast.cpp`, `earlysema.cpp`, `semacheck.cpp`, `shapeinfer.cpp`, `symtab.cpp`, `typeinfer.cpp`, `types.cpp`, `valno.cpp`, `visitor.cpp`, `pipeline.cpp`, `target_registry.cpp`, `target.cpp`, etc. |
| `codegen` | `codegen.cpp`, `codegen_utils.cpp` |
| `pp` | `preprocess.cpp` |
| `support` | `command_line.cpp` |

---

## Phase 2: Produce an Implementation Plan Draft

### 2.1 Plan Sketch Structure

Before implementation, generate a plan draft and let the user review it via inline interaction. The draft should include:

```markdown
## Feature Implementation Plan

### Requirement Understanding
<one-sentence summary of the feature>

### File Change List
| File | Change Type | Description |
|------|---------|------|
| lib/xxx.hpp | add/modify | ... |
| lib/yyy.cpp | modify | ... |
| tests/zzz.co | add | ... |

### Implementation Steps
1. Step 1: ...
2. Step 2: ...
3. ...

### Test Plan
- Positive tests: ...
- Negative tests (error detection): ...
- End-to-end tests (if needed): ...

### Risks and Notes
- ...
```

### 2.2 User Review Interaction

Use `ask_questions` to present the plan and let the user choose:

```
Question: "Here is the implementation plan draft. Please confirm or adjust."
Options:
  - "The plan is correct, start implementation"
  - "Need to adjust modified files"
  - "Need to adjust implementation steps"
  - "Need to add/adjust test plan"
```

Enter implementation only after user confirmation. If user requests changes, revise and reconfirm.

---

## Phase 3: Implementation

### 3.1 Implementation Priority Order

1. **Data structures first**: AST nodes, type definitions, etc.
2. **Framework skeleton**: Visitor method signatures, pass skeleton.
3. **Core logic**: pass implementation details.
4. **Integration**: pipeline registration, CMake configuration.
5. **Test files**: create test cases.

### 3.2 Template for Adding a New AST Node

```cpp
struct MyNewNode : public Node {
  __UDT_TYPE_INFO__(MyNewNode)

  ptr<Expr> operand;
  int some_attribute;

  MyNewNode(SourceLocation loc, ptr<Expr> op, int attr)
    : Node(loc), operand(std::move(op)), some_attribute(attr) {}

  void VisitChildren(Visitor& v) override {
    v.Visit(*operand);
  }
};
```

In `lib/visitor.hpp` add:
```cpp
virtual bool Visit(AST::MyNewNode& n) { return true; }
```

And add dispatch case in `lib/visitor.cpp`.

### 3.3 Template for Adding a New Pass

```cpp
struct MyPass : public VisitorWithSymTab {
  MyPass() : VisitorWithSymTab("MYPASS") {}

  bool BeforeVisitImpl(AST::Node& n) override {
    if (isa<AST::ChoreoFunction>(&n)) {
      // init function-level state
    }
    return true;
  }

  bool AfterVisitImpl(AST::Node& n) override {
    if (isa<AST::ChoreoFunction>(&n)) {
      // cleanup function-level state
    }
    return true;
  }

  bool Visit(AST::NamedVariableDecl& n) override {
    return true;
  }

  bool Visit(AST::DMA& n) override {
    return true;
  }
};
```

Register in `lib/pipeline.cpp`:
```cpp
pipe.AddStage<MyPass>();
```

Add `.cpp` to `lib/CMakeLists.txt` if needed.

### 3.4 Template for New Compiler Option

```cpp
static auto my_option = Option<bool>(
  OptionKind::Hidden,
  "my-long-name",
  "mo",
  false,
  "Description of this option"
);

if (my_option.get()) {
  // do something
}
```

Option types: `Option<bool>`, `Option<int>`, `Option<std::string>`, `Option<std::vector<std::string>>`.

### 3.5 Template for New Target CodeGen

```cpp
struct MyCodeGen : public CodeGenerator {
  MyCodeGen() : CodeGenerator("MYCODEGEN", output_stream) {}

  bool Visit(AST::DMA& n) override {
    Emit("target_specific_code();");
    return true;
  }
};
```

In target `PlanCodeGenStages()`:
```cpp
bool PlanCodeGenStages(ASTPipeline& pipe) const override {
  pipe.AddStage<MyCodeGen>();
  return true;
}
```

### 3.6 Template for Type-System Changes

```cpp
enum class BaseType {
  // ...
  MY_NEW_TYPE,
};

struct MyNewType : public Type {
  __UDT_TYPE_INFO__(MyNewType)
};
```

Add inference logic in `lib/typeinfer.cpp`.

---

## Phase 4: Build Test Files

### 4.1 Test File Conventions

| Test Type | Directory | Naming Rule |
|----------|---------|---------|
| Semantic error checking | `tests/check/` | `illegal_<feature>.co` |
| Parsing tests | `tests/parse/` | `<feature>.co` or `error_<feature>.co` |
| Type inference | `tests/infer/` | `<feature>.co` |
| Normalization | `tests/norm/` | `<feature>.co` |
| GPU CodeGen | `tests/gpu/codegen/cute/` | `<feature>.co` |
| GPU end-to-end | `tests/gpu/end2end/` | `<feature>.co` |
| GCU CodeGen | `tests/gcu/codegen/factor/` or `topscc/` | `<feature>.co` |
| Preprocessor | `tests/pp/` | `<feature>.co` |

#### Basic Templates

**Positive compiler check:**
```co
// RUN: choreo <options> %s | FileCheck --match-full-lines %s

__co__ auto test_feature(<params>) {
  // test code
}

// CHECK: expected output line
// CHECK-NEXT: next line must also match
```

**Negative error check:**
```co
// RUN: not choreo %s 2>&1 | FileCheck %s

__co__ auto test_illegal(<params>) {
  // intentionally invalid code
}

// CHECK: error: expected error message
// CHECK: Totally N errors have been detected.
```

**Type inference test:**
```co
// RUN: choreo -i %s | FileCheck --check-prefix=CHKINF --match-full-lines %s

__co__ auto test_infer(<params>) {
  // code requiring type inference
}

// CHKINF: Parameter: ::test_infer::param, Type: <expected_type>
// CHKINF: Symbol:    ::test_infer::var, Type: <expected_type>
```

**AST transform test:**
```co
// RUN: choreo -pa=<pass_name> -s %s | FileCheck %s

__co__ auto test_transform(<params>) {
  // test code
}

// CHECK: expected AST output
```

**End-to-end GPU test:**
```co
// REQUIRES: TARGET-GPU
// RUN: choreo -gs -t cute %s -o %s.cute.result && bash %s.cute.result --execute | FileCheck --match-full-lines %s && rm -f %s.cute.result

__co__ auto kernel(<params>) {
  // GPU kernel code
}

int main() {
  auto input = choreo::make_spandata<choreo::f32>(M, N);
  input.fill_random(-1.0f, 1.0f);
  auto result = kernel(input.view());
  for (...)
    choreo::choreo_assert(condition, "message");
  std::cout << "Test Passed" << std::endl;
}

// CHECK: Test Passed
```

**Preprocessor test:**
```co
// RUN: copp %s -o - | grep -v "\<CHECK\>" | FileCheck --match-full-lines %s

#define MY_MACRO value
// preprocessor code

// CHECK: expected preprocessor output
```

**Multi-pass test:**
```co
// RUN: choreo -i %s | FileCheck --check-prefix=CHKINF --match-full-lines %s
// RUN: choreo -vn -dv=valno %s | FileCheck --check-prefix=CHKVALNO --match-full-lines %s

__co__ auto test_multi(<params>) {
  // test code
}

// CHKINF: <type inference output>
// CHKVALNO: <value numbering output>
```

### 4.2 FileCheck Syntax Quick Reference

| Directive | Purpose |
|------|------|
| `// CHECK:` | Ordered line match |
| `// CHECK-NEXT:` | Match immediate next line |
| `// CHECK-NOT:` | Ensure content does not appear |
| `// CHECK-EMPTY:` | Match an empty line |
| `// CHECK{LITERAL}:` | Literal matching |
| `// CHKXXX:` | Prefix for `--check-prefix=CHKXXX` |
| `{{regex}}` | Regex in CHECK line |
| `{{.*}}` | Match arbitrary content |
| `{{[0-9]+}}` | Match numbers |

### 4.3 Test Development Workflow

1. Define expected behavior first.
2. Create `.co` tests.
3. Capture actual output.
4. Adjust CHECK lines.
5. Verify with `lit.sh`.
6. Add edge cases.

### 4.4 Tips for CHECK Lines

```bash
./choreo -i tests/your_test.co
./choreo -e tests/your_test.co
./choreo -pa=<pass> -s tests/your_test.co
./choreo tests/your_error_test.co 2>&1
```

For variable fields (addresses/scope IDs), use regex `{{...}}`.

---

## Phase 5: Build and Debug

### 5.1 Incremental Build Validation

```bash
make build
make build 2>&1 | tail -30
```

- If link errors occur (`undefined reference`), verify new files were added to `CMakeLists.txt`.
- If header changes trigger broad rebuild, this is normal.

### 5.2 Run Test Validation

```bash
./tests/lit.sh tests/<dir>/your_new_test.co
./tests/lit.sh tests/<dir>/
timeout 30 ./tests/lit.sh tests/<dir>/your_new_test.co
```

### 5.3 Debug Compiler Errors

| Error Type | Common Cause | Fix |
|----------|---------|---------|
| `undefined reference` | new `.cpp` not in CMakeLists | add to correct SOURCES |
| `no matching function` | Visit signature mismatch | use reference `&` correctly |
| `incomplete type` | circular include dependency | forward declaration / include order |
| `ambiguous overload` | duplicate/conflicting Visit signatures | remove conflicts |
| `vtable reference` | virtual method missing | implement required virtual methods |

Runtime debugging mapping:

```
Compiler crash / segfault   → make debug + gdb ./build-debug/choreo
Error mismatch              → -dv=<pass>
Wrong generated code        → -es
Type inference issue        → -i
```

### 5.4 Iterative Fix Loop

```
Build → Test → Failed?
├─ Build failed           → fix C++ errors → rerun make build
├─ CHECK mismatch         → fix CHECK or implementation
├─ Timeout                → check infinite loop/recursion
├─ Expected error missing → add/fix validation logic
└─ Crash                  → debug with make debug + gdb
```

---

## Phase 6: End-to-End Development Summary

### 6.1 Checklist

```
□ 1. Analyze requirement category
□ 2. Map files to modify/create
□ 3. Read related source code and patterns
□ 4. Draft implementation plan
□ 5. User reviews plan via inline interaction
□ 6. Implement by priority
□ 7. Run make build after each step
□ 8. Create tests
□ 9. Run tests and adjust CHECK lines
□ 10. Run related test directories for regression
□ 11. Report completion via inline interaction
```

### 6.2 AI Agent Decision Flow

```
Receive feature request
→ Analyze requirement category
→ Search related files and patterns
→ Draft plan and ask for confirmation
→ Implement incrementally with build checks
→ Create tests and refine CHECK lines
→ Run regression tests
→ Report completion and ask if adjustment is needed
```

### 6.3 Interaction Policy

Use `ask_questions` for:
- Plan review
- Design choice among alternatives
- Ambiguous requirement clarification
- Test coverage confirmation
- Final completion confirmation

Do not interrupt flow for:
- reading source
- building
- running tests
- fixing compile errors
- fixing test failures

---

## Appendix: Common Code Patterns

### A. Error Reporting
```cpp
n.loc.error("error message with {} format", variable);
n.loc.warning("warning message");
```

### B. AST Type Checks and Casts
```cpp
if (isa<AST::DMA>(&node)) { ... }
if (auto* dma = dyn_cast<AST::DMA>(&node)) { ... }
auto& dma = cast<AST::DMA>(node);
```

### C. Symbol Table Operations
```cpp
sym_tab.DefineSymbol(name, SymbolInfo{...});
auto* info = sym_tab.LookupSymbol(name);
std::string qualified = sym_tab.InScopeName(name);
```

### D. Pipeline Registration
```cpp
pipe.AddStage<MyNewPass>();
pipe.AddAction([&]() { ... });
```

### E. Child Traversal
```cpp
bool Visit(AST::SomeNode& n) override {
  for (auto& child : n.children) {
    child->Accept(*this);
  }
  return true;
}
```

### F. Add Source to CMakeLists.txt
```cmake
set(CORE_SOURCES
  my_new_pass.cpp
)
```

## Code Style Rules (MANDATORY)

### ASCII-only source files
- ALL source files (`lib/`, `tools/`, `tests/`) must contain only ASCII characters (0x00-0x7F).
- Do NOT use Unicode in comments, strings, or identifiers: use `--` not em-dash, `->` not right-arrow, `...` not ellipsis.
- After editing, verify with:
```bash
LC_ALL=C grep -Prn '[^\x00-\x7F]' lib/ tools/ tests/check/ --include="*.cpp" --include="*.hpp" --include="*.h" --include="*.co" 2>/dev/null | grep -v '\.swp'
```
  This must return no output.

### Diagnostic output: use errs()/dbgs(), never fprintf
- Use `errs()` for all compiler diagnostic messages, never `std::fprintf(stderr, ...)`.
- Use `dbgs()` for debug-only output guarded by `VST_DEBUG(...)`.
- After editing, verify:
```bash
grep -rn 'fprintf' lib/ tools/ --include="*.cpp" --include="*.hpp"
```
  This must return no output.
