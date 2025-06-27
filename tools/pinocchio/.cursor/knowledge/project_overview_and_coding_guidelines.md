# Project Overview and Coding Guidelines

## System Architecture

This project is structured as a modular compiler or DSL processing system, with clear separation between parsing, AST construction, semantic analysis, type/shape inference, optimization, and code generation. The architecture is designed for extensibility, maintainability, and support for advanced language features such as parallelism and symbolic computation.

### High-Level Components

- **Front-End (Parser & AST)**: Parses source code into an Abstract Syntax Tree (AST) using Bison/Flex and C++ classes.
- **Semantic Analysis & Inference**: Performs type and shape inference, semantic checks, and static analysis on the AST.
- **Symbolic Expression & Value Numbering**: Assigns unique numbers to expressions for equivalence, simplification, and optimization.
- **Intermediate Optimization**: Conducts liveness analysis, memory reuse, and normalization.
- **Back-End Code Generation**: Traverses the AST to generate code for various targets (CUDA, Factor, Topscc, etc.).
- **Main Control & Entry**: Orchestrates the compilation pipeline and handles command-line interaction.
- **Testing & Samples**: Provides comprehensive test coverage and usage examples.

### Component Dependency Graph

```
[parser.yy]
   |
   v
[ast.hpp] <------> [ast.cpp]
   |                  |
   |                  v
   |             [visitor.hpp]
   |
   v
[typeinfer.cpp] [shapeinfer.cpp] [earlysema.cpp] [semacheck.cpp]
   |
   v
[symbexpr.hpp] [valno.cpp]
   |
   v
[liveness_analysis.cpp] [mem_reuse.cpp] [normalize.hpp]
   |
   v
[codegen_cuda.cpp] [codegen_factor.cpp] [codegen_topscc.cpp]
   |
   v
[command_line.cpp] [choreo_main.cpp]
```

## Module Responsibilities

- **parser.yy**: Defines grammar and parsing rules, constructs AST nodes.
- **ast.hpp/ast.cpp**: Declares and implements all AST node types and their visitor interfaces.
- **visitor.hpp**: Declares the Visitor pattern for AST traversal.
- **typeinfer.cpp/shapeinfer.cpp**: Implements type and shape inference logic.
- **earlysema.cpp/semacheck.cpp**: Performs semantic analysis and static checks.
- **symbexpr.hpp/valno.cpp**: Handles symbolic expressions and value numbering for optimization and equivalence.
- **liveness_analysis.cpp/mem_reuse.cpp/normalize.hpp**: Conducts intermediate optimizations.
- **codegen_*.cpp**: Generates target-specific code from the AST.
- **command_line.cpp/choreo_main.cpp**: Main entry and orchestration.
- **tests/**: Test cases for all stages.

## Main Execution Flow

1. **Entry**: Main function parses command-line arguments and initializes the pipeline.
2. **Parsing**: Source code is parsed into an AST.
3. **Semantic Analysis & Inference**: AST is checked for semantic correctness, and types/shapes are inferred.
4. **Optimization**: Intermediate optimizations are performed.
5. **Code Generation**: Target code is generated from the AST.
6. **Testing**: Automated tests validate each stage.

---

## Code Style Guidelines

### Naming Conventions
- **Types/Classes**: PascalCase (e.g., `ValueNumbering`, `ShapeInference`)
- **Variables/Functions**: camelCase (e.g., `getValueNumber`, `tryToSimplifyBinary`)
- **Constants/Macros**: ALL_CAPS or `k` prefix (e.g., `UNKNOWN_VALUE`, `kMaxSize`)
- **Namespaces**: Lowercase (e.g., `namespace choreo`)
- **Member Variables**: Often prefixed with `m_` or suffixed with `_` (not strictly enforced)

### File Organization
- **Header/Source Separation**: `.hpp` for declarations, `.cpp` for implementations.
- **One Major Class/Module per File**: Each file focuses on a single responsibility.
- **Directory Structure**: Logical separation by function (e.g., `lib/`, `tests/`, `utils/`, `samples/`).

### Function Documentation Practices
- **Header Comments**: Major classes and functions are documented with purpose and usage.
- **Inline Comments**: Used for complex logic, edge cases, and non-obvious code paths.
- **Doxygen-style**: Not strictly enforced, but some functions use Doxygen-like comments.

### Error Handling Patterns
- **Assertions**: Extensive use of `assert()` for invariants and preconditions.
- **Custom Error Functions**: Use of `choreo_unreachable()` and `Error()` for fatal errors and diagnostics.
- **Graceful Degradation**: Some modules return `nullptr` or `std::optional` for recoverable errors.
- **Verbose Diagnostics**: Error messages often include context, location, and suggestions.

### Common Design Patterns
- **Visitor Pattern**: Central to AST traversal and processing.
- **Factory Functions**: For AST node creation (e.g., `AST::Make<T>()`).
- **Smart Pointers**: Extensive use of `std::shared_ptr` and custom `ptr<T>` for memory safety.
- **Modularization**: Each stage (parsing, analysis, codegen) is decoupled and extensible.

---

_This document is auto-generated to provide a concise yet comprehensive overview of the system architecture and coding standards for maintainers and contributors._
