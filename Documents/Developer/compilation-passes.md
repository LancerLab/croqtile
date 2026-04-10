# Compilation Passes

## Overview

The Croqtile compiler processes tileflow functions through a series of **passes**, each performing a specific transformation or analysis. Understanding these passes helps with debugging and understanding compiler behavior.

## Pass Pipeline

```
Source -> SEMA -> NORM -> VALNO -> INFER -> LATENORM -> CHECK -> CODEGEN -> Target
```

| Pass | Name | Description |
|------|------|-------------|
| `SEMA` | Early Semantic Analysis | Basic validation of syntax and semantic rules |
| `NORM` / `COMP_N` | AST Normalization | Normalizes parallel structures, fills implicit parallel variables |
| `VALNO` | Value Numbering | Assigns value numbers, simplifies shape expressions |
| `INFER` | Type Inference | Infers shapes, types, and storage for all expressions |
| `LATENORM` | Late Normalization | Post-inference normalizations |
| `CHECK` | Semantic Checking | Full semantic validation (shape consistency, storage rules, etc.) |
| `CODEGEN` | Code Generation | Generates target source code |

Additional target-specific passes may run between CHECK and CODEGEN (e.g., liveness analysis, vectorization).

## Debugging with Pass Options

### Dump AST After Parsing

```bash
./choreo -e your_program.co
```

Shows the raw AST structure before any passes run.

### Show Type Inference Results

```bash
./choreo -i your_program.co       # standard inference output
./choreo -ii your_program.co      # detailed inference output
```

Displays the inferred types, shapes, and storage for every symbol.

### Print AST After a Specific Pass

```bash
./choreo -pa=NORM your_program.co
./choreo -pa=INFER your_program.co
./choreo -pa=CHECK your_program.co
```

The `-pa=<PASS>` flag prints the AST state after the specified pass completes.

### Stop After a Specific Pass

```bash
./choreo -sa=INFER your_program.co
./choreo --stop-after=CHECK your_program.co
```

Stops compilation after the specified pass, useful for isolating issues.

### Value Numbering Trace

```bash
./choreo -vn your_program.co
./choreo -vn --simplify-fp-valno your_program.co
```

Shows the value numbering process and simplification results.

## What Each Pass Does

### SEMA (Early Semantic Analysis)

- Validates basic syntax rules.
- Checks that constructs appear in valid contexts (e.g., `call` inside `parallel-by`).
- Reports early errors before expensive analysis.

### NORM (Normalization)

- Fills in implicit parallel variables (anonymous `parallel by N`).
- Normalizes `parallel-by` annotations (`: block`, `: group`, `: thread`).
- Expands syntactic sugar (e.g., `foreach x in N` to `with x in N { foreach x }`).

*(Reference: `tests/norm/pb_norm_group.co`)*

### VALNO (Value Numbering)

- Assigns value numbers to all expressions.
- Performs constant folding and algebraic simplification.
- Builds the foundation for shape inference.

See [Value Numbering](value-numbering.md) for details.

### INFER (Type Inference)

- Infers shapes for all spanned data, futures, and DMA results.
- Resolves symbolic dimensions.
- Determines storage qualifiers for compiler-allocated buffers.
- Computes chunk shapes for `chunkat` expressions.

### CHECK (Semantic Checking)

- Validates shape consistency across DMA source and destination.
- Checks storage qualifier restrictions (e.g., `shared` inside `parallel-by`).
- Verifies bounded variable usage rules.
- Reports illegal operations (undefined variables, type mismatches, etc.).

*(Reference: the entire `tests/check/` directory)*

### CODEGEN (Code Generation)

- Generates target-specific source code.
- Maps `parallel-by` to kernel launches.
- Maps DMA operations to target memory transfer APIs.
- Maps MMA operations to target compute instructions.

## Using Passes for Debugging

A practical debugging workflow:

1. **Shape errors**: Use `-i` to see inferred shapes. Use `-pa=INFER` to see the AST state after inference.
2. **Normalization issues**: Use `-pa=NORM` to verify parallel structure normalization.
3. **Semantic errors**: Use `-sa=CHECK` to stop at the checking pass and see the full error report.
4. **Codegen issues**: Use `-es` to emit the generated source and inspect it directly.
5. **Value numbering**: Use `-vn` to trace shape simplification.

## Lit Test Infrastructure

The test suite uses these pass options extensively with FileCheck for regression testing:

```
// RUN: choreo -e %s | FileCheck %s
// RUN: choreo -i %s | FileCheck --match-full-lines %s
// RUN: choreo -pa=NORM %s | FileCheck %s
```

See the [Lit Test Runner](lit-test-runner.md) documentation for the full testing infrastructure.
