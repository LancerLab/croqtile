# Value Numbering

## Purpose

Croqtile's value numbering (VALNO) process is the foundation of **shape inference**. It assigns symbolic value numbers to every expression in the program, enabling the compiler to:

1. Determine shapes at compile time when possible.
2. Generate shape computation code when runtime evaluation is needed.
3. Simplify shape expressions through constant folding and algebraic simplification.

Unlike traditional value numbering (which targets redundant computation elimination), Croqtile's VALNO focuses on **simplifying mdspan expressions** -- the domain-specific shapes that drive all data movement decisions.

## How It Works

### Expressions to Value Numbers

The process traverses the AST in a syntax-directed manner, generating value numbers for each expression:

```choreo
mdspan a : [1, b, c + 3];
```

Produces:

```
VN #0:  const_1
VN #1:  b
VN #2:  c
VN #3:  const_3
VN #4:  +:#2:#3
VN #5:  #0,#1,#4    (shape of 'a')
```

Each value number represents either a constant, a symbolic variable, or a composition of other value numbers. Indexing operations (e.g., `a(0)`) resolve through this table to return `#0` (constant 1).

### Signatures and Lookup

For each expression, the process:

1. Generates a **signature** (currently a string representation).
2. Looks up the signature in the value number table.
3. If found, returns the existing value number (sharing).
4. If not found, creates a new value number and inserts it.

Symbol definitions associate the scoped symbol name with the expression's value number. Symbol references look up the scoped name.

### Simplification

Every time a signature is generated, the compiler attempts algebraic simplification:

```choreo
int a = 3 + 8;
```

Without simplification:
```
VN #0: const_3,  VN #1: const_8,  VN #2: +:#0:#1
```

After constant folding:
```
VN #0: const_3,  VN #1: const_8,  VN #2: const_11
```

The result `a` is known at compile time to be 11.

### Value Numbers Back to Expressions

After simplification, value numbers that remain symbolic (not fully constant) must be converted back into executable expressions for the generated code. This is the **regeneration** step -- it produces the minimal expression needed to compute each shape at runtime.

## Role in Shape Inference

The mdspan type determines:
- Memory sizes for buffer allocation.
- Chunk shapes for `chunkat` operations.
- DMA transfer configurations.

When shapes are fully constant, the compiler can:
- Allocate memory statically.
- Compute DMA configurations at compile time.
- Verify shape compatibility at compile time.

When shapes are symbolic (dynamic dimensions), the compiler generates host-side code to evaluate shapes at function entry.

## Scope Rules

Value numbering operates **globally** within a tileflow function but respects **lexical scopes**. Identical names in different scopes use their scoped name for signature generation, preventing collisions.

## Limitations

Because VALNO operates on the AST without data-flow analysis (no basic blocks, no SSA), it has two key restrictions:

1. **No bounded values in mdspan dimensions**: An mdspan cannot have dimensions derived from bounded integer types or bounded ituples (since their values vary across iterations/threads).

2. **No redefinition of integers and ituples**: Since there is no mutation tracking, redefining an `int` or `ituple` variable would break the value number table's assumptions. (This is enforced by Croqtile's immutability rules.)

## Inspecting Value Numbers

Use the `-vn` / `--print-valno` flag to see the value numbering output:

```bash
./choreo -vn your_program.co
```

The `--simplify-fp-valno` flag enables additional floating-point constant folding.

## Reference

Briggs P, Cooper K D, Simpson L T. *Value numbering*. Software: Practice and Experience, 1997, 27(6): 701-724.
