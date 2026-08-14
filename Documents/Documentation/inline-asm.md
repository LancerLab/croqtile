# Inline Assembly

Inline assembly is an escape hatch for emitting target-native instructions
directly inside device code. Croqtile adopts the GNU extended `asm` syntax
familiar from C and C++, so existing knowledge of GCC/Clang inline assembly
carries over unchanged. This is a low-level feature intended for advanced
users who need instructions that have no Croqtile-language equivalent; prefer
the first-class constructs (DMA, MMA, atomics, synchronization) whenever they
express the same operation.

## Syntax

An inline assembly statement has the following general form:

```
__asm__ [volatile] ( "template" [: outputs [: inputs [: clobbers ]]] )
```

Every part after the template string is optional, but the colons must be
present to reach a later part:

```choreo
__asm__("nop");                                // no operands, no clobbers
__asm__("nop" : : : "memory");                 // memory clobber only
__asm__("mov %0, %1" : "=r"(r) : "r"(v));      // output + input operands
```

The keyword `asm` is accepted as an alias for `__asm__`.

## `volatile`

The optional `volatile` qualifier tells the compiler not to move, delete, or
otherwise optimize the statement relative to other code with side effects.
Use it when the instruction has side effects the compiler cannot see, when the
statement must execute in a precise order relative to surrounding code, or
when the same instruction may be executed more than once and each execution
matters. Without `volatile`, the compiler is free to remove or reorder the
statement if it appears to have no observable effect.

## The Template String

The template string is the instruction text passed through to the target
assembler. Instructions themselves are target-specific; the string is emitted
verbatim, with operand placeholders substituted. Croqtile does not inspect the
template text.

Placeholders refer to operands by number or by name:

- **Positional references** use `%0`, `%1`, and so on. `%0` refers to the first
  operand in the output list, and inputs follow the outputs.
- **Named references** use `%[name]`, where `name` is the symbolic name
  declared on the corresponding operand. Named references make templates
  self-documenting and are less error-prone when operands are reordered.

```choreo
__asm__("mov %[dst], %[src]"
        : [dst] "=r"(result)
        : [src] "r"(val));
```

## Operands

Operands connect Croqtile variables to the register or memory slots the
instruction operates on. Each operand is written as:

```
constraint ( expression )
```

or, with a symbolic name:

```
[ name ] constraint ( expression )
```

The constraint is a string of letters describing how the value may be mapped
to a register or memory location. Common constraints include:

| Constraint | Meaning |
|------------|---------|
| `"r"` | Any general-purpose register |
| `"=r"` | Output written to a general-purpose register |
| `"+r"` | Read-write operand in a general-purpose register |
| `"m"` | A memory operand |
| `"=m"` | Output written to memory |

The exact set of accepted constraint letters is defined by the target
toolchain. Croqtile forwards the constraint verbatim and does not validate it
against the expression type.

### Output Operands

An output operand is written by the instruction. It appears in the first
colon-separated list and uses an `=` modifier (write-only) or a `+` modifier
(read-write):

```choreo
s32 result;
parallel p by 1 {
  __asm__("mov %0, %1" : "=r"(result) : "r"(val));
}
```

A `+` operand is both read and written by the instruction and must be
initialized before the statement:

```choreo
s32 x = val;
parallel p by 1 {
  __asm__("mov %0, %0" : "+r"(x));
}
```

### Input Operands

An input operand provides a value to the instruction and appears in the second
list. Input constraints do not carry a modifier:

```choreo
__asm__("mov %0, %1" : "=r"(result) : "r"(val));
```

### Operand Numbering

When positional `%N` references are used, operands are numbered in declaration
order: all output operands first (starting at `%0`), followed by all input
operands. For the example above, `%0` is the output `result` and `%1` is the
input `val`.

## Clobbers

The clobber list declares resources that the instruction modifies but that
are not listed as operands, so the compiler does not cache stale values across
the statement. Common clobbers are:

- `"memory"` -- the instruction reads or writes memory the compiler does not
  otherwise know about.
- `"cc"` -- the instruction modifies condition-code flags.
- Register names -- specific registers clobbered by the instruction.

```choreo
__asm__ volatile("fence" : : : "memory");
```

## Where Inline Assembly May Appear

Inline assembly is a device-code statement. It must appear inside a
`parallel-by` block, in the same position where a `call` statement or an
intrinsic passthrough call is allowed. It is not valid in host (non-parallel)
code.

```choreo
__co__ auto copy(s32 val) {
  s32 result;
  parallel p by 1 {
    __asm__ volatile(
      "mov %[dst], %[src]"
      : [dst] "=r"(result)
      : [src] "r"(val)
      : "memory"
    );
  }
  return result;
}
```

## Restrictions

- **`asm goto` is not supported.** The parser rejects the `goto` form with a
  compile-time error.
- **No template or constraint validation.** The template string, constraint
  letters, and clobber names are forwarded to the target toolchain verbatim.
  Croqtile does not check that they are valid, that the number of placeholders
  matches the number of operands, or that constraints are compatible with the
  expression type. Errors surface during target compilation.
- **Device-code only.** Inline assembly outside a `parallel-by` block is an
  error.
- **Support is target-dependent.** A target that cannot represent inline
  assembly rejects the statement at compile time with an "inline assembly is
  not supported" error.

## Comparison with Intrinsic Passthrough

Inline assembly and [intrinsic passthrough](call-and-device.md#intrinsic-passthrough)
are two escape hatches with different granularity:

- **Intrinsic passthrough** emits a whole call verbatim and is the right
  choice when a target intrinsic already expresses the operation as a
  function call.
- **Inline assembly** gives direct control over the instruction and its
  operands, registers, and clobbers, and is the right choice when no callable
  intrinsic exists or when precise register allocation matters.

*(Reference: `tests/cpu/end2end/asm_basic.co`, `tests/check/asm_edge_cases.co`,
`tests/check/asm_parse_errors.co`, `tests/check/asm_goto_rejected.co`.)*
