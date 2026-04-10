# Assertions

## Overview

Croqtile provides a single `assert` construct that handles both compile-time and runtime checking. The compiler decides at compile time whether a condition can be fully evaluated:

- **Statically evaluable**: If the condition is a constant or reducible to a constant (e.g., comparing two known shape values), it is checked at compile time. A passing assertion generates no code; a failing assertion is a compilation error.
- **Requires runtime**: If the condition depends on dynamic values (e.g., symbolic dimensions), the compiler generates a runtime check.

## Syntax

```choreo
assert(condition, "message");
```

Both arguments are required. The message is a string literal that appears in the error output if the assertion fails.

## Examples

### Statically Resolved

When the condition depends only on known values, the compiler evaluates it during compilation:

```choreo
__co__ void foo(f32 [128, 512] a) {
  parallel p by 1, q by 4 {
    // compiler knows a.span(0)=128 and a.span(1)=512 -- evaluates statically
    assert(a.span(0) < a.span(1), "M must be less than N");
    // passing: generates no code
  }
}
```

A statically false assertion is a compile-time error:

```choreo
assert(a.span(0) > a.span(1), "128 > 512");
// error: choreo assertion abort: 128 > 512
```

### Runtime Assertion (Symbolic Dimensions)

When dimensions are symbolic, the assertion becomes a runtime check:

```choreo
__co__ void foo(f32 [M, K] input) {
  assert(M % 16 == 0, "M must be a multiple of 16");
  assert(K > 0, "K must be positive");
}
```

The compiler generates the checks in the transpiled host code, defaulting to function entry.

## Assertion Hoisting

The compiler **hoists assertions to function entry** when possible. This means shape errors are reported immediately when the tileflow function is called, before any kernel launch or DMA operation, minimizing the cost and time-to-diagnosis.

An assertion can only be hoisted to entry if its condition depends only on function parameters and host-visible values (not values derived from inside `parallel-by` or loop bodies).

## Controlling Runtime Check Level

The `--runtime-check` (`-rtc`) flag controls which assertions are emitted and at what cost threshold:

| Level | Behavior |
|-------|----------|
| `none` | All runtime assertions disabled |
| `entry` | Host-side entry assertions only (default) |
| `low` | Entry + device-side assertions with low cost |
| `medium` | Entry + device-side assertions with medium cost |
| `high` / `all` | All assertions including high-cost device-side checks |

```bash
./choreo -t cute --runtime-check=all program.co    # enable all checks
./choreo -t cute --runtime-check=none program.co   # disable all checks
./choreo -t cute --runtime-check=medium program.co # balanced
./choreo -t cute --disable-runtime-check program.co  # legacy, same as none
```

The default (`entry`) is the right choice for production: catches the most common shape errors at function entry with zero device overhead.

Use `--runtime-check=all` during development and debugging to catch errors as close to their origin as possible.

## Compiler-Generated Assertions

Beyond user-written `assert` calls, the compiler automatically generates assertions for:

- **Zero-dimension guards**: Every derived shape must have positive dimensions.
- **Symbolic dimension consistency**: Dimensions sharing a symbolic name across operands must match.
- **Division validity**: Shape expressions that could produce division by zero.

These follow the same hoisting and cost-level rules as user-written assertions.

*(Reference: `tests/parse/assert.co`, `tests/check/illegal_static_assert.co`)*
