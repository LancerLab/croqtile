# Choreo Function Templates

Choreo functions support a deliberately small C++17-style template model for
writing one kernel definition for several scalar types and compile-time
configurations. Only concrete instances enter the normal Choreo semantic and
code-generation pipeline.

## Primary templates

A template declaration precedes `__co__` just as it precedes a C++ function:

```choreo
template <typename T, int Tile = 64, bool UseFastPath = true>
__co__ void scale(T [Tile] data) {
  if constexpr (__is_same<T, f16> && UseFastPath) {
    // f16 fast path
  } else {
    // generic path
  }
}
```

Supported template parameters are:

- `typename T` or `class T` for a Choreo scalar type;
- `int N` for a signed 32-bit compile-time integer;
- `bool B` for a compile-time Boolean.

Each parameter may have a default. As in C++, a parameter without a default
cannot follow one with a default.

## Explicit instantiation

A primary template does not create a kernel by itself. Request each concrete
instance with a Choreo explicit-instantiation declaration:

```choreo
template __co__ scale<f16>;
template __co__ scale<f32, 128, false>;
```

Omitted arguments use defaults. Repeating the same concrete request is
harmless and creates only one instance.

The host program calls the public template name:

```cpp
scale<choreo::f16>(input.view());
scale<float, 128, false>(other.view());
```

The compiler emits an internal ordinary host function for every requested
instance and a C++17 dispatcher under the public name. Calling an argument
combination that was not instantiated produces a host C++ `static_assert`.

## Full specialization

Use `template <>` to replace the primary body for one complete argument list:

```choreo
template <>
__co__ void scale<f32, 128, false>(f32 [128] data) {
  // specialized implementation
}
```

A full-specialization definition is already concrete, so it creates that
instance without a separate explicit-instantiation declaration. If both are
present, the specialization body wins and the instance is still emitted once.

Partial specialization is not supported.

## `if constexpr` and `__is_same`

Template bodies may use braced `if constexpr`, including an
`else if constexpr` chain. Conditions are evaluated after template arguments
and defaults are substituted. Supported condition operations include Boolean
logic, integer arithmetic, integer comparisons, parentheses, and:

```choreo
__is_same<T, f16>
```

`__is_same<A, B>` compares two Choreo scalar types. The inactive branch is
removed before Choreo parsing and semantic analysis, so target- or type-specific
code in that branch does not need to be valid for the selected instance.

The selected body retains a statically true lexical scope in the concrete AST.
Consequently, the existing type inference and value analysis see ordinary
static types and values such as `f16`, `64`, and `true`.

## Instance constraints with `static_assert`

Use `static_assert` in a template body to reject unsupported compile-time
configurations before the concrete function enters the ordinary Choreo
pipeline:

```choreo
template <typename T, int Tile>
__co__ void tiled(T [Tile] data) {
  static_assert(Tile > 0, "Tile must be positive");
  static_assert(Tile % 16 == 0, "Tile must be a multiple of 16");
}
```

The condition supports the same constant-expression subset as
`if constexpr`. The message is optional and, when present, must be a string
literal. Assertions in an inactive `if constexpr` branch are discarded.

A failed assertion reports both the assertion location and the explicit
instantiation that selected it. Errors reported later by Choreo semantic
passes also include the concrete template instance and request location.

## Compilation model

Template expansion happens after ordinary macro preprocessing and before the
Choreo parser:

```text
primary template + concrete requests
  -> argument binding and full-specialization selection
  -> identifier substitution and if-constexpr pruning
  -> ordinary concrete __co__ functions
  -> existing Choreo semantic, CoIR, and target pipelines
```

Template definitions that have no concrete instance never enter those
pipelines. This keeps template support independent of target lowering and
preserves the same static type and static value guarantees as handwritten
non-template kernels.

Build systems can query support without compiling a probe source:

```bash
choreo -t <target> --print-features | grep -qx FUNCTION_TEMPLATES
```

`FUNCTION_TEMPLATES` is a language feature and is reported for every target.

## Current restrictions

The following C++ template features are intentionally outside this model:

- partial specialization, parameter packs, concepts, and SFINAE;
- template overloading under one Choreo function name;
- non-type parameters other than `int` and `bool`;
- automatic discovery of template calls in arbitrary host C++;
- calls from one Choreo function to another Choreo function;
- recursive template instantiation.

Type arguments must be supported Choreo scalar type names. Integer template
arguments are decimal or hexadecimal literals in the signed 32-bit range;
Boolean arguments are `true`, `false`, `1`, or `0`.
