# Explicit Vectors

## Overview

Croqtile provides an explicit vector DSL for targets that expose vector
registers and vector memory operations. The DSL is built from logical vector
values, ordinary Croqtile expressions, and explicit vector memory accesses.
Target layout and legal widths are checked by the backend.

## Constructors

```choreo
lane = vec.iota<64>;                   // [0, 1, ..., 63]
stepped = vec.iota<64>(start);         // start, start + 1, ...
shifted = vec.iota<64>(start, stride); // start + i * stride
filled = vec.fill<64>(0.0f);           // every lane is 0.0f
```

- `vec.iota<N>` creates the logical lane sequence `[0, N-1]`.
- `vec.index<N>` is an alias for the zero-argument `vec.iota<N>` form.
- `vec.iota<N>(start)` defaults the stride to 1.
- A zero stride creates a splat; `vec.fill<N>(value)` is shorthand for
  `vec.iota<N>(value, 0)`.
- `N` is a positive compile-time integer. `start`, `stride`, and the fill value
  may be mutable scalar values and are evaluated when the vector value is
  created.
- `start` and `stride` must be `s32` scalar values for an affine iota.

## Values and mutability

Vector values are inferred; no public vector type annotation is required.

```choreo
lane = vec.iota<64>;
values = vec.load(buffer.at(lane));
values = values + 1.0f;
mask = values > 0.0f;
selected = mask ? values : 0.0f;
converted = __to<f32>(selected);
```

- Arithmetic and comparisons lift lane-wise.
- A scalar operand broadcasts to all lanes.
- A comparison returns `vector<bool, N>`.
- `&`, `|`, and `!` compose mask values. `&&` and `||` remain scalar
  short-circuit operators.
- `__to<T>(vector_value)` preserves the lane count and changes the element
  type.
- A mutable vector variable can be reassigned, but its lane count and element
  type cannot change.

## Vector memory access

```choreo
values = vec.load(buffer.at(coordinates));
masked = vec.load(buffer.at(coordinates), mask, other);
vec.store(buffer.at(coordinates), values);
vec.store(buffer.at(coordinates), values, mask);
```

- The address is a normal Croqtile element address ending in `.at(...)`.
- At least one coordinate must be a vector.
- Scalar coordinates broadcast; multiple vector coordinates are zipped
  lane-wise and must have the same lane count.
- A rank mismatch must be resolved with an explicit view or flattening.
- The load result element type is the buffer element type. Use an explicit
  `__to<T>` when a different type is required.
- The store value element type must match the buffer element type, unless an
  explicitly permitted conversion applies.
- A masked load requires `other`, the value returned for inactive lanes.
- Inactive lanes do not access memory. Coordinates for inactive lanes may be
  outside the buffer bounds.
- A mask is scalar `bool` or `vector<bool, N>`. Vector masks are not scalar
  control-flow conditions; use vector ternary or masked memory instead.

## Target limits

- Only targets that advertise explicit vector support accept this DSL.
- `N` must have a legal representation for the selected target and element type.
  Targets may accept partial-register values and one, two, or four register
  groups.
- Unsupported widths and element types produce a source diagnostic rather than
  reaching code generation.
- Alignment, masking, and contiguous versus indirect access may affect which
  target operation is selected, but do not change the source semantics.
- Vector parameters, vector returns, lane extraction/insertion, and a public
  vector type annotation are not part of this surface.

## Compatibility and migration

The parenthesized memory form is the canonical spelling. During the
compatibility window, the compiler also accepts the legacy spelling in a
complete assignment or statement position:

```choreo
values = vec.load buffer.at(lane);   // legacy
vec.store buffer.at(lane), values;   // legacy
```

The compatibility mode is selected with:

```text
-fvec-syntax=legacy   # current default; accepts legacy and canonical forms
-fvec-syntax=call     # canonical form only; rejects the legacy spelling
```

The default will move to the canonical form in a future release, after which
legacy mode will be available only through the compatibility flag. The legacy
spelling will be removed in a later release.

Migration tooling:

```bash
python3 scripts/migrate_vec_syntax.py path/to/kernels
python3 scripts/migrate_vec_syntax.py --write path/to/kernels
```

The first command reports changes; the second rewrites files in place.
