# Release Notes

## Explicit vector DSL syntax update

The explicit vector DSL has a canonical, parenthesized memory syntax:

```choreo
values = vec.load(buffer.at(lane));
vec.store(buffer.at(lane), values);
```

The following additions are part of the same update:

- `vec.iota<N>` as the canonical lane-sequence constructor;
- `vec.index<N>` as a zero-argument alias;
- `vec.iota<N>(start)` with a default stride of 1;
- `vec.iota<N>(start, stride)` for affine lane sequences;
- `vec.fill<N>(value)` as shorthand for a zero-stride splat.

### Compatibility window

- The compiler currently defaults to `-fvec-syntax=legacy`.
- The legacy no-parenthesis form is accepted only as a complete assignment or
  statement form.
- `-fvec-syntax=call` accepts only the canonical parenthesized form.
- The default will change to `call` in a future release.
- Legacy syntax will be removed in a later release after deprecation.
- `scripts/migrate_vec_syntax.py` migrates existing kernels to the canonical
  form.

See [Explicit Vectors](vector-dsl.md) for the complete syntax and semantics.
