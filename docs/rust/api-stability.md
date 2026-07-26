# Rust API stability

QMBED has one canonical Rust interface. Migration aliases are deliberately not
part of that interface: a scientific concept has one public name, and every
implementation route reaches the same mathematical kernel.

## Canonical vocabulary

| Concept | Public name |
|---|---|
| Coupled local product | `OperatorSpec` |
| Recoverable error | `QmbedError` |
| Fixed-width states | `U256`, `U1024`, `U4096`, `U16384` |
| Finite-group reduction | `SymmetryReducer` |
| Krylov time controls | `EvolutionOptions` |

Compact labels such as `"zz"` and the typed `OpProduct` representation are two
constructors for the same `OperatorSpec`; they are not separate execution
paths. Likewise, `eigsh` and `eigsh_with_workspace` share one selected-spectrum
implementation, with the latter retaining reusable state between related
solves.

## Evolution rules

- Extensible option structs are constructed with `new` or focused convenience
  constructors and refined with `with_*` methods.
- Extensible public enums and errors are non-exhaustive, so adding a backend,
  storage format, or target does not force a breaking downstream match.
- `basis`, `operator`, `solve`, `dynamics`, `measure`, `workflow`, and `archive`
  are the native scientific API.
- `interop` owns runtime-selected models for language frontends. It may add
  lifecycle conveniences but must not duplicate scientific kernels.
- Python's versioned QuSpin-compatible namespace is a language adapter, not a
  second Rust API. Julia exposes only its native QMBED interface.

Before QMBED 1.0, genuinely necessary breaking changes use a minor version
increment. Compatibility aliases are not accumulated across those releases.
The benchmark repository pins a candidate commit and exercises paper-shaped
workflows before a release is tagged.
