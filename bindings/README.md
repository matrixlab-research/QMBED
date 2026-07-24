# QMBED language bindings

Both language packages are thin request builders over `qmbed-capi`, which
links the same Rust core as native applications. The C boundary accepts one
typed JSON schema for built-in bases, `OpProduct` terms, materialization format,
and `eigsh` options. It returns dimensions, convergence evidence, eigenvalues,
residuals, and optionally eigenvectors.

Long-lived frontend objects use the same schema through a persistent-model
protocol: `create_model` returns an opaque decimal handle, model operations
reuse that handle, and `release_model` ends its lifetime. Handles are unique,
safe to use concurrently, and never expose a Rust address. Release is
deterministic when a frontend provides `close()` and is also backed by a
best-effort finalizer. A command which uses a released or unknown handle fails
explicitly. The Rust `PackedEdModel` also caches one assembled operator per
storage format, so repeated solver and export calls reuse both the basis and
the assembled matrix.

The same handle can apply its persistent terms or execute caller-supplied
temporary terms without replacing the model. `apply_model` reuses a cached
matrix-free operator, `materialize_terms_model` returns sparse triplets,
`apply_terms_model` assembles one temporary action for a batch of vectors, and
`bra_ket_terms_model` returns raw local transition tables. Algebraic actions
cover normal, transpose, conjugate, and adjoint forms. These commands are the
language-neutral protocol behind Python `dot`, `Op`, `inplace_Op`, and
`Op_bra_ket`; none materializes a dense matrix.

Cross-basis operations keep both bases alive as explicit handles.
`projector_model` exports the sparse isometry from a reduced model into a
caller-selected parent, while `apply_projector_model` performs batched lift or
projection. `apply_terms_between_models` streams typed terms directly from a
source sector into a target sector. This is the common Rust capability behind
Python `get_proj`, `project_from`, `project_to`, `get_vec`, and
`Op_shift_sector`; `pcon` only chooses which explicit parent model the Python
adapter requests.

`reduce_states_model` exposes the same reduction contract without building an
operator: for each physical state it returns the canonical representative,
orbit size, phase, and normalized coefficient. Projector construction consumes
that contract directly, so Python `representative`, `normalization`, and
`get_amp` cannot drift from `get_proj`.

- `python/` exposes the native `qmbed` module and the versioned
  `qmbed.compat.quspin` migration surface.
- `julia/` exposes only the native `QMBED` API.
- `capi/` owns serialization and the only unsafe pointer boundary.

Site indices are zero based in all three languages. Python compatibility
operator strings are parsed in the adapter and sent as typed local actions;
they do not select a separate Rust assembler. Julia callers construct
`OpProduct` and `OperatorSpec` values directly.

General packed bases use a serializable lattice-symmetry schema rather than
frontend callbacks. Each generator specifies a bijection of source sites,
optional per-site permutations of local states, and a character sector. Rust
validates the map, derives its finite period, and computes fermionic exchange
phases. `GeneralBasis` follows the closure of the generated group rather than a
fixed Cartesian product of generator powers. The same representation therefore
covers translations, reflections, local spin inversion, dihedral combinations,
and higher-dimensional lattice permutations; inconsistent one-dimensional
character requests produce valid empty sectors.

Spin requests also carry an explicit normalization instead of overloading one
boolean: angular-momentum operators, Pauli scaling of every non-identity
symbol, and Cartesian-only Pauli scaling are distinct Rust core conventions.

## Python compatibility contract

The directory `python/compat_tests/quspin-1.0.1/` is a byte-for-byte snapshot
of the 73 official Python tests from QuSpin 1.0.1 at commit
`5bf9e5b266e6d8b70e5cf5973c7c7d59d62e412f`. Its upstream BSD-3-Clause license,
file hashes, and an exhaustive compatibility status are committed beside the
tests.

CI runs every test marked `passing` without modifying its source. Tests whose
required object protocol has not yet been implemented remain explicitly
listed as `unsupported`; they are not silently skipped. The snapshot and
classification can be checked locally with:

```bash
python ci/freeze_upstream_quspin_tests.py --check
```
