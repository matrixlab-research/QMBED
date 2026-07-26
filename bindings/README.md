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
explicitly. The generic Rust `EdModel<B>` caches one assembled operator per
storage format; its packed and fixed-width-wide aliases therefore reuse the
same basis, assembly, solver, and cache implementation.

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

Basis-independent matrices use the same handle lifecycle rather than a second
frontend solver. `create_operator_model` accepts one fixed sparse operator and
named parameterized components; `materialize_model`, `apply_model`,
`eigh_model`, `eigsh_model`, and `evolve_model` accept the same complex
parameter map. Rust's `PackedOperatorModel` therefore covers direct
NumPy/SciPy matrices and named operator families without pretending that an
arbitrary matrix has local-basis transition semantics.

`project_operator_model` applies an arbitrary rectangular map as `P† A P` to
the fixed operator and every named component, returning another persistent
model. Dynamic coefficients, defaults, sparse storage, and solver behavior
therefore survive projection without a Python-side reimplementation.

Time-dependent coefficients use one explicit synchronous callback boundary.
`qmbed_evolve_model_with_drive_json` receives the ordered component names,
initial state columns, physical initial time, and output times. Rust calls the
frontend callback at every internal integration time; the callback writes one
complex coefficient per component in exactly the requested order. The
callback and its context remain borrowed only for the duration of that call,
and a nonzero callback status, missing coefficient, or non-finite coefficient
fails the evolution. Python selects this native path with
`solver_name="qmbed"`; other named solvers are intentionally delegated to
SciPy so that the QuSpin compatibility layer preserves the corresponding
solver's numerical behavior.

Reduction can also precede basis enumeration. `create_basis_plan` compiles a
serializable finite-group action into a `SymmetryReducer` handle;
`reduce_states_plan` returns the canonical representative, compatibility,
orbit size, character phase, physical map phase, and exact generator word.
`materialize_basis_plan` later enumerates the requested parent sector with that
same reducer and returns a normal model handle. A 64-site fixed-particle plan
is tested without attempting to enumerate its parent Hilbert space.

For an already materialized model, `reduce_states_model` exposes the matching
representative, orbit, phase, and normalized coefficient. Projector
construction consumes the same core reduction contract, so Python
`representative`, `normalization`, `get_amp`, and `get_proj` cannot drift.

`analyze_subsystem_model` combines that cached projector with an explicit
tensor-product parent. It accepts pure vectors, density matrices, and batches;
returns both reduced density matrices, spectra, and von Neumann or Renyi
entropies; and preserves the input norm or trace. An explicit fermionic flag
adds the occupation-dependent exchange phase induced by regrouping arbitrary
noncontiguous modes, rather than treating Fock modes as qubits. Local
dimensions and retained sites remain protocol data, so the same command
supports spin, boson, spinless-fermion, and spinful-fermion bases.

Krylov decompositions are persistent resources rather than transferred dense
bases. `lanczos_operator` evaluates any recursive operator expression once and
returns a `lanczos:*` handle plus the small Ritz eigensystem.
`lanczos_combine` and `lanczos_exponential` reconstruct vectors inside Rust;
`export_lanczos_basis` is an explicit opt-in for callers that truly need every
Krylov vector. Release and stale-handle behavior match the model lifecycle.

Floquet construction has two language-neutral entry points.
`analyze_floquet` accepts arbitrary piecewise operator expressions and step
durations, while `analyze_floquet_unitary` accepts a propagator produced by a
continuous-time integrator or another backend. Both use the same Rust analysis
for ordered quasienergies, eigenphases/states, residuals, the period unitary,
and the effective Hamiltonian; an optional physical period can differ from the
sum of explicit kick durations. `floquet_time_grid` supplies the uniform
ramp-up, constant, and ramp-down grid used by the Python compatibility view.

Recursive operator expressions also have a native complete Hermitian
eigensystem command. The Python compatibility layer uses that same expression
and action contract for Schrödinger batches, Liouville-von Neumann density
matrices, arbitrary user right-hand sides, ED reconstruction, and
time-dependent observable/entropy series. User-defined RHS integration remains
in the frontend by design; model-defined coefficient callbacks can still
select the fully native Rust integrator.

Additive quantum-number selections are also boundary data rather than frontend
enumeration. `up_sectors` and `particle_sectors` select validated unions for
spin, boson, spinless-fermion, and spinful-fermion bases. Site-local exclusions
use `allowed_local_occupancies`; Rust's reusable `LocalOccupationConstraint`
filters packed binary-species states before the same symmetry and operator
paths run. Python `double_occupancy=False` is only the two-species shorthand
for allowing local masks `[0, 1, 2]`.

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

CI runs every test marked `passing` without modifying its source. All 73
frozen files are currently classified as passing; no file is silently
skipped. If a future snapshot exposes an unsupported object protocol, it must
be classified explicitly rather than removed from the contract. The snapshot
and classification can be checked locally with:

```bash
python ci/freeze_upstream_quspin_tests.py --check
```

Passing that frozen public suite is not a claim of total implementation
completeness. Scalar symmetry-reduced spin-half models now run end to end
through `U16384`: the serialized request selects a `WidePackedBasis`, while
the persistent model reuses the generic `EdModel` assembly and solver path.
Same-storage wide parent projectors, cross-sector actions, and matrix-valued
symmetry subspaces now reuse the state-agnostic projector, runtime-erased
finite-map reducer, and generic `EdModel` paths. Wide subsystem analysis still
requires a sector-native sparse contraction instead of the current explicitly
enumerated full tensor product. Python also uses exact object-integer arrays
rather than registering QuSpin's custom
`uint256`, `uint1024`, `uint4096`, and `uint16384` NumPy dtype identities.
Other gaps are a versioned portable basis archive, a public constructor for
the core's arbitrary matrix-valued symmetry representations, exact routing of
every SciPy-specific solver-control keyword and legacy execution hint
(`parallel`, `sparse_diag`, `Ns_block_est`, and similar), and exact
reproduction of incidental warning/error text. The core itself supports branching local
transitions; QuSpin's documented higher-spin operators do not add a missing
fixed-row branching contract. `quspin.basis.transformations.square_lattice_trans`
is a thin map generator whose results enter the same general Rust symmetry
reducer. `user_basis(noncommuting_bits=...)` now carries arbitrary
validated unit-modulus exchange phases through the same pure- and
density-state subsystem kernels as the fermionic `-1` convention.
