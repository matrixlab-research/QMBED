# QuSpin 1.0.1 compatibility capability matrix

This matrix is derived from the unchanged 73-file upstream test snapshot. A
row is complete only when the original test executes through the public Python
compatibility package and the exercised behavior is covered by Rust tests.

## Implemented foundation

| Capability | Rust core | Language-neutral boundary | Upstream evidence |
|---|---|---|---|
| Runtime-selected packed basis | `PackedBasis` preserves concrete spin, boson, spinless-fermion, and spinful-fermion semantics | `describe_basis` | `test_version.py`, `test_boson_vs_ho.py`, `test_higher_spin.py` |
| Reusable ED model | `PackedEdModel` owns basis and typed terms, and caches one assembled operator per storage format | thread-safe `create_model`, `describe_model`, `materialize_model`, `eigh_model`, `eigsh_model`, and `release_model` handle protocol | direct cache-identity tests, concurrent C-ABI calls, stale-handle rejection, and Python lifecycle tests |
| Runtime lattice symmetries | `LatticeSymmetryMap` validates site/local-state permutations, derives their period, and computes fermionic exchange phases; `GeneralBasis` closes the generated finite group, including noncommuting generators whose requested phases define a one-dimensional character | built-in basis requests accept serializable symmetry generators | runtime translation equals the optimized spin sector; direct dihedral-group closure tests; unchanged 1D and 2D spin/boson/spinless/spinful decomposition tests |
| Low-level basis actions | temporary terms assemble once on an owned basis and support normal, transpose, conjugate, and adjoint batch actions; persistent terms reuse a cached matrix-free operator; raw bra-ket transitions use the same local semantics | `materialize_terms_model`, `apply_model`, `apply_terms_model`, and `bra_ket_terms_model` on persistent handles | direct Rust/C/Python tests and unchanged `test_pauli.py` plus `test_inplace_op.py` |
| Reduction metadata and explicit parent-space projectors | `ReductionImage` is the common representative/orbit/phase/normalized-amplitude contract; `BasisProjector::between` consumes it directly and supports full or particle-conserving parents plus batched lift/project | `reduce_states_model`, `projector_model`, and `apply_projector_model` operate on persistent handles | direct Rust/C/Python coefficient checks; unchanged `test_project_to.py`, `test_general_spin_get_vec.py`, `test_project_from_boson.py`, and `test_project_from_fermion.py` |
| Cross-sector actions | `PackedEdModel::apply_terms_from_batch` streams typed terms from a source basis into a target basis through the universal reduction path | `apply_terms_between_models` joins source and target handles without materializing a parent space | direct Rust/C/Python tests and unchanged module-level assertions in `test_Op_shift_sector.py` |
| Spin normalization | `SpinNormalization` distinguishes angular-momentum, all-symbol Pauli, and Cartesian-only Pauli conventions | serialized `normalization` basis field | direct amplitude tests for `z`, `x`, and ladder operators; unchanged `test_pauli.py` |
| Explicit basis-vector ordering | `PackedBasis::reversed` permutes `state`, `index`, and transition rows together | `reverse` basis option | all higher-spin `x/y/z/I` products |
| Explicit site convention | `OperatorSpec::with_site_permutation` relabels every coupling through one validated bijection | `site_permutation` command field | all higher-spin multi-site tensor products |
| Python drop-in namespace | Rust-backed `quspin.basis` and `quspin.operators` modules | same C ABI used by native Python and Julia | 15/73 files passing unchanged |

## Remaining foundation gaps

| Priority | Required behavior | Rust/core gap | Boundary or Python gap |
|---|---|---|---|
| P0 | Complete general-basis object protocol | finite generated-group closure and one-dimensional characters now cover commuting generators plus compatible noncommuting combinations; higher-dimensional non-Abelian irreps remain | `representative`, `normalization`, and `get_amp` work for constructed bases; deferred construction, generator-word/sign returns, sector unions, and fermionic particle-hole phases remain |
| P0 | Complete `hamiltonian` object protocol | persistent cached model now exists and operator algebra exists | no NumPy/SciPy input, sparse export, transpose/adjoint/algebra command set |
| P1 | Tensor, photon, user, and wide-state bases | native generic implementations exist with different state types | `PackedBasis` currently erases only `u128` packed bases |
| P1 | Python callables for `user_basis` and dynamic drives | Rust closures are supported | no callback/handle transport across the ABI |
| P1 | Evolution, Floquet, Lanczos, and exponential actions | native implementations exist | absent from the language-neutral command protocol |
| P1 | Entropy, partial trace, observables, and ensembles | native measurement implementations exist | absent from the language-neutral command protocol |
| P2 | QuSpin archive and internal helper fidelity | QMBED archive and generic utilities exist | exact QuSpin ZIP/layout, internal reshape, lattice helper, warning, and error semantics remain |

## Acceptance rule

Moving a file from `unsupported` to `passing` requires all of:

1. the copied upstream file remains byte-for-byte unchanged;
2. its original test entry points execute in CI;
3. new general Rust behavior has direct Rust tests;
4. the implementation does not branch on a model or test name;
5. the compatibility status remains exhaustive across all 73 files.
