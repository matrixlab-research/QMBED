# QMBED

**MATRIX / SIM · Quantum Many-Body Exact Diagonalization**

`qmbed` is a Rust-native exact-diagonalization toolkit derived from real
many-body workflows. Its native API is organized around mathematical
capabilities rather than Python class layout: a basis defines states and local
transitions, and every stored or matrix-free map implements one rectangular
`LinearOperator` interface. QuSpin-derived spellings remain available under
`qmbed::compat::quspin` during migration.

## Implemented capability surface

- Spin, boson, spinless/spinful fermion, arbitrary finite-symmetry, tensor,
  photon, callback-defined, and fixed-width wide-state bases.
- Higher spin, translation/parity sectors, fermionic momentum sectors,
  multi-sector spinful spaces, Majorana operators, branching user actions,
  projectors with leakage checks, and streamed source-to-target sector changes.
- Dense, CSC, CSR, DIA, and matrix-free operators; static, driven, and named
  parameterized Hamiltonians; nonzero-driven sparse algebra; reusable
  exponential grid/right action and low-level matvec plans; and safe versioned
  dense/sparse archives.
- Dense and selected Hermitian eigensolvers, shift-invert, fully
  reorthogonalized Lanczos, static/callable/batched/density evolution,
  FTLM/LTLM, reusable exponential plans, and cached shift-invert plans.
- Floquet systems, block operators, spectral and dynamical response,
  expectation values, arbitrary-site partial traces, pure/mixed entanglement,
  diagonal ensembles, level statistics, subspace tracking, and Lindblad
  generators.

The four fixed-width state types (`U256`, `U1024`, `U4096`, `U16384`) are
available independently of the small-system `u128` path and round-trip through
arbitrary-precision `BigUint` values. Optimized assembly is selected by
mathematical capabilities such as stored transitions or finite orbits, never
by model names. Fixed-particle basis enumeration scales with the requested
sector dimension instead of scanning the full parent Hilbert space.

## Minimal example

```rust
use qmbed::basis::SpinBasis1D;
use qmbed::operator::{
    Coupling, LocalOperator, MatrixFormat, OpProduct, OperatorBuilder, OperatorSpec,
};
use qmbed::solve::{eigsh, EigshOptions};

let basis = SpinBasis1D::builder(12).up(6).momentum(0).build()?;
let bonds = (0..12).map(|site| Coupling::new(1.0, vec![site, (site + 1) % 12]));
let zz = OpProduct::new([LocalOperator::Z, LocalOperator::Z])?;
let hamiltonian = OperatorBuilder::on(&basis)
    .term(OperatorSpec::from_product(zz, bonds)?)
    .build(MatrixFormat::Csc)?;
let low_energy = eigsh(&hamiltonian, EigshOptions::smallest_algebraic(4))?;
# Ok::<(), qmbed::QmbedError>(())
```

QuSpin operator strings remain accepted by `OperatorTerm::new` and by the
explicit functions in `qmbed::compat::quspin`. They are parsed once into the
same `OpProduct` used above; they do not select a second assembler.

Thin [Python and Julia bindings](bindings/README.md) construct the same typed
request through a shared C ABI. Python offers a versioned
`qmbed.compat.quspin` migration surface, tested against an unchanged snapshot
of the official QuSpin tests. Julia deliberately exposes only native QMBED
types and does not carry a second compatibility API.

The same `OperatorBuilder::between(source, target)` path constructs a
rectangular probe between particle-number or symmetry sectors. The same
operator can be converted among stored formats or consumed matrix-free by
Krylov algorithms.

## Runtime and numerical backend boundaries

Physics-facing basis and operator types depend only on the `LinearOperator`
contract. `qmbed::runtime` adds a second narrow waist for owned vectors and
coarse operations such as `apply`, `axpy`, `dotc`, and host transfer. The
built-in `CpuRuntime` is single-rank; an `ExecutionProfile` that requests a GPU
or multiple MPI ranks fails explicitly until an implementation of the same
`Runtime` contract is installed. No model name crosses this boundary.
`ExecutionProfile::throughput(n)` enables bounded shared-memory execution for
independent batches such as `ExpmMultiplyParallel::apply_batch_with_runtime`;
results retain input order. The serial profile uses the same path with one
worker.

Dense eigendecomposition, matrix products, and sparse shifted factorization
remain isolated in an internal numerical backend module. Krylov iterations
call one reusable factorization or dense kernel rather than dispatching inside
element or matvec loops. Real CSC Hamiltonians retain real arithmetic through
both the factorization and shift-invert Lanczos path; complex operators use the
same public solver contract.

## Benchmark and verification boundary

The public suite contains deterministic numerical properties and regressions.
The public
[QMBED Benchmark](https://github.com/matrixlab-research/QMBED-benchmark)
repository adds independent numerical oracles and twelve medium-size workflows
derived from published many-body calculations. These are reported separately:

1. public surface and properties;
2. independent numerical oracles;
3. complete paper-workflow composition;
4. representative sparse and symmetry-reduced scale.

A green API-presence test alone is not treated as parity. The complete design
and acceptance criteria are maintained in QMBED Benchmark's
`rust/full-taskdoc/` documents. The original frozen clean-room task for the
23-symbol workflow core remains available separately:

- [motivation](https://github.com/matrixlab-research/quspin-rust-task/blob/main/MOTIVATION.md)
- [API contract](https://github.com/matrixlab-research/quspin-rust-task/blob/main/CONTRACT.md)
- [visible examples](https://github.com/matrixlab-research/quspin-rust-task/blob/main/TESTS.md)

## Local gates

```bash
cargo fmt --check
cargo test --all-targets
cargo clippy --all-targets --all-features -- -D warnings
cargo test --release --test visible_contract -- --ignored --test-threads=1
```
