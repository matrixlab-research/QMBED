# QMBED

**MATRIX / SIM · Quantum Many-Body Exact Diagonalization**

[![Rust core](https://github.com/matrixlab-research/QMBED/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/matrixlab-research/QMBED/actions/workflows/ci.yml)
[![Language bindings](https://github.com/matrixlab-research/QMBED/actions/workflows/bindings.yml/badge.svg?branch=main)](https://github.com/matrixlab-research/QMBED/actions/workflows/bindings.yml)
[![Python wheels](https://github.com/matrixlab-research/QMBED/actions/workflows/wheels.yml/badge.svg?branch=main)](https://github.com/matrixlab-research/QMBED/actions/workflows/wheels.yml)
[![Documentation](https://github.com/matrixlab-research/QMBED/actions/workflows/docs.yml/badge.svg?branch=main)](https://github.com/matrixlab-research/QMBED/actions/workflows/docs.yml)
[![Paper workflows](https://github.com/matrixlab-research/QMBED-benchmark/actions/workflows/rust-contract.yml/badge.svg?branch=main)](https://github.com/matrixlab-research/QMBED-benchmark/actions/workflows/rust-contract.yml)
[![Release](https://img.shields.io/github/v/release/matrixlab-research/QMBED)](https://github.com/matrixlab-research/QMBED/releases)
[![crates.io](https://img.shields.io/crates/v/qmbed)](https://crates.io/crates/qmbed)
[![PyPI](https://img.shields.io/pypi/v/qmbed)](https://pypi.org/project/qmbed/)
[![License](https://img.shields.io/github/license/matrixlab-research/QMBED)](https://github.com/matrixlab-research/QMBED/blob/main/LICENSE)

QMBED is a Rust-native exact-diagonalization toolkit for quantum many-body
workflows. Rust, Python, and Julia all reach the same basis, operator, solver,
and measurement implementation. Optimized routes are selected from
mathematical capabilities—such as sparse storage, conserved sectors, or finite
symmetry orbits—rather than model names.

**[Documentation](https://matrixlab-research.github.io/QMBED/) ·
[Rust API](https://matrixlab-research.github.io/QMBED/rust/api/qmbed/) ·
[Python API](https://matrixlab-research.github.io/QMBED/python/api/) ·
[Julia API](https://matrixlab-research.github.io/QMBED/julia/api/) ·
[Benchmarks](https://github.com/matrixlab-research/QMBED-benchmark)**

## Choose an interface

| Interface | Intended use | API policy |
|---|---|---|
| [Rust](https://matrixlab-research.github.io/QMBED/rust/) | Native applications and reusable simulation components | One canonical typed API |
| [Python](https://matrixlab-research.github.io/QMBED/python/) | Python workflows and QuSpin migration | Native `qmbed` API plus versioned `quspin` compatibility |
| [Julia](https://matrixlab-research.github.io/QMBED/julia/) | Julia-native scientific workflows | Native `QMBED` API only |

All site indices are zero based. Python and Julia bindings are thin request
builders over `qmbed-capi`; they do not reimplement assembly or solvers.

## Rust quick start

Install the current release from crates.io:

```toml
[dependencies]
qmbed = "0.2.0"
```

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

Python 3.10–3.14 wheels are available from PyPI with
`python -m pip install "qmbed==0.2.0"`. Julia source-install instructions and
equivalent examples are in the
[getting-started guide](https://matrixlab-research.github.io/QMBED/getting-started/)
while the initial General-registry submission is pending.

Rust intentionally has no migration namespace or duplicate compatibility
aliases. See the
[Rust API stability policy](https://matrixlab-research.github.io/QMBED/rust/api-stability/)
for the canonical names and extension rules.

## What is implemented

- Spin, boson, spinless/spinful fermion, photon, tensor, callback-defined,
  symmetry-reduced, and fixed-width wide-state bases.
- Dense, CSC, CSR, DIA, and matrix-free operators, including rectangular
  operators between particle or symmetry sectors.
- Dense Hermitian eigensolvers, shift-invert, restarted Lanczos, reusable
  eigensolver workspaces, Krylov evolution, FTLM/LTLM, and exponential-action
  plans.
- Floquet spectra, spectral and dynamical response, expectation values,
  subsystem density matrices, entanglement, diagonal ensembles, state tracking,
  and Lindblad generators.
- Matrix-free selected Floquet quasienergies, sector-native wide-state
  entanglement contractions, and portable ordered basis manifests.
- Reusable parameter-scan operator plans and shared symmetry-orbit caches.
- Rust-native exact JVP/VJP rules for parameterized operator actions and
  gap-aware Hellmann--Feynman ground-state energy gradients; an optional
  `chainrules` feature adapts the same rules to `chainrules-core 0.2`.

The four fixed-width state types (`U256`, `U1024`, `U4096`, and `U16384`) are
independent of the small-system `u128` path. Fixed-particle enumeration scales
with the requested sector instead of scanning the full parent Hilbert space.
See the [capability guide](https://matrixlab-research.github.io/QMBED/capabilities/)
for module-level details and current boundaries.

Native AD is operation based: iterative solvers and sparse assembly are not
traced instruction by instruction. See the
[AD guide](https://matrixlab-research.github.io/QMBED/rust/ad/) for formulas,
diagnostics, examples, and explicit gaps. The benchmark repository separately
checks gradients against finite differences and reports both end-to-end time
and eigensolve counts across real ED workflows.

## Architecture

```text
Rust API ───────────────────────────────┐
                                       │
Python native / QuSpin compatibility ─ C ABI ─ QMBED core ─ LinearOperator
                                       │                    ├─ stored sparse/dense
Julia native API ───────────────────── C ABI                └─ matrix-free
```

The physics-facing narrow waist is `LinearOperator`. A second `Runtime`
boundary owns vectors and coarse operations. The built-in runtime is
single-rank CPU; GPU and MPI profiles fail explicitly until a backend
implementing that same contract is installed. Dense eigendecomposition,
matrix products, and shifted sparse factorization remain isolated behind the
numerical backend. More detail is available in
[Architecture](https://matrixlab-research.github.io/QMBED/architecture/).

## Verification and benchmarks

The repository CI checks formatting, Clippy, Rust 1.85, macOS/Windows
portability, public-API semver compatibility, crates.io packaging, unit and
integration tests, paper-scale visible contracts, the shared C boundary,
Python compatibility, Julia bindings, and all three documentation builds.

Independent verification lives in
[QMBED Benchmark](https://github.com/matrixlab-research/QMBED-benchmark). It
runs twelve medium-size, paper-shaped workflows on the same single-thread
runner, with one warm-up, five measured samples, and workflow-specific
residual, norm, or unitarity checks. The benchmark times basis construction,
Hamiltonian assembly, solver or evolution, and observable evaluation end to
end; it is not a microbenchmark or a claim of reproducing every paper.

See [Verification](https://matrixlab-research.github.io/QMBED/verification/)
for the exact test boundary and local commands.

## Contributing

```bash
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-targets
cargo test --release --test visible_contract -- --ignored --test-threads=1
```

Documentation build commands are documented in
[Contributing to the docs](https://matrixlab-research.github.io/QMBED/contributing/).

## License

QMBED is available under the
[MIT License](https://github.com/matrixlab-research/QMBED/blob/main/LICENSE).
The frozen upstream
QuSpin compatibility tests and compatibility package retain their upstream
BSD-3-Clause notices.
