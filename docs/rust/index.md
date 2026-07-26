# Rust interface

Rust exposes the complete QMBED core with typed bases and operators. Stored and
matrix-free maps implement the same `LinearOperator` contract, so solvers and
measurements do not depend on one sparse matrix type.

[:material-api: Open generated rustdoc](api/qmbed/){ .md-button .md-button--primary }

## Low-energy spectrum

```rust
use qmbed::basis::SpinBasis1D;
use qmbed::operator::{
    Coupling, LocalOperator, MatrixFormat, OpProduct, OperatorBuilder, OperatorSpec,
};
use qmbed::solve::{eigsh, EigshOptions};

let sites = 12;
let basis = SpinBasis1D::builder(sites).up(6).momentum(0).build()?;
let bonds = (0..sites)
    .map(|site| Coupling::new(1.0, vec![site, (site + 1) % sites]));
let zz = OpProduct::new([LocalOperator::Z, LocalOperator::Z])?;
let hamiltonian = OperatorBuilder::on(&basis)
    .term(OperatorSpec::from_product(zz, bonds)?)
    .build(MatrixFormat::Csc)?;
let result = eigsh(&hamiltonian, EigshOptions::smallest_algebraic(4))?;
assert!(result.converged);
# Ok::<(), qmbed::QmbedError>(())
```

## Module map

| Module | Responsibility |
|---|---|
| `basis` | State spaces, sectors, symmetries, projectors, wide states |
| `operator` | Typed local products, assembly, storage, matrix-free actions |
| `solve` | Dense/selected spectra, shift-invert, Lanczos, exponential action |
| `dynamics` | Floquet systems, spectral functions, correlators |
| `measure` | Expectations, partial traces, entanglement, ensembles |
| `runtime` | Vector ownership and coarse execution primitives |
| `interop` | Reusable owned models for language frontends |

New code should use these native modules. `qmbed::compat::quspin` retains older
Rust spellings during migration, but does not select another implementation.

## Reuse for scans

For related Hamiltonians, use `QuantumOperatorPlan` to retain one sparse
pattern and `EigshWorkspace` to carry the converged subspace between solves.
These are generic operator and solver facilities, not model-specific paths.
