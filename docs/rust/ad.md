# Rust-native automatic differentiation

QMBED defines derivatives at the scientific-operation boundary instead of
tracing sparse assembly or iterative solver internals. The native rules use the
same `QuantumOperator`, `LinearOperator`, and runtime contracts as ordinary
execution.

## Implemented rules

| Operation | Forward/reverse rule | Reliability evidence |
|---|---|---|
| `A(θ)x` | exact JVP and one-shot VJP | dimensions, domains, action counts |
| isolated ground-state energy | Hellmann--Feynman gradient | residual, spectral gap, solver iterations |
| ChainRules.rs | optional `chainrules-core 0.2` adapter | runs the native JVP/VJP, no duplicate derivative |

Rules for eigenvectors, degenerate eigenspaces, time evolution, thermal traces,
and response functions remain explicit capability gaps. In particular, QMBED
does not assign a derivative to an arbitrary vector in a degenerate eigenspace.

## Parameterized operator action

```rust
use qmbed::ad::{ParameterValues, apply_vjp};
use qmbed::runtime::{CpuRuntime, Runtime};

# fn example(
#   family: &qmbed::operator::QuantumOperator,
# ) -> Result<(), qmbed::QmbedError> {
let runtime = CpuRuntime::new(1)?;
let parameters = ParameterValues::real(family, [1.0, 0.7])?;
let state = runtime.upload(&vec![qmbed::Complex64::new(1.0, 0.0); family.shape().1])?;
let (value, pullback) = apply_vjp(&runtime, family, &parameters, &state)?;
let cotangents = pullback.backward(&value)?;
assert_eq!(cotangents.parameters.values().len(), 2);
# Ok(())
# }
```

Enable the protocol adapter only when needed:

```toml
qmbed = { version = "0.2", features = ["chainrules"] }
```

## Ground-state energy

`ground_state_energy_gradient` requests at least two lowest eigenpairs. This is
not extra decoration: the second eigenvalue supplies the gap needed to judge
whether the scalar eigenvalue derivative is well defined and numerically
resolved. For \(p\) parameters, the analytic rule uses one eigensolve followed
by \(p\) component actions; central finite differences require \(2p\)
additional eigensolves.

Benchmarks report both correctness against finite differences and the separate
solver/action counts. A speedup without agreement or without a resolved gap is
not accepted as AD evidence.
