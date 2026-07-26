# Julia interface

Julia exposes a small native `QMBED` API over the shared Rust core. It does not
mirror Python's QuSpin compatibility classes: new Julia programs construct
typed bases, local products, couplings, and solver options directly.

[:material-api: Open generated Julia reference](api/){ .md-button .md-button--primary }

## Example

```julia
using QMBED

sites = 12
basis = SpinBasis(sites=sites, up=6, momentum=0)
bonds = [
    Coupling(1.0, [site, mod(site + 1, sites)])
    for site in 0:(sites - 1)
]
zz = OperatorSpec(OpProduct([ZOp, ZOp]), bonds)
result = eigsh(basis, [zz], EigshOptions(eigenpairs=4))
@assert result.converged
```

## Conventions

- Sites are zero based so one operator request has identical meaning in Rust,
  Python, Julia, and the C schema.
- `SpinBasis`, `BosonBasis`, `SpinlessFermionBasis`, and
  `SpinfulFermionBasis` are immutable request values.
- `OpProduct` represents local actions; `OperatorSpec` attaches their
  couplings.
- `EigshOptions` makes convergence controls explicit.

The current Julia surface deliberately starts narrow. New capabilities should
be added as native QMBED concepts and routed through the shared protocol,
rather than recreating Python object structure.
