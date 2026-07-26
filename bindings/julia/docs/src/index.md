# QMBED.jl

```@docs
QMBED
```

`QMBED` is the native Julia interface to the shared Rust exact-diagonalization
core. It uses typed request values and deliberately does not reproduce
Python's QuSpin object hierarchy.

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
```

All sites are zero based. Build `bindings/capi` before the first numerical
call; importing the module and building this reference do not load the shared
library.
