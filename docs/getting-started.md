# Get started

QMBED 0.2 is published on crates.io and PyPI. The Julia package currently uses
a source checkout while its initial General-registry submission is pending.
All three interfaces execute the same Rust implementation.

## Prerequisites

- Rust 1.85 or newer
- Python 3.10 or newer for the Python package
- Julia 1.10 or newer for the Julia package

## Rust

Add the crates.io release to `Cargo.toml`:

```toml
[dependencies]
qmbed = "0.2.0"
```

The [Rust guide](rust/index.md) gives a complete low-energy spectrum example.

## Build the shared language boundary

Python and Julia both call `qmbed-capi`, which links the same Rust crate:

```bash
git clone https://github.com/matrixlab-research/QMBED.git
cd QMBED
cargo build --release --manifest-path bindings/capi/Cargo.toml
```

The source-tree layouts used below find this library automatically. Set
`QMBED_LIBRARY_PATH` only when the library is moved elsewhere.

## Python

```bash
python -m pip install "qmbed==0.2.0"
```

For an editable source checkout:

```bash
python -m pip install -e "bindings/python[test]"
python -c "import qmbed; print(qmbed.__all__)"
```

Use `qmbed` for new native code:

```python
from qmbed import (
    Coupling,
    EigshOptions,
    LocalOperator,
    OpProduct,
    OperatorSpec,
    SpinBasis,
    eigsh,
)

sites = 12
basis = SpinBasis(sites=sites, up=6, momentum=0)
bonds = tuple(
    Coupling(1.0, (site, (site + 1) % sites))
    for site in range(sites)
)
zz = OperatorSpec(
    OpProduct((LocalOperator.Z, LocalOperator.Z)),
    bonds,
)
result = eigsh(basis, [zz], EigshOptions(eigenpairs=4))
print(result.eigenvalues)
```

Existing QuSpin programs can instead import the compatibility package:

```python
from quspin.basis import spin_basis_1d
from quspin.operators import hamiltonian
```

Read [Python](python/index.md) before choosing between the native and
compatibility APIs.

## Julia

Instantiate the local package:

```bash
julia --project=bindings/julia -e 'using Pkg; Pkg.instantiate()'
```

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
println(result.eigenvalues)
```

QMBED site indices are zero based in Julia as well as Rust and Python. See the
[Julia guide](julia/index.md) for the API policy and generated reference.
