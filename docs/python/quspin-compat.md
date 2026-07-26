# QuSpin compatibility

QMBED's Python package provides a versioned compatibility layer for migrating
QuSpin 1.0.1 workflows:

```python
from quspin.basis import spin_basis_1d
from quspin.operators import hamiltonian
from quspin.tools.evolution import evolve
```

## What the contract means

- The compatibility package preserves the public imports exercised by the
  frozen upstream suite.
- Operator strings are parsed into the same typed `OpProduct` representation
  used by native QMBED.
- Persistent Python objects own opaque model handles and release them
  deterministically with `close()` or a best-effort finalizer.
- Sparse actions, projectors, sector changes, Krylov resources, Floquet
  analysis, and subsystem analysis execute in Rust where the shared protocol
  provides the operation.
- SciPy-specific solver choices remain delegated to SciPy when exact SciPy
  numerical behavior is part of the requested interface.

## Deliberate boundaries

Compatibility does not imply identity with every Python implementation detail.
Known boundaries include exact reproduction of custom wide NumPy dtype
identities, every legacy execution hint, incidental warning text, and a few
wide-state subsystem calls whose legacy two-sided dense return protocol has
not yet been routed to Rust's sector-native one-sided contraction.

The machine-readable status and source hashes live under
`bindings/python/compat_tests/quspin-1.0.1/`. CI refuses to silently remove an
upstream test from that contract.

## Validate locally

```bash
cargo build --release --manifest-path bindings/capi/Cargo.toml
python -m pip install -e "bindings/python[test]"
python ci/freeze_upstream_quspin_tests.py --check
python -m unittest discover -s bindings/python/tests -v
```
