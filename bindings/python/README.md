# QMBED for Python

QMBED is a Rust-native exact-diagonalization toolkit for quantum many-body
workflows. The Python distribution includes:

- the native `qmbed` request API;
- a versioned `quspin` compatibility surface for workflow migration; and
- the compiled QMBED Rust core inside each supported platform wheel.

After the first registry release:

```bash
python -m pip install qmbed
```

```python
import qmbed

basis = qmbed.SpinBasis(2)
terms = (
    qmbed.OperatorSpec(
        qmbed.OpProduct((qmbed.LocalOperator.Z, qmbed.LocalOperator.Z)),
        (qmbed.Coupling(1.0, (0, 1)),),
    ),
)
result = qmbed.eigsh(basis, terms, qmbed.EigshOptions(2))
print(result.eigenvalues)
```

See the
[QMBED documentation](https://matrixlab-research.github.io/QMBED/python/)
for supported bases, operators, solvers, observables, and compatibility
boundaries.
