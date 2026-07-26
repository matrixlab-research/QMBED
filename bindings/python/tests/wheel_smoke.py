"""Smoke test for an installed QMBED platform wheel."""

from __future__ import annotations

from pathlib import Path

import qmbed
from qmbed._ffi import _library_path


library = _library_path()
package = Path(qmbed.__file__).resolve().parent
assert library.parent == package, (library, package)

basis = qmbed.SpinBasis(2)
terms = (
    qmbed.OperatorSpec(
        qmbed.OpProduct((qmbed.LocalOperator.Z, qmbed.LocalOperator.Z)),
        (qmbed.Coupling(1.0, (0, 1)),),
    ),
)
result = qmbed.eigsh(basis, terms, qmbed.EigshOptions(2))
assert result.dimension == 4
assert result.converged
assert abs(result.eigenvalues[0] + 0.25) < 1.0e-10
