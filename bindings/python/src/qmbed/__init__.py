"""Native Python request types for the shared QMBED Rust core.

The classes exported here are immutable descriptions of bases, local operator
products, couplings, and eigensolver options. Numerical work is executed by
the same Rust implementation used by native QMBED applications.
"""

from ._ffi import QmbedError
from .model import (
    BasisSpec,
    BosonBasis,
    Coupling,
    Eigensystem,
    EigshOptions,
    LocalOperator,
    OpProduct,
    OperatorSpec,
    SpinBasis,
    SpinfulFermionBasis,
    SpinlessFermionBasis,
    eigsh,
)
from . import compat

__all__ = [
    "BasisSpec",
    "BosonBasis",
    "Coupling",
    "Eigensystem",
    "EigshOptions",
    "LocalOperator",
    "OpProduct",
    "OperatorSpec",
    "QmbedError",
    "SpinBasis",
    "SpinfulFermionBasis",
    "SpinlessFermionBasis",
    "compat",
    "eigsh",
]
