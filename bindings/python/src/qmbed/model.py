from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Sequence

from ._ffi import run


class LocalOperator(str, Enum):
    IDENTITY = "identity"
    NUMBER = "number"
    Z = "z"
    RAISING = "raising"
    LOWERING = "lowering"
    X = "x"
    Y = "y"


@dataclass(frozen=True)
class OpProduct:
    local: tuple[LocalOperator | str, ...]
    split: int | None = None
    splits: tuple[int, ...] = ()

    @classmethod
    def spinful(
        cls,
        up: Iterable[LocalOperator | str],
        down: Iterable[LocalOperator | str],
    ) -> "OpProduct":
        up = tuple(up)
        return cls(up + tuple(down), split=len(up))

    def request(self) -> dict:
        local = [
            operator.value if isinstance(operator, LocalOperator) else operator
            for operator in self.local
        ]
        result = {"local": local}
        if self.splits:
            if self.split is not None:
                raise ValueError("OpProduct cannot define both split and splits")
            result["splits"] = list(self.splits)
        elif self.split is not None:
            result["split"] = self.split
        return result


@dataclass(frozen=True)
class Coupling:
    coefficient: complex
    sites: tuple[int, ...]

    def request(self) -> dict:
        return {
            "coefficient": [self.coefficient.real, self.coefficient.imag],
            "sites": list(self.sites),
        }


@dataclass(frozen=True)
class OperatorSpec:
    product: OpProduct
    couplings: tuple[Coupling, ...]

    def request(self) -> dict:
        return {
            "product": self.product.request(),
            "couplings": [coupling.request() for coupling in self.couplings],
        }


class BasisSpec:
    def request(self) -> dict:
        raise NotImplementedError


@dataclass(frozen=True)
class SpinBasis(BasisSpec):
    sites: int
    spin_twice: int = 1
    up: int | None = None
    momentum: int | None = None
    parity: int | None = None
    pauli: bool = False

    def request(self) -> dict:
        return {
            "kind": "spin",
            "sites": self.sites,
            "spin_twice": self.spin_twice,
            "up": self.up,
            "momentum": self.momentum,
            "parity": self.parity,
            "pauli": self.pauli,
        }


@dataclass(frozen=True)
class BosonBasis(BasisSpec):
    sites: int
    states_per_site: int
    particles: int | None = None

    def request(self) -> dict:
        return {
            "kind": "boson",
            "sites": self.sites,
            "states_per_site": self.states_per_site,
            "particles": self.particles,
        }


@dataclass(frozen=True)
class SpinlessFermionBasis(BasisSpec):
    sites: int
    particles: int | None = None
    momentum: int | None = None

    def request(self) -> dict:
        return {
            "kind": "spinless_fermion",
            "sites": self.sites,
            "particles": self.particles,
            "momentum": self.momentum,
        }


@dataclass(frozen=True)
class SpinfulFermionBasis(BasisSpec):
    sites: int
    particles_up: int | None = None
    particles_down: int | None = None

    def request(self) -> dict:
        return {
            "kind": "spinful_fermion",
            "sites": self.sites,
            "particles_up": self.particles_up,
            "particles_down": self.particles_down,
        }


@dataclass(frozen=True)
class EigshOptions:
    eigenpairs: int
    target: str = "smallest_algebraic"
    shift: float | None = None
    krylov_dimension: int | None = None
    tolerance: float = 1.0e-10
    max_iterations: int = 1_000
    seed: int = 0
    eigenvectors: bool = False

    def request(self) -> dict:
        target = (
            {"kind": "shift", "value": self.shift}
            if self.target == "shift"
            else {"kind": self.target}
        )
        return {
            "eigenpairs": self.eigenpairs,
            "target": target,
            "krylov_dimension": self.krylov_dimension,
            "tolerance": self.tolerance,
            "max_iterations": self.max_iterations,
            "seed": self.seed,
            "eigenvectors": self.eigenvectors,
        }


@dataclass(frozen=True)
class Eigensystem:
    dimension: int
    eigenvalues: tuple[float, ...]
    residuals: tuple[float, ...]
    iterations: int
    converged: bool
    eigenvectors: tuple[tuple[complex, ...], ...] | None = None


def eigsh(
    basis: BasisSpec,
    terms: Sequence[OperatorSpec],
    options: EigshOptions,
    *,
    format: str = "csc",
) -> Eigensystem:
    result = run(
        {
            "basis": basis.request(),
            "terms": [term.request() for term in terms],
            "format": format,
            "solver": options.request(),
        }
    )
    vectors = result.get("eigenvectors")
    return Eigensystem(
        dimension=result["dimension"],
        eigenvalues=tuple(result["eigenvalues"]),
        residuals=tuple(result["residuals"]),
        iterations=result["iterations"],
        converged=result["converged"],
        eigenvectors=None
        if vectors is None
        else tuple(
            tuple(complex(real, imag) for real, imag in vector)
            for vector in vectors
        ),
    )
