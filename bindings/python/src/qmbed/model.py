"""Typed Python requests for bases, local operators, and sparse eigensolvers."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Sequence

from ._ffi import run


class LocalOperator(str, Enum):
    """Built-in local actions understood by the universal Rust assembler.

    Members describe one-site identity, number, Cartesian spin, and
    raising/lowering actions. A string may be used in :class:`OpProduct` for a
    custom action supported by a callback-defined basis.
    """

    IDENTITY = "identity"
    NUMBER = "number"
    Z = "z"
    RAISING = "raising"
    LOWERING = "lowering"
    X = "x"
    Y = "y"


@dataclass(frozen=True)
class OpProduct:
    """Ordered product of local actions.

    Args:
        local: Local actions in the same order as the coupling's site tuple.
        split: Boundary between the up and down species of a two-species
            product.
        splits: Boundaries for products with more than two species.

    ``split`` and ``splits`` are mutually exclusive. Site indices themselves
    live in :class:`Coupling` so one product can be reused over many bonds.
    """

    local: tuple[LocalOperator | str, ...]
    split: int | None = None
    splits: tuple[int, ...] = ()

    @classmethod
    def spinful(
        cls,
        up: Iterable[LocalOperator | str],
        down: Iterable[LocalOperator | str],
    ) -> "OpProduct":
        """Construct a two-species product from up- and down-sector actions.

        Args:
            up: Ordered local actions for the up species.
            down: Ordered local actions for the down species.

        Returns:
            A product whose species boundary is placed after ``up``.
        """

        up = tuple(up)
        return cls(up + tuple(down), split=len(up))

    def request(self) -> dict:
        """Serialize the product into the language-neutral C-ABI schema."""

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
    """One complex coefficient and the zero-based sites on which it acts.

    Args:
        coefficient: Scalar multiplying the local operator product.
        sites: Site indices in the same order as the product's local actions.
    """

    coefficient: complex
    sites: tuple[int, ...]

    def request(self) -> dict:
        """Serialize the coupling into the language-neutral C-ABI schema."""

        return {
            "coefficient": [self.coefficient.real, self.coefficient.imag],
            "sites": list(self.sites),
        }


@dataclass(frozen=True)
class OperatorSpec:
    """A reusable local product together with all of its couplings.

    The Rust core validates the product arity against every coupling and
    assembles the resulting term in the requested storage format.
    """

    product: OpProduct
    couplings: tuple[Coupling, ...]

    def request(self) -> dict:
        """Serialize this term into the language-neutral C-ABI schema."""

        return {
            "product": self.product.request(),
            "couplings": [coupling.request() for coupling in self.couplings],
        }


class BasisSpec:
    """Abstract base for immutable basis requests."""

    def request(self) -> dict:
        """Serialize this basis request for the Rust core."""

        raise NotImplementedError


@dataclass(frozen=True)
class SpinBasis(BasisSpec):
    """One-dimensional spin basis with optional conserved and symmetry sectors.

    Args:
        sites: Number of lattice sites.
        spin_twice: Twice the on-site spin quantum number; ``1`` means
            spin one half.
        up: Fixed total raising-quantum count, or ``None`` for the full space.
        momentum: Translation momentum sector, or ``None``.
        parity: Reflection eigenvalue, normally ``-1`` or ``1``.
        pauli: Use Pauli rather than angular-momentum normalization.
    """

    sites: int
    spin_twice: int = 1
    up: int | None = None
    momentum: int | None = None
    parity: int | None = None
    pauli: bool = False

    def request(self) -> dict:
        """Serialize this spin basis request."""

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
    """Bosonic lattice basis with a finite local cutoff.

    Args:
        sites: Number of lattice sites.
        states_per_site: Allowed local occupations ``0`` through
            ``states_per_site - 1``.
        particles: Fixed total boson number, or ``None`` for the full product
            space.
    """

    sites: int
    states_per_site: int
    particles: int | None = None

    def request(self) -> dict:
        """Serialize this boson basis request."""

        return {
            "kind": "boson",
            "sites": self.sites,
            "states_per_site": self.states_per_site,
            "particles": self.particles,
        }


@dataclass(frozen=True)
class SpinlessFermionBasis(BasisSpec):
    """Spinless-fermion Fock basis.

    Args:
        sites: Number of fermionic orbitals.
        particles: Fixed particle number, or ``None``.
        momentum: Translation momentum sector, or ``None``.
    """

    sites: int
    particles: int | None = None
    momentum: int | None = None

    def request(self) -> dict:
        """Serialize this spinless-fermion basis request."""

        return {
            "kind": "spinless_fermion",
            "sites": self.sites,
            "particles": self.particles,
            "momentum": self.momentum,
        }


@dataclass(frozen=True)
class SpinfulFermionBasis(BasisSpec):
    """Two-species fermion basis with independent number sectors.

    Args:
        sites: Number of spatial sites for each species.
        particles_up: Fixed up-species particle number, or ``None``.
        particles_down: Fixed down-species particle number, or ``None``.
    """

    sites: int
    particles_up: int | None = None
    particles_down: int | None = None

    def request(self) -> dict:
        """Serialize this spinful-fermion basis request."""

        return {
            "kind": "spinful_fermion",
            "sites": self.sites,
            "particles_up": self.particles_up,
            "particles_down": self.particles_down,
        }


@dataclass(frozen=True)
class EigshOptions:
    """Controls a selected Hermitian eigensolve.

    Args:
        eigenpairs: Number of requested eigenpairs.
        target: Spectral target such as ``"smallest_algebraic"`` or
            ``"shift"``.
        shift: Interior target used when ``target="shift"``.
        krylov_dimension: Maximum search-space dimension. ``None`` selects the
            core default.
        tolerance: Residual convergence tolerance.
        max_iterations: Maximum restart or iteration budget.
        seed: Deterministic initial-vector seed.
        eigenvectors: Return eigenvectors in addition to values and residuals.
    """

    eigenpairs: int
    target: str = "smallest_algebraic"
    shift: float | None = None
    krylov_dimension: int | None = None
    tolerance: float = 1.0e-10
    max_iterations: int = 1_000
    seed: int = 0
    eigenvectors: bool = False

    def request(self) -> dict:
        """Serialize and normalize the eigensolver options."""

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
    """Result of a QMBED Hermitian eigensolve.

    Attributes:
        dimension: Hilbert-space dimension.
        eigenvalues: Ordered real Ritz or exact eigenvalues.
        residuals: Norm of ``H v - λ v`` for each returned pair.
        iterations: Solver iteration count.
        converged: Whether every requested residual met the tolerance.
        eigenvectors: Eigenvectors when requested, otherwise ``None``.
    """

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
    """Solve selected eigenpairs of a Hamiltonian assembled from local terms.

    Args:
        basis: Typed Hilbert-space request.
        terms: Local operator specifications to sum into the Hamiltonian.
        options: Spectral target and convergence controls.
        format: Materialization format, normally ``"csc"``. Other core
            formats include ``"dense"``, ``"csr"``, ``"dia"``, and
            ``"matrix_free"`` when supported by the selected solver.

    Returns:
        Eigenvalues, residuals, convergence evidence, and optional
        eigenvectors from the Rust solver.

    Raises:
        QmbedError: If the basis, terms, storage route, or solver options are
            invalid, or if the native operation fails.
    """

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
