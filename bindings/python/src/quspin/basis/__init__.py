from __future__ import annotations

from copy import deepcopy
from fractions import Fraction
from functools import cached_property
import math
from typing import Any

import numpy as np
import scipy.sparse as sp

from qmbed._ffi import NativeBasisPlan, NativeModel, command
from qmbed.compat.quspin import operator_term
from qmbed.model import Coupling


def _reject_options(family: str, options: dict[str, Any]) -> None:
    unsupported = {
        name: value
        for name, value in options.items()
        if value is not None and name != "a"
    }
    if options.get("a", 1) != 1:
        unsupported["a"] = options["a"]
    if unsupported:
        names = ", ".join(sorted(unsupported))
        raise NotImplementedError(f"{family} does not support these blocks yet: {names}")


def _spin_twice(spin: str | int | float) -> int:
    value = Fraction(str(spin))
    doubled = value * 2
    if doubled.denominator != 1 or doubled <= 0:
        raise ValueError(f"invalid spin quantum number {spin!r}")
    return int(doubled)


def _spin_normalization(pauli: bool | int, spin_twice: int) -> str:
    if spin_twice != 1:
        return "angular_momentum"
    value = int(pauli)
    if value not in (-1, 0, 1):
        raise ValueError("pauli must be one of -1, 0, or 1")
    return {
        0: "angular_momentum",
        1: "pauli",
        -1: "pauli_cartesian",
    }[value]


def _single_species_sectors(
    value,
    name: str,
    *,
    negative_from: int | None = None,
) -> tuple[int | None, list[int] | None]:
    def normalize(sector) -> int:
        sector = int(sector)
        if sector < 0 and negative_from is not None:
            sector += int(negative_from)
        return sector

    if value is None:
        return None, None
    if isinstance(value, (int, np.integer)):
        return normalize(value), None
    try:
        sectors = [normalize(sector) for sector in value]
    except TypeError as error:
        raise TypeError(f"{name} must be an integer or an iterable of integers") from error
    if not sectors:
        raise ValueError(f"{name} sector union must be nonempty")
    return None, sectors


def _spinful_sectors(
    value,
) -> tuple[tuple[int, int] | None, list[list[int]] | None]:
    if value is None:
        return None, None
    if (
        isinstance(value, (tuple, list))
        and len(value) == 2
        and all(isinstance(count, (int, np.integer)) for count in value)
    ):
        return (int(value[0]), int(value[1])), None
    try:
        sectors = [[int(up), int(down)] for up, down in value]
    except (TypeError, ValueError) as error:
        raise TypeError(
            "Nf must be a pair or an iterable of (N_up, N_down) pairs"
        ) from error
    if not sectors:
        raise ValueError("Nf sector union must be nonempty")
    return None, sectors


def _rust_site_map(site_map, sites: int) -> tuple[list[int], list[bool]]:
    values = [int(value) for value in np.asarray(site_map).reshape(-1)]
    if len(values) != sites:
        raise ValueError(f"symmetry map has {len(values)} sites, expected {sites}")
    destinations = [0] * sites
    inverted = [False] * sites
    for python_source, encoded_destination in enumerate(values):
        is_inverted = encoded_destination < 0
        python_destination = (
            -encoded_destination - 1 if is_inverted else encoded_destination
        )
        if not 0 <= python_destination < sites:
            raise ValueError("symmetry map contains an out-of-range site")
        rust_source = sites - python_source - 1
        destinations[rust_source] = sites - python_destination - 1
        inverted[rust_source] = is_inverted
    if len(set(destinations)) != sites:
        raise ValueError("symmetry site map must be bijective")
    return destinations, inverted


def _symmetry_request(
    site_map,
    sector: int,
    *,
    sites: int,
    states_per_site: int,
    fermionic: bool = False,
) -> dict[str, Any]:
    destinations, inverted = _rust_site_map(site_map, sites)
    request: dict[str, Any] = {
        "destinations": destinations,
        "sector": int(sector),
    }
    if any(inverted):
        identity = list(range(states_per_site))
        reversed_digits = list(reversed(identity))
        request["local_permutations"] = [
            reversed_digits if flip else identity for flip in inverted
        ]
    return request


def _general_symmetries(
    blocks: dict[str, Any],
    *,
    sites: int,
    states_per_site: int,
    fermionic: bool = False,
) -> list[dict[str, Any]]:
    symmetries = []
    for name, block in blocks.items():
        if block is None:
            continue
        if not isinstance(block, (tuple, list)) or len(block) != 2:
            raise ValueError(f"{name} must be a (site_map, sector) pair")
        site_map, sector = block
        symmetries.append(
            _symmetry_request(
                site_map,
                sector,
                sites=sites,
                states_per_site=states_per_site,
                fermionic=fermionic,
            )
        )
    return symmetries


def _symmetry_period(request: dict[str, Any], states_per_site: int) -> int:
    destinations = [int(value) for value in request["destinations"]]
    identity = list(range(states_per_site))
    local_permutations = request.get("local_permutations")
    if local_permutations is None:
        local_permutations = [identity for _ in destinations]
    visited = [False] * (len(destinations) * states_per_site)
    period = 1
    for seed in range(len(visited)):
        if visited[seed]:
            continue
        current = seed
        cycle = 0
        while not visited[current]:
            visited[current] = True
            cycle += 1
            source, digit = divmod(current, states_per_site)
            current = (
                int(destinations[source]) * states_per_site
                + int(local_permutations[source][digit])
            )
        if current != seed:
            raise ValueError("symmetry map does not decompose into closed cycles")
        period = math.lcm(period, cycle)
    return period


def _one_dimensional_symmetries(
    sites: int,
    *,
    states_per_site: int,
    momentum: int | None,
    parity: int | None,
    fermionic: bool = False,
    translation_step: int = 1,
) -> list[dict[str, Any]]:
    blocks: dict[str, Any] = {}
    if momentum is not None:
        translation = (np.arange(sites) + translation_step) % sites
        blocks["translation"] = (translation, int(momentum))
    if parity is not None:
        if parity not in (-1, 1):
            raise ValueError("pblock must be either -1 or +1")
        reflection = np.arange(sites)[::-1]
        blocks["parity"] = (reflection, 0 if parity == 1 else 1)
    return _general_symmetries(
        blocks,
        sites=sites,
        states_per_site=states_per_site,
        fermionic=fermionic,
    )


def _matrix_payload(matrix: np.ndarray) -> list[list[list[float]]]:
    values = np.asarray(matrix, dtype=np.complex128)
    return [
        [[complex(value).real, complex(value).imag] for value in row]
        for row in values
    ]


def _generic_momentum_reflection_representation(
    symmetries: list[dict[str, Any]],
    *,
    momentum: int | None,
    reflection_indices: list[int],
    states_per_site: int,
) -> dict[str, Any] | None:
    """Promote noncommuting translation/reflection blocks to a 2D irrep."""

    if momentum is None or not reflection_indices:
        return None
    translation_index = 0
    period = _symmetry_period(symmetries[translation_index], states_per_site)
    normalized_momentum = int(momentum) % period
    if (2 * normalized_momentum) % period == 0:
        return None

    primary = symmetries[reflection_indices[0]]
    selected_row = 0 if int(primary["sector"]) % 2 == 0 else 1
    selected_reflection = 1.0 if selected_row == 0 else -1.0
    angle = 2.0 * np.pi * normalized_momentum / period
    cosine, sine = np.cos(angle), np.sin(angle)
    rotation = np.asarray(
        [[cosine, -sine], [sine, cosine]],
        dtype=np.complex128,
    )
    reflection = np.asarray([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
    identity = np.eye(2, dtype=np.complex128)
    reflection_indices = set(reflection_indices)
    generators = []
    for index, request in enumerate(symmetries):
        if index == translation_index:
            matrix = rotation
        elif index in reflection_indices:
            requested = 1.0 if int(request["sector"]) % 2 == 0 else -1.0
            matrix = (requested / selected_reflection) * reflection
        else:
            generator_period = _symmetry_period(request, states_per_site)
            phase = np.exp(
                2j
                * np.pi
                * (int(request["sector"]) % generator_period)
                / generator_period
            )
            matrix = phase * identity
        generators.append(
            {
                "destinations": list(request["destinations"]),
                "local_permutations": request.get("local_permutations"),
                "matrix": _matrix_payload(matrix),
            }
        )
    return {
        "dimension": 2,
        "selected_row": selected_row,
        "generators": generators,
    }


class _PackedBasis:
    _request: dict[str, Any]
    N: int
    _projector_embedding = False

    @cached_property
    def _model(self) -> NativeModel:
        if hasattr(self, "_deferred_basis") and not self._made_basis:
            raise AttributeError(
                "basis has not been constructed; call basis.make() first"
            )
        if hasattr(self, "_basis_plan"):
            if not self._made_basis:
                raise AttributeError(
                    "basis has not been constructed; call basis.make() first"
                )
            return self._materialized_model
        return NativeModel(
            {
                "basis": self._request,
                "terms": [],
                "site_permutation": self._site_permutation,
                "checks": {
                    "hermiticity": False,
                    "particle_conservation": False,
                    "symmetry_compatibility": False,
                },
            }
        )

    def _initialize_general_basis(self, *, make_basis: bool) -> None:
        self._made_basis = False
        uses_wide_state = (
            self._request.get("kind") == "spin"
            and int(self._request.get("spin_twice", 1)) == 1
            and int(self._request["sites"]) > 128
        )
        if self._request.get("symmetries") and not uses_wide_state:
            self._basis_plan = NativeBasisPlan(
                {
                    "basis": self._request,
                    "site_permutation": self._site_permutation,
                    "checks": {
                        "hermiticity": False,
                        "particle_conservation": False,
                        "symmetry_compatibility": False,
                    },
                }
            )
        else:
            self._deferred_basis = True
        if make_basis:
            self.make()

    def make(self, Ns_block_est=None, N_p=None):
        del Ns_block_est, N_p
        plan = getattr(self, "_basis_plan", None)
        if self._made_basis:
            return None
        if plan is not None:
            self._materialized_model = plan.materialize()
        self._made_basis = True
        self.__dict__.pop("_model", None)
        self.__dict__.pop("_description", None)
        return None

    @cached_property
    def _description(self) -> dict[str, Any]:
        if hasattr(self, "_deferred_basis") and not self._made_basis:
            raise AttributeError(
                "basis has not been constructed; call basis.make() first"
            )
        return command(
            {
                "operation": "describe_basis",
                "basis": self._request,
            }
        )

    @property
    def Ns(self) -> int:
        return int(self._description["dimension"])

    @property
    def _site_count(self) -> int:
        if self._request["kind"] == "spinful_fermion":
            return int(self._request["sites"])
        return int(self.N)

    @property
    def L(self) -> int:
        return self._site_count

    def _compat_state_encoding(self, state: int) -> int:
        value = int(state)
        if self._request["kind"] != "spinful_fermion":
            return value
        sites = self._site_count
        mask = (1 << sites) - 1
        return ((value & mask) << sites) | (value >> sites)

    @property
    def states(self) -> np.ndarray:
        return np.asarray(
            [
                self._compat_state_encoding(int(state))
                for state in self._description["states"]
            ],
            dtype=self.dtype,
        )

    @property
    def _basis(self) -> np.ndarray:
        return self.states

    @property
    def dtype(self) -> np.dtype:
        if self._request["kind"] == "spin":
            states_per_site = int(self._request.get("spin_twice", 1)) + 1
            encoded_sites = self._site_count
        elif self._request["kind"] == "boson":
            states_per_site = int(self._request["states_per_site"])
            encoded_sites = self._site_count
        elif self._request["kind"] == "spinful_fermion":
            states_per_site = 2
            encoded_sites = 2 * self._site_count
        else:
            states_per_site = 2
            encoded_sites = self._site_count
        maximum = states_per_site**encoded_sites - 1
        if maximum <= np.iinfo(np.uint32).max:
            return np.dtype(np.uint32)
        if maximum <= np.iinfo(np.uint64).max:
            return np.dtype(np.uint64)
        return np.dtype(object)

    @property
    def blocks(self) -> dict[str, Any]:
        return {
            name: value
            for name, value in self._request.items()
            if name in {"momentum", "parity", "symmetries"} and value not in (None, [])
        }

    def __len__(self) -> int:
        return self.Ns

    def __getitem__(self, index):
        return self.states[index]

    def int_to_state(self, state, bracket_notation: bool = True):
        if self._request["kind"] == "spin":
            base = int(self._request.get("spin_twice", 1)) + 1
            sites = self._site_count
        elif self._request["kind"] == "boson":
            base = int(self._request["states_per_site"])
            sites = self._site_count
        else:
            base = 2
            sites = (
                2 * self._site_count
                if self._request["kind"] == "spinful_fermion"
                else self._site_count
            )
        value = int(state)
        digits = []
        for _ in range(sites):
            digits.append(value % base)
            value //= base
        if value:
            raise ValueError("state exceeds the basis encoding width")
        body = " ".join(str(digit) for digit in reversed(digits))
        return f"|{body}>" if bracket_notation else body

    def state_to_int(self, state):
        if not isinstance(state, str):
            return int(state)
        text = (
            state.replace("|", " ")
            .replace(">", " ")
            .replace("<", " ")
            .replace(",", " ")
        )
        digits = [int(value) for value in text.split()]
        if self._request["kind"] == "spin":
            base = int(self._request.get("spin_twice", 1)) + 1
            sites = self._site_count
        elif self._request["kind"] == "boson":
            base = int(self._request["states_per_site"])
            sites = self._site_count
        else:
            base = 2
            sites = (
                2 * self._site_count
                if self._request["kind"] == "spinful_fermion"
                else self._site_count
            )
        if len(digits) != sites or any(digit < 0 or digit >= base for digit in digits):
            raise ValueError("state string does not match the basis encoding")
        value = 0
        for digit in digits:
            value = base * value + digit
        return value

    def index(self, state):
        value = self.state_to_int(state)
        matches = np.flatnonzero(self.states == value)
        if matches.size == 0:
            raise ValueError("state must be a representative state in the basis")
        return int(matches[0])

    @property
    def sps(self):
        if hasattr(self, "_sps_override"):
            return self._sps_override
        if self._request["kind"] == "spin":
            return int(self._request.get("spin_twice", 1)) + 1
        if self._request["kind"] == "boson":
            return int(self._request["states_per_site"])
        if self._request["kind"] == "spinful_fermion":
            return 4
        if self._request["kind"] == "tensor":
            raise NotImplementedError(
                "tensor_basis has no single local number of states per site"
            )
        return 2

    @sps.setter
    def sps(self, value):
        self._sps_override = int(value)

    @property
    def operators(self):
        kind = self._request["kind"]
        if kind in {"spin", "spinless_fermion", "spinful_fermion"}:
            return "I, +, -, n, z, x, y"
        if kind in {"boson", "photon"}:
            return "I, +, -, n, z"
        return "basis-defined local operators"

    @property
    def description(self):
        return f"{type(self).__name__}(N={self.N}, Ns={self.Ns}, sps={self.sps})"

    @property
    def noncommuting_bits(self):
        return tuple(getattr(self, "_noncommuting_bits", ()))

    def _check_operator_lists(self, static, dynamic, **checks):
        from quspin.operators import hamiltonian

        operator = hamiltonian(
            static,
            dynamic,
            basis=self,
            dtype=np.complex128,
            check_herm=checks.get("hermitian", False),
            check_pcon=checks.get("particle", False),
            check_symm=checks.get("symmetry", False),
        )
        operator.close()

    def check_hermitian(self, static, dynamic):
        self._check_operator_lists(static, dynamic, hermitian=True)

    def check_symm(self, static, dynamic):
        self._check_operator_lists(static, dynamic, symmetry=True)

    def check_pcon(self, static, dynamic):
        self._check_operator_lists(static, dynamic, particle=True)

    def make_basis_blocks(self, N_p=None):
        del N_p
        self.make()

    @property
    def _site_permutation(self) -> list[int]:
        if (
            self._request["kind"] == "spinful_fermion"
            and getattr(self, "_unified_orbitals", False)
        ):
            return list(range(self._site_count - 1, -1, -1)) + list(
                range(2 * self._site_count - 1, self._site_count - 1, -1)
            )
        return list(range(self._site_count - 1, -1, -1))

    def _new_empty_model(self, request: dict[str, Any]) -> NativeModel:
        return NativeModel(
            {
                "basis": request,
                "terms": [],
                "site_permutation": self._site_permutation,
                "checks": {
                    "hermiticity": False,
                    "particle_conservation": False,
                    "symmetry_compatibility": False,
                },
            }
        )

    def _parent_request(self, *, pcon: bool) -> dict[str, Any]:
        request = deepcopy(self._request)
        request["symmetries"] = []
        request["matrix_symmetry"] = None
        if request["kind"] == "spin":
            request["momentum"] = None
            request["parity"] = None
            if not pcon:
                request["up"] = None
                request["up_sectors"] = None
        elif request["kind"] == "boson":
            if not pcon:
                request["particles"] = None
                request["particle_sectors"] = None
        elif request["kind"] == "spinless_fermion":
            request["momentum"] = None
            if not pcon:
                request["particles"] = None
                request["particle_sectors"] = None
        elif request["kind"] == "spinful_fermion" and not pcon:
            request["particles_up"] = None
            request["particles_down"] = None
            request["particle_sectors"] = None
            request["allowed_local_occupancies"] = None
        return request

    @cached_property
    def _full_parent_model(self) -> NativeModel:
        return self._new_empty_model(self._parent_request(pcon=False))

    @cached_property
    def _particle_parent_model(self) -> NativeModel:
        return self._new_empty_model(self._parent_request(pcon=True))

    def _parent_model(self, *, pcon: bool) -> NativeModel:
        return self._particle_parent_model if pcon else self._full_parent_model

    def _reduction_entries(
        self, states
    ) -> tuple[np.ndarray, list[dict[str, Any]], int]:
        array = np.asarray(states, dtype=self.dtype, order="C")
        array = np.atleast_1d(array)
        if array.ndim != 1:
            raise TypeError("dimension of array_like states must not exceed 1.")
        encoded = [
            str(self._compat_state_encoding(int(state)))
            for state in array
        ]
        plan = getattr(self, "_basis_plan", None)
        if plan is None:
            result = self._model.execute("reduce_states_model", states=encoded)
            period_product = self._symmetry_period_product()
        else:
            result = plan.execute("reduce_states_plan", states=encoded)
            period_product = int(result["period_product"])
        return array, list(result["entries"]), period_product

    def representative(
        self,
        states,
        out=None,
        return_g: bool = False,
        return_sign: bool = False,
    ):
        array, entries, _period_product = self._reduction_entries(states)
        representatives = np.asarray(
            [
                self._compat_state_encoding(
                    int(entry.get("representative", entry["state"]))
                )
                for entry in entries
            ],
            dtype=self.dtype,
        )
        if out is not None:
            if not isinstance(out, np.ndarray):
                raise TypeError("out must be a numpy.ndarray")
            if out.shape != array.shape or out.dtype != self.dtype:
                raise TypeError("out must have the same shape and dtype as states")
            if not out.flags["CARRAY"]:
                raise ValueError("out must be a writable C-contiguous array")
            out[...] = representatives

        extras = []
        if return_g:
            generators = len(self._request.get("symmetries", []))
            generator_counts = np.zeros((array.size, generators), dtype=np.int32)
            for row, entry in enumerate(entries):
                for generator in entry.get("generator_word", []):
                    generator_counts[row, int(generator)] += 1
            extras.append(generator_counts)
        if return_sign:
            signs = np.ones(array.shape, dtype=np.int8)
            for row, entry in enumerate(entries):
                phase = complex(*entry.get("physical_phase_to_representative", [1.0, 0.0]))
                if abs(phase.imag) > 1.0e-10 or abs(abs(phase.real) - 1.0) > 1.0e-10:
                    raise ValueError(
                        "representative sign requires a real fermionic map phase"
                    )
                signs[row] = 1 if phase.real >= 0.0 else -1
            extras.append(signs)

        if out is not None:
            if not extras:
                return None
            return extras[0] if len(extras) == 1 else tuple(extras)
        if not extras:
            return representatives
        return (representatives, *extras)

    def _symmetry_period_product(self) -> int:
        if self._request["kind"] == "spin":
            states_per_site = int(self._request.get("spin_twice", 1)) + 1
        elif self._request["kind"] == "boson":
            states_per_site = int(self._request["states_per_site"])
        else:
            states_per_site = 2
        return math.prod(
            _symmetry_period(request, states_per_site)
            for request in self._request.get("symmetries", [])
        )

    @property
    def _pers(self) -> np.ndarray:
        if self._request["kind"] == "spin":
            states_per_site = int(self._request.get("spin_twice", 1)) + 1
        elif self._request["kind"] == "boson":
            states_per_site = int(self._request["states_per_site"])
        else:
            states_per_site = 2
        return np.asarray(
            [
                _symmetry_period(request, states_per_site)
                for request in self._request.get("symmetries", [])
            ],
            dtype=np.int64,
        )

    @property
    def _n(self) -> np.ndarray:
        period_product = int(self._pers.prod(dtype=np.int64))
        values = np.atleast_1d(
            np.asarray(self.normalization(self.states), dtype=np.uint64)
        )
        if period_product <= 0:
            raise ValueError("symmetry period product must be positive")
        return values // np.uint64(period_product)

    def _get_norms(self, dtype) -> np.ndarray:
        target = np.dtype(dtype)
        squared = self._n.astype(target, copy=False) * int(
            self._pers.prod(dtype=np.int64)
        )
        return np.sqrt(squared, dtype=target)

    def normalization(self, states, out=None):
        array, entries, period_product = self._reduction_entries(states)
        values = np.asarray(
            [
                0
                if not entry.get("compatible", "orbit_size" in entry)
                else period_product * period_product // int(entry["orbit_size"])
                for entry in entries
            ],
            dtype=np.uint64,
        )
        if out is None:
            maximum = int(values.max(initial=0))
            return values.astype(np.min_scalar_type(maximum)).squeeze()
        if out.shape != array.shape:
            raise TypeError("states and out must have same shape.")
        if not np.issubdtype(out.dtype, np.unsignedinteger):
            raise TypeError("out must have an unsigned integer datatype")
        if not out.flags["CARRAY"]:
            raise ValueError("out must be C-contiguous array.")
        out[...] = values
        return None

    def get_amp(self, states, out=None, amps=None, mode: str = "representative"):
        if mode not in {"representative", "full_basis"}:
            raise ValueError(
                "mode accepts only the values 'representative' and 'full_basis'."
            )
        array, entries, _period_product = self._reduction_entries(states)
        if out is None:
            output_dtype = np.complex128 if amps is None else np.asarray(amps).dtype
            out = np.zeros(array.shape, dtype=output_dtype)
        else:
            if out.shape != array.shape:
                raise TypeError("states and out must have same shape.")
            if out.dtype not in (
                np.dtype(np.float32),
                np.dtype(np.float64),
                np.dtype(np.complex64),
                np.dtype(np.complex128),
            ):
                raise TypeError("out must have a real or complex floating datatype")
            if not out.flags["CARRAY"]:
                raise ValueError("out must be C-contiguous array.")
        factors = np.asarray(
            [
                0.0
                if not entry.get("compatible", "amplitude" in entry)
                else complex(*entry["amplitude"])
                for entry in entries
            ],
            dtype=np.complex128,
        )
        out[...] = self._values_for_dtype(factors, out.dtype)
        if amps is not None:
            if np.shape(amps) != array.shape:
                raise TypeError("states and amps must have same shape.")
            if mode == "representative":
                amps *= out
            else:
                np.divide(amps, out, out=amps, where=out != 0)
        if mode == "full_basis" and isinstance(states, np.ndarray):
            states[...] = np.asarray(
                [
                    self._compat_state_encoding(
                        int(entry.get("representative", entry["state"]))
                    )
                    for entry in entries
                ],
                dtype=states.dtype,
            )
        return out.squeeze()

    @staticmethod
    def _complex_vectors(array: np.ndarray) -> list[list[list[float]]]:
        columns = int(np.prod(array.shape[1:], dtype=np.intp)) if array.ndim > 1 else 1
        matrix = array.reshape((array.shape[0], columns))
        return [
            [[complex(value).real, complex(value).imag] for value in matrix[:, column]]
            for column in range(matrix.shape[1])
        ]

    @staticmethod
    def _vectors_from_result(result: dict[str, Any]) -> np.ndarray:
        return np.column_stack(
            [
                np.asarray([complex(*value) for value in vector])
                for vector in result["vectors"]
            ]
        )

    def get_proj(self, dtype, pcon: bool = False):
        parent = self._parent_model(pcon=bool(pcon))
        result = self._model.execute(
            "projector_model",
            parent_handle=parent.handle,
            embedding=bool(self._projector_embedding),
        )
        entries = result["entries"]
        values = self._values_for_dtype(
            [complex(*entry["value"]) for entry in entries],
            dtype,
        )
        rows = np.asarray([entry["row"] for entry in entries], dtype=np.intp)
        columns = np.asarray([entry["column"] for entry in entries], dtype=np.intp)
        return sp.csc_matrix(
            (values, (rows, columns)),
            shape=tuple(result["shape"]),
            dtype=np.dtype(dtype),
        )

    def _apply_projector(self, vectors, *, pcon: bool, action: str) -> np.ndarray:
        array = np.asanyarray(vectors)
        expected = self.Ns if action == "lift" else self._parent_model(pcon=pcon).dimension
        if array.ndim == 0 or array.shape[0] != expected:
            raise ValueError("dimension mismatch")
        result_dtype = np.result_type(array.dtype, np.complex128)
        array = array.astype(result_dtype, order="C", copy=False)
        parent = self._parent_model(pcon=pcon)
        result = self._model.execute(
            "apply_projector_model",
            parent_handle=parent.handle,
            embedding=bool(self._projector_embedding),
            vectors=self._complex_vectors(array),
            action=action,
        )
        output = self._vectors_from_result(result)
        output = self._values_for_dtype(output, result_dtype)
        shape = (int(result["dimension"]), *array.shape[1:])
        return output.reshape(shape)

    def project_from(self, v0, sparse: bool = True, pcon: bool = False):
        output = self._apply_projector(v0, pcon=bool(pcon), action="lift")
        return sp.csc_matrix(output.reshape((output.shape[0], -1))) if sparse else output

    def get_vec(self, v0, sparse: bool = True, pcon: bool = False):
        return self.project_from(v0, sparse=sparse, pcon=pcon)

    def project_to(self, v0, sparse: bool = True, pcon: bool = False):
        output = self._apply_projector(v0, pcon=bool(pcon), action="project")
        return sp.csc_matrix(output.reshape((output.shape[0], -1))) if sparse else output

    def _subsystem_layout(
        self,
        sub_sys_A,
        *,
        subsys_ordering: bool,
    ) -> tuple[list[int], list[int], int, int]:
        kind = self._request["kind"]
        if kind == "spin":
            local_dimensions = [
                int(self._request.get("spin_twice", 1)) + 1
            ] * self._site_count
        elif kind == "boson":
            local_dimensions = [int(self._request["states_per_site"])] * self._site_count
        elif kind == "user":
            local_dimensions = [int(self.sps)] * self._site_count
        elif kind == "spinful_fermion":
            local_dimensions = [2] * (2 * self._site_count)
        else:
            local_dimensions = [2] * self._site_count

        if kind == "spinful_fermion":
            if sub_sys_A is None:
                half = tuple(range(self._site_count // 2))
                selected = [half, half]
            elif (
                isinstance(sub_sys_A, (tuple, list))
                and len(sub_sys_A) == 2
                and all(not isinstance(sites, (int, np.integer)) for sites in sub_sys_A)
            ):
                selected = [
                    tuple(int(site) for site in sub_sys_A[0]),
                    tuple(int(site) for site in sub_sys_A[1]),
                ]
            else:
                flat = tuple(int(site) for site in sub_sys_A)
                selected = [
                    tuple(site for site in flat if site < self._site_count),
                    tuple(
                        site - self._site_count
                        for site in flat
                        if site >= self._site_count
                    ),
                ]
            if subsys_ordering:
                selected = [tuple(sorted(sites)) for sites in selected]
            for sites in selected:
                if len(set(sites)) != len(sites) or any(
                    site < 0 or site >= self._site_count for site in sites
                ):
                    raise ValueError("sub_sys_A contains invalid or repeated sites")

            # The Rust mixed-radix kernel numbers the first retained site as
            # the least-significant local digit. Reverse each QuSpin site list
            # so the resulting subsystem index uses QuSpin's site convention.
            retained_sites = [
                self._site_count - 1 - site for site in reversed(selected[0])
            ] + [
                2 * self._site_count - 1 - site
                for site in reversed(selected[1])
            ]
            retained_count = len(selected[0]) + len(selected[1])
        else:
            if sub_sys_A is None:
                selected_sites = tuple(range(self._site_count // 2))
            else:
                selected_sites = tuple(int(site) for site in sub_sys_A)
            if subsys_ordering:
                selected_sites = tuple(sorted(selected_sites))
            if len(set(selected_sites)) != len(selected_sites) or any(
                site < 0 or site >= self._site_count for site in selected_sites
            ):
                raise ValueError("sub_sys_A contains invalid or repeated sites")
            retained_sites = [
                self._site_count - 1 - site for site in reversed(selected_sites)
            ]
            retained_count = len(selected_sites)

        environment_count = len(local_dimensions) - retained_count
        return (
            local_dimensions,
            retained_sites,
            retained_count,
            environment_count,
        )

    @staticmethod
    def _subsystem_samples(state, *, enforce_pure: bool):
        is_sparse = sp.issparse(state)
        array = state.toarray() if is_sparse else np.asanyarray(state)
        if array.ndim == 0 or array.shape[0] == 0:
            raise ValueError("state must have a nonempty leading Hilbert-space axis")

        samples: list[dict[str, Any]] = []
        if array.ndim == 1:
            samples.append({"kind": "pure", "values": array})
        elif array.ndim == 2:
            if array.shape[0] == array.shape[1] and not enforce_pure:
                samples.append({"kind": "density", "values": array.reshape(-1)})
            else:
                samples.extend(
                    {"kind": "pure", "values": array[:, column]}
                    for column in range(array.shape[1])
                )
        elif (
            array.ndim == 3
            and array.shape[0] == array.shape[1]
            and not enforce_pure
        ):
            samples.extend(
                {"kind": "density", "values": array[:, :, sample].reshape(-1)}
                for sample in range(array.shape[2])
            )
        else:
            raise ValueError(
                "state must be a vector, a column batch, a density matrix, "
                "or a density-matrix batch"
            )
        return samples, is_sparse

    @staticmethod
    def _complex_payload(values) -> list[list[float]]:
        return [
            [complex(value).real, complex(value).imag]
            for value in np.asarray(values).reshape(-1)
        ]

    def _analyze_subsystem(
        self,
        state,
        sub_sys_A=None,
        *,
        alpha: float = 1.0,
        enforce_pure: bool = False,
        subsys_ordering: bool = True,
    ) -> tuple[dict[str, Any], bool, int, int]:
        (
            local_dimensions,
            retained_sites,
            retained_count,
            environment_count,
        ) = self._subsystem_layout(
            sub_sys_A,
            subsys_ordering=bool(subsys_ordering),
        )
        samples, is_sparse = self._subsystem_samples(
            state,
            enforce_pure=bool(enforce_pure),
        )
        if any(
            np.asarray(sample["values"]).reshape(-1).size
            not in {self.Ns, self.Ns * self.Ns}
            for sample in samples
        ):
            raise ValueError("state shape does not match the basis dimension")
        payload = [
            {
                "kind": sample["kind"],
                "values": self._complex_payload(sample["values"]),
            }
            for sample in samples
        ]
        alpha = float(alpha)
        renyi_order = None if abs(alpha - 1.0) <= 1.0e-12 else alpha
        result = self._model.execute(
            "analyze_subsystem_model",
            parent_handle=self._full_parent_model.handle,
            embedding=bool(self._projector_embedding),
            local_dimensions=local_dimensions,
            retained_sites=retained_sites,
            fermionic=self._request["kind"]
            in {"spinless_fermion", "spinful_fermion"},
            noncommuting_groups=[
                {
                    "sites": [int(site) for site in sites],
                    "phase": [complex(phase).real, complex(phase).imag],
                }
                for sites, phase in getattr(self, "_noncommuting_bits", ())
            ],
            samples=payload,
            renyi_order=renyi_order,
        )
        return result, is_sparse, retained_count, environment_count

    @staticmethod
    def _density_batch(result: dict[str, Any], name: str, dimension: int) -> np.ndarray:
        matrices = np.stack(
            [
                np.asarray(
                    [complex(*value) for value in sample[name]],
                    dtype=np.complex128,
                ).reshape((dimension, dimension))[::-1, ::-1]
                for sample in result["samples"]
            ],
            axis=0,
        )
        if name == "density_b":
            # QuSpin's historical reshape convention reports the complement
            # density matrix in the conjugate orientation. Keep the Rust
            # subsystem result canonical and translate only at this adapter.
            matrices = matrices.conj()
        return matrices[0] if len(matrices) == 1 else matrices

    @staticmethod
    def _sparse_density_batch(matrices: np.ndarray):
        if matrices.ndim == 2:
            return sp.csr_matrix(matrices)
        output = np.empty(matrices.shape[0], dtype=object)
        for index, matrix in enumerate(matrices):
            output[index] = sp.csr_matrix(matrix)
        return output

    def partial_trace(
        self,
        state,
        sub_sys_A=None,
        return_rdm: str = "A",
        enforce_pure: bool = False,
        subsys_ordering: bool = True,
        **_options,
    ):
        if return_rdm not in {"A", "B", "both"}:
            raise ValueError("return_rdm must be 'A', 'B', or 'both'")
        result, is_sparse, _retained_count, _environment_count = (
            self._analyze_subsystem(
                state,
                sub_sys_A,
                enforce_pure=enforce_pure,
                subsys_ordering=subsys_ordering,
            )
        )
        density_a = self._density_batch(
            result,
            "density_a",
            int(result["subsystem_dimension"]),
        )
        density_b = self._density_batch(
            result,
            "density_b",
            int(result["environment_dimension"]),
        )
        if is_sparse:
            density_a = self._sparse_density_batch(density_a)
            density_b = self._sparse_density_batch(density_b)
        if return_rdm == "A":
            return density_a
        if return_rdm == "B":
            return density_b
        return density_a, density_b

    def ent_entropy(
        self,
        state,
        sub_sys_A=None,
        density: bool = True,
        alpha: float = 1.0,
        return_rdm: str | None = None,
        return_rdm_EVs: bool = False,
        enforce_pure: bool = False,
        sparse: bool = False,
        sparse_diag: bool = True,
        subsys_ordering: bool = True,
        **_options,
    ) -> dict[str, Any]:
        del sparse_diag
        if return_rdm not in {None, "A", "B", "both"}:
            raise ValueError("return_rdm must be None, 'A', 'B', or 'both'")
        result, input_sparse, retained_count, environment_count = (
            self._analyze_subsystem(
                state,
                sub_sys_A,
                alpha=alpha,
                enforce_pure=enforce_pure,
                subsys_ordering=subsys_ordering,
            )
        )
        entropy_a = np.asarray(
            [sample["entropy_a"] for sample in result["samples"]],
            dtype=np.float64,
        )
        entropy_b = np.asarray(
            [sample["entropy_b"] for sample in result["samples"]],
            dtype=np.float64,
        )
        if density:
            if retained_count:
                entropy_a /= retained_count
            if environment_count:
                entropy_b /= environment_count
        output: dict[str, Any] = {
            "Sent_A": entropy_a[0] if entropy_a.size == 1 else entropy_a,
        }
        if return_rdm in {"B", "both"}:
            output["Sent_B"] = entropy_b[0] if entropy_b.size == 1 else entropy_b

        if return_rdm_EVs:
            spectrum_a = np.stack(
                [
                    np.asarray(sample["spectrum_a"], dtype=np.float64)[::-1]
                    for sample in result["samples"]
                ],
                axis=0,
            )
            output["p_A"] = spectrum_a[0] if len(spectrum_a) == 1 else spectrum_a

        if return_rdm in {"A", "both"}:
            density_a = self._density_batch(
                result,
                "density_a",
                int(result["subsystem_dimension"]),
            )
            output["rdm_A"] = (
                self._sparse_density_batch(density_a)
                if input_sparse or sparse
                else density_a
            )
        if return_rdm in {"B", "both"}:
            density_b = self._density_batch(
                result,
                "density_b",
                int(result["environment_dimension"]),
            )
            output["rdm_B"] = (
                self._sparse_density_batch(density_b)
                if input_sparse or sparse
                else density_b
            )
        return output

    def _p_pure(
        self,
        state,
        sub_sys_A,
        return_rdm: str | None = None,
        **options,
    ):
        result = self.ent_entropy(
            state,
            sub_sys_A,
            density=False,
            return_rdm=return_rdm,
            return_rdm_EVs=True,
            enforce_pure=True,
            subsys_ordering=options.get("subsys_ordering", False),
        )
        probabilities = np.asarray(result["p_A"])
        if probabilities.ndim == 2:
            probabilities = probabilities.T
        return probabilities.squeeze(), result.get("rdm_A"), result.get("rdm_B")

    def _p_pure_sparse(
        self,
        state,
        sub_sys_A,
        return_rdm: str | None = None,
        **options,
    ):
        return self._p_pure(
            state,
            sub_sys_A,
            return_rdm=return_rdm,
            **options,
        )

    def Op_shift_sector(
        self,
        other_basis,
        op_list,
        v_in,
        v_out=None,
        dtype=None,
    ):
        if not isinstance(other_basis, _PackedBasis):
            raise TypeError("other_basis must be a QMBED-backed basis")
        input_array = np.asanyarray(v_in)
        if input_array.ndim == 0 or input_array.shape[0] != other_basis.Ns:
            raise ValueError("dimension mismatch")
        result_dtype = np.dtype(
            dtype
            if dtype is not None
            else np.result_type(input_array.dtype, np.complex128)
        )
        input_array = input_array.astype(result_dtype, order="C", copy=False)
        terms = [
            self._term_request(opstr, indx, coefficient)
            for opstr, indx, coefficient in op_list
        ]
        result = command(
            {
                "operation": "apply_terms_between_models",
                "source_handle": other_basis._model.handle,
                "target_handle": self._model.handle,
                "terms": terms,
                "vectors": self._complex_vectors(input_array),
            }
        )
        output = self._vectors_from_result(result)
        output = self._values_for_dtype(output, result_dtype)
        output = output.reshape((self.Ns, *input_array.shape[1:]))
        if v_out is None:
            return output
        destination = np.asanyarray(v_out)
        if destination.shape != output.shape:
            raise ValueError("v_out has incompatible dimensions with target basis")
        destination[...] = output
        return v_out

    def expanded_form(self, static=[], dynamic=[]):
        return static, dynamic

    @staticmethod
    def _term_request(opstr, indx, coefficient) -> dict[str, Any]:
        term = operator_term(
            str(opstr),
            [Coupling(complex(coefficient), tuple(int(site) for site in indx))],
        )
        return term.request()

    @staticmethod
    def _values_for_dtype(values, dtype) -> np.ndarray:
        target = np.dtype(dtype)
        values = np.asarray(values, dtype=np.complex128)
        if target.kind != "c":
            tolerance = 10 * np.finfo(np.float64).eps
            if np.any(np.abs(values.imag) > tolerance):
                raise TypeError("complex matrix elements cannot be represented by a real dtype")
            values = values.real
        return np.asarray(values, dtype=target)

    def Op(self, opstr, indx, J, dtype):
        term = self._term_request(opstr, indx, J)
        result = self._model.execute(
            "materialize_terms_model",
            terms=[term],
            format="csc",
            checks={
                "hermiticity": False,
                "particle_conservation": False,
                "symmetry_compatibility": False,
            },
        )
        entries = sorted(
            result["entries"],
            key=lambda entry: (entry["column"], entry["row"]),
        )
        matrix_elements = self._values_for_dtype(
            [complex(*entry["value"]) for entry in entries],
            dtype,
        )
        row = np.asarray([entry["row"] for entry in entries], dtype=np.intp)
        column = np.asarray([entry["column"] for entry in entries], dtype=np.intp)
        return matrix_elements, row, column

    def inplace_Op(
        self,
        v_in,
        op_list,
        dtype,
        transposed=False,
        conjugated=False,
        a=1.0,
        v_out=None,
    ):
        input_array = np.asanyarray(v_in)
        if input_array.ndim == 0 or input_array.shape[0] != self.Ns:
            raise ValueError("dimension mismatch")
        result_dtype = np.result_type(input_array.dtype, dtype)
        input_array = input_array.astype(result_dtype, order="C", copy=False)
        input_matrix = input_array.reshape((self.Ns, -1))

        if transposed and conjugated:
            action = "adjoint"
        elif transposed:
            action = "transpose"
        elif conjugated:
            action = "conjugate"
        else:
            action = "normal"

        terms = [
            self._term_request(opstr, indx, a * coefficient)
            for opstr, indx, coefficient in op_list
        ]
        vectors = self._complex_vectors(input_matrix)
        result = self._model.execute(
            "apply_terms_model",
            terms=terms,
            vectors=vectors,
            action=action,
        )
        applied = self._vectors_from_result(result)
        applied = self._values_for_dtype(applied, result_dtype).reshape(input_array.shape)

        if v_out is None:
            return applied.squeeze()
        if np.dtype(v_out.dtype) != np.dtype(result_dtype):
            raise TypeError("v_out does not have the correct data type.")
        if not v_out.flags["CARRAY"]:
            raise ValueError("v_out is not a writable C-contiguous array")
        if v_out.shape != input_array.shape:
            raise ValueError("invalid shape for v_out and v_in: v_in.shape != v_out.shape")
        v_out += applied
        return v_out.squeeze()

    def Op_bra_ket(
        self,
        opstr,
        indx,
        J,
        dtype,
        ket_states,
        reduce_output=True,
    ):
        kets = np.array(ket_states, dtype=object, ndmin=1)
        request = {
            "terms": [self._term_request(opstr, indx, J)],
            "kets": [
                str(self._compat_state_encoding(int(ket)))
                for ket in kets
            ],
        }
        plan = getattr(self, "_basis_plan", None)
        if plan is not None and not self._made_basis:
            result = plan.execute("bra_ket_terms_plan", **request)
        else:
            result = self._model.execute("bra_ket_terms_model", **request)
        grouped: list[list[dict[str, Any]]] = [[] for _ in range(kets.size)]
        for entry in result["entries"]:
            grouped[int(entry["input"])].append(entry)

        if reduce_output:
            entries = [entry for group in grouped for entry in group]
            matrix_elements = self._values_for_dtype(
                [complex(*entry["value"]) for entry in entries],
                dtype,
            )
            bras = np.asarray(
                [
                    self._compat_state_encoding(int(entry["bra"]))
                    for entry in entries
                ],
                dtype=self.dtype,
            )
            returned_kets = np.asarray(
                [
                    self._compat_state_encoding(int(entry["ket"]))
                    for entry in entries
                ],
                dtype=self.dtype,
            )
            return matrix_elements, bras, returned_kets

        if any(len(group) > 1 for group in grouped):
            raise NotImplementedError(
                "reduce_output=False cannot represent a branching local operator"
            )
        values = []
        bras = []
        for ket, group in zip(kets, grouped):
            if group:
                values.append(complex(*group[0]["value"]))
                bras.append(
                    self._compat_state_encoding(int(group[0]["bra"]))
                )
            else:
                values.append(0.0)
                bras.append(0)
        return (
            self._values_for_dtype(values, dtype),
            np.asarray(bras, dtype=self.dtype),
            kets.astype(self.dtype, copy=False),
        )


class spin_basis_1d(_PackedBasis):
    def __init__(
        self,
        L: int,
        Nup: int | None = None,
        m: float | None = None,
        S: str | int | float = "1/2",
        pauli: bool | int = True,
        kblock: int | None = None,
        pblock: int | None = None,
        zblock: int | None = None,
        a: int = 1,
        **blocks,
    ):
        spin_twice = _spin_twice(S)
        if m is not None:
            if Nup is not None:
                raise ValueError("Nup and m cannot both be specified")
            Nup = round((float(m) + spin_twice / 2) * L)
        fixed_up, up_sectors = _single_species_sectors(
            Nup,
            "Nup",
            negative_from=int(L) * spin_twice,
        )
        self.N = int(L)
        a = int(a)
        if a <= 0 or self.N % a:
            raise ValueError("a must be a positive divisor of L")
        pzblock = blocks.pop("pzblock", None)
        zAblock = blocks.pop("zAblock", None)
        zBblock = blocks.pop("zBblock", None)
        _reject_options("spin_basis_1d", blocks)
        symmetries = _one_dimensional_symmetries(
            self.N,
            states_per_site=spin_twice + 1,
            momentum=kblock,
            parity=pblock,
            translation_step=a,
        )
        reflection_indices = (
            [int(kblock is not None)] if pblock is not None else []
        )
        if zblock is not None:
            if zblock not in (-1, 1):
                raise ValueError("zblock must be either -1 or +1")
            inversion = -(np.arange(self.N) + 1)
            symmetries.extend(
                _general_symmetries(
                    {
                        "spin_inversion": (
                            inversion,
                            0 if zblock == 1 else 1,
                        )
                    },
                    sites=self.N,
                    states_per_site=spin_twice + 1,
                )
            )
        for name, block, site_map in (
            (
                "pzblock",
                pzblock,
                -(np.arange(self.N)[::-1] + 1),
            ),
            (
                "zAblock",
                zAblock,
                np.where(
                    np.arange(self.N) % 2 == 0,
                    -(np.arange(self.N) + 1),
                    np.arange(self.N),
                ),
            ),
            (
                "zBblock",
                zBblock,
                np.where(
                    np.arange(self.N) % 2 == 1,
                    -(np.arange(self.N) + 1),
                    np.arange(self.N),
                ),
            ),
        ):
            if block is None:
                continue
            if block not in (-1, 1):
                raise ValueError(f"{name} must be either -1 or +1")
            symmetry_index = len(symmetries)
            symmetries.extend(
                _general_symmetries(
                    {name: (site_map, 0 if block == 1 else 1)},
                    sites=self.N,
                    states_per_site=spin_twice + 1,
                )
            )
            if name == "pzblock":
                reflection_indices.append(symmetry_index)
        matrix_symmetry = _generic_momentum_reflection_representation(
            symmetries,
            momentum=kblock,
            reflection_indices=reflection_indices,
            states_per_site=spin_twice + 1,
        )
        self._request = {
            "kind": "spin",
            "sites": self.N,
            "spin_twice": spin_twice,
            "up": fixed_up,
            "up_sectors": up_sectors,
            "momentum": None,
            "parity": None,
            "normalization": _spin_normalization(pauli, spin_twice),
            "symmetries": [] if matrix_symmetry is not None else symmetries,
            "matrix_symmetry": matrix_symmetry,
            "reverse": True,
        }


class boson_basis_1d(_PackedBasis):
    def __init__(
        self,
        L: int,
        Nb: int | None = None,
        nb: float | None = None,
        sps: int | None = None,
        kblock: int | None = None,
        pblock: int | None = None,
        cblock: int | None = None,
        a: int = 1,
        **blocks,
    ):
        _reject_options("boson_basis_1d", blocks)
        self.N = int(L)
        if Nb is not None and nb is not None:
            raise ValueError("Nb and nb cannot both be specified")
        if Nb is None and nb is not None:
            Nb = int(float(nb) * self.N)
        if Nb is None and sps is None:
            raise ValueError("expecting value for 'Nb', 'nb', or 'sps'")
        fixed_particles, particle_sectors = _single_species_sectors(Nb, "Nb")
        a = int(a)
        if a <= 0 or self.N % a:
            raise ValueError("a must be a positive divisor of L")
        states_per_site = int(
            sps
            if sps is not None
            else (
                fixed_particles + 1
                if fixed_particles is not None
                else max(particle_sectors or [1]) + 1
            )
        )
        symmetries = _one_dimensional_symmetries(
            self.N,
            states_per_site=states_per_site,
            momentum=kblock,
            parity=pblock,
            translation_step=a,
        )
        if cblock is not None:
            if cblock not in (-1, 1):
                raise ValueError("cblock must be either -1 or +1")
            particle_hole = -(np.arange(self.N) + 1)
            symmetries.extend(
                _general_symmetries(
                    {
                        "particle_hole": (
                            particle_hole,
                            0 if cblock == 1 else 1,
                        )
                    },
                    sites=self.N,
                    states_per_site=states_per_site,
                )
            )
        matrix_symmetry = _generic_momentum_reflection_representation(
            symmetries,
            momentum=kblock,
            reflection_indices=(
                [int(kblock is not None)] if pblock is not None else []
            ),
            states_per_site=states_per_site,
        )
        self._request = {
            "kind": "boson",
            "sites": self.N,
            "particles": fixed_particles,
            "particle_sectors": particle_sectors,
            "states_per_site": states_per_site,
            "symmetries": [] if matrix_symmetry is not None else symmetries,
            "matrix_symmetry": matrix_symmetry,
            "reverse": True,
        }


class ho_basis(boson_basis_1d):
    def __init__(self, Np: int):
        self.Np = int(Np)
        super().__init__(1, sps=self.Np + 1)
        self._request["reverse"] = False


class spinless_fermion_basis_1d(_PackedBasis):
    def __init__(
        self,
        L: int,
        Nf: int | None = None,
        nf: float | None = None,
        kblock: int | None = None,
        pblock: int | None = None,
        a: int = 1,
        **blocks,
    ):
        _reject_options("spinless_fermion_basis_1d", blocks)
        self.N = int(L)
        if Nf is not None and nf is not None:
            raise ValueError("Nf and nf cannot both be specified")
        if Nf is None and nf is not None:
            density = float(nf)
            if not 0.0 <= density <= 1.0:
                raise ValueError("nf must be between 0 and 1")
            Nf = int(density * self.N)
        fixed_particles, particle_sectors = _single_species_sectors(Nf, "Nf")
        a = int(a)
        if a <= 0 or self.N % a:
            raise ValueError("a must be a positive divisor of L")
        symmetries = _one_dimensional_symmetries(
            self.N,
            states_per_site=2,
            momentum=kblock,
            parity=pblock,
            fermionic=True,
            translation_step=a,
        )
        matrix_symmetry = _generic_momentum_reflection_representation(
            symmetries,
            momentum=kblock,
            reflection_indices=(
                [int(kblock is not None)] if pblock is not None else []
            ),
            states_per_site=2,
        )
        self._request = {
            "kind": "spinless_fermion",
            "sites": self.N,
            "particles": fixed_particles,
            "particle_sectors": particle_sectors,
            "momentum": None,
            "symmetries": [] if matrix_symmetry is not None else symmetries,
            "matrix_symmetry": matrix_symmetry,
            "reverse": True,
        }


class spinful_fermion_basis_1d(_PackedBasis):
    def __init__(
        self,
        L: int,
        Nf: tuple[int, int] | None = None,
        nf: tuple[float, float] | None = None,
        kblock: int | None = None,
        pblock: int | None = None,
        a: int = 1,
        double_occupancy: bool = True,
        **blocks,
    ):
        sblock = blocks.pop("sblock", None)
        psblock = blocks.pop("psblock", None)
        _reject_options("spinful_fermion_basis_1d", blocks)
        sites = int(L)
        if Nf is not None and nf is not None:
            raise ValueError("Nf and nf cannot both be specified")
        if Nf is None and nf is not None:
            if len(nf) != 2 or any(
                not 0.0 <= float(value) <= 1.0 for value in nf
            ):
                raise ValueError("nf must contain two densities between 0 and 1")
            Nf = (int(float(nf[0]) * sites), int(float(nf[1]) * sites))
        a = int(a)
        if a <= 0 or sites % a:
            raise ValueError("a must be a positive divisor of L")
        fixed_particles, particle_sectors = _spinful_sectors(Nf)
        particles_up, particles_down = (
            (None, None) if fixed_particles is None else fixed_particles
        )
        spatial_symmetries = _one_dimensional_symmetries(
            sites,
            states_per_site=2,
            momentum=kblock,
            parity=pblock,
            fermionic=True,
            translation_step=a,
        )
        symmetries = []
        for symmetry in spatial_symmetries:
            destinations = symmetry["destinations"]
            symmetries.append(
                {
                    **symmetry,
                    "destinations": destinations
                    + [sites + destination for destination in destinations],
                }
            )
        reflection_indices = (
            [int(kblock is not None)] if pblock is not None else []
        )
        if sblock is not None:
            if sblock not in (-1, 1):
                raise ValueError("sblock must be either -1 or +1")
            species_exchange = list(range(sites, 2 * sites)) + list(
                range(sites)
            )
            symmetries.extend(
                _general_symmetries(
                    {
                        "species_exchange": (
                            np.asarray(species_exchange),
                            0 if sblock == 1 else 1,
                        )
                    },
                    sites=2 * sites,
                    states_per_site=2,
                    fermionic=True,
                )
            )
        if psblock is not None:
            if psblock not in (-1, 1):
                raise ValueError("psblock must be either -1 or +1")
            reflected = list(range(sites - 1, -1, -1))
            parity_species_exchange = [
                sites + destination for destination in reflected
            ] + reflected
            reflection_indices.append(len(symmetries))
            symmetries.extend(
                _general_symmetries(
                    {
                        "parity_species_exchange": (
                            np.asarray(parity_species_exchange),
                            0 if psblock == 1 else 1,
                        )
                    },
                    sites=2 * sites,
                    states_per_site=2,
                    fermionic=True,
                )
            )
        matrix_symmetry = _generic_momentum_reflection_representation(
            symmetries,
            momentum=kblock,
            reflection_indices=reflection_indices,
            states_per_site=2,
        )
        self._request = {
            "kind": "spinful_fermion",
            "sites": sites,
            "particles_up": particles_up,
            "particles_down": particles_down,
            "particle_sectors": particle_sectors,
            "allowed_local_occupancies": None
            if bool(double_occupancy)
            else [0, 1, 2],
            "symmetries": [] if matrix_symmetry is not None else symmetries,
            "matrix_symmetry": matrix_symmetry,
            "reverse": True,
        }
        self.N = 2 * sites

    def index(self, up_state, down_state=None):
        if down_state is None:
            return super().index(up_state)
        packed = (int(up_state) << self._site_count) | int(down_state)
        return super().index(packed)


class tensor_basis(_PackedBasis):
    """Runtime-erased direct product of two or more QMBED-backed factors."""

    def __init__(self, *factors):
        flattened = []
        for factor in factors:
            if isinstance(factor, tensor_basis):
                flattened.extend(factor._factors)
            elif isinstance(factor, _PackedBasis):
                flattened.append(factor)
            else:
                raise TypeError("tensor_basis factors must be QMBED-backed bases")
        if len(flattened) < 2:
            raise ValueError("tensor_basis requires at least two factors")
        self._factors = tuple(flattened)
        self.N = tuple(factor.N for factor in self._factors)
        self._request = {
            "kind": "tensor",
            "factors": [deepcopy(factor._request) for factor in self._factors],
        }

    @property
    def basis_left(self):
        return self._factors[0]

    @property
    def L(self):
        raise AttributeError("'tensor_basis' object has no attribute 'L'")

    @property
    def basis_right(self):
        if len(self._factors) == 2:
            return self._factors[1]
        return tensor_basis(*self._factors[1:])

    def index(self, *states):
        if len(states) != len(self._factors):
            raise ValueError("one state must be supplied for every tensor factor")
        index = 0
        for factor, state in zip(self._factors, states):
            index = index * factor.Ns + factor.index(state)
        return index

    def get_proj(self, dtype, full_left=True, full_right=True):
        left = (
            self.basis_left.get_proj(dtype)
            if bool(full_left)
            else sp.identity(self.basis_left.Ns, dtype=dtype, format="csc")
        )
        right_basis = self.basis_right
        right = (
            right_basis.get_proj(dtype)
            if bool(full_right)
            else sp.identity(right_basis.Ns, dtype=dtype, format="csc")
        )
        return sp.kron(left, right, format="csr")

    def project_from(
        self,
        v0,
        sparse=True,
        full_left=True,
        full_right=True,
    ):
        array = v0 if sp.issparse(v0) else np.asanyarray(v0)
        if array.ndim == 0 or array.shape[0] != self.Ns:
            raise ValueError("v0 has incompatible dimensions with basis")
        projector = self.get_proj(
            array.dtype,
            full_left=full_left,
            full_right=full_right,
        )
        output = projector.dot(array)
        if sparse:
            if sp.issparse(output):
                return output
            return sp.csc_matrix(np.asarray(output).reshape((output.shape[0], -1)))
        if sp.issparse(output):
            output = output.toarray()
        output = np.asarray(output)
        return output.reshape(-1) if np.ndim(v0) == 1 else output

    def get_vec(self, v0, sparse=True, full_left=True, full_right=True):
        return self.project_from(
            v0,
            sparse=sparse,
            full_left=full_left,
            full_right=full_right,
        )

    @property
    def dtype(self) -> np.dtype:
        maximum = max(self.Ns - 1, 0)
        if maximum <= np.iinfo(np.uint32).max:
            return np.dtype(np.uint32)
        if maximum <= np.iinfo(np.uint64).max:
            return np.dtype(np.uint64)
        return np.dtype(object)

    @property
    def _site_permutation(self):
        # Tensor operator sites are local to each factor and may repeat across
        # separators, so there is no single global site relabeling.
        return None

    def _parent_request(self, *, pcon: bool) -> dict[str, Any]:
        del pcon
        # Factor sectors are part of the tensor-product local dimensions.
        return deepcopy(self._request)

    def _subsystem_layout(
        self,
        sub_sys_A,
        *,
        subsys_ordering: bool,
    ) -> tuple[list[int], list[int], int, int]:
        factor_count = len(self._factors)
        if sub_sys_A is None or sub_sys_A == "left":
            selected = (0,)
        elif sub_sys_A == "right":
            selected = tuple(range(1, factor_count))
        elif isinstance(sub_sys_A, (int, np.integer)):
            selected = (int(sub_sys_A),)
        else:
            selected = tuple(int(factor) for factor in sub_sys_A)
        if subsys_ordering:
            selected = tuple(sorted(selected))
        if len(set(selected)) != len(selected) or any(
            factor < 0 or factor >= factor_count for factor in selected
        ):
            raise ValueError("sub_sys_A contains invalid or repeated tensor factors")

        # PackedTensorBasis encodes the first factor as the most-significant
        # mixed-radix digit, while the Rust subsystem kernel lists dimensions
        # from least to most significant.
        local_dimensions = [factor.Ns for factor in reversed(self._factors)]
        retained_sites = [
            factor_count - 1 - factor for factor in reversed(selected)
        ]
        return (
            local_dimensions,
            retained_sites,
            len(selected),
            factor_count - len(selected),
        )

    @staticmethod
    def _density_batch(result: dict[str, Any], name: str, dimension: int) -> np.ndarray:
        matrices = _PackedBasis._density_batch(result, name, dimension)
        # Factor basis rows already carry QuSpin's public ordering. Undo the
        # packed-state reversal used by ordinary site bases.
        return np.flip(matrices, axis=(-2, -1))

    def ent_entropy(
        self,
        state,
        sub_sys_A=None,
        density: bool = False,
        **options,
    ):
        return super().ent_entropy(
            state,
            sub_sys_A=sub_sys_A,
            density=density,
            **options,
        )


class photon_basis(_PackedBasis):
    """Matter basis coupled to one truncated photon mode."""

    def __init__(self, basis_constructor, *constructor_args, **blocks):
        Ntot = blocks.pop("Ntot", None)
        Nph = blocks.pop("Nph", None)
        if Ntot is None and Nph is None:
            raise TypeError("Either Ntot or Nph must be defined")
        selected = Ntot if Ntot is not None else Nph
        if type(selected) is not int:
            raise TypeError(f"{'Ntot' if Ntot is not None else 'Nph'} must be integer")
        if selected < 0:
            raise ValueError(
                f"{'Ntot' if Ntot is not None else 'Nph'} must be an integer >= 0"
            )

        matter_options = dict(blocks)
        if Ntot is not None:
            name = getattr(basis_constructor, "__name__", "")
            if name in {"spin_basis_1d", "spin_basis_general"} and not {
                "Nup",
                "m",
            }.intersection(matter_options):
                spin_twice = _spin_twice(matter_options.get("S", "1/2"))
                sites = int(
                    matter_options.get(
                        "L",
                        matter_options.get(
                            "N",
                            constructor_args[0] if constructor_args else 0,
                        ),
                    )
                )
                matter_options["Nup"] = range(
                    min(int(Ntot), sites * spin_twice) + 1
                )
            elif name in {"boson_basis_1d", "boson_basis_general"} and "Nb" not in matter_options:
                matter_options["Nb"] = range(int(Ntot) + 1)
                matter_options.setdefault("sps", int(Ntot) + 1)
            elif (
                name
                in {
                    "spinless_fermion_basis_1d",
                    "spinless_fermion_basis_general",
                }
                and "Nf" not in matter_options
            ):
                sites = int(
                    matter_options.get(
                        "L",
                        matter_options.get(
                            "N",
                            constructor_args[0] if constructor_args else 0,
                        ),
                    )
                )
                matter_options["Nf"] = range(min(int(Ntot), sites) + 1)
            elif (
                name
                in {
                    "spinful_fermion_basis_1d",
                    "spinful_fermion_basis_general",
                }
                and "Nf" not in matter_options
            ):
                sites = int(
                    matter_options.get(
                        "L",
                        matter_options.get(
                            "N",
                            constructor_args[0] if constructor_args else 0,
                        ),
                    )
                )
                matter_options["Nf"] = [
                    (up, down)
                    for up in range(min(int(Ntot), sites) + 1)
                    for down in range(min(int(Ntot) - up, sites) + 1)
                ]

        matter = basis_constructor(*constructor_args, **matter_options)
        if not isinstance(matter, _PackedBasis) or isinstance(
            matter, (tensor_basis, photon_basis)
        ):
            raise TypeError(
                "photon_basis requires a non-tensor QMBED-backed matter basis"
            )
        self._matter_basis = matter
        self._Ntot = Ntot
        self._Nph = Nph
        self._photon_cutoff = int(selected)
        self._photon_basis = ho_basis(self._photon_cutoff)
        self._parent_models: dict[tuple[int, bool], NativeModel] = {}
        self.N = (matter.N, 1)
        self._request = {
            "kind": "photon",
            "matter": deepcopy(matter._request),
            "photon_cutoff": self._photon_cutoff,
            "total_excitations": Ntot,
        }

    @property
    def dtype(self) -> np.dtype:
        maximum = max(
            (
                int(state) * (self._photon_cutoff + 1) + self._photon_cutoff
                for state in self._matter_basis.states
            ),
            default=0,
        )
        if maximum <= np.iinfo(np.uint32).max:
            return np.dtype(np.uint32)
        if maximum <= np.iinfo(np.uint64).max:
            return np.dtype(np.uint64)
        return np.dtype(object)

    @property
    def L(self):
        raise AttributeError("'photon_basis' object has no attribute 'L'")

    @property
    def _site_permutation(self):
        return None

    @property
    def Ntot(self):
        return self._Ntot

    @property
    def Nph(self):
        return self._Nph

    @property
    def basis_particle(self):
        return self._matter_basis

    @property
    def basis_photon(self):
        return self._photon_basis

    @property
    def basis_left(self):
        return self._matter_basis

    @property
    def basis_right(self):
        return self._photon_basis

    def index(self, *states):
        if len(states) != 2:
            raise ValueError("states must be list of atleast 2 elements long")
        matter_index = self._matter_basis.index(states[0])
        photon_index = self._photon_basis.index(states[1])
        return photon_index + self._photon_basis.Ns * matter_index

    @property
    def particle_Ns(self):
        return self._matter_basis.Ns

    @property
    def particle_N(self):
        return self._matter_basis.N

    @property
    def particle_sps(self):
        return self._matter_basis.sps

    @property
    def photon_Ns(self):
        return self._photon_cutoff + 1

    @property
    def photon_sps(self):
        return self._photon_cutoff + 1

    @property
    def chain_Ns(self):
        return self._matter_basis.Ns

    @property
    def chain_N(self):
        return self._matter_basis.N

    @staticmethod
    def _complete_operator_indices(opstr, indices):
        if opstr.count("|") != 1:
            raise ValueError("photon operator strings require exactly one '|' separator")
        matter_operator, photon_operator = opstr.split("|")
        indices = [int(site) for site in indices]
        if len(indices) == len(matter_operator):
            indices.extend([0] * len(photon_operator))
        elif len(indices) != len(matter_operator) + len(photon_operator):
            raise ValueError(
                "the number of indices must match the matter operators, "
                "with photon indices optional"
            )
        return indices

    def _term_request(self, opstr, indx, coefficient):
        return super()._term_request(
            opstr,
            self._complete_operator_indices(str(opstr), indx),
            coefficient,
        )

    def _normalize_operator_lists(self, static, dynamic):
        normalized_static = []
        for opstr, couplings in static:
            rows = [
                [
                    row[0],
                    *self._complete_operator_indices(opstr, row[1:]),
                ]
                for row in couplings
            ]
            normalized_static.append([opstr, rows])
        normalized_dynamic = []
        for opstr, couplings, drive, arguments in dynamic:
            rows = [
                [
                    row[0],
                    *self._complete_operator_indices(opstr, row[1:]),
                ]
                for row in couplings
            ]
            normalized_dynamic.append([opstr, rows, drive, arguments])
        return normalized_static, normalized_dynamic

    def _photon_parent_request(self, *, Nph: int, full_part: bool):
        matter = (
            self._matter_basis._parent_request(pcon=False)
            if full_part
            else deepcopy(self._matter_basis._request)
        )
        return {
            "kind": "photon",
            "matter": matter,
            "photon_cutoff": int(Nph),
            "total_excitations": None,
        }

    def _parent_request(self, *, pcon: bool) -> dict[str, Any]:
        del pcon
        return self._photon_parent_request(
            Nph=self._photon_cutoff,
            full_part=True,
        )

    def _projection_parent(self, *, Nph, full_part):
        cutoff = self._photon_cutoff if Nph is None else Nph
        if type(cutoff) is not int:
            raise TypeError("Nph must be integer")
        if cutoff < self._photon_cutoff:
            raise ValueError(f"Nph must be larger or equal to {self._photon_cutoff}")
        key = (cutoff, bool(full_part))
        if key not in self._parent_models:
            self._parent_models[key] = self._new_empty_model(
                self._photon_parent_request(
                    Nph=cutoff,
                    full_part=bool(full_part),
                )
            )
        return self._parent_models[key]

    def get_proj(self, dtype, Nph=None, full_part=True):
        parent = self._projection_parent(Nph=Nph, full_part=full_part)
        result = self._model.execute(
            "projector_model",
            parent_handle=parent.handle,
            embedding=not bool(full_part),
        )
        entries = result["entries"]
        values = self._values_for_dtype(
            [complex(*entry["value"]) for entry in entries],
            dtype,
        )
        rows = np.asarray([entry["row"] for entry in entries], dtype=np.intp)
        columns = np.asarray([entry["column"] for entry in entries], dtype=np.intp)
        return sp.csc_matrix(
            (values, (rows, columns)),
            shape=tuple(result["shape"]),
            dtype=np.dtype(dtype),
        )

    def project_from(self, v0, sparse=True, Nph=None, full_part=True):
        array = np.asanyarray(v0)
        if array.ndim == 0 or array.shape[0] != self.Ns:
            raise ValueError("v0 has incompatible dimensions with basis")
        result_dtype = np.result_type(array.dtype, np.complex128)
        array = array.astype(result_dtype, order="C", copy=False)
        parent = self._projection_parent(Nph=Nph, full_part=full_part)
        result = self._model.execute(
            "apply_projector_model",
            parent_handle=parent.handle,
            embedding=not bool(full_part),
            vectors=self._complex_vectors(array),
            action="lift",
        )
        output = self._values_for_dtype(
            self._vectors_from_result(result),
            result_dtype,
        ).reshape((int(result["dimension"]), *array.shape[1:]))
        if sparse:
            return sp.csc_matrix(output.reshape((output.shape[0], -1)))
        return output

    def get_vec(self, v0, sparse=True, Nph=None, full_part=True):
        return self.project_from(
            v0,
            sparse=sparse,
            Nph=Nph,
            full_part=full_part,
        )

    def _subsystem_layout(
        self,
        sub_sys_A,
        *,
        subsys_ordering: bool,
    ) -> tuple[list[int], list[int], int, int]:
        del subsys_ordering
        if sub_sys_A is None:
            sub_sys_A = "particles"
        if sub_sys_A in {"particles", "left"}:
            retained_sites = [1]
        elif sub_sys_A in {"photons", "right"}:
            retained_sites = [0]
        else:
            raise ValueError("sub_sys_A must be 'particles' or 'photons'")
        matter_dimension = self._matter_basis._parent_model(pcon=False).dimension
        return (
            [self._photon_cutoff + 1, matter_dimension],
            retained_sites,
            1,
            1,
        )

    @staticmethod
    def _density_batch(result: dict[str, Any], name: str, dimension: int) -> np.ndarray:
        matrices = _PackedBasis._density_batch(result, name, dimension)
        return np.flip(matrices, axis=(-2, -1))

    def ent_entropy(
        self,
        state,
        sub_sys_A="particles",
        density: bool = False,
        **options,
    ):
        return super().ent_entropy(
            state,
            sub_sys_A=sub_sys_A,
            density=density,
            **options,
        )


class spin_basis_general(_PackedBasis):
    def __init__(
        self,
        N: int,
        Nup: int | None = None,
        m: float | None = None,
        S: str | int | float = "1/2",
        pauli: bool | int = True,
        Ns_block_est: int | None = None,
        make_basis: bool = True,
        block_order=None,
        **blocks,
    ):
        del Ns_block_est
        spin_twice = _spin_twice(S)
        if block_order is not None:
            ordered = {
                name: blocks.pop(name)
                for name in block_order
                if name in blocks
            }
            ordered.update(blocks)
            blocks = ordered
        if m is not None:
            if Nup is not None:
                raise ValueError("Nup and m cannot both be specified")
            Nup = round((float(m) + spin_twice / 2) * N)
        fixed_up, up_sectors = _single_species_sectors(
            Nup,
            "Nup",
            negative_from=int(N) * spin_twice,
        )
        self.N = int(N)
        self._request = {
            "kind": "spin",
            "sites": self.N,
            "spin_twice": spin_twice,
            "up": fixed_up,
            "up_sectors": up_sectors,
            "momentum": None,
            "parity": None,
            "normalization": _spin_normalization(pauli, spin_twice),
            "symmetries": _general_symmetries(
                blocks,
                sites=self.N,
                states_per_site=spin_twice + 1,
            ),
            "reverse": True,
        }
        self._initialize_general_basis(make_basis=bool(make_basis))


class boson_basis_general(_PackedBasis):
    def __init__(
        self,
        N: int,
        Nb: int | None = None,
        nb: float | None = None,
        sps: int | None = None,
        Ns_block_est: int | None = None,
        make_basis: bool = True,
        block_order=None,
        **blocks,
    ):
        del Ns_block_est
        if block_order is not None:
            ordered = {
                name: blocks.pop(name)
                for name in block_order
                if name in blocks
            }
            ordered.update(blocks)
            blocks = ordered
        self.N = int(N)
        if Nb is not None and nb is not None:
            raise ValueError("Nb and nb cannot both be specified")
        if Nb is None and nb is not None:
            Nb = int(float(nb) * self.N)
        if Nb is None and sps is None:
            raise ValueError("expecting value for 'Nb', 'nb', or 'sps'")
        fixed_particles, particle_sectors = _single_species_sectors(Nb, "Nb")
        states_per_site = int(
            sps
            if sps is not None
            else (
                fixed_particles + 1
                if fixed_particles is not None
                else max(particle_sectors or [1]) + 1
            )
        )
        self._request = {
            "kind": "boson",
            "sites": self.N,
            "particles": fixed_particles,
            "particle_sectors": particle_sectors,
            "states_per_site": states_per_site,
            "symmetries": _general_symmetries(
                blocks,
                sites=self.N,
                states_per_site=states_per_site,
            ),
            "reverse": True,
        }
        self._initialize_general_basis(make_basis=bool(make_basis))


class spinless_fermion_basis_general(_PackedBasis):
    def __init__(
        self,
        N: int,
        Nf: int | None = None,
        nf: float | None = None,
        Ns_block_est: int | None = None,
        make_basis: bool = True,
        block_order=None,
        **blocks,
    ):
        del Ns_block_est
        if block_order is not None:
            ordered = {
                name: blocks.pop(name)
                for name in block_order
                if name in blocks
            }
            ordered.update(blocks)
            blocks = ordered
        self.N = int(N)
        if Nf is not None and nf is not None:
            raise ValueError("Nf and nf cannot both be specified")
        if Nf is None and nf is not None:
            density = float(nf)
            if not 0.0 <= density <= 1.0:
                raise ValueError("nf must be between 0 and 1")
            Nf = int(density * self.N)
        fixed_particles, particle_sectors = _single_species_sectors(Nf, "Nf")
        self._request = {
            "kind": "spinless_fermion",
            "sites": self.N,
            "particles": fixed_particles,
            "particle_sectors": particle_sectors,
            "momentum": None,
            "symmetries": _general_symmetries(
                blocks,
                sites=self.N,
                states_per_site=2,
                fermionic=True,
            ),
            "reverse": True,
        }
        self._initialize_general_basis(make_basis=bool(make_basis))


class spinful_fermion_basis_general(_PackedBasis):
    def __init__(
        self,
        N: int,
        Nf: tuple[int, int] | None = None,
        nf: tuple[float, float] | None = None,
        Ns_block_est: int | None = None,
        simple_symm: bool = True,
        double_occupancy: bool = True,
        make_basis: bool = True,
        block_order=None,
        **blocks,
    ):
        del Ns_block_est
        if "simple_symm" in blocks:
            simple_symm = blocks.pop("simple_symm")
        simple_symm = bool(simple_symm)
        self._unified_orbitals = not simple_symm
        if block_order is not None:
            ordered = {
                name: blocks.pop(name)
                for name in block_order
                if name in blocks
            }
            ordered.update(blocks)
            blocks = ordered
        sites = int(N)
        if Nf is not None and nf is not None:
            raise ValueError("Nf and nf cannot both be specified")
        if Nf is None and nf is not None:
            if len(nf) != 2 or any(
                not 0.0 <= float(value) <= 1.0 for value in nf
            ):
                raise ValueError("nf must contain two densities between 0 and 1")
            Nf = (int(float(nf[0]) * sites), int(float(nf[1]) * sites))
        fixed_particles, particle_sectors = _spinful_sectors(Nf)
        particles_up, particles_down = (
            (None, None) if fixed_particles is None else fixed_particles
        )
        if simple_symm:
            spatial_symmetries = _general_symmetries(
                blocks,
                sites=sites,
                states_per_site=2,
                fermionic=True,
            )
            symmetries = []
            for symmetry in spatial_symmetries:
                destinations = symmetry["destinations"]
                symmetries.append(
                    {
                        **symmetry,
                        "destinations": destinations
                        + [sites + destination for destination in destinations],
                    }
                )
        else:
            symmetries = _general_symmetries(
                blocks,
                sites=2 * sites,
                states_per_site=2,
                fermionic=True,
            )
        self._request = {
            "kind": "spinful_fermion",
            "sites": sites,
            "particles_up": None if particles_up is None else int(particles_up),
            "particles_down": None if particles_down is None else int(particles_down),
            "particle_sectors": particle_sectors,
            "allowed_local_occupancies": None
            if bool(double_occupancy)
            else [0, 1, 2],
            "symmetries": symmetries,
            "reverse": True,
        }
        self.N = 2 * sites
        self._initialize_general_basis(make_basis=bool(make_basis))

    def index(self, up_state, down_state=None):
        if down_state is None:
            return super().index(up_state)
        packed = (int(up_state) << self._site_count) | int(down_state)
        return super().index(packed)


def _bitwise_state_width(*arrays: np.ndarray) -> int:
    native_widths = [
        array.dtype.itemsize * 8
        for array in arrays
        if array.dtype != np.dtype(object)
    ]
    if len(native_widths) == len(arrays):
        return max(native_widths, default=1)
    maximum_bits = 1
    for array in arrays:
        for value in array.reshape(-1):
            integer = int(value)
            if integer < 0:
                raise ValueError("basis bitwise operations require nonnegative integers")
            maximum_bits = max(maximum_bits, integer.bit_length())
    for capacity in (128, 256, 1024, 4096, 16384):
        if maximum_bits <= capacity:
            return capacity
    raise OverflowError("basis integer requires more than 16384 bits")


def _bitwise_output(
    computed: np.ndarray,
    *,
    dtype: np.dtype,
    where,
    out,
):
    computed = np.asarray(computed, dtype=dtype)
    mask = np.broadcast_to(np.asarray(where, dtype=bool), computed.shape)
    if out is not None:
        if isinstance(out, tuple):
            if len(out) != 1:
                raise TypeError("bitwise out tuple must contain one array")
            out = out[0]
        if not isinstance(out, np.ndarray):
            raise TypeError("out must be a numpy.ndarray")
        if out.shape != computed.shape:
            raise ValueError("out and broadcast result have different shapes")
        np.copyto(out, computed, where=mask, casting="unsafe")
        return out
    result = np.zeros(computed.shape, dtype=dtype)
    np.copyto(result, computed, where=mask, casting="unsafe")
    return result


def _bitwise_command(operation: str, left, right=None, *, where=True, out=None):
    left = np.asarray(left)
    if right is None:
        arrays = [left]
    else:
        arrays = list(np.broadcast_arrays(left, np.asarray(right)))
        left = arrays[0]
    dtype = (
        np.dtype(object)
        if any(array.dtype == np.dtype(object) for array in arrays)
        else np.result_type(*[array.dtype for array in arrays])
    )
    width_bits = _bitwise_state_width(*arrays)
    mask = (1 << width_bits) - 1
    request = {
        "operation": "bitwise_states",
        "bitwise_operation": operation,
        "width_bits": width_bits,
        "left": [str(int(value) & mask) for value in left.reshape(-1)],
    }
    if operation in {"and", "or", "xor"}:
        request["right"] = [
            str(int(value) & mask) for value in arrays[1].reshape(-1)
        ]
    elif operation in {"left_shift", "right_shift"}:
        shifts = np.broadcast_to(np.asarray(right), left.shape)
        if np.any(shifts < 0):
            raise ValueError("bitwise shifts must be nonnegative")
        request["shifts"] = [int(value) for value in shifts.reshape(-1)]
        dtype = left.dtype
    response = command(request)
    values = np.asarray(
        [int(value) for value in response["values"]],
        dtype=object,
    ).reshape(left.shape)
    return _bitwise_output(values, dtype=dtype, where=where, out=out)


def bitwise_not(x, out=None, where=None):
    return _bitwise_command(
        "not",
        x,
        where=True if where is None else where,
        out=out,
    )


def bitwise_and(x1, x2, out=None, where=None):
    return _bitwise_command(
        "and",
        x1,
        x2,
        where=True if where is None else where,
        out=out,
    )


def bitwise_or(x1, x2, out=None, where=None):
    return _bitwise_command(
        "or",
        x1,
        x2,
        where=True if where is None else where,
        out=out,
    )


def bitwise_xor(x1, x2, out=None, where=None):
    return _bitwise_command(
        "xor",
        x1,
        x2,
        where=True if where is None else where,
        out=out,
    )


def bitwise_leftshift(x1, x2, out=None, where=None):
    return _bitwise_command(
        "left_shift",
        x1,
        x2,
        where=True if where is None else where,
        out=out,
    )


def bitwise_rightshift(x1, x2, out=None, where=None):
    return _bitwise_command(
        "right_shift",
        x1,
        x2,
        where=True if where is None else where,
        out=out,
    )


from quspin.basis.user import (
    count_particles_sig_32,
    count_particles_sig_64,
    map_sig_32,
    map_sig_64,
    next_state_sig_32,
    next_state_sig_64,
    op_sig_32,
    op_sig_64,
    pre_check_state_sig_32,
    pre_check_state_sig_64,
    user_basis,
)


basis = _PackedBasis
lattice_basis = _PackedBasis
uint32 = np.uint32
uint64 = np.uint64


def isbasis(obj):
    return isinstance(obj, _PackedBasis)


def get_basis_type(N, Np, sps):
    del Np
    maximum = int(sps) ** int(N) - 1
    if maximum <= np.iinfo(np.uint32).max:
        return np.uint32
    if maximum <= np.iinfo(np.uint64).max:
        return np.uint64
    return np.dtype(object)


def python_int_to_basis_int(python_int, dtype=None):
    value = int(python_int)
    if value < 0:
        raise OverflowError("basis integers must be nonnegative")
    if dtype is None:
        if value <= np.iinfo(np.uint32).max:
            dtype = np.uint32
        elif value <= np.iinfo(np.uint64).max:
            dtype = np.uint64
        else:
            dtype = object
    return np.asarray(value, dtype=dtype)


def basis_zeros(shape, dtype=np.uint32):
    return np.zeros(shape, dtype=dtype)


def basis_ones(shape, dtype=np.uint32):
    return np.ones(shape, dtype=dtype)


def coherent_state(a, n, dtype=np.float64):
    coefficient = complex(a)
    values = np.asarray(
        [
            np.exp(-0.5 * abs(coefficient) ** 2)
            * coefficient**occupation
            / math.sqrt(math.factorial(occupation))
            for occupation in range(int(n))
        ]
    )
    if np.dtype(dtype).kind != "c" and np.any(np.abs(values.imag) > 1.0e-14):
        raise TypeError("complex coherent-state amplitudes require a complex dtype")
    return np.asarray(np.real_if_close(values), dtype=dtype)


def photon_Hspace_dim(N, Ntot, Nph):
    sites = int(N)
    if Ntot is None:
        if Nph is None:
            raise TypeError("Either 'Ntot' or 'Nph' must be defined!")
        return (1 << sites) * (int(Nph) + 1)
    total = int(Ntot)
    return float(
        sum(
            math.comb(sites, particles)
            for particles in range(total + 1)
            if 0 <= particles <= sites
        )
    )


__all__ = [
    "basis",
    "basis_ones",
    "basis_zeros",
    "basis_int_to_python_int",
    "bitwise_and",
    "bitwise_leftshift",
    "bitwise_not",
    "bitwise_or",
    "bitwise_rightshift",
    "bitwise_xor",
    "boson_basis_1d",
    "boson_basis_general",
    "coherent_state",
    "count_particles_sig_32",
    "count_particles_sig_64",
    "get_basis_type",
    "ho_basis",
    "isbasis",
    "lattice_basis",
    "map_sig_32",
    "map_sig_64",
    "next_state_sig_32",
    "next_state_sig_64",
    "op_sig_32",
    "op_sig_64",
    "photon_basis",
    "photon_Hspace_dim",
    "pre_check_state_sig_32",
    "pre_check_state_sig_64",
    "python_int_to_basis_int",
    "spin_basis_1d",
    "spin_basis_general",
    "spinful_fermion_basis_1d",
    "spinful_fermion_basis_general",
    "spinless_fermion_basis_1d",
    "spinless_fermion_basis_general",
    "tensor_basis",
    "uint32",
    "uint64",
    "user_basis",
]


def basis_int_to_python_int(basis_int):
    array = np.asarray(basis_int, dtype=object)
    if array.ndim == 0:
        return int(array.item())
    converted = np.asarray(
        [int(item) for item in array.reshape(-1)],
        dtype=object,
    ).reshape(array.shape)
    if isinstance(basis_int, np.ndarray):
        return converted
    return converted.tolist()
