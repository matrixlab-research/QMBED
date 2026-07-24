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
        if fermionic:
            raise NotImplementedError(
                "fermionic particle-hole maps require an explicit phase convention"
            )
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


class _PackedBasis:
    _request: dict[str, Any]
    N: int

    @cached_property
    def _model(self) -> NativeModel:
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
        if make_basis:
            self.make()

    def make(self, Ns_block_est=None, N_p=None):
        del Ns_block_est, N_p
        plan = getattr(self, "_basis_plan", None)
        if plan is None or self._made_basis:
            return None
        self._materialized_model = plan.materialize()
        self._made_basis = True
        self.__dict__.pop("_model", None)
        self.__dict__.pop("_description", None)
        return None

    @cached_property
    def _description(self) -> dict[str, Any]:
        return self._model.execute("describe_model")

    @property
    def Ns(self) -> int:
        return int(self._description["dimension"])

    @property
    def states(self) -> np.ndarray:
        return np.asarray(
            [int(state) for state in self._description["states"]],
            dtype=self.dtype,
        )

    @property
    def dtype(self) -> np.dtype:
        if self._request["kind"] == "spin":
            states_per_site = int(self._request.get("spin_twice", 1)) + 1
            encoded_sites = self.N
        elif self._request["kind"] == "boson":
            states_per_site = int(self._request["states_per_site"])
            encoded_sites = self.N
        elif self._request["kind"] == "spinful_fermion":
            states_per_site = 2
            encoded_sites = 2 * self.N
        else:
            states_per_site = 2
            encoded_sites = self.N
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

    @property
    def _site_permutation(self) -> list[int]:
        return list(range(self.N - 1, -1, -1))

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
        if request["kind"] == "spin":
            request["momentum"] = None
            request["parity"] = None
            if not pcon:
                request["up"] = None
        elif request["kind"] == "boson":
            if not pcon:
                request["particles"] = None
        elif request["kind"] == "spinless_fermion":
            request["momentum"] = None
            if not pcon:
                request["particles"] = None
        elif request["kind"] == "spinful_fermion" and not pcon:
            request["particles_up"] = None
            request["particles_down"] = None
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
        encoded = [str(int(state)) for state in array]
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
                int(entry.get("representative", entry["state"]))
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
                    int(entry.get("representative", entry["state"]))
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

    def Op_shift_sector(self, other_basis, op_list, v_in):
        if not isinstance(other_basis, _PackedBasis):
            raise TypeError("other_basis must be a QMBED-backed basis")
        input_array = np.asanyarray(v_in)
        if input_array.ndim == 0 or input_array.shape[0] != other_basis.Ns:
            raise ValueError("dimension mismatch")
        result_dtype = np.result_type(input_array.dtype, np.complex128)
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
        return output.reshape((self.Ns, *input_array.shape[1:]))

    def expanded_form(self, static, dynamic):
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
        result = self._model.execute(
            "bra_ket_terms_model",
            terms=[self._term_request(opstr, indx, J)],
            kets=[str(int(ket)) for ket in kets],
        )
        grouped: list[list[dict[str, Any]]] = [[] for _ in range(kets.size)]
        for entry in result["entries"]:
            grouped[int(entry["input"])].append(entry)

        if reduce_output:
            entries = [entry for group in grouped for entry in group]
            matrix_elements = self._values_for_dtype(
                [complex(*entry["value"]) for entry in entries],
                dtype,
            )
            bras = np.asarray([int(entry["bra"]) for entry in entries], dtype=object)
            returned_kets = np.asarray([int(entry["ket"]) for entry in entries], dtype=object)
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
                bras.append(int(group[0]["bra"]))
            else:
                values.append(0.0)
                bras.append(0)
        return (
            self._values_for_dtype(values, dtype),
            np.asarray(bras, dtype=object),
            kets,
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
            symmetries.extend(
                _general_symmetries(
                    {name: (site_map, 0 if block == 1 else 1)},
                    sites=self.N,
                    states_per_site=spin_twice + 1,
                )
            )
        self._request = {
            "kind": "spin",
            "sites": self.N,
            "spin_twice": spin_twice,
            "up": Nup,
            "momentum": None,
            "parity": None,
            "normalization": _spin_normalization(pauli, spin_twice),
            "symmetries": symmetries,
            "reverse": True,
        }


class boson_basis_1d(_PackedBasis):
    def __init__(
        self,
        L: int,
        Nb: int | None = None,
        sps: int | None = None,
        kblock: int | None = None,
        pblock: int | None = None,
        a: int = 1,
        **blocks,
    ):
        _reject_options("boson_basis_1d", blocks)
        self.N = int(L)
        a = int(a)
        if a <= 0 or self.N % a:
            raise ValueError("a must be a positive divisor of L")
        states_per_site = int(
            sps if sps is not None else (Nb + 1 if Nb is not None else 2)
        )
        self._request = {
            "kind": "boson",
            "sites": self.N,
            "particles": Nb,
            "states_per_site": states_per_site,
            "symmetries": _one_dimensional_symmetries(
                self.N,
                states_per_site=states_per_site,
                momentum=kblock,
                parity=pblock,
                translation_step=a,
            ),
            "reverse": True,
        }


class ho_basis(boson_basis_1d):
    def __init__(self, N: int):
        super().__init__(1, sps=int(N) + 1)


class spinless_fermion_basis_1d(_PackedBasis):
    def __init__(
        self,
        L: int,
        Nf: int | None = None,
        kblock: int | None = None,
        pblock: int | None = None,
        a: int = 1,
        **blocks,
    ):
        _reject_options("spinless_fermion_basis_1d", blocks)
        self.N = int(L)
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
        self._request = {
            "kind": "spinless_fermion",
            "sites": self.N,
            "particles": Nf,
            "momentum": None,
            "symmetries": symmetries,
            "reverse": True,
        }


class spinful_fermion_basis_1d(_PackedBasis):
    def __init__(
        self,
        L: int,
        Nf: tuple[int, int] | None = None,
        **blocks,
    ):
        _reject_options("spinful_fermion_basis_1d", blocks)
        self.N = int(L)
        particles_up, particles_down = (None, None) if Nf is None else Nf
        self._request = {
            "kind": "spinful_fermion",
            "sites": self.N,
            "particles_up": particles_up,
            "particles_down": particles_down,
            "reverse": True,
        }


class spin_basis_general(_PackedBasis):
    def __init__(
        self,
        N: int,
        Nup: int | None = None,
        m: float | None = None,
        S: str | int | float = "1/2",
        pauli: bool | int = True,
        make_basis: bool = True,
        block_order=None,
        **blocks,
    ):
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
        if Nup is not None and not isinstance(Nup, (int, np.integer)):
            raise NotImplementedError("unions of spin particle sectors are not implemented")
        self.N = int(N)
        self._request = {
            "kind": "spin",
            "sites": self.N,
            "spin_twice": spin_twice,
            "up": None if Nup is None else int(Nup),
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
        sps: int | None = None,
        make_basis: bool = True,
        block_order=None,
        **blocks,
    ):
        if block_order is not None:
            ordered = {
                name: blocks.pop(name)
                for name in block_order
                if name in blocks
            }
            ordered.update(blocks)
            blocks = ordered
        self.N = int(N)
        states_per_site = int(
            sps if sps is not None else (Nb + 1 if Nb is not None else 2)
        )
        self._request = {
            "kind": "boson",
            "sites": self.N,
            "particles": Nb,
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
        make_basis: bool = True,
        block_order=None,
        **blocks,
    ):
        if block_order is not None:
            ordered = {
                name: blocks.pop(name)
                for name in block_order
                if name in blocks
            }
            ordered.update(blocks)
            blocks = ordered
        if Nf is not None and not isinstance(Nf, (int, np.integer)):
            raise NotImplementedError("unions of fermion particle sectors are not implemented")
        self.N = int(N)
        self._request = {
            "kind": "spinless_fermion",
            "sites": self.N,
            "particles": None if Nf is None else int(Nf),
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
        make_basis: bool = True,
        block_order=None,
        **blocks,
    ):
        if block_order is not None:
            ordered = {
                name: blocks.pop(name)
                for name in block_order
                if name in blocks
            }
            ordered.update(blocks)
            blocks = ordered
        self.N = int(N)
        particles_up, particles_down = (None, None) if Nf is None else Nf
        spatial_symmetries = _general_symmetries(
            blocks,
            sites=self.N,
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
                    + [self.N + destination for destination in destinations],
                }
            )
        self._request = {
            "kind": "spinful_fermion",
            "sites": self.N,
            "particles_up": None if particles_up is None else int(particles_up),
            "particles_down": None if particles_down is None else int(particles_down),
            "symmetries": symmetries,
            "reverse": True,
        }
        self._initialize_general_basis(make_basis=bool(make_basis))


__all__ = [
    "basis_int_to_python_int",
    "boson_basis_1d",
    "boson_basis_general",
    "ho_basis",
    "spin_basis_1d",
    "spin_basis_general",
    "spinful_fermion_basis_1d",
    "spinful_fermion_basis_general",
    "spinless_fermion_basis_1d",
    "spinless_fermion_basis_general",
]


def basis_int_to_python_int(value) -> int:
    return int(value)
