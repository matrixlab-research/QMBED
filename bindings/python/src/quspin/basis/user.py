from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from numba import types

from qmbed._ffi import NativeUserBasis32, NativeUserBasis64
from quspin.basis import _PackedBasis


map_sig_32 = types.uint32(
    types.uint32,
    types.intc,
    types.CPointer(types.int8),
    types.CPointer(types.uint32),
)
map_sig_64 = types.uint64(
    types.uint64,
    types.intc,
    types.CPointer(types.int8),
    types.CPointer(types.uint64),
)

next_state_sig_32 = types.uint32(
    types.uint32,
    types.uint32,
    types.uint32,
    types.CPointer(types.uint32),
)
next_state_sig_64 = types.uint64(
    types.uint64,
    types.uint64,
    types.uint64,
    types.CPointer(types.uint64),
)

pre_check_state_sig_32 = types.uint32(
    types.uint32,
    types.uint32,
    types.CPointer(types.uint32),
)
pre_check_state_sig_64 = types.uint64(
    types.uint64,
    types.uint64,
    types.CPointer(types.uint64),
)

op_results_32 = types.Record.make_c_struct(
    [
        ("matrix_ele", types.complex128),
        ("state", types.uint32),
    ]
)
op_results_64 = types.Record.make_c_struct(
    [
        ("matrix_ele", types.complex128),
        ("state", types.uint64),
    ]
)
op_sig_32 = types.intc(
    types.CPointer(op_results_32),
    types.char,
    types.intc,
    types.intc,
    types.CPointer(types.uint32),
)
op_sig_64 = types.intc(
    types.CPointer(op_results_64),
    types.char,
    types.intc,
    types.intc,
    types.CPointer(types.uint64),
)

count_particles_sig_32 = types.void(
    types.uint32,
    types.CPointer(types.intc),
    types.CPointer(types.intc),
)
count_particles_sig_64 = types.void(
    types.uint64,
    types.CPointer(types.intc),
    types.CPointer(types.intc),
)


def _require_cfunc(function, signature, name: str):
    if not hasattr(function, "address") or not hasattr(function, "_sig"):
        raise ValueError(f"{name} must be a numba.cfunc object")
    if function._sig != signature:
        raise ValueError(f"{name} does not have the required QuSpin signature")
    return function


def _uint_arguments(value, dtype, name: str) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise ValueError(f"{name} must be a C-contiguous numpy array")
    if value.dtype != np.dtype(dtype) or not value.flags["CARRAY"]:
        raise ValueError(
            f"{name} must be a C-contiguous numpy array with dtype {np.dtype(dtype)}"
        )
    return value


def _particle_sectors(value, n_sectors) -> list:
    if n_sectors is None:
        return [value]
    n_sectors = int(n_sectors)
    if n_sectors <= 0:
        raise ValueError("n_sectors must be positive")
    if n_sectors == 1:
        if isinstance(value, list):
            return value
        return [value]
    if isinstance(value, tuple) and len(value) == n_sectors:
        return [value]
    try:
        sectors = list(value)
    except TypeError as error:
        raise ValueError("Np does not match n_sectors") from error
    if not all(isinstance(sector, tuple) and len(sector) == n_sectors for sector in sectors):
        raise ValueError("each particle sector must match n_sectors")
    return sectors


def _normalize_noncommuting_bits(N: int, groups) -> list[tuple[np.ndarray, complex]]:
    normalized = []
    for sites, phase in groups:
        sites = np.asarray(sites, dtype=np.int64).reshape(-1)
        if len(set(int(site) for site in sites)) != sites.size:
            raise ValueError("noncommuting bit groups cannot repeat a site")
        if np.any(sites < 0) or np.any(sites >= N):
            raise ValueError("noncommuting bit group contains an invalid site")
        phase = complex(phase)
        if not np.isfinite(phase.real) or not np.isfinite(phase.imag):
            raise ValueError("noncommuting exchange phase must be finite")
        if not np.isclose(abs(phase), 1.0, rtol=0.0, atol=1.0e-12):
            raise ValueError("noncommuting exchange phase must have unit magnitude")
        if phase == 1:
            continue
        normalized.append((sites, phase))
    return normalized


class user_basis(_PackedBasis):
    """QuSpin-compatible callback basis backed by QMBED's Rust basis engine."""

    def __init__(
        self,
        basis_dtype,
        N,
        op_dict,
        sps=2,
        pcon_dict=None,
        pre_check_state=None,
        allowed_ops=None,
        parallel=False,
        Ns_block_est=None,
        _make_basis=True,
        block_order=None,
        noncommuting_bits=(),
        _Np=None,
        **blocks,
    ):
        del parallel, Ns_block_est
        if _Np is not None:
            raise ValueError("_Np is reserved for QuSpin internals")
        self.N = int(N)
        self.sps = int(sps)
        if self.N <= 0:
            raise ValueError("N must be positive")
        if self.sps < 2:
            raise ValueError("sps must be at least two")
        self._basis_dtype = np.dtype(basis_dtype)
        if self._basis_dtype not in {np.dtype(np.uint32), np.dtype(np.uint64)}:
            raise ValueError("basis_dtype must be numpy.uint32 or numpy.uint64")
        use_32bit = self._basis_dtype == np.dtype(np.uint32)
        uint_dtype = np.uint32 if use_32bit else np.uint64
        op_signature = op_sig_32 if use_32bit else op_sig_64
        next_signature = next_state_sig_32 if use_32bit else next_state_sig_64
        pre_check_signature = (
            pre_check_state_sig_32 if use_32bit else pre_check_state_sig_64
        )
        map_signature = map_sig_32 if use_32bit else map_sig_64
        registration_type = NativeUserBasis32 if use_32bit else NativeUserBasis64

        if not isinstance(op_dict, dict) or set(op_dict) != {"op", "op_args"}:
            raise ValueError("op_dict must contain exactly 'op' and 'op_args'")
        operator = _require_cfunc(op_dict["op"], op_signature, "op")
        operator_arguments = _uint_arguments(
            op_dict["op_args"], uint_dtype, "op_args"
        )
        if allowed_ops is None:
            raise ValueError("allowed_ops must be supplied")
        allowed_ops = set(str(symbol) for symbol in allowed_ops)
        if not allowed_ops or any(len(symbol) != 1 for symbol in allowed_ops):
            raise ValueError("allowed_ops must contain one-character operator names")

        callbacks = [operator]
        next_state = None
        state_segments = []
        next_state_arguments = np.asarray([], dtype=uint_dtype)
        if pcon_dict is not None:
            if not isinstance(pcon_dict, dict):
                raise ValueError("pcon_dict must be a dictionary")
            required = {
                "Np",
                "next_state",
                "next_state_args",
                "get_Ns_pcon",
                "get_s0_pcon",
            }
            missing = required - set(pcon_dict)
            if missing:
                raise ValueError(f"pcon_dict is missing {sorted(missing)}")
            next_state = _require_cfunc(
                pcon_dict["next_state"], next_signature, "next_state"
            )
            callbacks.append(next_state)
            next_state_arguments = _uint_arguments(
                pcon_dict["next_state_args"],
                uint_dtype,
                "next_state_args",
            )
            sectors = _particle_sectors(
                pcon_dict["Np"], pcon_dict.get("n_sectors")
            )
            for sector in sectors:
                count = int(pcon_dict["get_Ns_pcon"](self.N, sector))
                start = int(pcon_dict["get_s0_pcon"](self.N, sector))
                if count < 0:
                    raise ValueError("get_Ns_pcon returned a negative size")
                state_segments.append({"start": start, "count": count})

        pre_check = None
        pre_check_arguments = np.asarray([], dtype=uint_dtype)
        if pre_check_state is not None:
            if isinstance(pre_check_state, tuple):
                if len(pre_check_state) != 2:
                    raise ValueError("pre_check_state tuple must be (cfunc, args)")
                pre_check, pre_check_arguments = pre_check_state
                pre_check_arguments = _uint_arguments(
                    pre_check_arguments,
                    uint_dtype,
                    "pre_check_state args",
                )
            else:
                pre_check = pre_check_state
            pre_check = _require_cfunc(
                pre_check, pre_check_signature, "pre_check_state"
            )
            callbacks.append(pre_check)

        if block_order is None:
            block_items = sorted(
                blocks.items(),
                key=lambda item: int(item[1][1]),
                reverse=True,
            )
        else:
            names = list(block_order)
            if set(names) != set(blocks):
                raise ValueError("block_order must list every user symmetry exactly once")
            block_items = [(name, blocks[name]) for name in names]
        symmetry_callbacks = []
        symmetries = []
        self._blocks = {}
        for name, block in block_items:
            if not isinstance(block, tuple) or len(block) != 4:
                raise ValueError(
                    f"{name} must be (map_func, period, quantum_number, args)"
                )
            map_function, period, sector, arguments = block
            map_function = _require_cfunc(
                map_function, map_signature, f"{name} map"
            )
            arguments = _uint_arguments(
                arguments, uint_dtype, f"{name} map args"
            )
            period = int(period)
            if period <= 0:
                raise ValueError(f"{name} period must be positive")
            symmetry_callbacks.append(map_function)
            callbacks.append(map_function)
            symmetries.append(
                {
                    "period": period,
                    "sector": int(sector),
                    "arguments": [int(value) for value in arguments],
                }
            )
            self._blocks[name] = (-1) ** int(sector) if period == 2 else int(sector)

        self._noncommuting_bits = _normalize_noncommuting_bits(
            self.N, noncommuting_bits
        )
        registration = registration_type(
            {
                "sites": self.N,
                "states_per_site": self.sps,
                "allowed_ops": "".join(sorted(allowed_ops)),
                "state_segments": state_segments,
                "operator_arguments": [
                    int(value) for value in operator_arguments
                ],
                "next_state_arguments": [
                    int(value) for value in next_state_arguments
                ],
                "pre_check_arguments": [
                    int(value) for value in pre_check_arguments
                ],
                "symmetries": symmetries,
                "reverse": True,
            },
            operator_address=operator.address,
            next_state_address=None if next_state is None else next_state.address,
            pre_check_address=None if pre_check is None else pre_check.address,
            map_addresses=[function.address for function in symmetry_callbacks],
        )
        registration.keep_numba_callbacks_alive(callbacks)
        self._native_user_basis = registration
        self._request = {
            "kind": "user",
            "handle": registration.handle,
            "view": "primary",
        }
        self._made_basis = bool(_make_basis)
        if not self._made_basis:
            self._deferred_basis = True

    @property
    def dtype(self) -> np.dtype:
        return self._basis_dtype

    @property
    def blocks(self):
        return dict(self._blocks)

    @property
    def _site_permutation(self) -> list[int]:
        # QuSpin user callbacks receive public site indices and implement
        # their own integer-encoding convention.
        return list(range(self.N))

    def _parent_request(self, *, pcon: bool) -> dict:
        return {
            "kind": "user",
            "handle": self._native_user_basis.handle,
            "view": "constrained" if pcon else "full",
        }

    def int_to_state(self, state, bracket_notation: bool = True):
        value = int(state)
        digits = []
        for _ in range(self.N):
            digits.append(value % self.sps)
            value //= self.sps
        if value:
            raise ValueError("state exceeds the basis encoding width")
        body = " ".join(str(digit) for digit in reversed(digits))
        return f"|{body}>" if bracket_notation else body

    def __repr__(self) -> str:
        return (
            f"<user_basis N={self.N}, sps={self.sps}, Ns={self.Ns}, "
            f"dtype={self.dtype}>"
        )


__all__ = [
    "count_particles_sig_32",
    "count_particles_sig_64",
    "map_sig_32",
    "map_sig_64",
    "next_state_sig_32",
    "next_state_sig_64",
    "op_results_32",
    "op_results_64",
    "op_sig_32",
    "op_sig_64",
    "pre_check_state_sig_32",
    "pre_check_state_sig_64",
    "user_basis",
]
