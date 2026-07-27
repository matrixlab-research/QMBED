from __future__ import annotations

import ctypes
from functools import lru_cache
import json
import os
from pathlib import Path
import sys
from typing import Any
import weakref

import numpy as np


class QmbedError(RuntimeError):
    """Error reported by the QMBED native library or language boundary."""

    pass


class _QmbedComplex64(ctypes.Structure):
    _fields_ = [("real", ctypes.c_double), ("imaginary", ctypes.c_double)]


class _UserOpResult32(ctypes.Structure):
    _fields_ = [
        ("matrix_element", _QmbedComplex64),
        ("state", ctypes.c_uint32),
    ]


class _UserOpResult64(ctypes.Structure):
    _fields_ = [
        ("matrix_element", _QmbedComplex64),
        ("state", ctypes.c_uint64),
    ]


_DriveCallback = ctypes.CFUNCTYPE(
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_double,
    ctypes.POINTER(_QmbedComplex64),
    ctypes.c_size_t,
)

_UserOp32 = ctypes.CFUNCTYPE(
    ctypes.c_int,
    ctypes.POINTER(_UserOpResult32),
    ctypes.c_int8,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_uint32),
)
_UserNextState32 = ctypes.CFUNCTYPE(
    ctypes.c_uint32,
    ctypes.c_uint32,
    ctypes.c_uint32,
    ctypes.c_uint32,
    ctypes.POINTER(ctypes.c_uint32),
)
_UserPreCheck32 = ctypes.CFUNCTYPE(
    ctypes.c_uint32,
    ctypes.c_uint32,
    ctypes.c_uint32,
    ctypes.POINTER(ctypes.c_uint32),
)
_UserMap32 = ctypes.CFUNCTYPE(
    ctypes.c_uint32,
    ctypes.c_uint32,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_int8),
    ctypes.POINTER(ctypes.c_uint32),
)
_UserOp64 = ctypes.CFUNCTYPE(
    ctypes.c_int,
    ctypes.POINTER(_UserOpResult64),
    ctypes.c_int8,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_uint64),
)
_UserNextState64 = ctypes.CFUNCTYPE(
    ctypes.c_uint64,
    ctypes.c_uint64,
    ctypes.c_uint64,
    ctypes.c_uint64,
    ctypes.POINTER(ctypes.c_uint64),
)
_UserPreCheck64 = ctypes.CFUNCTYPE(
    ctypes.c_uint64,
    ctypes.c_uint64,
    ctypes.c_uint64,
    ctypes.POINTER(ctypes.c_uint64),
)
_UserMap64 = ctypes.CFUNCTYPE(
    ctypes.c_uint64,
    ctypes.c_uint64,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_int8),
    ctypes.POINTER(ctypes.c_uint64),
)


def _library_name() -> str:
    if sys.platform == "darwin":
        return "libqmbed_capi.dylib"
    if os.name == "nt":
        return "qmbed_capi.dll"
    return "libqmbed_capi.so"


def _bundled_library_path() -> Path | None:
    package = Path(__file__).resolve().parent
    if os.name == "nt":
        patterns = ("_native*.pyd", "_native*.dll")
    elif sys.platform == "darwin":
        patterns = ("_native*.so", "_native*.dylib")
    else:
        patterns = ("_native*.so",)
    for pattern in patterns:
        candidates = sorted(package.glob(pattern))
        if candidates:
            return candidates[0]
    return None


def _library_path() -> Path:
    configured = os.environ.get("QMBED_LIBRARY_PATH")
    if configured:
        return Path(configured).expanduser().resolve()
    bundled = _bundled_library_path()
    if bundled is not None:
        return bundled
    repository = Path(__file__).resolve().parents[4]
    for profile in ("release", "debug"):
        candidate = (
            repository
            / "bindings"
            / "capi"
            / "target"
            / profile
            / _library_name()
        )
        if candidate.exists():
            return candidate
    raise QmbedError(
        "QMBED native library not found in the installed package; reinstall a "
        "platform wheel, set QMBED_LIBRARY_PATH, or build bindings/capi with cargo"
    )


@lru_cache(maxsize=None)
def _load_library(path: str) -> ctypes.CDLL:
    library = ctypes.CDLL(path)
    for symbol in ("qmbed_run_json", "qmbed_command_json"):
        function = getattr(library, symbol)
        function.argtypes = [ctypes.c_char_p]
        function.restype = ctypes.c_void_p
    library.qmbed_evolve_model_with_drive_json.argtypes = [
        ctypes.c_char_p,
        _DriveCallback,
        ctypes.c_void_p,
    ]
    library.qmbed_evolve_model_with_drive_json.restype = ctypes.c_void_p
    library.qmbed_register_user_basis_32_json.argtypes = [
        ctypes.c_char_p,
        _UserOp32,
        _UserNextState32,
        _UserPreCheck32,
        ctypes.POINTER(_UserMap32),
        ctypes.c_size_t,
    ]
    library.qmbed_register_user_basis_32_json.restype = ctypes.c_void_p
    library.qmbed_register_user_basis_64_json.argtypes = [
        ctypes.c_char_p,
        _UserOp64,
        _UserNextState64,
        _UserPreCheck64,
        ctypes.POINTER(_UserMap64),
        ctypes.c_size_t,
    ]
    library.qmbed_register_user_basis_64_json.restype = ctypes.c_void_p
    library.qmbed_string_free.argtypes = [ctypes.c_void_p]
    library.qmbed_string_free.restype = None
    return library


def _call_json(
    symbol: str,
    request: dict[str, Any],
    *,
    library_path: str | None = None,
) -> dict[str, Any]:
    library = _load_library(library_path or str(_library_path()))
    function = getattr(library, symbol)
    encoded = json.dumps(request, separators=(",", ":")).encode()
    pointer = function(encoded)
    if not pointer:
        raise QmbedError("QMBED returned a null response")
    try:
        response = json.loads(ctypes.string_at(pointer))
    finally:
        library.qmbed_string_free(pointer)
    if response.get("status") != "ok":
        raise QmbedError(response.get("error", "unknown QMBED error"))
    return response["result"]


def run(request: dict[str, Any]) -> dict[str, Any]:
    return _call_json("qmbed_run_json", request)


def command(request: dict[str, Any]) -> dict[str, Any]:
    return _call_json("qmbed_command_json", request)


def _call_drive_json(
    request: dict[str, Any],
    coefficient_callback,
    *,
    library_path: str,
) -> dict[str, Any]:
    library = _load_library(library_path)
    callback_errors: list[BaseException] = []

    @_DriveCallback
    def bridge(_context, time, coefficients, count):
        try:
            values = list(coefficient_callback(float(time)))
            if len(values) != count:
                raise ValueError(
                    f"drive returned {len(values)} coefficients, expected {count}"
                )
            for index, value in enumerate(values):
                value = complex(value)
                coefficients[index].real = value.real
                coefficients[index].imaginary = value.imag
            return 0
        except BaseException as error:
            callback_errors.append(error)
            return 1

    encoded = json.dumps(request, separators=(",", ":")).encode()
    pointer = library.qmbed_evolve_model_with_drive_json(encoded, bridge, None)
    if not pointer:
        raise QmbedError("QMBED returned a null drive-evolution response")
    try:
        response = json.loads(ctypes.string_at(pointer))
    finally:
        library.qmbed_string_free(pointer)
    if callback_errors:
        raise QmbedError(f"drive callback failed: {callback_errors[0]}") from callback_errors[0]
    if response.get("status") != "ok":
        raise QmbedError(response.get("error", "unknown QMBED drive-evolution error"))
    return response["result"]


def _release_model_noexcept(library_path: str, handle: str) -> None:
    try:
        _call_json(
            "qmbed_command_json",
            {"operation": "release_model", "handle": handle},
            library_path=library_path,
        )
    except Exception:
        # Finalizers may run while Python is tearing down modules. Explicit
        # close() still reports native lifecycle errors to the caller.
        pass


def _release_basis_plan_noexcept(library_path: str, handle: str) -> None:
    try:
        _call_json(
            "qmbed_command_json",
            {"operation": "release_basis_plan", "plan_handle": handle},
            library_path=library_path,
        )
    except Exception:
        pass


def _release_expm_noexcept(library_path: str, handle: str) -> None:
    try:
        _call_json(
            "qmbed_command_json",
            {"operation": "release_expm_action", "expm_handle": handle},
            library_path=library_path,
        )
    except Exception:
        pass


def _release_user_basis_noexcept(library_path: str, handle: str) -> None:
    try:
        _call_json(
            "qmbed_command_json",
            {
                "operation": "release_user_basis",
                "user_basis_handle": handle,
            },
            library_path=library_path,
        )
    except Exception:
        pass


class NativeUserBasis32:
    """Owned registration of QuSpin-compatible 32-bit Numba callbacks."""

    def __init__(
        self,
        request: dict[str, Any],
        *,
        operator_address: int,
        next_state_address: int | None = None,
        pre_check_address: int | None = None,
        map_addresses=(),
    ):
        library_path = str(_library_path())
        library = _load_library(library_path)
        operator = _UserOp32(int(operator_address))
        next_state = (
            _UserNextState32(int(next_state_address))
            if next_state_address is not None
            else _UserNextState32()
        )
        pre_check = (
            _UserPreCheck32(int(pre_check_address))
            if pre_check_address is not None
            else _UserPreCheck32()
        )
        maps = tuple(_UserMap32(int(address)) for address in map_addresses)
        map_array = (_UserMap32 * len(maps))(*maps) if maps else None
        encoded = json.dumps(request, separators=(",", ":")).encode()
        pointer = library.qmbed_register_user_basis_32_json(
            encoded,
            operator,
            next_state,
            pre_check,
            map_array,
            len(maps),
        )
        if not pointer:
            raise QmbedError("QMBED returned a null user-basis response")
        try:
            response = json.loads(ctypes.string_at(pointer))
        finally:
            library.qmbed_string_free(pointer)
        if response.get("status") != "ok":
            raise QmbedError(response.get("error", "unknown QMBED user-basis error"))
        result = response["result"]
        self._library_path = library_path
        self._handle: str | None = str(result["handle"])
        self.dimension = int(result["dimension"])
        # Keep both the direct C wrappers and the original Numba objects (set
        # by the Python compatibility class) alive for the registry lifetime.
        self._callback_wrappers = (operator, next_state, pre_check, maps, map_array)
        self._finalizer = weakref.finalize(
            self,
            _release_user_basis_noexcept,
            library_path,
            self._handle,
        )

    @property
    def handle(self) -> str:
        if self._handle is None:
            raise QmbedError("QMBED user basis is closed")
        return self._handle

    def keep_numba_callbacks_alive(self, callbacks) -> None:
        self._numba_callbacks = tuple(callbacks)

    def close(self) -> None:
        if self._handle is None:
            return
        _call_json(
            "qmbed_command_json",
            {
                "operation": "release_user_basis",
                "user_basis_handle": self._handle,
            },
            library_path=self._library_path,
        )
        self._handle = None
        self._finalizer.detach()


class NativeUserBasis64:
    """Owned registration of QuSpin-compatible 64-bit Numba callbacks."""

    def __init__(
        self,
        request: dict[str, Any],
        *,
        operator_address: int,
        next_state_address: int | None = None,
        pre_check_address: int | None = None,
        map_addresses=(),
    ):
        library_path = str(_library_path())
        library = _load_library(library_path)
        operator = _UserOp64(int(operator_address))
        next_state = (
            _UserNextState64(int(next_state_address))
            if next_state_address is not None
            else _UserNextState64()
        )
        pre_check = (
            _UserPreCheck64(int(pre_check_address))
            if pre_check_address is not None
            else _UserPreCheck64()
        )
        maps = tuple(_UserMap64(int(address)) for address in map_addresses)
        map_array = (_UserMap64 * len(maps))(*maps) if maps else None
        encoded = json.dumps(request, separators=(",", ":")).encode()
        pointer = library.qmbed_register_user_basis_64_json(
            encoded,
            operator,
            next_state,
            pre_check,
            map_array,
            len(maps),
        )
        if not pointer:
            raise QmbedError("QMBED returned a null user-basis response")
        try:
            response = json.loads(ctypes.string_at(pointer))
        finally:
            library.qmbed_string_free(pointer)
        if response.get("status") != "ok":
            raise QmbedError(response.get("error", "unknown QMBED user-basis error"))
        result = response["result"]
        self._library_path = library_path
        self._handle: str | None = str(result["handle"])
        self.dimension = int(result["dimension"])
        self._callback_wrappers = (operator, next_state, pre_check, maps, map_array)
        self._finalizer = weakref.finalize(
            self,
            _release_user_basis_noexcept,
            library_path,
            self._handle,
        )

    @property
    def handle(self) -> str:
        if self._handle is None:
            raise QmbedError("QMBED user basis is closed")
        return self._handle

    def keep_numba_callbacks_alive(self, callbacks) -> None:
        self._numba_callbacks = tuple(callbacks)

    def close(self) -> None:
        if self._handle is None:
            return
        _call_json(
            "qmbed_command_json",
            {
                "operation": "release_user_basis",
                "user_basis_handle": self._handle,
            },
            library_path=self._library_path,
        )
        self._handle = None
        self._finalizer.detach()


class NativeModel:
    """Owned reference to one persistent Rust ED model."""

    def __init__(self, request: dict[str, Any]):
        library_path = str(_library_path())
        result = _call_json(
            "qmbed_command_json",
            {"operation": "create_model", **request},
            library_path=library_path,
        )
        self._attach(library_path, result)

    def _attach(self, library_path: str, result: dict[str, Any]) -> None:
        self._library_path = library_path
        self._handle: str | None = str(result["handle"])
        self.dimension = int(result["dimension"])
        self._finalizer = weakref.finalize(
            self,
            _release_model_noexcept,
            self._library_path,
            self._handle,
        )

    @classmethod
    def _from_registered(
        cls,
        library_path: str,
        result: dict[str, Any],
    ) -> NativeModel:
        model = cls.__new__(cls)
        model._attach(library_path, result)
        return model

    @property
    def handle(self) -> str:
        if self._handle is None:
            raise QmbedError("QMBED model is closed")
        return self._handle

    @property
    def closed(self) -> bool:
        return self._handle is None

    def execute(self, operation: str, **options: Any) -> dict[str, Any]:
        return _call_json(
            "qmbed_command_json",
            {
                "operation": operation,
                "handle": self.handle,
                **options,
            },
            library_path=self._library_path,
        )

    def evolve_with_drive(
        self,
        *,
        component_names,
        vectors,
        initial_time,
        evolution,
        coefficient_callback,
    ) -> dict[str, Any]:
        return _call_drive_json(
            {
                "handle": self.handle,
                "component_names": list(component_names),
                "vectors": vectors,
                "initial_time": float(initial_time),
                "evolution": evolution,
            },
            coefficient_callback,
            library_path=self._library_path,
        )

    def projected(self, projector: dict[str, Any]) -> NativeOperatorModel:
        result = self.execute(
            "project_operator_model",
            projector=projector,
        )
        return NativeOperatorModel._from_registered(self._library_path, result)

    def save_operator_archive(
        self,
        path: str | os.PathLike[str],
        *,
        formats: dict[str, str] | None = None,
        metadata: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        return self.execute(
            "save_operator_archive",
            path=os.fspath(path),
            formats={} if formats is None else dict(formats),
            metadata={} if metadata is None else dict(metadata),
        )

    def close(self) -> None:
        if self._handle is None:
            return
        _call_json(
            "qmbed_command_json",
            {"operation": "release_model", "handle": self._handle},
            library_path=self._library_path,
        )
        self._handle = None
        self._finalizer.detach()

    def __enter__(self) -> NativeModel:
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()


class NativeOperatorModel(NativeModel):
    """Owned reference to one basis-independent Rust operator family."""

    def __init__(self, request: dict[str, Any]):
        library_path = str(_library_path())
        result = _call_json(
            "qmbed_command_json",
            {"operation": "create_operator_model", **request},
            library_path=library_path,
        )
        self._attach(library_path, result)

    @classmethod
    def from_expression(
        cls,
        expression: dict[str, Any],
        *,
        format: str = "csc",
    ) -> NativeOperatorModel:
        library_path = str(_library_path())
        result = _call_json(
            "qmbed_command_json",
            {
                "operation": "create_operator_expression_model",
                "expression": expression,
                "format": format,
            },
            library_path=library_path,
        )
        return cls._from_registered(library_path, result)

    @classmethod
    def direct_sum(
        cls,
        models,
        *,
        format: str = "csc",
    ) -> NativeOperatorModel:
        models = tuple(models)
        if not models:
            raise ValueError("a block model requires at least one source model")
        library_path = models[0]._library_path
        if any(model._library_path != library_path for model in models):
            raise ValueError("all block models must use the same QMBED library")
        result = _call_json(
            "qmbed_command_json",
            {
                "operation": "create_block_model",
                "handles": [model.handle for model in models],
                "format": format,
            },
            library_path=library_path,
        )
        return cls._from_registered(library_path, result)

    @classmethod
    def projected_blocks(
        cls,
        blocks,
        *,
        tolerance: float = 1.0e-10,
        format: str = "csc",
    ) -> NativeOperatorModel:
        blocks = tuple(blocks)
        if not blocks:
            raise ValueError("a projected block model requires at least one block")
        library_path = blocks[0][0]._library_path
        if any(model._library_path != library_path for model, _ in blocks):
            raise ValueError("all projected blocks must use the same QMBED library")
        result = _call_json(
            "qmbed_command_json",
            {
                "operation": "create_projected_block_model",
                "blocks": [
                    {"handle": model.handle, "projector": projector}
                    for model, projector in blocks
                ],
                "tolerance": float(tolerance),
                "format": format,
            },
            library_path=library_path,
        )
        return cls._from_registered(library_path, result)

    @classmethod
    def load_operator_archive(
        cls,
        path: str | os.PathLike[str],
    ) -> NativeOperatorModel:
        library_path = str(_library_path())
        result = _call_json(
            "qmbed_command_json",
            {
                "operation": "load_operator_archive",
                "path": os.fspath(path),
            },
            library_path=library_path,
        )
        model = cls._from_registered(library_path, result)
        model.archive_components = tuple(result["components"])
        model.archive_metadata = dict(result["metadata"])
        return model


class NativeExpmAction:
    """Owned reusable Rust exponential-action plan."""

    def __init__(
        self,
        model: NativeModel,
        coefficient: complex,
        *,
        max_degree: int = 55,
        tolerance: float = 0.5 * np.finfo(np.float64).eps,
        max_substeps: int = 10_000,
        parameters: dict[str, complex] | None = None,
    ):
        self._library_path = model._library_path
        result = _call_json(
            "qmbed_command_json",
            {
                "operation": "create_expm_action",
                "handle": model.handle,
                "parameters": {
                    str(name): [complex(value).real, complex(value).imag]
                    for name, value in (parameters or {}).items()
                },
                "coefficient": [complex(coefficient).real, complex(coefficient).imag],
                "max_degree": int(max_degree),
                "tolerance": float(tolerance),
                "max_substeps": int(max_substeps),
            },
            library_path=self._library_path,
        )
        self._handle: str | None = str(result["handle"])
        self.dimension = int(result["dimension"])
        self._model = model
        self._finalizer = weakref.finalize(
            self,
            _release_expm_noexcept,
            self._library_path,
            self._handle,
        )

    @property
    def handle(self) -> str:
        if self._handle is None:
            raise QmbedError("QMBED exponential-action plan is closed")
        return self._handle

    @property
    def closed(self) -> bool:
        return self._handle is None

    def apply(self, vectors, *, threads: int | None = None) -> dict[str, Any]:
        request: dict[str, Any] = {
            "operation": "apply_expm_action",
            "expm_handle": self.handle,
            "vectors": vectors,
        }
        if threads is not None:
            request["threads"] = int(threads)
        return _call_json(
            "qmbed_command_json",
            request,
            library_path=self._library_path,
        )

    def close(self) -> None:
        if self._handle is None:
            return
        _call_json(
            "qmbed_command_json",
            {
                "operation": "release_expm_action",
                "expm_handle": self._handle,
            },
            library_path=self._library_path,
        )
        self._handle = None
        self._finalizer.detach()

    def __enter__(self) -> NativeExpmAction:
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()


class NativeBasisPlan:
    """Compiled symmetry action whose basis states have not been enumerated."""

    def __init__(self, request: dict[str, Any]):
        self._library_path = str(_library_path())
        result = _call_json(
            "qmbed_command_json",
            {"operation": "create_basis_plan", **request},
            library_path=self._library_path,
        )
        self._handle: str | None = str(result["handle"])
        self._finalizer = weakref.finalize(
            self,
            _release_basis_plan_noexcept,
            self._library_path,
            self._handle,
        )

    @property
    def handle(self) -> str:
        if self._handle is None:
            raise QmbedError("QMBED basis plan is closed")
        return self._handle

    @property
    def closed(self) -> bool:
        return self._handle is None

    def execute(self, operation: str, **options: Any) -> dict[str, Any]:
        return _call_json(
            "qmbed_command_json",
            {
                "operation": operation,
                "plan_handle": self.handle,
                **options,
            },
            library_path=self._library_path,
        )

    def materialize(self) -> NativeModel:
        result = self.execute("materialize_basis_plan")
        return NativeModel._from_registered(self._library_path, result)

    def close(self) -> None:
        if self._handle is None:
            return
        _call_json(
            "qmbed_command_json",
            {
                "operation": "release_basis_plan",
                "plan_handle": self._handle,
            },
            library_path=self._library_path,
        )
        self._handle = None
        self._finalizer.detach()

    def __enter__(self) -> NativeBasisPlan:
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()
