from __future__ import annotations

import ctypes
from functools import lru_cache
import json
import os
from pathlib import Path
import sys
from typing import Any
import weakref


class QmbedError(RuntimeError):
    pass


def _library_name() -> str:
    if sys.platform == "darwin":
        return "libqmbed_capi.dylib"
    if os.name == "nt":
        return "qmbed_capi.dll"
    return "libqmbed_capi.so"


def _library_path() -> Path:
    configured = os.environ.get("QMBED_LIBRARY_PATH")
    if configured:
        return Path(configured).expanduser().resolve()
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
        "QMBED native library not found; set QMBED_LIBRARY_PATH or build "
        "bindings/capi with cargo"
    )


@lru_cache(maxsize=None)
def _load_library(path: str) -> ctypes.CDLL:
    library = ctypes.CDLL(path)
    for symbol in ("qmbed_run_json", "qmbed_command_json"):
        function = getattr(library, symbol)
        function.argtypes = [ctypes.c_char_p]
        function.restype = ctypes.c_void_p
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
