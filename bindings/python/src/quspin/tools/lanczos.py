"""Lanczos compatibility API backed by persistent Rust decompositions."""

from __future__ import annotations

import weakref

import numpy as np

from qmbed._ffi import command
from quspin.operators import _as_operator_expression


def _complex_payload(values):
    return [
        [complex(value).real, complex(value).imag]
        for value in np.asarray(values).reshape(-1)
    ]


def _release_lanczos_noexcept(handle):
    try:
        command(
            {
                "operation": "release_lanczos",
                "lanczos_handle": handle,
            }
        )
    except Exception:
        pass


class _LanczosBasis:
    def __init__(self, result):
        self._handle = str(result["handle"])
        self.initial_norm = float(result["initial_norm"])
        self.dimension = int(result["dimension"])
        self.krylov_dimension = int(result["krylov_dimension"])
        self._finalizer = weakref.finalize(
            self,
            _release_lanczos_noexcept,
            self._handle,
        )

    @property
    def handle(self):
        if self._handle is None:
            raise RuntimeError("Lanczos decomposition is closed")
        return self._handle

    def close(self):
        if self._handle is None:
            return
        handle = self._handle
        command(
            {
                "operation": "release_lanczos",
                "lanczos_handle": handle,
            }
        )
        self._handle = None
        self._finalizer.detach()

    def _basis(self):
        result = command(
            {
                "operation": "export_lanczos_basis",
                "lanczos_handle": self.handle,
            }
        )
        return np.asarray(
            [
                [complex(*value) for value in vector]
                for vector in result["vectors"]
            ],
            dtype=np.complex128,
        )

    def __array__(self, dtype=None, copy=None):
        values = self._basis()
        if dtype is not None:
            values = values.astype(dtype, copy=False)
        if copy:
            values = values.copy()
        return values

    def __iter__(self):
        return iter(self._basis())

    def linear_combination(self, coefficients):
        result = command(
            {
                "operation": "lanczos_combine",
                "lanczos_handle": self.handle,
                "coefficients": _complex_payload(coefficients),
            }
        )
        return np.asarray(
            [complex(*value) for value in result["vectors"][0]],
            dtype=np.complex128,
        )


def _decompose(A, v0, m, *, eps=None):
    initial = np.asarray(v0)
    if initial.ndim != 1:
        raise ValueError("v0 must be one-dimensional")
    tolerance = 1.0e-13 if eps is None else float(eps)
    expression = _as_operator_expression(A)
    result = command(
        {
            "operation": "lanczos_operator",
            "expression": expression._request(),
            "initial": _complex_payload(initial),
            "krylov_dimension": int(m),
            "tolerance": tolerance,
        }
    )
    eigenvalues = np.asarray(result["eigenvalues"], dtype=np.float64)
    eigenvectors = np.column_stack(
        [np.asarray(vector, dtype=np.float64) for vector in result["eigenvectors"]]
    )
    return eigenvalues, eigenvectors, _LanczosBasis(result)


def lanczos_full(A, v0, m, full_ortho=False, out=None, eps=None):
    del full_ortho, out
    return _decompose(A, v0, m, eps=eps)


def lanczos_iter(
    A,
    v0,
    m,
    return_vec_iter=True,
    copy_v0=True,
    copy_A=False,
    eps=None,
):
    del copy_v0, copy_A
    decomposition = _decompose(A, v0, m, eps=eps)
    if return_vec_iter:
        return decomposition
    decomposition[2].close()
    return decomposition[:2]


def lin_comb_Q_T(coeff, Q_T, out=None):
    coefficients = np.asarray(coeff)
    if isinstance(Q_T, _LanczosBasis):
        values = Q_T.linear_combination(coefficients)
    elif isinstance(Q_T, np.ndarray):
        values = coefficients @ Q_T
    else:
        vectors = list(Q_T)
        if len(vectors) != coefficients.size:
            raise ValueError("coefficient and Lanczos-vector counts differ")
        values = np.zeros_like(np.asarray(vectors[0]), dtype=np.result_type(coefficients, complex))
        for coefficient, vector in zip(coefficients, vectors):
            values += coefficient * vector
    if out is not None:
        out[...] = values
        return out
    return values


def expm_lanczos(E, V, Q_T, a=1.0, out=None):
    eigenvalues = np.asarray(E, dtype=np.float64)
    eigenvectors = np.asarray(V)
    if eigenvectors.shape != (eigenvalues.size, eigenvalues.size):
        raise ValueError("V must be a square Ritz eigenvector matrix")
    coefficients = eigenvectors @ (
        np.exp(complex(a) * eigenvalues) * eigenvectors[0, :].conj()
    )
    if isinstance(Q_T, _LanczosBasis):
        coefficients *= Q_T.initial_norm
    return lin_comb_Q_T(coefficients, Q_T, out=out)


def _thermal_inputs(E, V, beta):
    eigenvalues = np.asarray(E, dtype=np.float64)
    if eigenvalues.ndim != 1 or eigenvalues.size == 0:
        raise ValueError("E must be a nonempty one-dimensional array")
    eigenvectors = np.asarray(V)
    if eigenvectors.shape != (eigenvalues.size, eigenvalues.size):
        raise ValueError("V must be a square Ritz eigenvector matrix")
    if np.iscomplexobj(eigenvectors):
        if np.any(np.abs(eigenvectors.imag) > 1.0e-14):
            raise ValueError("V must contain real Lanczos Ritz eigenvectors")
        eigenvectors = eigenvectors.real
    eigenvectors = np.asarray(eigenvectors, dtype=np.float64)
    if not np.all(np.isfinite(eigenvalues)) or not np.all(np.isfinite(eigenvectors)):
        raise ValueError("E and V must be finite")
    beta_values = np.asarray(beta, dtype=np.float64)
    if not np.all(np.isfinite(beta_values)):
        raise ValueError("beta must be finite")
    return eigenvalues, eigenvectors, beta_values


def _observable_action_matrix(observable, basis):
    try:
        applied = np.asarray(observable.dot(basis.T))
    except (TypeError, ValueError, NotImplementedError):
        applied = None
    if applied is None or applied.shape != basis.T.shape:
        columns = []
        for vector in basis:
            column = np.asarray(observable.dot(vector))
            if column.shape != (basis.shape[1],):
                raise ValueError("observable dot action returned an incompatible vector")
            columns.append(column)
        applied = np.column_stack(columns)
    return (basis.conj() @ applied).T


def _thermal_projected_observables(method, O_dict, basis):
    projected = []
    for index, observable in enumerate(O_dict.values()):
        if method == "ftlm":
            applied = np.asarray(observable.dot(basis[0]))
            if applied.shape != (basis.shape[1],):
                raise ValueError("observable dot action returned an incompatible vector")
            matrix_elements = basis.conj() @ applied
        else:
            matrix_elements = _observable_action_matrix(observable, basis).reshape(-1)
        projected.append(
            {
                "name": f"observable_{index}",
                "matrix_elements": _complex_payload(matrix_elements),
            }
        )
    return projected


def _thermal_static_iteration(method, O_dict, E, V, Q_T, beta):
    if not hasattr(O_dict, "items") or not O_dict:
        raise ValueError("O_dict must be a nonempty mapping")
    eigenvalues, eigenvectors, beta_values = _thermal_inputs(E, V, beta)
    observable_items = list(O_dict.items())
    request = {
        "operation": "lanczos_thermal",
        "method": method,
        "eigenvalues": eigenvalues.tolist(),
        "eigenvectors": eigenvectors.tolist(),
        "inverse_temperatures": np.atleast_1d(beta_values).reshape(-1).tolist(),
    }

    native_observables = []
    if isinstance(Q_T, _LanczosBasis):
        try:
            for index, (_, observable) in enumerate(observable_items):
                expression = _as_operator_expression(observable)
                native_observables.append(
                    {
                        "name": f"observable_{index}",
                        "expression": expression._request(),
                    }
                )
        except TypeError:
            native_observables = []
    if native_observables:
        request["lanczos_handle"] = Q_T.handle
        request["observables"] = native_observables
    else:
        if isinstance(Q_T, _LanczosBasis):
            basis = Q_T._basis()
        elif isinstance(Q_T, np.ndarray):
            basis = np.asarray(Q_T)
        else:
            basis = np.asarray(list(Q_T))
        if basis.ndim != 2 or basis.shape[0] != eigenvalues.size:
            raise ValueError("Q_T must contain one Lanczos row for each Ritz value")
        request["observables"] = _thermal_projected_observables(
            method,
            dict(observable_items),
            basis,
        )

    result = command(request)
    output = {}
    for index, (key, _observable) in enumerate(observable_items):
        values = np.asarray(
            [complex(*value) for value in result["values"][f"observable_{index}"]]
        )
        output[key] = np.real_if_close(values.reshape(beta_values.shape)).squeeze()
    identity = np.asarray(result["identity"], dtype=np.float64)
    return output, identity.reshape(beta_values.shape).squeeze()


def FTLM_static_iteration(O_dict, E, V, Q_T, beta=0):
    """One finite-temperature Lanczos observable iteration."""

    return _thermal_static_iteration("ftlm", O_dict, E, V, Q_T, beta)


def LTLM_static_iteration(O_dict, E, V, Q_T, beta=0):
    """One low-temperature Lanczos observable iteration."""

    return _thermal_static_iteration("ltlm", O_dict, E, V, Q_T, beta)


__all__ = [
    "expm_lanczos",
    "FTLM_static_iteration",
    "lanczos_full",
    "lanczos_iter",
    "lin_comb_Q_T",
    "LTLM_static_iteration",
]
