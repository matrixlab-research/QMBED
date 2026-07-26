"""General evolution helpers compatible with QuSpin's public tools API."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.integrate import solve_ivp

from qmbed._ffi import NativeExpmAction, NativeOperatorModel


def _complex_values(values):
    return [
        [complex(value).real, complex(value).imag]
        for value in np.asarray(values).reshape(-1)
    ]


class expm_multiply_parallel:
    """Reusable Rust-backed action of ``exp(a * A)`` on vectors or batches."""

    def __init__(
        self,
        A,
        a=1.0,
        dtype=None,
        copy=False,
        *,
        max_degree=55,
        tol=0.5 * np.finfo(np.float64).eps,
        max_substeps=10_000,
        n_jobs=None,
    ):
        from quspin.operators import _matrix_request

        if len(getattr(A, "shape", ())) != 2 or A.shape[0] != A.shape[1]:
            raise ValueError("A must be a square matrix")
        coefficient = complex(a)
        if not np.isfinite(coefficient.real) or not np.isfinite(coefficient.imag):
            raise ValueError("a must be finite")
        matrix_dtype = getattr(A, "dtype", np.asarray(A).dtype)
        self.dtype = np.dtype(
            dtype
            if dtype is not None
            else np.result_type(matrix_dtype, a, np.float64)
        )
        self.shape = tuple(int(value) for value in A.shape)
        self.A = sp.csr_matrix(A, copy=bool(copy))
        self.a = a
        self.n_jobs = None if n_jobs is None else int(n_jobs)
        self._model = NativeOperatorModel(
            {"static_operator": _matrix_request([self.A])}
        )
        self._plan_options = {
            "max_degree": int(max_degree),
            "tolerance": float(tol),
            "max_substeps": int(max_substeps),
        }
        self._plan = NativeExpmAction(
            self._model,
            coefficient,
            **self._plan_options,
        )

    @property
    def closed(self):
        return self._plan.closed

    def dot(self, v, out=None):
        values = np.asarray(v)
        if values.ndim not in {1, 2} or values.shape[0] != self.shape[1]:
            raise ValueError("v must be a vector or column batch matching A")
        columns = (
            [values]
            if values.ndim == 1
            else [values[:, column] for column in range(values.shape[1])]
        )
        result = self._plan.apply(
            [_complex_values(column) for column in columns],
            threads=self.n_jobs,
        )
        output = np.column_stack(
            [
                np.asarray(
                    [complex(*entry) for entry in column],
                    dtype=np.complex128,
                )
                for column in result["vectors"]
            ]
        )
        target_dtype = np.dtype(np.result_type(self.dtype, values.dtype))
        if target_dtype.kind != "c":
            if np.any(np.abs(output.imag) > 1.0e-12):
                target_dtype = np.dtype(np.result_type(target_dtype, np.complex128))
            else:
                output = output.real
        output = output.astype(target_dtype, copy=False)
        if values.ndim == 1:
            output = output[:, 0]
        if out is None:
            return output
        if out.shape != output.shape:
            raise ValueError("out has the wrong shape")
        if not np.can_cast(output.dtype, out.dtype, casting="same_kind"):
            raise TypeError("out has an incompatible dtype")
        out[...] = output
        return out

    def set_a(self, a):
        coefficient = complex(a)
        if not np.isfinite(coefficient.real) or not np.isfinite(coefficient.imag):
            raise ValueError("a must be finite")
        replacement = NativeExpmAction(
            self._model,
            coefficient,
            **self._plan_options,
        )
        self._plan.close()
        self._plan = replacement
        self.a = a

    def close(self):
        self._plan.close()
        self._model.close()

    def __enter__(self):
        return self

    def __exit__(self, *_exc_info):
        self.close()


def _time_values(times):
    if np.ndim(times) == 0:
        return np.asarray([times], dtype=np.float64), True
    return np.asarray(list(times), dtype=np.float64), False


def evolve(
    v0,
    t0,
    times,
    f,
    solver_name="dop853",
    real=False,
    stack_state=False,
    verbose=False,
    imag_time=False,
    iterate=False,
    f_params=(),
    **solver_args,
):
    del verbose
    requested_times, scalar_time = _time_values(times)
    if (
        requested_times.ndim != 1
        or requested_times.size == 0
        or np.any(np.diff(requested_times) < 0.0)
        or requested_times[0] < float(t0)
    ):
        raise ValueError("times must be nonempty, ordered, and no earlier than t0")

    initial = np.asarray(v0)
    complex_stacked = bool(real and stack_state and np.iscomplexobj(initial))
    if complex_stacked:
        solver_initial = np.concatenate((initial.real.reshape(-1), initial.imag.reshape(-1)))
    else:
        solver_initial = initial.reshape(-1)
    solver_initial = np.asarray(
        solver_initial,
        dtype=np.float64 if real else np.result_type(initial, np.complex64),
    )

    def rhs(time, values):
        derivative = np.asarray(f(time, values, *f_params))
        return derivative.reshape(-1)

    if requested_times[-1] == float(t0):
        raw_states = [solver_initial.copy() for _ in requested_times]
    else:
        method = {"dop853": "DOP853", "dopri5": "RK45"}.get(
            solver_name,
            solver_name,
        )
        atol = float(solver_args.pop("atol", 1.0e-9))
        rtol = float(solver_args.pop("rtol", 1.0e-9))
        max_step = float(solver_args.pop("max_step", np.inf))
        solver_args.pop("nsteps", None)
        if solver_args:
            names = ", ".join(sorted(solver_args))
            raise TypeError(f"unsupported evolution options: {names}")
        solution = solve_ivp(
            rhs,
            (float(t0), float(requested_times[-1])),
            solver_initial,
            method=method,
            t_eval=requested_times,
            atol=atol,
            rtol=rtol,
            max_step=max_step,
        )
        if not solution.success:
            raise RuntimeError(f"failed general evolution: {solution.message}")
        raw_states = [solution.y[:, index] for index in range(requested_times.size)]

    states = []
    for values in raw_states:
        if complex_stacked:
            size = initial.size
            state = (values[:size] + 1.0j * values[size:]).reshape(initial.shape)
        else:
            state = values.reshape(initial.shape)
        if imag_time:
            norm = np.linalg.norm(state)
            if norm == 0.0:
                raise RuntimeError("imaginary-time evolution produced a zero state")
            state = state / norm
        states.append(state)
    if iterate:
        return iter(states)
    if scalar_time:
        return states[0]
    return np.stack(states, axis=-1)


def ED_state_vs_time(psi, E, V, times, iterate=False):
    values, scalar_time = _time_values(times)
    energies = np.asarray(E, dtype=np.float64)
    eigenvectors = np.asarray(V)
    initial = np.asarray(psi)
    if eigenvectors.shape != (energies.size, energies.size):
        raise ValueError("V must be a square eigenvector matrix")

    states = []
    if initial.ndim == 1:
        coefficients = eigenvectors.conj().T @ initial
        for time in values:
            states.append(
                eigenvectors @ (np.exp(-1.0j * energies * time) * coefficients)
            )
    elif initial.ndim == 2 and initial.shape == eigenvectors.shape:
        eigenbasis_density = eigenvectors.conj().T @ initial @ eigenvectors
        for time in values:
            phases = np.exp(-1.0j * energies * time)
            states.append(
                eigenvectors
                @ (phases[:, None] * eigenbasis_density * phases.conj()[None, :])
                @ eigenvectors.conj().T
            )
    else:
        raise ValueError("psi must be a state vector or density matrix")

    if iterate:
        return iter(states)
    if scalar_time:
        return states[0]
    return np.stack(states, axis=-1)


ExpmMultiplyParallel = expm_multiply_parallel


__all__ = [
    "ED_state_vs_time",
    "evolve",
    "ExpmMultiplyParallel",
    "expm_multiply_parallel",
]
