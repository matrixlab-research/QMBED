"""Floquet compatibility adapters backed by QMBED dynamics primitives."""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

from qmbed._ffi import command
from quspin.operators import (
    _as_operator_expression,
    _dense_from_result,
    _matrix_request,
)


def _continuous_unitary(operator, period, *, atol, rtol):
    expression = _as_operator_expression(operator)
    dimension = int(expression.shape[0])
    if expression.shape != (dimension, dimension):
        raise ValueError("continuous Floquet generators must be square")
    initial = np.eye(dimension, dtype=np.complex128)

    def rhs(time, values):
        states = values.reshape((dimension, dimension))
        return (-1.0j * expression.dot(states, time=time)).reshape(-1)

    machine_floor = 100.0 * np.finfo(np.float64).eps
    solution = solve_ivp(
        rhs,
        (0.0, float(period)),
        initial.reshape(-1),
        method="DOP853",
        t_eval=[float(period)],
        atol=max(float(atol), np.finfo(np.float64).eps),
        rtol=max(float(rtol), machine_floor),
        max_step=float(period) / 32.0,
    )
    if not solution.success:
        raise RuntimeError(f"Floquet integration failed: {solution.message}")
    return solution.y[:, -1].reshape((dimension, dimension))


def _analyze_external_unitary(unitary, period):
    return command(
        {
            "operation": "analyze_floquet_unitary",
            "unitary": _matrix_request([unitary]),
            "period": float(period),
            "format": "dense",
        }
    )


def _analyze_steps(operators, times, durations, period):
    steps = [
        {
            "expression": _as_operator_expression(operator)._request(time),
            "duration": float(duration),
        }
        for operator, time, duration in zip(operators, times, durations)
    ]
    return command(
        {
            "operation": "analyze_floquet",
            "steps": steps,
            "period": float(period),
            "format": "dense",
        }
    )


class Floquet:
    """Floquet spectrum and propagator for continuous or stepped protocols."""

    def __init__(
        self,
        evo_dict,
        HF=False,
        UF=False,
        thetaF=False,
        VF=False,
        n_jobs=1,
        force_ONB=False,
    ):
        if not isinstance(evo_dict, dict):
            raise ValueError(f"evo_dict={evo_dict} is not correct format.")
        if not isinstance(n_jobs, int):
            raise TypeError("expecting integer value for optional variable 'n_jobs'!")

        keys = set(evo_dict)
        continuous_keys = [
            {"H", "T"},
            {"H", "T", "atol"},
            {"H", "T", "rtol"},
            {"H", "T", "atol", "rtol"},
        ]
        if keys in continuous_keys:
            self.T = float(evo_dict["T"])
            unitary = _continuous_unitary(
                evo_dict["H"],
                self.T,
                atol=evo_dict.get("atol", 1.0e-12),
                rtol=evo_dict.get("rtol", 1.0e-12),
            )
            result = _analyze_external_unitary(unitary, self.T)
        elif keys in ({"H", "t_list", "dt_list"}, {"H", "t_list", "dt_list", "T"}):
            evaluation_times = np.asarray(evo_dict["t_list"], dtype=np.float64)
            durations = np.asarray(evo_dict["dt_list"], dtype=np.float64)
            if evaluation_times.ndim != 1:
                raise ValueError("t_list must be 1d array.")
            if durations.ndim != 1:
                raise ValueError("dt_list must be 1d array.")
            if evaluation_times.size != durations.size:
                raise ValueError("t_list and dt_list must have the same length")
            self.T = float(evo_dict.get("T", durations.sum()))
            result = _analyze_steps(
                [evo_dict["H"]] * durations.size,
                evaluation_times,
                durations,
                self.T,
            )
        elif keys in ({"H_list", "dt_list"}, {"H_list", "dt_list", "T"}):
            operators = evo_dict["H_list"]
            durations = np.asarray(evo_dict["dt_list"], dtype=np.float64)
            if not isinstance(operators, (list, tuple)):
                raise ValueError("expecting list/tuple for H_list.")
            if durations.ndim != 1:
                raise ValueError("dt_list must be 1d array.")
            if len(operators) != durations.size:
                raise ValueError(
                    "Expecting arguments 'H_list' and 'dt_list' to have the same length!"
                )
            self.T = float(evo_dict.get("T", durations.sum()))
            result = _analyze_steps(
                operators,
                [None] * durations.size,
                durations,
                self.T,
            )
        else:
            raise ValueError(f"evo_dict={evo_dict} is not correct format.")

        self.EF = np.asarray(result["quasienergies"], dtype=np.float64)
        eigenvectors = np.column_stack(
            [
                np.asarray([complex(*value) for value in vector])
                for vector in result["eigenvectors"]
            ]
        )
        if force_ONB:
            eigenvectors, _ = np.linalg.qr(eigenvectors)
        if VF:
            self.VF = eigenvectors
        if thetaF:
            self.thetaF = np.asarray(
                [complex(*value) for value in result["eigenvalues"]],
                dtype=np.complex128,
            )
        if UF:
            self.UF = _dense_from_result(result["unitary"], dtype=np.complex128)
        if HF:
            self.HF = _dense_from_result(
                result["effective_hamiltonian"],
                dtype=np.complex128,
            )


class _ArrayView:
    def __iter__(self):
        return iter(self.vals)

    def __getitem__(self, key):
        return self.vals[key]

    def __str__(self):
        return str(self.vals)

    def __mul__(self, other):
        return self.vals * other

    def __div__(self, other):
        return self.vals / other

    def __truediv__(self, other):
        return self.vals / other

    def __len__(self):
        return len(self.vals)


class _StroboscopicTimes(_ArrayView):
    def __init__(self, values, points_per_cycle, index_offset=0):
        local_indices = np.arange(0, values.size, points_per_cycle, dtype=int)
        self.vals = values.take(local_indices)
        self.inds = local_indices + int(index_offset)


class _PeriodicStage(_ArrayView):
    def __init__(self, cycles, values, period, points_per_cycle, index_offset):
        self.N = int(cycles)
        self.vals = values
        self.i = float(values[0])
        self.f = float(values[-1])
        self.tot = self.N * float(period)
        self.len = int(values.size)
        self.strobo = _StroboscopicTimes(
            values,
            points_per_cycle,
            index_offset=index_offset,
        )


class Floquet_t_vec(_ArrayView):
    """Fixed-resolution periodic time grid with optional ramp stages."""

    def __init__(self, Omega, N_const, len_T=100, N_up=0, N_down=0):
        period = 2.0 * np.pi / float(Omega)
        result = command(
            {
                "operation": "floquet_time_grid",
                "period": period,
                "constant_cycles": int(N_const),
                "points_per_cycle": int(len_T),
                "ramp_up_cycles": int(N_up),
                "ramp_down_cycles": int(N_down),
            }
        )
        self.N = int(result["cycles"])
        self.len_T = int(result["points_per_cycle"])
        self.T = float(result["period"])
        self.vals = np.asarray(result["times"], dtype=np.float64)
        self.len = int(self.vals.size)
        self.shape = self.vals.shape
        self.dt = self.T / self.len_T
        self.i = float(self.vals[0])
        self.f = float(self.vals[-1])
        self.tot = self.f - self.i
        self.strobo = _StroboscopicTimes(self.vals, self.len_T)

        up_cycles = int(N_up)
        constant_cycles = int(N_const)
        down_cycles = int(N_down)
        constant_start = up_cycles * self.len_T
        constant_stop = (up_cycles + constant_cycles) * self.len_T + 1

        if up_cycles > 0:
            up_values = self.vals[:constant_start]
            self._up = _PeriodicStage(
                up_cycles,
                up_values,
                self.T,
                self.len_T,
                0,
            )

        if up_cycles > 0 or down_cycles > 0:
            constant_values = self.vals[constant_start:constant_stop]
            constant_offset = (
                self.up.strobo.inds[-1] + self.len_T
                if up_cycles > 0
                else 0
            )
            self._const = _PeriodicStage(
                constant_cycles,
                constant_values,
                self.T,
                self.len_T,
                constant_offset,
            )

        if down_cycles > 0:
            down_values = self.vals[constant_stop:]
            down_offset = self.const.strobo.inds[-1] + self.len_T
            self._down = _PeriodicStage(
                down_cycles,
                down_values,
                self.T,
                self.len_T,
                down_offset,
            )

    @property
    def up(self):
        return self._up

    @property
    def const(self):
        return self._const

    @property
    def down(self):
        return self._down

    def get_coordinates(self, index):
        time = self.vals[index]
        period_number = np.searchsorted(self.strobo.vals, time + 1.0e-15) - 1
        within_period = np.where(
            np.abs(
                time
                - period_number * self.T
                - self.vals[: self.strobo.inds[1]]
            )
            < 1.0e-15
        )[0][0]
        return int(period_number), int(within_period)


__all__ = ["Floquet", "Floquet_t_vec"]
