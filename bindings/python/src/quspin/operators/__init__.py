from __future__ import annotations

from copy import copy as shallow_copy
from typing import Any

import numpy as np
import scipy.sparse as sp
from scipy.integrate import solve_ivp

from qmbed._ffi import NativeModel, NativeOperatorModel, command
from qmbed.compat.quspin import terms_from_static


_TARGETS = {
    "SA": "smallest_algebraic",
    "LA": "largest_algebraic",
    "SM": "smallest_magnitude",
    "LM": "largest_magnitude",
    "BE": "both_ends",
}

def _matrix_request(matrices) -> dict[str, Any]:
    shape = None
    entries = []
    for matrix in matrices:
        if sp.issparse(matrix):
            coo = matrix.tocoo(copy=False)
            matrix_shape = tuple(int(value) for value in coo.shape)
            rows, columns, values = coo.row, coo.col, coo.data
        else:
            array = np.asanyarray(matrix)
            if array.ndim != 2:
                raise TypeError("direct operator inputs must be two-dimensional matrices")
            matrix_shape = tuple(int(value) for value in array.shape)
            rows, columns = np.nonzero(array)
            values = array[rows, columns]
        if shape is None:
            shape = matrix_shape
        elif matrix_shape != shape:
            raise ValueError("all direct operator inputs must have equal shapes")
        entries.extend(
            {
                "row": int(row),
                "column": int(column),
                "value": [complex(value).real, complex(value).imag],
            }
            for row, column, value in zip(rows, columns, values)
        )
    if shape is None:
        raise ValueError("at least one direct operator matrix is required")
    return {"shape": list(shape), "entries": entries}


def _parameter_payload(parameters) -> dict[str, list[float]]:
    if parameters is None:
        return {}
    return {
        str(name): [complex(value).real, complex(value).imag]
        for name, value in parameters.items()
    }


def _scaled_action_output(result, *, out=None, overwrite_out=True, a=1.0):
    scaled = np.asarray(result) * a
    if out is None:
        return scaled
    destination = np.asanyarray(out)
    if destination.shape != scaled.shape:
        raise ValueError("out has the wrong shape")
    if overwrite_out:
        destination[...] = scaled
    else:
        destination[...] += scaled
    return out


def _is_matrix_input(value) -> bool:
    if sp.issparse(value):
        return True
    try:
        return np.asanyarray(value).ndim == 2
    except (TypeError, ValueError):
        return False


def _term_requests(static) -> list[dict[str, Any]]:
    return [term.request() for term in terms_from_static(static)]


def _same_drive_arguments(left, right) -> bool:
    if len(left) != len(right):
        return False
    for left_value, right_value in zip(left, right):
        if left_value is right_value:
            continue
        try:
            if not bool(np.all(np.asarray(left_value) == np.asarray(right_value))):
                return False
        except (TypeError, ValueError):
            return False
    return True


def _group_dynamic_terms(dynamic):
    groups = []
    for entry in dynamic:
        if len(entry) != 4:
            raise ValueError(
                "dynamic basis terms must be [operator, couplings, drive, arguments]"
            )
        operator, couplings, drive, arguments = entry
        if not callable(drive):
            raise TypeError("dynamic drive must be callable")
        arguments = tuple(arguments)
        group = next(
            (
                candidate
                for candidate in groups
                if candidate["drive"] is drive
                and _same_drive_arguments(candidate["arguments"], arguments)
            ),
            None,
        )
        if group is None:
            group = {
                "drive": drive,
                "arguments": arguments,
                "static": [],
            }
            groups.append(group)
        group["static"].append([operator, couplings])
    return groups


def _sparse_from_result(result: dict[str, Any], dtype=np.complex128):
    entries = result["entries"]
    values = np.asarray(
        [complex(*entry["value"]) for entry in entries],
        dtype=np.complex128,
    )
    if np.dtype(dtype).kind != "c":
        if np.any(np.abs(values.imag) > 1.0e-12):
            raise TypeError("complex operator cannot be represented by a real dtype")
        values = values.real
    rows = np.asarray([entry["row"] for entry in entries], dtype=np.intp)
    columns = np.asarray([entry["column"] for entry in entries], dtype=np.intp)
    return sp.csr_matrix(
        (np.asarray(values, dtype=dtype), (rows, columns)),
        shape=tuple(result["shape"]),
    )


def _dense_from_result(result: dict[str, Any], dtype=np.complex128) -> np.ndarray:
    matrix = np.zeros(result["shape"], dtype=np.complex128)
    for entry in result["entries"]:
        matrix[entry["row"], entry["column"]] = complex(*entry["value"])
    target = np.dtype(dtype)
    if target.kind != "c":
        if np.any(np.abs(matrix.imag) > 1.0e-12):
            raise TypeError("complex operator cannot be represented by a real dtype")
        matrix = matrix.real
    return np.asarray(matrix, dtype=target)


def _is_scalar_operand(value) -> bool:
    return np.isscalar(value) or (
        isinstance(value, np.ndarray) and value.ndim == 0
    )


def _complex_values(values):
    return [[complex(value).real, complex(value).imag] for value in values]


def _eigsh_solver_request(
    *,
    dimension,
    k,
    target,
    ncv,
    tol,
    maxiter,
    return_eigenvectors,
    v0,
):
    request = {
        "eigenpairs": int(k),
        "target": target,
        "krylov_dimension": ncv,
        "tolerance": float(tol),
        "max_iterations": int(maxiter),
        "eigenvectors": bool(return_eigenvectors),
    }
    if v0 is None:
        return request
    initial = np.asarray(v0)
    if initial.ndim != 1 or initial.shape[0] != int(dimension):
        raise ValueError(f"v0 must have shape ({int(dimension)},)")
    try:
        initial = np.asarray(initial, dtype=np.complex128)
    except (TypeError, ValueError) as error:
        raise TypeError("v0 must contain numeric values") from error
    if not np.all(np.isfinite(initial)):
        raise ValueError("v0 must contain only finite values")
    if not np.any(np.abs(initial) > np.finfo(np.float64).eps):
        raise ValueError("v0 must have nonzero finite norm")
    request["initial_vector"] = _complex_values(initial)
    return request


def _vector_columns(values, dimension):
    if sp.issparse(values):
        array = values.toarray()
    else:
        array = np.asanyarray(values)
    if array.ndim == 1:
        array = array.reshape((array.shape[0], 1))
    if array.ndim != 2 or array.shape[0] != dimension:
        raise ValueError("vector batches must have shape (dimension, columns)")
    return array


def _matrix_elements(
    owner,
    left,
    right,
    *,
    diagonal,
    parameters,
):
    left = _vector_columns(left, owner.Ns)
    right = _vector_columns(right, owner.Ns)
    result = owner._model.execute(
        "matrix_elements_model",
        left_vectors=[
            _complex_values(left[:, column]) for column in range(left.shape[1])
        ],
        right_vectors=[
            _complex_values(right[:, column]) for column in range(right.shape[1])
        ],
        diagonal=bool(diagonal),
        parameters=parameters,
    )
    values = np.asarray(
        [complex(*value) for value in result["values"]],
        dtype=np.result_type(left.dtype, right.dtype, owner.dtype),
    )
    return values.reshape(result["shape"]).squeeze()


def _measurement_samples(values, dimension, *, enforce_pure):
    sparse = sp.issparse(values)
    array = values.toarray() if sparse else np.asanyarray(values)
    if array.ndim == 1:
        if array.shape[0] != dimension:
            raise ValueError("state dimension mismatch")
        return [("pure", array)], True
    if array.ndim == 2:
        if array.shape[0] != dimension:
            raise ValueError("state dimension mismatch")
        if not enforce_pure and array.shape[1] == dimension:
            return [("density", array.reshape(-1))], True
        return (
            [
                ("pure", array[:, column])
                for column in range(array.shape[1])
            ],
            bool(array.shape[1] == 1 and not enforce_pure),
        )
    if (
        array.ndim == 3
        and not enforce_pure
        and array.shape[0] == dimension
        and array.shape[1] == dimension
    ):
        return (
            [
                ("density", array[:, :, sample].reshape(-1))
                for sample in range(array.shape[2])
            ],
            False,
        )
    raise ValueError(
        "states must be vectors, column batches, density matrices, or density batches"
    )


def _measurement_parameters(owner, count, *, time=None, pars=None):
    if pars is not None:
        payload = _parameter_payload(pars)
        return [payload for _ in range(count)]
    if time is None or np.ndim(time) == 0:
        payload = owner._expression_parameters(time)
        return [payload for _ in range(count)]
    times = np.asarray(time, dtype=np.float64)
    if times.ndim != 1 or times.size != count:
        raise ValueError("time array must match the number of state samples")
    return [owner._expression_parameters(value) for value in times]


def _measure(
    owner,
    values,
    measurement,
    *,
    enforce_pure=False,
    time=None,
    pars=None,
):
    samples, scalar_result = _measurement_samples(
        values,
        owner.Ns,
        enforce_pure=bool(enforce_pure),
    )
    parameters = _measurement_parameters(
        owner,
        len(samples),
        time=time,
        pars=pars,
    )
    result = owner._model.execute(
        "measure_model",
        measurement=measurement,
        samples=[
            {
                "kind": kind,
                "values": _complex_values(sample),
                "parameters": sample_parameters,
            }
            for (kind, sample), sample_parameters in zip(samples, parameters)
        ],
    )
    input_dtype = values.dtype if hasattr(values, "dtype") else np.asanyarray(values).dtype
    target_dtype = np.dtype(np.result_type(input_dtype, owner.dtype))
    output = np.asarray(
        [complex(*value) for value in result["values"]],
        dtype=np.complex128,
    )
    if target_dtype.kind != "c" and np.all(np.abs(output.imag) <= 1.0e-12):
        output = output.real.astype(target_dtype, copy=False)
    else:
        output = output.astype(np.result_type(target_dtype, np.complex128), copy=False)
    return output[0] if scalar_result else output


def _evolve_operator_expression(
    expression,
    v0,
    initial_time,
    times,
    *,
    eom="SE",
    solver_name="dop853",
    stack_state=False,
    verbose=False,
    iterate=False,
    imag_time=False,
    **solver_args,
):
    del stack_state
    if eom not in {"SE", "LvNE"}:
        raise NotImplementedError("only SE and LvNE equations are implemented")
    if imag_time and eom != "SE":
        raise ValueError("imaginary-time evolution is only defined for state vectors")

    dimension = int(expression.shape[0])
    if expression.shape != (dimension, dimension):
        raise ValueError("evolution generator must be square")
    initial = np.asarray(v0, dtype=np.result_type(v0, np.complex64))
    if initial.ndim == 0 or initial.shape[0] != dimension:
        raise ValueError("dimension mismatch")
    if eom == "LvNE" and initial.shape != (dimension, dimension):
        raise ValueError("LvNE evolution requires a square density matrix")

    initial_time = float(initial_time)
    scalar_time = np.ndim(times) == 0
    if scalar_time:
        requested_times = np.asarray([times], dtype=np.float64)
    else:
        requested_times = np.asarray(list(times), dtype=np.float64)
    if (
        requested_times.ndim != 1
        or requested_times.size == 0
        or not np.isfinite(initial_time)
        or not np.all(np.isfinite(requested_times))
        or np.any(np.diff(requested_times) < 0.0)
        or requested_times[0] < initial_time
    ):
        raise ValueError(
            "times must be finite, nondecreasing, and no earlier than initial time"
        )

    def rhs(current_time, values):
        state = values.reshape(initial.shape)
        left = expression.dot(state, time=current_time)
        if eom == "LvNE":
            right = expression.T.dot(state.T, time=current_time).T
            return (-1.0j * (left - right)).reshape(-1)
        factor = -1.0 if imag_time else -1.0j
        return (factor * left).reshape(-1)

    if requested_times[-1] == initial_time:
        states = [initial.copy() for _ in requested_times]
    else:
        method = {"dop853": "DOP853", "dopri5": "RK45"}.get(
            solver_name,
            solver_name,
        )
        atol = float(solver_args.pop("atol", 1.0e-9))
        rtol = float(solver_args.pop("rtol", 1.0e-9))
        max_step = solver_args.pop("max_step", np.inf)
        solver_args.pop("nsteps", None)
        if solver_args:
            names = ", ".join(sorted(solver_args))
            raise TypeError(f"unsupported evolution options: {names}")
        solution = solve_ivp(
            rhs,
            (initial_time, float(requested_times[-1])),
            initial.reshape(-1),
            method=method,
            t_eval=requested_times,
            atol=atol,
            rtol=rtol,
            max_step=float(max_step),
        )
        if not solution.success:
            raise RuntimeError(f"failed state evolution: {solution.message}")
        states = [
            solution.y[:, index].reshape(initial.shape)
            for index in range(requested_times.size)
        ]

    if imag_time:
        normalized = []
        for state in states:
            if state.ndim == 1:
                norm = np.linalg.norm(state)
            else:
                norm = np.linalg.norm(state, axis=0)
            if np.any(np.asarray(norm) == 0.0):
                raise RuntimeError("imaginary-time evolution produced a zero state")
            normalized.append(state / norm)
        states = normalized
    if verbose:
        for target_time, state in zip(requested_times, states):
            print(
                f"evolved to time {target_time}, "
                f"norm of state(s) {np.linalg.norm(state, axis=0)}"
            )
    if iterate:
        return iter(states)
    if scalar_time:
        return states[0]
    return np.stack(states, axis=-1)


class _OperatorExpression:
    """Lazy algebra tree evaluated by the Rust operator boundary."""

    __array_priority__ = 10_000

    def __init__(self, kind, *, shape, dtype, **payload):
        self._kind = kind
        self._shape = tuple(int(value) for value in shape)
        self.dtype = np.dtype(dtype)
        self._payload = payload

    @classmethod
    def model(cls, owner, *, action="normal", parameters=None):
        shape = owner.shape[::-1] if action in {"transpose", "adjoint"} else owner.shape
        return cls(
            "model",
            shape=shape,
            dtype=owner.dtype,
            owner=owner,
            model=owner._model,
            action=action,
            parameters=parameters,
        )

    @classmethod
    def matrix(cls, value, *, action="normal"):
        if not _is_matrix_input(value):
            raise TypeError("operator algebra operands must be matrices or QMBED operators")
        shape = tuple(int(item) for item in value.shape)
        if action in {"transpose", "adjoint"}:
            shape = shape[::-1]
        return cls(
            "matrix",
            shape=shape,
            dtype=value.dtype,
            value=value,
            action=action,
        )

    @property
    def shape(self):
        return self._shape

    @property
    def get_shape(self):
        return self.shape

    def _request(self, time=None):
        if self._kind == "model":
            owner = self._payload["owner"]
            parameters = self._payload["parameters"]
            return {
                "kind": "model",
                "handle": self._payload["model"].handle,
                "parameters": (
                    owner._expression_parameters(time)
                    if parameters is None
                    else _parameter_payload(parameters)
                ),
                "action": self._payload["action"],
            }
        if self._kind == "matrix":
            return {
                "kind": "matrix",
                "operator": _matrix_request([self._payload["value"]]),
                "action": self._payload["action"],
            }
        if self._kind == "scale":
            coefficient = complex(self._payload["coefficient"])
            return {
                "kind": "scale",
                "coefficient": [coefficient.real, coefficient.imag],
                "operand": self._payload["operand"]._request(time),
            }
        if self._kind == "transform":
            return {
                "kind": "transform",
                "action": self._payload["action"],
                "operand": self._payload["operand"]._request(time),
            }
        return {
            "kind": "binary",
            "operation": self._payload["operation"],
            "left": self._payload["left"]._request(time),
            "right": self._payload["right"]._request(time),
        }

    def _materialize(self, format, time=None):
        return command(
            {
                "operation": "evaluate_operator_expression",
                "expression": self._request(time),
                "format": format,
            }
        )

    def _inspect(self, time=None):
        return command(
            {
                "operation": "inspect_operator_expression",
                "expression": self._request(time),
            }
        )

    def _apply(self, vector, time=None):
        values = np.asanyarray(vector)
        if values.ndim == 0 or values.shape[0] != self.shape[1]:
            raise ValueError("dimension mismatch")
        result_dtype = np.result_type(values.dtype, self.dtype)
        values = values.astype(result_dtype, order="C", copy=False)
        columns = values.reshape((self.shape[1], -1))
        result = command(
            {
                "operation": "apply_operator_expression",
                "expression": self._request(time),
                "vectors": [
                    _complex_values(columns[:, column])
                    for column in range(columns.shape[1])
                ],
            }
        )
        output = np.column_stack(
            [
                np.asarray([complex(*value) for value in column])
                for column in result["vectors"]
            ]
        ).reshape((self.shape[0], *values.shape[1:]))
        if np.dtype(result_dtype).kind != "c":
            if np.any(np.abs(output.imag) > 10 * np.finfo(np.float64).eps):
                raise TypeError("complex result cannot be represented by a real dtype")
            output = output.real
        return np.asarray(output, dtype=result_dtype)

    def toarray(self, time=None):
        return _dense_from_result(
            self._materialize("csc", time),
            dtype=self.dtype,
        )

    def todense(self, time=None):
        return np.asmatrix(self.toarray(time))

    def tocsr(self, time=None):
        return _sparse_from_result(
            self._materialize("csr", time),
            dtype=self.dtype,
        )

    def tocsc(self, time=None):
        return self.tocsr(time).tocsc()

    def dot(self, vector, time=None):
        return self._apply(vector, time)

    matvec = dot
    matmat = dot

    def rdot(self, vector, time=None):
        values = np.asanyarray(vector)
        if values.ndim == 0 or values.shape[-1] != self.shape[0]:
            raise ValueError("dimension mismatch")
        moved = np.moveaxis(values, -1, 0)
        applied = self.T.dot(moved, time=time)
        return np.moveaxis(applied, 0, -1)

    def rmatvec(self, vector, time=None):
        return self.H.dot(vector, time=time)

    rmatmat = rmatvec

    def diagonal(self, time=None):
        values = np.asarray(
            [complex(*value) for value in self._inspect(time)["diagonal"]]
        )
        return np.asarray(np.real_if_close(values), dtype=self.dtype)

    def trace(self, time=None):
        value = self._inspect(time)["trace"]
        if value is None:
            raise ValueError("operator trace requires a square operator")
        return np.asarray(complex(*value), dtype=self.dtype).item()

    def _binary(self, other, operation, *, reflected=False):
        right = _as_operator_expression(other)
        left = self
        if reflected:
            left, right = right, left
        if operation in {"add", "subtract"}:
            if left.shape != right.shape:
                raise ValueError("operator addition requires equal shapes")
            shape = left.shape
        else:
            if left.shape[1] != right.shape[0]:
                raise ValueError("operator product has incompatible dimensions")
            shape = (left.shape[0], right.shape[1])
        return _OperatorExpression(
            "binary",
            shape=shape,
            dtype=np.result_type(left.dtype, right.dtype),
            operation=operation,
            left=left,
            right=right,
        )

    def __add__(self, other):
        return self._binary(other, "add")

    def __radd__(self, other):
        return self._binary(other, "add", reflected=True)

    def __sub__(self, other):
        return self._binary(other, "subtract")

    def __rsub__(self, other):
        return self._binary(other, "subtract", reflected=True)

    def __mul__(self, other):
        if _is_scalar_operand(other):
            return _OperatorExpression(
                "scale",
                shape=self.shape,
                dtype=np.result_type(self.dtype, np.asarray(other).dtype),
                coefficient=other,
                operand=self,
            )
        return self._binary(other, "product")

    def __rmul__(self, other):
        if _is_scalar_operand(other):
            return self * other
        return self._binary(other, "product", reflected=True)

    def __neg__(self):
        return self * -1

    @property
    def static(self):
        return self.tocsr()

    @property
    def Ns(self):
        return self.shape[0]

    def eigvalsh(self, time=None):
        return self.eigh(time=time)[0]

    def eigh(self, time=None):
        result = command(
            {
                "operation": "eigh_operator_expression",
                "expression": self._request(time),
                "eigenvectors": True,
            }
        )
        vectors = np.column_stack(
            [
                np.asarray([complex(*value) for value in vector])
                for vector in result["eigenvectors"]
            ]
        )
        return np.asarray(result["eigenvalues"]), vectors

    def evolve(
        self,
        v0,
        time,
        times,
        eom="SE",
        solver_name="dop853",
        stack_state=False,
        verbose=False,
        iterate=False,
        imag_time=False,
        **solver_args,
    ):
        return _evolve_operator_expression(
            self,
            v0,
            time,
            times,
            eom=eom,
            solver_name=solver_name,
            stack_state=stack_state,
            verbose=verbose,
            iterate=iterate,
            imag_time=imag_time,
            **solver_args,
        )

    def eigsh(
        self,
        *,
        k: int,
        which: str = "SA",
        sigma: float | None = None,
        return_eigenvectors: bool = True,
        maxiter: int = 1_000,
        tol: float = 1.0e-10,
        ncv: int | None = None,
        v0=None,
        time: float | None = None,
        **_options,
    ):
        target = (
            {"kind": "shift", "value": float(sigma)}
            if sigma is not None
            else {"kind": _TARGETS[which]}
        )
        result = command(
            {
                "operation": "eigsh_operator_expression",
                "expression": self._request(time),
                "format": "csc",
                "solver": _eigsh_solver_request(
                    dimension=self.shape[0],
                    k=k,
                    target=target,
                    ncv=ncv,
                    tol=tol,
                    maxiter=maxiter,
                    return_eigenvectors=return_eigenvectors,
                    v0=v0,
                ),
            }
        )
        values = np.asarray(result["eigenvalues"])
        if not return_eigenvectors:
            return values
        vectors = np.column_stack(
            [
                np.asarray([complex(*value) for value in vector])
                for vector in result["eigenvectors"]
            ]
        )
        return values, vectors

    @property
    def T(self):
        return _OperatorExpression(
            "transform",
            shape=self.shape[::-1],
            dtype=self.dtype,
            action="transpose",
            operand=self,
        )

    @property
    def H(self):
        return _OperatorExpression(
            "transform",
            shape=self.shape[::-1],
            dtype=self.dtype,
            action="adjoint",
            operand=self,
        )

    def conj(self):
        return _OperatorExpression(
            "transform",
            shape=self.shape,
            dtype=self.dtype,
            action="conjugate",
            operand=self,
        )

    conjugate = conj

    def transpose(self, copy=False):
        del copy
        return self.T

    def getH(self, copy=False):
        del copy
        return self.H

    def copy(self):
        return shallow_copy(self)

    def astype(self, dtype, copy=False, casting="unsafe"):
        dtype = np.dtype(dtype)
        if not np.can_cast(self.dtype, dtype, casting=casting):
            raise TypeError(
                f"cannot cast operator from {self.dtype} to {dtype} "
                f"according to the rule {casting!r}"
            )
        result = self.copy() if copy or dtype != self.dtype else self
        result.dtype = dtype
        return result

    def aslinearoperator(self, time=0.0):
        return sp.linalg.LinearOperator(
            self.shape,
            matvec=lambda vector: self.dot(vector, time=time),
            rmatvec=lambda vector: self.H.dot(vector, time=time),
            matmat=lambda matrix: self.dot(matrix, time=time),
            dtype=self.dtype,
        )

    @property
    def ndim(self):
        return 2

    @property
    def is_dense(self):
        return False


def _as_operator_expression(value):
    if isinstance(value, _OperatorExpression):
        return value
    if isinstance(value, _OperatorView):
        return value._expression()
    if isinstance(value, hamiltonian):
        return value._expression()
    if isinstance(value, _EvaluatedQuantumOperator):
        return value._expression()
    if _is_matrix_input(value):
        return _OperatorExpression.matrix(value)
    raise TypeError("operator algebra operands must be matrices or QMBED operators")


class _OperatorView:
    def __init__(self, owner, *, transposed=False, conjugated=False):
        self._owner = owner
        self._transposed = bool(transposed)
        self._conjugated = bool(conjugated)

    @property
    def shape(self):
        return self._owner.shape[::-1] if self._transposed else self._owner.shape

    @property
    def dtype(self):
        return self._owner.dtype

    @property
    def T(self):
        return _OperatorView(
            self._owner,
            transposed=not self._transposed,
            conjugated=self._conjugated,
        )

    @property
    def H(self):
        return _OperatorView(
            self._owner,
            transposed=not self._transposed,
            conjugated=not self._conjugated,
        )

    def conj(self):
        return _OperatorView(
            self._owner,
            transposed=self._transposed,
            conjugated=not self._conjugated,
        )

    conjugate = conj

    def _expression(self):
        if self._transposed and self._conjugated:
            action = "adjoint"
        elif self._transposed:
            action = "transpose"
        elif self._conjugated:
            action = "conjugate"
        else:
            action = "normal"
        return _OperatorExpression.model(self._owner, action=action)

    def dot(self, vector, time=None):
        return self._owner._dot_action(
            vector,
            transposed=self._transposed,
            conjugated=self._conjugated,
            time=time,
        )

    def rdot(self, vector, time=None):
        return self._expression().rdot(vector, time=time)

    def diagonal(self, time=None):
        return self._expression().diagonal(time=time)

    def trace(self, time=None):
        return self._expression().trace(time=time)

    def transpose(self, copy=False):
        del copy
        return self.T

    def getH(self, copy=False):
        del copy
        return self.H

    def aslinearoperator(self, time=0.0):
        return self._expression().aslinearoperator(time=time)

    def toarray(self, time=None):
        return self._expression().toarray(time)

    def todense(self, time=None):
        return self._expression().todense(time)

    def tocsr(self, time=None):
        return self._expression().tocsr(time)

    def tocsc(self, time=None):
        return self._expression().tocsc(time)

    def __add__(self, other):
        return self._expression() + other

    def __radd__(self, other):
        return other + self._expression()

    def __sub__(self, other):
        return self._expression() - other

    def __rsub__(self, other):
        return other - self._expression()

    def __mul__(self, other):
        return self._expression() * other

    def __rmul__(self, other):
        return other * self._expression()

    def __neg__(self):
        return -self._expression()


class hamiltonian:
    __array_priority__ = 10_000

    def __init__(
        self,
        static_list=None,
        dynamic_list=None,
        N: int | None = None,
        basis=None,
        shape=None,
        Nup: int | None = None,
        S: str | int | float = "1/2",
        pauli: bool | int = True,
        dtype=np.complex128,
        static_fmt=None,
        dynamic_fmt=None,
        copy=True,
        check_herm: bool = True,
        check_pcon: bool = True,
        check_symm: bool = True,
        **basis_options,
    ):
        static = [] if static_list is None else static_list
        dynamic = [] if dynamic_list is None else dynamic_list
        if shape is not None:
            shape = tuple(int(value) for value in shape)
            if len(shape) != 2 or shape[0] != shape[1]:
                raise ValueError("hamiltonian must be a square matrix")
            if not static and not dynamic and basis is None and N is None:
                static = [sp.csr_matrix(shape, dtype=dtype)]
        self.dtype = np.dtype(dtype)
        self._drives = {}
        self._dynamic = {}
        self._static_format = "csr"
        self._dynamic_formats = {}
        direct_static = bool(static) and all(_is_matrix_input(value) for value in static)
        direct_dynamic = bool(dynamic) and all(
            len(value) == 3 and _is_matrix_input(value[0]) for value in dynamic
        )
        if basis is None and N is None and (
            direct_static or direct_dynamic or (not static and direct_dynamic)
        ):
            components = []
            for index, (matrix, drive, arguments) in enumerate(dynamic):
                if not callable(drive):
                    raise TypeError("dynamic drive must be callable")
                name = f"drive_{index}"
                arguments = tuple(arguments)
                components.append(
                    {
                        "name": name,
                        "operator": _matrix_request([matrix]),
                        "default": [
                            complex(drive(0.0, *arguments)).real,
                            complex(drive(0.0, *arguments)).imag,
                        ],
                    }
                )
                self._drives[name] = (drive, arguments)
                self._dynamic[name] = sp.csr_matrix(matrix, copy=bool(copy))
                self._dynamic_formats[name] = "csr"
            request = {"components": components}
            if direct_static:
                request["static_operator"] = _matrix_request(static)
            self.basis = None
            self._terms = ()
            self._checks = {
                "hermiticity": bool(check_herm),
                "particle_conservation": bool(check_pcon),
                "symmetry_compatibility": bool(check_symm),
            }
            self._model = NativeOperatorModel(request)
            self.Ns = self._model.dimension
            if shape is not None and self.shape != shape:
                raise ValueError("shape does not match the supplied operators")
            self.update_matrix_formats(
                "csr" if static_fmt is None else static_fmt,
                "csr" if dynamic_fmt is None else dynamic_fmt,
            )
            return
        if basis is None:
            if N is None:
                raise ValueError("basis or N must be supplied")
            from quspin.basis import spin_basis_1d

            basis = spin_basis_1d(
                N,
                Nup=Nup,
                S=S,
                pauli=pauli,
                **basis_options,
            )
        else:
            if N is not None and int(N) != int(basis.N):
                raise ValueError("N does not match the explicit basis")
            if Nup is not None or basis_options:
                raise ValueError(
                    "basis construction options cannot accompany an explicit basis"
                )

        normalize_lists = getattr(basis, "_normalize_operator_lists", None)
        if normalize_lists is not None:
            static, dynamic = normalize_lists(static, dynamic)
        self.basis = basis
        self._terms = tuple(terms_from_static(static))
        component_requests = []
        dynamic_groups = _group_dynamic_terms(dynamic)
        for index, group in enumerate(dynamic_groups):
            name = f"drive_{index}"
            drive = group["drive"]
            arguments = group["arguments"]
            default = complex(drive(0.0, *arguments))
            component_requests.append(
                {
                    "name": name,
                    "terms": _term_requests(group["static"]),
                    "default": [default.real, default.imag],
                }
            )
            self._drives[name] = (drive, arguments)
        self._checks = {
            "hermiticity": bool(check_herm),
            "particle_conservation": bool(check_pcon),
            "symmetry_compatibility": bool(check_symm),
        }
        request = {
            "basis": self.basis._request,
            "terms": [term.request() for term in self._terms],
            "components": component_requests,
            "site_permutation": self.basis._site_permutation,
            "checks": self._checks,
        }
        self._model = NativeModel(request)
        self.Ns = self._model.dimension
        for index, group in enumerate(dynamic_groups):
            result = self._model.execute(
                "materialize_component_model",
                name=f"drive_{index}",
                format="csr",
            )
            self._dynamic[f"drive_{index}"] = _sparse_from_result(
                result,
                dtype=self.dtype,
            )
            self._dynamic_formats[f"drive_{index}"] = "csr"
        if shape is not None and self.shape != shape:
            raise ValueError("shape does not match the supplied operators")
        self.update_matrix_formats(
            "csr" if static_fmt is None else static_fmt,
            "csr" if dynamic_fmt is None else dynamic_fmt,
        )

    @classmethod
    def _from_native_model(
        cls,
        model,
        *,
        dtype,
        drives=None,
        dynamic_matrices=None,
    ):
        """Create the compatibility view over an already registered Rust family."""
        result = cls.__new__(cls)
        result.dtype = np.dtype(dtype)
        result.basis = None
        result._terms = ()
        result._checks = {
            "hermiticity": False,
            "particle_conservation": False,
            "symmetry_compatibility": False,
        }
        result._model = model
        result.Ns = model.dimension
        result._drives = {} if drives is None else dict(drives)
        result._dynamic = (
            {} if dynamic_matrices is None else dict(dynamic_matrices)
        )
        result._static_format = "csr"
        result._dynamic_formats = {name: "csr" for name in result._dynamic}
        return result

    @property
    def shape(self) -> tuple[int, int]:
        return self.Ns, self.Ns

    @property
    def get_shape(self) -> tuple[int, int]:
        return self.shape

    @property
    def static(self):
        matrix_format = getattr(self, "_static_format", "csr")
        result = self._model.execute(
            "materialize_model",
            format="dense" if matrix_format == "dense" else matrix_format,
            parameters={
                name: [0.0, 0.0]
                for name in self._drives
            },
        )
        if matrix_format == "dense":
            return _dense_from_result(result, dtype=self.dtype)
        matrix = _sparse_from_result(result, dtype=self.dtype)
        return {
            "csc": matrix.tocsc,
            "csr": matrix.tocsr,
            "dia": matrix.todia,
        }[matrix_format]()

    @property
    def dynamic(self):
        def converted(name, matrix):
            matrix_format = self._dynamic_formats.get(name, "csr")
            if matrix_format == "dense":
                return matrix.toarray()
            return {
                "csc": matrix.tocsc,
                "csr": matrix.tocsr,
                "dia": matrix.todia,
            }[matrix_format]()

        return {
            (
                lambda time, drive=drive, arguments=arguments: drive(
                    time,
                    *arguments,
                )
            ): matrix
            for name, (drive, arguments) in self._drives.items()
            for matrix in [converted(name, self._dynamic[name])]
        }

    def __call__(self, time=0.0):
        return self.tocsr(time)

    @property
    def closed(self) -> bool:
        return self._model.closed

    def close(self) -> None:
        self._model.close()

    def __enter__(self) -> hamiltonian:
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()

    def _execute(self, operation: str, **options: Any) -> dict[str, Any]:
        time = options.pop("time", None)
        return self._model.execute(
            operation,
            parameters=self._expression_parameters(time),
            **options,
        )

    def _expression_parameters(self, time=None):
        parameters = {
            name: drive(0.0 if time is None else float(time), *arguments)
            for name, (drive, arguments) in self._drives.items()
        }
        return _parameter_payload(parameters)

    def _expression(self):
        return _OperatorExpression.model(self)

    def _coerce_matrix(self, result: dict[str, Any]) -> np.ndarray:
        return _dense_from_result(result, dtype=self.dtype)

    def _dot_action(
        self,
        vector,
        *,
        transposed=False,
        conjugated=False,
        time: float | None = None,
    ):
        input_array = np.asanyarray(vector)
        if input_array.ndim == 0 or input_array.shape[0] != self.Ns:
            raise ValueError("dimension mismatch")
        result_dtype = np.result_type(input_array.dtype, self.dtype)
        input_array = input_array.astype(result_dtype, order="C", copy=False)
        input_matrix = input_array.reshape((self.Ns, -1))
        vectors = [
            [[complex(value).real, complex(value).imag] for value in input_matrix[:, column]]
            for column in range(input_matrix.shape[1])
        ]
        if transposed and conjugated:
            action = "adjoint"
        elif transposed:
            action = "transpose"
        elif conjugated:
            action = "conjugate"
        else:
            action = "normal"
        result = self._execute(
            "apply_model",
            vectors=vectors,
            action=action,
            time=time,
        )
        applied = np.column_stack(
            [
                np.asarray([complex(*value) for value in output])
                for output in result["vectors"]
            ]
        ).reshape(input_array.shape)
        if np.dtype(result_dtype).kind != "c":
            tolerance = 10 * np.finfo(np.float64).eps
            if np.any(np.abs(applied.imag) > tolerance):
                raise TypeError("complex result cannot be represented by a real dtype")
            applied = applied.real
        return np.asarray(applied, dtype=result_dtype)

    def toarray(self, time=0, order=None, out=None) -> np.ndarray:
        matrix = self._coerce_matrix(
            self._execute("materialize_model", format="csc", time=time)
        )
        if order is not None:
            matrix = np.array(matrix, order=order, copy=True)
        if out is None:
            return matrix
        destination = np.asanyarray(out)
        if destination.shape != matrix.shape:
            raise ValueError("out has the wrong shape")
        destination[...] = matrix
        return out

    def todense(self, time=0, order=None, out=None) -> np.matrix:
        matrix = np.asmatrix(self.toarray(time=time, order=order))
        if out is None:
            return matrix
        destination = np.asanyarray(out)
        if destination.shape != matrix.shape:
            raise ValueError("out has the wrong shape")
        destination[...] = matrix
        return out

    def tocsr(self, time: float | None = None):
        return _sparse_from_result(
            self._execute("materialize_model", format="csr", time=time),
            dtype=self.dtype,
        )

    def tocsc(self, time: float | None = None):
        return self.tocsr(time).tocsc()

    def project_to(self, projector):
        if not _is_matrix_input(projector):
            raise TypeError("projector must be a dense or sparse matrix")
        shape = tuple(int(value) for value in projector.shape)
        if len(shape) != 2 or shape[0] != self.Ns:
            raise ValueError(
                f"projector must have shape ({self.Ns}, reduced_dimension)"
            )
        projected_model = self._model.projected(_matrix_request([projector]))
        projected = type(self).__new__(type(self))
        projected.dtype = np.dtype(
            np.result_type(self.dtype, getattr(projector, "dtype", self.dtype))
        )
        projected.basis = None
        projected._terms = ()
        projected._checks = dict(self._checks)
        projected._model = projected_model
        projected.Ns = projected_model.dimension
        projected._drives = dict(self._drives)
        adjoint = projector.conjugate().transpose()
        projected._dynamic = {
            name: sp.csr_matrix(adjoint @ matrix @ projector)
            for name, matrix in self._dynamic.items()
        }
        return projected

    def eigvalsh(self, time: float | None = None) -> np.ndarray:
        result = self._execute("eigh_model", eigenvectors=False, time=time)
        return np.asarray(result["eigenvalues"])

    def eigh(self, time: float | None = None):
        result = self._execute("eigh_model", eigenvectors=True, time=time)
        vectors = np.column_stack(
            [
                np.asarray([complex(*value) for value in vector])
                for vector in result["eigenvectors"]
            ]
        )
        return np.asarray(result["eigenvalues"]), vectors

    def eigsh(
        self,
        *,
        k: int,
        which: str = "SA",
        sigma: float | None = None,
        return_eigenvectors: bool = True,
        maxiter: int = 1_000,
        tol: float = 1.0e-10,
        ncv: int | None = None,
        v0=None,
        time: float | None = None,
        **_options,
    ):
        target = (
            {"kind": "shift", "value": float(sigma)}
            if sigma is not None
            else {"kind": _TARGETS[which]}
        )
        result = self._execute(
            "eigsh_model",
            format="csc",
            solver=_eigsh_solver_request(
                dimension=self.Ns,
                k=k,
                target=target,
                ncv=ncv,
                tol=tol,
                maxiter=maxiter,
                return_eigenvectors=return_eigenvectors,
                v0=v0,
            ),
            time=time,
        )
        values = np.asarray(result["eigenvalues"])
        if not return_eigenvectors:
            return values
        vectors = np.column_stack(
            [
                np.asarray([complex(*value) for value in vector])
                for vector in result["eigenvectors"]
            ]
        )
        return values, vectors

    def dot(
        self,
        V,
        time=0,
        check=True,
        out=None,
        overwrite_out=True,
        a=1.0,
    ):
        del check
        return _scaled_action_output(
            self._dot_action(V, time=time),
            out=out,
            overwrite_out=overwrite_out,
            a=a,
        )

    def rdot(
        self,
        vector,
        time=0,
        check=True,
        out=None,
        overwrite_out=True,
        a=1.0,
    ):
        del check
        return _scaled_action_output(
            self._expression().rdot(vector, time=time),
            out=out,
            overwrite_out=overwrite_out,
            a=a,
        )

    def matrix_ele(
        self,
        left,
        right,
        time: float | None = None,
        diagonal: bool = False,
        check: bool = True,
    ):
        del check
        if time is not None and np.ndim(time) != 0:
            raise ValueError("matrix_ele accepts one evaluation time")
        return _matrix_elements(
            self,
            left,
            right,
            diagonal=diagonal,
            parameters=self._expression_parameters(time),
        )

    def expt_value(self, values, time=None, check: bool = True, enforce_pure=False):
        del check
        return _measure(
            self,
            values,
            "expectation",
            enforce_pure=enforce_pure,
            time=time,
        )

    def quant_fluct(self, values, time=None, check: bool = True, enforce_pure=False):
        del check
        return _measure(
            self,
            values,
            "quantum_fluctuation",
            enforce_pure=enforce_pure,
            time=time,
        )

    def evolve(
        self,
        v0,
        time,
        times,
        eom: str = "SE",
        solver_name: str = "dop853",
        stack_state: bool = False,
        verbose: bool = False,
        iterate: bool = False,
        imag_time: bool = False,
        **solver_args,
    ):
        if eom == "LvNE":
            return _evolve_operator_expression(
                self._expression(),
                v0,
                time,
                times,
                eom=eom,
                solver_name=solver_name,
                stack_state=stack_state,
                verbose=verbose,
                iterate=iterate,
                imag_time=imag_time,
                **solver_args,
            )
        if eom != "SE":
            raise NotImplementedError("only SE and LvNE equations are implemented")
        del stack_state

        initial = np.asanyarray(v0)
        if initial.ndim == 0 or initial.shape[0] != self.Ns:
            raise ValueError("dimension mismatch")
        initial_shape = initial.shape
        columns = int(np.prod(initial_shape[1:], dtype=np.intp)) if initial.ndim > 1 else 1
        initial_matrix = initial.reshape((self.Ns, columns))

        initial_time = float(time)
        if not np.isfinite(initial_time):
            raise ValueError("initial time must be finite")
        scalar_time = np.ndim(times) == 0
        requested_times = np.atleast_1d(np.asarray(times, dtype=np.float64))
        if requested_times.ndim != 1 or requested_times.size == 0:
            raise ValueError("times must be a scalar or nonempty one-dimensional array")
        relative_times = requested_times - initial_time
        if (
            not np.all(np.isfinite(relative_times))
            or np.any(np.diff(relative_times) < 0.0)
            or relative_times[0] < 0.0
        ):
            raise ValueError(
                "times must be finite, nondecreasing, and no earlier than initial time"
            )

        if solver_name != "qmbed":
            from scipy.integrate import complex_ode

            scipy_options = dict(solver_args)
            if solver_name in {"dop853", "dopri5"}:
                scipy_options.setdefault("nsteps", np.iinfo(np.int32).max)
                scipy_options.setdefault("atol", 1.0e-9)
                scipy_options.setdefault("rtol", 1.0e-9)

            generator_factor = -1.0 if imag_time else -1.0j

            def rhs(current_time, flattened_state):
                state = np.asarray(flattened_state).reshape(initial_shape)
                return (
                    generator_factor * self.dot(state, time=current_time)
                ).reshape(-1)

            solver = complex_ode(rhs)
            solver.set_integrator(solver_name, **scipy_options)
            solver.set_initial_value(initial.reshape(-1), initial_time)
            states = []
            for target_time in requested_times:
                if target_time != initial_time:
                    solver.integrate(float(target_time))
                    if not solver.successful():
                        raise RuntimeError(
                            f"failed to evolve to time {target_time}, "
                            "nsteps might be too small"
                        )
                state = solver.y.reshape(initial_shape)
                if imag_time:
                    norms = np.linalg.norm(state, axis=0)
                    if np.any(norms == 0.0) or not np.all(np.isfinite(norms)):
                        raise RuntimeError(
                            "imaginary-time evolution produced a non-normalizable state"
                        )
                    state = state / norms
                    solver._y[:] = np.ascontiguousarray(
                        state.reshape(-1),
                        dtype=np.complex128,
                    ).view(np.float64)
                if verbose:
                    print(
                        f"evolved to time {target_time}, "
                        f"norm of state(s) {np.linalg.norm(state, axis=0)}"
                    )
                states.append(state.copy())

            if iterate:
                return iter(states)
            if scalar_time:
                return states[0]
            return np.stack(states, axis=-1)

        atol = float(solver_args.pop("atol", 1.0e-10))
        rtol = float(solver_args.pop("rtol", 1.0e-10))
        krylov_dimension = int(
            solver_args.pop("krylov_dimension", min(64, max(1, self.Ns)))
        )
        max_substeps = int(solver_args.pop("nsteps", 10_000))
        max_substeps = int(solver_args.pop("max_substeps", max_substeps))
        if solver_args:
            names = ", ".join(sorted(solver_args))
            raise TypeError(f"unsupported evolution options: {names}")
        tolerance = min(atol, rtol)
        if (
            not np.isfinite(tolerance)
            or tolerance <= 0.0
            or krylov_dimension <= 0
            or max_substeps <= 0
        ):
            raise ValueError("evolution tolerances and iteration controls must be positive")

        vectors = [
            [
                [complex(value).real, complex(value).imag]
                for value in initial_matrix[:, column]
            ]
            for column in range(columns)
        ]

        del verbose
        evolution = {
            "times": (
                requested_times.tolist()
                if self._drives
                else relative_times.tolist()
            ),
            "krylov_dimension": krylov_dimension,
            "tolerance": tolerance,
            "max_substeps": max_substeps,
            "imaginary_time": bool(imag_time),
        }
        if self._drives:
            drives = tuple(self._drives.items())
            result = self._model.evolve_with_drive(
                component_names=[name for name, _ in drives],
                vectors=vectors,
                initial_time=initial_time,
                evolution=evolution,
                coefficient_callback=lambda current_time: [
                    drive(current_time, *arguments)
                    for _, (drive, arguments) in drives
                ],
            )
        else:
            result = self._execute(
                "evolve_model",
                vectors=vectors,
                evolution=evolution,
            )
        result_dtype = np.result_type(initial.dtype, self.dtype, np.complex64)
        states = []
        for columns_at_time in result["states"]:
            matrix = np.column_stack(
                [
                    np.asarray(
                        [complex(*value) for value in column],
                        dtype=result_dtype,
                    )
                    for column in columns_at_time
                ]
            )
            states.append(matrix.reshape(initial_shape))

        if iterate:
            return iter(states)
        if scalar_time:
            return states[0]
        return np.stack(states, axis=-1)

    @property
    def T(self):
        return _OperatorView(self, transposed=True)

    @property
    def H(self):
        return _OperatorView(self, transposed=True, conjugated=True)

    def conj(self):
        return _OperatorView(self, conjugated=True)

    conjugate = conj

    def diagonal(self, time=0):
        return self._expression().diagonal(time=time)

    def trace(self, time=0):
        return self._expression().trace(time=time)

    def transpose(self, copy=False):
        del copy
        return self.T

    def getH(self, copy=False):
        del copy
        return self.H

    def aslinearoperator(self, time=0.0):
        return self._expression().aslinearoperator(time=time)

    def rotate_by(self, other, generator=False, **exp_op_kwargs):
        if generator:
            transformation = exp_op(other, **exp_op_kwargs).get_mat(dense=True)
            transformation = _as_operator_expression(transformation)
        else:
            transformation = _as_operator_expression(other)
        expression = transformation.H * self._expression() * transformation
        model = NativeOperatorModel.from_expression(expression._request(), format="csc")
        result = hamiltonian._from_native_model(
            model,
            dtype=np.result_type(self.dtype, transformation.dtype),
        )
        result.basis = self.basis
        result._static_format = self._static_format
        result._dynamic_formats = dict(self._dynamic_formats)
        return result

    def update_matrix_formats(self, static_fmt, dynamic_fmt):
        valid = {"dense", "csc", "csr", "dia"}
        if static_fmt is not None and str(static_fmt).lower() not in valid:
            raise ValueError(f"unsupported static matrix format {static_fmt!r}")
        if dynamic_fmt is None:
            dynamic_formats = dict(self._dynamic_formats)
        elif isinstance(dynamic_fmt, str):
            dynamic_formats = {name: dynamic_fmt for name in self._dynamic}
        elif hasattr(dynamic_fmt, "items"):
            dynamic_formats = dict(dynamic_fmt)
        else:
            dynamic_formats = {}
            for matrix_format, drive_key in dynamic_fmt:
                if not isinstance(drive_key, (tuple, list)) or len(drive_key) != 2:
                    raise ValueError(
                        "dynamic format entries must identify (drive, arguments)"
                    )
                drive, arguments = drive_key
                for name, (candidate, candidate_arguments) in self._drives.items():
                    if candidate is drive and tuple(arguments) == tuple(candidate_arguments):
                        dynamic_formats[name] = matrix_format
                        break
                else:
                    raise ValueError("dynamic format refers to an unknown drive")
        resolved_dynamic_formats = {}
        for name, (drive, arguments) in self._drives.items():
            matrix_format = dynamic_formats.get(name)
            if matrix_format is None:
                for key, candidate_format in dynamic_formats.items():
                    if (
                        isinstance(key, (tuple, list))
                        and len(key) == 2
                        and key[0] is drive
                        and tuple(key[1]) == tuple(arguments)
                    ):
                        matrix_format = candidate_format
                        break
            resolved_dynamic_formats[name] = (
                self._dynamic_formats.get(name, "csr")
                if matrix_format is None
                else matrix_format
            )
        for matrix_format in resolved_dynamic_formats.values():
            if str(matrix_format).lower() not in valid:
                raise ValueError(f"unsupported dynamic matrix format {matrix_format!r}")
        if static_fmt is not None:
            self._static_format = str(static_fmt).lower()
        self._dynamic_formats = {
            name: str(resolved_dynamic_formats.get(name, "csr")).lower()
            for name in self._dynamic
        }

    def as_dense_format(self, copy=False):
        result = self.copy() if copy else self
        result.update_matrix_formats(
            "dense",
            {name: "dense" for name in result._dynamic},
        )
        return result

    def as_sparse_format(self, static_fmt="csr", dynamic_fmt=None, copy=False):
        result = self.copy() if copy else self
        if dynamic_fmt is None:
            dynamic_fmt = {name: "csr" for name in result._dynamic}
        result.update_matrix_formats(static_fmt, dynamic_fmt)
        return result

    def copy(self):
        dynamic = [
            [self._dynamic[name].copy(), drive, arguments]
            for name, (drive, arguments) in self._drives.items()
        ]
        result = hamiltonian(
            [self.static.copy()],
            dynamic,
            dtype=self.dtype,
            check_herm=False,
            check_pcon=False,
            check_symm=False,
        )
        result.basis = self.basis
        result._static_format = self._static_format
        result._dynamic_formats = dict(self._dynamic_formats)
        return result

    def astype(self, dtype, copy=False, casting="unsafe"):
        dtype = np.dtype(dtype)
        if not np.can_cast(self.dtype, dtype, casting=casting):
            raise TypeError(
                f"cannot cast hamiltonian from {self.dtype} to {dtype} "
                f"according to the rule {casting!r}"
            )
        if dtype == self.dtype and not copy:
            return self
        dynamic = [
            [
                self._dynamic[name].astype(dtype, copy=True),
                drive,
                arguments,
            ]
            for name, (drive, arguments) in self._drives.items()
        ]
        result = hamiltonian(
            [self.static.astype(dtype, copy=True)],
            dynamic,
            dtype=dtype,
            check_herm=False,
            check_pcon=False,
            check_symm=False,
        )
        result.basis = self.basis
        return result

    @property
    def ndim(self):
        return 2

    @property
    def is_dense(self):
        return getattr(self, "_static_format", "csr") == "dense" and all(
            self._dynamic_formats.get(name, "csr") == "dense"
            for name in self._dynamic
        )

    def check_is_dense(self):
        return self.is_dense

    @property
    def nbytes(self):
        def sparse_nbytes(matrix):
            if isinstance(matrix, np.ndarray):
                return matrix.nbytes
            matrix = matrix.tocsr()
            return matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes

        return sparse_nbytes(self.static) + sum(
            sparse_nbytes(matrix) for matrix in self._dynamic.values()
        )

    def __add__(self, other):
        return self._expression() + other

    def __radd__(self, other):
        return other + self._expression()

    def __sub__(self, other):
        return self._expression() - other

    def __rsub__(self, other):
        return other - self._expression()

    def __mul__(self, other):
        return self._expression() * other

    def __rmul__(self, other):
        return other * self._expression()

    def __neg__(self):
        return -self._expression()


class exp_op:
    """Matrix exponential actions over one Rust-backed operator expression."""

    def __init__(
        self,
        O,
        a=1.0,
        start=None,
        stop=None,
        num=None,
        endpoint=None,
        iterate=False,
    ):
        if np.ndim(a) != 0:
            raise TypeError("expecting scalar argument for a")
        self._O = O
        self._operator = (
            None if isinstance(O, quantum_operator) else _as_operator_expression(O)
        )
        self._a = complex(a)
        self._grid = None
        self._step = None
        self._start = None
        self._stop = None
        self._num = None
        self._endpoint = None
        self._iterate = False
        if start is not None or stop is not None:
            if start is None or stop is None:
                raise ValueError("start and stop must be provided together")
            self.set_grid(start, stop, num=num, endpoint=endpoint)
        elif num is not None or endpoint is not None:
            raise ValueError("num and endpoint require start and stop")
        self.set_iterate(iterate)

    def _expression(self, *, time=None, pars=None):
        if self._operator is not None:
            return self._operator, time
        return self._O._expression(pars), None

    @staticmethod
    def _columns(values, dimension):
        values = np.asanyarray(values)
        if values.ndim == 0 or values.shape[0] != dimension:
            raise ValueError("dimension mismatch")
        return values, values.reshape((dimension, -1))

    def _apply(self, expression, values, coefficient, *, time=None):
        values, columns = self._columns(values, expression.shape[1])
        result = command(
            {
                "operation": "expm_operator_expression",
                "expression": expression._request(time),
                "coefficient": [complex(coefficient).real, complex(coefficient).imag],
                "vectors": [
                    _complex_values(columns[:, column])
                    for column in range(columns.shape[1])
                ],
            }
        )
        output = np.column_stack(
            [
                np.asarray([complex(*value) for value in column])
                for column in result["vectors"]
            ]
        )
        return output.reshape((expression.shape[0], *values.shape[1:]))

    @property
    def O(self):
        return self._O

    @property
    def a(self):
        return self._a

    @property
    def grid(self):
        return self._grid

    @property
    def step(self):
        return self._step

    @property
    def iterate(self):
        return self._iterate

    @property
    def Ns(self):
        return self.get_shape[0]

    @property
    def get_shape(self):
        return self._O.shape

    @property
    def ndim(self):
        return 2

    @property
    def T(self):
        return self.transpose(copy=False)

    @property
    def H(self):
        return self.getH(copy=False)

    def set_iterate(self, value):
        if not isinstance(value, (bool, np.bool_)):
            raise ValueError("iterate option must be true or false")
        if value and self._grid is None:
            raise ValueError("grid must be set in order to enable iteration")
        self._iterate = bool(value)

    def set_a(self, new_a):
        if np.ndim(new_a) != 0:
            raise ValueError("a must be scalar")
        self._a = complex(new_a)

    def set_grid(self, start, stop, num=None, endpoint=None):
        if np.ndim(start) != 0 or np.ndim(stop) != 0:
            raise ValueError("start and stop must be scalar")
        if not np.isreal(start) or not np.isreal(stop):
            raise ValueError("start and stop must be real")
        if num is not None and not isinstance(num, int):
            raise ValueError("num must be an integer")
        if endpoint is not None and not isinstance(endpoint, bool):
            raise ValueError("endpoint must be a boolean")
        self._start = float(start)
        self._stop = float(stop)
        self._num = 50 if num is None else int(num)
        self._endpoint = True if endpoint is None else bool(endpoint)
        self._grid, self._step = np.linspace(
            self._start,
            self._stop,
            num=self._num,
            endpoint=self._endpoint,
            retstep=True,
        )

    def unset_grid(self):
        self._iterate = False
        self._start = None
        self._stop = None
        self._num = None
        self._endpoint = None
        self._grid = None
        self._step = None

    def _factors(self):
        return np.asarray([1.0]) if self._grid is None else self._grid

    def dot(self, other, shift=None, **call_kwargs):
        expression, time = self._expression(
            time=call_kwargs.pop("time", None),
            pars=call_kwargs.pop("pars", None),
        )
        if call_kwargs:
            raise TypeError(
                f"unsupported exponential options: {', '.join(sorted(call_kwargs))}"
            )
        shift = 0.0 if shift is None else complex(shift)
        states = []
        for factor in self._factors():
            coefficient = self._a * factor
            state = self._apply(expression, other, coefficient, time=time)
            if shift != 0.0:
                state *= np.exp(coefficient * shift)
            states.append(state)
        if self._iterate:
            return iter(states)
        if self._grid is None:
            return states[0]
        return np.stack(states, axis=-1)

    def rdot(self, other, shift=None, **call_kwargs):
        expression, time = self._expression(
            time=call_kwargs.pop("time", None),
            pars=call_kwargs.pop("pars", None),
        )
        if call_kwargs:
            raise TypeError(
                f"unsupported exponential options: {', '.join(sorted(call_kwargs))}"
            )
        values = np.asanyarray(other)
        if values.ndim == 0 or values.shape[-1] != expression.shape[0]:
            raise ValueError("dimension mismatch")
        moved = np.moveaxis(values, -1, 0)
        shift = 0.0 if shift is None else complex(shift)
        states = []
        for factor in self._factors():
            coefficient = self._a * factor
            state = self._apply(expression.T, moved, coefficient, time=time)
            state = np.moveaxis(state, 0, -1)
            if shift != 0.0:
                state *= np.exp(coefficient * shift)
            states.append(state)
        if self._iterate:
            return iter(states)
        if self._grid is None:
            return states[0]
        return np.stack(states, axis=-1)

    def get_mat(self, dense=False, **call_kwargs):
        identity = np.eye(self.Ns, dtype=np.complex128)
        matrix = self.dot(identity, **call_kwargs)
        if self._grid is not None:
            if dense:
                return matrix
            return np.asarray(
                [sp.csc_matrix(matrix[..., index]) for index in range(matrix.shape[-1])],
                dtype=object,
            )
        return matrix if dense else sp.csc_matrix(matrix)

    def sandwich(self, other, shift=None, **call_kwargs):
        density = np.asarray(other)
        dimension = self.Ns
        if density.shape != (dimension, dimension):
            raise ValueError("sandwich requires a square density matrix")
        expression, time = self._expression(
            time=call_kwargs.pop("time", None),
            pars=call_kwargs.pop("pars", None),
        )
        if call_kwargs:
            raise TypeError(
                f"unsupported exponential options: {', '.join(sorted(call_kwargs))}"
            )
        shift = 0.0 if shift is None else complex(shift)
        states = []
        for factor in self._factors():
            coefficient = self._a * factor
            right = self._apply(expression.T, density.T, coefficient, time=time).T
            state = self._apply(
                expression.H,
                right,
                coefficient.conjugate(),
                time=time,
            )
            if shift != 0.0:
                state *= np.exp(coefficient * shift)
                state *= np.exp(coefficient * shift).conjugate()
            states.append(state)
        if self._iterate:
            return iter(states)
        if self._grid is None:
            return states[0]
        return np.stack(states, axis=-1)

    def transpose(self, copy=False):
        result = self.copy() if copy else self
        expression, _time = result._expression()
        result._operator = expression.T
        result._O = result._operator
        return result

    def conj(self):
        expression, _time = self._expression()
        self._operator = expression.conj()
        self._O = self._operator
        self._a = self._a.conjugate()
        return self

    conjugate = conj

    def getH(self, copy=False):
        result = self.copy() if copy else self
        expression, _time = result._expression()
        result._operator = expression.H
        result._O = result._operator
        result._a = result._a.conjugate()
        return result

    def copy(self):
        result = shallow_copy(self)
        if self._grid is not None:
            result._grid = self._grid.copy()
        return result


class quantum_LinearOperator(hamiltonian):
    """Matrix-free QuSpin-compatible view over one persistent Rust model."""

    def __init__(
        self,
        static_list,
        N=None,
        basis=None,
        diagonal=None,
        check_symm=True,
        check_herm=True,
        check_pcon=True,
        dtype=np.complex128,
        copy=False,
        **basis_args,
    ):
        static = tuple(static_list)
        self._static_list = static
        super().__init__(
            static,
            [],
            N=N,
            basis=basis,
            dtype=dtype,
            copy=copy,
            check_symm=check_symm,
            check_herm=check_herm,
            check_pcon=check_pcon,
            **basis_args,
        )
        self._base_expression = super()._expression()
        self._diagonal = None
        if diagonal is not None:
            self.set_diagonal(diagonal)

    @property
    def static_list(self):
        return list(self._static_list)

    @property
    def diagonal(self):
        if self._diagonal is None:
            return None
        values = self._diagonal.view()
        values.setflags(write=False)
        return values

    def set_diagonal(self, diagonal, copy=True):
        values = np.asarray(diagonal)
        if values.ndim != 1 or values.size != self.Ns:
            raise ValueError("diagonal must be one-dimensional with length Ns")
        values = values.copy() if copy else values
        diagonal_matrix = sp.diags(values, format="csr")
        expression = self._base_expression + diagonal_matrix
        replacement = NativeOperatorModel.from_expression(
            expression._request(),
            format="csr",
        )
        if self._model is not self._base_expression._payload["model"]:
            self._model.close()
        self._model = replacement
        self._diagonal = values

    def dot(self, other, out=None, a=1.0):
        return super().dot(
            other,
            time=0,
            check=False,
            out=out,
            overwrite_out=True,
            a=a,
        )

    def matvec(self, x):
        return self.dot(x)

    def matmat(self, X):
        return self.dot(X)

    def rmatvec(self, x):
        return self.H.dot(x)

    def rmatmat(self, X):
        return self.H.dot(X)

    def adjoint(self):
        return self.H

    def copy(self):
        return quantum_LinearOperator(
            self._static_list,
            basis=self.basis,
            diagonal=self._diagonal,
            dtype=self.dtype,
            check_symm=False,
            check_herm=False,
            check_pcon=False,
            copy=True,
        )

    def close(self):
        current = self._model
        base = self._base_expression._payload["model"]
        current.close()
        if base is not current:
            base.close()


class _QuantumOperatorDifference:
    """Compatibility view for an exactly empty named-family difference."""

    def __init__(self, dimension, dtype):
        self.Ns = int(dimension)
        self.dtype = np.dtype(dtype)
        self._quantum_operator = {}

    @property
    def shape(self):
        return self.Ns, self.Ns

    @property
    def get_shape(self):
        return self.shape

    def toarray(self, pars=None):
        if pars and any(complex(value) != 0.0 for value in pars.values()):
            name = next(iter(pars))
            raise ValueError(f"unknown operator parameter {name!r}")
        return np.zeros(self.shape, dtype=self.dtype)

    def todense(self, pars=None):
        return np.asmatrix(self.toarray(pars))


class quantum_operator:
    """Named linear combination backed by one Rust operator-model handle."""

    def __init__(
        self,
        input_dict,
        N=None,
        basis=None,
        shape=None,
        copy=True,
        check_symm=True,
        check_herm=True,
        check_pcon=True,
        matrix_formats=None,
        dtype=np.complex128,
        **basis_args,
    ):
        checks = {
            "hermiticity": bool(check_herm),
            "particle_conservation": bool(check_pcon),
            "symmetry_compatibility": bool(check_symm),
        }
        del copy
        if not input_dict:
            raise ValueError("quantum_operator requires at least one component")
        input_dict = {
            name: list(component_input)
            for name, component_input in input_dict.items()
        }
        component_names = [str(name) for name in input_dict]
        if len(set(component_names)) != len(component_names):
            raise ValueError("quantum_operator component names must be unique")
        if matrix_formats is None:
            matrix_formats = {}
        elif not hasattr(matrix_formats, "items"):
            raise TypeError("matrix_formats must be a mapping from component to format")
        matrix_formats = {
            str(name): str(matrix_format).lower()
            for name, matrix_format in matrix_formats.items()
        }
        unknown_formats = set(matrix_formats).difference(component_names)
        if unknown_formats:
            name = sorted(unknown_formats)[0]
            raise ValueError(f"matrix format supplied for unknown component {name!r}")
        valid_formats = {"dense", "csc", "csr", "dia"}
        invalid_formats = {
            name: matrix_format
            for name, matrix_format in matrix_formats.items()
            if matrix_format not in valid_formats
        }
        if invalid_formats:
            name, matrix_format = next(iter(invalid_formats.items()))
            raise ValueError(
                f"unsupported matrix format {matrix_format!r} for component {name!r}"
            )
        matrix_component_flags = []
        for component_input in input_dict.values():
            values = list(component_input)
            matrix_component_flags.append(bool(values) and all(
                _is_matrix_input(value) for value in values
            ))
        if basis is None and not all(matrix_component_flags):
            if N is None:
                raise ValueError(
                    "basis or N must be supplied for local operator components"
                )
            from quspin.basis import spin_basis_1d

            basis = spin_basis_1d(N, **basis_args)
        elif basis is not None:
            if N is not None and int(N) != int(basis.L):
                raise ValueError("N does not match the explicit basis")
            if basis_args:
                raise ValueError(
                    "basis construction options cannot accompany an explicit basis"
                )
        elif basis_args:
            names = ", ".join(sorted(basis_args))
            raise TypeError(f"unused basis options for matrix components: {names}")

        self.basis = basis
        if basis is None:
            components = [
                {
                    "name": str(name),
                    "operator": _matrix_request(matrices),
                    "default": [1.0, 0.0],
                }
                for name, matrices in input_dict.items()
            ]
            self._model = NativeOperatorModel({"components": components})
        else:
            components = []
            for name, component_input in input_dict.items():
                component_input = list(component_input)
                matrix_inputs = [
                    _is_matrix_input(value) for value in component_input
                ]
                if all(matrix_inputs):
                    payload = {
                        "operator": _matrix_request(component_input),
                    }
                elif any(matrix_inputs):
                    raise TypeError(
                        "one quantum_operator component cannot mix matrices and local terms"
                    )
                else:
                    payload = {
                        "terms": _term_requests(component_input),
                    }
                components.append(
                    {
                        "name": str(name),
                        "default": [1.0, 0.0],
                        **payload,
                    }
                )
            self._model = NativeOperatorModel(
                {
                    "basis": basis._request,
                    "components": components,
                    "site_permutation": basis._site_permutation,
                    "checks": checks,
                }
            )
        self.Ns = self._model.dimension
        if shape is not None:
            requested_shape = tuple(int(value) for value in shape)
            if len(requested_shape) != 2 or requested_shape[0] != requested_shape[1]:
                self._model.close()
                raise ValueError("quantum_operator must be square")
            if requested_shape != self.shape:
                self._model.close()
                raise ValueError("shape does not match the supplied operators")
        self.dtype = np.dtype(dtype)
        self._component_names = tuple(component_names)
        self._matrix_formats = {
            name: matrix_formats.get(name, "csc") for name in component_names
        }
        self._component_cache = None

    @classmethod
    def _from_archive(cls, path):
        model = NativeOperatorModel.load_operator_archive(path)
        result = cls.__new__(cls)
        result.basis = None
        result._model = model
        result.Ns = model.dimension
        result.dtype = np.dtype(
            model.archive_metadata.get("scalar_dtype", "complex128")
        )
        result._component_names = tuple(
            component["name"] for component in model.archive_components
        )
        result._matrix_formats = {
            component["name"]: component["format"]
            for component in model.archive_components
        }
        result._component_cache = None
        return result

    @staticmethod
    def _component_matrix(result, matrix_format, dtype):
        if matrix_format == "dense":
            return _dense_from_result(result, dtype=dtype)
        matrix = _sparse_from_result(result, dtype=dtype)
        return {
            "csc": matrix.tocsc,
            "csr": matrix.tocsr,
            "dia": matrix.todia,
        }[matrix_format]()

    @property
    def _quantum_operator(self):
        if self._component_cache is None:
            parameters = {
                name: 0.0 for name in self._component_names
            }
            components = {}
            for name in self._component_names:
                parameters[name] = 1.0
                matrix_format = self._matrix_formats[name]
                result = self._execute(
                    "materialize_model",
                    parameters,
                    format=matrix_format,
                )
                components[name] = self._component_matrix(
                    result,
                    matrix_format,
                    self.dtype,
                )
                parameters[name] = 0.0
            self._component_cache = components
        return self._component_cache

    @property
    def shape(self) -> tuple[int, int]:
        return self.Ns, self.Ns

    @property
    def get_shape(self) -> tuple[int, int]:
        return self.shape

    @property
    def closed(self) -> bool:
        return self._model.closed

    def close(self) -> None:
        self._model.close()

    def __enter__(self) -> quantum_operator:
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()

    def _execute(self, operation: str, pars=None, **options: Any) -> dict[str, Any]:
        return self._model.execute(
            operation,
            parameters=_parameter_payload(pars),
            **options,
        )

    def _expression(self, pars=None, action="normal"):
        return _OperatorExpression.model(
            self,
            action=action,
            parameters={} if pars is None else pars,
        )

    def _dot_action(self, vector, pars=None):
        input_array = np.asanyarray(vector)
        if input_array.ndim not in (1, 2) or input_array.shape[0] != self.Ns:
            raise ValueError("dimension mismatch")
        result_dtype = np.result_type(input_array.dtype, self.dtype)
        input_array = input_array.astype(result_dtype, order="C", copy=False)
        input_matrix = input_array.reshape((self.Ns, -1))
        result = self._execute(
            "apply_model",
            pars,
            vectors=[
                [
                    [complex(value).real, complex(value).imag]
                    for value in input_matrix[:, column]
                ]
                for column in range(input_matrix.shape[1])
            ],
        )
        applied = np.column_stack(
            [
                np.asarray([complex(*value) for value in output])
                for output in result["vectors"]
            ]
        ).reshape(input_array.shape)
        if np.dtype(result_dtype).kind != "c":
            if np.any(np.abs(applied.imag) > 10 * np.finfo(np.float64).eps):
                raise TypeError("complex result cannot be represented by a real dtype")
            applied = applied.real
        return np.asarray(applied, dtype=result_dtype)

    def dot(
        self,
        V,
        pars=None,
        check=True,
        out=None,
        overwrite_out=True,
        a=1.0,
    ):
        del check
        return _scaled_action_output(
            self._dot_action(V, pars),
            out=out,
            overwrite_out=overwrite_out,
            a=a,
        )

    def matvec(self, x):
        return self.dot(x)

    def matmat(self, X):
        return self.dot(X)

    def rdot(
        self,
        vector,
        pars=None,
        check=False,
        out=None,
        overwrite_out=True,
        a=1.0,
    ):
        del check
        return _scaled_action_output(
            self._expression(pars).rdot(vector),
            out=out,
            overwrite_out=overwrite_out,
            a=a,
        )

    def rmatvec(self, x):
        return self._expression(action="adjoint").dot(x)

    def rmatmat(self, X):
        return self._expression(action="adjoint").dot(X)

    def matrix_ele(self, left, right, pars=None, diagonal=False, check: bool = True):
        del check
        return _matrix_elements(
            self,
            left,
            right,
            diagonal=diagonal,
            parameters=_parameter_payload(pars),
        )

    def expt_value(self, values, pars=None, check: bool = True, enforce_pure=False):
        del check
        return _measure(
            self,
            values,
            "expectation",
            enforce_pure=enforce_pure,
            pars={} if pars is None else pars,
        )

    def quant_fluct(self, values, pars=None, check: bool = True, enforce_pure=False):
        del check
        return _measure(
            self,
            values,
            "quantum_fluctuation",
            enforce_pure=enforce_pure,
            pars={} if pars is None else pars,
        )

    def toarray(self, pars=None, out=None) -> np.ndarray:
        result = self._execute("materialize_model", pars, format="csc")
        matrix = np.zeros(result["shape"], dtype=np.complex128)
        for entry in result["entries"]:
            matrix[entry["row"], entry["column"]] = complex(*entry["value"])
        if self.dtype.kind != "c":
            if np.any(np.abs(matrix.imag) > 1.0e-12):
                raise TypeError("complex operator cannot be represented by a real dtype")
            matrix = matrix.real
        matrix = np.asarray(matrix, dtype=self.dtype)
        if out is None:
            return matrix
        destination = np.asanyarray(out)
        if destination.shape != matrix.shape:
            raise ValueError("out has the wrong shape")
        destination[...] = matrix
        return out

    def todense(self, pars=None, out=None) -> np.matrix:
        matrix = np.asmatrix(self.toarray(pars))
        if out is None:
            return matrix
        destination = np.asanyarray(out)
        if destination.shape != matrix.shape:
            raise ValueError("out has the wrong shape")
        destination[...] = matrix
        return out

    def eigvalsh(self, pars=None) -> np.ndarray:
        result = self._execute("eigh_model", pars, eigenvectors=False)
        return np.asarray(result["eigenvalues"])

    def eigh(self, pars=None):
        result = self._execute("eigh_model", pars, eigenvectors=True)
        vectors = np.column_stack(
            [
                np.asarray([complex(*value) for value in vector])
                for vector in result["eigenvectors"]
            ]
        )
        return np.asarray(result["eigenvalues"]), vectors

    def eigsh(
        self,
        *,
        k: int,
        pars=None,
        which: str = "LM",
        sigma: float | None = None,
        return_eigenvectors: bool = True,
        maxiter: int = 1_000,
        tol: float = 1.0e-10,
        ncv: int | None = None,
        v0=None,
        **_options,
    ):
        target = (
            {"kind": "shift", "value": float(sigma)}
            if sigma is not None
            else {"kind": _TARGETS[which]}
        )
        result = self._execute(
            "eigsh_model",
            pars,
            format="csc",
            solver=_eigsh_solver_request(
                dimension=self.Ns,
                k=k,
                target=target,
                ncv=ncv,
                tol=tol,
                maxiter=maxiter,
                return_eigenvectors=return_eigenvectors,
                v0=v0,
            ),
        )
        values = np.asarray(result["eigenvalues"])
        if not return_eigenvectors:
            return values
        vectors = np.column_stack(
            [
                np.asarray([complex(*value) for value in vector])
                for vector in result["eigenvectors"]
            ]
        )
        return values, vectors

    def tohamiltonian(self, pars=None):
        return _EvaluatedQuantumOperator(self, pars)

    def diagonal(self, pars=None):
        return self._expression(pars).diagonal()

    def trace(self, pars=None):
        return self._expression(pars).trace()

    def aslinearoperator(self, pars=None):
        return self._expression(pars).aslinearoperator()

    def get_operators(self, key):
        return self._quantum_operator[key]

    @property
    def ndim(self):
        return 2

    def tocsr(self, pars=None):
        return sp.csr_matrix(self.toarray(pars))

    def tocsc(self, pars=None):
        return self.tocsr(pars).tocsc()

    def _transformed_family(self, action):
        transform = {
            "transpose": lambda matrix: matrix.T,
            "conjugate": lambda matrix: matrix.conjugate(),
            "adjoint": lambda matrix: matrix.conjugate().T,
        }[action]
        result = quantum_operator(
            {
                name: [transform(matrix)]
                for name, matrix in self._quantum_operator.items()
            },
            basis=self.basis,
            dtype=self.dtype,
            check_herm=False,
            check_pcon=False,
            check_symm=False,
        )
        result._matrix_formats = dict(self._matrix_formats)
        return result

    def transpose(self, copy=False):
        del copy
        return self._transformed_family("transpose")

    def getH(self, copy=False):
        del copy
        return self._transformed_family("adjoint")

    def conj(self):
        return self._transformed_family("conjugate")

    conjugate = conj

    @property
    def T(self):
        return self.transpose()

    @property
    def H(self):
        return self.getH()

    def copy(self):
        result = quantum_operator(
            {
                name: [matrix.copy()]
                for name, matrix in self._quantum_operator.items()
            },
            basis=self.basis,
            dtype=self.dtype,
            check_herm=False,
            check_pcon=False,
            check_symm=False,
        )
        result._matrix_formats = dict(self._matrix_formats)
        return result

    def astype(self, dtype, copy=False, casting="unsafe"):
        dtype = np.dtype(dtype)
        if not np.can_cast(self.dtype, dtype, casting=casting):
            raise TypeError(
                f"cannot cast quantum_operator from {self.dtype} to {dtype} "
                f"according to the rule {casting!r}"
            )
        if dtype == self.dtype and not copy:
            return self
        result = quantum_operator(
            {
                name: [matrix.astype(dtype, copy=True)]
                for name, matrix in self._quantum_operator.items()
            },
            basis=self.basis,
            dtype=dtype,
            check_herm=False,
            check_pcon=False,
            check_symm=False,
        )
        result._matrix_formats = dict(self._matrix_formats)
        return result

    @property
    def is_dense(self):
        return all(
            not sp.issparse(matrix) for matrix in self._quantum_operator.values()
        )

    def update_matrix_formats(self, matrix_formats):
        unknown = set(matrix_formats).difference(self._component_names)
        if unknown:
            raise ValueError(
                f"matrix format supplied for unknown component {sorted(unknown)[0]!r}"
            )
        valid = {"dense", "csc", "csr", "dia"}
        for name, matrix_format in matrix_formats.items():
            matrix_format = str(matrix_format).lower()
            if matrix_format not in valid:
                raise ValueError(f"unsupported matrix format {matrix_format!r}")
            self._matrix_formats[name] = matrix_format
        self._component_cache = None

    def __sub__(self, other):
        if not isinstance(other, quantum_operator):
            return NotImplemented
        if self.shape != other.shape:
            raise ValueError("quantum_operator dimensions do not match")
        left = self._quantum_operator
        right = other._quantum_operator
        components = {}
        for name in dict.fromkeys((*left, *right)):
            if name in left and name in right:
                difference = left[name] - right[name]
            elif name in left:
                difference = left[name].copy()
            else:
                difference = -right[name]
            if sp.issparse(difference):
                compressed = difference.tocsr()
                compressed.eliminate_zeros()
                nonzero = compressed.nnz != 0
            else:
                nonzero = bool(np.any(np.asarray(difference) != 0))
            if nonzero:
                components[name] = [difference]
        if not components:
            return _QuantumOperatorDifference(
                self.Ns,
                np.result_type(self.dtype, other.dtype),
            )
        return quantum_operator(
            components,
            dtype=np.result_type(self.dtype, other.dtype),
            check_herm=False,
            check_pcon=False,
            check_symm=False,
        )


def save_zip(archive, op, save_basis=True):
    if not isinstance(op, quantum_operator):
        raise TypeError("save_zip currently requires a quantum_operator")
    if not isinstance(save_basis, (bool, np.bool_)):
        raise TypeError("save_basis must be a boolean")
    return op._model.save_operator_archive(
        archive,
        formats=op._matrix_formats,
        metadata={"scalar_dtype": op.dtype.name},
    )


def load_zip(archive):
    return quantum_operator._from_archive(archive)


class _EvaluatedQuantumOperator:
    """Fixed-parameter view over a persistent named operator family."""

    def __init__(self, owner: quantum_operator, pars=None):
        self._owner = owner
        self._pars = {} if pars is None else dict(pars)
        self.Ns = owner.Ns
        self.dtype = owner.dtype
        self.basis = owner.basis

    @property
    def shape(self) -> tuple[int, int]:
        return self._owner.shape

    @property
    def get_shape(self) -> tuple[int, int]:
        return self.shape

    def dot(self, vector):
        return self._owner.dot(vector, pars=self._pars)

    def _expression(self, action="normal"):
        return _OperatorExpression.model(
            self._owner,
            action=action,
            parameters=self._pars,
        )

    def toarray(self):
        return self._owner.toarray(pars=self._pars)

    def todense(self):
        return self._owner.todense(pars=self._pars)

    def eigvalsh(self):
        return self._owner.eigvalsh(pars=self._pars)

    def eigh(self):
        return self._owner.eigh(pars=self._pars)

    def eigsh(self, **options):
        return self._owner.eigsh(pars=self._pars, **options)

    @property
    def T(self):
        return _OperatorExpression.model(
            self._owner,
            action="transpose",
            parameters=self._pars,
        )

    @property
    def H(self):
        return _OperatorExpression.model(
            self._owner,
            action="adjoint",
            parameters=self._pars,
        )

    def conj(self):
        return _OperatorExpression.model(
            self._owner,
            action="conjugate",
            parameters=self._pars,
        )

    conjugate = conj

    def diagonal(self):
        return self._expression().diagonal()

    def trace(self):
        return self._expression().trace()

    def rdot(self, vector):
        return self._expression().rdot(vector)

    def aslinearoperator(self):
        return self._expression().aslinearoperator()


def commutator(H1, H2):
    """Return the lazy Rust-backed commutator ``left*right - right*left``."""

    left = _as_operator_expression(H1)
    right = _as_operator_expression(H2)
    return left * right - right * left


def anti_commutator(H1, H2):
    """Return the lazy Rust-backed anticommutator ``left*right + right*left``."""

    left = _as_operator_expression(H1)
    right = _as_operator_expression(H2)
    return left * right + right * left


def ishamiltonian(obj):
    return isinstance(obj, hamiltonian)


def isquantum_operator(obj):
    return isinstance(obj, quantum_operator)


def isquantum_LinearOperator(obj):
    return isinstance(obj, quantum_LinearOperator)


def isexp_op(obj):
    return isinstance(obj, exp_op)


from . import (
    exp_op_core,
    hamiltonian_core,
    quantum_LinearOperator_core,
    quantum_operator_core,
)


__all__ = [
    "anti_commutator",
    "commutator",
    "exp_op",
    "exp_op_core",
    "hamiltonian",
    "hamiltonian_core",
    "isexp_op",
    "ishamiltonian",
    "isquantum_LinearOperator",
    "isquantum_operator",
    "load_zip",
    "quantum_LinearOperator",
    "quantum_LinearOperator_core",
    "quantum_operator",
    "quantum_operator_core",
    "save_zip",
]
