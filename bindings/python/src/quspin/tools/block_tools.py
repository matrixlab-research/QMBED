"""Symmetry-block utilities backed by QMBED's reusable block operators."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from qmbed._ffi import NativeOperatorModel
from quspin.operators import _matrix_request, exp_op, hamiltonian


def _block_hamiltonian(hamiltonians, dtype):
    hamiltonians = tuple(hamiltonians)
    if not hamiltonians:
        raise ValueError("at least one nonempty Hamiltonian block is required")
    model = NativeOperatorModel.direct_sum(
        [operator._model for operator in hamiltonians],
        format="csc",
    )
    drives = dict(hamiltonians[0]._drives)
    for operator in hamiltonians[1:]:
        if tuple(operator._drives) != tuple(drives):
            raise ValueError("Hamiltonian blocks must have identical dynamic components")
    dynamic_matrices = {
        name: sp.block_diag(
            [
                operator._dynamic.get(
                    name,
                    sp.csr_matrix((operator.Ns, operator.Ns), dtype=dtype),
                )
                for operator in hamiltonians
            ],
            format="csr",
        )
        for name in drives
    }
    return hamiltonian._from_native_model(
        model,
        dtype=dtype,
        drives=drives,
        dynamic_matrices=dynamic_matrices,
    )


def _construct_blocks(
    blocks,
    static,
    dynamic,
    basis_con,
    basis_args,
    dtype,
    *,
    basis_kwargs,
    get_proj_kwargs,
    check_symm,
    check_herm,
    check_pcon,
):
    blocks = tuple(blocks)
    if not blocks:
        raise ValueError("blocks must contain at least one symmetry sector")
    if all(isinstance(block, hamiltonian) for block in blocks):
        return [(None, block.basis, block) for block in blocks]
    if not all(isinstance(block, dict) for block in blocks):
        raise ValueError(
            "blocks must be dictionaries containing symmetry sectors "
            "or Hamiltonian objects"
        )
    if basis_con is None or basis_args is None:
        raise ValueError("basis_con and basis_args are required for sector dictionaries")

    result = []
    checks = {
        "check_symm": bool(check_symm),
        "check_herm": bool(check_herm),
        "check_pcon": bool(check_pcon),
    }
    for index, block in enumerate(blocks):
        options = dict(basis_kwargs)
        options.update(block)
        basis = basis_con(*tuple(basis_args), **options)
        if basis.Ns == 0:
            continue
        operator = hamiltonian(
            static,
            dynamic,
            basis=basis,
            dtype=dtype,
            **(checks if index == 0 else {
                "check_symm": False,
                "check_herm": False,
                "check_pcon": False,
            }),
        )
        result.append((options, basis, operator))
    if not result:
        raise ValueError("all requested symmetry blocks are empty")
    return result


def block_diag_hamiltonian(
    blocks,
    static,
    dynamic,
    basis_con,
    basis_args,
    dtype,
    basis_kwargs={},
    get_proj_kwargs={},
    get_proj=True,
    check_symm=True,
    check_herm=True,
    check_pcon=True,
):
    """Build one parameterized direct-sum Hamiltonian and its block projector."""
    built = _construct_blocks(
        blocks,
        static,
        dynamic,
        basis_con,
        basis_args,
        dtype,
        basis_kwargs=dict(basis_kwargs),
        get_proj_kwargs=dict(get_proj_kwargs),
        check_symm=check_symm,
        check_herm=check_herm,
        check_pcon=check_pcon,
    )
    block_operator = _block_hamiltonian([entry[2] for entry in built], dtype)
    if not get_proj:
        return block_operator
    if any(entry[1] is None for entry in built):
        raise ValueError(
            "get_proj=True requires symmetry dictionaries with basis objects"
        )
    projectors = [
        basis.get_proj(dtype, **dict(get_proj_kwargs))
        for _, basis, _ in built
    ]
    return sp.hstack(projectors, format="csr"), block_operator


class block_ops:
    """Lazily decompose, evolve, and reconstruct states over symmetry sectors."""

    def __init__(
        self,
        blocks,
        static,
        dynamic,
        basis_con,
        basis_args,
        dtype,
        basis_kwargs={},
        get_proj_kwargs={},
        save_previous_data=True,
        compute_all_blocks=False,
        check_symm=True,
        check_herm=True,
        check_pcon=True,
    ):
        self._basis_dict = {}
        self._H_dict = {}
        self._P_dict = {}
        self._dtype = np.dtype(dtype)
        self._save = bool(save_previous_data)
        self._static = list(static)
        self._dynamic = list(dynamic)
        self._checks = {
            "check_symm": bool(check_symm),
            "check_herm": bool(check_herm),
            "check_pcon": bool(check_pcon),
        }
        self._no_checks = {
            "check_symm": False,
            "check_herm": False,
            "check_pcon": False,
        }
        self._checked = False
        self._get_proj_kwargs = dict(get_proj_kwargs)
        self._basis_kwargs = dict(basis_kwargs)
        self.update_blocks(blocks, basis_con, basis_args)
        if compute_all_blocks:
            self._save = True
            self.compute_all_blocks()

    @property
    def dtype(self):
        return self._dtype

    @property
    def save_previous_data(self):
        return self._save

    @property
    def H_dict(self):
        return self._H_dict

    @property
    def P_dict(self):
        return self._P_dict

    @property
    def basis_dict(self):
        return self._basis_dict

    @property
    def static(self):
        return list(self._static)

    @property
    def dynamic(self):
        return list(self._dynamic)

    def update_blocks(
        self,
        blocks,
        basis_con,
        basis_args,
        compute_all_blocks=False,
    ):
        for block in blocks:
            if not isinstance(block, dict):
                raise ValueError("block_ops sectors must be dictionaries")
            options = dict(self._basis_kwargs)
            options.update(block)
            key = str(options)
            if key in self._basis_dict:
                continue
            basis = basis_con(*tuple(basis_args), **options)
            if basis.Ns > 0:
                self._basis_dict[key] = basis
        if not self._basis_dict:
            raise ValueError("all requested symmetry blocks are empty")
        if compute_all_blocks:
            self.compute_all_blocks()

    def compute_all_blocks(self):
        self._save = True
        for key in self._basis_dict:
            self._get_P(key)
            self._get_H(key)

    def _get_P(self, key):
        projector = self._P_dict.get(key)
        if projector is None:
            projector = self._basis_dict[key].get_proj(
                self._dtype,
                **self._get_proj_kwargs,
            )
            if self._save:
                self._P_dict[key] = projector
        return projector

    def _get_H(self, key):
        operator = self._H_dict.get(key)
        if operator is None:
            checks = self._checks if not self._checked else self._no_checks
            operator = hamiltonian(
                self._static,
                self._dynamic,
                basis=self._basis_dict[key],
                dtype=self._dtype,
                **checks,
            )
            self._checked = True
            if self._save:
                self._H_dict[key] = operator
        return operator

    def _active_direct_sum(self, psi_0):
        state = (
            psi_0.toarray()
            if sp.issparse(psi_0)
            else np.asarray(psi_0)
        )
        if state.ndim == 2 and 1 in state.shape:
            state = state.reshape(-1)
        if state.ndim != 1:
            raise ValueError("block evolution requires one full-space state vector")

        projectors = []
        operators = []
        coordinates = []
        threshold = 1000 * np.finfo(self._dtype).eps
        for key in self._basis_dict:
            projector = self._get_P(key)
            if projector.shape[0] != state.size:
                raise ValueError(
                    "initial state dimension does not match the block projectors"
                )
            sector_state = np.asarray(projector.conjugate().T @ state).reshape(-1)
            if np.linalg.norm(sector_state) > threshold:
                projectors.append(projector)
                operators.append(self._get_H(key))
                coordinates.append(sector_state)
        if not operators:
            raise RuntimeError(
                "initial state has no projection on to specified blocks"
            )
        projector = sp.hstack(projectors, format="csr")
        coordinate_state = np.concatenate(coordinates)
        return projector, _block_hamiltonian(operators, self._dtype), coordinate_state

    @staticmethod
    def _lift_trajectory(projector, trajectory, iterate):
        if iterate:
            return (np.asarray(projector @ state) for state in trajectory)
        values = np.asarray(trajectory)
        if values.ndim == 1:
            return np.asarray(projector @ values)
        return np.asarray(projector @ values)

    def evolve(
        self,
        psi_0,
        t0,
        times,
        iterate=False,
        n_jobs=1,
        block_diag=False,
        stack_state=False,
        imag_time=False,
        solver_name="dop853",
        **solver_args,
    ):
        if imag_time:
            raise ValueError("imaginary time not supported for block evolution")
        if int(n_jobs) <= 0:
            raise ValueError("n_jobs must be positive")
        del block_diag
        if iterate and np.isscalar(times):
            raise ValueError("If iterate=True times must be a list/array")
        projector, operator, state = self._active_direct_sum(psi_0)
        trajectory = operator.evolve(
            state,
            t0,
            times,
            iterate=iterate,
            stack_state=stack_state,
            imag_time=False,
            solver_name=solver_name,
            **solver_args,
        )
        return self._lift_trajectory(projector, trajectory, iterate)

    def expm(
        self,
        psi_0,
        H_time_eval=0.0,
        iterate=False,
        n_jobs=1,
        block_diag=False,
        a=-1j,
        start=None,
        stop=None,
        endpoint=None,
        num=None,
        shift=None,
    ):
        if int(n_jobs) <= 0:
            raise ValueError("n_jobs must be positive")
        del block_diag
        projector, operator, state = self._active_direct_sum(psi_0)
        exponential = exp_op(
            operator,
            a=a,
            start=start,
            stop=stop,
            num=num,
            endpoint=endpoint,
            iterate=iterate,
        )
        trajectory = exponential.dot(
            state,
            time=H_time_eval,
            shift=0.0 if shift is None else shift,
        )
        return self._lift_trajectory(projector, trajectory, iterate)


__all__ = ["block_diag_hamiltonian", "block_ops"]
