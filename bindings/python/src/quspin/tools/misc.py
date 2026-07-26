"""Compatibility helpers built from QMBED's general operator contracts."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from .measurements import mean_level_spacing


def KL_div(p1, p2):
    left = np.asarray(p1, dtype=np.float64)
    right = np.asarray(p2, dtype=np.float64)
    if left.shape != right.shape:
        raise TypeError("Expecting p1 and p2 to have same shape")
    return np.sum(left * np.log(left / right), axis=0)


def project_op(Obs, proj, dtype=np.complex128):
    projector = proj.get_proj(dtype) if hasattr(proj, "get_proj") else proj
    projector = (
        projector.astype(dtype, copy=False)
        if sp.issparse(projector)
        else np.asarray(projector, dtype=dtype)
    )

    if hasattr(Obs, "project_to") and Obs.shape[0] == projector.shape[0]:
        projected = Obs.project_to(projector)
    else:
        operator = Obs if sp.issparse(Obs) else np.asarray(Obs)
        if operator.shape[0] == projector.shape[0]:
            projected = projector.conj().T @ operator @ projector
        elif operator.shape[0] == projector.shape[1]:
            projected = projector @ operator @ projector.conj().T
        else:
            raise ValueError("operator and projector dimensions do not match")
        if sp.issparse(projected):
            projected = projected.astype(dtype, copy=False)
        else:
            projected = np.asarray(projected, dtype=dtype)
    return {"Proj_Obs": projected}


def ints_to_array(basis_ints, N=None):
    values = np.asarray(basis_ints, dtype=object)
    if N is None:
        dtype = np.asarray(basis_ints).dtype
        if dtype != np.dtype(object) and np.issubdtype(dtype, np.integer):
            N = np.iinfo(dtype).bits
        else:
            maximum = max((int(value) for value in values.reshape(-1)), default=0)
            N = max(1, maximum.bit_length())
    flat = values.reshape(-1)
    rows = np.asarray(
        [
            [(int(value) >> (int(N) - site - 1)) & 1 for site in range(int(N))]
            for value in flat
        ],
        dtype=np.uint8,
    )
    return rows.reshape((*values.shape, int(N)))


def array_to_ints(state_array, dtype=None):
    rows = np.asarray(state_array, dtype=np.uint8)
    if rows.ndim == 1:
        rows = rows.reshape((1, -1))
    if rows.ndim != 2 or np.any(rows > 1):
        raise ValueError("state_array must be a two-dimensional binary array")
    values = [
        sum(int(bit) << (rows.shape[1] - site - 1) for site, bit in enumerate(row))
        for row in rows
    ]
    if dtype is None:
        dtype = object if rows.shape[1] > 64 else np.uint64
    return np.asarray(values, dtype=dtype)

from .matvec import get_matvec_function, matvec


__all__ = [
    "KL_div",
    "array_to_ints",
    "ints_to_array",
    "get_matvec_function",
    "matvec",
    "mean_level_spacing",
    "project_op",
]
