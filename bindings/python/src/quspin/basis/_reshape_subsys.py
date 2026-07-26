"""Compatibility reshapes for pure lattice states.

The numerical subsystem analysis lives in Rust.  These two private helpers
remain NumPy/SciPy views because their contract is only to rearrange an
already-materialized Python array.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def _layout(sub_sys_A, L: int, sps: int):
    sites_a = tuple(int(site) for site in sub_sys_A)
    if len(set(sites_a)) != len(sites_a) or any(
        site < 0 or site >= int(L) for site in sites_a
    ):
        raise ValueError("sub_sys_A contains invalid or repeated sites")
    sites_b = tuple(site for site in range(int(L)) if site not in set(sites_a))
    dimension_a = int(sps) ** len(sites_a)
    dimension_b = int(sps) ** len(sites_b)
    return sites_a, sites_b, dimension_a, dimension_b


def _lattice_reshape_pure(psi, sub_sys_A, L, sps):
    array = np.asanyarray(psi)
    sites_a, sites_b, dimension_a, dimension_b = _layout(sub_sys_A, L, sps)
    expected = int(sps) ** int(L)
    if array.ndim == 0 or array.shape[-1] != expected:
        raise ValueError("the final state axis does not match the lattice Hilbert space")
    extra = array.shape[:-1]
    axes = tuple(range(len(extra))) + tuple(
        len(extra) + site for site in sites_a + sites_b
    )
    return (
        array.reshape(extra + (int(sps),) * int(L))
        .transpose(axes)
        .reshape(extra + (dimension_a, dimension_b))
    )


def _lattice_reshape_sparse_pure(psi, sub_sys_A, L, sps):
    if not sp.issparse(psi):
        raise TypeError("psi must be a SciPy sparse matrix")
    sites_a, sites_b, dimension_a, dimension_b = _layout(sub_sys_A, L, sps)
    expected = int(sps) ** int(L)
    matrix = psi.tocoo(copy=True)
    if matrix.shape not in {(1, expected), (expected, 1)}:
        raise ValueError("sparse pure state must have one Hilbert-space axis")
    packed = matrix.col if matrix.shape[1] == expected else matrix.row
    reordered = np.zeros_like(packed)
    for new_site, old_site in enumerate(sites_a + sites_b):
        digit = (packed // (int(sps) ** (int(L) - old_site - 1))) % int(sps)
        reordered += digit * (int(sps) ** (int(L) - new_site - 1))
    rows = reordered // dimension_b
    columns = reordered % dimension_b
    return sp.csr_matrix(
        (matrix.data, (rows, columns)),
        shape=(dimension_a, dimension_b),
    )
