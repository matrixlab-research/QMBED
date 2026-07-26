"""Matrix-vector dispatch helpers over native or caller-owned operators."""

from __future__ import annotations

import numpy as np


def _apply(array, other):
    if hasattr(array, "dot"):
        return array.dot(other)
    return np.asarray(array) @ other


def _matvec(array, other, overwrite_out=False, out=None, a=1.0):
    result = complex(a) * _apply(array, other)
    result = np.real_if_close(result)
    if out is None:
        return result
    if np.shape(out) != np.shape(result):
        raise ValueError("out has the wrong shape")
    if overwrite_out:
        out[...] = result
    else:
        out[...] += result
    return out


def matvec(array, other, overwrite_out=False, out=None, a=1.0):
    return _matvec(
        array,
        other,
        overwrite_out=overwrite_out,
        out=out,
        a=a,
    )


def get_matvec_function(array):
    del array
    return _matvec


_get_matvec_function = get_matvec_function

__all__ = [
    "_get_matvec_function",
    "_matvec",
    "get_matvec_function",
    "matvec",
]
