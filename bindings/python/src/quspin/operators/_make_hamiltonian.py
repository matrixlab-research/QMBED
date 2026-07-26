from __future__ import annotations

import numpy as np


def _consolidate_static(static_list):
    """Coalesce duplicate couplings using QuSpin's public helper semantics."""

    tolerance = 10 * np.finfo(np.float64).eps
    grouped = {}
    for opstr, bonds in static_list:
        operator = grouped.setdefault(opstr, {})
        for bond in bonds:
            coefficient, *sites = bond
            key = tuple(int(site) for site in sites)
            operator[key] = operator.get(key, 0) + coefficient

    return [
        (opstr, sites, coefficient)
        for opstr, couplings in grouped.items()
        for sites, coefficient in couplings.items()
        if np.abs(coefficient) > tolerance
    ]


__all__ = ["_consolidate_static"]
