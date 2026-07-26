"""Read-only site-coordinate helpers used by lattice transformations."""

from __future__ import annotations

import numpy as np


class site_info:
    def __init__(self, N):
        self._N = int(N)
        self._sites = np.arange(self._N)

    @property
    def N(self):
        return self._N

    @property
    def sites(self):
        return self._sites


class site_info_2d(site_info):
    def __init__(self, Lx, Ly):
        super().__init__(int(Lx) * int(Ly))

    @property
    def coor_iter(self):
        return enumerate(zip(self._X, self._Y))
