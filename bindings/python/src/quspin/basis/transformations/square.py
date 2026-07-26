"""Square-lattice symmetry maps accepted by the general basis constructors."""

from __future__ import annotations

from itertools import product

import numpy as np

from .site_info import site_info_2d


class site_info_square(site_info_2d):
    def __init__(self, Lx, Ly):
        self._Lx = int(Lx)
        self._Ly = int(Ly)
        super().__init__(self._Lx, self._Ly)
        self._X = self.sites % self._Lx
        self._Y = self.sites // self._Lx

    @property
    def X(self):
        return self._X

    @property
    def Y(self):
        return self._Y


class square_lattice_trans:
    def __init__(self, Lx, Ly):
        self._Lx = int(Lx)
        self._Ly = int(Ly)
        if self._Lx <= 0 or self._Ly <= 0:
            raise ValueError("Lx and Ly must be positive")

        self._site_info = site_info_square(self._Lx, self._Ly)
        sites = self.site_info.sites
        X = self.site_info.X
        Y = self.site_info.Y

        self._Z = -(sites + 1)
        self._Z_A = np.asarray(
            [-(site + 1) if (x + y) % 2 == 0 else site
             for site, (x, y) in enumerate(zip(X, Y))]
        )
        self._Z_B = np.asarray(
            [-(site + 1) if (x + y) % 2 == 1 else site
             for site, (x, y) in enumerate(zip(X, Y))]
        )
        self._T_x = (X + 1) % self._Lx + Y * self._Lx
        self._T_y = X + ((Y + 1) % self._Ly) * self._Lx
        self._P_x = self._Lx - X - 1 + Y * self._Lx
        self._P_y = X + (self._Ly - Y - 1) * self._Lx
        if self._Lx == self._Ly:
            self._P_d = Y + self._Lx * X
            self._P_e = (
                self._Ly - Y - 1
                + self._Lx * (self._Lx - X - 1)
            )
        else:
            self._P_d = None
            self._P_e = None

    @property
    def site_info(self):
        return self._site_info

    @property
    def Z(self):
        return self._Z

    @property
    def Z_A(self):
        return self._Z_A

    @property
    def Z_B(self):
        return self._Z_B

    @property
    def T_x(self):
        return self._T_x

    @property
    def T_y(self):
        return self._T_y

    @property
    def P_x(self):
        return self._P_x

    @property
    def P_y(self):
        return self._P_y

    @property
    def P_e(self):
        if self._P_e is None:
            raise Exception("P_e symmetry only exsits for square lattice")
        return self._P_e

    @property
    def P_d(self):
        if self._P_d is None:
            raise Exception("P_d symmetry only exsits for square lattice")
        return self._P_d

    def allowed_blocks_spin_inversion_iter(self, Np, sps):
        half_filled = (
            Np == (int(sps) - 1) * (self._Lx * self._Ly) // 2
            and (self._Lx * self._Ly) % 2 == 0
        )
        if Np is None or half_filled:
            for blocks in self.allowed_blocks_iter():
                for zblock in range(2):
                    yield {**blocks, "zblock": (self._Z, zblock)}
        else:
            yield from self.allowed_blocks_iter()

    def allowed_blocks_iter_parity(self):
        for px, py in product(range(2), repeat=2):
            yield {
                "pxblock": (self._P_x, px),
                "pyblock": (self._P_y, py),
            }

    def allowed_blocks_iter(self):
        Lx = self._Lx
        Ly = self._Ly
        for kx, ky in product(
            range(-Lx // 2 + 1, Lx // 2 + 1),
            range(-Ly // 2 + 1, Ly // 2 + 1),
        ):
            common = {
                "kxblock": (self._T_x, kx),
                "kyblock": (self._T_y, ky),
            }
            if kx == 0:
                if ky == 0:
                    for px, py in product(range(2), repeat=2):
                        reflected = {
                            **common,
                            "pxblock": (self._P_x, px),
                            "pyblock": (self._P_y, py),
                        }
                        if px == py and Lx == Ly:
                            for pd in range(2):
                                yield {
                                    **reflected,
                                    "pdblock": (self._P_d, pd),
                                }
                        else:
                            yield reflected
                else:
                    for px in range(2):
                        yield {**common, "pxblock": (self._P_x, px)}
            elif kx == Lx // 2 and Lx % 2 == 0:
                if ky == Ly // 2 and Ly % 2 == 0:
                    for px, py in product(range(2), repeat=2):
                        reflected = {
                            **common,
                            "pxblock": (self._P_x, px),
                            "pyblock": (self._P_y, py),
                        }
                        if px == py and Lx == Ly:
                            for pd in range(2):
                                yield {
                                    **reflected,
                                    "pdblock": (self._P_d, pd),
                                }
                        else:
                            yield reflected
                else:
                    for px in range(2):
                        yield {**common, "pxblock": (self._P_x, px)}
            elif ky == 0 or (ky == Ly // 2 and Ly % 2 == 0):
                for py in range(2):
                    yield {**common, "pyblock": (self._P_y, py)}
            elif kx == ky and Lx == Ly:
                for pd in range(2):
                    yield {**common, "pdblock": (self._P_d, pd)}
            elif kx == -ky and Lx == Ly:
                for pe in range(2):
                    # QuSpin 1.0.1 uses the historical pdblock key here.
                    yield {**common, "pdblock": (self._P_e, pe)}
