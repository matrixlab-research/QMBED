import numpy as np

from quspin.basis import spin_basis_general
from quspin.basis.transformations import square_lattice_trans


def test_square_lattice_maps_feed_the_general_symmetry_reducer():
    transforms = square_lattice_trans(2, 2)

    np.testing.assert_array_equal(transforms.T_x, [1, 0, 3, 2])
    np.testing.assert_array_equal(transforms.T_y, [2, 3, 0, 1])
    np.testing.assert_array_equal(transforms.P_x, [1, 0, 3, 2])
    np.testing.assert_array_equal(transforms.P_y, [2, 3, 0, 1])
    np.testing.assert_array_equal(transforms.P_d, [0, 2, 1, 3])
    np.testing.assert_array_equal(transforms.P_e, [3, 1, 2, 0])
    np.testing.assert_array_equal(transforms.Z, [-1, -2, -3, -4])
    np.testing.assert_array_equal(transforms.Z_A, [-1, 1, 2, -4])
    np.testing.assert_array_equal(transforms.Z_B, [0, -2, -3, 3])

    basis = spin_basis_general(
        4,
        Nup=2,
        kxblock=(transforms.T_x, 0),
        kyblock=(transforms.T_y, 0),
    )
    assert basis.Ns == 3


def test_square_lattice_block_iterators_match_the_legacy_contract():
    transforms = square_lattice_trans(2, 2)

    parity_blocks = list(transforms.allowed_blocks_iter_parity())
    assert len(parity_blocks) == 4
    assert all(set(block) == {"pxblock", "pyblock"} for block in parity_blocks)

    blocks = list(transforms.allowed_blocks_iter())
    assert blocks
    assert all("kxblock" in block and "kyblock" in block for block in blocks)

    spin_blocks = list(transforms.allowed_blocks_spin_inversion_iter(2, 2))
    assert len(spin_blocks) == 2 * len(blocks)
    assert all("zblock" in block for block in spin_blocks)
