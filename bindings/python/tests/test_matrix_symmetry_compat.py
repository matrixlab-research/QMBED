import numpy as np
import scipy.sparse as sp

from quspin.basis import spin_basis_1d
from quspin.operators import hamiltonian


def _matrix_from_op(basis, opstr, sites, coefficient=1.0):
    values, rows, columns = basis.Op(
        opstr,
        sites,
        coefficient,
        np.complex128,
    )
    return sp.csc_matrix(
        (values, (rows, columns)),
        shape=(basis.Ns, basis.Ns),
    ).toarray()


def test_matrix_symmetry_keeps_physical_labels_projectors_and_temporary_terms():
    sector = spin_basis_1d(3, Nup=1, kblock=1, pblock=1)
    constrained = spin_basis_1d(3, Nup=1)
    full = spin_basis_1d(3)
    assert sector.Ns > 0
    assert not np.array_equal(sector.states, np.arange(sector.Ns))

    particle_projector = sector.get_proj(np.complex128, pcon=True)
    full_projector = sector.get_proj(np.complex128, pcon=False)
    assert particle_projector.shape == (constrained.Ns, sector.Ns)
    assert full_projector.shape == (full.Ns, sector.Ns)
    np.testing.assert_allclose(
        (particle_projector.conjugate().T @ particle_projector).toarray(),
        np.eye(sector.Ns),
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        (full_projector.conjugate().T @ full_projector).toarray(),
        np.eye(sector.Ns),
        atol=1.0e-12,
    )

    parent_local = _matrix_from_op(constrained, "z", [0])
    reduced_local = _matrix_from_op(sector, "z", [0])
    np.testing.assert_allclose(
        reduced_local,
        particle_projector.conjugate().T
        @ parent_local
        @ particle_projector,
        atol=1.0e-12,
    )
    vector = np.arange(1, sector.Ns + 1, dtype=np.complex128)
    vector += 0.2j
    np.testing.assert_allclose(
        sector.inplace_Op(
            vector,
            [["z", [0], 1.0]],
            np.complex128,
        ),
        reduced_local @ vector,
        atol=1.0e-12,
    )

    normalized = vector / np.linalg.norm(vector)
    reduced_entropy = sector.ent_entropy(
        normalized,
        [0],
        density=False,
        return_rdm="A",
    )
    parent_entropy = full.ent_entropy(
        np.asarray(full_projector @ normalized),
        [0],
        density=False,
        return_rdm="A",
    )
    np.testing.assert_allclose(
        reduced_entropy["Sent_A"],
        parent_entropy["Sent_A"],
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        reduced_entropy["rdm_A"],
        parent_entropy["rdm_A"],
        atol=1.0e-12,
    )


def test_matrix_symmetry_dynamic_family_and_evolution_match_parent_projection():
    sites = 3
    bonds = [[0.7, site, (site + 1) % sites] for site in range(sites)]
    fields = [[0.2, site] for site in range(sites)]
    static = [["zz", bonds]]
    dynamic = [["z", fields, np.cos, ()]]
    sector = spin_basis_1d(sites, Nup=1, kblock=1, pblock=-1)
    parent = spin_basis_1d(sites)
    projector = sector.get_proj(np.complex128)
    reduced_hamiltonian = hamiltonian(
        static,
        dynamic,
        basis=sector,
        dtype=np.complex128,
    )
    parent_hamiltonian = hamiltonian(
        static,
        dynamic,
        basis=parent,
        dtype=np.complex128,
        check_symm=False,
    )

    for time in (0.0, 0.31, 1.2):
        np.testing.assert_allclose(
            reduced_hamiltonian.toarray(time),
            projector.conjugate().T
            @ parent_hamiltonian.toarray(time)
            @ projector,
            atol=1.0e-12,
        )

    reduced_initial = np.arange(1, sector.Ns + 1, dtype=np.complex128)
    reduced_initial += 0.1j
    reduced_initial /= np.linalg.norm(reduced_initial)
    times = np.asarray([0.0, 0.03, 0.07])
    reduced_trajectory = reduced_hamiltonian.evolve(
        reduced_initial,
        0.0,
        times,
        solver_name="qmbed",
        atol=1.0e-11,
        rtol=1.0e-11,
    )
    parent_trajectory = parent_hamiltonian.evolve(
        np.asarray(projector @ reduced_initial),
        0.0,
        times,
        solver_name="qmbed",
        atol=1.0e-11,
        rtol=1.0e-11,
    )
    np.testing.assert_allclose(
        reduced_trajectory,
        projector.conjugate().T @ parent_trajectory,
        atol=2.0e-9,
    )


def test_matrix_symmetry_cross_sector_actions_share_the_projector_kernel():
    source = spin_basis_1d(3, Nup=1, kblock=1, pblock=1)
    target = spin_basis_1d(3, Nup=1, kblock=1, pblock=-1)
    parent = spin_basis_1d(3)
    source_projector = source.get_proj(np.complex128)
    target_projector = target.get_proj(np.complex128)
    terms = [["z", [0], 1.3 - 0.2j]]
    source_vector = np.arange(1, source.Ns + 1, dtype=np.complex128) + 0.4j

    parent_input = np.asarray(source_projector @ source_vector)
    parent_output = parent.inplace_Op(parent_input, terms, np.complex128)
    expected_target = np.asarray(target_projector.conjugate().T @ parent_output)

    np.testing.assert_allclose(
        target.Op_shift_sector(source, terms, source_vector),
        expected_target,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        parent.Op_shift_sector(source, terms, source_vector),
        parent_output,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        target.Op_shift_sector(parent, terms, parent_input),
        expected_target,
        atol=1.0e-12,
    )


def test_wide_matrix_symmetry_uses_the_same_projected_operator_contract():
    sites = 200
    sector = spin_basis_1d(sites, Nup=1, kblock=1, pblock=1)
    parent = spin_basis_1d(sites, Nup=1)

    assert sector.Ns == 1
    projector = sector.get_proj(np.complex128, pcon=True)
    assert projector.shape == (sites, 1)
    np.testing.assert_allclose(
        (projector.conjugate().T @ projector).toarray(),
        np.eye(1),
        atol=1.0e-12,
    )

    number = hamiltonian(
        [["n", [[1.0, site] for site in range(sites)]]],
        [],
        basis=sector,
        dtype=np.float64,
    )
    np.testing.assert_allclose(number.toarray(), [[1.0]], atol=1.0e-12)
    np.testing.assert_allclose(number.eigvalsh(), [1.0], atol=1.0e-12)

    reduced = np.asarray([1.0 + 0.2j])
    lifted = np.asarray(projector @ reduced)
    terms = [["z", [0], 0.7 - 0.1j]]
    parent_output = parent.inplace_Op(lifted, terms, np.complex128)
    np.testing.assert_allclose(
        sector.Op_shift_sector(parent, terms, lifted),
        np.asarray(projector.conjugate().T @ parent_output),
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        parent.Op_shift_sector(sector, terms, reduced),
        parent_output,
        atol=1.0e-12,
    )
