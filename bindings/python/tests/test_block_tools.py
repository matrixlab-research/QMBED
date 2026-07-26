import numpy as np

from quspin.basis import spin_basis_1d
from quspin.operators import exp_op, hamiltonian
from quspin.tools.block_tools import block_diag_hamiltonian, block_ops


def test_block_diagonal_family_is_unitarily_equivalent_to_the_full_model():
    sites = 4
    exchange = [[1.0, site, (site + 1) % sites] for site in range(sites)]
    static = [["xx", exchange], ["yy", exchange], ["zz", exchange]]
    full = hamiltonian(
        static,
        [],
        N=sites,
        dtype=np.complex128,
        check_symm=False,
    )
    projector, blocked = block_diag_hamiltonian(
        [{"Nup": particles} for particles in range(sites + 1)],
        static,
        [],
        spin_basis_1d,
        (sites,),
        np.complex128,
    )

    identity = np.eye(full.Ns)
    np.testing.assert_allclose(
        (projector.conjugate().T @ projector).toarray(),
        identity,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        (projector @ projector.conjugate().T).toarray(),
        identity,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        projector @ blocked.toarray() @ projector.conjugate().T,
        full.toarray(),
        atol=1.0e-12,
    )


def test_block_ops_dynamic_evolution_and_exponential_match_full_space():
    sites = 4
    bonds = [[1.0, site, (site + 1) % sites] for site in range(sites)]
    fields = [[0.4, site] for site in range(sites)]
    static = [["zz", bonds]]
    dynamic = [["x", fields, np.sin, ()]]
    full = hamiltonian(
        static,
        dynamic,
        N=sites,
        dtype=np.complex128,
        check_symm=False,
    )
    blocked = block_ops(
        [{"kblock": momentum} for momentum in range(sites)],
        static,
        dynamic,
        spin_basis_1d,
        (sites,),
        np.complex128,
        compute_all_blocks=True,
    )
    generator = np.random.default_rng(17)
    initial = generator.normal(size=full.Ns) + 1j * generator.normal(
        size=full.Ns
    )
    initial /= np.linalg.norm(initial)
    times = np.asarray([0.0, 0.04, 0.08])

    exact = full.evolve(
        initial,
        0.0,
        times,
        solver_name="qmbed",
        atol=1.0e-11,
        rtol=1.0e-11,
    )
    decomposed = blocked.evolve(
        initial,
        0.0,
        times,
        solver_name="qmbed",
        atol=1.0e-11,
        rtol=1.0e-11,
    )
    np.testing.assert_allclose(decomposed, exact, atol=2.0e-9)

    exact_exponential = exp_op(
        full,
        a=-1j,
        start=0.0,
        stop=0.08,
        num=3,
        endpoint=True,
    ).dot(initial, time=0.23)
    block_exponential = blocked.expm(
        initial,
        H_time_eval=0.23,
        a=-1j,
        start=0.0,
        stop=0.08,
        num=3,
        endpoint=True,
    )
    np.testing.assert_allclose(block_exponential, exact_exponential, atol=1.0e-10)
