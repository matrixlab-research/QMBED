import tempfile
import unittest

import numpy as np
import scipy.sparse as sp
import qmbed
from qmbed.compat import quspin
from qmbed._ffi import QmbedError, command
from quspin.basis import (
    basis_ones,
    basis_int_to_python_int,
    basis_zeros,
    bitwise_and,
    boson_basis_1d,
    boson_basis_general,
    coherent_state,
    get_basis_type,
    ho_basis,
    isbasis,
    photon_basis,
    photon_Hspace_dim,
    python_int_to_basis_int,
    spin_basis_1d,
    spin_basis_general,
    spinful_fermion_basis_1d,
    spinful_fermion_basis_general,
    spinless_fermion_basis_1d,
    spinless_fermion_basis_general,
    tensor_basis,
)
from quspin.operators import (
    anti_commutator,
    commutator,
    exp_op,
    hamiltonian,
    isexp_op,
    ishamiltonian,
    isquantum_operator,
    load_zip,
    quantum_LinearOperator,
    quantum_operator,
    save_zip,
)
from quspin.operators._make_hamiltonian import _consolidate_static
from quspin.tools.evolution import expm_multiply_parallel
from quspin.tools.lanczos import (
    FTLM_static_iteration,
    LTLM_static_iteration,
    lanczos_full,
    lanczos_iter,
)
from quspin.tools.measurements import diag_ensemble, ent_entropy, obs_vs_time
from quspin.tools.matvec import get_matvec_function, matvec


class QmbedBindingTests(unittest.TestCase):
    def test_native_and_compatibility_paths_share_the_rust_solver(self):
        basis = qmbed.SpinBasis(2)
        coupling = lambda value: qmbed.Coupling(value, (0, 1))
        terms = (
            qmbed.OperatorSpec(
                qmbed.OpProduct((qmbed.LocalOperator.Z, qmbed.LocalOperator.Z)),
                (coupling(1.0),),
            ),
            quspin.operator_term("+-", (coupling(0.5),)),
            quspin.operator_term("-+", (coupling(0.5),)),
        )
        result = qmbed.eigsh(basis, terms, qmbed.EigshOptions(2))
        self.assertEqual(result.dimension, 4)
        self.assertAlmostEqual(result.eigenvalues[0], -0.75, places=10)
        self.assertTrue(result.converged)

    def test_quspin_static_adapter(self):
        result = quspin.eigsh(
            qmbed.SpinBasis(2),
            [
                ("zz", [[1.0, 0, 1]]),
                ("+-", [[0.5, 0, 1]]),
                ("-+", [[0.5, 0, 1]]),
            ],
            k=2,
            which="SA",
        )
        self.assertAlmostEqual(result.eigenvalues[0], -0.75, places=10)

    def test_quspin_hamiltonian_reuses_and_releases_one_native_model(self):
        basis = spin_basis_1d(2)
        operator = hamiltonian(
            [["zz", [[1.0, 0, 1]]]],
            [],
            basis=basis,
            dtype=np.float64,
        )
        handle = operator._model.handle

        matrix = operator.toarray()
        eigenvalues = operator.eigvalsh()
        description = command({"operation": "describe_model", "handle": handle})

        self.assertEqual(matrix.shape, (4, 4))
        self.assertEqual(len(eigenvalues), 4)
        self.assertEqual(description["dimension"], 4)
        self.assertEqual(operator._model.handle, handle)

        operator.close()
        self.assertTrue(operator.closed)
        operator.close()
        with self.assertRaisesRegex(QmbedError, "model is closed"):
            operator.toarray()
        with self.assertRaisesRegex(QmbedError, "is not registered"):
            command({"operation": "describe_model", "handle": handle})

    def test_python_site_maps_reproduce_the_optimized_momentum_sector(self):
        sites = 6
        translation = (np.arange(sites) + 1) % sites
        general = spin_basis_general(
            sites,
            Nup=3,
            pauli=False,
            translation=(translation, 1),
        )
        optimized = spin_basis_1d(
            sites,
            Nup=3,
            pauli=False,
            kblock=1,
        )

        np.testing.assert_array_equal(general.states, optimized.states)
        static = [
            [
                "+-",
                [[1.0j, site, (site + 1) % sites] for site in range(sites)],
            ],
            [
                "-+",
                [[-1.0j, site, (site + 1) % sites] for site in range(sites)],
            ],
        ]
        general_operator = hamiltonian(
            static,
            [],
            basis=general,
            check_herm=False,
            check_pcon=False,
            check_symm=False,
        )
        optimized_operator = hamiltonian(
            static,
            [],
            basis=optimized,
            check_herm=False,
            check_pcon=False,
            check_symm=False,
        )
        np.testing.assert_allclose(
            general_operator.toarray(),
            optimized_operator.toarray(),
            atol=1.0e-12,
        )

    def test_wide_general_basis_builds_and_solves_through_one_native_model(self):
        sites = 200
        translation = (np.arange(sites) + 1) % sites
        basis = spin_basis_general(
            sites,
            Nup=1,
            pauli=False,
            translation=(translation, 0),
        )
        self.assertEqual(basis.Ns, 1)
        projector = basis.get_proj(np.complex128, pcon=True)
        self.assertEqual(projector.shape, (sites, 1))
        np.testing.assert_allclose(
            (projector.conjugate().T @ projector).toarray(),
            [[1.0]],
        )
        operator = hamiltonian(
            [["n", [[1.0, site] for site in range(sites)]]],
            [],
            basis=basis,
            dtype=np.float64,
        )

        np.testing.assert_allclose(operator.toarray(), [[1.0]])
        np.testing.assert_allclose(operator.eigvalsh(), [1.0])
        description = command(
            {"operation": "describe_model", "handle": operator._model.handle}
        )
        self.assertEqual(description["dimension"], 1)

    def test_general_fermion_particle_hole_maps_use_the_native_fock_phase(self):
        sites = 4
        permutation = (np.arange(sites) + 1) % sites
        particle_hole = -(permutation + 1)
        parent = spinless_fermion_basis_general(sites, Nf=sites // 2)
        parent_states = [int(state) for state in parent.states]
        rows = {state: index for index, state in enumerate(parent_states)}
        transformation = np.zeros((parent.Ns, parent.Ns), dtype=np.complex128)

        for column, state in enumerate(parent_states):
            digits = [
                (state >> (sites - source - 1)) & 1
                for source in range(sites)
            ]
            mapped_digits = [0] * sites
            occupied_destinations = []
            for source in range(sites - 1, -1, -1):
                destination = int(permutation[source])
                occupation = digits[source]
                mapped_digits[destination] = 1 - occupation
                if occupation:
                    occupied_destinations.append(destination)
            swaps = sum(
                left < right
                for index, left in enumerate(occupied_destinations)
                for right in occupied_destinations[index + 1 :]
            )
            mapped = sum(
                digit << (sites - destination - 1)
                for destination, digit in enumerate(mapped_digits)
            )
            transformation[rows[mapped], column] = -1.0 if swaps % 2 else 1.0

        projectors = []
        expected_representatives = ([12, 10], [12], [12, 5], [12])
        for sector in range(4):
            basis = spinless_fermion_basis_general(
                sites,
                Nf=sites // 2,
                phblock=(particle_hole, sector),
            )
            self.assertEqual(
                [int(state) for state in basis.states],
                expected_representatives[sector],
            )
            projector = basis.get_proj(np.complex128, pcon=True).toarray()
            eigenvalue = np.exp(2.0j * np.pi * sector / 4.0)
            np.testing.assert_allclose(
                transformation @ projector,
                eigenvalue * projector,
                atol=1.0e-12,
            )
            projectors.append(projector)

        self.assertEqual(sum(projector.shape[1] for projector in projectors), parent.Ns)
        combined = np.hstack(projectors)
        np.testing.assert_allclose(
            combined.conj().T @ combined,
            np.eye(parent.Ns),
            atol=1.0e-12,
        )

    def test_low_level_basis_operations_share_one_rust_action_protocol(self):
        basis = spin_basis_general(2, pauli=-1)
        static = [["y", [[0.75, 0]]], ["+", [[-0.5j, 1]]]]
        op_list = _consolidate_static(static)
        operator = hamiltonian(
            static,
            [],
            basis=basis,
            check_herm=False,
            check_pcon=False,
            check_symm=False,
        ).toarray()
        vector = np.asarray([1 + 0.5j, -2j, 0.25, -0.5 + 0.75j])

        actions = [
            (False, False, operator),
            (True, False, operator.T),
            (False, True, operator.conj()),
            (True, True, operator.conj().T),
        ]
        for transposed, conjugated, matrix in actions:
            actual = basis.inplace_Op(
                vector,
                op_list,
                np.complex128,
                transposed=transposed,
                conjugated=conjugated,
            )
            np.testing.assert_allclose(actual, matrix.dot(vector), atol=1.0e-12)

        initial = np.ones_like(vector)
        returned = basis.inplace_Op(
            vector,
            op_list,
            np.complex128,
            v_out=initial,
        )
        self.assertIs(returned, initial)
        np.testing.assert_allclose(returned, 1.0 + operator.dot(vector), atol=1.0e-12)

        elements, rows, columns = basis.Op("y", [0], 0.75, np.complex128)
        reconstructed = np.zeros_like(operator)
        reconstructed[rows, columns] = elements
        expected = hamiltonian(
            [["y", [[0.75, 0]]]],
            [],
            basis=basis,
            check_herm=False,
            check_pcon=False,
            check_symm=False,
        ).toarray()
        np.testing.assert_allclose(reconstructed, expected, atol=1.0e-12)

        elements, bras, kets = basis.Op_bra_ket(
            "+",
            [0],
            1.5,
            np.float64,
            basis.states,
        )
        self.assertTrue(np.all(elements == 1.5))
        self.assertTrue(np.all(bras > kets))
        self.assertTrue(all(basis_int_to_python_int(value) == int(value) for value in bras))

    def test_python_pauli_modes_map_to_distinct_rust_normalizations(self):
        spin = spin_basis_1d(1, pauli=0)
        pauli = spin_basis_1d(1, pauli=1)
        cartesian = spin_basis_1d(1, pauli=-1)

        spin_raising = spin.Op("+", [0], 1.0, np.float64)[0]
        pauli_raising = pauli.Op("+", [0], 1.0, np.float64)[0]
        cartesian_raising = cartesian.Op("+", [0], 1.0, np.float64)[0]
        np.testing.assert_allclose(pauli_raising, 2.0 * spin_raising)
        np.testing.assert_allclose(cartesian_raising, spin_raising)

        spin_x = spin.Op("x", [0], 1.0, np.float64)[0]
        pauli_x = pauli.Op("x", [0], 1.0, np.float64)[0]
        cartesian_x = cartesian.Op("x", [0], 1.0, np.float64)[0]
        np.testing.assert_allclose(pauli_x, 2.0 * spin_x)
        np.testing.assert_allclose(cartesian_x, 2.0 * spin_x)

    def test_recursive_tensor_basis_uses_one_runtime_rust_basis(self):
        factor = spin_basis_1d(1, pauli=False)
        basis = tensor_basis(tensor_basis(factor, factor), factor)
        operator = hamiltonian(
            [
                ["z||", [[1.0, 0]]],
                ["|z|", [[2.0, 0]]],
                ["||z", [[4.0, 0]]],
            ],
            [],
            basis=basis,
            check_pcon=False,
            check_symm=False,
        )

        self.assertEqual(basis.Ns, 8)
        np.testing.assert_allclose(
            operator.eigvalsh(),
            [-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5],
            atol=1.0e-12,
        )

    def test_photon_basis_uses_native_total_excitation_and_embedding_protocols(self):
        basis = photon_basis(spin_basis_1d, 2, Ntot=2, pauli=False)
        full = photon_basis(spin_basis_1d, 2, Nph=2, pauli=False)
        self.assertEqual(basis.Ns, 4)
        self.assertEqual(full.Ns, 12)

        projector = basis.get_proj(np.complex128, full_part=False)
        self.assertEqual(projector.shape, (12, 4))
        np.testing.assert_allclose(
            (projector.conj().T @ projector).toarray(),
            np.eye(4),
            atol=1.0e-12,
        )

        exchange = hamiltonian(
            [
                ["+|-", [[1.0, 0]]],
                ["-|+", [[1.0, 0]]],
                ["|n", [[0.25]]],
            ],
            [],
            basis=basis,
            check_symm=False,
        )
        self.assertGreater(exchange.tocsr().nnz, 0)

        state = np.asarray([1.0, 2.0j, -0.5, 0.25j], dtype=np.complex128)
        state /= np.linalg.norm(state)
        lifted = basis.get_vec(state, sparse=False, full_part=False)
        reduced = basis.ent_entropy(
            state,
            sub_sys_A="particles",
            return_rdm="both",
        )
        expanded = full.ent_entropy(
            lifted,
            sub_sys_A="particles",
            return_rdm="both",
        )
        np.testing.assert_allclose(reduced["Sent_A"], expanded["Sent_A"], atol=1.0e-12)
        np.testing.assert_allclose(reduced["rdm_A"], expanded["rdm_A"], atol=1.0e-12)
        np.testing.assert_allclose(reduced["rdm_B"], expanded["rdm_B"], atol=1.0e-12)

    def test_reusable_nonnormal_exponential_action_supports_column_batches(self):
        generator = np.asarray(
            [
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0],
            ]
        )
        vectors = np.asarray(
            [
                [0.0, 1.0],
                [0.0, 0.0],
                [1.0, 0.0],
            ]
        )
        with expm_multiply_parallel(generator, n_jobs=2) as action:
            actual = action.dot(vectors)
            np.testing.assert_allclose(
                actual,
                [[0.5, 1.0], [1.0, 0.0], [1.0, 0.0]],
                atol=1.0e-14,
            )
            self.assertIs(action.dot(vectors, out=actual), actual)
        self.assertTrue(action.closed)

    def test_projection_uses_explicit_full_and_particle_parent_models(self):
        sites = 4
        translation = (np.arange(sites) + 1) % sites
        basis = spin_basis_general(
            sites,
            Nup=2,
            pauli=False,
            translation=(translation, 1),
        )
        reduced = np.arange(1, basis.Ns + 1, dtype=np.complex128)

        full_projector = basis.get_proj(np.complex128)
        particle_projector = basis.get_proj(np.complex128, pcon=True)
        self.assertEqual(full_projector.shape, (2**sites, basis.Ns))
        self.assertEqual(particle_projector.shape, (6, basis.Ns))
        np.testing.assert_allclose(
            basis.project_from(reduced, sparse=False),
            full_projector.dot(reduced),
            atol=1.0e-12,
        )
        np.testing.assert_allclose(
            basis.project_from(reduced, sparse=False, pcon=True),
            particle_projector.dot(reduced),
            atol=1.0e-12,
        )
        np.testing.assert_allclose(
            basis.project_to(full_projector.dot(reduced), sparse=False),
            reduced,
            atol=1.0e-12,
        )

    def test_reduction_queries_share_projector_coefficients(self):
        sites = 4
        translation = (np.arange(sites) + 1) % sites
        basis = spin_basis_general(
            sites,
            Nup=2,
            pauli=False,
            translation=(translation, 1),
        )
        parent = spin_basis_general(sites, Nup=2, pauli=False)
        states = parent.states.copy()
        representatives = basis.representative(states)
        projector = basis.get_proj(np.complex128, pcon=True).toarray()
        factors = basis.get_amp(states.copy(), mode="full_basis")

        for row, (representative, factor) in enumerate(
            zip(representatives, factors)
        ):
            matches = np.flatnonzero(basis.states == representative)
            if matches.size:
                np.testing.assert_allclose(
                    factor,
                    projector[row, int(matches[0])],
                    atol=1.0e-12,
                )
            else:
                self.assertEqual(factor, 0.0)

        norms = basis.normalization(basis.states)
        self.assertTrue(np.all(norms > 0))
        representative_factors = basis.get_amp(
            basis.states.copy(),
            mode="representative",
        )
        np.testing.assert_allclose(
            np.abs(representative_factors),
            1.0 / np.sqrt((4 * 4) / norms),
            atol=1.0e-12,
        )

        inplace = np.zeros_like(states)
        self.assertIsNone(basis.representative(states, out=inplace))
        np.testing.assert_array_equal(inplace, representatives)

    def test_compatibility_sector_shortcuts_reuse_rust_reduction_metadata(self):
        sites = 6
        translation = (np.arange(sites) + 1) % sites
        negative = spin_basis_1d(sites, Nup=-2, kblock=0, pauli=False)
        explicit = spin_basis_1d(sites, Nup=sites - 2, kblock=0, pauli=False)
        general = spin_basis_general(
            sites,
            Nup=-2,
            translation=(translation, 0),
            pauli=False,
        )

        np.testing.assert_array_equal(negative.states, explicit.states)
        np.testing.assert_array_equal(negative.states, general.states)
        np.testing.assert_allclose(
            negative._get_norms(np.float64),
            general._get_norms(np.float64),
            atol=1.0e-12,
        )
        np.testing.assert_array_equal(negative._n, general._n)

        operator = hamiltonian(
            [["z", [[1.0, 0]]]],
            [],
            N=sites,
            basis=negative,
            check_symm=False,
        )
        self.assertEqual(operator.Ns, negative.Ns)
        with self.assertRaisesRegex(ValueError, "N does not match"):
            hamiltonian([], [], N=sites + 1, basis=negative)

    def test_sector_shift_runs_directly_between_persistent_models(self):
        source = spin_basis_general(3, Nup=0, pauli=False)
        target = spin_basis_general(3, Nup=1, pauli=False)
        output = target.Op_shift_sector(
            source,
            [["+", [0], 2.0]],
            np.asarray([1.0], dtype=np.complex128),
        )

        self.assertEqual(output.shape, (target.Ns,))
        self.assertEqual(np.count_nonzero(np.abs(output) > 1.0e-14), 1)
        np.testing.assert_allclose(np.linalg.norm(output), 2.0, atol=1.0e-12)

    def test_static_hamiltonian_evolution_uses_the_registered_rust_model(self):
        basis = spin_basis_1d(1)
        operator = hamiltonian(
            [["I", [[2.0, 0]]]],
            [],
            basis=basis,
            dtype=np.float64,
        )
        initial = np.eye(operator.Ns, dtype=np.complex128)
        times = np.asarray([0.0, np.pi / 4.0])

        trajectory = operator.evolve(initial, 0.0, times, solver_name="qmbed")
        self.assertEqual(trajectory.shape, (operator.Ns, operator.Ns, 2))
        np.testing.assert_allclose(trajectory[..., 0], initial, atol=1.0e-12)
        np.testing.assert_allclose(trajectory[..., 1], -1.0j * initial, atol=1.0e-12)

        iterated = list(
            operator.evolve(
                initial[:, 0],
                0.0,
                times,
                solver_name="qmbed",
                iterate=True,
            )
        )
        self.assertEqual(len(iterated), 2)
        np.testing.assert_allclose(iterated[0], initial[:, 0], atol=1.0e-12)
        np.testing.assert_allclose(iterated[1], -1.0j * initial[:, 0], atol=1.0e-12)

        scalar = operator.evolve(
            initial[:, 0],
            0.0,
            np.pi / 4.0,
            solver_name="qmbed",
        )
        self.assertEqual(scalar.shape, (operator.Ns,))
        np.testing.assert_allclose(scalar, -1.0j * initial[:, 0], atol=1.0e-12)

    def test_direct_matrix_operator_family_uses_named_rust_coefficients(self):
        fixed = np.diag([1.0, 2.0, 3.0])
        field = np.diag([-1.0, 0.0, 1.0])
        operator = quantum_operator({"fixed": [fixed], "field": [field]})
        parameters = {"fixed": 1.0, "field": 2.0}

        np.testing.assert_allclose(
            operator.toarray(parameters),
            np.diag([-1.0, 2.0, 5.0]),
            atol=1.0e-12,
        )
        np.testing.assert_allclose(
            operator.dot(np.ones(3), pars=parameters),
            [-1.0, 2.0, 5.0],
            atol=1.0e-12,
        )
        np.testing.assert_allclose(
            operator.eigvalsh(pars=parameters),
            [-1.0, 2.0, 5.0],
            atol=1.0e-12,
        )

    def test_all_operator_paths_pass_eigsh_v0_to_the_rust_solver(self):
        dimension = 129
        diagonal = np.diag(np.arange(dimension, dtype=np.float64))
        initial = np.zeros(dimension, dtype=np.complex128)
        initial[-2:] = 1.0
        options = {
            "k": 1,
            "which": "SA",
            "ncv": 2,
            "maxiter": 2,
            "tol": 1.0e-12,
            "return_eigenvectors": False,
            "v0": initial,
        }

        fixed = hamiltonian([diagonal], [], dtype=np.float64)
        parameterized = quantum_operator({"diagonal": [diagonal]})
        expression = fixed + 0.0 * fixed

        for operator in (fixed, parameterized, expression):
            values = operator.eigsh(**options)
            np.testing.assert_allclose(values, [dimension - 2], atol=1.0e-10)

        with self.assertRaisesRegex(ValueError, "v0 must have shape"):
            fixed.eigsh(**{**options, "v0": np.ones(2)})
        with self.assertRaisesRegex(ValueError, "nonzero finite norm"):
            fixed.eigsh(**{**options, "v0": np.zeros(dimension)})

    def test_operator_archive_round_trip_preserves_components_formats_and_actions(self):
        diagonal = np.diag([1.0, -1.0])
        exchange = np.asarray([[0.0, 2.0], [2.0, 0.0]])
        operator = quantum_operator(
            {"diagonal": [diagonal], "exchange": [exchange]},
            matrix_formats={"diagonal": "dia", "exchange": "csr"},
        )
        parameters = {"diagonal": 1.25, "exchange": -0.5}

        with tempfile.TemporaryDirectory() as directory:
            path = f"{directory}/operator.zip"
            saved = save_zip(archive=path, op=operator)
            self.assertEqual(
                {entry["name"]: entry["format"] for entry in saved["components"]},
                {"diagonal": "dia", "exchange": "csr"},
            )
            restored = load_zip(path)
            self.assertEqual(
                restored._matrix_formats,
                {"diagonal": "dia", "exchange": "csr"},
            )
            np.testing.assert_allclose(
                restored.toarray(parameters),
                operator.toarray(parameters),
                atol=1.0e-12,
            )
            np.testing.assert_allclose(
                restored.dot(np.asarray([1.0, -0.5]), pars=parameters),
                operator.dot(np.asarray([1.0, -0.5]), pars=parameters),
                atol=1.0e-12,
            )
            self.assertEqual(len((operator - restored)._quantum_operator), 0)

    def test_basis_operator_family_uses_named_rust_term_components(self):
        basis = spin_basis_1d(2, pauli=False)
        components = {
            "field": [["z", [[1.0, 0], [-0.5, 1]]]],
            "exchange": [
                ["+-", [[0.5, 0, 1]]],
                ["-+", [[0.5, 0, 1]]],
            ],
        }
        operator = quantum_operator(
            components,
            basis=basis,
            dtype=np.complex128,
        )
        parameters = {"field": 0.25, "exchange": 2.0}
        evaluated = operator.tohamiltonian(parameters)
        expected = hamiltonian(
            [
                ["z", [[0.25, 0], [-0.125, 1]]],
                ["+-", [[1.0, 0, 1]]],
                ["-+", [[1.0, 0, 1]]],
            ],
            [],
            basis=basis,
            dtype=np.complex128,
        )

        np.testing.assert_allclose(
            evaluated.toarray(),
            expected.toarray(),
            atol=1.0e-12,
        )
        np.testing.assert_allclose(
            evaluated.dot(np.ones(basis.Ns)),
            expected.dot(np.ones(basis.Ns)),
            atol=1.0e-12,
        )
        np.testing.assert_allclose(
            evaluated.eigvalsh(),
            expected.eigvalsh(),
            atol=1.0e-12,
        )

    def test_basis_dynamic_hamiltonian_groups_equal_drives_in_rust(self):
        basis = spin_basis_1d(2, Nup=1, pauli=False)

        def drive(time, frequency):
            return np.cos(frequency * time)

        arguments = (0.75,)
        hopping = [[1.0, 0, 1]]
        operator = hamiltonian(
            [["z", [[0.2, 0], [-0.1, 1]]]],
            [
                ["+-", hopping, drive, arguments],
                ["-+", hopping, drive, arguments],
            ],
            basis=basis,
            dtype=np.complex128,
        )
        time = 0.4
        coefficient = drive(time, *arguments)
        expected = hamiltonian(
            [
                ["z", [[0.2, 0], [-0.1, 1]]],
                ["+-", [[coefficient, 0, 1]]],
                ["-+", [[coefficient, 0, 1]]],
            ],
            [],
            basis=basis,
            dtype=np.complex128,
        )

        self.assertEqual(len(operator._dynamic), 1)
        dynamic_matrix = next(iter(operator._dynamic.values()))
        np.testing.assert_allclose(
            (dynamic_matrix - dynamic_matrix.T.conj()).toarray(),
            0.0,
            atol=1.0e-12,
        )
        np.testing.assert_allclose(
            operator.toarray(time),
            expected.toarray(),
            atol=1.0e-12,
        )
        np.testing.assert_allclose(
            operator.eigvalsh(time),
            expected.eigvalsh(),
            atol=1.0e-12,
        )

    def test_dynamic_evolution_calls_python_drive_at_rust_internal_times(self):
        matrix = np.diag([1.0, -1.0])
        operator = hamiltonian([], [[matrix, np.cos, ()]])
        initial = np.full(2, 1.0 / np.sqrt(2.0), dtype=np.complex128)
        initial_time = 0.3
        times = np.asarray([initial_time, 0.7, 1.1])

        trajectory = operator.evolve(
            initial,
            initial_time,
            times,
            solver_name="qmbed",
            atol=1.0e-10,
            rtol=1.0e-10,
        )
        phase = np.sin(times[-1]) - np.sin(initial_time)
        expected = initial * np.exp(-1.0j * np.asarray([phase, -phase]))
        self.assertEqual(trajectory.shape, (2, 3))
        np.testing.assert_allclose(trajectory[:, 0], initial, atol=1.0e-12)
        np.testing.assert_allclose(trajectory[:, -1], expected, atol=3.0e-9)

        imaginary = operator.evolve(
            initial,
            initial_time,
            times,
            solver_name="qmbed",
            imag_time=True,
            atol=1.0e-10,
            rtol=1.0e-10,
        )
        expected_imaginary = initial * np.exp(-np.asarray([phase, -phase]))
        expected_imaginary /= np.linalg.norm(expected_imaginary)
        np.testing.assert_allclose(
            imaginary[:, -1],
            expected_imaginary,
            atol=3.0e-9,
        )

    def test_recursive_expression_eigh_and_density_evolution_share_native_actions(self):
        matrix = np.diag([1.0, -1.0])
        operator = hamiltonian([matrix], [], dtype=np.float64)
        expression = 0.25 * operator + 0.75 * operator
        energies, _ = expression.eigh()
        np.testing.assert_allclose(energies, [-1.0, 1.0], atol=1.0e-12)

        pure = np.asarray([1.0, 1.0], dtype=np.complex128) / np.sqrt(2.0)
        density = np.outer(pure, pure.conj())
        times = np.asarray([0.0, 0.3])
        trajectory = expression.evolve(
            density,
            0.0,
            times,
            eom="LvNE",
            atol=1.0e-11,
            rtol=1.0e-11,
        )
        self.assertEqual(trajectory.shape, (2, 2, 2))
        np.testing.assert_allclose(
            trajectory[0, 1, -1],
            0.5 * np.exp(-0.6j),
            atol=1.0e-10,
        )
        np.testing.assert_allclose(
            np.trace(trajectory[..., -1]),
            1.0,
            atol=1.0e-12,
        )

        observed = obs_vs_time(
            trajectory,
            times,
            {"z": matrix},
            return_state=True,
        )
        np.testing.assert_allclose(observed["z"], 0.0, atol=1.0e-12)
        np.testing.assert_allclose(observed["psi_t"], trajectory, atol=1.0e-12)

    def test_diagonal_ensemble_composes_native_probabilities_and_observables(self):
        energies = np.asarray([-1.0, 1.0])
        eigenvectors = np.eye(2)
        initial = np.asarray([1.0, 1.0]) / np.sqrt(2.0)
        observable = np.asarray([[0.0, 1.0], [1.0, 0.0]])
        result = diag_ensemble(
            1,
            initial,
            energies,
            eigenvectors,
            Obs=observable,
            delta_t_Obs=True,
            delta_q_Obs=True,
            Sd_Renyi=True,
            rho_d=True,
        )
        np.testing.assert_allclose(result["rho_d"], [0.5, 0.5], atol=1.0e-12)
        self.assertAlmostEqual(result["Obs_pure"], 0.0, places=12)
        self.assertAlmostEqual(
            result["delta_t_Obs_pure"],
            np.sqrt(0.5),
            places=12,
        )
        self.assertAlmostEqual(
            result["delta_q_Obs_pure"],
            np.sqrt(0.5),
            places=12,
        )
        self.assertAlmostEqual(result["Sd_pure"], np.log(2.0), places=12)

    def test_thermal_lanczos_reuses_native_ritz_data_and_supports_dot_objects(self):
        class DotOnly:
            def __init__(self, matrix):
                self.matrix = np.asarray(matrix)
                self.dtype = self.matrix.dtype

            def dot(self, vector):
                return self.matrix @ vector

        hamiltonian_matrix = np.diag([-1.0, 2.0])
        observable = np.asarray(
            [[1.0, 0.5j], [-0.25j, 2.0]],
            dtype=np.complex128,
        )
        initial = np.asarray([0.6, 0.8], dtype=np.complex128)
        energies, ritz_vectors, lanczos_basis = lanczos_full(
            hamiltonian_matrix,
            initial,
            2,
        )
        betas = np.asarray([[0.0], [0.7]])
        native_ftlm = FTLM_static_iteration(
            {"O": observable},
            energies,
            ritz_vectors,
            lanczos_basis,
            beta=betas,
        )
        projected_ftlm = FTLM_static_iteration(
            {"O": DotOnly(observable)},
            energies,
            ritz_vectors,
            lanczos_basis,
            beta=betas,
        )
        native_ltlm = LTLM_static_iteration(
            {"O": observable},
            energies,
            ritz_vectors,
            lanczos_basis,
            beta=betas,
        )
        projected_ltlm = LTLM_static_iteration(
            {"O": DotOnly(observable)},
            energies,
            ritz_vectors,
            np.asarray(lanczos_basis),
            beta=betas,
        )

        np.testing.assert_allclose(
            native_ftlm[0]["O"],
            projected_ftlm[0]["O"],
            atol=1.0e-12,
        )
        np.testing.assert_allclose(
            native_ftlm[1],
            projected_ftlm[1],
            atol=1.0e-12,
        )
        np.testing.assert_allclose(
            native_ltlm[0]["O"],
            projected_ltlm[0]["O"],
            atol=1.0e-12,
        )
        np.testing.assert_allclose(
            native_ltlm[1],
            projected_ltlm[1],
            atol=1.0e-12,
        )
        self.assertEqual(np.shape(native_ftlm[0]["O"]), (2,))
        scalar, identity = FTLM_static_iteration(
            {"O": observable},
            energies,
            ritz_vectors,
            lanczos_basis,
            beta=0.3,
        )
        self.assertEqual(np.shape(scalar["O"]), ())
        self.assertEqual(np.shape(identity), ())

    def test_operator_protocol_reuses_native_expression_actions_and_inspection(self):
        from scipy.linalg import expm

        left_matrix = np.asarray(
            [[1.0, 2.0 + 0.5j], [0.0, -1.0]],
            dtype=np.complex128,
        )
        right_matrix = np.asarray(
            [[0.0, 1.0], [3.0, 0.5]],
            dtype=np.complex128,
        )
        left = hamiltonian([left_matrix], [], dtype=np.complex128)
        right = hamiltonian([right_matrix], [], dtype=np.complex128)
        self.assertTrue(ishamiltonian(left))
        np.testing.assert_allclose(
            commutator(H1=left, H2=right).toarray(),
            left_matrix @ right_matrix - right_matrix @ left_matrix,
            atol=1.0e-12,
        )
        np.testing.assert_allclose(
            anti_commutator(H1=left, H2=right).toarray(),
            left_matrix @ right_matrix + right_matrix @ left_matrix,
            atol=1.0e-12,
        )
        vector = np.asarray([0.25 - 0.5j, 1.0])
        batch = np.column_stack([vector, vector.conj()])
        output = np.ones(2, dtype=np.complex128)
        left.dot(
            vector,
            out=output,
            overwrite_out=False,
            a=0.5,
        )
        np.testing.assert_allclose(output, 1.0 + 0.5 * left_matrix @ vector)
        np.testing.assert_allclose(
            (left * right).dot(batch),
            left_matrix @ right_matrix @ batch,
            atol=1.0e-12,
        )
        np.testing.assert_allclose(left.diagonal(), np.diag(left_matrix))
        self.assertAlmostEqual(left.trace(), np.trace(left_matrix))
        np.testing.assert_allclose(
            left.rdot(batch.T),
            batch.T @ left_matrix,
            atol=1.0e-12,
        )
        np.testing.assert_allclose(
            left.aslinearoperator().matmat(batch),
            left_matrix @ batch,
            atol=1.0e-12,
        )
        copied = left.copy()
        self.assertNotEqual(copied._model.handle, left._model.handle)
        np.testing.assert_allclose(copied.toarray(), left_matrix)
        dense = left.copy().as_dense_format()
        self.assertTrue(dense.is_dense)
        self.assertIsInstance(dense.static, np.ndarray)
        dense.as_sparse_format("csc")
        self.assertFalse(dense.is_dense)
        self.assertTrue(sp.isspmatrix_csc(dense.static))
        unitary = np.asarray([[0.0, 1.0], [1.0, 0.0]])
        np.testing.assert_allclose(
            left.rotate_by(unitary).toarray(),
            unitary.conj().T @ left_matrix @ unitary,
            atol=1.0e-12,
        )

        family = quantum_operator(
            {"left": [left_matrix], "right": [right_matrix]},
            dtype=np.complex128,
        )
        self.assertTrue(isquantum_operator(family))
        parameters = {"left": 0.25, "right": -0.5}
        evaluated = 0.25 * left_matrix - 0.5 * right_matrix
        np.testing.assert_allclose(family.diagonal(parameters), np.diag(evaluated))
        self.assertAlmostEqual(family.trace(parameters), np.trace(evaluated))
        np.testing.assert_allclose(
            family.rdot(batch.T, parameters),
            batch.T @ evaluated,
            atol=1.0e-12,
        )
        np.testing.assert_allclose(
            family.getH().toarray(parameters),
            evaluated.conj().T,
            atol=1.0e-12,
        )
        family_output = np.ones(2, dtype=np.complex128)
        self.assertIs(
            family.dot(
                vector,
                parameters,
                out=family_output,
                overwrite_out=False,
                a=0.5,
            ),
            family_output,
        )
        np.testing.assert_allclose(
            family_output,
            1.0 + 0.5 * evaluated @ vector,
        )
        family_dense = np.empty_like(evaluated)
        self.assertIs(family.toarray(parameters, out=family_dense), family_dense)
        np.testing.assert_allclose(family_dense, evaluated)

        linear = quantum_LinearOperator(
            [left_matrix],
            diagonal=np.asarray([0.5, -0.25]),
            dtype=np.complex128,
        )
        corrected = left_matrix + np.diag([0.5, -0.25])
        np.testing.assert_allclose(linear.matmat(batch), corrected @ batch)
        np.testing.assert_allclose(linear.rmatvec(vector), corrected.conj().T @ vector)
        np.testing.assert_allclose(linear.toarray(), corrected)
        np.testing.assert_allclose(linear.diagonal, [0.5, -0.25])
        linear.set_diagonal(np.asarray([-0.5, 0.25]))
        np.testing.assert_allclose(
            linear.toarray(),
            left_matrix + np.diag([-0.5, 0.25]),
        )
        linear_output = np.ones(2, dtype=np.complex128)
        self.assertIs(
            linear.dot(vector, out=linear_output, a=0.5),
            linear_output,
        )
        np.testing.assert_allclose(
            linear_output,
            0.5 * linear.toarray() @ vector,
        )
        np.testing.assert_allclose(linear.copy().toarray(), linear.toarray())

        exponential = exp_op(left_matrix, a=0.2 - 0.1j)
        self.assertTrue(isexp_op(exponential))
        expected_exponential = expm((0.2 - 0.1j) * left_matrix)
        np.testing.assert_allclose(
            exponential.get_mat(dense=True),
            expected_exponential,
            atol=1.0e-12,
        )
        np.testing.assert_allclose(
            exponential.rdot(batch.T),
            batch.T @ expected_exponential,
            atol=1.0e-12,
        )
        transposed = exponential.transpose(copy=True)
        np.testing.assert_allclose(
            transposed.get_mat(dense=True),
            expected_exponential.T,
            atol=1.0e-12,
        )
        exponential.set_grid(0.0, 1.0, num=3, endpoint=True)
        self.assertAlmostEqual(exponential.step, 0.5)
        self.assertEqual(exponential.dot(vector).shape, (2, 3))
        exponential.unset_grid()
        self.assertIsNone(exponential.grid)

        output = np.ones(2, dtype=np.complex128)
        matvec(left, vector, out=output, overwrite_out=True, a=0.5)
        np.testing.assert_allclose(output, 0.5 * left_matrix @ vector)
        selected_matvec = get_matvec_function(left)
        np.testing.assert_allclose(
            selected_matvec(left, vector),
            left_matrix @ vector,
        )

    def test_basis_convenience_protocol_matches_integer_and_photon_conventions(self):
        self.assertIs(get_basis_type(4, None, 2), np.uint32)
        self.assertIs(get_basis_type(40, None, 2), np.uint64)
        self.assertEqual(int(python_int_to_basis_int(2**40)), 2**40)
        np.testing.assert_array_equal(basis_zeros((2, 2)), np.zeros((2, 2)))
        np.testing.assert_array_equal(basis_ones(3), np.ones(3))
        np.testing.assert_array_equal(
            bitwise_and(
                x1=np.asarray([1, 2], dtype=np.uint32),
                x2=np.asarray([3, 3], dtype=np.uint32),
                where=None,
            ),
            [1, 2],
        )
        np.testing.assert_allclose(
            coherent_state(0.5, 4),
            [0.8824969025845955, 0.4412484512922977, 0.15600488604843067, 0.04503472926396058],
            atol=1.0e-14,
        )
        self.assertEqual(photon_Hspace_dim(4, None, 3), 64)
        self.assertEqual(photon_Hspace_dim(4, 3, 1), 15.0)
        self.assertEqual(photon_Hspace_dim(4, 3, 5), 15.0)
        basis = spin_basis_1d(4, Nup=2)
        for index, state in enumerate(basis.states):
            encoded = basis.int_to_state(state)
            self.assertEqual(basis.state_to_int(encoded), int(state))
            self.assertEqual(basis.index(encoded), index)

        entropy_basis = spin_basis_1d(2)
        bell = np.asarray([1.0, 0.0, 0.0, 1.0]) / np.sqrt(2.0)
        entropy = ent_entropy(
            bell,
            entropy_basis,
            chain_subsys=[0],
            DM=False,
            svd_return_vec=[False, True, False],
        )
        np.testing.assert_allclose(
            np.sort(entropy["lmbda"]),
            [1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0)],
            atol=1.0e-12,
        )
        mixed = ent_entropy(
            {"rho_d": [1.0, 0.0, 0.0, 0.0], "V_rho": np.eye(4)},
            entropy_basis,
            chain_subsys=[0],
        )
        self.assertIn("Sent_A", mixed)

    def test_documented_constructor_and_tool_protocols_reuse_native_models(self):
        self.assertEqual(boson_basis_1d(4, nb=0.5, sps=3).Ns, 10)
        self.assertEqual(
            boson_basis_general(4, nb=0.5, sps=3, Ns_block_est=10).Ns,
            10,
        )
        self.assertEqual(spinless_fermion_basis_1d(4, nf=0.5).Ns, 6)
        spinful_1d = spinful_fermion_basis_1d(3, nf=(1 / 3, 2 / 3))
        self.assertEqual((spinful_1d.N, spinful_1d.L, spinful_1d.Ns), (6, 3, 9))
        self.assertEqual(spinful_fermion_basis_1d(3, Nf=(1, 1)).index(1, 1), 8)
        spinful_general = spinful_fermion_basis_general(
            3,
            nf=(1 / 3, 2 / 3),
            Ns_block_est=9,
            simple_symm=True,
        )
        self.assertEqual((spinful_general.N, spinful_general.Ns), (6, 9))
        product = tensor_basis(spin_basis_1d(2), spinless_fermion_basis_1d(3))
        self.assertEqual((product.N, product.Ns), ((2, 3), 32))
        reduced_product = tensor_basis(
            spin_basis_1d(2, Nup=1),
            spin_basis_1d(1),
        )
        product_projector = reduced_product.get_proj(
            np.float64,
            full_left=True,
            full_right=False,
        )
        self.assertEqual(product_projector.shape, (8, 4))
        np.testing.assert_allclose(
            (product_projector.T @ product_projector).toarray(),
            np.eye(4),
        )
        coupled = photon_basis(spin_basis_1d, 2, Ntot=1, Nph=2)
        self.assertEqual((coupled.N, coupled.Ns, coupled.photon_Ns), ((2, 1), 3, 2))
        self.assertEqual(coupled.index(0, 0), 4)
        oscillator = ho_basis(Np=4)
        self.assertEqual(oscillator.Np, 4)
        self.assertEqual(oscillator.Ns, 5)
        self.assertTrue(isbasis(oscillator))

        zero = hamiltonian([], [], shape=(3, 3), static_fmt="dense")
        self.assertEqual(zero.shape, (3, 3))
        self.assertTrue(zero.is_dense)
        np.testing.assert_array_equal(zero.toarray(), np.zeros((3, 3)))

        family = quantum_operator(
            {"field": [["z", [[1.0, 0]]]]},
            N=2,
            dtype=np.float64,
        )
        self.assertEqual(family.shape, (4, 4))
        np.testing.assert_allclose(
            family.toarray({"field": 1.0}),
            np.diag([1.0, 1.0, -1.0, -1.0]),
        )

        matrix = sp.csr_matrix([[0.0, 1.0], [0.0, 0.0]])
        exponential = expm_multiply_parallel(matrix, a=0.0, copy=True)
        self.assertTrue(sp.isspmatrix_csr(exponential.A))
        np.testing.assert_allclose(exponential.dot([0.0, 1.0]), [0.0, 1.0])
        exponential.set_a(1.0)
        self.assertEqual(exponential.a, 1.0)
        np.testing.assert_allclose(exponential.dot([0.0, 1.0]), [1.0, 1.0])

        eigenvalues, eigenvectors = lanczos_iter(
            np.diag([0.0, 1.0, 2.0]),
            np.asarray([1.0, 1.0, 1.0]),
            2,
            return_vec_iter=False,
            copy_v0=True,
            copy_A=True,
        )
        self.assertEqual(eigenvalues.shape, (2,))
        self.assertEqual(eigenvectors.shape, (2, 2))


if __name__ == "__main__":
    unittest.main()
