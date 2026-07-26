use std::collections::HashMap;
use std::sync::Arc;

use approx::assert_abs_diff_eq;
use qmbed::basis::{SpinBasis1D, SpinfulFermionBasis1D};
use qmbed::operator::{
    AssemblyChecks, Coupling, DynamicTerm, ExpGrid, ExpOp, Hamiltonian, LinearOperator,
    MatrixFormat, Operator, OperatorBuilder, OperatorSpec, QuantumComponent, QuantumLinearOperator,
    QuantumOperator, Static, TimeDependentOperator, TimeOperator, anticommutator, commutator,
    get_matvec_function, is_exp_op, is_hamiltonian, is_quantum_linear_operator,
    is_quantum_operator, matmat, matvec, rmatmat, rmatvec,
};
use qmbed::solve::eigh;
use qmbed::{Complex64, QmbedError};

fn assert_complex_close(actual: Complex64, expected: Complex64) {
    assert_abs_diff_eq!(actual.re, expected.re, epsilon = 1.0e-12);
    assert_abs_diff_eq!(actual.im, expected.im, epsilon = 1.0e-12);
}

#[test]
fn dynamic_hamiltonian_evaluation_and_action_share_one_semantics() {
    let basis = SpinBasis1D::builder(1).pauli(true).build().unwrap();
    let static_term = OperatorSpec::new("z", [Coupling::new(0.25, vec![0])]).unwrap();
    let driven_term = DynamicTerm::new(
        OperatorSpec::new("x", [Coupling::new(1.0, vec![0])]).unwrap(),
        |time| Complex64::new(time.sin(), 0.0),
    );
    let hamiltonian = OperatorBuilder::on(&basis)
        .term(static_term)
        .build_dynamic([driven_term], MatrixFormat::Csc)
        .unwrap();
    assert_eq!(hamiltonian.dynamic_components(), 1);

    let time = std::f64::consts::FRAC_PI_2;
    let evaluated = hamiltonian.evaluate(time, MatrixFormat::Dense).unwrap();
    let mut matrix_free = vec![Complex64::new(0.0, 0.0); 2];
    hamiltonian
        .apply_at(
            time,
            &[Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
            &mut matrix_free,
        )
        .unwrap();
    let mut materialized = vec![Complex64::new(0.0, 0.0); 2];
    evaluated
        .apply(
            &[Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
            &mut materialized,
        )
        .unwrap();
    for (actual, expected) in matrix_free.into_iter().zip(materialized) {
        assert_complex_close(actual, expected);
    }
    for (transformed, expected) in [
        (
            hamiltonian.transpose().unwrap(),
            evaluated.transpose().unwrap(),
        ),
        (
            hamiltonian.conjugated().unwrap(),
            evaluated.conjugated().unwrap(),
        ),
        (hamiltonian.adjoint().unwrap(), evaluated.adjoint().unwrap()),
    ] {
        assert_eq!(
            transformed
                .evaluate(time, MatrixFormat::Dense)
                .unwrap()
                .to_dense(),
            expected.to_dense()
        );
    }
}

#[test]
fn quantum_operator_uses_required_and_default_parameters() {
    let identity = Operator::from_dense(
        2,
        2,
        vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
        ],
    )
    .unwrap();
    let sigma_z = Operator::from_dense(
        2,
        2,
        vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(-1.0, 0.0),
        ],
    )
    .unwrap();
    let parameterized = QuantumOperator::new([
        QuantumComponent::required("field", sigma_z),
        QuantumComponent::with_default("offset", identity, 2.0),
    ])
    .unwrap();

    let parameters = HashMap::from([("field".to_string(), Complex64::new(3.0, 0.0))]);
    let evaluated = parameterized
        .evaluate(&parameters, MatrixFormat::Csr)
        .unwrap();
    let diagonal = evaluated.diagonal();
    assert_complex_close(diagonal[0], Complex64::new(5.0, 0.0));
    assert_complex_close(diagonal[1], Complex64::new(-1.0, 0.0));

    let missing = parameterized
        .evaluate(&HashMap::new(), MatrixFormat::Dense)
        .unwrap_err();
    assert!(matches!(missing, QmbedError::InvalidOptions(_)));
}

#[test]
fn quantum_operator_plan_reuses_one_pattern_across_parameter_points() {
    let diagonal = Operator::from_triplets(
        3,
        3,
        [
            (0, 0, Complex64::new(1.0, 0.0)),
            (1, 1, Complex64::new(-2.0, 0.0)),
            (2, 2, Complex64::new(3.0, 0.0)),
        ],
        MatrixFormat::Csc,
    )
    .unwrap();
    let hopping = Operator::from_triplets(
        3,
        3,
        [
            (0, 1, Complex64::new(1.0, 0.0)),
            (1, 0, Complex64::new(1.0, 0.0)),
            (1, 2, Complex64::new(1.0, 0.0)),
            (2, 1, Complex64::new(1.0, 0.0)),
        ],
        MatrixFormat::Csc,
    )
    .unwrap();
    let family = QuantumOperator::new([
        QuantumComponent::required("field", diagonal),
        QuantumComponent::required("hop", hopping),
    ])
    .unwrap();
    let mut plan = family.plan(MatrixFormat::Csc).unwrap();
    let structural_bytes = plan.operator().memory_bytes();
    for (field, hop) in [(0.5, -1.0), (1.25, 0.3), (-0.7, 2.0)] {
        let parameters = HashMap::from([
            ("field".to_string(), Complex64::new(field, 0.0)),
            ("hop".to_string(), Complex64::new(hop, 0.0)),
        ]);
        let expected = family
            .evaluate(&parameters, MatrixFormat::Dense)
            .unwrap()
            .to_dense();
        let actual = plan.evaluate(&parameters).unwrap().to_dense();
        assert_eq!(actual, expected);
        assert_eq!(plan.operator().memory_bytes(), structural_bytes);
    }
}

#[test]
fn spinful_basis_accepts_unified_orbital_labels_without_losing_sector_checks() {
    let basis = SpinfulFermionBasis1D::builder(2)
        .particles(1, 1)
        .build()
        .unwrap();
    let component_checks = AssemblyChecks {
        hermiticity: false,
        particle_conservation: true,
        symmetry_compatibility: true,
    };
    let unified_down = OperatorBuilder::on(&basis)
        .term(OperatorSpec::new("+-", [Coupling::new(1.0, vec![2, 3])]).unwrap())
        .checks(component_checks)
        .build(MatrixFormat::Csc)
        .unwrap();
    let split_down = OperatorBuilder::on(&basis)
        .term(OperatorSpec::new("|+-", [Coupling::new(1.0, vec![0, 1])]).unwrap())
        .checks(component_checks)
        .build(MatrixFormat::Csc)
        .unwrap();
    assert_eq!(unified_down.triplets(), split_down.triplets());

    let cross_species = OperatorSpec::new("+-", [Coupling::new(1.0, vec![0, 2])]).unwrap();
    assert!(matches!(
        OperatorBuilder::on(&basis)
            .term(cross_species)
            .checks(component_checks)
            .build(MatrixFormat::Csc),
        Err(QmbedError::InvalidSector(_))
    ));
}

#[test]
fn operator_algebra_obeys_pauli_identities_and_format_conversion() {
    let x = Operator::from_dense(
        2,
        2,
        vec![
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
        ],
    )
    .unwrap()
    .converted(MatrixFormat::Csc)
    .unwrap();
    let y = Operator::from_dense(
        2,
        2,
        vec![
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, -1.0),
            Complex64::new(0.0, 1.0),
            Complex64::new(0.0, 0.0),
        ],
    )
    .unwrap()
    .converted(MatrixFormat::Csc)
    .unwrap();

    let xy_commutator = commutator(&x, &y).unwrap().to_dense();
    assert_complex_close(xy_commutator[0], Complex64::new(0.0, 2.0));
    assert_complex_close(xy_commutator[3], Complex64::new(0.0, -2.0));
    let xy_anticommutator = anticommutator(&x, &y).unwrap();
    assert_eq!(xy_anticommutator.nnz(), 0);
    let identity = x.pow(2).unwrap().to_dense();
    assert_complex_close(identity[0], Complex64::new(1.0, 0.0));
    assert_complex_close(identity[3], Complex64::new(1.0, 0.0));
    assert_eq!(x.adjoint().unwrap().to_dense(), x.to_dense());
    assert_eq!(x.transpose().unwrap().to_dense(), x.to_dense());
    assert_complex_close(x.trace().unwrap(), Complex64::new(0.0, 0.0));

    let rotation = Operator::from_dense(
        2,
        2,
        vec![
            Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
            Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
            Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
            Complex64::new(-1.0 / 2.0_f64.sqrt(), 0.0),
        ],
    )
    .unwrap();
    let rotated = x.rotated(&rotation, 1.0e-12).unwrap();
    assert_complex_close(rotated.diagonal()[0], Complex64::new(1.0, 0.0));
    assert_complex_close(rotated.diagonal()[1], Complex64::new(-1.0, 0.0));
}

#[test]
fn runtime_compatibility_predicates_identify_public_operator_families() {
    let operator = Operator::from_dense(1, 1, vec![Complex64::new(1.0, 0.0)]).unwrap();
    let exp = ExpOp::new(
        Arc::new(operator.clone()),
        Complex64::new(0.0, -1.0),
        4,
        1.0e-12,
        10,
    )
    .unwrap();
    let hamiltonian = Hamiltonian::<Static>::new(operator.clone()).unwrap();
    let quantum =
        QuantumOperator::new([QuantumComponent::with_default("a", operator, 1.0)]).unwrap();
    let linear = QuantumLinearOperator::from_operator(
        Operator::from_dense(1, 1, vec![Complex64::new(1.0, 0.0)]).unwrap(),
    )
    .unwrap();
    assert!(is_exp_op(&exp));
    assert!(is_hamiltonian(&hamiltonian));
    assert!(is_quantum_operator(&quantum));
    assert!(!is_quantum_linear_operator(&quantum));
    assert!(is_quantum_linear_operator(&linear));
    assert_eq!(quantum.component_names().collect::<Vec<_>>(), vec!["a"]);
}

#[test]
fn quantum_linear_operator_applies_and_replaces_its_diagonal_correction() {
    let base = Operator::from_dense(
        2,
        2,
        vec![
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
        ],
    )
    .unwrap();
    let mut linear = QuantumLinearOperator::new(
        base,
        vec![Complex64::new(0.2, 0.0), Complex64::new(-0.3, 0.0)],
    )
    .unwrap();
    let mut output = vec![Complex64::new(0.0, 0.0); 2];
    linear
        .apply(
            &[Complex64::new(2.0, 0.0), Complex64::new(-1.0, 0.0)],
            &mut output,
        )
        .unwrap();
    assert_complex_close(output[0], Complex64::new(-0.6, 0.0));
    assert_complex_close(output[1], Complex64::new(2.3, 0.0));
    linear
        .set_diagonal(vec![Complex64::new(0.0, 0.0); 2])
        .unwrap();
    assert_eq!(
        linear.materialize(MatrixFormat::Dense).unwrap().to_dense(),
        vec![
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
        ]
    );
}

#[test]
fn time_operator_algebra_matches_materialized_matrix_arithmetic() {
    let basis = SpinBasis1D::builder(1).pauli(true).build().unwrap();
    let dynamic = OperatorBuilder::on(&basis)
        .build_dynamic(
            [DynamicTerm::new(
                OperatorSpec::new("x", [Coupling::new(1.0, vec![0])]).unwrap(),
                |time| Complex64::new(time.sin(), 0.0),
            )],
            MatrixFormat::Csc,
        )
        .unwrap();
    let static_h = Hamiltonian::<Static>::new(
        OperatorBuilder::on(&basis)
            .term(OperatorSpec::new("z", [Coupling::new(0.3, vec![0])]).unwrap())
            .build(MatrixFormat::Csc)
            .unwrap(),
    )
    .unwrap();
    let left = TimeOperator::from_operator(Arc::new(dynamic));
    let right = TimeOperator::from_operator(Arc::new(static_h));
    let sum = left.add(&right).unwrap();
    let product = left.product(&right).unwrap();
    let commuted = left.commutator(&right).unwrap();
    for time in [0.0, 0.2, 0.9] {
        let left_matrix = left.evaluate(time, MatrixFormat::Dense).unwrap();
        let right_matrix = right.evaluate(time, MatrixFormat::Dense).unwrap();
        assert_eq!(
            sum.evaluate(time, MatrixFormat::Dense).unwrap().to_dense(),
            left_matrix.add(&right_matrix).unwrap().to_dense()
        );
        assert_eq!(
            product
                .evaluate(time, MatrixFormat::Dense)
                .unwrap()
                .to_dense(),
            left_matrix.product(&right_matrix).unwrap().to_dense()
        );
        assert_eq!(
            commuted
                .evaluate(time, MatrixFormat::Dense)
                .unwrap()
                .to_dense(),
            commutator(&left_matrix, &right_matrix).unwrap().to_dense()
        );
    }
}

#[test]
fn low_level_matvec_supports_scaled_overwrite_and_accumulation() {
    let operator = Arc::new(
        Operator::from_triplets(
            2,
            2,
            [
                (0, 0, Complex64::new(2.0, 0.0)),
                (1, 0, Complex64::new(3.0, 0.0)),
                (1, 1, Complex64::new(-1.0, 0.0)),
            ],
            MatrixFormat::Csc,
        )
        .unwrap(),
    );
    let input = [Complex64::new(1.0, 1.0), Complex64::new(2.0, 0.0)];
    let mut output = vec![Complex64::new(7.0, 0.0); 2];
    matvec(
        operator.as_ref(),
        &input,
        &mut output,
        Complex64::new(0.5, 0.0),
        true,
    )
    .unwrap();
    assert_eq!(
        output,
        vec![Complex64::new(1.0, 1.0), Complex64::new(0.5, 1.5)]
    );

    get_matvec_function(operator.clone())
        .apply(&input, &mut output, Complex64::new(2.0, 0.0), false)
        .unwrap();
    assert_eq!(
        output,
        vec![Complex64::new(5.0, 5.0), Complex64::new(2.5, 7.5)]
    );
    let columns = vec![input.to_vec(), vec![Complex64::new(0.0, 0.0); 2]];
    assert_eq!(
        matmat(operator.as_ref(), &columns).unwrap()[0][0],
        Complex64::new(2.0, 2.0)
    );
    assert_eq!(
        rmatvec(operator.as_ref(), &input).unwrap()[0],
        Complex64::new(8.0, 2.0)
    );
    assert_eq!(rmatmat(operator.as_ref(), &columns).unwrap().len(), 2);
}

#[test]
fn exp_op_grid_and_right_action_match_explicit_matrices() {
    let generator = Arc::new(
        Operator::from_dense(
            2,
            2,
            vec![
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
            ],
        )
        .unwrap(),
    );
    let exponential = ExpOp::new(generator, Complex64::new(1.0, 0.0), 8, 1.0e-13, 32).unwrap();
    let state = [Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)];
    let grid = ExpGrid::new(0.0, 1.0, 3, true).unwrap();
    let states = exponential.apply_grid(&state, grid).unwrap();
    assert_eq!(states.len(), 3);
    assert_complex_close(states[0][0], Complex64::new(1.0, 0.0));
    assert_complex_close(states[0][1], Complex64::new(2.0, 0.0));
    assert_complex_close(states[1][0], Complex64::new(2.0, 0.0));
    assert_complex_close(states[2][0], Complex64::new(3.0, 0.0));

    let explicit = exponential.matrix(MatrixFormat::Dense).unwrap();
    let right = exponential.right_apply(&state).unwrap();
    let expected = explicit.right_apply(&state).unwrap();
    for (actual, expected) in right.into_iter().zip(expected) {
        assert_complex_close(actual, expected);
    }
    let transformations = [
        (
            exponential.transpose().unwrap(),
            explicit.transpose().unwrap(),
        ),
        (
            exponential.conjugated().unwrap(),
            explicit.conjugated().unwrap(),
        ),
        (exponential.adjoint().unwrap(), explicit.adjoint().unwrap()),
    ];
    for (transformed, expected) in transformations {
        let actual = transformed.matrix(MatrixFormat::Dense).unwrap().to_dense();
        for (actual, expected) in actual.into_iter().zip(expected.to_dense()) {
            assert_complex_close(actual, expected);
        }
    }
}

#[test]
fn assembly_particle_check_rejects_sector_leakage_and_can_be_explicitly_disabled() {
    let basis = SpinBasis1D::builder(4).up(2).build().unwrap();
    let transverse = OperatorSpec::new("x", [Coupling::new(1.0, vec![0])]).unwrap();
    let error = OperatorBuilder::on(&basis)
        .term(transverse.clone())
        .build(MatrixFormat::Csc)
        .unwrap_err();
    assert!(matches!(error, QmbedError::InvalidSector(_)));

    let projected = OperatorBuilder::on(&basis)
        .checks(AssemblyChecks {
            hermiticity: false,
            particle_conservation: false,
            symmetry_compatibility: true,
        })
        .term(transverse)
        .build(MatrixFormat::Csc)
        .unwrap();
    assert_eq!(projected.nnz(), 0);
}

#[test]
fn assembly_particle_check_accounts_for_cancellation_across_the_operator_sum() {
    let basis = SpinBasis1D::builder(4).up(2).pauli(false).build().unwrap();
    let coupling = Coupling::new(0.5, vec![0, 1]);
    let cartesian = OperatorBuilder::on(&basis)
        .terms([
            OperatorSpec::new("xx", [coupling.clone()]).unwrap(),
            OperatorSpec::new("yy", [coupling]).unwrap(),
        ])
        .build(MatrixFormat::Dense)
        .unwrap();
    let ladder = OperatorBuilder::on(&basis)
        .terms([
            OperatorSpec::new("+-", [Coupling::new(0.25, vec![0, 1])]).unwrap(),
            OperatorSpec::new("-+", [Coupling::new(0.25, vec![0, 1])]).unwrap(),
        ])
        .build(MatrixFormat::Dense)
        .unwrap();
    assert_eq!(cartesian.to_dense(), ladder.to_dense());

    let residual_leakage = OperatorBuilder::on(&basis)
        .terms([
            OperatorSpec::new("xx", [Coupling::new(0.5, vec![0, 1])]).unwrap(),
            OperatorSpec::new("yy", [Coupling::new(0.25, vec![0, 1])]).unwrap(),
        ])
        .build(MatrixFormat::Dense)
        .unwrap_err();
    assert!(matches!(residual_leakage, QmbedError::InvalidSector(_)));
}

#[test]
fn dense_eigendecomposition_defines_the_empty_sector_spectrum() {
    let empty = Operator::from_dense(0, 0, Vec::new()).unwrap();
    let spectrum = eigh(&empty).unwrap();
    assert!(spectrum.eigenvalues.is_empty());
    assert!(spectrum.eigenvectors.is_empty());
    assert!(spectrum.residuals.is_empty());
    assert_eq!(spectrum.iterations, 0);
    assert!(spectrum.converged);
}
