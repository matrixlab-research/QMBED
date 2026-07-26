use std::sync::Arc;

use approx::assert_abs_diff_eq;
use qmbed::dynamics::{
    CallableDriveStep, DriveStep, Floquet, FloquetSpectrumOptions, FloquetTimeVector,
    analyze_floquet_unitary, dynamical_correlator,
};
use qmbed::operator::{
    Dynamic, DynamicComponent, Hamiltonian, LinearOperator, MatrixFormat, Operator,
};
use qmbed::solve::EvolutionOptions;
use qmbed::{Complex64, QmbedError};

fn diagonal(values: &[f64]) -> Operator {
    let dimension = values.len();
    let mut dense = vec![Complex64::new(0.0, 0.0); dimension * dimension];
    for (index, value) in values.iter().enumerate() {
        dense[index * dimension + index] = Complex64::new(*value, 0.0);
    }
    Operator::from_dense(dimension, dimension, dense).unwrap()
}

#[test]
fn floquet_builds_unitary_quasienergies_and_effective_hamiltonian() {
    let hamiltonian = diagonal(&[-1.0, 1.0]);
    let floquet =
        Floquet::new([DriveStep::new(Arc::new(hamiltonian.clone()), 0.7).unwrap()]).unwrap();
    assert_abs_diff_eq!(floquet.period(), 0.7, epsilon = 1.0e-15);
    let unitary = floquet.full_unitary(MatrixFormat::Csc).unwrap();
    let values = unitary.to_dense();
    assert_abs_diff_eq!(values[0].re, 0.7_f64.cos(), epsilon = 1.0e-12);
    assert_abs_diff_eq!(values[0].im, 0.7_f64.sin(), epsilon = 1.0e-12);
    assert_abs_diff_eq!(values[3].im, -0.7_f64.sin(), epsilon = 1.0e-12);

    let eigensystem = floquet.eigensystem().unwrap();
    assert_abs_diff_eq!(eigensystem.quasienergies[0], -1.0, epsilon = 1.0e-12);
    assert_abs_diff_eq!(eigensystem.quasienergies[1], 1.0, epsilon = 1.0e-12);
    assert!(
        eigensystem
            .residuals
            .iter()
            .all(|residual| *residual < 1.0e-12)
    );
    let effective = floquet.effective_hamiltonian(MatrixFormat::Dense).unwrap();
    for (actual, expected) in effective.to_dense().iter().zip(hamiltonian.to_dense()) {
        assert_abs_diff_eq!(actual.re, expected.re, epsilon = 1.0e-12);
        assert_abs_diff_eq!(actual.im, expected.im, epsilon = 1.0e-12);
    }
}

#[test]
fn floquet_analysis_reuses_one_propagator_and_supports_kicked_periods() {
    let hamiltonian = diagonal(&[-1.0, 1.0]);
    let floquet = Floquet::new([DriveStep::new(Arc::new(hamiltonian), 0.25).unwrap()])
        .unwrap()
        .with_period(1.0)
        .unwrap();
    let analysis = floquet.analyze(MatrixFormat::Dense).unwrap();
    assert_abs_diff_eq!(analysis.period, 1.0, epsilon = 1.0e-15);
    assert_abs_diff_eq!(analysis.protocol_duration, 0.25, epsilon = 1.0e-15);
    assert_abs_diff_eq!(
        analysis.eigensystem.quasienergies[0],
        -0.25,
        epsilon = 1.0e-12
    );
    assert_abs_diff_eq!(
        analysis.eigensystem.quasienergies[1],
        0.25,
        epsilon = 1.0e-12
    );
    let effective = analysis.effective_hamiltonian.to_dense();
    assert_abs_diff_eq!(effective[0].re, -0.25, epsilon = 1.0e-12);
    assert_abs_diff_eq!(effective[3].re, 0.25, epsilon = 1.0e-12);
}

#[test]
fn selected_floquet_spectrum_uses_matrix_free_period_actions() {
    let dimension = 129;
    let mut energies = vec![-2.0; dimension];
    energies[..3].copy_from_slice(&[0.31, 0.2, 0.45]);
    let hamiltonian = Operator::from_triplets(
        dimension,
        dimension,
        energies
            .iter()
            .copied()
            .enumerate()
            .map(|(index, energy)| (index, index, Complex64::new(energy, 0.0))),
        MatrixFormat::Csc,
    )
    .unwrap();
    let floquet = Floquet::new([DriveStep::new(Arc::new(hamiltonian), 1.0).unwrap()]).unwrap();
    let mut seed = vec![Complex64::new(0.0, 0.0); dimension];
    seed[0] = Complex64::new(0.6, 0.2);
    seed[7] = Complex64::new(-0.3, 0.7);
    let mut forward = vec![Complex64::new(0.0, 0.0); dimension];
    let mut restored = vec![Complex64::new(0.0, 0.0); dimension];
    floquet.apply_period(&seed, &mut forward).unwrap();
    floquet
        .apply_adjoint_period(&forward, &mut restored)
        .unwrap();
    assert!(
        seed.iter()
            .zip(restored)
            .all(|(expected, actual)| (*expected - actual).norm() < 1.0e-10)
    );
    let target = 0.31;
    let selected = floquet
        .selected_eigensystem(
            FloquetSpectrumOptions::new(3, target)
                .with_search_dimension(5)
                .with_krylov_dimension(12)
                .with_tolerance(1.0e-11)
                .with_max_iterations(2_000),
        )
        .unwrap();
    let mut expected = energies.clone();
    expected.sort_by(|left, right| (left - target).abs().total_cmp(&(right - target).abs()));
    for (actual, expected) in selected.quasienergies.iter().zip(expected) {
        assert_abs_diff_eq!(actual, &expected, epsilon = 1.0e-8);
    }
    assert!(selected.residuals.iter().all(|residual| *residual < 1.0e-8));
}

#[test]
fn floquet_spectrum_owns_the_reused_unitary_and_backend_diagnostics() {
    let hamiltonian = diagonal(&[-1.0, 1.0]);
    let floquet = Floquet::new([DriveStep::new(Arc::new(hamiltonian), 0.4).unwrap()]).unwrap();
    let spectrum = floquet.spectrum(MatrixFormat::Csc).unwrap();
    assert_eq!(spectrum.unitary.format(), MatrixFormat::Csc);
    assert!(spectrum.unitarity_residual < 1.0e-12);
    assert_abs_diff_eq!(
        spectrum.eigensystem.quasienergies[0],
        -1.0,
        epsilon = 1.0e-12
    );
    assert_abs_diff_eq!(
        spectrum.eigensystem.quasienergies[1],
        1.0,
        epsilon = 1.0e-12
    );
}

#[test]
fn externally_constructed_unitary_uses_the_same_floquet_analysis() {
    let unitary = Operator::from_dense(
        2,
        2,
        vec![
            Complex64::new(0.3_f64.cos(), 0.3_f64.sin()),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.3_f64.cos(), -0.3_f64.sin()),
        ],
    )
    .unwrap();
    let analysis = analyze_floquet_unitary(&unitary, 0.6, MatrixFormat::Dense).unwrap();
    assert_abs_diff_eq!(
        analysis.eigensystem.quasienergies[0],
        -0.5,
        epsilon = 1.0e-12
    );
    assert_abs_diff_eq!(
        analysis.eigensystem.quasienergies[1],
        0.5,
        epsilon = 1.0e-12
    );
}

#[test]
fn callable_floquet_drive_integrates_within_the_period() {
    let zero = diagonal(&[0.0, 0.0]);
    let driven = Hamiltonian::<Dynamic>::new(
        zero,
        vec![DynamicComponent::new(diagonal(&[-1.0, 1.0]), |time| {
            Complex64::new(time, 0.0)
        })],
    )
    .unwrap();
    let floquet =
        Floquet::from_callable([CallableDriveStep::new(Arc::new(driven), 1.0).unwrap()]).unwrap();
    let unitary = floquet
        .full_unitary(MatrixFormat::Dense)
        .unwrap()
        .to_dense();
    assert_abs_diff_eq!(unitary[0].re, 0.5_f64.cos(), epsilon = 2.0e-9);
    assert_abs_diff_eq!(unitary[0].im, 0.5_f64.sin(), epsilon = 2.0e-9);
    assert_abs_diff_eq!(unitary[3].im, -0.5_f64.sin(), epsilon = 2.0e-9);
}

#[test]
fn floquet_time_vector_has_exact_cycle_coordinates() {
    let times = FloquetTimeVector::new(2.0, 2, 4, true).unwrap();
    assert_eq!(times.times().len(), 9);
    assert_abs_diff_eq!(times.times()[8], 4.0, epsilon = 1.0e-15);
    assert_eq!(times.coordinate(5).unwrap().cycle, 1);
    assert_abs_diff_eq!(
        times.coordinate(5).unwrap().within_cycle,
        0.5,
        epsilon = 1.0e-15
    );
    assert_eq!(times.coordinate(8).unwrap().cycle, 2);
    assert!(matches!(
        times.coordinate(9).unwrap_err(),
        QmbedError::InvalidOptions(_)
    ));
}

#[test]
fn staged_floquet_time_vector_includes_ramps_and_both_endpoints() {
    let times = FloquetTimeVector::staged(0.5, 2, 3, 1, 4).unwrap();
    assert_eq!(times.cycles(), 6);
    assert_eq!(times.points_per_cycle(), 4);
    assert_eq!(times.times().len(), 25);
    assert_abs_diff_eq!(times.times()[0], -1.0, epsilon = 1.0e-15);
    assert_abs_diff_eq!(times.times()[24], 2.0, epsilon = 1.0e-15);
}

#[test]
fn dynamical_correlator_matches_a_two_level_lehmann_phase() {
    let hamiltonian = diagonal(&[-1.0, 1.0]);
    let sigma_x = Operator::from_dense(
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
    let times = vec![0.0, 0.3, 0.8];
    let values = dynamical_correlator(
        &hamiltonian,
        &[Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
        &sigma_x,
        &sigma_x,
        EvolutionOptions::new(times.clone())
            .with_krylov_dimension(8)
            .with_tolerance(1.0e-12)
            .with_max_substeps(100),
    )
    .unwrap();
    for (value, time) in values.iter().zip(times) {
        let expected = Complex64::new(0.0, -2.0 * time).exp();
        assert_abs_diff_eq!(value.re, expected.re, epsilon = 1.0e-12);
        assert_abs_diff_eq!(value.im, expected.im, epsilon = 1.0e-12);
    }
}
