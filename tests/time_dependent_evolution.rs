use approx::assert_abs_diff_eq;
use qmbed::Complex64;
use qmbed::operator::{Dynamic, DynamicComponent, Hamiltonian, MatrixFormat, Operator};
use qmbed::solve::{
    EvolutionOptions, RhsEvolutionOptions, evolve_batch, evolve_density, evolve_rhs,
    evolve_time_dependent, evolve_time_dependent_batch,
};

fn options(times: Vec<f64>) -> EvolutionOptions {
    EvolutionOptions::new(times).with_krylov_dimension(24)
}

fn sigma_z_drive() -> Hamiltonian<Dynamic> {
    let zero = Operator::from_dense(2, 2, vec![Complex64::new(0.0, 0.0); 4]).unwrap();
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
    .unwrap()
    .converted(MatrixFormat::Csc)
    .unwrap();
    Hamiltonian::<Dynamic>::new(
        zero,
        vec![DynamicComponent::new(sigma_z, |time| {
            Complex64::new(time, 0.0)
        })],
    )
    .unwrap()
}

fn norm(state: &[Complex64]) -> f64 {
    state.iter().map(Complex64::norm_sqr).sum::<f64>().sqrt()
}

#[test]
fn driven_diagonal_hamiltonian_matches_the_analytic_phase() {
    let hamiltonian = sigma_z_drive();
    let initial = vec![
        Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
        Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
    ];
    let trajectory =
        evolve_time_dependent(&hamiltonian, &initial, options(vec![0.0, 0.4, 1.0])).unwrap();
    let phase = 0.5_f64;
    let expected = [
        Complex64::from_polar(1.0 / 2.0_f64.sqrt(), -phase),
        Complex64::from_polar(1.0 / 2.0_f64.sqrt(), phase),
    ];
    for (actual, expected) in trajectory.states[2].iter().zip(expected) {
        assert_abs_diff_eq!(actual.re, expected.re, epsilon = 2.0e-9);
        assert_abs_diff_eq!(actual.im, expected.im, epsilon = 2.0e-9);
    }
    assert_abs_diff_eq!(norm(&trajectory.states[2]), 1.0, epsilon = 1.0e-12);
}

#[test]
fn static_and_dynamic_batches_equal_independent_columns() {
    let diagonal = Operator::from_dense(
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
    let columns = vec![
        vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
        vec![Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
    ];
    let static_batch = evolve_batch(&diagonal, &columns, options(vec![0.0, 0.5])).unwrap();
    assert_eq!(static_batch.states.len(), 2);
    assert_eq!(static_batch.states[1].len(), 2);
    assert_abs_diff_eq!(norm(&static_batch.states[1][0]), 1.0, epsilon = 1.0e-12);
    assert_abs_diff_eq!(norm(&static_batch.states[1][1]), 1.0, epsilon = 1.0e-12);

    let dynamic_batch =
        evolve_time_dependent_batch(&sigma_z_drive(), &columns, options(vec![0.0, 0.5])).unwrap();
    assert_abs_diff_eq!(norm(&dynamic_batch.states[1][0]), 1.0, epsilon = 1.0e-12);
    assert_abs_diff_eq!(norm(&dynamic_batch.states[1][1]), 1.0, epsilon = 1.0e-12);
}

#[test]
fn callable_rhs_and_density_modes_preserve_their_physical_invariants() {
    let rhs_options = RhsEvolutionOptions::new(vec![0.0, 0.2, 1.0], 0.002).with_max_substeps(2_000);
    let trajectory = evolve_rhs(
        &[Complex64::new(1.0, 0.0)],
        0.0,
        rhs_options.clone(),
        |_, state, output| {
            output[0] = Complex64::new(0.0, -2.0) * state[0];
            Ok(())
        },
    )
    .unwrap();
    let expected = Complex64::new(0.0, -2.0).exp();
    assert_abs_diff_eq!(trajectory.states[2][0].re, expected.re, epsilon = 1.0e-11);
    assert_abs_diff_eq!(trajectory.states[2][0].im, expected.im, epsilon = 1.0e-11);

    let hamiltonian = Operator::from_dense(
        2,
        2,
        vec![
            Complex64::new(-1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
        ],
    )
    .unwrap();
    let density = vec![
        Complex64::new(0.5, 0.0),
        Complex64::new(0.5, 0.0),
        Complex64::new(0.5, 0.0),
        Complex64::new(0.5, 0.0),
    ];
    let evolved = evolve_density(&hamiltonian, &density, rhs_options).unwrap();
    let final_density = &evolved.states[2];
    assert_abs_diff_eq!(
        final_density[0].re + final_density[3].re,
        1.0,
        epsilon = 1.0e-11
    );
    assert_abs_diff_eq!(final_density[1].re, final_density[2].re, epsilon = 1.0e-12);
    assert_abs_diff_eq!(final_density[1].im, -final_density[2].im, epsilon = 1.0e-12);
}
