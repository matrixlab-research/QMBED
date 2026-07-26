use approx::assert_abs_diff_eq;
use qmbed::Complex64;
use qmbed::measure::{
    EntropyOrder, NoncommutingGroup, analyze_diagonal_ensemble, apply_fermionic_subsystem_phases,
    apply_noncommuting_subsystem_exchange_phases,
    apply_noncommuting_subsystem_exchange_phases_density, apply_noncommuting_subsystem_phases,
    array_to_ints, array_to_states, canonical_schmidt_spectrum_subsystem, density_expectation,
    density_matrix_spectrum, density_quantum_fluctuation, diagonal_ensemble,
    diagonal_ensemble_density, diagonal_ensemble_observable, ed_density_vs_time, ed_state_vs_time,
    energy_window_indices, entanglement_entropy, entanglement_entropy_batch,
    entanglement_entropy_density, entanglement_entropy_density_subsystem,
    entanglement_entropy_subsystem, entanglement_spectrum, entanglement_spectrum_density,
    entanglement_spectrum_subsystem, entropy_from_spectrum, expectation, ints_to_array,
    kl_divergence, matrix_element, mean_level_spacing, observables_vs_time, partial_trace,
    partial_trace_density, partial_trace_density_subsystem, partial_trace_subsystem,
    quantum_fluctuation, raw_quantum_fluctuation, states_to_array,
};
use qmbed::operator::Operator;
use qmbed::solve::StateTrajectory;

fn sigma_z() -> Operator {
    Operator::from_dense(
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
}

#[test]
fn observables_and_fluctuations_match_two_level_anchors() {
    let operator = sigma_z();
    let plus = vec![
        Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
        Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
    ];
    assert_abs_diff_eq!(
        expectation(&operator, &plus).unwrap().re,
        0.0,
        epsilon = 1.0e-12
    );
    assert_abs_diff_eq!(
        matrix_element(&plus, &operator, &plus).unwrap().re,
        0.0,
        epsilon = 1.0e-12
    );
    assert_abs_diff_eq!(
        quantum_fluctuation(&operator, &plus).unwrap(),
        1.0,
        epsilon = 1.0e-12
    );
    assert_eq!(
        raw_quantum_fluctuation(
            &operator,
            &[Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0)],
        )
        .unwrap(),
        Complex64::new(0.0, 0.0)
    );

    let trajectory = StateTrajectory {
        times: vec![0.0, 1.0],
        states: vec![
            plus.clone(),
            vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
        ],
    };
    let values = observables_vs_time(&trajectory, &[("z".to_string(), &operator)]).unwrap();
    assert_abs_diff_eq!(values["z"][0].re, 0.0, epsilon = 1.0e-12);
    assert_abs_diff_eq!(values["z"][1].re, 1.0, epsilon = 1.0e-12);
}

#[test]
fn partial_trace_and_entropy_distinguish_product_and_bell_states() {
    let product = vec![
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
    ];
    assert_abs_diff_eq!(
        entanglement_entropy(&product, 2, 2, EntropyOrder::VonNeumann).unwrap(),
        0.0,
        epsilon = 1.0e-12
    );

    let amplitude = 1.0 / 2.0_f64.sqrt();
    let bell = vec![
        Complex64::new(amplitude, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(amplitude, 0.0),
    ];
    let reduced = partial_trace(&bell, 2, 2).unwrap();
    assert_abs_diff_eq!(reduced[0].re, 0.5, epsilon = 1.0e-12);
    assert_abs_diff_eq!(reduced[3].re, 0.5, epsilon = 1.0e-12);
    assert_abs_diff_eq!(
        entanglement_entropy(&bell, 2, 2, EntropyOrder::VonNeumann).unwrap(),
        2.0_f64.ln(),
        epsilon = 1.0e-12
    );
    assert_abs_diff_eq!(
        entanglement_entropy(&bell, 2, 2, EntropyOrder::Renyi(2.0)).unwrap(),
        2.0_f64.ln(),
        epsilon = 1.0e-12
    );
}

#[test]
fn ensemble_statistics_and_state_conversions_are_deterministic() {
    let eigenvectors = vec![
        vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
        vec![Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
    ];
    let initial = vec![
        Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
        Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
    ];
    let ensemble = diagonal_ensemble(&[-1.0, 1.0], &eigenvectors, &initial).unwrap();
    assert_abs_diff_eq!(ensemble.mean_energy, 0.0, epsilon = 1.0e-12);
    assert_abs_diff_eq!(ensemble.energy_variance, 1.0, epsilon = 1.0e-12);
    assert_abs_diff_eq!(ensemble.entropy, 2.0_f64.ln(), epsilon = 1.0e-12);
    assert_abs_diff_eq!(
        kl_divergence(&[0.5, 0.5], &[0.5, 0.5]).unwrap(),
        0.0,
        epsilon = 1.0e-12
    );
    assert_abs_diff_eq!(
        mean_level_spacing(&[0.0, 1.0, 3.0, 6.0]).unwrap(),
        (0.5 + 2.0 / 3.0) / 2.0,
        epsilon = 1.0e-12
    );

    let states = vec![0_u128, 5, 15];
    let occupations = states_to_array(&states, 4, 2).unwrap();
    assert_eq!(array_to_states(&occupations, 2).unwrap(), states);
    let binary = ints_to_array(&states, 4).unwrap();
    assert_eq!(binary[1], vec![0, 1, 0, 1]);
    assert_eq!(array_to_ints(&binary).unwrap(), states);
}

#[test]
fn exact_eigenbasis_evolution_supports_pure_and_mixed_states() {
    let eigenvalues = [-1.0, 1.0];
    let eigenvectors = vec![
        vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
        vec![Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
    ];
    let initial = vec![
        Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
        Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
    ];
    let trajectory = ed_state_vs_time(&initial, &eigenvalues, &eigenvectors, &[0.0, 0.5]).unwrap();
    assert_abs_diff_eq!(trajectory.states[1][0].arg(), 0.5, epsilon = 1.0e-12);
    assert_abs_diff_eq!(trajectory.states[1][1].arg(), -0.5, epsilon = 1.0e-12);

    let density = vec![
        Complex64::new(0.5, 0.0),
        Complex64::new(0.5, 0.0),
        Complex64::new(0.5, 0.0),
        Complex64::new(0.5, 0.0),
    ];
    let evolved = ed_density_vs_time(&density, &eigenvalues, &eigenvectors, &[0.5]).unwrap();
    assert_abs_diff_eq!(evolved[0][0].re + evolved[0][3].re, 1.0, epsilon = 1.0e-12);
    assert_abs_diff_eq!(evolved[0][1].norm(), 0.5, epsilon = 1.0e-12);
}

#[test]
fn mixed_and_batched_measurements_share_the_pure_state_limits() {
    let amplitude = 1.0 / 2.0_f64.sqrt();
    let bell = vec![
        Complex64::new(amplitude, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(amplitude, 0.0),
    ];
    let density: Vec<_> = bell
        .iter()
        .flat_map(|left| bell.iter().map(move |right| *left * right.conj()))
        .collect();
    assert_eq!(
        partial_trace_density(&density, 2, 2).unwrap(),
        partial_trace(&bell, 2, 2).unwrap()
    );
    let pure_spectrum = entanglement_spectrum(&bell, 2, 2).unwrap();
    let mixed_spectrum = entanglement_spectrum_density(&density, 2, 2).unwrap();
    for (actual, expected) in mixed_spectrum.iter().zip(pure_spectrum) {
        assert_abs_diff_eq!(actual, &expected, epsilon = 1.0e-12);
    }
    assert_abs_diff_eq!(
        entanglement_entropy_density(&density, 2, 2, EntropyOrder::VonNeumann).unwrap(),
        2.0_f64.ln(),
        epsilon = 1.0e-12
    );
    let product = vec![
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
    ];
    let batch =
        entanglement_entropy_batch(&[bell, product], 2, 2, EntropyOrder::VonNeumann).unwrap();
    assert_abs_diff_eq!(batch[0], 2.0_f64.ln(), epsilon = 1.0e-12);
    assert_abs_diff_eq!(batch[1], 0.0, epsilon = 1.0e-12);
}

#[test]
fn density_diagonal_ensemble_and_observable_are_consistent() {
    let eigenvectors = vec![
        vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
        vec![Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
    ];
    let density = vec![
        Complex64::new(0.75, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.25, 0.0),
    ];
    let ensemble = diagonal_ensemble_density(&[-1.0, 1.0], &eigenvectors, &density).unwrap();
    assert_eq!(ensemble.probabilities, vec![0.75, 0.25]);
    let z = sigma_z();
    assert_abs_diff_eq!(
        density_expectation(&z, &density).unwrap().re,
        0.5,
        epsilon = 1.0e-12
    );
    assert_abs_diff_eq!(
        density_quantum_fluctuation(&z, &density).unwrap().re,
        0.75,
        epsilon = 1.0e-12
    );
    assert_abs_diff_eq!(
        diagonal_ensemble_observable(&ensemble, &eigenvectors, &z)
            .unwrap()
            .re,
        0.5,
        epsilon = 1.0e-12
    );
    assert_eq!(
        energy_window_indices(&[-2.0, -0.1, 0.2, 3.0], 0.0, 0.25).unwrap(),
        vec![1, 2]
    );
}

#[test]
fn diagonal_probability_batches_share_matrix_free_observable_statistics() {
    let eigenvectors = vec![
        vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
        vec![Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
    ];
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
    let analyses = analyze_diagonal_ensemble(
        &[-1.0, 1.0],
        &eigenvectors,
        &[vec![3.0, 1.0], vec![1.0, 1.0]],
        Some(&sigma_x),
        2.0,
    )
    .unwrap();
    assert_eq!(analyses[0].ensemble.probabilities, vec![0.75, 0.25]);
    assert_abs_diff_eq!(
        analyses[0].diagonal_entropy,
        -(0.625_f64).ln(),
        epsilon = 1.0e-12
    );
    assert_abs_diff_eq!(analyses[0].observable.unwrap(), 0.0, epsilon = 1.0e-12);
    assert_abs_diff_eq!(
        analyses[0].temporal_fluctuation.unwrap(),
        0.375_f64.sqrt(),
        epsilon = 1.0e-12
    );
    assert_abs_diff_eq!(
        analyses[0].quantum_fluctuation.unwrap(),
        0.625_f64.sqrt(),
        epsilon = 1.0e-12
    );
    assert_abs_diff_eq!(
        analyses[1].diagonal_entropy,
        2.0_f64.ln(),
        epsilon = 1.0e-12
    );
}

#[test]
fn arbitrary_site_partial_trace_supports_noncontiguous_subsystems() {
    let amplitude = 1.0 / 2.0_f64.sqrt();
    let mut ghz = vec![Complex64::new(0.0, 0.0); 8];
    ghz[0] = Complex64::new(amplitude, 0.0);
    ghz[7] = Complex64::new(amplitude, 0.0);
    let reduced = partial_trace_subsystem(&ghz, &[2, 2, 2], &[0, 2]).unwrap();
    assert_eq!(reduced.len(), 16);
    assert_abs_diff_eq!(reduced[0].re, 0.5, epsilon = 1.0e-12);
    assert_abs_diff_eq!(reduced[15].re, 0.5, epsilon = 1.0e-12);
    assert_abs_diff_eq!(reduced[3].norm(), 0.0, epsilon = 1.0e-12);
    assert_abs_diff_eq!(
        entanglement_entropy_subsystem(&ghz, &[2, 2, 2], &[0, 2], EntropyOrder::VonNeumann)
            .unwrap(),
        2.0_f64.ln(),
        epsilon = 1.0e-12
    );
    let spectrum = entanglement_spectrum_subsystem(&ghz, &[2, 2, 2], &[0, 2]).unwrap();
    assert_abs_diff_eq!(spectrum[2], 0.5, epsilon = 1.0e-12);
    assert_abs_diff_eq!(spectrum[3], 0.5, epsilon = 1.0e-12);

    let density: Vec<_> = ghz
        .iter()
        .flat_map(|left| ghz.iter().map(move |right| *left * right.conj()))
        .collect();
    let mixed = partial_trace_density_subsystem(&density, &[2, 2, 2], &[0, 2]).unwrap();
    for (actual, expected) in mixed.iter().zip(reduced) {
        assert_abs_diff_eq!(actual.re, expected.re, epsilon = 1.0e-12);
        assert_abs_diff_eq!(actual.im, expected.im, epsilon = 1.0e-12);
    }
    assert_abs_diff_eq!(
        entanglement_entropy_density_subsystem(
            &density,
            &[2, 2, 2],
            &[0, 2],
            EntropyOrder::Renyi(2.0),
        )
        .unwrap(),
        2.0_f64.ln(),
        epsilon = 1.0e-12
    );
}

#[test]
fn large_rank_deficient_density_spectrum_survives_backend_degeneracy() {
    let dimension = 192;
    let norm = (dimension as f64).sqrt();
    let first = vec![Complex64::new(1.0 / norm, 0.0); dimension];
    let second = (0..dimension)
        .map(|index| {
            let phase = std::f64::consts::TAU * index as f64 / dimension as f64;
            Complex64::from_polar(1.0 / norm, phase)
        })
        .collect::<Vec<_>>();
    let density = (0..dimension)
        .flat_map(|row| {
            let first = &first;
            let second = &second;
            (0..dimension).map(move |column| {
                0.3 * first[row] * first[column].conj() + 0.7 * second[row] * second[column].conj()
            })
        })
        .collect::<Vec<_>>();

    let spectrum = density_matrix_spectrum(density, dimension).unwrap();
    assert_eq!(spectrum.len(), dimension);
    assert!(
        spectrum[..dimension - 2]
            .iter()
            .all(|value| *value < 1.0e-10)
    );
    assert_abs_diff_eq!(spectrum[dimension - 2], 0.3, epsilon = 1.0e-10);
    assert_abs_diff_eq!(spectrum[dimension - 1], 0.7, epsilon = 1.0e-10);
}

#[test]
fn canonical_schmidt_spectrum_is_bitwise_identical_for_complementary_calls() {
    let local_dimensions = [2; 6];
    let mut state = (0..64)
        .map(|index| {
            Complex64::new(
                ((index * 17 + 3) % 29) as f64 - 14.0,
                ((index * 11 + 5) % 31) as f64 - 15.0,
            )
        })
        .collect::<Vec<_>>();
    let norm = state.iter().map(Complex64::norm_sqr).sum::<f64>().sqrt();
    for value in &mut state {
        *value /= norm;
    }
    for (retained, environment) in [
        (vec![2, 0, 4, 1, 3], vec![5]),
        (vec![4, 0, 2], vec![1, 3, 5]),
        (vec![5, 3, 1], vec![0, 2, 4]),
    ] {
        let retained_spectrum =
            canonical_schmidt_spectrum_subsystem(&state, &local_dimensions, &retained, &[])
                .unwrap();
        let environment_spectrum =
            canonical_schmidt_spectrum_subsystem(&state, &local_dimensions, &environment, &[])
                .unwrap();
        assert_eq!(retained_spectrum, environment_spectrum);
        assert_eq!(
            entropy_from_spectrum(&retained_spectrum, EntropyOrder::VonNeumann).unwrap(),
            entropy_from_spectrum(&environment_spectrum, EntropyOrder::VonNeumann).unwrap()
        );
    }
}

#[test]
fn fermionic_subsystem_reordering_tracks_occupied_mode_swaps() {
    let mut state = vec![Complex64::new(0.0, 0.0); 16];
    state[0b0110] = Complex64::new(0.25, -0.5);
    state[0b1010] = Complex64::new(-0.75, 0.125);
    apply_fermionic_subsystem_phases(&mut state, &[2, 2, 2, 2], &[1, 3]).unwrap();
    assert_eq!(state[0b0110], Complex64::new(-0.25, 0.5));
    assert_eq!(state[0b1010], Complex64::new(-0.75, 0.125));
}

#[test]
fn noncommuting_subsystem_reordering_keeps_distinct_species_commuting() {
    let mut state = vec![Complex64::new(1.0, 0.0); 16];
    apply_noncommuting_subsystem_phases(
        &mut state,
        &[2, 2, 2, 2],
        &[1, 2],
        &[vec![0, 1], vec![2, 3]],
    )
    .unwrap();
    assert_eq!(state[0b0110], Complex64::new(1.0, 0.0));
    assert_eq!(state[0b1100], Complex64::new(-1.0, 0.0));
}

#[test]
fn noncommuting_subsystem_reordering_supports_unit_modulus_exchange_phases() {
    let group = NoncommutingGroup::new([0, 1], Complex64::new(0.0, 1.0)).unwrap();
    let mut state = vec![Complex64::new(1.0, 0.0); 4];
    apply_noncommuting_subsystem_exchange_phases(
        &mut state,
        &[2, 2],
        &[0],
        std::slice::from_ref(&group),
    )
    .unwrap();
    assert_eq!(state[0b00], Complex64::new(1.0, 0.0));
    assert_eq!(state[0b11], Complex64::new(0.0, 1.0));

    let mut density = vec![Complex64::new(0.0, 0.0); 16];
    density[0b11 * 4] = Complex64::new(1.0, 0.0);
    density[0b11] = Complex64::new(1.0, 0.0);
    apply_noncommuting_subsystem_exchange_phases_density(&mut density, &[2, 2], &[0], &[group])
        .unwrap();
    assert_eq!(density[0b11 * 4], Complex64::new(0.0, 1.0));
    assert_eq!(density[0b11], Complex64::new(0.0, -1.0));

    assert!(NoncommutingGroup::new([0, 1], Complex64::new(2.0, 0.0)).is_err());
}
