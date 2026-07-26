use approx::assert_abs_diff_eq;
use qmbed::basis::{Basis, SpinBasis1D, SpinlessFermionBasis1D};
use qmbed::operator::{
    Coupling, LinearOperator, MatrixFormat, Operator, OperatorBuilder, OperatorTerm,
};
use qmbed::solve::{
    EighOptions, EigshOptions, SpectrumTarget, eigh, eigh_with_options, eigsh, eigsh_values,
    eigsh_with_initial,
};
use qmbed::{Complex64, QmbedError};

fn single_site_operator(basis: &SpinBasis1D, operator: &str) -> Vec<Complex64> {
    OperatorBuilder::on(basis)
        .term(OperatorTerm::new(operator, [Coupling::new(1.0, vec![0])]).unwrap())
        .build(MatrixFormat::Dense)
        .unwrap()
        .to_dense()
}

fn entry(matrix: &[Complex64], dimension: usize, row: usize, column: usize) -> Complex64 {
    matrix[row * dimension + column]
}

#[test]
fn spin_one_cartesian_operator_uses_branching_transitions() {
    let basis = SpinBasis1D::builder(1).spin_twice(2).build().unwrap();
    assert_eq!(basis.len(), 3);
    assert_eq!(basis.state(0).unwrap(), 0);
    assert_eq!(basis.state(1).unwrap(), 1);
    assert_eq!(basis.state(2).unwrap(), 2);

    let transitions = basis.apply_local_transitions(1, "x", &[0]).unwrap();
    assert_eq!(transitions.len(), 2);
    assert!(!transitions.spilled());
    let mut streamed = Vec::new();
    basis
        .visit_local_unreduced_transitions(1, "x", &[0], |target, amplitude| {
            streamed.push((target, amplitude));
            Ok(())
        })
        .unwrap();
    assert_eq!(streamed.as_slice(), transitions.as_slice());
    let expected = 1.0 / 2.0_f64.sqrt();
    assert_eq!(transitions[0].0, 2);
    assert_abs_diff_eq!(transitions[0].1.re, expected, epsilon = 1.0e-14);
    assert_eq!(transitions[1].0, 0);
    assert_abs_diff_eq!(transitions[1].1.re, expected, epsilon = 1.0e-14);

    let x = single_site_operator(&basis, "x");
    for (row, column) in [(0, 1), (1, 0), (1, 2), (2, 1)] {
        assert_abs_diff_eq!(
            entry(&x, basis.len(), row, column).re,
            expected,
            epsilon = 1.0e-14
        );
    }
    assert_abs_diff_eq!(entry(&x, basis.len(), 0, 2).norm(), 0.0, epsilon = 1.0e-14);

    let z = single_site_operator(&basis, "z");
    assert_abs_diff_eq!(entry(&z, 3, 0, 0).re, -1.0, epsilon = 1.0e-14);
    assert_abs_diff_eq!(entry(&z, 3, 1, 1).re, 0.0, epsilon = 1.0e-14);
    assert_abs_diff_eq!(entry(&z, 3, 2, 2).re, 1.0, epsilon = 1.0e-14);
}

#[test]
fn higher_spin_fixed_magnetization_and_translation_sectors_are_enumerated() {
    let parent = SpinBasis1D::builder(2).spin_twice(2).up(2).build().unwrap();
    assert_eq!(parent.len(), 3);

    let momentum_zero = SpinBasis1D::builder(2)
        .spin_twice(2)
        .up(2)
        .momentum(0)
        .build()
        .unwrap();
    let momentum_one = SpinBasis1D::builder(2)
        .spin_twice(2)
        .up(2)
        .momentum(1)
        .build()
        .unwrap();
    assert_eq!(momentum_zero.len(), 2);
    assert_eq!(momentum_one.len(), 1);

    let hamiltonian = OperatorBuilder::on(&momentum_zero)
        .term(
            OperatorTerm::new(
                "zz",
                [
                    Coupling::new(1.0, vec![0, 1]),
                    Coupling::new(1.0, vec![1, 0]),
                ],
            )
            .unwrap(),
        )
        .build(MatrixFormat::Csc)
        .unwrap();
    assert_eq!(hamiltonian.shape(), (2, 2));
    assert!(hamiltonian.is_hermitian(1.0e-13));
}

#[test]
fn higher_spin_rejects_the_spin_half_pauli_convention() {
    let error = SpinBasis1D::builder(2)
        .spin_twice(2)
        .pauli(true)
        .build()
        .unwrap_err();
    assert!(matches!(error, QmbedError::InvalidOptions(_)));
}

#[test]
fn dense_eigensolvers_support_complex_hermitian_operators() {
    let sigma_y = Operator::from_dense(
        2,
        2,
        vec![
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, -1.0),
            Complex64::new(0.0, 1.0),
            Complex64::new(0.0, 0.0),
        ],
    )
    .unwrap();
    let full = eigh(&sigma_y).unwrap();
    assert_abs_diff_eq!(full.eigenvalues[0], -1.0, epsilon = 1.0e-12);
    assert_abs_diff_eq!(full.eigenvalues[1], 1.0, epsilon = 1.0e-12);
    assert!(full.residuals.iter().all(|residual| *residual < 1.0e-12));

    let partial = eigsh(
        &sigma_y,
        EigshOptions {
            eigenpairs: 1,
            target: SpectrumTarget::SmallestAlgebraic,
            krylov_dimension: None,
            tolerance: 1.0e-12,
            max_iterations: 20,
            seed: 9,
        },
    )
    .unwrap();
    assert_abs_diff_eq!(partial.eigenvalues[0], -1.0, epsilon = 1.0e-12);
    assert!(partial.residuals[0] < 1.0e-12);
}

fn periodic_heisenberg(sites: usize) -> Vec<OperatorTerm> {
    let mut zz = Vec::new();
    let mut plus_minus = Vec::new();
    let mut minus_plus = Vec::new();
    for site in 0..sites {
        let next = (site + 1) % sites;
        zz.push(Coupling::new(1.0, vec![site, next]));
        plus_minus.push(Coupling::new(0.5, vec![site, next]));
        minus_plus.push(Coupling::new(0.5, vec![site, next]));
    }
    vec![
        OperatorTerm::new("zz", zz).unwrap(),
        OperatorTerm::new("+-", plus_minus).unwrap(),
        OperatorTerm::new("-+", minus_plus).unwrap(),
    ]
}

#[test]
fn parity_sectors_reconstruct_the_full_spin_spectrum() {
    let full_basis = SpinBasis1D::builder(4).up(2).build().unwrap();
    let even_basis = SpinBasis1D::builder(4).up(2).parity(1).build().unwrap();
    let odd_basis = SpinBasis1D::builder(4).up(2).parity(-1).build().unwrap();
    assert_eq!(even_basis.len(), 4);
    assert_eq!(odd_basis.len(), 2);
    assert_eq!(even_basis.parity(), Some(1));
    assert_eq!(odd_basis.parity(), Some(-1));

    let terms = periodic_heisenberg(4);
    let full = OperatorBuilder::on(&full_basis)
        .terms(terms.clone())
        .build(MatrixFormat::Dense)
        .unwrap();
    let even = OperatorBuilder::on(&even_basis)
        .terms(terms.clone())
        .build(MatrixFormat::Dense)
        .unwrap();
    let odd = OperatorBuilder::on(&odd_basis)
        .terms(terms)
        .build(MatrixFormat::Dense)
        .unwrap();

    let full_values = eigh(&full).unwrap().eigenvalues;
    let mut reconstructed = eigh(&even).unwrap().eigenvalues;
    reconstructed.extend(eigh(&odd).unwrap().eigenvalues);
    reconstructed.sort_by(f64::total_cmp);
    for (actual, expected) in reconstructed.iter().zip(full_values) {
        assert_abs_diff_eq!(*actual, expected, epsilon = 1.0e-11);
    }

    let incompatible = SpinBasis1D::builder(4)
        .up(2)
        .momentum(1)
        .parity(1)
        .build()
        .unwrap_err();
    assert!(matches!(incompatible, QmbedError::IncompatibleSymmetry(_)));
}

fn periodic_spinless_hopping(sites: usize) -> Vec<OperatorTerm> {
    let mut forward = Vec::new();
    let mut backward = Vec::new();
    for site in 0..sites {
        let next = (site + 1) % sites;
        forward.push(Coupling::new(-1.0, vec![site, next]));
        backward.push(Coupling::new(1.0, vec![site, next]));
    }
    vec![
        OperatorTerm::new("+-", forward).unwrap(),
        OperatorTerm::new("-+", backward).unwrap(),
    ]
}

#[test]
fn spinless_momentum_sectors_reconstruct_the_full_spectrum() {
    let full_basis = SpinlessFermionBasis1D::builder(4)
        .particles(2)
        .build()
        .unwrap();
    let terms = periodic_spinless_hopping(4);
    let full = OperatorBuilder::on(&full_basis)
        .terms(terms.clone())
        .build(MatrixFormat::Dense)
        .unwrap();
    let full_values = eigh(&full).unwrap().eigenvalues;

    let mut reconstructed = Vec::new();
    let mut sector_dimension = 0;
    for momentum in 0..4 {
        let basis = SpinlessFermionBasis1D::builder(4)
            .particles(2)
            .momentum(momentum)
            .build()
            .unwrap();
        assert_eq!(basis.momentum(), Some(momentum as usize));
        sector_dimension += basis.len();
        let operator = OperatorBuilder::on(&basis)
            .terms(terms.clone())
            .build(MatrixFormat::Dense)
            .unwrap();
        reconstructed.extend(eigh(&operator).unwrap().eigenvalues);
    }

    assert_eq!(sector_dimension, full_basis.len());
    reconstructed.sort_by(f64::total_cmp);
    for (actual, expected) in reconstructed.iter().zip(full_values) {
        assert_abs_diff_eq!(*actual, expected, epsilon = 1.0e-11);
    }
}

fn target_values(operator: &Operator, target: SpectrumTarget, eigenpairs: usize) -> Vec<f64> {
    eigsh(
        operator,
        EigshOptions {
            eigenpairs,
            target,
            krylov_dimension: None,
            tolerance: 1.0e-12,
            max_iterations: 50,
            seed: 3,
        },
    )
    .unwrap()
    .eigenvalues
}

#[test]
fn eigsh_covers_magnitude_and_both_end_targets() {
    let diagonal_values = [-5.0, -2.0, 0.25, 3.0, 7.0];
    let mut dense = vec![Complex64::new(0.0, 0.0); 25];
    for (index, value) in diagonal_values.into_iter().enumerate() {
        dense[index * 5 + index] = Complex64::new(value, 0.0);
    }
    let operator = Operator::from_dense(5, 5, dense).unwrap();

    for (actual, expected) in target_values(&operator, SpectrumTarget::SmallestMagnitude, 2)
        .into_iter()
        .zip([0.25, -2.0])
    {
        assert_abs_diff_eq!(actual, expected, epsilon = 1.0e-12);
    }
    for (actual, expected) in target_values(&operator, SpectrumTarget::LargestMagnitude, 2)
        .into_iter()
        .zip([7.0, -5.0])
    {
        assert_abs_diff_eq!(actual, expected, epsilon = 1.0e-12);
    }
    for (actual, expected) in target_values(&operator, SpectrumTarget::BothEnds, 3)
        .into_iter()
        .zip([-5.0, 3.0, 7.0])
    {
        assert_abs_diff_eq!(actual, expected, epsilon = 1.0e-12);
    }
    let without_vectors = eigh_with_options(
        &operator,
        EighOptions {
            return_eigenvectors: false,
        },
    )
    .unwrap();
    assert!(without_vectors.eigenvectors.is_empty());
    assert_eq!(without_vectors.eigenvalues.len(), 5);
    for (actual, expected) in eigsh_values(&operator, EigshOptions::smallest_algebraic(2))
        .unwrap()
        .into_iter()
        .zip([-5.0, -2.0])
    {
        assert_abs_diff_eq!(actual, expected, epsilon = 1.0e-12);
    }

    let large = Operator::from_triplets(
        129,
        129,
        (0..129).map(|index| (index, index, Complex64::new(index as f64, 0.0))),
        MatrixFormat::MatrixFree,
    )
    .unwrap();
    let invalid_initial = eigsh_with_initial(
        &large,
        EigshOptions::smallest_algebraic(1),
        &[Complex64::new(1.0, 0.0); 2],
    )
    .unwrap_err();
    assert!(matches!(invalid_initial, QmbedError::DimensionMismatch(_)));

    let zero_initial = eigsh_with_initial(
        &large,
        EigshOptions::smallest_algebraic(1),
        &[Complex64::new(0.0, 0.0); 129],
    )
    .unwrap_err();
    assert!(matches!(zero_initial, QmbedError::InvalidOptions(_)));

    let nonfinite_initial = eigsh_with_initial(
        &large,
        EigshOptions::smallest_algebraic(1),
        &[Complex64::new(f64::NAN, 0.0); 129],
    )
    .unwrap_err();
    assert!(matches!(nonfinite_initial, QmbedError::InvalidOptions(_)));

    let mut selected_subspace = vec![Complex64::new(0.0, 0.0); 129];
    selected_subspace[127] = Complex64::new(1.0, 0.0);
    selected_subspace[128] = Complex64::new(1.0, 0.0);
    let selected = eigsh_with_initial(
        &large,
        EigshOptions {
            eigenpairs: 1,
            target: SpectrumTarget::SmallestAlgebraic,
            krylov_dimension: Some(2),
            tolerance: 1.0e-12,
            max_iterations: 2,
            seed: 0,
        },
        &selected_subspace,
    )
    .unwrap();
    assert_abs_diff_eq!(selected.eigenvalues[0], 127.0, epsilon = 1.0e-10);
}
