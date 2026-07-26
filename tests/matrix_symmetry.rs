use approx::assert_abs_diff_eq;
use qmbed::Complex64;
use qmbed::basis::{
    Basis, ExchangeStatistics, GeneralBasis, LatticeSymmetryMap, MatrixSymmetryReducer,
    SpinBasis1D, SymmetryReducer,
};
use qmbed::operator::{Coupling, LinearOperator, MatrixFormat, OperatorBuilder, OperatorTerm};
use qmbed::solve::eigh;

fn dihedral_reducer(
    sites: usize,
    momentum: i32,
    selected_row: usize,
) -> MatrixSymmetryReducer<u128> {
    let angle = std::f64::consts::TAU * f64::from(momentum) / sites as f64;
    let (sine, cosine) = angle.sin_cos();
    let translation_representation = vec![
        Complex64::new(cosine, 0.0),
        Complex64::new(-sine, 0.0),
        Complex64::new(sine, 0.0),
        Complex64::new(cosine, 0.0),
    ];
    let reflection_representation = vec![
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(-1.0, 0.0),
    ];
    let translation = LatticeSymmetryMap::new(
        2,
        (0..sites)
            .map(|site| (site + 1) % sites)
            .collect::<Vec<_>>(),
        None,
        ExchangeStatistics::Distinguishable,
    )
    .unwrap();
    let reflection = LatticeSymmetryMap::new(
        2,
        (0..sites).rev().collect::<Vec<_>>(),
        None,
        ExchangeStatistics::Distinguishable,
    )
    .unwrap();
    MatrixSymmetryReducer::new(2, selected_row)
        .unwrap()
        .with_map(translation, translation_representation)
        .unwrap()
        .with_map(reflection, reflection_representation)
        .unwrap()
}

fn invariant_spin_terms(sites: usize) -> Vec<OperatorTerm> {
    vec![
        OperatorTerm::new(
            "zz",
            (0..sites).map(|site| Coupling::new(0.73, vec![site, (site + 1) % sites])),
        )
        .unwrap(),
        OperatorTerm::new("x", (0..sites).map(|site| Coupling::new(-0.41, vec![site]))).unwrap(),
    ]
}

#[test]
fn matrix_representation_rows_reproduce_the_generic_momentum_block() {
    let sites = 3;
    let parent = SpinBasis1D::builder(sites).pauli(true).build().unwrap();
    let full_operator = OperatorBuilder::on(&parent)
        .terms(invariant_spin_terms(sites))
        .build(MatrixFormat::Csc)
        .unwrap();

    let mut row_spectra = Vec::new();
    for selected_row in 0..2 {
        let subspace = dihedral_reducer(sites, 1, selected_row)
            .subspace(&parent)
            .unwrap();
        assert_eq!(subspace.dimension(), 2);
        let projector = subspace.projector(&parent, MatrixFormat::Csc).unwrap();
        assert_eq!(projector.shape(), (parent.len(), 2));

        let gram = projector.adjoint().unwrap().product(&projector).unwrap();
        let dense = gram.to_dense();
        for row in 0..2 {
            for column in 0..2 {
                let expected = if row == column { 1.0 } else { 0.0 };
                assert_abs_diff_eq!(dense[row * 2 + column].re, expected, epsilon = 1.0e-10);
                assert_abs_diff_eq!(dense[row * 2 + column].im, 0.0, epsilon = 1.0e-10);
            }
        }
        row_spectra.push(
            eigh(&full_operator.projected_by(&projector).unwrap())
                .unwrap()
                .eigenvalues,
        );
    }

    let translation = LatticeSymmetryMap::site_permutation(
        2,
        (0..sites)
            .map(|site| (site + 1) % sites)
            .collect::<Vec<_>>(),
    )
    .unwrap();
    let momentum_basis =
        GeneralBasis::from_reducer(parent, SymmetryReducer::new().with_map(translation, 1))
            .unwrap();
    let momentum_operator = OperatorBuilder::on(&momentum_basis)
        .terms(invariant_spin_terms(sites))
        .build(MatrixFormat::Csc)
        .unwrap();
    let momentum_spectrum = eigh(&momentum_operator).unwrap().eigenvalues;

    for spectrum in row_spectra {
        assert_eq!(spectrum.len(), momentum_spectrum.len());
        for (actual, expected) in spectrum.into_iter().zip(&momentum_spectrum) {
            assert_abs_diff_eq!(actual, expected, epsilon = 1.0e-10);
        }
    }
}
