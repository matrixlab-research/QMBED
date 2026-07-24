use approx::assert_abs_diff_eq;
use qmbed::Complex64;
use qmbed::basis::{
    Basis, BosonBasis1D, GeneralBasis, LatticeSymmetryMap, PackedBasis, SpinBasis1D,
    SpinNormalization, SymmetrySector,
};
use qmbed::interop::{OperatorAction, PackedEdModel};
use qmbed::operator::{
    AssemblyChecks, Coupling, LinearOperator, LocalOperator, MatrixFormat, OpProduct,
    OperatorBuilder, OperatorSpec,
};
use qmbed::solve::EighOptions;
use std::sync::Arc;

#[test]
fn packed_basis_preserves_concrete_operator_semantics() {
    let basis = SpinBasis1D::builder(3).up(1).build().unwrap();
    let packed = PackedBasis::from(basis.clone());
    let term = OperatorSpec::from_product(
        OpProduct::new([LocalOperator::Z]).unwrap(),
        (0..3).map(|site| Coupling::new(site as f64 + 1.0, vec![site])),
    )
    .unwrap();

    let concrete = OperatorBuilder::on(&basis)
        .term(term.clone())
        .build(MatrixFormat::Csc)
        .unwrap();
    let erased = OperatorBuilder::on(&packed)
        .term(term)
        .build(MatrixFormat::Csc)
        .unwrap();

    assert_eq!(packed.len(), basis.len());
    assert_eq!(erased.triplets(), concrete.triplets());
}

#[test]
fn reversed_packed_basis_reorders_states_and_operator_rows_together() {
    let natural = PackedBasis::from(SpinBasis1D::builder(1).build().unwrap());
    let reversed = natural.clone().reversed();
    let term = OperatorSpec::from_product(
        OpProduct::new([LocalOperator::Y]).unwrap(),
        [Coupling::new(1.0, vec![0])],
    )
    .unwrap();
    let natural_operator = OperatorBuilder::on(&natural)
        .term(term.clone())
        .build(MatrixFormat::Dense)
        .unwrap();
    let reversed_operator = OperatorBuilder::on(&reversed)
        .term(term)
        .build(MatrixFormat::Dense)
        .unwrap();

    assert_eq!(natural.state(0).unwrap(), reversed.state(1).unwrap());
    assert_eq!(natural.state(1).unwrap(), reversed.state(0).unwrap());
    let natural_dense = natural_operator.to_dense();
    let reversed_dense = reversed_operator.to_dense();
    assert_eq!(reversed_dense[0], natural_dense[3]);
    assert_eq!(reversed_dense[1], natural_dense[2]);
    assert_eq!(reversed_dense[2], natural_dense[1]);
    assert_eq!(reversed_dense[3], natural_dense[0]);
}

#[test]
fn operator_site_permutation_changes_labels_not_assembly_code() {
    let basis = SpinBasis1D::builder(2).build().unwrap();
    let left = OperatorSpec::from_product(
        OpProduct::new([LocalOperator::X, LocalOperator::Identity]).unwrap(),
        [Coupling::new(1.0, vec![0, 1])],
    )
    .unwrap();
    let right = left.with_site_permutation(&[1, 0]).unwrap();
    let left = OperatorBuilder::on(&basis)
        .term(left)
        .build(MatrixFormat::Dense)
        .unwrap();
    let right = OperatorBuilder::on(&basis)
        .term(right)
        .build(MatrixFormat::Dense)
        .unwrap();

    assert_eq!(left.to_dense()[4].re, 0.5);
    assert_eq!(right.to_dense()[8].re, 0.5);
    assert_ne!(left.triplets(), right.triplets());
}

#[test]
fn packed_model_reuses_one_spec_for_states_matrix_and_eigh() {
    let basis = BosonBasis1D::builder(1, 4).build().unwrap();
    let terms = ["+", "-", "n"].into_iter().map(|operator| {
        let local = match operator {
            "+" => LocalOperator::Raising,
            "-" => LocalOperator::Lowering,
            "n" => LocalOperator::Number,
            _ => unreachable!(),
        };
        let coefficient = if operator == "n" { 0.25 } else { 1.0 };
        OperatorSpec::from_product(
            OpProduct::new([local]).unwrap(),
            [Coupling::new(coefficient, vec![0])],
        )
        .unwrap()
    });
    let model = PackedEdModel::new(basis, terms);

    assert_eq!(model.dimension(), 4);
    assert_eq!(model.states().unwrap(), vec![0, 1, 2, 3]);
    let operator = model.materialize(MatrixFormat::Csc).unwrap();
    let result = model
        .eigh(EighOptions {
            return_eigenvectors: false,
        })
        .unwrap();

    assert_eq!(operator.shape(), (4, 4));
    assert_eq!(result.eigenvalues.len(), 4);
    assert!(result.eigenvectors.is_empty());
    assert_abs_diff_eq!(result.eigenvalues[0], -1.885007105857148, epsilon = 1.0e-12);
}

#[test]
fn packed_model_caches_each_materialized_format() {
    let basis = SpinBasis1D::builder(2).build().unwrap();
    let term = OperatorSpec::from_product(
        OpProduct::new([LocalOperator::Z]).unwrap(),
        [Coupling::new(1.0, vec![0])],
    )
    .unwrap();
    let model = PackedEdModel::new(basis, [term]);

    let first = model.materialized(MatrixFormat::Csc).unwrap();
    let second = model.materialized(MatrixFormat::Csc).unwrap();
    let dense = model.materialized(MatrixFormat::Dense).unwrap();

    assert!(Arc::ptr_eq(&first, &second));
    assert!(!Arc::ptr_eq(&first, &dense));
    assert_eq!(first.to_dense(), dense.to_dense());
}

#[test]
fn transformed_model_does_not_reuse_a_stale_operator() {
    let basis = SpinBasis1D::builder(2).build().unwrap();
    let term = OperatorSpec::from_product(
        OpProduct::new([LocalOperator::Z]).unwrap(),
        [Coupling::new(1.0, vec![0])],
    )
    .unwrap();
    let model = PackedEdModel::new(basis, [term]);
    let original = model.materialized(MatrixFormat::Csc).unwrap();
    let permuted_model = model.with_site_permutation(&[1, 0]).unwrap();
    let permuted = permuted_model.materialized(MatrixFormat::Csc).unwrap();

    assert!(!Arc::ptr_eq(&original, &permuted));
    assert_ne!(original.triplets(), permuted.triplets());
}

#[test]
fn spin_normalization_distinguishes_ladder_and_cartesian_conventions() {
    let angular = SpinBasis1D::builder(1)
        .normalization(SpinNormalization::AngularMomentum)
        .build()
        .unwrap();
    let pauli = SpinBasis1D::builder(1)
        .normalization(SpinNormalization::Pauli)
        .build()
        .unwrap();
    let cartesian = SpinBasis1D::builder(1)
        .normalization(SpinNormalization::PauliCartesian)
        .build()
        .unwrap();

    let amplitude =
        |basis: &SpinBasis1D, operator| basis.apply_local(0, operator, &[0]).unwrap().unwrap().1.re;
    assert_abs_diff_eq!(amplitude(&angular, "+"), 1.0);
    assert_abs_diff_eq!(amplitude(&pauli, "+"), 2.0);
    assert_abs_diff_eq!(amplitude(&cartesian, "+"), 1.0);
    assert_abs_diff_eq!(amplitude(&angular, "x"), 0.5);
    assert_abs_diff_eq!(amplitude(&pauli, "x"), 1.0);
    assert_abs_diff_eq!(amplitude(&cartesian, "x"), 1.0);
    assert_abs_diff_eq!(
        angular.apply_local(0, "z", &[0]).unwrap().unwrap().1.re,
        -0.5
    );
    assert_abs_diff_eq!(pauli.apply_local(0, "z", &[0]).unwrap().unwrap().1.re, -1.0);
}

#[test]
fn temporary_terms_reuse_basis_and_support_all_algebraic_actions() {
    let basis = SpinBasis1D::builder(1).build().unwrap();
    let model = PackedEdModel::new(basis, []);
    let term = OperatorSpec::from_product(
        OpProduct::new([LocalOperator::Y]).unwrap(),
        [Coupling::new(2.0, vec![0])],
    )
    .unwrap();
    let operator = model
        .assemble_terms([term.clone()], AssemblyChecks::none(), MatrixFormat::Csc)
        .unwrap();
    let inputs = vec![vec![Complex64::new(1.0, 0.5), Complex64::new(-0.25, 2.0)]];

    for action in [
        OperatorAction::Normal,
        OperatorAction::Transpose,
        OperatorAction::Conjugate,
        OperatorAction::Adjoint,
    ] {
        let actual = model
            .apply_terms_batch([term.clone()], &inputs, action)
            .unwrap();
        let mut expected = vec![Complex64::new(0.0, 0.0); 2];
        match action {
            OperatorAction::Normal => operator.apply(&inputs[0], &mut expected).unwrap(),
            OperatorAction::Transpose => {
                operator.apply_transpose(&inputs[0], &mut expected).unwrap()
            }
            OperatorAction::Conjugate => {
                let conjugated = inputs[0]
                    .iter()
                    .map(|value| value.conj())
                    .collect::<Vec<_>>();
                operator.apply(&conjugated, &mut expected).unwrap();
                expected.iter_mut().for_each(|value| *value = value.conj());
            }
            OperatorAction::Adjoint => operator.apply_adjoint(&inputs[0], &mut expected).unwrap(),
        }
        assert_eq!(actual, vec![expected]);
    }

    let fixed_model = PackedEdModel::new(SpinBasis1D::builder(1).build().unwrap(), [term.clone()]);
    assert_eq!(
        fixed_model
            .apply_batch(&inputs, OperatorAction::Normal)
            .unwrap(),
        model
            .apply_terms_batch([term], &inputs, OperatorAction::Normal)
            .unwrap()
    );
}

#[test]
fn temporary_terms_and_bra_ket_share_the_models_site_convention() {
    let basis = SpinBasis1D::builder(2).build().unwrap();
    let model = PackedEdModel::new(basis, [])
        .with_site_permutation(&[1, 0])
        .unwrap();
    let term = OperatorSpec::from_product(
        OpProduct::new([LocalOperator::Raising]).unwrap(),
        [Coupling::new(3.0, vec![0])],
    )
    .unwrap();

    let operator = model
        .assemble_terms([term.clone()], AssemblyChecks::none(), MatrixFormat::Csc)
        .unwrap();
    assert_eq!(
        operator.triplets(),
        vec![
            (2, 0, Complex64::new(3.0, 0.0)),
            (3, 1, Complex64::new(3.0, 0.0))
        ]
    );
    let transitions = model.bra_ket_terms([term], &[0, 1]).unwrap();
    assert_eq!(transitions[0][0].bra, 2);
    assert_eq!(transitions[0][0].matrix_element, Complex64::new(3.0, 0.0));
    assert_eq!(transitions[1][0].bra, 3);

    let invalid = PackedEdModel::new(SpinBasis1D::builder(2).build().unwrap(), std::iter::empty())
        .with_site_permutation(&[0, 0]);
    assert!(invalid.is_err());
}

#[test]
fn packed_models_project_between_explicit_parent_spaces() {
    let translation = LatticeSymmetryMap::site_permutation(2, vec![1, 2, 3, 0]).unwrap();
    let reduced = GeneralBasis::new(
        SpinBasis1D::builder(4).up(2).build().unwrap(),
        SymmetrySector::new().with_map(translation, 1),
    )
    .unwrap();
    let reduced = PackedEdModel::new(reduced, []);
    let fixed_parent = PackedEdModel::new(SpinBasis1D::builder(4).up(2).build().unwrap(), []);
    let full_parent = PackedEdModel::new(SpinBasis1D::builder(4).build().unwrap(), []);
    let vectors = vec![vec![Complex64::new(0.25, -0.5); reduced.dimension()]];
    let images = reduced.reduction_images(&[3, 6, 5, 0]).unwrap();

    let fixed_lifted = reduced.lift_to_batch(&fixed_parent, &vectors).unwrap();
    let full_lifted = reduced.lift_to_batch(&full_parent, &vectors).unwrap();
    assert_eq!(fixed_lifted[0].len(), 6);
    assert_eq!(full_lifted[0].len(), 16);
    assert_eq!(
        reduced
            .project_from_batch(&fixed_parent, &fixed_lifted)
            .unwrap(),
        vectors
    );
    assert_eq!(
        reduced
            .project_from_batch(&full_parent, &full_lifted)
            .unwrap(),
        vectors
    );
    let representative = images[0].unwrap();
    assert_eq!(*representative.representative(), 3);
    assert_eq!(representative.orbit_size(), 4);
    assert!((representative.amplitude().norm() - 0.5).abs() < 1.0e-12);
    assert_eq!(
        images[1].unwrap().representative(),
        representative.representative()
    );
    assert!(images[2].is_none());
    assert!(images[3].is_none());
}

#[test]
fn packed_models_apply_terms_directly_between_sectors() {
    let source = PackedEdModel::new(
        PackedBasis::from(SpinBasis1D::builder(3).up(0).build().unwrap()).reversed(),
        [],
    )
    .with_site_permutation(&[2, 1, 0])
    .unwrap();
    let target = PackedEdModel::new(
        PackedBasis::from(SpinBasis1D::builder(3).up(1).build().unwrap()).reversed(),
        [],
    )
    .with_site_permutation(&[2, 1, 0])
    .unwrap();
    let term = OperatorSpec::from_product(
        OpProduct::new([LocalOperator::Raising]).unwrap(),
        [Coupling::new(2.0, vec![0])],
    )
    .unwrap();
    let actual = target
        .apply_terms_from_batch(&source, [term], &[vec![Complex64::new(1.0, 0.0)]])
        .unwrap();

    assert_eq!(actual[0].len(), 3);
    assert_eq!(
        actual[0]
            .iter()
            .filter(|value| value.norm() > f64::EPSILON)
            .count(),
        1
    );
    assert_eq!(
        actual[0]
            .iter()
            .copied()
            .find(|value| value.norm() > f64::EPSILON)
            .unwrap(),
        Complex64::new(2.0, 0.0)
    );
}

#[test]
fn packed_spin_basis_can_represent_an_empty_symmetry_sector() {
    let basis = SpinBasis1D::builder(4)
        .momentum(0)
        .parity(-1)
        .build()
        .unwrap();
    let model = PackedEdModel::new(basis, []);

    assert_eq!(model.dimension(), 0);
    assert!(model.states().unwrap().is_empty());
}
