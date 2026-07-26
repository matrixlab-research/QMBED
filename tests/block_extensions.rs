use std::collections::HashMap;
use std::sync::Arc;

use approx::assert_abs_diff_eq;
use qmbed::Complex64;
use qmbed::block::{
    BlockOps, DynamicBlockOps, ProjectedBlockOps, ProjectedDynamicBlockOps, block_diag_hamiltonian,
};
use qmbed::interop::PackedOperatorModel;
use qmbed::operator::{
    Dynamic, DynamicComponent, Hamiltonian, LinearOperator, MatrixFormat, Operator,
    QuantumComponent, TimeDependentOperator,
};

fn diagonal(values: &[f64]) -> Operator {
    let dimension = values.len();
    let mut dense = vec![Complex64::new(0.0, 0.0); dimension * dimension];
    for (index, value) in values.iter().enumerate() {
        dense[index * dimension + index] = Complex64::new(*value, 0.0);
    }
    Operator::from_dense(dimension, dimension, dense).unwrap()
}

fn projector(rows: usize, columns: usize, entries: &[(usize, usize, f64)]) -> Operator {
    Operator::from_triplets(
        rows,
        columns,
        entries
            .iter()
            .map(|&(row, column, value)| (row, column, Complex64::new(value, 0.0))),
        MatrixFormat::Csc,
    )
    .unwrap()
}

#[test]
fn delayed_and_materialized_static_blocks_agree() {
    let blocks: Vec<Arc<dyn LinearOperator>> =
        vec![Arc::new(diagonal(&[-2.0, 1.0])), Arc::new(diagonal(&[3.0]))];
    let delayed = BlockOps::new(blocks.clone()).unwrap();
    let materialized = block_diag_hamiltonian(blocks, MatrixFormat::Csc).unwrap();
    assert_eq!(delayed.shape(), (3, 3));
    let input = vec![
        Complex64::new(1.0, 0.0),
        Complex64::new(-0.5, 0.0),
        Complex64::new(2.0, 0.0),
    ];
    let mut left = vec![Complex64::new(0.0, 0.0); 3];
    let mut right = left.clone();
    delayed.apply(&input, &mut left).unwrap();
    materialized.apply(&input, &mut right).unwrap();
    assert_eq!(left, right);
    assert_eq!(
        delayed.materialize(MatrixFormat::Csr).unwrap().to_dense(),
        materialized.to_dense()
    );
}

#[test]
fn dynamic_blocks_apply_each_sector_at_the_same_time() {
    let first = Hamiltonian::<Dynamic>::new(
        diagonal(&[0.0]),
        vec![DynamicComponent::new(diagonal(&[2.0]), |time| {
            Complex64::new(time, 0.0)
        })],
    )
    .unwrap();
    let second = Hamiltonian::<Dynamic>::new(
        diagonal(&[1.0]),
        vec![DynamicComponent::new(diagonal(&[-1.0]), |time| {
            Complex64::new(time, 0.0)
        })],
    )
    .unwrap();
    let blocks: Vec<Arc<dyn TimeDependentOperator>> = vec![Arc::new(first), Arc::new(second)];
    let dynamic = DynamicBlockOps::new(blocks).unwrap();
    let mut output = vec![Complex64::new(0.0, 0.0); 2];
    dynamic
        .apply_at(
            0.25,
            &[Complex64::new(2.0, 0.0), Complex64::new(4.0, 0.0)],
            &mut output,
        )
        .unwrap();
    assert_abs_diff_eq!(output[0].re, 1.0, epsilon = 1.0e-12);
    assert_abs_diff_eq!(output[1].re, 3.0, epsilon = 1.0e-12);
}

#[test]
fn projected_blocks_round_trip_and_act_in_the_parent_space() {
    let blocks: Vec<Arc<dyn LinearOperator>> =
        vec![Arc::new(diagonal(&[2.0, 5.0])), Arc::new(diagonal(&[3.0]))];
    let projectors: Vec<Arc<dyn LinearOperator>> = vec![
        Arc::new(projector(3, 2, &[(0, 0, 1.0), (2, 1, 1.0)])),
        Arc::new(projector(3, 1, &[(1, 0, 1.0)])),
    ];
    let projected = ProjectedBlockOps::new(blocks, projectors, 1.0e-12).unwrap();
    assert_eq!(projected.blocks(), 2);
    assert_eq!(projected.full_dimension(), 3);
    assert_eq!(projected.block_dimension(), 3);
    assert_abs_diff_eq!(
        projected.completeness_residual().unwrap(),
        0.0,
        epsilon = 1.0e-12
    );

    let parent = vec![
        Complex64::new(1.0, 0.0),
        Complex64::new(2.0, 0.0),
        Complex64::new(3.0, 0.0),
    ];
    let mut coordinates = vec![Complex64::new(0.0, 0.0); 3];
    projected.project(&parent, &mut coordinates).unwrap();
    assert_eq!(
        coordinates,
        vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(3.0, 0.0),
            Complex64::new(2.0, 0.0)
        ]
    );
    let mut reconstructed = vec![Complex64::new(0.0, 0.0); 3];
    projected.lift(&coordinates, &mut reconstructed).unwrap();
    assert_eq!(reconstructed, parent);

    let mut output = vec![Complex64::new(0.0, 0.0); 3];
    projected.apply(&parent, &mut output).unwrap();
    assert_eq!(
        output,
        vec![
            Complex64::new(2.0, 0.0),
            Complex64::new(6.0, 0.0),
            Complex64::new(15.0, 0.0)
        ]
    );
    assert_eq!(
        projected.materialize(MatrixFormat::Csc).unwrap().to_dense(),
        diagonal(&[2.0, 3.0, 5.0]).to_dense()
    );
}

#[test]
fn projected_blocks_reject_overlapping_sector_projectors() {
    let blocks: Vec<Arc<dyn LinearOperator>> =
        vec![Arc::new(diagonal(&[1.0])), Arc::new(diagonal(&[2.0]))];
    let projectors: Vec<Arc<dyn LinearOperator>> = vec![
        Arc::new(projector(2, 1, &[(0, 0, 1.0)])),
        Arc::new(projector(2, 1, &[(0, 0, 1.0)])),
    ];
    let error = ProjectedBlockOps::new(blocks, projectors, 1.0e-12)
        .err()
        .expect("overlapping sector projectors must be rejected");
    assert!(error.to_string().contains("not an isometry"));
}

#[test]
fn projected_dynamic_blocks_share_one_physical_time_and_parent_space() {
    let first = Hamiltonian::<Dynamic>::new(
        diagonal(&[1.0]),
        vec![DynamicComponent::new(diagonal(&[2.0]), |time| {
            Complex64::new(time, 0.0)
        })],
    )
    .unwrap();
    let second = Hamiltonian::<Dynamic>::new(
        diagonal(&[-1.0]),
        vec![DynamicComponent::new(diagonal(&[4.0]), |time| {
            Complex64::new(time, 0.0)
        })],
    )
    .unwrap();
    let blocks: Vec<Arc<dyn TimeDependentOperator>> = vec![Arc::new(first), Arc::new(second)];
    let projectors: Vec<Arc<dyn LinearOperator>> = vec![
        Arc::new(projector(2, 1, &[(1, 0, 1.0)])),
        Arc::new(projector(2, 1, &[(0, 0, 1.0)])),
    ];
    let projected = ProjectedDynamicBlockOps::new(blocks, projectors, 1.0e-12).unwrap();
    let input = vec![Complex64::new(3.0, 0.0), Complex64::new(5.0, 0.0)];
    let mut output = vec![Complex64::new(0.0, 0.0); 2];
    projected.apply_at(0.5, &input, &mut output).unwrap();
    assert_abs_diff_eq!(output[0].re, 3.0, epsilon = 1.0e-12);
    assert_abs_diff_eq!(output[1].re, 10.0, epsilon = 1.0e-12);
    assert_eq!(
        projected
            .materialize(0.5, MatrixFormat::Csr)
            .unwrap()
            .to_dense(),
        diagonal(&[1.0, 2.0]).to_dense()
    );
}

#[test]
fn projected_block_family_preserves_named_dynamic_components() {
    let first = PackedOperatorModel::with_components(
        diagonal(&[1.0, 2.0]),
        [QuantumComponent::required("drive", diagonal(&[10.0, 20.0]))],
    )
    .unwrap();
    let second = PackedOperatorModel::with_components(
        diagonal(&[3.0]),
        [QuantumComponent::required("drive", diagonal(&[30.0]))],
    )
    .unwrap();
    let family = PackedOperatorModel::from_projected_blocks(
        [
            (first, projector(3, 2, &[(0, 0, 1.0), (2, 1, 1.0)])),
            (second, projector(3, 1, &[(1, 0, 1.0)])),
        ],
        1.0e-12,
        MatrixFormat::Csc,
    )
    .unwrap();
    assert_eq!(family.component_names().collect::<Vec<_>>(), vec!["drive"]);
    let parameters = HashMap::from([("drive".to_string(), Complex64::new(0.5, 0.0))]);
    assert_eq!(
        family
            .materialize(&parameters, MatrixFormat::Csc)
            .unwrap()
            .to_dense(),
        diagonal(&[6.0, 18.0, 12.0]).to_dense()
    );
}
