use qmbed::basis::{Basis, SpinBasis1D};
use qmbed::operator::{
    Coupling, LinearOperator, LocalOperator, MatrixFormat, OpProduct, OperatorBuilder, OperatorSpec,
};
use qmbed::{Complex64, QmbedError};

#[test]
fn typed_operator_product_uses_the_universal_assembler() {
    let basis = SpinBasis1D::builder(2).build().unwrap();
    let product = OpProduct::new([LocalOperator::Z, LocalOperator::Z]).unwrap();
    let term = OperatorSpec::from_product(product, [Coupling::new(1.0, vec![0, 1])]).unwrap();
    let operator = OperatorBuilder::on(&basis)
        .term(term)
        .build(MatrixFormat::Csc)
        .unwrap();

    assert_eq!(operator.shape(), (basis.len(), basis.len()));
    assert_eq!(
        operator.diagonal(),
        vec![
            Complex64::new(0.25, 0.0),
            Complex64::new(-0.25, 0.0),
            Complex64::new(-0.25, 0.0),
            Complex64::new(0.25, 0.0),
        ]
    );
}

#[test]
fn compact_labels_parse_into_the_same_typed_product() {
    let parsed = OpProduct::parse("+-|nI").unwrap();
    let typed = OpProduct::with_splits(
        [
            LocalOperator::Raising,
            LocalOperator::Lowering,
            LocalOperator::Number,
            LocalOperator::Identity,
        ],
        [2],
    )
    .unwrap();
    assert_eq!(parsed, typed);
    assert_eq!(parsed.label(), "+-|nI");
    assert!(matches!(
        OpProduct::parse("||"),
        Err(QmbedError::InvalidOperator(_))
    ));
}
