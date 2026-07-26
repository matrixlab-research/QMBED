use qmbed::basis::{Basis, SpinBasis1D};
use qmbed::compat::quspin::parse_operator_product;
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
fn quspin_strings_are_parsed_once_at_the_compatibility_boundary() {
    let product = parse_operator_product("+-|nI").unwrap();
    assert_eq!(product.split(), Some(2));
    assert_eq!(
        product.local_operators(),
        &[
            LocalOperator::Raising,
            LocalOperator::Lowering,
            LocalOperator::Number,
            LocalOperator::Identity,
        ]
    );
    assert_eq!(product.label(), "+-|nI");

    let recursive = parse_operator_product("+||-").unwrap();
    assert_eq!(recursive.splits(), &[1, 1]);
    assert_eq!(recursive.label(), "+||-");
    assert!(matches!(
        parse_operator_product("||"),
        Err(QmbedError::InvalidOperator(_))
    ));
}
