//! Compatibility namespaces for APIs that predate QMBED's native surface.

/// QuSpin-derived Rust spellings retained during the QMBED migration.
///
/// New code should import QMBED modules directly. This namespace is an adapter:
/// it does not contain a second implementation or a model-specific execution
/// path.
pub mod quspin {
    use crate::operator::{Coupling, LocalOperator, OpProduct, OperatorTerm};

    pub use crate::archive;
    pub use crate::basis;
    pub use crate::block;
    pub use crate::dynamics;
    pub use crate::error::QuSpinError;
    pub use crate::measure;
    pub use crate::operator;
    pub use crate::solve;
    pub use crate::workflow;
    pub use crate::{Complex64, Result, VERSION};

    /// Parse the compact QuSpin operator-string grammar once at the
    /// compatibility boundary.
    pub fn parse_operator_product(operator: impl AsRef<str>) -> Result<OpProduct> {
        let operator = operator.as_ref();
        let mut splits = Vec::new();
        let mut local = Vec::with_capacity(operator.chars().count());
        for symbol in operator.chars() {
            if symbol == '|' {
                splits.push(local.len());
                continue;
            }
            let typed = match symbol {
                'I' => LocalOperator::Identity,
                'n' => LocalOperator::Number,
                'z' => LocalOperator::Z,
                '+' => LocalOperator::Raising,
                '-' => LocalOperator::Lowering,
                'x' => LocalOperator::X,
                'y' => LocalOperator::Y,
                custom => LocalOperator::Custom(custom),
            };
            local.push(typed);
        }
        OpProduct::with_splits(local, splits)
    }

    pub fn operator_term(
        operator: impl AsRef<str>,
        couplings: impl IntoIterator<Item = Coupling>,
    ) -> Result<OperatorTerm> {
        OperatorTerm::from_product(parse_operator_product(operator)?, couplings)
    }
}

impl crate::operator::OperatorTerm {
    /// Compatibility constructor for QuSpin's compact operator-string grammar.
    ///
    /// Native QMBED code should prefer [`Self::from_product`].
    pub fn new(
        operator: impl AsRef<str>,
        couplings: impl IntoIterator<Item = crate::operator::Coupling>,
    ) -> crate::Result<Self> {
        quspin::operator_term(operator, couplings)
    }
}
