//! Runtime-owned exact-diagonalization model shared by language frontends.
//!
//! The native generic API remains the zero-cost path. This module provides a
//! small owned narrow waist for frontends that select a packed basis at
//! runtime and need to reuse one mathematical model across materialization and
//! solver operations.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

use num_complex::Complex64;

use crate::basis::{Basis, BasisProjector, PackedBasis, ReductionImage};
use crate::operator::{
    AssemblyChecks, BraKetTransition, LinearOperator, MatrixFormat, Operator, OperatorBuilder,
    OperatorSpec, apply_sector_shift,
};
use crate::solve::{Eigensystem, EighOptions, EigshOptions, eigh_with_options, eigsh};
use crate::{QmbedError, Result};

/// One owned basis and operator specification reusable across frontend calls.
#[derive(Clone, Debug)]
pub struct PackedEdModel {
    basis: PackedBasis,
    terms: Vec<OperatorSpec>,
    checks: AssemblyChecks,
    site_permutation: Option<Vec<usize>>,
    operators: Arc<Mutex<HashMap<MatrixFormat, Arc<Operator>>>>,
}

/// Algebraic view used when applying a temporary operator.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum OperatorAction {
    #[default]
    Normal,
    Transpose,
    Conjugate,
    Adjoint,
}

impl PackedEdModel {
    pub fn new(
        basis: impl Into<PackedBasis>,
        terms: impl IntoIterator<Item = OperatorSpec>,
    ) -> Self {
        Self {
            basis: basis.into(),
            terms: terms.into_iter().collect(),
            checks: AssemblyChecks::all(),
            site_permutation: None,
            operators: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    pub fn with_checks(mut self, checks: AssemblyChecks) -> Self {
        self.checks = checks;
        self.operators = Arc::new(Mutex::new(HashMap::new()));
        self
    }

    pub fn with_site_permutation(mut self, permutation: &[usize]) -> Result<Self> {
        validate_site_permutation(permutation)?;
        self.terms = self
            .terms
            .iter()
            .map(|term| term.with_site_permutation(permutation))
            .collect::<Result<Vec<_>>>()?;
        self.site_permutation = Some(match self.site_permutation {
            Some(previous) => previous.into_iter().map(|site| permutation[site]).collect(),
            None => permutation.to_vec(),
        });
        self.operators = Arc::new(Mutex::new(HashMap::new()));
        Ok(self)
    }

    pub const fn basis(&self) -> &PackedBasis {
        &self.basis
    }

    pub fn terms(&self) -> &[OperatorSpec] {
        &self.terms
    }

    pub fn dimension(&self) -> usize {
        self.basis.len()
    }

    pub fn states(&self) -> Result<Vec<u128>> {
        (0..self.basis.len())
            .map(|index| self.basis.state(index))
            .collect()
    }

    /// Query canonical representatives and normalized orbit coefficients.
    ///
    /// This works for both explicit and symmetry-reduced bases and does not
    /// materialize an operator or projector.
    pub fn reduction_images(&self, states: &[u128]) -> Result<Vec<Option<ReductionImage<u128>>>> {
        states
            .iter()
            .map(|&state| self.basis.reduction_image(state))
            .collect()
    }

    /// Build the sparse isometry from this model's basis into an explicit
    /// parent model's basis.
    ///
    /// The parent is deliberately explicit: frontends may choose either a
    /// particle-conserving parent or the unrestricted physical Hilbert space
    /// without encoding either policy in the Rust core.
    pub fn projector_to(&self, parent: &Self) -> Result<BasisProjector> {
        self.ensure_same_site_convention(parent)?;
        BasisProjector::between(&self.basis, &parent.basis)
    }

    /// Lift a batch of reduced-space vectors into an explicit parent model.
    pub fn lift_to_batch(
        &self,
        parent: &Self,
        vectors: &[Vec<Complex64>],
    ) -> Result<Vec<Vec<Complex64>>> {
        self.projector_to(parent)?.lift_batch(vectors)
    }

    /// Project a batch of parent-space vectors into this model's basis.
    pub fn project_from_batch(
        &self,
        parent: &Self,
        vectors: &[Vec<Complex64>],
    ) -> Result<Vec<Vec<Complex64>>> {
        self.projector_to(parent)?.project_batch(vectors)
    }

    /// Apply temporary terms directly from a source model into this target
    /// model without materializing either physical parent space.
    pub fn apply_terms_from_batch(
        &self,
        source: &Self,
        terms: impl IntoIterator<Item = OperatorSpec>,
        inputs: &[Vec<Complex64>],
    ) -> Result<Vec<Vec<Complex64>>> {
        self.ensure_same_site_convention(source)?;
        let terms = self.prepare_terms(terms)?;
        inputs
            .iter()
            .map(|input| {
                let mut output = vec![Complex64::new(0.0, 0.0); self.dimension()];
                apply_sector_shift(&source.basis, &self.basis, &terms, input, &mut output)?;
                Ok(output)
            })
            .collect()
    }

    /// Return the assembled operator shared by all calls using this model.
    ///
    /// Assembly is performed at most once per storage format. Clones of an
    /// unchanged model share the same cache; model-transforming builders reset
    /// it before changing checks or site labels.
    pub fn materialized(&self, format: MatrixFormat) -> Result<Arc<Operator>> {
        let mut operators = self.operators.lock().map_err(|_| {
            QmbedError::InternalState("materialized-operator cache lock is poisoned".into())
        })?;
        if let Some(operator) = operators.get(&format) {
            return Ok(Arc::clone(operator));
        }
        let operator = Arc::new(
            OperatorBuilder::on(&self.basis)
                .terms(self.terms.clone())
                .checks(self.checks)
                .build(format)?,
        );
        operators.insert(format, Arc::clone(&operator));
        Ok(operator)
    }

    /// Materialize an owned operator for callers that do not need reuse.
    pub fn materialize(&self, format: MatrixFormat) -> Result<Operator> {
        Ok((*self.materialized(format)?).clone())
    }

    /// Assemble caller-supplied terms on this model's already-owned basis.
    ///
    /// This is the native narrow waist for low-level basis operations. The
    /// terms use the model's original site convention and are relabeled by the
    /// same permutation as its persistent terms.
    pub fn assemble_terms(
        &self,
        terms: impl IntoIterator<Item = OperatorSpec>,
        checks: AssemblyChecks,
        format: MatrixFormat,
    ) -> Result<Operator> {
        OperatorBuilder::on(&self.basis)
            .terms(self.prepare_terms(terms)?)
            .checks(checks)
            .build(format)
    }

    /// Apply one temporary operator to a batch of column vectors.
    ///
    /// The operator is assembled once for the whole batch and never converted
    /// to a dense matrix.
    pub fn apply_terms_batch(
        &self,
        terms: impl IntoIterator<Item = OperatorSpec>,
        inputs: &[Vec<Complex64>],
        action: OperatorAction,
    ) -> Result<Vec<Vec<Complex64>>> {
        let operator =
            self.assemble_terms(terms, AssemblyChecks::none(), MatrixFormat::MatrixFree)?;
        apply_operator_batch(&operator, inputs, action)
    }

    /// Apply the model's persistent terms without dense materialization.
    ///
    /// The matrix-free representation is cached after the first call and
    /// reused by subsequent vectors and algebraic views.
    pub fn apply_batch(
        &self,
        inputs: &[Vec<Complex64>],
        action: OperatorAction,
    ) -> Result<Vec<Vec<Complex64>>> {
        let operator = self.materialized(MatrixFormat::MatrixFree)?;
        apply_operator_batch(operator.as_ref(), inputs, action)
    }

    /// Return raw local transitions grouped by input ket.
    ///
    /// Unlike square operator assembly, this operation intentionally does not
    /// reduce destination states into the model's symmetry sector.
    pub fn bra_ket_terms(
        &self,
        terms: impl IntoIterator<Item = OperatorSpec>,
        kets: &[u128],
    ) -> Result<Vec<Vec<BraKetTransition<u128>>>> {
        let terms = self.prepare_terms(terms)?;
        kets.iter()
            .copied()
            .map(|ket| {
                let mut transitions = Vec::new();
                for term in &terms {
                    for coupling in term.couplings() {
                        self.basis.visit_preparsed_local_unreduced_transitions(
                            ket,
                            term.operator(),
                            term.symbols(),
                            term.split(),
                            &coupling.sites,
                            |bra, amplitude| {
                                let matrix_element = coupling.coefficient * amplitude;
                                if matrix_element.norm() > f64::EPSILON {
                                    transitions.push(BraKetTransition {
                                        bra,
                                        ket,
                                        matrix_element,
                                    });
                                }
                                Ok(())
                            },
                        )?;
                    }
                }
                Ok(transitions)
            })
            .collect()
    }

    pub fn eigh(&self, options: EighOptions) -> Result<Eigensystem> {
        let operator = self.materialized(MatrixFormat::Dense)?;
        eigh_with_options(operator.as_ref(), options)
    }

    pub fn eigsh(&self, format: MatrixFormat, options: EigshOptions) -> Result<Eigensystem> {
        let operator = self.materialized(format)?;
        eigsh(operator.as_ref(), options)
    }

    fn prepare_terms(
        &self,
        terms: impl IntoIterator<Item = OperatorSpec>,
    ) -> Result<Vec<OperatorSpec>> {
        let terms = terms.into_iter();
        match &self.site_permutation {
            Some(permutation) => terms
                .map(|term| term.with_site_permutation(permutation))
                .collect(),
            None => Ok(terms.collect()),
        }
    }

    fn ensure_same_site_convention(&self, other: &Self) -> Result<()> {
        if self.site_permutation != other.site_permutation {
            return Err(QmbedError::InvalidOptions(
                "models must use the same site permutation for cross-basis operations".into(),
            ));
        }
        Ok(())
    }
}

fn validate_site_permutation(permutation: &[usize]) -> Result<()> {
    if let Some(site) = permutation
        .iter()
        .copied()
        .find(|&site| site >= permutation.len())
    {
        return Err(QmbedError::InvalidSite {
            site,
            sites: permutation.len(),
        });
    }
    if permutation.iter().copied().collect::<HashSet<_>>().len() != permutation.len() {
        return Err(QmbedError::InvalidOptions(
            "site permutation must be bijective".into(),
        ));
    }
    Ok(())
}

fn apply_operator_batch(
    operator: &dyn LinearOperator,
    inputs: &[Vec<Complex64>],
    action: OperatorAction,
) -> Result<Vec<Vec<Complex64>>> {
    let (rows, columns) = operator.shape();
    let (input_dimension, output_dimension) = match action {
        OperatorAction::Normal | OperatorAction::Conjugate => (columns, rows),
        OperatorAction::Transpose | OperatorAction::Adjoint => (rows, columns),
    };
    inputs
        .iter()
        .map(|input| {
            if input.len() != input_dimension {
                return Err(QmbedError::DimensionMismatch(format!(
                    "operator action needs input length {input_dimension}, got {}",
                    input.len()
                )));
            }
            let mut output = vec![Complex64::new(0.0, 0.0); output_dimension];
            match action {
                OperatorAction::Normal => operator.apply(input, &mut output)?,
                OperatorAction::Transpose => operator.apply_transpose(input, &mut output)?,
                OperatorAction::Conjugate => {
                    let conjugated: Vec<_> = input.iter().map(|value| value.conj()).collect();
                    operator.apply(&conjugated, &mut output)?;
                    output.iter_mut().for_each(|value| *value = value.conj());
                }
                OperatorAction::Adjoint => operator.apply_adjoint(input, &mut output)?,
            }
            Ok(output)
        })
        .collect()
}
