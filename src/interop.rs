//! Runtime-owned exact-diagonalization model shared by language frontends.
//!
//! The native generic API remains the zero-cost path. This module provides a
//! small owned narrow waist for frontends that select a packed basis at
//! runtime and need to reuse one mathematical model across materialization and
//! solver operations.

use std::collections::{HashMap, HashSet};
use std::hash::Hash;
use std::sync::{Arc, Mutex};

use num_complex::Complex64;

use crate::archive::OperatorArchive;
use crate::basis::{Basis, BasisProjector, PackedBasis, ReductionImage, WidePackedBasis};
use crate::block::{BlockOps, ProjectedBlockOps};
use crate::operator::{
    AssemblyChecks, BraKetTransition, LinearOperator, MatrixFormat, Operator, OperatorBuilder,
    OperatorSpec, QuantumComponent, QuantumOperator, TimeOperator, apply_sector_shift,
};
use crate::solve::{
    Eigensystem, EighOptions, EigshOptions, EvolutionOptions, StateBatchTrajectory,
    eigh_with_options, eigsh, eigsh_with_initial, evolve_batch as evolve_operator_batch,
    evolve_time_dependent_batch_from,
};
use crate::{QmbedError, Result};

/// One owned basis and operator specification reusable across frontend calls.
#[derive(Clone, Debug)]
pub struct EdModel<B> {
    basis: B,
    terms: Vec<OperatorSpec>,
    components: Vec<PackedTermComponent>,
    checks: AssemblyChecks,
    site_permutation: Option<Vec<usize>>,
    operators: Arc<Mutex<HashMap<MatrixFormat, Arc<Operator>>>>,
    operator_families: Arc<Mutex<HashMap<MatrixFormat, Arc<PackedOperatorModel>>>>,
}

pub type PackedEdModel = EdModel<PackedBasis>;
pub type WideEdModel = EdModel<WidePackedBasis>;

/// Algebraic view used when applying a temporary operator.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum OperatorAction {
    #[default]
    Normal,
    Transpose,
    Conjugate,
    Adjoint,
}

/// One named operator component expressed in the local terms of a packed
/// basis.
///
/// The component stays as a basis-level specification until a storage format
/// is requested. This lets the same parameterized model reuse the universal
/// [`OperatorBuilder`] for dense, sparse, and matrix-free execution instead of
/// introducing a frontend-specific dynamic-Hamiltonian path.
#[derive(Clone, Debug)]
pub struct PackedTermComponent {
    name: String,
    terms: Vec<OperatorSpec>,
    default: Option<Complex64>,
}

impl PackedTermComponent {
    /// Construct a component whose coefficient must be supplied by name.
    pub fn required(
        name: impl Into<String>,
        terms: impl IntoIterator<Item = OperatorSpec>,
    ) -> Self {
        Self {
            name: name.into(),
            terms: terms.into_iter().collect(),
            default: None,
        }
    }

    /// Construct a component with a finite default coefficient.
    pub fn with_default(
        name: impl Into<String>,
        terms: impl IntoIterator<Item = OperatorSpec>,
        default: impl Into<Complex64>,
    ) -> Self {
        Self {
            name: name.into(),
            terms: terms.into_iter().collect(),
            default: Some(default.into()),
        }
    }

    /// Python-compatible parameter component: an omitted coefficient equals
    /// one.
    pub fn parameter(
        name: impl Into<String>,
        terms: impl IntoIterator<Item = OperatorSpec>,
    ) -> Self {
        Self::with_default(name, terms, Complex64::new(1.0, 0.0))
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn terms(&self) -> &[OperatorSpec] {
        &self.terms
    }

    pub const fn default(&self) -> Option<Complex64> {
        self.default
    }
}

/// Runtime-owned finite operator family independent of any local-basis
/// representation.
///
/// This is the language-neutral model for operators supplied directly as
/// dense or sparse matrices. A fixed part may be combined with named
/// parameterized components, and every frontend operation evaluates the same
/// family before applying, materializing, or solving it. Basis-aware local
/// term assembly remains in [`PackedEdModel`].
#[derive(Clone, Debug)]
pub struct PackedOperatorModel {
    static_part: Operator,
    parameterized_part: Option<QuantumOperator>,
}

impl PackedOperatorModel {
    /// Construct a fixed square operator model.
    pub fn new(static_part: Operator) -> Result<Self> {
        let shape = static_part.shape();
        if shape.0 != shape.1 {
            return Err(QmbedError::DimensionMismatch(
                "a persistent operator model must be square".into(),
            ));
        }
        Ok(Self {
            static_part,
            parameterized_part: None,
        })
    }

    /// Construct a named operator family with an explicit fixed part.
    pub fn with_components(
        static_part: Operator,
        components: impl IntoIterator<Item = QuantumComponent>,
    ) -> Result<Self> {
        let mut model = Self::new(static_part)?;
        let parameterized_part = QuantumOperator::new(components)?;
        if parameterized_part.shape() != model.static_part.shape() {
            return Err(QmbedError::DimensionMismatch(
                "fixed and parameterized operators must have equal shapes".into(),
            ));
        }
        model.parameterized_part = Some(parameterized_part);
        Ok(model)
    }

    /// Construct a purely parameterized operator family.
    pub fn parameterized(
        components: impl IntoIterator<Item = QuantumComponent>,
        format: MatrixFormat,
    ) -> Result<Self> {
        let parameterized_part = QuantumOperator::new(components)?;
        let shape = parameterized_part.shape();
        let static_part = Operator::from_triplets(shape.0, shape.1, std::iter::empty(), format)?;
        Ok(Self {
            static_part,
            parameterized_part: Some(parameterized_part),
        })
    }

    pub fn dimension(&self) -> usize {
        self.static_part.shape().0
    }

    pub fn component_names(&self) -> impl Iterator<Item = &str> {
        self.parameterized_part
            .iter()
            .flat_map(QuantumOperator::component_names)
    }

    /// Return one named component without evaluating the fixed part.
    pub fn component_operator(&self, name: &str, format: MatrixFormat) -> Result<Operator> {
        let parameterized = self.parameterized_part.as_ref().ok_or_else(|| {
            QmbedError::InvalidOptions(format!("fixed operator model has no component {name:?}"))
        })?;
        parameterized.component(name)?.converted(format)
    }

    /// Export the named part of this family as a storage-preserving archive.
    ///
    /// Component formats may be selected independently. Names omitted from
    /// `formats` retain their current representation. Unknown names are
    /// rejected so a frontend typo cannot silently produce a different
    /// archive layout.
    pub fn component_archive(
        &self,
        formats: &HashMap<String, MatrixFormat>,
    ) -> Result<OperatorArchive> {
        if !self.static_part.triplets().is_empty() {
            return Err(QmbedError::InvalidOptions(
                "the named-component archive cannot omit a nonzero fixed operator".into(),
            ));
        }
        let parameterized = self.parameterized_part.as_ref().ok_or_else(|| {
            QmbedError::InvalidOptions(
                "an operator archive requires at least one named component".into(),
            )
        })?;
        if let Some(name) = formats.keys().find(|name| {
            !parameterized
                .component_names()
                .any(|candidate| candidate == name.as_str())
        }) {
            return Err(QmbedError::InvalidOptions(format!(
                "unknown operator component format {name:?}"
            )));
        }
        let mut archive = OperatorArchive::new();
        for component in parameterized.components() {
            let operator = match formats.get(component.name()) {
                Some(format) => component.operator().converted(*format)?,
                None => component.operator().clone(),
            };
            archive.insert(component.name(), operator, component.default())?;
        }
        Ok(archive)
    }

    /// Reconstruct a basis-independent parameterized family from an archive.
    ///
    /// Archived component formats are retained. The fixed part is the exact
    /// zero operator because [`OperatorArchive`] represents named families;
    /// fixed-plus-parameterized snapshots use a separate model archive
    /// contract.
    pub fn from_component_archive(
        archive: OperatorArchive,
        static_format: MatrixFormat,
    ) -> Result<Self> {
        let operator = archive.into_quantum_operator()?;
        Self::parameterized(operator.components().iter().cloned(), static_format)
    }

    /// Project the fixed part and every named component through the same
    /// rectangular map, preserving component names and default coefficients.
    pub fn projected_by(&self, projector: &Operator) -> Result<Self> {
        let static_part = self.static_part.projected_by(projector)?;
        let Some(parameterized) = &self.parameterized_part else {
            return Self::new(static_part);
        };
        let components = parameterized
            .components()
            .iter()
            .map(|component| {
                let operator = component.operator().projected_by(projector)?;
                Ok(match component.default() {
                    Some(default) => {
                        QuantumComponent::with_default(component.name(), operator, default)
                    }
                    None => QuantumComponent::required(component.name(), operator),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Self::with_components(static_part, components)
    }

    fn combine_block_families<F>(
        blocks: Vec<Self>,
        lift_operators: F,
        format: MatrixFormat,
    ) -> Result<Self>
    where
        F: Fn(Vec<Operator>) -> Result<Operator>,
    {
        if blocks.is_empty() {
            return Err(QmbedError::InvalidOptions(
                "a block family requires at least one sector model".into(),
            ));
        }
        let static_part = lift_operators(
            blocks
                .iter()
                .map(|model| model.static_part.clone())
                .collect(),
        )?;

        let mut names = Vec::<String>::new();
        let mut defaults = HashMap::<String, Option<Complex64>>::new();
        for model in &blocks {
            let Some(parameterized) = &model.parameterized_part else {
                continue;
            };
            for component in parameterized.components() {
                match defaults.get(component.name()) {
                    Some(existing) if *existing != component.default() => {
                        return Err(QmbedError::InvalidOptions(format!(
                            "block component {:?} has inconsistent defaults",
                            component.name()
                        )));
                    }
                    Some(_) => {}
                    None => {
                        names.push(component.name().to_owned());
                        defaults.insert(component.name().to_owned(), component.default());
                    }
                }
            }
        }
        if names.is_empty() {
            return Self::new(static_part);
        }

        let components = names
            .into_iter()
            .map(|name| {
                let operators = blocks
                    .iter()
                    .map(|model| {
                        model
                            .parameterized_part
                            .as_ref()
                            .and_then(|operator| {
                                operator
                                    .components()
                                    .iter()
                                    .find(|component| component.name() == name)
                            })
                            .map(|component| component.operator().clone())
                            .map_or_else(
                                || {
                                    Operator::from_triplets(
                                        model.dimension(),
                                        model.dimension(),
                                        std::iter::empty(),
                                        format,
                                    )
                                },
                                Ok,
                            )
                    })
                    .collect::<Result<Vec<_>>>()?;
                let operator = lift_operators(operators)?;
                Ok(match defaults[&name] {
                    Some(default) => QuantumComponent::with_default(name, operator, default),
                    None => QuantumComponent::required(name, operator),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Self::with_components(static_part, components)
    }

    /// Assemble the direct sum of independent fixed or parameterized models.
    ///
    /// Named components are combined by name and retain one shared coefficient
    /// contract, so the result can be evaluated, evolved, or exponentiated
    /// through the ordinary persistent-model API.
    pub fn from_blocks(
        blocks: impl IntoIterator<Item = Self>,
        format: MatrixFormat,
    ) -> Result<Self> {
        let blocks: Vec<_> = blocks.into_iter().collect();
        Self::combine_block_families(
            blocks,
            |operators| {
                let operators = operators
                    .into_iter()
                    .map(|operator| Arc::new(operator) as Arc<dyn LinearOperator>)
                    .collect::<Vec<_>>();
                BlockOps::new(operators)?.materialize(format)
            },
            format,
        )
    }

    /// Assemble a shared parent-space family from independently reduced
    /// sector models and their embedding projectors.
    ///
    /// The result preserves the fixed part and the union of named components.
    /// Components with the same name are lifted from every sector into one
    /// parent-space operator; a sector which does not define that name
    /// contributes an exact zero block. Defaults for a shared name must agree,
    /// because one parameter value controls that physical component across all
    /// sectors.
    pub fn from_projected_blocks(
        blocks: impl IntoIterator<Item = (Self, Operator)>,
        tolerance: f64,
        format: MatrixFormat,
    ) -> Result<Self> {
        let blocks: Vec<_> = blocks.into_iter().collect();
        if blocks.is_empty() {
            return Err(QmbedError::InvalidOptions(
                "a projected block family requires at least one sector model".into(),
            ));
        }
        let projectors: Vec<_> = blocks
            .iter()
            .map(|(_, projector)| Arc::new(projector.clone()) as Arc<dyn LinearOperator>)
            .collect();
        let models = blocks.into_iter().map(|(model, _)| model).collect();
        Self::combine_block_families(
            models,
            |operators| {
                let operators = operators
                    .into_iter()
                    .map(|operator| Arc::new(operator) as Arc<dyn LinearOperator>)
                    .collect::<Vec<_>>();
                ProjectedBlockOps::new(operators, projectors.clone(), tolerance)?
                    .materialize(format)
            },
            format,
        )
    }

    pub fn materialize(
        &self,
        parameters: &HashMap<String, Complex64>,
        format: MatrixFormat,
    ) -> Result<Operator> {
        let fixed = self.static_part.converted(format)?;
        match &self.parameterized_part {
            Some(parameterized) => fixed.add(&parameterized.evaluate(parameters, format)?),
            None if parameters.is_empty() => Ok(fixed),
            None => {
                let name = parameters.keys().next().expect("nonempty map was checked");
                Err(QmbedError::InvalidOptions(format!(
                    "unknown operator parameter {name:?}"
                )))
            }
        }
    }

    pub fn apply_batch(
        &self,
        parameters: &HashMap<String, Complex64>,
        inputs: &[Vec<Complex64>],
        action: OperatorAction,
    ) -> Result<Vec<Vec<Complex64>>> {
        if action != OperatorAction::Normal {
            let operator = self.materialize(parameters, MatrixFormat::MatrixFree)?;
            return apply_operator_batch(&operator, inputs, action);
        }
        inputs
            .iter()
            .map(|input| {
                let mut output = vec![Complex64::new(0.0, 0.0); self.dimension()];
                self.apply(parameters, input, &mut output)?;
                Ok(output)
            })
            .collect()
    }

    /// Apply one evaluated family member directly, without assembling its
    /// weighted sparse sum.
    pub fn apply(
        &self,
        parameters: &HashMap<String, Complex64>,
        input: &[Complex64],
        output: &mut [Complex64],
    ) -> Result<()> {
        let Some(parameterized) = &self.parameterized_part else {
            if let Some(name) = parameters.keys().next() {
                return Err(QmbedError::InvalidOptions(format!(
                    "unknown operator parameter {name:?}"
                )));
            }
            self.static_part.apply(input, output)?;
            return Ok(());
        };
        let coefficients = parameterized.resolve_coefficients(parameters)?;
        self.apply_coefficients(&coefficients, input, output)
    }

    /// Apply ordered component coefficients directly. The order is exactly
    /// [`Self::component_names`].
    pub fn apply_coefficients(
        &self,
        coefficients: &[Complex64],
        input: &[Complex64],
        output: &mut [Complex64],
    ) -> Result<()> {
        self.static_part.apply(input, output)?;
        let Some(parameterized) = &self.parameterized_part else {
            if coefficients.is_empty() {
                return Ok(());
            }
            return Err(QmbedError::DimensionMismatch(
                "a fixed operator model accepts no component coefficients".into(),
            ));
        };
        let mut contribution = vec![Complex64::new(0.0, 0.0); output.len()];
        parameterized.apply_coefficients(coefficients, input, &mut contribution)?;
        for (value, addition) in output.iter_mut().zip(contribution) {
            *value += addition;
        }
        Ok(())
    }

    /// Construct a matrix-free time operator whose callback fills coefficients
    /// in the stable component order.
    pub fn time_operator<F>(&self, coefficients_at: F) -> Result<TimeOperator>
    where
        F: Fn(f64, &mut [Complex64]) -> Result<()> + Send + Sync + 'static,
    {
        self.time_operator_scaled(Complex64::new(1.0, 0.0), coefficients_at)
    }

    /// Construct a scaled matrix-free time operator. The scale applies to the
    /// complete fixed-plus-parameterized family.
    pub fn time_operator_scaled<F>(
        &self,
        operator_scale: Complex64,
        coefficients_at: F,
    ) -> Result<TimeOperator>
    where
        F: Fn(f64, &mut [Complex64]) -> Result<()> + Send + Sync + 'static,
    {
        if !operator_scale.re.is_finite() || !operator_scale.im.is_finite() {
            return Err(QmbedError::InvalidOptions(
                "time-dependent operator scale must be finite".into(),
            ));
        }
        let component_count = self.component_names().count();
        if component_count == 0 {
            return Err(QmbedError::InvalidOptions(
                "time-dependent evolution requires at least one operator component".into(),
            ));
        }
        let model = self.clone();
        TimeOperator::new(
            (self.dimension(), self.dimension()),
            move |time, input, output| {
                let mut coefficients = vec![Complex64::new(f64::NAN, f64::NAN); component_count];
                coefficients_at(time, &mut coefficients)?;
                if coefficients
                    .iter()
                    .any(|value| !value.re.is_finite() || !value.im.is_finite())
                {
                    return Err(QmbedError::InvalidOptions(format!(
                        "time-dependent coefficient callback did not return {component_count} finite values"
                    )));
                }
                model.apply_coefficients(&coefficients, input, output)?;
                for value in output {
                    *value *= operator_scale;
                }
                Ok(())
            },
        )
    }

    /// Evolve a batch while evaluating named component coefficients at the
    /// integrator's internal physical times.
    pub fn evolve_time_dependent_batch<F>(
        &self,
        initial_columns: &[Vec<Complex64>],
        initial_time: f64,
        options: EvolutionOptions,
        coefficients_at: F,
    ) -> Result<StateBatchTrajectory>
    where
        F: Fn(f64, &mut [Complex64]) -> Result<()> + Send + Sync + 'static,
    {
        self.evolve_time_dependent_batch_scaled(
            initial_columns,
            initial_time,
            options,
            Complex64::new(1.0, 0.0),
            coefficients_at,
        )
    }

    pub fn evolve_time_dependent_batch_scaled<F>(
        &self,
        initial_columns: &[Vec<Complex64>],
        initial_time: f64,
        options: EvolutionOptions,
        operator_scale: Complex64,
        coefficients_at: F,
    ) -> Result<StateBatchTrajectory>
    where
        F: Fn(f64, &mut [Complex64]) -> Result<()> + Send + Sync + 'static,
    {
        let operator = self.time_operator_scaled(operator_scale, coefficients_at)?;
        evolve_time_dependent_batch_from(&operator, initial_columns, initial_time, options)
    }

    pub fn eigh(
        &self,
        parameters: &HashMap<String, Complex64>,
        options: EighOptions,
    ) -> Result<Eigensystem> {
        let operator = self.materialize(parameters, MatrixFormat::Dense)?;
        if !operator.is_hermitian(1.0e-12) {
            return Err(QmbedError::NonHermitian);
        }
        eigh_with_options(&operator, options)
    }

    pub fn eigsh(
        &self,
        parameters: &HashMap<String, Complex64>,
        format: MatrixFormat,
        options: EigshOptions,
    ) -> Result<Eigensystem> {
        let operator = self.materialize(parameters, format)?;
        if !operator.is_hermitian(1.0e-12) {
            return Err(QmbedError::NonHermitian);
        }
        eigsh(&operator, options)
    }

    pub fn eigsh_with_initial(
        &self,
        parameters: &HashMap<String, Complex64>,
        format: MatrixFormat,
        options: EigshOptions,
        initial: &[Complex64],
    ) -> Result<Eigensystem> {
        let operator = self.materialize(parameters, format)?;
        if !operator.is_hermitian(1.0e-12) {
            return Err(QmbedError::NonHermitian);
        }
        eigsh_with_initial(&operator, options, initial)
    }

    pub fn evolve_batch(
        &self,
        parameters: &HashMap<String, Complex64>,
        initial_columns: &[Vec<Complex64>],
        options: EvolutionOptions,
    ) -> Result<StateBatchTrajectory> {
        let operator = self.materialize(parameters, MatrixFormat::MatrixFree)?;
        if options.hamiltonian && !operator.is_hermitian(1.0e-12) {
            return Err(QmbedError::NonHermitian);
        }
        evolve_operator_batch(&operator, initial_columns, options)
    }
}

impl<B> EdModel<B>
where
    B: Basis + Clone,
    B::State: Hash + Ord + 'static,
{
    pub fn new(basis: impl Into<B>, terms: impl IntoIterator<Item = OperatorSpec>) -> Self {
        Self {
            basis: basis.into(),
            terms: terms.into_iter().collect(),
            components: Vec::new(),
            checks: AssemblyChecks::all(),
            site_permutation: None,
            operators: Arc::new(Mutex::new(HashMap::new())),
            operator_families: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    pub fn with_checks(mut self, checks: AssemblyChecks) -> Self {
        self.checks = checks;
        self.operators = Arc::new(Mutex::new(HashMap::new()));
        self.operator_families = Arc::new(Mutex::new(HashMap::new()));
        self
    }

    /// Add named local-term components to this basis-owned model.
    ///
    /// Names and defaults are validated before any operator assembly. The
    /// components may be non-Hermitian individually; Hermiticity is checked
    /// after evaluating a concrete parameter set.
    pub fn with_components(
        mut self,
        components: impl IntoIterator<Item = PackedTermComponent>,
    ) -> Result<Self> {
        let components: Vec<_> = components.into_iter().collect();
        validate_term_components(&components)?;
        self.components = components;
        self.operator_families = Arc::new(Mutex::new(HashMap::new()));
        Ok(self)
    }

    pub fn with_site_permutation(mut self, permutation: &[usize]) -> Result<Self> {
        validate_site_permutation(permutation)?;
        self.terms = self
            .terms
            .iter()
            .map(|term| term.with_site_permutation(permutation))
            .collect::<Result<Vec<_>>>()?;
        self.components = self
            .components
            .into_iter()
            .map(|mut component| {
                component.terms = component
                    .terms
                    .iter()
                    .map(|term| term.with_site_permutation(permutation))
                    .collect::<Result<Vec<_>>>()?;
                Ok(component)
            })
            .collect::<Result<Vec<_>>>()?;
        self.site_permutation = Some(match self.site_permutation {
            Some(previous) => previous.into_iter().map(|site| permutation[site]).collect(),
            None => permutation.to_vec(),
        });
        self.operators = Arc::new(Mutex::new(HashMap::new()));
        self.operator_families = Arc::new(Mutex::new(HashMap::new()));
        Ok(self)
    }

    pub const fn basis(&self) -> &B {
        &self.basis
    }

    pub fn terms(&self) -> &[OperatorSpec] {
        &self.terms
    }

    pub fn components(&self) -> &[PackedTermComponent] {
        &self.components
    }

    pub fn component_names(&self) -> impl Iterator<Item = &str> {
        self.components.iter().map(PackedTermComponent::name)
    }

    pub fn dimension(&self) -> usize {
        self.basis.len()
    }

    pub fn states(&self) -> Result<Vec<B::State>> {
        (0..self.basis.len())
            .map(|index| self.basis.state(index))
            .collect()
    }

    /// Scatter a basis-ordered vector into packed-state index order.
    ///
    /// Explicit tensor-product parents may expose a frontend-compatible row
    /// ordering rather than ascending packed states. Subsystem kernels use the
    /// packed state itself as the mixed-radix row index, so this method makes
    /// that conversion explicit and reusable.
    pub fn scatter_state_vector(
        &self,
        values: &[Complex64],
        full_dimension: usize,
    ) -> Result<Vec<Complex64>>
    where
        usize: TryFrom<B::State>,
    {
        if values.len() != self.dimension() {
            return Err(QmbedError::DimensionMismatch(
                "state vector does not match the basis dimension".into(),
            ));
        }
        let mut scattered = vec![Complex64::new(0.0, 0.0); full_dimension];
        for (row, value) in values.iter().enumerate() {
            let state = usize::try_from(self.basis.state(row)?).map_err(|_| {
                QmbedError::DimensionMismatch(
                    "packed state does not fit the tensor-product index space".into(),
                )
            })?;
            if state >= full_dimension {
                return Err(QmbedError::DimensionMismatch(
                    "packed state exceeds the tensor-product index space".into(),
                ));
            }
            scattered[state] = *value;
        }
        Ok(scattered)
    }

    /// Scatter both axes of a row-major density matrix by packed state.
    pub fn scatter_density(
        &self,
        values: &[Complex64],
        full_dimension: usize,
    ) -> Result<Vec<Complex64>>
    where
        usize: TryFrom<B::State>,
    {
        let dimension = self.dimension();
        if values.len() != dimension.saturating_mul(dimension) {
            return Err(QmbedError::DimensionMismatch(
                "density matrix does not match the basis dimension".into(),
            ));
        }
        let mut indices = Vec::with_capacity(dimension);
        for state in self.states()? {
            let state = usize::try_from(state).map_err(|_| {
                QmbedError::DimensionMismatch(
                    "packed state does not fit the tensor-product index space".into(),
                )
            })?;
            if state >= full_dimension {
                return Err(QmbedError::DimensionMismatch(
                    "packed state exceeds the tensor-product index space".into(),
                ));
            }
            indices.push(state);
        }
        let mut scattered =
            vec![Complex64::new(0.0, 0.0); full_dimension.saturating_mul(full_dimension)];
        for row in 0..dimension {
            for column in 0..dimension {
                scattered[indices[row] * full_dimension + indices[column]] =
                    values[row * dimension + column];
            }
        }
        Ok(scattered)
    }

    /// Query canonical representatives and normalized orbit coefficients.
    ///
    /// This works for both explicit and symmetry-reduced bases and does not
    /// materialize an operator or projector.
    pub fn reduction_images(
        &self,
        states: &[B::State],
    ) -> Result<Vec<Option<ReductionImage<B::State>>>> {
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

    /// Build a one-hot embedding when this basis is an explicitly selected
    /// subset of a parent that uses the same physical state identifiers.
    ///
    /// Unlike [`PackedEdModel::projector_to`], this does not apply symmetry
    /// orbit amplitudes. Frontends use it for filters such as a fixed total
    /// excitation inside an otherwise identical product basis.
    pub fn embedding_to(&self, parent: &Self) -> Result<BasisProjector> {
        self.ensure_same_site_convention(parent)?;
        BasisProjector::from_embedding(&self.basis, &parent.basis)
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

    /// Apply temporary local terms between arbitrary isometric subspaces.
    ///
    /// `source` and `target` own the physical local algebras. Optional
    /// projectors describe reduced coordinates inside those explicit parent
    /// bases. Omitting a projector selects the entire corresponding parent
    /// basis, so this one operation covers ordinary sector shifts,
    /// reduced-to-full, full-to-reduced, and reduced-to-reduced actions:
    ///
    /// `output = P_target† O P_source input`.
    ///
    /// The local operator is streamed through [`PackedEdModel::apply_terms_from_batch`];
    /// neither a square parent-space operator nor a dense projector product is
    /// materialized.
    pub fn apply_terms_between_subspaces_batch(
        source: &Self,
        source_projector: Option<&BasisProjector>,
        target: &Self,
        target_projector: Option<&BasisProjector>,
        terms: impl IntoIterator<Item = OperatorSpec>,
        inputs: &[Vec<Complex64>],
    ) -> Result<Vec<Vec<Complex64>>> {
        if source_projector
            .is_some_and(|projector| projector.source_dimension() != source.dimension())
        {
            return Err(QmbedError::DimensionMismatch(
                "source projector rows do not match the source parent basis".into(),
            ));
        }
        if target_projector
            .is_some_and(|projector| projector.source_dimension() != target.dimension())
        {
            return Err(QmbedError::DimensionMismatch(
                "target projector rows do not match the target parent basis".into(),
            ));
        }
        let lifted;
        let parent_inputs = match source_projector {
            Some(projector) => {
                lifted = projector.lift_batch(inputs)?;
                lifted.as_slice()
            }
            None => inputs,
        };
        let parent_outputs = target.apply_terms_from_batch(source, terms, parent_inputs)?;
        match target_projector {
            Some(projector) => projector.project_batch(&parent_outputs),
            None => Ok(parent_outputs),
        }
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
        self.materialize_with(&HashMap::new(), format)
    }

    /// Evaluate this basis-owned fixed/parameterized operator family.
    pub fn materialize_with(
        &self,
        parameters: &HashMap<String, Complex64>,
        format: MatrixFormat,
    ) -> Result<Operator> {
        if self.components.is_empty() {
            if let Some(name) = parameters.keys().next() {
                return Err(QmbedError::InvalidOptions(format!(
                    "unknown operator parameter {name:?}"
                )));
            }
            return Ok((*self.materialized(format)?).clone());
        }
        self.operator_family(format)?
            .materialize(parameters, format)
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
        self.apply_batch_with(&HashMap::new(), inputs, action)
    }

    /// Apply one evaluated member of this basis-owned operator family.
    pub fn apply_batch_with(
        &self,
        parameters: &HashMap<String, Complex64>,
        inputs: &[Vec<Complex64>],
        action: OperatorAction,
    ) -> Result<Vec<Vec<Complex64>>> {
        if self.components.is_empty() {
            if let Some(name) = parameters.keys().next() {
                return Err(QmbedError::InvalidOptions(format!(
                    "unknown operator parameter {name:?}"
                )));
            }
            let operator = self.materialized(MatrixFormat::MatrixFree)?;
            return apply_operator_batch(operator.as_ref(), inputs, action);
        }
        self.operator_family(MatrixFormat::MatrixFree)?
            .apply_batch(parameters, inputs, action)
    }

    /// Return raw local transitions grouped by input ket.
    ///
    /// Unlike square operator assembly, this operation intentionally does not
    /// reduce destination states into the model's symmetry sector.
    pub fn bra_ket_terms(
        &self,
        terms: impl IntoIterator<Item = OperatorSpec>,
        kets: &[B::State],
    ) -> Result<Vec<Vec<BraKetTransition<B::State>>>> {
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
        self.eigh_with(&HashMap::new(), options)
    }

    pub fn eigh_with(
        &self,
        parameters: &HashMap<String, Complex64>,
        options: EighOptions,
    ) -> Result<Eigensystem> {
        if self.components.is_empty() {
            if let Some(name) = parameters.keys().next() {
                return Err(QmbedError::InvalidOptions(format!(
                    "unknown operator parameter {name:?}"
                )));
            }
            let operator = self.materialized(MatrixFormat::Dense)?;
            return eigh_with_options(operator.as_ref(), options);
        }
        self.operator_family(MatrixFormat::Dense)?
            .eigh(parameters, options)
    }

    pub fn eigsh(&self, format: MatrixFormat, options: EigshOptions) -> Result<Eigensystem> {
        self.eigsh_with(&HashMap::new(), format, options)
    }

    pub fn eigsh_with(
        &self,
        parameters: &HashMap<String, Complex64>,
        format: MatrixFormat,
        options: EigshOptions,
    ) -> Result<Eigensystem> {
        self.eigsh_with_optional_initial(parameters, format, options, None)
    }

    pub fn eigsh_with_initial(
        &self,
        format: MatrixFormat,
        options: EigshOptions,
        initial: &[Complex64],
    ) -> Result<Eigensystem> {
        self.eigsh_with_initial_and_parameters(&HashMap::new(), format, options, initial)
    }

    pub fn eigsh_with_initial_and_parameters(
        &self,
        parameters: &HashMap<String, Complex64>,
        format: MatrixFormat,
        options: EigshOptions,
        initial: &[Complex64],
    ) -> Result<Eigensystem> {
        self.eigsh_with_optional_initial(parameters, format, options, Some(initial))
    }

    fn eigsh_with_optional_initial(
        &self,
        parameters: &HashMap<String, Complex64>,
        format: MatrixFormat,
        options: EigshOptions,
        initial: Option<&[Complex64]>,
    ) -> Result<Eigensystem> {
        if self.components.is_empty() {
            if let Some(name) = parameters.keys().next() {
                return Err(QmbedError::InvalidOptions(format!(
                    "unknown operator parameter {name:?}"
                )));
            }
            let operator = self.materialized(format)?;
            return match initial {
                Some(initial) => eigsh_with_initial(operator.as_ref(), options, initial),
                None => eigsh(operator.as_ref(), options),
            };
        }
        let family = self.operator_family(format)?;
        match initial {
            Some(initial) => family.eigsh_with_initial(parameters, format, options, initial),
            None => family.eigsh(parameters, format, options),
        }
    }

    /// Evolve independent state columns under this model's static Hamiltonian.
    ///
    /// The matrix-free operator is cached and shared with normal operator
    /// applications. Numerical evolution remains a Rust-core capability; the
    /// language bindings only convert arrays and time grids.
    pub fn evolve_batch(
        &self,
        initial_columns: &[Vec<Complex64>],
        options: EvolutionOptions,
    ) -> Result<StateBatchTrajectory> {
        self.evolve_batch_with(&HashMap::new(), initial_columns, options)
    }

    pub fn evolve_batch_with(
        &self,
        parameters: &HashMap<String, Complex64>,
        initial_columns: &[Vec<Complex64>],
        options: EvolutionOptions,
    ) -> Result<StateBatchTrajectory> {
        if self.components.is_empty() {
            if let Some(name) = parameters.keys().next() {
                return Err(QmbedError::InvalidOptions(format!(
                    "unknown operator parameter {name:?}"
                )));
            }
            let operator = self.materialized(MatrixFormat::MatrixFree)?;
            return evolve_operator_batch(operator.as_ref(), initial_columns, options);
        }
        self.operator_family(MatrixFormat::MatrixFree)?
            .evolve_batch(parameters, initial_columns, options)
    }

    /// Convert this basis-owned model into a basis-independent operator family
    /// without losing named components or their defaults.
    pub fn operator_model(&self, format: MatrixFormat) -> Result<PackedOperatorModel> {
        if self.components.is_empty() {
            return PackedOperatorModel::new(self.materialized(format)?.as_ref().clone());
        }
        Ok(self.operator_family(format)?.as_ref().clone())
    }

    /// Evolve this basis-owned parameterized family with coefficients supplied
    /// at the integrator's internal physical times.
    pub fn evolve_time_dependent_batch<F>(
        &self,
        initial_columns: &[Vec<Complex64>],
        initial_time: f64,
        options: EvolutionOptions,
        coefficients_at: F,
    ) -> Result<StateBatchTrajectory>
    where
        F: Fn(f64, &mut [Complex64]) -> Result<()> + Send + Sync + 'static,
    {
        self.evolve_time_dependent_batch_scaled(
            initial_columns,
            initial_time,
            options,
            Complex64::new(1.0, 0.0),
            coefficients_at,
        )
    }

    pub fn evolve_time_dependent_batch_scaled<F>(
        &self,
        initial_columns: &[Vec<Complex64>],
        initial_time: f64,
        options: EvolutionOptions,
        operator_scale: Complex64,
        coefficients_at: F,
    ) -> Result<StateBatchTrajectory>
    where
        F: Fn(f64, &mut [Complex64]) -> Result<()> + Send + Sync + 'static,
    {
        self.operator_family(MatrixFormat::MatrixFree)?
            .evolve_time_dependent_batch_scaled(
                initial_columns,
                initial_time,
                options,
                operator_scale,
                coefficients_at,
            )
    }

    fn operator_family(&self, format: MatrixFormat) -> Result<Arc<PackedOperatorModel>> {
        let mut families = self.operator_families.lock().map_err(|_| {
            QmbedError::InternalState("parameterized-operator cache lock is poisoned".into())
        })?;
        if let Some(family) = families.get(&format) {
            return Ok(Arc::clone(family));
        }
        let component_checks = AssemblyChecks {
            hermiticity: false,
            particle_conservation: self.checks.particle_conservation,
            symmetry_compatibility: self.checks.symmetry_compatibility,
        };
        let components = self
            .components
            .iter()
            .map(|component| {
                let operator = OperatorBuilder::on(&self.basis)
                    .terms(component.terms.clone())
                    .checks(component_checks)
                    .build(format)?;
                Ok(match component.default {
                    Some(default) => {
                        QuantumComponent::with_default(component.name.clone(), operator, default)
                    }
                    None => QuantumComponent::required(component.name.clone(), operator),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let family = Arc::new(PackedOperatorModel::with_components(
            (*self.materialized(format)?).clone(),
            components,
        )?);
        families.insert(format, Arc::clone(&family));
        Ok(family)
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

fn validate_term_components(components: &[PackedTermComponent]) -> Result<()> {
    let mut names = HashSet::new();
    for component in components {
        if component.name.is_empty() || !names.insert(component.name.clone()) {
            return Err(QmbedError::InvalidOptions(
                "component names must be nonempty and unique".into(),
            ));
        }
        if component
            .default
            .is_some_and(|value| !value.re.is_finite() || !value.im.is_finite())
        {
            return Err(QmbedError::InvalidOptions(
                "component defaults must be finite".into(),
            ));
        }
    }
    Ok(())
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
