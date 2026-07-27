//! Rust-native differentiation primitives for QMBED scientific operations.
//!
//! The native layer owns the mathematical derivative semantics. Optional
//! adapters such as the optional `chainrules` module only translate these tested primitives into
//! another protocol; they do not reimplement the derivatives.

use std::collections::HashSet;
use std::sync::Arc;

use crate::operator::{LinearOperator, MatrixFormat, Operator, QuantumOperator};
use crate::runtime::{Runtime, RuntimeAdjointLinearOperator, RuntimeBuffer, RuntimeLinearOperator};
use crate::solve::{EigshOptions, EigshWorkspace, SpectrumTarget, eigsh_with_workspace};
use crate::{Complex64, QmbedError, Result};

/// Differential domain of one ordered operator parameter.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ParameterDomain {
    /// A real parameter represented by a complex value with zero imaginary part.
    Real,
    /// A complex parameter differentiated under the real pairing `Re(x†y)`.
    Complex,
}

/// Stable names and differential domains for an ordered parameter vector.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ParameterSchema {
    names: Arc<[String]>,
    domains: Arc<[ParameterDomain]>,
}

impl ParameterSchema {
    /// Construct a schema after validating names and domain count.
    pub fn new(
        names: impl IntoIterator<Item = String>,
        domains: impl IntoIterator<Item = ParameterDomain>,
    ) -> Result<Self> {
        let names: Vec<_> = names.into_iter().collect();
        let domains: Vec<_> = domains.into_iter().collect();
        if names.len() != domains.len() {
            return Err(QmbedError::DimensionMismatch(format!(
                "parameter schema has {} names and {} domains",
                names.len(),
                domains.len()
            )));
        }
        let mut unique = HashSet::with_capacity(names.len());
        if let Some(name) = names
            .iter()
            .find(|name| name.is_empty() || !unique.insert(name.as_str()))
        {
            return Err(QmbedError::InvalidOptions(format!(
                "parameter names must be nonempty and unique; invalid name {name:?}"
            )));
        }
        Ok(Self {
            names: names.into(),
            domains: domains.into(),
        })
    }

    /// Create a uniform schema from a [`QuantumOperator`]'s stable component order.
    pub fn for_operator(operator: &QuantumOperator, domain: ParameterDomain) -> Self {
        let names: Vec<_> = operator.component_names().map(str::to_owned).collect();
        let domains = vec![domain; names.len()];
        Self {
            names: names.into(),
            domains: domains.into(),
        }
    }

    /// Create a mixed real/complex schema for an operator.
    pub fn for_operator_with_domains(
        operator: &QuantumOperator,
        domains: impl IntoIterator<Item = ParameterDomain>,
    ) -> Result<Self> {
        Self::new(operator.component_names().map(str::to_owned), domains)
    }

    /// Ordered parameter names.
    pub fn names(&self) -> &[String] {
        &self.names
    }

    /// Ordered differential domains.
    pub fn domains(&self) -> &[ParameterDomain] {
        &self.domains
    }

    /// Number of parameters.
    pub fn len(&self) -> usize {
        self.names.len()
    }

    /// Whether the schema is empty.
    pub fn is_empty(&self) -> bool {
        self.names.is_empty()
    }

    /// Return the stable index of one parameter name.
    pub fn index_of(&self, name: &str) -> Option<usize> {
        self.names.iter().position(|candidate| candidate == name)
    }
}

/// Ordered primal values for a parameterized operator.
#[derive(Clone, Debug, PartialEq)]
pub struct ParameterValues {
    schema: Arc<ParameterSchema>,
    values: Vec<Complex64>,
}

impl ParameterValues {
    /// Construct values under an existing schema.
    pub fn new(
        schema: Arc<ParameterSchema>,
        values: impl IntoIterator<Item = Complex64>,
    ) -> Result<Self> {
        let values: Vec<_> = values.into_iter().collect();
        validate_parameter_vector(&schema, &values, "parameter values")?;
        Ok(Self { schema, values })
    }

    /// Construct an all-real parameter vector in operator component order.
    pub fn real(operator: &QuantumOperator, values: impl IntoIterator<Item = f64>) -> Result<Self> {
        let schema = Arc::new(ParameterSchema::for_operator(
            operator,
            ParameterDomain::Real,
        ));
        Self::new(
            schema,
            values.into_iter().map(|value| Complex64::new(value, 0.0)),
        )
    }

    /// Construct an all-complex parameter vector in operator component order.
    pub fn complex(
        operator: &QuantumOperator,
        values: impl IntoIterator<Item = Complex64>,
    ) -> Result<Self> {
        let schema = Arc::new(ParameterSchema::for_operator(
            operator,
            ParameterDomain::Complex,
        ));
        Self::new(schema, values)
    }

    /// Shared parameter schema.
    pub fn schema(&self) -> &Arc<ParameterSchema> {
        &self.schema
    }

    /// Ordered numeric values.
    pub fn values(&self) -> &[Complex64] {
        &self.values
    }

    /// Construct a direction in the same differential space.
    pub fn direction(
        &self,
        values: impl IntoIterator<Item = Complex64>,
    ) -> Result<ParameterDirection> {
        ParameterDirection::new(Arc::clone(&self.schema), values)
    }

    /// Return one named value.
    pub fn get(&self, name: &str) -> Option<Complex64> {
        self.schema.index_of(name).map(|index| self.values[index])
    }
}

/// Ordered forward perturbation of [`ParameterValues`].
#[derive(Clone, Debug, PartialEq)]
pub struct ParameterDirection {
    schema: Arc<ParameterSchema>,
    values: Vec<Complex64>,
}

impl ParameterDirection {
    /// Construct a direction after checking its schema and domains.
    pub fn new(
        schema: Arc<ParameterSchema>,
        values: impl IntoIterator<Item = Complex64>,
    ) -> Result<Self> {
        let values: Vec<_> = values.into_iter().collect();
        validate_parameter_vector(&schema, &values, "parameter direction")?;
        Ok(Self { schema, values })
    }

    /// Shared parameter schema.
    pub fn schema(&self) -> &Arc<ParameterSchema> {
        &self.schema
    }

    /// Ordered direction values.
    pub fn values(&self) -> &[Complex64] {
        &self.values
    }
}

/// Ordered reverse sensitivity of [`ParameterValues`].
#[derive(Clone, Debug, PartialEq)]
pub struct ParameterGradient {
    schema: Arc<ParameterSchema>,
    values: Vec<Complex64>,
}

impl ParameterGradient {
    fn new(schema: Arc<ParameterSchema>, values: Vec<Complex64>) -> Self {
        Self { schema, values }
    }

    /// Shared parameter schema.
    pub fn schema(&self) -> &Arc<ParameterSchema> {
        &self.schema
    }

    /// Ordered gradient values.
    pub fn values(&self) -> &[Complex64] {
        &self.values
    }

    /// Return one named gradient.
    pub fn get(&self, name: &str) -> Option<Complex64> {
        self.schema.index_of(name).map(|index| self.values[index])
    }
}

/// Reliability classification attached to a scientific gradient.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GradientStatus {
    /// The rule is exact or all requested residual and conditioning checks passed.
    Reliable,
    /// The derivative exists, but diagnostics indicate poor conditioning.
    IllConditioned,
    /// The requested mathematical derivative is undefined.
    NonDifferentiable,
}

/// Backend-independent evidence returned with a native derivative.
#[derive(Clone, Debug, PartialEq)]
pub struct GradientDiagnostics {
    /// Reliability classification.
    pub status: GradientStatus,
    /// Optional primal numerical residual.
    pub primal_residual: Option<f64>,
    /// Optional reverse numerical residual.
    pub backward_residual: Option<f64>,
    /// Optional spectral gap or separation diagnostic.
    pub spectral_gap: Option<f64>,
    /// Number of component actions used by the primal/JVP computation.
    pub primal_applications: usize,
    /// Number of component actions used by the reverse computation.
    pub backward_applications: usize,
    /// Number of forward states or actions recomputed by the reverse pass.
    pub recomputations: usize,
}

/// Ground-state energy, Hellmann--Feynman gradient, and solver evidence.
///
/// This rule differentiates an isolated lowest eigenvalue of a real-parameter
/// Hermitian operator family. It deliberately does not return a derivative of
/// an arbitrarily phased eigenvector.
#[derive(Clone, Debug)]
pub struct GroundStateEnergyGradient {
    /// Algebraically smallest eigenvalue.
    pub energy: f64,
    /// Normalized eigenvector used by the Hellmann--Feynman rule.
    pub state: Vec<Complex64>,
    /// Ordered `dE/dθᵢ = ⟨ψ|Aᵢ|ψ⟩`.
    pub gradient: ParameterGradient,
    /// Residual and spectral-separation evidence.
    pub diagnostics: GradientDiagnostics,
    /// Eigensolver iteration count.
    pub eigensolver_iterations: usize,
}

struct BoundQuantumOperator<'a> {
    family: &'a QuantumOperator,
    coefficients: &'a [Complex64],
}

impl LinearOperator for BoundQuantumOperator<'_> {
    fn shape(&self) -> (usize, usize) {
        self.family.shape()
    }

    fn format(&self) -> MatrixFormat {
        MatrixFormat::MatrixFree
    }

    fn apply(&self, input: &[Complex64], output: &mut [Complex64]) -> Result<()> {
        self.family
            .apply_coefficients(self.coefficients, input, output)
    }

    fn is_real(&self) -> bool {
        self.coefficients.iter().all(|value| value.im == 0.0)
            && self
                .family
                .components()
                .iter()
                .all(|component| component.operator().is_real())
    }
}

/// Differentiate the isolated algebraic ground-state energy.
///
/// The forward solve requests at least two eigenpairs so the returned
/// diagnostics can expose the spectral gap. Reverse work then requires one
/// component action per parameter, instead of the two additional eigensolves
/// per parameter required by central finite differences.
///
/// All parameters and every component must be real and Hermitian. Degenerate
/// or numerically unresolved ground states are reported as
/// [`GradientStatus::NonDifferentiable`] rather than silently assigning a
/// derivative to an arbitrary eigenvector.
pub fn ground_state_energy_gradient(
    operator: &QuantumOperator,
    parameters: &ParameterValues,
    options: EigshOptions,
    workspace: &mut EigshWorkspace,
) -> Result<GroundStateEnergyGradient> {
    validate_ground_state_arguments(operator, parameters, &options)?;
    let bound = BoundQuantumOperator {
        family: operator,
        coefficients: parameters.values(),
    };
    let eigensystem = eigsh_with_workspace(&bound, options, workspace)?;
    let energy = eigensystem.eigenvalues[0];
    let gap = eigensystem.eigenvalues[1] - energy;
    let state = eigensystem.eigenvectors[0].clone();
    let mut contribution = vec![Complex64::new(0.0, 0.0); state.len()];
    let mut gradient = Vec::with_capacity(operator.components().len());
    for component in operator.components() {
        component.operator().apply(&state, &mut contribution)?;
        let derivative = state
            .iter()
            .zip(&contribution)
            .map(|(left, right)| left.conj() * right)
            .sum::<Complex64>();
        gradient.push(Complex64::new(derivative.re, 0.0));
    }

    let residual = eigensystem.residuals.iter().copied().fold(0.0, f64::max);
    let scale = energy.abs().max(1.0);
    let unresolved = (10.0 * residual).max(64.0 * f64::EPSILON * scale);
    let status = if gap <= unresolved {
        GradientStatus::NonDifferentiable
    } else if gap <= scale * 1.0e-8 {
        GradientStatus::IllConditioned
    } else {
        GradientStatus::Reliable
    };
    Ok(GroundStateEnergyGradient {
        energy,
        state,
        gradient: ParameterGradient::new(Arc::clone(parameters.schema()), gradient),
        diagnostics: GradientDiagnostics {
            status,
            primal_residual: Some(residual),
            backward_residual: None,
            spectral_gap: Some(gap),
            primal_applications: 0,
            backward_applications: operator.components().len(),
            recomputations: 0,
        },
        eigensolver_iterations: eigensystem.iterations,
    })
}

fn validate_ground_state_arguments(
    operator: &QuantumOperator,
    parameters: &ParameterValues,
    options: &EigshOptions,
) -> Result<()> {
    validate_operator_schema(operator, parameters)?;
    let (rows, columns) = operator.shape();
    if rows != columns {
        return Err(QmbedError::DimensionMismatch(
            "ground-state differentiation requires a square operator family".into(),
        ));
    }
    if options.eigenpairs < 2 {
        return Err(QmbedError::InvalidOptions(
            "ground-state differentiation requires at least two eigenpairs for gap diagnostics"
                .into(),
        ));
    }
    if !matches!(options.target, SpectrumTarget::SmallestAlgebraic) {
        return Err(QmbedError::InvalidOptions(
            "ground-state differentiation requires the SmallestAlgebraic target".into(),
        ));
    }
    if parameters
        .schema()
        .domains()
        .iter()
        .any(|domain| *domain != ParameterDomain::Real)
    {
        return Err(QmbedError::InvalidOptions(
            "ground-state energy gradients currently require real parameters".into(),
        ));
    }
    if let Some(component) = operator
        .components()
        .iter()
        .find(|component| !component.operator().is_hermitian(1.0e-12))
    {
        return Err(QmbedError::InvalidOptions(format!(
            "operator component {:?} is not Hermitian",
            component.name()
        )));
    }
    Ok(())
}

impl GradientDiagnostics {
    fn exact_operator(primal_applications: usize, backward_applications: usize) -> Self {
        Self {
            status: GradientStatus::Reliable,
            primal_residual: Some(0.0),
            backward_residual: Some(0.0),
            spectral_gap: None,
            primal_applications,
            backward_applications,
            recomputations: backward_applications / 2,
        }
    }
}

/// Primal value and forward tangent from parameterized operator application.
#[derive(Clone, Debug)]
pub struct ApplyJvp<B> {
    /// `A(θ)x`.
    pub value: B,
    /// `A(θ)dx + dA(θ)[dθ]x`.
    pub tangent: B,
    /// Work and reliability evidence.
    pub diagnostics: GradientDiagnostics,
}

/// State and parameter cotangents returned by an operator pullback.
#[derive(Clone, Debug)]
pub struct ApplyCotangents<B> {
    /// Cotangent with respect to the ordered parameters.
    pub parameters: ParameterGradient,
    /// Cotangent with respect to the input state.
    pub state: B,
    /// Work and reliability evidence.
    pub diagnostics: GradientDiagnostics,
}

/// Evaluate a parameterized operator and its JVP without materializing `A(θ)`.
pub fn apply_jvp<R>(
    runtime: &R,
    operator: &QuantumOperator,
    parameters: &ParameterValues,
    parameter_direction: &ParameterDirection,
    input: &R::Buffer,
    input_direction: &R::Buffer,
) -> Result<ApplyJvp<R::Buffer>>
where
    R: Runtime,
    Operator: RuntimeLinearOperator<R>,
{
    validate_operator_arguments::<R>(
        operator,
        parameters,
        Some(parameter_direction),
        input,
        Some(input_direction),
    )?;

    let (rows, _) = operator.shape();
    let mut value = runtime.zeros(rows)?;
    let mut tangent = runtime.zeros(rows)?;
    let mut input_contribution = runtime.zeros(rows)?;
    let mut direction_contribution = runtime.zeros(rows)?;

    for ((component, coefficient), parameter_tangent) in operator
        .components()
        .iter()
        .zip(parameters.values())
        .zip(parameter_direction.values())
    {
        runtime.fill(&mut input_contribution, Complex64::new(0.0, 0.0))?;
        component
            .operator()
            .apply_on(runtime, input, &mut input_contribution)?;
        runtime.axpy(*coefficient, &input_contribution, &mut value)?;
        runtime.axpy(*parameter_tangent, &input_contribution, &mut tangent)?;

        runtime.fill(&mut direction_contribution, Complex64::new(0.0, 0.0))?;
        component
            .operator()
            .apply_on(runtime, input_direction, &mut direction_contribution)?;
        runtime.axpy(*coefficient, &direction_contribution, &mut tangent)?;
    }

    Ok(ApplyJvp {
        value,
        tangent,
        diagnostics: GradientDiagnostics::exact_operator(2 * operator.components().len(), 0),
    })
}

/// Evaluate a parameterized operator and prepare a one-shot reverse pullback.
pub fn apply_vjp<'a, R>(
    runtime: &'a R,
    operator: &'a QuantumOperator,
    parameters: &'a ParameterValues,
    input: &'a R::Buffer,
) -> Result<(R::Buffer, ApplyPullback<'a, R>)>
where
    R: Runtime,
    Operator: RuntimeLinearOperator<R> + RuntimeAdjointLinearOperator<R>,
{
    validate_operator_arguments::<R>(operator, parameters, None, input, None)?;
    let (rows, _) = operator.shape();
    let mut value = runtime.zeros(rows)?;
    let mut contribution = runtime.zeros(rows)?;
    for (component, coefficient) in operator.components().iter().zip(parameters.values()) {
        runtime.fill(&mut contribution, Complex64::new(0.0, 0.0))?;
        component
            .operator()
            .apply_on(runtime, input, &mut contribution)?;
        runtime.axpy(*coefficient, &contribution, &mut value)?;
    }
    let pullback = ApplyPullback {
        runtime,
        operator,
        parameters,
        input,
    };
    Ok((value, pullback))
}

/// One-shot pullback for [`apply_vjp`].
pub struct ApplyPullback<'a, R>
where
    R: Runtime,
{
    runtime: &'a R,
    operator: &'a QuantumOperator,
    parameters: &'a ParameterValues,
    input: &'a R::Buffer,
}

impl<R> ApplyPullback<'_, R>
where
    R: Runtime,
    Operator: RuntimeLinearOperator<R> + RuntimeAdjointLinearOperator<R>,
{
    /// Pull an output cotangent back to state and parameter cotangents.
    pub fn backward(self, output_cotangent: &R::Buffer) -> Result<ApplyCotangents<R::Buffer>> {
        let (rows, columns) = self.operator.shape();
        if output_cotangent.len() != rows {
            return Err(QmbedError::DimensionMismatch(format!(
                "operator pullback requires output cotangent length {rows}, got {}",
                output_cotangent.len()
            )));
        }

        let mut state = self.runtime.zeros(columns)?;
        let mut state_contribution = self.runtime.zeros(columns)?;
        let mut parameter_contribution = self.runtime.zeros(rows)?;
        let mut gradient = Vec::with_capacity(self.operator.components().len());

        for ((component, coefficient), domain) in self
            .operator
            .components()
            .iter()
            .zip(self.parameters.values())
            .zip(self.parameters.schema.domains())
        {
            self.runtime
                .fill(&mut state_contribution, Complex64::new(0.0, 0.0))?;
            component.operator().apply_adjoint_on(
                self.runtime,
                output_cotangent,
                &mut state_contribution,
            )?;
            self.runtime
                .axpy(coefficient.conj(), &state_contribution, &mut state)?;

            self.runtime
                .fill(&mut parameter_contribution, Complex64::new(0.0, 0.0))?;
            component
                .operator()
                .apply_on(self.runtime, self.input, &mut parameter_contribution)?;
            let cotangent = self
                .runtime
                .dotc(&parameter_contribution, output_cotangent)?;
            gradient.push(match domain {
                ParameterDomain::Real => Complex64::new(cotangent.re, 0.0),
                ParameterDomain::Complex => cotangent,
            });
        }

        let applications = 2 * self.operator.components().len();
        Ok(ApplyCotangents {
            parameters: ParameterGradient::new(Arc::clone(&self.parameters.schema), gradient),
            state,
            diagnostics: GradientDiagnostics::exact_operator(
                self.operator.components().len(),
                applications,
            ),
        })
    }
}

fn validate_parameter_vector(
    schema: &ParameterSchema,
    values: &[Complex64],
    label: &str,
) -> Result<()> {
    if values.len() != schema.len() {
        return Err(QmbedError::DimensionMismatch(format!(
            "{label} has length {}, expected {}",
            values.len(),
            schema.len()
        )));
    }
    for ((name, domain), value) in schema.names().iter().zip(schema.domains()).zip(values) {
        if !value.re.is_finite() || !value.im.is_finite() {
            return Err(QmbedError::InvalidOptions(format!(
                "{label} for {name:?} must be finite"
            )));
        }
        if matches!(domain, ParameterDomain::Real) && value.im != 0.0 {
            return Err(QmbedError::InvalidOptions(format!(
                "{label} for real parameter {name:?} must have zero imaginary part"
            )));
        }
    }
    Ok(())
}

fn validate_operator_arguments<R>(
    operator: &QuantumOperator,
    parameters: &ParameterValues,
    direction: Option<&ParameterDirection>,
    input: &R::Buffer,
    input_direction: Option<&R::Buffer>,
) -> Result<()>
where
    R: Runtime,
{
    validate_operator_schema(operator, parameters)?;
    if let Some(direction) = direction {
        if direction.schema.as_ref() != parameters.schema.as_ref() {
            return Err(QmbedError::InvalidOptions(
                "parameter direction uses a different schema".into(),
            ));
        }
    }
    let (_, columns) = operator.shape();
    if input.len() != columns {
        return Err(QmbedError::DimensionMismatch(format!(
            "parameterized operator requires input length {columns}, got {}",
            input.len()
        )));
    }
    if let Some(input_direction) = input_direction {
        if input_direction.len() != columns {
            return Err(QmbedError::DimensionMismatch(format!(
                "operator JVP requires input direction length {columns}, got {}",
                input_direction.len()
            )));
        }
    }
    Ok(())
}

fn validate_operator_schema(
    operator: &QuantumOperator,
    parameters: &ParameterValues,
) -> Result<()> {
    let operator_names: Vec<_> = operator.component_names().collect();
    if operator_names.len() != parameters.schema.len()
        || operator_names
            .iter()
            .zip(parameters.schema.names())
            .any(|(operator_name, parameter_name)| operator_name != parameter_name)
    {
        return Err(QmbedError::InvalidOptions(
            "parameter schema does not match QuantumOperator component order".into(),
        ));
    }
    Ok(())
}

/// `chainrules-core 0.2` adapters for the Rust-native operator rules.
#[cfg(feature = "chainrules")]
pub mod chainrules {
    use std::sync::Arc;

    use chainrules_core::{Differentiable, JvpRule, Pullback, VjpRule};

    use super::{
        ApplyCotangents, ApplyPullback, GradientStatus, ParameterDirection, ParameterGradient,
        ParameterValues, apply_jvp, apply_vjp, ground_state_energy_gradient,
    };
    use crate::operator::{Operator, QuantumOperator};
    use crate::runtime::{Runtime, RuntimeAdjointLinearOperator, RuntimeLinearOperator};
    use crate::solve::{EigshOptions, EigshWorkspace};
    use crate::{QmbedError, Result};

    impl Differentiable for ParameterValues {
        type Tangent = ParameterDirection;
        type Cotangent = ParameterGradient;
    }

    /// Runtime-owned state returned through `chainrules-core`.
    #[derive(Clone, Debug)]
    pub struct State<B>(pub B);

    impl<B> State<B> {
        /// Consume the wrapper.
        pub fn into_inner(self) -> B {
            self.0
        }
    }

    impl<B> Differentiable for State<B> {
        type Tangent = B;
        type Cotangent = B;
    }

    /// Borrowed primal arguments for parameterized operator application.
    pub struct ApplyArguments<'a, R>
    where
        R: Runtime,
    {
        /// Ordered operator parameters.
        pub parameters: &'a ParameterValues,
        /// Runtime-owned input state.
        pub state: &'a R::Buffer,
    }

    /// Borrowed forward perturbations corresponding to [`ApplyArguments`].
    pub struct ApplyArgumentTangent<'a, R>
    where
        R: Runtime,
    {
        /// Parameter direction.
        pub parameters: &'a ParameterDirection,
        /// Input-state direction.
        pub state: &'a R::Buffer,
    }

    impl<'a, R> Differentiable for ApplyArguments<'a, R>
    where
        R: Runtime,
    {
        type Tangent = ApplyArgumentTangent<'a, R>;
        type Cotangent = ApplyCotangents<R::Buffer>;
    }

    /// Explicit rule object connecting QMBED native application to ChainRules.
    pub struct ApplyRule<'a, R>
    where
        R: Runtime,
    {
        /// Execution runtime.
        pub runtime: &'a R,
        /// Parameterized operator family.
        pub operator: &'a QuantumOperator,
    }

    impl<'rule, 'args, R> JvpRule<ApplyArguments<'args, R>> for ApplyRule<'rule, R>
    where
        R: Runtime,
        Operator: RuntimeLinearOperator<R>,
    {
        type Output = State<R::Buffer>;
        type Error = QmbedError;

        fn jvp(
            &self,
            args: &ApplyArguments<'args, R>,
            tangent: &ApplyArgumentTangent<'args, R>,
        ) -> Result<(Self::Output, <Self::Output as Differentiable>::Tangent)> {
            let result = apply_jvp(
                self.runtime,
                self.operator,
                args.parameters,
                tangent.parameters,
                args.state,
                tangent.state,
            )?;
            Ok((State(result.value), result.tangent))
        }
    }

    /// ChainRules wrapper around QMBED's native one-shot pullback.
    pub struct RulePullback<'a, R>
    where
        R: Runtime,
    {
        inner: ApplyPullback<'a, R>,
    }

    impl<R> Pullback<R::Buffer, ApplyCotangents<R::Buffer>> for RulePullback<'_, R>
    where
        R: Runtime,
        Operator: RuntimeLinearOperator<R> + RuntimeAdjointLinearOperator<R>,
    {
        type Error = QmbedError;

        fn apply(self, cotangent: R::Buffer) -> Result<ApplyCotangents<R::Buffer>> {
            self.inner.backward(&cotangent)
        }
    }

    impl<'rule, 'args, R> VjpRule<ApplyArguments<'args, R>> for ApplyRule<'rule, R>
    where
        R: Runtime,
        Operator: RuntimeLinearOperator<R> + RuntimeAdjointLinearOperator<R>,
    {
        type Output = State<R::Buffer>;
        type Error = QmbedError;
        type Pullback<'a>
            = RulePullback<'a, R>
        where
            Self: 'a,
            ApplyArguments<'args, R>: 'a;

        fn vjp<'a>(
            &'a self,
            args: &'a ApplyArguments<'args, R>,
        ) -> Result<(Self::Output, Self::Pullback<'a>)> {
            let (value, pullback) =
                apply_vjp(self.runtime, self.operator, args.parameters, args.state)?;
            Ok((State(value), RulePullback { inner: pullback }))
        }
    }

    /// ChainRules protocol object for an isolated ground-state energy.
    pub struct GroundStateEnergyRule<'a> {
        /// Real-parameter Hermitian operator family.
        pub operator: &'a QuantumOperator,
        /// Solver and convergence controls, including two or more eigenpairs.
        pub options: EigshOptions,
    }

    /// One-shot scalar-energy pullback owning the computed gradient.
    pub struct GroundStateEnergyPullback {
        gradient: ParameterGradient,
    }

    impl Pullback<f64, ParameterGradient> for GroundStateEnergyPullback {
        type Error = QmbedError;

        fn apply(self, cotangent: f64) -> Result<ParameterGradient> {
            if !cotangent.is_finite() {
                return Err(QmbedError::InvalidOptions(
                    "ground-state energy cotangent must be finite".into(),
                ));
            }
            Ok(ParameterGradient::new(
                Arc::clone(self.gradient.schema()),
                self.gradient
                    .values()
                    .iter()
                    .map(|value| cotangent * *value)
                    .collect(),
            ))
        }
    }

    impl JvpRule<ParameterValues> for GroundStateEnergyRule<'_> {
        type Output = f64;
        type Error = QmbedError;

        fn jvp(
            &self,
            parameters: &ParameterValues,
            tangent: &ParameterDirection,
        ) -> Result<(f64, f64)> {
            if parameters.schema().as_ref() != tangent.schema().as_ref() {
                return Err(QmbedError::InvalidOptions(
                    "ground-state tangent uses a different parameter schema".into(),
                ));
            }
            let result = ground_state_energy_gradient(
                self.operator,
                parameters,
                self.options.clone(),
                &mut EigshWorkspace::new(),
            )?;
            if result.diagnostics.status != GradientStatus::Reliable {
                return Err(QmbedError::InvalidOptions(format!(
                    "ground-state derivative is not reliable: {:?}",
                    result.diagnostics
                )));
            }
            let directional = result
                .gradient
                .values()
                .iter()
                .zip(tangent.values())
                .map(|(gradient, direction)| (direction.conj() * gradient).re)
                .sum();
            Ok((result.energy, directional))
        }
    }

    impl VjpRule<ParameterValues> for GroundStateEnergyRule<'_> {
        type Output = f64;
        type Error = QmbedError;
        type Pullback<'a>
            = GroundStateEnergyPullback
        where
            Self: 'a,
            ParameterValues: 'a;

        fn vjp<'a>(&'a self, parameters: &'a ParameterValues) -> Result<(f64, Self::Pullback<'a>)> {
            let result = ground_state_energy_gradient(
                self.operator,
                parameters,
                self.options.clone(),
                &mut EigshWorkspace::new(),
            )?;
            if result.diagnostics.status != GradientStatus::Reliable {
                return Err(QmbedError::InvalidOptions(format!(
                    "ground-state derivative is not reliable: {:?}",
                    result.diagnostics
                )));
            }
            Ok((
                result.energy,
                GroundStateEnergyPullback {
                    gradient: result.gradient,
                },
            ))
        }
    }
}
