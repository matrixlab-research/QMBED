use std::sync::Arc;

use nalgebra::{DMatrix, DVector, SymmetricEigen};
use num_complex::Complex64;

use crate::backend;
use crate::operator::{
    ExpOp, LinearOperator, MatrixFormat, ShiftedLinearSolver, TimeDependentOperator,
    materialize_dense,
};
use crate::runtime::CpuRuntime;
use crate::{QmbedError, Result};

const EXPM_TAYLOR_THETA: &[(usize, f64)] = &[
    (1, 2.29e-16),
    (2, 2.58e-8),
    (3, 1.39e-5),
    (4, 3.40e-4),
    (5, 2.40e-3),
    (6, 9.07e-3),
    (7, 2.38e-2),
    (8, 5.00e-2),
    (9, 8.96e-2),
    (10, 1.44e-1),
    (11, 2.14e-1),
    (12, 3.00e-1),
    (13, 4.00e-1),
    (14, 5.14e-1),
    (15, 6.41e-1),
    (16, 7.81e-1),
    (17, 9.31e-1),
    (18, 1.09),
    (19, 1.26),
    (20, 1.44),
    (21, 1.62),
    (22, 1.82),
    (23, 2.01),
    (24, 2.22),
    (25, 2.43),
    (26, 2.64),
    (27, 2.86),
    (28, 3.08),
    (29, 3.31),
    (30, 3.54),
    (35, 4.7),
    (40, 6.0),
    (45, 7.2),
    (50, 8.5),
    (55, 9.9),
];

// Daniel--Gragg--Kaufman--Stewart reorthogonalization criterion. A second
// modified Gram--Schmidt pass is useful only when the first pass removes a
// substantial fraction of the vector norm. Keeping the criterion here makes
// clustered-spectrum robustness a property of the generic Lanczos backend
// without charging every iteration for two unconditional O(m n) passes.
const DGKS_REORTHOGONALIZATION_THRESHOLD: f64 = std::f64::consts::FRAC_1_SQRT_2;

fn shifted_trace_and_one_norm(
    operator: &(impl LinearOperator + ?Sized),
) -> Result<(Complex64, f64)> {
    let (rows, columns) = operator.shape();
    if rows != columns {
        return Err(QmbedError::DimensionMismatch(
            "exponential action requires a square operator".into(),
        ));
    }
    if rows == 0 {
        return Ok((Complex64::new(0.0, 0.0), 0.0));
    }
    let entries = match operator.stored_triplets()? {
        Some(entries) => entries,
        None => {
            let dense = materialize_dense(operator)?;
            dense
                .into_iter()
                .enumerate()
                .filter_map(|(offset, value)| {
                    (value.norm() > 0.0).then_some((offset / columns, offset % columns, value))
                })
                .collect()
        }
    };
    let mut diagonal = vec![Complex64::new(0.0, 0.0); rows];
    let mut column_sums = vec![0.0; columns];
    for (row, column, value) in entries {
        if !value.re.is_finite() || !value.im.is_finite() {
            return Err(QmbedError::InvalidOptions(
                "exponential generator entries must be finite".into(),
            ));
        }
        if row == column {
            diagonal[row] += value;
        } else {
            column_sums[column] += value.norm();
        }
    }
    let trace = diagonal.iter().copied().sum::<Complex64>();
    let shift = trace / rows as f64;
    for (column, value) in diagonal.into_iter().enumerate() {
        column_sums[column] += (value - shift).norm();
    }
    Ok((shift, column_sums.into_iter().fold(0.0_f64, f64::max)))
}

fn vector_infinity_norm(vector: &[Complex64]) -> f64 {
    vector.iter().map(|value| value.norm()).fold(0.0, f64::max)
}

/// Prepared Al-Mohy--Higham exponential-action plan.
///
/// Construction computes the trace shift, exact stored one-norm, Taylor
/// degree, and scaling count once. Repeated vector and batch applications then
/// perform only matrix-vector products and vector updates.
#[derive(Clone, Debug)]
pub struct ExpmActionPlan {
    dimension: usize,
    coefficient: Complex64,
    shift: Complex64,
    degree: usize,
    scaling: usize,
    tolerance: f64,
}

impl ExpmActionPlan {
    pub fn new(
        operator: &(impl LinearOperator + ?Sized),
        coefficient: Complex64,
        max_degree: usize,
        tolerance: f64,
        max_substeps: usize,
    ) -> Result<Self> {
        let (rows, columns) = operator.shape();
        if rows != columns {
            return Err(QmbedError::DimensionMismatch(
                "exponential action requires a square operator".into(),
            ));
        }
        if !coefficient.re.is_finite()
            || !coefficient.im.is_finite()
            || max_degree == 0
            || !tolerance.is_finite()
            || tolerance <= 0.0
            || max_substeps == 0
        {
            return Err(QmbedError::InvalidOptions(
                "invalid exponential coefficient or numerical controls".into(),
            ));
        }
        let (shift, shifted_one_norm) = shifted_trace_and_one_norm(operator)?;
        let scaled_norm = coefficient.norm() * shifted_one_norm;
        let (degree, scaling) = if scaled_norm == 0.0 {
            (0, 1)
        } else {
            EXPM_TAYLOR_THETA
                .iter()
                .copied()
                .filter(|(degree, _)| *degree <= max_degree)
                .map(|(degree, theta)| {
                    let scaling = (scaled_norm / theta).ceil().max(1.0) as usize;
                    (degree, scaling)
                })
                .min_by_key(|(degree, scaling)| degree.saturating_mul(*scaling))
                .ok_or_else(|| {
                    QmbedError::InvalidOptions(
                        "Taylor degree is below the minimum supported degree".into(),
                    )
                })?
        };
        if scaling > max_substeps {
            return Err(QmbedError::NonConvergence {
                iterations: max_substeps,
                residual: scaled_norm,
            });
        }
        Ok(Self {
            dimension: rows,
            coefficient,
            shift,
            degree,
            scaling,
            tolerance,
        })
    }

    pub const fn coefficient(&self) -> Complex64 {
        self.coefficient
    }

    pub const fn degree(&self) -> usize {
        self.degree
    }

    pub const fn scaling(&self) -> usize {
        self.scaling
    }

    pub fn apply(
        &self,
        operator: &(impl LinearOperator + ?Sized),
        initial: &[Complex64],
    ) -> Result<Vec<Complex64>> {
        if operator.shape() != (self.dimension, self.dimension) || initial.len() != self.dimension {
            return Err(QmbedError::DimensionMismatch(
                "exponential plan, operator, and state dimensions do not match".into(),
            ));
        }
        if self.coefficient.norm() <= f64::EPSILON || vector_infinity_norm(initial) == 0.0 {
            return Ok(initial.to_vec());
        }
        let factor = self.coefficient / self.scaling as f64;
        let eta = (factor * self.shift).exp();
        if !eta.re.is_finite() || !eta.im.is_finite() {
            return Err(QmbedError::UnsupportedBackend(
                "exponential action overflowed its scalar trace shift".into(),
            ));
        }
        let mut state = initial.to_vec();
        let mut applied = vec![Complex64::new(0.0, 0.0); self.dimension];
        for _ in 0..self.scaling {
            let mut term = state.clone();
            let mut sum = state.clone();
            let mut previous_norm = vector_infinity_norm(&term);
            for order in 1..=self.degree {
                operator.apply(&term, &mut applied)?;
                let scale = factor / order as f64;
                for index in 0..self.dimension {
                    applied[index] = scale * (applied[index] - self.shift * term[index]);
                }
                std::mem::swap(&mut term, &mut applied);
                for (total, value) in sum.iter_mut().zip(&term) {
                    *total += *value;
                }
                let term_norm = vector_infinity_norm(&term);
                if previous_norm + term_norm
                    <= self.tolerance * vector_infinity_norm(&sum).max(f64::MIN_POSITIVE)
                {
                    break;
                }
                previous_norm = term_norm;
            }
            for (value, total) in state.iter_mut().zip(sum) {
                *value = eta * total;
            }
        }
        Ok(state)
    }
}

/// Reusable exponential-action plan for vectors and batches.
#[derive(Clone, Debug)]
pub struct ExpmMultiplyParallel {
    inner: ExpOp,
}

impl ExpmMultiplyParallel {
    pub fn new(
        operator: Arc<dyn LinearOperator>,
        coefficient: Complex64,
        krylov_dimension: usize,
        tolerance: f64,
        max_substeps: usize,
    ) -> Result<Self> {
        Ok(Self {
            inner: ExpOp::new(
                operator,
                coefficient,
                krylov_dimension,
                tolerance,
                max_substeps,
            )?,
        })
    }

    pub const fn coefficient(&self) -> Complex64 {
        self.inner.exponent()
    }

    pub fn set_coefficient(&mut self, coefficient: Complex64) -> Result<()> {
        self.inner.set_exponent(coefficient)
    }

    pub fn apply_in_place(&self, state: &mut [Complex64]) -> Result<()> {
        let input = state.to_vec();
        self.inner.apply(&input, state)
    }

    pub fn apply_batch(&self, states: &[Vec<Complex64>]) -> Result<Vec<Vec<Complex64>>> {
        self.apply_batch_with_runtime(
            &CpuRuntime::from_profile(crate::runtime::ExecutionProfile::serial())?,
            states,
        )
    }

    pub fn apply_batch_with_runtime(
        &self,
        runtime: &CpuRuntime,
        states: &[Vec<Complex64>],
    ) -> Result<Vec<Vec<Complex64>>> {
        let dimension = self.inner.shape().1;
        runtime.map_ordered(states, |state| {
            if state.len() != dimension {
                return Err(QmbedError::DimensionMismatch(
                    "exponential batch column has the wrong length".into(),
                ));
            }
            let mut output = vec![Complex64::new(0.0, 0.0); dimension];
            self.inner.apply(state, &mut output)?;
            Ok(output)
        })
    }
}

impl LinearOperator for ExpmMultiplyParallel {
    fn shape(&self) -> (usize, usize) {
        self.inner.shape()
    }

    fn format(&self) -> MatrixFormat {
        MatrixFormat::MatrixFree
    }

    fn apply(&self, input: &[Complex64], output: &mut [Complex64]) -> Result<()> {
        self.inner.apply(input, output)
    }
}

/// Spectral region requested from a selected Hermitian eigensolver.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub enum SpectrumTarget {
    /// Algebraically smallest eigenvalues.
    SmallestAlgebraic,
    /// Algebraically largest eigenvalues.
    LargestAlgebraic,
    /// Eigenvalues with smallest absolute value.
    SmallestMagnitude,
    /// Eigenvalues with largest absolute value.
    LargestMagnitude,
    /// A balanced selection from both algebraic ends.
    BothEnds,
    /// Eigenvalues nearest the supplied real shift.
    Shift(f64),
}

/// Controls target selection, search-space size, and convergence for [`eigsh`].
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct EigshOptions {
    /// Number of requested eigenpairs; must be smaller than the dimension.
    pub eigenpairs: usize,
    /// Region of the real Hermitian spectrum to target.
    pub target: SpectrumTarget,
    /// Optional Lanczos or restart window dimension.
    pub krylov_dimension: Option<usize>,
    /// Required residual norm for convergence.
    pub tolerance: f64,
    /// Maximum Lanczos or restart iteration budget.
    pub max_iterations: usize,
    /// Deterministic seed for the initial vector.
    pub seed: u64,
}

const GUARANTEED_DENSE_EIGSH_CROSSOVER: usize = 128;
const AUTOMATIC_DENSE_EIGSH_CROSSOVER: usize = 256;
const FULL_KRYLOV_DENSE_FALLBACK: usize = 2_048;
const TRIDIAGONAL_LANCZOS_WINDOW: usize = 96;

fn use_dense_eigsh(dimension: usize, options: &EigshOptions) -> bool {
    dimension <= GUARANTEED_DENSE_EIGSH_CROSSOVER
        || (dimension <= AUTOMATIC_DENSE_EIGSH_CROSSOVER && options.krylov_dimension.is_none())
}

impl EigshOptions {
    /// Construct solver controls for an arbitrary spectral target.
    pub fn new(eigenpairs: usize, target: SpectrumTarget) -> Self {
        Self {
            eigenpairs,
            target,
            krylov_dimension: None,
            tolerance: 1.0e-10,
            max_iterations: 1_000,
            seed: 0,
        }
    }

    /// Construct default controls for the algebraically lowest eigenpairs.
    pub fn smallest_algebraic(eigenpairs: usize) -> Self {
        Self::new(eigenpairs, SpectrumTarget::SmallestAlgebraic)
    }

    /// Construct default controls for eigenpairs nearest a real shift.
    pub fn near_shift(eigenpairs: usize, shift: f64) -> Self {
        Self::new(eigenpairs, SpectrumTarget::Shift(shift))
    }

    /// Set the optional Lanczos or restart window dimension.
    #[must_use]
    pub fn with_krylov_dimension(mut self, dimension: usize) -> Self {
        self.krylov_dimension = Some(dimension);
        self
    }

    /// Set the required residual norm.
    #[must_use]
    pub fn with_tolerance(mut self, tolerance: f64) -> Self {
        self.tolerance = tolerance;
        self
    }

    /// Set the maximum Lanczos or restart iteration budget.
    #[must_use]
    pub fn with_max_iterations(mut self, max_iterations: usize) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    /// Set the deterministic initial-vector seed.
    #[must_use]
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    fn validate(&self, dimension: usize) -> Result<()> {
        if self.eigenpairs == 0 || self.eigenpairs >= dimension {
            return Err(QmbedError::InvalidOptions(
                "eigenpairs must be positive and smaller than the operator dimension".into(),
            ));
        }
        if !self.tolerance.is_finite() || self.tolerance <= 0.0 || self.max_iterations == 0 {
            return Err(QmbedError::InvalidOptions(
                "tolerance and max_iterations must be positive".into(),
            ));
        }
        if self
            .krylov_dimension
            .is_some_and(|size| size <= self.eigenpairs || size > dimension)
        {
            return Err(QmbedError::InvalidOptions(
                "krylov_dimension must exceed eigenpairs and not exceed dimension".into(),
            ));
        }
        if matches!(self.target, SpectrumTarget::Shift(value) if !value.is_finite()) {
            return Err(QmbedError::InvalidOptions("shift must be finite".into()));
        }
        Ok(())
    }
}

/// Values, vectors, residual evidence, and work counters from an eigensolve.
#[derive(Clone, Debug)]
pub struct Eigensystem {
    /// Ordered real eigenvalues or Ritz values.
    pub eigenvalues: Vec<f64>,
    /// Column vectors corresponding to `eigenvalues`.
    pub eigenvectors: Vec<Vec<Complex64>>,
    /// Norm of `A v - λ v` for each returned pair.
    pub residuals: Vec<f64>,
    /// Algorithm iteration count.
    pub iterations: usize,
    /// Total modified Gram-Schmidt passes.
    pub reorthogonalization_passes: usize,
    /// Selective DGKS second passes triggered by loss of norm.
    pub conditional_second_passes: usize,
    /// Whether every requested residual met the tolerance.
    pub converged: bool,
}

/// Reusable state for a sequence of related Hermitian eigenproblems.
///
/// The workspace keeps the complete converged invariant subspace from the
/// previous solve. [`eigsh_with_workspace`] preserves it as a thick initial
/// subspace for restarted windows and combines all of its vectors into a
/// balanced warm start when the cheaper tridiagonal Lanczos path applies.
#[derive(Clone, Debug, Default)]
pub struct EigshWorkspace {
    dimension: usize,
    initial_subspace: Vec<Vec<Complex64>>,
}

impl EigshWorkspace {
    /// Create an empty workspace that accepts the first solved dimension.
    pub const fn new() -> Self {
        Self {
            dimension: 0,
            initial_subspace: Vec::new(),
        }
    }

    /// Forget the stored dimension and converged invariant subspace.
    pub fn clear(&mut self) {
        self.dimension = 0;
        self.initial_subspace.clear();
    }

    /// Return the operator dimension associated with the stored subspace.
    pub const fn dimension(&self) -> usize {
        self.dimension
    }

    /// Borrow the complete converged subspace from the previous solve.
    pub fn initial_subspace(&self) -> &[Vec<Complex64>] {
        &self.initial_subspace
    }

    /// Install a user-provided warm-start subspace after validating dimensions.
    pub fn set_initial_subspace(
        &mut self,
        dimension: usize,
        vectors: impl IntoIterator<Item = Vec<Complex64>>,
    ) -> Result<()> {
        let vectors: Vec<_> = vectors.into_iter().collect();
        for vector in &vectors {
            validate_eigsh_initial(vector, dimension)?;
        }
        self.dimension = dimension;
        self.initial_subspace = vectors;
        Ok(())
    }

    fn update(&mut self, dimension: usize, eigensystem: &Eigensystem) {
        self.dimension = dimension;
        self.initial_subspace.clone_from(&eigensystem.eigenvectors);
    }
}

/// Controls whether [`eigh_with_options`] retains the complete eigenvector matrix.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub struct EighOptions {
    /// Return eigenvectors when `true`; values and residuals are always returned.
    pub return_eigenvectors: bool,
}

impl Default for EighOptions {
    fn default() -> Self {
        Self {
            return_eigenvectors: true,
        }
    }
}

impl EighOptions {
    /// Construct controls which retain or discard the full eigenvector matrix.
    pub const fn new(return_eigenvectors: bool) -> Self {
        Self {
            return_eigenvectors,
        }
    }

    /// Compute eigenvalues and residual evidence without returning vectors.
    pub const fn values_only() -> Self {
        Self::new(false)
    }
}

pub(crate) fn hermitian_eigenpairs_all(
    operator: &(impl LinearOperator + ?Sized),
) -> Result<(Vec<f64>, Vec<Vec<Complex64>>)> {
    let shape = operator.shape();
    if shape.0 != shape.1 {
        return Err(QmbedError::DimensionMismatch(
            "a square operator is required".into(),
        ));
    }
    let dense = materialize_dense(operator)?;
    let dimension = shape.0;
    if dimension == 0 {
        return Ok((Vec::new(), Vec::new()));
    }
    for row in 0..dimension {
        for column in 0..dimension {
            if (dense[row * dimension + column] - dense[column * dimension + row].conj()).norm()
                > 1.0e-12
            {
                return Err(QmbedError::NonHermitian);
            }
        }
    }
    let eigensystem = backend::hermitian_eigenpairs(&dense, dimension)?;
    Ok((eigensystem.eigenvalues, eigensystem.eigenvectors))
}

/// Complete eigendecomposition of a finite Hermitian operator.
pub fn eigh<O>(operator: &O) -> Result<Eigensystem>
where
    O: LinearOperator + ?Sized,
{
    let (eigenvalues, eigenvectors) = hermitian_eigenpairs_all(operator)?;
    let residuals = eigenvalues
        .iter()
        .zip(&eigenvectors)
        .map(|(&value, vector)| residual_norm(operator, value, vector))
        .collect::<Result<Vec<_>>>()?;
    Ok(Eigensystem {
        eigenvalues,
        eigenvectors,
        residuals,
        iterations: usize::from(operator.shape().0 != 0),
        reorthogonalization_passes: 0,
        conditional_second_passes: 0,
        converged: true,
    })
}

/// Compute the complete Hermitian spectrum with optional vector elision.
///
/// The operator is materialized once as dense storage. This routine is for
/// finite systems where the cubic full-spectrum cost is intentional.
pub fn eigh_with_options<O>(operator: &O, options: EighOptions) -> Result<Eigensystem>
where
    O: LinearOperator + ?Sized,
{
    let mut result = eigh(operator)?;
    if !options.return_eigenvectors {
        result.eigenvectors.clear();
    }
    Ok(result)
}

fn residual_norm(
    operator: &(impl LinearOperator + ?Sized),
    eigenvalue: f64,
    vector: &[Complex64],
) -> Result<f64> {
    let mut applied = vec![Complex64::new(0.0, 0.0); vector.len()];
    operator.apply(vector, &mut applied)?;
    Ok(applied
        .iter()
        .zip(vector)
        .map(|(actual, component)| (*actual - eigenvalue * *component).norm_sqr())
        .sum::<f64>()
        .sqrt())
}

fn rayleigh_value_and_residual(
    operator: &(impl LinearOperator + ?Sized),
    vector: &[Complex64],
) -> Result<(f64, f64)> {
    let mut applied = vec![Complex64::new(0.0, 0.0); vector.len()];
    operator.apply(vector, &mut applied)?;
    let norm_squared = inner(vector, vector).re;
    if !norm_squared.is_finite() || norm_squared <= f64::EPSILON {
        return Err(QmbedError::NonConvergence {
            iterations: 0,
            residual: norm_squared.sqrt(),
        });
    }
    let eigenvalue = inner(vector, &applied).re / norm_squared;
    let residual = applied
        .iter()
        .zip(vector)
        .map(|(actual, component)| (*actual - eigenvalue * *component).norm_sqr())
        .sum::<f64>()
        .sqrt();
    Ok((eigenvalue, residual))
}

fn inner(left: &[Complex64], right: &[Complex64]) -> Complex64 {
    left.iter()
        .zip(right)
        .map(|(left_value, right_value)| left_value.conj() * *right_value)
        .sum()
}

fn vector_norm(vector: &[Complex64]) -> f64 {
    vector.iter().map(Complex64::norm_sqr).sum::<f64>().sqrt()
}

fn validate_eigsh_initial(vector: &[Complex64], dimension: usize) -> Result<()> {
    if vector.len() != dimension {
        return Err(QmbedError::DimensionMismatch(
            "eigsh initial vector does not match the operator".into(),
        ));
    }
    if vector
        .iter()
        .any(|value| !value.re.is_finite() || !value.im.is_finite())
    {
        return Err(QmbedError::InvalidOptions(
            "eigsh initial vector must contain only finite values".into(),
        ));
    }
    let norm = vector_norm(vector);
    if !norm.is_finite() || norm <= f64::EPSILON {
        return Err(QmbedError::InvalidOptions(
            "eigsh initial vector must have nonzero finite norm".into(),
        ));
    }
    Ok(())
}

fn normalize(vector: &mut [Complex64]) -> Result<()> {
    let norm = vector_norm(vector);
    if !norm.is_finite() || norm <= f64::EPSILON {
        return Err(QmbedError::NonConvergence {
            iterations: 0,
            residual: norm,
        });
    }
    for value in vector {
        *value /= norm;
    }
    Ok(())
}

fn balanced_subspace_start(initial_subspace: &[Vec<Complex64>]) -> Result<Vec<Complex64>> {
    let dimension = initial_subspace
        .first()
        .map(Vec::len)
        .ok_or_else(|| QmbedError::InvalidOptions("initial subspace must not be empty".into()))?;
    let mut combined = vec![Complex64::new(0.0, 0.0); dimension];
    let count = initial_subspace.len() as f64;
    for (index, vector) in initial_subspace.iter().enumerate() {
        let phase = std::f64::consts::TAU * index as f64 / count;
        let coefficient = Complex64::from_polar(count.sqrt().recip(), phase);
        for (value, component) in combined.iter_mut().zip(vector) {
            *value += coefficient * *component;
        }
    }
    if normalize(&mut combined).is_err() {
        combined.clone_from(&initial_subspace[0]);
        normalize(&mut combined)?;
    }
    Ok(combined)
}

fn deterministic_start(dimension: usize, seed: u64) -> Result<Vec<Complex64>> {
    let mut state = seed | 1;
    let mut vector = Vec::with_capacity(dimension);
    for _ in 0..dimension {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let mantissa = state >> 11;
        let value = mantissa as f64 / ((1_u64 << 53) as f64) - 0.5;
        vector.push(Complex64::new(value, 0.0));
    }
    normalize(&mut vector)?;
    Ok(vector)
}

fn shifted_apply<O>(
    operator: &O,
    shift: f64,
    input: &[Complex64],
    output: &mut [Complex64],
) -> Result<()>
where
    O: LinearOperator + ?Sized,
{
    operator.apply(input, output)?;
    for (value, input_value) in output.iter_mut().zip(input) {
        *value -= shift * *input_value;
    }
    Ok(())
}

fn gmres_shift_invert<O>(
    operator: &O,
    shift: f64,
    right_hand_side: &[Complex64],
    tolerance: f64,
    max_iterations: usize,
) -> Result<Vec<Complex64>>
where
    O: LinearOperator + ?Sized,
{
    let dimension = right_hand_side.len();
    let right_norm = vector_norm(right_hand_side);
    if right_norm <= f64::EPSILON {
        return Ok(vec![Complex64::new(0.0, 0.0); dimension]);
    }
    let restart = dimension.clamp(1, 256);
    let mut solution = vec![Complex64::new(0.0, 0.0); dimension];
    let mut residual = right_hand_side.to_vec();
    let mut applied = vec![Complex64::new(0.0, 0.0); dimension];
    let mut iterations = 0;

    while iterations < max_iterations {
        let beta = vector_norm(&residual);
        if beta <= tolerance * right_norm {
            return Ok(solution);
        }
        let mut first = residual.clone();
        for value in &mut first {
            *value /= beta;
        }
        let mut basis = vec![first];
        let cycle = restart.min(max_iterations - iterations);
        let mut hessenberg = vec![vec![Complex64::new(0.0, 0.0); cycle]; cycle + 1];
        let mut columns = 0;

        for column in 0..cycle {
            shifted_apply(operator, shift, &basis[column], &mut applied)?;
            for _ in 0..2 {
                for (row, vector) in basis.iter().enumerate() {
                    let overlap = inner(vector, &applied);
                    hessenberg[row][column] += overlap;
                    for (value, basis_value) in applied.iter_mut().zip(vector) {
                        *value -= overlap * *basis_value;
                    }
                }
            }
            let next_norm = vector_norm(&applied);
            hessenberg[column + 1][column] = Complex64::new(next_norm, 0.0);
            columns = column + 1;
            if next_norm <= 1.0e-14 {
                break;
            }
            let mut next = applied.clone();
            for value in &mut next {
                *value /= next_norm;
            }
            basis.push(next);
        }

        let mut normal = DMatrix::<Complex64>::zeros(columns, columns);
        let mut projected_rhs = DVector::<Complex64>::zeros(columns);
        for row in 0..columns {
            projected_rhs[row] = hessenberg[0][row].conj() * beta;
            for column in 0..columns {
                normal[(row, column)] = (0..=columns)
                    .map(|index| hessenberg[index][row].conj() * hessenberg[index][column])
                    .sum();
            }
        }
        let coefficients = normal
            .lu()
            .solve(&projected_rhs)
            .ok_or(QmbedError::NonConvergence {
                iterations,
                residual: beta,
            })?;
        for (coefficient, vector) in coefficients.iter().zip(&basis) {
            for (value, basis_value) in solution.iter_mut().zip(vector) {
                *value += *coefficient * *basis_value;
            }
        }
        shifted_apply(operator, shift, &solution, &mut applied)?;
        for ((value, right_value), applied_value) in
            residual.iter_mut().zip(right_hand_side).zip(&applied)
        {
            *value = *right_value - *applied_value;
        }
        iterations += columns;
    }
    Err(QmbedError::NonConvergence {
        iterations,
        residual: vector_norm(&residual),
    })
}

/// Reusable action of `(A - shift I)^{-1}`. Stored CSC operators cache one
/// sparse factorization; other operators reuse the plan and solve with
/// restarted GMRES without materializing `A`.
pub struct ShiftInvertPlan {
    operator: Arc<dyn LinearOperator>,
    shift: f64,
    tolerance: f64,
    max_iterations: usize,
    factorization: Option<Box<dyn ShiftedLinearSolver>>,
}

impl std::fmt::Debug for ShiftInvertPlan {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ShiftInvertPlan")
            .field("shape", &self.operator.shape())
            .field("shift", &self.shift)
            .field("tolerance", &self.tolerance)
            .field("max_iterations", &self.max_iterations)
            .field("factorized", &self.factorization.is_some())
            .finish()
    }
}

impl ShiftInvertPlan {
    pub fn new(
        operator: Arc<dyn LinearOperator>,
        shift: f64,
        tolerance: f64,
        max_iterations: usize,
    ) -> Result<Self> {
        let shape = operator.shape();
        if shape.0 != shape.1
            || !shift.is_finite()
            || !tolerance.is_finite()
            || tolerance <= 0.0
            || max_iterations == 0
        {
            return Err(QmbedError::InvalidOptions(
                "shift-invert needs a square operator, finite shift, positive tolerance, and positive iteration cap"
                    .into(),
            ));
        }
        let factorization = operator.shifted_solver(shift)?;
        Ok(Self {
            operator,
            shift,
            tolerance,
            max_iterations,
            factorization,
        })
    }

    pub const fn shift(&self) -> f64 {
        self.shift
    }

    pub const fn is_factorized(&self) -> bool {
        self.factorization.is_some()
    }

    pub fn solve(&self, input: &[Complex64], output: &mut [Complex64]) -> Result<()> {
        if input.len() != self.operator.shape().0 || output.len() != input.len() {
            return Err(QmbedError::DimensionMismatch(
                "shift-invert input and output must match the operator dimension".into(),
            ));
        }
        if let Some(factorization) = &self.factorization {
            return factorization.solve(input, output);
        }
        let solved = gmres_shift_invert(
            self.operator.as_ref(),
            self.shift,
            input,
            self.tolerance,
            self.max_iterations,
        )?;
        output.copy_from_slice(&solved);
        Ok(())
    }
}

impl LinearOperator for ShiftInvertPlan {
    fn shape(&self) -> (usize, usize) {
        self.operator.shape()
    }

    fn format(&self) -> MatrixFormat {
        MatrixFormat::MatrixFree
    }

    fn apply(&self, input: &[Complex64], output: &mut [Complex64]) -> Result<()> {
        self.solve(input, output)
    }
}

fn transformed_apply<O>(
    operator: &O,
    options: &EigshOptions,
    shifted_solver: Option<&dyn ShiftedLinearSolver>,
    input: &[Complex64],
    output: &mut [Complex64],
) -> Result<()>
where
    O: LinearOperator + ?Sized,
{
    match options.target {
        SpectrumTarget::Shift(shift) => {
            if let Some(solver) = shifted_solver {
                return solver.solve(input, output);
            }
            let solved = gmres_shift_invert(
                operator,
                shift,
                input,
                (options.tolerance * 0.1).min(1.0e-10),
                options.max_iterations.max(128),
            )?;
            output.copy_from_slice(&solved);
            Ok(())
        }
        _ => operator.apply(input, output),
    }
}

fn transformed_apply_real<O>(
    operator: &O,
    options: &EigshOptions,
    shifted_solver: Option<&dyn ShiftedLinearSolver>,
    input: &[f64],
    output: &mut [f64],
) -> Result<()>
where
    O: LinearOperator + ?Sized,
{
    match options.target {
        SpectrumTarget::Shift(_) => shifted_solver
            .ok_or_else(|| {
                QmbedError::UnsupportedBackend(
                    "real shift-invert requires a reusable real factorization".into(),
                )
            })?
            .solve_real(input, output),
        _ => operator.apply_real(input, output),
    }
}

fn select_indices(values: &[f64], target: SpectrumTarget, count: usize) -> Vec<usize> {
    if target == SpectrumTarget::BothEnds {
        let mut ordered: Vec<_> = (0..values.len()).collect();
        ordered.sort_by(|&left, &right| {
            values[left]
                .total_cmp(&values[right])
                .then_with(|| left.cmp(&right))
        });
        let lower = count / 2;
        let upper = count - lower;
        let mut selected = Vec::with_capacity(count);
        selected.extend(ordered.iter().take(lower).copied());
        selected.extend(ordered.iter().rev().take(upper).copied());
        selected.sort_by(|&left, &right| values[left].total_cmp(&values[right]));
        return selected;
    }
    let mut indices: Vec<_> = (0..values.len()).collect();
    indices.sort_by(|&left, &right| {
        let left_value = values[left];
        let right_value = values[right];
        let ordering = match target {
            SpectrumTarget::SmallestAlgebraic => left_value.total_cmp(&right_value),
            SpectrumTarget::LargestAlgebraic => right_value.total_cmp(&left_value),
            SpectrumTarget::SmallestMagnitude => left_value.abs().total_cmp(&right_value.abs()),
            SpectrumTarget::LargestMagnitude => right_value.abs().total_cmp(&left_value.abs()),
            SpectrumTarget::BothEnds => unreachable!(),
            SpectrumTarget::Shift(shift) => (left_value - shift)
                .abs()
                .total_cmp(&(right_value - shift).abs()),
        };
        ordering.then_with(|| left.cmp(&right))
    });
    indices.truncate(count);
    indices
}

fn real_inner(left: &[f64], right: &[f64]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(left_value, right_value)| left_value * right_value)
        .sum()
}

fn real_vector_norm(vector: &[f64]) -> f64 {
    real_inner(vector, vector).sqrt()
}

#[cfg_attr(not(test), allow(dead_code))]
fn dgks_reorthogonalize_real(basis: &[Vec<f64>], output: &mut [f64]) -> (f64, bool) {
    let norm_before_reorthogonalization = real_vector_norm(output);
    for vector in basis {
        let overlap = real_inner(vector, output);
        for (value, basis_value) in output.iter_mut().zip(vector) {
            *value -= overlap * *basis_value;
        }
    }
    let norm_after_first_pass = real_vector_norm(output);
    if norm_before_reorthogonalization > f64::EPSILON
        && norm_after_first_pass
            <= DGKS_REORTHOGONALIZATION_THRESHOLD * norm_before_reorthogonalization
    {
        for vector in basis {
            let overlap = real_inner(vector, output);
            for (value, basis_value) in output.iter_mut().zip(vector) {
                *value -= overlap * *basis_value;
            }
        }
        (real_vector_norm(output), true)
    } else {
        (norm_after_first_pass, false)
    }
}

#[cfg_attr(not(test), allow(dead_code))]
fn dgks_reorthogonalize_complex(basis: &[Vec<Complex64>], output: &mut [Complex64]) -> (f64, bool) {
    let norm_before_reorthogonalization = vector_norm(output);
    for vector in basis {
        let overlap = inner(vector, output);
        for (value, basis_value) in output.iter_mut().zip(vector) {
            *value -= overlap * *basis_value;
        }
    }
    let norm_after_first_pass = vector_norm(output);
    if norm_before_reorthogonalization > f64::EPSILON
        && norm_after_first_pass
            <= DGKS_REORTHOGONALIZATION_THRESHOLD * norm_before_reorthogonalization
    {
        for vector in basis {
            let overlap = inner(vector, output);
            for (value, basis_value) in output.iter_mut().zip(vector) {
                *value -= overlap * *basis_value;
            }
        }
        (vector_norm(output), true)
    } else {
        (norm_after_first_pass, false)
    }
}

fn normalize_real(vector: &mut [f64]) -> Result<()> {
    let norm = real_vector_norm(vector);
    if !norm.is_finite() || norm <= f64::EPSILON {
        return Err(QmbedError::NonConvergence {
            iterations: 0,
            residual: norm,
        });
    }
    for value in vector {
        *value /= norm;
    }
    Ok(())
}

#[allow(dead_code)]
fn lanczos_eigsh_real<O>(
    operator: &O,
    options: &EigshOptions,
    initial: Option<&[Complex64]>,
    shifted_solver: Option<Box<dyn ShiftedLinearSolver>>,
) -> Result<Eigensystem>
where
    O: LinearOperator + ?Sized,
{
    let dimension = operator.shape().0;
    let requested_dimension = options
        .krylov_dimension
        .unwrap_or_else(|| (8 * options.eigenpairs + 64).max(256));
    let krylov_dimension = requested_dimension
        .min(options.max_iterations)
        .min(dimension);
    if krylov_dimension <= options.eigenpairs {
        return Err(QmbedError::InvalidOptions(
            "the effective Krylov dimension must exceed eigenpairs".into(),
        ));
    }

    let first = if let Some(initial) = initial {
        if initial.len() != dimension {
            return Err(QmbedError::DimensionMismatch(
                "eigsh initial vector does not match the operator".into(),
            ));
        }
        let mut first: Vec<_> = initial.iter().map(|value| value.re).collect();
        normalize_real(&mut first)?;
        first
    } else {
        deterministic_start(dimension, options.seed)?
            .into_iter()
            .map(|value| value.re)
            .collect()
    };
    let mut basis = Vec::with_capacity(krylov_dimension);
    basis.push(first);
    let mut alphas = Vec::with_capacity(krylov_dimension);
    let mut betas = Vec::with_capacity(krylov_dimension.saturating_sub(1));
    let mut output = vec![0.0; dimension];
    let mut reorthogonalization_passes = 0;
    let mut conditional_second_passes = 0;

    for iteration in 0..krylov_dimension {
        transformed_apply_real(
            operator,
            options,
            shifted_solver.as_deref(),
            &basis[iteration],
            &mut output,
        )?;
        let alpha = real_inner(&basis[iteration], &output);
        alphas.push(alpha);
        for (value, basis_value) in output.iter_mut().zip(&basis[iteration]) {
            *value -= alpha * *basis_value;
        }
        if iteration > 0 {
            let beta = betas[iteration - 1];
            for (value, previous) in output.iter_mut().zip(&basis[iteration - 1]) {
                *value -= beta * *previous;
            }
        }

        let (beta, second_pass) = dgks_reorthogonalize_real(&basis, &mut output);
        reorthogonalization_passes += 1 + usize::from(second_pass);
        if second_pass {
            conditional_second_passes += 1;
        }
        if iteration + 1 == krylov_dimension || beta <= 1.0e-14 {
            break;
        }
        betas.push(beta);
        for value in &mut output {
            *value /= beta;
        }
        basis.push(output.clone());
    }

    if basis.len() <= options.eigenpairs {
        return Err(QmbedError::NonConvergence {
            iterations: basis.len(),
            residual: f64::INFINITY,
        });
    }
    let size = basis.len();
    let mut tridiagonal = DMatrix::<f64>::zeros(size, size);
    for index in 0..size {
        tridiagonal[(index, index)] = alphas[index];
        if index + 1 < size {
            tridiagonal[(index, index + 1)] = betas[index];
            tridiagonal[(index + 1, index)] = betas[index];
        }
    }
    let decomposition = SymmetricEigen::new(tridiagonal);
    let transformed_target = if matches!(options.target, SpectrumTarget::Shift(_)) {
        SpectrumTarget::LargestMagnitude
    } else {
        options.target
    };
    let indices = select_indices(
        decomposition.eigenvalues.as_slice(),
        transformed_target,
        options.eigenpairs,
    );

    let mut candidates = Vec::with_capacity(options.eigenpairs);
    for index in indices {
        let mut vector = vec![0.0; dimension];
        for (basis_index, basis_vector) in basis.iter().enumerate() {
            let coefficient = decomposition.eigenvectors[(basis_index, index)];
            for (value, basis_value) in vector.iter_mut().zip(basis_vector) {
                *value += coefficient * *basis_value;
            }
        }
        normalize_real(&mut vector)?;
        operator.apply_real(&vector, &mut output)?;
        let eigenvalue = real_inner(&vector, &output);
        let residual = output
            .iter()
            .zip(&vector)
            .map(|(actual, component)| (actual - eigenvalue * component).powi(2))
            .sum::<f64>()
            .sqrt();
        candidates.push((eigenvalue, vector, residual));
    }
    candidates.sort_by(|left, right| match options.target {
        SpectrumTarget::SmallestAlgebraic => left.0.total_cmp(&right.0),
        SpectrumTarget::LargestAlgebraic => right.0.total_cmp(&left.0),
        SpectrumTarget::SmallestMagnitude => left.0.abs().total_cmp(&right.0.abs()),
        SpectrumTarget::LargestMagnitude => right.0.abs().total_cmp(&left.0.abs()),
        SpectrumTarget::BothEnds => left.0.total_cmp(&right.0),
        SpectrumTarget::Shift(shift) => (left.0 - shift).abs().total_cmp(&(right.0 - shift).abs()),
    });
    let residuals: Vec<_> = candidates.iter().map(|candidate| candidate.2).collect();
    let failure_residual = candidates
        .iter()
        .filter_map(|candidate| {
            (candidate.2 > options.tolerance * candidate.0.abs().max(1.0)).then_some(candidate.2)
        })
        .fold(0.0_f64, f64::max);
    if failure_residual > 0.0 {
        return Err(QmbedError::NonConvergence {
            iterations: size,
            residual: failure_residual,
        });
    }
    Ok(Eigensystem {
        eigenvalues: candidates.iter().map(|candidate| candidate.0).collect(),
        eigenvectors: candidates
            .into_iter()
            .map(|candidate| {
                candidate
                    .1
                    .into_iter()
                    .map(|value| Complex64::new(value, 0.0))
                    .collect()
            })
            .collect(),
        residuals,
        iterations: size,
        reorthogonalization_passes,
        conditional_second_passes,
        converged: true,
    })
}

#[allow(dead_code)]
fn lanczos_eigsh<O>(
    operator: &O,
    options: &EigshOptions,
    initial: Option<&[Complex64]>,
) -> Result<Eigensystem>
where
    O: LinearOperator + ?Sized,
{
    let real_compatible = operator.is_real()
        && initial.is_none_or(|vector| vector.iter().all(|value| value.im.abs() <= 1.0e-14));
    let mut shifted_solver = match options.target {
        SpectrumTarget::Shift(shift) if real_compatible => operator.shifted_solver(shift)?,
        _ => None,
    };
    if real_compatible
        && (!matches!(options.target, SpectrumTarget::Shift(_))
            || shifted_solver
                .as_ref()
                .is_some_and(|solver| solver.supports_real()))
    {
        return lanczos_eigsh_real(operator, options, initial, shifted_solver);
    }
    let dimension = operator.shape().0;
    let requested_dimension = options
        .krylov_dimension
        .unwrap_or_else(|| (8 * options.eigenpairs + 64).max(256));
    let krylov_dimension = requested_dimension
        .min(options.max_iterations)
        .min(dimension);
    if krylov_dimension <= options.eigenpairs {
        return Err(QmbedError::InvalidOptions(
            "the effective Krylov dimension must exceed eigenpairs".into(),
        ));
    }

    let mut basis = Vec::with_capacity(krylov_dimension);
    basis.push(if let Some(initial) = initial {
        if initial.len() != dimension {
            return Err(QmbedError::DimensionMismatch(
                "eigsh initial vector does not match the operator".into(),
            ));
        }
        let mut initial = initial.to_vec();
        normalize(&mut initial)?;
        initial
    } else {
        deterministic_start(dimension, options.seed)?
    });
    let mut alphas = Vec::with_capacity(krylov_dimension);
    let mut betas = Vec::with_capacity(krylov_dimension.saturating_sub(1));
    let mut output = vec![Complex64::new(0.0, 0.0); dimension];
    let mut reorthogonalization_passes = 0;
    let mut conditional_second_passes = 0;
    if let SpectrumTarget::Shift(shift) = options.target {
        if shifted_solver.is_none() {
            shifted_solver = operator.shifted_solver(shift)?;
        }
    }

    for iteration in 0..krylov_dimension {
        transformed_apply(
            operator,
            options,
            shifted_solver.as_deref(),
            &basis[iteration],
            &mut output,
        )?;
        let alpha = inner(&basis[iteration], &output).re;
        alphas.push(alpha);
        for (value, basis_value) in output.iter_mut().zip(&basis[iteration]) {
            *value -= alpha * *basis_value;
        }
        if iteration > 0 {
            let beta = betas[iteration - 1];
            for (value, previous) in output.iter_mut().zip(&basis[iteration - 1]) {
                *value -= beta * *previous;
            }
        }

        let (beta, second_pass) = dgks_reorthogonalize_complex(&basis, &mut output);
        reorthogonalization_passes += 1 + usize::from(second_pass);
        if second_pass {
            conditional_second_passes += 1;
        }
        if iteration + 1 == krylov_dimension || beta <= 1.0e-14 {
            break;
        }
        betas.push(beta);
        for value in &mut output {
            *value /= beta;
        }
        basis.push(output.clone());
    }

    if basis.len() <= options.eigenpairs {
        return Err(QmbedError::NonConvergence {
            iterations: basis.len(),
            residual: f64::INFINITY,
        });
    }
    let size = basis.len();
    let mut tridiagonal = DMatrix::<f64>::zeros(size, size);
    for index in 0..size {
        tridiagonal[(index, index)] = alphas[index];
        if index + 1 < size {
            tridiagonal[(index, index + 1)] = betas[index];
            tridiagonal[(index + 1, index)] = betas[index];
        }
    }
    let decomposition = SymmetricEigen::new(tridiagonal);
    let transformed_values = decomposition.eigenvalues.as_slice();
    let transformed_target = if matches!(options.target, SpectrumTarget::Shift(_)) {
        SpectrumTarget::LargestMagnitude
    } else {
        options.target
    };
    let indices = select_indices(transformed_values, transformed_target, options.eigenpairs);

    let mut candidates = Vec::with_capacity(options.eigenpairs);
    for index in indices {
        let mut vector = vec![Complex64::new(0.0, 0.0); dimension];
        for (basis_index, basis_vector) in basis.iter().enumerate() {
            let coefficient = decomposition.eigenvectors[(basis_index, index)];
            for (value, basis_value) in vector.iter_mut().zip(basis_vector) {
                *value += coefficient * *basis_value;
            }
        }
        normalize(&mut vector)?;
        operator.apply(&vector, &mut output)?;
        let eigenvalue = inner(&vector, &output).re;
        let residual = output
            .iter()
            .zip(&vector)
            .map(|(actual, component)| (*actual - eigenvalue * *component).norm_sqr())
            .sum::<f64>()
            .sqrt();
        candidates.push((eigenvalue, vector, residual));
    }
    candidates.sort_by(|left, right| match options.target {
        SpectrumTarget::SmallestAlgebraic => left.0.total_cmp(&right.0),
        SpectrumTarget::LargestAlgebraic => right.0.total_cmp(&left.0),
        SpectrumTarget::SmallestMagnitude => left.0.abs().total_cmp(&right.0.abs()),
        SpectrumTarget::LargestMagnitude => right.0.abs().total_cmp(&left.0.abs()),
        SpectrumTarget::BothEnds => left.0.total_cmp(&right.0),
        SpectrumTarget::Shift(shift) => (left.0 - shift).abs().total_cmp(&(right.0 - shift).abs()),
    });
    let residuals: Vec<_> = candidates.iter().map(|candidate| candidate.2).collect();
    let failure_residual = candidates
        .iter()
        .filter_map(|candidate| {
            (candidate.2 > options.tolerance * candidate.0.abs().max(1.0)).then_some(candidate.2)
        })
        .fold(0.0_f64, f64::max);
    if failure_residual > 0.0 {
        return Err(QmbedError::NonConvergence {
            iterations: size,
            residual: failure_residual,
        });
    }
    Ok(Eigensystem {
        eigenvalues: candidates.iter().map(|candidate| candidate.0).collect(),
        eigenvectors: candidates
            .into_iter()
            .map(|candidate| candidate.1)
            .collect(),
        residuals,
        iterations: size,
        reorthogonalization_passes,
        conditional_second_passes,
        converged: true,
    })
}

#[derive(Clone)]
struct RealRitzCandidate {
    value: f64,
    vector: Vec<f64>,
    residual: f64,
    transformed_action: Vec<f64>,
}

#[derive(Clone)]
struct ComplexRitzCandidate {
    value: f64,
    vector: Vec<Complex64>,
    residual: f64,
    transformed_action: Vec<Complex64>,
}

struct RetainedRealVector {
    vector: Vec<f64>,
    transformed_action: Option<Vec<f64>>,
}

struct RetainedComplexVector {
    vector: Vec<Complex64>,
    transformed_action: Option<Vec<Complex64>>,
}

fn sort_real_candidates(candidates: &mut [RealRitzCandidate], target: SpectrumTarget) {
    candidates.sort_by(|left, right| match target {
        SpectrumTarget::SmallestAlgebraic => left.value.total_cmp(&right.value),
        SpectrumTarget::LargestAlgebraic => right.value.total_cmp(&left.value),
        SpectrumTarget::SmallestMagnitude => left.value.abs().total_cmp(&right.value.abs()),
        SpectrumTarget::LargestMagnitude => right.value.abs().total_cmp(&left.value.abs()),
        SpectrumTarget::BothEnds => left.value.total_cmp(&right.value),
        SpectrumTarget::Shift(shift) => (left.value - shift)
            .abs()
            .total_cmp(&(right.value - shift).abs()),
    });
}

fn sort_complex_candidates(candidates: &mut [ComplexRitzCandidate], target: SpectrumTarget) {
    candidates.sort_by(|left, right| match target {
        SpectrumTarget::SmallestAlgebraic => left.value.total_cmp(&right.value),
        SpectrumTarget::LargestAlgebraic => right.value.total_cmp(&left.value),
        SpectrumTarget::SmallestMagnitude => left.value.abs().total_cmp(&right.value.abs()),
        SpectrumTarget::LargestMagnitude => right.value.abs().total_cmp(&left.value.abs()),
        SpectrumTarget::BothEnds => left.value.total_cmp(&right.value),
        SpectrumTarget::Shift(shift) => (left.value - shift)
            .abs()
            .total_cmp(&(right.value - shift).abs()),
    });
}

fn orthonormalize_real(
    locked: &[RealRitzCandidate],
    basis: &[Vec<f64>],
    vector: Vec<f64>,
) -> (Option<Vec<f64>>, usize, usize) {
    let (vector, passes, second_passes, _) =
        orthonormalize_real_with_projection(locked, basis, vector);
    (vector, passes, second_passes)
}

fn orthonormalize_real_with_projection(
    locked: &[RealRitzCandidate],
    basis: &[Vec<f64>],
    mut vector: Vec<f64>,
) -> (Option<Vec<f64>>, usize, usize, Vec<f64>) {
    let norm_before = real_vector_norm(&vector);
    for candidate in locked {
        let overlap = real_inner(&candidate.vector, &vector);
        for (value, locked_value) in vector.iter_mut().zip(&candidate.vector) {
            *value -= overlap * *locked_value;
        }
    }
    let mut projection = Vec::with_capacity(basis.len());
    for basis_vector in basis {
        let overlap = real_inner(basis_vector, &vector);
        projection.push(overlap);
        for (value, basis_value) in vector.iter_mut().zip(basis_vector) {
            *value -= overlap * *basis_value;
        }
    }
    let mut norm = real_vector_norm(&vector);
    let second_pass =
        norm_before > f64::EPSILON && norm <= DGKS_REORTHOGONALIZATION_THRESHOLD * norm_before;
    if second_pass {
        for candidate in locked {
            let overlap = real_inner(&candidate.vector, &vector);
            for (value, locked_value) in vector.iter_mut().zip(&candidate.vector) {
                *value -= overlap * *locked_value;
            }
        }
        for basis_vector in basis {
            let overlap = real_inner(basis_vector, &vector);
            for (value, basis_value) in vector.iter_mut().zip(basis_vector) {
                *value -= overlap * *basis_value;
            }
        }
        norm = real_vector_norm(&vector);
    }
    if !norm.is_finite() || norm <= 1.0e-14 {
        return (
            None,
            1 + usize::from(second_pass),
            usize::from(second_pass),
            projection,
        );
    }
    for value in &mut vector {
        *value /= norm;
    }
    (
        Some(vector),
        1 + usize::from(second_pass),
        usize::from(second_pass),
        projection,
    )
}

fn orthonormalize_complex(
    locked: &[ComplexRitzCandidate],
    basis: &[Vec<Complex64>],
    vector: Vec<Complex64>,
) -> (Option<Vec<Complex64>>, usize, usize) {
    let (vector, passes, second_passes, _) =
        orthonormalize_complex_with_projection(locked, basis, vector);
    (vector, passes, second_passes)
}

fn orthonormalize_complex_with_projection(
    locked: &[ComplexRitzCandidate],
    basis: &[Vec<Complex64>],
    mut vector: Vec<Complex64>,
) -> (Option<Vec<Complex64>>, usize, usize, Vec<Complex64>) {
    let norm_before = vector_norm(&vector);
    for candidate in locked {
        let overlap = inner(&candidate.vector, &vector);
        for (value, locked_value) in vector.iter_mut().zip(&candidate.vector) {
            *value -= overlap * *locked_value;
        }
    }
    let mut projection = Vec::with_capacity(basis.len());
    for basis_vector in basis {
        let overlap = inner(basis_vector, &vector);
        projection.push(overlap);
        for (value, basis_value) in vector.iter_mut().zip(basis_vector) {
            *value -= overlap * *basis_value;
        }
    }
    let mut norm = vector_norm(&vector);
    let second_pass =
        norm_before > f64::EPSILON && norm <= DGKS_REORTHOGONALIZATION_THRESHOLD * norm_before;
    if second_pass {
        for candidate in locked {
            let overlap = inner(&candidate.vector, &vector);
            for (value, locked_value) in vector.iter_mut().zip(&candidate.vector) {
                *value -= overlap * *locked_value;
            }
        }
        for basis_vector in basis {
            let overlap = inner(basis_vector, &vector);
            for (value, basis_value) in vector.iter_mut().zip(basis_vector) {
                *value -= overlap * *basis_value;
            }
        }
        norm = vector_norm(&vector);
    }
    if !norm.is_finite() || norm <= 1.0e-14 {
        return (
            None,
            1 + usize::from(second_pass),
            usize::from(second_pass),
            projection,
        );
    }
    for value in &mut vector {
        *value /= norm;
    }
    (
        Some(vector),
        1 + usize::from(second_pass),
        usize::from(second_pass),
        projection,
    )
}

fn thick_restart_dimension(options: &EigshOptions, dimension: usize) -> usize {
    options
        .krylov_dimension
        .unwrap_or_else(|| (4 * options.eigenpairs + 24).max(32))
        .min(dimension)
}

fn thick_restarted_eigsh_real<O>(
    operator: &O,
    options: &EigshOptions,
    initial_subspace: Option<&[Vec<Complex64>]>,
    shifted_solver: Option<Box<dyn ShiftedLinearSolver>>,
) -> Result<Eigensystem>
where
    O: LinearOperator + ?Sized,
{
    let dimension = operator.shape().0;
    let restart_dimension = thick_restart_dimension(options, dimension);
    if restart_dimension <= options.eigenpairs {
        return Err(QmbedError::InvalidOptions(
            "the effective Krylov dimension must exceed eigenpairs".into(),
        ));
    }
    let mut retained = initial_subspace
        .unwrap_or_default()
        .iter()
        .map(|vector| RetainedRealVector {
            vector: vector.iter().map(|value| value.re).collect::<Vec<_>>(),
            transformed_action: None,
        })
        .collect::<Vec<_>>();
    let mut locked = Vec::<RealRitzCandidate>::new();
    let mut iterations = 0usize;
    let mut reorthogonalization_passes = 0usize;
    let mut conditional_second_passes = 0usize;
    let mut last_residual = f64::INFINITY;
    let mut cycle = 0_u64;

    while iterations < options.max_iterations && locked.len() < options.eigenpairs {
        let mut basis = Vec::with_capacity(restart_dimension);
        let mut applied = Vec::with_capacity(restart_dimension);
        let reuse_actions = !retained.is_empty()
            && retained
                .iter()
                .all(|vector| vector.transformed_action.is_some());
        for mut retained_vector in retained.drain(..) {
            if reuse_actions {
                let norm = real_vector_norm(&retained_vector.vector);
                if norm.is_finite() && norm > 1.0e-14 {
                    for value in &mut retained_vector.vector {
                        *value /= norm;
                    }
                    let mut action = retained_vector.transformed_action.take().unwrap();
                    for value in &mut action {
                        *value /= norm;
                    }
                    basis.push(retained_vector.vector);
                    applied.push(action);
                }
            } else {
                let (vector, passes, second_passes) =
                    orthonormalize_real(&locked, &basis, retained_vector.vector);
                reorthogonalization_passes += passes;
                conditional_second_passes += second_passes;
                if let Some(vector) = vector {
                    basis.push(vector);
                }
            }
            if basis.len() >= restart_dimension / 2 {
                break;
            }
        }
        if basis.is_empty() {
            let start = deterministic_start(
                dimension,
                options.seed.wrapping_add(cycle.wrapping_mul(0x9e37_79b9)),
            )?
            .into_iter()
            .map(|value| value.re)
            .collect();
            let (start, passes, second_passes) = orthonormalize_real(&locked, &basis, start);
            reorthogonalization_passes += passes;
            conditional_second_passes += second_passes;
            if let Some(start) = start {
                basis.push(start);
            }
        }
        if basis.is_empty() {
            break;
        }

        let known_actions = applied.len();
        let mut projection_columns = Vec::with_capacity(restart_dimension);
        if reuse_actions {
            for (column, action) in applied[..known_actions].iter().cloned().enumerate() {
                if basis.len() >= restart_dimension {
                    break;
                }
                let (residual_direction, passes, second_passes, projection) =
                    orthonormalize_real_with_projection(&locked, &basis, action);
                reorthogonalization_passes += passes;
                conditional_second_passes += second_passes;
                projection_columns.push(projection[..=column].to_vec());
                if let Some(residual_direction) = residual_direction {
                    basis.push(residual_direction);
                }
            }
        }
        let mut cursor = known_actions;
        while cursor < basis.len()
            && iterations < options.max_iterations
            && applied.len() < restart_dimension
        {
            let mut output = vec![0.0; dimension];
            transformed_apply_real(
                operator,
                options,
                shifted_solver.as_deref(),
                &basis[cursor],
                &mut output,
            )?;
            iterations += 1;
            applied.push(output.clone());
            let (next, passes, second_passes, projection) =
                orthonormalize_real_with_projection(&locked, &basis, output);
            reorthogonalization_passes += passes;
            conditional_second_passes += second_passes;
            projection_columns.push(projection[..=cursor].to_vec());
            if basis.len() < restart_dimension {
                if let Some(next) = next {
                    basis.push(next);
                }
            }
            cursor += 1;
        }
        basis.truncate(applied.len());
        if basis.is_empty() {
            break;
        }

        let size = basis.len();
        let mut projected = DMatrix::<f64>::zeros(size, size);
        for (column, projection) in projection_columns.iter().enumerate().take(size) {
            for (row, &value) in projection.iter().enumerate().take(column + 1) {
                projected[(row, column)] = value;
                projected[(column, row)] = value;
            }
        }
        let decomposition = SymmetricEigen::new(projected);
        let transformed_target = if matches!(options.target, SpectrumTarget::Shift(_)) {
            SpectrumTarget::LargestMagnitude
        } else {
            options.target
        };
        let active_needed = options.eigenpairs - locked.len();
        let keep_limit = active_needed.min(restart_dimension / 2).max(1);
        let indices = select_indices(
            decomposition.eigenvalues.as_slice(),
            transformed_target,
            keep_limit.min(size),
        );
        let mut candidates = Vec::with_capacity(indices.len());
        let mut original_action = vec![0.0; dimension];
        for index in indices {
            let mut vector = vec![0.0; dimension];
            let mut transformed_action = vec![0.0; dimension];
            for (basis_index, basis_vector) in basis.iter().enumerate() {
                let coefficient = decomposition.eigenvectors[(basis_index, index)];
                for (value, basis_value) in vector.iter_mut().zip(basis_vector) {
                    *value += coefficient * *basis_value;
                }
                for (value, applied_value) in
                    transformed_action.iter_mut().zip(&applied[basis_index])
                {
                    *value += coefficient * *applied_value;
                }
            }
            let ritz_norm = real_vector_norm(&vector);
            if !ritz_norm.is_finite() || ritz_norm <= f64::EPSILON {
                return Err(QmbedError::NonConvergence {
                    iterations,
                    residual: ritz_norm,
                });
            }
            for value in &mut vector {
                *value /= ritz_norm;
            }
            for value in &mut transformed_action {
                *value /= ritz_norm;
            }
            if matches!(options.target, SpectrumTarget::Shift(_)) {
                operator.apply_real(&vector, &mut original_action)?;
            } else {
                original_action.copy_from_slice(&transformed_action);
            }
            let value = real_inner(&vector, &original_action);
            let residual = original_action
                .iter()
                .zip(&vector)
                .map(|(actual, component)| (actual - value * component).powi(2))
                .sum::<f64>()
                .sqrt();
            candidates.push(RealRitzCandidate {
                value,
                vector,
                residual,
                transformed_action,
            });
        }
        sort_real_candidates(&mut candidates, options.target);
        last_residual = candidates
            .iter()
            .take(active_needed)
            .map(|candidate| candidate.residual)
            .fold(0.0_f64, f64::max);

        let mut next_retained = Vec::new();
        for (rank, candidate) in candidates.into_iter().enumerate() {
            let converged =
                candidate.residual <= options.tolerance * candidate.value.abs().max(1.0);
            if rank < active_needed && converged && locked.len() < options.eigenpairs {
                let (vector, passes, second_passes) =
                    orthonormalize_real(&locked, &[], candidate.vector);
                reorthogonalization_passes += passes;
                conditional_second_passes += second_passes;
                if let Some(vector) = vector {
                    locked.push(RealRitzCandidate {
                        vector,
                        ..candidate
                    });
                }
            } else if next_retained.len() < keep_limit {
                next_retained.push(RetainedRealVector {
                    vector: candidate.vector,
                    transformed_action: Some(candidate.transformed_action),
                });
            }
        }
        retained = next_retained;
        cycle = cycle.wrapping_add(1);
    }

    if locked.len() < options.eigenpairs {
        return Err(QmbedError::NonConvergence {
            iterations,
            residual: last_residual,
        });
    }
    sort_real_candidates(&mut locked, options.target);
    locked.truncate(options.eigenpairs);
    Ok(Eigensystem {
        eigenvalues: locked.iter().map(|candidate| candidate.value).collect(),
        eigenvectors: locked
            .iter()
            .map(|candidate| {
                candidate
                    .vector
                    .iter()
                    .map(|value| Complex64::new(*value, 0.0))
                    .collect()
            })
            .collect(),
        residuals: locked.iter().map(|candidate| candidate.residual).collect(),
        iterations,
        reorthogonalization_passes,
        conditional_second_passes,
        converged: true,
    })
}

fn thick_restarted_eigsh_complex<O>(
    operator: &O,
    options: &EigshOptions,
    initial_subspace: Option<&[Vec<Complex64>]>,
    shifted_solver: Option<Box<dyn ShiftedLinearSolver>>,
) -> Result<Eigensystem>
where
    O: LinearOperator + ?Sized,
{
    let dimension = operator.shape().0;
    let restart_dimension = thick_restart_dimension(options, dimension);
    if restart_dimension <= options.eigenpairs {
        return Err(QmbedError::InvalidOptions(
            "the effective Krylov dimension must exceed eigenpairs".into(),
        ));
    }
    let mut retained = initial_subspace
        .unwrap_or_default()
        .iter()
        .cloned()
        .map(|vector| RetainedComplexVector {
            vector,
            transformed_action: None,
        })
        .collect::<Vec<_>>();
    let mut locked = Vec::<ComplexRitzCandidate>::new();
    let mut iterations = 0usize;
    let mut reorthogonalization_passes = 0usize;
    let mut conditional_second_passes = 0usize;
    let mut last_residual = f64::INFINITY;
    let mut cycle = 0_u64;

    while iterations < options.max_iterations && locked.len() < options.eigenpairs {
        let mut basis = Vec::with_capacity(restart_dimension);
        let mut applied = Vec::with_capacity(restart_dimension);
        let reuse_actions = !retained.is_empty()
            && retained
                .iter()
                .all(|vector| vector.transformed_action.is_some());
        for mut retained_vector in retained.drain(..) {
            if reuse_actions {
                let norm = vector_norm(&retained_vector.vector);
                if norm.is_finite() && norm > 1.0e-14 {
                    for value in &mut retained_vector.vector {
                        *value /= norm;
                    }
                    let mut action = retained_vector.transformed_action.take().unwrap();
                    for value in &mut action {
                        *value /= norm;
                    }
                    basis.push(retained_vector.vector);
                    applied.push(action);
                }
            } else {
                let (vector, passes, second_passes) =
                    orthonormalize_complex(&locked, &basis, retained_vector.vector);
                reorthogonalization_passes += passes;
                conditional_second_passes += second_passes;
                if let Some(vector) = vector {
                    basis.push(vector);
                }
            }
            if basis.len() >= restart_dimension / 2 {
                break;
            }
        }
        if basis.is_empty() {
            let start = deterministic_start(
                dimension,
                options.seed.wrapping_add(cycle.wrapping_mul(0x9e37_79b9)),
            )?;
            let (start, passes, second_passes) = orthonormalize_complex(&locked, &basis, start);
            reorthogonalization_passes += passes;
            conditional_second_passes += second_passes;
            if let Some(start) = start {
                basis.push(start);
            }
        }
        if basis.is_empty() {
            break;
        }

        let known_actions = applied.len();
        let mut projection_columns = Vec::with_capacity(restart_dimension);
        if reuse_actions {
            for (column, action) in applied[..known_actions].iter().cloned().enumerate() {
                if basis.len() >= restart_dimension {
                    break;
                }
                let (residual_direction, passes, second_passes, projection) =
                    orthonormalize_complex_with_projection(&locked, &basis, action);
                reorthogonalization_passes += passes;
                conditional_second_passes += second_passes;
                projection_columns.push(projection[..=column].to_vec());
                if let Some(residual_direction) = residual_direction {
                    basis.push(residual_direction);
                }
            }
        }
        let mut cursor = known_actions;
        while cursor < basis.len()
            && iterations < options.max_iterations
            && applied.len() < restart_dimension
        {
            let mut output = vec![Complex64::new(0.0, 0.0); dimension];
            transformed_apply(
                operator,
                options,
                shifted_solver.as_deref(),
                &basis[cursor],
                &mut output,
            )?;
            iterations += 1;
            applied.push(output.clone());
            let (next, passes, second_passes, projection) =
                orthonormalize_complex_with_projection(&locked, &basis, output);
            reorthogonalization_passes += passes;
            conditional_second_passes += second_passes;
            projection_columns.push(projection[..=cursor].to_vec());
            if basis.len() < restart_dimension {
                if let Some(next) = next {
                    basis.push(next);
                }
            }
            cursor += 1;
        }
        basis.truncate(applied.len());
        if basis.is_empty() {
            break;
        }

        let size = basis.len();
        let mut projected = DMatrix::<Complex64>::zeros(size, size);
        for (column, projection) in projection_columns.iter().enumerate().take(size) {
            for (row, &value) in projection.iter().enumerate().take(column + 1) {
                projected[(row, column)] = value;
                projected[(column, row)] = value.conj();
            }
        }
        let decomposition = SymmetricEigen::new(projected);
        let transformed_target = if matches!(options.target, SpectrumTarget::Shift(_)) {
            SpectrumTarget::LargestMagnitude
        } else {
            options.target
        };
        let active_needed = options.eigenpairs - locked.len();
        let keep_limit = active_needed.min(restart_dimension / 2).max(1);
        let indices = select_indices(
            decomposition.eigenvalues.as_slice(),
            transformed_target,
            keep_limit.min(size),
        );
        let mut candidates = Vec::with_capacity(indices.len());
        let mut original_action = vec![Complex64::new(0.0, 0.0); dimension];
        for index in indices {
            let mut vector = vec![Complex64::new(0.0, 0.0); dimension];
            let mut transformed_action = vec![Complex64::new(0.0, 0.0); dimension];
            for (basis_index, basis_vector) in basis.iter().enumerate() {
                let coefficient = decomposition.eigenvectors[(basis_index, index)];
                for (value, basis_value) in vector.iter_mut().zip(basis_vector) {
                    *value += coefficient * *basis_value;
                }
                for (value, applied_value) in
                    transformed_action.iter_mut().zip(&applied[basis_index])
                {
                    *value += coefficient * *applied_value;
                }
            }
            let ritz_norm = vector_norm(&vector);
            if !ritz_norm.is_finite() || ritz_norm <= f64::EPSILON {
                return Err(QmbedError::NonConvergence {
                    iterations,
                    residual: ritz_norm,
                });
            }
            for value in &mut vector {
                *value /= ritz_norm;
            }
            for value in &mut transformed_action {
                *value /= ritz_norm;
            }
            if matches!(options.target, SpectrumTarget::Shift(_)) {
                operator.apply(&vector, &mut original_action)?;
            } else {
                original_action.copy_from_slice(&transformed_action);
            }
            let value = inner(&vector, &original_action).re;
            let residual = original_action
                .iter()
                .zip(&vector)
                .map(|(actual, component)| (*actual - value * *component).norm_sqr())
                .sum::<f64>()
                .sqrt();
            candidates.push(ComplexRitzCandidate {
                value,
                vector,
                residual,
                transformed_action,
            });
        }
        sort_complex_candidates(&mut candidates, options.target);
        last_residual = candidates
            .iter()
            .take(active_needed)
            .map(|candidate| candidate.residual)
            .fold(0.0_f64, f64::max);

        let mut next_retained = Vec::new();
        for (rank, candidate) in candidates.into_iter().enumerate() {
            let converged =
                candidate.residual <= options.tolerance * candidate.value.abs().max(1.0);
            if rank < active_needed && converged && locked.len() < options.eigenpairs {
                let (vector, passes, second_passes) =
                    orthonormalize_complex(&locked, &[], candidate.vector);
                reorthogonalization_passes += passes;
                conditional_second_passes += second_passes;
                if let Some(vector) = vector {
                    locked.push(ComplexRitzCandidate {
                        vector,
                        ..candidate
                    });
                }
            } else if next_retained.len() < keep_limit {
                next_retained.push(RetainedComplexVector {
                    vector: candidate.vector,
                    transformed_action: Some(candidate.transformed_action),
                });
            }
        }
        retained = next_retained;
        cycle = cycle.wrapping_add(1);
    }

    if locked.len() < options.eigenpairs {
        return Err(QmbedError::NonConvergence {
            iterations,
            residual: last_residual,
        });
    }
    sort_complex_candidates(&mut locked, options.target);
    locked.truncate(options.eigenpairs);
    Ok(Eigensystem {
        eigenvalues: locked.iter().map(|candidate| candidate.value).collect(),
        eigenvectors: locked
            .iter()
            .map(|candidate| candidate.vector.clone())
            .collect(),
        residuals: locked.iter().map(|candidate| candidate.residual).collect(),
        iterations,
        reorthogonalization_passes,
        conditional_second_passes,
        converged: true,
    })
}

fn thick_restarted_eigsh<O>(
    operator: &O,
    options: &EigshOptions,
    initial_subspace: Option<&[Vec<Complex64>]>,
) -> Result<Eigensystem>
where
    O: LinearOperator + ?Sized,
{
    let real_compatible = operator.is_real()
        && initial_subspace.is_none_or(|vectors| {
            vectors
                .iter()
                .all(|vector| vector.iter().all(|value| value.im.abs() <= 1.0e-14))
        });
    let shifted_solver = match options.target {
        SpectrumTarget::Shift(shift) => operator.shifted_solver(shift)?,
        _ => None,
    };
    if real_compatible
        && (!matches!(options.target, SpectrumTarget::Shift(_))
            || shifted_solver
                .as_ref()
                .is_some_and(|solver| solver.supports_real()))
    {
        return thick_restarted_eigsh_real(operator, options, initial_subspace, shifted_solver);
    }
    thick_restarted_eigsh_complex(operator, options, initial_subspace, shifted_solver)
}

fn dense_eigsh<O>(operator: &O, options: &EigshOptions) -> Result<Eigensystem>
where
    O: LinearOperator + ?Sized,
{
    let (values, vectors) = hermitian_eigenpairs_all(operator)?;
    let indices = match options.target {
        SpectrumTarget::Shift(shift) => {
            let mut indices: Vec<_> = (0..values.len()).collect();
            indices.sort_by(|&left, &right| {
                (values[left] - shift)
                    .abs()
                    .total_cmp(&(values[right] - shift).abs())
                    .then_with(|| left.cmp(&right))
            });
            indices.truncate(options.eigenpairs);
            indices
        }
        _ => select_indices(&values, options.target, options.eigenpairs),
    };
    let eigenvectors: Vec<_> = indices
        .iter()
        .map(|&index| vectors[index].clone())
        .collect();
    // Use the selected dense eigenvectors to refine each value with its
    // Rayleigh quotient. This shares the operator applications already needed
    // for residuals and avoids backend/platform rounding differences at tight
    // compatibility tolerances.
    let refined = eigenvectors
        .iter()
        .map(|vector| rayleigh_value_and_residual(operator, vector))
        .collect::<Result<Vec<_>>>()?;
    let eigenvalues = refined.iter().map(|&(value, _)| value).collect();
    let residuals = refined.iter().map(|&(_, residual)| residual).collect();
    Ok(Eigensystem {
        eigenvalues,
        eigenvectors,
        residuals,
        iterations: 1,
        reorthogonalization_passes: 0,
        conditional_second_passes: 0,
        converged: true,
    })
}

fn expanding_lanczos_eigsh<O>(
    operator: &O,
    options: &EigshOptions,
    initial_subspace: Option<&[Vec<Complex64>]>,
) -> Result<Eigensystem>
where
    O: LinearOperator + ?Sized,
{
    if options.krylov_dimension.is_none()
        && initial_subspace.is_none_or(|vectors| vectors.len() <= 1)
    {
        let initial = initial_subspace
            .and_then(|vectors| vectors.first())
            .map(Vec::as_slice);
        let dimension = operator.shape().0;
        let mut subspace_dimension = (8 * options.eigenpairs + 64)
            .max(256)
            .min(dimension)
            .min(options.max_iterations);
        let mut spent_iterations = 0usize;
        loop {
            let mut attempt = options.clone();
            attempt.krylov_dimension = Some(subspace_dimension);
            attempt.max_iterations = subspace_dimension;
            match lanczos_eigsh(operator, &attempt, initial) {
                Ok(mut result) => {
                    result.iterations += spent_iterations;
                    return Ok(result);
                }
                Err(QmbedError::NonConvergence {
                    iterations,
                    residual,
                }) => {
                    spent_iterations = spent_iterations.saturating_add(iterations);
                    if subspace_dimension == dimension && dimension <= FULL_KRYLOV_DENSE_FALLBACK {
                        let mut result = dense_eigsh(operator, options)?;
                        result.iterations = result.iterations.saturating_add(spent_iterations);
                        return Ok(result);
                    }
                    let remaining = options.max_iterations.saturating_sub(spent_iterations);
                    let next_dimension = subspace_dimension
                        .saturating_mul(2)
                        .min(dimension)
                        .min(remaining);
                    if next_dimension <= subspace_dimension {
                        return Err(QmbedError::NonConvergence {
                            iterations: spent_iterations,
                            residual,
                        });
                    }
                    subspace_dimension = next_dimension;
                }
                Err(error) => return Err(error),
            }
        }
    }
    // A large enough unrestarted window is still the cheapest path: its
    // three-term projection is tridiagonal and avoids the dense Rayleigh--Ritz
    // projection required after a thick restart.  Small windows and genuine
    // multi-vector starts use the restarted backend below.
    if options
        .krylov_dimension
        .is_some_and(|dimension| dimension >= TRIDIAGONAL_LANCZOS_WINDOW)
        && initial_subspace.is_none_or(|vectors| vectors.len() <= 1)
    {
        let initial = initial_subspace
            .and_then(|vectors| vectors.first())
            .map(Vec::as_slice);
        return lanczos_eigsh(operator, options, initial);
    }
    match thick_restarted_eigsh(operator, options, initial_subspace) {
        Err(QmbedError::NonConvergence {
            iterations,
            residual: _,
        }) if operator.shape().0 <= FULL_KRYLOV_DENSE_FALLBACK
            && options.max_iterations >= operator.shape().0
            && options
                .krylov_dimension
                .is_none_or(|dimension| dimension == operator.shape().0) =>
        {
            let mut result = dense_eigsh(operator, options)?;
            result.iterations = iterations.saturating_add(result.iterations);
            Ok(result)
        }
        result => result,
    }
}

/// Selected Hermitian eigenpairs.
///
/// Small problems use a dense real-symmetric decomposition. Larger problems
/// use a matrix-free, fully reorthogonalized Lanczos backend; shift targets
/// apply a restarted GMRES inverse without materializing the operator.
pub fn eigsh<O>(operator: &O, options: EigshOptions) -> Result<Eigensystem>
where
    O: LinearOperator + ?Sized,
{
    let shape = operator.shape();
    if shape.0 != shape.1 {
        return Err(QmbedError::DimensionMismatch(
            "eigsh requires a square operator".into(),
        ));
    }
    options.validate(shape.0)?;
    if !use_dense_eigsh(shape.0, &options) {
        return expanding_lanczos_eigsh(operator, &options, None);
    }
    dense_eigsh(operator, &options)
}

/// Compute selected eigenpairs from one explicit initial vector.
///
/// The vector must match the square operator dimension and have nonzero finite
/// norm. Small problems may still choose the exact dense route.
pub fn eigsh_with_initial<O>(
    operator: &O,
    options: EigshOptions,
    initial: &[Complex64],
) -> Result<Eigensystem>
where
    O: LinearOperator + ?Sized,
{
    let shape = operator.shape();
    if shape.0 != shape.1 {
        return Err(QmbedError::DimensionMismatch(
            "eigsh requires a square operator".into(),
        ));
    }
    options.validate(shape.0)?;
    validate_eigsh_initial(initial, shape.0)?;
    if use_dense_eigsh(shape.0, &options) {
        return eigsh(operator, options);
    }
    let initial_subspace = vec![initial.to_vec()];
    expanding_lanczos_eigsh(operator, &options, Some(&initial_subspace))
}

/// Solve from a complete user-supplied initial subspace.
///
/// Related parameter points should pass all previously converged target
/// vectors here.  The thick-restart backend preserves the subspace as a
/// subspace, so rotations inside a degenerate multiplet do not discard useful
/// information.
pub fn eigsh_with_initial_subspace<O>(
    operator: &O,
    options: EigshOptions,
    initial_subspace: &[Vec<Complex64>],
) -> Result<Eigensystem>
where
    O: LinearOperator + ?Sized,
{
    let shape = operator.shape();
    if shape.0 != shape.1 {
        return Err(QmbedError::DimensionMismatch(
            "eigsh requires a square operator".into(),
        ));
    }
    options.validate(shape.0)?;
    if initial_subspace.is_empty() {
        return eigsh(operator, options);
    }
    for vector in initial_subspace {
        validate_eigsh_initial(vector, shape.0)?;
    }
    if use_dense_eigsh(shape.0, &options) {
        return eigsh(operator, options);
    }
    expanding_lanczos_eigsh(operator, &options, Some(initial_subspace))
}

/// Solve one member of a related operator family and update its reusable
/// invariant-subspace workspace.
///
/// A compatible workspace retains all previously converged vectors. Small
/// restart windows preserve them as a thick subspace; the cheaper large-window
/// tridiagonal route forms a balanced warm start without replacing the generic
/// Lanczos algorithm.
pub fn eigsh_with_workspace<O>(
    operator: &O,
    options: EigshOptions,
    workspace: &mut EigshWorkspace,
) -> Result<Eigensystem>
where
    O: LinearOperator + ?Sized,
{
    let dimension = operator.shape().0;
    let result = if workspace.dimension == dimension && !workspace.initial_subspace.is_empty() {
        if options
            .krylov_dimension
            .is_none_or(|size| size >= TRIDIAGONAL_LANCZOS_WINDOW)
            && workspace.initial_subspace.len() > 1
        {
            let initial = balanced_subspace_start(&workspace.initial_subspace)?;
            eigsh_with_initial(operator, options, &initial)?
        } else {
            eigsh_with_initial_subspace(operator, options, &workspace.initial_subspace)?
        }
    } else {
        eigsh(operator, options)?
    };
    workspace.update(dimension, &result);
    Ok(result)
}

/// Return only selected eigenvalues while using the same convergence checks as [`eigsh`].
pub fn eigsh_values<O>(operator: &O, options: EigshOptions) -> Result<Vec<f64>>
where
    O: LinearOperator + ?Sized,
{
    Ok(eigsh(operator, options)?.eigenvalues)
}

/// Time grid and adaptive Krylov controls for state evolution.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct EvolutionOptions {
    /// Finite nondecreasing output times measured from the initial state.
    pub times: Vec<f64>,
    /// Maximum dimension of each Krylov projection.
    pub krylov_dimension: usize,
    /// Local error tolerance.
    pub tolerance: f64,
    /// Maximum accepted and rejected trial intervals.
    pub max_substeps: usize,
    /// Interpret the generator as `H` and apply `exp(-i H t)` when true.
    pub hamiltonian: bool,
}

impl EvolutionOptions {
    /// Construct Hamiltonian evolution controls on the supplied output grid.
    pub fn new(times: impl Into<Vec<f64>>) -> Self {
        Self {
            times: times.into(),
            krylov_dimension: 30,
            tolerance: 1.0e-10,
            max_substeps: 10_000,
            hamiltonian: true,
        }
    }

    /// Set the maximum dimension of each Krylov projection.
    #[must_use]
    pub fn with_krylov_dimension(mut self, dimension: usize) -> Self {
        self.krylov_dimension = dimension;
        self
    }

    /// Set the local error tolerance.
    #[must_use]
    pub fn with_tolerance(mut self, tolerance: f64) -> Self {
        self.tolerance = tolerance;
        self
    }

    /// Set the maximum accepted and rejected trial intervals.
    #[must_use]
    pub fn with_max_substeps(mut self, max_substeps: usize) -> Self {
        self.max_substeps = max_substeps;
        self
    }

    /// Select Hamiltonian (`exp(-iHt)`) or direct-generator evolution.
    #[must_use]
    pub fn with_hamiltonian(mut self, hamiltonian: bool) -> Self {
        self.hamiltonian = hamiltonian;
        self
    }

    fn validate(&self) -> Result<()> {
        if self.times.is_empty()
            || self.times.iter().any(|time| !time.is_finite())
            || self.times.windows(2).any(|pair| pair[0] > pair[1])
        {
            return Err(QmbedError::InvalidOptions(
                "times must be a nonempty finite nondecreasing grid".into(),
            ));
        }
        if self.krylov_dimension == 0
            || !self.tolerance.is_finite()
            || self.tolerance <= 0.0
            || self.max_substeps == 0
        {
            return Err(QmbedError::InvalidOptions(
                "evolution controls must be positive".into(),
            ));
        }
        Ok(())
    }
}

/// State vectors sampled at the requested physical times.
#[derive(Clone, Debug)]
pub struct StateTrajectory {
    /// Output times copied from the validated request.
    pub times: Vec<f64>,
    /// One state vector per output time.
    pub states: Vec<Vec<Complex64>>,
}

/// Algorithmic work performed by Hermitian Krylov time evolution.
///
/// Wall time depends on the host and sparse backend. These counters expose the
/// portable cost drivers needed by verification and performance-regression
/// suites without changing the existing [`evolve`] return contract.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct EvolutionDiagnostics {
    pub lanczos_projections: usize,
    pub matrix_vector_products: usize,
    pub real_lanczos_projections: usize,
    pub real_matrix_vector_products: usize,
    pub accepted_substeps: usize,
    pub rejected_trial_intervals: usize,
    pub maximum_estimated_error: f64,
}

/// Column-oriented batch trajectory: `states[time_index][column_index]`.
#[derive(Clone, Debug)]
pub struct StateBatchTrajectory {
    pub times: Vec<f64>,
    pub states: Vec<Vec<Vec<Complex64>>>,
}

/// Dimension and breakdown tolerance for a Lanczos decomposition.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct LanczosOptions {
    /// Maximum number of orthonormal Krylov vectors.
    pub krylov_dimension: usize,
    /// Threshold for norm breakdown and Hermiticity checks.
    pub tolerance: f64,
}

impl LanczosOptions {
    /// Construct a decomposition request with the default breakdown tolerance.
    pub const fn new(krylov_dimension: usize) -> Self {
        Self {
            krylov_dimension,
            tolerance: 1.0e-12,
        }
    }

    /// Set the norm-breakdown and Hermiticity tolerance.
    #[must_use]
    pub const fn with_tolerance(mut self, tolerance: f64) -> Self {
        self.tolerance = tolerance;
        self
    }

    fn validate(&self) -> Result<()> {
        if self.krylov_dimension == 0 || !self.tolerance.is_finite() || self.tolerance <= 0.0 {
            return Err(QmbedError::InvalidOptions(
                "Lanczos dimension and tolerance must be positive".into(),
            ));
        }
        Ok(())
    }
}

/// One streamed vector and neighboring coefficients of a Lanczos recurrence.
#[derive(Clone, Debug)]
pub struct LanczosVector {
    /// Zero-based position in the Krylov basis.
    pub index: usize,
    /// Normalized Krylov vector.
    pub vector: Vec<Complex64>,
    /// Diagonal tridiagonal coefficient.
    pub diagonal: f64,
    /// Off-diagonal coefficient leading to the next vector.
    pub next_off_diagonal: f64,
}

/// Complete orthonormal basis and real symmetric tridiagonal projection.
#[derive(Clone, Debug)]
pub struct LanczosDecomposition {
    /// Norm removed from the caller's initial vector.
    pub initial_norm: f64,
    /// Orthonormal Krylov vectors.
    pub basis: Vec<Vec<Complex64>>,
    /// Diagonal of the projected tridiagonal matrix.
    pub diagonal: Vec<f64>,
    /// First off-diagonal of the projected tridiagonal matrix.
    pub off_diagonal: Vec<f64>,
}

#[derive(Clone, Debug)]
pub struct LanczosRitzDecomposition {
    pub decomposition: LanczosDecomposition,
    pub eigenvalues: Vec<f64>,
    /// Column-oriented eigenvectors of the real symmetric tridiagonal matrix.
    pub eigenvectors: Vec<Vec<f64>>,
}

impl LanczosRitzDecomposition {
    /// Lift coefficients in Krylov-basis order back to the full Hilbert space.
    pub fn linear_combination(&self, coefficients: &[Complex64]) -> Result<Vec<Complex64>> {
        linear_combination_qt(&self.decomposition.basis, coefficients)
    }

    /// Apply `exp(coefficient * T)` to the first Krylov vector and lift it
    /// through the stored Lanczos basis.
    pub fn exponential_action(&self, coefficient: Complex64) -> Result<Vec<Complex64>> {
        if !coefficient.re.is_finite() || !coefficient.im.is_finite() {
            return Err(QmbedError::InvalidOptions(
                "Lanczos exponential coefficient must be finite".into(),
            ));
        }
        let dimension = self.eigenvalues.len();
        if dimension == 0
            || self.eigenvectors.len() != dimension
            || self
                .eigenvectors
                .iter()
                .any(|vector| vector.len() != dimension)
        {
            return Err(QmbedError::InternalState(
                "Lanczos Ritz eigensystem is inconsistent".into(),
            ));
        }
        let mut coefficients = vec![Complex64::new(0.0, 0.0); dimension];
        for (eigenvalue, eigenvector) in self.eigenvalues.iter().zip(&self.eigenvectors) {
            let weight = Complex64::new(eigenvector[0], 0.0)
                * (coefficient * *eigenvalue).exp()
                * self.decomposition.initial_norm;
            for (value, component) in coefficients.iter_mut().zip(eigenvector) {
                *value += weight * *component;
            }
        }
        self.linear_combination(&coefficients)
    }
}

/// Iterator that streams a fully reorthogonalized Lanczos recurrence.
pub struct LanczosIter<'a, O>
where
    O: LinearOperator + ?Sized,
{
    operator: &'a O,
    options: LanczosOptions,
    index: usize,
    previous: Option<Vec<Complex64>>,
    current: Option<Vec<Complex64>>,
    previous_beta: f64,
    history: Vec<Vec<Complex64>>,
    failed: bool,
}

impl<O> Iterator for LanczosIter<'_, O>
where
    O: LinearOperator + ?Sized,
{
    type Item = Result<LanczosVector>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.failed || self.index >= self.options.krylov_dimension {
            return None;
        }
        let current = self.current.take()?;
        let mut applied = vec![Complex64::new(0.0, 0.0); current.len()];
        if let Err(error) = self.operator.apply(&current, &mut applied) {
            self.failed = true;
            return Some(Err(error));
        }
        let alpha = inner(&current, &applied);
        if alpha.im.abs() > self.options.tolerance.max(1.0e-10) {
            self.failed = true;
            return Some(Err(QmbedError::NonHermitian));
        }
        for (value, basis_value) in applied.iter_mut().zip(&current) {
            *value -= alpha.re * *basis_value;
        }
        if let Some(previous) = &self.previous {
            for (value, basis_value) in applied.iter_mut().zip(previous) {
                *value -= self.previous_beta * *basis_value;
            }
        }
        let norm_before_reorthogonalization = vector_norm(&applied);
        for basis_vector in self.history.iter().chain(std::iter::once(&current)) {
            let correction = inner(basis_vector, &applied);
            for (value, basis_value) in applied.iter_mut().zip(basis_vector) {
                *value -= correction * *basis_value;
            }
        }
        let norm_after_first_pass = vector_norm(&applied);
        if norm_before_reorthogonalization > f64::EPSILON
            && norm_after_first_pass
                <= DGKS_REORTHOGONALIZATION_THRESHOLD * norm_before_reorthogonalization
        {
            for basis_vector in self.history.iter().chain(std::iter::once(&current)) {
                let correction = inner(basis_vector, &applied);
                for (value, basis_value) in applied.iter_mut().zip(basis_vector) {
                    *value -= correction * *basis_value;
                }
            }
        }
        let beta = vector_norm(&applied);
        let output = LanczosVector {
            index: self.index,
            vector: current.clone(),
            diagonal: alpha.re,
            next_off_diagonal: beta,
        };
        self.index += 1;
        self.history.push(current.clone());
        if self.index < self.options.krylov_dimension && beta > self.options.tolerance {
            for value in &mut applied {
                *value /= beta;
            }
            self.previous = Some(current);
            self.current = Some(applied);
            self.previous_beta = beta;
        } else {
            self.current = None;
        }
        Some(Ok(output))
    }
}

/// Start a validated streaming Lanczos decomposition.
pub fn lanczos_iter<'a, O>(
    operator: &'a O,
    initial: &'a [Complex64],
    options: LanczosOptions,
) -> Result<LanczosIter<'a, O>>
where
    O: LinearOperator + ?Sized,
{
    options.validate()?;
    let shape = operator.shape();
    if shape.0 != shape.1 || initial.len() != shape.0 {
        return Err(QmbedError::DimensionMismatch(
            "Lanczos operator and initial vector do not match".into(),
        ));
    }
    let mut current = initial.to_vec();
    normalize(&mut current)?;
    let capacity = options.krylov_dimension;
    Ok(LanczosIter {
        operator,
        options,
        index: 0,
        previous: None,
        current: Some(current),
        previous_beta: 0.0,
        history: Vec::with_capacity(capacity),
        failed: false,
    })
}

/// Materialize the full Lanczos basis and tridiagonal coefficients.
pub fn lanczos_full<O>(
    operator: &O,
    initial: &[Complex64],
    options: LanczosOptions,
) -> Result<LanczosDecomposition>
where
    O: LinearOperator + ?Sized,
{
    options.validate()?;
    let initial_norm = vector_norm(initial);
    let vectors: Vec<_> = lanczos_iter(operator, initial, options)?.collect::<Result<_>>()?;
    let off_diagonal = vectors
        .iter()
        .take(vectors.len().saturating_sub(1))
        .map(|vector| vector.next_off_diagonal)
        .collect();
    Ok(LanczosDecomposition {
        initial_norm,
        basis: vectors.iter().map(|vector| vector.vector.clone()).collect(),
        diagonal: vectors.iter().map(|vector| vector.diagonal).collect(),
        off_diagonal,
    })
}

/// Full Lanczos basis plus the sorted eigensystem of its tridiagonal
/// projection. This is the reusable native object behind compatibility
/// interfaces which expose both Ritz data and later reconstruction.
pub fn lanczos_ritz<O>(
    operator: &O,
    initial: &[Complex64],
    options: LanczosOptions,
) -> Result<LanczosRitzDecomposition>
where
    O: LinearOperator + ?Sized,
{
    let decomposition = lanczos_full(operator, initial, options)?;
    let dimension = decomposition.diagonal.len();
    let mut tridiagonal = DMatrix::<f64>::zeros(dimension, dimension);
    for index in 0..dimension {
        tridiagonal[(index, index)] = decomposition.diagonal[index];
        if index + 1 < dimension {
            tridiagonal[(index, index + 1)] = decomposition.off_diagonal[index];
            tridiagonal[(index + 1, index)] = decomposition.off_diagonal[index];
        }
    }
    let eigensystem = SymmetricEigen::new(tridiagonal);
    let mut order = (0..dimension).collect::<Vec<_>>();
    order.sort_by(|left, right| {
        eigensystem.eigenvalues[*left].total_cmp(&eigensystem.eigenvalues[*right])
    });
    let eigenvalues = order
        .iter()
        .map(|index| eigensystem.eigenvalues[*index])
        .collect();
    let eigenvectors = order
        .iter()
        .map(|column| {
            (0..dimension)
                .map(|row| eigensystem.eigenvectors[(row, *column)])
                .collect()
        })
        .collect();
    Ok(LanczosRitzDecomposition {
        decomposition,
        eigenvalues,
        eigenvectors,
    })
}

struct LanczosProjection {
    initial_norm: f64,
    basis: ProjectedBasis,
    residual_beta: f64,
    eigenvalues: Vec<f64>,
    eigenvectors: DMatrix<f64>,
}

enum ProjectedBasis {
    Real(Vec<Vec<f64>>),
    Complex(Vec<Vec<Complex64>>),
}

impl ProjectedBasis {
    fn len(&self) -> usize {
        match self {
            Self::Real(basis) => basis.len(),
            Self::Complex(basis) => basis.len(),
        }
    }

    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

pub(crate) fn lanczos_spectral_measure(
    operator: &(impl LinearOperator + ?Sized),
    source: &[Complex64],
    krylov_dimension: usize,
) -> Result<(Vec<f64>, Vec<f64>)> {
    let shape = operator.shape();
    if shape.0 != shape.1 || source.len() != shape.0 {
        return Err(QmbedError::DimensionMismatch(
            "spectral Lanczos requires a square operator matching the source".into(),
        ));
    }
    if krylov_dimension == 0 {
        return Err(QmbedError::InvalidOptions(
            "spectral Krylov dimension must be positive".into(),
        ));
    }
    let projection = lanczos_projection(operator, source, krylov_dimension)?;
    let weights = (0..projection.eigenvalues.len())
        .map(|index| projection.initial_norm.powi(2) * projection.eigenvectors[(0, index)].powi(2))
        .collect();
    Ok((projection.eigenvalues, weights))
}

fn lanczos_projection(
    operator: &(impl LinearOperator + ?Sized),
    initial: &[Complex64],
    dimension: usize,
) -> Result<LanczosProjection> {
    let initial_norm = vector_norm(initial);
    if initial_norm <= f64::EPSILON {
        return Ok(LanczosProjection {
            initial_norm,
            basis: ProjectedBasis::Complex(Vec::new()),
            residual_beta: 0.0,
            eigenvalues: Vec::new(),
            eigenvectors: DMatrix::zeros(0, 0),
        });
    }
    if operator.is_real() && initial.iter().all(|value| value.im == 0.0) {
        return lanczos_projection_real(operator, initial, dimension);
    }
    let krylov_dimension = dimension.min(initial.len()).max(1);
    let mut first = initial.to_vec();
    for value in &mut first {
        *value /= initial_norm;
    }
    let mut basis = Vec::with_capacity(krylov_dimension);
    basis.push(first);
    let mut alphas = Vec::with_capacity(krylov_dimension);
    let mut betas = Vec::with_capacity(krylov_dimension.saturating_sub(1));
    let mut residual_beta = 0.0;
    let mut applied = vec![Complex64::new(0.0, 0.0); initial.len()];

    for iteration in 0..krylov_dimension {
        operator.apply(&basis[iteration], &mut applied)?;
        let alpha = inner(&basis[iteration], &applied);
        if alpha.im.abs() > 1.0e-10 {
            return Err(QmbedError::NonHermitian);
        }
        alphas.push(alpha.re);
        for (value, basis_value) in applied.iter_mut().zip(&basis[iteration]) {
            *value -= alpha.re * *basis_value;
        }
        if iteration > 0 {
            for (value, previous) in applied.iter_mut().zip(&basis[iteration - 1]) {
                *value -= betas[iteration - 1] * *previous;
            }
        }
        // Exponential actions use the Hermitian three-term recurrence. Unlike
        // the multi-eigenpair solver, they do not need global Ritz-vector
        // orthogonality, so avoiding O(m^2 n) reorthogonalization is essential
        // for the 100-step paper workflows.
        let beta = vector_norm(&applied);
        residual_beta = beta;
        if iteration + 1 == krylov_dimension || beta <= 1.0e-14 {
            break;
        }
        betas.push(beta);
        for value in &mut applied {
            *value /= beta;
        }
        basis.push(applied.clone());
    }

    let size = basis.len();
    let mut tridiagonal = DMatrix::<f64>::zeros(size, size);
    for index in 0..size {
        tridiagonal[(index, index)] = alphas[index];
        if index + 1 < size {
            tridiagonal[(index, index + 1)] = betas[index];
            tridiagonal[(index + 1, index)] = betas[index];
        }
    }
    let decomposition = SymmetricEigen::new(tridiagonal);
    Ok(LanczosProjection {
        initial_norm,
        basis: ProjectedBasis::Complex(basis),
        residual_beta,
        eigenvalues: decomposition.eigenvalues.as_slice().to_vec(),
        eigenvectors: decomposition.eigenvectors,
    })
}

fn lanczos_projection_real(
    operator: &(impl LinearOperator + ?Sized),
    initial: &[Complex64],
    dimension: usize,
) -> Result<LanczosProjection> {
    let initial_norm = initial
        .iter()
        .map(|value| value.re * value.re)
        .sum::<f64>()
        .sqrt();
    let krylov_dimension = dimension.min(initial.len()).max(1);
    let mut basis = Vec::with_capacity(krylov_dimension);
    basis.push(
        initial
            .iter()
            .map(|value| value.re / initial_norm)
            .collect::<Vec<_>>(),
    );
    let mut alphas = Vec::with_capacity(krylov_dimension);
    let mut betas = Vec::with_capacity(krylov_dimension.saturating_sub(1));
    let mut residual_beta = 0.0;
    let mut applied = vec![0.0; initial.len()];

    for iteration in 0..krylov_dimension {
        operator.apply_real(&basis[iteration], &mut applied)?;
        let alpha = real_inner(&basis[iteration], &applied);
        alphas.push(alpha);
        for (value, basis_value) in applied.iter_mut().zip(&basis[iteration]) {
            *value -= alpha * *basis_value;
        }
        if iteration > 0 {
            let previous_beta = betas[iteration - 1];
            for (value, basis_value) in applied.iter_mut().zip(&basis[iteration - 1]) {
                *value -= previous_beta * *basis_value;
            }
        }
        let beta = real_vector_norm(&applied);
        residual_beta = beta;
        if iteration + 1 == krylov_dimension || beta <= 1.0e-14 {
            break;
        }
        betas.push(beta);
        for value in &mut applied {
            *value /= beta;
        }
        basis.push(applied.clone());
    }

    let size = basis.len();
    let mut tridiagonal = DMatrix::<f64>::zeros(size, size);
    for index in 0..size {
        tridiagonal[(index, index)] = alphas[index];
        if index + 1 < size {
            tridiagonal[(index, index + 1)] = betas[index];
            tridiagonal[(index + 1, index)] = betas[index];
        }
    }
    let decomposition = SymmetricEigen::new(tridiagonal);
    Ok(LanczosProjection {
        initial_norm,
        basis: ProjectedBasis::Real(basis),
        residual_beta,
        eigenvalues: decomposition.eigenvalues.as_slice().to_vec(),
        eigenvectors: decomposition.eigenvectors,
    })
}

fn projected_exponential_coefficients(
    projection: &LanczosProjection,
    interval: f64,
    hamiltonian: bool,
) -> Vec<Complex64> {
    let size = projection.basis.len();
    let mut coefficients = vec![Complex64::new(0.0, 0.0); size];
    for eigen_index in 0..size {
        let exponent = if hamiltonian {
            Complex64::new(0.0, -interval * projection.eigenvalues[eigen_index]).exp()
        } else {
            Complex64::new(interval * projection.eigenvalues[eigen_index], 0.0).exp()
        };
        let weight = projection.initial_norm * projection.eigenvectors[(0, eigen_index)] * exponent;
        for (basis_index, coefficient) in coefficients.iter_mut().enumerate() {
            *coefficient += projection.eigenvectors[(basis_index, eigen_index)] * weight;
        }
    }
    coefficients
}

fn projected_exponential_action_from_coefficients(
    projection: &LanczosProjection,
    coefficients: &[Complex64],
    hamiltonian: bool,
    ambient_dimension: usize,
) -> Vec<Complex64> {
    if projection.basis.is_empty() {
        return vec![Complex64::new(0.0, 0.0); ambient_dimension];
    }
    let mut output = vec![Complex64::new(0.0, 0.0); ambient_dimension];
    match &projection.basis {
        ProjectedBasis::Real(basis) => {
            for (coefficient, vector) in coefficients.iter().zip(basis) {
                for (value, basis_value) in output.iter_mut().zip(vector) {
                    value.re += coefficient.re * *basis_value;
                    value.im += coefficient.im * *basis_value;
                }
            }
        }
        ProjectedBasis::Complex(basis) => {
            for (coefficient, vector) in coefficients.iter().zip(basis) {
                for (value, basis_value) in output.iter_mut().zip(vector) {
                    *value += *coefficient * *basis_value;
                }
            }
        }
    }
    if hamiltonian {
        let output_norm = vector_norm(&output);
        if output_norm > f64::EPSILON && output_norm.is_finite() {
            let scale = projection.initial_norm / output_norm;
            for value in &mut output {
                *value *= scale;
            }
        }
    }
    output
}

fn projected_exponential_action(
    projection: &LanczosProjection,
    interval: f64,
    hamiltonian: bool,
    ambient_dimension: usize,
) -> Vec<Complex64> {
    let coefficients = projected_exponential_coefficients(projection, interval, hamiltonian);
    projected_exponential_action_from_coefficients(
        projection,
        &coefficients,
        hamiltonian,
        ambient_dimension,
    )
}

fn projected_residual_amplitude(projection: &LanczosProjection, interval: f64) -> f64 {
    let Some(last_index) = projection.basis.len().checked_sub(1) else {
        return 0.0;
    };
    let mut amplitude = Complex64::new(0.0, 0.0);
    for eigen_index in 0..projection.eigenvalues.len() {
        let phase = Complex64::new(0.0, -interval * projection.eigenvalues[eigen_index]).exp();
        amplitude += projection.initial_norm
            * projection.eigenvectors[(0, eigen_index)]
            * projection.eigenvectors[(last_index, eigen_index)]
            * phase;
    }
    amplitude.norm()
}

/// A posteriori Hermitian Krylov error estimate.
///
/// For `V_m exp(-i t T_m) e_1`, the residual norm is
/// `beta_m |e_m^T exp(-i t T_m) e_1|`. Duhamel's formula bounds the state
/// error by its time integral because the exact Hermitian propagator is
/// unitary. The integral is evaluated entirely in the small projected space,
/// so rejected trial intervals never materialize ambient-dimension vectors.
fn projected_exponential_error_bound(
    projection: &LanczosProjection,
    interval: f64,
    ambient_dimension: usize,
) -> f64 {
    let duration = interval.abs();
    if duration <= f64::EPSILON
        || projection.basis.is_empty()
        || projection.basis.len() == ambient_dimension
        || projection.residual_beta <= 1.0e-14
    {
        return 0.0;
    }

    let (minimum, maximum) = projection.eigenvalues.iter().copied().fold(
        (f64::INFINITY, f64::NEG_INFINITY),
        |(minimum, maximum), value| (minimum.min(value), maximum.max(value)),
    );
    let phase_span = duration * (maximum - minimum).max(0.0);
    let oscillations = (phase_span / std::f64::consts::PI).ceil().min(60.0) as usize;
    let mut intervals = (16 + 4 * oscillations).min(256);
    if intervals % 2 != 0 {
        intervals += 1;
    }
    let direction = interval.signum();
    let step = duration / intervals as f64;
    let mut integral = 0.0;
    for index in 0..=intervals {
        let weight = if index == 0 || index == intervals {
            1.0
        } else if index % 2 == 0 {
            2.0
        } else {
            4.0
        };
        integral +=
            weight * projected_residual_amplitude(projection, direction * step * index as f64);
    }
    // Simpson quadrature is highly accurate for the smooth projected
    // residual. A modest safety factor protects the accept/reject boundary
    // from quadrature and finite-precision Lanczos error.
    1.25 * projection.residual_beta * step * integral / 3.0
}

fn evolve_hermitian_adaptive_grid(
    operator: &(impl LinearOperator + ?Sized),
    initial: &[Complex64],
    options: &EvolutionOptions,
) -> Result<(StateTrajectory, EvolutionDiagnostics)> {
    let mut states = Vec::with_capacity(options.times.len());
    let mut current_state = initial.to_vec();
    let mut current_time = 0.0;
    let mut output_index = 0;
    let mut substeps = 0;
    let mut diagnostics = EvolutionDiagnostics::default();
    let local_tolerance = options.tolerance / options.times.len().max(1) as f64;

    while output_index < options.times.len() {
        let target_time = options.times[output_index];
        if (target_time - current_time).abs() <= 16.0 * f64::EPSILON * target_time.abs().max(1.0) {
            states.push(current_state.clone());
            current_time = target_time;
            output_index += 1;
            continue;
        }
        if substeps >= options.max_substeps {
            return Err(QmbedError::NonConvergence {
                iterations: substeps,
                residual: (target_time - current_time).abs(),
            });
        }

        let projection = lanczos_projection(operator, &current_state, options.krylov_dimension)?;
        diagnostics.lanczos_projections += 1;
        diagnostics.matrix_vector_products += projection.basis.len();
        if matches!(&projection.basis, ProjectedBasis::Real(_)) {
            diagnostics.real_lanczos_projections += 1;
            diagnostics.real_matrix_vector_products += projection.basis.len();
        }
        let scale = vector_norm(&current_state).max(1.0);
        let threshold = local_tolerance * scale;

        let mut accepted = Vec::new();
        for &time in &options.times[output_index..] {
            let interval = time - current_time;
            let error =
                projected_exponential_error_bound(&projection, interval, current_state.len());
            diagnostics.maximum_estimated_error = diagnostics.maximum_estimated_error.max(error);
            if error > threshold {
                diagnostics.rejected_trial_intervals += 1;
                break;
            }
            let candidate =
                projected_exponential_action(&projection, interval, true, current_state.len());
            accepted.push((time, candidate));
        }
        if !accepted.is_empty() {
            for (_, candidate) in &accepted {
                states.push(candidate.clone());
            }
            let (time, state) = accepted.pop().expect("accepted is nonempty");
            current_time = time;
            current_state = state;
            output_index += accepted.len() + 1;
            substeps += 1;
            diagnostics.accepted_substeps += 1;
            continue;
        }

        let target_interval = target_time - current_time;
        let direction = target_interval.signum();
        let minimum_interval = 16.0 * f64::EPSILON * target_time.abs().max(1.0);
        let mut rejected_magnitude = target_interval.abs();
        let mut accepted_magnitude = rejected_magnitude;
        loop {
            let interval = direction * accepted_magnitude;
            let error =
                projected_exponential_error_bound(&projection, interval, current_state.len());
            diagnostics.maximum_estimated_error = diagnostics.maximum_estimated_error.max(error);
            if error <= threshold || accepted_magnitude <= minimum_interval {
                break;
            }
            diagnostics.rejected_trial_intervals += 1;
            rejected_magnitude = accepted_magnitude;
            accepted_magnitude *= 0.5;
        }

        // Halving finds a safe bracket but can undershoot the largest stable
        // step by almost a factor of two. Refine only in projected space so a
        // projection advances as far as its error budget permits.
        if accepted_magnitude > minimum_interval && rejected_magnitude > accepted_magnitude {
            for _ in 0..8 {
                let trial_magnitude = 0.5 * (accepted_magnitude + rejected_magnitude);
                let error = projected_exponential_error_bound(
                    &projection,
                    direction * trial_magnitude,
                    current_state.len(),
                );
                diagnostics.maximum_estimated_error =
                    diagnostics.maximum_estimated_error.max(error);
                if error <= threshold {
                    accepted_magnitude = trial_magnitude;
                } else {
                    diagnostics.rejected_trial_intervals += 1;
                    rejected_magnitude = trial_magnitude;
                }
            }
        }

        let interval = direction * accepted_magnitude;
        current_time += interval;
        current_state =
            projected_exponential_action(&projection, interval, true, current_state.len());
        substeps += 1;
        diagnostics.accepted_substeps += 1;
    }
    Ok((
        StateTrajectory {
            times: options.times.clone(),
            states,
        },
        diagnostics,
    ))
}

pub(crate) fn expm_action(
    operator: &(impl LinearOperator + ?Sized),
    initial: &[Complex64],
    interval: f64,
    options: &EvolutionOptions,
) -> Result<Vec<Complex64>> {
    let shape = operator.shape();
    if shape.0 != shape.1 || initial.len() != shape.0 {
        return Err(QmbedError::DimensionMismatch(
            "evolution requires a square operator matching the state".into(),
        ));
    }
    if interval == 0.0 {
        return Ok(initial.to_vec());
    }
    if options.hamiltonian {
        let projection = lanczos_projection(operator, initial, options.krylov_dimension)?;
        return Ok(projected_exponential_action(
            &projection,
            interval,
            true,
            initial.len(),
        ));
    }
    let exponent = Complex64::new(interval, 0.0);
    expm_action_complex(
        operator,
        initial,
        exponent,
        options.krylov_dimension,
        options.tolerance,
        options.max_substeps,
    )
}

pub(crate) fn expm_action_complex(
    operator: &(impl LinearOperator + ?Sized),
    initial: &[Complex64],
    exponent: Complex64,
    krylov_dimension: usize,
    tolerance: f64,
    max_substeps: usize,
) -> Result<Vec<Complex64>> {
    ExpmActionPlan::new(
        operator,
        exponent,
        krylov_dimension,
        tolerance,
        max_substeps,
    )?
    .apply(operator, initial)
}

/// Time evolution on an arbitrary square stored or matrix-free operator.
pub fn evolve_with_diagnostics<O>(
    operator: &O,
    initial: &[Complex64],
    options: EvolutionOptions,
) -> Result<(StateTrajectory, EvolutionDiagnostics)>
where
    O: LinearOperator + ?Sized,
{
    options.validate()?;
    let shape = operator.shape();
    if shape.0 != shape.1 || initial.len() != shape.0 {
        return Err(QmbedError::DimensionMismatch(
            "evolution operator and initial state do not match".into(),
        ));
    }
    let mut states = Vec::with_capacity(options.times.len());
    if options.hamiltonian {
        return evolve_hermitian_adaptive_grid(operator, initial, &options);
    }
    let mut state = initial.to_vec();
    let mut previous_time = 0.0;
    for &time in &options.times {
        state = expm_action(operator, &state, time - previous_time, &options)?;
        states.push(state.clone());
        previous_time = time;
    }
    Ok((
        StateTrajectory {
            times: options.times,
            states,
        },
        EvolutionDiagnostics::default(),
    ))
}

/// Time evolution on an arbitrary square stored or matrix-free operator.
pub fn evolve<O>(
    operator: &O,
    initial: &[Complex64],
    options: EvolutionOptions,
) -> Result<StateTrajectory>
where
    O: LinearOperator + ?Sized,
{
    Ok(evolve_with_diagnostics(operator, initial, options)?.0)
}

/// Exponential action over a time grid. Hermitian Hamiltonians reuse each
/// residual-controlled Lanczos projection across the longest accepted prefix.
pub fn expm_multiply<O>(
    operator: &O,
    initial: &[Complex64],
    options: EvolutionOptions,
) -> Result<StateTrajectory>
where
    O: LinearOperator + ?Sized,
{
    evolve(operator, initial, options)
}

pub fn expm_lanczos<O>(
    operator: &O,
    initial: &[Complex64],
    time: f64,
    options: LanczosOptions,
) -> Result<Vec<Complex64>>
where
    O: LinearOperator + ?Sized,
{
    options.validate()?;
    if !time.is_finite() {
        return Err(QmbedError::InvalidOptions(
            "exponential time must be finite".into(),
        ));
    }
    let projection = lanczos_projection(operator, initial, options.krylov_dimension)?;
    Ok(projected_exponential_action(
        &projection,
        time,
        true,
        initial.len(),
    ))
}

#[derive(Clone, Debug)]
pub struct ThermalIteration {
    pub inverse_temperatures: Vec<f64>,
    pub log_partition: Vec<f64>,
    pub mean_energy: Vec<f64>,
    pub krylov_dimension: usize,
}

fn thermal_lanczos_iteration<O>(
    operator: &O,
    initial: &[Complex64],
    inverse_temperatures: &[f64],
    options: LanczosOptions,
) -> Result<ThermalIteration>
where
    O: LinearOperator + ?Sized,
{
    options.validate()?;
    if inverse_temperatures.is_empty()
        || inverse_temperatures
            .iter()
            .any(|beta| !beta.is_finite() || *beta < 0.0)
    {
        return Err(QmbedError::InvalidOptions(
            "inverse temperatures must be nonempty, finite, and nonnegative".into(),
        ));
    }
    let decomposition = lanczos_full(operator, initial, options)?;
    let size = decomposition.diagonal.len();
    let mut tridiagonal = DMatrix::<f64>::zeros(size, size);
    for index in 0..size {
        tridiagonal[(index, index)] = decomposition.diagonal[index];
        if index + 1 < size {
            tridiagonal[(index, index + 1)] = decomposition.off_diagonal[index];
            tridiagonal[(index + 1, index)] = decomposition.off_diagonal[index];
        }
    }
    let eigensystem = SymmetricEigen::new(tridiagonal);
    let weights: Vec<_> = (0..size)
        .map(|index| {
            decomposition.initial_norm.powi(2) * eigensystem.eigenvectors[(0, index)].powi(2)
        })
        .collect();
    let hilbert_dimension = operator.shape().0 as f64;
    let minimum_energy = eigensystem
        .eigenvalues
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);
    let mut log_partition = Vec::with_capacity(inverse_temperatures.len());
    let mut mean_energy = Vec::with_capacity(inverse_temperatures.len());
    for &beta in inverse_temperatures {
        let boltzmann: Vec<_> = eigensystem
            .eigenvalues
            .iter()
            .zip(&weights)
            .map(|(energy, weight)| weight * (-beta * (*energy - minimum_energy)).exp())
            .collect();
        let projected_partition = boltzmann.iter().sum::<f64>();
        if projected_partition <= f64::EPSILON {
            return Err(QmbedError::NonConvergence {
                iterations: size,
                residual: projected_partition,
            });
        }
        log_partition.push(
            (hilbert_dimension / decomposition.initial_norm.powi(2)).ln() - beta * minimum_energy
                + projected_partition.ln(),
        );
        mean_energy.push(
            boltzmann
                .iter()
                .zip(eigensystem.eigenvalues.iter())
                .map(|(weight, energy)| weight * energy)
                .sum::<f64>()
                / projected_partition,
        );
    }
    Ok(ThermalIteration {
        inverse_temperatures: inverse_temperatures.to_vec(),
        log_partition,
        mean_energy,
        krylov_dimension: size,
    })
}

/// One finite-temperature Lanczos random-vector iteration.
pub fn ftlm_static_iteration<O>(
    operator: &O,
    initial: &[Complex64],
    inverse_temperatures: &[f64],
    options: LanczosOptions,
) -> Result<ThermalIteration>
where
    O: LinearOperator + ?Sized,
{
    thermal_lanczos_iteration(operator, initial, inverse_temperatures, options)
}

/// Low-temperature Lanczos iteration using a ground-energy shifted Boltzmann
/// evaluation for numerical stability.
pub fn ltlm_static_iteration<O>(
    operator: &O,
    initial: &[Complex64],
    inverse_temperatures: &[f64],
    options: LanczosOptions,
) -> Result<ThermalIteration>
where
    O: LinearOperator + ?Sized,
{
    thermal_lanczos_iteration(operator, initial, inverse_temperatures, options)
}

#[derive(Clone, Debug)]
pub struct ThermalObservableIteration {
    pub inverse_temperatures: Vec<f64>,
    pub values: std::collections::HashMap<String, Vec<Complex64>>,
    pub identity: Vec<f64>,
}

/// Finite-temperature contraction applied to a precomputed Lanczos
/// eigensystem and observable projections.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ThermalLanczosMethod {
    Ftlm,
    Ltlm,
}

/// Observable data after projecting the language- or backend-owned action
/// into a Lanczos basis.
///
/// FTLM stores `m` overlaps `⟨q_i|A|q_0⟩`. LTLM stores the `m × m` matrix
/// `⟨q_j|A|q_i⟩` in input-major order (`i * m + j`). This is the narrow
/// interface needed by the thermal contraction and does not require Rust to
/// own the original observable.
#[derive(Clone, Debug)]
pub struct ProjectedThermalObservable {
    pub name: String,
    pub matrix_elements: Vec<Complex64>,
}

fn validate_thermal_ritz_data(
    eigenvalues: &[f64],
    eigenvectors: &[Vec<f64>],
    observables: &[ProjectedThermalObservable],
    inverse_temperatures: &[f64],
    method: ThermalLanczosMethod,
) -> Result<()> {
    let dimension = eigenvalues.len();
    if dimension == 0
        || eigenvalues.iter().any(|value| !value.is_finite())
        || eigenvectors.len() != dimension
        || eigenvectors.iter().any(|vector| {
            vector.len() != dimension || vector.iter().any(|value| !value.is_finite())
        })
    {
        return Err(QmbedError::DimensionMismatch(
            "thermal Lanczos contraction requires a finite square Ritz eigensystem".into(),
        ));
    }
    if observables.is_empty()
        || inverse_temperatures.is_empty()
        || inverse_temperatures.iter().any(|beta| !beta.is_finite())
    {
        return Err(QmbedError::InvalidOptions(
            "thermal observables and inverse temperatures must be nonempty and valid".into(),
        ));
    }
    let mut names = std::collections::HashSet::new();
    let expected = match method {
        ThermalLanczosMethod::Ftlm => dimension,
        ThermalLanczosMethod::Ltlm => dimension
            .checked_mul(dimension)
            .ok_or_else(|| QmbedError::DimensionMismatch("Lanczos dimension overflows".into()))?,
    };
    for observable in observables {
        if observable.name.is_empty()
            || !names.insert(observable.name.as_str())
            || observable.matrix_elements.len() != expected
            || observable
                .matrix_elements
                .iter()
                .any(|value| !value.re.is_finite() || !value.im.is_finite())
        {
            return Err(QmbedError::DimensionMismatch(
                "thermal observables require unique names and matching finite projections".into(),
            ));
        }
    }
    Ok(())
}

fn ftlm_projected_contraction(
    eigenvalues: &[f64],
    eigenvectors: &[Vec<f64>],
    observables: &[ProjectedThermalObservable],
    inverse_temperatures: &[f64],
) -> ThermalObservableIteration {
    let dimension = eigenvalues.len();
    let coefficients = inverse_temperatures
        .iter()
        .map(|beta| {
            (0..dimension)
                .map(|row| {
                    (0..dimension)
                        .map(|eigen| {
                            eigenvectors[eigen][row]
                                * eigenvectors[eigen][0]
                                * (-beta * eigenvalues[eigen]).exp()
                        })
                        .sum::<f64>()
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let identity = coefficients
        .iter()
        .map(|coefficient| coefficient[0])
        .collect();
    let values = observables
        .iter()
        .map(|observable| {
            let estimates = coefficients
                .iter()
                .map(|coefficient| {
                    observable
                        .matrix_elements
                        .iter()
                        .zip(coefficient)
                        .map(|(overlap, coefficient)| *overlap * *coefficient)
                        .sum()
                })
                .collect();
            (observable.name.clone(), estimates)
        })
        .collect();
    ThermalObservableIteration {
        inverse_temperatures: inverse_temperatures.to_vec(),
        values,
        identity,
    }
}

fn ltlm_effective_dimension(
    eigenvalues: &[f64],
    eigenvectors: &[Vec<f64>],
    inverse_temperatures: &[f64],
) -> usize {
    let minimum_beta = inverse_temperatures
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);
    let maximum_first_component = eigenvectors
        .iter()
        .map(|vector| vector[0].abs())
        .fold(0.0_f64, f64::max);
    eigenvalues
        .iter()
        .position(|energy| (-energy * minimum_beta).exp() * maximum_first_component < f64::EPSILON)
        .unwrap_or(eigenvalues.len())
}

fn ltlm_projected_contraction(
    eigenvalues: &[f64],
    eigenvectors: &[Vec<f64>],
    observables: &[ProjectedThermalObservable],
    inverse_temperatures: &[f64],
) -> Result<ThermalObservableIteration> {
    let full_dimension = eigenvalues.len();
    let dimension = ltlm_effective_dimension(eigenvalues, eigenvectors, inverse_temperatures);
    if dimension == 0 {
        return Err(QmbedError::NonConvergence {
            iterations: 0,
            residual: 0.0,
        });
    }
    let identity = inverse_temperatures
        .iter()
        .map(|beta| {
            (0..dimension)
                .map(|eigen| eigenvectors[eigen][0].powi(2) * (-beta * eigenvalues[eigen]).exp())
                .sum()
        })
        .collect();
    let mut values = std::collections::HashMap::new();
    for observable in observables {
        let mut transformed = vec![Complex64::new(0.0, 0.0); dimension * dimension];
        for left in 0..dimension {
            for right in 0..dimension {
                let mut value = Complex64::new(0.0, 0.0);
                for input in 0..dimension {
                    for bra in 0..dimension {
                        value += eigenvectors[left][input]
                            * observable.matrix_elements[input * full_dimension + bra]
                            * eigenvectors[right][bra];
                    }
                }
                transformed[left * dimension + right] = value;
            }
        }
        let estimates = inverse_temperatures
            .iter()
            .map(|beta| {
                let weights = (0..dimension)
                    .map(|eigen| eigenvectors[eigen][0] * (-0.5 * beta * eigenvalues[eigen]).exp())
                    .collect::<Vec<_>>();
                let mut estimate = Complex64::new(0.0, 0.0);
                for left in 0..dimension {
                    for right in 0..dimension {
                        estimate +=
                            weights[left] * transformed[left * dimension + right] * weights[right];
                    }
                }
                estimate
            })
            .collect();
        values.insert(observable.name.clone(), estimates);
    }
    Ok(ThermalObservableIteration {
        inverse_temperatures: inverse_temperatures.to_vec(),
        values,
        identity,
    })
}

/// Contract projected observables with an existing Lanczos Ritz eigensystem.
///
/// `eigenvectors` are column-oriented, matching [`LanczosRitzDecomposition`].
/// This function is useful at language boundaries where an observable may be
/// callback-owned but its action can still be projected into the Krylov basis.
pub fn thermal_observable_contraction(
    method: ThermalLanczosMethod,
    eigenvalues: &[f64],
    eigenvectors: &[Vec<f64>],
    observables: &[ProjectedThermalObservable],
    inverse_temperatures: &[f64],
) -> Result<ThermalObservableIteration> {
    validate_thermal_ritz_data(
        eigenvalues,
        eigenvectors,
        observables,
        inverse_temperatures,
        method,
    )?;
    match method {
        ThermalLanczosMethod::Ftlm => Ok(ftlm_projected_contraction(
            eigenvalues,
            eigenvectors,
            observables,
            inverse_temperatures,
        )),
        ThermalLanczosMethod::Ltlm => {
            ltlm_projected_contraction(eigenvalues, eigenvectors, observables, inverse_temperatures)
        }
    }
}

impl LanczosRitzDecomposition {
    /// Apply observables to the stored Krylov basis and return only the
    /// projected data required by the selected thermal contraction.
    pub fn project_thermal_observables(
        &self,
        observables: &[(String, &dyn LinearOperator)],
        method: ThermalLanczosMethod,
    ) -> Result<Vec<ProjectedThermalObservable>> {
        let dimension = self.decomposition.basis.first().map_or(0, Vec::len);
        if observables.is_empty() {
            return Err(QmbedError::InvalidOptions(
                "thermal observables must be nonempty".into(),
            ));
        }
        let mut names = std::collections::HashSet::new();
        let mut projected = Vec::with_capacity(observables.len());
        for (name, observable) in observables {
            if name.is_empty()
                || !names.insert(name.as_str())
                || observable.shape() != (dimension, dimension)
            {
                return Err(QmbedError::DimensionMismatch(
                    "thermal observables require unique names and matching square shapes".into(),
                ));
            }
            let input_count = match method {
                ThermalLanczosMethod::Ftlm => 1,
                ThermalLanczosMethod::Ltlm => self.decomposition.basis.len(),
            };
            let mut matrix_elements =
                Vec::with_capacity(input_count * self.decomposition.basis.len());
            let mut applied = vec![Complex64::new(0.0, 0.0); dimension];
            for input in 0..input_count {
                observable.apply(&self.decomposition.basis[input], &mut applied)?;
                matrix_elements.extend(
                    self.decomposition
                        .basis
                        .iter()
                        .map(|bra| inner(bra, &applied)),
                );
            }
            projected.push(ProjectedThermalObservable {
                name: name.clone(),
                matrix_elements,
            });
        }
        Ok(projected)
    }

    pub fn thermal_observable_iteration(
        &self,
        method: ThermalLanczosMethod,
        observables: &[(String, &dyn LinearOperator)],
        inverse_temperatures: &[f64],
    ) -> Result<ThermalObservableIteration> {
        let projected = self.project_thermal_observables(observables, method)?;
        thermal_observable_contraction(
            method,
            &self.eigenvalues,
            &self.eigenvectors,
            &projected,
            inverse_temperatures,
        )
    }
}

/// QuSpin-compatible one-sided FTLM observable estimates.
pub fn ftlm_observable_iteration<O>(
    hamiltonian: &O,
    initial: &[Complex64],
    observables: &[(String, &dyn LinearOperator)],
    inverse_temperatures: &[f64],
    options: LanczosOptions,
) -> Result<ThermalObservableIteration>
where
    O: LinearOperator + ?Sized,
{
    let decomposition = lanczos_ritz(hamiltonian, initial, options)?;
    decomposition.thermal_observable_iteration(
        ThermalLanczosMethod::Ftlm,
        observables,
        inverse_temperatures,
    )
}

/// Symmetric low-temperature Lanczos observable estimates.
pub fn ltlm_observable_iteration<O>(
    hamiltonian: &O,
    initial: &[Complex64],
    observables: &[(String, &dyn LinearOperator)],
    inverse_temperatures: &[f64],
    options: LanczosOptions,
) -> Result<ThermalObservableIteration>
where
    O: LinearOperator + ?Sized,
{
    let decomposition = lanczos_ritz(hamiltonian, initial, options)?;
    decomposition.thermal_observable_iteration(
        ThermalLanczosMethod::Ltlm,
        observables,
        inverse_temperatures,
    )
}

pub fn linear_combination_qt(
    basis: &[Vec<Complex64>],
    coefficients: &[Complex64],
) -> Result<Vec<Complex64>> {
    if basis.len() != coefficients.len() || basis.is_empty() {
        return Err(QmbedError::DimensionMismatch(
            "basis and coefficient counts must be equal and nonzero".into(),
        ));
    }
    let dimension = basis[0].len();
    if basis.iter().any(|vector| vector.len() != dimension) {
        return Err(QmbedError::DimensionMismatch(
            "linear-combination basis vectors must have equal lengths".into(),
        ));
    }
    let mut output = vec![Complex64::new(0.0, 0.0); dimension];
    for (coefficient, vector) in coefficients.iter().zip(basis) {
        for (value, basis_value) in output.iter_mut().zip(vector) {
            *value += *coefficient * *basis_value;
        }
    }
    Ok(output)
}

fn time_derivative<O>(
    operator: &O,
    time: f64,
    state: &[Complex64],
    hamiltonian: bool,
) -> Result<Vec<Complex64>>
where
    O: TimeDependentOperator + ?Sized,
{
    let mut derivative = vec![Complex64::new(0.0, 0.0); state.len()];
    operator.apply_at(time, state, &mut derivative)?;
    if hamiltonian {
        for value in &mut derivative {
            *value *= Complex64::new(0.0, -1.0);
        }
    }
    Ok(derivative)
}

fn rk4_step<O>(
    operator: &O,
    time: f64,
    state: &[Complex64],
    step: f64,
    hamiltonian: bool,
) -> Result<Vec<Complex64>>
where
    O: TimeDependentOperator + ?Sized,
{
    let k1 = time_derivative(operator, time, state, hamiltonian)?;
    let stage: Vec<_> = state
        .iter()
        .zip(&k1)
        .map(|(value, derivative)| *value + 0.5 * step * *derivative)
        .collect();
    let k2 = time_derivative(operator, time + 0.5 * step, &stage, hamiltonian)?;
    let stage: Vec<_> = state
        .iter()
        .zip(&k2)
        .map(|(value, derivative)| *value + 0.5 * step * *derivative)
        .collect();
    let k3 = time_derivative(operator, time + 0.5 * step, &stage, hamiltonian)?;
    let stage: Vec<_> = state
        .iter()
        .zip(&k3)
        .map(|(value, derivative)| *value + step * *derivative)
        .collect();
    let k4 = time_derivative(operator, time + step, &stage, hamiltonian)?;
    Ok(state
        .iter()
        .zip(k1.iter().zip(k2.iter().zip(k3.iter().zip(&k4))))
        .map(|(value, (first, (second, (third, fourth))))| {
            *value + step * (*first + 2.0 * *second + 2.0 * *third + *fourth) / 6.0
        })
        .collect())
}

fn adaptive_time_interval<O>(
    operator: &O,
    initial_time: f64,
    initial: &[Complex64],
    interval: f64,
    options: &EvolutionOptions,
) -> Result<Vec<Complex64>>
where
    O: TimeDependentOperator + ?Sized,
{
    if interval == 0.0 {
        return Ok(initial.to_vec());
    }
    let target_time = initial_time + interval;
    let direction = interval.signum();
    let mut step = direction * interval.abs().min(0.1);
    let mut time = initial_time;
    let mut state = initial.to_vec();
    let mut steps = 0;
    let interval_scale = interval.abs().max(1.0);
    while direction * (target_time - time) > 16.0 * f64::EPSILON * target_time.abs().max(1.0) {
        if steps >= options.max_substeps {
            return Err(QmbedError::NonConvergence {
                iterations: steps,
                residual: (target_time - time).abs(),
            });
        }
        if direction * (time + step - target_time) > 0.0 {
            step = target_time - time;
        }
        let full = rk4_step(operator, time, &state, step, options.hamiltonian)?;
        let first_half = rk4_step(operator, time, &state, 0.5 * step, options.hamiltonian)?;
        let two_halves = rk4_step(
            operator,
            time + 0.5 * step,
            &first_half,
            0.5 * step,
            options.hamiltonian,
        )?;
        let error = full
            .iter()
            .zip(&two_halves)
            .map(|(coarse, fine)| (*coarse - *fine).norm_sqr())
            .sum::<f64>()
            .sqrt();
        let scale = vector_norm(&two_halves).max(1.0);
        // Treat `tolerance` as an interval-level budget rather than allowing
        // every accepted RK step to spend the full requested error. The
        // step/interval fraction makes accumulated long-time error track the
        // public tolerance while retaining the usual relative state scaling.
        let threshold =
            options.tolerance * scale * (step.abs() / interval_scale).clamp(f64::EPSILON, 1.0);
        if error <= threshold || step.abs() <= f64::EPSILON * time.abs().max(1.0) {
            state = two_halves;
            time += step;
            steps += 1;
            let growth = if error <= f64::EPSILON {
                2.0
            } else {
                (0.9 * (threshold / error).powf(0.2)).clamp(1.0, 2.0)
            };
            step *= growth;
        } else {
            let shrink = (0.9 * (threshold / error).powf(0.2)).clamp(0.1, 0.8);
            step *= shrink;
        }
    }
    Ok(state)
}

/// Adaptive fourth-order evolution for an explicitly time-dependent operator.
pub fn evolve_time_dependent<O>(
    operator: &O,
    initial: &[Complex64],
    options: EvolutionOptions,
) -> Result<StateTrajectory>
where
    O: TimeDependentOperator + ?Sized,
{
    evolve_time_dependent_from(operator, initial, 0.0, options)
}

/// Evolve an explicitly time-dependent operator from an arbitrary absolute
/// start time.
///
/// Unlike shifting the output grid to zero in a language adapter, this keeps
/// the times observed by the operator callback in the caller's physical time
/// coordinate.
pub fn evolve_time_dependent_from<O>(
    operator: &O,
    initial: &[Complex64],
    initial_time: f64,
    options: EvolutionOptions,
) -> Result<StateTrajectory>
where
    O: TimeDependentOperator + ?Sized,
{
    options.validate()?;
    if !initial_time.is_finite() || options.times[0] < initial_time {
        return Err(QmbedError::InvalidOptions(
            "initial time must be finite and no later than the first output time".into(),
        ));
    }
    let shape = operator.shape();
    if shape.0 != shape.1 || initial.len() != shape.0 {
        return Err(QmbedError::DimensionMismatch(
            "time-dependent operator and initial state do not match".into(),
        ));
    }
    let mut states = Vec::with_capacity(options.times.len());
    let mut state = initial.to_vec();
    let requested_norm = vector_norm(initial);
    let mut previous_time = initial_time;
    for &time in &options.times {
        state = adaptive_time_interval(
            operator,
            previous_time,
            &state,
            time - previous_time,
            &options,
        )?;
        if options.hamiltonian && requested_norm > f64::EPSILON {
            let norm = vector_norm(&state);
            if norm <= f64::EPSILON || !norm.is_finite() {
                return Err(QmbedError::NonConvergence {
                    iterations: options.max_substeps,
                    residual: norm,
                });
            }
            for value in &mut state {
                *value *= requested_norm / norm;
            }
        }
        states.push(state.clone());
        previous_time = time;
    }
    Ok(StateTrajectory {
        times: options.times,
        states,
    })
}

/// Evolve independent column states without changing their column ordering.
pub fn evolve_batch<O>(
    operator: &O,
    initial_columns: &[Vec<Complex64>],
    options: EvolutionOptions,
) -> Result<StateBatchTrajectory>
where
    O: LinearOperator + ?Sized,
{
    if initial_columns.is_empty() {
        return Err(QmbedError::InvalidOptions(
            "a state batch must contain at least one column".into(),
        ));
    }
    let mut by_column = Vec::with_capacity(initial_columns.len());
    for initial in initial_columns {
        by_column.push(evolve(operator, initial, options.clone())?);
    }
    let states = (0..options.times.len())
        .map(|time_index| {
            by_column
                .iter()
                .map(|trajectory| trajectory.states[time_index].clone())
                .collect()
        })
        .collect();
    Ok(StateBatchTrajectory {
        times: options.times,
        states,
    })
}

/// Batched counterpart of [`evolve_time_dependent`].
pub fn evolve_time_dependent_batch<O>(
    operator: &O,
    initial_columns: &[Vec<Complex64>],
    options: EvolutionOptions,
) -> Result<StateBatchTrajectory>
where
    O: TimeDependentOperator + ?Sized,
{
    evolve_time_dependent_batch_from(operator, initial_columns, 0.0, options)
}

/// Batched counterpart of [`evolve_time_dependent_from`].
pub fn evolve_time_dependent_batch_from<O>(
    operator: &O,
    initial_columns: &[Vec<Complex64>],
    initial_time: f64,
    options: EvolutionOptions,
) -> Result<StateBatchTrajectory>
where
    O: TimeDependentOperator + ?Sized,
{
    if initial_columns.is_empty() {
        return Err(QmbedError::InvalidOptions(
            "a state batch must contain at least one column".into(),
        ));
    }
    let mut by_column = Vec::with_capacity(initial_columns.len());
    for initial in initial_columns {
        by_column.push(evolve_time_dependent_from(
            operator,
            initial,
            initial_time,
            options.clone(),
        )?);
    }
    let states = (0..options.times.len())
        .map(|time_index| {
            by_column
                .iter()
                .map(|trajectory| trajectory.states[time_index].clone())
                .collect()
        })
        .collect();
    Ok(StateBatchTrajectory {
        times: options.times,
        states,
    })
}

/// Time grid and explicit-step controls for a caller-defined right-hand side.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct RhsEvolutionOptions {
    /// Finite nondecreasing output times.
    pub times: Vec<f64>,
    /// Maximum Runge--Kutta step.
    pub max_step: f64,
    /// Maximum integration substeps.
    pub max_substeps: usize,
    /// Normalize each returned state.
    pub normalize: bool,
}

impl RhsEvolutionOptions {
    /// Construct callable-RHS integration controls.
    pub fn new(times: impl Into<Vec<f64>>, max_step: f64) -> Self {
        Self {
            times: times.into(),
            max_step,
            max_substeps: 100_000,
            normalize: false,
        }
    }

    /// Set the maximum number of integration substeps.
    #[must_use]
    pub fn with_max_substeps(mut self, max_substeps: usize) -> Self {
        self.max_substeps = max_substeps;
        self
    }

    /// Normalize each accepted output state.
    #[must_use]
    pub fn with_normalization(mut self, normalize: bool) -> Self {
        self.normalize = normalize;
        self
    }

    fn validate(&self, initial_time: f64) -> Result<()> {
        if !initial_time.is_finite()
            || self.times.is_empty()
            || self.times.iter().any(|time| !time.is_finite())
            || self.times.windows(2).any(|pair| pair[0] > pair[1])
            || self.times[0] < initial_time
            || !self.max_step.is_finite()
            || self.max_step <= 0.0
            || self.max_substeps == 0
        {
            return Err(QmbedError::InvalidOptions(
                "invalid callable-RHS time grid or integration controls".into(),
            ));
        }
        Ok(())
    }
}

fn rhs_rk4_step<F>(
    derivative: &F,
    time: f64,
    state: &[Complex64],
    step: f64,
) -> Result<Vec<Complex64>>
where
    F: Fn(f64, &[Complex64], &mut [Complex64]) -> Result<()>,
{
    let dimension = state.len();
    let mut k1 = vec![Complex64::new(0.0, 0.0); dimension];
    derivative(time, state, &mut k1)?;
    let stage: Vec<_> = state
        .iter()
        .zip(&k1)
        .map(|(value, slope)| *value + 0.5 * step * *slope)
        .collect();
    let mut k2 = vec![Complex64::new(0.0, 0.0); dimension];
    derivative(time + 0.5 * step, &stage, &mut k2)?;
    let stage: Vec<_> = state
        .iter()
        .zip(&k2)
        .map(|(value, slope)| *value + 0.5 * step * *slope)
        .collect();
    let mut k3 = vec![Complex64::new(0.0, 0.0); dimension];
    derivative(time + 0.5 * step, &stage, &mut k3)?;
    let stage: Vec<_> = state
        .iter()
        .zip(&k3)
        .map(|(value, slope)| *value + step * *slope)
        .collect();
    let mut k4 = vec![Complex64::new(0.0, 0.0); dimension];
    derivative(time + step, &stage, &mut k4)?;
    Ok(state
        .iter()
        .zip(k1.iter().zip(k2.iter().zip(k3.iter().zip(&k4))))
        .map(|(value, (first, (second, (third, fourth))))| {
            *value + step * (*first + 2.0 * *second + 2.0 * *third + *fourth) / 6.0
        })
        .collect())
}

/// Integrate an arbitrary complex right-hand side `dstate/dt = f(t, state)`.
pub fn evolve_rhs<F>(
    initial: &[Complex64],
    initial_time: f64,
    options: RhsEvolutionOptions,
    derivative: F,
) -> Result<StateTrajectory>
where
    F: Fn(f64, &[Complex64], &mut [Complex64]) -> Result<()>,
{
    options.validate(initial_time)?;
    if initial.is_empty() {
        return Err(QmbedError::DimensionMismatch(
            "callable-RHS state must be nonempty".into(),
        ));
    }
    let mut state = initial.to_vec();
    let normalization = vector_norm(initial);
    let mut current_time = initial_time;
    let mut used_steps = 0_usize;
    let mut states = Vec::with_capacity(options.times.len());
    for &target_time in &options.times {
        let interval = target_time - current_time;
        let steps = (interval.abs() / options.max_step).ceil().max(1.0) as usize;
        if used_steps.saturating_add(steps) > options.max_substeps {
            return Err(QmbedError::NonConvergence {
                iterations: used_steps,
                residual: interval.abs(),
            });
        }
        let step = interval / steps as f64;
        for _ in 0..steps {
            state = rhs_rk4_step(&derivative, current_time, &state, step)?;
            current_time += step;
        }
        if options.normalize && normalization > f64::EPSILON {
            let norm = vector_norm(&state);
            if norm <= f64::EPSILON || !norm.is_finite() {
                return Err(QmbedError::NonConvergence {
                    iterations: used_steps + steps,
                    residual: norm,
                });
            }
            for value in &mut state {
                *value *= normalization / norm;
            }
        }
        used_steps += steps;
        current_time = target_time;
        states.push(state.clone());
    }
    Ok(StateTrajectory {
        times: options.times,
        states,
    })
}

/// Liouville-von Neumann evolution of a row-major density matrix under a
/// Hermitian static Hamiltonian.
pub fn evolve_density<O>(
    hamiltonian: &O,
    initial_density: &[Complex64],
    mut options: RhsEvolutionOptions,
) -> Result<StateTrajectory>
where
    O: LinearOperator + ?Sized,
{
    let shape = hamiltonian.shape();
    if shape.0 != shape.1 || initial_density.len() != shape.0.saturating_mul(shape.0) {
        return Err(QmbedError::DimensionMismatch(
            "density evolution requires a square Hamiltonian and density matrix".into(),
        ));
    }
    let dimension = shape.0;
    options.normalize = false;
    evolve_rhs(initial_density, 0.0, options, |_, density, output| {
        let mut column = vec![Complex64::new(0.0, 0.0); dimension];
        let mut applied = vec![Complex64::new(0.0, 0.0); dimension];
        let mut h_rho = vec![Complex64::new(0.0, 0.0); density.len()];
        let mut rho_h = vec![Complex64::new(0.0, 0.0); density.len()];
        for column_index in 0..dimension {
            for row in 0..dimension {
                column[row] = density[row * dimension + column_index];
            }
            hamiltonian.apply(&column, &mut applied)?;
            for row in 0..dimension {
                h_rho[row * dimension + column_index] = applied[row];
            }
        }
        // For Hermitian H, conjugating a row turns right multiplication by H
        // into the same column action used above.
        for row in 0..dimension {
            for column_index in 0..dimension {
                column[column_index] = density[row * dimension + column_index].conj();
            }
            hamiltonian.apply(&column, &mut applied)?;
            for column_index in 0..dimension {
                rho_h[row * dimension + column_index] = applied[column_index].conj();
            }
        }
        for index in 0..output.len() {
            output[index] = Complex64::new(0.0, -1.0) * (h_rho[index] - rho_h[index]);
        }
        Ok(())
    })
}

#[cfg(test)]
mod tests {
    use super::{
        Complex64, ProjectedBasis, dgks_reorthogonalize_complex, dgks_reorthogonalize_real,
        lanczos_projection, projected_exponential_action, projected_exponential_error_bound,
    };
    use crate::Result;
    use crate::operator::{LinearOperator, MatrixFormat, Operator};

    struct ComplexView<'a>(&'a Operator);

    impl LinearOperator for ComplexView<'_> {
        fn shape(&self) -> (usize, usize) {
            self.0.shape()
        }

        fn format(&self) -> MatrixFormat {
            self.0.format()
        }

        fn apply(&self, input: &[Complex64], output: &mut [Complex64]) -> Result<()> {
            self.0.apply(input, output)
        }
    }

    #[test]
    fn dgks_second_pass_is_conditional_for_real_and_complex_vectors() {
        let real_basis = vec![vec![1.0, 0.0]];
        let mut contaminated_real = vec![1.0, 1.0e-12];
        let (_, repeated_real) = dgks_reorthogonalize_real(&real_basis, &mut contaminated_real);
        assert!(repeated_real);
        let mut orthogonal_real = vec![0.0, 1.0];
        let (_, repeated_real) = dgks_reorthogonalize_real(&real_basis, &mut orthogonal_real);
        assert!(!repeated_real);

        let complex_basis = vec![vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)]];
        let mut contaminated_complex = vec![Complex64::new(0.0, 1.0), Complex64::new(1.0e-12, 0.0)];
        let (_, repeated_complex) =
            dgks_reorthogonalize_complex(&complex_basis, &mut contaminated_complex);
        assert!(repeated_complex);
        let mut orthogonal_complex = vec![Complex64::new(0.0, 0.0), Complex64::new(0.0, 1.0)];
        let (_, repeated_complex) =
            dgks_reorthogonalize_complex(&complex_basis, &mut orthogonal_complex);
        assert!(!repeated_complex);
    }

    #[test]
    fn real_and_complex_projected_bases_define_the_same_krylov_action() {
        let operator = Operator::from_triplets(
            4,
            4,
            [
                (0, 0, Complex64::new(-1.0, 0.0)),
                (0, 1, Complex64::new(0.5, 0.0)),
                (1, 0, Complex64::new(0.5, 0.0)),
                (1, 1, Complex64::new(0.25, 0.0)),
                (1, 2, Complex64::new(-0.7, 0.0)),
                (2, 1, Complex64::new(-0.7, 0.0)),
                (2, 3, Complex64::new(0.3, 0.0)),
                (3, 2, Complex64::new(0.3, 0.0)),
            ],
            MatrixFormat::Csc,
        )
        .unwrap();
        let initial = [
            Complex64::new(1.0, 0.0),
            Complex64::new(-0.5, 0.0),
            Complex64::new(0.25, 0.0),
            Complex64::new(0.75, 0.0),
        ];
        let real = lanczos_projection(&operator, &initial, 4).unwrap();
        let complex = lanczos_projection(&ComplexView(&operator), &initial, 4).unwrap();
        assert!(matches!(&real.basis, ProjectedBasis::Real(_)));
        assert!(matches!(&complex.basis, ProjectedBasis::Complex(_)));

        for interval in [0.1, 1.7] {
            let real_state = projected_exponential_action(&real, interval, true, initial.len());
            let complex_state =
                projected_exponential_action(&complex, interval, true, initial.len());
            for (real_value, complex_value) in real_state.iter().zip(complex_state) {
                assert!((*real_value - complex_value).norm() < 1.0e-12);
            }
            let real_error = projected_exponential_error_bound(&real, interval, initial.len());
            let complex_error =
                projected_exponential_error_bound(&complex, interval, initial.len());
            assert!((real_error - complex_error).abs() < 1.0e-12);
        }
    }
}
