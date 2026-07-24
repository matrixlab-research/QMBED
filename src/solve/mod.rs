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

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SpectrumTarget {
    SmallestAlgebraic,
    LargestAlgebraic,
    SmallestMagnitude,
    LargestMagnitude,
    BothEnds,
    Shift(f64),
}

#[derive(Clone, Debug, PartialEq)]
pub struct EigshOptions {
    pub eigenpairs: usize,
    pub target: SpectrumTarget,
    pub krylov_dimension: Option<usize>,
    pub tolerance: f64,
    pub max_iterations: usize,
    pub seed: u64,
}

const GUARANTEED_DENSE_EIGSH_CROSSOVER: usize = 128;
const AUTOMATIC_DENSE_EIGSH_CROSSOVER: usize = 256;

fn use_dense_eigsh(dimension: usize, options: &EigshOptions) -> bool {
    dimension <= GUARANTEED_DENSE_EIGSH_CROSSOVER
        || (dimension <= AUTOMATIC_DENSE_EIGSH_CROSSOVER && options.krylov_dimension.is_none())
}

impl EigshOptions {
    pub fn smallest_algebraic(eigenpairs: usize) -> Self {
        Self {
            eigenpairs,
            target: SpectrumTarget::SmallestAlgebraic,
            krylov_dimension: None,
            tolerance: 1.0e-10,
            max_iterations: 1_000,
            seed: 0,
        }
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

#[derive(Clone, Debug)]
pub struct Eigensystem {
    pub eigenvalues: Vec<f64>,
    pub eigenvectors: Vec<Vec<Complex64>>,
    pub residuals: Vec<f64>,
    pub iterations: usize,
    pub converged: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EighOptions {
    pub return_eigenvectors: bool,
}

impl Default for EighOptions {
    fn default() -> Self {
        Self {
            return_eigenvectors: true,
        }
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
        iterations: 1,
        converged: true,
    })
}

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

fn inner(left: &[Complex64], right: &[Complex64]) -> Complex64 {
    left.iter()
        .zip(right)
        .map(|(left_value, right_value)| left_value.conj() * *right_value)
        .sum()
}

fn vector_norm(vector: &[Complex64]) -> f64 {
    vector.iter().map(Complex64::norm_sqr).sum::<f64>().sqrt()
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

        // A second modified Gram-Schmidt pass is necessary when extremal
        // Ritz values are clustered. Without it, a nominally complete
        // Krylov space can still return residuals above the requested
        // tolerance because roundoff reintroduces earlier Lanczos vectors.
        for _ in 0..2 {
            for vector in &basis {
                let overlap = real_inner(vector, &output);
                for (value, basis_value) in output.iter_mut().zip(vector) {
                    *value -= overlap * *basis_value;
                }
            }
        }
        let beta = real_vector_norm(&output);
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
    let failure_residual = residuals.iter().copied().fold(0.0_f64, f64::max);
    let accepted_residual = options.tolerance.max(1.0e-7);
    if failure_residual > accepted_residual {
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
        converged: true,
    })
}

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
    if let SpectrumTarget::Shift(shift) = options.target
        && shifted_solver.is_none()
    {
        shifted_solver = operator.shifted_solver(shift)?;
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

        // Two-pass modified Gram-Schmidt keeps clustered, multiple, and
        // interior Ritz vectors reliable at the requested residual.
        for _ in 0..2 {
            for vector in &basis {
                let overlap = inner(vector, &output);
                for (value, basis_value) in output.iter_mut().zip(vector) {
                    *value -= overlap * *basis_value;
                }
            }
        }
        let beta = vector_norm(&output);
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
    let failure_residual = residuals.iter().copied().fold(0.0_f64, f64::max);
    let accepted_residual = options.tolerance.max(1.0e-7);
    if failure_residual > accepted_residual {
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
        converged: true,
    })
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
        return lanczos_eigsh(operator, &options, None);
    }
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
    let eigenvalues: Vec<_> = indices.iter().map(|&index| values[index]).collect();
    let eigenvectors: Vec<_> = indices
        .iter()
        .map(|&index| vectors[index].clone())
        .collect();
    let residuals = eigenvalues
        .iter()
        .zip(&eigenvectors)
        .map(|(&value, vector)| residual_norm(operator, value, vector))
        .collect::<Result<Vec<_>>>()?;
    Ok(Eigensystem {
        eigenvalues,
        eigenvectors,
        residuals,
        iterations: 1,
        converged: true,
    })
}

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
    if initial.len() != shape.0 {
        return Err(QmbedError::DimensionMismatch(
            "eigsh initial vector does not match the operator".into(),
        ));
    }
    if use_dense_eigsh(shape.0, &options) {
        return eigsh(operator, options);
    }
    lanczos_eigsh(operator, &options, Some(initial))
}

pub fn eigsh_values<O>(operator: &O, options: EigshOptions) -> Result<Vec<f64>>
where
    O: LinearOperator + ?Sized,
{
    Ok(eigsh(operator, options)?.eigenvalues)
}

#[derive(Clone, Debug, PartialEq)]
pub struct EvolutionOptions {
    pub times: Vec<f64>,
    pub krylov_dimension: usize,
    pub tolerance: f64,
    pub max_substeps: usize,
    pub hamiltonian: bool,
}

impl EvolutionOptions {
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

#[derive(Clone, Debug)]
pub struct StateTrajectory {
    pub times: Vec<f64>,
    pub states: Vec<Vec<Complex64>>,
}

/// Column-oriented batch trajectory: `states[time_index][column_index]`.
#[derive(Clone, Debug)]
pub struct StateBatchTrajectory {
    pub times: Vec<f64>,
    pub states: Vec<Vec<Vec<Complex64>>>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct LanczosOptions {
    pub krylov_dimension: usize,
    pub tolerance: f64,
}

impl LanczosOptions {
    fn validate(&self) -> Result<()> {
        if self.krylov_dimension == 0 || !self.tolerance.is_finite() || self.tolerance <= 0.0 {
            return Err(QmbedError::InvalidOptions(
                "Lanczos dimension and tolerance must be positive".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct LanczosVector {
    pub index: usize,
    pub vector: Vec<Complex64>,
    pub diagonal: f64,
    pub next_off_diagonal: f64,
}

#[derive(Clone, Debug)]
pub struct LanczosDecomposition {
    pub initial_norm: f64,
    pub basis: Vec<Vec<Complex64>>,
    pub diagonal: Vec<f64>,
    pub off_diagonal: Vec<f64>,
}

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
        // Two-pass full reorthogonalization keeps the public basis usable for
        // degenerate and long Krylov runs, not only for tridiagonal scalars.
        for _ in 0..2 {
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

#[derive(Clone, Debug, PartialEq)]
pub struct ExpmOptions {
    pub times: Vec<f64>,
    pub krylov_dimension: usize,
    pub tolerance: f64,
    pub max_substeps: usize,
    pub hamiltonian: bool,
}

impl From<ExpmOptions> for EvolutionOptions {
    fn from(options: ExpmOptions) -> Self {
        Self {
            times: options.times,
            krylov_dimension: options.krylov_dimension,
            tolerance: options.tolerance,
            max_substeps: options.max_substeps,
            hamiltonian: options.hamiltonian,
        }
    }
}

struct LanczosProjection {
    initial_norm: f64,
    basis: Vec<Vec<Complex64>>,
    eigenvalues: Vec<f64>,
    eigenvectors: DMatrix<f64>,
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
            basis: Vec::new(),
            eigenvalues: Vec::new(),
            eigenvectors: DMatrix::zeros(0, 0),
        });
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
        basis,
        eigenvalues: decomposition.eigenvalues.as_slice().to_vec(),
        eigenvectors: decomposition.eigenvectors,
    })
}

fn projected_exponential_action(
    projection: &LanczosProjection,
    interval: f64,
    hamiltonian: bool,
    ambient_dimension: usize,
) -> Vec<Complex64> {
    if projection.basis.is_empty() {
        return vec![Complex64::new(0.0, 0.0); ambient_dimension];
    }
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
    let mut output = vec![Complex64::new(0.0, 0.0); ambient_dimension];
    for (coefficient, vector) in coefficients.iter().zip(&projection.basis) {
        for (value, basis_value) in output.iter_mut().zip(vector) {
            *value += *coefficient * *basis_value;
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
    let shape = operator.shape();
    if shape.0 != shape.1 || initial.len() != shape.0 {
        return Err(QmbedError::DimensionMismatch(
            "exponential action requires a square operator matching the state".into(),
        ));
    }
    if !exponent.re.is_finite()
        || !exponent.im.is_finite()
        || krylov_dimension == 0
        || !tolerance.is_finite()
        || tolerance <= 0.0
        || max_substeps == 0
    {
        return Err(QmbedError::InvalidOptions(
            "invalid exponential coefficient or numerical controls".into(),
        ));
    }
    if exponent.norm() <= f64::EPSILON {
        return Ok(initial.to_vec());
    }
    let requested_steps = exponent.norm().ceil().max(1.0) as usize;
    if requested_steps > max_substeps {
        return Err(QmbedError::NonConvergence {
            iterations: max_substeps,
            residual: exponent.norm(),
        });
    }
    let factor = exponent / requested_steps as f64;
    let mut state = initial.to_vec();
    let mut applied = vec![Complex64::new(0.0, 0.0); shape.0];
    for _ in 0..requested_steps {
        let mut sum = state.clone();
        let mut term = state.clone();
        for order in 1..=krylov_dimension {
            operator.apply(&term, &mut applied)?;
            let scale = factor / order as f64;
            for (next, value) in term.iter_mut().zip(&applied) {
                *next = scale * *value;
            }
            for (total, value) in sum.iter_mut().zip(&term) {
                *total += *value;
            }
            if vector_norm(&term) <= tolerance * vector_norm(&sum).max(1.0) {
                break;
            }
            if order == krylov_dimension {
                return Err(QmbedError::NonConvergence {
                    iterations: order,
                    residual: vector_norm(&term),
                });
            }
        }
        state = sum;
    }
    Ok(state)
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
    options.validate()?;
    let shape = operator.shape();
    if shape.0 != shape.1 || initial.len() != shape.0 {
        return Err(QmbedError::DimensionMismatch(
            "evolution operator and initial state do not match".into(),
        ));
    }
    let mut states = Vec::with_capacity(options.times.len());
    if options.hamiltonian {
        let projection = lanczos_projection(operator, initial, options.krylov_dimension)?;
        for &time in &options.times {
            states.push(projected_exponential_action(
                &projection,
                time,
                true,
                initial.len(),
            ));
        }
        return Ok(StateTrajectory {
            times: options.times,
            states,
        });
    }
    let mut state = initial.to_vec();
    let mut previous_time = 0.0;
    for &time in &options.times {
        state = expm_action(operator, &state, time - previous_time, &options)?;
        states.push(state.clone());
        previous_time = time;
    }
    Ok(StateTrajectory {
        times: options.times,
        states,
    })
}

/// Exponential action over a time grid. Hermitian Hamiltonians use one
/// reusable Lanczos projection for the complete grid.
pub fn expm_multiply<O>(
    operator: &O,
    initial: &[Complex64],
    options: ExpmOptions,
) -> Result<StateTrajectory>
where
    O: LinearOperator + ?Sized,
{
    evolve(operator, initial, options.into())
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

fn lanczos_ritz_data<O>(
    hamiltonian: &O,
    initial: &[Complex64],
    options: LanczosOptions,
) -> Result<(LanczosDecomposition, SymmetricEigen<f64, nalgebra::Dyn>)>
where
    O: LinearOperator + ?Sized,
{
    let decomposition = lanczos_full(hamiltonian, initial, options)?;
    let size = decomposition.diagonal.len();
    let mut tridiagonal = DMatrix::<f64>::zeros(size, size);
    for index in 0..size {
        tridiagonal[(index, index)] = decomposition.diagonal[index];
        if index + 1 < size {
            tridiagonal[(index, index + 1)] = decomposition.off_diagonal[index];
            tridiagonal[(index + 1, index)] = decomposition.off_diagonal[index];
        }
    }
    Ok((decomposition, SymmetricEigen::new(tridiagonal)))
}

fn validate_thermal_observables(
    observables: &[(String, &dyn LinearOperator)],
    dimension: usize,
    inverse_temperatures: &[f64],
) -> Result<()> {
    if observables.is_empty()
        || inverse_temperatures.is_empty()
        || inverse_temperatures
            .iter()
            .any(|beta| !beta.is_finite() || *beta < 0.0)
    {
        return Err(QmbedError::InvalidOptions(
            "thermal observables and inverse temperatures must be nonempty and valid".into(),
        ));
    }
    let mut names = std::collections::HashSet::new();
    for (name, observable) in observables {
        if name.is_empty() || !names.insert(name) || observable.shape() != (dimension, dimension) {
            return Err(QmbedError::DimensionMismatch(
                "thermal observables require unique names and matching square shapes".into(),
            ));
        }
    }
    Ok(())
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
    validate_thermal_observables(observables, initial.len(), inverse_temperatures)?;
    let (decomposition, ritz) = lanczos_ritz_data(hamiltonian, initial, options)?;
    let size = decomposition.basis.len();
    let identity = inverse_temperatures
        .iter()
        .map(|beta| {
            (0..size)
                .map(|eigen| {
                    ritz.eigenvectors[(0, eigen)].powi(2) * (-beta * ritz.eigenvalues[eigen]).exp()
                })
                .sum()
        })
        .collect();
    let mut values = std::collections::HashMap::new();
    for (name, observable) in observables {
        let mut applied = vec![Complex64::new(0.0, 0.0); initial.len()];
        observable.apply(&decomposition.basis[0], &mut applied)?;
        let overlaps: Vec<_> = decomposition
            .basis
            .iter()
            .map(|vector| inner(vector, &applied))
            .collect();
        let estimates = inverse_temperatures
            .iter()
            .map(|beta| {
                (0..size)
                    .map(|row| {
                        let coefficient = (0..size)
                            .map(|eigen| {
                                ritz.eigenvectors[(row, eigen)]
                                    * ritz.eigenvectors[(0, eigen)]
                                    * (-beta * ritz.eigenvalues[eigen]).exp()
                            })
                            .sum::<f64>();
                        overlaps[row] * coefficient
                    })
                    .sum()
            })
            .collect();
        values.insert(name.clone(), estimates);
    }
    Ok(ThermalObservableIteration {
        inverse_temperatures: inverse_temperatures.to_vec(),
        values,
        identity,
    })
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
    validate_thermal_observables(observables, initial.len(), inverse_temperatures)?;
    let (decomposition, ritz) = lanczos_ritz_data(hamiltonian, initial, options)?;
    let size = decomposition.basis.len();
    let identity = inverse_temperatures
        .iter()
        .map(|beta| {
            (0..size)
                .map(|eigen| {
                    ritz.eigenvectors[(0, eigen)].powi(2) * (-beta * ritz.eigenvalues[eigen]).exp()
                })
                .sum()
        })
        .collect();
    let mut values = std::collections::HashMap::new();
    for (name, observable) in observables {
        let mut matrix_elements = vec![Complex64::new(0.0, 0.0); size * size];
        let mut applied = vec![Complex64::new(0.0, 0.0); initial.len()];
        for row in 0..size {
            observable.apply(&decomposition.basis[row], &mut applied)?;
            for column in 0..size {
                matrix_elements[row * size + column] =
                    inner(&decomposition.basis[column], &applied);
            }
        }
        let estimates = inverse_temperatures
            .iter()
            .map(|beta| {
                let weights: Vec<_> = (0..size)
                    .map(|eigen| {
                        ritz.eigenvectors[(0, eigen)]
                            * (-0.5 * beta * ritz.eigenvalues[eigen]).exp()
                    })
                    .collect();
                let mut estimate = Complex64::new(0.0, 0.0);
                for left in 0..size {
                    for right in 0..size {
                        let mut projected = Complex64::new(0.0, 0.0);
                        for row in 0..size {
                            for column in 0..size {
                                projected += ritz.eigenvectors[(row, left)]
                                    * matrix_elements[row * size + column]
                                    * ritz.eigenvectors[(column, right)];
                            }
                        }
                        estimate += weights[left] * projected * weights[right];
                    }
                }
                estimate
            })
            .collect();
        values.insert(name.clone(), estimates);
    }
    Ok(ThermalObservableIteration {
        inverse_temperatures: inverse_temperatures.to_vec(),
        values,
        identity,
    })
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
    let initial_norm = vector_norm(initial);
    let mut steps = 0;
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
        let threshold = options.tolerance * scale;
        if error <= threshold || step.abs() <= f64::EPSILON * time.abs().max(1.0) {
            state = two_halves;
            if options.hamiltonian && initial_norm > f64::EPSILON {
                let norm = vector_norm(&state);
                if norm > f64::EPSILON && norm.is_finite() {
                    for value in &mut state {
                        *value *= initial_norm / norm;
                    }
                }
            }
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
    options.validate()?;
    let shape = operator.shape();
    if shape.0 != shape.1 || initial.len() != shape.0 {
        return Err(QmbedError::DimensionMismatch(
            "time-dependent operator and initial state do not match".into(),
        ));
    }
    let mut states = Vec::with_capacity(options.times.len());
    let mut state = initial.to_vec();
    let mut previous_time = 0.0;
    for &time in &options.times {
        state = adaptive_time_interval(
            operator,
            previous_time,
            &state,
            time - previous_time,
            &options,
        )?;
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
    if initial_columns.is_empty() {
        return Err(QmbedError::InvalidOptions(
            "a state batch must contain at least one column".into(),
        ));
    }
    let mut by_column = Vec::with_capacity(initial_columns.len());
    for initial in initial_columns {
        by_column.push(evolve_time_dependent(operator, initial, options.clone())?);
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

#[derive(Clone, Debug, PartialEq)]
pub struct RhsEvolutionOptions {
    pub times: Vec<f64>,
    pub max_step: f64,
    pub max_substeps: usize,
    pub normalize: bool,
}

impl RhsEvolutionOptions {
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
