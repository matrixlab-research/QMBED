use std::f64::consts::PI;
use std::sync::Arc;

use num_complex::Complex64;

use crate::backend;
use crate::operator::{
    LinearOperator, MatrixFormat, Operator, TimeDependentOperator, materialize_dense,
};
use crate::solve::{
    EvolutionOptions, evolve, evolve_time_dependent, expm_action, hermitian_eigenpairs_all,
    lanczos_spectral_measure,
};
use crate::{QmbedError, Result};

const DENSE_PROPAGATOR_CUTOFF: usize = 128;

pub struct DriveStep {
    pub hamiltonian: Arc<dyn LinearOperator>,
    pub duration: f64,
}

pub struct CallableDriveStep {
    pub hamiltonian: Arc<dyn TimeDependentOperator>,
    pub duration: f64,
}

impl CallableDriveStep {
    pub fn new(hamiltonian: Arc<dyn TimeDependentOperator>, duration: f64) -> Result<Self> {
        let shape = hamiltonian.shape();
        if shape.0 != shape.1 {
            return Err(QmbedError::DimensionMismatch(
                "a callable drive Hamiltonian must be square".into(),
            ));
        }
        if !duration.is_finite() || duration < 0.0 {
            return Err(QmbedError::InvalidOptions(
                "drive duration must be finite and nonnegative".into(),
            ));
        }
        Ok(Self {
            hamiltonian,
            duration,
        })
    }
}

enum FloquetStep {
    Static(DriveStep),
    Callable(CallableDriveStep),
}

impl DriveStep {
    pub fn new(hamiltonian: Arc<dyn LinearOperator>, duration: f64) -> Result<Self> {
        let shape = hamiltonian.shape();
        if shape.0 != shape.1 {
            return Err(QmbedError::DimensionMismatch(
                "a drive Hamiltonian must be square".into(),
            ));
        }
        if !duration.is_finite() || duration < 0.0 {
            return Err(QmbedError::InvalidOptions(
                "drive duration must be finite and nonnegative".into(),
            ));
        }
        Ok(Self {
            hamiltonian,
            duration,
        })
    }
}

/// One period of a piecewise-constant drive.
pub struct Floquet {
    steps: Vec<FloquetStep>,
    dimension: usize,
    evolution: EvolutionOptions,
    analysis_period: Option<f64>,
}

impl Floquet {
    pub fn new(steps: impl IntoIterator<Item = DriveStep>) -> Result<Self> {
        let steps: Vec<_> = steps.into_iter().map(FloquetStep::Static).collect();
        let first = steps.first().ok_or_else(|| {
            QmbedError::InvalidOptions("Floquet requires at least one drive step".into())
        })?;
        let dimension = first.shape().0;
        if steps
            .iter()
            .any(|step| step.shape() != (dimension, dimension))
        {
            return Err(QmbedError::DimensionMismatch(
                "all drive steps must have the same square shape".into(),
            ));
        }
        Ok(Self {
            steps,
            dimension,
            evolution: EvolutionOptions {
                times: vec![0.0],
                krylov_dimension: 64,
                tolerance: 1.0e-12,
                max_substeps: 10_000,
                hamiltonian: true,
            },
            analysis_period: None,
        })
    }

    pub fn from_callable(steps: impl IntoIterator<Item = CallableDriveStep>) -> Result<Self> {
        let steps: Vec<_> = steps.into_iter().map(FloquetStep::Callable).collect();
        let first = steps.first().ok_or_else(|| {
            QmbedError::InvalidOptions("Floquet requires at least one drive step".into())
        })?;
        let dimension = first.shape().0;
        if steps
            .iter()
            .any(|step| step.shape() != (dimension, dimension))
        {
            return Err(QmbedError::DimensionMismatch(
                "all drive steps must have the same square shape".into(),
            ));
        }
        Ok(Self {
            steps,
            dimension,
            evolution: EvolutionOptions {
                times: vec![0.0],
                krylov_dimension: 64,
                tolerance: 1.0e-12,
                max_substeps: 10_000,
                hamiltonian: true,
            },
            analysis_period: None,
        })
    }

    pub fn with_evolution_options(mut self, options: EvolutionOptions) -> Self {
        self.evolution = options;
        self.evolution.hamiltonian = true;
        self
    }

    /// Override the physical Floquet period used for quasienergies and the
    /// effective Hamiltonian without changing the step durations.
    ///
    /// This supports kicked protocols whose explicit evolution intervals do
    /// not add up to the declared drive period.
    pub fn with_period(mut self, period: f64) -> Result<Self> {
        if !period.is_finite() || period <= 0.0 {
            return Err(QmbedError::InvalidOptions(
                "Floquet analysis period must be finite and positive".into(),
            ));
        }
        self.analysis_period = Some(period);
        Ok(self)
    }

    pub fn apply_period(&self, input: &[Complex64], output: &mut [Complex64]) -> Result<()> {
        if input.len() != self.dimension || output.len() != self.dimension {
            return Err(QmbedError::DimensionMismatch(
                "Floquet input or output length does not match".into(),
            ));
        }
        let mut state = input.to_vec();
        for step in &self.steps {
            match step {
                FloquetStep::Static(step) => {
                    state = expm_action(
                        step.hamiltonian.as_ref(),
                        &state,
                        step.duration,
                        &self.evolution,
                    )?;
                }
                FloquetStep::Callable(step) => {
                    let mut options = self.evolution.clone();
                    options.times = vec![step.duration];
                    state = evolve_time_dependent(step.hamiltonian.as_ref(), &state, options)?
                        .states
                        .pop()
                        .ok_or(QmbedError::NonConvergence {
                            iterations: 0,
                            residual: f64::INFINITY,
                        })?;
                }
            }
        }
        output.copy_from_slice(&state);
        Ok(())
    }

    pub fn period(&self) -> f64 {
        self.analysis_period
            .unwrap_or_else(|| self.protocol_duration())
    }

    /// Sum of the explicit piecewise evolution intervals.
    pub fn protocol_duration(&self) -> f64 {
        self.steps.iter().map(FloquetStep::duration).sum()
    }

    pub fn full_unitary(&self, format: MatrixFormat) -> Result<Operator> {
        let dense = if self
            .steps
            .iter()
            .all(|step| matches!(step, FloquetStep::Static(_)))
            && self.dimension <= DENSE_PROPAGATOR_CUTOFF
        {
            let mut total = vec![Complex64::new(0.0, 0.0); self.dimension * self.dimension];
            for index in 0..self.dimension {
                total[index * self.dimension + index] = Complex64::new(1.0, 0.0);
            }
            for step in &self.steps {
                let FloquetStep::Static(step) = step else {
                    unreachable!("all Floquet steps were checked as static");
                };
                let hamiltonian = materialize_dense(step.hamiltonian.as_ref())?;
                let propagator = backend::hermitian_exponential(
                    &hamiltonian,
                    self.dimension,
                    Complex64::new(0.0, -step.duration),
                )?;
                total = backend::square_matmul(&propagator, &total, self.dimension)?;
            }
            total
        } else {
            materialize_dense(self)?
        };
        Operator::from_dense(self.dimension, self.dimension, dense)?.converted(format)
    }

    pub fn eigensystem(&self) -> Result<FloquetEigensystem> {
        let period = self.period();
        if period <= 0.0 {
            return Err(QmbedError::InvalidOptions(
                "Floquet eigensystems require a positive period".into(),
            ));
        }
        let dense_unitary = self.full_unitary(MatrixFormat::Dense)?.to_dense();
        floquet_eigensystem_from_dense(&dense_unitary, self.dimension, period)
    }

    pub fn effective_hamiltonian(&self, format: MatrixFormat) -> Result<Operator> {
        let eigensystem = self.eigensystem()?;
        self.effective_hamiltonian_from_eigensystem(&eigensystem, format)
    }

    fn effective_hamiltonian_from_eigensystem(
        &self,
        eigensystem: &FloquetEigensystem,
        format: MatrixFormat,
    ) -> Result<Operator> {
        floquet_effective_hamiltonian(eigensystem, self.dimension, format)
    }

    /// Materialize one period once and compute its complete dense spectrum.
    ///
    /// The returned object owns both products, so callers that need unitarity
    /// and quasienergy checks never need to rebuild the propagator or bring a
    /// second dense eigensolver into the workflow.
    pub fn spectrum(&self, format: MatrixFormat) -> Result<FloquetSpectrum> {
        let period = self.period();
        if period <= 0.0 {
            return Err(QmbedError::InvalidOptions(
                "Floquet spectra require a positive period".into(),
            ));
        }
        let dense_unitary = self.full_unitary(MatrixFormat::Dense)?.to_dense();
        let unitarity_residual = backend::unitarity_residual(&dense_unitary, self.dimension)?;
        let eigensystem = floquet_eigensystem_from_dense(&dense_unitary, self.dimension, period)?;
        Ok(FloquetSpectrum {
            period,
            protocol_duration: self.protocol_duration(),
            unitary: Operator::from_dense(self.dimension, self.dimension, dense_unitary)?
                .converted(format)?,
            unitarity_residual,
            eigensystem,
        })
    }

    /// Compute the period propagator and all spectral products while
    /// materializing the propagator only once.
    pub fn analyze(&self, format: MatrixFormat) -> Result<FloquetAnalysis> {
        let spectrum = self.spectrum(format)?;
        let effective_hamiltonian =
            self.effective_hamiltonian_from_eigensystem(&spectrum.eigensystem, format)?;
        Ok(FloquetAnalysis {
            period: spectrum.period,
            protocol_duration: spectrum.protocol_duration,
            unitary: spectrum.unitary,
            unitarity_residual: spectrum.unitarity_residual,
            eigensystem: spectrum.eigensystem,
            effective_hamiltonian,
        })
    }
}

impl FloquetStep {
    fn shape(&self) -> (usize, usize) {
        match self {
            Self::Static(step) => step.hamiltonian.shape(),
            Self::Callable(step) => step.hamiltonian.shape(),
        }
    }

    fn duration(&self) -> f64 {
        match self {
            Self::Static(step) => step.duration,
            Self::Callable(step) => step.duration,
        }
    }
}

#[derive(Clone, Debug)]
pub struct FloquetEigensystem {
    pub quasienergies: Vec<f64>,
    pub eigenvalues: Vec<Complex64>,
    pub eigenvectors: Vec<Vec<Complex64>>,
    pub residuals: Vec<f64>,
}

#[derive(Clone, Debug)]
pub struct FloquetSpectrum {
    pub period: f64,
    pub protocol_duration: f64,
    pub unitary: Operator,
    pub unitarity_residual: f64,
    pub eigensystem: FloquetEigensystem,
}

#[derive(Clone, Debug)]
pub struct FloquetAnalysis {
    pub period: f64,
    pub protocol_duration: f64,
    pub unitary: Operator,
    pub unitarity_residual: f64,
    pub eigensystem: FloquetEigensystem,
    pub effective_hamiltonian: Operator,
}

fn floquet_eigensystem_from_dense(
    unitary: &[Complex64],
    dimension: usize,
    period: f64,
) -> Result<FloquetEigensystem> {
    let eigensystem = backend::complex_eigenpairs(unitary, dimension)?;
    let mut entries = Vec::with_capacity(dimension);
    for column in 0..dimension {
        let eigenvalue = eigensystem.eigenvalues[column];
        if (eigenvalue.norm() - 1.0).abs() > 1.0e-8 {
            return Err(QmbedError::NonConvergence {
                iterations: 1,
                residual: (eigenvalue.norm() - 1.0).abs(),
            });
        }
        let vector = eigensystem.eigenvectors[column].clone();
        let mut applied = vec![Complex64::new(0.0, 0.0); dimension];
        for row in 0..dimension {
            applied[row] = (0..dimension)
                .map(|inner| unitary[row * dimension + inner] * vector[inner])
                .sum();
        }
        let residual = applied
            .iter()
            .zip(&vector)
            .map(|(actual, component)| (*actual - eigenvalue * *component).norm_sqr())
            .sum::<f64>()
            .sqrt();
        entries.push((-eigenvalue.arg() / period, eigenvalue, vector, residual));
    }
    entries.sort_by(|left, right| left.0.total_cmp(&right.0));
    Ok(FloquetEigensystem {
        quasienergies: entries.iter().map(|entry| entry.0).collect(),
        eigenvalues: entries.iter().map(|entry| entry.1).collect(),
        eigenvectors: entries.iter().map(|entry| entry.2.clone()).collect(),
        residuals: entries.into_iter().map(|entry| entry.3).collect(),
    })
}

fn floquet_effective_hamiltonian(
    eigensystem: &FloquetEigensystem,
    dimension: usize,
    format: MatrixFormat,
) -> Result<Operator> {
    let mut values = vec![Complex64::new(0.0, 0.0); dimension * dimension];
    for (energy, vector) in eigensystem
        .quasienergies
        .iter()
        .zip(&eigensystem.eigenvectors)
    {
        for row in 0..dimension {
            for column in 0..dimension {
                values[row * dimension + column] += *energy * vector[row] * vector[column].conj();
            }
        }
    }
    Operator::from_dense(dimension, dimension, values)?.converted(format)
}

fn analyze_floquet_dense_unitary(
    dense_unitary: Vec<Complex64>,
    dimension: usize,
    period: f64,
    protocol_duration: f64,
    format: MatrixFormat,
) -> Result<FloquetAnalysis> {
    if !period.is_finite() || period <= 0.0 {
        return Err(QmbedError::InvalidOptions(
            "Floquet analyses require a finite positive period".into(),
        ));
    }
    let unitarity_residual = backend::unitarity_residual(&dense_unitary, dimension)?;
    let eigensystem = floquet_eigensystem_from_dense(&dense_unitary, dimension, period)?;
    let effective_hamiltonian = floquet_effective_hamiltonian(&eigensystem, dimension, format)?;
    Ok(FloquetAnalysis {
        period,
        protocol_duration,
        unitary: Operator::from_dense(dimension, dimension, dense_unitary)?.converted(format)?,
        unitarity_residual,
        eigensystem,
        effective_hamiltonian,
    })
}

/// Analyze an already-constructed one-period propagator.
///
/// This is the common boundary for continuous-time integrators, tensor-network
/// propagators, and externally supplied unitaries.
pub fn analyze_floquet_unitary(
    unitary: &dyn LinearOperator,
    period: f64,
    format: MatrixFormat,
) -> Result<FloquetAnalysis> {
    let (rows, columns) = unitary.shape();
    if rows != columns {
        return Err(QmbedError::DimensionMismatch(
            "a Floquet propagator must be square".into(),
        ));
    }
    analyze_floquet_dense_unitary(materialize_dense(unitary)?, rows, period, period, format)
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FloquetCoordinate {
    pub cycle: usize,
    pub within_cycle: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct FloquetTimeVector {
    period: f64,
    cycles: usize,
    points_per_cycle: usize,
    times: Vec<f64>,
}

impl FloquetTimeVector {
    pub fn new(
        period: f64,
        cycles: usize,
        points_per_cycle: usize,
        include_endpoint: bool,
    ) -> Result<Self> {
        if !period.is_finite() || period <= 0.0 || cycles == 0 || points_per_cycle == 0 {
            return Err(QmbedError::InvalidOptions(
                "Floquet time-vector controls must be positive".into(),
            ));
        }
        let points = cycles
            .checked_mul(points_per_cycle)
            .and_then(|value| value.checked_add(usize::from(include_endpoint)))
            .ok_or_else(|| QmbedError::InvalidOptions("Floquet time-vector overflow".into()))?;
        let step = period / points_per_cycle as f64;
        let times = (0..points).map(|index| index as f64 * step).collect();
        Ok(Self {
            period,
            cycles,
            points_per_cycle,
            times,
        })
    }

    /// Time grid for ramp-up, constant, and ramp-down Floquet stages.
    ///
    /// The grid starts at `-ramp_up_cycles * period`, includes the final
    /// endpoint, and retains a uniform number of points per period.
    pub fn staged(
        period: f64,
        ramp_up_cycles: usize,
        constant_cycles: usize,
        ramp_down_cycles: usize,
        points_per_cycle: usize,
    ) -> Result<Self> {
        let cycles = ramp_up_cycles
            .checked_add(constant_cycles)
            .and_then(|value| value.checked_add(ramp_down_cycles))
            .ok_or_else(|| QmbedError::InvalidOptions("Floquet cycle count overflow".into()))?;
        if !period.is_finite()
            || period <= 0.0
            || constant_cycles == 0
            || cycles == 0
            || points_per_cycle == 0
        {
            return Err(QmbedError::InvalidOptions(
                "Floquet staged-grid controls must be positive".into(),
            ));
        }
        let points = cycles
            .checked_mul(points_per_cycle)
            .and_then(|value| value.checked_add(1))
            .ok_or_else(|| QmbedError::InvalidOptions("Floquet time-vector overflow".into()))?;
        let step = period / points_per_cycle as f64;
        let start = -(ramp_up_cycles as f64) * period;
        let times = (0..points)
            .map(|index| start + index as f64 * step)
            .collect();
        Ok(Self {
            period,
            cycles,
            points_per_cycle,
            times,
        })
    }

    pub const fn period(&self) -> f64 {
        self.period
    }

    pub const fn cycles(&self) -> usize {
        self.cycles
    }

    pub const fn points_per_cycle(&self) -> usize {
        self.points_per_cycle
    }

    pub fn times(&self) -> &[f64] {
        &self.times
    }

    pub fn coordinate(&self, index: usize) -> Result<FloquetCoordinate> {
        if index >= self.times.len() {
            return Err(QmbedError::InvalidOptions(
                "Floquet time index is out of bounds".into(),
            ));
        }
        Ok(FloquetCoordinate {
            cycle: index / self.points_per_cycle,
            within_cycle: (index % self.points_per_cycle) as f64 * self.period
                / self.points_per_cycle as f64,
        })
    }
}

impl LinearOperator for Floquet {
    fn shape(&self) -> (usize, usize) {
        (self.dimension, self.dimension)
    }

    fn format(&self) -> MatrixFormat {
        MatrixFormat::MatrixFree
    }

    fn apply(&self, input: &[Complex64], output: &mut [Complex64]) -> Result<()> {
        self.apply_period(input, output)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct SpectrumOptions {
    pub frequencies: Vec<f64>,
    pub reference_energy: f64,
    pub broadening: f64,
    pub krylov_dimension: usize,
    pub tolerance: f64,
}

impl SpectrumOptions {
    fn validate(&self) -> Result<()> {
        if self.frequencies.is_empty()
            || self.frequencies.iter().any(|value| !value.is_finite())
            || !self.reference_energy.is_finite()
            || !self.broadening.is_finite()
            || self.broadening <= 0.0
            || self.krylov_dimension == 0
            || !self.tolerance.is_finite()
            || self.tolerance <= 0.0
        {
            return Err(QmbedError::InvalidOptions(
                "invalid spectrum frequency grid or numerical controls".into(),
            ));
        }
        Ok(())
    }
}

/// Lorentzian-broadened spectral density in a same or different target sector.
pub fn spectral_function<H, P>(
    target_hamiltonian: &H,
    source: &[Complex64],
    probe: &P,
    options: SpectrumOptions,
) -> Result<Vec<f64>>
where
    H: LinearOperator + ?Sized,
    P: LinearOperator + ?Sized,
{
    options.validate()?;
    let target_shape = target_hamiltonian.shape();
    let probe_shape = probe.shape();
    if target_shape.0 != target_shape.1
        || probe_shape.0 != target_shape.0
        || probe_shape.1 != source.len()
    {
        return Err(QmbedError::DimensionMismatch(
            "target Hamiltonian, source, and probe shapes are incompatible".into(),
        ));
    }
    let mut created = vec![Complex64::new(0.0, 0.0); probe_shape.0];
    probe.apply(source, &mut created)?;
    let (energies, weights) = if target_shape.0 <= 128 {
        let (energies, eigenvectors) = hermitian_eigenpairs_all(target_hamiltonian)?;
        let weights = eigenvectors
            .iter()
            .map(|vector| {
                vector
                    .iter()
                    .zip(&created)
                    .map(|(left, right)| left.conj() * *right)
                    .sum::<Complex64>()
                    .norm_sqr()
            })
            .collect();
        (energies, weights)
    } else {
        lanczos_spectral_measure(target_hamiltonian, &created, options.krylov_dimension)?
    };
    Ok(options
        .frequencies
        .iter()
        .map(|&frequency| {
            energies
                .iter()
                .zip(&weights)
                .map(|(&energy, &weight)| {
                    let detuning = frequency + options.reference_energy - energy;
                    weight * options.broadening
                        / (PI * (detuning * detuning + options.broadening.powi(2)))
                })
                .sum()
        })
        .collect())
}

/// Real-time two-point function `<psi|A(t) B(0)|psi>`.
pub fn dynamical_correlator<H, A, B>(
    hamiltonian: &H,
    state: &[Complex64],
    left_probe: &A,
    right_probe: &B,
    mut options: EvolutionOptions,
) -> Result<Vec<Complex64>>
where
    H: LinearOperator + ?Sized,
    A: LinearOperator + ?Sized,
    B: LinearOperator + ?Sized,
{
    let dimension = state.len();
    if hamiltonian.shape() != (dimension, dimension)
        || left_probe.shape() != (dimension, dimension)
        || right_probe.shape() != (dimension, dimension)
    {
        return Err(QmbedError::DimensionMismatch(
            "correlator Hamiltonian, probes, and state dimensions do not match".into(),
        ));
    }
    options.hamiltonian = true;
    let mut created = vec![Complex64::new(0.0, 0.0); dimension];
    right_probe.apply(state, &mut created)?;
    let reference = evolve(hamiltonian, state, options.clone())?;
    let excited = evolve(hamiltonian, &created, options)?;
    reference
        .states
        .iter()
        .zip(excited.states)
        .map(|(reference_state, excited_state)| {
            let mut probed = vec![Complex64::new(0.0, 0.0); dimension];
            left_probe.apply(&excited_state, &mut probed)?;
            Ok(reference_state
                .iter()
                .zip(probed)
                .map(|(left, right)| left.conj() * right)
                .sum())
        })
        .collect()
}
