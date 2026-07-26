use std::collections::HashMap;

use num_complex::Complex64;

use crate::basis::{BasisProjector, BinaryState};
use crate::operator::{LinearOperator, MatrixFormat, Operator};
use crate::solve::StateTrajectory;
use crate::{QmbedError, Result};

/// Gauge-independent finite-dimensional subspace.
#[derive(Clone, Debug)]
pub struct Subspace {
    ambient_dimension: usize,
    columns: Vec<Vec<Complex64>>,
}

impl Subspace {
    pub fn from_columns(
        ambient_dimension: usize,
        rank: usize,
        column_major_vectors: Vec<Complex64>,
    ) -> Result<Self> {
        if ambient_dimension == 0
            || rank == 0
            || column_major_vectors.len() != ambient_dimension.saturating_mul(rank)
        {
            return Err(QmbedError::DimensionMismatch(
                "subspace storage must contain ambient_dimension * rank entries".into(),
            ));
        }
        let mut columns: Vec<Vec<Complex64>> = Vec::with_capacity(rank);
        for column in 0..rank {
            let mut vector = column_major_vectors
                [column * ambient_dimension..(column + 1) * ambient_dimension]
                .to_vec();
            for previous in &columns {
                let overlap = inner(previous, &vector);
                for (value, basis_value) in vector.iter_mut().zip(previous) {
                    *value -= overlap * *basis_value;
                }
            }
            let norm = vector.iter().map(Complex64::norm_sqr).sum::<f64>().sqrt();
            if norm <= 1.0e-13 {
                return Err(QmbedError::RankDeficient);
            }
            for value in &mut vector {
                *value /= norm;
            }
            columns.push(vector);
        }
        Ok(Self {
            ambient_dimension,
            columns,
        })
    }

    pub const fn ambient_dimension(&self) -> usize {
        self.ambient_dimension
    }

    pub fn rank(&self) -> usize {
        self.columns.len()
    }

    pub fn columns(&self) -> &[Vec<Complex64>] {
        &self.columns
    }
}

fn inner(left: &[Complex64], right: &[Complex64]) -> Complex64 {
    left.iter()
        .zip(right)
        .map(|(left_value, right_value)| left_value.conj() * *right_value)
        .sum()
}

/// Mean squared principal-angle cosine between two subspaces.
pub fn subspace_fidelity(left: &Subspace, right: &Subspace) -> Result<f64> {
    if left.ambient_dimension != right.ambient_dimension {
        return Err(QmbedError::DimensionMismatch(
            "subspaces must share an ambient dimension".into(),
        ));
    }
    let denominator = left.rank().min(right.rank());
    if denominator == 0 {
        return Err(QmbedError::RankDeficient);
    }
    let overlap_norm: f64 = left
        .columns
        .iter()
        .flat_map(|left_vector| {
            right
                .columns
                .iter()
                .map(move |right_vector| inner(left_vector, right_vector).norm_sqr())
        })
        .sum();
    Ok((overlap_norm / denominator as f64).clamp(0.0, 1.0))
}

pub fn matrix_element(
    left: &[Complex64],
    operator: &(impl LinearOperator + ?Sized),
    right: &[Complex64],
) -> Result<Complex64> {
    let shape = operator.shape();
    if left.len() != shape.0 || right.len() != shape.1 {
        return Err(QmbedError::DimensionMismatch(
            "matrix-element vectors do not match the operator shape".into(),
        ));
    }
    let mut applied = vec![Complex64::new(0.0, 0.0); shape.0];
    operator.apply(right, &mut applied)?;
    Ok(inner(left, &applied))
}

pub fn expectation(
    operator: &(impl LinearOperator + ?Sized),
    state: &[Complex64],
) -> Result<Complex64> {
    matrix_element(state, operator, state)
}

/// Variance `||A psi||^2 - |<psi|A|psi>|^2` for a normalized state.
pub fn quantum_fluctuation(
    operator: &(impl LinearOperator + ?Sized),
    state: &[Complex64],
) -> Result<f64> {
    let shape = operator.shape();
    if shape.0 != shape.1 || state.len() != shape.0 {
        return Err(QmbedError::DimensionMismatch(
            "quantum fluctuation requires a square operator matching the state".into(),
        ));
    }
    let norm = inner(state, state).re;
    if !norm.is_finite() || norm <= f64::EPSILON {
        return Err(QmbedError::InvalidOptions(
            "state must have positive finite norm".into(),
        ));
    }
    let mut applied = vec![Complex64::new(0.0, 0.0); state.len()];
    operator.apply(state, &mut applied)?;
    let mean = inner(state, &applied) / norm;
    let second = inner(&applied, &applied).re / norm;
    Ok((second - mean.norm_sqr()).max(0.0))
}

/// Raw pure-state fluctuation
/// `⟨ψ|A²|ψ⟩ - ⟨ψ|A|ψ⟩²` without normalizing `ψ`.
///
/// This preserves array-oriented compatibility semantics, including the zero
/// vector, while [`quantum_fluctuation`] remains the strict normalized-state
/// variance API.
pub fn raw_quantum_fluctuation(
    operator: &(impl LinearOperator + ?Sized),
    state: &[Complex64],
) -> Result<Complex64> {
    let shape = operator.shape();
    if shape.0 != shape.1 || state.len() != shape.0 {
        return Err(QmbedError::DimensionMismatch(
            "raw quantum fluctuation requires a square operator matching the state".into(),
        ));
    }
    let mut applied = vec![Complex64::new(0.0, 0.0); state.len()];
    let mut applied_twice = vec![Complex64::new(0.0, 0.0); state.len()];
    operator.apply(state, &mut applied)?;
    operator.apply(&applied, &mut applied_twice)?;
    let mean = inner(state, &applied);
    Ok(inner(state, &applied_twice) - mean * mean)
}

/// Reduced density matrix of the first factor of a pure bipartite state.
pub fn partial_trace(
    state: &[Complex64],
    subsystem_dimension: usize,
    environment_dimension: usize,
) -> Result<Vec<Complex64>> {
    if subsystem_dimension == 0
        || environment_dimension == 0
        || state.len()
            != subsystem_dimension
                .checked_mul(environment_dimension)
                .ok_or_else(|| {
                    QmbedError::DimensionMismatch("tensor-product dimension overflow".into())
                })?
    {
        return Err(QmbedError::DimensionMismatch(
            "state length must equal subsystem_dimension * environment_dimension".into(),
        ));
    }
    let norm = state.iter().map(Complex64::norm_sqr).sum::<f64>();
    if !norm.is_finite() || norm <= f64::EPSILON {
        return Err(QmbedError::InvalidOptions(
            "state must have positive finite norm".into(),
        ));
    }
    let mut density = vec![Complex64::new(0.0, 0.0); subsystem_dimension * subsystem_dimension];
    for left in 0..subsystem_dimension {
        for right in 0..subsystem_dimension {
            density[left * subsystem_dimension + right] = (0..environment_dimension)
                .map(|environment| {
                    state[left * environment_dimension + environment]
                        * state[right * environment_dimension + environment].conj()
                        / norm
                })
                .sum();
        }
    }
    Ok(density)
}

/// Reduced density matrix of the first factor for a row-major mixed state.
pub fn partial_trace_density(
    density: &[Complex64],
    subsystem_dimension: usize,
    environment_dimension: usize,
) -> Result<Vec<Complex64>> {
    let dimension = subsystem_dimension
        .checked_mul(environment_dimension)
        .ok_or_else(|| QmbedError::DimensionMismatch("tensor-product dimension overflow".into()))?;
    if subsystem_dimension == 0
        || environment_dimension == 0
        || density.len() != dimension.saturating_mul(dimension)
    {
        return Err(QmbedError::DimensionMismatch(
            "density shape must match the bipartite Hilbert space".into(),
        ));
    }
    let trace: Complex64 = (0..dimension)
        .map(|index| density[index * dimension + index])
        .sum();
    if trace.im.abs() > 1.0e-10 || !trace.re.is_finite() || trace.re <= f64::EPSILON {
        return Err(QmbedError::InvalidOptions(
            "density matrix must have a positive real trace".into(),
        ));
    }
    for row in 0..dimension {
        for column in 0..dimension {
            if (density[row * dimension + column] - density[column * dimension + row].conj()).norm()
                > 1.0e-10
            {
                return Err(QmbedError::InvalidOptions(
                    "density matrix must be Hermitian".into(),
                ));
            }
        }
    }
    let mut reduced = vec![Complex64::new(0.0, 0.0); subsystem_dimension * subsystem_dimension];
    for left in 0..subsystem_dimension {
        for right in 0..subsystem_dimension {
            for environment in 0..environment_dimension {
                let row = left * environment_dimension + environment;
                let column = right * environment_dimension + environment;
                reduced[left * subsystem_dimension + right] +=
                    density[row * dimension + column] / trace.re;
            }
        }
    }
    Ok(reduced)
}

#[derive(Clone, Debug)]
struct SubsystemIndexMap {
    full_dimension: usize,
    subsystem_dimension: usize,
    environment_dimension: usize,
    subsystem_indices: Vec<usize>,
    environment_indices: Vec<usize>,
}

fn subsystem_index_map(
    local_dimensions: &[usize],
    retained_sites: &[usize],
) -> Result<SubsystemIndexMap> {
    if local_dimensions.is_empty() || local_dimensions.contains(&0) {
        return Err(QmbedError::InvalidOptions(
            "local dimensions must be a nonempty list of positive values".into(),
        ));
    }
    let mut retained = vec![false; local_dimensions.len()];
    for &site in retained_sites {
        if site >= local_dimensions.len() {
            return Err(QmbedError::InvalidSite {
                site,
                sites: local_dimensions.len(),
            });
        }
        if std::mem::replace(&mut retained[site], true) {
            return Err(QmbedError::InvalidOptions(
                "retained subsystem sites must be unique".into(),
            ));
        }
    }
    let product = |sites: &[usize]| -> Result<usize> {
        sites.iter().try_fold(1_usize, |dimension, &site| {
            dimension
                .checked_mul(local_dimensions[site])
                .ok_or_else(|| QmbedError::DimensionMismatch("Hilbert-space size overflow".into()))
        })
    };
    let environment_sites: Vec<_> = (0..local_dimensions.len())
        .filter(|site| !retained[*site])
        .collect();
    let subsystem_dimension = product(retained_sites)?;
    let environment_dimension = product(&environment_sites)?;
    let full_dimension = subsystem_dimension
        .checked_mul(environment_dimension)
        .ok_or_else(|| QmbedError::DimensionMismatch("Hilbert-space size overflow".into()))?;

    let mut subsystem_indices = vec![0; full_dimension];
    let mut environment_indices = vec![0; full_dimension];
    for global in 0..full_dimension {
        let mut value = global;
        let mut digits = Vec::with_capacity(local_dimensions.len());
        for &dimension in local_dimensions {
            digits.push(value % dimension);
            value /= dimension;
        }
        let mut stride = 1;
        for &site in retained_sites {
            subsystem_indices[global] += digits[site] * stride;
            stride *= local_dimensions[site];
        }
        stride = 1;
        for &site in &environment_sites {
            environment_indices[global] += digits[site] * stride;
            stride *= local_dimensions[site];
        }
    }
    Ok(SubsystemIndexMap {
        full_dimension,
        subsystem_dimension,
        environment_dimension,
        subsystem_indices,
        environment_indices,
    })
}

/// Tensor-product dimensions selected by an arbitrary retained site set.
pub fn subsystem_dimensions(
    local_dimensions: &[usize],
    retained_sites: &[usize],
) -> Result<(usize, usize)> {
    let layout = subsystem_index_map(local_dimensions, retained_sites)?;
    Ok((layout.subsystem_dimension, layout.environment_dimension))
}

/// Fock-space signs induced by grouping selected fermionic modes into a
/// subsystem. The returned phase is indexed by the original packed state.
///
/// Modes inside the environment and subsystem keep the orders supplied by the
/// ordinary subsystem layout; only the swaps needed to group the two factors
/// contribute a sign.
pub fn fermionic_subsystem_phases(
    local_dimensions: &[usize],
    retained_sites: &[usize],
) -> Result<Vec<f64>> {
    let all_sites = (0..local_dimensions.len()).collect::<Vec<_>>();
    noncommuting_subsystem_phases(local_dimensions, retained_sites, &[all_sites])
}

/// One disjoint set of binary modes sharing an exchange phase.
#[derive(Clone, Debug, PartialEq)]
pub struct NoncommutingGroup {
    sites: Vec<usize>,
    exchange_phase: Complex64,
}

impl NoncommutingGroup {
    pub fn new(sites: impl Into<Vec<usize>>, exchange_phase: impl Into<Complex64>) -> Result<Self> {
        let exchange_phase = exchange_phase.into();
        if !exchange_phase.re.is_finite()
            || !exchange_phase.im.is_finite()
            || (exchange_phase.norm() - 1.0).abs() > 1.0e-12
        {
            return Err(QmbedError::InvalidOptions(
                "a noncommuting exchange phase must be finite and have unit magnitude".into(),
            ));
        }
        Ok(Self {
            sites: sites.into(),
            exchange_phase,
        })
    }

    pub fn sites(&self) -> &[usize] {
        &self.sites
    }

    pub const fn exchange_phase(&self) -> Complex64 {
        self.exchange_phase
    }
}

/// Exchange phases induced by regrouping a tensor product with selected
/// mutually anticommuting site groups.
///
/// Sites outside these groups commute. Distinct groups also commute with one
/// another, while occupied binary sites inside the same group contribute one
/// minus sign for every order inversion. This is the minimal generalization
/// needed for mixed spin/boson/fermion user bases without assigning global
/// fermionic statistics to the entire product space.
pub fn noncommuting_subsystem_phases(
    local_dimensions: &[usize],
    retained_sites: &[usize],
    noncommuting_groups: &[Vec<usize>],
) -> Result<Vec<f64>> {
    let groups = noncommuting_groups
        .iter()
        .map(|sites| NoncommutingGroup::new(sites.clone(), Complex64::new(-1.0, 0.0)))
        .collect::<Result<Vec<_>>>()?;
    noncommuting_subsystem_exchange_phases(local_dimensions, retained_sites, &groups)
        .map(|phases| phases.into_iter().map(|phase| phase.re).collect::<Vec<_>>())
}

/// General exchange phases induced by regrouping disjoint binary-mode groups.
///
/// Each inversion inside a group contributes that group's unit-modulus phase.
/// Distinct groups and sites outside all groups commute.
pub fn noncommuting_subsystem_exchange_phases(
    local_dimensions: &[usize],
    retained_sites: &[usize],
    noncommuting_groups: &[NoncommutingGroup],
) -> Result<Vec<Complex64>> {
    let layout = subsystem_index_map(local_dimensions, retained_sites)?;
    let mut retained = vec![false; local_dimensions.len()];
    for &site in retained_sites {
        retained[site] = true;
    }
    let mut new_order = (0..local_dimensions.len())
        .filter(|site| !retained[*site])
        .collect::<Vec<_>>();
    new_order.extend_from_slice(retained_sites);
    let mut new_position = vec![0; local_dimensions.len()];
    for (position, &site) in new_order.iter().enumerate() {
        new_position[site] = position;
    }
    let mut group_for_site = vec![None; local_dimensions.len()];
    for (group, noncommuting_group) in noncommuting_groups.iter().enumerate() {
        let mut seen = std::collections::HashSet::with_capacity(noncommuting_group.sites().len());
        for &site in noncommuting_group.sites() {
            if site >= local_dimensions.len() {
                return Err(QmbedError::InvalidSite {
                    site,
                    sites: local_dimensions.len(),
                });
            }
            if local_dimensions[site] != 2 {
                return Err(QmbedError::InvalidOptions(format!(
                    "noncommuting site {site} must have binary local dimension"
                )));
            }
            if !seen.insert(site) {
                return Err(QmbedError::InvalidOptions(
                    "a noncommuting group cannot repeat a site".into(),
                ));
            }
            if group_for_site[site].replace(group).is_some() {
                return Err(QmbedError::InvalidOptions(
                    "noncommuting groups must be disjoint".into(),
                ));
            }
        }
    }
    let mut strides = Vec::with_capacity(local_dimensions.len());
    let mut stride = 1_usize;
    for &dimension in local_dimensions {
        strides.push(stride);
        stride = stride
            .checked_mul(dimension)
            .ok_or_else(|| QmbedError::DimensionMismatch("Hilbert-space size overflow".into()))?;
    }

    Ok((0..layout.full_dimension)
        .map(|state| {
            let mut inversions = vec![0_u32; noncommuting_groups.len()];
            for left in 0..local_dimensions.len() {
                let Some(group) = group_for_site[left] else {
                    continue;
                };
                if (state / strides[left]) % local_dimensions[left] == 0 {
                    continue;
                }
                for right in (left + 1)..local_dimensions.len() {
                    if group_for_site[right] == Some(group)
                        && (state / strides[right]) % local_dimensions[right] != 0
                        && new_position[left] > new_position[right]
                    {
                        inversions[group] += 1;
                    }
                }
            }
            inversions
                .into_iter()
                .zip(noncommuting_groups)
                .fold(Complex64::new(1.0, 0.0), |phase, (count, group)| {
                    phase * group.exchange_phase().powu(count)
                })
        })
        .collect())
}

/// Apply arbitrary selected exchange phases to a pure state.
pub fn apply_noncommuting_subsystem_exchange_phases(
    state: &mut [Complex64],
    local_dimensions: &[usize],
    retained_sites: &[usize],
    noncommuting_groups: &[NoncommutingGroup],
) -> Result<()> {
    let phases = noncommuting_subsystem_exchange_phases(
        local_dimensions,
        retained_sites,
        noncommuting_groups,
    )?;
    if state.len() != phases.len() {
        return Err(QmbedError::DimensionMismatch(
            "state length does not match the noncommuting subsystem layout".into(),
        ));
    }
    for (value, phase) in state.iter_mut().zip(phases) {
        *value *= phase;
    }
    Ok(())
}

/// Apply selected noncommuting-group exchange phases to a pure state.
pub fn apply_noncommuting_subsystem_phases(
    state: &mut [Complex64],
    local_dimensions: &[usize],
    retained_sites: &[usize],
    noncommuting_groups: &[Vec<usize>],
) -> Result<()> {
    let phases =
        noncommuting_subsystem_phases(local_dimensions, retained_sites, noncommuting_groups)?;
    if state.len() != phases.len() {
        return Err(QmbedError::DimensionMismatch(
            "state length does not match the noncommuting subsystem layout".into(),
        ));
    }
    for (value, phase) in state.iter_mut().zip(phases) {
        *value *= phase;
    }
    Ok(())
}

/// Apply fermionic subsystem-ordering phases to a packed pure state.
pub fn apply_fermionic_subsystem_phases(
    state: &mut [Complex64],
    local_dimensions: &[usize],
    retained_sites: &[usize],
) -> Result<()> {
    let phases = fermionic_subsystem_phases(local_dimensions, retained_sites)?;
    if state.len() != phases.len() {
        return Err(QmbedError::DimensionMismatch(
            "state length does not match the fermionic subsystem layout".into(),
        ));
    }
    for (value, phase) in state.iter_mut().zip(phases) {
        *value *= phase;
    }
    Ok(())
}

/// Apply selected noncommuting-group exchange phases to both density axes.
pub fn apply_noncommuting_subsystem_phases_density(
    density: &mut [Complex64],
    local_dimensions: &[usize],
    retained_sites: &[usize],
    noncommuting_groups: &[Vec<usize>],
) -> Result<()> {
    let phases =
        noncommuting_subsystem_phases(local_dimensions, retained_sites, noncommuting_groups)?;
    let dimension = phases.len();
    if density.len() != dimension.saturating_mul(dimension) {
        return Err(QmbedError::DimensionMismatch(
            "density matrix does not match the noncommuting subsystem layout".into(),
        ));
    }
    for row in 0..dimension {
        for column in 0..dimension {
            density[row * dimension + column] *= phases[row] * phases[column];
        }
    }
    Ok(())
}

/// Apply arbitrary selected exchange phases to both density-matrix axes.
pub fn apply_noncommuting_subsystem_exchange_phases_density(
    density: &mut [Complex64],
    local_dimensions: &[usize],
    retained_sites: &[usize],
    noncommuting_groups: &[NoncommutingGroup],
) -> Result<()> {
    let phases = noncommuting_subsystem_exchange_phases(
        local_dimensions,
        retained_sites,
        noncommuting_groups,
    )?;
    let dimension = phases.len();
    if density.len() != dimension.saturating_mul(dimension) {
        return Err(QmbedError::DimensionMismatch(
            "density matrix does not match the noncommuting subsystem layout".into(),
        ));
    }
    for row in 0..dimension {
        for column in 0..dimension {
            density[row * dimension + column] *= phases[row] * phases[column].conj();
        }
    }
    Ok(())
}

/// Apply the same fermionic mode permutation to both density-matrix axes.
pub fn apply_fermionic_subsystem_phases_density(
    density: &mut [Complex64],
    local_dimensions: &[usize],
    retained_sites: &[usize],
) -> Result<()> {
    let phases = fermionic_subsystem_phases(local_dimensions, retained_sites)?;
    let dimension = phases.len();
    if density.len() != dimension.saturating_mul(dimension) {
        return Err(QmbedError::DimensionMismatch(
            "density matrix does not match the fermionic subsystem layout".into(),
        ));
    }
    for row in 0..dimension {
        for column in 0..dimension {
            density[row * dimension + column] *= phases[row] * phases[column];
        }
    }
    Ok(())
}

/// Reduced density matrix for an arbitrary retained set of mixed-radix sites.
/// Site zero is the least-significant local digit, matching the basis encodings.
pub fn partial_trace_subsystem(
    state: &[Complex64],
    local_dimensions: &[usize],
    retained_sites: &[usize],
) -> Result<Vec<Complex64>> {
    let layout = subsystem_index_map(local_dimensions, retained_sites)?;
    if state.len() != layout.full_dimension {
        return Err(QmbedError::DimensionMismatch(
            "state length does not match the product of local dimensions".into(),
        ));
    }
    let norm = state.iter().map(Complex64::norm_sqr).sum::<f64>();
    if !norm.is_finite() || norm <= f64::EPSILON {
        return Err(QmbedError::InvalidOptions(
            "state must have positive finite norm".into(),
        ));
    }
    let mut amplitudes = vec![Complex64::new(0.0, 0.0); layout.full_dimension];
    for (global, &value) in state.iter().enumerate() {
        let subsystem = layout.subsystem_indices[global];
        let environment = layout.environment_indices[global];
        amplitudes[subsystem * layout.environment_dimension + environment] = value;
    }
    let mut reduced =
        vec![Complex64::new(0.0, 0.0); layout.subsystem_dimension * layout.subsystem_dimension];
    for left in 0..layout.subsystem_dimension {
        for right in 0..layout.subsystem_dimension {
            reduced[left * layout.subsystem_dimension + right] = (0..layout.environment_dimension)
                .map(|environment| {
                    amplitudes[left * layout.environment_dimension + environment]
                        * amplitudes[right * layout.environment_dimension + environment].conj()
                        / norm
                })
                .sum();
        }
    }
    Ok(reduced)
}

/// Mixed-state partial trace for an arbitrary retained site set.
pub fn partial_trace_density_subsystem(
    density: &[Complex64],
    local_dimensions: &[usize],
    retained_sites: &[usize],
) -> Result<Vec<Complex64>> {
    let layout = subsystem_index_map(local_dimensions, retained_sites)?;
    if density.len() != layout.full_dimension.saturating_mul(layout.full_dimension) {
        return Err(QmbedError::DimensionMismatch(
            "density shape does not match the product of local dimensions".into(),
        ));
    }
    let trace: Complex64 = (0..layout.full_dimension)
        .map(|index| density[index * layout.full_dimension + index])
        .sum();
    if trace.im.abs() > 1.0e-10 || !trace.re.is_finite() || trace.re <= f64::EPSILON {
        return Err(QmbedError::InvalidOptions(
            "density matrix must have a positive real trace".into(),
        ));
    }
    for row in 0..layout.full_dimension {
        for column in 0..layout.full_dimension {
            if (density[row * layout.full_dimension + column]
                - density[column * layout.full_dimension + row].conj())
            .norm()
                > 1.0e-10
            {
                return Err(QmbedError::InvalidOptions(
                    "density matrix must be Hermitian".into(),
                ));
            }
        }
    }
    let mut reduced =
        vec![Complex64::new(0.0, 0.0); layout.subsystem_dimension * layout.subsystem_dimension];
    for row in 0..layout.full_dimension {
        for column in 0..layout.full_dimension {
            if layout.environment_indices[row] == layout.environment_indices[column] {
                let reduced_row = layout.subsystem_indices[row];
                let reduced_column = layout.subsystem_indices[column];
                reduced[reduced_row * layout.subsystem_dimension + reduced_column] +=
                    density[row * layout.full_dimension + column] / trace.re;
            }
        }
    }
    Ok(reduced)
}

#[derive(Clone, Debug)]
struct BinarySectorSubsystemLayout {
    total_sites: usize,
    retained_sites: Vec<usize>,
    environment_sites: Vec<usize>,
    subsystem_dimension: usize,
    group_for_site: Vec<Option<usize>>,
    new_position: Vec<usize>,
}

#[derive(Clone, Debug)]
struct SplitBinarySectorState {
    subsystem: usize,
    environment: Vec<u64>,
    exchange_phase: Complex64,
}

fn binary_sector_subsystem_layout(
    total_sites: usize,
    retained_sites: &[usize],
    noncommuting_groups: &[NoncommutingGroup],
) -> Result<BinarySectorSubsystemLayout> {
    if total_sites == 0 {
        return Err(QmbedError::InvalidOptions(
            "a binary subsystem requires at least one site".into(),
        ));
    }
    let mut retained = vec![false; total_sites];
    for &site in retained_sites {
        if site >= total_sites {
            return Err(QmbedError::InvalidSite {
                site,
                sites: total_sites,
            });
        }
        if std::mem::replace(&mut retained[site], true) {
            return Err(QmbedError::InvalidOptions(
                "retained subsystem sites must be unique".into(),
            ));
        }
    }
    let subsystem_dimension =
        1_usize
            .checked_shl(u32::try_from(retained_sites.len()).map_err(|_| {
                QmbedError::DimensionMismatch("retained subsystem is too large".into())
            })?)
            .ok_or_else(|| {
                QmbedError::DimensionMismatch(
                    "retained subsystem is too large for a dense reduced density matrix".into(),
                )
            })?;
    subsystem_dimension
        .checked_mul(subsystem_dimension)
        .ok_or_else(|| {
            QmbedError::DimensionMismatch("reduced density-matrix allocation would overflow".into())
        })?;

    let environment_sites = (0..total_sites)
        .filter(|site| !retained[*site])
        .collect::<Vec<_>>();
    let mut new_order = environment_sites.clone();
    new_order.extend_from_slice(retained_sites);
    let mut new_position = vec![0; total_sites];
    for (position, &site) in new_order.iter().enumerate() {
        new_position[site] = position;
    }
    let mut group_for_site = vec![None; total_sites];
    for (group, noncommuting_group) in noncommuting_groups.iter().enumerate() {
        let mut seen = std::collections::HashSet::with_capacity(noncommuting_group.sites().len());
        for &site in noncommuting_group.sites() {
            if site >= total_sites {
                return Err(QmbedError::InvalidSite {
                    site,
                    sites: total_sites,
                });
            }
            if !seen.insert(site) {
                return Err(QmbedError::InvalidOptions(
                    "a noncommuting group cannot repeat a site".into(),
                ));
            }
            if group_for_site[site].replace(group).is_some() {
                return Err(QmbedError::InvalidOptions(
                    "noncommuting groups must be disjoint".into(),
                ));
            }
        }
    }
    Ok(BinarySectorSubsystemLayout {
        total_sites,
        retained_sites: retained_sites.to_vec(),
        environment_sites,
        subsystem_dimension,
        group_for_site,
        new_position,
    })
}

fn split_binary_sector_state<State>(
    state: State,
    layout: &BinarySectorSubsystemLayout,
    noncommuting_groups: &[NoncommutingGroup],
) -> Result<SplitBinarySectorState>
where
    State: BinaryState,
{
    let mut subsystem = 0_usize;
    for (position, &site) in layout.retained_sites.iter().enumerate() {
        if state.bit(site)? {
            subsystem |= 1_usize << position;
        }
    }
    let mut environment = vec![0_u64; layout.environment_sites.len().div_ceil(64)];
    for (position, &site) in layout.environment_sites.iter().enumerate() {
        if state.bit(site)? {
            environment[position / 64] |= 1_u64 << (position % 64);
        }
    }
    let mut inversions = vec![0_u32; noncommuting_groups.len()];
    for left in 0..layout.total_sites {
        let Some(group) = layout.group_for_site[left] else {
            continue;
        };
        if !state.bit(left)? {
            continue;
        }
        for right in (left + 1)..layout.total_sites {
            if layout.group_for_site[right] == Some(group)
                && state.bit(right)?
                && layout.new_position[left] > layout.new_position[right]
            {
                inversions[group] += 1;
            }
        }
    }
    let exchange_phase = inversions
        .into_iter()
        .zip(noncommuting_groups)
        .fold(Complex64::new(1.0, 0.0), |phase, (count, group)| {
            phase * group.exchange_phase().powu(count)
        });
    Ok(SplitBinarySectorState {
        subsystem,
        environment,
        exchange_phase,
    })
}

fn split_binary_sector_states<State>(
    states: &[State],
    layout: &BinarySectorSubsystemLayout,
    noncommuting_groups: &[NoncommutingGroup],
) -> Result<Vec<SplitBinarySectorState>>
where
    State: BinaryState,
{
    let mut seen = std::collections::HashSet::with_capacity(states.len());
    states
        .iter()
        .copied()
        .map(|state| {
            if !seen.insert(state) {
                return Err(QmbedError::InvalidOptions(
                    "sector basis states must be unique".into(),
                ));
            }
            split_binary_sector_state(state, layout, noncommuting_groups)
        })
        .collect()
}

fn zero_complex_square(dimension: usize) -> Result<Vec<Complex64>> {
    let length = dimension.checked_mul(dimension).ok_or_else(|| {
        QmbedError::DimensionMismatch("reduced density-matrix allocation would overflow".into())
    })?;
    let mut values = Vec::new();
    values.try_reserve_exact(length).map_err(|_| {
        QmbedError::UnsupportedBackend(format!(
            "a dense {dimension}x{dimension} reduced density matrix does not fit in memory"
        ))
    })?;
    values.resize(length, Complex64::new(0.0, 0.0));
    Ok(values)
}

/// Reduced density matrix of a pure state stored in an arbitrary binary
/// sector basis.
///
/// The contraction groups amplitudes by the environment bit pattern.  It
/// never enumerates or allocates the unrestricted `2^total_sites` parent
/// space; memory is proportional to the supplied sector and the requested
/// dense subsystem matrix.
pub fn partial_trace_sector_state<State>(
    amplitudes: &[Complex64],
    basis_states: &[State],
    total_sites: usize,
    retained_sites: &[usize],
    noncommuting_groups: &[NoncommutingGroup],
) -> Result<Vec<Complex64>>
where
    State: BinaryState,
{
    if amplitudes.len() != basis_states.len() {
        return Err(QmbedError::DimensionMismatch(
            "sector amplitudes and basis states must have the same length".into(),
        ));
    }
    let norm = amplitudes.iter().map(Complex64::norm_sqr).sum::<f64>();
    if !norm.is_finite() || norm <= f64::EPSILON {
        return Err(QmbedError::InvalidOptions(
            "state must have positive finite norm".into(),
        ));
    }
    let layout = binary_sector_subsystem_layout(total_sites, retained_sites, noncommuting_groups)?;
    let split = split_binary_sector_states(basis_states, &layout, noncommuting_groups)?;
    let mut by_environment = std::collections::HashMap::<Vec<u64>, Vec<(usize, Complex64)>>::new();
    for (state, amplitude) in split.iter().zip(amplitudes) {
        by_environment
            .entry(state.environment.clone())
            .or_default()
            .push((state.subsystem, state.exchange_phase * *amplitude));
    }
    let mut reduced = zero_complex_square(layout.subsystem_dimension)?;
    for amplitudes in by_environment.values() {
        for &(left, left_value) in amplitudes {
            for &(right, right_value) in amplitudes {
                reduced[left * layout.subsystem_dimension + right] +=
                    left_value * right_value.conj() / norm;
            }
        }
    }
    Ok(reduced)
}

/// Reduced density matrix of a mixed state stored in an arbitrary binary
/// sector basis.
///
/// Only pairs with the same environment key are visited.  The input remains a
/// dense matrix in the selected sector, but no unrestricted parent density
/// matrix is formed.
pub fn partial_trace_sector_density<State>(
    density: &[Complex64],
    basis_states: &[State],
    total_sites: usize,
    retained_sites: &[usize],
    noncommuting_groups: &[NoncommutingGroup],
) -> Result<Vec<Complex64>>
where
    State: BinaryState,
{
    let dimension = basis_states.len();
    if density.len() != dimension.saturating_mul(dimension) {
        return Err(QmbedError::DimensionMismatch(
            "sector density matrix does not match the basis dimension".into(),
        ));
    }
    let trace = (0..dimension)
        .map(|index| density[index * dimension + index])
        .sum::<Complex64>();
    if trace.im.abs() > 1.0e-10 || !trace.re.is_finite() || trace.re <= f64::EPSILON {
        return Err(QmbedError::InvalidOptions(
            "density matrix must have a positive real trace".into(),
        ));
    }
    for row in 0..dimension {
        for column in 0..dimension {
            if (density[row * dimension + column] - density[column * dimension + row].conj()).norm()
                > 1.0e-10
            {
                return Err(QmbedError::InvalidOptions(
                    "density matrix must be Hermitian".into(),
                ));
            }
        }
    }
    let layout = binary_sector_subsystem_layout(total_sites, retained_sites, noncommuting_groups)?;
    let split = split_binary_sector_states(basis_states, &layout, noncommuting_groups)?;
    let mut rows_by_environment =
        std::collections::HashMap::<Vec<u64>, Vec<usize>>::with_capacity(dimension);
    for (row, state) in split.iter().enumerate() {
        rows_by_environment
            .entry(state.environment.clone())
            .or_default()
            .push(row);
    }
    let mut reduced = zero_complex_square(layout.subsystem_dimension)?;
    for rows in rows_by_environment.values() {
        for &row in rows {
            for &column in rows {
                let left = &split[row];
                let right = &split[column];
                reduced[left.subsystem * layout.subsystem_dimension + right.subsystem] += left
                    .exchange_phase
                    * density[row * dimension + column]
                    * right.exchange_phase.conj()
                    / trace.re;
            }
        }
    }
    Ok(reduced)
}

/// Entanglement spectrum of a pure state represented in a binary sector.
pub fn entanglement_spectrum_sector<State>(
    amplitudes: &[Complex64],
    basis_states: &[State],
    total_sites: usize,
    retained_sites: &[usize],
    noncommuting_groups: &[NoncommutingGroup],
) -> Result<Vec<f64>>
where
    State: BinaryState,
{
    let dimension = 1_usize
        .checked_shl(u32::try_from(retained_sites.len()).unwrap_or(u32::MAX))
        .ok_or_else(|| {
            QmbedError::DimensionMismatch(
                "retained subsystem is too large for a dense entanglement spectrum".into(),
            )
        })?;
    density_matrix_spectrum(
        partial_trace_sector_state(
            amplitudes,
            basis_states,
            total_sites,
            retained_sites,
            noncommuting_groups,
        )?,
        dimension,
    )
}

/// Entanglement entropy of a pure state represented in a binary sector.
pub fn entanglement_entropy_sector<State>(
    amplitudes: &[Complex64],
    basis_states: &[State],
    total_sites: usize,
    retained_sites: &[usize],
    noncommuting_groups: &[NoncommutingGroup],
    order: EntropyOrder,
) -> Result<f64>
where
    State: BinaryState,
{
    entropy_from_spectrum(
        &entanglement_spectrum_sector(
            amplitudes,
            basis_states,
            total_sites,
            retained_sites,
            noncommuting_groups,
        )?,
        order,
    )
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum EntropyOrder {
    VonNeumann,
    Renyi(f64),
}

/// Entropy of an already validated density-matrix spectrum.
pub fn entropy_from_spectrum(probabilities: &[f64], order: EntropyOrder) -> Result<f64> {
    match order {
        EntropyOrder::VonNeumann => Ok(-probabilities
            .iter()
            .copied()
            .filter(|probability| *probability > f64::EPSILON)
            .map(|probability| probability * probability.ln())
            .sum::<f64>()),
        EntropyOrder::Renyi(alpha)
            if alpha.is_finite() && alpha > 0.0 && (alpha - 1.0).abs() > 1.0e-12 =>
        {
            Ok(probabilities
                .iter()
                .copied()
                .map(|probability| probability.powf(alpha))
                .sum::<f64>()
                .ln()
                / (1.0 - alpha))
        }
        EntropyOrder::Renyi(_) => Err(QmbedError::InvalidOptions(
            "Renyi order must be positive, finite, and different from one".into(),
        )),
    }
}

pub fn entanglement_entropy(
    state: &[Complex64],
    subsystem_dimension: usize,
    environment_dimension: usize,
    order: EntropyOrder,
) -> Result<f64> {
    if matches!(order, EntropyOrder::Renyi(alpha) if !alpha.is_finite() || alpha <= 0.0 || (alpha - 1.0).abs() <= 1.0e-12)
    {
        return Err(QmbedError::InvalidOptions(
            "Renyi order must be positive, finite, and different from one".into(),
        ));
    }
    let probabilities = density_matrix_spectrum(
        partial_trace(state, subsystem_dimension, environment_dimension)?,
        subsystem_dimension,
    )?;
    entropy_from_spectrum(&probabilities, order)
}

pub fn entanglement_spectrum_subsystem(
    state: &[Complex64],
    local_dimensions: &[usize],
    retained_sites: &[usize],
) -> Result<Vec<f64>> {
    let layout = subsystem_index_map(local_dimensions, retained_sites)?;
    density_matrix_spectrum(
        partial_trace_subsystem(state, local_dimensions, retained_sites)?,
        layout.subsystem_dimension,
    )
}

/// Canonical Schmidt spectrum for a pure bipartition.
///
/// A pure state has the same nonzero reduced spectrum on both sides of a
/// bipartition. Computing two density-matrix decompositions independently is
/// both wasteful and capable of producing last-bit entropy differences. This
/// routine chooses the smaller factor, with a lexicographic tie-break on
/// sorted site sets, so complementary calls take the identical numerical
/// path. Optional noncommuting groups are reordered against that same
/// canonical factor before the spectrum is evaluated.
pub fn canonical_schmidt_spectrum_subsystem(
    state: &[Complex64],
    local_dimensions: &[usize],
    retained_sites: &[usize],
    noncommuting_groups: &[Vec<usize>],
) -> Result<Vec<f64>> {
    let groups = noncommuting_groups
        .iter()
        .map(|sites| NoncommutingGroup::new(sites.clone(), Complex64::new(-1.0, 0.0)))
        .collect::<Result<Vec<_>>>()?;
    canonical_schmidt_spectrum_subsystem_with_exchange_phases(
        state,
        local_dimensions,
        retained_sites,
        &groups,
    )
}

pub fn canonical_schmidt_spectrum_subsystem_with_exchange_phases(
    state: &[Complex64],
    local_dimensions: &[usize],
    retained_sites: &[usize],
    noncommuting_groups: &[NoncommutingGroup],
) -> Result<Vec<f64>> {
    let layout = subsystem_index_map(local_dimensions, retained_sites)?;
    let mut retained = retained_sites.to_vec();
    retained.sort_unstable();
    let mut selected = vec![false; local_dimensions.len()];
    for &site in &retained {
        selected[site] = true;
    }
    let environment = (0..local_dimensions.len())
        .filter(|site| !selected[*site])
        .collect::<Vec<_>>();
    let canonical_sites = if layout.subsystem_dimension < layout.environment_dimension
        || (layout.subsystem_dimension == layout.environment_dimension && retained <= environment)
    {
        retained
    } else {
        environment
    };
    let mut canonical_state = state.to_vec();
    if !noncommuting_groups.is_empty() {
        apply_noncommuting_subsystem_exchange_phases(
            &mut canonical_state,
            local_dimensions,
            &canonical_sites,
            noncommuting_groups,
        )?;
    }
    entanglement_spectrum_subsystem(&canonical_state, local_dimensions, &canonical_sites)
}

pub fn entanglement_spectrum_density_subsystem(
    density: &[Complex64],
    local_dimensions: &[usize],
    retained_sites: &[usize],
) -> Result<Vec<f64>> {
    let layout = subsystem_index_map(local_dimensions, retained_sites)?;
    density_matrix_spectrum(
        partial_trace_density_subsystem(density, local_dimensions, retained_sites)?,
        layout.subsystem_dimension,
    )
}

pub fn entanglement_entropy_subsystem(
    state: &[Complex64],
    local_dimensions: &[usize],
    retained_sites: &[usize],
    order: EntropyOrder,
) -> Result<f64> {
    entropy_from_spectrum(
        &entanglement_spectrum_subsystem(state, local_dimensions, retained_sites)?,
        order,
    )
}

pub fn entanglement_entropy_density(
    density: &[Complex64],
    subsystem_dimension: usize,
    environment_dimension: usize,
    order: EntropyOrder,
) -> Result<f64> {
    entropy_from_spectrum(
        &entanglement_spectrum_density(density, subsystem_dimension, environment_dimension)?,
        order,
    )
}

pub fn entanglement_entropy_density_subsystem(
    density: &[Complex64],
    local_dimensions: &[usize],
    retained_sites: &[usize],
    order: EntropyOrder,
) -> Result<f64> {
    entropy_from_spectrum(
        &entanglement_spectrum_density_subsystem(density, local_dimensions, retained_sites)?,
        order,
    )
}

/// Sorted eigenvalue spectrum of a row-major positive semidefinite density matrix.
pub fn density_matrix_spectrum(density: Vec<Complex64>, dimension: usize) -> Result<Vec<f64>> {
    if density.len() != dimension.saturating_mul(dimension) {
        return Err(QmbedError::DimensionMismatch(
            "density matrix shape does not match its dimension".into(),
        ));
    }
    let mut trace = Complex64::new(0.0, 0.0);
    let mut scale = 0.0_f64;
    for row in 0..dimension {
        trace += density[row * dimension + row];
        for column in 0..dimension {
            let value = density[row * dimension + column];
            if !value.re.is_finite() || !value.im.is_finite() {
                return Err(QmbedError::InvalidOptions(
                    "density matrix entries must be finite".into(),
                ));
            }
            scale = scale.max(value.norm());
            if (value - density[column * dimension + row].conj()).norm() > 1.0e-10 * scale.max(1.0)
            {
                return Err(QmbedError::InvalidOptions(
                    "density matrix must be Hermitian".into(),
                ));
            }
        }
    }
    let mut probabilities = crate::backend::singular_values(&density, dimension, dimension)?;
    probabilities.sort_by(f64::total_cmp);
    let nuclear_norm = probabilities.iter().sum::<f64>();
    let tolerance =
        (256.0 * f64::EPSILON * dimension as f64 * nuclear_norm.max(scale).max(1.0)).max(1.0e-10);
    if trace.im.abs() > tolerance
        || trace.re < -tolerance
        || (nuclear_norm - trace.re).abs() > tolerance
    {
        if trace.re >= -tolerance && nuclear_norm >= trace.re {
            let negative_mass = 0.5 * (nuclear_norm - trace.re);
            return Err(QmbedError::InvalidOptions(format!(
                "density matrix is not positive semidefinite; negative spectral mass is \
                 {negative_mass:e}"
            )));
        }
        return Err(QmbedError::InvalidOptions(
            "density matrix must have a finite nonnegative real trace".into(),
        ));
    }
    Ok(probabilities)
}

pub fn entanglement_spectrum(
    state: &[Complex64],
    subsystem_dimension: usize,
    environment_dimension: usize,
) -> Result<Vec<f64>> {
    density_matrix_spectrum(
        partial_trace(state, subsystem_dimension, environment_dimension)?,
        subsystem_dimension,
    )
}

pub fn entanglement_spectrum_density(
    density: &[Complex64],
    subsystem_dimension: usize,
    environment_dimension: usize,
) -> Result<Vec<f64>> {
    density_matrix_spectrum(
        partial_trace_density(density, subsystem_dimension, environment_dimension)?,
        subsystem_dimension,
    )
}

pub fn entanglement_entropy_batch(
    states: &[Vec<Complex64>],
    subsystem_dimension: usize,
    environment_dimension: usize,
    order: EntropyOrder,
) -> Result<Vec<f64>> {
    states
        .iter()
        .map(|state| entanglement_entropy(state, subsystem_dimension, environment_dimension, order))
        .collect()
}

pub fn density_expectation(
    operator: &(impl LinearOperator + ?Sized),
    density: &[Complex64],
) -> Result<Complex64> {
    let shape = operator.shape();
    if shape.0 != shape.1 || density.len() != shape.0.saturating_mul(shape.0) {
        return Err(QmbedError::DimensionMismatch(
            "density expectation requires a square operator and matching density".into(),
        ));
    }
    let dimension = shape.0;
    let mut column = vec![Complex64::new(0.0, 0.0); dimension];
    let mut applied = vec![Complex64::new(0.0, 0.0); dimension];
    let mut trace = Complex64::new(0.0, 0.0);
    for density_column in 0..dimension {
        for row in 0..dimension {
            column[row] = density[row * dimension + density_column];
        }
        operator.apply(&column, &mut applied)?;
        trace += applied[density_column];
    }
    Ok(trace)
}

/// Density-matrix fluctuation `Tr(ρ A²) - Tr(ρ A)²`.
///
/// The density matrix follows the same row-major convention as
/// [`density_expectation`]. Normalization is deliberately not imposed: callers
/// may use normalized density matrices or preserve QuSpin's direct linear
/// algebra semantics for weighted ensembles.
pub fn density_quantum_fluctuation(
    operator: &(impl LinearOperator + ?Sized),
    density: &[Complex64],
) -> Result<Complex64> {
    let shape = operator.shape();
    if shape.0 != shape.1 || density.len() != shape.0.saturating_mul(shape.0) {
        return Err(QmbedError::DimensionMismatch(
            "density fluctuation requires a square operator and matching density".into(),
        ));
    }
    let dimension = shape.0;
    let mean = density_expectation(operator, density)?;
    let mut column = vec![Complex64::new(0.0, 0.0); dimension];
    let mut applied = vec![Complex64::new(0.0, 0.0); dimension];
    let mut applied_twice = vec![Complex64::new(0.0, 0.0); dimension];
    let mut second = Complex64::new(0.0, 0.0);
    for density_column in 0..dimension {
        for row in 0..dimension {
            column[row] = density[row * dimension + density_column];
        }
        operator.apply(&column, &mut applied)?;
        operator.apply(&applied, &mut applied_twice)?;
        second += applied_twice[density_column];
    }
    Ok(second - mean * mean)
}

pub fn observables_vs_time(
    trajectory: &StateTrajectory,
    observables: &[(String, &dyn LinearOperator)],
) -> Result<HashMap<String, Vec<Complex64>>> {
    if trajectory.times.len() != trajectory.states.len() {
        return Err(QmbedError::DimensionMismatch(
            "trajectory times and states must have equal lengths".into(),
        ));
    }
    let mut result = HashMap::with_capacity(observables.len());
    for (name, operator) in observables {
        if name.is_empty() || result.contains_key(name) {
            return Err(QmbedError::InvalidOptions(
                "observable names must be nonempty and unique".into(),
            ));
        }
        let values = trajectory
            .states
            .iter()
            .map(|state| expectation(*operator, state))
            .collect::<Result<_>>()?;
        result.insert(name.clone(), values);
    }
    Ok(result)
}

/// Exact time evolution from a complete eigendecomposition.
pub fn ed_state_vs_time(
    initial: &[Complex64],
    eigenvalues: &[f64],
    eigenvectors: &[Vec<Complex64>],
    times: &[f64],
) -> Result<StateTrajectory> {
    let dimension = initial.len();
    if times.is_empty()
        || times.iter().any(|time| !time.is_finite())
        || eigenvalues.len() != dimension
        || eigenvectors.len() != dimension
        || eigenvectors.iter().any(|vector| vector.len() != dimension)
    {
        return Err(QmbedError::DimensionMismatch(
            "complete eigensystem, state, and finite nonempty times are required".into(),
        ));
    }
    let coefficients: Vec<_> = eigenvectors
        .iter()
        .map(|vector| inner(vector, initial))
        .collect();
    let states = times
        .iter()
        .map(|time| {
            let mut state = vec![Complex64::new(0.0, 0.0); dimension];
            for ((energy, vector), coefficient) in
                eigenvalues.iter().zip(eigenvectors).zip(&coefficients)
            {
                let weight = coefficient * Complex64::new(0.0, -*time * energy).exp();
                for (value, eigenvector_value) in state.iter_mut().zip(vector) {
                    *value += weight * *eigenvector_value;
                }
            }
            state
        })
        .collect();
    Ok(StateTrajectory {
        times: times.to_vec(),
        states,
    })
}

/// Density-matrix counterpart of [`ed_state_vs_time`], with row-major input
/// and output matrices.
pub fn ed_density_vs_time(
    initial: &[Complex64],
    eigenvalues: &[f64],
    eigenvectors: &[Vec<Complex64>],
    times: &[f64],
) -> Result<Vec<Vec<Complex64>>> {
    let dimension = eigenvalues.len();
    if initial.len() != dimension.saturating_mul(dimension)
        || eigenvectors.len() != dimension
        || eigenvectors.iter().any(|vector| vector.len() != dimension)
        || times.is_empty()
        || times.iter().any(|time| !time.is_finite())
    {
        return Err(QmbedError::DimensionMismatch(
            "density matrix and complete eigensystem dimensions do not match".into(),
        ));
    }
    let mut eigen_density = vec![Complex64::new(0.0, 0.0); dimension * dimension];
    for left in 0..dimension {
        for right in 0..dimension {
            for row in 0..dimension {
                for column in 0..dimension {
                    eigen_density[left * dimension + right] += eigenvectors[left][row].conj()
                        * initial[row * dimension + column]
                        * eigenvectors[right][column];
                }
            }
        }
    }
    Ok(times
        .iter()
        .map(|time| {
            let mut density = vec![Complex64::new(0.0, 0.0); dimension * dimension];
            for row in 0..dimension {
                for column in 0..dimension {
                    for left in 0..dimension {
                        for right in 0..dimension {
                            let phase = Complex64::new(
                                0.0,
                                -*time * (eigenvalues[left] - eigenvalues[right]),
                            )
                            .exp();
                            density[row * dimension + column] += eigenvectors[left][row]
                                * phase
                                * eigen_density[left * dimension + right]
                                * eigenvectors[right][column].conj();
                        }
                    }
                }
            }
            density
        })
        .collect())
}

#[derive(Clone, Debug)]
pub struct DiagonalEnsemble {
    pub probabilities: Vec<f64>,
    pub mean_energy: f64,
    pub energy_variance: f64,
    pub entropy: f64,
}

#[derive(Clone, Debug)]
pub struct DiagonalEnsembleColumn {
    pub ensemble: DiagonalEnsemble,
    pub diagonal_entropy: f64,
    pub observable: Option<f64>,
    pub temporal_fluctuation: Option<f64>,
    pub quantum_fluctuation: Option<f64>,
}

fn summarize_diagonal_probabilities(
    eigenvalues: &[f64],
    mut probabilities: Vec<f64>,
) -> Result<DiagonalEnsemble> {
    let probability_sum = probabilities.iter().sum::<f64>();
    if probability_sum <= f64::EPSILON || !probability_sum.is_finite() {
        return Err(QmbedError::InvalidOptions(
            "eigenvectors have no finite overlap with the initial state".into(),
        ));
    }
    for probability in &mut probabilities {
        *probability /= probability_sum;
    }
    let mean_energy = probabilities
        .iter()
        .zip(eigenvalues)
        .map(|(probability, energy)| probability * energy)
        .sum::<f64>();
    let energy_variance = probabilities
        .iter()
        .zip(eigenvalues)
        .map(|(probability, energy)| probability * (energy - mean_energy).powi(2))
        .sum();
    let entropy = -probabilities
        .iter()
        .filter(|probability| **probability > f64::EPSILON)
        .map(|probability| probability * probability.ln())
        .sum::<f64>();
    Ok(DiagonalEnsemble {
        probabilities,
        mean_energy,
        energy_variance,
        entropy,
    })
}

pub fn diagonal_ensemble_from_probabilities(
    eigenvalues: &[f64],
    probabilities: &[f64],
) -> Result<DiagonalEnsemble> {
    if eigenvalues.len() != probabilities.len()
        || probabilities
            .iter()
            .any(|probability| !probability.is_finite() || *probability < 0.0)
    {
        return Err(QmbedError::InvalidOptions(
            "diagonal probabilities must be finite, nonnegative, and match the spectrum".into(),
        ));
    }
    summarize_diagonal_probabilities(eigenvalues, probabilities.to_vec())
}

pub fn diagonal_density_matrix(
    eigenvectors: &[Vec<Complex64>],
    probabilities: &[f64],
) -> Result<Vec<Complex64>> {
    let dimension = eigenvectors.len();
    if probabilities.len() != dimension
        || eigenvectors.iter().any(|vector| vector.len() != dimension)
    {
        return Err(QmbedError::DimensionMismatch(
            "diagonal density probabilities and eigenvectors do not match".into(),
        ));
    }
    if probabilities
        .iter()
        .any(|probability| !probability.is_finite() || *probability < 0.0)
    {
        return Err(QmbedError::InvalidOptions(
            "diagonal density probabilities must be finite and nonnegative".into(),
        ));
    }
    let mut density = vec![Complex64::new(0.0, 0.0); dimension * dimension];
    for (probability, vector) in probabilities.iter().zip(eigenvectors) {
        for row in 0..dimension {
            for column in 0..dimension {
                density[row * dimension + column] +=
                    *probability * vector[row] * vector[column].conj();
            }
        }
    }
    Ok(density)
}

fn distribution_entropy(probabilities: &[f64], alpha: f64) -> Result<f64> {
    if !alpha.is_finite() || alpha < 0.0 {
        return Err(QmbedError::InvalidOptions(
            "Renyi alpha must be finite and nonnegative".into(),
        ));
    }
    if (alpha - 1.0).abs() <= f64::EPSILON {
        return Ok(-probabilities
            .iter()
            .filter(|probability| **probability > f64::EPSILON)
            .map(|probability| probability * probability.ln())
            .sum::<f64>());
    }
    let moment = probabilities
        .iter()
        .filter(|probability| **probability > f64::EPSILON)
        .map(|probability| probability.powf(alpha))
        .sum::<f64>();
    if moment <= 0.0 || !moment.is_finite() {
        return Err(QmbedError::InvalidOptions(
            "Renyi probability moment is not positive and finite".into(),
        ));
    }
    Ok(moment.ln() / (1.0 - alpha))
}

/// Analyze one or more diagonal probability distributions in a shared
/// eigensystem. Optional observable statistics are evaluated matrix-free in
/// that eigensystem, so stored and matrix-free operators share one path.
pub fn analyze_diagonal_ensemble(
    eigenvalues: &[f64],
    eigenvectors: &[Vec<Complex64>],
    probability_columns: &[Vec<f64>],
    observable: Option<&dyn LinearOperator>,
    alpha: f64,
) -> Result<Vec<DiagonalEnsembleColumn>> {
    let dimension = eigenvalues.len();
    let mut sorted_eigenvalues = eigenvalues.to_vec();
    sorted_eigenvalues.sort_by(f64::total_cmp);
    if sorted_eigenvalues.windows(2).any(|window| {
        let scale = window[0].abs().max(window[1].abs()).max(1.0);
        (window[1] - window[0]).abs() <= 1.0e-12 * scale
    }) {
        return Err(QmbedError::InvalidOptions(
            "diagonal-ensemble formulas require a nondegenerate spectrum".into(),
        ));
    }
    if eigenvectors.len() != dimension
        || eigenvectors.iter().any(|vector| vector.len() != dimension)
    {
        return Err(QmbedError::DimensionMismatch(
            "diagonal-ensemble eigensystem must be square".into(),
        ));
    }
    if probability_columns.is_empty() {
        return Err(QmbedError::InvalidOptions(
            "diagonal-ensemble analysis needs at least one probability column".into(),
        ));
    }

    type ObservableStatistics = (Vec<f64>, Vec<Vec<f64>>, Vec<f64>);
    let observable_statistics = observable
        .map(|observable| -> Result<ObservableStatistics> {
            if observable.shape() != (dimension, dimension) {
                return Err(QmbedError::DimensionMismatch(
                    "diagonal-ensemble observable and eigensystem do not match".into(),
                ));
            }
            let mut applied = Vec::with_capacity(dimension);
            for vector in eigenvectors {
                let mut output = vec![Complex64::new(0.0, 0.0); dimension];
                observable.apply(vector, &mut output)?;
                applied.push(output);
            }
            let mut diagonal = vec![0.0; dimension];
            let mut squared_elements = vec![vec![0.0; dimension]; dimension];
            let mut squared_diagonal = vec![0.0; dimension];
            for row in 0..dimension {
                for column in 0..dimension {
                    let element = inner(&eigenvectors[row], &applied[column]);
                    if row == column {
                        if element.im.abs() > 1.0e-9 {
                            return Err(QmbedError::InvalidOptions(
                                "diagonal-ensemble observable must be Hermitian".into(),
                            ));
                        }
                        diagonal[row] = element.re;
                    }
                    let magnitude = element.norm_sqr();
                    squared_elements[row][column] = magnitude;
                    squared_diagonal[row] += magnitude;
                }
            }
            Ok((diagonal, squared_elements, squared_diagonal))
        })
        .transpose()?;

    probability_columns
        .iter()
        .map(|probabilities| {
            let ensemble = diagonal_ensemble_from_probabilities(eigenvalues, probabilities)?;
            let diagonal_entropy = distribution_entropy(&ensemble.probabilities, alpha)?;
            let (observable, temporal_fluctuation, quantum_fluctuation) = if let Some((
                diagonal,
                squared_elements,
                squared_diagonal,
            )) =
                &observable_statistics
            {
                let expectation = diagonal
                    .iter()
                    .zip(&ensemble.probabilities)
                    .map(|(value, probability)| value * probability)
                    .sum::<f64>();
                let mut temporal_variance = 0.0;
                for (row, squared_row) in squared_elements.iter().enumerate() {
                    for (column, squared_element) in squared_row.iter().enumerate() {
                        if row != column {
                            temporal_variance += ensemble.probabilities[row]
                                * squared_element
                                * ensemble.probabilities[column];
                        }
                    }
                }
                let total_second_moment = squared_diagonal
                    .iter()
                    .zip(&ensemble.probabilities)
                    .map(|(value, probability)| value * probability)
                    .sum::<f64>();
                let quantum_variance =
                    total_second_moment - temporal_variance - expectation.powi(2);
                (
                    Some(expectation),
                    Some(temporal_variance.max(0.0).sqrt()),
                    Some(quantum_variance.max(0.0).sqrt()),
                )
            } else {
                (None, None, None)
            };
            Ok(DiagonalEnsembleColumn {
                ensemble,
                diagonal_entropy,
                observable,
                temporal_fluctuation,
                quantum_fluctuation,
            })
        })
        .collect()
}

pub fn diagonal_ensemble(
    eigenvalues: &[f64],
    eigenvectors: &[Vec<Complex64>],
    initial: &[Complex64],
) -> Result<DiagonalEnsemble> {
    if eigenvalues.len() != eigenvectors.len()
        || eigenvectors
            .iter()
            .any(|vector| vector.len() != initial.len())
    {
        return Err(QmbedError::DimensionMismatch(
            "eigensystem and initial state dimensions do not match".into(),
        ));
    }
    let initial_norm = inner(initial, initial).re;
    if !initial_norm.is_finite() || initial_norm <= f64::EPSILON {
        return Err(QmbedError::InvalidOptions(
            "initial state must have positive finite norm".into(),
        ));
    }
    let probabilities: Vec<_> = eigenvectors
        .iter()
        .map(|vector| inner(vector, initial).norm_sqr() / initial_norm)
        .collect();
    summarize_diagonal_probabilities(eigenvalues, probabilities)
}

pub fn diagonal_ensemble_density(
    eigenvalues: &[f64],
    eigenvectors: &[Vec<Complex64>],
    initial_density: &[Complex64],
) -> Result<DiagonalEnsemble> {
    let dimension = eigenvalues.len();
    if eigenvectors.len() != dimension
        || eigenvectors.iter().any(|vector| vector.len() != dimension)
        || initial_density.len() != dimension.saturating_mul(dimension)
    {
        return Err(QmbedError::DimensionMismatch(
            "eigensystem and initial density dimensions do not match".into(),
        ));
    }
    let trace: Complex64 = (0..dimension)
        .map(|index| initial_density[index * dimension + index])
        .sum();
    if trace.im.abs() > 1.0e-10 || !trace.re.is_finite() || trace.re <= f64::EPSILON {
        return Err(QmbedError::InvalidOptions(
            "initial density must have a positive real trace".into(),
        ));
    }
    let probabilities = eigenvectors
        .iter()
        .map(|vector| {
            let mut value = Complex64::new(0.0, 0.0);
            for row in 0..dimension {
                for column in 0..dimension {
                    value += vector[row].conj()
                        * initial_density[row * dimension + column]
                        * vector[column];
                }
            }
            if value.im.abs() > 1.0e-10 || value.re < -1.0e-10 {
                Err(QmbedError::InvalidOptions(
                    "initial density is not positive in the supplied eigenbasis".into(),
                ))
            } else {
                Ok(value.re.max(0.0) / trace.re)
            }
        })
        .collect::<Result<Vec<_>>>()?;
    summarize_diagonal_probabilities(eigenvalues, probabilities)
}

pub fn diagonal_ensemble_observable(
    ensemble: &DiagonalEnsemble,
    eigenvectors: &[Vec<Complex64>],
    observable: &(impl LinearOperator + ?Sized),
) -> Result<Complex64> {
    if ensemble.probabilities.len() != eigenvectors.len()
        || eigenvectors.iter().any(|vector| {
            vector.len() != observable.shape().0 || observable.shape().0 != observable.shape().1
        })
    {
        return Err(QmbedError::DimensionMismatch(
            "diagonal ensemble, eigenvectors, and observable do not match".into(),
        ));
    }
    ensemble
        .probabilities
        .iter()
        .zip(eigenvectors)
        .try_fold(Complex64::new(0.0, 0.0), |total, (probability, vector)| {
            Ok(total + *probability * expectation(observable, vector)?)
        })
}

pub fn energy_window_indices(
    eigenvalues: &[f64],
    center: f64,
    half_width: f64,
) -> Result<Vec<usize>> {
    if !center.is_finite()
        || !half_width.is_finite()
        || half_width < 0.0
        || eigenvalues.iter().any(|value| !value.is_finite())
    {
        return Err(QmbedError::InvalidOptions(
            "energy-window inputs must be finite and the half-width nonnegative".into(),
        ));
    }
    Ok(eigenvalues
        .iter()
        .enumerate()
        .filter_map(|(index, value)| ((*value - center).abs() <= half_width).then_some(index))
        .collect())
}

pub fn kl_divergence(left: &[f64], right: &[f64]) -> Result<f64> {
    if left.len() != right.len()
        || left.is_empty()
        || left
            .iter()
            .chain(right)
            .any(|value| !value.is_finite() || *value <= 0.0)
    {
        return Err(QmbedError::InvalidOptions(
            "KL distributions must be nonempty, equal-length, finite, and strictly positive".into(),
        ));
    }
    let left_sum = left.iter().sum::<f64>();
    let right_sum = right.iter().sum::<f64>();
    if (left_sum - 1.0).abs() > 1.0e-13 || (right_sum - 1.0).abs() > 1.0e-13 {
        return Err(QmbedError::InvalidOptions(
            "KL distributions must be normalized".into(),
        ));
    }
    Ok(left
        .iter()
        .zip(right)
        .map(|(probability, reference)| probability * (probability / reference).ln())
        .sum::<f64>()
        .max(0.0))
}

/// Mean adjacent-gap ratio of an ordered spectrum.
pub fn mean_level_spacing(eigenvalues: &[f64]) -> Result<f64> {
    if eigenvalues.len() < 3 || eigenvalues.iter().any(|value| !value.is_finite()) {
        return Err(QmbedError::InvalidOptions(
            "level-spacing statistics require at least three finite values".into(),
        ));
    }
    if eigenvalues.windows(2).any(|pair| pair[0] > pair[1]) {
        return Err(QmbedError::InvalidOptions(
            "level spectrum must be sorted in ascending order".into(),
        ));
    }
    let gaps: Vec<_> = eigenvalues
        .windows(2)
        .map(|pair| pair[1] - pair[0])
        .collect();
    if gaps.contains(&0.0) {
        return Ok(f64::NAN);
    }
    Ok(gaps
        .windows(2)
        .map(|pair| pair[0].min(pair[1]) / pair[0].max(pair[1]))
        .sum::<f64>()
        / (gaps.len() - 1) as f64)
}

pub fn states_to_array(
    states: &[u128],
    sites: usize,
    local_dimension: usize,
) -> Result<Vec<Vec<usize>>> {
    if local_dimension < 2 {
        return Err(QmbedError::InvalidSector(
            "local dimension must be at least two".into(),
        ));
    }
    let base = local_dimension as u128;
    let mut result = Vec::with_capacity(states.len());
    for &state in states {
        let mut value = state;
        let mut occupations = Vec::with_capacity(sites);
        for _ in 0..sites {
            occupations.push((value % base) as usize);
            value /= base;
        }
        if value != 0 {
            return Err(QmbedError::StateNotInBasis);
        }
        result.push(occupations);
    }
    Ok(result)
}

pub fn array_to_states(arrays: &[Vec<usize>], local_dimension: usize) -> Result<Vec<u128>> {
    if local_dimension < 2 {
        return Err(QmbedError::InvalidSector(
            "local dimension must be at least two".into(),
        ));
    }
    let sites = arrays.first().map_or(0, Vec::len);
    let base = local_dimension as u128;
    arrays
        .iter()
        .map(|occupations| {
            if occupations.len() != sites
                || occupations
                    .iter()
                    .any(|occupation| *occupation >= local_dimension)
            {
                return Err(QmbedError::InvalidSector(
                    "occupation arrays must have equal length and valid digits".into(),
                ));
            }
            let mut state = 0_u128;
            let mut place = 1_u128;
            for &occupation in occupations {
                state = state
                    .checked_add(place.checked_mul(occupation as u128).ok_or_else(|| {
                        QmbedError::UnsupportedBackend("state encoding overflow".into())
                    })?)
                    .ok_or_else(|| {
                        QmbedError::UnsupportedBackend("state encoding overflow".into())
                    })?;
                place = place.checked_mul(base).ok_or_else(|| {
                    QmbedError::UnsupportedBackend("state encoding overflow".into())
                })?;
            }
            Ok(state)
        })
        .collect()
}

/// Convert integer states to most-significant-site-first binary rows.
pub fn ints_to_array(states: &[u128], sites: usize) -> Result<Vec<Vec<u8>>> {
    if sites > 128 {
        return Err(QmbedError::UnsupportedBackend(
            "u128 binary conversion supports at most 128 sites".into(),
        ));
    }
    if states
        .iter()
        .any(|state| sites < 128 && *state >= (1_u128 << sites))
    {
        return Err(QmbedError::StateNotInBasis);
    }
    Ok(states
        .iter()
        .map(|state| {
            (0..sites)
                .map(|column| ((state >> (sites - column - 1)) & 1) as u8)
                .collect()
        })
        .collect())
}

/// Convert most-significant-site-first binary rows to integer states.
pub fn array_to_ints(arrays: &[Vec<u8>]) -> Result<Vec<u128>> {
    let sites = arrays.first().map_or(0, Vec::len);
    if sites > 128 {
        return Err(QmbedError::UnsupportedBackend(
            "u128 binary conversion supports at most 128 sites".into(),
        ));
    }
    arrays
        .iter()
        .map(|row| {
            if row.len() != sites || row.iter().any(|bit| *bit > 1) {
                return Err(QmbedError::InvalidSector(
                    "binary state rows must have equal lengths and contain only zero or one".into(),
                ));
            }
            Ok(row
                .iter()
                .fold(0_u128, |state, bit| (state << 1) | u128::from(*bit)))
        })
        .collect()
}

/// Compute `P† A P` one reduced column at a time.
pub fn project_operator(
    operator: &(impl LinearOperator + ?Sized),
    projector: &BasisProjector,
    format: MatrixFormat,
) -> Result<Operator> {
    let source_dimension = projector.source_dimension();
    let reduced_dimension = projector.reduced_dimension();
    let operator_dimension = operator.shape();
    if operator_dimension.0 != operator_dimension.1
        || (operator_dimension.0 != source_dimension && operator_dimension.0 != reduced_dimension)
    {
        return Err(QmbedError::DimensionMismatch(
            "operator dimension must match the parent or reduced projector space".into(),
        ));
    }
    if operator_dimension.0 == reduced_dimension {
        let mut parent_input = vec![Complex64::new(0.0, 0.0); source_dimension];
        let mut reduced_input = vec![Complex64::new(0.0, 0.0); reduced_dimension];
        let mut reduced_output = vec![Complex64::new(0.0, 0.0); reduced_dimension];
        let mut parent_output = vec![Complex64::new(0.0, 0.0); source_dimension];
        let mut triplets = Vec::new();
        for column in 0..source_dimension {
            parent_input.fill(Complex64::new(0.0, 0.0));
            parent_input[column] = Complex64::new(1.0, 0.0);
            projector.project(&parent_input, &mut reduced_input)?;
            operator.apply(&reduced_input, &mut reduced_output)?;
            projector.apply(&reduced_output, &mut parent_output)?;
            for (row, &value) in parent_output.iter().enumerate() {
                if value.norm() > f64::EPSILON {
                    triplets.push((row, column, value));
                }
            }
        }
        return Operator::from_triplets(source_dimension, source_dimension, triplets, format);
    }
    let mut reduced_input = vec![Complex64::new(0.0, 0.0); reduced_dimension];
    let mut parent_input = vec![Complex64::new(0.0, 0.0); source_dimension];
    let mut parent_output = vec![Complex64::new(0.0, 0.0); source_dimension];
    let mut reduced_output = vec![Complex64::new(0.0, 0.0); reduced_dimension];
    let mut triplets = Vec::new();
    for column in 0..reduced_dimension {
        reduced_input.fill(Complex64::new(0.0, 0.0));
        reduced_input[column] = Complex64::new(1.0, 0.0);
        projector.apply(&reduced_input, &mut parent_input)?;
        operator.apply(&parent_input, &mut parent_output)?;
        projector.project(&parent_output, &mut reduced_output)?;
        for (row, &value) in reduced_output.iter().enumerate() {
            if value.norm() > f64::EPSILON {
                triplets.push((row, column, value));
            }
        }
    }
    Operator::from_triplets(reduced_dimension, reduced_dimension, triplets, format)
}
