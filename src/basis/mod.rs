use std::collections::{HashMap, HashSet, VecDeque};
use std::hash::Hash;
use std::sync::Arc;

use num_bigint::BigUint;
use num_complex::Complex64;
use smallvec::SmallVec;

use crate::operator::{LinearOperator, MatrixFormat, check_apply_shape};
use crate::{QmbedError, Result};

/// Compact collection of local-operator destinations.
///
/// The common zero-, one-, and two-destination cases stay inline. Operators
/// with wider branching use the same interface and spill to heap storage
/// automatically.
pub type LocalTransitions<State> = SmallVec<[(State, Complex64); 2]>;

/// Canonical image of one physical state under a basis reduction.
///
/// `phase / sqrt(orbit_size)` is the coefficient of `state` in the normalized
/// reduced vector labelled by `representative`. Keeping this convention at the
/// basis boundary lets projectors, cross-sector operators, and language
/// bindings share one source of truth.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ReductionImage<State> {
    representative: State,
    phase: Complex64,
    orbit_size: usize,
}

impl<State> ReductionImage<State> {
    pub fn new(representative: State, phase: Complex64, orbit_size: usize) -> Result<Self> {
        if orbit_size == 0 || !phase.re.is_finite() || !phase.im.is_finite() {
            return Err(QmbedError::InternalState(
                "a reduction image requires a positive orbit and finite phase".into(),
            ));
        }
        if (phase.norm() - 1.0).abs() > 1.0e-10 {
            return Err(QmbedError::InternalState(
                "a reduction-image phase must have unit magnitude".into(),
            ));
        }
        Ok(Self {
            representative,
            phase,
            orbit_size,
        })
    }

    pub const fn representative(&self) -> &State {
        &self.representative
    }

    pub const fn phase(&self) -> Complex64 {
        self.phase
    }

    pub const fn orbit_size(&self) -> usize {
        self.orbit_size
    }

    /// Normalized coefficient of the physical state in its reduced vector.
    pub fn amplitude(&self) -> Complex64 {
        self.phase / (self.orbit_size as f64).sqrt()
    }
}

/// Finite Hilbert-space basis and its local operator semantics.
pub trait Basis: Send + Sync {
    type State: Copy + Eq + Send + Sync;

    fn len(&self) -> usize;
    fn state(&self, index: usize) -> Result<Self::State>;
    fn index(&self, state: Self::State) -> Result<usize>;
    fn apply_local(
        &self,
        state: Self::State,
        operator: &str,
        sites: &[usize],
    ) -> Result<Option<(Self::State, Complex64)>>;

    /// Applies a local operator and returns every nonzero destination.
    ///
    /// Most spin-one-half, boson ladder, and fermion strings are deterministic,
    /// so the default implementation wraps [`Basis::apply_local`]. Higher-spin
    /// Cartesian operators and general user-defined local matrices can branch
    /// and override this method without changing the universal assembler.
    fn apply_local_transitions(
        &self,
        state: Self::State,
        operator: &str,
        sites: &[usize],
    ) -> Result<LocalTransitions<Self::State>> {
        Ok(self
            .apply_local(state, operator, sites)?
            .into_iter()
            .collect())
    }

    /// Local action before symmetry-sector reduction. Cross-sector builders
    /// use this path and let the target basis perform the reduction.
    fn apply_local_unreduced_transitions(
        &self,
        state: Self::State,
        operator: &str,
        sites: &[usize],
    ) -> Result<LocalTransitions<Self::State>> {
        self.apply_local_transitions(state, operator, sites)
    }

    /// Streams unreduced destinations directly to a consumer.
    ///
    /// This is the universal hot-path interface used by assemblers. The
    /// default covers deterministic bases without constructing an intermediate
    /// collection; branching and symmetry-reduced bases override it while
    /// preserving the same consumer contract.
    fn visit_local_unreduced_transitions<F>(
        &self,
        state: Self::State,
        operator: &str,
        sites: &[usize],
        mut visit: F,
    ) -> Result<()>
    where
        Self: Sized,
        F: FnMut(Self::State, Complex64) -> Result<()>,
    {
        if let Some((target, amplitude)) = self.apply_local(state, operator, sites)? {
            visit(target, amplitude)?;
        }
        Ok(())
    }

    /// Streams a local action whose operator symbols were parsed at the API
    /// boundary.
    ///
    /// The default preserves compatibility with custom bases. Built-in bases
    /// override this method so repeated state/coupling actions do not rescan
    /// the same operator string.
    #[doc(hidden)]
    fn visit_preparsed_local_unreduced_transitions<F>(
        &self,
        state: Self::State,
        operator: &str,
        symbols: &[char],
        split: Option<usize>,
        sites: &[usize],
        visit: F,
    ) -> Result<()>
    where
        Self: Sized,
        F: FnMut(Self::State, Complex64) -> Result<()>,
    {
        let _ = (symbols, split);
        self.visit_local_unreduced_transitions(state, operator, sites, visit)
    }

    /// Orbit size of a canonical source state used in projector normalization.
    fn transition_orbit_size(&self, _state: Self::State) -> Result<usize> {
        Ok(1)
    }

    /// Return the canonical reduction metadata for a physical state.
    ///
    /// Explicit bases use a one-state orbit. Symmetry-reduced bases override
    /// this query without exposing their lookup-table representation.
    fn reduction_image(&self, state: Self::State) -> Result<Option<ReductionImage<Self::State>>> {
        match self.index(state) {
            Ok(_) => Ok(Some(ReductionImage::new(
                state,
                Complex64::new(1.0, 0.0),
                1,
            )?)),
            Err(QmbedError::StateNotInBasis) => Ok(None),
            Err(error) => Err(error),
        }
    }

    /// Map an unreduced physical target state into this basis.
    fn reduce_transition(
        &self,
        state: Self::State,
        source_orbit_size: usize,
    ) -> Result<Option<(Self::State, Complex64)>> {
        Ok(self.reduction_image(state)?.map(|image| {
            (
                image.representative,
                (source_orbit_size as f64 / image.orbit_size as f64).sqrt() * image.phase.conj(),
            )
        }))
    }

    /// Reduces a physical target and locates its row in one operation.
    ///
    /// Assemblers use this fused boundary to avoid looking up the same state
    /// once during reduction and again during indexing.
    fn index_transition(
        &self,
        state: Self::State,
        source_orbit_size: usize,
    ) -> Result<Option<(usize, Complex64)>> {
        let Some((representative, amplitude)) = self.reduce_transition(state, source_orbit_size)?
        else {
            return Ok(None);
        };
        Ok(Some((self.index(representative)?, amplitude)))
    }

    /// Whether a local operator string preserves the particle-sector
    /// constraints represented by this basis. Unconstrained and custom bases
    /// accept every syntactically valid string by default.
    fn operator_preserves_particle_sector(&self, _operator: &str) -> Result<bool> {
        Ok(true)
    }

    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

fn operator_number_change(operator: &str) -> Result<Option<i32>> {
    let mut change = 0_i32;
    for character in operator.chars().filter(|character| *character != '|') {
        match character {
            '+' => change += 1,
            '-' => change -= 1,
            'x' | 'y' => return Ok(None),
            'I' | 'n' | 'z' => {}
            _ => return Err(QmbedError::InvalidOperator(operator.into())),
        }
    }
    Ok(Some(change))
}

fn fixed_weight_states(sites: usize, particles: Option<usize>) -> Result<Vec<u128>> {
    if sites > 128 {
        return Err(QmbedError::UnsupportedBackend(
            "the initial u128 state backend supports at most 128 orbitals".into(),
        ));
    }
    if particles.is_some_and(|count| count > sites) {
        return Err(QmbedError::InvalidSector(
            "particle count exceeds site count".into(),
        ));
    }
    let Some(particles) = particles else {
        let limit = 1_u128
            .checked_shl(u32::try_from(sites).unwrap_or(u32::MAX))
            .ok_or_else(|| {
                QmbedError::UnsupportedBackend(
                    "enumerating the unconstrained 128-site Hilbert space is infeasible".into(),
                )
            })?;
        return Ok((0..limit).collect());
    };
    if particles == 0 {
        return Ok(vec![0]);
    }
    if particles == sites {
        let state = if sites == 128 {
            u128::MAX
        } else {
            (1_u128 << sites) - 1
        };
        return Ok(vec![state]);
    }

    // Gosper's hack enumerates only C(sites, particles) states instead of
    // scanning the complete 2^sites parent space.
    let mut state = (1_u128 << particles) - 1;
    let limit = (sites < 128).then(|| 1_u128 << sites);
    let mut states = Vec::new();
    loop {
        states.push(state);
        let low_bit = state & state.wrapping_neg();
        let Some(ripple) = state.checked_add(low_bit) else {
            break;
        };
        let next = (((ripple ^ state) >> 2) / low_bit) | ripple;
        if limit.is_some_and(|upper| next >= upper) {
            break;
        }
        state = next;
    }
    Ok(states)
}

fn fixed_digit_sum_states(
    sites: usize,
    states_per_site: usize,
    total: Option<usize>,
) -> Result<Vec<u128>> {
    if sites == 0 || states_per_site == 0 {
        return Err(QmbedError::InvalidSector(
            "sites and local state count must be positive".into(),
        ));
    }
    if total.is_some_and(|value| value > sites.saturating_mul(states_per_site - 1)) {
        return Err(QmbedError::InvalidSector(
            "requested occupation exceeds the local spin capacity".into(),
        ));
    }
    let base = states_per_site as u128;
    let exponent = u32::try_from(sites)
        .map_err(|_| QmbedError::UnsupportedBackend("site count is too large".into()))?;
    let limit = base.checked_pow(exponent).ok_or_else(|| {
        QmbedError::UnsupportedBackend("mixed-radix state encoding overflow".into())
    })?;
    if total.is_none() {
        return Ok((0..limit).collect());
    }

    fn enumerate(
        site: usize,
        sites: usize,
        states_per_site: usize,
        remaining: usize,
        place: u128,
        encoded: u128,
        output: &mut Vec<u128>,
    ) {
        if site == sites {
            if remaining == 0 {
                output.push(encoded);
            }
            return;
        }
        let remaining_sites = sites - site - 1;
        let maximum_tail = remaining_sites.saturating_mul(states_per_site - 1);
        for digit in 0..states_per_site {
            if digit > remaining || remaining - digit > maximum_tail {
                continue;
            }
            enumerate(
                site + 1,
                sites,
                states_per_site,
                remaining - digit,
                place * states_per_site as u128,
                encoded + digit as u128 * place,
                output,
            );
        }
    }

    let mut states = Vec::new();
    enumerate(
        0,
        sites,
        states_per_site,
        total.unwrap_or_default(),
        1,
        0,
        &mut states,
    );
    states.sort_unstable();
    Ok(states)
}

fn state_index(states: &[u128], state: u128) -> Result<usize> {
    states
        .binary_search(&state)
        .map_err(|_| QmbedError::StateNotInBasis)
}

fn direct_state_index(states: &[u128], state: u128) -> Result<usize> {
    let index = usize::try_from(state).map_err(|_| QmbedError::StateNotInBasis)?;
    if index < states.len() {
        Ok(index)
    } else {
        Err(QmbedError::StateNotInBasis)
    }
}

/// Rank a fixed-weight bit string in the colexicographic order generated by
/// Gosper's hack in [`fixed_weight_states`].
fn fixed_weight_state_index(state: u128, sites: usize, particles: usize) -> Result<usize> {
    if particles > sites
        || state.count_ones() as usize != particles
        || (sites < 128 && state >= (1_u128 << sites))
    {
        return Err(QmbedError::StateNotInBasis);
    }
    let mut rank = 0_usize;
    let mut ordinal = 1_usize;
    let mut remaining = state;
    while remaining != 0 {
        let position = remaining.trailing_zeros() as usize;
        if position >= ordinal {
            rank = rank.saturating_add(binomial(position, ordinal));
        }
        ordinal += 1;
        remaining &= remaining - 1;
    }
    Ok(rank)
}

fn rotate_lattice_state(state: u128, shift: usize, sites: usize, base: u128) -> u128 {
    if sites == 0 {
        return state;
    }
    let shift = shift % sites;
    if shift == 0 {
        return state;
    }
    if base == 2 {
        let mask = if sites == 128 {
            u128::MAX
        } else {
            (1_u128 << sites) - 1
        };
        return ((state << shift) & mask) | (state >> (sites - shift));
    }
    let mut translated = 0_u128;
    let mut source_place = 1_u128;
    for site in 0..sites {
        let digit = (state / source_place) % base;
        let target_site = (site + shift) % sites;
        translated += digit * base.pow(u32::try_from(target_site).unwrap_or(u32::MAX));
        source_place *= base;
    }
    translated
}

fn reflect_lattice_state(state: u128, sites: usize, base: u128) -> u128 {
    let mut reflected = 0_u128;
    let mut source_place = 1_u128;
    for site in 0..sites {
        let digit = (state / source_place) % base;
        let target_site = sites - site - 1;
        reflected += digit * base.pow(u32::try_from(target_site).unwrap_or(u32::MAX));
        source_place *= base;
    }
    reflected
}

#[derive(Clone, Copy, Debug)]
struct SymmetryImage {
    representative: u128,
    phase: Complex64,
    orbit_size: usize,
}

type SymmetrySectorData = (
    Vec<u128>,
    Vec<usize>,
    HashMap<u128, SymmetryImage>,
    Option<usize>,
    Option<i8>,
);

fn spin_symmetry_sector(
    parent_states: Vec<u128>,
    sites: usize,
    base: u128,
    momentum: Option<i32>,
    parity: Option<i8>,
) -> Result<SymmetrySectorData> {
    if sites == 0 {
        return Err(QmbedError::InvalidSector(
            "symmetry sectors require at least one site".into(),
        ));
    }
    if parity.is_some_and(|value| value != -1 && value != 1) {
        return Err(QmbedError::InvalidSector(
            "parity must be either -1 or +1".into(),
        ));
    }
    let sites_i64 = i64::try_from(sites)
        .map_err(|_| QmbedError::UnsupportedBackend("site count is too large".into()))?;
    let normalized_momentum = momentum.map(|value| i64::from(value).rem_euclid(sites_i64) as usize);
    if parity.is_some() && normalized_momentum.is_some_and(|value| value != 0 && 2 * value != sites)
    {
        return Err(QmbedError::IncompatibleSymmetry(
            "parity can share a one-dimensional sector with momentum only at k=0 or k=pi".into(),
        ));
    }

    if momentum.is_none() && parity.is_none() {
        let orbit_sizes = vec![1; parent_states.len()];
        let lookup = parent_states
            .iter()
            .copied()
            .map(|state| {
                (
                    state,
                    SymmetryImage {
                        representative: state,
                        phase: Complex64::new(1.0, 0.0),
                        orbit_size: 1,
                    },
                )
            })
            .collect();
        return Ok((parent_states, orbit_sizes, lookup, None, None));
    }

    let parent_lookup: HashSet<_> = parent_states.iter().copied().collect();
    let translations = if momentum.is_some() { sites } else { 1 };
    let mut visited = HashSet::with_capacity(parent_states.len());
    let mut sectors = Vec::<(u128, usize)>::new();
    let mut lookup = HashMap::with_capacity(parent_states.len());

    for seed in parent_states {
        if visited.contains(&seed) {
            continue;
        }
        let mut orbit = HashSet::new();
        for shift in 0..translations {
            let translated = rotate_lattice_state(seed, shift, sites, base);
            orbit.insert(translated);
            if parity.is_some() {
                orbit.insert(reflect_lattice_state(translated, sites, base));
            }
        }
        if orbit.iter().any(|state| !parent_lookup.contains(state)) {
            return Err(QmbedError::IncompatibleSymmetry(
                "symmetry map leaves the selected magnetization sector".into(),
            ));
        }
        visited.extend(orbit.iter().copied());
        let representative = *orbit
            .iter()
            .min()
            .ok_or_else(|| QmbedError::InvalidSector("symmetry generated an empty orbit".into()))?;

        let mut coefficients = HashMap::<u128, Complex64>::new();
        for shift in 0..translations {
            let angle = normalized_momentum.map_or(0.0, |value| {
                -std::f64::consts::TAU * (value * shift) as f64 / sites as f64
            });
            let character = Complex64::from_polar(1.0, angle);
            let translated = rotate_lattice_state(representative, shift, sites, base);
            *coefficients
                .entry(translated)
                .or_insert(Complex64::new(0.0, 0.0)) += character;
            if let Some(parity_value) = parity {
                let reflected = reflect_lattice_state(translated, sites, base);
                *coefficients
                    .entry(reflected)
                    .or_insert(Complex64::new(0.0, 0.0)) += f64::from(parity_value) * character;
            }
        }
        coefficients.retain(|_, coefficient| coefficient.norm() > 1.0e-12);
        if coefficients.is_empty() {
            continue;
        }
        let representative_coefficient =
            coefficients
                .get(&representative)
                .copied()
                .ok_or(QmbedError::IncompatibleSymmetry(
                    "symmetry projection removed its orbit representative".into(),
                ))?;
        let gauge = representative_coefficient / representative_coefficient.norm();
        let norm = coefficients
            .values()
            .map(Complex64::norm_sqr)
            .sum::<f64>()
            .sqrt();
        let orbit_size = coefficients.len();
        let expected_magnitude = 1.0 / (orbit_size as f64).sqrt();
        for (&state, coefficient) in &coefficients {
            let normalized = *coefficient / (gauge * norm);
            if (normalized.norm() - expected_magnitude).abs() > 1.0e-10 {
                return Err(QmbedError::IncompatibleSymmetry(
                    "symmetry projection does not define a one-dimensional orbit sector".into(),
                ));
            }
            lookup.insert(
                state,
                SymmetryImage {
                    representative,
                    phase: normalized / expected_magnitude,
                    orbit_size,
                },
            );
        }
        sectors.push((representative, orbit_size));
    }

    sectors.sort_by_key(|(representative, _)| *representative);
    let (states, orbit_sizes) = sectors.into_iter().unzip();
    Ok((states, orbit_sizes, lookup, normalized_momentum, parity))
}

fn translate_fermion_state(state: u128, shift: usize, sites: usize) -> (u128, f64) {
    let normalized = shift % sites;
    if normalized == 0 {
        return (state, 1.0);
    }
    let translated = rotate_lattice_state(state, normalized, sites, 2);
    let wrapped_mask = ((1_u128 << normalized) - 1) << (sites - normalized);
    let wrapped = (state & wrapped_mask).count_ones() as usize;
    let retained = state.count_ones() as usize - wrapped;
    let sign = if wrapped * retained % 2 == 0 {
        1.0
    } else {
        -1.0
    };
    (translated, sign)
}

type FermionTranslationSector = (
    Vec<u128>,
    Vec<usize>,
    HashMap<u128, SymmetryImage>,
    Option<usize>,
);

fn fermion_translation_sector(
    parent_states: Vec<u128>,
    sites: usize,
    momentum: Option<i32>,
) -> Result<FermionTranslationSector> {
    if momentum.is_none() {
        let orbit_sizes = vec![1; parent_states.len()];
        let lookup = parent_states
            .iter()
            .copied()
            .map(|state| {
                (
                    state,
                    SymmetryImage {
                        representative: state,
                        phase: Complex64::new(1.0, 0.0),
                        orbit_size: 1,
                    },
                )
            })
            .collect();
        return Ok((parent_states, orbit_sizes, lookup, None));
    }
    if sites == 0 {
        return Err(QmbedError::InvalidSector(
            "translation sectors require at least one site".into(),
        ));
    }
    let sites_i64 = i64::try_from(sites)
        .map_err(|_| QmbedError::UnsupportedBackend("site count is too large".into()))?;
    let normalized = i64::from(momentum.unwrap_or_default()).rem_euclid(sites_i64) as usize;
    let mut visited = HashSet::with_capacity(parent_states.len());
    let mut sectors = Vec::<(u128, usize)>::new();
    let mut lookup = HashMap::with_capacity(parent_states.len());

    for seed in parent_states {
        if visited.contains(&seed) {
            continue;
        }
        let orbit: HashSet<_> = (0..sites)
            .map(|shift| translate_fermion_state(seed, shift, sites).0)
            .collect();
        visited.extend(orbit.iter().copied());
        let representative = *orbit.iter().min().ok_or_else(|| {
            QmbedError::InvalidSector("translation generated an empty fermion orbit".into())
        })?;
        let mut coefficients = HashMap::<u128, Complex64>::new();
        for shift in 0..sites {
            let (translated, sign) = translate_fermion_state(representative, shift, sites);
            let angle = -std::f64::consts::TAU * (normalized * shift) as f64 / sites as f64;
            *coefficients
                .entry(translated)
                .or_insert(Complex64::new(0.0, 0.0)) += sign * Complex64::from_polar(1.0, angle);
        }
        coefficients.retain(|_, coefficient| coefficient.norm() > 1.0e-12);
        if coefficients.is_empty() {
            continue;
        }
        let representative_coefficient =
            coefficients
                .get(&representative)
                .copied()
                .ok_or(QmbedError::IncompatibleSymmetry(
                    "translation projection removed its fermion representative".into(),
                ))?;
        let gauge = representative_coefficient / representative_coefficient.norm();
        let norm = coefficients
            .values()
            .map(Complex64::norm_sqr)
            .sum::<f64>()
            .sqrt();
        let orbit_size = coefficients.len();
        let expected_magnitude = 1.0 / (orbit_size as f64).sqrt();
        for (&state, coefficient) in &coefficients {
            let projected = *coefficient / (gauge * norm);
            if (projected.norm() - expected_magnitude).abs() > 1.0e-10 {
                return Err(QmbedError::IncompatibleSymmetry(
                    "fermion translation does not define a one-dimensional orbit sector".into(),
                ));
            }
            lookup.insert(
                state,
                SymmetryImage {
                    representative,
                    phase: projected / expected_magnitude,
                    orbit_size,
                },
            );
        }
        sectors.push((representative, orbit_size));
    }
    sectors.sort_by_key(|(representative, _)| *representative);
    if sectors.is_empty() {
        return Err(QmbedError::InvalidSector(
            "the requested fermion momentum sector is empty".into(),
        ));
    }
    let (states, orbit_sizes) = sectors.into_iter().unzip();
    Ok((states, orbit_sizes, lookup, Some(normalized)))
}

fn checked_site(site: usize, sites: usize) -> Result<()> {
    if site >= sites {
        Err(QmbedError::InvalidSite { site, sites })
    } else {
        Ok(())
    }
}

fn operator_chars(operator: &str, sites: &[usize]) -> Result<SmallVec<[char; 8]>> {
    let chars: SmallVec<[char; 8]> = operator
        .chars()
        .filter(|character| *character != '|')
        .collect();
    if chars.len() != sites.len() {
        return Err(QmbedError::InvalidCoupling(format!(
            "operator arity {} does not match {} sites",
            chars.len(),
            sites.len()
        )));
    }
    Ok(chars)
}

/// Normalization of local spin operators.
///
/// The distinction matters for spin one-half because two common interfaces
/// assign different meanings to the ladder symbols. `PauliCartesian` keeps
/// the conventional unit-amplitude sigma-plus/minus operators, while `Pauli`
/// scales every non-identity spin symbol by two relative to angular-momentum
/// operators.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum SpinNormalization {
    #[default]
    AngularMomentum,
    Pauli,
    PauliCartesian,
}

/// Spin-chain basis for the full or fixed-magnetization spin space.
#[derive(Clone, Debug)]
pub struct SpinBasis1D {
    sites: usize,
    spin_twice: u16,
    states_per_site: u128,
    radix_bits: Option<u32>,
    up: Option<usize>,
    normalization: SpinNormalization,
    place_values: Vec<u128>,
    z_factors: Vec<f64>,
    raise_factors: Vec<f64>,
    lower_factors: Vec<f64>,
    momentum: Option<usize>,
    parity: Option<i8>,
    orbit_lengths: Vec<usize>,
    symmetry_lookup: HashMap<u128, SymmetryImage>,
    states: Vec<u128>,
}

impl SpinBasis1D {
    pub fn builder(sites: usize) -> SpinBasisBuilder {
        SpinBasisBuilder {
            sites,
            spin_twice: 1,
            up: None,
            momentum: None,
            parity: None,
            normalization: SpinNormalization::AngularMomentum,
        }
    }

    pub const fn sites(&self) -> usize {
        self.sites
    }

    pub const fn spin_twice(&self) -> u16 {
        self.spin_twice
    }

    pub const fn up(&self) -> Option<usize> {
        self.up
    }

    pub const fn pauli(&self) -> bool {
        !matches!(self.normalization, SpinNormalization::AngularMomentum)
    }

    pub const fn normalization(&self) -> SpinNormalization {
        self.normalization
    }

    pub const fn momentum(&self) -> Option<usize> {
        self.momentum
    }

    pub const fn parity(&self) -> Option<i8> {
        self.parity
    }

    fn unreduced_local_transitions(
        &self,
        state: u128,
        operator: &str,
        sites: &[usize],
    ) -> Result<LocalTransitions<u128>> {
        let mut transitions = LocalTransitions::new();
        self.visit_unreduced_local_transitions(state, operator, sites, |target, amplitude| {
            transitions.push((target, amplitude));
            Ok(())
        })?;
        Ok(transitions)
    }

    fn visit_unreduced_local_transitions<F>(
        &self,
        state: u128,
        operator: &str,
        sites: &[usize],
        visit: F,
    ) -> Result<()>
    where
        F: FnMut(u128, Complex64) -> Result<()>,
    {
        let symbols = operator_chars(operator, sites)?;
        self.visit_unreduced_local_transitions_with_symbols(state, &symbols, sites, visit)
    }

    fn visit_unreduced_local_transitions_with_symbols<F>(
        &self,
        state: u128,
        symbols: &[char],
        sites: &[usize],
        mut visit: F,
    ) -> Result<()>
    where
        F: FnMut(u128, Complex64) -> Result<()>,
    {
        if symbols.len() != sites.len() {
            return Err(QmbedError::InvalidCoupling(format!(
                "operator arity {} does not match {} sites",
                symbols.len(),
                sites.len()
            )));
        }
        let mut pending = SmallVec::<[(u128, Complex64, usize); 2]>::new();
        pending.push((state, Complex64::new(1.0, 0.0), symbols.len()));
        while let Some((mut encoded, mut amplitude, mut remaining)) = pending.pop() {
            loop {
                if remaining == 0 {
                    visit(encoded, amplitude)?;
                    break;
                }
                let position = remaining - 1;
                let site = sites[position];
                let op = symbols[position];
                checked_site(site, self.sites)?;
                let place = self.place_values[site];
                let encoded_digit = self.radix_bits.map_or_else(
                    || (encoded / place) % self.states_per_site,
                    |bits| {
                        (encoded >> (bits * u32::try_from(site).unwrap_or(u32::MAX)))
                            & (self.states_per_site - 1)
                    },
                );
                let digit =
                    usize::try_from(encoded_digit).map_err(|_| QmbedError::StateNotInBasis)?;
                let raise_factor = self.raise_factors[digit];
                let lower_factor = self.lower_factors[digit];
                match op {
                    'I' => {}
                    'z' => {
                        let factor = self.z_factors[digit];
                        if factor == 0.0 {
                            break;
                        }
                        amplitude *= factor;
                    }
                    '+' => {
                        if raise_factor == 0.0 {
                            break;
                        }
                        encoded += place;
                        if raise_factor != 1.0 {
                            amplitude *= raise_factor;
                        }
                    }
                    '-' => {
                        if lower_factor == 0.0 {
                            break;
                        }
                        encoded -= place;
                        if lower_factor != 1.0 {
                            amplitude *= lower_factor;
                        }
                    }
                    'x' | 'y' => {
                        let scale = match self.normalization {
                            SpinNormalization::AngularMomentum | SpinNormalization::Pauli => 0.5,
                            SpinNormalization::PauliCartesian => 1.0,
                        };
                        let raise_phase = if op == 'x' {
                            Complex64::new(scale, 0.0)
                        } else {
                            Complex64::new(0.0, -scale)
                        };
                        let lower_phase = if op == 'x' {
                            Complex64::new(scale, 0.0)
                        } else {
                            Complex64::new(0.0, scale)
                        };
                        match (raise_factor != 0.0, lower_factor != 0.0) {
                            (true, true) => {
                                pending.push((
                                    encoded - place,
                                    amplitude * lower_phase * lower_factor,
                                    position,
                                ));
                                encoded += place;
                                amplitude *= raise_phase * raise_factor;
                            }
                            (true, false) => {
                                encoded += place;
                                amplitude *= raise_phase * raise_factor;
                            }
                            (false, true) => {
                                encoded -= place;
                                amplitude *= lower_phase * lower_factor;
                            }
                            (false, false) => break,
                        }
                    }
                    _ => return Err(QmbedError::InvalidOperator(op.to_string())),
                }
                remaining = position;
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct SpinBasisBuilder {
    sites: usize,
    spin_twice: u16,
    up: Option<usize>,
    momentum: Option<i32>,
    parity: Option<i8>,
    normalization: SpinNormalization,
}

impl SpinBasisBuilder {
    pub const fn spin_twice(mut self, spin_twice: u16) -> Self {
        self.spin_twice = spin_twice;
        self
    }

    pub const fn up(mut self, up: usize) -> Self {
        self.up = Some(up);
        self
    }

    pub const fn magnetization(mut self, up: usize) -> Self {
        self.up = Some(up);
        self
    }

    pub const fn momentum(mut self, momentum: i32) -> Self {
        self.momentum = Some(momentum);
        self
    }

    pub const fn parity(mut self, parity: i8) -> Self {
        self.parity = Some(parity);
        self
    }

    pub const fn pauli(mut self, pauli: bool) -> Self {
        self.normalization = if pauli {
            SpinNormalization::PauliCartesian
        } else {
            SpinNormalization::AngularMomentum
        };
        self
    }

    pub const fn normalization(mut self, normalization: SpinNormalization) -> Self {
        self.normalization = normalization;
        self
    }

    pub fn build(self) -> Result<SpinBasis1D> {
        if self.spin_twice == 0 {
            return Err(QmbedError::InvalidSector(
                "spin_twice must be positive".into(),
            ));
        }
        if self.normalization != SpinNormalization::AngularMomentum && self.spin_twice != 1 {
            return Err(QmbedError::InvalidOptions(
                "the Pauli convention is defined only for spin one-half".into(),
            ));
        }
        let states_per_site = usize::from(self.spin_twice) + 1;
        let states_per_site_u128 = states_per_site as u128;
        let radix_bits = states_per_site_u128
            .is_power_of_two()
            .then_some(states_per_site_u128.trailing_zeros());
        let place_values = (0..self.sites)
            .map(|site| {
                states_per_site_u128
                    .checked_pow(u32::try_from(site).unwrap_or(u32::MAX))
                    .ok_or_else(|| {
                        QmbedError::UnsupportedBackend(
                            "spin-state place value exceeds the u128 backend".into(),
                        )
                    })
            })
            .collect::<Result<Vec<_>>>()?;
        let spin = f64::from(self.spin_twice) * 0.5;
        let mut z_factors = Vec::with_capacity(states_per_site);
        let mut raise_factors = Vec::with_capacity(states_per_site);
        let mut lower_factors = Vec::with_capacity(states_per_site);
        for digit in 0..states_per_site {
            let magnetic = digit as f64 - spin;
            z_factors.push(
                if self.normalization == SpinNormalization::AngularMomentum {
                    magnetic
                } else {
                    2.0 * magnetic
                },
            );
            let ladder_scale = if self.normalization == SpinNormalization::Pauli {
                2.0
            } else {
                1.0
            };
            raise_factors.push(
                if digit + 1 < states_per_site {
                    (spin * (spin + 1.0) - magnetic * (magnetic + 1.0)).sqrt()
                } else {
                    0.0
                } * ladder_scale,
            );
            lower_factors.push(
                if digit > 0 {
                    (spin * (spin + 1.0) - magnetic * (magnetic - 1.0)).sqrt()
                } else {
                    0.0
                } * ladder_scale,
            );
        }
        let parent_states = if self.spin_twice == 1 {
            fixed_weight_states(self.sites, self.up)?
        } else {
            fixed_digit_sum_states(self.sites, states_per_site, self.up)?
        };
        let (states, orbit_lengths, symmetry_lookup, momentum, parity) = spin_symmetry_sector(
            parent_states,
            self.sites,
            states_per_site_u128,
            self.momentum,
            self.parity,
        )?;
        Ok(SpinBasis1D {
            sites: self.sites,
            spin_twice: self.spin_twice,
            states_per_site: states_per_site_u128,
            radix_bits,
            up: self.up,
            normalization: self.normalization,
            place_values,
            z_factors,
            raise_factors,
            lower_factors,
            momentum,
            parity,
            orbit_lengths,
            symmetry_lookup,
            states,
        })
    }
}

impl Basis for SpinBasis1D {
    type State = u128;

    fn len(&self) -> usize {
        self.states.len()
    }

    fn state(&self, index: usize) -> Result<Self::State> {
        self.states
            .get(index)
            .copied()
            .ok_or(QmbedError::StateNotInBasis)
    }

    fn index(&self, state: Self::State) -> Result<usize> {
        if self.momentum.is_none() && self.parity.is_none() {
            if self.spin_twice == 1 {
                return self.up.map_or_else(
                    || direct_state_index(&self.states, state),
                    |up| fixed_weight_state_index(state, self.sites, up),
                );
            }
            if self.up.is_none() {
                return direct_state_index(&self.states, state);
            }
        }
        state_index(&self.states, state)
    }

    fn apply_local(
        &self,
        state: Self::State,
        operator: &str,
        sites: &[usize],
    ) -> Result<Option<(Self::State, Complex64)>> {
        let transitions = self.apply_local_transitions(state, operator, sites)?;
        match transitions.as_slice() {
            [] => Ok(None),
            [transition] => Ok(Some(*transition)),
            _ => Err(QmbedError::UnsupportedBackend(
                "this higher-spin local action branches; use apply_local_transitions".into(),
            )),
        }
    }

    fn apply_local_transitions(
        &self,
        state: Self::State,
        operator: &str,
        sites: &[usize],
    ) -> Result<LocalTransitions<Self::State>> {
        let source_state = state;
        let branches = self.unreduced_local_transitions(state, operator, sites)?;
        if self.momentum.is_some() || self.parity.is_some() {
            let source_index = self.index(source_state)?;
            let source_orbit = self.orbit_lengths[source_index];
            let mut reduced = HashMap::<u128, Complex64>::new();
            for (encoded, mut amplitude) in branches {
                let Some(image) = self.symmetry_lookup.get(&encoded) else {
                    continue;
                };
                amplitude *=
                    (source_orbit as f64 / image.orbit_size as f64).sqrt() * image.phase.conj();
                *reduced
                    .entry(image.representative)
                    .or_insert(Complex64::new(0.0, 0.0)) += amplitude;
            }
            let mut transitions: LocalTransitions<_> = reduced
                .into_iter()
                .filter(|(_, amplitude)| amplitude.norm() > f64::EPSILON)
                .collect();
            transitions.sort_by_key(|(encoded, _)| *encoded);
            return Ok(transitions);
        }
        Ok(branches)
    }

    fn apply_local_unreduced_transitions(
        &self,
        state: Self::State,
        operator: &str,
        sites: &[usize],
    ) -> Result<LocalTransitions<Self::State>> {
        self.unreduced_local_transitions(state, operator, sites)
    }

    fn visit_local_unreduced_transitions<F>(
        &self,
        state: Self::State,
        operator: &str,
        sites: &[usize],
        visit: F,
    ) -> Result<()>
    where
        F: FnMut(Self::State, Complex64) -> Result<()>,
    {
        self.visit_unreduced_local_transitions(state, operator, sites, visit)
    }

    fn visit_preparsed_local_unreduced_transitions<F>(
        &self,
        state: Self::State,
        _operator: &str,
        symbols: &[char],
        _split: Option<usize>,
        sites: &[usize],
        visit: F,
    ) -> Result<()>
    where
        F: FnMut(Self::State, Complex64) -> Result<()>,
    {
        self.visit_unreduced_local_transitions_with_symbols(state, symbols, sites, visit)
    }

    fn transition_orbit_size(&self, state: Self::State) -> Result<usize> {
        if self.momentum.is_none() && self.parity.is_none() {
            return Ok(1);
        }
        Ok(self.orbit_lengths[self.index(state)?])
    }

    fn reduction_image(&self, state: Self::State) -> Result<Option<ReductionImage<Self::State>>> {
        let Some(image) = self.symmetry_lookup.get(&state) else {
            return Ok(None);
        };
        Ok(Some(ReductionImage::new(
            image.representative,
            image.phase,
            image.orbit_size,
        )?))
    }

    fn reduce_transition(
        &self,
        state: Self::State,
        source_orbit_size: usize,
    ) -> Result<Option<(Self::State, Complex64)>> {
        let Some(image) = self.symmetry_lookup.get(&state) else {
            return Ok(None);
        };
        Ok(Some((
            image.representative,
            (source_orbit_size as f64 / image.orbit_size as f64).sqrt() * image.phase.conj(),
        )))
    }

    fn index_transition(
        &self,
        state: Self::State,
        source_orbit_size: usize,
    ) -> Result<Option<(usize, Complex64)>> {
        if self.momentum.is_none() && self.parity.is_none() {
            return match self.index(state) {
                Ok(index) => Ok(Some((index, Complex64::new(1.0, 0.0)))),
                Err(QmbedError::StateNotInBasis) => Ok(None),
                Err(error) => Err(error),
            };
        }
        let Some(image) = self.symmetry_lookup.get(&state) else {
            return Ok(None);
        };
        Ok(Some((
            self.index(image.representative)?,
            (source_orbit_size as f64 / image.orbit_size as f64).sqrt() * image.phase.conj(),
        )))
    }

    fn operator_preserves_particle_sector(&self, operator: &str) -> Result<bool> {
        Ok(self.up.is_none() || operator_number_change(operator)? == Some(0))
    }
}

/// Truncated on-site boson basis.
#[derive(Clone, Debug)]
pub struct BosonBasis1D {
    sites: usize,
    particles: Option<usize>,
    states_per_site: usize,
    states: Vec<u128>,
}

impl BosonBasis1D {
    pub fn builder(sites: usize, states_per_site: usize) -> BosonBasisBuilder {
        BosonBasisBuilder {
            sites,
            particles: None,
            states_per_site,
        }
    }

    pub const fn sites(&self) -> usize {
        self.sites
    }

    pub const fn particles(&self) -> Option<usize> {
        self.particles
    }

    pub const fn states_per_site(&self) -> usize {
        self.states_per_site
    }

    fn apply_local_symbols(
        &self,
        mut state: u128,
        symbols: &[char],
        sites: &[usize],
    ) -> Result<Option<(u128, Complex64)>> {
        if symbols.len() != sites.len() {
            return Err(QmbedError::InvalidCoupling(format!(
                "operator arity {} does not match {} sites",
                symbols.len(),
                sites.len()
            )));
        }
        let base = self.states_per_site as u128;
        let mut amplitude = Complex64::new(1.0, 0.0);
        for (&site, &op) in sites.iter().zip(symbols).rev() {
            checked_site(site, self.sites)?;
            let place = base.pow(u32::try_from(site).unwrap_or(u32::MAX));
            let occupation = (state / place) % base;
            match op {
                'I' => {}
                'n' => amplitude *= occupation as f64,
                '+' if occupation + 1 < base => {
                    state += place;
                    amplitude *= ((occupation + 1) as f64).sqrt();
                }
                '-' if occupation > 0 => {
                    state -= place;
                    amplitude *= (occupation as f64).sqrt();
                }
                '+' | '-' => return Ok(None),
                _ => return Err(QmbedError::InvalidOperator(op.to_string())),
            }
        }
        Ok(Some((state, amplitude)))
    }
}

#[derive(Clone, Debug)]
pub struct BosonBasisBuilder {
    sites: usize,
    particles: Option<usize>,
    states_per_site: usize,
}

impl BosonBasisBuilder {
    pub const fn particles(mut self, particles: usize) -> Self {
        self.particles = Some(particles);
        self
    }

    pub fn build(self) -> Result<BosonBasis1D> {
        if self.sites == 0 || self.states_per_site == 0 {
            return Err(QmbedError::InvalidSector(
                "boson sites and states_per_site must be positive".into(),
            ));
        }
        if self
            .particles
            .is_some_and(|count| count > self.sites * (self.states_per_site - 1))
        {
            return Err(QmbedError::InvalidSector(
                "particle count exceeds the local cutoff".into(),
            ));
        }
        let states = fixed_digit_sum_states(self.sites, self.states_per_site, self.particles)?;
        if states.is_empty() {
            return Err(QmbedError::InvalidSector("empty boson sector".into()));
        }
        Ok(BosonBasis1D {
            sites: self.sites,
            particles: self.particles,
            states_per_site: self.states_per_site,
            states,
        })
    }
}

impl Basis for BosonBasis1D {
    type State = u128;

    fn len(&self) -> usize {
        self.states.len()
    }

    fn state(&self, index: usize) -> Result<Self::State> {
        self.states
            .get(index)
            .copied()
            .ok_or(QmbedError::StateNotInBasis)
    }

    fn index(&self, state: Self::State) -> Result<usize> {
        if self.particles.is_none() {
            return direct_state_index(&self.states, state);
        }
        state_index(&self.states, state)
    }

    fn apply_local(
        &self,
        state: Self::State,
        operator: &str,
        sites: &[usize],
    ) -> Result<Option<(Self::State, Complex64)>> {
        let symbols = operator_chars(operator, sites)?;
        self.apply_local_symbols(state, &symbols, sites)
    }

    fn visit_preparsed_local_unreduced_transitions<F>(
        &self,
        state: Self::State,
        _operator: &str,
        symbols: &[char],
        _split: Option<usize>,
        sites: &[usize],
        mut visit: F,
    ) -> Result<()>
    where
        F: FnMut(Self::State, Complex64) -> Result<()>,
    {
        if let Some((target, amplitude)) = self.apply_local_symbols(state, symbols, sites)? {
            visit(target, amplitude)?;
        }
        Ok(())
    }

    fn operator_preserves_particle_sector(&self, operator: &str) -> Result<bool> {
        Ok(self.particles.is_none() || operator_number_change(operator)? == Some(0))
    }
}

/// Single-flavor fermion basis.
#[derive(Clone, Debug)]
pub struct SpinlessFermionBasis1D {
    sites: usize,
    particles: Option<usize>,
    momentum: Option<usize>,
    orbit_lengths: Vec<usize>,
    symmetry_lookup: HashMap<u128, SymmetryImage>,
    states: Vec<u128>,
}

impl SpinlessFermionBasis1D {
    pub fn builder(sites: usize) -> SpinlessFermionBasisBuilder {
        SpinlessFermionBasisBuilder {
            sites,
            particles: None,
            momentum: None,
        }
    }

    pub const fn sites(&self) -> usize {
        self.sites
    }

    pub const fn particles(&self) -> Option<usize> {
        self.particles
    }

    pub const fn momentum(&self) -> Option<usize> {
        self.momentum
    }

    fn unreduced_local_transition(
        &self,
        state: u128,
        operator: &str,
        sites: &[usize],
    ) -> Result<Option<(u128, Complex64)>> {
        let symbols = operator_chars(operator, sites)?;
        self.unreduced_local_transition_with_symbols(state, &symbols, sites)
    }

    fn unreduced_local_transition_with_symbols(
        &self,
        mut state: u128,
        symbols: &[char],
        sites: &[usize],
    ) -> Result<Option<(u128, Complex64)>> {
        if symbols.len() != sites.len() {
            return Err(QmbedError::InvalidCoupling(format!(
                "operator arity {} does not match {} sites",
                symbols.len(),
                sites.len()
            )));
        }
        let mut amplitude = Complex64::new(1.0, 0.0);
        for (&site, &op) in sites.iter().zip(symbols).rev() {
            checked_site(site, self.sites)?;
            let Some((next, local)) = apply_fermion(state, site, op)? else {
                return Ok(None);
            };
            state = next;
            amplitude *= local;
        }
        Ok(Some((state, amplitude)))
    }
}

#[derive(Clone, Debug)]
pub struct SpinlessFermionBasisBuilder {
    sites: usize,
    particles: Option<usize>,
    momentum: Option<i32>,
}

impl SpinlessFermionBasisBuilder {
    pub const fn particles(mut self, particles: usize) -> Self {
        self.particles = Some(particles);
        self
    }

    pub const fn momentum(mut self, momentum: i32) -> Self {
        self.momentum = Some(momentum);
        self
    }

    pub fn build(self) -> Result<SpinlessFermionBasis1D> {
        let parent_states = fixed_weight_states(self.sites, self.particles)?;
        let (states, orbit_lengths, symmetry_lookup, momentum) =
            fermion_translation_sector(parent_states, self.sites, self.momentum)?;
        Ok(SpinlessFermionBasis1D {
            sites: self.sites,
            particles: self.particles,
            momentum,
            orbit_lengths,
            symmetry_lookup,
            states,
        })
    }
}

fn apply_fermion(mut state: u128, orbital: usize, op: char) -> Result<Option<(u128, Complex64)>> {
    let mask = 1_u128 << orbital;
    let occupied = state & mask != 0;
    let prior_mask = mask - 1;
    let sign = if (state & prior_mask).count_ones() % 2 == 0 {
        1.0
    } else {
        -1.0
    };
    let amplitude = match op {
        'I' => 1.0,
        'n' => return Ok(occupied.then_some((state, Complex64::new(1.0, 0.0)))),
        'z' => {
            return Ok(Some((
                state,
                Complex64::new(if occupied { 1.0 } else { -1.0 }, 0.0),
            )));
        }
        '+' if !occupied => {
            state |= mask;
            sign
        }
        '-' if occupied => {
            state &= !mask;
            sign
        }
        'x' => {
            state ^= mask;
            sign
        }
        'y' => {
            state ^= mask;
            return Ok(Some((
                state,
                Complex64::new(0.0, if occupied { sign } else { -sign }),
            )));
        }
        '+' | '-' => return Ok(None),
        _ => return Err(QmbedError::InvalidOperator(op.to_string())),
    };
    Ok(Some((state, Complex64::new(amplitude, 0.0))))
}

impl Basis for SpinlessFermionBasis1D {
    type State = u128;

    fn len(&self) -> usize {
        self.states.len()
    }

    fn state(&self, index: usize) -> Result<Self::State> {
        self.states
            .get(index)
            .copied()
            .ok_or(QmbedError::StateNotInBasis)
    }

    fn index(&self, state: Self::State) -> Result<usize> {
        if self.momentum.is_none() {
            return self.particles.map_or_else(
                || direct_state_index(&self.states, state),
                |particles| fixed_weight_state_index(state, self.sites, particles),
            );
        }
        state_index(&self.states, state)
    }

    fn apply_local(
        &self,
        mut state: Self::State,
        operator: &str,
        sites: &[usize],
    ) -> Result<Option<(Self::State, Complex64)>> {
        let source_state = state;
        let chars = operator_chars(operator, sites)?;
        let mut amplitude = Complex64::new(1.0, 0.0);
        for (&site, op) in sites.iter().zip(chars).rev() {
            checked_site(site, self.sites)?;
            let Some((next, local)) = apply_fermion(state, site, op)? else {
                return Ok(None);
            };
            state = next;
            amplitude *= local;
        }
        if self.momentum.is_some() {
            let source_index = self.index(source_state)?;
            let source_orbit = self.orbit_lengths[source_index];
            let Some(image) = self.symmetry_lookup.get(&state) else {
                return Ok(None);
            };
            amplitude *=
                (source_orbit as f64 / image.orbit_size as f64).sqrt() * image.phase.conj();
            state = image.representative;
        }
        Ok(Some((state, amplitude)))
    }

    fn apply_local_unreduced_transitions(
        &self,
        state: Self::State,
        operator: &str,
        sites: &[usize],
    ) -> Result<LocalTransitions<Self::State>> {
        Ok(self
            .unreduced_local_transition(state, operator, sites)?
            .into_iter()
            .collect())
    }

    fn visit_local_unreduced_transitions<F>(
        &self,
        state: Self::State,
        operator: &str,
        sites: &[usize],
        mut visit: F,
    ) -> Result<()>
    where
        F: FnMut(Self::State, Complex64) -> Result<()>,
    {
        if let Some((target, amplitude)) =
            self.unreduced_local_transition(state, operator, sites)?
        {
            visit(target, amplitude)?;
        }
        Ok(())
    }

    fn visit_preparsed_local_unreduced_transitions<F>(
        &self,
        state: Self::State,
        _operator: &str,
        symbols: &[char],
        _split: Option<usize>,
        sites: &[usize],
        mut visit: F,
    ) -> Result<()>
    where
        F: FnMut(Self::State, Complex64) -> Result<()>,
    {
        if let Some((target, amplitude)) =
            self.unreduced_local_transition_with_symbols(state, symbols, sites)?
        {
            visit(target, amplitude)?;
        }
        Ok(())
    }

    fn transition_orbit_size(&self, state: Self::State) -> Result<usize> {
        if self.momentum.is_none() {
            return Ok(1);
        }
        Ok(self.orbit_lengths[self.index(state)?])
    }

    fn reduction_image(&self, state: Self::State) -> Result<Option<ReductionImage<Self::State>>> {
        let Some(image) = self.symmetry_lookup.get(&state) else {
            return Ok(None);
        };
        Ok(Some(ReductionImage::new(
            image.representative,
            image.phase,
            image.orbit_size,
        )?))
    }

    fn reduce_transition(
        &self,
        state: Self::State,
        source_orbit_size: usize,
    ) -> Result<Option<(Self::State, Complex64)>> {
        let Some(image) = self.symmetry_lookup.get(&state) else {
            return Ok(None);
        };
        Ok(Some((
            image.representative,
            (source_orbit_size as f64 / image.orbit_size as f64).sqrt() * image.phase.conj(),
        )))
    }

    fn index_transition(
        &self,
        state: Self::State,
        source_orbit_size: usize,
    ) -> Result<Option<(usize, Complex64)>> {
        if self.momentum.is_none() {
            return match self.index(state) {
                Ok(index) => Ok(Some((index, Complex64::new(1.0, 0.0)))),
                Err(QmbedError::StateNotInBasis) => Ok(None),
                Err(error) => Err(error),
            };
        }
        let Some(image) = self.symmetry_lookup.get(&state) else {
            return Ok(None);
        };
        Ok(Some((
            self.index(image.representative)?,
            (source_orbit_size as f64 / image.orbit_size as f64).sqrt() * image.phase.conj(),
        )))
    }

    fn operator_preserves_particle_sector(&self, operator: &str) -> Result<bool> {
        Ok(self.particles.is_none() || operator_number_change(operator)? == Some(0))
    }
}

/// Two-flavor fermion basis with all up orbitals ordered before all down orbitals.
#[derive(Clone, Debug)]
pub struct SpinfulFermionBasis1D {
    sites: usize,
    particles_up: Option<usize>,
    particles_down: Option<usize>,
    particle_sectors: Option<Vec<(usize, usize)>>,
    states: Vec<u128>,
}

impl SpinfulFermionBasis1D {
    pub fn builder(sites: usize) -> SpinfulFermionBasisBuilder {
        SpinfulFermionBasisBuilder {
            sites,
            particles_up: None,
            particles_down: None,
            particle_sectors: None,
        }
    }

    pub const fn sites(&self) -> usize {
        self.sites
    }

    pub const fn particles_up(&self) -> Option<usize> {
        self.particles_up
    }

    pub const fn particles_down(&self) -> Option<usize> {
        self.particles_down
    }

    pub fn particle_sectors(&self) -> Option<&[(usize, usize)]> {
        self.particle_sectors.as_deref()
    }

    fn apply_local_symbols(
        &self,
        mut state: u128,
        symbols: &[char],
        split: usize,
        sites: &[usize],
    ) -> Result<Option<(u128, Complex64)>> {
        if symbols.len() != sites.len() || split > symbols.len() {
            return Err(QmbedError::InvalidCoupling(format!(
                "operator arity {} does not match {} sites",
                symbols.len(),
                sites.len()
            )));
        }
        let mut amplitude = Complex64::new(1.0, 0.0);
        for (position, (&site, &op)) in sites.iter().zip(symbols).enumerate().rev() {
            checked_site(site, self.sites)?;
            let orbital = if position < split {
                site
            } else {
                self.sites + site
            };
            let Some((next, local)) = apply_fermion(state, orbital, op)? else {
                return Ok(None);
            };
            state = next;
            amplitude *= local;
        }
        Ok(Some((state, amplitude)))
    }
}

#[derive(Clone, Debug)]
pub struct SpinfulFermionBasisBuilder {
    sites: usize,
    particles_up: Option<usize>,
    particles_down: Option<usize>,
    particle_sectors: Option<Vec<(usize, usize)>>,
}

impl SpinfulFermionBasisBuilder {
    pub const fn particles_up(mut self, particles: usize) -> Self {
        self.particles_up = Some(particles);
        self
    }

    pub const fn particles_down(mut self, particles: usize) -> Self {
        self.particles_down = Some(particles);
        self
    }

    pub fn particles(mut self, up: usize, down: usize) -> Self {
        self.particles_up = Some(up);
        self.particles_down = Some(down);
        self.particle_sectors = None;
        self
    }

    /// Select a union of fixed `(N_up, N_down)` sectors.
    pub fn particle_sectors(mut self, sectors: impl IntoIterator<Item = (usize, usize)>) -> Self {
        self.particle_sectors = Some(sectors.into_iter().collect());
        self.particles_up = None;
        self.particles_down = None;
        self
    }

    pub fn build(self) -> Result<SpinfulFermionBasis1D> {
        if self.sites > 64 {
            return Err(QmbedError::UnsupportedBackend(
                "the packed spinful backend supports at most 64 sites".into(),
            ));
        }
        let sectors = match &self.particle_sectors {
            Some(sectors) if sectors.is_empty() => {
                return Err(QmbedError::InvalidSector(
                    "spinful particle-sector union must be nonempty".into(),
                ));
            }
            Some(sectors) => sectors.clone(),
            None => vec![(
                self.particles_up.unwrap_or(usize::MAX),
                self.particles_down.unwrap_or(usize::MAX),
            )],
        };
        let mut states = Vec::new();
        for (up_count, down_count) in sectors {
            let up_states =
                fixed_weight_states(self.sites, (up_count != usize::MAX).then_some(up_count))?;
            let down_states =
                fixed_weight_states(self.sites, (down_count != usize::MAX).then_some(down_count))?;
            states.reserve(up_states.len().saturating_mul(down_states.len()));
            for down in down_states {
                for &up in &up_states {
                    states.push(up | (down << self.sites));
                }
            }
        }
        states.sort_unstable();
        states.dedup();
        Ok(SpinfulFermionBasis1D {
            sites: self.sites,
            particles_up: self.particles_up,
            particles_down: self.particles_down,
            particle_sectors: self.particle_sectors,
            states,
        })
    }
}

impl Basis for SpinfulFermionBasis1D {
    type State = u128;

    fn len(&self) -> usize {
        self.states.len()
    }

    fn state(&self, index: usize) -> Result<Self::State> {
        self.states
            .get(index)
            .copied()
            .ok_or(QmbedError::StateNotInBasis)
    }

    fn index(&self, state: Self::State) -> Result<usize> {
        if self.particle_sectors.is_none() {
            let mask = if self.sites == 128 {
                u128::MAX
            } else {
                (1_u128 << self.sites) - 1
            };
            let up_state = state & mask;
            let down_state = state >> self.sites;
            let up_index = self.particles_up.map_or_else(
                || usize::try_from(up_state).map_err(|_| QmbedError::StateNotInBasis),
                |particles| fixed_weight_state_index(up_state, self.sites, particles),
            )?;
            let down_index = self.particles_down.map_or_else(
                || usize::try_from(down_state).map_err(|_| QmbedError::StateNotInBasis),
                |particles| fixed_weight_state_index(down_state, self.sites, particles),
            )?;
            let up_dimension = match self.particles_up {
                Some(particles) => binomial(self.sites, particles),
                None => 1_usize
                    .checked_shl(u32::try_from(self.sites).unwrap_or(u32::MAX))
                    .ok_or(QmbedError::StateNotInBasis)?,
            };
            let index = down_index
                .checked_mul(up_dimension)
                .and_then(|offset| offset.checked_add(up_index))
                .ok_or(QmbedError::StateNotInBasis)?;
            if index < self.states.len() {
                return Ok(index);
            }
            return Err(QmbedError::StateNotInBasis);
        }
        state_index(&self.states, state)
    }

    fn apply_local(
        &self,
        state: Self::State,
        operator: &str,
        sites: &[usize],
    ) -> Result<Option<(Self::State, Complex64)>> {
        let symbols = operator_chars(operator, sites)?;
        let split = operator.find('|').map_or(symbols.len(), |position| {
            operator[..position].chars().count()
        });
        self.apply_local_symbols(state, &symbols, split, sites)
    }

    fn visit_preparsed_local_unreduced_transitions<F>(
        &self,
        state: Self::State,
        _operator: &str,
        symbols: &[char],
        split: Option<usize>,
        sites: &[usize],
        mut visit: F,
    ) -> Result<()>
    where
        F: FnMut(Self::State, Complex64) -> Result<()>,
    {
        if let Some((target, amplitude)) =
            self.apply_local_symbols(state, symbols, split.unwrap_or(symbols.len()), sites)?
        {
            visit(target, amplitude)?;
        }
        Ok(())
    }

    fn operator_preserves_particle_sector(&self, operator: &str) -> Result<bool> {
        let (up_operator, down_operator) = operator.split_once('|').unwrap_or((operator, ""));
        if down_operator.contains('|') {
            return Err(QmbedError::InvalidOperator(operator.into()));
        }
        let Some(up_change) = operator_number_change(up_operator)? else {
            return Ok(self.particles_up.is_none()
                && self.particle_sectors.is_none()
                && self.particles_down.is_none());
        };
        let Some(down_change) = operator_number_change(down_operator)? else {
            return Ok(self.particles_up.is_none()
                && self.particle_sectors.is_none()
                && self.particles_down.is_none());
        };
        if let Some(sectors) = &self.particle_sectors {
            let sectors: HashSet<_> = sectors.iter().copied().collect();
            return Ok(sectors.iter().all(|&(up, down)| {
                let target_up = up as i32 + up_change;
                let target_down = down as i32 + down_change;
                target_up >= 0
                    && target_down >= 0
                    && target_up <= self.sites as i32
                    && target_down <= self.sites as i32
                    && sectors.contains(&(target_up as usize, target_down as usize))
            }));
        }
        Ok(self.particles_up.is_none_or(|_| up_change == 0)
            && self.particles_down.is_none_or(|_| down_change == 0))
    }
}

type UserAction<State> = Arc<
    dyn Fn(State, usize, &mut dyn FnMut(State, Complex64) -> Result<()>) -> Result<()>
        + Send
        + Sync,
>;
type UserStateFactory<State> = Arc<dyn Fn() -> Result<Vec<State>> + Send + Sync>;

/// Callback-defined constrained basis using the same assembly path as built-ins.
pub struct UserBasis<State>
where
    State: Copy + Eq + Hash + Send + Sync,
{
    sites: usize,
    states: Vec<State>,
    indices: HashMap<State, usize>,
    operators: HashMap<char, UserAction<State>>,
}

impl<State> UserBasis<State>
where
    State: Copy + Eq + Hash + Send + Sync,
{
    pub fn builder(sites: usize) -> UserBasisBuilder<State> {
        UserBasisBuilder {
            sites,
            states: Vec::new(),
            state_factory: None,
            operators: HashMap::new(),
        }
    }
}

pub struct UserBasisBuilder<State>
where
    State: Copy + Eq + Hash + Send + Sync,
{
    sites: usize,
    states: Vec<State>,
    state_factory: Option<UserStateFactory<State>>,
    operators: HashMap<char, UserAction<State>>,
}

impl<State> UserBasisBuilder<State>
where
    State: Copy + Eq + Hash + Send + Sync + 'static,
{
    pub fn states(mut self, states: impl IntoIterator<Item = State>) -> Self {
        self.states = states.into_iter().collect();
        self.state_factory = None;
        self
    }

    /// Defer potentially expensive state enumeration until `materialize` or
    /// `build` is called.
    pub fn deferred_states<F>(mut self, factory: F) -> Self
    where
        F: Fn() -> Result<Vec<State>> + Send + Sync + 'static,
    {
        self.states.clear();
        self.state_factory = Some(Arc::new(factory));
        self
    }

    pub fn operator<F>(mut self, name: char, action: F) -> Self
    where
        F: Fn(State, usize) -> Result<Option<(State, Complex64)>> + Send + Sync + 'static,
    {
        self.operators.insert(
            name,
            Arc::new(move |state, site, visit| {
                if let Some((target, amplitude)) = action(state, site)? {
                    visit(target, amplitude)?;
                }
                Ok(())
            }),
        );
        self
    }

    /// Register a local action with more than one nonzero destination.
    pub fn branching_operator<F>(mut self, name: char, action: F) -> Self
    where
        F: Fn(State, usize) -> Result<Vec<(State, Complex64)>> + Send + Sync + 'static,
    {
        self.operators.insert(
            name,
            Arc::new(move |state, site, visit| {
                for (target, amplitude) in action(state, site)? {
                    visit(target, amplitude)?;
                }
                Ok(())
            }),
        );
        self
    }

    pub fn build(mut self) -> Result<UserBasis<State>> {
        if self.states.is_empty() {
            if let Some(factory) = self.state_factory.take() {
                self.states = factory()?;
            }
        }
        if self.states.is_empty() {
            return Err(QmbedError::InvalidSector(
                "UserBasis requires at least one accepted state".into(),
            ));
        }
        let mut indices = HashMap::with_capacity(self.states.len());
        for (index, state) in self.states.iter().copied().enumerate() {
            if indices.insert(state, index).is_some() {
                return Err(QmbedError::InvalidSector(
                    "UserBasis states must be unique".into(),
                ));
            }
        }
        Ok(UserBasis {
            sites: self.sites,
            states: self.states,
            indices,
            operators: self.operators,
        })
    }

    pub fn materialize(self) -> Result<UserBasis<State>> {
        self.build()
    }
}

impl UserBasisBuilder<u128> {
    pub fn state_filter<F>(mut self, keep: F) -> Result<Self>
    where
        F: Fn(u128) -> bool,
    {
        if self.sites > 127 {
            return Err(QmbedError::UnsupportedBackend(
                "u128 UserBasis filters support at most 127 sites".into(),
            ));
        }
        let limit = 1_u128 << self.sites;
        self.states = (0..limit).filter(|state| keep(*state)).collect();
        Ok(self)
    }

    /// Deterministically enumerate a filtered binary state space in parallel.
    pub fn state_filter_parallel<F>(mut self, keep: F) -> Result<Self>
    where
        F: Fn(u128) -> bool + Sync,
    {
        if self.sites > 127 {
            return Err(QmbedError::UnsupportedBackend(
                "u128 UserBasis filters support at most 127 sites".into(),
            ));
        }
        let limit = 1_u128 << self.sites;
        let workers = std::thread::available_parallelism()
            .map_or(1, std::num::NonZeroUsize::get)
            .min(usize::try_from(limit).unwrap_or(usize::MAX).max(1));
        let stride = limit.div_ceil(workers as u128);
        let mut chunks = std::thread::scope(|scope| {
            let mut handles = Vec::with_capacity(workers);
            let keep = &keep;
            for worker in 0..workers {
                let start = worker as u128 * stride;
                let end = (start + stride).min(limit);
                handles.push(scope.spawn(move || {
                    (start..end)
                        .filter(|state| keep(*state))
                        .collect::<Vec<_>>()
                }));
            }
            handles
                .into_iter()
                .map(|handle| handle.join().unwrap_or_default())
                .collect::<Vec<_>>()
        });
        self.states.clear();
        for chunk in &mut chunks {
            self.states.append(chunk);
        }
        Ok(self)
    }
}

impl<State> UserBasis<State>
where
    State: Copy + Eq + Hash + Send + Sync + 'static,
{
    fn visit_user_transitions<F>(
        &self,
        state: State,
        operator: &str,
        sites: &[usize],
        visit: F,
    ) -> Result<()>
    where
        F: FnMut(State, Complex64) -> Result<()>,
    {
        let symbols = operator_chars(operator, sites)?;
        self.visit_user_transitions_with_symbols(state, &symbols, sites, visit)
    }

    fn visit_user_transitions_with_symbols<F>(
        &self,
        state: State,
        symbols: &[char],
        sites: &[usize],
        mut visit: F,
    ) -> Result<()>
    where
        F: FnMut(State, Complex64) -> Result<()>,
    {
        if symbols.len() != sites.len() {
            return Err(QmbedError::InvalidCoupling(format!(
                "operator arity {} does not match {} sites",
                symbols.len(),
                sites.len()
            )));
        }
        self.visit_user_branch(
            state,
            Complex64::new(1.0, 0.0),
            symbols,
            sites,
            symbols.len(),
            &mut visit,
        )
    }

    fn visit_user_branch<F>(
        &self,
        state: State,
        amplitude: Complex64,
        chars: &[char],
        sites: &[usize],
        remaining: usize,
        visit: &mut F,
    ) -> Result<()>
    where
        F: FnMut(State, Complex64) -> Result<()>,
    {
        if remaining == 0 {
            return visit(state, amplitude);
        }
        let position = remaining - 1;
        let site = sites[position];
        let op = chars[position];
        checked_site(site, self.sites)?;
        let action = self
            .operators
            .get(&op)
            .ok_or_else(|| QmbedError::InvalidOperator(op.to_string()))?;
        action(state, site, &mut |target, local| {
            if local.norm() <= f64::EPSILON {
                return Ok(());
            }
            self.visit_user_branch(target, amplitude * local, chars, sites, position, visit)
        })
    }
}

impl<State> Basis for UserBasis<State>
where
    State: Copy + Eq + Hash + Send + Sync + 'static,
{
    type State = State;

    fn len(&self) -> usize {
        self.states.len()
    }

    fn state(&self, index: usize) -> Result<Self::State> {
        self.states
            .get(index)
            .copied()
            .ok_or(QmbedError::StateNotInBasis)
    }

    fn index(&self, state: Self::State) -> Result<usize> {
        self.indices
            .get(&state)
            .copied()
            .ok_or(QmbedError::StateNotInBasis)
    }

    fn apply_local(
        &self,
        state: Self::State,
        operator: &str,
        sites: &[usize],
    ) -> Result<Option<(Self::State, Complex64)>> {
        let transitions = self.apply_local_transitions(state, operator, sites)?;
        match transitions.as_slice() {
            [] => Ok(None),
            [transition] => Ok(Some(*transition)),
            _ => Err(QmbedError::UnsupportedBackend(
                "this user local action branches; use apply_local_transitions".into(),
            )),
        }
    }

    fn apply_local_transitions(
        &self,
        state: Self::State,
        operator: &str,
        sites: &[usize],
    ) -> Result<LocalTransitions<Self::State>> {
        let mut accumulated = HashMap::<State, Complex64>::new();
        self.visit_user_transitions(state, operator, sites, |target, amplitude| {
            *accumulated
                .entry(target)
                .or_insert(Complex64::new(0.0, 0.0)) += amplitude;
            Ok(())
        })?;
        Ok(accumulated
            .into_iter()
            .filter(|(_, amplitude)| amplitude.norm() > f64::EPSILON)
            .collect())
    }

    fn visit_local_unreduced_transitions<F>(
        &self,
        state: Self::State,
        operator: &str,
        sites: &[usize],
        visit: F,
    ) -> Result<()>
    where
        F: FnMut(Self::State, Complex64) -> Result<()>,
    {
        self.visit_user_transitions(state, operator, sites, visit)
    }

    fn visit_preparsed_local_unreduced_transitions<F>(
        &self,
        state: Self::State,
        _operator: &str,
        symbols: &[char],
        _split: Option<usize>,
        sites: &[usize],
        visit: F,
    ) -> Result<()>
    where
        F: FnMut(Self::State, Complex64) -> Result<()>,
    {
        self.visit_user_transitions_with_symbols(state, symbols, sites, visit)
    }
}

/// Finite symmetry action, including any phase acquired by the state.
pub trait SymmetryMap<State>: Send + Sync {
    fn period(&self) -> usize;
    fn apply(&self, state: State) -> Result<(State, Complex64)>;
}

/// Exchange statistics used when a lattice map reorders local degrees of freedom.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExchangeStatistics {
    Distinguishable,
    Fermionic,
}

/// Runtime-owned finite symmetry of a packed lattice state.
///
/// `destinations[source]` gives the target site. Each source site also owns a
/// permutation of its local digits, allowing the same representation to cover
/// translations, reflections, sublattice maps, and local spin inversion.
/// Fermionic maps compute the parity of the occupied-orbital permutation
/// instead of requiring a frontend-provided phase callback.
#[derive(Clone, Debug)]
pub struct LatticeSymmetryMap {
    states_per_site: usize,
    destinations: Vec<usize>,
    local_permutations: Vec<Vec<usize>>,
    statistics: ExchangeStatistics,
    place_values: Vec<u128>,
    state_limit: Option<u128>,
    period: usize,
}

impl LatticeSymmetryMap {
    pub fn new(
        states_per_site: usize,
        destinations: impl Into<Vec<usize>>,
        local_permutations: Option<Vec<Vec<usize>>>,
        statistics: ExchangeStatistics,
    ) -> Result<Self> {
        let destinations = destinations.into();
        if destinations.is_empty() || states_per_site == 0 {
            return Err(QmbedError::InvalidSector(
                "a lattice symmetry requires sites and local states".into(),
            ));
        }
        let unique_destinations: HashSet<_> = destinations.iter().copied().collect();
        if unique_destinations.len() != destinations.len()
            || destinations
                .iter()
                .any(|&destination| destination >= destinations.len())
        {
            return Err(QmbedError::IncompatibleSymmetry(
                "symmetry site destinations must be a bijection".into(),
            ));
        }

        let identity = (0..states_per_site).collect::<Vec<_>>();
        let local_permutations =
            local_permutations.unwrap_or_else(|| vec![identity.clone(); destinations.len()]);
        if local_permutations.len() != destinations.len()
            || local_permutations.iter().any(|permutation| {
                permutation.len() != states_per_site
                    || permutation.iter().copied().collect::<HashSet<_>>().len() != states_per_site
                    || permutation.iter().any(|&digit| digit >= states_per_site)
            })
        {
            return Err(QmbedError::IncompatibleSymmetry(
                "every local-state map must be a permutation".into(),
            ));
        }
        if statistics == ExchangeStatistics::Fermionic
            && (states_per_site != 2
                || local_permutations
                    .iter()
                    .any(|permutation| permutation != &identity))
        {
            return Err(QmbedError::InvalidOptions(
                "fermionic lattice maps require binary occupation and cannot change it".into(),
            ));
        }

        let base = states_per_site as u128;
        let mut place_values = Vec::with_capacity(destinations.len());
        let mut place = 1_u128;
        for site in 0..destinations.len() {
            place_values.push(place);
            if site + 1 < destinations.len() {
                place = place.checked_mul(base).ok_or_else(|| {
                    QmbedError::UnsupportedBackend(
                        "lattice-symmetry state encoding exceeds u128".into(),
                    )
                })?;
            }
        }
        let state_limit = place.checked_mul(base);
        let exact_full_range = base.is_power_of_two()
            && (base.trailing_zeros() as usize).checked_mul(destinations.len()) == Some(128);
        if state_limit.is_none() && !exact_full_range {
            return Err(QmbedError::UnsupportedBackend(
                "lattice-symmetry state encoding exceeds u128".into(),
            ));
        }
        let period =
            combined_permutation_period(&destinations, &local_permutations, states_per_site)?;
        Ok(Self {
            states_per_site,
            destinations,
            local_permutations,
            statistics,
            place_values,
            state_limit,
            period,
        })
    }

    pub fn site_permutation(
        states_per_site: usize,
        destinations: impl Into<Vec<usize>>,
    ) -> Result<Self> {
        Self::new(
            states_per_site,
            destinations,
            None,
            ExchangeStatistics::Distinguishable,
        )
    }

    pub fn fermionic_orbital_permutation(destinations: impl Into<Vec<usize>>) -> Result<Self> {
        Self::new(2, destinations, None, ExchangeStatistics::Fermionic)
    }

    pub fn sites(&self) -> usize {
        self.destinations.len()
    }

    pub const fn states_per_site(&self) -> usize {
        self.states_per_site
    }

    pub fn destinations(&self) -> &[usize] {
        &self.destinations
    }

    pub fn local_permutations(&self) -> &[Vec<usize>] {
        &self.local_permutations
    }

    pub const fn statistics(&self) -> ExchangeStatistics {
        self.statistics
    }

    fn mapped_state(&self, state: u128) -> Result<(u128, Complex64)> {
        if self.state_limit.is_some_and(|limit| state >= limit) {
            return Err(QmbedError::StateNotInBasis);
        }
        let base = self.states_per_site as u128;
        let mut mapped = 0_u128;
        let mut occupied_destinations = 0_u128;
        let mut odd_fermion_permutation = false;
        for source in 0..self.destinations.len() {
            let digit = usize::try_from((state / self.place_values[source]) % base)
                .map_err(|_| QmbedError::StateNotInBasis)?;
            let mapped_digit = self.local_permutations[source][digit];
            let destination = self.destinations[source];
            mapped += mapped_digit as u128 * self.place_values[destination];
            if self.statistics == ExchangeStatistics::Fermionic && digit == 1 {
                let occupied_after = if destination == 127 {
                    0
                } else {
                    (occupied_destinations >> (destination + 1)).count_ones()
                };
                odd_fermion_permutation ^= occupied_after % 2 == 1;
                occupied_destinations |= 1_u128 << destination;
            }
        }
        Ok((
            mapped,
            Complex64::new(if odd_fermion_permutation { -1.0 } else { 1.0 }, 0.0),
        ))
    }
}

impl SymmetryMap<u128> for LatticeSymmetryMap {
    fn period(&self) -> usize {
        self.period
    }

    fn apply(&self, state: u128) -> Result<(u128, Complex64)> {
        self.mapped_state(state)
    }
}

fn combined_permutation_period(
    destinations: &[usize],
    local_permutations: &[Vec<usize>],
    states_per_site: usize,
) -> Result<usize> {
    let elements = destinations
        .len()
        .checked_mul(states_per_site)
        .ok_or_else(|| {
            QmbedError::UnsupportedBackend("lattice-symmetry permutation is too large".into())
        })?;
    let mut visited = vec![false; elements];
    let mut period = 1_usize;
    for seed in 0..elements {
        if visited[seed] {
            continue;
        }
        let mut current = seed;
        let mut cycle = 0_usize;
        loop {
            if visited[current] {
                if current != seed {
                    return Err(QmbedError::IncompatibleSymmetry(
                        "lattice symmetry does not decompose into closed cycles".into(),
                    ));
                }
                break;
            }
            visited[current] = true;
            cycle += 1;
            let source = current / states_per_site;
            let digit = current % states_per_site;
            current = destinations[source] * states_per_site + local_permutations[source][digit];
        }
        period = checked_lcm(period, cycle)?;
    }
    Ok(period)
}

fn checked_lcm(left: usize, right: usize) -> Result<usize> {
    fn gcd(mut left: usize, mut right: usize) -> usize {
        while right != 0 {
            (left, right) = (right, left % right);
        }
        left
    }

    left.checked_div(gcd(left, right))
        .and_then(|reduced| reduced.checked_mul(right))
        .ok_or_else(|| QmbedError::UnsupportedBackend("symmetry period is too large".into()))
}

type SymmetryAction<State> = Arc<dyn Fn(State) -> Result<(State, Complex64)> + Send + Sync>;

/// Closure-backed finite map for lattice, particle-hole, or user symmetries.
pub struct ClosureSymmetryMap<State> {
    period: usize,
    action: SymmetryAction<State>,
}

impl<State> ClosureSymmetryMap<State> {
    pub fn new<F>(period: usize, action: F) -> Result<Self>
    where
        F: Fn(State) -> Result<(State, Complex64)> + Send + Sync + 'static,
    {
        if period == 0 {
            return Err(QmbedError::InvalidSector(
                "a symmetry-map period must be positive".into(),
            ));
        }
        Ok(Self {
            period,
            action: Arc::new(action),
        })
    }
}

impl<State> SymmetryMap<State> for ClosureSymmetryMap<State>
where
    State: Copy,
{
    fn period(&self) -> usize {
        self.period
    }

    fn apply(&self, state: State) -> Result<(State, Complex64)> {
        (self.action)(state)
    }
}

struct SymmetryGenerator<State> {
    map: Arc<dyn SymmetryMap<State>>,
    sector: i32,
}

/// Finite generators and sector phases that extend to a one-dimensional
/// character of the generated group.
pub struct SymmetrySector<State> {
    generators: Vec<SymmetryGenerator<State>>,
}

impl<State> SymmetrySector<State> {
    pub fn new() -> Self {
        Self {
            generators: Vec::new(),
        }
    }

    pub fn with_map<M>(mut self, map: M, sector: i32) -> Self
    where
        M: SymmetryMap<State> + 'static,
    {
        self.generators.push(SymmetryGenerator {
            map: Arc::new(map),
            sector,
        });
        self
    }

    pub fn generators(&self) -> usize {
        self.generators.len()
    }
}

impl<State> Default for SymmetrySector<State> {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone, Copy, Debug)]
struct GeneralSymmetryImage<State> {
    representative: State,
    phase: Complex64,
    orbit_size: usize,
}

fn trace_symmetry_orbit<State>(
    state: State,
    generators: &[SymmetryGenerator<State>],
) -> Result<(HashMap<State, Complex64>, bool)>
where
    State: Copy + Eq + Hash,
{
    let mut sector_steps = Vec::with_capacity(generators.len());
    for generator in generators {
        let period = generator.map.period();
        if period == 0 {
            return Err(QmbedError::InvalidSector(
                "a symmetry-map period must be positive".into(),
            ));
        }
        let normalized_sector = i64::from(generator.sector).rem_euclid(period as i64) as usize;
        let angle = -std::f64::consts::TAU * normalized_sector as f64 / period as f64;
        sector_steps.push(Complex64::from_polar(1.0, angle));

        let mut closure_state = state;
        let mut closure_phase = Complex64::new(1.0, 0.0);
        for _ in 0..period {
            let (next, phase) = generator.map.apply(closure_state)?;
            if !phase.re.is_finite() || !phase.im.is_finite() {
                return Err(QmbedError::IncompatibleSymmetry(
                    "a symmetry map returned a non-finite phase".into(),
                ));
            }
            closure_state = next;
            closure_phase *= phase;
        }
        if closure_state != state || (closure_phase - Complex64::new(1.0, 0.0)).norm() > 1.0e-10 {
            return Err(QmbedError::IncompatibleSymmetry(
                "a symmetry map does not close at its declared period".into(),
            ));
        }
    }

    let mut coefficients = HashMap::new();
    coefficients.insert(state, Complex64::new(1.0, 0.0));
    let mut queue = VecDeque::from([state]);
    let mut compatible = true;
    while let Some(source) = queue.pop_front() {
        let source_coefficient = coefficients[&source];
        for (generator, &sector_step) in generators.iter().zip(&sector_steps) {
            let (target, map_phase) = generator.map.apply(source)?;
            if !map_phase.re.is_finite() || !map_phase.im.is_finite() {
                return Err(QmbedError::IncompatibleSymmetry(
                    "a symmetry map returned a non-finite phase".into(),
                ));
            }
            let target_coefficient = source_coefficient * map_phase * sector_step;
            if let Some(previous) = coefficients.get(&target) {
                if (*previous - target_coefficient).norm() > 1.0e-10 {
                    compatible = false;
                }
            } else {
                coefficients.insert(target, target_coefficient);
                queue.push_back(target);
            }
        }
    }
    Ok((coefficients, compatible))
}

/// Arbitrary finite-map reduction of any concrete parent basis.
#[derive(Clone, Debug)]
pub struct GeneralBasis<Parent>
where
    Parent: Basis,
    Parent::State: Hash + Ord,
{
    parent: Parent,
    states: Vec<Parent::State>,
    orbit_lengths: Vec<usize>,
    lookup: HashMap<Parent::State, GeneralSymmetryImage<Parent::State>>,
}

impl<Parent> GeneralBasis<Parent>
where
    Parent: Basis,
    Parent::State: Hash + Ord,
{
    pub fn new(parent: Parent, sector: SymmetrySector<Parent::State>) -> Result<Self> {
        if sector.generators.is_empty() {
            let mut states = Vec::with_capacity(parent.len());
            let mut lookup = HashMap::with_capacity(parent.len());
            for index in 0..parent.len() {
                let state = parent.state(index)?;
                states.push(state);
                lookup.insert(
                    state,
                    GeneralSymmetryImage {
                        representative: state,
                        phase: Complex64::new(1.0, 0.0),
                        orbit_size: 1,
                    },
                );
            }
            states.sort_unstable();
            return Ok(Self {
                parent,
                orbit_lengths: vec![1; states.len()],
                states,
                lookup,
            });
        }

        let mut visited = HashSet::with_capacity(parent.len());
        let mut representatives = Vec::new();
        let mut lookup = HashMap::with_capacity(parent.len());
        for index in 0..parent.len() {
            let seed = parent.state(index)?;
            if visited.contains(&seed) {
                continue;
            }
            let (coefficients, compatible) = trace_symmetry_orbit(seed, &sector.generators)?;
            for state in coefficients.keys() {
                parent.index(*state).map_err(|_| {
                    QmbedError::IncompatibleSymmetry(
                        "a symmetry map leaves the parent basis".into(),
                    )
                })?;
                visited.insert(*state);
            }
            if !compatible {
                continue;
            }
            let representative = *coefficients.keys().min().ok_or_else(|| {
                QmbedError::InvalidSector("symmetry projection generated no state".into())
            })?;
            let representative_coefficient = coefficients[&representative];
            let gauge = representative_coefficient / representative_coefficient.norm();
            let norm = coefficients
                .values()
                .map(Complex64::norm_sqr)
                .sum::<f64>()
                .sqrt();
            let orbit_size = coefficients.len();
            let expected_magnitude = 1.0 / (orbit_size as f64).sqrt();
            for (&state, &coefficient) in &coefficients {
                let normalized = coefficient / (gauge * norm);
                if (normalized.norm() - expected_magnitude).abs() > 1.0e-10 {
                    return Err(QmbedError::IncompatibleSymmetry(
                        "symmetry maps do not define a one-dimensional orbit sector".into(),
                    ));
                }
                lookup.insert(
                    state,
                    GeneralSymmetryImage {
                        representative,
                        phase: normalized / expected_magnitude,
                        orbit_size,
                    },
                );
            }
            representatives.push((representative, orbit_size));
        }
        representatives.sort_by_key(|(state, _)| *state);
        let (states, orbit_lengths) = representatives.into_iter().unzip();
        Ok(Self {
            parent,
            states,
            orbit_lengths,
            lookup,
        })
    }

    pub fn parent(&self) -> &Parent {
        &self.parent
    }

    pub fn representative(&self, state: Parent::State) -> Result<Parent::State> {
        self.lookup
            .get(&state)
            .map(|image| image.representative)
            .ok_or(QmbedError::StateNotInBasis)
    }

    pub fn orbit_size(&self, state: Parent::State) -> Result<usize> {
        self.lookup
            .get(&state)
            .map(|image| image.orbit_size)
            .ok_or(QmbedError::StateNotInBasis)
    }

    /// Normalized coefficient of a parent state in its reduced representative.
    pub fn symmetry_amplitude(&self, state: Parent::State) -> Result<Complex64> {
        self.lookup
            .get(&state)
            .map(|image| image.phase / (image.orbit_size as f64).sqrt())
            .ok_or(QmbedError::StateNotInBasis)
    }
}

impl<Parent> Basis for GeneralBasis<Parent>
where
    Parent: Basis,
    Parent::State: Hash + Ord + 'static,
{
    type State = Parent::State;

    fn len(&self) -> usize {
        self.states.len()
    }

    fn state(&self, index: usize) -> Result<Self::State> {
        self.states
            .get(index)
            .copied()
            .ok_or(QmbedError::StateNotInBasis)
    }

    fn index(&self, state: Self::State) -> Result<usize> {
        self.states
            .binary_search(&state)
            .map_err(|_| QmbedError::StateNotInBasis)
    }

    fn apply_local(
        &self,
        state: Self::State,
        operator: &str,
        sites: &[usize],
    ) -> Result<Option<(Self::State, Complex64)>> {
        let transitions = self.apply_local_transitions(state, operator, sites)?;
        match transitions.as_slice() {
            [] => Ok(None),
            [transition] => Ok(Some(*transition)),
            _ => Err(QmbedError::UnsupportedBackend(
                "this reduced local action branches; use apply_local_transitions".into(),
            )),
        }
    }

    fn apply_local_transitions(
        &self,
        state: Self::State,
        operator: &str,
        sites: &[usize],
    ) -> Result<LocalTransitions<Self::State>> {
        let source_index = self.index(state)?;
        let source_orbit = self.orbit_lengths[source_index];
        let mut reduced = HashMap::<Self::State, Complex64>::new();
        for (target, mut amplitude) in self
            .parent
            .apply_local_transitions(state, operator, sites)?
        {
            let Some(image) = self.lookup.get(&target) else {
                continue;
            };
            amplitude *=
                (source_orbit as f64 / image.orbit_size as f64).sqrt() * image.phase.conj();
            *reduced
                .entry(image.representative)
                .or_insert(Complex64::new(0.0, 0.0)) += amplitude;
        }
        let mut transitions: LocalTransitions<_> = reduced
            .into_iter()
            .filter(|(_, amplitude)| amplitude.norm() > f64::EPSILON)
            .collect();
        transitions.sort_by_key(|(state, _)| *state);
        Ok(transitions)
    }

    fn apply_local_unreduced_transitions(
        &self,
        state: Self::State,
        operator: &str,
        sites: &[usize],
    ) -> Result<LocalTransitions<Self::State>> {
        self.parent
            .apply_local_unreduced_transitions(state, operator, sites)
    }

    fn visit_local_unreduced_transitions<F>(
        &self,
        state: Self::State,
        operator: &str,
        sites: &[usize],
        visit: F,
    ) -> Result<()>
    where
        F: FnMut(Self::State, Complex64) -> Result<()>,
    {
        self.parent
            .visit_local_unreduced_transitions(state, operator, sites, visit)
    }

    fn visit_preparsed_local_unreduced_transitions<F>(
        &self,
        state: Self::State,
        operator: &str,
        symbols: &[char],
        split: Option<usize>,
        sites: &[usize],
        visit: F,
    ) -> Result<()>
    where
        F: FnMut(Self::State, Complex64) -> Result<()>,
    {
        self.parent.visit_preparsed_local_unreduced_transitions(
            state, operator, symbols, split, sites, visit,
        )
    }

    fn transition_orbit_size(&self, state: Self::State) -> Result<usize> {
        Ok(self.orbit_lengths[self.index(state)?])
    }

    fn reduction_image(&self, state: Self::State) -> Result<Option<ReductionImage<Self::State>>> {
        let Some(image) = self.lookup.get(&state) else {
            return Ok(None);
        };
        Ok(Some(ReductionImage::new(
            image.representative,
            image.phase,
            image.orbit_size,
        )?))
    }

    fn reduce_transition(
        &self,
        state: Self::State,
        source_orbit_size: usize,
    ) -> Result<Option<(Self::State, Complex64)>> {
        let Some(image) = self.lookup.get(&state) else {
            return Ok(None);
        };
        Ok(Some((
            image.representative,
            (source_orbit_size as f64 / image.orbit_size as f64).sqrt() * image.phase.conj(),
        )))
    }

    fn index_transition(
        &self,
        state: Self::State,
        source_orbit_size: usize,
    ) -> Result<Option<(usize, Complex64)>> {
        let Some(image) = self.lookup.get(&state) else {
            return Ok(None);
        };
        Ok(Some((
            self.index(image.representative)?,
            (source_orbit_size as f64 / image.orbit_size as f64).sqrt() * image.phase.conj(),
        )))
    }

    fn operator_preserves_particle_sector(&self, operator: &str) -> Result<bool> {
        self.parent.operator_preserves_particle_sector(operator)
    }
}

pub type SpinBasisGeneral = GeneralBasis<SpinBasis1D>;
pub type BosonBasisGeneral = GeneralBasis<BosonBasis1D>;
pub type SpinlessFermionBasisGeneral = GeneralBasis<SpinlessFermionBasis1D>;
pub type SpinfulFermionBasisGeneral = GeneralBasis<SpinfulFermionBasis1D>;

/// Direct-product basis. Operator strings use `left|right` factor syntax.
#[derive(Clone, Debug)]
pub struct TensorBasis<Left, Right> {
    left: Left,
    right: Right,
}

impl<Left, Right> TensorBasis<Left, Right>
where
    Left: Basis,
    Right: Basis,
{
    pub fn new(left: Left, right: Right) -> Result<Self> {
        left.len()
            .checked_mul(right.len())
            .ok_or_else(|| QmbedError::UnsupportedBackend("tensor-basis size overflow".into()))?;
        Ok(Self { left, right })
    }

    pub fn left(&self) -> &Left {
        &self.left
    }

    pub fn right(&self) -> &Right {
        &self.right
    }
}

impl<Left, Right> Basis for TensorBasis<Left, Right>
where
    Left: Basis,
    Right: Basis,
    Left::State: 'static,
    Right::State: 'static,
{
    type State = (Left::State, Right::State);

    fn len(&self) -> usize {
        self.left.len() * self.right.len()
    }

    fn state(&self, index: usize) -> Result<Self::State> {
        if index >= self.len() {
            return Err(QmbedError::StateNotInBasis);
        }
        Ok((
            self.left.state(index / self.right.len())?,
            self.right.state(index % self.right.len())?,
        ))
    }

    fn index(&self, state: Self::State) -> Result<usize> {
        Ok(self.left.index(state.0)? * self.right.len() + self.right.index(state.1)?)
    }

    fn apply_local(
        &self,
        state: Self::State,
        operator: &str,
        sites: &[usize],
    ) -> Result<Option<(Self::State, Complex64)>> {
        let transitions = self.apply_local_transitions(state, operator, sites)?;
        match transitions.as_slice() {
            [] => Ok(None),
            [transition] => Ok(Some(*transition)),
            _ => Err(QmbedError::UnsupportedBackend(
                "this tensor local action branches; use apply_local_transitions".into(),
            )),
        }
    }

    fn apply_local_transitions(
        &self,
        state: Self::State,
        operator: &str,
        sites: &[usize],
    ) -> Result<LocalTransitions<Self::State>> {
        let (left_operator, right_operator) = operator.split_once('|').ok_or_else(|| {
            QmbedError::InvalidOperator(
                "tensor-basis operator strings must contain one `|` separator".into(),
            )
        })?;
        if right_operator.contains('|') {
            return Err(QmbedError::InvalidOperator(
                "a two-factor tensor operator contains too many separators".into(),
            ));
        }
        let left_arity = left_operator.chars().count();
        let right_arity = right_operator.chars().count();
        if sites.len() != left_arity + right_arity {
            return Err(QmbedError::InvalidCoupling(
                "tensor operator arity does not match its sites".into(),
            ));
        }
        let left_transitions = if left_operator.is_empty() {
            LocalTransitions::from_iter([(state.0, Complex64::new(1.0, 0.0))])
        } else {
            self.left
                .apply_local_transitions(state.0, left_operator, &sites[..left_arity])?
        };
        let right_transitions = if right_operator.is_empty() {
            LocalTransitions::from_iter([(state.1, Complex64::new(1.0, 0.0))])
        } else {
            self.right
                .apply_local_transitions(state.1, right_operator, &sites[left_arity..])?
        };
        let mut transitions = LocalTransitions::new();
        for &(left_state, left_amplitude) in &left_transitions {
            for &(right_state, right_amplitude) in &right_transitions {
                transitions.push(((left_state, right_state), left_amplitude * right_amplitude));
            }
        }
        Ok(transitions)
    }

    fn visit_local_unreduced_transitions<F>(
        &self,
        state: Self::State,
        operator: &str,
        sites: &[usize],
        mut visit: F,
    ) -> Result<()>
    where
        F: FnMut(Self::State, Complex64) -> Result<()>,
    {
        for (target, amplitude) in self.apply_local_transitions(state, operator, sites)? {
            visit(target, amplitude)?;
        }
        Ok(())
    }

    fn operator_preserves_particle_sector(&self, operator: &str) -> Result<bool> {
        let (left_operator, right_operator) = operator
            .split_once('|')
            .ok_or_else(|| QmbedError::InvalidOperator(operator.into()))?;
        if right_operator.contains('|') {
            return Err(QmbedError::InvalidOperator(operator.into()));
        }
        Ok(self
            .left
            .operator_preserves_particle_sector(left_operator)?
            && self
                .right
                .operator_preserves_particle_sector(right_operator)?)
    }
}

/// Matter basis tensored with one truncated photon mode, optionally at fixed
/// total excitation number.
pub struct PhotonBasis<Matter>
where
    Matter: Basis,
    Matter::State: Hash,
{
    tensor: TensorBasis<Matter, BosonBasis1D>,
    states: Vec<(Matter::State, u128)>,
    indices: HashMap<(Matter::State, u128), usize>,
    total_excitations: Option<usize>,
}

impl<Matter> PhotonBasis<Matter>
where
    Matter: Basis,
    Matter::State: Hash + 'static,
{
    pub fn new(matter: Matter, photon: BosonBasis1D) -> Result<Self> {
        Self::build(matter, photon, None, |_| 0)
    }

    pub fn fixed_total_excitations<F>(
        matter: Matter,
        photon: BosonBasis1D,
        total: usize,
        matter_excitations: F,
    ) -> Result<Self>
    where
        F: Fn(Matter::State) -> usize,
    {
        Self::build(matter, photon, Some(total), matter_excitations)
    }

    fn build<F>(
        matter: Matter,
        photon: BosonBasis1D,
        total_excitations: Option<usize>,
        matter_excitations: F,
    ) -> Result<Self>
    where
        F: Fn(Matter::State) -> usize,
    {
        if photon.sites() != 1 {
            return Err(QmbedError::InvalidSector(
                "PhotonBasis requires a one-mode boson basis".into(),
            ));
        }
        let tensor = TensorBasis::new(matter, photon)?;
        let mut states = Vec::new();
        for index in 0..tensor.len() {
            let state = tensor.state(index)?;
            if total_excitations
                .is_none_or(|total| matter_excitations(state.0) + state.1 as usize == total)
            {
                states.push(state);
            }
        }
        if states.is_empty() {
            return Err(QmbedError::InvalidSector(
                "the requested photon sector is empty".into(),
            ));
        }
        let indices = states
            .iter()
            .copied()
            .enumerate()
            .map(|(index, state)| (state, index))
            .collect();
        Ok(Self {
            tensor,
            states,
            indices,
            total_excitations,
        })
    }

    pub const fn total_excitations(&self) -> Option<usize> {
        self.total_excitations
    }

    pub fn matter(&self) -> &Matter {
        self.tensor.left()
    }

    pub fn photon(&self) -> &BosonBasis1D {
        self.tensor.right()
    }
}

impl<Matter> Basis for PhotonBasis<Matter>
where
    Matter: Basis,
    Matter::State: Hash + 'static,
{
    type State = (Matter::State, u128);

    fn len(&self) -> usize {
        self.states.len()
    }

    fn state(&self, index: usize) -> Result<Self::State> {
        self.states
            .get(index)
            .copied()
            .ok_or(QmbedError::StateNotInBasis)
    }

    fn index(&self, state: Self::State) -> Result<usize> {
        self.indices
            .get(&state)
            .copied()
            .ok_or(QmbedError::StateNotInBasis)
    }

    fn apply_local(
        &self,
        state: Self::State,
        operator: &str,
        sites: &[usize],
    ) -> Result<Option<(Self::State, Complex64)>> {
        let transitions = self.apply_local_transitions(state, operator, sites)?;
        match transitions.as_slice() {
            [] => Ok(None),
            [transition] => Ok(Some(*transition)),
            _ => Err(QmbedError::UnsupportedBackend(
                "this photon-basis action branches; use apply_local_transitions".into(),
            )),
        }
    }

    fn apply_local_transitions(
        &self,
        state: Self::State,
        operator: &str,
        sites: &[usize],
    ) -> Result<LocalTransitions<Self::State>> {
        Ok(self
            .tensor
            .apply_local_transitions(state, operator, sites)?
            .into_iter()
            .filter(|(target, _)| self.indices.contains_key(target))
            .collect())
    }

    fn visit_local_unreduced_transitions<F>(
        &self,
        state: Self::State,
        operator: &str,
        sites: &[usize],
        mut visit: F,
    ) -> Result<()>
    where
        F: FnMut(Self::State, Complex64) -> Result<()>,
    {
        for (target, amplitude) in self.apply_local_transitions(state, operator, sites)? {
            visit(target, amplitude)?;
        }
        Ok(())
    }

    fn operator_preserves_particle_sector(&self, operator: &str) -> Result<bool> {
        Ok(self.total_excitations.is_none() || operator_number_change(operator)? == Some(0))
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StateStorage {
    U128,
    U256,
    U1024,
    U4096,
    U16384,
}

/// Fixed-width state used by wide user and general bases.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct WideState<const WORDS: usize> {
    words: [u64; WORDS],
}

impl<const WORDS: usize> WideState<WORDS> {
    pub const fn zero() -> Self {
        Self { words: [0; WORDS] }
    }

    pub const fn capacity_bits() -> usize {
        WORDS * 64
    }

    pub fn from_words(words: [u64; WORDS]) -> Self {
        Self { words }
    }

    pub fn words(&self) -> &[u64; WORDS] {
        &self.words
    }

    pub fn bit(&self, index: usize) -> Result<bool> {
        if index >= Self::capacity_bits() {
            return Err(QmbedError::InvalidSite {
                site: index,
                sites: Self::capacity_bits(),
            });
        }
        Ok(self.words[index / 64] & (1_u64 << (index % 64)) != 0)
    }

    pub fn with_bit(mut self, index: usize, occupied: bool) -> Result<Self> {
        if index >= Self::capacity_bits() {
            return Err(QmbedError::InvalidSite {
                site: index,
                sites: Self::capacity_bits(),
            });
        }
        let mask = 1_u64 << (index % 64);
        if occupied {
            self.words[index / 64] |= mask;
        } else {
            self.words[index / 64] &= !mask;
        }
        Ok(self)
    }

    pub fn count_ones(&self) -> usize {
        self.words
            .iter()
            .map(|word| word.count_ones() as usize)
            .sum()
    }

    pub fn bitwise_and(self, right: Self) -> Self {
        Self::from_words(std::array::from_fn(|index| {
            self.words[index] & right.words[index]
        }))
    }

    pub fn bitwise_or(self, right: Self) -> Self {
        Self::from_words(std::array::from_fn(|index| {
            self.words[index] | right.words[index]
        }))
    }

    pub fn bitwise_xor(self, right: Self) -> Self {
        Self::from_words(std::array::from_fn(|index| {
            self.words[index] ^ right.words[index]
        }))
    }

    pub fn bitwise_not(self) -> Self {
        Self::from_words(std::array::from_fn(|index| !self.words[index]))
    }

    pub fn left_shift(self, shift: usize) -> Self {
        if shift >= Self::capacity_bits() {
            return Self::zero();
        }
        let word_shift = shift / 64;
        let bit_shift = shift % 64;
        let mut words = [0_u64; WORDS];
        for target in (word_shift..WORDS).rev() {
            let source = target - word_shift;
            words[target] |= self.words[source] << bit_shift;
            if bit_shift > 0 && source > 0 {
                words[target] |= self.words[source - 1] >> (64 - bit_shift);
            }
        }
        Self::from_words(words)
    }

    pub fn right_shift(self, shift: usize) -> Self {
        if shift >= Self::capacity_bits() {
            return Self::zero();
        }
        let word_shift = shift / 64;
        let bit_shift = shift % 64;
        let mut words = [0_u64; WORDS];
        for (target, word) in words.iter_mut().enumerate().take(WORDS - word_shift) {
            let source = target + word_shift;
            *word |= self.words[source] >> bit_shift;
            if bit_shift > 0 && source + 1 < WORDS {
                *word |= self.words[source + 1] << (64 - bit_shift);
            }
        }
        Self::from_words(words)
    }
}

pub type U256 = WideState<4>;
pub type U1024 = WideState<16>;
pub type U4096 = WideState<64>;
pub type U16384 = WideState<256>;
pub type UInt256 = U256;
pub type UInt1024 = U1024;
pub type UInt4096 = U4096;
pub type UInt16384 = U16384;

/// Spin-half basis backed by a fixed-width state, including sites above 127.
#[derive(Clone, Debug)]
pub struct WideSpinBasis<const WORDS: usize> {
    sites: usize,
    particles: Option<usize>,
    pauli: bool,
    states: Vec<WideState<WORDS>>,
}

impl<const WORDS: usize> WideSpinBasis<WORDS> {
    pub fn new(sites: usize, particles: Option<usize>, pauli: bool) -> Result<Self> {
        if sites == 0 || sites > WideState::<WORDS>::capacity_bits() {
            return Err(QmbedError::UnsupportedBackend(format!(
                "wide spin basis needs 1..={} sites",
                WideState::<WORDS>::capacity_bits()
            )));
        }
        if particles.is_some_and(|count| count > sites) {
            return Err(QmbedError::InvalidSector(
                "particle count exceeds the wide spin site count".into(),
            ));
        }
        let mut states = Vec::new();
        if let Some(count) = particles {
            fn enumerate<const WORDS: usize>(
                next_site: usize,
                sites: usize,
                remaining: usize,
                state: WideState<WORDS>,
                output: &mut Vec<WideState<WORDS>>,
            ) -> Result<()> {
                if remaining == 0 {
                    output.push(state);
                    return Ok(());
                }
                if sites.saturating_sub(next_site) < remaining {
                    return Ok(());
                }
                for site in next_site..=sites - remaining {
                    enumerate(
                        site + 1,
                        sites,
                        remaining - 1,
                        state.with_bit(site, true)?,
                        output,
                    )?;
                }
                Ok(())
            }
            enumerate(0, sites, count, WideState::zero(), &mut states)?;
        } else {
            if sites > 24 {
                return Err(QmbedError::InvalidOptions(
                    "an unrestricted wide spin basis above 24 sites is not enumerable; select a particle sector"
                        .into(),
                ));
            }
            let limit = 1_u128 << sites;
            states.extend((0..limit).map(python_int_to_basis_int));
        }
        states.sort_unstable();
        Ok(Self {
            sites,
            particles,
            pauli,
            states,
        })
    }

    pub const fn sites(&self) -> usize {
        self.sites
    }

    pub const fn particles(&self) -> Option<usize> {
        self.particles
    }
}

impl<const WORDS: usize> Basis for WideSpinBasis<WORDS> {
    type State = WideState<WORDS>;

    fn len(&self) -> usize {
        self.states.len()
    }

    fn state(&self, index: usize) -> Result<Self::State> {
        self.states
            .get(index)
            .copied()
            .ok_or(QmbedError::StateNotInBasis)
    }

    fn index(&self, state: Self::State) -> Result<usize> {
        self.states
            .binary_search(&state)
            .map_err(|_| QmbedError::StateNotInBasis)
    }

    fn apply_local(
        &self,
        mut state: Self::State,
        operator: &str,
        sites: &[usize],
    ) -> Result<Option<(Self::State, Complex64)>> {
        let chars = operator_chars(operator, sites)?;
        let mut amplitude = Complex64::new(1.0, 0.0);
        for (&site, op) in sites.iter().zip(chars).rev() {
            checked_site(site, self.sites)?;
            let occupied = state.bit(site)?;
            let scale = if self.pauli { 1.0 } else { 0.5 };
            match op {
                'I' => {}
                'n' => {
                    if !occupied {
                        return Ok(None);
                    }
                }
                'z' => amplitude *= if occupied { scale } else { -scale },
                '+' if !occupied => state = state.with_bit(site, true)?,
                '-' if occupied => state = state.with_bit(site, false)?,
                'x' => {
                    state = state.with_bit(site, !occupied)?;
                    amplitude *= scale;
                }
                'y' => {
                    state = state.with_bit(site, !occupied)?;
                    amplitude *= Complex64::new(0.0, if occupied { scale } else { -scale });
                }
                '+' | '-' => return Ok(None),
                _ => return Err(QmbedError::InvalidOperator(op.to_string())),
            }
        }
        Ok(Some((state, amplitude)))
    }

    fn operator_preserves_particle_sector(&self, operator: &str) -> Result<bool> {
        Ok(self.particles.is_none() || operator_number_change(operator)? == Some(0))
    }
}

pub type WideSpinBasis256 = WideSpinBasis<4>;
pub type WideSpinBasis1024 = WideSpinBasis<16>;
pub type WideSpinBasis4096 = WideSpinBasis<64>;
pub type WideSpinBasis16384 = WideSpinBasis<256>;

pub fn basis_zeros<const WORDS: usize>(length: usize) -> Vec<WideState<WORDS>> {
    vec![WideState::zero(); length]
}

pub fn basis_ones<const WORDS: usize>(length: usize) -> Vec<WideState<WORDS>> {
    vec![WideState::zero().bitwise_not(); length]
}

pub fn bitwise_and<const WORDS: usize>(
    left: WideState<WORDS>,
    right: WideState<WORDS>,
) -> WideState<WORDS> {
    left.bitwise_and(right)
}

pub fn bitwise_or<const WORDS: usize>(
    left: WideState<WORDS>,
    right: WideState<WORDS>,
) -> WideState<WORDS> {
    left.bitwise_or(right)
}

pub fn bitwise_xor<const WORDS: usize>(
    left: WideState<WORDS>,
    right: WideState<WORDS>,
) -> WideState<WORDS> {
    left.bitwise_xor(right)
}

pub fn bitwise_not<const WORDS: usize>(value: WideState<WORDS>) -> WideState<WORDS> {
    value.bitwise_not()
}

pub fn bitwise_leftshift<const WORDS: usize>(
    value: WideState<WORDS>,
    shift: usize,
) -> WideState<WORDS> {
    value.left_shift(shift)
}

pub fn bitwise_rightshift<const WORDS: usize>(
    value: WideState<WORDS>,
    shift: usize,
) -> WideState<WORDS> {
    value.right_shift(shift)
}

pub fn python_int_to_basis_int<const WORDS: usize>(value: u128) -> WideState<WORDS> {
    let mut words = [0_u64; WORDS];
    if WORDS > 0 {
        words[0] = value as u64;
    }
    if WORDS > 1 {
        words[1] = (value >> 64) as u64;
    }
    WideState::from_words(words)
}

pub fn basis_int_to_python_int<const WORDS: usize>(value: WideState<WORDS>) -> Result<u128> {
    if value.words.iter().skip(2).any(|word| *word != 0) {
        return Err(QmbedError::UnsupportedBackend(
            "wide basis integer does not fit into a Python-compatible u128".into(),
        ));
    }
    Ok(u128::from(value.words.first().copied().unwrap_or_default())
        | (u128::from(value.words.get(1).copied().unwrap_or_default()) << 64))
}

/// Convert an arbitrary-precision nonnegative integer into a fixed-width basis
/// state without truncating high words.
pub fn state_from_biguint<const WORDS: usize>(value: &BigUint) -> Result<WideState<WORDS>> {
    let digits = value.to_u64_digits();
    if digits.len() > WORDS {
        return Err(QmbedError::UnsupportedBackend(format!(
            "integer needs {} bits but this state stores {} bits",
            value.bits(),
            WideState::<WORDS>::capacity_bits()
        )));
    }
    let mut words = [0_u64; WORDS];
    words[..digits.len()].copy_from_slice(&digits);
    Ok(WideState::from_words(words))
}

/// Convert a fixed-width state to an arbitrary-precision integer.
pub fn state_to_biguint<const WORDS: usize>(value: WideState<WORDS>) -> BigUint {
    let bytes: Vec<_> = value
        .words()
        .iter()
        .flat_map(|word| word.to_le_bytes())
        .collect();
    BigUint::from_bytes_le(&bytes)
}

pub fn get_basis_type(
    sites: usize,
    _particles: Option<usize>,
    states_per_site: usize,
) -> Result<StateStorage> {
    if states_per_site < 2 {
        return Err(QmbedError::InvalidSector(
            "states_per_site must be at least two".into(),
        ));
    }
    let bits_per_site =
        usize::try_from(usize::BITS - (states_per_site - 1).leading_zeros()).unwrap_or(usize::MAX);
    let bits = sites
        .checked_mul(bits_per_site)
        .ok_or_else(|| QmbedError::UnsupportedBackend("basis bit width overflow".into()))?;
    match bits {
        0..=128 => Ok(StateStorage::U128),
        129..=256 => Ok(StateStorage::U256),
        257..=1024 => Ok(StateStorage::U1024),
        1025..=4096 => Ok(StateStorage::U4096),
        4097..=16384 => Ok(StateStorage::U16384),
        _ => Err(QmbedError::UnsupportedBackend(
            "basis requires more than 16384 state bits".into(),
        )),
    }
}

pub fn coherent_state(amplitude: Complex64, states: usize) -> Result<Vec<Complex64>> {
    if states == 0 || !amplitude.re.is_finite() || !amplitude.im.is_finite() {
        return Err(QmbedError::InvalidOptions(
            "coherent-state amplitude must be finite and the cutoff positive".into(),
        ));
    }
    let mut coefficients = Vec::with_capacity(states);
    let mut coefficient = Complex64::new((-0.5 * amplitude.norm_sqr()).exp(), 0.0);
    coefficients.push(coefficient);
    for occupation in 1..states {
        coefficient *= amplitude / (occupation as f64).sqrt();
        coefficients.push(coefficient);
    }
    Ok(coefficients)
}

fn binomial(n: usize, k: usize) -> usize {
    let k = k.min(n.saturating_sub(k));
    (0..k).fold(1_usize, |value, index| {
        value.saturating_mul(n - index) / (index + 1)
    })
}

/// Dimension of a spin-half chain plus one photon mode at fixed excitation.
pub fn photon_hspace_dim(
    sites: usize,
    total_excitations: Option<usize>,
    photon_cutoff: Option<usize>,
) -> Result<usize> {
    match (total_excitations, photon_cutoff) {
        (None, Some(cutoff)) => 1_usize
            .checked_shl(u32::try_from(sites).unwrap_or(u32::MAX))
            .and_then(|matter| matter.checked_mul(cutoff.saturating_add(1)))
            .ok_or_else(|| QmbedError::UnsupportedBackend("photon dimension overflow".into())),
        (Some(total), cutoff) => {
            let minimum_matter =
                cutoff.map_or(0, |maximum_photons| total.saturating_sub(maximum_photons));
            let maximum_matter = sites.min(total);
            Ok((minimum_matter..=maximum_matter)
                .map(|matter| binomial(sites, matter))
                .sum())
        }
        (None, None) => Err(QmbedError::InvalidSector(
            "either total excitation or photon cutoff must be finite".into(),
        )),
    }
}

/// Sparse isometric lift from a symmetry-reduced basis to its parent basis.
#[derive(Clone, Debug)]
pub struct BasisProjector {
    source_dimension: usize,
    reduced_dimension: usize,
    column_offsets: Vec<usize>,
    row_indices: Vec<usize>,
    values: Vec<Complex64>,
}

impl BasisProjector {
    fn from_columns(
        source_dimension: usize,
        mut columns: Vec<Vec<(usize, Complex64)>>,
    ) -> Result<Self> {
        if columns.iter().flatten().any(|(row, value)| {
            *row >= source_dimension || !value.re.is_finite() || !value.im.is_finite()
        }) {
            return Err(QmbedError::DimensionMismatch(
                "projector contains an invalid parent-space row or coefficient".into(),
            ));
        }
        let reduced_dimension = columns.len();
        let mut column_offsets = Vec::with_capacity(reduced_dimension + 1);
        let mut row_indices = Vec::new();
        let mut values = Vec::new();
        column_offsets.push(0);
        for column in &mut columns {
            column.sort_by_key(|(row, _)| *row);
            for &(row, value) in column.iter() {
                row_indices.push(row);
                values.push(value);
            }
            column_offsets.push(row_indices.len());
        }
        Ok(Self {
            source_dimension,
            reduced_dimension,
            column_offsets,
            row_indices,
            values,
        })
    }

    /// One-hot embedding of a selected basis into a compatible parent basis.
    pub fn from_embedding<Reduced, Parent>(reduced: &Reduced, parent: &Parent) -> Result<Self>
    where
        Reduced: Basis,
        Parent: Basis<State = Reduced::State>,
    {
        let columns = (0..reduced.len())
            .map(|column| {
                let state = reduced.state(column)?;
                Ok(vec![(parent.index(state)?, Complex64::new(1.0, 0.0))])
            })
            .collect::<Result<Vec<_>>>()?;
        Self::from_columns(parent.len(), columns)
    }

    /// Isometric lift from any basis into an explicitly selected parent basis.
    ///
    /// The reduced basis owns the symmetry-reduction convention through
    /// [`Basis::reduction_image`]. Iterating the explicit parent states makes
    /// this equally useful for built-in and runtime symmetry sectors, fixed
    /// particle subspaces, and unrestricted parent spaces.
    pub fn between<Reduced, Parent>(reduced: &Reduced, parent: &Parent) -> Result<Self>
    where
        Reduced: Basis,
        Parent: Basis<State = Reduced::State>,
    {
        let mut columns = vec![Vec::<(usize, Complex64)>::new(); reduced.len()];
        for row in 0..parent.len() {
            let state = parent.state(row)?;
            let Some(image) = reduced.reduction_image(state)? else {
                continue;
            };
            let column = reduced.index(*image.representative())?;
            columns[column].push((row, image.amplitude()));
        }
        Self::from_columns(parent.len(), columns)
    }

    pub fn from_general<Parent>(basis: &GeneralBasis<Parent>) -> Result<Self>
    where
        Parent: Basis,
        Parent::State: Hash + Ord + 'static,
    {
        Self::between(basis, &basis.parent)
    }

    pub const fn source_dimension(&self) -> usize {
        self.source_dimension
    }

    pub const fn reduced_dimension(&self) -> usize {
        self.reduced_dimension
    }

    /// Apply the adjoint projector to a parent-space vector.
    pub fn project(&self, parent: &[Complex64], reduced: &mut [Complex64]) -> Result<()> {
        if parent.len() != self.source_dimension || reduced.len() != self.reduced_dimension {
            return Err(QmbedError::DimensionMismatch(
                "projector input or output length does not match".into(),
            ));
        }
        reduced.fill(Complex64::new(0.0, 0.0));
        for (column, reduced_value) in reduced.iter_mut().enumerate() {
            for position in self.column_offsets[column]..self.column_offsets[column + 1] {
                *reduced_value += self.values[position].conj() * parent[self.row_indices[position]];
            }
        }
        Ok(())
    }

    pub fn lifted(&self, reduced: &[Complex64]) -> Result<Vec<Complex64>> {
        let mut parent = vec![Complex64::new(0.0, 0.0); self.source_dimension];
        self.apply(reduced, &mut parent)?;
        Ok(parent)
    }

    pub fn projected(&self, parent: &[Complex64]) -> Result<Vec<Complex64>> {
        let mut reduced = vec![Complex64::new(0.0, 0.0); self.reduced_dimension];
        self.project(parent, &mut reduced)?;
        Ok(reduced)
    }

    pub fn lift_batch(&self, reduced: &[Vec<Complex64>]) -> Result<Vec<Vec<Complex64>>> {
        reduced.iter().map(|state| self.lifted(state)).collect()
    }

    pub fn project_batch(&self, parent: &[Vec<Complex64>]) -> Result<Vec<Vec<Complex64>>> {
        parent.iter().map(|state| self.projected(state)).collect()
    }

    /// Frobenius norm of `(I - P P†) A P`, evaluated one reduced column at a
    /// time. Zero means the parent-space operator preserves this symmetry
    /// sector; no parent-space square projector is formed.
    pub fn symmetry_leakage_norm(&self, operator: &(impl LinearOperator + ?Sized)) -> Result<f64> {
        if operator.shape() != (self.source_dimension, self.source_dimension) {
            return Err(QmbedError::DimensionMismatch(
                "symmetry check requires a square parent-space operator".into(),
            ));
        }
        let mut total = 0.0;
        let mut reduced_basis = vec![Complex64::new(0.0, 0.0); self.reduced_dimension];
        let mut applied = vec![Complex64::new(0.0, 0.0); self.source_dimension];
        for column in 0..self.reduced_dimension {
            reduced_basis.fill(Complex64::new(0.0, 0.0));
            reduced_basis[column] = Complex64::new(1.0, 0.0);
            let lifted = self.lifted(&reduced_basis)?;
            operator.apply(&lifted, &mut applied)?;
            let projected = self.projected(&applied)?;
            let invariant_component = self.lifted(&projected)?;
            total += applied
                .iter()
                .zip(invariant_component)
                .map(|(value, invariant)| (*value - invariant).norm_sqr())
                .sum::<f64>();
        }
        Ok(total.sqrt())
    }

    pub fn preserves_operator_symmetry(
        &self,
        operator: &(impl LinearOperator + ?Sized),
        tolerance: f64,
    ) -> Result<bool> {
        if !tolerance.is_finite() || tolerance < 0.0 {
            return Err(QmbedError::InvalidOptions(
                "symmetry-check tolerance must be finite and nonnegative".into(),
            ));
        }
        Ok(self.symmetry_leakage_norm(operator)? <= tolerance)
    }
}

impl LinearOperator for BasisProjector {
    fn shape(&self) -> (usize, usize) {
        (self.source_dimension, self.reduced_dimension)
    }

    fn format(&self) -> MatrixFormat {
        MatrixFormat::Csc
    }

    fn apply(&self, input: &[Complex64], output: &mut [Complex64]) -> Result<()> {
        check_apply_shape(self.shape(), input, output)?;
        output.fill(Complex64::new(0.0, 0.0));
        for (column, &input_value) in input.iter().enumerate() {
            for position in self.column_offsets[column]..self.column_offsets[column + 1] {
                output[self.row_indices[position]] += self.values[position] * input_value;
            }
        }
        Ok(())
    }

    fn stored_triplets(&self) -> Result<Option<Vec<(usize, usize, Complex64)>>> {
        let mut entries = Vec::with_capacity(self.values.len());
        for column in 0..self.reduced_dimension {
            for position in self.column_offsets[column]..self.column_offsets[column + 1] {
                entries.push((self.row_indices[position], column, self.values[position]));
            }
        }
        Ok(Some(entries))
    }
}

/// Owned type-erased basis for frontends that choose a built-in basis at runtime.
///
/// Native Rust callers can continue using the concrete generic basis types.
/// Language bindings, configuration-driven workflows, and other runtime
/// frontends use this enum without duplicating the universal assembly logic.
#[derive(Clone, Debug)]
pub enum PackedBasis {
    Spin(SpinBasis1D),
    Boson(BosonBasis1D),
    SpinlessFermion(SpinlessFermionBasis1D),
    SpinfulFermion(SpinfulFermionBasis1D),
    GeneralSpin(SpinBasisGeneral),
    GeneralBoson(BosonBasisGeneral),
    GeneralSpinlessFermion(SpinlessFermionBasisGeneral),
    GeneralSpinfulFermion(SpinfulFermionBasisGeneral),
    Reversed(Box<PackedBasis>),
}

impl PackedBasis {
    /// Reverse the public basis-vector order without changing physical states.
    ///
    /// This is a general permutation view: `state`, `index`, transition row
    /// lookup, and universal operator assembly all observe the same order.
    pub fn reversed(self) -> Self {
        match self {
            Self::Reversed(inner) => *inner,
            basis => Self::Reversed(Box::new(basis)),
        }
    }
}

impl From<SpinBasis1D> for PackedBasis {
    fn from(basis: SpinBasis1D) -> Self {
        Self::Spin(basis)
    }
}

impl From<BosonBasis1D> for PackedBasis {
    fn from(basis: BosonBasis1D) -> Self {
        Self::Boson(basis)
    }
}

impl From<SpinlessFermionBasis1D> for PackedBasis {
    fn from(basis: SpinlessFermionBasis1D) -> Self {
        Self::SpinlessFermion(basis)
    }
}

impl From<SpinfulFermionBasis1D> for PackedBasis {
    fn from(basis: SpinfulFermionBasis1D) -> Self {
        Self::SpinfulFermion(basis)
    }
}

impl From<SpinBasisGeneral> for PackedBasis {
    fn from(basis: SpinBasisGeneral) -> Self {
        Self::GeneralSpin(basis)
    }
}

impl From<BosonBasisGeneral> for PackedBasis {
    fn from(basis: BosonBasisGeneral) -> Self {
        Self::GeneralBoson(basis)
    }
}

impl From<SpinlessFermionBasisGeneral> for PackedBasis {
    fn from(basis: SpinlessFermionBasisGeneral) -> Self {
        Self::GeneralSpinlessFermion(basis)
    }
}

impl From<SpinfulFermionBasisGeneral> for PackedBasis {
    fn from(basis: SpinfulFermionBasisGeneral) -> Self {
        Self::GeneralSpinfulFermion(basis)
    }
}

impl Basis for PackedBasis {
    type State = u128;

    fn len(&self) -> usize {
        match self {
            Self::Spin(basis) => basis.len(),
            Self::Boson(basis) => basis.len(),
            Self::SpinlessFermion(basis) => basis.len(),
            Self::SpinfulFermion(basis) => basis.len(),
            Self::GeneralSpin(basis) => basis.len(),
            Self::GeneralBoson(basis) => basis.len(),
            Self::GeneralSpinlessFermion(basis) => basis.len(),
            Self::GeneralSpinfulFermion(basis) => basis.len(),
            Self::Reversed(basis) => basis.len(),
        }
    }

    fn state(&self, index: usize) -> Result<Self::State> {
        match self {
            Self::Spin(basis) => basis.state(index),
            Self::Boson(basis) => basis.state(index),
            Self::SpinlessFermion(basis) => basis.state(index),
            Self::SpinfulFermion(basis) => basis.state(index),
            Self::GeneralSpin(basis) => basis.state(index),
            Self::GeneralBoson(basis) => basis.state(index),
            Self::GeneralSpinlessFermion(basis) => basis.state(index),
            Self::GeneralSpinfulFermion(basis) => basis.state(index),
            Self::Reversed(basis) => {
                let reversed = basis
                    .len()
                    .checked_sub(index + 1)
                    .ok_or(QmbedError::StateNotInBasis)?;
                basis.state(reversed)
            }
        }
    }

    fn index(&self, state: Self::State) -> Result<usize> {
        match self {
            Self::Spin(basis) => basis.index(state),
            Self::Boson(basis) => basis.index(state),
            Self::SpinlessFermion(basis) => basis.index(state),
            Self::SpinfulFermion(basis) => basis.index(state),
            Self::GeneralSpin(basis) => basis.index(state),
            Self::GeneralBoson(basis) => basis.index(state),
            Self::GeneralSpinlessFermion(basis) => basis.index(state),
            Self::GeneralSpinfulFermion(basis) => basis.index(state),
            Self::Reversed(basis) => basis.index(state).map(|index| basis.len() - index - 1),
        }
    }

    fn apply_local(
        &self,
        state: Self::State,
        operator: &str,
        sites: &[usize],
    ) -> Result<Option<(Self::State, Complex64)>> {
        match self {
            Self::Spin(basis) => basis.apply_local(state, operator, sites),
            Self::Boson(basis) => basis.apply_local(state, operator, sites),
            Self::SpinlessFermion(basis) => basis.apply_local(state, operator, sites),
            Self::SpinfulFermion(basis) => basis.apply_local(state, operator, sites),
            Self::GeneralSpin(basis) => basis.apply_local(state, operator, sites),
            Self::GeneralBoson(basis) => basis.apply_local(state, operator, sites),
            Self::GeneralSpinlessFermion(basis) => basis.apply_local(state, operator, sites),
            Self::GeneralSpinfulFermion(basis) => basis.apply_local(state, operator, sites),
            Self::Reversed(basis) => basis.apply_local(state, operator, sites),
        }
    }

    fn apply_local_transitions(
        &self,
        state: Self::State,
        operator: &str,
        sites: &[usize],
    ) -> Result<LocalTransitions<Self::State>> {
        match self {
            Self::Spin(basis) => basis.apply_local_transitions(state, operator, sites),
            Self::Boson(basis) => basis.apply_local_transitions(state, operator, sites),
            Self::SpinlessFermion(basis) => basis.apply_local_transitions(state, operator, sites),
            Self::SpinfulFermion(basis) => basis.apply_local_transitions(state, operator, sites),
            Self::GeneralSpin(basis) => basis.apply_local_transitions(state, operator, sites),
            Self::GeneralBoson(basis) => basis.apply_local_transitions(state, operator, sites),
            Self::GeneralSpinlessFermion(basis) => {
                basis.apply_local_transitions(state, operator, sites)
            }
            Self::GeneralSpinfulFermion(basis) => {
                basis.apply_local_transitions(state, operator, sites)
            }
            Self::Reversed(basis) => basis.apply_local_transitions(state, operator, sites),
        }
    }

    fn apply_local_unreduced_transitions(
        &self,
        state: Self::State,
        operator: &str,
        sites: &[usize],
    ) -> Result<LocalTransitions<Self::State>> {
        match self {
            Self::Spin(basis) => basis.apply_local_unreduced_transitions(state, operator, sites),
            Self::Boson(basis) => basis.apply_local_unreduced_transitions(state, operator, sites),
            Self::SpinlessFermion(basis) => {
                basis.apply_local_unreduced_transitions(state, operator, sites)
            }
            Self::SpinfulFermion(basis) => {
                basis.apply_local_unreduced_transitions(state, operator, sites)
            }
            Self::GeneralSpin(basis) => {
                basis.apply_local_unreduced_transitions(state, operator, sites)
            }
            Self::GeneralBoson(basis) => {
                basis.apply_local_unreduced_transitions(state, operator, sites)
            }
            Self::GeneralSpinlessFermion(basis) => {
                basis.apply_local_unreduced_transitions(state, operator, sites)
            }
            Self::GeneralSpinfulFermion(basis) => {
                basis.apply_local_unreduced_transitions(state, operator, sites)
            }
            Self::Reversed(basis) => {
                basis.apply_local_unreduced_transitions(state, operator, sites)
            }
        }
    }

    fn visit_local_unreduced_transitions<F>(
        &self,
        state: Self::State,
        operator: &str,
        sites: &[usize],
        visit: F,
    ) -> Result<()>
    where
        Self: Sized,
        F: FnMut(Self::State, Complex64) -> Result<()>,
    {
        match self {
            Self::Spin(basis) => {
                basis.visit_local_unreduced_transitions(state, operator, sites, visit)
            }
            Self::Boson(basis) => {
                basis.visit_local_unreduced_transitions(state, operator, sites, visit)
            }
            Self::SpinlessFermion(basis) => {
                basis.visit_local_unreduced_transitions(state, operator, sites, visit)
            }
            Self::SpinfulFermion(basis) => {
                basis.visit_local_unreduced_transitions(state, operator, sites, visit)
            }
            Self::GeneralSpin(basis) => {
                basis.visit_local_unreduced_transitions(state, operator, sites, visit)
            }
            Self::GeneralBoson(basis) => {
                basis.visit_local_unreduced_transitions(state, operator, sites, visit)
            }
            Self::GeneralSpinlessFermion(basis) => {
                basis.visit_local_unreduced_transitions(state, operator, sites, visit)
            }
            Self::GeneralSpinfulFermion(basis) => {
                basis.visit_local_unreduced_transitions(state, operator, sites, visit)
            }
            Self::Reversed(basis) => {
                basis.visit_local_unreduced_transitions(state, operator, sites, visit)
            }
        }
    }

    fn visit_preparsed_local_unreduced_transitions<F>(
        &self,
        state: Self::State,
        operator: &str,
        symbols: &[char],
        split: Option<usize>,
        sites: &[usize],
        visit: F,
    ) -> Result<()>
    where
        Self: Sized,
        F: FnMut(Self::State, Complex64) -> Result<()>,
    {
        match self {
            Self::Spin(basis) => basis.visit_preparsed_local_unreduced_transitions(
                state, operator, symbols, split, sites, visit,
            ),
            Self::Boson(basis) => basis.visit_preparsed_local_unreduced_transitions(
                state, operator, symbols, split, sites, visit,
            ),
            Self::SpinlessFermion(basis) => basis.visit_preparsed_local_unreduced_transitions(
                state, operator, symbols, split, sites, visit,
            ),
            Self::SpinfulFermion(basis) => basis.visit_preparsed_local_unreduced_transitions(
                state, operator, symbols, split, sites, visit,
            ),
            Self::GeneralSpin(basis) => basis.visit_preparsed_local_unreduced_transitions(
                state, operator, symbols, split, sites, visit,
            ),
            Self::GeneralBoson(basis) => basis.visit_preparsed_local_unreduced_transitions(
                state, operator, symbols, split, sites, visit,
            ),
            Self::GeneralSpinlessFermion(basis) => basis
                .visit_preparsed_local_unreduced_transitions(
                    state, operator, symbols, split, sites, visit,
                ),
            Self::GeneralSpinfulFermion(basis) => basis
                .visit_preparsed_local_unreduced_transitions(
                    state, operator, symbols, split, sites, visit,
                ),
            Self::Reversed(basis) => basis.visit_preparsed_local_unreduced_transitions(
                state, operator, symbols, split, sites, visit,
            ),
        }
    }

    fn transition_orbit_size(&self, state: Self::State) -> Result<usize> {
        match self {
            Self::Spin(basis) => basis.transition_orbit_size(state),
            Self::Boson(basis) => basis.transition_orbit_size(state),
            Self::SpinlessFermion(basis) => basis.transition_orbit_size(state),
            Self::SpinfulFermion(basis) => basis.transition_orbit_size(state),
            Self::GeneralSpin(basis) => basis.transition_orbit_size(state),
            Self::GeneralBoson(basis) => basis.transition_orbit_size(state),
            Self::GeneralSpinlessFermion(basis) => basis.transition_orbit_size(state),
            Self::GeneralSpinfulFermion(basis) => basis.transition_orbit_size(state),
            Self::Reversed(basis) => basis.transition_orbit_size(state),
        }
    }

    fn reduction_image(&self, state: Self::State) -> Result<Option<ReductionImage<Self::State>>> {
        match self {
            Self::Spin(basis) => basis.reduction_image(state),
            Self::Boson(basis) => basis.reduction_image(state),
            Self::SpinlessFermion(basis) => basis.reduction_image(state),
            Self::SpinfulFermion(basis) => basis.reduction_image(state),
            Self::GeneralSpin(basis) => basis.reduction_image(state),
            Self::GeneralBoson(basis) => basis.reduction_image(state),
            Self::GeneralSpinlessFermion(basis) => basis.reduction_image(state),
            Self::GeneralSpinfulFermion(basis) => basis.reduction_image(state),
            Self::Reversed(basis) => basis.reduction_image(state),
        }
    }

    fn reduce_transition(
        &self,
        state: Self::State,
        source_orbit_size: usize,
    ) -> Result<Option<(Self::State, Complex64)>> {
        match self {
            Self::Spin(basis) => basis.reduce_transition(state, source_orbit_size),
            Self::Boson(basis) => basis.reduce_transition(state, source_orbit_size),
            Self::SpinlessFermion(basis) => basis.reduce_transition(state, source_orbit_size),
            Self::SpinfulFermion(basis) => basis.reduce_transition(state, source_orbit_size),
            Self::GeneralSpin(basis) => basis.reduce_transition(state, source_orbit_size),
            Self::GeneralBoson(basis) => basis.reduce_transition(state, source_orbit_size),
            Self::GeneralSpinlessFermion(basis) => {
                basis.reduce_transition(state, source_orbit_size)
            }
            Self::GeneralSpinfulFermion(basis) => basis.reduce_transition(state, source_orbit_size),
            Self::Reversed(basis) => basis.reduce_transition(state, source_orbit_size),
        }
    }

    fn index_transition(
        &self,
        state: Self::State,
        source_orbit_size: usize,
    ) -> Result<Option<(usize, Complex64)>> {
        match self {
            Self::Spin(basis) => basis.index_transition(state, source_orbit_size),
            Self::Boson(basis) => basis.index_transition(state, source_orbit_size),
            Self::SpinlessFermion(basis) => basis.index_transition(state, source_orbit_size),
            Self::SpinfulFermion(basis) => basis.index_transition(state, source_orbit_size),
            Self::GeneralSpin(basis) => basis.index_transition(state, source_orbit_size),
            Self::GeneralBoson(basis) => basis.index_transition(state, source_orbit_size),
            Self::GeneralSpinlessFermion(basis) => basis.index_transition(state, source_orbit_size),
            Self::GeneralSpinfulFermion(basis) => basis.index_transition(state, source_orbit_size),
            Self::Reversed(basis) => Ok(basis
                .index_transition(state, source_orbit_size)?
                .map(|(index, amplitude)| (basis.len() - index - 1, amplitude))),
        }
    }

    fn operator_preserves_particle_sector(&self, operator: &str) -> Result<bool> {
        match self {
            Self::Spin(basis) => basis.operator_preserves_particle_sector(operator),
            Self::Boson(basis) => basis.operator_preserves_particle_sector(operator),
            Self::SpinlessFermion(basis) => basis.operator_preserves_particle_sector(operator),
            Self::SpinfulFermion(basis) => basis.operator_preserves_particle_sector(operator),
            Self::GeneralSpin(basis) => basis.operator_preserves_particle_sector(operator),
            Self::GeneralBoson(basis) => basis.operator_preserves_particle_sector(operator),
            Self::GeneralSpinlessFermion(basis) => {
                basis.operator_preserves_particle_sector(operator)
            }
            Self::GeneralSpinfulFermion(basis) => {
                basis.operator_preserves_particle_sector(operator)
            }
            Self::Reversed(basis) => basis.operator_preserves_particle_sector(operator),
        }
    }
}
