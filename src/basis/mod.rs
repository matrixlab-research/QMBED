use std::cmp::Ordering;
use std::collections::{HashMap, HashSet, VecDeque};
use std::hash::Hash;
use std::sync::{Arc, RwLock};

use num_bigint::BigUint;
use num_complex::Complex64;
use smallvec::SmallVec;

use crate::operator::{LinearOperator, MatrixFormat, Operator, check_apply_shape};
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

    /// Site-aware particle-sector check for bases whose local labels encode
    /// species as well as position.
    ///
    /// Most bases depend only on the operator string and inherit this default.
    /// Multi-species bases may override it so a unified orbital convention
    /// remains physically meaningful without inserting a species separator.
    fn operator_preserves_particle_sector_on_sites(
        &self,
        operator: &str,
        _sites: &[usize],
    ) -> Result<bool> {
        self.operator_preserves_particle_sector(operator)
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

fn operator_number_changes(operator: &str) -> Result<Vec<i32>> {
    let mut changes = vec![0_i32];
    for character in operator.chars().filter(|character| *character != '|') {
        match character {
            '+' => {
                for change in &mut changes {
                    *change += 1;
                }
            }
            '-' => {
                for change in &mut changes {
                    *change -= 1;
                }
            }
            'x' | 'y' => {
                let mut lowered = changes.clone();
                for change in &mut changes {
                    *change += 1;
                }
                for change in &mut lowered {
                    *change -= 1;
                }
                changes.extend(lowered);
                changes.sort_unstable();
                changes.dedup();
            }
            'I' | 'n' | 'z' => {}
            _ => return Err(QmbedError::InvalidOperator(operator.into())),
        }
    }
    Ok(changes)
}

fn canonical_particle_sectors(
    sectors: impl IntoIterator<Item = usize>,
    maximum: usize,
    family: &str,
) -> Result<Vec<usize>> {
    let mut sectors = sectors.into_iter().collect::<Vec<_>>();
    if sectors.is_empty() {
        return Err(QmbedError::InvalidSector(format!(
            "{family} particle-sector union must be nonempty"
        )));
    }
    if sectors.iter().any(|&sector| sector > maximum) {
        return Err(QmbedError::InvalidSector(format!(
            "{family} particle sector exceeds the local capacity"
        )));
    }
    sectors.sort_unstable();
    sectors.dedup();
    Ok(sectors)
}

fn selected_sectors_preserve_changes(
    fixed: Option<usize>,
    sectors: Option<&[usize]>,
    maximum: usize,
    changes: &[i32],
) -> bool {
    let contains = |sector: usize| {
        sectors.map_or_else(
            || fixed == Some(sector),
            |selected| selected.binary_search(&sector).is_ok(),
        )
    };
    let preserves_source = |source: usize| {
        changes.iter().copied().all(|change| {
            let target = source as i128 + i128::from(change);
            target < 0 || target > maximum as i128 || contains(target as usize)
        })
    };
    match (fixed, sectors) {
        (_, Some(sectors)) => sectors.iter().copied().all(preserves_source),
        (Some(fixed), None) => preserves_source(fixed),
        (None, None) => true,
    }
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

fn fixed_weight_sector_states(
    sites: usize,
    particles: Option<usize>,
    particle_sectors: Option<&[usize]>,
) -> Result<Vec<u128>> {
    let Some(sectors) = particle_sectors else {
        return fixed_weight_states(sites, particles);
    };
    let mut states = Vec::new();
    for &sector in sectors {
        states.extend(fixed_weight_states(sites, Some(sector))?);
    }
    states.sort_unstable();
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

fn fixed_digit_sum_sector_states(
    sites: usize,
    states_per_site: usize,
    total: Option<usize>,
    particle_sectors: Option<&[usize]>,
) -> Result<Vec<u128>> {
    let Some(sectors) = particle_sectors else {
        return fixed_digit_sum_states(sites, states_per_site, total);
    };
    let mut states = Vec::new();
    for &sector in sectors {
        states.extend(fixed_digit_sum_states(
            sites,
            states_per_site,
            Some(sector),
        )?);
    }
    states.sort_unstable();
    Ok(states)
}

/// Site-local constraint for packed binary species.
///
/// A local state is encoded as a bit mask over species: bit `s` is the
/// occupation of species `s` at the site. This represents exclusions such as
/// no-double-occupancy without embedding a model-specific predicate in the
/// fermion basis or an interop layer.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LocalOccupationConstraint {
    species: usize,
    allowed_local_states: Vec<usize>,
}

impl LocalOccupationConstraint {
    pub fn new(
        species: usize,
        allowed_local_states: impl IntoIterator<Item = usize>,
    ) -> Result<Self> {
        let local_dimension = 1_usize
            .checked_shl(u32::try_from(species).unwrap_or(u32::MAX))
            .ok_or_else(|| {
                QmbedError::UnsupportedBackend(
                    "the local binary-species dimension exceeds usize".into(),
                )
            })?;
        if species == 0 {
            return Err(QmbedError::InvalidSector(
                "a local occupation constraint needs at least one species".into(),
            ));
        }
        let mut allowed_local_states = allowed_local_states.into_iter().collect::<Vec<_>>();
        if allowed_local_states.is_empty() {
            return Err(QmbedError::InvalidSector(
                "a local occupation constraint must allow at least one state".into(),
            ));
        }
        if allowed_local_states
            .iter()
            .any(|&state| state >= local_dimension)
        {
            return Err(QmbedError::InvalidSector(
                "an allowed local occupation is outside the binary-species space".into(),
            ));
        }
        allowed_local_states.sort_unstable();
        allowed_local_states.dedup();
        Ok(Self {
            species,
            allowed_local_states,
        })
    }

    pub const fn species(&self) -> usize {
        self.species
    }

    pub fn allowed_local_states(&self) -> &[usize] {
        &self.allowed_local_states
    }

    pub fn accepts_packed_state(&self, state: u128, sites: usize) -> Result<bool> {
        let orbitals = self.species.checked_mul(sites).ok_or_else(|| {
            QmbedError::UnsupportedBackend("binary-species orbital count overflow".into())
        })?;
        if orbitals > 128 {
            return Err(QmbedError::UnsupportedBackend(
                "the packed local-constraint backend supports at most 128 orbitals".into(),
            ));
        }
        for site in 0..sites {
            let mut local_state = 0_usize;
            for species in 0..self.species {
                let orbital = species * sites + site;
                if state & (1_u128 << orbital) != 0 {
                    local_state |= 1_usize << species;
                }
            }
            if self
                .allowed_local_states
                .binary_search(&local_state)
                .is_err()
            {
                return Ok(false);
            }
        }
        Ok(true)
    }
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
    particle_sectors: Option<Vec<usize>>,
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
            particle_sectors: None,
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

    /// Allowed additive spin-occupation sectors when the basis is a union.
    pub fn particle_sectors(&self) -> Option<&[usize]> {
        self.particle_sectors.as_deref()
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
    particle_sectors: Option<Vec<usize>>,
    momentum: Option<i32>,
    parity: Option<i8>,
    normalization: SpinNormalization,
}

impl SpinBasisBuilder {
    pub const fn spin_twice(mut self, spin_twice: u16) -> Self {
        self.spin_twice = spin_twice;
        self
    }

    pub fn up(mut self, up: usize) -> Self {
        self.up = Some(up);
        self.particle_sectors = None;
        self
    }

    pub fn magnetization(mut self, up: usize) -> Self {
        self.up = Some(up);
        self.particle_sectors = None;
        self
    }

    /// Select a union of additive spin-occupation sectors.
    pub fn particle_sectors(mut self, sectors: impl IntoIterator<Item = usize>) -> Self {
        self.particle_sectors = Some(sectors.into_iter().collect());
        self.up = None;
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
        let maximum_particles = self.sites.saturating_mul(usize::from(self.spin_twice));
        let particle_sectors = self
            .particle_sectors
            .map(|sectors| canonical_particle_sectors(sectors, maximum_particles, "spin"))
            .transpose()?;
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
            fixed_weight_sector_states(self.sites, self.up, particle_sectors.as_deref())?
        } else {
            fixed_digit_sum_sector_states(
                self.sites,
                states_per_site,
                self.up,
                particle_sectors.as_deref(),
            )?
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
            particle_sectors,
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
            if self.spin_twice == 1 && self.particle_sectors.is_none() {
                return self.up.map_or_else(
                    || direct_state_index(&self.states, state),
                    |up| fixed_weight_state_index(state, self.sites, up),
                );
            }
            if self.up.is_none() && self.particle_sectors.is_none() {
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
        Ok(selected_sectors_preserve_changes(
            self.up,
            self.particle_sectors.as_deref(),
            self.sites.saturating_mul(usize::from(self.spin_twice)),
            &operator_number_changes(operator)?,
        ))
    }
}

/// Truncated on-site boson basis.
#[derive(Clone, Debug)]
pub struct BosonBasis1D {
    sites: usize,
    particles: Option<usize>,
    particle_sectors: Option<Vec<usize>>,
    states_per_site: usize,
    states: Vec<u128>,
}

impl BosonBasis1D {
    pub fn builder(sites: usize, states_per_site: usize) -> BosonBasisBuilder {
        BosonBasisBuilder {
            sites,
            particles: None,
            particle_sectors: None,
            states_per_site,
        }
    }

    pub const fn sites(&self) -> usize {
        self.sites
    }

    pub const fn particles(&self) -> Option<usize> {
        self.particles
    }

    /// Allowed total-occupation sectors when the basis is a union.
    pub fn particle_sectors(&self) -> Option<&[usize]> {
        self.particle_sectors.as_deref()
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
                'z' => {
                    amplitude *=
                        occupation as f64 - 0.5 * (self.states_per_site.saturating_sub(1)) as f64;
                }
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
    particle_sectors: Option<Vec<usize>>,
    states_per_site: usize,
}

impl BosonBasisBuilder {
    pub fn particles(mut self, particles: usize) -> Self {
        self.particles = Some(particles);
        self.particle_sectors = None;
        self
    }

    /// Select a union of total boson-occupation sectors.
    pub fn particle_sectors(mut self, sectors: impl IntoIterator<Item = usize>) -> Self {
        self.particle_sectors = Some(sectors.into_iter().collect());
        self.particles = None;
        self
    }

    pub fn build(self) -> Result<BosonBasis1D> {
        if self.sites == 0 || self.states_per_site == 0 {
            return Err(QmbedError::InvalidSector(
                "boson sites and states_per_site must be positive".into(),
            ));
        }
        let maximum_particles = self.sites * (self.states_per_site - 1);
        if self
            .particles
            .is_some_and(|count| count > maximum_particles)
        {
            return Err(QmbedError::InvalidSector(
                "particle count exceeds the local cutoff".into(),
            ));
        }
        let particle_sectors = self
            .particle_sectors
            .map(|sectors| canonical_particle_sectors(sectors, maximum_particles, "boson"))
            .transpose()?;
        let states = fixed_digit_sum_sector_states(
            self.sites,
            self.states_per_site,
            self.particles,
            particle_sectors.as_deref(),
        )?;
        if states.is_empty() {
            return Err(QmbedError::InvalidSector("empty boson sector".into()));
        }
        Ok(BosonBasis1D {
            sites: self.sites,
            particles: self.particles,
            particle_sectors,
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
        if self.particles.is_none() && self.particle_sectors.is_none() {
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
        Ok(selected_sectors_preserve_changes(
            self.particles,
            self.particle_sectors.as_deref(),
            self.sites * (self.states_per_site - 1),
            &operator_number_changes(operator)?,
        ))
    }
}

/// Single-flavor fermion basis.
#[derive(Clone, Debug)]
pub struct SpinlessFermionBasis1D {
    sites: usize,
    particles: Option<usize>,
    particle_sectors: Option<Vec<usize>>,
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
            particle_sectors: None,
            momentum: None,
        }
    }

    pub const fn sites(&self) -> usize {
        self.sites
    }

    pub const fn particles(&self) -> Option<usize> {
        self.particles
    }

    /// Allowed particle-number sectors when the basis is a union.
    pub fn particle_sectors(&self) -> Option<&[usize]> {
        self.particle_sectors.as_deref()
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
    particle_sectors: Option<Vec<usize>>,
    momentum: Option<i32>,
}

impl SpinlessFermionBasisBuilder {
    pub fn particles(mut self, particles: usize) -> Self {
        self.particles = Some(particles);
        self.particle_sectors = None;
        self
    }

    /// Select a union of particle-number sectors.
    pub fn particle_sectors(mut self, sectors: impl IntoIterator<Item = usize>) -> Self {
        self.particle_sectors = Some(sectors.into_iter().collect());
        self.particles = None;
        self
    }

    pub const fn momentum(mut self, momentum: i32) -> Self {
        self.momentum = Some(momentum);
        self
    }

    pub fn build(self) -> Result<SpinlessFermionBasis1D> {
        let particle_sectors = self
            .particle_sectors
            .map(|sectors| canonical_particle_sectors(sectors, self.sites, "fermion"))
            .transpose()?;
        let parent_states =
            fixed_weight_sector_states(self.sites, self.particles, particle_sectors.as_deref())?;
        let (states, orbit_lengths, symmetry_lookup, momentum) =
            fermion_translation_sector(parent_states, self.sites, self.momentum)?;
        Ok(SpinlessFermionBasis1D {
            sites: self.sites,
            particles: self.particles,
            particle_sectors,
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
                Complex64::new(if occupied { 0.5 } else { -0.5 }, 0.0),
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
                Complex64::new(0.0, if occupied { -sign } else { sign }),
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
            if self.particle_sectors.is_some() {
                return state_index(&self.states, state);
            }
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
        Ok(selected_sectors_preserve_changes(
            self.particles,
            self.particle_sectors.as_deref(),
            self.sites,
            &operator_number_changes(operator)?,
        ))
    }
}

/// Two-flavor fermion basis with all up orbitals ordered before all down orbitals.
///
/// Rows follow the direct-product convention `up ⊗ down`: the up-basis row
/// is the major index and the down-basis row is the minor index.
#[derive(Clone, Debug)]
pub struct SpinfulFermionBasis1D {
    sites: usize,
    particles_up: Option<usize>,
    particles_down: Option<usize>,
    particle_sectors: Option<Vec<(usize, usize)>>,
    local_occupation_constraint: Option<LocalOccupationConstraint>,
    states: Vec<u128>,
    indices: Option<HashMap<u128, usize>>,
}

impl SpinfulFermionBasis1D {
    pub fn builder(sites: usize) -> SpinfulFermionBasisBuilder {
        SpinfulFermionBasisBuilder {
            sites,
            particles_up: None,
            particles_down: None,
            particle_sectors: None,
            local_occupation_constraint: None,
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

    pub const fn local_occupation_constraint(&self) -> Option<&LocalOccupationConstraint> {
        self.local_occupation_constraint.as_ref()
    }

    fn apply_local_symbols(
        &self,
        mut state: u128,
        symbols: &[char],
        split: Option<usize>,
        sites: &[usize],
    ) -> Result<Option<(u128, Complex64)>> {
        if symbols.len() != sites.len() || split.is_some_and(|value| value > symbols.len()) {
            return Err(QmbedError::InvalidCoupling(format!(
                "operator arity {} does not match {} sites",
                symbols.len(),
                sites.len()
            )));
        }
        let mut amplitude = Complex64::new(1.0, 0.0);
        for (position, (&site, &op)) in sites.iter().zip(symbols).enumerate().rev() {
            let orbital = match split {
                Some(boundary) => {
                    checked_site(site, self.sites)?;
                    if position < boundary {
                        site
                    } else {
                        self.sites + site
                    }
                }
                None => {
                    let orbitals = self.sites.checked_mul(2).ok_or_else(|| {
                        QmbedError::UnsupportedBackend("spinful orbital count is too large".into())
                    })?;
                    checked_site(site, orbitals)?;
                    site
                }
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
    local_occupation_constraint: Option<LocalOccupationConstraint>,
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

    /// Restrict the allowed site-local occupation masks for the two binary
    /// fermion species.
    pub fn local_occupation_constraint(mut self, constraint: LocalOccupationConstraint) -> Self {
        self.local_occupation_constraint = Some(constraint);
        self
    }

    pub fn build(self) -> Result<SpinfulFermionBasis1D> {
        if self.sites > 64 {
            return Err(QmbedError::UnsupportedBackend(
                "the packed spinful backend supports at most 64 sites".into(),
            ));
        }
        if self
            .local_occupation_constraint
            .as_ref()
            .is_some_and(|constraint| constraint.species() != 2)
        {
            return Err(QmbedError::InvalidSector(
                "spinful fermions require a two-species local occupation constraint".into(),
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
        // Keep the row order identical to the direct product
        // `up_basis ⊗ down_basis`: the up-sector row is the major index and
        // the down-sector row is the minor index. This is observable for
        // state vectors and density matrices, and avoids making a spinful
        // basis silently disagree with the equivalent `PackedTensorBasis`.
        let mask = if self.sites == 128 {
            u128::MAX
        } else {
            (1_u128 << self.sites) - 1
        };
        states.sort_unstable_by_key(|state| ((*state & mask), (*state >> self.sites)));
        states.dedup();
        if let Some(constraint) = &self.local_occupation_constraint {
            let mut filtered = Vec::with_capacity(states.len());
            for state in states {
                if constraint.accepts_packed_state(state, self.sites)? {
                    filtered.push(state);
                }
            }
            states = filtered;
        }
        let indices = (self.particle_sectors.is_some()
            || self.local_occupation_constraint.is_some())
        .then(|| {
            states
                .iter()
                .copied()
                .enumerate()
                .map(|(index, state)| (state, index))
                .collect()
        });
        Ok(SpinfulFermionBasis1D {
            sites: self.sites,
            particles_up: self.particles_up,
            particles_down: self.particles_down,
            particle_sectors: self.particle_sectors,
            local_occupation_constraint: self.local_occupation_constraint,
            states,
            indices,
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
        if self.particle_sectors.is_none() && self.local_occupation_constraint.is_none() {
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
            let down_dimension = match self.particles_down {
                Some(particles) => binomial(self.sites, particles),
                None => 1_usize
                    .checked_shl(u32::try_from(self.sites).unwrap_or(u32::MAX))
                    .ok_or(QmbedError::StateNotInBasis)?,
            };
            let index = up_index
                .checked_mul(down_dimension)
                .and_then(|offset| offset.checked_add(down_index))
                .ok_or(QmbedError::StateNotInBasis)?;
            if index < self.states.len() {
                return Ok(index);
            }
            return Err(QmbedError::StateNotInBasis);
        }
        self.indices
            .as_ref()
            .and_then(|indices| indices.get(&state).copied())
            .ok_or(QmbedError::StateNotInBasis)
    }

    fn apply_local(
        &self,
        state: Self::State,
        operator: &str,
        sites: &[usize],
    ) -> Result<Option<(Self::State, Complex64)>> {
        let symbols = operator_chars(operator, sites)?;
        let split = operator
            .find('|')
            .map(|position| operator[..position].chars().count());
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
        if let Some((target, amplitude)) = self.apply_local_symbols(state, symbols, split, sites)? {
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

    fn operator_preserves_particle_sector_on_sites(
        &self,
        operator: &str,
        sites: &[usize],
    ) -> Result<bool> {
        if operator.contains('|') {
            return self.operator_preserves_particle_sector(operator);
        }
        let symbols = operator_chars(operator, sites)?;
        let mut changes = [Some(0_i32), Some(0_i32)];
        for (&symbol, &orbital) in symbols.iter().zip(sites) {
            let orbitals = self.sites.checked_mul(2).ok_or_else(|| {
                QmbedError::UnsupportedBackend("spinful orbital count is too large".into())
            })?;
            checked_site(orbital, orbitals)?;
            let species = usize::from(orbital >= self.sites);
            match symbol {
                '+' => {
                    if let Some(change) = &mut changes[species] {
                        *change += 1;
                    }
                }
                '-' => {
                    if let Some(change) = &mut changes[species] {
                        *change -= 1;
                    }
                }
                'x' | 'y' => changes[species] = None,
                'I' | 'n' | 'z' => {}
                _ => return Err(QmbedError::InvalidOperator(operator.into())),
            }
        }
        if let Some(sectors) = &self.particle_sectors {
            let sectors: HashSet<_> = sectors.iter().copied().collect();
            return Ok(sectors.iter().all(|&(up, down)| {
                let Some(target_up) = changes[0].and_then(|change| (up as i32).checked_add(change))
                else {
                    return false;
                };
                let Some(target_down) =
                    changes[1].and_then(|change| (down as i32).checked_add(change))
                else {
                    return false;
                };
                target_up >= 0
                    && target_down >= 0
                    && target_up <= self.sites as i32
                    && target_down <= self.sites as i32
                    && sectors.contains(&(target_up as usize, target_down as usize))
            }));
        }
        Ok(self
            .particles_up
            .is_none_or(|_| changes[0].is_some_and(|change| change == 0))
            && self
                .particles_down
                .is_none_or(|_| changes[1].is_some_and(|change| change == 0)))
    }
}

type UserAction<State> = Arc<
    dyn Fn(State, usize, &mut dyn FnMut(State, Complex64) -> Result<()>) -> Result<()>
        + Send
        + Sync,
>;
type UserStateFactory<State> = Arc<dyn Fn() -> Result<Vec<State>> + Send + Sync>;

/// Callback-defined constrained basis using the same assembly path as built-ins.
#[derive(Clone)]
pub struct UserBasis<State>
where
    State: Copy + Eq + Hash + Send + Sync,
{
    sites: usize,
    states: Vec<State>,
    indices: HashMap<State, usize>,
    operators: HashMap<char, UserAction<State>>,
}

impl<State> std::fmt::Debug for UserBasis<State>
where
    State: Copy + Eq + Hash + Send + Sync,
{
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut operators: Vec<_> = self.operators.keys().copied().collect();
        operators.sort_unstable();
        formatter
            .debug_struct("UserBasis")
            .field("sites", &self.sites)
            .field("states", &self.states.len())
            .field("operators", &operators)
            .finish()
    }
}

impl<State> UserBasis<State>
where
    State: Copy + Eq + Hash + Send + Sync + 'static,
{
    pub fn builder(sites: usize) -> UserBasisBuilder<State> {
        UserBasisBuilder {
            sites,
            states: Vec::new(),
            state_factory: None,
            operators: HashMap::new(),
        }
    }

    pub const fn sites(&self) -> usize {
        self.sites
    }

    /// Reuse the same callback-defined local algebra on another explicit
    /// state set.
    ///
    /// This separates the local operator semantics from state-space
    /// constraints. Language bindings use it to construct the full,
    /// constrained, and symmetry-reduced views required by projection and
    /// subsystem analysis without duplicating callback adapters.
    pub fn with_states(&self, states: impl IntoIterator<Item = State>) -> Result<UserBasis<State>> {
        let mut indices = HashMap::new();
        let states: Vec<_> = states.into_iter().collect();
        if states.is_empty() {
            return Err(QmbedError::InvalidSector(
                "UserBasis requires at least one accepted state".into(),
            ));
        }
        indices.reserve(states.len());
        for (index, state) in states.iter().copied().enumerate() {
            if indices.insert(state, index).is_some() {
                return Err(QmbedError::InvalidSector(
                    "UserBasis states must be unique".into(),
                ));
            }
        }
        Ok(UserBasis {
            sites: self.sites,
            states,
            indices,
            operators: self.operators.clone(),
        })
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
/// Fermionic maps compute the parity of the originally occupied-orbital
/// permutation instead of requiring a frontend-provided phase callback.
/// Binary local permutations may also exchange empty and occupied digits.
/// Such particle-hole flips change the mapped occupation but introduce no
/// additional Fock-state phase; the orbital reordering alone fixes the sign.
#[derive(Clone, Debug)]
pub struct LatticeSymmetryMap {
    states_per_site: usize,
    destinations: Vec<usize>,
    local_permutations: Vec<Vec<usize>>,
    statistics: ExchangeStatistics,
    place_values: Option<Vec<u128>>,
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
        if statistics == ExchangeStatistics::Fermionic && states_per_site != 2 {
            return Err(QmbedError::InvalidOptions(
                "fermionic lattice maps require binary occupation".into(),
            ));
        }

        let base = states_per_site as u128;
        let mut packed_place_values = Vec::with_capacity(destinations.len());
        let mut place = Some(1_u128);
        for site in 0..destinations.len() {
            if let Some(value) = place {
                packed_place_values.push(value);
            }
            if site + 1 < destinations.len() {
                place = place.and_then(|value| value.checked_mul(base));
            }
        }
        let place_values =
            (packed_place_values.len() == destinations.len()).then_some(packed_place_values);
        let state_limit = place.and_then(|value| value.checked_mul(base));
        let exact_full_range = base.is_power_of_two()
            && (base.trailing_zeros() as usize).checked_mul(destinations.len()) == Some(128);
        let wide_binary = states_per_site == 2 && destinations.len() <= 16_384;
        if place_values.is_none() && !wide_binary {
            return Err(QmbedError::UnsupportedBackend(
                "lattice-symmetry state encoding exceeds the supported fixed-width backends".into(),
            ));
        }
        if place_values.is_some() && state_limit.is_none() && !exact_full_range {
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

    pub const fn period(&self) -> usize {
        self.period
    }

    fn mapped_state(&self, state: u128) -> Result<(u128, Complex64)> {
        let place_values = self.place_values.as_ref().ok_or_else(|| {
            QmbedError::UnsupportedBackend(
                "this lattice symmetry requires a wide fixed-width state".into(),
            )
        })?;
        if self.state_limit.is_some_and(|limit| state >= limit) {
            return Err(QmbedError::StateNotInBasis);
        }
        let base = self.states_per_site as u128;
        let mut mapped = 0_u128;
        let mut occupied_destinations = 0_u128;
        let mut odd_fermion_permutation = false;
        for source in 0..self.destinations.len() {
            let digit = usize::try_from((state / place_values[source]) % base)
                .map_err(|_| QmbedError::StateNotInBasis)?;
            let mapped_digit = self.local_permutations[source][digit];
            let destination = self.destinations[source];
            mapped += mapped_digit as u128 * place_values[destination];
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

impl<const WORDS: usize> SymmetryMap<WideState<WORDS>> for LatticeSymmetryMap {
    fn period(&self) -> usize {
        self.period
    }

    fn apply(&self, state: WideState<WORDS>) -> Result<(WideState<WORDS>, Complex64)> {
        if self.states_per_site != 2
            || self.destinations.len() > WideState::<WORDS>::capacity_bits()
        {
            return Err(QmbedError::UnsupportedBackend(
                "wide lattice symmetries currently require a binary fixed-width state".into(),
            ));
        }
        if state.has_bits_at_or_above(self.destinations.len()) {
            return Err(QmbedError::StateNotInBasis);
        }
        let mut mapped = WideState::<WORDS>::zero();
        let mut occupied_destinations = WideState::<WORDS>::zero();
        let mut odd_fermion_permutation = false;
        for source in 0..self.destinations.len() {
            let occupied = state.bit(source)?;
            let digit = usize::from(occupied);
            let mapped_digit = self.local_permutations[source][digit];
            let destination = self.destinations[source];
            mapped = mapped.with_bit(destination, mapped_digit != 0)?;
            if self.statistics == ExchangeStatistics::Fermionic && occupied {
                odd_fermion_permutation ^=
                    occupied_destinations.count_ones_after(destination) % 2 == 1;
                occupied_destinations = occupied_destinations.with_bit(destination, true)?;
            }
        }
        Ok((
            mapped,
            Complex64::new(if odd_fermion_permutation { -1.0 } else { 1.0 }, 0.0),
        ))
    }
}

impl SymmetryMap<ErasedState> for LatticeSymmetryMap {
    fn period(&self) -> usize {
        self.period
    }

    fn apply(&self, state: ErasedState) -> Result<(ErasedState, Complex64)> {
        if state.width_bits != self.destinations.len() {
            return Err(QmbedError::StateNotInBasis);
        }
        let (value, phase) = match state.value {
            ErasedStateValue::U128(value) => {
                let (value, phase) = self.mapped_state(value)?;
                (ErasedStateValue::U128(value), phase)
            }
            ErasedStateValue::U256(value) => {
                let (value, phase) = <Self as SymmetryMap<U256>>::apply(self, value)?;
                (ErasedStateValue::U256(value), phase)
            }
            ErasedStateValue::U1024(value) => {
                let (value, phase) = <Self as SymmetryMap<U1024>>::apply(self, value)?;
                (ErasedStateValue::U1024(value), phase)
            }
            ErasedStateValue::U4096(value) => {
                let (value, phase) = <Self as SymmetryMap<U4096>>::apply(self, value)?;
                (ErasedStateValue::U4096(value), phase)
            }
            ErasedStateValue::U16384(value) => {
                let (value, phase) = <Self as SymmetryMap<U16384>>::apply(self, value)?;
                (ErasedStateValue::U16384(value), phase)
            }
        };
        Ok((
            ErasedState {
                width_bits: state.width_bits,
                value,
            },
            phase,
        ))
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

#[derive(Clone)]
struct SymmetryGenerator<State> {
    map: Arc<dyn SymmetryMap<State>>,
    sector: i32,
}

/// Finite generators and sector phases that extend to a one-dimensional
/// character of the generated group.
#[derive(Clone)]
pub struct SymmetryReducer<State> {
    generators: Vec<SymmetryGenerator<State>>,
    orbit_cache: Arc<RwLock<HashMap<State, Arc<SymmetryTrace<State>>>>>,
}

impl<State> std::fmt::Debug for SymmetryReducer<State> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("SymmetryReducer")
            .field("generators", &self.generators.len())
            .field("cached_orbits", &self.cached_orbits())
            .finish()
    }
}

impl<State> SymmetryReducer<State> {
    pub fn new() -> Self {
        Self {
            generators: Vec::new(),
            orbit_cache: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub fn with_map<M>(mut self, map: M, sector: i32) -> Self
    where
        M: SymmetryMap<State> + 'static,
    {
        // A changed generator list defines a different group action.  Clones
        // of an unchanged reducer share their cache; derived reducers do not.
        self.orbit_cache = Arc::new(RwLock::new(HashMap::new()));
        self.generators.push(SymmetryGenerator {
            map: Arc::new(map),
            sector,
        });
        self
    }

    pub fn generators(&self) -> usize {
        self.generators.len()
    }

    pub fn cached_orbits(&self) -> usize {
        self.orbit_cache.read().map_or(0, |cache| cache.len())
    }

    pub fn clear_cache(&self) {
        if let Ok(mut cache) = self.orbit_cache.write() {
            cache.clear();
        }
    }

    /// Product of declared generator periods.
    ///
    /// This is not generally the order of a non-Abelian generated group. It is
    /// exposed because compatibility layers such as QuSpin define their
    /// historical normalization convention from this product.
    pub fn period_product(&self) -> Result<usize> {
        self.generators
            .iter()
            .try_fold(1_usize, |product, generator| {
                product.checked_mul(generator.map.period()).ok_or_else(|| {
                    QmbedError::UnsupportedBackend("symmetry period product is too large".into())
                })
            })
    }
}

impl<State> Default for SymmetryReducer<State> {
    fn default() -> Self {
        Self::new()
    }
}

/// Backward-compatible name for a reducer configured with one sector
/// character per generator.
pub type SymmetrySector<State> = SymmetryReducer<State>;

/// Result of reducing one physical state without materializing a basis.
#[derive(Clone, Debug, PartialEq)]
pub struct SymmetryOrbit<State> {
    representative: State,
    orbit_size: usize,
    compatible: bool,
    phase: Option<Complex64>,
    physical_phase_to_representative: Complex64,
    generator_word: Vec<usize>,
}

impl<State> SymmetryOrbit<State> {
    pub const fn representative(&self) -> &State {
        &self.representative
    }

    pub const fn orbit_size(&self) -> usize {
        self.orbit_size
    }

    pub const fn is_compatible(&self) -> bool {
        self.compatible
    }

    /// Unit phase of the queried physical state in the representative vector.
    pub const fn phase(&self) -> Option<Complex64> {
        self.phase
    }

    /// Physical map phase acquired along the returned generator word.
    pub const fn physical_phase_to_representative(&self) -> Complex64 {
        self.physical_phase_to_representative
    }

    /// Generator indices applied in order to reach the representative.
    pub fn generator_word(&self) -> &[usize] {
        &self.generator_word
    }
}

#[derive(Clone, Copy, Debug)]
struct GeneralSymmetryImage<State> {
    representative: State,
    phase: Complex64,
    orbit_size: usize,
}

#[derive(Clone)]
struct SymmetryTrace<State> {
    coefficients: HashMap<State, Complex64>,
    physical_phases: HashMap<State, Complex64>,
    predecessors: HashMap<State, (State, usize)>,
    compatible: bool,
}

fn trace_symmetry_orbit<State>(
    state: State,
    generators: &[SymmetryGenerator<State>],
) -> Result<SymmetryTrace<State>>
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
    let mut physical_phases = HashMap::new();
    physical_phases.insert(state, Complex64::new(1.0, 0.0));
    let mut predecessors = HashMap::new();
    let mut queue = VecDeque::from([state]);
    let mut compatible = true;
    while let Some(source) = queue.pop_front() {
        let source_coefficient = coefficients[&source];
        let source_physical_phase = physical_phases[&source];
        for (generator_index, (generator, &sector_step)) in
            generators.iter().zip(&sector_steps).enumerate()
        {
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
                physical_phases.insert(target, source_physical_phase * map_phase);
                predecessors.insert(target, (source, generator_index));
                queue.push_back(target);
            }
        }
    }
    Ok(SymmetryTrace {
        coefficients,
        physical_phases,
        predecessors,
        compatible,
    })
}

impl<State> SymmetryReducer<State>
where
    State: Copy + Eq + Hash + Ord,
{
    fn trace(&self, state: State) -> Result<Arc<SymmetryTrace<State>>> {
        if let Some(trace) = self
            .orbit_cache
            .read()
            .map_err(|_| QmbedError::InternalState("symmetry orbit cache was poisoned".into()))?
            .get(&state)
            .cloned()
        {
            return Ok(trace);
        }
        let trace = Arc::new(trace_symmetry_orbit(state, &self.generators)?);
        self.orbit_cache
            .write()
            .map_err(|_| QmbedError::InternalState("symmetry orbit cache was poisoned".into()))?
            .entry(state)
            .or_insert_with(|| trace.clone());
        Ok(trace)
    }

    /// Enumerate the finite physical orbit of one state in canonical order.
    ///
    /// This stays independent of any materialized basis. Deferred operator
    /// paths use it to test whether an output orbit intersects the seed
    /// constraint, including when the canonical representative itself lies
    /// outside that constraint.
    pub fn orbit_states(&self, state: State) -> Result<Vec<State>> {
        Ok(self.orbit_with_states(state)?.1)
    }

    /// Reduce a state and return its physical orbit from one group traversal.
    ///
    /// Consumers that need both results should prefer this method over
    /// separate [`SymmetryReducer::orbit`] and
    /// [`SymmetryReducer::orbit_states`] calls.
    pub fn orbit_with_states_and_ordering(
        &self,
        state: State,
        ordering: RepresentativeOrdering,
    ) -> Result<(SymmetryOrbit<State>, Vec<State>)> {
        let trace = self.trace(state)?;
        let representative = *match ordering {
            RepresentativeOrdering::Minimum => trace.coefficients.keys().min(),
            RepresentativeOrdering::Maximum => trace.coefficients.keys().max(),
        }
        .ok_or_else(|| QmbedError::InternalState("a symmetry orbit contains no states".into()))?;
        let representative_coefficient = trace.coefficients[&representative];
        let phase = trace.compatible.then(|| {
            let gauge = representative_coefficient / representative_coefficient.norm();
            gauge.conj()
        });
        let mut generator_word = Vec::new();
        let mut cursor = representative;
        while cursor != state {
            let &(previous, generator) = trace.predecessors.get(&cursor).ok_or_else(|| {
                QmbedError::InternalState(
                    "a symmetry-orbit representative has no predecessor path".into(),
                )
            })?;
            generator_word.push(generator);
            cursor = previous;
        }
        generator_word.reverse();
        let mut states: Vec<_> = trace.coefficients.keys().copied().collect();
        states.sort_unstable();
        Ok((
            SymmetryOrbit {
                representative,
                orbit_size: trace.coefficients.len(),
                compatible: trace.compatible,
                phase,
                physical_phase_to_representative: trace.physical_phases[&representative],
                generator_word,
            },
            states,
        ))
    }

    pub fn orbit_with_states(&self, state: State) -> Result<(SymmetryOrbit<State>, Vec<State>)> {
        self.orbit_with_states_and_ordering(state, RepresentativeOrdering::Minimum)
    }

    /// Reduce one state using only the finite group action.
    ///
    /// This query does not enumerate a parent Hilbert space or construct the
    /// reduced basis. Incompatible character requests still return the
    /// canonical orbit representative, but have no reduced-vector phase.
    pub fn orbit(&self, state: State) -> Result<SymmetryOrbit<State>> {
        Ok(self.orbit_with_states(state)?.0)
    }

    /// Reduce one state using an explicit representative convention.
    pub fn orbit_with_ordering(
        &self,
        state: State,
        ordering: RepresentativeOrdering,
    ) -> Result<SymmetryOrbit<State>> {
        Ok(self.orbit_with_states_and_ordering(state, ordering)?.0)
    }
}

#[derive(Clone)]
struct MatrixSymmetryGenerator<State> {
    map: Arc<dyn SymmetryMap<State>>,
    representation: Vec<Complex64>,
}

/// Finite symmetry generators carrying a common unitary matrix
/// representation.
///
/// Scalar characters are the one-dimensional special case handled more
/// efficiently by [`SymmetryReducer`]. This type covers irreducible
/// representations of arbitrary finite dimension and constructs one selected
/// representation row without assuming that the generators commute.
#[derive(Clone)]
pub struct MatrixSymmetryReducer<State> {
    dimension: usize,
    selected_row: usize,
    generators: Vec<MatrixSymmetryGenerator<State>>,
}

impl<State> std::fmt::Debug for MatrixSymmetryReducer<State> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("MatrixSymmetryReducer")
            .field("dimension", &self.dimension)
            .field("selected_row", &self.selected_row)
            .field("generators", &self.generators.len())
            .finish()
    }
}

impl<State> MatrixSymmetryReducer<State> {
    pub fn new(dimension: usize, selected_row: usize) -> Result<Self> {
        if dimension == 0 || selected_row >= dimension {
            return Err(QmbedError::InvalidSector(
                "a matrix symmetry representation needs a positive dimension and valid row".into(),
            ));
        }
        Ok(Self {
            dimension,
            selected_row,
            generators: Vec::new(),
        })
    }

    pub const fn dimension(&self) -> usize {
        self.dimension
    }

    pub const fn selected_row(&self) -> usize {
        self.selected_row
    }

    pub fn with_map<M>(mut self, map: M, representation: impl Into<Vec<Complex64>>) -> Result<Self>
    where
        M: SymmetryMap<State> + 'static,
    {
        let representation = representation.into();
        let entries = self.dimension.checked_mul(self.dimension).ok_or_else(|| {
            QmbedError::UnsupportedBackend("matrix symmetry representation is too large".into())
        })?;
        if representation.len() != entries {
            return Err(QmbedError::InvalidSector(format!(
                "matrix symmetry generator has {} entries, expected {entries}",
                representation.len()
            )));
        }
        validate_unitary_matrix(&representation, self.dimension)?;
        let mut power = identity_matrix(self.dimension);
        for _ in 0..map.period() {
            power = multiply_square_matrices(&representation, &power, self.dimension);
        }
        if !matrices_close(
            &power,
            &identity_matrix(self.dimension),
            MATRIX_SYMMETRY_TOLERANCE,
        ) {
            return Err(QmbedError::IncompatibleSymmetry(
                "a representation generator does not close at the physical map period".into(),
            ));
        }
        self.generators.push(MatrixSymmetryGenerator {
            map: Arc::new(map),
            representation,
        });
        Ok(self)
    }
}

const MATRIX_SYMMETRY_TOLERANCE: f64 = 1.0e-10;
const MAX_MATRIX_SYMMETRY_GROUP_ORDER: usize = 65_536;

fn identity_matrix(dimension: usize) -> Vec<Complex64> {
    let mut matrix = vec![Complex64::new(0.0, 0.0); dimension * dimension];
    for index in 0..dimension {
        matrix[index * dimension + index] = Complex64::new(1.0, 0.0);
    }
    matrix
}

fn multiply_square_matrices(
    left: &[Complex64],
    right: &[Complex64],
    dimension: usize,
) -> Vec<Complex64> {
    let mut product = vec![Complex64::new(0.0, 0.0); dimension * dimension];
    for row in 0..dimension {
        for middle in 0..dimension {
            let coefficient = left[row * dimension + middle];
            if coefficient.norm() <= f64::EPSILON {
                continue;
            }
            for column in 0..dimension {
                product[row * dimension + column] +=
                    coefficient * right[middle * dimension + column];
            }
        }
    }
    product
}

fn matrices_close(left: &[Complex64], right: &[Complex64], tolerance: f64) -> bool {
    left.len() == right.len()
        && left
            .iter()
            .zip(right)
            .all(|(left, right)| (*left - *right).norm() <= tolerance)
}

fn validate_unitary_matrix(matrix: &[Complex64], dimension: usize) -> Result<()> {
    if matrix
        .iter()
        .any(|value| !value.re.is_finite() || !value.im.is_finite())
    {
        return Err(QmbedError::InvalidSector(
            "matrix symmetry generators must contain finite values".into(),
        ));
    }
    for left in 0..dimension {
        for right in 0..dimension {
            let overlap = (0..dimension).fold(Complex64::new(0.0, 0.0), |sum, row| {
                sum + matrix[row * dimension + left].conj() * matrix[row * dimension + right]
            });
            let expected = if left == right {
                Complex64::new(1.0, 0.0)
            } else {
                Complex64::new(0.0, 0.0)
            };
            if (overlap - expected).norm() > MATRIX_SYMMETRY_TOLERANCE {
                return Err(QmbedError::InvalidSector(
                    "matrix symmetry generators must be unitary".into(),
                ));
            }
        }
    }
    Ok(())
}

#[derive(Clone)]
struct OrbitGroupElement {
    destinations: Vec<usize>,
    phases: Vec<Complex64>,
    representation: Vec<Complex64>,
}

type OrbitColumn<State> = (State, Vec<(State, Complex64)>);

fn orbit_group_elements_close(left: &OrbitGroupElement, right: &OrbitGroupElement) -> bool {
    left.destinations == right.destinations
        && matrices_close(&left.phases, &right.phases, MATRIX_SYMMETRY_TOLERANCE)
        && matrices_close(
            &left.representation,
            &right.representation,
            MATRIX_SYMMETRY_TOLERANCE,
        )
}

fn compose_orbit_group_elements(
    left: &OrbitGroupElement,
    right: &OrbitGroupElement,
    representation_dimension: usize,
) -> OrbitGroupElement {
    let mut destinations = Vec::with_capacity(right.destinations.len());
    let mut phases = Vec::with_capacity(right.phases.len());
    for source in 0..right.destinations.len() {
        let middle = right.destinations[source];
        destinations.push(left.destinations[middle]);
        phases.push(right.phases[source] * left.phases[middle]);
    }
    OrbitGroupElement {
        destinations,
        phases,
        representation: multiply_square_matrices(
            &left.representation,
            &right.representation,
            representation_dimension,
        ),
    }
}

/// Orthonormal columns spanning one selected row of a finite-group matrix
/// representation.
#[derive(Clone, Debug)]
pub struct MatrixSymmetrySubspace<State> {
    physical_states: Vec<State>,
    labels: Vec<State>,
    columns: Vec<Vec<(State, Complex64)>>,
}

impl<State> MatrixSymmetrySubspace<State>
where
    State: Copy + Eq + Hash + Ord,
{
    pub fn dimension(&self) -> usize {
        self.columns.len()
    }

    pub fn physical_dimension(&self) -> usize {
        self.physical_states.len()
    }

    pub fn physical_states(&self) -> &[State] {
        &self.physical_states
    }

    /// Deterministic physical seed label attached to each orthonormal column.
    pub fn labels(&self) -> &[State] {
        &self.labels
    }

    pub fn columns(&self) -> &[Vec<(State, Complex64)>] {
        &self.columns
    }

    /// Embed the selected representation row into an explicit parent basis.
    pub fn projector<Parent>(&self, parent: &Parent, format: MatrixFormat) -> Result<Operator>
    where
        Parent: Basis<State = State>,
    {
        Operator::from_triplets(
            parent.len(),
            self.columns.len(),
            self.columns
                .iter()
                .enumerate()
                .flat_map(|(column, entries)| {
                    entries
                        .iter()
                        .map(move |&(state, value)| (state, column, value))
                })
                .map(|(state, column, value)| Ok((parent.index(state)?, column, value)))
                .collect::<Result<Vec<_>>>()?,
            format,
        )
    }
}

impl<State> MatrixSymmetryReducer<State>
where
    State: Copy + Eq + Hash + Ord,
{
    fn physical_orbit(&self, seed: State) -> Result<Vec<State>> {
        let mut visited = HashSet::from([seed]);
        let mut queue = VecDeque::from([seed]);
        while let Some(source) = queue.pop_front() {
            for generator in &self.generators {
                let (target, phase) = generator.map.apply(source)?;
                if !phase.re.is_finite() || !phase.im.is_finite() {
                    return Err(QmbedError::IncompatibleSymmetry(
                        "a symmetry map returned a non-finite phase".into(),
                    ));
                }
                if visited.insert(target) {
                    queue.push_back(target);
                }
            }
        }
        let mut orbit: Vec<_> = visited.into_iter().collect();
        orbit.sort_unstable();
        Ok(orbit)
    }

    fn orbit_group(&self, orbit: &[State]) -> Result<Vec<OrbitGroupElement>> {
        let indices: HashMap<_, _> = orbit
            .iter()
            .copied()
            .enumerate()
            .map(|(index, state)| (state, index))
            .collect();
        let mut generators = Vec::with_capacity(self.generators.len());
        for generator in &self.generators {
            let mut destinations = Vec::with_capacity(orbit.len());
            let mut phases = Vec::with_capacity(orbit.len());
            for &state in orbit {
                let (target, phase) = generator.map.apply(state)?;
                let target = indices.get(&target).copied().ok_or_else(|| {
                    QmbedError::IncompatibleSymmetry(
                        "a matrix symmetry generator leaves its physical orbit".into(),
                    )
                })?;
                destinations.push(target);
                phases.push(phase);
            }
            generators.push(OrbitGroupElement {
                destinations,
                phases,
                representation: generator.representation.clone(),
            });
        }

        let identity = OrbitGroupElement {
            destinations: (0..orbit.len()).collect(),
            phases: vec![Complex64::new(1.0, 0.0); orbit.len()],
            representation: identity_matrix(self.dimension),
        };
        let mut group = vec![identity];
        let mut cursor = 0;
        while cursor < group.len() {
            let current = group[cursor].clone();
            cursor += 1;
            for generator in &generators {
                let candidate = compose_orbit_group_elements(generator, &current, self.dimension);
                if group
                    .iter()
                    .any(|element| orbit_group_elements_close(element, &candidate))
                {
                    continue;
                }
                if group.len() >= MAX_MATRIX_SYMMETRY_GROUP_ORDER {
                    return Err(QmbedError::UnsupportedBackend(
                        "matrix symmetry group exceeds the finite closure limit".into(),
                    ));
                }
                group.push(candidate);
            }
        }
        Ok(group)
    }

    fn selected_orbit_columns(&self, orbit: &[State]) -> Result<Vec<OrbitColumn<State>>> {
        let group = self.orbit_group(orbit)?;
        let scale = self.dimension as f64 / group.len() as f64;
        let mut projector = vec![vec![Complex64::new(0.0, 0.0); orbit.len()]; orbit.len()];
        let diagonal = self.selected_row * self.dimension + self.selected_row;
        for element in &group {
            let weight = scale * element.representation[diagonal].conj();
            for source in 0..orbit.len() {
                projector[element.destinations[source]][source] += weight * element.phases[source];
            }
        }

        let mut orthonormal = Vec::<(State, Vec<Complex64>)>::new();
        for source in 0..orbit.len() {
            let mut vector: Vec<_> = projector.iter().map(|row| row[source]).collect();
            // A second modified-Gram-Schmidt pass keeps nearly dependent
            // projector columns stable on short symmetry orbits.
            for _ in 0..2 {
                for (_, previous) in &orthonormal {
                    let overlap = previous
                        .iter()
                        .zip(&vector)
                        .fold(Complex64::new(0.0, 0.0), |sum, (left, right)| {
                            sum + left.conj() * right
                        });
                    for (value, basis_value) in vector.iter_mut().zip(previous) {
                        *value -= overlap * basis_value;
                    }
                }
            }
            let norm = vector.iter().map(Complex64::norm_sqr).sum::<f64>().sqrt();
            if norm <= MATRIX_SYMMETRY_TOLERANCE {
                continue;
            }
            vector.iter_mut().for_each(|value| *value /= norm);
            if let Some(pivot) = vector
                .iter()
                .copied()
                .find(|value| value.norm() > MATRIX_SYMMETRY_TOLERANCE)
            {
                let gauge = pivot / pivot.norm();
                vector.iter_mut().for_each(|value| *value *= gauge.conj());
            }
            orthonormal.push((orbit[source], vector));
        }

        Ok(orthonormal
            .into_iter()
            .map(|(label, vector)| {
                (
                    label,
                    orbit
                        .iter()
                        .copied()
                        .zip(vector)
                        .filter(|(_, value)| value.norm() > MATRIX_SYMMETRY_TOLERANCE)
                        .collect(),
                )
            })
            .collect())
    }

    /// Build the selected representation row from constrained seed states.
    ///
    /// Each seed orbit is completed before projection, so generators may
    /// exchange additive sectors without forcing an unrestricted seed
    /// enumeration.
    pub fn subspace<Seed>(&self, seeds: &Seed) -> Result<MatrixSymmetrySubspace<State>>
    where
        Seed: Basis<State = State>,
    {
        if self.generators.is_empty() {
            return Err(QmbedError::InvalidSector(
                "a matrix symmetry representation requires at least one generator".into(),
            ));
        }
        let mut visited = HashSet::new();
        let mut physical_states = Vec::new();
        let mut labels = Vec::new();
        let mut columns = Vec::new();
        for index in 0..seeds.len() {
            let seed = seeds.state(index)?;
            if visited.contains(&seed) {
                continue;
            }
            let orbit = self.physical_orbit(seed)?;
            visited.extend(orbit.iter().copied());
            physical_states.extend(orbit.iter().copied());
            for (label, column) in self.selected_orbit_columns(&orbit)? {
                labels.push(label);
                columns.push(column);
            }
        }
        physical_states.sort_unstable();
        physical_states.dedup();
        Ok(MatrixSymmetrySubspace {
            physical_states,
            labels,
            columns,
        })
    }
}

/// Arbitrary finite-map reduction of any concrete parent basis.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum RepresentativeOrdering {
    #[default]
    Minimum,
    Maximum,
}

#[derive(Clone, Debug)]
pub struct GeneralBasis<Parent>
where
    Parent: Basis,
    Parent::State: Hash + Ord,
{
    parent: Parent,
    reducer: SymmetryReducer<Parent::State>,
    states: Vec<Parent::State>,
    indices: HashMap<Parent::State, usize>,
    orbit_lengths: Vec<usize>,
    lookup: HashMap<Parent::State, GeneralSymmetryImage<Parent::State>>,
}

impl<Parent> GeneralBasis<Parent>
where
    Parent: Basis,
    Parent::State: Hash + Ord,
{
    pub fn new(parent: Parent, sector: SymmetrySector<Parent::State>) -> Result<Self> {
        Self::from_reducer(parent, sector)
    }

    /// Materialize a reduced basis from an independently reusable reducer.
    pub fn from_reducer(parent: Parent, reducer: SymmetryReducer<Parent::State>) -> Result<Self> {
        Self::from_reducer_with_ordering(parent, reducer, RepresentativeOrdering::Minimum)
    }

    /// Materialize a reduced basis with an explicit canonical representative
    /// convention. The native default is the minimum state; compatibility
    /// frontends with descending basis order may select the maximum state
    /// without changing the physical symmetry subspace.
    pub fn from_reducer_with_ordering(
        parent: Parent,
        reducer: SymmetryReducer<Parent::State>,
        ordering: RepresentativeOrdering,
    ) -> Result<Self> {
        Self::from_reducer_impl(parent, reducer, true, ordering)
    }

    /// Materialize symmetry vectors whose seed states come from `parent`.
    ///
    /// Unlike [`GeneralBasis::from_reducer`], the finite symmetry orbit may
    /// contain states outside the parent's enumerated constraint. The parent
    /// still supplies the local operator algebra, while its state list selects
    /// which physical orbits enter the reduced space. This is the natural
    /// construction for particle-hole or species-exchange symmetries acting
    /// on a fixed additive sector: one seed sector selects an orbit, and the
    /// normalized reduced vector contains the complete symmetry orbit.
    ///
    /// The parent's local action must therefore be defined for every state
    /// generated by the reducer. Built-in packed bases and callback bases
    /// satisfy this contract because local actions operate on the encoded
    /// state rather than its row index.
    pub fn from_orbit_seeds(
        parent: Parent,
        reducer: SymmetryReducer<Parent::State>,
    ) -> Result<Self> {
        Self::from_orbit_seeds_with_ordering(parent, reducer, RepresentativeOrdering::Minimum)
    }

    pub fn from_orbit_seeds_with_ordering(
        parent: Parent,
        reducer: SymmetryReducer<Parent::State>,
        ordering: RepresentativeOrdering,
    ) -> Result<Self> {
        Self::from_reducer_impl(parent, reducer, false, ordering)
    }

    fn from_reducer_impl(
        parent: Parent,
        reducer: SymmetryReducer<Parent::State>,
        require_parent_membership: bool,
        ordering: RepresentativeOrdering,
    ) -> Result<Self> {
        if reducer.generators.is_empty() {
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
            let indices = states
                .iter()
                .copied()
                .enumerate()
                .map(|(index, state)| (state, index))
                .collect();
            return Ok(Self {
                parent,
                reducer,
                orbit_lengths: vec![1; states.len()],
                states,
                indices,
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
            let trace = reducer.trace(seed)?;
            for state in trace.coefficients.keys() {
                if require_parent_membership {
                    parent.index(*state).map_err(|_| {
                        QmbedError::IncompatibleSymmetry(
                            "a symmetry map leaves the parent basis".into(),
                        )
                    })?;
                }
                visited.insert(*state);
            }
            if !trace.compatible {
                continue;
            }
            let representative = *match ordering {
                RepresentativeOrdering::Minimum => trace.coefficients.keys().min(),
                RepresentativeOrdering::Maximum => trace.coefficients.keys().max(),
            }
            .ok_or_else(|| {
                QmbedError::InvalidSector("symmetry projection generated no state".into())
            })?;
            let representative_coefficient = trace.coefficients[&representative];
            let gauge = representative_coefficient / representative_coefficient.norm();
            let norm = trace
                .coefficients
                .values()
                .map(Complex64::norm_sqr)
                .sum::<f64>()
                .sqrt();
            let orbit_size = trace.coefficients.len();
            let expected_magnitude = 1.0 / (orbit_size as f64).sqrt();
            for (&state, &coefficient) in &trace.coefficients {
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
        // A basis row is an ordered coordinate, not merely a packed integer.
        // Preserve the parent basis' public row convention whenever every
        // selected representative belongs to it. This matters for composite
        // encodings such as spinful fermions, whose native integer layout is
        // intentionally different from their `up ⊗ down` row order.
        let parent_rows = representatives
            .iter()
            .map(|(state, _)| parent.index(*state).ok())
            .collect::<Vec<_>>();
        if parent_rows.iter().all(Option::is_some) {
            let mut with_rows = representatives
                .into_iter()
                .zip(parent_rows)
                .map(|((state, orbit_size), row)| (row.unwrap(), state, orbit_size))
                .collect::<Vec<_>>();
            with_rows.sort_by_key(|(row, _, _)| *row);
            representatives = with_rows
                .into_iter()
                .map(|(_, state, orbit_size)| (state, orbit_size))
                .collect();
        } else {
            // Orbit-seed reductions may deliberately choose a representative
            // outside the enumerated parent sector. There is then no parent
            // row to inherit, so retain the state type's canonical ordering.
            representatives.sort_by_key(|(state, _)| *state);
        }
        let (states, orbit_lengths): (Vec<Parent::State>, Vec<usize>) =
            representatives.into_iter().unzip();
        let indices = states
            .iter()
            .copied()
            .enumerate()
            .map(|(index, state)| (state, index))
            .collect();
        Ok(Self {
            parent,
            reducer,
            states,
            indices,
            orbit_lengths,
            lookup,
        })
    }

    pub fn parent(&self) -> &Parent {
        &self.parent
    }

    pub const fn reducer(&self) -> &SymmetryReducer<Parent::State> {
        &self.reducer
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

    fn operator_preserves_particle_sector_on_sites(
        &self,
        operator: &str,
        sites: &[usize],
    ) -> Result<bool> {
        self.parent
            .operator_preserves_particle_sector_on_sites(operator, sites)
    }
}

pub type SpinBasisGeneral = GeneralBasis<SpinBasis1D>;
pub type BosonBasisGeneral = GeneralBasis<BosonBasis1D>;
pub type SpinlessFermionBasisGeneral = GeneralBasis<SpinlessFermionBasis1D>;
pub type SpinfulFermionBasisGeneral = GeneralBasis<SpinfulFermionBasis1D>;
pub type UserBasisGeneral = GeneralBasis<UserBasis<u128>>;

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

impl StateStorage {
    pub fn for_bits(bits: usize) -> Result<Self> {
        match bits {
            1..=128 => Ok(Self::U128),
            129..=256 => Ok(Self::U256),
            257..=1024 => Ok(Self::U1024),
            1025..=4096 => Ok(Self::U4096),
            4097..=16384 => Ok(Self::U16384),
            0 => Err(QmbedError::InvalidSector(
                "a basis state needs at least one bit".into(),
            )),
            _ => Err(QmbedError::UnsupportedBackend(
                "basis state requires more than 16384 bits".into(),
            )),
        }
    }

    pub const fn capacity_bits(self) -> usize {
        match self {
            Self::U128 => 128,
            Self::U256 => 256,
            Self::U1024 => 1024,
            Self::U4096 => 4096,
            Self::U16384 => 16384,
        }
    }
}

/// Fixed-width state used by wide user and general bases.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct WideState<const WORDS: usize> {
    words: [u64; WORDS],
}

impl<const WORDS: usize> Ord for WideState<WORDS> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.words.iter().rev().cmp(other.words.iter().rev())
    }
}

impl<const WORDS: usize> PartialOrd for WideState<WORDS> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
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

    pub fn count_ones_after(&self, index: usize) -> usize {
        if index + 1 >= Self::capacity_bits() {
            return 0;
        }
        let first = (index + 1) / 64;
        let offset = (index + 1) % 64;
        let mut count = 0_usize;
        if offset != 0 {
            count += (self.words[first] & (!0_u64 << offset)).count_ones() as usize;
        } else {
            count += self.words[first].count_ones() as usize;
        }
        count
            + self.words[first + 1..]
                .iter()
                .map(|word| word.count_ones() as usize)
                .sum::<usize>()
    }

    pub fn has_bits_at_or_above(&self, index: usize) -> bool {
        if index >= Self::capacity_bits() {
            return false;
        }
        let first = index / 64;
        let offset = index % 64;
        let first_mask = !0_u64 << offset;
        self.words[first] & first_mask != 0 || self.words[first + 1..].iter().any(|word| *word != 0)
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

/// Runtime-selected fixed-width basis state.
///
/// This is the type-erased state boundary used by language bindings and
/// runtime-selected bases. Arithmetic remains delegated to the same concrete
/// fixed-width implementations used by native Rust callers.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ErasedState {
    width_bits: usize,
    value: ErasedStateValue,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[allow(clippy::large_enum_variant)] // Fixed-width values stay Copy and allocation-free at the FFI boundary.
enum ErasedStateValue {
    U128(u128),
    U256(U256),
    U1024(U1024),
    U4096(U4096),
    U16384(U16384),
}

impl Ord for ErasedState {
    fn cmp(&self, other: &Self) -> Ordering {
        self.width_bits
            .cmp(&other.width_bits)
            .then_with(|| match (&self.value, &other.value) {
                (ErasedStateValue::U128(left), ErasedStateValue::U128(right)) => left.cmp(right),
                (ErasedStateValue::U256(left), ErasedStateValue::U256(right)) => left.cmp(right),
                (ErasedStateValue::U1024(left), ErasedStateValue::U1024(right)) => left.cmp(right),
                (ErasedStateValue::U4096(left), ErasedStateValue::U4096(right)) => left.cmp(right),
                (ErasedStateValue::U16384(left), ErasedStateValue::U16384(right)) => {
                    left.cmp(right)
                }
                (left, right) => erased_state_value_rank(left).cmp(&erased_state_value_rank(right)),
            })
    }
}

impl PartialOrd for ErasedState {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl std::fmt::Display for ErasedState {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.to_decimal())
    }
}

const fn erased_state_value_rank(value: &ErasedStateValue) -> u8 {
    match value {
        ErasedStateValue::U128(_) => 0,
        ErasedStateValue::U256(_) => 1,
        ErasedStateValue::U1024(_) => 2,
        ErasedStateValue::U4096(_) => 3,
        ErasedStateValue::U16384(_) => 4,
    }
}

trait FixedWidthState: Copy {
    const STORAGE: StateStorage;

    fn into_erased_value(self) -> ErasedStateValue;
    fn from_erased_value(value: ErasedStateValue) -> Option<Self>;
}

macro_rules! impl_fixed_width_state {
    ($state:ty, $storage:ident, $variant:ident) => {
        impl FixedWidthState for $state {
            const STORAGE: StateStorage = StateStorage::$storage;

            fn into_erased_value(self) -> ErasedStateValue {
                ErasedStateValue::$variant(self)
            }

            fn from_erased_value(value: ErasedStateValue) -> Option<Self> {
                match value {
                    ErasedStateValue::$variant(value) => Some(value),
                    _ => None,
                }
            }
        }
    };
}

impl_fixed_width_state!(U256, U256, U256);
impl_fixed_width_state!(U1024, U1024, U1024);
impl_fixed_width_state!(U4096, U4096, U4096);
impl_fixed_width_state!(U16384, U16384, U16384);

fn erase_fixed_width_state<State: FixedWidthState>(width_bits: usize, state: State) -> ErasedState {
    ErasedState {
        width_bits,
        value: state.into_erased_value(),
    }
}

fn restore_fixed_width_state<State: FixedWidthState>(
    state: ErasedState,
    width_bits: usize,
) -> Result<State> {
    if state.width_bits != width_bits || state.storage() != State::STORAGE {
        return Err(QmbedError::StateNotInBasis);
    }
    State::from_erased_value(state.value).ok_or(QmbedError::StateNotInBasis)
}

impl ErasedState {
    pub fn from_decimal(width_bits: usize, value: &str) -> Result<Self> {
        let value = BigUint::parse_bytes(value.as_bytes(), 10).ok_or_else(|| {
            QmbedError::InvalidOptions(format!(
                "basis state {value:?} is not a nonnegative decimal integer"
            ))
        })?;
        Self::from_biguint(width_bits, &value)
    }

    pub fn from_biguint(width_bits: usize, value: &BigUint) -> Result<Self> {
        if value.bits() > u64::try_from(width_bits).unwrap_or(u64::MAX) {
            return Err(QmbedError::UnsupportedBackend(format!(
                "integer needs {} bits but the requested state width is {width_bits}",
                value.bits()
            )));
        }
        let storage = StateStorage::for_bits(width_bits)?;
        let value = match storage {
            StateStorage::U128 => {
                let digits = value.to_u64_digits();
                let low = u128::from(digits.first().copied().unwrap_or_default());
                let high = u128::from(digits.get(1).copied().unwrap_or_default()) << 64;
                ErasedStateValue::U128(low | high)
            }
            StateStorage::U256 => ErasedStateValue::U256(state_from_biguint(value)?),
            StateStorage::U1024 => ErasedStateValue::U1024(state_from_biguint(value)?),
            StateStorage::U4096 => ErasedStateValue::U4096(state_from_biguint(value)?),
            StateStorage::U16384 => ErasedStateValue::U16384(state_from_biguint(value)?),
        };
        Ok(Self { width_bits, value })
    }

    pub const fn width_bits(&self) -> usize {
        self.width_bits
    }

    pub const fn storage(&self) -> StateStorage {
        match self.value {
            ErasedStateValue::U128(_) => StateStorage::U128,
            ErasedStateValue::U256(_) => StateStorage::U256,
            ErasedStateValue::U1024(_) => StateStorage::U1024,
            ErasedStateValue::U4096(_) => StateStorage::U4096,
            ErasedStateValue::U16384(_) => StateStorage::U16384,
        }
    }

    pub fn to_biguint(&self) -> BigUint {
        match &self.value {
            ErasedStateValue::U128(value) => BigUint::from(*value),
            ErasedStateValue::U256(value) => state_to_biguint(*value),
            ErasedStateValue::U1024(value) => state_to_biguint(*value),
            ErasedStateValue::U4096(value) => state_to_biguint(*value),
            ErasedStateValue::U16384(value) => state_to_biguint(*value),
        }
    }

    pub fn to_decimal(&self) -> String {
        self.to_biguint().to_str_radix(10)
    }

    fn ensure_compatible(&self, right: &Self) -> Result<()> {
        if self.width_bits != right.width_bits || self.storage() != right.storage() {
            return Err(QmbedError::DimensionMismatch(
                "bitwise basis states must have the same logical width".into(),
            ));
        }
        Ok(())
    }

    fn remask(self) -> Result<Self> {
        let mask = (BigUint::from(1_u8) << self.width_bits) - BigUint::from(1_u8);
        Self::from_biguint(self.width_bits, &(self.to_biguint() & mask))
    }

    pub fn bitwise_and(&self, right: &Self) -> Result<Self> {
        self.ensure_compatible(right)?;
        let value = match (&self.value, &right.value) {
            (ErasedStateValue::U128(left), ErasedStateValue::U128(right)) => {
                ErasedStateValue::U128(left & right)
            }
            (ErasedStateValue::U256(left), ErasedStateValue::U256(right)) => {
                ErasedStateValue::U256(left.bitwise_and(*right))
            }
            (ErasedStateValue::U1024(left), ErasedStateValue::U1024(right)) => {
                ErasedStateValue::U1024(left.bitwise_and(*right))
            }
            (ErasedStateValue::U4096(left), ErasedStateValue::U4096(right)) => {
                ErasedStateValue::U4096(left.bitwise_and(*right))
            }
            (ErasedStateValue::U16384(left), ErasedStateValue::U16384(right)) => {
                ErasedStateValue::U16384(left.bitwise_and(*right))
            }
            _ => unreachable!("compatible erased states have matching storage"),
        };
        Ok(Self {
            width_bits: self.width_bits,
            value,
        })
    }

    pub fn bitwise_or(&self, right: &Self) -> Result<Self> {
        self.ensure_compatible(right)?;
        let value = match (&self.value, &right.value) {
            (ErasedStateValue::U128(left), ErasedStateValue::U128(right)) => {
                ErasedStateValue::U128(left | right)
            }
            (ErasedStateValue::U256(left), ErasedStateValue::U256(right)) => {
                ErasedStateValue::U256(left.bitwise_or(*right))
            }
            (ErasedStateValue::U1024(left), ErasedStateValue::U1024(right)) => {
                ErasedStateValue::U1024(left.bitwise_or(*right))
            }
            (ErasedStateValue::U4096(left), ErasedStateValue::U4096(right)) => {
                ErasedStateValue::U4096(left.bitwise_or(*right))
            }
            (ErasedStateValue::U16384(left), ErasedStateValue::U16384(right)) => {
                ErasedStateValue::U16384(left.bitwise_or(*right))
            }
            _ => unreachable!("compatible erased states have matching storage"),
        };
        Ok(Self {
            width_bits: self.width_bits,
            value,
        })
    }

    pub fn bitwise_xor(&self, right: &Self) -> Result<Self> {
        self.ensure_compatible(right)?;
        let value = match (&self.value, &right.value) {
            (ErasedStateValue::U128(left), ErasedStateValue::U128(right)) => {
                ErasedStateValue::U128(left ^ right)
            }
            (ErasedStateValue::U256(left), ErasedStateValue::U256(right)) => {
                ErasedStateValue::U256(left.bitwise_xor(*right))
            }
            (ErasedStateValue::U1024(left), ErasedStateValue::U1024(right)) => {
                ErasedStateValue::U1024(left.bitwise_xor(*right))
            }
            (ErasedStateValue::U4096(left), ErasedStateValue::U4096(right)) => {
                ErasedStateValue::U4096(left.bitwise_xor(*right))
            }
            (ErasedStateValue::U16384(left), ErasedStateValue::U16384(right)) => {
                ErasedStateValue::U16384(left.bitwise_xor(*right))
            }
            _ => unreachable!("compatible erased states have matching storage"),
        };
        Ok(Self {
            width_bits: self.width_bits,
            value,
        })
    }

    pub fn bitwise_not(&self) -> Result<Self> {
        let value = match &self.value {
            ErasedStateValue::U128(value) => ErasedStateValue::U128(!value),
            ErasedStateValue::U256(value) => ErasedStateValue::U256(value.bitwise_not()),
            ErasedStateValue::U1024(value) => ErasedStateValue::U1024(value.bitwise_not()),
            ErasedStateValue::U4096(value) => ErasedStateValue::U4096(value.bitwise_not()),
            ErasedStateValue::U16384(value) => ErasedStateValue::U16384(value.bitwise_not()),
        };
        Self {
            width_bits: self.width_bits,
            value,
        }
        .remask()
    }

    pub fn left_shift(&self, shift: usize) -> Result<Self> {
        let value = match &self.value {
            ErasedStateValue::U128(value) => ErasedStateValue::U128(
                u32::try_from(shift)
                    .ok()
                    .and_then(|shift| value.checked_shl(shift))
                    .unwrap_or_default(),
            ),
            ErasedStateValue::U256(value) => ErasedStateValue::U256(value.left_shift(shift)),
            ErasedStateValue::U1024(value) => ErasedStateValue::U1024(value.left_shift(shift)),
            ErasedStateValue::U4096(value) => ErasedStateValue::U4096(value.left_shift(shift)),
            ErasedStateValue::U16384(value) => ErasedStateValue::U16384(value.left_shift(shift)),
        };
        Self {
            width_bits: self.width_bits,
            value,
        }
        .remask()
    }

    pub fn right_shift(&self, shift: usize) -> Self {
        let value = match &self.value {
            ErasedStateValue::U128(value) => ErasedStateValue::U128(
                u32::try_from(shift)
                    .ok()
                    .and_then(|shift| value.checked_shr(shift))
                    .unwrap_or_default(),
            ),
            ErasedStateValue::U256(value) => ErasedStateValue::U256(value.right_shift(shift)),
            ErasedStateValue::U1024(value) => ErasedStateValue::U1024(value.right_shift(shift)),
            ErasedStateValue::U4096(value) => ErasedStateValue::U4096(value.right_shift(shift)),
            ErasedStateValue::U16384(value) => ErasedStateValue::U16384(value.right_shift(shift)),
        };
        Self {
            width_bits: self.width_bits,
            value,
        }
    }
}

/// Spin-half basis backed by a fixed-width state, including sites above 127.
#[derive(Clone, Debug)]
pub struct WideSpinBasis<const WORDS: usize> {
    sites: usize,
    particle_sectors: Option<Vec<usize>>,
    normalization: SpinNormalization,
    states: Vec<WideState<WORDS>>,
}

impl<const WORDS: usize> WideSpinBasis<WORDS> {
    pub fn new(sites: usize, particles: Option<usize>, pauli: bool) -> Result<Self> {
        let normalization = if pauli {
            SpinNormalization::PauliCartesian
        } else {
            SpinNormalization::AngularMomentum
        };
        match particles {
            Some(particles) => {
                Self::from_particle_sectors_with_normalization(sites, [particles], normalization)
            }
            None => Self::from_optional_particle_sectors(sites, None, normalization),
        }
    }

    /// Construct a wide basis from a nonempty union of fixed-particle sectors.
    pub fn from_particle_sectors(
        sites: usize,
        sectors: impl IntoIterator<Item = usize>,
        pauli: bool,
    ) -> Result<Self> {
        let normalization = if pauli {
            SpinNormalization::PauliCartesian
        } else {
            SpinNormalization::AngularMomentum
        };
        Self::from_particle_sectors_with_normalization(sites, sectors, normalization)
    }

    pub fn with_normalization(
        sites: usize,
        particles: Option<usize>,
        normalization: SpinNormalization,
    ) -> Result<Self> {
        match particles {
            Some(particles) => {
                Self::from_particle_sectors_with_normalization(sites, [particles], normalization)
            }
            None => Self::from_optional_particle_sectors(sites, None, normalization),
        }
    }

    pub fn from_particle_sectors_with_normalization(
        sites: usize,
        sectors: impl IntoIterator<Item = usize>,
        normalization: SpinNormalization,
    ) -> Result<Self> {
        let mut sectors: Vec<_> = sectors.into_iter().collect();
        sectors.sort_unstable();
        sectors.dedup();
        if sectors.is_empty() {
            return Err(QmbedError::InvalidSector(
                "wide spin sector union must be nonempty".into(),
            ));
        }
        Self::from_optional_particle_sectors(sites, Some(sectors), normalization)
    }

    /// Construct the same spin algebra on an explicitly selected physical
    /// state set.
    ///
    /// This is useful for finite-group orbit completions whose physical span
    /// is enumerable even when the unrestricted `2^sites` parent is not. The
    /// selected states remain an ordinary basis: local actions leaving the set
    /// are rejected by assembly in exactly the same way as any sector basis.
    pub fn from_explicit_states_with_normalization(
        sites: usize,
        states: impl IntoIterator<Item = WideState<WORDS>>,
        normalization: SpinNormalization,
    ) -> Result<Self> {
        if sites == 0 || sites > WideState::<WORDS>::capacity_bits() {
            return Err(QmbedError::UnsupportedBackend(format!(
                "wide spin basis needs 1..={} sites",
                WideState::<WORDS>::capacity_bits()
            )));
        }
        let mut states: Vec<_> = states.into_iter().collect();
        if states.iter().any(|state| state.has_bits_at_or_above(sites)) {
            return Err(QmbedError::InvalidSector(
                "an explicit wide spin state exceeds the requested site width".into(),
            ));
        }
        states.sort_unstable();
        states.dedup();
        let mut particle_sectors: Vec<_> = states.iter().map(WideState::count_ones).collect();
        particle_sectors.sort_unstable();
        particle_sectors.dedup();
        Ok(Self {
            sites,
            particle_sectors: Some(particle_sectors),
            normalization,
            states,
        })
    }

    fn from_optional_particle_sectors(
        sites: usize,
        particle_sectors: Option<Vec<usize>>,
        normalization: SpinNormalization,
    ) -> Result<Self> {
        if sites == 0 || sites > WideState::<WORDS>::capacity_bits() {
            return Err(QmbedError::UnsupportedBackend(format!(
                "wide spin basis needs 1..={} sites",
                WideState::<WORDS>::capacity_bits()
            )));
        }
        if particle_sectors
            .as_ref()
            .is_some_and(|sectors| sectors.iter().any(|&count| count > sites))
        {
            return Err(QmbedError::InvalidSector(
                "particle count exceeds the wide spin site count".into(),
            ));
        }
        let mut states = Vec::new();
        if let Some(sectors) = &particle_sectors {
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
            for &count in sectors {
                enumerate(0, sites, count, WideState::zero(), &mut states)?;
            }
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
            particle_sectors,
            normalization,
            states,
        })
    }

    pub const fn sites(&self) -> usize {
        self.sites
    }

    pub fn particles(&self) -> Option<usize> {
        match self.particle_sectors.as_deref() {
            Some([particles]) => Some(*particles),
            _ => None,
        }
    }

    pub fn particle_sectors(&self) -> Option<&[usize]> {
        self.particle_sectors.as_deref()
    }

    pub const fn normalization(&self) -> SpinNormalization {
        self.normalization
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
            let cartesian_scale = match self.normalization {
                SpinNormalization::AngularMomentum | SpinNormalization::Pauli => 0.5,
                SpinNormalization::PauliCartesian => 1.0,
            };
            let z_scale = if self.normalization == SpinNormalization::AngularMomentum {
                0.5
            } else {
                1.0
            };
            let ladder_scale = if self.normalization == SpinNormalization::Pauli {
                2.0
            } else {
                1.0
            };
            match op {
                'I' => {}
                'n' => {
                    if !occupied {
                        return Ok(None);
                    }
                }
                'z' => amplitude *= if occupied { z_scale } else { -z_scale },
                '+' if !occupied => {
                    state = state.with_bit(site, true)?;
                    amplitude *= ladder_scale;
                }
                '-' if occupied => {
                    state = state.with_bit(site, false)?;
                    amplitude *= ladder_scale;
                }
                'x' => {
                    state = state.with_bit(site, !occupied)?;
                    amplitude *= cartesian_scale;
                }
                'y' => {
                    state = state.with_bit(site, !occupied)?;
                    amplitude *= Complex64::new(
                        0.0,
                        if occupied {
                            cartesian_scale
                        } else {
                            -cartesian_scale
                        },
                    );
                }
                '+' | '-' => return Ok(None),
                _ => return Err(QmbedError::InvalidOperator(op.to_string())),
            }
        }
        Ok(Some((state, amplitude)))
    }

    fn operator_preserves_particle_sector(&self, operator: &str) -> Result<bool> {
        Ok(self.particle_sectors.is_none() || operator_number_change(operator)? == Some(0))
    }
}

pub type WideSpinBasis256 = WideSpinBasis<4>;
pub type WideSpinBasis1024 = WideSpinBasis<16>;
pub type WideSpinBasis4096 = WideSpinBasis<64>;
pub type WideSpinBasis16384 = WideSpinBasis<256>;
pub type WideSpinBasisGeneral256 = GeneralBasis<WideSpinBasis256>;
pub type WideSpinBasisGeneral1024 = GeneralBasis<WideSpinBasis1024>;
pub type WideSpinBasisGeneral4096 = GeneralBasis<WideSpinBasis4096>;
pub type WideSpinBasisGeneral16384 = GeneralBasis<WideSpinBasis16384>;

/// Runtime-selected wide spin basis used by persistent language-neutral models.
///
/// The enum erases only the fixed-width state storage. Local operator
/// semantics, symmetry reduction, and universal assembly remain implemented
/// by the same typed [`WideSpinBasis`] and [`GeneralBasis`] kernels used by
/// native Rust callers.
#[derive(Clone, Debug)]
pub enum WidePackedBasis {
    Spin256(WideSpinBasis256),
    Spin1024(WideSpinBasis1024),
    Spin4096(WideSpinBasis4096),
    Spin16384(WideSpinBasis16384),
    GeneralSpin256(WideSpinBasisGeneral256),
    GeneralSpin1024(WideSpinBasisGeneral1024),
    GeneralSpin4096(WideSpinBasisGeneral4096),
    GeneralSpin16384(WideSpinBasisGeneral16384),
    Reversed(Box<WidePackedBasis>),
}

impl WidePackedBasis {
    pub fn reversed(self) -> Self {
        match self {
            Self::Reversed(inner) => *inner,
            basis => Self::Reversed(Box::new(basis)),
        }
    }

    /// Reuse this runtime-selected spin algebra on an explicitly selected set
    /// of physical states while preserving its public ordering convention.
    pub fn explicit_spin_subspace(&self, states: &[ErasedState]) -> Result<Self> {
        macro_rules! explicit {
            ($basis:expr, $state:ty, $variant:ident) => {{
                let basis = $basis;
                let typed = states
                    .iter()
                    .copied()
                    .map(|state| restore_fixed_width_state::<$state>(state, basis.sites()))
                    .collect::<Result<Vec<_>>>()?;
                Ok(Self::$variant(
                    WideSpinBasis::from_explicit_states_with_normalization(
                        basis.sites(),
                        typed,
                        basis.normalization(),
                    )?,
                ))
            }};
        }
        match self {
            Self::Spin256(basis) => explicit!(basis, U256, Spin256),
            Self::Spin1024(basis) => explicit!(basis, U1024, Spin1024),
            Self::Spin4096(basis) => explicit!(basis, U4096, Spin4096),
            Self::Spin16384(basis) => explicit!(basis, U16384, Spin16384),
            Self::Reversed(inner) => Ok(inner.explicit_spin_subspace(states)?.reversed()),
            Self::GeneralSpin256(_)
            | Self::GeneralSpin1024(_)
            | Self::GeneralSpin4096(_)
            | Self::GeneralSpin16384(_) => Err(QmbedError::InvalidOptions(
                "an explicit wide spin subspace must start from an unreduced spin algebra".into(),
            )),
        }
    }

    pub fn width_bits(&self) -> usize {
        match self {
            Self::Spin256(basis) => basis.sites(),
            Self::Spin1024(basis) => basis.sites(),
            Self::Spin4096(basis) => basis.sites(),
            Self::Spin16384(basis) => basis.sites(),
            Self::GeneralSpin256(basis) => basis.parent().sites(),
            Self::GeneralSpin1024(basis) => basis.parent().sites(),
            Self::GeneralSpin4096(basis) => basis.parent().sites(),
            Self::GeneralSpin16384(basis) => basis.parent().sites(),
            Self::Reversed(basis) => basis.width_bits(),
        }
    }

    pub fn storage(&self) -> StateStorage {
        match self {
            Self::Spin256(_) | Self::GeneralSpin256(_) => StateStorage::U256,
            Self::Spin1024(_) | Self::GeneralSpin1024(_) => StateStorage::U1024,
            Self::Spin4096(_) | Self::GeneralSpin4096(_) => StateStorage::U4096,
            Self::Spin16384(_) | Self::GeneralSpin16384(_) => StateStorage::U16384,
            Self::Reversed(basis) => basis.storage(),
        }
    }
}

macro_rules! dispatch_wide_basis {
    ($basis:expr, $inner:ident => $body:expr) => {
        match $basis {
            WidePackedBasis::Spin256($inner) => $body,
            WidePackedBasis::Spin1024($inner) => $body,
            WidePackedBasis::Spin4096($inner) => $body,
            WidePackedBasis::Spin16384($inner) => $body,
            WidePackedBasis::GeneralSpin256($inner) => $body,
            WidePackedBasis::GeneralSpin1024($inner) => $body,
            WidePackedBasis::GeneralSpin4096($inner) => $body,
            WidePackedBasis::GeneralSpin16384($inner) => $body,
            WidePackedBasis::Reversed(_) => {
                unreachable!("reversed wide bases are handled before typed dispatch")
            }
        }
    };
}

impl Basis for WidePackedBasis {
    type State = ErasedState;

    fn len(&self) -> usize {
        if let Self::Reversed(basis) = self {
            return basis.len();
        }
        dispatch_wide_basis!(self, basis => basis.len())
    }

    fn state(&self, index: usize) -> Result<Self::State> {
        if let Self::Reversed(basis) = self {
            let reversed = basis
                .len()
                .checked_sub(index + 1)
                .ok_or(QmbedError::StateNotInBasis)?;
            return basis.state(reversed);
        }
        let width_bits = self.width_bits();
        dispatch_wide_basis!(
            self,
            basis => basis
                .state(index)
                .map(|state| erase_fixed_width_state(width_bits, state))
        )
    }

    fn index(&self, state: Self::State) -> Result<usize> {
        if let Self::Reversed(basis) = self {
            return basis.index(state).map(|index| basis.len() - index - 1);
        }
        let width_bits = self.width_bits();
        dispatch_wide_basis!(
            self,
            basis => basis.index(restore_fixed_width_state(state, width_bits)?)
        )
    }

    fn apply_local(
        &self,
        state: Self::State,
        operator: &str,
        sites: &[usize],
    ) -> Result<Option<(Self::State, Complex64)>> {
        if let Self::Reversed(basis) = self {
            return basis.apply_local(state, operator, sites);
        }
        let width_bits = self.width_bits();
        dispatch_wide_basis!(
            self,
            basis => basis
                .apply_local(
                    restore_fixed_width_state(state, width_bits)?,
                    operator,
                    sites,
                )
                .map(|transition| {
                    transition.map(|(target, amplitude)| {
                        (
                            erase_fixed_width_state(width_bits, target),
                            amplitude,
                        )
                    })
                })
        )
    }

    fn apply_local_transitions(
        &self,
        state: Self::State,
        operator: &str,
        sites: &[usize],
    ) -> Result<LocalTransitions<Self::State>> {
        if let Self::Reversed(basis) = self {
            return basis.apply_local_transitions(state, operator, sites);
        }
        let width_bits = self.width_bits();
        dispatch_wide_basis!(
            self,
            basis => basis
                .apply_local_transitions(
                    restore_fixed_width_state(state, width_bits)?,
                    operator,
                    sites,
                )
                .map(|transitions| {
                    transitions
                        .into_iter()
                        .map(|(target, amplitude)| {
                            (
                                erase_fixed_width_state(width_bits, target),
                                amplitude,
                            )
                        })
                        .collect()
                })
        )
    }

    fn apply_local_unreduced_transitions(
        &self,
        state: Self::State,
        operator: &str,
        sites: &[usize],
    ) -> Result<LocalTransitions<Self::State>> {
        if let Self::Reversed(basis) = self {
            return basis.apply_local_unreduced_transitions(state, operator, sites);
        }
        let width_bits = self.width_bits();
        dispatch_wide_basis!(
            self,
            basis => basis
                .apply_local_unreduced_transitions(
                    restore_fixed_width_state(state, width_bits)?,
                    operator,
                    sites,
                )
                .map(|transitions| {
                    transitions
                        .into_iter()
                        .map(|(target, amplitude)| {
                            (
                                erase_fixed_width_state(width_bits, target),
                                amplitude,
                            )
                        })
                        .collect()
                })
        )
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
        if let Self::Reversed(basis) = self {
            return basis.visit_local_unreduced_transitions(state, operator, sites, visit);
        }
        let width_bits = self.width_bits();
        dispatch_wide_basis!(
            self,
            basis => basis.visit_local_unreduced_transitions(
                restore_fixed_width_state(state, width_bits)?,
                operator,
                sites,
                |target, amplitude| {
                    visit(erase_fixed_width_state(width_bits, target), amplitude)
                },
            )
        )
    }

    fn visit_preparsed_local_unreduced_transitions<F>(
        &self,
        state: Self::State,
        operator: &str,
        symbols: &[char],
        split: Option<usize>,
        sites: &[usize],
        mut visit: F,
    ) -> Result<()>
    where
        F: FnMut(Self::State, Complex64) -> Result<()>,
    {
        if let Self::Reversed(basis) = self {
            return basis.visit_preparsed_local_unreduced_transitions(
                state, operator, symbols, split, sites, visit,
            );
        }
        let width_bits = self.width_bits();
        dispatch_wide_basis!(
            self,
            basis => basis.visit_preparsed_local_unreduced_transitions(
                restore_fixed_width_state(state, width_bits)?,
                operator,
                symbols,
                split,
                sites,
                |target, amplitude| {
                    visit(erase_fixed_width_state(width_bits, target), amplitude)
                },
            )
        )
    }

    fn reduction_image(&self, state: Self::State) -> Result<Option<ReductionImage<Self::State>>> {
        if let Self::Reversed(basis) = self {
            return basis.reduction_image(state);
        }
        let width_bits = self.width_bits();
        dispatch_wide_basis!(
            self,
            basis => basis
                .reduction_image(restore_fixed_width_state(state, width_bits)?)?
                .map(|image| {
                    ReductionImage::new(
                        erase_fixed_width_state(width_bits, *image.representative()),
                        image.phase(),
                        image.orbit_size(),
                    )
                })
                .transpose()
        )
    }

    fn transition_orbit_size(&self, state: Self::State) -> Result<usize> {
        if let Self::Reversed(basis) = self {
            return basis.transition_orbit_size(state);
        }
        let width_bits = self.width_bits();
        dispatch_wide_basis!(
            self,
            basis => basis.transition_orbit_size(restore_fixed_width_state(state, width_bits)?)
        )
    }

    fn reduce_transition(
        &self,
        state: Self::State,
        source_orbit_size: usize,
    ) -> Result<Option<(Self::State, Complex64)>> {
        if let Self::Reversed(basis) = self {
            return basis.reduce_transition(state, source_orbit_size);
        }
        let width_bits = self.width_bits();
        dispatch_wide_basis!(
            self,
            basis => Ok(basis
                .reduce_transition(
                    restore_fixed_width_state(state, width_bits)?,
                    source_orbit_size,
                )?
                .map(|(representative, amplitude)| {
                    (
                        erase_fixed_width_state(width_bits, representative),
                        amplitude,
                    )
                }))
        )
    }

    fn index_transition(
        &self,
        state: Self::State,
        source_orbit_size: usize,
    ) -> Result<Option<(usize, Complex64)>> {
        if let Self::Reversed(basis) = self {
            return Ok(basis
                .index_transition(state, source_orbit_size)?
                .map(|(index, amplitude)| (basis.len() - index - 1, amplitude)));
        }
        let width_bits = self.width_bits();
        dispatch_wide_basis!(
            self,
            basis => basis.index_transition(
                restore_fixed_width_state(state, width_bits)?,
                source_orbit_size,
            )
        )
    }

    fn operator_preserves_particle_sector(&self, operator: &str) -> Result<bool> {
        if let Self::Reversed(basis) = self {
            return basis.operator_preserves_particle_sector(operator);
        }
        dispatch_wide_basis!(
            self,
            basis => basis.operator_preserves_particle_sector(operator)
        )
    }
}

impl From<WideSpinBasis256> for WidePackedBasis {
    fn from(basis: WideSpinBasis256) -> Self {
        Self::Spin256(basis)
    }
}

impl From<WideSpinBasis1024> for WidePackedBasis {
    fn from(basis: WideSpinBasis1024) -> Self {
        Self::Spin1024(basis)
    }
}

impl From<WideSpinBasis4096> for WidePackedBasis {
    fn from(basis: WideSpinBasis4096) -> Self {
        Self::Spin4096(basis)
    }
}

impl From<WideSpinBasis16384> for WidePackedBasis {
    fn from(basis: WideSpinBasis16384) -> Self {
        Self::Spin16384(basis)
    }
}

impl From<WideSpinBasisGeneral256> for WidePackedBasis {
    fn from(basis: WideSpinBasisGeneral256) -> Self {
        Self::GeneralSpin256(basis)
    }
}

impl From<WideSpinBasisGeneral1024> for WidePackedBasis {
    fn from(basis: WideSpinBasisGeneral1024) -> Self {
        Self::GeneralSpin1024(basis)
    }
}

impl From<WideSpinBasisGeneral4096> for WidePackedBasis {
    fn from(basis: WideSpinBasisGeneral4096) -> Self {
        Self::GeneralSpin4096(basis)
    }
}

impl From<WideSpinBasisGeneral16384> for WidePackedBasis {
    fn from(basis: WideSpinBasisGeneral16384) -> Self {
        Self::GeneralSpin16384(basis)
    }
}

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
    StateStorage::for_bits(bits.max(1))
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

    /// Convert any explicit rectangular isometry into a reusable basis
    /// projector.
    ///
    /// This is the narrow waist for reduced spaces whose columns are not
    /// representable by a one-dimensional orbit character, including
    /// higher-dimensional finite-group representations.
    pub fn from_operator(
        projector: &(impl LinearOperator + ?Sized),
        tolerance: f64,
    ) -> Result<Self> {
        if !tolerance.is_finite() || tolerance <= 0.0 {
            return Err(QmbedError::InvalidOptions(
                "projector-isometry tolerance must be positive and finite".into(),
            ));
        }
        let (source_dimension, reduced_dimension) = projector.shape();
        let mut columns = vec![Vec::new(); reduced_dimension];
        if let Some(entries) = projector.stored_triplets()? {
            for (row, column, value) in entries {
                if value.norm() > f64::EPSILON {
                    columns[column].push((row, value));
                }
            }
        } else {
            let mut input = vec![Complex64::new(0.0, 0.0); reduced_dimension];
            let mut output = vec![Complex64::new(0.0, 0.0); source_dimension];
            for column in 0..reduced_dimension {
                input.fill(Complex64::new(0.0, 0.0));
                input[column] = Complex64::new(1.0, 0.0);
                projector.apply(&input, &mut output)?;
                columns[column].extend(
                    output
                        .iter()
                        .copied()
                        .enumerate()
                        .filter(|(_, value)| value.norm() > f64::EPSILON),
                );
            }
        }
        let projector = Self::from_columns(source_dimension, columns)?;
        let mut coordinate = vec![Complex64::new(0.0, 0.0); reduced_dimension];
        let mut parent = vec![Complex64::new(0.0, 0.0); source_dimension];
        let mut recovered = vec![Complex64::new(0.0, 0.0); reduced_dimension];
        for column in 0..reduced_dimension {
            coordinate.fill(Complex64::new(0.0, 0.0));
            coordinate[column] = Complex64::new(1.0, 0.0);
            projector.apply(&coordinate, &mut parent)?;
            projector.project(&parent, &mut recovered)?;
            for (row, value) in recovered.iter().copied().enumerate() {
                let expected = if row == column {
                    Complex64::new(1.0, 0.0)
                } else {
                    Complex64::new(0.0, 0.0)
                };
                if (value - expected).norm() > tolerance {
                    return Err(QmbedError::InvalidOptions(format!(
                        "operator columns are not isometric within tolerance {tolerance}"
                    )));
                }
            }
        }
        Ok(projector)
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

    /// Lift a row-major reduced density matrix as `P ρ P†`.
    pub fn lift_density(&self, reduced: &[Complex64]) -> Result<Vec<Complex64>> {
        if reduced.len()
            != self
                .reduced_dimension
                .saturating_mul(self.reduced_dimension)
        {
            return Err(QmbedError::DimensionMismatch(
                "reduced density matrix does not match the projector domain".into(),
            ));
        }
        let mut parent =
            vec![Complex64::new(0.0, 0.0); self.source_dimension * self.source_dimension];
        for reduced_row in 0..self.reduced_dimension {
            for reduced_column in 0..self.reduced_dimension {
                let density = reduced[reduced_row * self.reduced_dimension + reduced_column];
                if density.norm_sqr() <= f64::EPSILON {
                    continue;
                }
                for left in self.column_offsets[reduced_row]..self.column_offsets[reduced_row + 1] {
                    for right in
                        self.column_offsets[reduced_column]..self.column_offsets[reduced_column + 1]
                    {
                        let row = self.row_indices[left];
                        let column = self.row_indices[right];
                        parent[row * self.source_dimension + column] +=
                            self.values[left] * density * self.values[right].conj();
                    }
                }
            }
        }
        Ok(parent)
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
pub struct PackedTensorBasis {
    factors: Vec<PackedBasis>,
    dimensions: Vec<usize>,
    strides: Vec<usize>,
    dimension: usize,
}

impl PackedTensorBasis {
    pub fn new(factors: impl IntoIterator<Item = PackedBasis>) -> Result<Self> {
        let mut flattened = Vec::new();
        for factor in factors {
            match factor {
                PackedBasis::Tensor(tensor) => flattened.extend(tensor.factors),
                other => flattened.push(other),
            }
        }
        if flattened.len() < 2 {
            return Err(QmbedError::InvalidSector(
                "a packed tensor basis requires at least two factors".into(),
            ));
        }
        let dimensions: Vec<_> = flattened.iter().map(Basis::len).collect();
        if dimensions.contains(&0) {
            return Err(QmbedError::InvalidSector(
                "tensor-basis factors must be nonempty".into(),
            ));
        }
        let mut strides = vec![1; dimensions.len()];
        let mut dimension = 1_usize;
        for index in (0..dimensions.len()).rev() {
            strides[index] = dimension;
            dimension = dimension.checked_mul(dimensions[index]).ok_or_else(|| {
                QmbedError::UnsupportedBackend("tensor-basis size overflow".into())
            })?;
        }
        Ok(Self {
            factors: flattened,
            dimensions,
            strides,
            dimension,
        })
    }

    pub fn factors(&self) -> &[PackedBasis] {
        &self.factors
    }

    pub fn dimensions(&self) -> &[usize] {
        &self.dimensions
    }

    fn row(&self, state: u128) -> Result<usize> {
        let row = usize::try_from(state).map_err(|_| QmbedError::StateNotInBasis)?;
        (row < self.dimension)
            .then_some(row)
            .ok_or(QmbedError::StateNotInBasis)
    }

    fn factor_rows(&self, state: u128) -> Result<Vec<usize>> {
        let row = self.row(state)?;
        Ok(self
            .dimensions
            .iter()
            .zip(&self.strides)
            .map(|(&dimension, &stride)| (row / stride) % dimension)
            .collect())
    }

    fn operator_factors<'a>(&self, operator: &'a str) -> Result<Vec<&'a str>> {
        let factors: Vec<_> = operator.split('|').collect();
        if factors.len() != self.factors.len() {
            return Err(QmbedError::InvalidOperator(format!(
                "tensor operator has {} factors, expected {}",
                factors.len(),
                self.factors.len()
            )));
        }
        Ok(factors)
    }
}

impl Basis for PackedTensorBasis {
    type State = u128;

    fn len(&self) -> usize {
        self.dimension
    }

    fn state(&self, index: usize) -> Result<Self::State> {
        if index >= self.dimension {
            return Err(QmbedError::StateNotInBasis);
        }
        Ok(index as u128)
    }

    fn index(&self, state: Self::State) -> Result<usize> {
        self.row(state)
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
        let operators = self.operator_factors(operator)?;
        let source_rows = self.factor_rows(state)?;
        let expected_arity: usize = operators.iter().map(|part| part.chars().count()).sum();
        if sites.len() != expected_arity {
            return Err(QmbedError::InvalidCoupling(format!(
                "tensor operator arity {expected_arity} does not match {} sites",
                sites.len()
            )));
        }

        let mut partial = vec![(0_usize, Complex64::new(1.0, 0.0))];
        let mut site_offset = 0;
        for (factor_index, ((factor, part), &source_row)) in self
            .factors
            .iter()
            .zip(&operators)
            .zip(&source_rows)
            .enumerate()
        {
            let arity = part.chars().count();
            let source_state = factor.state(source_row)?;
            let transitions = if part.is_empty() {
                LocalTransitions::from_iter([(source_state, Complex64::new(1.0, 0.0))])
            } else {
                factor.apply_local_transitions(
                    source_state,
                    part,
                    &sites[site_offset..site_offset + arity],
                )?
            };
            site_offset += arity;
            let mut next = Vec::with_capacity(partial.len().saturating_mul(transitions.len()));
            for &(row, amplitude) in &partial {
                for &(target_state, local) in &transitions {
                    let target_row = factor.index(target_state)?;
                    next.push((
                        row + target_row * self.strides[factor_index],
                        amplitude * local,
                    ));
                }
            }
            partial = next;
        }

        let mut accumulated = HashMap::<usize, Complex64>::new();
        for (row, amplitude) in partial {
            *accumulated.entry(row).or_insert(Complex64::new(0.0, 0.0)) += amplitude;
        }
        let mut transitions: Vec<_> = accumulated
            .into_iter()
            .filter(|(_, amplitude)| amplitude.norm() > f64::EPSILON)
            .collect();
        transitions.sort_unstable_by_key(|(row, _)| *row);
        Ok(transitions
            .into_iter()
            .map(|(row, amplitude)| (row as u128, amplitude))
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
        self.operator_factors(operator)?
            .into_iter()
            .zip(&self.factors)
            .try_fold(true, |preserves, (part, factor)| {
                Ok(preserves && factor.operator_preserves_particle_sector(part)?)
            })
    }

    fn operator_preserves_particle_sector_on_sites(
        &self,
        operator: &str,
        sites: &[usize],
    ) -> Result<bool> {
        let operators = self.operator_factors(operator)?;
        let expected_arity: usize = operators.iter().map(|part| part.chars().count()).sum();
        if sites.len() != expected_arity {
            return Err(QmbedError::InvalidCoupling(
                "tensor operator arity does not match its sites".into(),
            ));
        }
        let mut offset = 0;
        for (part, factor) in operators.into_iter().zip(&self.factors) {
            let arity = part.chars().count();
            if !factor
                .operator_preserves_particle_sector_on_sites(part, &sites[offset..offset + arity])?
            {
                return Ok(false);
            }
            offset += arity;
        }
        Ok(true)
    }
}

/// Runtime-erased matter basis tensored with one truncated photon mode.
///
/// Public states are rows in the unfiltered direct product, with the matter
/// row as the major index and photon occupation as the minor index. Keeping
/// these identifiers unchanged when a fixed-total-excitation filter is
/// applied lets [`BasisProjector`] connect the filtered and full spaces
/// without a photon-specific projection path.
#[derive(Clone, Debug)]
pub struct PackedPhotonBasis {
    matter: Box<PackedBasis>,
    photon: BosonBasis1D,
    photon_dimension: usize,
    full_dimension: usize,
    states: Vec<u128>,
    indices: HashMap<u128, usize>,
    total_excitations: Option<usize>,
}

impl PackedPhotonBasis {
    /// Construct a matter-photon product with photon occupations
    /// `0..=photon_cutoff`, optionally filtered by a total additive quantum
    /// number.
    pub fn new(
        matter: PackedBasis,
        photon_cutoff: usize,
        total_excitations: Option<usize>,
    ) -> Result<Self> {
        let photon_dimension = photon_cutoff
            .checked_add(1)
            .ok_or_else(|| QmbedError::UnsupportedBackend("photon cutoff is too large".into()))?;
        let photon = BosonBasis1D::builder(1, photon_dimension).build()?;
        let full_dimension = matter.len().checked_mul(photon_dimension).ok_or_else(|| {
            QmbedError::UnsupportedBackend("matter-photon basis size overflow".into())
        })?;

        let mut states = Vec::with_capacity(full_dimension);
        for matter_row in 0..matter.len() {
            let matter_state = matter.state(matter_row)?;
            let matter_excitations = matter.additive_quantum_number(matter_state)?;
            for photon_occupation in 0..photon_dimension {
                if total_excitations.is_none_or(|total| {
                    matter_excitations
                        .checked_add(photon_occupation)
                        .is_some_and(|value| value == total)
                }) {
                    states.push(Self::encode_product_state(
                        matter_state,
                        photon_occupation,
                        photon_dimension,
                    )?);
                }
            }
        }
        if states.is_empty() {
            return Err(QmbedError::InvalidSector(
                "the requested total-excitation photon sector is empty".into(),
            ));
        }
        let indices = states
            .iter()
            .copied()
            .enumerate()
            .map(|(index, state)| (state, index))
            .collect();
        Ok(Self {
            matter: Box::new(matter),
            photon,
            photon_dimension,
            full_dimension,
            states,
            indices,
            total_excitations,
        })
    }

    pub fn matter(&self) -> &PackedBasis {
        &self.matter
    }

    pub const fn photon(&self) -> &BosonBasis1D {
        &self.photon
    }

    pub const fn photon_dimension(&self) -> usize {
        self.photon_dimension
    }

    pub const fn full_dimension(&self) -> usize {
        self.full_dimension
    }

    pub const fn total_excitations(&self) -> Option<usize> {
        self.total_excitations
    }

    fn encode_product_state(
        matter_state: u128,
        photon_occupation: usize,
        photon_dimension: usize,
    ) -> Result<u128> {
        matter_state
            .checked_mul(photon_dimension as u128)
            .and_then(|value| value.checked_add(photon_occupation as u128))
            .ok_or_else(|| {
                QmbedError::UnsupportedBackend("matter-photon state encoding exceeds u128".into())
            })
    }

    fn product_state(&self, state: u128) -> Result<(u128, usize)> {
        let photon_occupation = usize::try_from(state % self.photon_dimension as u128)
            .map_err(|_| QmbedError::StateNotInBasis)?;
        Ok((state / self.photon_dimension as u128, photon_occupation))
    }

    fn product_rows(&self, state: u128) -> Result<(usize, usize)> {
        let (matter_state, photon_occupation) = self.product_state(state)?;
        Ok((self.matter.index(matter_state)?, photon_occupation))
    }

    fn operator_parts<'a>(&self, operator: &'a str) -> Result<(&'a str, &'a str)> {
        let (matter, photon) = operator
            .split_once('|')
            .ok_or_else(|| QmbedError::InvalidOperator(operator.into()))?;
        if photon.contains('|') {
            return Err(QmbedError::InvalidOperator(operator.into()));
        }
        Ok((matter, photon))
    }

    fn raw_local_transitions(
        &self,
        state: u128,
        operator: &str,
        sites: &[usize],
    ) -> Result<LocalTransitions<u128>> {
        self.index(state)?;
        let (matter_operator, photon_operator) = self.operator_parts(operator)?;
        let matter_arity = matter_operator.chars().count();
        let photon_arity = photon_operator.chars().count();
        if sites.len() != matter_arity + photon_arity {
            return Err(QmbedError::InvalidCoupling(format!(
                "matter-photon operator arity {} does not match {} sites",
                matter_arity + photon_arity,
                sites.len()
            )));
        }
        let (matter_row, photon_row) = self.product_rows(state)?;
        let matter_state = self.matter.state(matter_row)?;
        let photon_state = self.photon.state(photon_row)?;
        let matter_transitions = if matter_operator.is_empty() {
            LocalTransitions::from_iter([(matter_state, Complex64::new(1.0, 0.0))])
        } else {
            self.matter.apply_local_transitions(
                matter_state,
                matter_operator,
                &sites[..matter_arity],
            )?
        };
        let photon_transitions = if photon_operator.is_empty() {
            LocalTransitions::from_iter([(photon_state, Complex64::new(1.0, 0.0))])
        } else {
            self.photon.apply_local_transitions(
                photon_state,
                photon_operator,
                &sites[matter_arity..],
            )?
        };

        let mut accumulated = HashMap::<u128, Complex64>::new();
        for &(target_matter, matter_amplitude) in &matter_transitions {
            for &(target_photon, photon_amplitude) in &photon_transitions {
                let target_photon_row = self.photon.index(target_photon)?;
                let target = Self::encode_product_state(
                    target_matter,
                    target_photon_row,
                    self.photon_dimension,
                )?;
                *accumulated
                    .entry(target)
                    .or_insert(Complex64::new(0.0, 0.0)) += matter_amplitude * photon_amplitude;
            }
        }
        let mut transitions: Vec<_> = accumulated
            .into_iter()
            .filter(|(_, amplitude)| amplitude.norm() > f64::EPSILON)
            .collect();
        transitions.sort_unstable_by_key(|(row, _)| *row);
        Ok(transitions.into())
    }
}

impl Basis for PackedPhotonBasis {
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
                "this matter-photon action branches; use apply_local_transitions".into(),
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
            .raw_local_transitions(state, operator, sites)?
            .into_iter()
            .filter(|(target, _)| self.indices.contains_key(target))
            .collect())
    }

    fn apply_local_unreduced_transitions(
        &self,
        state: Self::State,
        operator: &str,
        sites: &[usize],
    ) -> Result<LocalTransitions<Self::State>> {
        self.raw_local_transitions(state, operator, sites)
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
        for (target, amplitude) in self.raw_local_transitions(state, operator, sites)? {
            visit(target, amplitude)?;
        }
        Ok(())
    }

    fn transition_orbit_size(&self, state: Self::State) -> Result<usize> {
        let (matter_state, _) = self.product_state(state)?;
        self.matter.transition_orbit_size(matter_state)
    }

    fn reduction_image(&self, state: Self::State) -> Result<Option<ReductionImage<Self::State>>> {
        let (matter_state, photon_occupation) = self.product_state(state)?;
        let matter_excitations = match self.matter.additive_quantum_number(matter_state) {
            Ok(value) => value,
            Err(QmbedError::StateNotInBasis) => return Ok(None),
            Err(error) => return Err(error),
        };
        if self.total_excitations.is_some_and(|total| {
            matter_excitations
                .checked_add(photon_occupation)
                .is_none_or(|value| value != total)
        }) {
            return Ok(None);
        }
        let Some(matter_image) = self.matter.reduction_image(matter_state)? else {
            return Ok(None);
        };
        let representative = Self::encode_product_state(
            *matter_image.representative(),
            photon_occupation,
            self.photon_dimension,
        )?;
        if !self.indices.contains_key(&representative) {
            return Ok(None);
        }
        Ok(Some(ReductionImage::new(
            representative,
            matter_image.phase(),
            matter_image.orbit_size(),
        )?))
    }

    fn operator_preserves_particle_sector(&self, operator: &str) -> Result<bool> {
        self.operator_parts(operator)?;
        Ok(self.total_excitations.is_none() || operator_number_change(operator)? == Some(0))
    }
}

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
    Tensor(Box<PackedTensorBasis>),
    Photon(Box<PackedPhotonBasis>),
    User(Box<UserBasisGeneral>),
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

    /// Additive occupation/excitation quantum number of a concrete state.
    ///
    /// This is the runtime-selected counterpart of the typed sector metadata
    /// used by concrete bases. Composite bases sum their factor quantum
    /// numbers, so consumers such as [`PackedPhotonBasis`] do not need to
    /// special-case spin, boson, or fermion encodings.
    pub fn additive_quantum_number(&self, state: u128) -> Result<usize> {
        fn digit_sum(mut state: u128, sites: usize, base: usize) -> Result<usize> {
            let base = base as u128;
            let mut total = 0_usize;
            for _ in 0..sites {
                total = total.checked_add((state % base) as usize).ok_or_else(|| {
                    QmbedError::UnsupportedBackend("additive quantum number overflow".into())
                })?;
                state /= base;
            }
            Ok(total)
        }

        match self {
            Self::Spin(basis) => {
                basis.index(state)?;
                digit_sum(state, basis.sites(), usize::from(basis.spin_twice()) + 1)
            }
            Self::Boson(basis) => {
                basis.index(state)?;
                digit_sum(state, basis.sites(), basis.states_per_site())
            }
            Self::SpinlessFermion(basis) => {
                basis.index(state)?;
                Ok(state.count_ones() as usize)
            }
            Self::SpinfulFermion(basis) => {
                basis.index(state)?;
                Ok(state.count_ones() as usize)
            }
            Self::GeneralSpin(basis) => {
                basis.parent().index(state)?;
                digit_sum(
                    state,
                    basis.parent().sites(),
                    usize::from(basis.parent().spin_twice()) + 1,
                )
            }
            Self::GeneralBoson(basis) => {
                basis.parent().index(state)?;
                digit_sum(
                    state,
                    basis.parent().sites(),
                    basis.parent().states_per_site(),
                )
            }
            Self::GeneralSpinlessFermion(basis) => {
                basis.parent().index(state)?;
                Ok(state.count_ones() as usize)
            }
            Self::GeneralSpinfulFermion(basis) => {
                basis.parent().index(state)?;
                Ok(state.count_ones() as usize)
            }
            Self::Tensor(basis) => {
                basis.index(state)?;
                let rows = basis.factor_rows(state)?;
                basis
                    .factors()
                    .iter()
                    .zip(rows)
                    .try_fold(0_usize, |total, (factor, row)| {
                        let factor_state = factor.state(row)?;
                        total
                            .checked_add(factor.additive_quantum_number(factor_state)?)
                            .ok_or_else(|| {
                                QmbedError::UnsupportedBackend(
                                    "additive quantum number overflow".into(),
                                )
                            })
                    })
            }
            Self::Photon(basis) => {
                let (matter_state, photon_row) = basis.product_state(state)?;
                basis
                    .matter
                    .additive_quantum_number(matter_state)?
                    .checked_add(photon_row)
                    .ok_or_else(|| {
                        QmbedError::UnsupportedBackend("additive quantum number overflow".into())
                    })
            }
            Self::User(basis) => {
                basis.parent().index(state)?;
                Err(QmbedError::UnsupportedBackend(
                    "a callback-defined basis must provide an additive quantum-number callback"
                        .into(),
                ))
            }
            Self::Reversed(basis) => basis.additive_quantum_number(state),
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

impl From<PackedTensorBasis> for PackedBasis {
    fn from(basis: PackedTensorBasis) -> Self {
        Self::Tensor(Box::new(basis))
    }
}

impl From<PackedPhotonBasis> for PackedBasis {
    fn from(basis: PackedPhotonBasis) -> Self {
        Self::Photon(Box::new(basis))
    }
}

impl From<UserBasisGeneral> for PackedBasis {
    fn from(basis: UserBasisGeneral) -> Self {
        Self::User(Box::new(basis))
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
            Self::Tensor(basis) => basis.len(),
            Self::Photon(basis) => basis.len(),
            Self::User(basis) => basis.len(),
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
            Self::Tensor(basis) => basis.state(index),
            Self::Photon(basis) => basis.state(index),
            Self::User(basis) => basis.state(index),
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
            Self::Tensor(basis) => basis.index(state),
            Self::Photon(basis) => basis.index(state),
            Self::User(basis) => basis.index(state),
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
            Self::Tensor(basis) => basis.apply_local(state, operator, sites),
            Self::Photon(basis) => basis.apply_local(state, operator, sites),
            Self::User(basis) => basis.apply_local(state, operator, sites),
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
            Self::Tensor(basis) => basis.apply_local_transitions(state, operator, sites),
            Self::Photon(basis) => basis.apply_local_transitions(state, operator, sites),
            Self::User(basis) => basis.apply_local_transitions(state, operator, sites),
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
            Self::Tensor(basis) => basis.apply_local_unreduced_transitions(state, operator, sites),
            Self::Photon(basis) => basis.apply_local_unreduced_transitions(state, operator, sites),
            Self::User(basis) => basis.apply_local_unreduced_transitions(state, operator, sites),
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
            Self::Tensor(basis) => {
                basis.visit_local_unreduced_transitions(state, operator, sites, visit)
            }
            Self::Photon(basis) => {
                basis.visit_local_unreduced_transitions(state, operator, sites, visit)
            }
            Self::User(basis) => {
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
            Self::Tensor(basis) => {
                basis.visit_local_unreduced_transitions(state, operator, sites, visit)
            }
            Self::Photon(basis) => {
                basis.visit_local_unreduced_transitions(state, operator, sites, visit)
            }
            Self::User(basis) => basis.visit_preparsed_local_unreduced_transitions(
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
            Self::Tensor(basis) => basis.transition_orbit_size(state),
            Self::Photon(basis) => basis.transition_orbit_size(state),
            Self::User(basis) => basis.transition_orbit_size(state),
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
            Self::Tensor(basis) => basis.reduction_image(state),
            Self::Photon(basis) => basis.reduction_image(state),
            Self::User(basis) => basis.reduction_image(state),
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
            Self::Tensor(basis) => basis.reduce_transition(state, source_orbit_size),
            Self::Photon(basis) => basis.reduce_transition(state, source_orbit_size),
            Self::User(basis) => basis.reduce_transition(state, source_orbit_size),
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
            Self::Tensor(basis) => basis.index_transition(state, source_orbit_size),
            Self::Photon(basis) => basis.index_transition(state, source_orbit_size),
            Self::User(basis) => basis.index_transition(state, source_orbit_size),
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
            Self::Tensor(basis) => basis.operator_preserves_particle_sector(operator),
            Self::Photon(basis) => basis.operator_preserves_particle_sector(operator),
            Self::User(basis) => basis.operator_preserves_particle_sector(operator),
            Self::Reversed(basis) => basis.operator_preserves_particle_sector(operator),
        }
    }

    fn operator_preserves_particle_sector_on_sites(
        &self,
        operator: &str,
        sites: &[usize],
    ) -> Result<bool> {
        match self {
            Self::Spin(basis) => basis.operator_preserves_particle_sector_on_sites(operator, sites),
            Self::Boson(basis) => {
                basis.operator_preserves_particle_sector_on_sites(operator, sites)
            }
            Self::SpinlessFermion(basis) => {
                basis.operator_preserves_particle_sector_on_sites(operator, sites)
            }
            Self::SpinfulFermion(basis) => {
                basis.operator_preserves_particle_sector_on_sites(operator, sites)
            }
            Self::GeneralSpin(basis) => {
                basis.operator_preserves_particle_sector_on_sites(operator, sites)
            }
            Self::GeneralBoson(basis) => {
                basis.operator_preserves_particle_sector_on_sites(operator, sites)
            }
            Self::GeneralSpinlessFermion(basis) => {
                basis.operator_preserves_particle_sector_on_sites(operator, sites)
            }
            Self::GeneralSpinfulFermion(basis) => {
                basis.operator_preserves_particle_sector_on_sites(operator, sites)
            }
            Self::Tensor(basis) => {
                basis.operator_preserves_particle_sector_on_sites(operator, sites)
            }
            Self::Photon(basis) => {
                basis.operator_preserves_particle_sector_on_sites(operator, sites)
            }
            Self::User(basis) => basis.operator_preserves_particle_sector_on_sites(operator, sites),
            Self::Reversed(basis) => {
                basis.operator_preserves_particle_sector_on_sites(operator, sites)
            }
        }
    }
}
