//! Stable, language-neutral entry point shared by the Python and Julia layers.

use std::collections::HashMap;
use std::ffi::{CStr, CString, c_char};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock, RwLock};

use qmbed::basis::{
    Basis, BasisProjector, BosonBasis1D, ExchangeStatistics, GeneralBasis, LatticeSymmetryMap,
    PackedBasis, SpinBasis1D, SpinNormalization, SpinfulFermionBasis1D, SpinlessFermionBasis1D,
    SymmetrySector,
};
use qmbed::interop::{OperatorAction, PackedEdModel};
use qmbed::operator::{
    AssemblyChecks, Coupling, LocalOperator, MatrixFormat, OpProduct, OperatorSpec,
};
use qmbed::solve::{EighOptions, EigshOptions, SpectrumTarget};
use qmbed::{Complex64, QmbedError, Result};
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize)]
pub struct SolveRequest {
    basis: BasisRequest,
    terms: Vec<TermRequest>,
    #[serde(default)]
    format: StorageFormat,
    solver: SolverRequest,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "operation", rename_all = "snake_case")]
enum CommandRequest {
    DescribeBasis {
        basis: BasisRequest,
    },
    Materialize {
        basis: BasisRequest,
        terms: Vec<TermRequest>,
        site_permutation: Option<Vec<usize>>,
        #[serde(default)]
        format: StorageFormat,
        #[serde(default)]
        checks: ChecksRequest,
    },
    Eigh {
        basis: BasisRequest,
        terms: Vec<TermRequest>,
        site_permutation: Option<Vec<usize>>,
        #[serde(default)]
        eigenvectors: bool,
        #[serde(default)]
        checks: ChecksRequest,
    },
    Eigsh {
        basis: BasisRequest,
        terms: Vec<TermRequest>,
        site_permutation: Option<Vec<usize>>,
        #[serde(default)]
        format: StorageFormat,
        solver: SolverRequest,
        #[serde(default)]
        checks: ChecksRequest,
    },
    CreateModel {
        basis: BasisRequest,
        terms: Vec<TermRequest>,
        site_permutation: Option<Vec<usize>>,
        #[serde(default)]
        checks: ChecksRequest,
    },
    DescribeModel {
        handle: String,
    },
    MaterializeModel {
        handle: String,
        #[serde(default)]
        format: StorageFormat,
    },
    MaterializeTermsModel {
        handle: String,
        terms: Vec<TermRequest>,
        #[serde(default)]
        format: StorageFormat,
        #[serde(default)]
        checks: ChecksRequest,
    },
    ApplyModel {
        handle: String,
        vectors: Vec<Vec<[f64; 2]>>,
        #[serde(default)]
        action: OperatorActionRequest,
    },
    ApplyTermsModel {
        handle: String,
        terms: Vec<TermRequest>,
        vectors: Vec<Vec<[f64; 2]>>,
        #[serde(default)]
        action: OperatorActionRequest,
    },
    BraKetTermsModel {
        handle: String,
        terms: Vec<TermRequest>,
        kets: Vec<String>,
    },
    ReduceStatesModel {
        handle: String,
        states: Vec<String>,
    },
    ProjectorModel {
        handle: String,
        parent_handle: String,
    },
    ApplyProjectorModel {
        handle: String,
        parent_handle: String,
        vectors: Vec<Vec<[f64; 2]>>,
        #[serde(default)]
        action: ProjectorActionRequest,
    },
    ApplyTermsBetweenModels {
        source_handle: String,
        target_handle: String,
        terms: Vec<TermRequest>,
        vectors: Vec<Vec<[f64; 2]>>,
    },
    EighModel {
        handle: String,
        #[serde(default)]
        eigenvectors: bool,
    },
    EigshModel {
        handle: String,
        #[serde(default)]
        format: StorageFormat,
        solver: SolverRequest,
    },
    ReleaseModel {
        handle: String,
    },
}

#[derive(Debug, Default, Deserialize)]
struct ChecksRequest {
    hermiticity: Option<bool>,
    particle_conservation: Option<bool>,
    symmetry_compatibility: Option<bool>,
}

impl From<ChecksRequest> for AssemblyChecks {
    fn from(checks: ChecksRequest) -> Self {
        Self {
            hermiticity: checks.hermiticity.unwrap_or(true),
            particle_conservation: checks.particle_conservation.unwrap_or(true),
            symmetry_compatibility: checks.symmetry_compatibility.unwrap_or(true),
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum BasisRequest {
    Spin {
        sites: usize,
        #[serde(default = "default_spin_twice")]
        spin_twice: u16,
        up: Option<usize>,
        momentum: Option<i32>,
        parity: Option<i8>,
        #[serde(default)]
        pauli: bool,
        normalization: Option<SpinNormalizationRequest>,
        #[serde(default)]
        symmetries: Vec<SymmetryRequest>,
        #[serde(default)]
        reverse: bool,
    },
    Boson {
        sites: usize,
        particles: Option<usize>,
        states_per_site: usize,
        #[serde(default)]
        symmetries: Vec<SymmetryRequest>,
        #[serde(default)]
        reverse: bool,
    },
    SpinlessFermion {
        sites: usize,
        particles: Option<usize>,
        momentum: Option<i32>,
        #[serde(default)]
        symmetries: Vec<SymmetryRequest>,
        #[serde(default)]
        reverse: bool,
    },
    SpinfulFermion {
        sites: usize,
        particles_up: Option<usize>,
        particles_down: Option<usize>,
        #[serde(default)]
        symmetries: Vec<SymmetryRequest>,
        #[serde(default)]
        reverse: bool,
    },
}

const fn default_spin_twice() -> u16 {
    1
}

#[derive(Debug, Deserialize)]
struct TermRequest {
    product: ProductRequest,
    couplings: Vec<CouplingRequest>,
}

#[derive(Debug, Deserialize)]
struct ProductRequest {
    local: Vec<String>,
    split: Option<usize>,
}

#[derive(Debug, Deserialize)]
struct CouplingRequest {
    coefficient: [f64; 2],
    sites: Vec<usize>,
}

#[derive(Debug, Deserialize)]
struct SymmetryRequest {
    destinations: Vec<usize>,
    local_permutations: Option<Vec<Vec<usize>>>,
    sector: i32,
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
enum StorageFormat {
    Dense,
    #[default]
    Csc,
    Csr,
    Dia,
    MatrixFree,
}

#[derive(Clone, Copy, Debug, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
enum OperatorActionRequest {
    #[default]
    Normal,
    Transpose,
    Conjugate,
    Adjoint,
}

impl From<OperatorActionRequest> for OperatorAction {
    fn from(value: OperatorActionRequest) -> Self {
        match value {
            OperatorActionRequest::Normal => Self::Normal,
            OperatorActionRequest::Transpose => Self::Transpose,
            OperatorActionRequest::Conjugate => Self::Conjugate,
            OperatorActionRequest::Adjoint => Self::Adjoint,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
enum ProjectorActionRequest {
    #[default]
    Lift,
    Project,
}

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
enum SpinNormalizationRequest {
    AngularMomentum,
    Pauli,
    PauliCartesian,
}

impl From<SpinNormalizationRequest> for SpinNormalization {
    fn from(value: SpinNormalizationRequest) -> Self {
        match value {
            SpinNormalizationRequest::AngularMomentum => Self::AngularMomentum,
            SpinNormalizationRequest::Pauli => Self::Pauli,
            SpinNormalizationRequest::PauliCartesian => Self::PauliCartesian,
        }
    }
}

#[derive(Debug, Serialize)]
struct MatrixEntry {
    row: usize,
    column: usize,
    value: [f64; 2],
}

#[derive(Debug, Serialize)]
struct TransitionEntry {
    input: usize,
    bra: String,
    ket: String,
    value: [f64; 2],
}

#[derive(Debug, Serialize)]
struct ReductionEntry {
    state: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    representative: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    phase: Option<[f64; 2]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    amplitude: Option<[f64; 2]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    orbit_size: Option<usize>,
}

#[derive(Debug, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum CommandResult {
    Basis {
        dimension: usize,
        states: Vec<String>,
    },
    Operator {
        shape: [usize; 2],
        format: StorageFormat,
        entries: Vec<MatrixEntry>,
    },
    Vectors {
        dimension: usize,
        vectors: Vec<Vec<[f64; 2]>>,
    },
    Transitions {
        entries: Vec<TransitionEntry>,
    },
    Reductions {
        entries: Vec<ReductionEntry>,
    },
    Eigensystem {
        dimension: usize,
        eigenvalues: Vec<f64>,
        residuals: Vec<f64>,
        iterations: usize,
        converged: bool,
        #[serde(skip_serializing_if = "Option::is_none")]
        eigenvectors: Option<Vec<Vec<[f64; 2]>>>,
    },
    Model {
        handle: String,
        dimension: usize,
    },
    Released {
        handle: String,
    },
}

static NEXT_MODEL_HANDLE: AtomicU64 = AtomicU64::new(1);
static MODEL_REGISTRY: OnceLock<RwLock<HashMap<u64, Arc<PackedEdModel>>>> = OnceLock::new();
type ProjectorKey = (u64, u64);
type ProjectorCache = HashMap<ProjectorKey, Arc<BasisProjector>>;
static PROJECTOR_REGISTRY: OnceLock<RwLock<ProjectorCache>> = OnceLock::new();

fn model_registry() -> &'static RwLock<HashMap<u64, Arc<PackedEdModel>>> {
    MODEL_REGISTRY.get_or_init(|| RwLock::new(HashMap::new()))
}

fn projector_registry() -> &'static RwLock<ProjectorCache> {
    PROJECTOR_REGISTRY.get_or_init(|| RwLock::new(HashMap::new()))
}

fn parse_model_handle(handle: &str) -> Result<u64> {
    let parsed = handle.parse::<u64>().map_err(|_| {
        QmbedError::InvalidOptions(format!("model handle {handle:?} is not a positive integer"))
    })?;
    if parsed == 0 {
        return Err(QmbedError::InvalidOptions(
            "model handle must be positive".into(),
        ));
    }
    Ok(parsed)
}

fn register_model(model: PackedEdModel) -> Result<String> {
    let handle = NEXT_MODEL_HANDLE
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |next| {
            next.checked_add(1)
        })
        .map_err(|_| QmbedError::InvalidOptions("model handle space is exhausted".into()))?;
    let previous = model_registry()
        .write()
        .map_err(|_| QmbedError::InternalState("model registry lock is poisoned".into()))?
        .insert(handle, Arc::new(model));
    if previous.is_some() {
        return Err(QmbedError::InvalidOptions(format!(
            "model handle {handle} is already registered"
        )));
    }
    Ok(handle.to_string())
}

fn registered_model(handle: &str) -> Result<Arc<PackedEdModel>> {
    let handle = parse_model_handle(handle)?;
    model_registry()
        .read()
        .map_err(|_| QmbedError::InternalState("model registry lock is poisoned".into()))?
        .get(&handle)
        .cloned()
        .ok_or_else(|| {
            QmbedError::InvalidOptions(format!("model handle {handle} is not registered"))
        })
}

fn cached_projector(handle: &str, parent_handle: &str) -> Result<Arc<BasisProjector>> {
    let handle = parse_model_handle(handle)?;
    let parent_handle = parse_model_handle(parent_handle)?;
    let models = model_registry()
        .read()
        .map_err(|_| QmbedError::InternalState("model registry lock is poisoned".into()))?;
    let model = models.get(&handle).ok_or_else(|| {
        QmbedError::InvalidOptions(format!("model handle {handle} is not registered"))
    })?;
    let parent = models.get(&parent_handle).ok_or_else(|| {
        QmbedError::InvalidOptions(format!("model handle {parent_handle} is not registered"))
    })?;
    let mut projectors = projector_registry()
        .write()
        .map_err(|_| QmbedError::InternalState("projector cache lock is poisoned".into()))?;
    if let Some(projector) = projectors.get(&(handle, parent_handle)) {
        return Ok(Arc::clone(projector));
    }
    let projector = Arc::new(model.projector_to(parent)?);
    projectors.insert((handle, parent_handle), Arc::clone(&projector));
    Ok(projector)
}

fn release_model(handle: &str) -> Result<String> {
    let parsed = parse_model_handle(handle)?;
    let mut models = model_registry()
        .write()
        .map_err(|_| QmbedError::InternalState("model registry lock is poisoned".into()))?;
    let removed = models.remove(&parsed);
    if removed.is_none() {
        return Err(QmbedError::InvalidOptions(format!(
            "model handle {parsed} is not registered"
        )));
    }
    projector_registry()
        .write()
        .map_err(|_| QmbedError::InternalState("projector cache lock is poisoned".into()))?
        .retain(|&(reduced, parent), _| reduced != parsed && parent != parsed);
    Ok(parsed.to_string())
}

impl From<StorageFormat> for MatrixFormat {
    fn from(value: StorageFormat) -> Self {
        match value {
            StorageFormat::Dense => Self::Dense,
            StorageFormat::Csc => Self::Csc,
            StorageFormat::Csr => Self::Csr,
            StorageFormat::Dia => Self::Dia,
            StorageFormat::MatrixFree => Self::MatrixFree,
        }
    }
}

#[derive(Debug, Deserialize)]
struct SolverRequest {
    eigenpairs: usize,
    target: TargetRequest,
    krylov_dimension: Option<usize>,
    #[serde(default = "default_tolerance")]
    tolerance: f64,
    #[serde(default = "default_iterations")]
    max_iterations: usize,
    #[serde(default)]
    seed: u64,
    #[serde(default)]
    eigenvectors: bool,
}

const fn default_tolerance() -> f64 {
    1.0e-10
}

const fn default_iterations() -> usize {
    1_000
}

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum TargetRequest {
    SmallestAlgebraic,
    LargestAlgebraic,
    SmallestMagnitude,
    LargestMagnitude,
    BothEnds,
    Shift { value: f64 },
}

impl From<TargetRequest> for SpectrumTarget {
    fn from(value: TargetRequest) -> Self {
        match value {
            TargetRequest::SmallestAlgebraic => Self::SmallestAlgebraic,
            TargetRequest::LargestAlgebraic => Self::LargestAlgebraic,
            TargetRequest::SmallestMagnitude => Self::SmallestMagnitude,
            TargetRequest::LargestMagnitude => Self::LargestMagnitude,
            TargetRequest::BothEnds => Self::BothEnds,
            TargetRequest::Shift { value } => Self::Shift(value),
        }
    }
}

#[derive(Debug, Serialize)]
struct SolveResult {
    dimension: usize,
    eigenvalues: Vec<f64>,
    residuals: Vec<f64>,
    iterations: usize,
    converged: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    eigenvectors: Option<Vec<Vec<[f64; 2]>>>,
}

#[derive(Debug, Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
enum Response {
    Ok { result: SolveResult },
    Error { error: String },
}

#[derive(Debug, Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
enum CommandResponse {
    Ok { result: CommandResult },
    Error { error: String },
}

pub fn run_json(request: &str) -> String {
    let response = serde_json::from_str::<SolveRequest>(request)
        .map_err(|error| QmbedError::InvalidOptions(format!("invalid binding request: {error}")))
        .and_then(execute)
        .map_or_else(
            |error| Response::Error {
                error: error.to_string(),
            },
            |result| Response::Ok { result },
        );
    serde_json::to_string(&response).unwrap_or_else(|error| {
        format!(r#"{{"status":"error","error":"response serialization failed: {error}"}}"#)
    })
}

pub fn run_command_json(request: &str) -> String {
    let response = serde_json::from_str::<CommandRequest>(request)
        .map_err(|error| QmbedError::InvalidOptions(format!("invalid binding command: {error}")))
        .and_then(execute_command)
        .map_or_else(
            |error| CommandResponse::Error {
                error: error.to_string(),
            },
            |result| CommandResponse::Ok { result },
        );
    serde_json::to_string(&response).unwrap_or_else(|error| {
        format!(r#"{{"status":"error","error":"response serialization failed: {error}"}}"#)
    })
}

fn execute(request: SolveRequest) -> Result<SolveResult> {
    let model = build_model(&request.basis, request.terms, AssemblyChecks::all(), None)?;
    solve_model(&model, request.format, &request.solver)
}

fn build_basis(request: &BasisRequest) -> Result<PackedBasis> {
    match request {
        BasisRequest::Spin { .. } => build_spin_basis(request),
        BasisRequest::Boson { .. } => build_boson_basis(request),
        BasisRequest::SpinlessFermion { .. } => build_spinless_basis(request),
        BasisRequest::SpinfulFermion { .. } => build_spinful_basis(request),
    }
}

fn build_spin_basis(request: &BasisRequest) -> Result<PackedBasis> {
    let BasisRequest::Spin {
        sites,
        spin_twice,
        up,
        momentum,
        parity,
        pauli,
        normalization,
        symmetries,
        reverse,
    } = request
    else {
        unreachable!("build_spin_basis requires a spin request");
    };
    if !symmetries.is_empty() && (momentum.is_some() || parity.is_some()) {
        return Err(QmbedError::InvalidOptions(
            "built-in and general spin symmetries cannot be mixed".into(),
        ));
    }
    let mut builder = SpinBasis1D::builder(*sites).spin_twice(*spin_twice);
    builder = match normalization {
        Some(normalization) => builder.normalization((*normalization).into()),
        None => builder.pauli(*pauli),
    };
    if let Some(up) = up {
        builder = builder.up(*up);
    }
    if let Some(momentum) = momentum {
        builder = builder.momentum(*momentum);
    }
    if let Some(parity) = parity {
        builder = builder.parity(*parity);
    }
    let basis = builder.build()?;
    let packed = if symmetries.is_empty() {
        basis.into()
    } else {
        GeneralBasis::new(
            basis,
            runtime_symmetry_sector(
                *sites,
                usize::from(*spin_twice) + 1,
                ExchangeStatistics::Distinguishable,
                symmetries,
            )?,
        )?
        .into()
    };
    Ok(ordered_basis(packed, *reverse))
}

fn build_boson_basis(request: &BasisRequest) -> Result<PackedBasis> {
    let BasisRequest::Boson {
        sites,
        particles,
        states_per_site,
        symmetries,
        reverse,
    } = request
    else {
        unreachable!("build_boson_basis requires a boson request");
    };
    let mut builder = BosonBasis1D::builder(*sites, *states_per_site);
    if let Some(particles) = particles {
        builder = builder.particles(*particles);
    }
    let basis = builder.build()?;
    let packed = if symmetries.is_empty() {
        basis.into()
    } else {
        GeneralBasis::new(
            basis,
            runtime_symmetry_sector(
                *sites,
                *states_per_site,
                ExchangeStatistics::Distinguishable,
                symmetries,
            )?,
        )?
        .into()
    };
    Ok(ordered_basis(packed, *reverse))
}

fn build_spinless_basis(request: &BasisRequest) -> Result<PackedBasis> {
    let BasisRequest::SpinlessFermion {
        sites,
        particles,
        momentum,
        symmetries,
        reverse,
    } = request
    else {
        unreachable!("build_spinless_basis requires a spinless request");
    };
    if !symmetries.is_empty() && momentum.is_some() {
        return Err(QmbedError::InvalidOptions(
            "built-in and general fermion symmetries cannot be mixed".into(),
        ));
    }
    let mut builder = SpinlessFermionBasis1D::builder(*sites);
    if let Some(particles) = particles {
        builder = builder.particles(*particles);
    }
    if let Some(momentum) = momentum {
        builder = builder.momentum(*momentum);
    }
    let basis = builder.build()?;
    let packed = if symmetries.is_empty() {
        basis.into()
    } else {
        GeneralBasis::new(
            basis,
            runtime_symmetry_sector(*sites, 2, ExchangeStatistics::Fermionic, symmetries)?,
        )?
        .into()
    };
    Ok(ordered_basis(packed, *reverse))
}

fn build_spinful_basis(request: &BasisRequest) -> Result<PackedBasis> {
    let BasisRequest::SpinfulFermion {
        sites,
        particles_up,
        particles_down,
        symmetries,
        reverse,
    } = request
    else {
        unreachable!("build_spinful_basis requires a spinful request");
    };
    let mut builder = SpinfulFermionBasis1D::builder(*sites);
    if let Some(particles) = particles_up {
        builder = builder.particles_up(*particles);
    }
    if let Some(particles) = particles_down {
        builder = builder.particles_down(*particles);
    }
    let basis = builder.build()?;
    let packed = if symmetries.is_empty() {
        basis.into()
    } else {
        GeneralBasis::new(
            basis,
            runtime_symmetry_sector(
                sites.checked_mul(2).ok_or_else(|| {
                    QmbedError::UnsupportedBackend("spinful orbital count is too large".into())
                })?,
                2,
                ExchangeStatistics::Fermionic,
                symmetries,
            )?,
        )?
        .into()
    };
    Ok(ordered_basis(packed, *reverse))
}

fn runtime_symmetry_sector(
    encoded_sites: usize,
    states_per_site: usize,
    statistics: ExchangeStatistics,
    requests: &[SymmetryRequest],
) -> Result<SymmetrySector<u128>> {
    let mut sector = SymmetrySector::new();
    for request in requests {
        if request.destinations.len() != encoded_sites {
            return Err(QmbedError::InvalidOptions(format!(
                "symmetry map has {} sites, expected {encoded_sites}",
                request.destinations.len()
            )));
        }
        let map = LatticeSymmetryMap::new(
            states_per_site,
            request.destinations.clone(),
            request.local_permutations.clone(),
            statistics,
        )?;
        sector = sector.with_map(map, request.sector);
    }
    Ok(sector)
}

fn ordered_basis(basis: PackedBasis, reverse: bool) -> PackedBasis {
    if reverse { basis.reversed() } else { basis }
}

fn build_model(
    basis: &BasisRequest,
    terms: Vec<TermRequest>,
    checks: AssemblyChecks,
    site_permutation: Option<Vec<usize>>,
) -> Result<PackedEdModel> {
    let terms = terms
        .into_iter()
        .map(typed_term)
        .collect::<Result<Vec<_>>>()?;
    let model = PackedEdModel::new(build_basis(basis)?, terms).with_checks(checks);
    match site_permutation {
        Some(permutation) => model.with_site_permutation(&permutation),
        None => Ok(model),
    }
}

fn solve_model(
    model: &PackedEdModel,
    format: StorageFormat,
    solver: &SolverRequest,
) -> Result<SolveResult> {
    let include_vectors = solver.eigenvectors;
    let result = model.eigsh(
        format.into(),
        EigshOptions {
            eigenpairs: solver.eigenpairs,
            target: solver.target.into(),
            krylov_dimension: solver.krylov_dimension,
            tolerance: solver.tolerance,
            max_iterations: solver.max_iterations,
            seed: solver.seed,
        },
    )?;
    Ok(SolveResult {
        dimension: model.dimension(),
        eigenvalues: result.eigenvalues,
        residuals: result.residuals,
        iterations: result.iterations,
        converged: result.converged,
        eigenvectors: include_vectors.then(|| {
            result
                .eigenvectors
                .into_iter()
                .map(|vector| {
                    vector
                        .into_iter()
                        .map(|value| [value.re, value.im])
                        .collect()
                })
                .collect()
        }),
    })
}

fn command_eigensystem(
    dimension: usize,
    result: qmbed::solve::Eigensystem,
    include_vectors: bool,
) -> CommandResult {
    CommandResult::Eigensystem {
        dimension,
        eigenvalues: result.eigenvalues,
        residuals: result.residuals,
        iterations: result.iterations,
        converged: result.converged,
        eigenvectors: include_vectors.then(|| {
            result
                .eigenvectors
                .into_iter()
                .map(|vector| {
                    vector
                        .into_iter()
                        .map(|value| [value.re, value.im])
                        .collect()
                })
                .collect()
        }),
    }
}

fn command_operator(model: &PackedEdModel, format: StorageFormat) -> Result<CommandResult> {
    let operator = model.materialized(format.into())?;
    Ok(command_operator_value(operator.as_ref(), format))
}

fn command_operator_value(
    operator: &qmbed::operator::Operator,
    format: StorageFormat,
) -> CommandResult {
    let (rows, columns) = qmbed::operator::LinearOperator::shape(operator);
    let entries = operator
        .triplets()
        .into_iter()
        .map(|(row, column, value)| MatrixEntry {
            row,
            column,
            value: [value.re, value.im],
        })
        .collect();
    CommandResult::Operator {
        shape: [rows, columns],
        format,
        entries,
    }
}

fn command_apply_terms(
    model: &PackedEdModel,
    terms: Vec<TermRequest>,
    vectors: Vec<Vec<[f64; 2]>>,
    action: OperatorActionRequest,
) -> Result<CommandResult> {
    let terms = terms
        .into_iter()
        .map(typed_term)
        .collect::<Result<Vec<_>>>()?;
    let vectors = complex_vectors(vectors);
    let vectors = model.apply_terms_batch(terms, &vectors, action.into())?;
    Ok(command_vectors(model, vectors))
}

fn command_apply_model(
    model: &PackedEdModel,
    vectors: Vec<Vec<[f64; 2]>>,
    action: OperatorActionRequest,
) -> Result<CommandResult> {
    let vectors = complex_vectors(vectors);
    let vectors = model.apply_batch(&vectors, action.into())?;
    Ok(command_vectors(model, vectors))
}

fn complex_vectors(vectors: Vec<Vec<[f64; 2]>>) -> Vec<Vec<Complex64>> {
    vectors
        .into_iter()
        .map(|vector| {
            vector
                .into_iter()
                .map(|[real, imaginary]| Complex64::new(real, imaginary))
                .collect()
        })
        .collect()
}

fn command_vectors(model: &PackedEdModel, vectors: Vec<Vec<Complex64>>) -> CommandResult {
    command_vectors_with_dimension(model.dimension(), vectors)
}

fn command_vectors_with_dimension(
    default_dimension: usize,
    vectors: Vec<Vec<Complex64>>,
) -> CommandResult {
    let dimension = vectors.first().map_or(default_dimension, Vec::len);
    CommandResult::Vectors {
        dimension,
        vectors: vectors
            .into_iter()
            .map(|vector| {
                vector
                    .into_iter()
                    .map(|value| [value.re, value.im])
                    .collect()
            })
            .collect(),
    }
}

fn command_projector(projector: &BasisProjector) -> Result<CommandResult> {
    let entries = qmbed::operator::LinearOperator::stored_triplets(projector)?
        .ok_or_else(|| QmbedError::InternalState("projector has no sparse entries".into()))?
        .into_iter()
        .map(|(row, column, value)| MatrixEntry {
            row,
            column,
            value: [value.re, value.im],
        })
        .collect();
    Ok(CommandResult::Operator {
        shape: [projector.source_dimension(), projector.reduced_dimension()],
        format: StorageFormat::Csc,
        entries,
    })
}

fn command_apply_projector(
    projector: &BasisProjector,
    vectors: Vec<Vec<[f64; 2]>>,
    action: ProjectorActionRequest,
) -> Result<CommandResult> {
    let vectors = complex_vectors(vectors);
    let (default_dimension, vectors) = match action {
        ProjectorActionRequest::Lift => (
            projector.source_dimension(),
            projector.lift_batch(&vectors)?,
        ),
        ProjectorActionRequest::Project => (
            projector.reduced_dimension(),
            projector.project_batch(&vectors)?,
        ),
    };
    Ok(command_vectors_with_dimension(default_dimension, vectors))
}

fn command_apply_terms_between(
    source: &PackedEdModel,
    target: &PackedEdModel,
    terms: Vec<TermRequest>,
    vectors: Vec<Vec<[f64; 2]>>,
) -> Result<CommandResult> {
    let terms = terms
        .into_iter()
        .map(typed_term)
        .collect::<Result<Vec<_>>>()?;
    let vectors = complex_vectors(vectors);
    let vectors = target.apply_terms_from_batch(source, terms, &vectors)?;
    Ok(command_vectors(target, vectors))
}

fn registered_projector(handle: &str, parent_handle: &str) -> Result<CommandResult> {
    let projector = cached_projector(handle, parent_handle)?;
    command_projector(&projector)
}

fn registered_projector_action(
    handle: &str,
    parent_handle: &str,
    vectors: Vec<Vec<[f64; 2]>>,
    action: ProjectorActionRequest,
) -> Result<CommandResult> {
    let projector = cached_projector(handle, parent_handle)?;
    command_apply_projector(&projector, vectors, action)
}

fn registered_cross_sector_action(
    source_handle: &str,
    target_handle: &str,
    terms: Vec<TermRequest>,
    vectors: Vec<Vec<[f64; 2]>>,
) -> Result<CommandResult> {
    let source = registered_model(source_handle)?;
    let target = registered_model(target_handle)?;
    command_apply_terms_between(&source, &target, terms, vectors)
}

fn registered_bra_ket_terms(
    handle: &str,
    terms: Vec<TermRequest>,
    kets: Vec<String>,
) -> Result<CommandResult> {
    let model = registered_model(handle)?;
    command_bra_ket_terms(&model, terms, kets)
}

fn registered_reduce_states(handle: &str, states: Vec<String>) -> Result<CommandResult> {
    let model = registered_model(handle)?;
    command_reduce_states(&model, states)
}

fn command_bra_ket_terms(
    model: &PackedEdModel,
    terms: Vec<TermRequest>,
    kets: Vec<String>,
) -> Result<CommandResult> {
    let terms = terms
        .into_iter()
        .map(typed_term)
        .collect::<Result<Vec<_>>>()?;
    let kets = kets
        .into_iter()
        .map(|ket| {
            ket.parse::<u128>().map_err(|_| {
                QmbedError::InvalidOptions(format!(
                    "ket state {ket:?} is not an unsigned 128-bit integer"
                ))
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let entries = model
        .bra_ket_terms(terms, &kets)?
        .into_iter()
        .enumerate()
        .flat_map(|(input, transitions)| {
            transitions
                .into_iter()
                .map(move |transition| TransitionEntry {
                    input,
                    bra: transition.bra.to_string(),
                    ket: transition.ket.to_string(),
                    value: [transition.matrix_element.re, transition.matrix_element.im],
                })
        })
        .collect();
    Ok(CommandResult::Transitions { entries })
}

fn command_reduce_states(model: &PackedEdModel, states: Vec<String>) -> Result<CommandResult> {
    let states = states
        .into_iter()
        .map(|state| {
            state.parse::<u128>().map_err(|_| {
                QmbedError::InvalidOptions(format!(
                    "physical state {state:?} is not an unsigned 128-bit integer"
                ))
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let images = model.reduction_images(&states)?;
    let entries = states
        .into_iter()
        .zip(images)
        .map(|(state, image)| {
            let representative = image.map(|value| value.representative().to_string());
            let phase = image.map(|value| [value.phase().re, value.phase().im]);
            let amplitude = image.map(|value| [value.amplitude().re, value.amplitude().im]);
            let orbit_size = image.map(|value| value.orbit_size());
            ReductionEntry {
                state: state.to_string(),
                representative,
                phase,
                amplitude,
                orbit_size,
            }
        })
        .collect();
    Ok(CommandResult::Reductions { entries })
}

fn command_eigh(model: &PackedEdModel, eigenvectors: bool) -> Result<CommandResult> {
    let result = model.eigh(EighOptions {
        return_eigenvectors: eigenvectors,
    })?;
    Ok(command_eigensystem(model.dimension(), result, eigenvectors))
}

fn command_eigsh(
    model: &PackedEdModel,
    format: StorageFormat,
    solver: &SolverRequest,
) -> Result<CommandResult> {
    let include_vectors = solver.eigenvectors;
    let result = model.eigsh(
        format.into(),
        EigshOptions {
            eigenpairs: solver.eigenpairs,
            target: solver.target.into(),
            krylov_dimension: solver.krylov_dimension,
            tolerance: solver.tolerance,
            max_iterations: solver.max_iterations,
            seed: solver.seed,
        },
    )?;
    Ok(command_eigensystem(
        model.dimension(),
        result,
        include_vectors,
    ))
}

fn execute_command(request: CommandRequest) -> Result<CommandResult> {
    match request {
        CommandRequest::DescribeBasis { basis } => {
            let basis = build_basis(&basis)?;
            let states = (0..basis.len())
                .map(|index| basis.state(index).map(|state| state.to_string()))
                .collect::<Result<Vec<_>>>()?;
            Ok(CommandResult::Basis {
                dimension: basis.len(),
                states,
            })
        }
        CommandRequest::Materialize {
            basis,
            terms,
            site_permutation,
            format,
            checks,
        } => {
            let model = build_model(&basis, terms, checks.into(), site_permutation)?;
            command_operator(&model, format)
        }
        CommandRequest::Eigh {
            basis,
            terms,
            site_permutation,
            eigenvectors,
            checks,
        } => {
            let model = build_model(&basis, terms, checks.into(), site_permutation)?;
            command_eigh(&model, eigenvectors)
        }
        CommandRequest::Eigsh {
            basis,
            terms,
            site_permutation,
            format,
            solver,
            checks,
        } => {
            let model = build_model(&basis, terms, checks.into(), site_permutation)?;
            command_eigsh(&model, format, &solver)
        }
        CommandRequest::CreateModel {
            basis,
            terms,
            site_permutation,
            checks,
        } => {
            let model = build_model(&basis, terms, checks.into(), site_permutation)?;
            let dimension = model.dimension();
            let handle = register_model(model)?;
            Ok(CommandResult::Model { handle, dimension })
        }
        request => execute_registered_command(request),
    }
}

fn execute_registered_command(request: CommandRequest) -> Result<CommandResult> {
    match request {
        CommandRequest::DescribeModel { handle } => {
            let model = registered_model(&handle)?;
            let states = model
                .states()?
                .into_iter()
                .map(|state| state.to_string())
                .collect();
            Ok(CommandResult::Basis {
                dimension: model.dimension(),
                states,
            })
        }
        CommandRequest::MaterializeModel { handle, format } => {
            let model = registered_model(&handle)?;
            command_operator(&model, format)
        }
        CommandRequest::MaterializeTermsModel {
            handle,
            terms,
            format,
            checks,
        } => {
            let model = registered_model(&handle)?;
            let terms = terms
                .into_iter()
                .map(typed_term)
                .collect::<Result<Vec<_>>>()?;
            let operator = model.assemble_terms(terms, checks.into(), format.into())?;
            Ok(command_operator_value(&operator, format))
        }
        CommandRequest::ApplyModel {
            handle,
            vectors,
            action,
        } => {
            let model = registered_model(&handle)?;
            command_apply_model(&model, vectors, action)
        }
        CommandRequest::ApplyTermsModel {
            handle,
            terms,
            vectors,
            action,
        } => {
            let model = registered_model(&handle)?;
            command_apply_terms(&model, terms, vectors, action)
        }
        CommandRequest::BraKetTermsModel {
            handle,
            terms,
            kets,
        } => registered_bra_ket_terms(&handle, terms, kets),
        CommandRequest::ReduceStatesModel { handle, states } => {
            registered_reduce_states(&handle, states)
        }
        CommandRequest::ProjectorModel {
            handle,
            parent_handle,
        } => registered_projector(&handle, &parent_handle),
        CommandRequest::ApplyProjectorModel {
            handle,
            parent_handle,
            vectors,
            action,
        } => registered_projector_action(&handle, &parent_handle, vectors, action),
        CommandRequest::ApplyTermsBetweenModels {
            source_handle,
            target_handle,
            terms,
            vectors,
        } => registered_cross_sector_action(&source_handle, &target_handle, terms, vectors),
        CommandRequest::EighModel {
            handle,
            eigenvectors,
        } => {
            let model = registered_model(&handle)?;
            command_eigh(&model, eigenvectors)
        }
        CommandRequest::EigshModel {
            handle,
            format,
            solver,
        } => {
            let model = registered_model(&handle)?;
            command_eigsh(&model, format, &solver)
        }
        CommandRequest::ReleaseModel { handle } => {
            let handle = release_model(&handle)?;
            Ok(CommandResult::Released { handle })
        }
        CommandRequest::DescribeBasis { .. }
        | CommandRequest::Materialize { .. }
        | CommandRequest::Eigh { .. }
        | CommandRequest::Eigsh { .. }
        | CommandRequest::CreateModel { .. } => {
            unreachable!("stateless command was pre-dispatched")
        }
    }
}

fn typed_term(term: TermRequest) -> Result<OperatorSpec> {
    let local = term
        .product
        .local
        .iter()
        .map(|name| typed_local_operator(name))
        .collect::<Result<Vec<_>>>()?;
    let product = OpProduct::with_split(local, term.product.split)?;
    let couplings = term.couplings.into_iter().map(|coupling| {
        Coupling::new(
            Complex64::new(coupling.coefficient[0], coupling.coefficient[1]),
            coupling.sites,
        )
    });
    OperatorSpec::from_product(product, couplings)
}

fn typed_local_operator(name: &str) -> Result<LocalOperator> {
    match name {
        "identity" => Ok(LocalOperator::Identity),
        "number" => Ok(LocalOperator::Number),
        "z" => Ok(LocalOperator::Z),
        "raising" => Ok(LocalOperator::Raising),
        "lowering" => Ok(LocalOperator::Lowering),
        "x" => Ok(LocalOperator::X),
        "y" => Ok(LocalOperator::Y),
        custom if custom.starts_with("custom:") => {
            let mut symbols = custom["custom:".len()..].chars();
            match (symbols.next(), symbols.next()) {
                (Some(symbol), None) => Ok(LocalOperator::Custom(symbol)),
                _ => Err(QmbedError::InvalidOperator(custom.into())),
            }
        }
        unknown => Err(QmbedError::InvalidOperator(unknown.into())),
    }
}

/// Execute one JSON request and return an owned UTF-8 JSON response.
///
/// # Safety
///
/// `request` must point to a valid NUL-terminated string for the duration of
/// this call. The returned pointer must be released exactly once with
/// [`qmbed_string_free`].
///
/// # Panics
///
/// This function panics only if Rust's JSON serializer violates its guarantee
/// to escape interior NUL characters.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qmbed_run_json(request: *const c_char) -> *mut c_char {
    let response = catch_unwind(AssertUnwindSafe(|| {
        if request.is_null() {
            return r#"{"status":"error","error":"request pointer is null"}"#.to_string();
        }
        // SAFETY: The caller contract requires a live NUL-terminated string.
        let request = unsafe { CStr::from_ptr(request) };
        match request.to_str() {
            Ok(request) => run_json(request),
            Err(error) => {
                format!(r#"{{"status":"error","error":"request is not UTF-8: {error}"}}"#)
            }
        }
    }))
    .unwrap_or_else(|_| r#"{"status":"error","error":"Rust binding panic"}"#.to_string());
    CString::new(response)
        .expect("serialized JSON never contains an interior NUL")
        .into_raw()
}

/// Execute a reusable-model command encoded as JSON.
///
/// # Safety
///
/// `request` must point to a valid NUL-terminated string for the duration of
/// this call. The returned pointer must be released exactly once with
/// [`qmbed_string_free`].
///
/// # Panics
///
/// This function panics only if Rust's JSON serializer violates its guarantee
/// to escape interior NUL characters.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qmbed_command_json(request: *const c_char) -> *mut c_char {
    let response = catch_unwind(AssertUnwindSafe(|| {
        if request.is_null() {
            return r#"{"status":"error","error":"request pointer is null"}"#.to_string();
        }
        // SAFETY: The caller contract requires a live NUL-terminated string.
        let request = unsafe { CStr::from_ptr(request) };
        match request.to_str() {
            Ok(request) => run_command_json(request),
            Err(error) => {
                format!(r#"{{"status":"error","error":"request is not UTF-8: {error}"}}"#)
            }
        }
    }))
    .unwrap_or_else(|_| r#"{"status":"error","error":"Rust binding panic"}"#.to_string());
    CString::new(response)
        .expect("serialized JSON never contains an interior NUL")
        .into_raw()
}

/// Release a response returned by [`qmbed_run_json`].
///
/// # Safety
///
/// `response` must be null or a pointer returned by [`qmbed_run_json`] that
/// has not already been released.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qmbed_string_free(response: *mut c_char) {
    if !response.is_null() {
        // SAFETY: The caller contract transfers the unique owned pointer back.
        drop(unsafe { CString::from_raw(response) });
    }
}

#[cfg(test)]
mod tests {
    use serde_json::Value;
    use std::sync::Arc;
    use std::thread;

    use super::{cached_projector, run_command_json, run_json};

    #[test]
    fn typed_json_request_reaches_the_rust_solver() {
        let response = run_json(
            r#"{
                "basis":{"kind":"spin","sites":2,"pauli":false},
                "terms":[
                    {"product":{"local":["z","z"]},"couplings":[{"coefficient":[1.0,0.0],"sites":[0,1]}]},
                    {"product":{"local":["raising","lowering"]},"couplings":[{"coefficient":[0.5,0.0],"sites":[0,1]}]},
                    {"product":{"local":["lowering","raising"]},"couplings":[{"coefficient":[0.5,0.0],"sites":[0,1]}]}
                ],
                "format":"csc",
                "solver":{"eigenpairs":2,"target":{"kind":"smallest_algebraic"}}
            }"#,
        );
        let response: Value = serde_json::from_str(&response).unwrap();
        assert_eq!(response["status"], "ok");
        assert_eq!(response["result"]["dimension"], 4);
        assert!((response["result"]["eigenvalues"][0].as_f64().unwrap() + 0.75).abs() < 1.0e-12);
    }

    #[test]
    fn command_protocol_reuses_one_model_shape_for_basis_operator_and_eigh() {
        let basis = r#"{"kind":"boson","sites":1,"states_per_site":4}"#;
        let terms = r#"[
            {"product":{"local":["raising"]},"couplings":[{"coefficient":[1.0,0.0],"sites":[0]}]},
            {"product":{"local":["lowering"]},"couplings":[{"coefficient":[1.0,0.0],"sites":[0]}]},
            {"product":{"local":["number"]},"couplings":[{"coefficient":[0.25,0.0],"sites":[0]}]}
        ]"#;

        let describe = run_command_json(&format!(
            r#"{{"operation":"describe_basis","basis":{basis}}}"#
        ));
        let describe: Value = serde_json::from_str(&describe).unwrap();
        assert_eq!(describe["status"], "ok");
        assert_eq!(describe["result"]["dimension"], 4);
        assert_eq!(describe["result"]["states"][3], "3");

        let materialize = run_command_json(&format!(
            r#"{{"operation":"materialize","basis":{basis},"terms":{terms},"format":"csc"}}"#
        ));
        let materialize: Value = serde_json::from_str(&materialize).unwrap();
        assert_eq!(materialize["status"], "ok");
        assert_eq!(materialize["result"]["shape"], serde_json::json!([4, 4]));
        assert_eq!(
            materialize["result"]["entries"].as_array().unwrap().len(),
            9
        );

        let eigh = run_command_json(&format!(
            r#"{{"operation":"eigh","basis":{basis},"terms":{terms}}}"#
        ));
        let eigh: Value = serde_json::from_str(&eigh).unwrap();
        assert_eq!(eigh["status"], "ok");
        assert_eq!(eigh["result"]["eigenvalues"].as_array().unwrap().len(), 4);
        assert!(eigh["result"].get("eigenvectors").is_none());
    }

    #[test]
    fn registered_model_is_thread_safe_and_rejects_stale_handles() {
        let create = run_command_json(
            r#"{
                "operation":"create_model",
                "basis":{"kind":"spin","sites":3,"reverse":true},
                "terms":[
                    {"product":{"local":["z"]},"couplings":[
                        {"coefficient":[1.0,0.0],"sites":[0]},
                        {"coefficient":[2.0,0.0],"sites":[1]}
                    ]}
                ],
                "site_permutation":[2,1,0]
            }"#,
        );
        let create: Value = serde_json::from_str(&create).unwrap();
        assert_eq!(create["status"], "ok");
        assert_eq!(create["result"]["dimension"], 8);
        let handle = create["result"]["handle"].as_str().unwrap().to_owned();

        let workers = (0..4)
            .map(|_| {
                let handle = handle.clone();
                thread::spawn(move || {
                    run_command_json(&format!(
                        r#"{{"operation":"materialize_model","handle":"{handle}","format":"csc"}}"#
                    ))
                })
            })
            .collect::<Vec<_>>();
        for worker in workers {
            let response: Value = serde_json::from_str(&worker.join().unwrap()).unwrap();
            assert_eq!(response["status"], "ok");
            assert_eq!(response["result"]["shape"], serde_json::json!([8, 8]));
        }

        let release = run_command_json(&format!(
            r#"{{"operation":"release_model","handle":"{handle}"}}"#
        ));
        let release: Value = serde_json::from_str(&release).unwrap();
        assert_eq!(release["status"], "ok");
        assert_eq!(release["result"]["handle"], handle);

        let stale = run_command_json(&format!(
            r#"{{"operation":"eigh_model","handle":"{handle}"}}"#
        ));
        let stale: Value = serde_json::from_str(&stale).unwrap();
        assert_eq!(stale["status"], "error");
        assert!(
            stale["error"]
                .as_str()
                .unwrap()
                .contains("is not registered")
        );
    }

    #[test]
    fn serialized_runtime_symmetry_matches_the_builtin_translation_sector() {
        let builtin = run_command_json(
            r#"{
                "operation":"describe_basis",
                "basis":{
                    "kind":"spin",
                    "sites":6,
                    "up":3,
                    "momentum":1
                }
            }"#,
        );
        let general = run_command_json(
            r#"{
                "operation":"describe_basis",
                "basis":{
                    "kind":"spin",
                    "sites":6,
                    "up":3,
                    "symmetries":[{
                        "destinations":[1,2,3,4,5,0],
                        "sector":1
                    }]
                }
            }"#,
        );
        let builtin: Value = serde_json::from_str(&builtin).unwrap();
        let general: Value = serde_json::from_str(&general).unwrap();
        assert_eq!(builtin["status"], "ok");
        assert_eq!(general["status"], "ok");
        assert_eq!(general["result"], builtin["result"]);

        let invalid = run_command_json(
            r#"{
                "operation":"describe_basis",
                "basis":{
                    "kind":"boson",
                    "sites":3,
                    "states_per_site":2,
                    "symmetries":[{
                        "destinations":[1,0],
                        "sector":0
                    }]
                }
            }"#,
        );
        let invalid: Value = serde_json::from_str(&invalid).unwrap();
        assert_eq!(invalid["status"], "error");
        assert!(invalid["error"].as_str().unwrap().contains("expected 3"));
    }

    #[test]
    fn registered_basis_executes_temporary_terms_vectors_and_transition_tables() {
        let create = run_command_json(
            r#"{
                "operation":"create_model",
                "basis":{
                    "kind":"spin",
                    "sites":1,
                    "normalization":"pauli",
                    "reverse":true
                },
                "terms":[],
                "site_permutation":[0],
                "checks":{
                    "hermiticity":false,
                    "particle_conservation":false,
                    "symmetry_compatibility":false
                }
            }"#,
        );
        let create: Value = serde_json::from_str(&create).unwrap();
        assert_eq!(create["status"], "ok");
        let handle = create["result"]["handle"].as_str().unwrap();
        let raising = r#"[
            {
                "product":{"local":["raising"]},
                "couplings":[{"coefficient":[1.0,0.0],"sites":[0]}]
            }
        ]"#;

        let materialize = run_command_json(&format!(
            r#"{{
                "operation":"materialize_terms_model",
                "handle":"{handle}",
                "terms":{raising},
                "format":"csc",
                "checks":{{
                    "hermiticity":false,
                    "particle_conservation":false,
                    "symmetry_compatibility":false
                }}
            }}"#
        ));
        let materialize: Value = serde_json::from_str(&materialize).unwrap();
        assert_eq!(materialize["status"], "ok");
        assert_eq!(materialize["result"]["entries"][0]["value"][0], 2.0);

        let apply = run_command_json(&format!(
            r#"{{
                "operation":"apply_terms_model",
                "handle":"{handle}",
                "terms":{raising},
                "vectors":[[[0.0,0.0],[1.0,0.0]]],
                "action":"normal"
            }}"#
        ));
        let apply: Value = serde_json::from_str(&apply).unwrap();
        assert_eq!(apply["status"], "ok");
        assert_eq!(
            apply["result"]["vectors"],
            serde_json::json!([[[2.0, 0.0], [0.0, 0.0]]])
        );

        let transitions = run_command_json(&format!(
            r#"{{
                "operation":"bra_ket_terms_model",
                "handle":"{handle}",
                "terms":{raising},
                "kets":["0","1"]
            }}"#
        ));
        let transitions: Value = serde_json::from_str(&transitions).unwrap();
        assert_eq!(transitions["status"], "ok");
        assert_eq!(
            transitions["result"]["entries"].as_array().unwrap().len(),
            1
        );
        assert_eq!(transitions["result"]["entries"][0]["input"], 0);
        assert_eq!(transitions["result"]["entries"][0]["bra"], "1");
        assert_eq!(transitions["result"]["entries"][0]["value"][0], 2.0);

        let release = run_command_json(&format!(
            r#"{{"operation":"release_model","handle":"{handle}"}}"#
        ));
        let release: Value = serde_json::from_str(&release).unwrap();
        assert_eq!(release["status"], "ok");
    }

    #[test]
    fn registered_models_share_projectors_and_cross_sector_actions() {
        let source = run_command_json(
            r#"{
                "operation":"create_model",
                "basis":{"kind":"spin","sites":3,"up":0,"reverse":true},
                "terms":[],
                "site_permutation":[2,1,0]
            }"#,
        );
        let source: Value = serde_json::from_str(&source).unwrap();
        let source_handle = source["result"]["handle"].as_str().unwrap();
        let target = run_command_json(
            r#"{
                "operation":"create_model",
                "basis":{"kind":"spin","sites":3,"up":1,"reverse":true},
                "terms":[],
                "site_permutation":[2,1,0]
            }"#,
        );
        let target: Value = serde_json::from_str(&target).unwrap();
        let target_handle = target["result"]["handle"].as_str().unwrap();
        let parent = run_command_json(
            r#"{
                "operation":"create_model",
                "basis":{"kind":"spin","sites":3,"reverse":true},
                "terms":[],
                "site_permutation":[2,1,0]
            }"#,
        );
        let parent: Value = serde_json::from_str(&parent).unwrap();
        let parent_handle = parent["result"]["handle"].as_str().unwrap();
        let first_projector = cached_projector(target_handle, parent_handle).unwrap();
        let second_projector = cached_projector(target_handle, parent_handle).unwrap();
        assert!(Arc::ptr_eq(&first_projector, &second_projector));

        let projector = run_command_json(&format!(
            r#"{{
                "operation":"projector_model",
                "handle":"{target_handle}",
                "parent_handle":"{parent_handle}"
            }}"#
        ));
        let projector: Value = serde_json::from_str(&projector).unwrap();
        assert_eq!(projector["status"], "ok");
        assert_eq!(projector["result"]["shape"], serde_json::json!([8, 3]));
        assert_eq!(projector["result"]["entries"].as_array().unwrap().len(), 3);

        let shifted = run_command_json(&format!(
            r#"{{
                "operation":"apply_terms_between_models",
                "source_handle":"{source_handle}",
                "target_handle":"{target_handle}",
                "terms":[{{
                    "product":{{"local":["raising"]}},
                    "couplings":[{{"coefficient":[2.0,0.0],"sites":[0]}}]
                }}],
                "vectors":[[[1.0,0.0]]]
            }}"#
        ));
        let shifted: Value = serde_json::from_str(&shifted).unwrap();
        assert_eq!(shifted["status"], "ok");
        assert_eq!(shifted["result"]["dimension"], 3);
        let nonzero = shifted["result"]["vectors"][0]
            .as_array()
            .unwrap()
            .iter()
            .filter(|value| value[0].as_f64().unwrap().abs() > f64::EPSILON)
            .count();
        assert_eq!(nonzero, 1);
    }

    #[test]
    fn registered_model_reports_reduction_metadata_for_physical_states() {
        let create = run_command_json(
            r#"{
                "operation":"create_model",
                "basis":{
                    "kind":"spin",
                    "sites":4,
                    "up":2,
                    "symmetries":[{
                        "destinations":[1,2,3,0],
                        "sector":1
                    }]
                },
                "terms":[]
            }"#,
        );
        let create: Value = serde_json::from_str(&create).unwrap();
        assert_eq!(create["status"], "ok");
        let handle = create["result"]["handle"].as_str().unwrap();

        let reduced = run_command_json(&format!(
            r#"{{
                "operation":"reduce_states_model",
                "handle":"{handle}",
                "states":["3","6","5","0"]
            }}"#
        ));
        let reduced: Value = serde_json::from_str(&reduced).unwrap();
        assert_eq!(reduced["status"], "ok");
        let entries = reduced["result"]["entries"].as_array().unwrap();
        assert_eq!(entries[0]["representative"], "3");
        assert_eq!(entries[0]["orbit_size"], 4);
        assert!((entries[0]["amplitude"][0].as_f64().unwrap() - 0.5).abs() < 1.0e-12);
        assert_eq!(entries[1]["representative"], "3");
        assert!(entries[2].get("representative").is_none());
        assert!(entries[3].get("representative").is_none());
    }
}
