//! Stable, language-neutral entry point shared by the Python and Julia layers.
#![allow(clippy::borrow_as_ptr)] // Callback ABIs require explicit raw-pointer coercions.
#![allow(clippy::cast_possible_truncation)] // Protocol widths are validated at their boundaries.
#![allow(clippy::cast_possible_wrap)] // Operator bytes intentionally preserve their C signed bit pattern.
#![allow(clippy::cast_precision_loss)] // Dimensions are converted only for normalized scientific scalars.
#![allow(clippy::needless_pass_by_value)] // Deserialized command payloads transfer ownership into dispatch.
#![allow(clippy::too_many_lines)] // Command and FFI dispatchers mirror the versioned wire protocol.

use std::collections::{HashMap, HashSet};
use std::ffi::{CStr, CString, c_char, c_void};
use std::hash::Hash;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock, RwLock};

use qmbed::archive::{OperatorArchive, load_zip, save_zip};
use qmbed::basis::{
    Basis, BasisProjector, BosonBasis1D, ClosureSymmetryMap, ErasedState, ExchangeStatistics,
    GeneralBasis, LatticeSymmetryMap, LocalOccupationConstraint, MatrixSymmetryReducer,
    MatrixSymmetrySubspace, PackedBasis, PackedPhotonBasis, PackedTensorBasis, ReductionImage,
    RepresentativeOrdering, SpinBasis1D, SpinNormalization, SpinfulFermionBasis1D,
    SpinlessFermionBasis1D, StateStorage, SymmetryMap, SymmetryReducer, SymmetrySector, UserBasis,
    WidePackedBasis, WideSpinBasis, WideState, get_basis_type,
};
use qmbed::dynamics::{
    DriveStep, Floquet, FloquetAnalysis, FloquetTimeVector, analyze_floquet_unitary,
};
use qmbed::interop::{
    EdModel, OperatorAction, PackedEdModel, PackedOperatorModel, PackedTermComponent, WideEdModel,
};
use qmbed::measure::{
    EntropyOrder, NoncommutingGroup, analyze_diagonal_ensemble,
    apply_noncommuting_subsystem_exchange_phases,
    apply_noncommuting_subsystem_exchange_phases_density,
    canonical_schmidt_spectrum_subsystem_with_exchange_phases, density_expectation,
    density_matrix_spectrum, density_quantum_fluctuation, diagonal_density_matrix,
    diagonal_ensemble, diagonal_ensemble_density, entropy_from_spectrum, expectation,
    matrix_element, mean_level_spacing, partial_trace_density_subsystem, partial_trace_subsystem,
    raw_quantum_fluctuation, subsystem_dimensions,
};
use qmbed::operator::{
    AssemblyChecks, Coupling, LinearOperator, LocalOperator, MatrixFormat, OpProduct, Operator,
    OperatorSpec, QuantumComponent,
};
use qmbed::runtime::CpuRuntime;
use qmbed::solve::{
    EighOptions, EigshOptions, EvolutionOptions, ExpmMultiplyParallel, LanczosOptions,
    LanczosRitzDecomposition, ProjectedThermalObservable, SpectrumTarget, ThermalLanczosMethod,
    eigh_with_options, eigsh, eigsh_with_initial, lanczos_ritz, thermal_observable_contraction,
};
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
    BitwiseStates {
        bitwise_operation: BitwiseOperationRequest,
        width_bits: usize,
        left: Vec<String>,
        #[serde(default)]
        right: Vec<String>,
        #[serde(default)]
        shifts: Vec<usize>,
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
        #[serde(default)]
        components: Vec<TermComponentRequest>,
        site_permutation: Option<Vec<usize>>,
        #[serde(default)]
        checks: ChecksRequest,
    },
    CreateOperatorModel {
        static_operator: Option<MatrixRequest>,
        #[serde(default)]
        components: Vec<OperatorComponentRequest>,
        basis: Option<BasisRequest>,
        site_permutation: Option<Vec<usize>>,
        #[serde(default)]
        checks: ChecksRequest,
    },
    CreateProjectedBlockModel {
        blocks: Vec<ProjectedBlockRequest>,
        #[serde(default = "default_tolerance")]
        tolerance: f64,
        #[serde(default)]
        format: StorageFormat,
    },
    CreateBlockModel {
        handles: Vec<String>,
        #[serde(default)]
        format: StorageFormat,
    },
    LoadOperatorArchive {
        path: String,
    },
    CreateBasisPlan {
        basis: BasisRequest,
        site_permutation: Option<Vec<usize>>,
        #[serde(default)]
        checks: ChecksRequest,
    },
    ReduceStatesPlan {
        plan_handle: String,
        states: Vec<String>,
    },
    MaterializeBasisPlan {
        plan_handle: String,
    },
    ReleaseBasisPlan {
        plan_handle: String,
    },
    BraKetTermsPlan {
        plan_handle: String,
        terms: Vec<TermRequest>,
        kets: Vec<String>,
    },
    ReleaseUserBasis {
        user_basis_handle: String,
    },
    DescribeModel {
        handle: String,
    },
    SaveOperatorArchive {
        handle: String,
        path: String,
        #[serde(default)]
        formats: HashMap<String, StorageFormat>,
        #[serde(default)]
        metadata: HashMap<String, String>,
    },
    MaterializeModel {
        handle: String,
        #[serde(default)]
        format: StorageFormat,
        #[serde(default)]
        parameters: HashMap<String, [f64; 2]>,
    },
    EvaluateOperatorExpression {
        expression: OperatorExpressionRequest,
        #[serde(default)]
        format: StorageFormat,
    },
    CreateOperatorExpressionModel {
        expression: OperatorExpressionRequest,
        #[serde(default)]
        format: StorageFormat,
    },
    ApplyOperatorExpression {
        expression: OperatorExpressionRequest,
        vectors: Vec<Vec<[f64; 2]>>,
    },
    InspectOperatorExpression {
        expression: OperatorExpressionRequest,
    },
    ExpmOperatorExpression {
        expression: OperatorExpressionRequest,
        coefficient: [f64; 2],
        vectors: Vec<Vec<[f64; 2]>>,
        #[serde(default = "default_expm_degree")]
        max_degree: usize,
        #[serde(default = "default_expm_tolerance")]
        tolerance: f64,
        #[serde(default = "default_expm_substeps")]
        max_substeps: usize,
        threads: Option<usize>,
    },
    EighOperatorExpression {
        expression: OperatorExpressionRequest,
        #[serde(default)]
        eigenvectors: bool,
    },
    EigshOperatorExpression {
        expression: OperatorExpressionRequest,
        #[serde(default)]
        format: StorageFormat,
        solver: SolverRequest,
    },
    LanczosOperator {
        expression: OperatorExpressionRequest,
        initial: Vec<[f64; 2]>,
        krylov_dimension: usize,
        #[serde(default = "default_lanczos_tolerance")]
        tolerance: f64,
    },
    LanczosCombine {
        lanczos_handle: String,
        coefficients: Vec<[f64; 2]>,
    },
    LanczosExponential {
        lanczos_handle: String,
        coefficient: [f64; 2],
    },
    LanczosThermal {
        #[serde(default)]
        lanczos_handle: Option<String>,
        method: ThermalLanczosMethodRequest,
        eigenvalues: Vec<f64>,
        eigenvectors: Vec<Vec<f64>>,
        inverse_temperatures: Vec<f64>,
        observables: Vec<ThermalObservableRequest>,
    },
    ExportLanczosBasis {
        lanczos_handle: String,
    },
    ReleaseLanczos {
        lanczos_handle: String,
    },
    CreateExpmAction {
        handle: String,
        #[serde(default)]
        parameters: HashMap<String, [f64; 2]>,
        coefficient: [f64; 2],
        #[serde(default = "default_expm_degree")]
        max_degree: usize,
        #[serde(default = "default_expm_tolerance")]
        tolerance: f64,
        #[serde(default = "default_expm_substeps")]
        max_substeps: usize,
    },
    ApplyExpmAction {
        expm_handle: String,
        vectors: Vec<Vec<[f64; 2]>>,
        threads: Option<usize>,
    },
    ReleaseExpmAction {
        expm_handle: String,
    },
    ProjectOperatorModel {
        handle: String,
        projector: MatrixRequest,
    },
    MaterializeTermsModel {
        handle: String,
        terms: Vec<TermRequest>,
        #[serde(default)]
        format: StorageFormat,
        #[serde(default)]
        checks: ChecksRequest,
    },
    MaterializeComponentModel {
        handle: String,
        name: String,
        #[serde(default)]
        format: StorageFormat,
    },
    ApplyModel {
        handle: String,
        vectors: Vec<Vec<[f64; 2]>>,
        #[serde(default)]
        action: OperatorActionRequest,
        #[serde(default)]
        parameters: HashMap<String, [f64; 2]>,
    },
    MatrixElementsModel {
        handle: String,
        left_vectors: Vec<Vec<[f64; 2]>>,
        right_vectors: Vec<Vec<[f64; 2]>>,
        #[serde(default)]
        diagonal: bool,
        #[serde(default)]
        parameters: HashMap<String, [f64; 2]>,
    },
    MeasureModel {
        handle: String,
        measurement: MeasurementRequest,
        samples: Vec<MeasurementSampleRequest>,
    },
    MeanLevelSpacing {
        eigenvalues: Vec<f64>,
    },
    AnalyzeDiagonalEnsemble {
        eigenvalues: Vec<f64>,
        eigenvectors: Vec<Vec<[f64; 2]>>,
        input: DiagonalEnsembleInputRequest,
        observable: Option<OperatorExpressionRequest>,
        #[serde(default = "default_renyi_alpha")]
        alpha: f64,
        #[serde(default)]
        reconstruct_density: bool,
    },
    FloquetTimeGrid {
        period: f64,
        constant_cycles: usize,
        points_per_cycle: usize,
        #[serde(default)]
        ramp_up_cycles: usize,
        #[serde(default)]
        ramp_down_cycles: usize,
    },
    AnalyzeFloquet {
        steps: Vec<FloquetStepRequest>,
        period: Option<f64>,
        #[serde(default)]
        format: StorageFormat,
    },
    AnalyzeFloquetUnitary {
        unitary: MatrixRequest,
        period: f64,
        #[serde(default)]
        format: StorageFormat,
    },
    AnalyzeSubsystemModel {
        handle: String,
        parent_handle: String,
        #[serde(default)]
        embedding: bool,
        local_dimensions: Vec<usize>,
        retained_sites: Vec<usize>,
        #[serde(default)]
        fermionic: bool,
        #[serde(default)]
        noncommuting_groups: Vec<NoncommutingGroupRequest>,
        samples: Vec<SubsystemSampleRequest>,
        renyi_order: Option<f64>,
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
        #[serde(default)]
        embedding: bool,
    },
    ApplyProjectorModel {
        handle: String,
        parent_handle: String,
        #[serde(default)]
        embedding: bool,
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
        #[serde(default)]
        parameters: HashMap<String, [f64; 2]>,
    },
    EigshModel {
        handle: String,
        #[serde(default)]
        format: StorageFormat,
        solver: SolverRequest,
        #[serde(default)]
        parameters: HashMap<String, [f64; 2]>,
    },
    EvolveModel {
        handle: String,
        vectors: Vec<Vec<[f64; 2]>>,
        evolution: EvolutionRequest,
        #[serde(default)]
        parameters: HashMap<String, [f64; 2]>,
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

#[derive(Clone, Debug, Deserialize)]
struct EvolutionRequest {
    times: Vec<f64>,
    #[serde(default = "default_evolution_krylov_dimension")]
    krylov_dimension: usize,
    #[serde(default = "default_evolution_tolerance")]
    tolerance: f64,
    #[serde(default = "default_evolution_max_substeps")]
    max_substeps: usize,
    #[serde(default)]
    imaginary_time: bool,
}

#[derive(Clone, Debug, Deserialize)]
struct DriveEvolutionRequest {
    handle: String,
    component_names: Vec<String>,
    vectors: Vec<Vec<[f64; 2]>>,
    initial_time: f64,
    evolution: EvolutionRequest,
}

/// ABI-stable complex scalar used by coefficient callbacks.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct QmbedComplex64 {
    pub real: f64,
    pub imaginary: f64,
}

/// Fill `count` ordered component coefficients for one physical time.
///
/// Returning zero reports success. Any nonzero status aborts evolution. The
/// component order is supplied and validated in [`DriveEvolutionRequest`].
pub type QmbedDriveCallback = extern "C" fn(
    context: *mut c_void,
    time: f64,
    coefficients: *mut QmbedComplex64,
    count: usize,
) -> i32;

/// Mutable local-operator result matching `QuSpin`'s `op_results_32` C layout.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct QmbedUserOpResult32 {
    pub matrix_element: QmbedComplex64,
    pub state: u32,
}

/// QuSpin-compatible local-operator callback for 32-bit user states.
pub type QmbedUserOp32Callback = extern "C" fn(
    result: *mut QmbedUserOpResult32,
    operator: i8,
    site: i32,
    sites: i32,
    arguments: *const u32,
) -> i32;

/// QuSpin-compatible constrained-state successor for 32-bit user states.
pub type QmbedUserNextState32Callback =
    extern "C" fn(state: u32, counter: u32, sites: u32, arguments: *const u32) -> u32;

/// QuSpin-compatible state predicate for 32-bit user states.
pub type QmbedUserPreCheck32Callback =
    extern "C" fn(state: u32, sites: u32, arguments: *const u32) -> u32;

/// QuSpin-compatible finite symmetry map for 32-bit user states.
pub type QmbedUserMap32Callback =
    extern "C" fn(state: u32, sites: i32, sign: *mut i8, arguments: *const u32) -> u32;

/// Mutable local-operator result matching `QuSpin`'s `op_results_64` C layout.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct QmbedUserOpResult64 {
    pub matrix_element: QmbedComplex64,
    pub state: u64,
}

pub type QmbedUserOp64Callback = extern "C" fn(
    result: *mut QmbedUserOpResult64,
    operator: i8,
    site: i32,
    sites: i32,
    arguments: *const u64,
) -> i32;
pub type QmbedUserNextState64Callback =
    extern "C" fn(state: u64, counter: u64, sites: u64, arguments: *const u64) -> u64;
pub type QmbedUserPreCheck64Callback =
    extern "C" fn(state: u64, sites: u64, arguments: *const u64) -> u64;
pub type QmbedUserMap64Callback =
    extern "C" fn(state: u64, sites: i32, sign: *mut i8, arguments: *const u64) -> u64;

#[derive(Clone, Debug, Deserialize)]
struct UserStateSegment32 {
    start: u32,
    count: usize,
}

#[derive(Clone, Debug, Deserialize)]
struct UserSymmetry32Request {
    period: usize,
    sector: i32,
    #[serde(default)]
    arguments: Vec<u32>,
}

#[derive(Clone, Debug, Deserialize)]
struct UserBasis32RegistrationRequest {
    sites: usize,
    states_per_site: usize,
    allowed_ops: String,
    #[serde(default)]
    explicit_states: Vec<u32>,
    #[serde(default)]
    state_segments: Vec<UserStateSegment32>,
    #[serde(default)]
    operator_arguments: Vec<u32>,
    #[serde(default)]
    next_state_arguments: Vec<u32>,
    #[serde(default)]
    pre_check_arguments: Vec<u32>,
    #[serde(default)]
    symmetries: Vec<UserSymmetry32Request>,
    #[serde(default = "default_true")]
    reverse: bool,
}

#[derive(Clone, Debug, Deserialize)]
struct UserStateSegment64 {
    start: u64,
    count: usize,
}

#[derive(Clone, Debug, Deserialize)]
struct UserSymmetry64Request {
    period: usize,
    sector: i32,
    #[serde(default)]
    arguments: Vec<u64>,
}

#[derive(Clone, Debug, Deserialize)]
struct UserBasis64RegistrationRequest {
    sites: usize,
    states_per_site: usize,
    allowed_ops: String,
    #[serde(default)]
    explicit_states: Vec<u64>,
    #[serde(default)]
    state_segments: Vec<UserStateSegment64>,
    #[serde(default)]
    operator_arguments: Vec<u64>,
    #[serde(default)]
    next_state_arguments: Vec<u64>,
    #[serde(default)]
    pre_check_arguments: Vec<u64>,
    #[serde(default)]
    symmetries: Vec<UserSymmetry64Request>,
    #[serde(default = "default_true")]
    reverse: bool,
}

const fn default_true() -> bool {
    true
}

const fn default_evolution_krylov_dimension() -> usize {
    64
}

const fn default_evolution_tolerance() -> f64 {
    1.0e-10
}

const fn default_evolution_max_substeps() -> usize {
    10_000
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

#[derive(Clone, Debug, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum BasisRequest {
    Spin {
        sites: usize,
        #[serde(default = "default_spin_twice")]
        spin_twice: u16,
        up: Option<usize>,
        up_sectors: Option<Vec<usize>>,
        momentum: Option<i32>,
        parity: Option<i8>,
        #[serde(default)]
        pauli: bool,
        normalization: Option<SpinNormalizationRequest>,
        #[serde(default)]
        symmetries: Vec<SymmetryRequest>,
        matrix_symmetry: Option<MatrixSymmetryRequest>,
        #[serde(default)]
        reverse: bool,
    },
    Boson {
        sites: usize,
        particles: Option<usize>,
        particle_sectors: Option<Vec<usize>>,
        states_per_site: usize,
        #[serde(default)]
        symmetries: Vec<SymmetryRequest>,
        matrix_symmetry: Option<MatrixSymmetryRequest>,
        #[serde(default)]
        reverse: bool,
    },
    SpinlessFermion {
        sites: usize,
        particles: Option<usize>,
        particle_sectors: Option<Vec<usize>>,
        momentum: Option<i32>,
        #[serde(default)]
        symmetries: Vec<SymmetryRequest>,
        matrix_symmetry: Option<MatrixSymmetryRequest>,
        #[serde(default)]
        reverse: bool,
    },
    SpinfulFermion {
        sites: usize,
        particles_up: Option<usize>,
        particles_down: Option<usize>,
        particle_sectors: Option<Vec<[usize; 2]>>,
        allowed_local_occupancies: Option<Vec<usize>>,
        #[serde(default)]
        symmetries: Vec<SymmetryRequest>,
        matrix_symmetry: Option<MatrixSymmetryRequest>,
        #[serde(default)]
        reverse: bool,
    },
    Tensor {
        factors: Vec<BasisRequest>,
    },
    Photon {
        matter: Box<BasisRequest>,
        photon_cutoff: usize,
        total_excitations: Option<usize>,
    },
    User {
        handle: String,
        #[serde(default)]
        view: UserBasisViewRequest,
    },
}

#[derive(Clone, Copy, Debug, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
enum UserBasisViewRequest {
    #[default]
    Primary,
    Constrained,
    Full,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(untagged)]
enum NoncommutingGroupRequest {
    Fermionic(Vec<usize>),
    WithPhase { sites: Vec<usize>, phase: [f64; 2] },
}

impl TryFrom<NoncommutingGroupRequest> for NoncommutingGroup {
    type Error = QmbedError;

    fn try_from(group: NoncommutingGroupRequest) -> Result<Self> {
        match group {
            NoncommutingGroupRequest::Fermionic(sites) => {
                NoncommutingGroup::new(sites, Complex64::new(-1.0, 0.0))
            }
            NoncommutingGroupRequest::WithPhase {
                sites,
                phase: [real, imaginary],
            } => NoncommutingGroup::new(sites, Complex64::new(real, imaginary)),
        }
    }
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
    #[serde(default)]
    splits: Vec<usize>,
}

#[derive(Debug, Deserialize)]
struct CouplingRequest {
    coefficient: [f64; 2],
    sites: Vec<usize>,
}

#[derive(Clone, Debug, Deserialize)]
struct MatrixRequest {
    shape: [usize; 2],
    entries: Vec<MatrixEntryRequest>,
}

#[derive(Clone, Debug, Deserialize)]
struct MatrixEntryRequest {
    row: usize,
    column: usize,
    value: [f64; 2],
}

#[derive(Clone, Debug, Deserialize)]
struct MatrixComponentRequest {
    name: String,
    operator: MatrixRequest,
    default: Option<[f64; 2]>,
}

#[derive(Clone, Debug, Deserialize)]
struct ProjectedBlockRequest {
    handle: String,
    projector: MatrixRequest,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum OperatorComponentRequest {
    Matrix(MatrixComponentRequest),
    Terms(TermComponentRequest),
}

#[derive(Debug, Deserialize)]
struct FloquetStepRequest {
    expression: OperatorExpressionRequest,
    duration: f64,
}

#[derive(Debug, Deserialize)]
struct TermComponentRequest {
    name: String,
    terms: Vec<TermRequest>,
    default: Option<[f64; 2]>,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum OperatorExpressionRequest {
    Model {
        handle: String,
        #[serde(default)]
        parameters: HashMap<String, [f64; 2]>,
        #[serde(default)]
        action: OperatorActionRequest,
    },
    Matrix {
        operator: MatrixRequest,
        #[serde(default)]
        action: OperatorActionRequest,
    },
    Scale {
        coefficient: [f64; 2],
        operand: Box<OperatorExpressionRequest>,
    },
    Transform {
        action: OperatorActionRequest,
        operand: Box<OperatorExpressionRequest>,
    },
    Binary {
        operation: AlgebraOperationRequest,
        left: Box<OperatorExpressionRequest>,
        right: Box<OperatorExpressionRequest>,
    },
}

#[derive(Debug, Deserialize)]
struct ThermalObservableRequest {
    name: String,
    #[serde(default)]
    expression: Option<OperatorExpressionRequest>,
    #[serde(default)]
    matrix_elements: Vec<[f64; 2]>,
}

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
enum ThermalLanczosMethodRequest {
    Ftlm,
    Ltlm,
}

impl From<ThermalLanczosMethodRequest> for ThermalLanczosMethod {
    fn from(value: ThermalLanczosMethodRequest) -> Self {
        match value {
            ThermalLanczosMethodRequest::Ftlm => Self::Ftlm,
            ThermalLanczosMethodRequest::Ltlm => Self::Ltlm,
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
enum AlgebraOperationRequest {
    Add,
    Subtract,
    Product,
}

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
enum MeasurementRequest {
    Expectation,
    QuantumFluctuation,
}

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
enum BitwiseOperationRequest {
    Not,
    And,
    Or,
    Xor,
    LeftShift,
    RightShift,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum MeasurementSampleRequest {
    Pure {
        values: Vec<[f64; 2]>,
        #[serde(default)]
        parameters: HashMap<String, [f64; 2]>,
    },
    Density {
        values: Vec<[f64; 2]>,
        #[serde(default)]
        parameters: HashMap<String, [f64; 2]>,
    },
}

#[derive(Debug, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum SubsystemSampleRequest {
    Pure { values: Vec<[f64; 2]> },
    Density { values: Vec<[f64; 2]> },
}

#[derive(Debug, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum DiagonalEnsembleInputRequest {
    Pure { values: Vec<[f64; 2]> },
    PureColumns { vectors: Vec<Vec<[f64; 2]>> },
    Density { values: Vec<[f64; 2]> },
    Probabilities { columns: Vec<Vec<f64>> },
}

const fn default_renyi_alpha() -> f64 {
    1.0
}

#[derive(Clone, Debug, Deserialize)]
struct SymmetryRequest {
    destinations: Vec<usize>,
    local_permutations: Option<Vec<Vec<usize>>>,
    sector: i32,
}

#[derive(Clone, Debug, Deserialize)]
struct MatrixSymmetryRequest {
    dimension: usize,
    selected_row: usize,
    generators: Vec<MatrixSymmetryGeneratorRequest>,
}

#[derive(Clone, Debug, Deserialize)]
struct MatrixSymmetryGeneratorRequest {
    destinations: Vec<usize>,
    local_permutations: Option<Vec<Vec<usize>>>,
    matrix: Vec<Vec<[f64; 2]>>,
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
struct MatrixPayload {
    shape: [usize; 2],
    format: StorageFormat,
    entries: Vec<MatrixEntry>,
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
    compatible: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    physical_phase_to_representative: Option<[f64; 2]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    generator_word: Option<Vec<usize>>,
}

#[derive(Debug, Serialize)]
struct SubsystemAnalysisEntry {
    entropy_a: f64,
    entropy_b: f64,
    spectrum_a: Vec<f64>,
    spectrum_b: Vec<f64>,
    density_a: Vec<[f64; 2]>,
    density_b: Vec<[f64; 2]>,
}

#[derive(Debug, Serialize)]
struct ArchiveComponentResult {
    name: String,
    format: StorageFormat,
    #[serde(skip_serializing_if = "Option::is_none")]
    default: Option<[f64; 2]>,
}

#[derive(Debug, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum CommandResult {
    Basis {
        dimension: usize,
        states: Vec<String>,
    },
    Integers {
        width_bits: usize,
        values: Vec<String>,
    },
    Operator {
        shape: [usize; 2],
        format: StorageFormat,
        entries: Vec<MatrixEntry>,
    },
    OperatorSummary {
        shape: [usize; 2],
        diagonal: Vec<[f64; 2]>,
        #[serde(skip_serializing_if = "Option::is_none")]
        trace: Option<[f64; 2]>,
        nonzeros: usize,
    },
    Vectors {
        dimension: usize,
        vectors: Vec<Vec<[f64; 2]>>,
    },
    Measurements {
        shape: Vec<usize>,
        values: Vec<[f64; 2]>,
    },
    Statistic {
        value: f64,
    },
    DiagonalEnsemble {
        probabilities: Vec<Vec<f64>>,
        mean_energies: Vec<f64>,
        energy_variances: Vec<f64>,
        von_neumann_entropies: Vec<f64>,
        diagonal_entropies: Vec<f64>,
        #[serde(skip_serializing_if = "Option::is_none")]
        observables: Option<Vec<f64>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        temporal_fluctuations: Option<Vec<f64>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        quantum_fluctuations: Option<Vec<f64>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        density_matrices: Option<Vec<Vec<[f64; 2]>>>,
    },
    TimeGrid {
        period: f64,
        cycles: usize,
        points_per_cycle: usize,
        times: Vec<f64>,
    },
    FloquetAnalysis {
        period: f64,
        protocol_duration: f64,
        unitary: MatrixPayload,
        quasienergies: Vec<f64>,
        eigenvalues: Vec<[f64; 2]>,
        eigenvectors: Vec<Vec<[f64; 2]>>,
        residuals: Vec<f64>,
        effective_hamiltonian: MatrixPayload,
    },
    Lanczos {
        handle: String,
        dimension: usize,
        krylov_dimension: usize,
        initial_norm: f64,
        eigenvalues: Vec<f64>,
        eigenvectors: Vec<Vec<f64>>,
    },
    LanczosBasis {
        dimension: usize,
        vectors: Vec<Vec<[f64; 2]>>,
    },
    ThermalLanczos {
        values: HashMap<String, Vec<[f64; 2]>>,
        identity: Vec<f64>,
    },
    ExpmPlan {
        handle: String,
        dimension: usize,
    },
    SubsystemAnalysis {
        subsystem_dimension: usize,
        environment_dimension: usize,
        samples: Vec<SubsystemAnalysisEntry>,
    },
    Trajectory {
        dimension: usize,
        times: Vec<f64>,
        states: Vec<Vec<Vec<[f64; 2]>>>,
    },
    Transitions {
        entries: Vec<TransitionEntry>,
    },
    Reductions {
        #[serde(skip_serializing_if = "Option::is_none")]
        period_product: Option<usize>,
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
    Archive {
        path: String,
        components: Vec<ArchiveComponentResult>,
        metadata: HashMap<String, String>,
    },
    ArchivedModel {
        handle: String,
        dimension: usize,
        components: Vec<ArchiveComponentResult>,
        metadata: HashMap<String, String>,
    },
    BasisPlan {
        handle: String,
    },
    UserBasis {
        handle: String,
        dimension: usize,
    },
    Released {
        handle: String,
    },
}

static NEXT_MODEL_HANDLE: AtomicU64 = AtomicU64::new(1);
static MODEL_REGISTRY: OnceLock<RwLock<HashMap<u64, Arc<RegisteredModel>>>> = OnceLock::new();
static NEXT_LANCZOS_HANDLE: AtomicU64 = AtomicU64::new(1);
static LANCZOS_REGISTRY: OnceLock<RwLock<HashMap<u64, Arc<LanczosRitzDecomposition>>>> =
    OnceLock::new();
static NEXT_EXPM_HANDLE: AtomicU64 = AtomicU64::new(1);
static EXPM_REGISTRY: OnceLock<RwLock<HashMap<u64, Arc<ExpmMultiplyParallel>>>> = OnceLock::new();
static NEXT_BASIS_PLAN_HANDLE: AtomicU64 = AtomicU64::new(1);
static BASIS_PLAN_REGISTRY: OnceLock<RwLock<HashMap<u64, Arc<RegisteredBasisPlan>>>> =
    OnceLock::new();
static NEXT_USER_BASIS_HANDLE: AtomicU64 = AtomicU64::new(1);
static USER_BASIS_REGISTRY: OnceLock<RwLock<HashMap<u64, Arc<RegisteredUserBasis>>>> =
    OnceLock::new();
type ProjectorKey = (u64, u64, bool);
type ProjectorCache = HashMap<ProjectorKey, Arc<BasisProjector>>;
static PROJECTOR_REGISTRY: OnceLock<RwLock<ProjectorCache>> = OnceLock::new();

fn model_registry() -> &'static RwLock<HashMap<u64, Arc<RegisteredModel>>> {
    MODEL_REGISTRY.get_or_init(|| RwLock::new(HashMap::new()))
}

fn lanczos_registry() -> &'static RwLock<HashMap<u64, Arc<LanczosRitzDecomposition>>> {
    LANCZOS_REGISTRY.get_or_init(|| RwLock::new(HashMap::new()))
}

fn expm_registry() -> &'static RwLock<HashMap<u64, Arc<ExpmMultiplyParallel>>> {
    EXPM_REGISTRY.get_or_init(|| RwLock::new(HashMap::new()))
}

#[derive(Clone, Debug)]
enum RegisteredModel {
    Ed(Box<PackedEdModel>),
    Wide(Box<WideEdModel>),
    Operator(Box<PackedOperatorModel>),
    Projected(Box<RegisteredProjectedBasisModel<PackedBasis>>),
    WideProjected(Box<RegisteredProjectedBasisModel<WidePackedBasis>>),
}

#[derive(Clone, Debug)]
struct RegisteredProjectedBasisModel<B>
where
    B: Basis,
{
    operator: Box<PackedOperatorModel>,
    assembly_parent: Box<EdModel<B>>,
    projector: Operator,
    basis_projector: BasisProjector,
    labels: Vec<B::State>,
    columns: Vec<Vec<(B::State, Complex64)>>,
}

impl<B> RegisteredProjectedBasisModel<B>
where
    B: Basis + Clone,
    B::State: Copy + Eq + Hash + Ord + 'static,
{
    fn basis_projector_to(&self, parent: &EdModel<B>) -> Result<BasisProjector> {
        let operator = Operator::from_triplets(
            parent.dimension(),
            self.columns.len(),
            self.columns
                .iter()
                .enumerate()
                .flat_map(|(column, entries)| {
                    entries
                        .iter()
                        .map(move |&(state, value)| (state, column, value))
                })
                .map(|(state, column, value)| Ok((parent.basis().index(state)?, column, value)))
                .collect::<Result<Vec<_>>>()?,
            MatrixFormat::Csc,
        )?;
        BasisProjector::from_operator(&operator, default_tolerance())
    }

    fn temporary_operator(
        &self,
        terms: Vec<OperatorSpec>,
        checks: AssemblyChecks,
        format: MatrixFormat,
    ) -> Result<Operator> {
        self.assembly_parent
            .assemble_terms(terms, checks, MatrixFormat::Csc)?
            .projected_by(&self.projector)?
            .converted(format)
    }
}

impl From<PackedEdModel> for RegisteredModel {
    fn from(model: PackedEdModel) -> Self {
        Self::Ed(Box::new(model))
    }
}

impl From<WideEdModel> for RegisteredModel {
    fn from(model: WideEdModel) -> Self {
        Self::Wide(Box::new(model))
    }
}

impl From<PackedOperatorModel> for RegisteredModel {
    fn from(model: PackedOperatorModel) -> Self {
        Self::Operator(Box::new(model))
    }
}

struct RegisteredBasisPlan {
    basis: BasisRequest,
    reducer: SymmetryReducer<u128>,
    representative_ordering: RepresentativeOrdering,
    site_permutation: Option<Vec<usize>>,
    checks: AssemblyChecks,
}

#[derive(Clone, Debug)]
struct RegisteredUserBasis {
    primary: PackedBasis,
    constrained: PackedBasis,
    callback_basis: UserBasis<u128>,
    full_dimension: Option<usize>,
    reverse: bool,
}

impl RegisteredUserBasis {
    fn basis(&self, view: UserBasisViewRequest) -> Result<PackedBasis> {
        match view {
            UserBasisViewRequest::Primary => Ok(self.primary.clone()),
            UserBasisViewRequest::Constrained => Ok(self.constrained.clone()),
            UserBasisViewRequest::Full => {
                let full_dimension = self.full_dimension.ok_or_else(|| {
                    QmbedError::UnsupportedBackend(
                        "the callback state width can represent this basis, but its full Hilbert \
                         space does not fit the runtime index type"
                            .into(),
                    )
                })?;
                let parent = self
                    .callback_basis
                    .with_states((0..full_dimension).map(|state| state as u128))?;
                let basis = GeneralBasis::from_reducer(parent, SymmetryReducer::new())?.into();
                Ok(ordered_basis(basis, self.reverse))
            }
        }
    }
}

impl RegisteredModel {
    fn dimension(&self) -> usize {
        match self {
            Self::Ed(model) => model.dimension(),
            Self::Wide(model) => model.dimension(),
            Self::Operator(model) => model.dimension(),
            Self::Projected(model) => model.operator.dimension(),
            Self::WideProjected(model) => model.operator.dimension(),
        }
    }

    fn states(&self) -> Result<Vec<String>> {
        match self {
            Self::Ed(model) => Ok(model
                .states()?
                .into_iter()
                .map(|state| state.to_string())
                .collect()),
            Self::Wide(model) => Ok(model
                .states()?
                .into_iter()
                .map(|state| state.to_decimal())
                .collect()),
            Self::Operator(model) => Ok((0..model.dimension())
                .map(|index| index.to_string())
                .collect()),
            Self::Projected(model) => Ok(model.labels.iter().map(ToString::to_string).collect()),
            Self::WideProjected(model) => {
                Ok(model.labels.iter().map(ToString::to_string).collect())
            }
        }
    }

    fn ed(&self) -> Result<&PackedEdModel> {
        match self {
            Self::Ed(model) => Ok(model),
            Self::Wide(_) => Err(QmbedError::InvalidOptions(
                "wide-state models do not expose a u128 physical basis".into(),
            )),
            Self::Operator(_) => Err(QmbedError::InvalidOptions(
                "this operator model does not own a physical basis".into(),
            )),
            Self::Projected(_) => Err(QmbedError::InvalidOptions(
                "a matrix-representation model has a multi-state physical basis; \
                 use its explicit projector-aware operations"
                    .into(),
            )),
            Self::WideProjected(_) => Err(QmbedError::InvalidOptions(
                "a wide matrix-representation model has a multi-state physical basis; \
                 use its explicit projector-aware operations"
                    .into(),
            )),
        }
    }

    fn local_subspace(&self) -> Result<(&PackedEdModel, Option<&BasisProjector>)> {
        match self {
            Self::Ed(model) => Ok((model, None)),
            Self::Projected(model) => Ok((&model.assembly_parent, Some(&model.basis_projector))),
            Self::Wide(_) | Self::WideProjected(_) => Err(QmbedError::InvalidOptions(
                "this model uses wide-state storage rather than the packed local subspace".into(),
            )),
            Self::Operator(_) => Err(QmbedError::InvalidOptions(
                "basis-independent operator models do not define local cross-basis actions".into(),
            )),
        }
    }

    fn wide_local_subspace(&self) -> Result<(&WideEdModel, Option<&BasisProjector>)> {
        match self {
            Self::Wide(model) => Ok((model, None)),
            Self::WideProjected(model) => {
                Ok((&model.assembly_parent, Some(&model.basis_projector)))
            }
            Self::Ed(_) | Self::Projected(_) => Err(QmbedError::InvalidOptions(
                "this model uses packed storage rather than the wide local subspace".into(),
            )),
            Self::Operator(_) => Err(QmbedError::InvalidOptions(
                "basis-independent operator models do not define local cross-basis actions".into(),
            )),
        }
    }

    fn materialize(
        &self,
        parameters: &HashMap<String, Complex64>,
        format: MatrixFormat,
    ) -> Result<Operator> {
        match self {
            Self::Ed(model) => model.materialize_with(parameters, format),
            Self::Wide(model) => model.materialize_with(parameters, format),
            Self::Operator(model) => model.materialize(parameters, format),
            Self::Projected(model) => model.operator.materialize(parameters, format),
            Self::WideProjected(model) => model.operator.materialize(parameters, format),
        }
    }

    fn apply_batch(
        &self,
        parameters: &HashMap<String, Complex64>,
        vectors: &[Vec<Complex64>],
        action: OperatorAction,
    ) -> Result<Vec<Vec<Complex64>>> {
        match self {
            Self::Ed(model) => model.apply_batch_with(parameters, vectors, action),
            Self::Wide(model) => model.apply_batch_with(parameters, vectors, action),
            Self::Operator(model) => model.apply_batch(parameters, vectors, action),
            Self::Projected(model) => model.operator.apply_batch(parameters, vectors, action),
            Self::WideProjected(model) => model.operator.apply_batch(parameters, vectors, action),
        }
    }

    fn eigh(
        &self,
        parameters: &HashMap<String, Complex64>,
        options: EighOptions,
    ) -> Result<qmbed::solve::Eigensystem> {
        match self {
            Self::Ed(model) => model.eigh_with(parameters, options),
            Self::Wide(model) => model.eigh_with(parameters, options),
            Self::Operator(model) => model.eigh(parameters, options),
            Self::Projected(model) => model.operator.eigh(parameters, options),
            Self::WideProjected(model) => model.operator.eigh(parameters, options),
        }
    }

    fn eigsh(
        &self,
        parameters: &HashMap<String, Complex64>,
        format: MatrixFormat,
        options: EigshOptions,
        initial: Option<&[Complex64]>,
    ) -> Result<qmbed::solve::Eigensystem> {
        match self {
            Self::Ed(model) => match initial {
                Some(initial) => {
                    model.eigsh_with_initial_and_parameters(parameters, format, options, initial)
                }
                None => model.eigsh_with(parameters, format, options),
            },
            Self::Wide(model) => match initial {
                Some(initial) => {
                    model.eigsh_with_initial_and_parameters(parameters, format, options, initial)
                }
                None => model.eigsh_with(parameters, format, options),
            },
            Self::Operator(model) => match initial {
                Some(initial) => model.eigsh_with_initial(parameters, format, options, initial),
                None => model.eigsh(parameters, format, options),
            },
            Self::Projected(model) => match initial {
                Some(initial) => model
                    .operator
                    .eigsh_with_initial(parameters, format, options, initial),
                None => model.operator.eigsh(parameters, format, options),
            },
            Self::WideProjected(model) => match initial {
                Some(initial) => model
                    .operator
                    .eigsh_with_initial(parameters, format, options, initial),
                None => model.operator.eigsh(parameters, format, options),
            },
        }
    }

    fn evolve_batch(
        &self,
        parameters: &HashMap<String, Complex64>,
        vectors: &[Vec<Complex64>],
        options: EvolutionOptions,
    ) -> Result<qmbed::solve::StateBatchTrajectory> {
        match self {
            Self::Ed(model) => model.evolve_batch_with(parameters, vectors, options),
            Self::Wide(model) => model.evolve_batch_with(parameters, vectors, options),
            Self::Operator(model) => model.evolve_batch(parameters, vectors, options),
            Self::Projected(model) => model.operator.evolve_batch(parameters, vectors, options),
            Self::WideProjected(model) => model.operator.evolve_batch(parameters, vectors, options),
        }
    }

    fn component_names(&self) -> Vec<String> {
        match self {
            Self::Ed(model) => model.component_names().map(str::to_owned).collect(),
            Self::Wide(model) => model.component_names().map(str::to_owned).collect(),
            Self::Operator(model) => model.component_names().map(str::to_owned).collect(),
            Self::Projected(model) => model
                .operator
                .component_names()
                .map(str::to_owned)
                .collect(),
            Self::WideProjected(model) => model
                .operator
                .component_names()
                .map(str::to_owned)
                .collect(),
        }
    }

    fn component_archive(
        &self,
        formats: &HashMap<String, MatrixFormat>,
    ) -> Result<OperatorArchive> {
        match self {
            Self::Ed(model) => model
                .operator_model(MatrixFormat::Csc)?
                .component_archive(formats),
            Self::Wide(model) => model
                .operator_model(MatrixFormat::Csc)?
                .component_archive(formats),
            Self::Operator(model) => model.component_archive(formats),
            Self::Projected(model) => model.operator.component_archive(formats),
            Self::WideProjected(model) => model.operator.component_archive(formats),
        }
    }

    fn projected_by(&self, projector: &Operator) -> Result<PackedOperatorModel> {
        match self {
            Self::Ed(model) => model
                .operator_model(MatrixFormat::Csc)?
                .projected_by(projector),
            Self::Wide(model) => model
                .operator_model(MatrixFormat::Csc)?
                .projected_by(projector),
            Self::Operator(model) => model.projected_by(projector),
            Self::Projected(model) => model.operator.projected_by(projector),
            Self::WideProjected(model) => model.operator.projected_by(projector),
        }
    }

    fn operator_model(&self, format: MatrixFormat) -> Result<PackedOperatorModel> {
        match self {
            Self::Ed(model) => model.operator_model(format),
            Self::Wide(model) => model.operator_model(format),
            Self::Operator(model) => Ok(model.as_ref().clone()),
            Self::Projected(model) => Ok(model.operator.as_ref().clone()),
            Self::WideProjected(model) => Ok(model.operator.as_ref().clone()),
        }
    }

    fn component_operator(&self, name: &str, format: MatrixFormat) -> Result<Operator> {
        match self {
            Self::Ed(model) => model
                .operator_model(format)?
                .component_operator(name, format),
            Self::Wide(model) => model
                .operator_model(format)?
                .component_operator(name, format),
            Self::Operator(model) => model.component_operator(name, format),
            Self::Projected(model) => model.operator.component_operator(name, format),
            Self::WideProjected(model) => model.operator.component_operator(name, format),
        }
    }

    fn temporary_operator(
        &self,
        terms: Vec<OperatorSpec>,
        checks: AssemblyChecks,
        format: MatrixFormat,
    ) -> Result<Operator> {
        match self {
            Self::Ed(model) => model.assemble_terms(terms, checks, format),
            Self::Wide(model) => model.assemble_terms(terms, checks, format),
            Self::Operator(_) => Err(QmbedError::InvalidOptions(
                "this operator model does not own a physical local algebra".into(),
            )),
            Self::Projected(model) => model.temporary_operator(terms, checks, format),
            Self::WideProjected(model) => model.temporary_operator(terms, checks, format),
        }
    }

    fn evolve_time_dependent_batch<F>(
        &self,
        vectors: &[Vec<Complex64>],
        initial_time: f64,
        options: EvolutionOptions,
        operator_scale: Complex64,
        coefficients_at: F,
    ) -> Result<qmbed::solve::StateBatchTrajectory>
    where
        F: Fn(f64, &mut [Complex64]) -> Result<()> + Send + Sync + 'static,
    {
        match self {
            Self::Ed(model) => model.evolve_time_dependent_batch_scaled(
                vectors,
                initial_time,
                options,
                operator_scale,
                coefficients_at,
            ),
            Self::Wide(model) => model.evolve_time_dependent_batch_scaled(
                vectors,
                initial_time,
                options,
                operator_scale,
                coefficients_at,
            ),
            Self::Operator(model) => model.evolve_time_dependent_batch_scaled(
                vectors,
                initial_time,
                options,
                operator_scale,
                coefficients_at,
            ),
            Self::Projected(model) => model.operator.evolve_time_dependent_batch_scaled(
                vectors,
                initial_time,
                options,
                operator_scale,
                coefficients_at,
            ),
            Self::WideProjected(model) => model.operator.evolve_time_dependent_batch_scaled(
                vectors,
                initial_time,
                options,
                operator_scale,
                coefficients_at,
            ),
        }
    }
}

fn basis_plan_registry() -> &'static RwLock<HashMap<u64, Arc<RegisteredBasisPlan>>> {
    BASIS_PLAN_REGISTRY.get_or_init(|| RwLock::new(HashMap::new()))
}

fn user_basis_registry() -> &'static RwLock<HashMap<u64, Arc<RegisteredUserBasis>>> {
    USER_BASIS_REGISTRY.get_or_init(|| RwLock::new(HashMap::new()))
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

fn parse_lanczos_handle(handle: &str) -> Result<u64> {
    let value = handle.strip_prefix("lanczos:").ok_or_else(|| {
        QmbedError::InvalidOptions(format!(
            "Lanczos handle {handle:?} does not use the lanczos: prefix"
        ))
    })?;
    let parsed = value
        .parse::<u64>()
        .map_err(|_| QmbedError::InvalidOptions(format!("Lanczos handle {handle:?} is invalid")))?;
    if parsed == 0 {
        return Err(QmbedError::InvalidOptions(
            "Lanczos handle must be positive".into(),
        ));
    }
    Ok(parsed)
}

fn parse_expm_handle(handle: &str) -> Result<u64> {
    let value = handle.strip_prefix("expm:").ok_or_else(|| {
        QmbedError::InvalidOptions(format!(
            "exponential-action handle {handle:?} does not use the expm: prefix"
        ))
    })?;
    let parsed = value.parse::<u64>().map_err(|_| {
        QmbedError::InvalidOptions(format!("exponential-action handle {handle:?} is invalid"))
    })?;
    if parsed == 0 {
        return Err(QmbedError::InvalidOptions(
            "exponential-action handle must be positive".into(),
        ));
    }
    Ok(parsed)
}

fn parse_basis_plan_handle(handle: &str) -> Result<u64> {
    let parsed = handle.parse::<u64>().map_err(|_| {
        QmbedError::InvalidOptions(format!(
            "basis-plan handle {handle:?} is not a positive integer"
        ))
    })?;
    if parsed == 0 {
        return Err(QmbedError::InvalidOptions(
            "basis-plan handle must be positive".into(),
        ));
    }
    Ok(parsed)
}

fn parse_user_basis_handle(handle: &str) -> Result<u64> {
    let value = handle.strip_prefix("user:").ok_or_else(|| {
        QmbedError::InvalidOptions(format!(
            "user-basis handle {handle:?} does not use the user: prefix"
        ))
    })?;
    let parsed = value.parse::<u64>().map_err(|_| {
        QmbedError::InvalidOptions(format!("user-basis handle {handle:?} is invalid"))
    })?;
    if parsed == 0 {
        return Err(QmbedError::InvalidOptions(
            "user-basis handle must be positive".into(),
        ));
    }
    Ok(parsed)
}

fn register_user_basis(basis: RegisteredUserBasis) -> Result<String> {
    let handle = NEXT_USER_BASIS_HANDLE
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |next| {
            next.checked_add(1)
        })
        .map_err(|_| QmbedError::InvalidOptions("user-basis handle space is exhausted".into()))?;
    let previous = user_basis_registry()
        .write()
        .map_err(|_| QmbedError::InternalState("user-basis registry lock is poisoned".into()))?
        .insert(handle, Arc::new(basis));
    if previous.is_some() {
        return Err(QmbedError::InvalidOptions(format!(
            "user-basis handle {handle} is already registered"
        )));
    }
    Ok(format!("user:{handle}"))
}

fn registered_user_basis(handle: &str) -> Result<Arc<RegisteredUserBasis>> {
    let parsed = parse_user_basis_handle(handle)?;
    user_basis_registry()
        .read()
        .map_err(|_| QmbedError::InternalState("user-basis registry lock is poisoned".into()))?
        .get(&parsed)
        .cloned()
        .ok_or_else(|| {
            QmbedError::InvalidOptions(format!("user-basis handle {handle:?} is not registered"))
        })
}

fn release_user_basis(handle: &str) -> Result<String> {
    let parsed = parse_user_basis_handle(handle)?;
    let removed = user_basis_registry()
        .write()
        .map_err(|_| QmbedError::InternalState("user-basis registry lock is poisoned".into()))?
        .remove(&parsed);
    if removed.is_none() {
        return Err(QmbedError::InvalidOptions(format!(
            "user-basis handle {handle:?} is not registered"
        )));
    }
    Ok(handle.to_owned())
}

fn register_basis_plan(plan: RegisteredBasisPlan) -> Result<String> {
    let handle = NEXT_BASIS_PLAN_HANDLE
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |next| {
            next.checked_add(1)
        })
        .map_err(|_| QmbedError::InvalidOptions("basis-plan handle space is exhausted".into()))?;
    let previous = basis_plan_registry()
        .write()
        .map_err(|_| QmbedError::InternalState("basis-plan registry lock is poisoned".into()))?
        .insert(handle, Arc::new(plan));
    if previous.is_some() {
        return Err(QmbedError::InvalidOptions(format!(
            "basis-plan handle {handle} is already registered"
        )));
    }
    Ok(handle.to_string())
}

fn registered_basis_plan(handle: &str) -> Result<Arc<RegisteredBasisPlan>> {
    let handle = parse_basis_plan_handle(handle)?;
    basis_plan_registry()
        .read()
        .map_err(|_| QmbedError::InternalState("basis-plan registry lock is poisoned".into()))?
        .get(&handle)
        .cloned()
        .ok_or_else(|| {
            QmbedError::InvalidOptions(format!("basis-plan handle {handle} is not registered"))
        })
}

fn release_basis_plan(handle: &str) -> Result<String> {
    let parsed = parse_basis_plan_handle(handle)?;
    let removed = basis_plan_registry()
        .write()
        .map_err(|_| QmbedError::InternalState("basis-plan registry lock is poisoned".into()))?
        .remove(&parsed);
    if removed.is_none() {
        return Err(QmbedError::InvalidOptions(format!(
            "basis-plan handle {parsed} is not registered"
        )));
    }
    Ok(parsed.to_string())
}

fn register_model(model: impl Into<RegisteredModel>) -> Result<String> {
    let model = model.into();
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

fn register_lanczos(decomposition: LanczosRitzDecomposition) -> Result<String> {
    let handle = NEXT_LANCZOS_HANDLE
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |next| {
            next.checked_add(1)
        })
        .map_err(|_| QmbedError::InvalidOptions("Lanczos handle space is exhausted".into()))?;
    let previous = lanczos_registry()
        .write()
        .map_err(|_| QmbedError::InternalState("Lanczos registry lock is poisoned".into()))?
        .insert(handle, Arc::new(decomposition));
    if previous.is_some() {
        return Err(QmbedError::InvalidOptions(format!(
            "Lanczos handle {handle} is already registered"
        )));
    }
    Ok(format!("lanczos:{handle}"))
}

fn registered_lanczos(handle: &str) -> Result<Arc<LanczosRitzDecomposition>> {
    let parsed = parse_lanczos_handle(handle)?;
    lanczos_registry()
        .read()
        .map_err(|_| QmbedError::InternalState("Lanczos registry lock is poisoned".into()))?
        .get(&parsed)
        .cloned()
        .ok_or_else(|| {
            QmbedError::InvalidOptions(format!("Lanczos handle {handle} is not registered"))
        })
}

fn release_lanczos(handle: &str) -> Result<String> {
    let parsed = parse_lanczos_handle(handle)?;
    let removed = lanczos_registry()
        .write()
        .map_err(|_| QmbedError::InternalState("Lanczos registry lock is poisoned".into()))?
        .remove(&parsed);
    if removed.is_none() {
        return Err(QmbedError::InvalidOptions(format!(
            "Lanczos handle {handle} is not registered"
        )));
    }
    Ok(format!("lanczos:{parsed}"))
}

fn register_expm(plan: ExpmMultiplyParallel) -> Result<String> {
    let handle = NEXT_EXPM_HANDLE
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |next| {
            next.checked_add(1)
        })
        .map_err(|_| {
            QmbedError::InvalidOptions("exponential-action handle space is exhausted".into())
        })?;
    let previous = expm_registry()
        .write()
        .map_err(|_| {
            QmbedError::InternalState("exponential-action registry lock is poisoned".into())
        })?
        .insert(handle, Arc::new(plan));
    if previous.is_some() {
        return Err(QmbedError::InvalidOptions(format!(
            "exponential-action handle {handle} is already registered"
        )));
    }
    Ok(format!("expm:{handle}"))
}

fn registered_expm(handle: &str) -> Result<Arc<ExpmMultiplyParallel>> {
    let parsed = parse_expm_handle(handle)?;
    expm_registry()
        .read()
        .map_err(|_| {
            QmbedError::InternalState("exponential-action registry lock is poisoned".into())
        })?
        .get(&parsed)
        .cloned()
        .ok_or_else(|| {
            QmbedError::InvalidOptions(format!(
                "exponential-action handle {handle} is not registered"
            ))
        })
}

fn release_expm(handle: &str) -> Result<String> {
    let parsed = parse_expm_handle(handle)?;
    let removed = expm_registry()
        .write()
        .map_err(|_| {
            QmbedError::InternalState("exponential-action registry lock is poisoned".into())
        })?
        .remove(&parsed);
    if removed.is_none() {
        return Err(QmbedError::InvalidOptions(format!(
            "exponential-action handle {handle} is not registered"
        )));
    }
    Ok(format!("expm:{parsed}"))
}

fn registered_model(handle: &str) -> Result<Arc<RegisteredModel>> {
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

fn cached_projector(
    handle: &str,
    parent_handle: &str,
    embedding: bool,
) -> Result<Arc<BasisProjector>> {
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
    if let Some(projector) = projectors.get(&(handle, parent_handle, embedding)) {
        return Ok(Arc::clone(projector));
    }
    let projector = Arc::new(match (model.as_ref(), parent.as_ref()) {
        (RegisteredModel::Ed(model), RegisteredModel::Ed(parent)) if embedding => {
            model.embedding_to(parent)?
        }
        (RegisteredModel::Ed(model), RegisteredModel::Ed(parent)) => model.projector_to(parent)?,
        (RegisteredModel::Wide(model), RegisteredModel::Wide(parent)) if embedding => {
            model.embedding_to(parent)?
        }
        (RegisteredModel::Wide(model), RegisteredModel::Wide(parent)) => {
            model.projector_to(parent)?
        }
        (RegisteredModel::Projected(_) | RegisteredModel::WideProjected(_), _) if embedding => {
            return Err(QmbedError::InvalidOptions(
                "matrix-representation sectors require an isometric projector, \
                 not a one-hot embedding"
                    .into(),
            ));
        }
        (RegisteredModel::Projected(model), RegisteredModel::Ed(parent)) => {
            model.basis_projector_to(parent)?
        }
        (RegisteredModel::WideProjected(model), RegisteredModel::Wide(parent)) => {
            model.basis_projector_to(parent)?
        }
        (RegisteredModel::Operator(_), _) | (_, RegisteredModel::Operator(_)) => {
            return Err(QmbedError::InvalidOptions(
                "basis-independent operator models do not define basis projectors".into(),
            ));
        }
        _ => {
            return Err(QmbedError::InvalidOptions(
                "basis projectors require source and parent models with the same state storage"
                    .into(),
            ));
        }
    });
    projectors.insert((handle, parent_handle, embedding), Arc::clone(&projector));
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
        .retain(|&(reduced, parent, _embedding), _| reduced != parsed && parent != parsed);
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

impl From<MatrixFormat> for StorageFormat {
    fn from(value: MatrixFormat) -> Self {
        match value {
            MatrixFormat::Dense => Self::Dense,
            MatrixFormat::Csc => Self::Csc,
            MatrixFormat::Csr => Self::Csr,
            MatrixFormat::Dia => Self::Dia,
            MatrixFormat::MatrixFree => Self::MatrixFree,
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
    #[serde(default, alias = "v0")]
    initial_vector: Option<Vec<[f64; 2]>>,
}

impl SolverRequest {
    fn options(&self) -> EigshOptions {
        EigshOptions {
            eigenpairs: self.eigenpairs,
            target: self.target.into(),
            krylov_dimension: self.krylov_dimension,
            tolerance: self.tolerance,
            max_iterations: self.max_iterations,
            seed: self.seed,
        }
    }

    fn initial_vector(&self) -> Option<Vec<Complex64>> {
        self.initial_vector.as_ref().map(|values| {
            values
                .iter()
                .map(|[real, imaginary]| Complex64::new(*real, *imaginary))
                .collect()
        })
    }
}

const fn default_tolerance() -> f64 {
    1.0e-10
}

const fn default_lanczos_tolerance() -> f64 {
    1.0e-13
}

const fn default_expm_degree() -> usize {
    55
}

const fn default_expm_tolerance() -> f64 {
    0.5 * f64::EPSILON
}

const fn default_expm_substeps() -> usize {
    10_000
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

fn run_drive_evolution_json(
    request: &str,
    callback: QmbedDriveCallback,
    context_address: usize,
) -> String {
    let response = serde_json::from_str::<DriveEvolutionRequest>(request)
        .map_err(|error| {
            QmbedError::InvalidOptions(format!("invalid drive evolution request: {error}"))
        })
        .and_then(|request| execute_drive_evolution(request, callback, context_address))
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

fn checked_user_full_dimension(sites: usize, states_per_site: usize) -> Result<usize> {
    if states_per_site < 2 {
        return Err(QmbedError::InvalidSector(
            "a user basis requires at least two states per site".into(),
        ));
    }
    let exponent = u32::try_from(sites).map_err(|_| {
        QmbedError::UnsupportedBackend("the user-basis site count is too large".into())
    })?;
    let dimension = states_per_site.checked_pow(exponent).ok_or_else(|| {
        QmbedError::UnsupportedBackend("the user-basis full dimension overflows usize".into())
    })?;
    if (dimension as u128) > u128::from(u32::MAX) + 1 {
        return Err(QmbedError::UnsupportedBackend(
            "the 32-bit user callback ABI cannot represent the requested local state space".into(),
        ));
    }
    Ok(dimension)
}

fn enumerate_user_states_32(
    request: &UserBasis32RegistrationRequest,
    next_state: Option<QmbedUserNextState32Callback>,
    pre_check: Option<QmbedUserPreCheck32Callback>,
) -> Result<(Vec<u128>, usize)> {
    let sites = u32::try_from(request.sites).map_err(|_| {
        QmbedError::UnsupportedBackend("the 32-bit user callback ABI requires N <= 2^32-1".into())
    })?;
    let full_dimension = checked_user_full_dimension(request.sites, request.states_per_site)?;
    if !request.explicit_states.is_empty() && !request.state_segments.is_empty() {
        return Err(QmbedError::InvalidOptions(
            "a user basis accepts either explicit_states or state_segments, not both".into(),
        ));
    }
    let mut states = if !request.explicit_states.is_empty() {
        request.explicit_states.clone()
    } else if !request.state_segments.is_empty() {
        let callback = next_state.ok_or_else(|| {
            QmbedError::InvalidOptions("state_segments require a next_state callback".into())
        })?;
        let capacity = request
            .state_segments
            .iter()
            .try_fold(0_usize, |total, segment| {
                total.checked_add(segment.count).ok_or_else(|| {
                    QmbedError::UnsupportedBackend(
                        "the user-basis state count overflows usize".into(),
                    )
                })
            })?;
        let mut states = Vec::with_capacity(capacity);
        for segment in &request.state_segments {
            if segment.count == 0 {
                continue;
            }
            let mut state = segment.start;
            states.push(state);
            for counter in 0..segment.count - 1 {
                let counter = u32::try_from(counter).map_err(|_| {
                    QmbedError::UnsupportedBackend(
                        "the 32-bit next_state counter exceeds u32".into(),
                    )
                })?;
                state = callback(state, counter, sites, request.next_state_arguments.as_ptr());
                states.push(state);
            }
        }
        states
    } else {
        (0..full_dimension)
            .map(|state| state as u32)
            .collect::<Vec<_>>()
    };
    if let Some(callback) = pre_check {
        states.retain(|&state| callback(state, sites, request.pre_check_arguments.as_ptr()) != 0);
    }
    Ok((states.into_iter().map(u128::from).collect(), full_dimension))
}

fn build_user_basis_32(
    request: UserBasis32RegistrationRequest,
    operator: QmbedUserOp32Callback,
    next_state: Option<QmbedUserNextState32Callback>,
    pre_check: Option<QmbedUserPreCheck32Callback>,
    map_callbacks: &[QmbedUserMap32Callback],
) -> Result<CommandResult> {
    if map_callbacks.len() != request.symmetries.len() {
        return Err(QmbedError::InvalidOptions(format!(
            "received {} user symmetry callbacks for {} symmetry requests",
            map_callbacks.len(),
            request.symmetries.len()
        )));
    }
    let sites_i32 = i32::try_from(request.sites).map_err(|_| {
        QmbedError::UnsupportedBackend("the user operator callback requires N <= i32::MAX".into())
    })?;
    let (states, full_dimension) = enumerate_user_states_32(&request, next_state, pre_check)?;
    let operator_arguments = Arc::new(request.operator_arguments.clone());
    let mut builder = UserBasis::<u128>::builder(request.sites).states(states);
    let mut allowed_ops: Vec<_> = request.allowed_ops.chars().collect();
    allowed_ops.sort_unstable();
    allowed_ops.dedup();
    if allowed_ops.is_empty() {
        return Err(QmbedError::InvalidOptions(
            "allowed_ops must contain at least one operator character".into(),
        ));
    }
    for symbol in allowed_ops {
        if !symbol.is_ascii() {
            return Err(QmbedError::InvalidOperator(format!(
                "the 32-bit user callback ABI requires ASCII operator symbols, got {symbol:?}"
            )));
        }
        let arguments = Arc::clone(&operator_arguments);
        builder = builder.operator(symbol, move |state, site| {
            let state = u32::try_from(state).map_err(|_| {
                QmbedError::UnsupportedBackend(
                    "a user operator produced a state outside the 32-bit ABI".into(),
                )
            })?;
            let site = i32::try_from(site).map_err(|_| {
                QmbedError::UnsupportedBackend("a user operator site exceeds i32".into())
            })?;
            let mut result = QmbedUserOpResult32 {
                matrix_element: QmbedComplex64 {
                    real: 1.0,
                    imaginary: 0.0,
                },
                state,
            };
            let callback_status = operator(
                &mut result,
                symbol as u8 as i8,
                site,
                sites_i32,
                arguments.as_ptr(),
            );
            if callback_status != 0 {
                return Err(QmbedError::InvalidOperator(format!(
                    "user operator {symbol:?} returned status {callback_status}"
                )));
            }
            let amplitude =
                Complex64::new(result.matrix_element.real, result.matrix_element.imaginary);
            if !amplitude.re.is_finite() || !amplitude.im.is_finite() {
                return Err(QmbedError::InvalidOperator(format!(
                    "user operator {symbol:?} returned a non-finite matrix element"
                )));
            }
            Ok((amplitude.norm() > f64::EPSILON).then_some((u128::from(result.state), amplitude)))
        });
    }
    let callback_basis = builder.build()?;
    let ordering = representative_ordering(request.reverse);
    let constrained = GeneralBasis::from_reducer_with_ordering(
        callback_basis.clone(),
        SymmetryReducer::new(),
        ordering,
    )?;
    let mut reducer = SymmetryReducer::new();
    for ((symmetry, &callback), index) in
        request.symmetries.iter().zip(map_callbacks).zip(0_usize..)
    {
        let arguments = Arc::new(symmetry.arguments.clone());
        let map = ClosureSymmetryMap::new(symmetry.period, move |state: u128| {
            let state = u32::try_from(state).map_err(|_| {
                QmbedError::UnsupportedBackend(format!(
                    "user symmetry {index} received a state outside the 32-bit ABI"
                ))
            })?;
            let mut sign = 1_i8;
            let target = callback(state, sites_i32, &mut sign, arguments.as_ptr());
            if sign == 0 {
                return Err(QmbedError::IncompatibleSymmetry(format!(
                    "user symmetry {index} returned zero phase"
                )));
            }
            Ok((u128::from(target), Complex64::new(f64::from(sign), 0.0)))
        })?;
        reducer = reducer.with_map(map, symmetry.sector);
    }
    let primary =
        GeneralBasis::from_reducer_with_ordering(callback_basis.clone(), reducer, ordering)?;
    let primary: PackedBasis = ordered_basis(primary.into(), request.reverse);
    let dimension = primary.len();
    let constrained: PackedBasis = ordered_basis(constrained.into(), request.reverse);
    let handle = register_user_basis(RegisteredUserBasis {
        primary,
        constrained,
        callback_basis,
        full_dimension: Some(full_dimension),
        reverse: request.reverse,
    })?;
    Ok(CommandResult::UserBasis { handle, dimension })
}

fn run_user_basis_32_registration_json(
    request: &str,
    operator: QmbedUserOp32Callback,
    next_state: Option<QmbedUserNextState32Callback>,
    pre_check: Option<QmbedUserPreCheck32Callback>,
    map_callbacks: &[QmbedUserMap32Callback],
) -> String {
    let response = serde_json::from_str::<UserBasis32RegistrationRequest>(request)
        .map_err(|error| {
            QmbedError::InvalidOptions(format!(
                "invalid 32-bit user-basis registration request: {error}"
            ))
        })
        .and_then(|request| {
            build_user_basis_32(request, operator, next_state, pre_check, map_callbacks)
        })
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

fn user_full_dimension_usize(sites: usize, states_per_site: usize) -> Result<Option<usize>> {
    if states_per_site < 2 {
        return Err(QmbedError::InvalidSector(
            "a user basis requires at least two states per site".into(),
        ));
    }
    let exponent = u32::try_from(sites).map_err(|_| {
        QmbedError::UnsupportedBackend("the user-basis site count is too large".into())
    })?;
    Ok(states_per_site.checked_pow(exponent))
}

fn enumerate_user_states_64(
    request: &UserBasis64RegistrationRequest,
    next_state: Option<QmbedUserNextState64Callback>,
    pre_check: Option<QmbedUserPreCheck64Callback>,
) -> Result<(Vec<u128>, Option<usize>)> {
    let sites = u64::try_from(request.sites).map_err(|_| {
        QmbedError::UnsupportedBackend("the 64-bit user callback ABI cannot represent N".into())
    })?;
    let full_dimension = user_full_dimension_usize(request.sites, request.states_per_site)?;
    if !request.explicit_states.is_empty() && !request.state_segments.is_empty() {
        return Err(QmbedError::InvalidOptions(
            "a user basis accepts either explicit_states or state_segments, not both".into(),
        ));
    }
    let mut states = if !request.explicit_states.is_empty() {
        request.explicit_states.clone()
    } else if !request.state_segments.is_empty() {
        let callback = next_state.ok_or_else(|| {
            QmbedError::InvalidOptions("state_segments require a next_state callback".into())
        })?;
        let capacity = request
            .state_segments
            .iter()
            .try_fold(0_usize, |total, segment| {
                total.checked_add(segment.count).ok_or_else(|| {
                    QmbedError::UnsupportedBackend(
                        "the user-basis state count overflows usize".into(),
                    )
                })
            })?;
        let mut states = Vec::with_capacity(capacity);
        for segment in &request.state_segments {
            if segment.count == 0 {
                continue;
            }
            let mut state = segment.start;
            states.push(state);
            for counter in 0..segment.count - 1 {
                let counter = u64::try_from(counter).map_err(|_| {
                    QmbedError::UnsupportedBackend(
                        "the 64-bit next_state counter exceeds u64".into(),
                    )
                })?;
                state = callback(state, counter, sites, request.next_state_arguments.as_ptr());
                states.push(state);
            }
        }
        states
    } else {
        let dimension = full_dimension.ok_or_else(|| {
            QmbedError::UnsupportedBackend(
                "the unconstrained user basis does not fit the runtime index type; \
                 supply constrained state segments"
                    .into(),
            )
        })?;
        (0..dimension).map(|state| state as u64).collect()
    };
    if let Some(callback) = pre_check {
        states.retain(|&state| callback(state, sites, request.pre_check_arguments.as_ptr()) != 0);
    }
    Ok((states.into_iter().map(u128::from).collect(), full_dimension))
}

fn build_user_basis_64(
    request: UserBasis64RegistrationRequest,
    operator: QmbedUserOp64Callback,
    next_state: Option<QmbedUserNextState64Callback>,
    pre_check: Option<QmbedUserPreCheck64Callback>,
    map_callbacks: &[QmbedUserMap64Callback],
) -> Result<CommandResult> {
    if map_callbacks.len() != request.symmetries.len() {
        return Err(QmbedError::InvalidOptions(format!(
            "received {} user symmetry callbacks for {} symmetry requests",
            map_callbacks.len(),
            request.symmetries.len()
        )));
    }
    let sites_i32 = i32::try_from(request.sites).map_err(|_| {
        QmbedError::UnsupportedBackend("the user operator callback requires N <= i32::MAX".into())
    })?;
    let (states, full_dimension) = enumerate_user_states_64(&request, next_state, pre_check)?;
    let operator_arguments = Arc::new(request.operator_arguments.clone());
    let mut builder = UserBasis::<u128>::builder(request.sites).states(states);
    let mut allowed_ops: Vec<_> = request.allowed_ops.chars().collect();
    allowed_ops.sort_unstable();
    allowed_ops.dedup();
    if allowed_ops.is_empty() {
        return Err(QmbedError::InvalidOptions(
            "allowed_ops must contain at least one operator character".into(),
        ));
    }
    for symbol in allowed_ops {
        if !symbol.is_ascii() {
            return Err(QmbedError::InvalidOperator(format!(
                "the 64-bit user callback ABI requires ASCII operator symbols, got {symbol:?}"
            )));
        }
        let arguments = Arc::clone(&operator_arguments);
        builder = builder.operator(symbol, move |state, site| {
            let state = u64::try_from(state).map_err(|_| {
                QmbedError::UnsupportedBackend(
                    "a user operator produced a state outside the 64-bit ABI".into(),
                )
            })?;
            let site = i32::try_from(site).map_err(|_| {
                QmbedError::UnsupportedBackend("a user operator site exceeds i32".into())
            })?;
            let mut result = QmbedUserOpResult64 {
                matrix_element: QmbedComplex64 {
                    real: 1.0,
                    imaginary: 0.0,
                },
                state,
            };
            let callback_status = operator(
                &mut result,
                symbol as u8 as i8,
                site,
                sites_i32,
                arguments.as_ptr(),
            );
            if callback_status != 0 {
                return Err(QmbedError::InvalidOperator(format!(
                    "user operator {symbol:?} returned status {callback_status}"
                )));
            }
            let amplitude =
                Complex64::new(result.matrix_element.real, result.matrix_element.imaginary);
            if !amplitude.re.is_finite() || !amplitude.im.is_finite() {
                return Err(QmbedError::InvalidOperator(format!(
                    "user operator {symbol:?} returned a non-finite matrix element"
                )));
            }
            Ok((amplitude.norm() > f64::EPSILON).then_some((u128::from(result.state), amplitude)))
        });
    }
    let callback_basis = builder.build()?;
    let ordering = representative_ordering(request.reverse);
    let constrained = GeneralBasis::from_reducer_with_ordering(
        callback_basis.clone(),
        SymmetryReducer::new(),
        ordering,
    )?;
    let mut reducer = SymmetryReducer::new();
    for ((symmetry, &callback), index) in
        request.symmetries.iter().zip(map_callbacks).zip(0_usize..)
    {
        let arguments = Arc::new(symmetry.arguments.clone());
        let map = ClosureSymmetryMap::new(symmetry.period, move |state: u128| {
            let state = u64::try_from(state).map_err(|_| {
                QmbedError::UnsupportedBackend(format!(
                    "user symmetry {index} received a state outside the 64-bit ABI"
                ))
            })?;
            let mut sign = 1_i8;
            let target = callback(state, sites_i32, &mut sign, arguments.as_ptr());
            if sign == 0 {
                return Err(QmbedError::IncompatibleSymmetry(format!(
                    "user symmetry {index} returned zero phase"
                )));
            }
            Ok((u128::from(target), Complex64::new(f64::from(sign), 0.0)))
        })?;
        reducer = reducer.with_map(map, symmetry.sector);
    }
    let primary =
        GeneralBasis::from_reducer_with_ordering(callback_basis.clone(), reducer, ordering)?;
    let primary: PackedBasis = ordered_basis(primary.into(), request.reverse);
    let dimension = primary.len();
    let constrained: PackedBasis = ordered_basis(constrained.into(), request.reverse);
    let handle = register_user_basis(RegisteredUserBasis {
        primary,
        constrained,
        callback_basis,
        full_dimension,
        reverse: request.reverse,
    })?;
    Ok(CommandResult::UserBasis { handle, dimension })
}

fn run_user_basis_64_registration_json(
    request: &str,
    operator: QmbedUserOp64Callback,
    next_state: Option<QmbedUserNextState64Callback>,
    pre_check: Option<QmbedUserPreCheck64Callback>,
    map_callbacks: &[QmbedUserMap64Callback],
) -> String {
    let response = serde_json::from_str::<UserBasis64RegistrationRequest>(request)
        .map_err(|error| {
            QmbedError::InvalidOptions(format!(
                "invalid 64-bit user-basis registration request: {error}"
            ))
        })
        .and_then(|request| {
            build_user_basis_64(request, operator, next_state, pre_check, map_callbacks)
        })
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
    build_basis_with_reducer(request, None)
}

fn basis_storage(request: &BasisRequest) -> Result<StateStorage> {
    match request {
        BasisRequest::Spin {
            sites, spin_twice, ..
        } => get_basis_type(*sites, None, usize::from(*spin_twice) + 1),
        BasisRequest::Boson { .. }
        | BasisRequest::SpinlessFermion { .. }
        | BasisRequest::SpinfulFermion { .. }
        | BasisRequest::Tensor { .. }
        | BasisRequest::Photon { .. }
        | BasisRequest::User { .. } => Ok(StateStorage::U128),
    }
}

fn build_wide_spin_parent<const WORDS: usize>(
    request: &BasisRequest,
) -> Result<WideSpinBasis<WORDS>> {
    let BasisRequest::Spin {
        sites,
        spin_twice,
        up,
        up_sectors,
        momentum,
        parity,
        pauli,
        normalization,
        ..
    } = request
    else {
        return Err(QmbedError::UnsupportedBackend(
            "wide runtime storage currently supports spin bases".into(),
        ));
    };
    if *spin_twice != 1 {
        return Err(QmbedError::UnsupportedBackend(
            "wide runtime storage currently requires spin one-half".into(),
        ));
    }
    if up.is_some() && up_sectors.is_some() {
        return Err(QmbedError::InvalidOptions(
            "spin basis accepts either up or up_sectors, not both".into(),
        ));
    }
    let _ = (momentum, parity);
    let normalization = normalization.map_or_else(
        || {
            if *pauli {
                SpinNormalization::PauliCartesian
            } else {
                SpinNormalization::AngularMomentum
            }
        },
        Into::into,
    );
    Ok(match (up, up_sectors.as_deref()) {
        (Some(up), None) => {
            WideSpinBasis::<WORDS>::with_normalization(*sites, Some(*up), normalization)?
        }
        (None, Some(sectors)) => WideSpinBasis::<WORDS>::from_particle_sectors_with_normalization(
            *sites,
            sectors.iter().copied(),
            normalization,
        )?,
        (None, None) => WideSpinBasis::<WORDS>::with_normalization(*sites, None, normalization)?,
        (Some(_), Some(_)) => unreachable!("conflicting sectors were rejected"),
    })
}

fn build_wide_spin_basis<const WORDS: usize>(request: &BasisRequest) -> Result<WidePackedBasis>
where
    WidePackedBasis: From<WideSpinBasis<WORDS>> + From<GeneralBasis<WideSpinBasis<WORDS>>>,
    LatticeSymmetryMap: SymmetryMap<WideState<WORDS>>,
{
    let BasisRequest::Spin {
        sites,
        momentum,
        parity,
        symmetries,
        matrix_symmetry,
        reverse,
        ..
    } = request
    else {
        unreachable!("wide spin parent validates the request kind")
    };
    if momentum.is_some() || parity.is_some() {
        return Err(QmbedError::UnsupportedBackend(
            "built-in one-dimensional momentum/parity blocks above 128 encoded bits are not \
             implemented; use the general lattice symmetry maps"
                .into(),
        ));
    }
    if matrix_symmetry.is_some() {
        return Err(QmbedError::InvalidOptions(
            "matrix-valued symmetry requests use the projected wide-model path".into(),
        ));
    }
    let basis = build_wide_spin_parent::<WORDS>(request)?;
    let basis: WidePackedBasis = if symmetries.is_empty() {
        basis.into()
    } else {
        GeneralBasis::from_orbit_seeds_with_ordering(
            basis,
            runtime_symmetry_sector(*sites, 2, ExchangeStatistics::Distinguishable, symmetries)?,
            representative_ordering(*reverse),
        )?
        .into()
    };
    Ok(if *reverse { basis.reversed() } else { basis })
}

fn build_wide_basis(request: &BasisRequest) -> Result<WidePackedBasis> {
    match basis_storage(request)? {
        StateStorage::U128 => Err(QmbedError::InvalidOptions(
            "wide basis construction requires more than 128 encoded bits".into(),
        )),
        StateStorage::U256 => build_wide_spin_basis::<4>(request),
        StateStorage::U1024 => build_wide_spin_basis::<16>(request),
        StateStorage::U4096 => build_wide_spin_basis::<64>(request),
        StateStorage::U16384 => build_wide_spin_basis::<256>(request),
    }
}

fn build_basis_with_reducer(
    request: &BasisRequest,
    reducer: Option<&SymmetryReducer<u128>>,
) -> Result<PackedBasis> {
    match request {
        BasisRequest::Spin { .. } => build_spin_basis(request, reducer),
        BasisRequest::Boson { .. } => build_boson_basis(request, reducer),
        BasisRequest::SpinlessFermion { .. } => build_spinless_basis(request, reducer),
        BasisRequest::SpinfulFermion { .. } => build_spinful_basis(request, reducer),
        BasisRequest::Tensor { factors } => {
            if reducer.is_some() {
                return Err(QmbedError::InvalidOptions(
                    "tensor bases do not accept an outer symmetry reducer".into(),
                ));
            }
            PackedTensorBasis::new(
                factors
                    .iter()
                    .map(build_basis)
                    .collect::<Result<Vec<_>>>()?,
            )
            .map(Into::into)
        }
        BasisRequest::Photon {
            matter,
            photon_cutoff,
            total_excitations,
        } => {
            if reducer.is_some() {
                return Err(QmbedError::InvalidOptions(
                    "photon bases do not accept an outer symmetry reducer".into(),
                ));
            }
            PackedPhotonBasis::new(build_basis(matter)?, *photon_cutoff, *total_excitations)
                .map(Into::into)
        }
        BasisRequest::User { handle, view } => {
            if reducer.is_some() {
                return Err(QmbedError::InvalidOptions(
                    "registered user bases do not accept an outer symmetry reducer".into(),
                ));
            }
            registered_user_basis(handle)?.basis(*view)
        }
    }
}

fn build_spin_basis(
    request: &BasisRequest,
    reducer: Option<&SymmetryReducer<u128>>,
) -> Result<PackedBasis> {
    let BasisRequest::Spin {
        sites,
        spin_twice,
        up,
        up_sectors,
        momentum,
        parity,
        pauli,
        normalization,
        symmetries,
        matrix_symmetry,
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
    if matrix_symmetry.is_some() {
        return Err(QmbedError::InvalidOptions(
            "matrix-valued symmetry sectors materialize as projected operator models".into(),
        ));
    }
    let mut builder = SpinBasis1D::builder(*sites).spin_twice(*spin_twice);
    builder = match normalization {
        Some(normalization) => builder.normalization((*normalization).into()),
        None => builder.pauli(*pauli),
    };
    if up.is_some() && up_sectors.is_some() {
        return Err(QmbedError::InvalidOptions(
            "spin basis accepts either up or up_sectors, not both".into(),
        ));
    }
    if let Some(up) = up {
        builder = builder.up(*up);
    } else if let Some(sectors) = up_sectors {
        builder = builder.particle_sectors(sectors.iter().copied());
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
        GeneralBasis::from_orbit_seeds_with_ordering(
            basis,
            match reducer {
                Some(reducer) => reducer.clone(),
                None => runtime_symmetry_sector(
                    *sites,
                    usize::from(*spin_twice) + 1,
                    ExchangeStatistics::Distinguishable,
                    symmetries,
                )?,
            },
            representative_ordering(*reverse),
        )?
        .into()
    };
    Ok(ordered_basis(packed, *reverse))
}

fn build_boson_basis(
    request: &BasisRequest,
    reducer: Option<&SymmetryReducer<u128>>,
) -> Result<PackedBasis> {
    let BasisRequest::Boson {
        sites,
        particles,
        particle_sectors,
        states_per_site,
        symmetries,
        matrix_symmetry,
        reverse,
    } = request
    else {
        unreachable!("build_boson_basis requires a boson request");
    };
    if matrix_symmetry.is_some() {
        return Err(QmbedError::InvalidOptions(
            "matrix-valued symmetry sectors materialize as projected operator models".into(),
        ));
    }
    let mut builder = BosonBasis1D::builder(*sites, *states_per_site);
    if particles.is_some() && particle_sectors.is_some() {
        return Err(QmbedError::InvalidOptions(
            "boson basis accepts either particles or particle_sectors, not both".into(),
        ));
    }
    if let Some(particles) = particles {
        builder = builder.particles(*particles);
    } else if let Some(sectors) = particle_sectors {
        builder = builder.particle_sectors(sectors.iter().copied());
    }
    let basis = builder.build()?;
    let packed = if symmetries.is_empty() {
        basis.into()
    } else {
        GeneralBasis::from_orbit_seeds_with_ordering(
            basis,
            match reducer {
                Some(reducer) => reducer.clone(),
                None => runtime_symmetry_sector(
                    *sites,
                    *states_per_site,
                    ExchangeStatistics::Distinguishable,
                    symmetries,
                )?,
            },
            representative_ordering(*reverse),
        )?
        .into()
    };
    Ok(ordered_basis(packed, *reverse))
}

fn build_spinless_basis(
    request: &BasisRequest,
    reducer: Option<&SymmetryReducer<u128>>,
) -> Result<PackedBasis> {
    let BasisRequest::SpinlessFermion {
        sites,
        particles,
        particle_sectors,
        momentum,
        symmetries,
        matrix_symmetry,
        reverse,
    } = request
    else {
        unreachable!("build_spinless_basis requires a spinless request");
    };
    if matrix_symmetry.is_some() {
        return Err(QmbedError::InvalidOptions(
            "matrix-valued symmetry sectors materialize as projected operator models".into(),
        ));
    }
    if !symmetries.is_empty() && momentum.is_some() {
        return Err(QmbedError::InvalidOptions(
            "built-in and general fermion symmetries cannot be mixed".into(),
        ));
    }
    let mut builder = SpinlessFermionBasis1D::builder(*sites);
    if particles.is_some() && particle_sectors.is_some() {
        return Err(QmbedError::InvalidOptions(
            "spinless basis accepts either particles or particle_sectors, not both".into(),
        ));
    }
    if let Some(particles) = particles {
        builder = builder.particles(*particles);
    } else if let Some(sectors) = particle_sectors {
        builder = builder.particle_sectors(sectors.iter().copied());
    }
    if let Some(momentum) = momentum {
        builder = builder.momentum(*momentum);
    }
    let basis = builder.build()?;
    let packed = if symmetries.is_empty() {
        basis.into()
    } else {
        GeneralBasis::from_orbit_seeds_with_ordering(
            basis,
            match reducer {
                Some(reducer) => reducer.clone(),
                None => {
                    runtime_symmetry_sector(*sites, 2, ExchangeStatistics::Fermionic, symmetries)?
                }
            },
            representative_ordering(*reverse),
        )?
        .into()
    };
    Ok(ordered_basis(packed, *reverse))
}

fn build_spinful_basis(
    request: &BasisRequest,
    reducer: Option<&SymmetryReducer<u128>>,
) -> Result<PackedBasis> {
    let BasisRequest::SpinfulFermion {
        sites,
        particles_up,
        particles_down,
        particle_sectors,
        allowed_local_occupancies,
        symmetries,
        matrix_symmetry,
        reverse,
    } = request
    else {
        unreachable!("build_spinful_basis requires a spinful request");
    };
    if matrix_symmetry.is_some() {
        return Err(QmbedError::InvalidOptions(
            "matrix-valued symmetry sectors materialize as projected operator models".into(),
        ));
    }
    let mut builder = SpinfulFermionBasis1D::builder(*sites);
    if particle_sectors.is_some() && (particles_up.is_some() || particles_down.is_some()) {
        return Err(QmbedError::InvalidOptions(
            "spinful basis accepts fixed particles or particle_sectors, not both".into(),
        ));
    }
    if let Some(sectors) = particle_sectors {
        builder = builder.particle_sectors(sectors.iter().map(|sector| (sector[0], sector[1])));
    } else {
        if let Some(particles) = particles_up {
            builder = builder.particles_up(*particles);
        }
        if let Some(particles) = particles_down {
            builder = builder.particles_down(*particles);
        }
    }
    if let Some(allowed) = allowed_local_occupancies {
        builder = builder.local_occupation_constraint(LocalOccupationConstraint::new(
            2,
            allowed.iter().copied(),
        )?);
    }
    let basis = builder.build()?;
    let packed = if symmetries.is_empty() {
        basis.into()
    } else {
        GeneralBasis::from_orbit_seeds_with_ordering(
            basis,
            match reducer {
                Some(reducer) => reducer.clone(),
                None => runtime_symmetry_sector(
                    sites.checked_mul(2).ok_or_else(|| {
                        QmbedError::UnsupportedBackend("spinful orbital count is too large".into())
                    })?,
                    2,
                    ExchangeStatistics::Fermionic,
                    symmetries,
                )?,
            },
            representative_ordering(*reverse),
        )?
        .into()
    };
    Ok(ordered_basis(packed, *reverse))
}

fn runtime_symmetry_sector<State>(
    encoded_sites: usize,
    states_per_site: usize,
    statistics: ExchangeStatistics,
    requests: &[SymmetryRequest],
) -> Result<SymmetrySector<State>>
where
    LatticeSymmetryMap: SymmetryMap<State>,
{
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

fn matrix_symmetry_request(request: &BasisRequest) -> Option<&MatrixSymmetryRequest> {
    match request {
        BasisRequest::Spin {
            matrix_symmetry, ..
        }
        | BasisRequest::Boson {
            matrix_symmetry, ..
        }
        | BasisRequest::SpinlessFermion {
            matrix_symmetry, ..
        }
        | BasisRequest::SpinfulFermion {
            matrix_symmetry, ..
        } => matrix_symmetry.as_ref(),
        BasisRequest::Tensor { .. } | BasisRequest::Photon { .. } | BasisRequest::User { .. } => {
            None
        }
    }
}

fn matrix_symmetry_geometry(request: &BasisRequest) -> Result<(usize, usize, ExchangeStatistics)> {
    match request {
        BasisRequest::Spin {
            sites, spin_twice, ..
        } => Ok((
            *sites,
            usize::from(*spin_twice) + 1,
            ExchangeStatistics::Distinguishable,
        )),
        BasisRequest::Boson {
            sites,
            states_per_site,
            ..
        } => Ok((
            *sites,
            *states_per_site,
            ExchangeStatistics::Distinguishable,
        )),
        BasisRequest::SpinlessFermion { sites, .. } => {
            Ok((*sites, 2, ExchangeStatistics::Fermionic))
        }
        BasisRequest::SpinfulFermion { sites, .. } => Ok((
            sites.checked_mul(2).ok_or_else(|| {
                QmbedError::UnsupportedBackend("spinful orbital count is too large".into())
            })?,
            2,
            ExchangeStatistics::Fermionic,
        )),
        BasisRequest::Tensor { .. } | BasisRequest::Photon { .. } | BasisRequest::User { .. } => {
            Err(QmbedError::InvalidOptions(
                "matrix-valued symmetry sectors currently require one packed lattice basis".into(),
            ))
        }
    }
}

fn without_runtime_symmetries(request: &BasisRequest, unrestricted_parent: bool) -> BasisRequest {
    let mut request = request.clone();
    match &mut request {
        BasisRequest::Spin {
            up,
            up_sectors,
            momentum,
            parity,
            symmetries,
            matrix_symmetry,
            ..
        } => {
            *momentum = None;
            *parity = None;
            symmetries.clear();
            *matrix_symmetry = None;
            if unrestricted_parent {
                *up = None;
                *up_sectors = None;
            }
        }
        BasisRequest::Boson {
            particles,
            particle_sectors,
            symmetries,
            matrix_symmetry,
            ..
        } => {
            symmetries.clear();
            *matrix_symmetry = None;
            if unrestricted_parent {
                *particles = None;
                *particle_sectors = None;
            }
        }
        BasisRequest::SpinlessFermion {
            particles,
            particle_sectors,
            momentum,
            symmetries,
            matrix_symmetry,
            ..
        } => {
            *momentum = None;
            symmetries.clear();
            *matrix_symmetry = None;
            if unrestricted_parent {
                *particles = None;
                *particle_sectors = None;
            }
        }
        BasisRequest::SpinfulFermion {
            particles_up,
            particles_down,
            particle_sectors,
            symmetries,
            matrix_symmetry,
            ..
        } => {
            symmetries.clear();
            *matrix_symmetry = None;
            if unrestricted_parent {
                *particles_up = None;
                *particles_down = None;
                *particle_sectors = None;
            }
        }
        BasisRequest::Tensor { .. } | BasisRequest::Photon { .. } | BasisRequest::User { .. } => {}
    }
    request
}

fn matrix_symmetry_reducer<State>(request: &BasisRequest) -> Result<MatrixSymmetryReducer<State>>
where
    LatticeSymmetryMap: SymmetryMap<State>,
{
    let specification = matrix_symmetry_request(request).ok_or_else(|| {
        QmbedError::InvalidOptions("basis request has no matrix symmetry representation".into())
    })?;
    let scalar_symmetries = match request {
        BasisRequest::Spin { symmetries, .. }
        | BasisRequest::Boson { symmetries, .. }
        | BasisRequest::SpinlessFermion { symmetries, .. }
        | BasisRequest::SpinfulFermion { symmetries, .. } => symmetries,
        BasisRequest::Tensor { .. } | BasisRequest::Photon { .. } | BasisRequest::User { .. } => {
            unreachable!("matrix representation was checked above")
        }
    };
    if !scalar_symmetries.is_empty() {
        return Err(QmbedError::InvalidOptions(
            "scalar and matrix symmetry generators must be combined into one matrix representation"
                .into(),
        ));
    }
    let (encoded_sites, states_per_site, statistics) = matrix_symmetry_geometry(request)?;
    let mut reducer =
        MatrixSymmetryReducer::new(specification.dimension, specification.selected_row)?;
    for generator in &specification.generators {
        if generator.destinations.len() != encoded_sites {
            return Err(QmbedError::InvalidOptions(format!(
                "matrix symmetry map has {} sites, expected {encoded_sites}",
                generator.destinations.len()
            )));
        }
        if generator.matrix.len() != specification.dimension
            || generator
                .matrix
                .iter()
                .any(|row| row.len() != specification.dimension)
        {
            return Err(QmbedError::InvalidSector(format!(
                "matrix symmetry generator must have shape {}x{}",
                specification.dimension, specification.dimension
            )));
        }
        let map = LatticeSymmetryMap::new(
            states_per_site,
            generator.destinations.clone(),
            generator.local_permutations.clone(),
            statistics,
        )?;
        let matrix = generator
            .matrix
            .iter()
            .flatten()
            .map(|[real, imaginary]| Complex64::new(*real, *imaginary))
            .collect::<Vec<_>>();
        reducer = reducer.with_map(map, matrix)?;
    }
    Ok(reducer)
}

fn matrix_symmetry_subspace(
    request: &BasisRequest,
) -> Result<(PackedBasis, MatrixSymmetrySubspace<u128>)> {
    let reducer = matrix_symmetry_reducer::<u128>(request)?;
    let seeds = build_basis(&without_runtime_symmetries(request, false))?;
    let parent = build_basis(&without_runtime_symmetries(request, true))?;
    let subspace = reducer.subspace(&seeds)?;
    Ok((parent, subspace))
}

fn wide_matrix_symmetry_subspace(
    request: &BasisRequest,
) -> Result<(WidePackedBasis, MatrixSymmetrySubspace<ErasedState>)> {
    let reducer = matrix_symmetry_reducer::<ErasedState>(request)?;
    let seeds = build_wide_basis(&without_runtime_symmetries(request, false))?;
    let subspace = reducer.subspace(&seeds)?;
    let parent = seeds.explicit_spin_subspace(subspace.physical_states())?;
    Ok((parent, subspace))
}

fn request_symmetry_reducer(request: &BasisRequest) -> Result<SymmetryReducer<u128>> {
    match request {
        BasisRequest::Spin {
            sites,
            spin_twice,
            momentum,
            parity,
            symmetries,
            ..
        } => {
            if momentum.is_some() || parity.is_some() {
                return Err(QmbedError::InvalidOptions(
                    "deferred basis plans require serializable symmetry generators; \
                     encode built-in momentum/parity as lattice maps"
                        .into(),
                ));
            }
            runtime_symmetry_sector(
                *sites,
                usize::from(*spin_twice) + 1,
                ExchangeStatistics::Distinguishable,
                symmetries,
            )
        }
        BasisRequest::Boson {
            sites,
            states_per_site,
            symmetries,
            ..
        } => runtime_symmetry_sector(
            *sites,
            *states_per_site,
            ExchangeStatistics::Distinguishable,
            symmetries,
        ),
        BasisRequest::SpinlessFermion {
            sites,
            momentum,
            symmetries,
            ..
        } => {
            if momentum.is_some() {
                return Err(QmbedError::InvalidOptions(
                    "deferred basis plans require momentum as a serializable lattice map".into(),
                ));
            }
            runtime_symmetry_sector(*sites, 2, ExchangeStatistics::Fermionic, symmetries)
        }
        BasisRequest::SpinfulFermion {
            sites, symmetries, ..
        } => runtime_symmetry_sector(
            sites.checked_mul(2).ok_or_else(|| {
                QmbedError::UnsupportedBackend("spinful orbital count is too large".into())
            })?,
            2,
            ExchangeStatistics::Fermionic,
            symmetries,
        ),
        BasisRequest::Tensor { .. } => Err(QmbedError::InvalidOptions(
            "deferred symmetry plans are defined on tensor factors, not on the tensor product"
                .into(),
        )),
        BasisRequest::Photon { .. } => Err(QmbedError::InvalidOptions(
            "deferred symmetry plans are defined on the matter basis, not on the matter-photon product"
                .into(),
        )),
        BasisRequest::User { .. } => Err(QmbedError::InvalidOptions(
            "callback-defined user bases are materialized when their callback ABI is registered"
                .into(),
        )),
    }
}

fn ordered_basis(basis: PackedBasis, reverse: bool) -> PackedBasis {
    if reverse { basis.reversed() } else { basis }
}

const fn representative_ordering(reverse: bool) -> RepresentativeOrdering {
    if reverse {
        RepresentativeOrdering::Maximum
    } else {
        RepresentativeOrdering::Minimum
    }
}

fn request_representative_ordering(request: &BasisRequest) -> RepresentativeOrdering {
    match request {
        BasisRequest::Spin { reverse, .. }
        | BasisRequest::Boson { reverse, .. }
        | BasisRequest::SpinlessFermion { reverse, .. }
        | BasisRequest::SpinfulFermion { reverse, .. } => representative_ordering(*reverse),
        BasisRequest::Tensor { .. } | BasisRequest::Photon { .. } | BasisRequest::User { .. } => {
            RepresentativeOrdering::Minimum
        }
    }
}

fn build_model(
    basis: &BasisRequest,
    terms: Vec<TermRequest>,
    checks: AssemblyChecks,
    site_permutation: Option<Vec<usize>>,
) -> Result<PackedEdModel> {
    build_parameterized_model(basis, terms, Vec::new(), checks, site_permutation)
}

fn build_parameterized_model(
    basis: &BasisRequest,
    terms: Vec<TermRequest>,
    components: Vec<TermComponentRequest>,
    checks: AssemblyChecks,
    site_permutation: Option<Vec<usize>>,
) -> Result<PackedEdModel> {
    build_parameterized_model_on_basis(
        build_basis(basis)?,
        terms,
        components,
        checks,
        site_permutation,
    )
}

fn build_parameterized_model_on_basis<B>(
    basis: B,
    terms: Vec<TermRequest>,
    components: Vec<TermComponentRequest>,
    checks: AssemblyChecks,
    site_permutation: Option<Vec<usize>>,
) -> Result<EdModel<B>>
where
    B: Basis + Clone,
    B::State: Hash + Ord + 'static,
{
    let terms = terms
        .into_iter()
        .map(typed_term)
        .collect::<Result<Vec<_>>>()?;
    let components = components
        .into_iter()
        .map(|component| {
            let terms = component
                .terms
                .into_iter()
                .map(typed_term)
                .collect::<Result<Vec<_>>>()?;
            Ok(match component.default {
                Some([real, imaginary]) => PackedTermComponent::with_default(
                    component.name,
                    terms,
                    Complex64::new(real, imaginary),
                ),
                None => PackedTermComponent::required(component.name, terms),
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let model = EdModel::new(basis, terms)
        .with_checks(checks)
        .with_components(components)?;
    match site_permutation {
        Some(permutation) => model.with_site_permutation(&permutation),
        None => Ok(model),
    }
}

fn build_registered_parameterized_model(
    basis: &BasisRequest,
    terms: Vec<TermRequest>,
    components: Vec<TermComponentRequest>,
    checks: AssemblyChecks,
    site_permutation: Option<Vec<usize>>,
) -> Result<RegisteredModel> {
    if matrix_symmetry_request(basis).is_none() {
        if basis_storage(basis)? != StateStorage::U128 {
            let basis = build_wide_basis(basis)?;
            return build_parameterized_model_on_basis(
                basis,
                terms,
                components,
                checks,
                site_permutation,
            )
            .map(Into::into);
        }
        return build_parameterized_model(basis, terms, components, checks, site_permutation)
            .map(Into::into);
    }
    if basis_storage(basis)? != StateStorage::U128 {
        let (parent, subspace) = wide_matrix_symmetry_subspace(basis)?;
        let projector = subspace.projector(&parent, MatrixFormat::Csc)?;
        let basis_projector = BasisProjector::from_operator(&projector, default_tolerance())?;
        let labels = subspace.labels().to_vec();
        let columns = subspace.columns().to_vec();
        let parent_model = build_parameterized_model_on_basis(
            parent,
            terms,
            components,
            checks,
            site_permutation,
        )?;
        let operator = parent_model
            .operator_model(MatrixFormat::Csc)?
            .projected_by(&projector)?;
        return Ok(RegisteredModel::WideProjected(Box::new(
            RegisteredProjectedBasisModel {
                operator: Box::new(operator),
                assembly_parent: Box::new(parent_model),
                projector,
                basis_projector,
                labels,
                columns,
            },
        )));
    }
    let (parent, subspace) = matrix_symmetry_subspace(basis)?;
    let projector = subspace.projector(&parent, MatrixFormat::Csc)?;
    let basis_projector = BasisProjector::from_operator(&projector, default_tolerance())?;
    let labels = subspace.labels().to_vec();
    let columns = subspace.columns().to_vec();
    let parent_model =
        build_parameterized_model_on_basis(parent, terms, components, checks, site_permutation)?;
    let operator = parent_model
        .operator_model(MatrixFormat::Csc)?
        .projected_by(&projector)?;
    Ok(RegisteredModel::Projected(Box::new(
        RegisteredProjectedBasisModel {
            operator: Box::new(operator),
            assembly_parent: Box::new(parent_model),
            projector,
            basis_projector,
            labels,
            columns,
        },
    )))
}

fn matrix_operator(request: MatrixRequest, format: MatrixFormat) -> Result<Operator> {
    Operator::from_triplets(
        request.shape[0],
        request.shape[1],
        request.entries.into_iter().map(|entry| {
            (
                entry.row,
                entry.column,
                Complex64::new(entry.value[0], entry.value[1]),
            )
        }),
        format,
    )
}

fn build_operator_model(
    static_operator: Option<MatrixRequest>,
    components: Vec<OperatorComponentRequest>,
    basis: Option<&BasisRequest>,
    checks: AssemblyChecks,
    site_permutation: Option<Vec<usize>>,
) -> Result<PackedOperatorModel> {
    let needs_basis = components
        .iter()
        .any(|component| matches!(component, OperatorComponentRequest::Terms(_)));
    let basis_model = match (needs_basis, basis) {
        (true, Some(basis)) => Some(build_model(basis, Vec::new(), checks, site_permutation)?),
        (true, None) => {
            return Err(QmbedError::InvalidOptions(
                "local-term operator components require a basis".into(),
            ));
        }
        (false, _) => None,
    };
    let component_checks = AssemblyChecks {
        hermiticity: false,
        particle_conservation: checks.particle_conservation,
        symmetry_compatibility: checks.symmetry_compatibility,
    };
    let components = components
        .into_iter()
        .map(|component| {
            let (name, operator, default) = match component {
                OperatorComponentRequest::Matrix(component) => (
                    component.name,
                    matrix_operator(component.operator, MatrixFormat::Csc)?,
                    component.default,
                ),
                OperatorComponentRequest::Terms(component) => {
                    let terms = component
                        .terms
                        .into_iter()
                        .map(typed_term)
                        .collect::<Result<Vec<_>>>()?;
                    (
                        component.name,
                        basis_model
                            .as_ref()
                            .expect("term components require a basis model")
                            .assemble_terms(terms, component_checks, MatrixFormat::Csc)?,
                        component.default,
                    )
                }
            };
            Ok(match default {
                Some([real, imaginary]) => {
                    QuantumComponent::with_default(name, operator, Complex64::new(real, imaginary))
                }
                None => QuantumComponent::required(name, operator),
            })
        })
        .collect::<Result<Vec<_>>>()?;
    match (static_operator, components.is_empty()) {
        (Some(operator), true) => {
            PackedOperatorModel::new(matrix_operator(operator, MatrixFormat::Csc)?)
        }
        (Some(operator), false) => PackedOperatorModel::with_components(
            matrix_operator(operator, MatrixFormat::Csc)?,
            components,
        ),
        (None, false) => PackedOperatorModel::parameterized(components, MatrixFormat::Csc),
        (None, true) => Err(QmbedError::InvalidOptions(
            "an operator model requires a fixed operator or named components".into(),
        )),
    }
}

fn archive_component_results(archive: &OperatorArchive) -> Vec<ArchiveComponentResult> {
    archive
        .iter()
        .map(|(name, entry)| ArchiveComponentResult {
            name: name.to_string(),
            format: entry.operator.format().into(),
            default: entry.default.map(|value| [value.re, value.im]),
        })
        .collect()
}

fn archive_metadata(archive: &OperatorArchive) -> HashMap<String, String> {
    archive
        .metadata()
        .map(|(key, value)| (key.to_string(), value.to_string()))
        .collect()
}

fn command_load_operator_archive(path: String) -> Result<CommandResult> {
    let archive = load_zip(&path)?;
    let components = archive_component_results(&archive);
    let metadata = archive_metadata(&archive);
    let model = PackedOperatorModel::from_component_archive(archive, MatrixFormat::Csc)?;
    let dimension = model.dimension();
    let handle = register_model(model)?;
    Ok(CommandResult::ArchivedModel {
        handle,
        dimension,
        components,
        metadata,
    })
}

fn create_projected_block_model(
    blocks: Vec<ProjectedBlockRequest>,
    tolerance: f64,
    format: StorageFormat,
) -> Result<CommandResult> {
    let blocks = blocks
        .into_iter()
        .map(|block| {
            let model = registered_model(&block.handle)?;
            Ok((
                model.operator_model(MatrixFormat::Csc)?,
                matrix_operator(block.projector, MatrixFormat::Csc)?,
            ))
        })
        .collect::<Result<Vec<_>>>()?;
    let model = PackedOperatorModel::from_projected_blocks(blocks, tolerance, format.into())?;
    let dimension = model.dimension();
    let handle = register_model(model)?;
    Ok(CommandResult::Model { handle, dimension })
}

fn create_block_model(handles: Vec<String>, format: StorageFormat) -> Result<CommandResult> {
    let blocks = handles
        .into_iter()
        .map(|handle| registered_model(&handle)?.operator_model(MatrixFormat::Csc))
        .collect::<Result<Vec<_>>>()?;
    let model = PackedOperatorModel::from_blocks(blocks, format.into())?;
    let dimension = model.dimension();
    let handle = register_model(model)?;
    Ok(CommandResult::Model { handle, dimension })
}

fn registered_save_operator_archive(
    handle: &str,
    path: String,
    formats: HashMap<String, StorageFormat>,
    metadata: HashMap<String, String>,
) -> Result<CommandResult> {
    let model = registered_model(handle)?;
    let formats = formats
        .into_iter()
        .map(|(name, format)| (name, format.into()))
        .collect();
    let mut archive = model.component_archive(&formats)?;
    for (key, value) in metadata {
        archive.insert_metadata(key, value)?;
    }
    let components = archive_component_results(&archive);
    let metadata = archive_metadata(&archive);
    save_zip(&path, &archive)?;
    Ok(CommandResult::Archive {
        path,
        components,
        metadata,
    })
}

fn create_basis_plan(
    basis: BasisRequest,
    site_permutation: Option<Vec<usize>>,
    checks: AssemblyChecks,
) -> Result<CommandResult> {
    let reducer = request_symmetry_reducer(&basis)?;
    let representative_ordering = request_representative_ordering(&basis);
    let handle = register_basis_plan(RegisteredBasisPlan {
        basis,
        reducer,
        representative_ordering,
        site_permutation,
        checks,
    })?;
    Ok(CommandResult::BasisPlan { handle })
}

fn materialize_basis_plan(handle: &str) -> Result<CommandResult> {
    let plan = registered_basis_plan(handle)?;
    let basis = build_basis_with_reducer(&plan.basis, Some(&plan.reducer))?;
    let model = PackedEdModel::new(basis, []).with_checks(plan.checks);
    let model = match &plan.site_permutation {
        Some(permutation) => model.with_site_permutation(permutation)?,
        None => model,
    };
    let dimension = model.dimension();
    let handle = register_model(model)?;
    Ok(CommandResult::Model { handle, dimension })
}

fn command_bra_ket_terms_plan(
    plan: &RegisteredBasisPlan,
    terms: Vec<TermRequest>,
    kets: Vec<String>,
) -> Result<CommandResult> {
    let kets = parse_physical_states(kets)?;
    let parent_basis = build_basis_with_reducer(&plan.basis, Some(&SymmetryReducer::new()))?;
    let parent = PackedEdModel::new(parent_basis, []).with_checks(AssemblyChecks::none());
    let parent = match &plan.site_permutation {
        Some(permutation) => parent.with_site_permutation(permutation)?,
        None => parent,
    };
    let terms = terms
        .into_iter()
        .map(typed_term)
        .collect::<Result<Vec<_>>>()?;
    let raw = parent.bra_ket_terms(terms, &kets)?;
    let mut entries = Vec::new();
    for (input, (&ket, transitions)) in kets.iter().zip(raw).enumerate() {
        let (source, source_states) = plan
            .reducer
            .orbit_with_states_and_ordering(ket, plan.representative_ordering)?;
        let mut source_selected = false;
        for state in source_states {
            if parent.basis().reduction_image(state)?.is_some() {
                source_selected = true;
                break;
            }
        }
        if !source.is_compatible() || !source_selected {
            continue;
        }
        let mut reduced = HashMap::<u128, Complex64>::new();
        for transition in transitions {
            let (target, target_states) = plan
                .reducer
                .orbit_with_states_and_ordering(transition.bra, plan.representative_ordering)?;
            let mut target_selected = false;
            for state in target_states {
                if parent.basis().reduction_image(state)?.is_some() {
                    target_selected = true;
                    break;
                }
            }
            if !target_selected {
                continue;
            }
            let Some(phase) = target.phase() else {
                continue;
            };
            let factor =
                (source.orbit_size() as f64 / target.orbit_size() as f64).sqrt() * phase.conj();
            *reduced
                .entry(*target.representative())
                .or_insert(Complex64::new(0.0, 0.0)) += transition.matrix_element * factor;
        }
        let mut reduced: Vec<_> = reduced
            .into_iter()
            .filter(|(_, value)| value.norm() > f64::EPSILON)
            .collect();
        reduced.sort_by_key(|(bra, _)| *bra);
        entries.extend(reduced.into_iter().map(|(bra, value)| TransitionEntry {
            input,
            bra: bra.to_string(),
            ket: ket.to_string(),
            value: [value.re, value.im],
        }));
    }
    Ok(CommandResult::Transitions { entries })
}

fn solve_model(
    model: &PackedEdModel,
    format: StorageFormat,
    solver: &SolverRequest,
) -> Result<SolveResult> {
    let include_vectors = solver.eigenvectors;
    let options = solver.options();
    let initial = solver.initial_vector();
    let result = match initial.as_deref() {
        Some(initial) => model.eigsh_with_initial(format.into(), options, initial),
        None => model.eigsh(format.into(), options),
    }?;
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

fn command_operator(
    model: &RegisteredModel,
    parameters: HashMap<String, [f64; 2]>,
    format: StorageFormat,
) -> Result<CommandResult> {
    let operator = model.materialize(&complex_parameters(parameters), format.into())?;
    Ok(command_operator_value(&operator, format))
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

fn matrix_payload(operator: &Operator, format: StorageFormat) -> MatrixPayload {
    let (rows, columns) = qmbed::operator::LinearOperator::shape(operator);
    MatrixPayload {
        shape: [rows, columns],
        format,
        entries: operator
            .triplets()
            .into_iter()
            .map(|(row, column, value)| MatrixEntry {
                row,
                column,
                value: [value.re, value.im],
            })
            .collect(),
    }
}

fn command_analyze_floquet(
    steps: Vec<FloquetStepRequest>,
    period: Option<f64>,
    format: StorageFormat,
) -> Result<CommandResult> {
    let steps = steps
        .into_iter()
        .map(|step| {
            let operator = evaluate_operator_expression(step.expression)?;
            DriveStep::new(Arc::new(operator), step.duration)
        })
        .collect::<Result<Vec<_>>>()?;
    let floquet = Floquet::new(steps)?;
    let floquet = match period {
        Some(period) => floquet.with_period(period)?,
        None => floquet,
    };
    let analysis = floquet.analyze(format.into())?;
    Ok(command_floquet_analysis(analysis, format))
}

fn command_floquet_analysis(analysis: FloquetAnalysis, format: StorageFormat) -> CommandResult {
    CommandResult::FloquetAnalysis {
        period: analysis.period,
        protocol_duration: analysis.protocol_duration,
        unitary: matrix_payload(&analysis.unitary, format),
        quasienergies: analysis.eigensystem.quasienergies,
        eigenvalues: analysis
            .eigensystem
            .eigenvalues
            .into_iter()
            .map(|value| [value.re, value.im])
            .collect(),
        eigenvectors: analysis
            .eigensystem
            .eigenvectors
            .into_iter()
            .map(|vector| {
                vector
                    .into_iter()
                    .map(|value| [value.re, value.im])
                    .collect()
            })
            .collect(),
        residuals: analysis.eigensystem.residuals,
        effective_hamiltonian: matrix_payload(&analysis.effective_hamiltonian, format),
    }
}

fn evaluate_operator_expression(expression: OperatorExpressionRequest) -> Result<Operator> {
    match expression {
        OperatorExpressionRequest::Model {
            handle,
            parameters,
            action,
        } => {
            let model = registered_model(&handle)?;
            let operator = model.materialize(&complex_parameters(parameters), MatrixFormat::Csc)?;
            transform_operator(operator, action)
        }
        OperatorExpressionRequest::Matrix { operator, action } => {
            transform_operator(matrix_operator(operator, MatrixFormat::Csc)?, action)
        }
        OperatorExpressionRequest::Scale {
            coefficient: [real, imaginary],
            operand,
        } => evaluate_operator_expression(*operand)?.scaled(Complex64::new(real, imaginary)),
        OperatorExpressionRequest::Transform { action, operand } => {
            transform_operator(evaluate_operator_expression(*operand)?, action)
        }
        OperatorExpressionRequest::Binary {
            operation,
            left,
            right,
        } => {
            let left = evaluate_operator_expression(*left)?;
            let right = evaluate_operator_expression(*right)?;
            match operation {
                AlgebraOperationRequest::Add => left.add(&right),
                AlgebraOperationRequest::Subtract => left.subtract(&right),
                AlgebraOperationRequest::Product => left.product(&right),
            }
        }
    }
}

fn command_lanczos_operator(
    expression: OperatorExpressionRequest,
    initial: Vec<[f64; 2]>,
    krylov_dimension: usize,
    tolerance: f64,
) -> Result<CommandResult> {
    let operator = evaluate_operator_expression(expression)?;
    let initial = initial
        .into_iter()
        .map(|[real, imaginary]| Complex64::new(real, imaginary))
        .collect::<Vec<_>>();
    let decomposition = lanczos_ritz(
        &operator,
        &initial,
        LanczosOptions {
            krylov_dimension,
            tolerance,
        },
    )?;
    let projected_dimension = decomposition.eigenvalues.len();
    let initial_norm = decomposition.decomposition.initial_norm;
    let eigenvalues = decomposition.eigenvalues.clone();
    let eigenvectors = decomposition.eigenvectors.clone();
    let handle = register_lanczos(decomposition)?;
    let result = CommandResult::Lanczos {
        handle,
        dimension: initial.len(),
        krylov_dimension: projected_dimension,
        initial_norm,
        eigenvalues,
        eigenvectors,
    };
    Ok(result)
}

fn command_apply_operator_expression(
    expression: OperatorExpressionRequest,
    vectors: Vec<Vec<[f64; 2]>>,
) -> Result<CommandResult> {
    let operator = evaluate_operator_expression(expression)?;
    let shape = LinearOperator::shape(&operator);
    let vectors = vectors
        .into_iter()
        .map(|vector| {
            vector
                .into_iter()
                .map(|[real, imaginary]| Complex64::new(real, imaginary))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    if vectors.is_empty() || vectors.iter().any(|vector| vector.len() != shape.1) {
        return Err(QmbedError::DimensionMismatch(
            "operator expression inputs must be a nonempty batch matching its columns".into(),
        ));
    }
    let mut outputs = Vec::with_capacity(vectors.len());
    for vector in vectors {
        let mut output = vec![Complex64::new(0.0, 0.0); shape.0];
        operator.apply(&vector, &mut output)?;
        outputs.push(output);
    }
    Ok(command_vectors_with_dimension(shape.0, outputs))
}

fn command_create_operator_expression_model(
    expression: OperatorExpressionRequest,
    format: StorageFormat,
) -> Result<CommandResult> {
    let operator = evaluate_operator_expression(expression)?.converted(format.into())?;
    let model = PackedOperatorModel::new(operator)?;
    let dimension = model.dimension();
    let handle = register_model(model)?;
    Ok(CommandResult::Model { handle, dimension })
}

fn command_inspect_operator_expression(
    expression: OperatorExpressionRequest,
) -> Result<CommandResult> {
    let operator = evaluate_operator_expression(expression)?;
    let shape = LinearOperator::shape(&operator);
    let trace = (shape.0 == shape.1)
        .then(|| operator.trace())
        .transpose()?
        .map(|value| [value.re, value.im]);
    Ok(CommandResult::OperatorSummary {
        shape: [shape.0, shape.1],
        diagonal: complex_payload(operator.diagonal()),
        trace,
        nonzeros: operator.nnz(),
    })
}

fn command_expm_operator_expression(
    expression: OperatorExpressionRequest,
    [real, imaginary]: [f64; 2],
    vectors: Vec<Vec<[f64; 2]>>,
    max_degree: usize,
    tolerance: f64,
    max_substeps: usize,
    threads: Option<usize>,
) -> Result<CommandResult> {
    let operator = evaluate_operator_expression(expression)?;
    let shape = LinearOperator::shape(&operator);
    if shape.0 != shape.1 {
        return Err(QmbedError::DimensionMismatch(
            "operator exponential requires a square expression".into(),
        ));
    }
    let plan = ExpmMultiplyParallel::new(
        Arc::new(operator),
        Complex64::new(real, imaginary),
        max_degree,
        tolerance,
        max_substeps,
    )?;
    let vectors = complex_vectors(vectors);
    if vectors.is_empty() {
        return Err(QmbedError::InvalidOptions(
            "operator exponential requires a nonempty vector batch".into(),
        ));
    }
    let threads = threads.unwrap_or_else(|| {
        std::thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get)
    });
    let runtime = CpuRuntime::new(threads)?;
    let vectors = plan.apply_batch_with_runtime(&runtime, &vectors)?;
    Ok(command_vectors_with_dimension(shape.0, vectors))
}

fn command_eigsh_operator_expression(
    expression: OperatorExpressionRequest,
    format: StorageFormat,
    solver: &SolverRequest,
) -> Result<CommandResult> {
    let operator = evaluate_operator_expression(expression)?.converted(format.into())?;
    let include_vectors = solver.eigenvectors;
    let options = solver.options();
    let initial = solver.initial_vector();
    let result = match initial.as_deref() {
        Some(initial) => eigsh_with_initial(&operator, options, initial),
        None => eigsh(&operator, options),
    }?;
    Ok(command_eigensystem(
        qmbed::operator::LinearOperator::shape(&operator).0,
        result,
        include_vectors,
    ))
}

fn command_lanczos_combine(
    lanczos_handle: &str,
    coefficients: Vec<[f64; 2]>,
) -> Result<CommandResult> {
    let decomposition = registered_lanczos(lanczos_handle)?;
    let coefficients = coefficients
        .into_iter()
        .map(|[real, imaginary]| Complex64::new(real, imaginary))
        .collect::<Vec<_>>();
    let vector = decomposition.linear_combination(&coefficients)?;
    Ok(command_vectors_with_dimension(vector.len(), vec![vector]))
}

fn command_lanczos_exponential(
    lanczos_handle: &str,
    [real, imaginary]: [f64; 2],
) -> Result<CommandResult> {
    let decomposition = registered_lanczos(lanczos_handle)?;
    let vector = decomposition.exponential_action(Complex64::new(real, imaginary))?;
    Ok(command_vectors_with_dimension(vector.len(), vec![vector]))
}

fn command_lanczos_thermal(
    lanczos_handle: Option<&str>,
    method: ThermalLanczosMethodRequest,
    eigenvalues: Vec<f64>,
    eigenvectors: Vec<Vec<f64>>,
    inverse_temperatures: Vec<f64>,
    observables: Vec<ThermalObservableRequest>,
) -> Result<CommandResult> {
    let dimension = eigenvalues.len();
    if eigenvectors.len() != dimension || eigenvectors.iter().any(|row| row.len() != dimension) {
        return Err(QmbedError::DimensionMismatch(
            "Lanczos Ritz eigenvectors must be a square row-major matrix".into(),
        ));
    }
    let eigenvector_columns = (0..dimension)
        .map(|column| {
            (0..dimension)
                .map(|row| eigenvectors[row][column])
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let method = ThermalLanczosMethod::from(method);
    let mut projected = Vec::with_capacity(observables.len());
    let mut native_observables = Vec::new();
    for observable in observables {
        match (observable.expression, observable.matrix_elements.is_empty()) {
            (Some(expression), true) => {
                native_observables
                    .push((observable.name, evaluate_operator_expression(expression)?));
            }
            (None, false) => projected.push(ProjectedThermalObservable {
                name: observable.name,
                matrix_elements: observable
                    .matrix_elements
                    .into_iter()
                    .map(|[real, imaginary]| Complex64::new(real, imaginary))
                    .collect(),
            }),
            _ => {
                return Err(QmbedError::InvalidOptions(
                    "each thermal observable needs exactly one expression or projected matrix"
                        .into(),
                ));
            }
        }
    }
    if !native_observables.is_empty() {
        let handle = lanczos_handle.ok_or_else(|| {
            QmbedError::InvalidOptions("native thermal observables require a Lanczos handle".into())
        })?;
        let decomposition = registered_lanczos(handle)?;
        if decomposition.decomposition.basis.len() != dimension {
            return Err(QmbedError::DimensionMismatch(
                "Lanczos handle and supplied Ritz eigensystem dimensions differ".into(),
            ));
        }
        let borrowed = native_observables
            .iter()
            .map(|(name, operator)| (name.clone(), operator as &dyn LinearOperator))
            .collect::<Vec<_>>();
        projected.extend(decomposition.project_thermal_observables(&borrowed, method)?);
    }
    let result = thermal_observable_contraction(
        method,
        &eigenvalues,
        &eigenvector_columns,
        &projected,
        &inverse_temperatures,
    )?;
    Ok(CommandResult::ThermalLanczos {
        values: result
            .values
            .into_iter()
            .map(|(name, values)| (name, complex_payload(values)))
            .collect(),
        identity: result.identity,
    })
}

fn command_export_lanczos_basis(lanczos_handle: &str) -> Result<CommandResult> {
    let decomposition = registered_lanczos(lanczos_handle)?;
    let dimension = decomposition
        .decomposition
        .basis
        .first()
        .map_or(0, Vec::len);
    Ok(CommandResult::LanczosBasis {
        dimension,
        vectors: decomposition
            .decomposition
            .basis
            .iter()
            .map(|vector| vector.iter().map(|value| [value.re, value.im]).collect())
            .collect(),
    })
}

fn transform_operator(operator: Operator, action: OperatorActionRequest) -> Result<Operator> {
    match action {
        OperatorActionRequest::Normal => Ok(operator),
        OperatorActionRequest::Transpose => operator.transpose(),
        OperatorActionRequest::Conjugate => operator.conjugated(),
        OperatorActionRequest::Adjoint => operator.adjoint(),
    }
}

fn command_apply_terms(
    model: &RegisteredModel,
    terms: Vec<TermRequest>,
    vectors: Vec<Vec<[f64; 2]>>,
    action: OperatorActionRequest,
) -> Result<CommandResult> {
    let terms = terms
        .into_iter()
        .map(typed_term)
        .collect::<Result<Vec<_>>>()?;
    let vectors = complex_vectors(vectors);
    if let RegisteredModel::Ed(model) = model {
        let vectors = model.apply_terms_batch(terms, &vectors, action.into())?;
        return Ok(command_vectors(model, vectors));
    }
    let operator = transform_operator(
        model.temporary_operator(terms, AssemblyChecks::none(), MatrixFormat::MatrixFree)?,
        action,
    )?;
    let output_dimension = operator.shape().0;
    let vectors = vectors
        .into_iter()
        .map(|vector| {
            let mut output = vec![Complex64::new(0.0, 0.0); output_dimension];
            operator.apply(&vector, &mut output)?;
            Ok(output)
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(command_vectors_with_dimension(output_dimension, vectors))
}

fn command_apply_model(
    model: &RegisteredModel,
    vectors: Vec<Vec<[f64; 2]>>,
    action: OperatorActionRequest,
    parameters: HashMap<String, [f64; 2]>,
) -> Result<CommandResult> {
    let vectors = complex_vectors(vectors);
    let vectors = model.apply_batch(&complex_parameters(parameters), &vectors, action.into())?;
    Ok(command_vectors_with_dimension(model.dimension(), vectors))
}

fn command_matrix_elements(
    model: &RegisteredModel,
    left_vectors: Vec<Vec<[f64; 2]>>,
    right_vectors: Vec<Vec<[f64; 2]>>,
    diagonal: bool,
    parameters: HashMap<String, [f64; 2]>,
) -> Result<CommandResult> {
    if diagonal && left_vectors.len() != right_vectors.len() {
        return Err(QmbedError::DimensionMismatch(
            "diagonal matrix elements require equal vector batches".into(),
        ));
    }
    let operator = model.materialize(&complex_parameters(parameters), MatrixFormat::MatrixFree)?;
    let left_vectors = complex_vectors(left_vectors);
    let right_vectors = complex_vectors(right_vectors);
    let mut values = Vec::new();
    let shape = if diagonal {
        values.reserve(left_vectors.len());
        for (left, right) in left_vectors.iter().zip(&right_vectors) {
            let value = matrix_element(left, &operator, right)?;
            values.push([value.re, value.im]);
        }
        vec![left_vectors.len()]
    } else {
        values.reserve(left_vectors.len().saturating_mul(right_vectors.len()));
        for left in &left_vectors {
            for right in &right_vectors {
                let value = matrix_element(left, &operator, right)?;
                values.push([value.re, value.im]);
            }
        }
        vec![left_vectors.len(), right_vectors.len()]
    };
    Ok(CommandResult::Measurements { shape, values })
}

fn command_measure_model(
    model: &RegisteredModel,
    measurement: MeasurementRequest,
    samples: Vec<MeasurementSampleRequest>,
) -> Result<CommandResult> {
    let mut values = Vec::with_capacity(samples.len());
    for sample in samples {
        let (is_density, sample, parameters) = match sample {
            MeasurementSampleRequest::Pure { values, parameters } => (false, values, parameters),
            MeasurementSampleRequest::Density { values, parameters } => (true, values, parameters),
        };
        let operator =
            model.materialize(&complex_parameters(parameters), MatrixFormat::MatrixFree)?;
        let sample = sample
            .into_iter()
            .map(|[real, imaginary]| Complex64::new(real, imaginary))
            .collect::<Vec<_>>();
        let value = match (measurement, is_density) {
            (MeasurementRequest::Expectation, false) => expectation(&operator, &sample)?,
            (MeasurementRequest::Expectation, true) => density_expectation(&operator, &sample)?,
            (MeasurementRequest::QuantumFluctuation, false) => {
                raw_quantum_fluctuation(&operator, &sample)?
            }
            (MeasurementRequest::QuantumFluctuation, true) => {
                density_quantum_fluctuation(&operator, &sample)?
            }
        };
        values.push([value.re, value.im]);
    }
    Ok(CommandResult::Measurements {
        shape: vec![values.len()],
        values,
    })
}

fn complex_payload(values: Vec<Complex64>) -> Vec<[f64; 2]> {
    values
        .into_iter()
        .map(|value| [value.re, value.im])
        .collect()
}

struct SubsystemAnalysisOptions {
    local_dimensions: Vec<usize>,
    retained_sites: Vec<usize>,
    fermionic: bool,
    noncommuting_groups: Vec<NoncommutingGroupRequest>,
    renyi_order: Option<f64>,
}

struct SubsystemAnalysisLayout {
    local_dimensions: Vec<usize>,
    retained_sites: Vec<usize>,
    environment_sites: Vec<usize>,
    subsystem_dimension: usize,
    environment_dimension: usize,
    full_dimension: usize,
    noncommuting_groups: Vec<NoncommutingGroup>,
    order: EntropyOrder,
}

struct SubsystemSampleDensities {
    density_a: Vec<Complex64>,
    density_b: Vec<Complex64>,
    trace_scale: f64,
    canonical_pure_spectrum: Option<Vec<f64>>,
}

fn subsystem_sample_densities(
    model: &RegisteredModel,
    parent: &PackedEdModel,
    projector: &BasisProjector,
    layout: &SubsystemAnalysisLayout,
    sample: SubsystemSampleRequest,
) -> Result<SubsystemSampleDensities> {
    match sample {
        SubsystemSampleRequest::Pure { values } => {
            let reduced = values
                .into_iter()
                .map(|[real, imaginary]| Complex64::new(real, imaginary))
                .collect::<Vec<_>>();
            if reduced.len() != model.dimension() {
                return Err(QmbedError::DimensionMismatch(
                    "pure subsystem sample does not match the source model".into(),
                ));
            }
            let norm = reduced.iter().map(Complex64::norm_sqr).sum::<f64>();
            let lifted = projector.lifted(&reduced)?;
            let state = parent.scatter_state_vector(&lifted, layout.full_dimension)?;
            let canonical_pure_spectrum =
                canonical_schmidt_spectrum_subsystem_with_exchange_phases(
                    &state,
                    &layout.local_dimensions,
                    &layout.retained_sites,
                    &layout.noncommuting_groups,
                )?;
            let mut requested_state = state;
            if !layout.noncommuting_groups.is_empty() {
                apply_noncommuting_subsystem_exchange_phases(
                    &mut requested_state,
                    &layout.local_dimensions,
                    &layout.retained_sites,
                    &layout.noncommuting_groups,
                )?;
            }
            Ok(SubsystemSampleDensities {
                density_a: partial_trace_subsystem(
                    &requested_state,
                    &layout.local_dimensions,
                    &layout.retained_sites,
                )?,
                density_b: partial_trace_subsystem(
                    &requested_state,
                    &layout.local_dimensions,
                    &layout.environment_sites,
                )?,
                trace_scale: norm,
                canonical_pure_spectrum: Some(canonical_pure_spectrum),
            })
        }
        SubsystemSampleRequest::Density { values } => {
            let reduced = values
                .into_iter()
                .map(|[real, imaginary]| Complex64::new(real, imaginary))
                .collect::<Vec<_>>();
            if reduced.len() != model.dimension().saturating_mul(model.dimension()) {
                return Err(QmbedError::DimensionMismatch(
                    "density subsystem sample does not match the source model".into(),
                ));
            }
            let trace = (0..model.dimension())
                .map(|index| reduced[index * model.dimension() + index])
                .sum::<Complex64>();
            let lifted = projector.lift_density(&reduced)?;
            let mut density = parent.scatter_density(&lifted, layout.full_dimension)?;
            if !layout.noncommuting_groups.is_empty() {
                apply_noncommuting_subsystem_exchange_phases_density(
                    &mut density,
                    &layout.local_dimensions,
                    &layout.retained_sites,
                    &layout.noncommuting_groups,
                )?;
            }
            Ok(SubsystemSampleDensities {
                density_a: partial_trace_density_subsystem(
                    &density,
                    &layout.local_dimensions,
                    &layout.retained_sites,
                )?,
                density_b: partial_trace_density_subsystem(
                    &density,
                    &layout.local_dimensions,
                    &layout.environment_sites,
                )?,
                trace_scale: trace.re,
                canonical_pure_spectrum: None,
            })
        }
    }
}

fn pad_spectrum(probabilities: &[f64], dimension: usize) -> Result<Vec<f64>> {
    if probabilities.len() > dimension {
        return Err(QmbedError::InternalState(
            "Schmidt spectrum exceeds a bipartition factor dimension".into(),
        ));
    }
    let mut padded = vec![0.0; dimension];
    let start = dimension - probabilities.len();
    padded[start..].copy_from_slice(probabilities);
    Ok(padded)
}

fn command_analyze_subsystem(
    model: &RegisteredModel,
    parent: &RegisteredModel,
    projector: &BasisProjector,
    options: SubsystemAnalysisOptions,
    samples: Vec<SubsystemSampleRequest>,
) -> Result<CommandResult> {
    let noncommuting_groups = if options.noncommuting_groups.is_empty() && options.fermionic {
        vec![NoncommutingGroup::new(
            (0..options.local_dimensions.len()).collect::<Vec<_>>(),
            Complex64::new(-1.0, 0.0),
        )?]
    } else {
        options
            .noncommuting_groups
            .into_iter()
            .map(TryInto::try_into)
            .collect::<Result<Vec<_>>>()?
    };
    let (subsystem_dimension, environment_dimension) =
        subsystem_dimensions(&options.local_dimensions, &options.retained_sites)?;
    let full_dimension = subsystem_dimension
        .checked_mul(environment_dimension)
        .ok_or_else(|| QmbedError::DimensionMismatch("Hilbert-space size overflow".into()))?;
    if parent.dimension() != full_dimension {
        return Err(QmbedError::DimensionMismatch(
            "explicit parent dimension does not match the local tensor product".into(),
        ));
    }
    let mut retained = vec![false; options.local_dimensions.len()];
    for &site in &options.retained_sites {
        retained[site] = true;
    }
    let environment_sites = (0..options.local_dimensions.len())
        .filter(|site| !retained[*site])
        .collect::<Vec<_>>();
    let layout = SubsystemAnalysisLayout {
        local_dimensions: options.local_dimensions,
        retained_sites: options.retained_sites,
        environment_sites,
        subsystem_dimension,
        environment_dimension,
        full_dimension,
        noncommuting_groups,
        order: options
            .renyi_order
            .map_or(EntropyOrder::VonNeumann, EntropyOrder::Renyi),
    };
    let parent = parent.ed()?;
    let mut analyses = Vec::with_capacity(samples.len());
    for sample in samples {
        let SubsystemSampleDensities {
            mut density_a,
            mut density_b,
            trace_scale,
            canonical_pure_spectrum,
        } = subsystem_sample_densities(model, parent, projector, &layout, sample)?;
        let is_pure = canonical_pure_spectrum.is_some();
        let (mut spectrum_a, mut spectrum_b) = match canonical_pure_spectrum {
            Some(probabilities) => (
                pad_spectrum(&probabilities, layout.subsystem_dimension)?,
                pad_spectrum(&probabilities, layout.environment_dimension)?,
            ),
            None => (
                density_matrix_spectrum(density_a.clone(), layout.subsystem_dimension)?,
                density_matrix_spectrum(density_b.clone(), layout.environment_dimension)?,
            ),
        };
        for value in &mut density_a {
            *value *= trace_scale;
        }
        for value in &mut density_b {
            *value *= trace_scale;
        }
        for value in &mut spectrum_a {
            *value *= trace_scale;
        }
        for value in &mut spectrum_b {
            *value *= trace_scale;
        }
        let entropy_a = entropy_from_spectrum(&spectrum_a, layout.order)?;
        let entropy_b = if is_pure {
            entropy_a
        } else {
            entropy_from_spectrum(&spectrum_b, layout.order)?
        };
        analyses.push(SubsystemAnalysisEntry {
            entropy_a,
            entropy_b,
            spectrum_a,
            spectrum_b,
            density_a: complex_payload(density_a),
            density_b: complex_payload(density_b),
        });
    }
    Ok(CommandResult::SubsystemAnalysis {
        subsystem_dimension: layout.subsystem_dimension,
        environment_dimension: layout.environment_dimension,
        samples: analyses,
    })
}

fn command_evolve_model(
    model: &RegisteredModel,
    vectors: Vec<Vec<[f64; 2]>>,
    evolution: EvolutionRequest,
    parameters: HashMap<String, [f64; 2]>,
) -> Result<CommandResult> {
    if evolution.imaginary_time {
        return Err(QmbedError::InvalidOptions(
            "static imaginary-time evolution is not exposed by this command yet".into(),
        ));
    }
    let trajectory = model.evolve_batch(
        &complex_parameters(parameters),
        &complex_vectors(vectors),
        EvolutionOptions {
            times: evolution.times,
            krylov_dimension: evolution.krylov_dimension,
            tolerance: evolution.tolerance,
            max_substeps: evolution.max_substeps,
            hamiltonian: true,
        },
    )?;
    Ok(command_trajectory(model.dimension(), trajectory))
}

fn execute_drive_evolution(
    request: DriveEvolutionRequest,
    callback: QmbedDriveCallback,
    context_address: usize,
) -> Result<CommandResult> {
    let model = registered_model(&request.handle)?;
    let expected_names = model.component_names();
    if request.component_names != expected_names {
        return Err(QmbedError::InvalidOptions(format!(
            "drive component order mismatch: expected {expected_names:?}, received {:?}",
            request.component_names
        )));
    }
    let trajectory = model.evolve_time_dependent_batch(
        &complex_vectors(request.vectors),
        request.initial_time,
        EvolutionOptions {
            times: request.evolution.times,
            krylov_dimension: request.evolution.krylov_dimension,
            tolerance: request.evolution.tolerance,
            max_substeps: request.evolution.max_substeps,
            hamiltonian: true,
        },
        if request.evolution.imaginary_time {
            Complex64::new(0.0, -1.0)
        } else {
            Complex64::new(1.0, 0.0)
        },
        move |time, coefficients| {
            let mut abi_coefficients = vec![
                QmbedComplex64 {
                    real: f64::NAN,
                    imaginary: f64::NAN,
                };
                coefficients.len()
            ];
            let status = callback(
                context_address as *mut c_void,
                time,
                abi_coefficients.as_mut_ptr(),
                abi_coefficients.len(),
            );
            if status != 0 {
                return Err(QmbedError::InvalidOptions(format!(
                    "drive callback returned status {status} at time {time}"
                )));
            }
            for (coefficient, value) in coefficients.iter_mut().zip(abi_coefficients) {
                *coefficient = Complex64::new(value.real, value.imaginary);
            }
            Ok(())
        },
    )?;
    Ok(command_trajectory(model.dimension(), trajectory))
}

fn command_trajectory(
    dimension: usize,
    trajectory: qmbed::solve::StateBatchTrajectory,
) -> CommandResult {
    CommandResult::Trajectory {
        dimension,
        times: trajectory.times,
        states: trajectory
            .states
            .into_iter()
            .map(|columns| {
                columns
                    .into_iter()
                    .map(|column| {
                        column
                            .into_iter()
                            .map(|value| [value.re, value.im])
                            .collect()
                    })
                    .collect()
            })
            .collect(),
    }
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

fn complex_parameters(parameters: HashMap<String, [f64; 2]>) -> HashMap<String, Complex64> {
    parameters
        .into_iter()
        .map(|(name, [real, imaginary])| (name, Complex64::new(real, imaginary)))
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

fn registered_projector(
    handle: &str,
    parent_handle: &str,
    embedding: bool,
) -> Result<CommandResult> {
    let projector = cached_projector(handle, parent_handle, embedding)?;
    command_projector(&projector)
}

fn registered_projector_action(
    handle: &str,
    parent_handle: &str,
    embedding: bool,
    vectors: Vec<Vec<[f64; 2]>>,
    action: ProjectorActionRequest,
) -> Result<CommandResult> {
    let projector = cached_projector(handle, parent_handle, embedding)?;
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
    let terms = terms
        .into_iter()
        .map(typed_term)
        .collect::<Result<Vec<_>>>()?;
    let vectors = complex_vectors(vectors);
    let source_is_wide = matches!(
        source.as_ref(),
        RegisteredModel::Wide(_) | RegisteredModel::WideProjected(_)
    );
    let target_is_wide = matches!(
        target.as_ref(),
        RegisteredModel::Wide(_) | RegisteredModel::WideProjected(_)
    );
    if source_is_wide && target_is_wide {
        let (source_parent, source_projector) = source.wide_local_subspace()?;
        let (target_parent, target_projector) = target.wide_local_subspace()?;
        let vectors = WideEdModel::apply_terms_between_subspaces_batch(
            source_parent,
            source_projector,
            target_parent,
            target_projector,
            terms,
            &vectors,
        )?;
        return Ok(command_vectors_with_dimension(target.dimension(), vectors));
    }
    let (source_parent, source_projector) = source.local_subspace()?;
    let (target_parent, target_projector) = target.local_subspace()?;
    let vectors = PackedEdModel::apply_terms_between_subspaces_batch(
        source_parent,
        source_projector,
        target_parent,
        target_projector,
        terms,
        &vectors,
    )?;
    Ok(command_vectors_with_dimension(target.dimension(), vectors))
}

fn registered_bra_ket_terms(
    handle: &str,
    terms: Vec<TermRequest>,
    kets: Vec<String>,
) -> Result<CommandResult> {
    let model = registered_model(handle)?;
    match model.as_ref() {
        RegisteredModel::Ed(model) => command_bra_ket_terms(model, terms, kets),
        RegisteredModel::Wide(model) => command_wide_bra_ket_terms(model, terms, kets),
        RegisteredModel::Operator(_) => Err(QmbedError::InvalidOptions(
            "basis-independent operator models do not define local bra-ket actions".into(),
        )),
        RegisteredModel::Projected(model) => {
            let terms = terms
                .into_iter()
                .map(typed_term)
                .collect::<Result<Vec<_>>>()?;
            let operator = model.temporary_operator(
                terms,
                AssemblyChecks::none(),
                MatrixFormat::MatrixFree,
            )?;
            let mut label_indices = HashMap::new();
            for (index, label) in model.labels.iter().copied().enumerate() {
                if label_indices.insert(label, index).is_some() {
                    return Err(QmbedError::InternalState(
                        "matrix-representation labels must be unique".into(),
                    ));
                }
            }
            let mut entries = Vec::new();
            for (input, ket) in kets.into_iter().enumerate() {
                let ket_value = ket.parse::<u128>().map_err(|_| {
                    QmbedError::InvalidOptions(format!(
                        "ket state {ket:?} is not an unsigned 128-bit integer"
                    ))
                })?;
                let Some(&column) = label_indices.get(&ket_value) else {
                    continue;
                };
                let mut source = vec![Complex64::new(0.0, 0.0); model.labels.len()];
                source[column] = Complex64::new(1.0, 0.0);
                let mut target = source.clone();
                operator.apply(&source, &mut target)?;
                entries.extend(
                    target
                        .into_iter()
                        .enumerate()
                        .filter(|(_, value)| value.norm() > f64::EPSILON)
                        .map(|(row, value)| TransitionEntry {
                            input,
                            bra: model.labels[row].to_string(),
                            ket: ket_value.to_string(),
                            value: [value.re, value.im],
                        }),
                );
            }
            Ok(CommandResult::Transitions { entries })
        }
        RegisteredModel::WideProjected(model) => {
            let terms = terms
                .into_iter()
                .map(typed_term)
                .collect::<Result<Vec<_>>>()?;
            let operator = model.temporary_operator(
                terms,
                AssemblyChecks::none(),
                MatrixFormat::MatrixFree,
            )?;
            let mut label_indices = HashMap::new();
            for (index, label) in model.labels.iter().copied().enumerate() {
                if label_indices.insert(label, index).is_some() {
                    return Err(QmbedError::InternalState(
                        "matrix-representation labels must be unique".into(),
                    ));
                }
            }
            let width_bits = model
                .labels
                .first()
                .copied()
                .or_else(|| model.assembly_parent.basis().state(0).ok())
                .map(|state| state.width_bits())
                .ok_or_else(|| {
                    QmbedError::InvalidSector(
                        "a wide matrix-representation parent has no physical states".into(),
                    )
                })?;
            let mut entries = Vec::new();
            for (input, ket) in kets.into_iter().enumerate() {
                let ket_value = ErasedState::from_decimal(width_bits, &ket)?;
                let Some(&column) = label_indices.get(&ket_value) else {
                    continue;
                };
                let mut source = vec![Complex64::new(0.0, 0.0); model.labels.len()];
                source[column] = Complex64::new(1.0, 0.0);
                let mut target = source.clone();
                operator.apply(&source, &mut target)?;
                entries.extend(
                    target
                        .into_iter()
                        .enumerate()
                        .filter(|(_, value)| value.norm() > f64::EPSILON)
                        .map(|(row, value)| TransitionEntry {
                            input,
                            bra: model.labels[row].to_string(),
                            ket: ket_value.to_string(),
                            value: [value.re, value.im],
                        }),
                );
            }
            Ok(CommandResult::Transitions { entries })
        }
    }
}

fn registered_reduce_states(handle: &str, states: Vec<String>) -> Result<CommandResult> {
    let model = registered_model(handle)?;
    match model.as_ref() {
        RegisteredModel::Ed(model) => command_reduce_states(model, states),
        RegisteredModel::Wide(model) => command_reduce_wide_states(model, states),
        RegisteredModel::Operator(_) => Err(QmbedError::InvalidOptions(
            "basis-independent operator models do not define physical-state reduction".into(),
        )),
        RegisteredModel::Projected(model) => {
            let labels: HashSet<_> = model.labels.iter().copied().collect();
            let entries = states
                .into_iter()
                .map(|state| {
                    let value = state.parse::<u128>().map_err(|_| {
                        QmbedError::InvalidOptions(format!(
                            "state {state:?} is not an unsigned 128-bit integer"
                        ))
                    })?;
                    let compatible = labels.contains(&value);
                    Ok(ReductionEntry {
                        state: value.to_string(),
                        representative: compatible.then_some(value.to_string()),
                        phase: compatible.then_some([1.0, 0.0]),
                        amplitude: compatible.then_some([1.0, 0.0]),
                        orbit_size: compatible.then_some(1),
                        compatible,
                        physical_phase_to_representative: compatible.then_some([1.0, 0.0]),
                        generator_word: compatible.then(Vec::new),
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            Ok(CommandResult::Reductions {
                period_product: None,
                entries,
            })
        }
        RegisteredModel::WideProjected(model) => {
            let labels: HashSet<_> = model.labels.iter().copied().collect();
            let width_bits = model
                .labels
                .first()
                .copied()
                .or_else(|| model.assembly_parent.basis().state(0).ok())
                .map(|state| state.width_bits())
                .ok_or_else(|| {
                    QmbedError::InvalidSector(
                        "a wide matrix-representation parent has no physical states".into(),
                    )
                })?;
            let entries = states
                .into_iter()
                .map(|state| {
                    let value = ErasedState::from_decimal(width_bits, &state)?;
                    let compatible = labels.contains(&value);
                    Ok(ReductionEntry {
                        state: value.to_string(),
                        representative: compatible.then_some(value.to_string()),
                        phase: compatible.then_some([1.0, 0.0]),
                        amplitude: compatible.then_some([1.0, 0.0]),
                        orbit_size: compatible.then_some(1),
                        compatible,
                        physical_phase_to_representative: compatible.then_some([1.0, 0.0]),
                        generator_word: compatible.then(Vec::new),
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            Ok(CommandResult::Reductions {
                period_product: None,
                entries,
            })
        }
    }
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

fn command_wide_bra_ket_terms(
    model: &WideEdModel,
    terms: Vec<TermRequest>,
    kets: Vec<String>,
) -> Result<CommandResult> {
    let terms = terms
        .into_iter()
        .map(typed_term)
        .collect::<Result<Vec<_>>>()?;
    let width_bits = model.basis().width_bits();
    let kets = kets
        .into_iter()
        .map(|ket| ErasedState::from_decimal(width_bits, &ket))
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
                    bra: transition.bra.to_decimal(),
                    ket: transition.ket.to_decimal(),
                    value: [transition.matrix_element.re, transition.matrix_element.im],
                })
        })
        .collect();
    Ok(CommandResult::Transitions { entries })
}

fn parse_physical_states(states: Vec<String>) -> Result<Vec<u128>> {
    states
        .into_iter()
        .map(|state| {
            state.parse::<u128>().map_err(|_| {
                QmbedError::InvalidOptions(format!(
                    "physical state {state:?} is not an unsigned 128-bit integer"
                ))
            })
        })
        .collect()
}

fn command_reduce_states(model: &PackedEdModel, states: Vec<String>) -> Result<CommandResult> {
    let states = parse_physical_states(states)?;
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
                compatible: image.is_some(),
                physical_phase_to_representative: None,
                generator_word: None,
            }
        })
        .collect();
    Ok(CommandResult::Reductions {
        period_product: None,
        entries,
    })
}

fn command_reduce_wide_states(model: &WideEdModel, states: Vec<String>) -> Result<CommandResult> {
    let width_bits = model.basis().width_bits();
    let states = states
        .into_iter()
        .map(|state| ErasedState::from_decimal(width_bits, &state))
        .collect::<Result<Vec<_>>>()?;
    let images = model.reduction_images(&states)?;
    let entries = states
        .into_iter()
        .zip(images)
        .map(|(state, image)| {
            let representative = image.map(|value| value.representative().to_decimal());
            let phase = image.map(|value| [value.phase().re, value.phase().im]);
            let amplitude = image.map(|value| [value.amplitude().re, value.amplitude().im]);
            let orbit_size = image.map(|value| value.orbit_size());
            ReductionEntry {
                state: state.to_decimal(),
                representative,
                phase,
                amplitude,
                orbit_size,
                compatible: image.is_some(),
                physical_phase_to_representative: None,
                generator_word: None,
            }
        })
        .collect();
    Ok(CommandResult::Reductions {
        period_product: None,
        entries,
    })
}

fn command_reduce_states_plan(
    plan: &RegisteredBasisPlan,
    states: Vec<String>,
) -> Result<CommandResult> {
    let states = parse_physical_states(states)?;
    let period_product = plan.reducer.period_product()?;
    let entries = states
        .into_iter()
        .map(|state| {
            let orbit = plan
                .reducer
                .orbit_with_ordering(state, plan.representative_ordering)?;
            let phase = orbit.phase();
            let amplitude = phase
                .map(|value| {
                    ReductionImage::new(state, value, orbit.orbit_size())
                        .map(|image| image.amplitude())
                })
                .transpose()?;
            let physical_phase = orbit.physical_phase_to_representative();
            Ok(ReductionEntry {
                state: state.to_string(),
                representative: Some(orbit.representative().to_string()),
                phase: phase.map(|value| [value.re, value.im]),
                amplitude: amplitude.map(|value| [value.re, value.im]),
                orbit_size: Some(orbit.orbit_size()),
                compatible: orbit.is_compatible(),
                physical_phase_to_representative: Some([physical_phase.re, physical_phase.im]),
                generator_word: Some(orbit.generator_word().to_vec()),
            })
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(CommandResult::Reductions {
        period_product: Some(period_product),
        entries,
    })
}

fn command_eigh(
    model: &RegisteredModel,
    parameters: HashMap<String, [f64; 2]>,
    eigenvectors: bool,
) -> Result<CommandResult> {
    let result = model.eigh(
        &complex_parameters(parameters),
        EighOptions {
            return_eigenvectors: eigenvectors,
        },
    )?;
    Ok(command_eigensystem(model.dimension(), result, eigenvectors))
}

fn command_eigsh(
    model: &RegisteredModel,
    parameters: HashMap<String, [f64; 2]>,
    format: StorageFormat,
    solver: &SolverRequest,
) -> Result<CommandResult> {
    let include_vectors = solver.eigenvectors;
    let initial = solver.initial_vector();
    let result = model.eigsh(
        &complex_parameters(parameters),
        format.into(),
        solver.options(),
        initial.as_deref(),
    )?;
    Ok(command_eigensystem(
        model.dimension(),
        result,
        include_vectors,
    ))
}

fn describe_basis(request: &BasisRequest) -> Result<CommandResult> {
    if matrix_symmetry_request(request).is_some() {
        if basis_storage(request)? != StateStorage::U128 {
            let (_parent, subspace) = wide_matrix_symmetry_subspace(request)?;
            return Ok(CommandResult::Basis {
                dimension: subspace.dimension(),
                states: subspace.labels().iter().map(ToString::to_string).collect(),
            });
        }
        let (_parent, subspace) = matrix_symmetry_subspace(request)?;
        return Ok(CommandResult::Basis {
            dimension: subspace.dimension(),
            states: subspace.labels().iter().map(ToString::to_string).collect(),
        });
    }
    if basis_storage(request)? != StateStorage::U128 {
        let basis = build_wide_basis(request)?;
        let states = (0..basis.len())
            .map(|index| basis.state(index).map(|state| state.to_decimal()))
            .collect::<Result<Vec<_>>>()?;
        return Ok(CommandResult::Basis {
            dimension: basis.len(),
            states,
        });
    }
    let basis = build_basis(request)?;
    let states = (0..basis.len())
        .map(|index| basis.state(index).map(|state| state.to_string()))
        .collect::<Result<Vec<_>>>()?;
    Ok(CommandResult::Basis {
        dimension: basis.len(),
        states,
    })
}

fn command_bitwise_states(
    operation: BitwiseOperationRequest,
    width_bits: usize,
    left: Vec<String>,
    right: Vec<String>,
    shifts: Vec<usize>,
) -> Result<CommandResult> {
    let left = left
        .iter()
        .map(|value| ErasedState::from_decimal(width_bits, value))
        .collect::<Result<Vec<_>>>()?;
    let needs_right = matches!(
        operation,
        BitwiseOperationRequest::And | BitwiseOperationRequest::Or | BitwiseOperationRequest::Xor
    );
    let needs_shift = matches!(
        operation,
        BitwiseOperationRequest::LeftShift | BitwiseOperationRequest::RightShift
    );
    if needs_right && right.len() != left.len() {
        return Err(QmbedError::DimensionMismatch(
            "binary bitwise operands must have matching lengths".into(),
        ));
    }
    if needs_shift && shifts.len() != left.len() {
        return Err(QmbedError::DimensionMismatch(
            "bitwise values and shifts must have matching lengths".into(),
        ));
    }
    if !needs_right && !right.is_empty() {
        return Err(QmbedError::InvalidOptions(
            "this bitwise operation does not accept a right operand".into(),
        ));
    }
    if !needs_shift && !shifts.is_empty() {
        return Err(QmbedError::InvalidOptions(
            "this bitwise operation does not accept shifts".into(),
        ));
    }
    let right = if needs_right {
        right
            .iter()
            .map(|value| ErasedState::from_decimal(width_bits, value))
            .collect::<Result<Vec<_>>>()?
    } else {
        Vec::new()
    };
    let values = left
        .iter()
        .enumerate()
        .map(|(index, value)| {
            let output = match operation {
                BitwiseOperationRequest::Not => value.bitwise_not()?,
                BitwiseOperationRequest::And => value.bitwise_and(&right[index])?,
                BitwiseOperationRequest::Or => value.bitwise_or(&right[index])?,
                BitwiseOperationRequest::Xor => value.bitwise_xor(&right[index])?,
                BitwiseOperationRequest::LeftShift => value.left_shift(shifts[index])?,
                BitwiseOperationRequest::RightShift => value.right_shift(shifts[index]),
            };
            Ok(output.to_decimal())
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(CommandResult::Integers { width_bits, values })
}

fn command_analyze_diagonal_ensemble(
    eigenvalues: Vec<f64>,
    eigenvectors: Vec<Vec<[f64; 2]>>,
    input: DiagonalEnsembleInputRequest,
    observable: Option<OperatorExpressionRequest>,
    alpha: f64,
    reconstruct_density: bool,
) -> Result<CommandResult> {
    let eigenvectors = complex_vectors(eigenvectors);
    let probability_columns = match input {
        DiagonalEnsembleInputRequest::Pure { values } => {
            let initial = complex_vectors(vec![values]).remove(0);
            vec![diagonal_ensemble(&eigenvalues, &eigenvectors, &initial)?.probabilities]
        }
        DiagonalEnsembleInputRequest::PureColumns { vectors } => complex_vectors(vectors)
            .iter()
            .map(|initial| {
                Ok(diagonal_ensemble(&eigenvalues, &eigenvectors, initial)?.probabilities)
            })
            .collect::<Result<Vec<_>>>()?,
        DiagonalEnsembleInputRequest::Density { values } => {
            let density = complex_vectors(vec![values]).remove(0);
            vec![diagonal_ensemble_density(&eigenvalues, &eigenvectors, &density)?.probabilities]
        }
        DiagonalEnsembleInputRequest::Probabilities { columns } => columns,
    };
    let observable = observable.map(evaluate_operator_expression).transpose()?;
    let analyses = analyze_diagonal_ensemble(
        &eigenvalues,
        &eigenvectors,
        &probability_columns,
        observable
            .as_ref()
            .map(|operator| operator as &dyn qmbed::operator::LinearOperator),
        alpha,
    )?;
    let has_observable = observable.is_some();
    Ok(CommandResult::DiagonalEnsemble {
        probabilities: analyses
            .iter()
            .map(|analysis| analysis.ensemble.probabilities.clone())
            .collect(),
        mean_energies: analyses
            .iter()
            .map(|analysis| analysis.ensemble.mean_energy)
            .collect(),
        energy_variances: analyses
            .iter()
            .map(|analysis| analysis.ensemble.energy_variance)
            .collect(),
        von_neumann_entropies: analyses
            .iter()
            .map(|analysis| analysis.ensemble.entropy)
            .collect(),
        diagonal_entropies: analyses
            .iter()
            .map(|analysis| analysis.diagonal_entropy)
            .collect(),
        observables: has_observable.then(|| {
            analyses
                .iter()
                .map(|analysis| analysis.observable.expect("observable analysis"))
                .collect()
        }),
        temporal_fluctuations: has_observable.then(|| {
            analyses
                .iter()
                .map(|analysis| analysis.temporal_fluctuation.expect("observable analysis"))
                .collect()
        }),
        quantum_fluctuations: has_observable.then(|| {
            analyses
                .iter()
                .map(|analysis| analysis.quantum_fluctuation.expect("observable analysis"))
                .collect()
        }),
        density_matrices: reconstruct_density
            .then(|| {
                analyses
                    .iter()
                    .map(|analysis| {
                        diagonal_density_matrix(&eigenvectors, &analysis.ensemble.probabilities)
                            .map(complex_payload)
                    })
                    .collect::<Result<Vec<_>>>()
            })
            .transpose()?,
    })
}

fn execute_command(request: CommandRequest) -> Result<CommandResult> {
    match request {
        CommandRequest::DescribeBasis { basis } => describe_basis(&basis),
        CommandRequest::BitwiseStates {
            bitwise_operation,
            width_bits,
            left,
            right,
            shifts,
        } => command_bitwise_states(bitwise_operation, width_bits, left, right, shifts),
        CommandRequest::AnalyzeDiagonalEnsemble {
            eigenvalues,
            eigenvectors,
            input,
            observable,
            alpha,
            reconstruct_density,
        } => command_analyze_diagonal_ensemble(
            eigenvalues,
            eigenvectors,
            input,
            observable,
            alpha,
            reconstruct_density,
        ),
        CommandRequest::Materialize {
            basis,
            terms,
            site_permutation,
            format,
            checks,
        } => {
            let model = build_registered_parameterized_model(
                &basis,
                terms,
                Vec::new(),
                checks.into(),
                site_permutation,
            )?;
            command_operator(&model, HashMap::new(), format)
        }
        CommandRequest::Eigh {
            basis,
            terms,
            site_permutation,
            eigenvectors,
            checks,
        } => {
            let model = build_registered_parameterized_model(
                &basis,
                terms,
                Vec::new(),
                checks.into(),
                site_permutation,
            )?;
            command_eigh(&model, HashMap::new(), eigenvectors)
        }
        CommandRequest::Eigsh {
            basis,
            terms,
            site_permutation,
            format,
            solver,
            checks,
        } => {
            let model = build_registered_parameterized_model(
                &basis,
                terms,
                Vec::new(),
                checks.into(),
                site_permutation,
            )?;
            command_eigsh(&model, HashMap::new(), format, &solver)
        }
        CommandRequest::CreateModel {
            basis,
            terms,
            components,
            site_permutation,
            checks,
        } => {
            let model = build_registered_parameterized_model(
                &basis,
                terms,
                components,
                checks.into(),
                site_permutation,
            )?;
            let dimension = model.dimension();
            let handle = register_model(model)?;
            Ok(CommandResult::Model { handle, dimension })
        }
        CommandRequest::CreateOperatorModel {
            static_operator,
            components,
            basis,
            site_permutation,
            checks,
        } => {
            let model = build_operator_model(
                static_operator,
                components,
                basis.as_ref(),
                checks.into(),
                site_permutation,
            )?;
            let dimension = model.dimension();
            let handle = register_model(model)?;
            Ok(CommandResult::Model { handle, dimension })
        }
        CommandRequest::CreateOperatorExpressionModel { expression, format } => {
            command_create_operator_expression_model(expression, format)
        }
        CommandRequest::CreateProjectedBlockModel {
            blocks,
            tolerance,
            format,
        } => create_projected_block_model(blocks, tolerance, format),
        CommandRequest::CreateBlockModel { handles, format } => create_block_model(handles, format),
        CommandRequest::LoadOperatorArchive { path } => command_load_operator_archive(path),
        CommandRequest::CreateBasisPlan {
            basis,
            site_permutation,
            checks,
        } => create_basis_plan(basis, site_permutation, checks.into()),
        request => execute_registered_command(request),
    }
}

#[allow(clippy::too_many_lines)]
fn execute_registered_command(request: CommandRequest) -> Result<CommandResult> {
    match request {
        CommandRequest::DescribeModel { handle } => registered_describe_model(&handle),
        CommandRequest::SaveOperatorArchive {
            handle,
            path,
            formats,
            metadata,
        } => registered_save_operator_archive(&handle, path, formats, metadata),
        CommandRequest::MaterializeModel {
            handle,
            format,
            parameters,
        } => {
            let model = registered_model(&handle)?;
            command_operator(&model, parameters, format)
        }
        CommandRequest::EvaluateOperatorExpression { expression, format } => {
            let operator = evaluate_operator_expression(expression)?.converted(format.into())?;
            Ok(command_operator_value(&operator, format))
        }
        CommandRequest::ApplyOperatorExpression {
            expression,
            vectors,
        } => command_apply_operator_expression(expression, vectors),
        CommandRequest::InspectOperatorExpression { expression } => {
            command_inspect_operator_expression(expression)
        }
        CommandRequest::ExpmOperatorExpression {
            expression,
            coefficient,
            vectors,
            max_degree,
            tolerance,
            max_substeps,
            threads,
        } => command_expm_operator_expression(
            expression,
            coefficient,
            vectors,
            max_degree,
            tolerance,
            max_substeps,
            threads,
        ),
        CommandRequest::EighOperatorExpression {
            expression,
            eigenvectors,
        } => {
            let operator = evaluate_operator_expression(expression)?;
            let dimension = qmbed::operator::LinearOperator::shape(&operator).0;
            let result = eigh_with_options(
                &operator,
                EighOptions {
                    return_eigenvectors: eigenvectors,
                },
            )?;
            Ok(command_eigensystem(dimension, result, eigenvectors))
        }
        CommandRequest::EigshOperatorExpression {
            expression,
            format,
            solver,
        } => command_eigsh_operator_expression(expression, format, &solver),
        CommandRequest::LanczosOperator {
            expression,
            initial,
            krylov_dimension,
            tolerance,
        } => command_lanczos_operator(expression, initial, krylov_dimension, tolerance),
        CommandRequest::LanczosCombine {
            lanczos_handle,
            coefficients,
        } => command_lanczos_combine(&lanczos_handle, coefficients),
        CommandRequest::LanczosExponential {
            lanczos_handle,
            coefficient,
        } => command_lanczos_exponential(&lanczos_handle, coefficient),
        CommandRequest::LanczosThermal {
            lanczos_handle,
            method,
            eigenvalues,
            eigenvectors,
            inverse_temperatures,
            observables,
        } => command_lanczos_thermal(
            lanczos_handle.as_deref(),
            method,
            eigenvalues,
            eigenvectors,
            inverse_temperatures,
            observables,
        ),
        CommandRequest::ExportLanczosBasis { lanczos_handle } => {
            command_export_lanczos_basis(&lanczos_handle)
        }
        CommandRequest::ReleaseLanczos { lanczos_handle } => Ok(CommandResult::Released {
            handle: release_lanczos(&lanczos_handle)?,
        }),
        CommandRequest::CreateExpmAction {
            handle,
            parameters,
            coefficient: [real, imaginary],
            max_degree,
            tolerance,
            max_substeps,
        } => {
            let model = registered_model(&handle)?;
            let operator = model.materialize(&complex_parameters(parameters), MatrixFormat::Csr)?;
            let dimension = operator.shape().0;
            let plan = ExpmMultiplyParallel::new(
                Arc::new(operator),
                Complex64::new(real, imaginary),
                max_degree,
                tolerance,
                max_substeps,
            )?;
            let handle = register_expm(plan)?;
            Ok(CommandResult::ExpmPlan { handle, dimension })
        }
        CommandRequest::ApplyExpmAction {
            expm_handle,
            vectors,
            threads,
        } => {
            let plan = registered_expm(&expm_handle)?;
            let vectors = complex_vectors(vectors);
            let threads = threads.unwrap_or_else(|| {
                std::thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get)
            });
            let runtime = CpuRuntime::new(threads)?;
            let vectors = plan.apply_batch_with_runtime(&runtime, &vectors)?;
            Ok(command_vectors_with_dimension(plan.shape().0, vectors))
        }
        CommandRequest::ReleaseExpmAction { expm_handle } => Ok(CommandResult::Released {
            handle: release_expm(&expm_handle)?,
        }),
        CommandRequest::ProjectOperatorModel { handle, projector } => {
            let source = registered_model(&handle)?;
            let projector = matrix_operator(projector, MatrixFormat::Csc)?;
            let projected = source.projected_by(&projector)?;
            let dimension = projected.dimension();
            let handle = register_model(projected)?;
            Ok(CommandResult::Model { handle, dimension })
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
            let operator = model.temporary_operator(terms, checks.into(), format.into())?;
            Ok(command_operator_value(&operator, format))
        }
        CommandRequest::MaterializeComponentModel {
            handle,
            name,
            format,
        } => {
            let model = registered_model(&handle)?;
            let operator = model.component_operator(&name, format.into())?;
            Ok(command_operator_value(&operator, format))
        }
        CommandRequest::ApplyModel {
            handle,
            vectors,
            action,
            parameters,
        } => {
            let model = registered_model(&handle)?;
            command_apply_model(&model, vectors, action, parameters)
        }
        CommandRequest::MatrixElementsModel {
            handle,
            left_vectors,
            right_vectors,
            diagonal,
            parameters,
        } => {
            let model = registered_model(&handle)?;
            command_matrix_elements(&model, left_vectors, right_vectors, diagonal, parameters)
        }
        CommandRequest::MeasureModel {
            handle,
            measurement,
            samples,
        } => {
            let model = registered_model(&handle)?;
            command_measure_model(&model, measurement, samples)
        }
        CommandRequest::MeanLevelSpacing { eigenvalues } => Ok(CommandResult::Statistic {
            value: mean_level_spacing(&eigenvalues)?,
        }),
        CommandRequest::FloquetTimeGrid {
            period,
            constant_cycles,
            points_per_cycle,
            ramp_up_cycles,
            ramp_down_cycles,
        } => {
            let grid = FloquetTimeVector::staged(
                period,
                ramp_up_cycles,
                constant_cycles,
                ramp_down_cycles,
                points_per_cycle,
            )?;
            Ok(CommandResult::TimeGrid {
                period: grid.period(),
                cycles: grid.cycles(),
                points_per_cycle: grid.points_per_cycle(),
                times: grid.times().to_vec(),
            })
        }
        CommandRequest::AnalyzeFloquet {
            steps,
            period,
            format,
        } => command_analyze_floquet(steps, period, format),
        CommandRequest::AnalyzeFloquetUnitary {
            unitary,
            period,
            format,
        } => {
            let unitary = matrix_operator(unitary, MatrixFormat::Csc)?;
            let analysis = analyze_floquet_unitary(&unitary, period, format.into())?;
            Ok(command_floquet_analysis(analysis, format))
        }
        CommandRequest::AnalyzeSubsystemModel {
            handle,
            parent_handle,
            embedding,
            local_dimensions,
            retained_sites,
            fermionic,
            noncommuting_groups,
            samples,
            renyi_order,
        } => {
            let model = registered_model(&handle)?;
            let parent = registered_model(&parent_handle)?;
            let projector = cached_projector(&handle, &parent_handle, embedding)?;
            command_analyze_subsystem(
                &model,
                &parent,
                &projector,
                SubsystemAnalysisOptions {
                    local_dimensions,
                    retained_sites,
                    fermionic,
                    noncommuting_groups,
                    renyi_order,
                },
                samples,
            )
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
        request @ (CommandRequest::ReduceStatesPlan { .. }
        | CommandRequest::MaterializeBasisPlan { .. }
        | CommandRequest::ReleaseBasisPlan { .. }
        | CommandRequest::BraKetTermsPlan { .. }) => execute_basis_plan_command(request),
        CommandRequest::ProjectorModel {
            handle,
            parent_handle,
            embedding,
        } => registered_projector(&handle, &parent_handle, embedding),
        CommandRequest::ApplyProjectorModel {
            handle,
            parent_handle,
            embedding,
            vectors,
            action,
        } => registered_projector_action(&handle, &parent_handle, embedding, vectors, action),
        CommandRequest::ApplyTermsBetweenModels {
            source_handle,
            target_handle,
            terms,
            vectors,
        } => registered_cross_sector_action(&source_handle, &target_handle, terms, vectors),
        CommandRequest::EighModel {
            handle,
            eigenvectors,
            parameters,
        } => {
            let model = registered_model(&handle)?;
            command_eigh(&model, parameters, eigenvectors)
        }
        CommandRequest::EigshModel {
            handle,
            format,
            solver,
            parameters,
        } => {
            let model = registered_model(&handle)?;
            command_eigsh(&model, parameters, format, &solver)
        }
        CommandRequest::EvolveModel {
            handle,
            vectors,
            evolution,
            parameters,
        } => {
            let model = registered_model(&handle)?;
            command_evolve_model(&model, vectors, evolution, parameters)
        }
        CommandRequest::ReleaseModel { handle } => {
            let handle = release_model(&handle)?;
            Ok(CommandResult::Released { handle })
        }
        CommandRequest::ReleaseUserBasis { user_basis_handle } => {
            let handle = release_user_basis(&user_basis_handle)?;
            Ok(CommandResult::Released { handle })
        }
        CommandRequest::DescribeBasis { .. }
        | CommandRequest::BitwiseStates { .. }
        | CommandRequest::AnalyzeDiagonalEnsemble { .. }
        | CommandRequest::Materialize { .. }
        | CommandRequest::Eigh { .. }
        | CommandRequest::Eigsh { .. }
        | CommandRequest::CreateModel { .. }
        | CommandRequest::CreateOperatorModel { .. }
        | CommandRequest::CreateOperatorExpressionModel { .. }
        | CommandRequest::CreateProjectedBlockModel { .. }
        | CommandRequest::CreateBlockModel { .. }
        | CommandRequest::LoadOperatorArchive { .. }
        | CommandRequest::CreateBasisPlan { .. } => {
            unreachable!("stateless command was pre-dispatched")
        }
    }
}

fn registered_describe_model(handle: &str) -> Result<CommandResult> {
    let model = registered_model(handle)?;
    let states = model.states()?;
    Ok(CommandResult::Basis {
        dimension: model.dimension(),
        states,
    })
}

fn execute_basis_plan_command(request: CommandRequest) -> Result<CommandResult> {
    match request {
        CommandRequest::ReduceStatesPlan {
            plan_handle,
            states,
        } => {
            let plan = registered_basis_plan(&plan_handle)?;
            command_reduce_states_plan(&plan, states)
        }
        CommandRequest::MaterializeBasisPlan { plan_handle } => {
            materialize_basis_plan(&plan_handle)
        }
        CommandRequest::ReleaseBasisPlan { plan_handle } => {
            let handle = release_basis_plan(&plan_handle)?;
            Ok(CommandResult::Released { handle })
        }
        CommandRequest::BraKetTermsPlan {
            plan_handle,
            terms,
            kets,
        } => {
            let plan = registered_basis_plan(&plan_handle)?;
            command_bra_ket_terms_plan(&plan, terms, kets)
        }
        _ => unreachable!("only basis-plan commands are routed here"),
    }
}

fn typed_term(term: TermRequest) -> Result<OperatorSpec> {
    let local = term
        .product
        .local
        .iter()
        .map(|name| typed_local_operator(name))
        .collect::<Result<Vec<_>>>()?;
    let product = if term.product.splits.is_empty() {
        OpProduct::with_split(local, term.product.split)?
    } else {
        if term.product.split.is_some() {
            return Err(QmbedError::InvalidOptions(
                "operator product must use either split or splits, not both".into(),
            ));
        }
        OpProduct::with_splits(local, term.product.splits)?
    };
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

fn into_c_response(response: String) -> *mut c_char {
    CString::new(response).unwrap_or_default().into_raw()
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
    into_c_response(response)
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
    into_c_response(response)
}

/// Evolve a registered parameterized model while synchronously obtaining its
/// component coefficients from a language-neutral C callback.
///
/// # Safety
///
/// `request` must point to a live NUL-terminated UTF-8 string. `callback` must
/// remain callable for the duration of this function, and `context` must
/// remain valid according to the callback's own contract. The returned string
/// must be released exactly once with [`qmbed_string_free`].
///
/// # Panics
///
/// Panics only if Rust's JSON serializer produces an interior NUL byte; the
/// serialized response schema cannot contain one.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qmbed_evolve_model_with_drive_json(
    request: *const c_char,
    callback: Option<QmbedDriveCallback>,
    context: *mut c_void,
) -> *mut c_char {
    let response = catch_unwind(AssertUnwindSafe(|| {
        if request.is_null() {
            return r#"{"status":"error","error":"request pointer is null"}"#.to_string();
        }
        let Some(callback) = callback else {
            return r#"{"status":"error","error":"drive callback is null"}"#.to_string();
        };
        // SAFETY: The caller contract requires a live NUL-terminated string.
        let request = unsafe { CStr::from_ptr(request) };
        match request.to_str() {
            Ok(request) => run_drive_evolution_json(request, callback, context as usize),
            Err(error) => {
                format!(r#"{{"status":"error","error":"request is not UTF-8: {error}"}}"#)
            }
        }
    }))
    .unwrap_or_else(|_| r#"{"status":"error","error":"Rust binding panic"}"#.to_string());
    into_c_response(response)
}

/// Register a callback-defined 32-bit user basis.
///
/// The callback ABI intentionally matches `QuSpin`'s public Numba signatures,
/// while the resulting basis is stored behind QMBED's ordinary runtime basis
/// interface. This keeps operator assembly, symmetry reduction, projection,
/// and measurements on the same Rust execution path as built-in bases.
///
/// # Safety
///
/// `request` must be a live NUL-terminated UTF-8 string. `operator` must be a
/// valid callback. Optional callbacks must either be null or callable with the
/// documented ABI. `map_callbacks` must point to `map_count` valid callback
/// entries (or be null when `map_count == 0`). Every callback must remain
/// callable until the returned user-basis handle is released and all models
/// created from it have finished using the basis.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qmbed_register_user_basis_32_json(
    request: *const c_char,
    operator: Option<QmbedUserOp32Callback>,
    next_state: Option<QmbedUserNextState32Callback>,
    pre_check: Option<QmbedUserPreCheck32Callback>,
    map_callbacks: *const QmbedUserMap32Callback,
    map_count: usize,
) -> *mut c_char {
    let response = catch_unwind(AssertUnwindSafe(|| {
        if request.is_null() {
            return r#"{"status":"error","error":"request pointer is null"}"#.to_string();
        }
        let Some(operator) = operator else {
            return r#"{"status":"error","error":"user operator callback is null"}"#.to_string();
        };
        if map_count > 0 && map_callbacks.is_null() {
            return r#"{"status":"error","error":"user symmetry callback array is null"}"#
                .to_string();
        }
        // SAFETY: The caller contract provides `map_count` live callback
        // entries, or a zero-length slice represented by a dangling pointer.
        let maps = if map_count == 0 {
            &[]
        } else {
            unsafe { std::slice::from_raw_parts(map_callbacks, map_count) }
        };
        // SAFETY: The caller contract requires a live NUL-terminated string.
        let request = unsafe { CStr::from_ptr(request) };
        match request.to_str() {
            Ok(request) => {
                run_user_basis_32_registration_json(request, operator, next_state, pre_check, maps)
            }
            Err(error) => {
                format!(r#"{{"status":"error","error":"request is not UTF-8: {error}"}}"#)
            }
        }
    }))
    .unwrap_or_else(|_| r#"{"status":"error","error":"Rust binding panic"}"#.to_string());
    into_c_response(response)
}

/// Register a callback-defined 64-bit user basis using `QuSpin`'s public ABI.
///
/// # Safety
///
/// The pointer and lifetime requirements are identical to
/// [`qmbed_register_user_basis_32_json`], with every state and argument value
/// widened to `u64`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qmbed_register_user_basis_64_json(
    request: *const c_char,
    operator: Option<QmbedUserOp64Callback>,
    next_state: Option<QmbedUserNextState64Callback>,
    pre_check: Option<QmbedUserPreCheck64Callback>,
    map_callbacks: *const QmbedUserMap64Callback,
    map_count: usize,
) -> *mut c_char {
    let response = catch_unwind(AssertUnwindSafe(|| {
        if request.is_null() {
            return r#"{"status":"error","error":"request pointer is null"}"#.to_string();
        }
        let Some(operator) = operator else {
            return r#"{"status":"error","error":"user operator callback is null"}"#.to_string();
        };
        if map_count > 0 && map_callbacks.is_null() {
            return r#"{"status":"error","error":"user symmetry callback array is null"}"#
                .to_string();
        }
        // SAFETY: The caller supplies `map_count` live callback entries.
        let maps = if map_count == 0 {
            &[]
        } else {
            unsafe { std::slice::from_raw_parts(map_callbacks, map_count) }
        };
        // SAFETY: The caller supplies a live NUL-terminated string.
        let request = unsafe { CStr::from_ptr(request) };
        match request.to_str() {
            Ok(request) => {
                run_user_basis_64_registration_json(request, operator, next_state, pre_check, maps)
            }
            Err(error) => {
                format!(r#"{{"status":"error","error":"request is not UTF-8: {error}"}}"#)
            }
        }
    }))
    .unwrap_or_else(|_| r#"{"status":"error","error":"Rust binding panic"}"#.to_string());
    into_c_response(response)
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
    use std::ffi::c_void;
    use std::sync::Arc;
    use std::thread;
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::{
        Complex64, QmbedComplex64, cached_projector, run_command_json, run_drive_evolution_json,
        run_json,
    };

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
    fn command_eigsh_passes_initial_vectors_to_the_rust_krylov_solver() {
        let dimension = 129;
        let entries = (0..dimension)
            .map(|index| {
                serde_json::json!({
                    "row": index,
                    "column": index,
                    "value": [index as f64, 0.0],
                })
            })
            .collect::<Vec<_>>();
        let mut initial = vec![[0.0, 0.0]; dimension];
        initial[dimension - 2] = [1.0, 0.0];
        initial[dimension - 1] = [1.0, 0.0];
        let request = serde_json::json!({
            "operation": "eigsh_operator_expression",
            "expression": {
                "kind": "matrix",
                "operator": {
                    "shape": [dimension, dimension],
                    "entries": entries,
                },
            },
            "format": "csc",
            "solver": {
                "eigenpairs": 1,
                "target": {"kind": "smallest_algebraic"},
                "krylov_dimension": 2,
                "max_iterations": 2,
                "tolerance": 1.0e-12,
                "initial_vector": initial,
            },
        });
        let response: Value =
            serde_json::from_str(&run_command_json(&request.to_string())).unwrap();

        assert_eq!(response["status"], "ok");
        assert!(
            (response["result"]["eigenvalues"][0].as_f64().unwrap() - (dimension - 2) as f64).abs()
                < 1.0e-10
        );
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
    fn command_protocol_describes_wide_bases_and_applies_erased_bitwise_actions() {
        let described: Value = serde_json::from_str(&run_command_json(
            r#"{
                "operation":"describe_basis",
                "basis":{"kind":"spin","sites":400,"up":1,"reverse":true}
            }"#,
        ))
        .unwrap();
        assert_eq!(described["status"], "ok");
        assert_eq!(described["result"]["dimension"], 400);
        assert_eq!(
            described["result"]["states"][0],
            qmbed::basis::ErasedState::from_decimal(400, "1")
                .unwrap()
                .left_shift(399)
                .unwrap()
                .to_decimal()
        );

        let shifted: Value = serde_json::from_str(&run_command_json(
            r#"{
                "operation":"bitwise_states",
                "bitwise_operation":"left_shift",
                "width_bits":600,
                "left":["1","3"],
                "shifts":[599,598]
            }"#,
        ))
        .unwrap();
        assert_eq!(shifted["status"], "ok");
        assert_eq!(
            shifted["result"]["values"][0],
            qmbed::basis::ErasedState::from_decimal(600, "1")
                .unwrap()
                .left_shift(599)
                .unwrap()
                .to_decimal()
        );
        assert_eq!(
            shifted["result"]["values"][1],
            qmbed::basis::ErasedState::from_decimal(600, "3")
                .unwrap()
                .left_shift(598)
                .unwrap()
                .to_decimal()
        );

        let inverted: Value = serde_json::from_str(&run_command_json(
            r#"{
                "operation":"bitwise_states",
                "bitwise_operation":"not",
                "width_bits":8,
                "left":["1"]
            }"#,
        ))
        .unwrap();
        assert_eq!(inverted["status"], "ok");
        assert_eq!(inverted["result"]["values"][0], "254");
    }

    #[test]
    fn wide_registered_model_reuses_symmetry_assembly_and_solver_commands() {
        let sites = 200;
        let destinations = (0..sites)
            .map(|site| (site + 1) % sites)
            .collect::<Vec<_>>();
        let number_couplings = (0..sites)
            .map(|site| {
                serde_json::json!({
                    "coefficient": [1.0, 0.0],
                    "sites": [site],
                })
            })
            .collect::<Vec<_>>();
        let create = serde_json::json!({
            "operation": "create_model",
            "basis": {
                "kind": "spin",
                "sites": sites,
                "up": 1,
                "symmetries": [{
                    "destinations": destinations,
                    "sector": 0,
                }],
            },
            "terms": [{
                "product": {"local": ["number"]},
                "couplings": number_couplings,
            }],
        });
        let create: Value = serde_json::from_str(&run_command_json(&create.to_string())).unwrap();
        assert_eq!(create["status"], "ok");
        assert_eq!(create["result"]["dimension"], 1);
        let handle = create["result"]["handle"].as_str().unwrap();

        let materialized: Value = serde_json::from_str(&run_command_json(&format!(
            r#"{{"operation":"materialize_model","handle":"{handle}","format":"csc"}}"#
        )))
        .unwrap();
        assert_eq!(materialized["status"], "ok");
        assert_eq!(materialized["result"]["shape"], serde_json::json!([1, 1]));
        assert!(
            (materialized["result"]["entries"][0]["value"][0]
                .as_f64()
                .unwrap()
                - 1.0)
                .abs()
                < 1.0e-12
        );

        let high_state = qmbed::basis::ErasedState::from_decimal(sites, "1")
            .unwrap()
            .left_shift(sites - 1)
            .unwrap()
            .to_decimal();
        let reduced: Value = serde_json::from_str(&run_command_json(&format!(
            r#"{{
                "operation":"reduce_states_model",
                "handle":"{handle}",
                "states":["{high_state}"]
            }}"#
        )))
        .unwrap();
        assert_eq!(reduced["status"], "ok");
        assert_eq!(reduced["result"]["entries"][0]["representative"], "1");
        assert_eq!(reduced["result"]["entries"][0]["orbit_size"], sites);

        let transitions: Value = serde_json::from_str(&run_command_json(&format!(
            r#"{{
                "operation":"bra_ket_terms_model",
                "handle":"{handle}",
                "terms":[{{
                    "product":{{"local":["number"]}},
                    "couplings":[{{
                        "coefficient":[1.0,0.0],
                        "sites":[199]
                    }}]
                }}],
                "kets":["{high_state}"]
            }}"#
        )))
        .unwrap();
        assert_eq!(transitions["status"], "ok");
        assert_eq!(transitions["result"]["entries"][0]["bra"], high_state);
        assert_eq!(transitions["result"]["entries"][0]["ket"], high_state);

        let parent = serde_json::json!({
            "operation": "create_model",
            "basis": {
                "kind": "spin",
                "sites": sites,
                "up": 1,
            },
            "terms": [],
        });
        let parent: Value = serde_json::from_str(&run_command_json(&parent.to_string())).unwrap();
        assert_eq!(parent["status"], "ok");
        assert_eq!(parent["result"]["dimension"], sites);
        let parent_handle = parent["result"]["handle"].as_str().unwrap();
        let projector: Value = serde_json::from_str(&run_command_json(&format!(
            r#"{{
                "operation":"projector_model",
                "handle":"{handle}",
                "parent_handle":"{parent_handle}"
            }}"#
        )))
        .unwrap();
        assert_eq!(projector["status"], "ok");
        assert_eq!(projector["result"]["shape"], serde_json::json!([sites, 1]));
        assert_eq!(
            projector["result"]["entries"].as_array().unwrap().len(),
            sites
        );

        let source = serde_json::json!({
            "operation": "create_model",
            "basis": {
                "kind": "spin",
                "sites": sites,
                "up": 0,
                "symmetries": [{
                    "destinations": (0..sites)
                        .map(|site| (site + 1) % sites)
                        .collect::<Vec<_>>(),
                    "sector": 0,
                }],
            },
            "terms": [],
        });
        let source: Value = serde_json::from_str(&run_command_json(&source.to_string())).unwrap();
        assert_eq!(source["status"], "ok");
        let source_handle = source["result"]["handle"].as_str().unwrap();
        let shifted = serde_json::json!({
            "operation": "apply_terms_between_models",
            "source_handle": source_handle,
            "target_handle": handle,
            "terms": [{
                "product": {"local": ["raising"]},
                "couplings": (0..sites)
                    .map(|site| serde_json::json!({
                        "coefficient": [1.0, 0.0],
                        "sites": [site],
                    }))
                    .collect::<Vec<_>>(),
            }],
            "vectors": [[[1.0, 0.0]]],
        });
        let shifted: Value = serde_json::from_str(&run_command_json(&shifted.to_string())).unwrap();
        assert_eq!(shifted["status"], "ok");
        assert!(
            (shifted["result"]["vectors"][0][0][0].as_f64().unwrap() - (sites as f64).sqrt()).abs()
                < 1.0e-12
        );

        let eigh: Value = serde_json::from_str(&run_command_json(&format!(
            r#"{{"operation":"eigh_model","handle":"{handle}"}}"#
        )))
        .unwrap();
        assert_eq!(eigh["status"], "ok");
        assert!((eigh["result"]["eigenvalues"][0].as_f64().unwrap() - 1.0).abs() < 1.0e-12);
    }

    #[test]
    fn wide_matrix_symmetry_reuses_the_erased_state_projected_model_path() {
        let sites = 200;
        let angle = 2.0 * std::f64::consts::PI / 200.0;
        let phase = Complex64::from_polar(1.0, angle);
        let translation = (0..sites)
            .map(|site| (site + 1) % sites)
            .collect::<Vec<_>>();
        let reflection = (0..sites).map(|site| sites - site - 1).collect::<Vec<_>>();
        let matrix_symmetry = serde_json::json!({
            "dimension": 2,
            "selected_row": 0,
            "generators": [
                {
                    "destinations": translation,
                    "matrix": [
                        [[phase.re, phase.im], [0.0, 0.0]],
                        [[0.0, 0.0], [phase.re, -phase.im]]
                    ]
                },
                {
                    "destinations": reflection,
                    "matrix": [
                        [[0.0, 0.0], [1.0, 0.0]],
                        [[1.0, 0.0], [0.0, 0.0]]
                    ]
                }
            ]
        });
        let request = serde_json::json!({
            "kind": "spin",
            "sites": sites,
            "up": 1,
            "matrix_symmetry": matrix_symmetry,
            "reverse": true
        });
        let described: Value = serde_json::from_str(&run_command_json(
            &serde_json::json!({
                "operation": "describe_basis",
                "basis": request.clone()
            })
            .to_string(),
        ))
        .unwrap();
        assert_eq!(described["status"], "ok");
        assert_eq!(described["result"]["dimension"], 1);

        let create: Value = serde_json::from_str(&run_command_json(
            &serde_json::json!({
                "operation": "create_model",
                "basis": request,
                "terms": [{
                    "product": {"local": ["number"]},
                    "couplings": (0..sites)
                        .map(|site| serde_json::json!({
                            "coefficient": [1.0, 0.0],
                            "sites": [site]
                        }))
                        .collect::<Vec<_>>()
                }]
            })
            .to_string(),
        ))
        .unwrap();
        assert_eq!(create["status"], "ok");
        assert_eq!(create["result"]["dimension"], 1);
        let handle = create["result"]["handle"].as_str().unwrap();

        let parent: Value = serde_json::from_str(&run_command_json(
            &serde_json::json!({
                "operation": "create_model",
                "basis": {"kind": "spin", "sites": sites, "up": 1},
                "terms": []
            })
            .to_string(),
        ))
        .unwrap();
        assert_eq!(parent["status"], "ok");
        let parent_handle = parent["result"]["handle"].as_str().unwrap();
        let projector: Value = serde_json::from_str(&run_command_json(&format!(
            r#"{{
                "operation":"projector_model",
                "handle":"{handle}",
                "parent_handle":"{parent_handle}"
            }}"#
        )))
        .unwrap();
        assert_eq!(projector["status"], "ok");
        assert_eq!(projector["result"]["shape"], serde_json::json!([sites, 1]));
        assert_eq!(
            projector["result"]["entries"].as_array().unwrap().len(),
            sites
        );

        let eigensystem: Value = serde_json::from_str(&run_command_json(&format!(
            r#"{{"operation":"eigh_model","handle":"{handle}"}}"#
        )))
        .unwrap();
        assert_eq!(eigensystem["status"], "ok");
        assert!((eigensystem["result"]["eigenvalues"][0].as_f64().unwrap() - 1.0).abs() < 1.0e-12);
    }

    #[test]
    fn command_protocol_analyzes_diagonal_ensembles_over_operator_expressions() {
        let response: Value = serde_json::from_str(&run_command_json(
            r#"{
                "operation":"analyze_diagonal_ensemble",
                "eigenvalues":[-1.0,1.0],
                "eigenvectors":[
                    [[1.0,0.0],[0.0,0.0]],
                    [[0.0,0.0],[1.0,0.0]]
                ],
                "input":{
                    "kind":"pure",
                    "values":[[0.7071067811865475,0.0],[0.7071067811865475,0.0]]
                },
                "observable":{
                    "kind":"matrix",
                    "operator":{
                        "shape":[2,2],
                        "entries":[
                            {"row":0,"column":1,"value":[1.0,0.0]},
                            {"row":1,"column":0,"value":[1.0,0.0]}
                        ]
                    }
                },
                "alpha":2.0
            }"#,
        ))
        .unwrap();
        assert_eq!(response["status"], "ok");
        assert_eq!(
            response["result"]["probabilities"][0],
            serde_json::json!([0.5, 0.5])
        );
        assert!(
            (response["result"]["diagonal_entropies"][0]
                .as_f64()
                .unwrap()
                - 2.0_f64.ln())
            .abs()
                < 1.0e-12
        );
        assert_eq!(response["result"]["observables"][0], 0.0);
        assert!(
            (response["result"]["temporal_fluctuations"][0]
                .as_f64()
                .unwrap()
                - 0.5_f64.sqrt())
            .abs()
                < 1.0e-12
        );
        assert!(
            (response["result"]["quantum_fluctuations"][0]
                .as_f64()
                .unwrap()
                - 0.5_f64.sqrt())
            .abs()
                < 1.0e-12
        );
    }

    #[test]
    fn command_protocol_returns_an_empty_spectrum_for_an_empty_valid_sector() {
        let create = run_command_json(
            r#"{
                "operation":"create_model",
                "basis":{
                    "kind":"spin",
                    "sites":2,
                    "up":0,
                    "symmetries":[{"destinations":[1,0],"sector":1}]
                },
                "terms":[]
            }"#,
        );
        let create: Value = serde_json::from_str(&create).unwrap();
        assert_eq!(create["status"], "ok");
        assert_eq!(create["result"]["dimension"], 0);
        let handle = create["result"]["handle"].as_str().unwrap();

        let eigh = run_command_json(&format!(
            r#"{{"operation":"eigh_model","handle":"{handle}","eigenvectors":true}}"#
        ));
        let eigh: Value = serde_json::from_str(&eigh).unwrap();
        assert_eq!(eigh["status"], "ok");
        assert_eq!(eigh["result"]["dimension"], 0);
        assert_eq!(eigh["result"]["eigenvalues"], serde_json::json!([]));
        assert_eq!(eigh["result"]["eigenvectors"], serde_json::json!([]));
        assert_eq!(eigh["result"]["iterations"], 0);

        let release = run_command_json(&format!(
            r#"{{"operation":"release_model","handle":"{handle}"}}"#
        ));
        let release: Value = serde_json::from_str(&release).unwrap();
        assert_eq!(release["status"], "ok");
    }

    #[test]
    fn registered_model_evolves_a_column_batch_with_one_static_contract() {
        let create = run_command_json(
            r#"{
                "operation":"create_model",
                "basis":{"kind":"spin","sites":1},
                "terms":[{
                    "product":{"local":["identity"]},
                    "couplings":[{"coefficient":[2.0,0.0],"sites":[0]}]
                }]
            }"#,
        );
        let create: Value = serde_json::from_str(&create).unwrap();
        assert_eq!(create["status"], "ok");
        let handle = create["result"]["handle"].as_str().unwrap();

        let time = std::f64::consts::FRAC_PI_4;
        let evolved = run_command_json(&format!(
            r#"{{
                "operation":"evolve_model",
                "handle":"{handle}",
                "vectors":[
                    [[1.0,0.0],[0.0,0.0]],
                    [[0.0,0.0],[1.0,0.0]]
                ],
                "evolution":{{
                    "times":[0.0,{time}],
                    "krylov_dimension":8,
                    "tolerance":1e-12,
                    "max_substeps":100
                }}
            }}"#
        ));
        let evolved: Value = serde_json::from_str(&evolved).unwrap();
        assert_eq!(evolved["status"], "ok");
        assert_eq!(evolved["result"]["kind"], "trajectory");
        assert_eq!(evolved["result"]["dimension"], 2);
        assert_eq!(
            evolved["result"]["times"],
            serde_json::json!([0.0, std::f64::consts::FRAC_PI_4])
        );
        let states = evolved["result"]["states"].as_array().unwrap();
        assert_eq!(states.len(), 2);
        assert_eq!(states[0].as_array().unwrap().len(), 2);
        for (column, initial_column) in states[0].as_array().unwrap().iter().enumerate() {
            for (component, initial) in initial_column.as_array().unwrap().iter().enumerate() {
                let expected_real = if column == component { 1.0 } else { 0.0 };
                assert!((initial[0].as_f64().unwrap() - expected_real).abs() < 1.0e-12);
                assert!(initial[1].as_f64().unwrap().abs() < 1.0e-12);

                let final_value = &states[1][column][component];
                assert!(final_value[0].as_f64().unwrap().abs() < 1.0e-12);
                assert!((final_value[1].as_f64().unwrap() + expected_real).abs() < 1.0e-12);
            }
        }

        let release = run_command_json(&format!(
            r#"{{"operation":"release_model","handle":"{handle}"}}"#
        ));
        let release: Value = serde_json::from_str(&release).unwrap();
        assert_eq!(release["status"], "ok");
    }

    #[test]
    fn c_drive_callback_observes_absolute_internal_times() {
        extern "C" fn linear_drive(
            _context: *mut c_void,
            time: f64,
            coefficients: *mut QmbedComplex64,
            count: usize,
        ) -> i32 {
            if count != 1 || coefficients.is_null() {
                return 1;
            }
            // SAFETY: The callback contract supplies one writable coefficient.
            unsafe {
                (*coefficients).real = time;
                (*coefficients).imaginary = 0.0;
            }
            0
        }

        let create = run_command_json(
            r#"{
                "operation":"create_operator_model",
                "static_operator":{"shape":[2,2],"entries":[]},
                "components":[{
                    "name":"field",
                    "operator":{
                        "shape":[2,2],
                        "entries":[
                            {"row":0,"column":0,"value":[1.0,0.0]},
                            {"row":1,"column":1,"value":[-1.0,0.0]}
                        ]
                    }
                }]
            }"#,
        );
        let create: Value = serde_json::from_str(&create).unwrap();
        assert_eq!(create["status"], "ok");
        let handle = create["result"]["handle"].as_str().unwrap();
        let amplitude = 1.0 / 2.0_f64.sqrt();
        let response = run_drive_evolution_json(
            &format!(
                r#"{{
                    "handle":"{handle}",
                    "component_names":["field"],
                    "vectors":[[[{amplitude},0.0],[{amplitude},0.0]]],
                    "initial_time":2.0,
                    "evolution":{{
                        "times":[2.0,2.5],
                        "krylov_dimension":16,
                        "tolerance":1e-10,
                        "max_substeps":10000
                    }}
                }}"#
            ),
            linear_drive,
            std::ptr::null_mut::<c_void>() as usize,
        );
        let response: Value = serde_json::from_str(&response).unwrap();
        assert_eq!(response["status"], "ok");
        let final_state = &response["result"]["states"][1][0];
        let phase = (2.5_f64.powi(2) - 2.0_f64.powi(2)) / 2.0;
        let expected = [
            [amplitude * phase.cos(), -amplitude * phase.sin()],
            [amplitude * phase.cos(), amplitude * phase.sin()],
        ];
        for (index, expected) in expected.iter().enumerate() {
            let actual = &final_state[index];
            assert!((actual[0].as_f64().unwrap() - expected[0]).abs() < 2.0e-9);
            assert!((actual[1].as_f64().unwrap() - expected[1]).abs() < 2.0e-9);
        }

        let release = run_command_json(&format!(
            r#"{{"operation":"release_model","handle":"{handle}"}}"#
        ));
        let release: Value = serde_json::from_str(&release).unwrap();
        assert_eq!(release["status"], "ok");
    }

    #[test]
    fn direct_operator_model_reuses_named_components_across_commands() {
        let create = run_command_json(
            r#"{
                "operation":"create_operator_model",
                "static_operator":{
                    "shape":[3,3],
                    "entries":[
                        {"row":0,"column":0,"value":[1.0,0.0]},
                        {"row":1,"column":1,"value":[2.0,0.0]},
                        {"row":2,"column":2,"value":[3.0,0.0]}
                    ]
                },
                "components":[{
                    "name":"field",
                    "default":[1.0,0.0],
                    "operator":{
                        "shape":[3,3],
                        "entries":[
                            {"row":0,"column":0,"value":[-1.0,0.0]},
                            {"row":2,"column":2,"value":[1.0,0.0]}
                        ]
                    }
                }]
            }"#,
        );
        let create: Value = serde_json::from_str(&create).unwrap();
        assert_eq!(create["status"], "ok");
        assert_eq!(create["result"]["dimension"], 3);
        let handle = create["result"]["handle"].as_str().unwrap();

        let parameters = r#"{"field":[2.0,0.0]}"#;
        let materialized = run_command_json(&format!(
            r#"{{
                "operation":"materialize_model",
                "handle":"{handle}",
                "format":"csc",
                "parameters":{parameters}
            }}"#
        ));
        let materialized: Value = serde_json::from_str(&materialized).unwrap();
        assert_eq!(materialized["status"], "ok");
        let entries = materialized["result"]["entries"].as_array().unwrap();
        let diagonal = entries
            .iter()
            .map(|entry| entry["value"][0].as_f64().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(diagonal, [-1.0, 2.0, 5.0]);

        let eigh = run_command_json(&format!(
            r#"{{
                "operation":"eigh_model",
                "handle":"{handle}",
                "parameters":{parameters}
            }}"#
        ));
        let eigh: Value = serde_json::from_str(&eigh).unwrap();
        assert_eq!(eigh["status"], "ok");
        assert_eq!(
            eigh["result"]["eigenvalues"],
            serde_json::json!([-1.0, 2.0, 5.0])
        );

        let applied = run_command_json(&format!(
            r#"{{
                "operation":"apply_model",
                "handle":"{handle}",
                "parameters":{parameters},
                "vectors":[[[1.0,0.0],[1.0,0.0],[1.0,0.0]]]
            }}"#
        ));
        let applied: Value = serde_json::from_str(&applied).unwrap();
        assert_eq!(applied["status"], "ok");
        assert_eq!(
            applied["result"]["vectors"][0],
            serde_json::json!([[-1.0, 0.0], [2.0, 0.0], [5.0, 0.0]])
        );

        let release = run_command_json(&format!(
            r#"{{"operation":"release_model","handle":"{handle}"}}"#
        ));
        let release: Value = serde_json::from_str(&release).unwrap();
        assert_eq!(release["status"], "ok");
    }

    #[test]
    fn command_protocol_round_trips_parameterized_operator_archives() {
        let create = run_command_json(
            r#"{
                "operation":"create_operator_model",
                "basis":{"kind":"spin","sites":1,"pauli":true,"reverse":true},
                "components":[
                    {
                        "name":"diagonal",
                        "default":[0.5,0.0],
                        "terms":[{
                            "product":{"local":["z"]},
                            "couplings":[{"coefficient":[1.0,0.0],"sites":[0]}]
                        }]
                    },
                    {
                        "name":"exchange",
                        "operator":{
                            "shape":[2,2],
                            "entries":[
                                {"row":0,"column":1,"value":[2.0,0.0]},
                                {"row":1,"column":0,"value":[2.0,0.0]}
                            ]
                        }
                    }
                ]
            }"#,
        );
        let create: Value = serde_json::from_str(&create).unwrap();
        assert_eq!(create["status"], "ok");
        let original_handle = create["result"]["handle"].as_str().unwrap();
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "qmbed-capi-operator-archive-{}-{nonce}.zip",
            std::process::id()
        ));

        let saved = run_command_json(
            &serde_json::json!({
                "operation": "save_operator_archive",
                "handle": original_handle,
                "path": path,
                "formats": {
                    "diagonal": "dia",
                    "exchange": "csr"
                },
                "metadata": {"scalar_dtype": "float32"}
            })
            .to_string(),
        );
        let saved: Value = serde_json::from_str(&saved).unwrap();
        assert_eq!(saved["status"], "ok");
        assert_eq!(saved["result"]["kind"], "archive");
        assert_eq!(saved["result"]["components"][0]["name"], "diagonal");
        assert_eq!(saved["result"]["components"][0]["format"], "dia");
        assert_eq!(saved["result"]["components"][1]["name"], "exchange");
        assert_eq!(saved["result"]["components"][1]["format"], "csr");
        assert_eq!(saved["result"]["metadata"]["scalar_dtype"], "float32");

        let loaded = run_command_json(
            &serde_json::json!({
                "operation": "load_operator_archive",
                "path": path,
            })
            .to_string(),
        );
        let loaded: Value = serde_json::from_str(&loaded).unwrap();
        assert_eq!(loaded["status"], "ok");
        assert_eq!(loaded["result"]["kind"], "archived_model");
        assert_eq!(loaded["result"]["dimension"], 2);
        assert_eq!(loaded["result"]["components"][0]["format"], "dia");
        assert_eq!(loaded["result"]["components"][1]["format"], "csr");
        assert_eq!(loaded["result"]["metadata"]["scalar_dtype"], "float32");
        let loaded_handle = loaded["result"]["handle"].as_str().unwrap();

        let parameters = serde_json::json!({
            "diagonal": [1.25, 0.0],
            "exchange": [-0.5, 0.0],
        });
        let materialize = |handle: &str| {
            let response = run_command_json(
                &serde_json::json!({
                    "operation": "materialize_model",
                    "handle": handle,
                    "format": "dense",
                    "parameters": parameters,
                })
                .to_string(),
            );
            let response: Value = serde_json::from_str(&response).unwrap();
            assert_eq!(response["status"], "ok");
            response["result"]["entries"].clone()
        };
        assert_eq!(materialize(original_handle), materialize(loaded_handle));

        for handle in [original_handle, loaded_handle] {
            let released = run_command_json(
                &serde_json::json!({
                    "operation": "release_model",
                    "handle": handle,
                })
                .to_string(),
            );
            let released: Value = serde_json::from_str(&released).unwrap();
            assert_eq!(released["status"], "ok");
        }
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn basis_model_reuses_named_local_term_components_across_commands() {
        let create = run_command_json(
            r#"{
                "operation":"create_model",
                "basis":{"kind":"spin","sites":1,"reverse":true},
                "terms":[{
                    "product":{"local":["identity"]},
                    "couplings":[{"coefficient":[1.0,0.0],"sites":[0]}]
                }],
                "components":[{
                    "name":"field",
                    "default":[1.0,0.0],
                    "terms":[{
                        "product":{"local":["z"]},
                        "couplings":[{"coefficient":[1.0,0.0],"sites":[0]}]
                    }]
                }]
            }"#,
        );
        let create: Value = serde_json::from_str(&create).unwrap();
        assert_eq!(create["status"], "ok");
        assert_eq!(create["result"]["dimension"], 2);
        let handle = create["result"]["handle"].as_str().unwrap();

        let parameters = r#"{"field":[2.0,0.0]}"#;
        let eigh = run_command_json(&format!(
            r#"{{
                "operation":"eigh_model",
                "handle":"{handle}",
                "parameters":{parameters}
            }}"#
        ));
        let eigh: Value = serde_json::from_str(&eigh).unwrap();
        assert_eq!(eigh["status"], "ok");
        assert_eq!(eigh["result"]["eigenvalues"], serde_json::json!([0.0, 2.0]));

        let described = run_command_json(&format!(
            r#"{{"operation":"describe_model","handle":"{handle}"}}"#
        ));
        let described: Value = serde_json::from_str(&described).unwrap();
        assert_eq!(described["status"], "ok");
        assert_eq!(described["result"]["states"], serde_json::json!(["1", "0"]));

        let temporary = run_command_json(&format!(
            r#"{{
                "operation":"materialize_terms_model",
                "handle":"{handle}",
                "terms":[{{
                    "product":{{"local":["x"]}},
                    "couplings":[{{"coefficient":[1.0,0.0],"sites":[0]}}]
                }}],
                "checks":{{
                    "hermiticity":false,
                    "particle_conservation":false,
                    "symmetry_compatibility":false
                }}
            }}"#
        ));
        let temporary: Value = serde_json::from_str(&temporary).unwrap();
        assert_eq!(temporary["status"], "ok");
        assert_eq!(temporary["result"]["entries"].as_array().unwrap().len(), 2);

        let release = run_command_json(&format!(
            r#"{{"operation":"release_model","handle":"{handle}"}}"#
        ));
        let release: Value = serde_json::from_str(&release).unwrap();
        assert_eq!(release["status"], "ok");
    }

    #[test]
    fn operator_projection_preserves_basis_model_components() {
        let create = run_command_json(
            r#"{
                "operation":"create_model",
                "basis":{"kind":"spin","sites":1,"reverse":true},
                "terms":[{
                    "product":{"local":["identity"]},
                    "couplings":[{"coefficient":[1.0,0.0],"sites":[0]}]
                }],
                "components":[{
                    "name":"field",
                    "default":[1.0,0.0],
                    "terms":[{
                        "product":{"local":["z"]},
                        "couplings":[{"coefficient":[1.0,0.0],"sites":[0]}]
                    }]
                }]
            }"#,
        );
        let create: Value = serde_json::from_str(&create).unwrap();
        assert_eq!(create["status"], "ok");
        let source = create["result"]["handle"].as_str().unwrap();

        let projected = run_command_json(&format!(
            r#"{{
                "operation":"project_operator_model",
                "handle":"{source}",
                "projector":{{
                    "shape":[2,1],
                    "entries":[{{"row":0,"column":0,"value":[1.0,0.0]}}]
                }}
            }}"#
        ));
        let projected: Value = serde_json::from_str(&projected).unwrap();
        assert_eq!(projected["status"], "ok");
        assert_eq!(projected["result"]["dimension"], 1);
        let handle = projected["result"]["handle"].as_str().unwrap();

        let materialized = run_command_json(&format!(
            r#"{{
                "operation":"materialize_model",
                "handle":"{handle}",
                "parameters":{{"field":[2.0,0.0]}}
            }}"#
        ));
        let materialized: Value = serde_json::from_str(&materialized).unwrap();
        assert_eq!(materialized["status"], "ok");
        assert_eq!(
            materialized["result"]["entries"][0]["value"],
            serde_json::json!([2.0, 0.0])
        );
    }

    #[test]
    fn recursive_operator_expression_evaluates_model_parameters_and_matrix_algebra() {
        let create = run_command_json(
            r#"{
                "operation":"create_operator_model",
                "static_operator":{
                    "shape":[2,2],
                    "entries":[
                        {"row":0,"column":0,"value":[1.0,0.0]},
                        {"row":1,"column":1,"value":[2.0,0.0]}
                    ]
                },
                "components":[{
                    "name":"field",
                    "default":[1.0,0.0],
                    "operator":{
                        "shape":[2,2],
                        "entries":[
                            {"row":0,"column":0,"value":[1.0,0.0]},
                            {"row":1,"column":1,"value":[-1.0,0.0]}
                        ]
                    }
                }]
            }"#,
        );
        let create: Value = serde_json::from_str(&create).unwrap();
        assert_eq!(create["status"], "ok");
        let handle = create["result"]["handle"].as_str().unwrap();

        let evaluated = run_command_json(&format!(
            r#"{{
                "operation":"evaluate_operator_expression",
                "format":"csr",
                "expression":{{
                    "kind":"transform",
                    "action":"adjoint",
                    "operand":{{
                        "kind":"binary",
                        "operation":"product",
                        "left":{{
                            "kind":"model",
                            "handle":"{handle}",
                            "parameters":{{"field":[2.0,0.0]}}
                        }},
                        "right":{{
                            "kind":"matrix",
                            "operator":{{
                                "shape":[2,2],
                                "entries":[
                                    {{"row":0,"column":1,"value":[1.0,0.0]}},
                                    {{"row":1,"column":0,"value":[1.0,0.0]}}
                                ]
                            }}
                        }}
                    }}
                }}
            }}"#
        ));
        let evaluated: Value = serde_json::from_str(&evaluated).unwrap();
        assert_eq!(evaluated["status"], "ok");
        assert_eq!(evaluated["result"]["format"], "csr");
        assert_eq!(
            evaluated["result"]["entries"],
            serde_json::json!([{
                "row": 1,
                "column": 0,
                "value": [3.0, 0.0]
            }])
        );

        let release = run_command_json(&format!(
            r#"{{"operation":"release_model","handle":"{handle}"}}"#
        ));
        let release: Value = serde_json::from_str(&release).unwrap();
        assert_eq!(release["status"], "ok");
    }

    #[test]
    fn recursive_operator_expression_has_a_native_dense_eigensystem() {
        let result = run_command_json(
            r#"{
                "operation":"eigh_operator_expression",
                "eigenvectors":true,
                "expression":{
                    "kind":"scale",
                    "coefficient":[2.0,0.0],
                    "operand":{
                        "kind":"matrix",
                        "operator":{
                            "shape":[2,2],
                            "entries":[
                                {"row":0,"column":0,"value":[-1.0,0.0]},
                                {"row":1,"column":1,"value":[3.0,0.0]}
                            ]
                        }
                    }
                }
            }"#,
        );
        let result: Value = serde_json::from_str(&result).unwrap();
        assert_eq!(result["status"], "ok");
        assert_eq!(
            result["result"]["eigenvalues"],
            serde_json::json!([-2.0, 6.0])
        );
        assert_eq!(
            result["result"]["eigenvectors"].as_array().unwrap().len(),
            2
        );
    }

    #[test]
    fn recursive_operator_expression_applies_and_inspects_without_export() {
        let expression = serde_json::json!({
            "kind": "binary",
            "operation": "product",
            "left": {
                "kind": "matrix",
                "operator": {
                    "shape": [2, 2],
                    "entries": [
                        {"row": 0, "column": 1, "value": [2.0, 1.0]},
                        {"row": 1, "column": 0, "value": [3.0, 0.0]}
                    ]
                }
            },
            "right": {
                "kind": "matrix",
                "operator": {
                    "shape": [2, 2],
                    "entries": [
                        {"row": 0, "column": 1, "value": [1.0, 0.0]},
                        {"row": 1, "column": 0, "value": [1.0, 0.0]}
                    ]
                }
            }
        });
        let applied = run_command_json(
            &serde_json::json!({
                "operation": "apply_operator_expression",
                "expression": expression.clone(),
                "vectors": [
                    [[1.0, 0.0], [0.0, 0.0]],
                    [[0.0, 0.0], [1.0, 0.0]]
                ]
            })
            .to_string(),
        );
        let applied: Value = serde_json::from_str(&applied).unwrap();
        assert_eq!(applied["status"], "ok");
        assert_eq!(
            applied["result"]["vectors"],
            serde_json::json!([[[2.0, 1.0], [0.0, 0.0]], [[0.0, 0.0], [3.0, 0.0]]])
        );

        let summary = run_command_json(
            &serde_json::json!({
                "operation": "inspect_operator_expression",
                "expression": expression
            })
            .to_string(),
        );
        let summary: Value = serde_json::from_str(&summary).unwrap();
        assert_eq!(summary["status"], "ok");
        assert_eq!(
            summary["result"]["diagonal"],
            serde_json::json!([[2.0, 1.0], [3.0, 0.0]])
        );
        assert_eq!(summary["result"]["trace"], serde_json::json!([5.0, 1.0]));
        assert_eq!(summary["result"]["nonzeros"], 2);

        let exponential = run_command_json(
            &serde_json::json!({
                "operation": "expm_operator_expression",
                "expression": expression,
                "coefficient": [0.1, 0.0],
                "vectors": [[[1.0, 0.0], [0.0, 0.0]]]
            })
            .to_string(),
        );
        let exponential: Value = serde_json::from_str(&exponential).unwrap();
        assert_eq!(exponential["status"], "ok");
        let actual = &exponential["result"]["vectors"][0][0];
        let expected = Complex64::new(0.2, 0.1).exp();
        assert!((actual[0].as_f64().unwrap() - expected.re).abs() < 1.0e-12);
        assert!((actual[1].as_f64().unwrap() - expected.im).abs() < 1.0e-12);
    }

    #[test]
    fn lanczos_resource_reconstructs_and_releases_without_exporting_the_basis() {
        let decomposition = run_command_json(
            r#"{
                "operation":"lanczos_operator",
                "expression":{
                    "kind":"matrix",
                    "operator":{
                        "shape":[2,2],
                        "entries":[
                            {"row":0,"column":0,"value":[-1.0,0.0]},
                            {"row":1,"column":1,"value":[2.0,0.0]}
                        ]
                    }
                },
                "initial":[[0.6,0.0],[0.8,0.0]],
                "krylov_dimension":2,
                "tolerance":1e-13
            }"#,
        );
        let decomposition: Value = serde_json::from_str(&decomposition).unwrap();
        assert_eq!(decomposition["status"], "ok");
        let handle = decomposition["result"]["handle"].as_str().unwrap();
        assert_eq!(decomposition["result"]["krylov_dimension"], 2);

        let exponential = run_command_json(&format!(
            r#"{{
                "operation":"lanczos_exponential",
                "lanczos_handle":"{handle}",
                "coefficient":[0.0,-0.25]
            }}"#
        ));
        let exponential: Value = serde_json::from_str(&exponential).unwrap();
        assert_eq!(exponential["status"], "ok");
        let values = exponential["result"]["vectors"][0].as_array().unwrap();
        let expected = [
            Complex64::new(0.6, 0.0) * Complex64::new(0.0, 0.25).exp(),
            Complex64::new(0.8, 0.0) * Complex64::new(0.0, -0.5).exp(),
        ];
        for (actual, expected) in values.iter().zip(expected) {
            assert!((actual[0].as_f64().unwrap() - expected.re).abs() < 1.0e-12);
            assert!((actual[1].as_f64().unwrap() - expected.im).abs() < 1.0e-12);
        }

        let exported = run_command_json(&format!(
            r#"{{"operation":"export_lanczos_basis","lanczos_handle":"{handle}"}}"#
        ));
        let exported: Value = serde_json::from_str(&exported).unwrap();
        assert_eq!(exported["status"], "ok");
        assert_eq!(exported["result"]["vectors"].as_array().unwrap().len(), 2);

        let thermal = run_command_json(
            &serde_json::json!({
                "operation": "lanczos_thermal",
                "lanczos_handle": handle,
                "method": "ftlm",
                "eigenvalues": decomposition["result"]["eigenvalues"],
                "eigenvectors": decomposition["result"]["eigenvectors"],
                "inverse_temperatures": [0.0, 0.7],
                "observables": [{
                    "name": "I",
                    "expression": {
                        "kind": "matrix",
                        "operator": {
                            "shape": [2, 2],
                            "entries": [
                                {"row": 0, "column": 0, "value": [1.0, 0.0]},
                                {"row": 1, "column": 1, "value": [1.0, 0.0]}
                            ]
                        }
                    }
                }]
            })
            .to_string(),
        );
        let thermal: Value = serde_json::from_str(&thermal).unwrap();
        assert_eq!(thermal["status"], "ok");
        assert_eq!(
            thermal["result"]["values"]["I"],
            serde_json::json!([
                [thermal["result"]["identity"][0].as_f64().unwrap(), 0.0],
                [thermal["result"]["identity"][1].as_f64().unwrap(), 0.0]
            ])
        );

        let release = run_command_json(&format!(
            r#"{{"operation":"release_lanczos","lanczos_handle":"{handle}"}}"#
        ));
        let release: Value = serde_json::from_str(&release).unwrap();
        assert_eq!(release["status"], "ok");
        let stale = run_command_json(&format!(
            r#"{{"operation":"export_lanczos_basis","lanczos_handle":"{handle}"}}"#
        ));
        let stale: Value = serde_json::from_str(&stale).unwrap();
        assert_eq!(stale["status"], "error");
    }

    #[test]
    fn reusable_nonnormal_exponential_action_crosses_the_c_boundary() {
        let create = run_command_json(
            r#"{
                "operation":"create_operator_model",
                "static_operator":{
                    "shape":[3,3],
                    "entries":[
                        {"row":0,"column":1,"value":[1.0,0.0]},
                        {"row":1,"column":2,"value":[1.0,0.0]}
                    ]
                }
            }"#,
        );
        let create: Value = serde_json::from_str(&create).unwrap();
        assert_eq!(create["status"], "ok");
        let model_handle = create["result"]["handle"].as_str().unwrap();

        let prepared = run_command_json(&format!(
            r#"{{
                "operation":"create_expm_action",
                "handle":"{model_handle}",
                "coefficient":[1.0,0.0],
                "max_degree":55,
                "tolerance":1e-15,
                "max_substeps":1000
            }}"#
        ));
        let prepared: Value = serde_json::from_str(&prepared).unwrap();
        assert_eq!(prepared["status"], "ok");
        assert_eq!(prepared["result"]["dimension"], 3);
        let expm_handle = prepared["result"]["handle"].as_str().unwrap();

        let applied = run_command_json(&format!(
            r#"{{
                "operation":"apply_expm_action",
                "expm_handle":"{expm_handle}",
                "vectors":[
                    [[0.0,0.0],[0.0,0.0],[1.0,0.0]],
                    [[1.0,0.0],[0.0,0.0],[0.0,0.0]]
                ],
                "threads":2
            }}"#
        ));
        let applied: Value = serde_json::from_str(&applied).unwrap();
        assert_eq!(applied["status"], "ok");
        let vectors = applied["result"]["vectors"].as_array().unwrap();
        for (actual, expected) in vectors[0].as_array().unwrap().iter().zip([0.5, 1.0, 1.0]) {
            assert!((actual[0].as_f64().unwrap() - expected).abs() < 1.0e-13);
            assert!(actual[1].as_f64().unwrap().abs() < 1.0e-13);
        }
        assert_eq!(
            vectors[1],
            serde_json::json!([[1.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
        );

        let release = run_command_json(&format!(
            r#"{{"operation":"release_expm_action","expm_handle":"{expm_handle}"}}"#
        ));
        let release: Value = serde_json::from_str(&release).unwrap();
        assert_eq!(release["status"], "ok");
        let release = run_command_json(&format!(
            r#"{{"operation":"release_model","handle":"{model_handle}"}}"#
        ));
        let release: Value = serde_json::from_str(&release).unwrap();
        assert_eq!(release["status"], "ok");
    }

    #[test]
    fn floquet_time_grid_preserves_stage_origin_and_final_endpoint() {
        let grid = run_command_json(
            r#"{
                "operation":"floquet_time_grid",
                "period":0.5,
                "constant_cycles":3,
                "points_per_cycle":4,
                "ramp_up_cycles":2,
                "ramp_down_cycles":1
            }"#,
        );
        let grid: Value = serde_json::from_str(&grid).unwrap();
        assert_eq!(grid["status"], "ok");
        assert_eq!(grid["result"]["kind"], "time_grid");
        assert_eq!(grid["result"]["cycles"], 6);
        assert_eq!(grid["result"]["points_per_cycle"], 4);
        let times = grid["result"]["times"].as_array().unwrap();
        assert_eq!(times.len(), 25);
        assert!((times[0].as_f64().unwrap() + 1.0).abs() < 1.0e-15);
        assert!((times[24].as_f64().unwrap() - 2.0).abs() < 1.0e-15);
    }

    #[test]
    fn floquet_analysis_accepts_generic_operator_expressions_and_period_override() {
        let analysis = run_command_json(
            r#"{
                "operation":"analyze_floquet",
                "steps":[{
                    "expression":{
                        "kind":"matrix",
                        "operator":{
                            "shape":[2,2],
                            "entries":[
                                {"row":0,"column":0,"value":[-1.0,0.0]},
                                {"row":1,"column":1,"value":[1.0,0.0]}
                            ]
                        }
                    },
                    "duration":0.25
                }],
                "period":1.0,
                "format":"dense"
            }"#,
        );
        let analysis: Value = serde_json::from_str(&analysis).unwrap();
        assert_eq!(analysis["status"], "ok");
        assert_eq!(analysis["result"]["kind"], "floquet_analysis");
        assert_eq!(
            analysis["result"]["unitary"]["shape"],
            serde_json::json!([2, 2])
        );
        assert_eq!(
            analysis["result"]["effective_hamiltonian"]["shape"],
            serde_json::json!([2, 2])
        );
        let energies = analysis["result"]["quasienergies"].as_array().unwrap();
        assert!((energies[0].as_f64().unwrap() + 0.25).abs() < 1.0e-12);
        assert!((energies[1].as_f64().unwrap() - 0.25).abs() < 1.0e-12);
    }

    #[test]
    fn floquet_analysis_accepts_an_externally_integrated_unitary() {
        let analysis = run_command_json(
            r#"{
                "operation":"analyze_floquet_unitary",
                "unitary":{
                    "shape":[2,2],
                    "entries":[
                        {"row":0,"column":0,"value":[0.955336489125606,0.29552020666133955]},
                        {"row":1,"column":1,"value":[0.955336489125606,-0.29552020666133955]}
                    ]
                },
                "period":0.6,
                "format":"dense"
            }"#,
        );
        let analysis: Value = serde_json::from_str(&analysis).unwrap();
        assert_eq!(analysis["status"], "ok");
        let energies = analysis["result"]["quasienergies"].as_array().unwrap();
        assert!((energies[0].as_f64().unwrap() + 0.5).abs() < 1.0e-12);
        assert!((energies[1].as_f64().unwrap() - 0.5).abs() < 1.0e-12);
    }

    #[test]
    fn measurement_commands_cover_vector_batches_and_density_samples() {
        let create = run_command_json(
            r#"{
                "operation":"create_operator_model",
                "static_operator":{
                    "shape":[2,2],
                    "entries":[
                        {"row":0,"column":0,"value":[1.0,0.0]},
                        {"row":1,"column":1,"value":[-1.0,0.0]}
                    ]
                }
            }"#,
        );
        let create: Value = serde_json::from_str(&create).unwrap();
        assert_eq!(create["status"], "ok");
        let handle = create["result"]["handle"].as_str().unwrap();

        let matrix_elements = run_command_json(&format!(
            r#"{{
                "operation":"matrix_elements_model",
                "handle":"{handle}",
                "left_vectors":[
                    [[1.0,0.0],[0.0,0.0]],
                    [[0.0,0.0],[1.0,0.0]]
                ],
                "right_vectors":[
                    [[1.0,0.0],[0.0,0.0]],
                    [[0.0,0.0],[1.0,0.0]]
                ]
            }}"#
        ));
        let matrix_elements: Value = serde_json::from_str(&matrix_elements).unwrap();
        assert_eq!(matrix_elements["status"], "ok");
        assert_eq!(
            matrix_elements["result"]["shape"],
            serde_json::json!([2, 2])
        );
        assert_eq!(
            matrix_elements["result"]["values"],
            serde_json::json!([[1.0, 0.0], [0.0, 0.0], [0.0, 0.0], [-1.0, 0.0]])
        );

        let amplitude = 1.0 / 2.0_f64.sqrt();
        let samples = format!(
            r#"[
                {{
                    "kind":"pure",
                    "values":[[{amplitude},0.0],[{amplitude},0.0]]
                }},
                {{
                    "kind":"density",
                    "values":[
                        [0.75,0.0],[0.0,0.0],
                        [0.0,0.0],[0.25,0.0]
                    ]
                }}
            ]"#
        );
        let expectation = run_command_json(&format!(
            r#"{{
                "operation":"measure_model",
                "handle":"{handle}",
                "measurement":"expectation",
                "samples":{samples}
            }}"#
        ));
        let expectation: Value = serde_json::from_str(&expectation).unwrap();
        assert_eq!(expectation["status"], "ok");
        assert!(
            expectation["result"]["values"][0][0]
                .as_f64()
                .unwrap()
                .abs()
                < 1.0e-12
        );
        assert_eq!(
            expectation["result"]["values"][1],
            serde_json::json!([0.5, 0.0])
        );

        let fluctuation = run_command_json(&format!(
            r#"{{
                "operation":"measure_model",
                "handle":"{handle}",
                "measurement":"quantum_fluctuation",
                "samples":{samples}
            }}"#
        ));
        let fluctuation: Value = serde_json::from_str(&fluctuation).unwrap();
        assert_eq!(fluctuation["status"], "ok");
        assert!((fluctuation["result"]["values"][0][0].as_f64().unwrap() - 1.0).abs() < 1.0e-12);
        assert_eq!(
            fluctuation["result"]["values"][1],
            serde_json::json!([0.75, 0.0])
        );

        let release = run_command_json(&format!(
            r#"{{"operation":"release_model","handle":"{handle}"}}"#
        ));
        let release: Value = serde_json::from_str(&release).unwrap();
        assert_eq!(release["status"], "ok");
    }

    #[test]
    fn subsystem_analysis_lifts_basis_order_before_tracing() {
        let create = run_command_json(
            r#"{
                "operation":"create_model",
                "basis":{"kind":"spin","sites":2,"reverse":true},
                "terms":[],
                "site_permutation":[1,0],
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
        let amplitude = 1.0 / 2.0_f64.sqrt();
        let analysis = run_command_json(&format!(
            r#"{{
                "operation":"analyze_subsystem_model",
                "handle":"{handle}",
                "parent_handle":"{handle}",
                "local_dimensions":[2,2],
                "retained_sites":[0],
                "samples":[{{
                    "kind":"pure",
                    "values":[
                        [{amplitude},0.0],[0.0,0.0],
                        [0.0,0.0],[{amplitude},0.0]
                    ]
                }}]
            }}"#
        ));
        let analysis: Value = serde_json::from_str(&analysis).unwrap();
        assert_eq!(analysis["status"], "ok");
        assert_eq!(analysis["result"]["subsystem_dimension"], 2);
        let density = analysis["result"]["samples"][0]["density_a"]
            .as_array()
            .unwrap();
        for (entry, expected) in density.iter().zip([0.5, 0.0, 0.0, 0.5]) {
            assert!((entry[0].as_f64().unwrap() - expected).abs() < 1.0e-12);
            assert!(entry[1].as_f64().unwrap().abs() < 1.0e-12);
        }
        let entropy = analysis["result"]["samples"][0]["entropy_a"]
            .as_f64()
            .unwrap();
        assert!((entropy - 2.0_f64.ln()).abs() < 1.0e-12);
    }

    #[test]
    fn recursive_tensor_basis_crosses_the_c_boundary_with_all_factor_splits() {
        let create = run_command_json(
            r#"{
                "operation":"create_model",
                "basis":{
                    "kind":"tensor",
                    "factors":[
                        {"kind":"spin","sites":1,"reverse":true},
                        {"kind":"spin","sites":1,"reverse":true},
                        {"kind":"spin","sites":1,"reverse":true}
                    ]
                },
                "terms":[
                    {
                        "product":{"local":["z"],"splits":[1,1]},
                        "couplings":[{"coefficient":[1.0,0.0],"sites":[0]}]
                    },
                    {
                        "product":{"local":["z"],"splits":[0,1]},
                        "couplings":[{"coefficient":[2.0,0.0],"sites":[0]}]
                    },
                    {
                        "product":{"local":["z"],"splits":[0,0]},
                        "couplings":[{"coefficient":[4.0,0.0],"sites":[0]}]
                    }
                ],
                "checks":{
                    "hermiticity":true,
                    "particle_conservation":false,
                    "symmetry_compatibility":false
                }
            }"#,
        );
        let create: Value = serde_json::from_str(&create).unwrap();
        assert_eq!(create["status"], "ok");
        assert_eq!(create["result"]["dimension"], 8);
        let handle = create["result"]["handle"].as_str().unwrap();

        let spectrum = run_command_json(&format!(
            r#"{{"operation":"eigh_model","handle":"{handle}"}}"#
        ));
        let spectrum: Value = serde_json::from_str(&spectrum).unwrap();
        assert_eq!(spectrum["status"], "ok");
        let eigenvalues = spectrum["result"]["eigenvalues"].as_array().unwrap();
        for (actual, expected) in eigenvalues
            .iter()
            .zip([-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5])
        {
            assert!((actual.as_f64().unwrap() - expected).abs() < 1.0e-12);
        }

        let release = run_command_json(&format!(
            r#"{{"operation":"release_model","handle":"{handle}"}}"#
        ));
        let release: Value = serde_json::from_str(&release).unwrap();
        assert_eq!(release["status"], "ok");
    }

    #[test]
    fn photon_basis_and_explicit_embedding_cross_the_c_boundary() {
        let fixed = run_command_json(
            r#"{
                "operation":"create_model",
                "basis":{
                    "kind":"photon",
                    "matter":{"kind":"spin","sites":2,"reverse":true},
                    "photon_cutoff":2,
                    "total_excitations":2
                },
                "terms":[
                    {
                        "product":{"local":["raising","lowering"],"split":1},
                        "couplings":[{"coefficient":[1.0,0.0],"sites":[0,0]}]
                    },
                    {
                        "product":{"local":["lowering","raising"],"split":1},
                        "couplings":[{"coefficient":[1.0,0.0],"sites":[0,0]}]
                    }
                ],
                "checks":{
                    "hermiticity":true,
                    "particle_conservation":true,
                    "symmetry_compatibility":false
                }
            }"#,
        );
        let fixed: Value = serde_json::from_str(&fixed).unwrap();
        assert_eq!(fixed["status"], "ok");
        assert_eq!(fixed["result"]["dimension"], 4);
        let fixed_handle = fixed["result"]["handle"].as_str().unwrap();

        let parent = run_command_json(
            r#"{
                "operation":"create_model",
                "basis":{
                    "kind":"photon",
                    "matter":{"kind":"spin","sites":2,"reverse":true},
                    "photon_cutoff":2
                },
                "terms":[]
            }"#,
        );
        let parent: Value = serde_json::from_str(&parent).unwrap();
        assert_eq!(parent["status"], "ok");
        assert_eq!(parent["result"]["dimension"], 12);
        let parent_handle = parent["result"]["handle"].as_str().unwrap();

        let embedding = run_command_json(&format!(
            r#"{{
                "operation":"projector_model",
                "handle":"{fixed_handle}",
                "parent_handle":"{parent_handle}",
                "embedding":true
            }}"#
        ));
        let embedding: Value = serde_json::from_str(&embedding).unwrap();
        assert_eq!(embedding["status"], "ok");
        assert_eq!(embedding["result"]["shape"], serde_json::json!([12, 4]));
        assert_eq!(embedding["result"]["entries"].as_array().unwrap().len(), 4);

        for handle in [fixed_handle, parent_handle] {
            let release = run_command_json(&format!(
                r#"{{"operation":"release_model","handle":"{handle}"}}"#
            ));
            let release: Value = serde_json::from_str(&release).unwrap();
            assert_eq!(release["status"], "ok");
        }
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
    fn basis_requests_materialize_single_and_multicomponent_sector_unions() {
        let cases = [
            (
                r#"{
                    "operation":"describe_basis",
                    "basis":{
                        "kind":"spin",
                        "sites":4,
                        "up_sectors":[0,2],
                        "symmetries":[{"destinations":[1,2,3,0],"sector":0}]
                    }
                }"#,
                None,
            ),
            (
                r#"{
                    "operation":"describe_basis",
                    "basis":{
                        "kind":"boson",
                        "sites":4,
                        "states_per_site":5,
                        "particle_sectors":[0,2]
                    }
                }"#,
                Some(11),
            ),
            (
                r#"{
                    "operation":"describe_basis",
                    "basis":{
                        "kind":"spinless_fermion",
                        "sites":4,
                        "particle_sectors":[0,2]
                    }
                }"#,
                Some(7),
            ),
            (
                r#"{
                    "operation":"describe_basis",
                    "basis":{
                        "kind":"spinful_fermion",
                        "sites":3,
                        "particle_sectors":[[0,0],[0,2],[2,0],[2,2]]
                    }
                }"#,
                Some(16),
            ),
            (
                r#"{
                    "operation":"describe_basis",
                    "basis":{
                        "kind":"spinful_fermion",
                        "sites":3,
                        "particles_up":2,
                        "particles_down":1,
                        "allowed_local_occupancies":[0,1,2]
                    }
                }"#,
                Some(3),
            ),
        ];
        for (request, expected_dimension) in cases {
            let response: Value = serde_json::from_str(&run_command_json(request)).unwrap();
            assert_eq!(response["status"], "ok");
            if let Some(expected_dimension) = expected_dimension {
                assert_eq!(response["result"]["dimension"], expected_dimension);
            }
        }
    }

    #[test]
    fn basis_plan_reduces_large_states_without_enumerating_the_parent_basis() {
        let destinations = (1..64).chain(std::iter::once(0)).collect::<Vec<_>>();
        let create = run_command_json(&format!(
            r#"{{
                "operation":"create_basis_plan",
                "basis":{{
                    "kind":"spin",
                    "sites":64,
                    "up":32,
                    "symmetries":[{{
                        "destinations":{destinations:?},
                        "sector":0
                    }}]
                }}
            }}"#
        ));
        let create: Value = serde_json::from_str(&create).unwrap();
        assert_eq!(create["status"], "ok");
        let handle = create["result"]["handle"].as_str().unwrap();

        let reduce = run_command_json(&format!(
            r#"{{
                "operation":"reduce_states_plan",
                "plan_handle":"{handle}",
                "states":["1"]
            }}"#
        ));
        let reduce: Value = serde_json::from_str(&reduce).unwrap();
        assert_eq!(reduce["status"], "ok");
        assert_eq!(reduce["result"]["period_product"], 64);
        assert_eq!(reduce["result"]["entries"][0]["representative"], "1");
        assert_eq!(reduce["result"]["entries"][0]["orbit_size"], 64);
        assert_eq!(reduce["result"]["entries"][0]["compatible"], true);

        let release = run_command_json(&format!(
            r#"{{"operation":"release_basis_plan","plan_handle":"{handle}"}}"#
        ));
        let release: Value = serde_json::from_str(&release).unwrap();
        assert_eq!(release["status"], "ok");

        let stale = run_command_json(&format!(
            r#"{{
                "operation":"reduce_states_plan",
                "plan_handle":"{handle}",
                "states":["1"]
            }}"#
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
    fn basis_plan_materializes_the_same_reducer_it_queries() {
        let create = run_command_json(
            r#"{
                "operation":"create_basis_plan",
                "basis":{
                    "kind":"spin",
                    "sites":4,
                    "symmetries":[
                        {"destinations":[1,2,3,0],"sector":0},
                        {"destinations":[3,2,1,0],"sector":0}
                    ]
                }
            }"#,
        );
        let create: Value = serde_json::from_str(&create).unwrap();
        assert_eq!(create["status"], "ok");
        let plan_handle = create["result"]["handle"].as_str().unwrap();

        let reduce = run_command_json(&format!(
            r#"{{
                "operation":"reduce_states_plan",
                "plan_handle":"{plan_handle}",
                "states":["8"]
            }}"#
        ));
        let reduce: Value = serde_json::from_str(&reduce).unwrap();
        assert_eq!(reduce["status"], "ok");
        assert_eq!(reduce["result"]["entries"][0]["representative"], "1");
        assert_eq!(
            reduce["result"]["entries"][0]["generator_word"],
            serde_json::json!([0])
        );

        let materialize = run_command_json(&format!(
            r#"{{
                "operation":"materialize_basis_plan",
                "plan_handle":"{plan_handle}"
            }}"#
        ));
        let materialize: Value = serde_json::from_str(&materialize).unwrap();
        assert_eq!(materialize["status"], "ok");
        assert_eq!(materialize["result"]["dimension"], 6);
        let model_handle = materialize["result"]["handle"].as_str().unwrap();

        let describe = run_command_json(&format!(
            r#"{{"operation":"describe_model","handle":"{model_handle}"}}"#
        ));
        let describe: Value = serde_json::from_str(&describe).unwrap();
        assert_eq!(
            describe["result"]["states"],
            serde_json::json!(["0", "1", "3", "5", "7", "15"])
        );

        let release_model = run_command_json(&format!(
            r#"{{"operation":"release_model","handle":"{model_handle}"}}"#
        ));
        let release_model: Value = serde_json::from_str(&release_model).unwrap();
        assert_eq!(release_model["status"], "ok");
        let release_plan = run_command_json(&format!(
            r#"{{"operation":"release_basis_plan","plan_handle":"{plan_handle}"}}"#
        ));
        let release_plan: Value = serde_json::from_str(&release_plan).unwrap();
        assert_eq!(release_plan["status"], "ok");
    }

    #[test]
    fn basis_plan_completes_seed_orbits_for_deferred_bra_ket_actions() {
        let create = run_command_json(
            r#"{
                "operation":"create_basis_plan",
                "basis":{
                    "kind":"spin",
                    "sites":4,
                    "up":1,
                    "normalization":"pauli",
                    "symmetries":[{
                        "destinations":[0,1,2,3],
                        "local_permutations":[[1,0],[1,0],[1,0],[1,0]],
                        "sector":0
                    }]
                }
            }"#,
        );
        let create: Value = serde_json::from_str(&create).unwrap();
        assert_eq!(create["status"], "ok");
        let plan_handle = create["result"]["handle"].as_str().unwrap();

        let materialize = run_command_json(&format!(
            r#"{{
                "operation":"materialize_basis_plan",
                "plan_handle":"{plan_handle}"
            }}"#
        ));
        let materialize: Value = serde_json::from_str(&materialize).unwrap();
        assert_eq!(materialize["status"], "ok");
        assert_eq!(materialize["result"]["dimension"], 4);
        let model_handle = materialize["result"]["handle"].as_str().unwrap();

        let describe: Value = serde_json::from_str(&run_command_json(&format!(
            r#"{{"operation":"describe_model","handle":"{model_handle}"}}"#
        )))
        .unwrap();
        assert_eq!(
            describe["result"]["states"],
            serde_json::json!(["1", "2", "4", "7"])
        );

        // State 7 is outside the one-particle seed sector, but it is the
        // canonical representative of the completed {7, 8} inversion orbit.
        let transitions: Value = serde_json::from_str(&run_command_json(&format!(
            r#"{{
                "operation":"bra_ket_terms_plan",
                "plan_handle":"{plan_handle}",
                "terms":[{{
                    "product":{{"local":["z","z"]}},
                    "couplings":[{{"coefficient":[1.0,0.0],"sites":[0,1]}}]
                }}],
                "kets":["7"]
            }}"#
        )))
        .unwrap();
        assert_eq!(transitions["status"], "ok");
        assert_eq!(transitions["result"]["entries"][0]["bra"], "7");
        assert_eq!(transitions["result"]["entries"][0]["ket"], "7");
        assert_eq!(
            transitions["result"]["entries"][0]["value"],
            serde_json::json!([1.0, 0.0])
        );

        let release_model: Value = serde_json::from_str(&run_command_json(&format!(
            r#"{{"operation":"release_model","handle":"{model_handle}"}}"#
        )))
        .unwrap();
        assert_eq!(release_model["status"], "ok");
        let release_plan: Value = serde_json::from_str(&run_command_json(&format!(
            r#"{{"operation":"release_basis_plan","plan_handle":"{plan_handle}"}}"#
        )))
        .unwrap();
        assert_eq!(release_plan["status"], "ok");
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
        let first_projector = cached_projector(target_handle, parent_handle, false).unwrap();
        let second_projector = cached_projector(target_handle, parent_handle, false).unwrap();
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
    fn registered_models_stream_terms_across_matrix_representation_subspaces() {
        let matrix_basis = |selected_row: usize| {
            serde_json::json!({
                "kind": "spin",
                "sites": 3,
                "up": 1,
                "pauli": true,
                "symmetries": [],
                "matrix_symmetry": {
                    "dimension": 2,
                    "selected_row": selected_row,
                    "generators": [
                        {
                            "destinations": [1, 2, 0],
                            "local_permutations": null,
                            "matrix": [
                                [[-0.5, 0.0], [-0.866_025_403_784_438_6, 0.0]],
                                [[0.866_025_403_784_438_6, 0.0], [-0.5, 0.0]]
                            ]
                        },
                        {
                            "destinations": [2, 1, 0],
                            "local_permutations": null,
                            "matrix": [
                                [[1.0, 0.0], [0.0, 0.0]],
                                [[0.0, 0.0], [-1.0, 0.0]]
                            ]
                        }
                    ]
                },
                "reverse": true
            })
        };
        let create_model = |basis: Value| {
            let response: Value = serde_json::from_str(&run_command_json(
                &serde_json::json!({
                    "operation": "create_model",
                    "basis": basis,
                    "terms": [],
                    "site_permutation": null,
                    "checks": {
                        "hermiticity": false,
                        "particle_conservation": false,
                        "symmetry_compatibility": false
                    }
                })
                .to_string(),
            ))
            .unwrap();
            assert_eq!(response["status"], "ok", "{response}");
            (
                response["result"]["handle"].as_str().unwrap().to_owned(),
                response["result"]["dimension"].as_u64().unwrap() as usize,
            )
        };
        let (source_handle, source_dimension) = create_model(matrix_basis(0));
        let (target_handle, target_dimension) = create_model(matrix_basis(1));
        let (parent_handle, parent_dimension) = create_model(serde_json::json!({
            "kind": "spin",
            "sites": 3,
            "pauli": true,
            "reverse": true
        }));
        assert_eq!(source_dimension, 1);
        assert_eq!(target_dimension, 1);
        assert_eq!(parent_dimension, 8);

        let terms = serde_json::json!([{
            "product": {"local": ["z"]},
            "couplings": [{"coefficient": [1.3, -0.2], "sites": [0]}]
        }]);
        let source_vectors = serde_json::json!([[[0.7, -0.4]]]);
        let apply_between = |source: &str, target: &str, vectors: Value| {
            let response: Value = serde_json::from_str(&run_command_json(
                &serde_json::json!({
                    "operation": "apply_terms_between_models",
                    "source_handle": source,
                    "target_handle": target,
                    "terms": terms.clone(),
                    "vectors": vectors
                })
                .to_string(),
            ))
            .unwrap();
            assert_eq!(response["status"], "ok", "{response}");
            response
        };
        let project_to_target = |vectors: Value| {
            let response: Value = serde_json::from_str(&run_command_json(
                &serde_json::json!({
                    "operation": "apply_projector_model",
                    "handle": target_handle,
                    "parent_handle": parent_handle,
                    "vectors": vectors,
                    "action": "project"
                })
                .to_string(),
            ))
            .unwrap();
            assert_eq!(response["status"], "ok", "{response}");
            response
        };

        let projected_to_projected =
            apply_between(&source_handle, &target_handle, source_vectors.clone());
        let projected_to_parent = apply_between(&source_handle, &parent_handle, source_vectors);
        let staged = project_to_target(projected_to_parent["result"]["vectors"].clone());
        let direct_values = projected_to_projected["result"]["vectors"][0]
            .as_array()
            .unwrap();
        let staged_values = staged["result"]["vectors"][0].as_array().unwrap();
        assert_eq!(direct_values.len(), staged_values.len());
        for (direct, staged) in direct_values.iter().zip(staged_values) {
            assert!((direct[0].as_f64().unwrap() - staged[0].as_f64().unwrap()).abs() < 1.0e-12);
            assert!((direct[1].as_f64().unwrap() - staged[1].as_f64().unwrap()).abs() < 1.0e-12);
        }

        let parent_vectors = serde_json::json!([[
            [0.1, 0.0],
            [0.2, -0.1],
            [0.3, 0.2],
            [0.4, -0.2],
            [0.5, 0.1],
            [0.6, -0.3],
            [0.7, 0.4],
            [0.8, -0.5]
        ]]);
        let parent_to_projected =
            apply_between(&parent_handle, &target_handle, parent_vectors.clone());
        let parent_to_parent = apply_between(&parent_handle, &parent_handle, parent_vectors);
        let staged = project_to_target(parent_to_parent["result"]["vectors"].clone());
        let direct_values = parent_to_projected["result"]["vectors"][0]
            .as_array()
            .unwrap();
        let staged_values = staged["result"]["vectors"][0].as_array().unwrap();
        for (direct, staged) in direct_values.iter().zip(staged_values) {
            assert!((direct[0].as_f64().unwrap() - staged[0].as_f64().unwrap()).abs() < 1.0e-12);
            assert!((direct[1].as_f64().unwrap() - staged[1].as_f64().unwrap()).abs() < 1.0e-12);
        }
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

    #[test]
    fn command_protocol_builds_a_projected_parameterized_block_family() {
        let first: Value = serde_json::from_str(&run_command_json(
            r#"{
                "operation":"create_operator_model",
                "static_operator":{
                    "shape":[2,2],
                    "entries":[
                        {"row":0,"column":0,"value":[1.0,0.0]},
                        {"row":1,"column":1,"value":[2.0,0.0]}
                    ]
                },
                "components":[{
                    "name":"drive",
                    "operator":{
                        "shape":[2,2],
                        "entries":[
                            {"row":0,"column":0,"value":[10.0,0.0]},
                            {"row":1,"column":1,"value":[20.0,0.0]}
                        ]
                    }
                }],
                "basis":null,
                "site_permutation":null
            }"#,
        ))
        .unwrap();
        assert_eq!(first["status"], "ok");
        let first_handle = first["result"]["handle"].as_str().unwrap();

        let second: Value = serde_json::from_str(&run_command_json(
            r#"{
                "operation":"create_operator_model",
                "static_operator":{
                    "shape":[1,1],
                    "entries":[{"row":0,"column":0,"value":[3.0,0.0]}]
                },
                "components":[{
                    "name":"drive",
                    "operator":{
                        "shape":[1,1],
                        "entries":[{"row":0,"column":0,"value":[30.0,0.0]}]
                    }
                }],
                "basis":null,
                "site_permutation":null
            }"#,
        ))
        .unwrap();
        assert_eq!(second["status"], "ok");
        let second_handle = second["result"]["handle"].as_str().unwrap();

        let projected: Value = serde_json::from_str(&run_command_json(&format!(
            r#"{{
                "operation":"create_projected_block_model",
                "blocks":[
                    {{
                        "handle":"{first_handle}",
                        "projector":{{
                            "shape":[3,2],
                            "entries":[
                                {{"row":0,"column":0,"value":[1.0,0.0]}},
                                {{"row":2,"column":1,"value":[1.0,0.0]}}
                            ]
                        }}
                    }},
                    {{
                        "handle":"{second_handle}",
                        "projector":{{
                            "shape":[3,1],
                            "entries":[
                                {{"row":1,"column":0,"value":[1.0,0.0]}}
                            ]
                        }}
                    }}
                ],
                "tolerance":1e-12,
                "format":"csc"
            }}"#
        )))
        .unwrap();
        assert_eq!(projected["status"], "ok");
        assert_eq!(projected["result"]["dimension"], 3);
        let projected_handle = projected["result"]["handle"].as_str().unwrap();

        let applied: Value = serde_json::from_str(&run_command_json(&format!(
            r#"{{
                "operation":"apply_model",
                "handle":"{projected_handle}",
                "vectors":[[[1.0,0.0],[1.0,0.0],[1.0,0.0]]],
                "parameters":{{"drive":[0.5,0.0]}}
            }}"#
        )))
        .unwrap();
        assert_eq!(applied["status"], "ok");
        assert_eq!(
            applied["result"]["vectors"][0],
            serde_json::json!([[6.0, 0.0], [18.0, 0.0], [12.0, 0.0]])
        );

        for handle in [projected_handle, first_handle, second_handle] {
            let released: Value = serde_json::from_str(&run_command_json(&format!(
                r#"{{"operation":"release_model","handle":"{handle}"}}"#
            )))
            .unwrap();
            assert_eq!(released["status"], "ok");
        }
    }
}
