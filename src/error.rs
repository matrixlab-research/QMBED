use thiserror::Error;

/// Stable failure categories for all recoverable public operations.
#[derive(Debug, Error)]
pub enum QmbedError {
    #[error("invalid or unregistered operator: {0}")]
    InvalidOperator(String),
    #[error("invalid coupling: {0}")]
    InvalidCoupling(String),
    #[error("site {site} is outside a lattice with {sites} sites")]
    InvalidSite { site: usize, sites: usize },
    #[error("state is not represented by this basis")]
    StateNotInBasis,
    #[error("invalid Hilbert-space sector: {0}")]
    InvalidSector(String),
    #[error("incompatible symmetry request: {0}")]
    IncompatibleSymmetry(String),
    #[error("a Hermitian operator is required")]
    NonHermitian,
    #[error("invalid options: {0}")]
    InvalidOptions(String),
    #[error("dimension mismatch: {0}")]
    DimensionMismatch(String),
    #[error("the supplied vectors are numerically rank deficient")]
    RankDeficient,
    #[error("unsupported backend: {0}")]
    UnsupportedBackend(String),
    #[error("archive error: {0}")]
    Archive(String),
    #[error("internal state error: {0}")]
    InternalState(String),
    #[error("solver did not converge after {iterations} iterations; residual={residual:e}")]
    NonConvergence { iterations: usize, residual: f64 },
}

/// Crate-wide result type.
pub type Result<T> = std::result::Result<T, QmbedError>;

/// Compatibility spelling retained for source migrations from the earlier
/// QuSpin-derived Rust API.
pub type QuSpinError = QmbedError;
