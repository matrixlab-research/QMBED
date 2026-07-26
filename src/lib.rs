#![doc = include_str!("../README.md")]
#![forbid(unsafe_code)]

mod backend;

/// Versioned dense and sparse operator serialization.
pub mod archive;
/// Hilbert-space bases, sectors, symmetries, and projectors.
pub mod basis;
/// Static and time-dependent block operators.
pub mod block;
/// Time-dependent, Floquet, spectral, and correlation workflows.
pub mod dynamics;
/// Structured public error and result types.
pub mod error;
/// Runtime-owned models used by language frontends.
pub mod interop;
/// Observables, reduced states, entanglement, and ensemble analysis.
pub mod measure;
/// Typed local terms, universal assembly, and linear-operator storage.
pub mod operator;
/// Execution profiles, vector buffers, and backend extension points.
pub mod runtime;
/// Hermitian eigensolvers, Krylov evolution, and thermal Lanczos methods.
pub mod solve;
/// Composite workflows such as state tracking and Lindblad generators.
pub mod workflow;

pub use error::{QmbedError, Result};
pub use num_complex::Complex64;

/// Crate version used by verification adapters.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
