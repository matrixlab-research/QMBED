#![doc = include_str!("../README.md")]
#![forbid(unsafe_code)]

mod backend;

pub mod archive;
pub mod basis;
pub mod block;
pub mod compat;
pub mod dynamics;
pub mod error;
pub mod interop;
pub mod measure;
pub mod operator;
pub mod runtime;
pub mod solve;
pub mod workflow;

pub use error::{QmbedError, QuSpinError, Result};
pub use num_complex::Complex64;

/// Crate version used by verification adapters.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
