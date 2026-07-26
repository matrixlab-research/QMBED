//! Execution and vector-storage boundary.
//!
//! Physics-facing code depends on [`crate::operator::LinearOperator`]. A
//! runtime owns vectors and coarse-grained vector primitives. The built-in CPU
//! implementation is complete; accelerator and multi-rank profiles are
//! explicit extension points rather than silently falling back to the host.

use std::sync::mpsc;
use std::thread;

use num_complex::Complex64;

use crate::operator::LinearOperator;
use crate::{QmbedError, Result};

/// Accelerator requested by an execution profile.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Accelerator {
    /// Host CPU execution.
    Cpu,
    /// GPU execution on a zero-based device index.
    Gpu { device: usize },
}

/// Language-neutral execution request.
///
/// `ranks > 1` denotes a distributed run. Each rank may in turn use the
/// requested accelerator and number of host threads.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExecutionProfile {
    pub accelerator: Accelerator,
    pub ranks: usize,
    pub threads_per_rank: usize,
}

impl ExecutionProfile {
    /// Request one CPU rank and one host thread.
    pub const fn serial() -> Self {
        Self::local_cpu(1)
    }

    /// Request bounded host parallelism for independent work items.
    pub const fn throughput(threads: usize) -> Self {
        Self::local_cpu(threads)
    }

    /// Request one CPU rank with `threads` host workers.
    pub const fn local_cpu(threads: usize) -> Self {
        Self {
            accelerator: Accelerator::Cpu,
            ranks: 1,
            threads_per_rank: threads,
        }
    }

    /// Request a distributed CPU run.
    ///
    /// The built-in runtime rejects this profile until an MPI implementation
    /// of [`Runtime`] is installed.
    pub const fn distributed_cpu(ranks: usize, threads_per_rank: usize) -> Self {
        Self {
            accelerator: Accelerator::Cpu,
            ranks,
            threads_per_rank,
        }
    }

    /// Request one GPU device with the supplied host-thread allowance.
    pub const fn local_gpu(device: usize, threads: usize) -> Self {
        Self {
            accelerator: Accelerator::Gpu { device },
            ranks: 1,
            threads_per_rank: threads,
        }
    }

    /// Validate that rank and thread counts are positive.
    pub fn validate(self) -> Result<Self> {
        if self.ranks == 0 || self.threads_per_rank == 0 {
            return Err(QmbedError::InvalidOptions(
                "execution profiles require at least one rank and one thread per rank".into(),
            ));
        }
        Ok(self)
    }
}

/// Concrete resources supplied by one runtime implementation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RuntimeCapabilities {
    /// Active accelerator.
    pub accelerator: Accelerator,
    /// Number of cooperating process ranks.
    pub ranks: usize,
    /// Host threads available to each rank.
    pub threads_per_rank: usize,
}

/// Vector storage owned by a [`Runtime`].
pub trait RuntimeBuffer: Send + Sync {
    /// Return the number of complex elements.
    fn len(&self) -> usize;

    /// Return whether the buffer has no elements.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Coarse-grained storage and vector primitives required by iterative methods.
///
/// A future GPU or MPI backend implements this trait with its native buffer.
/// No basis, model, or operator-string type crosses this boundary.
pub trait Runtime: Send + Sync {
    /// Backend-native vector type.
    type Buffer: RuntimeBuffer;

    /// Report the execution resources supplied by this runtime.
    fn capabilities(&self) -> RuntimeCapabilities;
    /// Allocate a zero-filled complex vector.
    fn zeros(&self, length: usize) -> Result<Self::Buffer>;
    /// Copy a host slice into backend-owned storage.
    fn upload(&self, values: &[Complex64]) -> Result<Self::Buffer>;
    /// Copy backend-owned storage to the host.
    fn to_host(&self, buffer: &Self::Buffer) -> Result<Vec<Complex64>>;
    /// Fill a backend buffer with one complex value.
    fn fill(&self, buffer: &mut Self::Buffer, value: Complex64) -> Result<()>;
    /// Compute `output += alpha * input`.
    fn axpy(&self, alpha: Complex64, input: &Self::Buffer, output: &mut Self::Buffer)
    -> Result<()>;
    /// Scale a vector in place.
    fn scale(&self, alpha: Complex64, buffer: &mut Self::Buffer) -> Result<()>;
    /// Return the conjugating inner product of two buffers.
    fn dotc(&self, left: &Self::Buffer, right: &Self::Buffer) -> Result<Complex64>;

    /// Return the Euclidean norm derived from [`Runtime::dotc`].
    fn norm(&self, buffer: &Self::Buffer) -> Result<f64> {
        Ok(self.dotc(buffer, buffer)?.re.max(0.0).sqrt())
    }
}

/// Runtime-aware action without changing the scientific operator contract.
pub trait RuntimeLinearOperator<R>
where
    R: Runtime,
{
    /// Return `(rows, columns)` in runtime coordinates.
    fn runtime_shape(&self) -> (usize, usize);
    /// Apply this map to backend-owned buffers.
    fn apply_on(&self, runtime: &R, input: &R::Buffer, output: &mut R::Buffer) -> Result<()>;
}

/// Host-resident complex vector used by [`CpuRuntime`].
#[derive(Clone, Debug, PartialEq)]
pub struct CpuBuffer {
    values: Vec<Complex64>,
}

impl CpuBuffer {
    /// Borrow the host values.
    pub fn as_slice(&self) -> &[Complex64] {
        &self.values
    }

    /// Mutably borrow the host values.
    pub fn as_mut_slice(&mut self) -> &mut [Complex64] {
        &mut self.values
    }
}

impl RuntimeBuffer for CpuBuffer {
    fn len(&self) -> usize {
        self.values.len()
    }
}

/// Built-in single-rank CPU runtime.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CpuRuntime {
    profile: ExecutionProfile,
}

impl CpuRuntime {
    /// Construct a local CPU runtime with a positive worker count.
    pub fn new(threads: usize) -> Result<Self> {
        Self::from_profile(ExecutionProfile::local_cpu(threads))
    }

    /// Resolve a profile against the built-in runtime.
    ///
    /// Unsupported profiles fail explicitly so a requested GPU or MPI run
    /// cannot be mistaken for a successful serial CPU calculation.
    pub fn from_profile(profile: ExecutionProfile) -> Result<Self> {
        let profile = profile.validate()?;
        if profile.accelerator != Accelerator::Cpu {
            return Err(QmbedError::UnsupportedBackend(
                "the built-in runtime is CPU-only; install a GPU runtime implementation".into(),
            ));
        }
        if profile.ranks != 1 {
            return Err(QmbedError::UnsupportedBackend(
                "the built-in runtime is single-rank; install an MPI runtime implementation".into(),
            ));
        }
        Ok(Self { profile })
    }

    /// Return the validated profile used to construct this runtime.
    pub const fn profile(self) -> ExecutionProfile {
        self.profile
    }

    /// Apply an independent operation with bounded shared-memory parallelism.
    ///
    /// Results and errors are returned in input order regardless of worker
    /// scheduling. A one-thread profile executes through the same contract.
    pub fn map_ordered<T, U, F>(&self, items: &[T], operation: F) -> Result<Vec<U>>
    where
        T: Sync,
        U: Send,
        F: Fn(&T) -> Result<U> + Sync,
    {
        if items.is_empty() {
            return Ok(Vec::new());
        }
        let workers = self.profile.threads_per_rank.min(items.len());
        let mut ordered = (0..items.len()).map(|_| None).collect::<Vec<_>>();
        thread::scope(|scope| {
            let (sender, receiver) = mpsc::channel();
            let mut handles = Vec::with_capacity(workers);
            for worker in 0..workers {
                let sender = sender.clone();
                let operation = &operation;
                handles.push(scope.spawn(move || {
                    for index in (worker..items.len()).step_by(workers) {
                        if sender.send((index, operation(&items[index]))).is_err() {
                            break;
                        }
                    }
                }));
            }
            drop(sender);
            for (index, result) in receiver {
                ordered[index] = Some(result);
            }
            if handles.into_iter().any(|handle| handle.join().is_err()) {
                return Err(QmbedError::UnsupportedBackend(
                    "a CPU runtime worker panicked".into(),
                ));
            }
            let mut output = Vec::with_capacity(items.len());
            for result in ordered {
                let result = result.ok_or_else(|| {
                    QmbedError::UnsupportedBackend(
                        "a CPU runtime worker did not return a result".into(),
                    )
                })?;
                output.push(result?);
            }
            Ok(output)
        })
    }
}

impl Runtime for CpuRuntime {
    type Buffer = CpuBuffer;

    fn capabilities(&self) -> RuntimeCapabilities {
        RuntimeCapabilities {
            accelerator: Accelerator::Cpu,
            ranks: 1,
            threads_per_rank: self.profile.threads_per_rank,
        }
    }

    fn zeros(&self, length: usize) -> Result<Self::Buffer> {
        Ok(CpuBuffer {
            values: vec![Complex64::new(0.0, 0.0); length],
        })
    }

    fn upload(&self, values: &[Complex64]) -> Result<Self::Buffer> {
        Ok(CpuBuffer {
            values: values.to_vec(),
        })
    }

    fn to_host(&self, buffer: &Self::Buffer) -> Result<Vec<Complex64>> {
        Ok(buffer.values.clone())
    }

    fn fill(&self, buffer: &mut Self::Buffer, value: Complex64) -> Result<()> {
        buffer.values.fill(value);
        Ok(())
    }

    fn axpy(
        &self,
        alpha: Complex64,
        input: &Self::Buffer,
        output: &mut Self::Buffer,
    ) -> Result<()> {
        equal_lengths(input, output)?;
        for (target, source) in output.values.iter_mut().zip(&input.values) {
            *target += alpha * *source;
        }
        Ok(())
    }

    fn scale(&self, alpha: Complex64, buffer: &mut Self::Buffer) -> Result<()> {
        for value in &mut buffer.values {
            *value *= alpha;
        }
        Ok(())
    }

    fn dotc(&self, left: &Self::Buffer, right: &Self::Buffer) -> Result<Complex64> {
        equal_lengths(left, right)?;
        Ok(left
            .values
            .iter()
            .zip(&right.values)
            .map(|(left, right)| left.conj() * *right)
            .sum())
    }
}

impl<O> RuntimeLinearOperator<CpuRuntime> for O
where
    O: LinearOperator + ?Sized,
{
    fn runtime_shape(&self) -> (usize, usize) {
        self.shape()
    }

    fn apply_on(
        &self,
        _runtime: &CpuRuntime,
        input: &CpuBuffer,
        output: &mut CpuBuffer,
    ) -> Result<()> {
        self.apply(input.as_slice(), output.as_mut_slice())
    }
}

fn equal_lengths(left: &impl RuntimeBuffer, right: &impl RuntimeBuffer) -> Result<()> {
    if left.len() != right.len() {
        return Err(QmbedError::DimensionMismatch(format!(
            "runtime vector lengths differ: {} and {}",
            left.len(),
            right.len()
        )));
    }
    Ok(())
}
