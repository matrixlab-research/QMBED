use std::collections::HashMap;
use std::marker::PhantomData;
use std::sync::Arc;

use num_complex::Complex64;
use smallvec::SmallVec;

pub use crate::backend::ShiftedLinearSolver;
use crate::backend::factor_shifted_csc;
use crate::basis::Basis;
use crate::{QmbedError, Result};

/// One typed local action in an operator product.
///
/// This is the native QMBED spelling. Character strings belong to compatibility
/// adapters and are parsed once into this representation before assembly.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum LocalOperator {
    Identity,
    Number,
    Z,
    Raising,
    Lowering,
    X,
    Y,
    Custom(char),
}

impl LocalOperator {
    pub const fn symbol(self) -> char {
        match self {
            Self::Identity => 'I',
            Self::Number => 'n',
            Self::Z => 'z',
            Self::Raising => '+',
            Self::Lowering => '-',
            Self::X => 'x',
            Self::Y => 'y',
            Self::Custom(symbol) => symbol,
        }
    }
}

/// Typed ordered product of local actions.
///
/// `splits` identify factor boundaries. A single split retains the traditional
/// spinful-fermion meaning; multiple (including repeated) boundaries describe
/// arbitrary tensor products with empty factor actions.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OpProduct {
    local: SmallVec<[LocalOperator; 8]>,
    symbols: SmallVec<[char; 8]>,
    split: Option<usize>,
    splits: SmallVec<[usize; 4]>,
    label: String,
}

impl OpProduct {
    pub fn new(local: impl IntoIterator<Item = LocalOperator>) -> Result<Self> {
        Self::with_split(local, None)
    }

    pub fn spinful(
        up: impl IntoIterator<Item = LocalOperator>,
        down: impl IntoIterator<Item = LocalOperator>,
    ) -> Result<Self> {
        let mut local: SmallVec<[LocalOperator; 8]> = up.into_iter().collect();
        let split = local.len();
        local.extend(down);
        Self::with_split(local, Some(split))
    }

    pub fn with_split(
        local: impl IntoIterator<Item = LocalOperator>,
        split: Option<usize>,
    ) -> Result<Self> {
        Self::with_splits(local, split)
    }

    /// Construct an operator product with any number of ordered factor
    /// boundaries. Duplicate boundaries preserve empty tensor factors such as
    /// the middle factor in `x||z`.
    pub fn with_splits(
        local: impl IntoIterator<Item = LocalOperator>,
        splits: impl IntoIterator<Item = usize>,
    ) -> Result<Self> {
        let local: SmallVec<[LocalOperator; 8]> = local.into_iter().collect();
        if local.is_empty() {
            return Err(QmbedError::InvalidOperator(
                "operator products cannot be empty".into(),
            ));
        }
        if local.contains(&LocalOperator::Custom('|')) {
            return Err(QmbedError::InvalidOperator(
                "the species separator is not a local operator".into(),
            ));
        }
        let splits: SmallVec<[usize; 4]> = splits.into_iter().collect();
        if splits.iter().any(|&boundary| boundary > local.len())
            || splits.windows(2).any(|pair| pair[0] > pair[1])
        {
            return Err(QmbedError::InvalidOperator(
                "operator factor boundaries must be ordered within the product arity".into(),
            ));
        }
        let symbols: SmallVec<[char; 8]> = local.iter().map(|operator| operator.symbol()).collect();
        let mut label = String::with_capacity(symbols.len() + splits.len());
        let mut boundary = 0;
        for position in 0..=symbols.len() {
            while splits.get(boundary) == Some(&position) {
                label.push('|');
                boundary += 1;
            }
            if let Some(symbol) = symbols.get(position) {
                label.push(*symbol);
            }
        }
        let split = (splits.len() == 1).then(|| splits[0]);
        Ok(Self {
            local,
            symbols,
            split,
            splits,
            label,
        })
    }

    pub fn local_operators(&self) -> &[LocalOperator] {
        &self.local
    }

    pub const fn split(&self) -> Option<usize> {
        self.split
    }

    pub fn splits(&self) -> &[usize] {
        &self.splits
    }

    pub fn label(&self) -> &str {
        &self.label
    }

    pub(crate) fn symbols(&self) -> &[char] {
        &self.symbols
    }
}

/// One complex coefficient and its ordered zero-based sites.
#[derive(Clone, Debug, PartialEq)]
pub struct Coupling {
    pub coefficient: Complex64,
    pub sites: Vec<usize>,
}

impl Coupling {
    pub fn new(coefficient: impl Into<Complex64>, sites: impl Into<Vec<usize>>) -> Self {
        Self {
            coefficient: coefficient.into(),
            sites: sites.into(),
        }
    }
}

/// Parsed-once local operator string and its couplings.
#[derive(Clone, Debug, PartialEq)]
pub struct OperatorTerm {
    product: OpProduct,
    couplings: Vec<Coupling>,
}

/// Native name for one typed product and its spatial couplings.
///
/// `OperatorTerm` remains as the compatibility spelling during migration.
pub type OperatorSpec = OperatorTerm;

impl OperatorTerm {
    pub fn from_product(
        product: OpProduct,
        couplings: impl IntoIterator<Item = Coupling>,
    ) -> Result<Self> {
        let arity = product.local_operators().len();
        let couplings: Vec<_> = couplings.into_iter().collect();
        for coupling in &couplings {
            if coupling.sites.len() != arity {
                return Err(QmbedError::InvalidCoupling(format!(
                    "operator {:?} has arity {arity}, but a coupling has {} sites",
                    product.label(),
                    coupling.sites.len()
                )));
            }
            if !coupling.coefficient.re.is_finite() || !coupling.coefficient.im.is_finite() {
                return Err(QmbedError::InvalidCoupling(
                    "coupling coefficients must be finite".into(),
                ));
            }
        }
        Ok(Self { product, couplings })
    }

    pub fn operator(&self) -> &str {
        self.product.label()
    }

    pub const fn product(&self) -> &OpProduct {
        &self.product
    }

    pub fn couplings(&self) -> &[Coupling] {
        &self.couplings
    }

    /// Return the same local action after a bijective site relabeling.
    ///
    /// `permutation[old_site]` is the site used by the returned term. This
    /// keeps frontend indexing conventions and lattice embeddings outside the
    /// universal assembler.
    pub fn with_site_permutation(&self, permutation: &[usize]) -> Result<Self> {
        if permutation.iter().any(|&site| site >= permutation.len()) {
            return Err(QmbedError::InvalidSite {
                site: permutation
                    .iter()
                    .copied()
                    .find(|&site| site >= permutation.len())
                    .unwrap_or(permutation.len()),
                sites: permutation.len(),
            });
        }
        let unique = permutation
            .iter()
            .copied()
            .collect::<std::collections::HashSet<_>>();
        if unique.len() != permutation.len() {
            return Err(QmbedError::InvalidOptions(
                "site permutation must be bijective".into(),
            ));
        }
        let couplings = self
            .couplings
            .iter()
            .map(|coupling| {
                let sites = coupling
                    .sites
                    .iter()
                    .map(|&site| {
                        permutation
                            .get(site)
                            .copied()
                            .ok_or(QmbedError::InvalidSite {
                                site,
                                sites: permutation.len(),
                            })
                    })
                    .collect::<Result<Vec<_>>>()?;
                Ok(Coupling::new(coupling.coefficient, sites))
            })
            .collect::<Result<Vec<_>>>()?;
        Self::from_product(self.product.clone(), couplings)
    }

    pub(crate) fn symbols(&self) -> &[char] {
        self.product.symbols()
    }

    pub(crate) const fn split(&self) -> Option<usize> {
        self.product.split()
    }
}

/// Requested materialization backend.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum MatrixFormat {
    Dense,
    Csc,
    Csr,
    Dia,
    MatrixFree,
}

/// Rectangular-capable narrow waist shared by stored and matrix-free maps.
pub trait LinearOperator: Send + Sync {
    fn shape(&self) -> (usize, usize);
    fn format(&self) -> MatrixFormat;
    fn apply(&self, input: &[Complex64], output: &mut [Complex64]) -> Result<()>;

    /// Whether this operator preserves real vectors exactly.
    ///
    /// Eigensolvers use this capability to retain the public complex-valued
    /// interface while avoiding complex arithmetic for real Hamiltonians.
    fn is_real(&self) -> bool {
        false
    }

    /// Apply this operator to a real vector.
    ///
    /// Custom operators can opt into the real fast path by overriding
    /// [`LinearOperator::is_real`] and this method. The compatibility fallback
    /// remains correct, but stored operators provide an allocation-free
    /// implementation below.
    fn apply_real(&self, input: &[f64], output: &mut [f64]) -> Result<()> {
        let shape = self.shape();
        if input.len() != shape.1 || output.len() != shape.0 {
            return Err(QmbedError::DimensionMismatch(format!(
                "shape {shape:?} requires real input length {} and output length {}, got {} and {}",
                shape.1,
                shape.0,
                input.len(),
                output.len()
            )));
        }
        let complex_input: Vec<_> = input
            .iter()
            .map(|&value| Complex64::new(value, 0.0))
            .collect();
        let mut complex_output = vec![Complex64::new(0.0, 0.0); shape.0];
        self.apply(&complex_input, &mut complex_output)?;
        for (real, complex) in output.iter_mut().zip(complex_output) {
            if complex.im.abs() > 1.0e-12 {
                return Err(QmbedError::UnsupportedBackend(
                    "operator declared a real action but produced an imaginary component".into(),
                ));
            }
            *real = complex.re;
        }
        Ok(())
    }

    /// Apply the algebraic transpose without conjugating either operand.
    ///
    /// Stored operators use their canonical triplets directly. A genuinely
    /// matrix-free implementation may override this method; the default falls
    /// back to column actions without ever retaining a full square matrix.
    fn apply_transpose(&self, input: &[Complex64], output: &mut [Complex64]) -> Result<()> {
        let (rows, columns) = self.shape();
        if input.len() != rows || output.len() != columns {
            return Err(QmbedError::DimensionMismatch(format!(
                "transpose of shape ({rows}, {columns}) requires input length {rows} and output length {columns}"
            )));
        }
        output.fill(Complex64::new(0.0, 0.0));
        if let Some(entries) = self.stored_triplets()? {
            for (row, column, value) in entries {
                output[column] += value * input[row];
            }
            return Ok(());
        }

        let mut basis_vector = vec![Complex64::new(0.0, 0.0); columns];
        let mut column_values = vec![Complex64::new(0.0, 0.0); rows];
        for column in 0..columns {
            basis_vector.fill(Complex64::new(0.0, 0.0));
            basis_vector[column] = Complex64::new(1.0, 0.0);
            self.apply(&basis_vector, &mut column_values)?;
            output[column] = column_values
                .iter()
                .zip(input.iter())
                .map(|(value, input_value)| *value * *input_value)
                .sum();
        }
        Ok(())
    }

    /// Apply the conjugate transpose.
    fn apply_adjoint(&self, input: &[Complex64], output: &mut [Complex64]) -> Result<()> {
        let conjugated_input: Vec<_> = input.iter().map(|value| value.conj()).collect();
        self.apply_transpose(&conjugated_input, output)?;
        for value in output {
            *value = value.conj();
        }
        Ok(())
    }

    /// Return canonical stored nonzeros when the representation already owns them.
    fn stored_triplets(&self) -> Result<Option<Vec<(usize, usize, Complex64)>>> {
        Ok(None)
    }

    fn shifted_solver(&self, _shift: f64) -> Result<Option<Box<dyn ShiftedLinearSolver>>> {
        Ok(None)
    }
}

/// Narrow waist for operators whose action depends explicitly on time.
pub trait TimeDependentOperator: Send + Sync {
    fn shape(&self) -> (usize, usize);
    fn apply_at(&self, time: f64, input: &[Complex64], output: &mut [Complex64]) -> Result<()>;
}

type TimedApply = Arc<dyn Fn(f64, &[Complex64], &mut [Complex64]) -> Result<()> + Send + Sync>;

/// Composable explicitly time-dependent linear map.
#[derive(Clone)]
pub struct TimeOperator {
    shape: (usize, usize),
    action: TimedApply,
}

impl std::fmt::Debug for TimeOperator {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("TimeOperator")
            .field("shape", &self.shape)
            .finish_non_exhaustive()
    }
}

impl TimeOperator {
    pub fn new<F>(shape: (usize, usize), action: F) -> Result<Self>
    where
        F: Fn(f64, &[Complex64], &mut [Complex64]) -> Result<()> + Send + Sync + 'static,
    {
        if shape.0 == 0 || shape.1 == 0 {
            return Err(QmbedError::DimensionMismatch(
                "time-dependent operator dimensions must be positive".into(),
            ));
        }
        Ok(Self {
            shape,
            action: Arc::new(action),
        })
    }

    pub fn from_operator<O>(operator: Arc<O>) -> Self
    where
        O: TimeDependentOperator + 'static,
    {
        let shape = operator.shape();
        Self {
            shape,
            action: Arc::new(move |time, input, output| operator.apply_at(time, input, output)),
        }
    }

    pub fn evaluate(&self, time: f64, format: MatrixFormat) -> Result<Operator> {
        if !time.is_finite() {
            return Err(QmbedError::InvalidOptions(
                "evaluation time must be finite".into(),
            ));
        }
        let mut input = vec![Complex64::new(0.0, 0.0); self.shape.1];
        let mut output = vec![Complex64::new(0.0, 0.0); self.shape.0];
        let mut triplets = Vec::new();
        for column in 0..self.shape.1 {
            input.fill(Complex64::new(0.0, 0.0));
            input[column] = Complex64::new(1.0, 0.0);
            self.apply_at(time, &input, &mut output)?;
            for (row, value) in output.iter().copied().enumerate() {
                if value.norm() > f64::EPSILON {
                    triplets.push((row, column, value));
                }
            }
        }
        Operator::from_triplets(self.shape.0, self.shape.1, triplets, format)
    }

    pub fn scaled(&self, coefficient: impl Into<Complex64>) -> Result<Self> {
        let coefficient = coefficient.into();
        if !coefficient.re.is_finite() || !coefficient.im.is_finite() {
            return Err(QmbedError::InvalidOptions(
                "time-operator scale must be finite".into(),
            ));
        }
        let operator = self.clone();
        Self::new(self.shape, move |time, input, output| {
            operator.apply_at(time, input, output)?;
            for value in output {
                *value *= coefficient;
            }
            Ok(())
        })
    }

    pub fn add(&self, right: &Self) -> Result<Self> {
        if self.shape != right.shape {
            return Err(QmbedError::DimensionMismatch(
                "time-dependent sums require equal shapes".into(),
            ));
        }
        let left = self.clone();
        let right = right.clone();
        Self::new(self.shape, move |time, input, output| {
            left.apply_at(time, input, output)?;
            let mut contribution = vec![Complex64::new(0.0, 0.0); output.len()];
            right.apply_at(time, input, &mut contribution)?;
            for (value, addition) in output.iter_mut().zip(contribution) {
                *value += addition;
            }
            Ok(())
        })
    }

    pub fn subtract(&self, right: &Self) -> Result<Self> {
        self.add(&right.scaled(-1.0)?)
    }

    pub fn product(&self, right: &Self) -> Result<Self> {
        if self.shape.1 != right.shape.0 {
            return Err(QmbedError::DimensionMismatch(
                "time-dependent product inner dimensions do not match".into(),
            ));
        }
        let left = self.clone();
        let right = right.clone();
        Self::new((self.shape.0, right.shape.1), move |time, input, output| {
            let mut intermediate = vec![Complex64::new(0.0, 0.0); right.shape.0];
            right.apply_at(time, input, &mut intermediate)?;
            left.apply_at(time, &intermediate, output)
        })
    }

    pub fn commutator(&self, right: &Self) -> Result<Self> {
        self.product(right)?.subtract(&right.product(self)?)
    }

    pub fn anticommutator(&self, right: &Self) -> Result<Self> {
        self.product(right)?.add(&right.product(self)?)
    }

    pub fn pow(&self, exponent: u32) -> Result<Self> {
        if self.shape.0 != self.shape.1 {
            return Err(QmbedError::DimensionMismatch(
                "time-dependent powers require a square operator".into(),
            ));
        }
        if exponent == 0 {
            let dimension = self.shape.0;
            return Self::new(self.shape, move |_time, input, output| {
                check_apply_shape((dimension, dimension), input, output)?;
                output.copy_from_slice(input);
                Ok(())
            });
        }
        let mut result = self.clone();
        for _ in 1..exponent {
            result = result.product(self)?;
        }
        Ok(result)
    }

    pub fn rotated(&self, unitary: &Operator, tolerance: f64) -> Result<Self> {
        if self.shape.0 != self.shape.1 || unitary.shape() != self.shape {
            return Err(QmbedError::DimensionMismatch(
                "time-dependent rotation needs equal square shapes".into(),
            ));
        }
        let adjoint = unitary.adjoint()?;
        let identity = adjoint.product(unitary)?;
        for row in 0..self.shape.0 {
            for column in 0..self.shape.1 {
                let expected = if row == column { 1.0 } else { 0.0 };
                if (identity.value_at(row, column) - expected).norm() > tolerance {
                    return Err(QmbedError::InvalidOptions(
                        "rotation matrix must be unitary".into(),
                    ));
                }
            }
        }
        let operator = self.clone();
        let unitary = unitary.clone();
        Self::new(self.shape, move |time, input, output| {
            let mut rotated_input = vec![Complex64::new(0.0, 0.0); input.len()];
            let mut applied = vec![Complex64::new(0.0, 0.0); output.len()];
            unitary.apply(input, &mut rotated_input)?;
            operator.apply_at(time, &rotated_input, &mut applied)?;
            adjoint.apply(&applied, output)
        })
    }
}

impl TimeDependentOperator for TimeOperator {
    fn shape(&self) -> (usize, usize) {
        self.shape
    }

    fn apply_at(&self, time: f64, input: &[Complex64], output: &mut [Complex64]) -> Result<()> {
        if !time.is_finite() {
            return Err(QmbedError::InvalidOptions(
                "time-dependent operator time must be finite".into(),
            ));
        }
        check_apply_shape(self.shape, input, output)?;
        (self.action)(time, input, output)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AssemblyChecks {
    pub hermiticity: bool,
    pub particle_conservation: bool,
    pub symmetry_compatibility: bool,
}

impl AssemblyChecks {
    pub const fn all() -> Self {
        Self {
            hermiticity: true,
            particle_conservation: true,
            symmetry_compatibility: true,
        }
    }

    pub const fn rectangular() -> Self {
        Self {
            hermiticity: false,
            particle_conservation: false,
            symmetry_compatibility: true,
        }
    }

    /// Disable Hamiltonian-specific validation for a general linear action.
    ///
    /// Individual local operators, cross-sector probes, and matrix-free
    /// actions are not necessarily Hermitian or particle conserving.
    pub const fn none() -> Self {
        Self {
            hermiticity: false,
            particle_conservation: false,
            symmetry_compatibility: false,
        }
    }
}

impl Default for AssemblyChecks {
    fn default() -> Self {
        Self::all()
    }
}

#[derive(Clone, Debug)]
struct Triplet {
    row: usize,
    column: usize,
    value: Complex64,
}

#[derive(Clone, Debug)]
enum Storage {
    Dense(Vec<Complex64>),
    Csc {
        column_offsets: Vec<usize>,
        row_indices: Vec<usize>,
        values: Vec<Complex64>,
    },
    Csr {
        row_offsets: Vec<usize>,
        column_indices: Vec<usize>,
        values: Vec<Complex64>,
    },
    Dia {
        offsets: Vec<isize>,
        values_by_row: Vec<Complex64>,
    },
    MatrixFree(Vec<Triplet>),
}

/// Concrete operator returned by universal assembly.
#[derive(Clone, Debug)]
pub struct Operator {
    shape: (usize, usize),
    format: MatrixFormat,
    storage: Storage,
    real: bool,
}

impl Operator {
    pub fn from_dense(
        rows: usize,
        columns: usize,
        values_row_major: Vec<Complex64>,
    ) -> Result<Self> {
        if values_row_major.len() != rows.saturating_mul(columns) {
            return Err(QmbedError::DimensionMismatch(format!(
                "dense storage has {} entries for shape ({rows}, {columns})",
                values_row_major.len()
            )));
        }
        let storage = Storage::Dense(values_row_major);
        Ok(Self {
            shape: (rows, columns),
            format: MatrixFormat::Dense,
            real: storage_is_real(&storage),
            storage,
        })
    }

    pub fn from_triplets(
        rows: usize,
        columns: usize,
        triplets: impl IntoIterator<Item = (usize, usize, Complex64)>,
        format: MatrixFormat,
    ) -> Result<Self> {
        if format == MatrixFormat::Dense {
            let mut values = vec![Complex64::new(0.0, 0.0); rows.saturating_mul(columns)];
            for (row, column, value) in triplets {
                if row >= rows
                    || column >= columns
                    || !value.re.is_finite()
                    || !value.im.is_finite()
                {
                    return Err(QmbedError::InvalidCoupling(
                        "triplet index is out of bounds or its value is non-finite".into(),
                    ));
                }
                values[row * columns + column] += value;
            }
            return Self::from_dense(rows, columns, values);
        }
        let mut accumulated = HashMap::new();
        for (row, column, value) in triplets {
            if row >= rows || column >= columns || !value.re.is_finite() || !value.im.is_finite() {
                return Err(QmbedError::InvalidCoupling(
                    "triplet index is out of bounds or its value is non-finite".into(),
                ));
            }
            *accumulated
                .entry((row, column))
                .or_insert(Complex64::new(0.0, 0.0)) += value;
        }
        let mut entries: Vec<_> = accumulated
            .into_iter()
            .filter_map(|((row, column), value)| {
                (value.norm() > f64::EPSILON).then_some(Triplet { row, column, value })
            })
            .collect();
        match format {
            MatrixFormat::Csc => entries.sort_by_key(|entry| (entry.column, entry.row)),
            MatrixFormat::Csr => entries.sort_by_key(|entry| (entry.row, entry.column)),
            MatrixFormat::Dia | MatrixFormat::MatrixFree => {
                entries.sort_by_key(|entry| (entry.row, entry.column));
            }
            MatrixFormat::Dense => unreachable!(),
        }
        let shape = (rows, columns);
        let storage = match format {
            MatrixFormat::Csc => csc_storage(&entries, columns),
            MatrixFormat::Csr => csr_storage(&entries, rows),
            MatrixFormat::Dia => dia_storage(&entries, rows),
            MatrixFormat::MatrixFree => Storage::MatrixFree(entries),
            MatrixFormat::Dense => unreachable!(),
        };
        Ok(Self {
            shape,
            format,
            real: storage_is_real(&storage),
            storage,
        })
    }

    pub fn to_dense(&self) -> Vec<Complex64> {
        if let Storage::Dense(values) = &self.storage {
            return values.clone();
        }
        let mut dense = vec![Complex64::new(0.0, 0.0); self.shape.0 * self.shape.1];
        match &self.storage {
            Storage::Dense(_) => unreachable!(),
            Storage::Csc {
                column_offsets,
                row_indices,
                values,
            } => {
                for column in 0..self.shape.1 {
                    for position in column_offsets[column]..column_offsets[column + 1] {
                        dense[row_indices[position] * self.shape.1 + column] = values[position];
                    }
                }
            }
            Storage::Csr {
                row_offsets,
                column_indices,
                values,
            } => {
                for row in 0..self.shape.0 {
                    for position in row_offsets[row]..row_offsets[row + 1] {
                        dense[row * self.shape.1 + column_indices[position]] = values[position];
                    }
                }
            }
            Storage::Dia {
                offsets,
                values_by_row,
            } => {
                for (diagonal, &offset) in offsets.iter().enumerate() {
                    for row in 0..self.shape.0 {
                        let Some(column) = row.checked_add_signed(-offset) else {
                            continue;
                        };
                        if column < self.shape.1 {
                            dense[row * self.shape.1 + column] =
                                values_by_row[diagonal * self.shape.0 + row];
                        }
                    }
                }
            }
            Storage::MatrixFree(entries) => {
                for entry in entries {
                    dense[entry.row * self.shape.1 + entry.column] = entry.value;
                }
            }
        }
        dense
    }

    pub fn conjugated(&self) -> Result<Self> {
        Self::from_triplets(
            self.shape.0,
            self.shape.1,
            self.triplets()
                .into_iter()
                .map(|(row, column, value)| (row, column, value.conj())),
            self.format,
        )
    }

    /// Row-vector action `inputᵀ A` without conjugating the input.
    pub fn right_apply(&self, input: &[Complex64]) -> Result<Vec<Complex64>> {
        if input.len() != self.shape.0 {
            return Err(QmbedError::DimensionMismatch(
                "right-apply input must match the operator row count".into(),
            ));
        }
        let mut output = vec![Complex64::new(0.0, 0.0); self.shape.1];
        self.apply_transpose(input, &mut output)?;
        Ok(output)
    }

    pub fn memory_bytes(&self) -> usize {
        let complex = std::mem::size_of::<Complex64>();
        let index = std::mem::size_of::<usize>();
        match &self.storage {
            Storage::Dense(values) => values.len() * complex,
            Storage::Csc {
                column_offsets,
                row_indices,
                values,
            } => column_offsets.len() * index + row_indices.len() * index + values.len() * complex,
            Storage::Csr {
                row_offsets,
                column_indices,
                values,
            } => row_offsets.len() * index + column_indices.len() * index + values.len() * complex,
            Storage::Dia {
                offsets,
                values_by_row,
            } => offsets.len() * std::mem::size_of::<isize>() + values_by_row.len() * complex,
            Storage::MatrixFree(entries) => entries.len() * std::mem::size_of::<Triplet>(),
        }
    }

    pub fn nnz(&self) -> usize {
        match &self.storage {
            Storage::Dense(values) => values
                .iter()
                .filter(|value| value.norm() > f64::EPSILON)
                .count(),
            Storage::Csc { values, .. } | Storage::Csr { values, .. } => values.len(),
            Storage::Dia { values_by_row, .. } => values_by_row
                .iter()
                .filter(|value| value.norm() > f64::EPSILON)
                .count(),
            Storage::MatrixFree(entries) => entries.len(),
        }
    }

    /// Canonical nonzero `(row, column, value)` entries without dense materialization.
    pub fn triplets(&self) -> Vec<(usize, usize, Complex64)> {
        let mut entries = match &self.storage {
            Storage::Dense(values) => values
                .iter()
                .enumerate()
                .filter_map(|(index, value)| {
                    (value.norm() > f64::EPSILON).then_some((
                        index / self.shape.1,
                        index % self.shape.1,
                        *value,
                    ))
                })
                .collect(),
            Storage::Csc {
                column_offsets,
                row_indices,
                values,
            } => {
                let mut entries = Vec::with_capacity(values.len());
                for column in 0..self.shape.1 {
                    for position in column_offsets[column]..column_offsets[column + 1] {
                        entries.push((row_indices[position], column, values[position]));
                    }
                }
                entries
            }
            Storage::Csr {
                row_offsets,
                column_indices,
                values,
            } => {
                let mut entries = Vec::with_capacity(values.len());
                for row in 0..self.shape.0 {
                    for position in row_offsets[row]..row_offsets[row + 1] {
                        entries.push((row, column_indices[position], values[position]));
                    }
                }
                entries
            }
            Storage::Dia {
                offsets,
                values_by_row,
            } => {
                let mut entries = Vec::new();
                for (diagonal, &offset) in offsets.iter().enumerate() {
                    for row in 0..self.shape.0 {
                        let Some(column) = row.checked_add_signed(-offset) else {
                            continue;
                        };
                        if column < self.shape.1 {
                            let value = values_by_row[diagonal * self.shape.0 + row];
                            if value.norm() > f64::EPSILON {
                                entries.push((row, column, value));
                            }
                        }
                    }
                }
                entries
            }
            Storage::MatrixFree(entries) => entries
                .iter()
                .map(|entry| (entry.row, entry.column, entry.value))
                .collect(),
        };
        entries.sort_by_key(|(row, column, _)| (*row, *column));
        entries
    }

    /// Convert the operator without changing its numerical values.
    pub fn converted(&self, format: MatrixFormat) -> Result<Self> {
        if format == self.format {
            return Ok(self.clone());
        }
        Self::from_triplets(self.shape.0, self.shape.1, self.triplets(), format)
    }

    pub fn diagonal(&self) -> Vec<Complex64> {
        let dimension = self.shape.0.min(self.shape.1);
        (0..dimension)
            .map(|index| self.value_at(index, index))
            .collect()
    }

    pub fn trace(&self) -> Result<Complex64> {
        if self.shape.0 != self.shape.1 {
            return Err(QmbedError::DimensionMismatch(
                "operator trace requires a square operator".into(),
            ));
        }
        Ok(self.diagonal().into_iter().sum())
    }

    pub fn scaled(&self, coefficient: impl Into<Complex64>) -> Result<Self> {
        let coefficient = coefficient.into();
        if !coefficient.re.is_finite() || !coefficient.im.is_finite() {
            return Err(QmbedError::InvalidOptions(
                "operator scale must be finite".into(),
            ));
        }
        Self::from_triplets(
            self.shape.0,
            self.shape.1,
            self.triplets()
                .into_iter()
                .map(|(row, column, value)| (row, column, coefficient * value)),
            self.format,
        )
    }

    pub fn add(&self, right: &Self) -> Result<Self> {
        self.combine(right, Complex64::new(1.0, 0.0))
    }

    pub fn subtract(&self, right: &Self) -> Result<Self> {
        self.combine(right, Complex64::new(-1.0, 0.0))
    }

    fn combine(&self, right: &Self, right_scale: Complex64) -> Result<Self> {
        if self.shape != right.shape {
            return Err(QmbedError::DimensionMismatch(
                "operator addition requires equal shapes".into(),
            ));
        }
        let entries = self.triplets().into_iter().chain(
            right
                .triplets()
                .into_iter()
                .map(|(row, column, value)| (row, column, right_scale * value)),
        );
        Self::from_triplets(self.shape.0, self.shape.1, entries, self.format)
    }

    /// Matrix product `self * right`.
    pub fn product(&self, right: &Self) -> Result<Self> {
        if self.shape.1 != right.shape.0 {
            return Err(QmbedError::DimensionMismatch(
                "operator product has incompatible inner dimensions".into(),
            ));
        }
        let mut right_by_row = vec![Vec::new(); right.shape.0];
        for (row, column, value) in right.triplets() {
            right_by_row[row].push((column, value));
        }
        let mut accumulated = HashMap::new();
        for (row, middle, left_value) in self.triplets() {
            for &(column, right_value) in &right_by_row[middle] {
                *accumulated
                    .entry((row, column))
                    .or_insert(Complex64::new(0.0, 0.0)) += left_value * right_value;
            }
        }
        Self::from_triplets(
            self.shape.0,
            right.shape.1,
            accumulated
                .into_iter()
                .map(|((row, column), value)| (row, column, value)),
            self.format,
        )
    }

    /// Project a square operator into the column space of `projector`.
    ///
    /// For a projector-like rectangular matrix `P` with shape `(N, M)`, this
    /// returns `P† self P` with shape `(M, M)`. The operation intentionally
    /// accepts any rectangular linear map; callers that require an isometry
    /// may validate `P†P = I` separately.
    pub fn projected_by(&self, projector: &Self) -> Result<Self> {
        if self.shape.0 != self.shape.1 || projector.shape.0 != self.shape.0 {
            return Err(QmbedError::DimensionMismatch(
                "operator projection requires a square source and matching projector rows".into(),
            ));
        }
        let applied = self.product(projector)?;
        projector.adjoint()?.product(&applied)
    }

    pub fn pow(&self, exponent: u32) -> Result<Self> {
        if self.shape.0 != self.shape.1 {
            return Err(QmbedError::DimensionMismatch(
                "operator powers require a square operator".into(),
            ));
        }
        let dimension = self.shape.0;
        let mut result = Self::from_triplets(
            dimension,
            dimension,
            (0..dimension).map(|index| (index, index, Complex64::new(1.0, 0.0))),
            self.format,
        )?;
        let mut base = self.clone();
        let mut remaining = exponent;
        while remaining > 0 {
            if remaining & 1 == 1 {
                result = result.product(&base)?;
            }
            remaining >>= 1;
            if remaining > 0 {
                base = base.product(&base)?;
            }
        }
        result.converted(self.format)
    }

    pub fn adjoint(&self) -> Result<Self> {
        Self::from_triplets(
            self.shape.1,
            self.shape.0,
            self.triplets()
                .into_iter()
                .map(|(row, column, value)| (column, row, value.conj())),
            self.format,
        )
    }

    pub fn transpose(&self) -> Result<Self> {
        Self::from_triplets(
            self.shape.1,
            self.shape.0,
            self.triplets()
                .into_iter()
                .map(|(row, column, value)| (column, row, value)),
            self.format,
        )
    }

    /// Similarity rotation `U† A U` after validating unitarity.
    pub fn rotated(&self, unitary: &Self, tolerance: f64) -> Result<Self> {
        if self.shape.0 != self.shape.1 || unitary.shape != self.shape {
            return Err(QmbedError::DimensionMismatch(
                "operator rotation requires equal square shapes".into(),
            ));
        }
        if !tolerance.is_finite() || tolerance <= 0.0 {
            return Err(QmbedError::InvalidOptions(
                "unitarity tolerance must be positive".into(),
            ));
        }
        let identity = unitary.adjoint()?.product(unitary)?;
        for row in 0..self.shape.0 {
            for column in 0..self.shape.1 {
                let expected = if row == column { 1.0 } else { 0.0 };
                if (identity.value_at(row, column) - expected).norm() > tolerance {
                    return Err(QmbedError::InvalidOptions(
                        "rotation matrix must be unitary".into(),
                    ));
                }
            }
        }
        unitary.adjoint()?.product(self)?.product(unitary)
    }

    fn value_at(&self, row: usize, column: usize) -> Complex64 {
        match &self.storage {
            Storage::Dense(values) => values[row * self.shape.1 + column],
            Storage::Csc {
                column_offsets,
                row_indices,
                values,
            } => {
                let range = column_offsets[column]..column_offsets[column + 1];
                row_indices[range.clone()]
                    .binary_search(&row)
                    .map_or(Complex64::new(0.0, 0.0), |position| {
                        values[range.start + position]
                    })
            }
            Storage::Csr {
                row_offsets,
                column_indices,
                values,
            } => {
                let range = row_offsets[row]..row_offsets[row + 1];
                column_indices[range.clone()]
                    .binary_search(&column)
                    .map_or(Complex64::new(0.0, 0.0), |position| {
                        values[range.start + position]
                    })
            }
            Storage::Dia {
                offsets,
                values_by_row,
            } => {
                let offset = row as isize - column as isize;
                offsets
                    .binary_search(&offset)
                    .map_or(Complex64::new(0.0, 0.0), |diagonal| {
                        values_by_row[diagonal * self.shape.0 + row]
                    })
            }
            Storage::MatrixFree(entries) => entries
                .binary_search_by_key(&(row, column), |entry| (entry.row, entry.column))
                .map_or(Complex64::new(0.0, 0.0), |position| entries[position].value),
        }
    }

    fn entry_is_hermitian(
        &self,
        row: usize,
        column: usize,
        value: Complex64,
        tolerance: f64,
    ) -> bool {
        (value - self.value_at(column, row).conj()).norm() <= tolerance
    }

    fn apply_real_coefficients(&self, input: &[Complex64], output: &mut [Complex64]) {
        match &self.storage {
            Storage::Dense(values) => {
                for row in 0..self.shape.0 {
                    let target = &mut output[row];
                    for column in 0..self.shape.1 {
                        let coefficient = values[row * self.shape.1 + column].re;
                        target.re += coefficient * input[column].re;
                        target.im += coefficient * input[column].im;
                    }
                }
            }
            Storage::Csc {
                column_offsets,
                row_indices,
                values,
            } => {
                for column in 0..self.shape.1 {
                    let input_value = input[column];
                    for position in column_offsets[column]..column_offsets[column + 1] {
                        let target = &mut output[row_indices[position]];
                        let coefficient = values[position].re;
                        target.re += coefficient * input_value.re;
                        target.im += coefficient * input_value.im;
                    }
                }
            }
            Storage::Csr {
                row_offsets,
                column_indices,
                values,
            } => {
                for row in 0..self.shape.0 {
                    let target = &mut output[row];
                    for position in row_offsets[row]..row_offsets[row + 1] {
                        let input_value = input[column_indices[position]];
                        let coefficient = values[position].re;
                        target.re += coefficient * input_value.re;
                        target.im += coefficient * input_value.im;
                    }
                }
            }
            Storage::Dia {
                offsets,
                values_by_row,
            } => {
                for (diagonal, &offset) in offsets.iter().enumerate() {
                    for row in 0..self.shape.0 {
                        let Some(column) = row.checked_add_signed(-offset) else {
                            continue;
                        };
                        if column < self.shape.1 {
                            let input_value = input[column];
                            let coefficient = values_by_row[diagonal * self.shape.0 + row].re;
                            output[row].re += coefficient * input_value.re;
                            output[row].im += coefficient * input_value.im;
                        }
                    }
                }
            }
            Storage::MatrixFree(entries) => {
                for entry in entries {
                    output[entry.row].re += entry.value.re * input[entry.column].re;
                    output[entry.row].im += entry.value.re * input[entry.column].im;
                }
            }
        }
    }

    pub fn is_hermitian(&self, tolerance: f64) -> bool {
        if self.shape.0 != self.shape.1 {
            return false;
        }
        match &self.storage {
            Storage::Dense(values) => {
                for row in 0..self.shape.0 {
                    for column in 0..self.shape.1 {
                        if !self.entry_is_hermitian(
                            row,
                            column,
                            values[row * self.shape.1 + column],
                            tolerance,
                        ) {
                            return false;
                        }
                    }
                }
            }
            Storage::Csc {
                column_offsets,
                row_indices,
                values,
            } => {
                for column in 0..self.shape.1 {
                    for position in column_offsets[column]..column_offsets[column + 1] {
                        if !self.entry_is_hermitian(
                            row_indices[position],
                            column,
                            values[position],
                            tolerance,
                        ) {
                            return false;
                        }
                    }
                }
            }
            Storage::Csr {
                row_offsets,
                column_indices,
                values,
            } => {
                for row in 0..self.shape.0 {
                    for position in row_offsets[row]..row_offsets[row + 1] {
                        if !self.entry_is_hermitian(
                            row,
                            column_indices[position],
                            values[position],
                            tolerance,
                        ) {
                            return false;
                        }
                    }
                }
            }
            Storage::Dia {
                offsets,
                values_by_row,
            } => {
                for (diagonal, &offset) in offsets.iter().enumerate() {
                    for row in 0..self.shape.0 {
                        let value = values_by_row[diagonal * self.shape.0 + row];
                        if value.norm() <= f64::EPSILON {
                            continue;
                        }
                        let Some(column) = row.checked_add_signed(-offset) else {
                            continue;
                        };
                        if column < self.shape.1
                            && !self.entry_is_hermitian(row, column, value, tolerance)
                        {
                            return false;
                        }
                    }
                }
            }
            Storage::MatrixFree(entries) => {
                for entry in entries {
                    if !self.entry_is_hermitian(entry.row, entry.column, entry.value, tolerance) {
                        return false;
                    }
                }
            }
        }
        true
    }
}

impl LinearOperator for Operator {
    fn shape(&self) -> (usize, usize) {
        self.shape
    }

    fn format(&self) -> MatrixFormat {
        self.format
    }

    fn is_real(&self) -> bool {
        self.real
    }

    fn apply_real(&self, input: &[f64], output: &mut [f64]) -> Result<()> {
        if input.len() != self.shape.1 || output.len() != self.shape.0 {
            return Err(QmbedError::DimensionMismatch(format!(
                "shape {:?} requires real input length {} and output length {}, got {} and {}",
                self.shape,
                self.shape.1,
                self.shape.0,
                input.len(),
                output.len()
            )));
        }
        if !self.real {
            return Err(QmbedError::UnsupportedBackend(
                "real action requires an operator with real-valued storage".into(),
            ));
        }
        output.fill(0.0);
        match &self.storage {
            Storage::Dense(values) => {
                for row in 0..self.shape.0 {
                    for column in 0..self.shape.1 {
                        output[row] += values[row * self.shape.1 + column].re * input[column];
                    }
                }
            }
            Storage::Csc {
                column_offsets,
                row_indices,
                values,
            } => {
                for column in 0..self.shape.1 {
                    let input_value = input[column];
                    for position in column_offsets[column]..column_offsets[column + 1] {
                        output[row_indices[position]] += values[position].re * input_value;
                    }
                }
            }
            Storage::Csr {
                row_offsets,
                column_indices,
                values,
            } => {
                for row in 0..self.shape.0 {
                    for position in row_offsets[row]..row_offsets[row + 1] {
                        output[row] += values[position].re * input[column_indices[position]];
                    }
                }
            }
            Storage::Dia {
                offsets,
                values_by_row,
            } => {
                for (diagonal, &offset) in offsets.iter().enumerate() {
                    for row in 0..self.shape.0 {
                        let Some(column) = row.checked_add_signed(-offset) else {
                            continue;
                        };
                        if column < self.shape.1 {
                            output[row] +=
                                values_by_row[diagonal * self.shape.0 + row].re * input[column];
                        }
                    }
                }
            }
            Storage::MatrixFree(entries) => {
                for entry in entries {
                    output[entry.row] += entry.value.re * input[entry.column];
                }
            }
        }
        Ok(())
    }

    fn apply(&self, input: &[Complex64], output: &mut [Complex64]) -> Result<()> {
        check_apply_shape(self.shape, input, output)?;
        output.fill(Complex64::new(0.0, 0.0));
        if self.real {
            self.apply_real_coefficients(input, output);
            return Ok(());
        }
        match &self.storage {
            Storage::Dense(values) => {
                for row in 0..self.shape.0 {
                    for column in 0..self.shape.1 {
                        output[row] += values[row * self.shape.1 + column] * input[column];
                    }
                }
            }
            Storage::Csc {
                column_offsets,
                row_indices,
                values,
            } => {
                for column in 0..self.shape.1 {
                    let input_value = input[column];
                    for position in column_offsets[column]..column_offsets[column + 1] {
                        output[row_indices[position]] += values[position] * input_value;
                    }
                }
            }
            Storage::Csr {
                row_offsets,
                column_indices,
                values,
            } => {
                for row in 0..self.shape.0 {
                    for position in row_offsets[row]..row_offsets[row + 1] {
                        output[row] += values[position] * input[column_indices[position]];
                    }
                }
            }
            Storage::Dia {
                offsets,
                values_by_row,
            } => {
                for (diagonal, &offset) in offsets.iter().enumerate() {
                    for row in 0..self.shape.0 {
                        let Some(column) = row.checked_add_signed(-offset) else {
                            continue;
                        };
                        if column < self.shape.1 {
                            output[row] +=
                                values_by_row[diagonal * self.shape.0 + row] * input[column];
                        }
                    }
                }
            }
            Storage::MatrixFree(entries) => {
                for entry in entries {
                    output[entry.row] += entry.value * input[entry.column];
                }
            }
        }
        Ok(())
    }

    fn stored_triplets(&self) -> Result<Option<Vec<(usize, usize, Complex64)>>> {
        Ok(Some(self.triplets()))
    }

    fn shifted_solver(&self, shift: f64) -> Result<Option<Box<dyn ShiftedLinearSolver>>> {
        if self.shape.0 != self.shape.1 || !shift.is_finite() {
            return Err(QmbedError::InvalidOptions(
                "shifted factorization requires a square operator and finite shift".into(),
            ));
        }
        let Storage::Csc {
            column_offsets,
            row_indices,
            values,
        } = &self.storage
        else {
            return Ok(None);
        };
        factor_shifted_csc(
            self.shape.0,
            column_offsets,
            row_indices,
            values,
            shift,
            self.real,
        )
        .map(Some)
    }
}

fn csc_storage(entries: &[Triplet], columns: usize) -> Storage {
    let mut column_offsets = vec![0_usize; columns + 1];
    for entry in entries {
        column_offsets[entry.column + 1] += 1;
    }
    for column in 0..columns {
        column_offsets[column + 1] += column_offsets[column];
    }
    Storage::Csc {
        column_offsets,
        row_indices: entries.iter().map(|entry| entry.row).collect(),
        values: entries.iter().map(|entry| entry.value).collect(),
    }
}

fn storage_is_real(storage: &Storage) -> bool {
    match storage {
        Storage::Dense(values)
        | Storage::Csc { values, .. }
        | Storage::Csr { values, .. }
        | Storage::Dia {
            values_by_row: values,
            ..
        } => values.iter().all(|value| value.im == 0.0),
        Storage::MatrixFree(entries) => entries.iter().all(|entry| entry.value.im == 0.0),
    }
}

fn csr_storage(entries: &[Triplet], rows: usize) -> Storage {
    let mut row_offsets = vec![0_usize; rows + 1];
    for entry in entries {
        row_offsets[entry.row + 1] += 1;
    }
    for row in 0..rows {
        row_offsets[row + 1] += row_offsets[row];
    }
    Storage::Csr {
        row_offsets,
        column_indices: entries.iter().map(|entry| entry.column).collect(),
        values: entries.iter().map(|entry| entry.value).collect(),
    }
}

fn dia_storage(entries: &[Triplet], rows: usize) -> Storage {
    let mut offsets: Vec<_> = entries
        .iter()
        .map(|entry| entry.row as isize - entry.column as isize)
        .collect();
    offsets.sort_unstable();
    offsets.dedup();
    let diagonal_index: HashMap<_, _> = offsets
        .iter()
        .copied()
        .enumerate()
        .map(|(index, offset)| (offset, index))
        .collect();
    let mut values_by_row = vec![Complex64::new(0.0, 0.0); offsets.len() * rows];
    for entry in entries {
        let offset = entry.row as isize - entry.column as isize;
        let diagonal = diagonal_index[&offset];
        values_by_row[diagonal * rows + entry.row] = entry.value;
    }
    Storage::Dia {
        offsets,
        values_by_row,
    }
}

pub(crate) fn check_apply_shape(
    shape: (usize, usize),
    input: &[Complex64],
    output: &[Complex64],
) -> Result<()> {
    if input.len() != shape.1 || output.len() != shape.0 {
        return Err(QmbedError::DimensionMismatch(format!(
            "shape {shape:?} requires input length {} and output length {}, got {} and {}",
            shape.1,
            shape.0,
            input.len(),
            output.len()
        )));
    }
    Ok(())
}

pub(crate) fn materialize_dense(
    operator: &(impl LinearOperator + ?Sized),
) -> Result<Vec<Complex64>> {
    let (rows, columns) = operator.shape();
    let mut dense = vec![Complex64::new(0.0, 0.0); rows * columns];
    let mut input = vec![Complex64::new(0.0, 0.0); columns];
    let mut output = vec![Complex64::new(0.0, 0.0); rows];
    for column in 0..columns {
        input.fill(Complex64::new(0.0, 0.0));
        input[column] = Complex64::new(1.0, 0.0);
        operator.apply(&input, &mut output)?;
        for row in 0..rows {
            dense[row * columns + column] = output[row];
        }
    }
    Ok(dense)
}

/// Shared scaled matrix-vector adapter corresponding to QuSpin's low-level
/// `matvec` helper. When `overwrite` is false the result is accumulated into
/// the existing output buffer.
pub fn matvec(
    operator: &(impl LinearOperator + ?Sized),
    input: &[Complex64],
    output: &mut [Complex64],
    coefficient: Complex64,
    overwrite: bool,
) -> Result<()> {
    if !coefficient.re.is_finite() || !coefficient.im.is_finite() {
        return Err(QmbedError::InvalidOptions(
            "matvec coefficient must be finite".into(),
        ));
    }
    if overwrite {
        operator.apply(input, output)?;
        for value in output {
            *value *= coefficient;
        }
        return Ok(());
    }

    let mut contribution = vec![Complex64::new(0.0, 0.0); output.len()];
    operator.apply(input, &mut contribution)?;
    for (value, addition) in output.iter_mut().zip(contribution) {
        *value += coefficient * addition;
    }
    Ok(())
}

/// Apply an operator to a column batch without constructing a dense matrix.
pub fn matmat(
    operator: &(impl LinearOperator + ?Sized),
    columns: &[Vec<Complex64>],
) -> Result<Vec<Vec<Complex64>>> {
    let (rows, input_dimension) = operator.shape();
    columns
        .iter()
        .map(|column| {
            if column.len() != input_dimension {
                return Err(QmbedError::DimensionMismatch(
                    "matrix column does not match the operator input dimension".into(),
                ));
            }
            let mut output = vec![Complex64::new(0.0, 0.0); rows];
            operator.apply(column, &mut output)?;
            Ok(output)
        })
        .collect()
}

/// Row-vector action through the transpose narrow waist.
pub fn rmatvec(
    operator: &(impl LinearOperator + ?Sized),
    input: &[Complex64],
) -> Result<Vec<Complex64>> {
    let mut output = vec![Complex64::new(0.0, 0.0); operator.shape().1];
    operator.apply_transpose(input, &mut output)?;
    Ok(output)
}

pub fn rmatmat(
    operator: &(impl LinearOperator + ?Sized),
    rows: &[Vec<Complex64>],
) -> Result<Vec<Vec<Complex64>>> {
    rows.iter().map(|row| rmatvec(operator, row)).collect()
}

/// Reusable owned matrix-vector plan for callback-driven algorithms.
#[derive(Clone)]
pub struct MatVecPlan {
    operator: Arc<dyn LinearOperator>,
}

impl std::fmt::Debug for MatVecPlan {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("MatVecPlan")
            .field("shape", &self.operator.shape())
            .finish()
    }
}

impl MatVecPlan {
    pub fn new(operator: Arc<dyn LinearOperator>) -> Self {
        Self { operator }
    }

    pub fn apply(
        &self,
        input: &[Complex64],
        output: &mut [Complex64],
        coefficient: Complex64,
        overwrite: bool,
    ) -> Result<()> {
        matvec(
            self.operator.as_ref(),
            input,
            output,
            coefficient,
            overwrite,
        )
    }

    pub fn operator(&self) -> &Arc<dyn LinearOperator> {
        &self.operator
    }
}

pub fn get_matvec_function(operator: Arc<dyn LinearOperator>) -> MatVecPlan {
    MatVecPlan::new(operator)
}

/// Matrix-free algebraic transpose view of an owned linear operator.
#[derive(Clone)]
pub struct TransposedLinearOperator {
    operator: Arc<dyn LinearOperator>,
}

impl TransposedLinearOperator {
    pub fn new(operator: Arc<dyn LinearOperator>) -> Self {
        Self { operator }
    }
}

impl std::fmt::Debug for TransposedLinearOperator {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("TransposedLinearOperator")
            .field("shape", &self.shape())
            .finish()
    }
}

impl LinearOperator for TransposedLinearOperator {
    fn shape(&self) -> (usize, usize) {
        let (rows, columns) = self.operator.shape();
        (columns, rows)
    }

    fn format(&self) -> MatrixFormat {
        MatrixFormat::MatrixFree
    }

    fn apply(&self, input: &[Complex64], output: &mut [Complex64]) -> Result<()> {
        self.operator.apply_transpose(input, output)
    }

    fn apply_transpose(&self, input: &[Complex64], output: &mut [Complex64]) -> Result<()> {
        self.operator.apply(input, output)
    }

    fn stored_triplets(&self) -> Result<Option<Vec<(usize, usize, Complex64)>>> {
        Ok(self.operator.stored_triplets()?.map(|entries| {
            entries
                .into_iter()
                .map(|(row, column, value)| (column, row, value))
                .collect()
        }))
    }
}

/// Matrix-free elementwise-conjugate view of an owned linear operator.
#[derive(Clone)]
pub struct ConjugatedLinearOperator {
    operator: Arc<dyn LinearOperator>,
}

impl ConjugatedLinearOperator {
    pub fn new(operator: Arc<dyn LinearOperator>) -> Self {
        Self { operator }
    }
}

impl std::fmt::Debug for ConjugatedLinearOperator {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ConjugatedLinearOperator")
            .field("shape", &self.shape())
            .finish()
    }
}

impl LinearOperator for ConjugatedLinearOperator {
    fn shape(&self) -> (usize, usize) {
        self.operator.shape()
    }

    fn format(&self) -> MatrixFormat {
        MatrixFormat::MatrixFree
    }

    fn apply(&self, input: &[Complex64], output: &mut [Complex64]) -> Result<()> {
        let conjugated_input: Vec<_> = input.iter().map(|value| value.conj()).collect();
        self.operator.apply(&conjugated_input, output)?;
        for value in output {
            *value = value.conj();
        }
        Ok(())
    }

    fn stored_triplets(&self) -> Result<Option<Vec<(usize, usize, Complex64)>>> {
        Ok(self.operator.stored_triplets()?.map(|entries| {
            entries
                .into_iter()
                .map(|(row, column, value)| (row, column, value.conj()))
                .collect()
        }))
    }
}

/// Matrix-free conjugate-transpose view of an owned linear operator.
#[derive(Clone)]
pub struct AdjointLinearOperator {
    operator: Arc<dyn LinearOperator>,
}

impl AdjointLinearOperator {
    pub fn new(operator: Arc<dyn LinearOperator>) -> Self {
        Self { operator }
    }
}

impl std::fmt::Debug for AdjointLinearOperator {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("AdjointLinearOperator")
            .field("shape", &self.shape())
            .finish()
    }
}

impl LinearOperator for AdjointLinearOperator {
    fn shape(&self) -> (usize, usize) {
        let (rows, columns) = self.operator.shape();
        (columns, rows)
    }

    fn format(&self) -> MatrixFormat {
        MatrixFormat::MatrixFree
    }

    fn apply(&self, input: &[Complex64], output: &mut [Complex64]) -> Result<()> {
        self.operator.apply_adjoint(input, output)
    }

    fn stored_triplets(&self) -> Result<Option<Vec<(usize, usize, Complex64)>>> {
        Ok(self.operator.stored_triplets()?.map(|entries| {
            entries
                .into_iter()
                .map(|(row, column, value)| (column, row, value.conj()))
                .collect()
        }))
    }
}

/// Universal square or cross-sector operator builder.
pub struct OperatorBuilder<'a, Source, Target>
where
    Source: Basis,
    Target: Basis<State = Source::State>,
{
    source: &'a Source,
    target: &'a Target,
    terms: Vec<OperatorTerm>,
    checks: AssemblyChecks,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct ParticleConservationComponent {
    /// Cartesian operators share one normalization within a concrete basis,
    /// but that normalization can differ from explicit ladder operators.
    /// Keeping the degree in the key permits exact `xx + yy` cancellation
    /// without assuming a particular spin normalization.
    cartesian_degree: usize,
    operator: String,
    sites: Vec<usize>,
}

fn definite_number_change_components(operator: &str) -> Result<Vec<(usize, String, Complex64)>> {
    let mut components = vec![(0_usize, String::new(), Complex64::new(1.0, 0.0))];
    for symbol in operator.chars() {
        match symbol {
            'x' | 'y' => {
                let mut expanded = Vec::with_capacity(components.len().saturating_mul(2));
                for (degree, prefix, coefficient) in components {
                    let mut raised = prefix.clone();
                    raised.push('+');
                    let mut lowered = prefix;
                    lowered.push('-');
                    let (raise_phase, lower_phase) = if symbol == 'x' {
                        (Complex64::new(1.0, 0.0), Complex64::new(1.0, 0.0))
                    } else {
                        (Complex64::new(0.0, -1.0), Complex64::new(0.0, 1.0))
                    };
                    expanded.push((degree + 1, raised, coefficient * raise_phase));
                    expanded.push((degree + 1, lowered, coefficient * lower_phase));
                }
                components = expanded;
            }
            'I' | 'n' | 'z' | '+' | '-' | '|' => {
                for (_, prefix, _) in &mut components {
                    prefix.push(symbol);
                }
            }
            _ => return Err(QmbedError::InvalidOperator(operator.into())),
        }
    }
    Ok(components)
}

/// Check additive-sector conservation for a complete operator sum.
///
/// Cartesian products are first decomposed into definite raising/lowering
/// components. Coefficients are accumulated for identical spatial actions
/// before sector closure is tested, so cancellations such as `xx + yy` are
/// recognized. The Cartesian degree remains part of the canonical key because
/// a basis may normalize `x/y` differently from explicit `+/-` symbols.
pub fn operator_sum_preserves_particle_sector<B>(basis: &B, terms: &[OperatorTerm]) -> Result<bool>
where
    B: Basis,
{
    let mut components = HashMap::<ParticleConservationComponent, Complex64>::new();
    for term in terms {
        for coupling in term.couplings() {
            for (cartesian_degree, operator, coefficient) in
                definite_number_change_components(term.operator())?
            {
                *components
                    .entry(ParticleConservationComponent {
                        cartesian_degree,
                        operator,
                        sites: coupling.sites.clone(),
                    })
                    .or_insert(Complex64::new(0.0, 0.0)) += coupling.coefficient * coefficient;
            }
        }
    }
    for (component, coefficient) in components {
        if coefficient.norm() > f64::EPSILON
            && !basis.operator_preserves_particle_sector_on_sites(
                &component.operator,
                &component.sites,
            )?
        {
            return Ok(false);
        }
    }
    Ok(true)
}

impl<'a, BasisType> OperatorBuilder<'a, BasisType, BasisType>
where
    BasisType: Basis,
{
    pub fn on(basis: &'a BasisType) -> Self {
        Self {
            source: basis,
            target: basis,
            terms: Vec::new(),
            checks: AssemblyChecks::all(),
        }
    }
}

impl<'a, Source, Target> OperatorBuilder<'a, Source, Target>
where
    Source: Basis,
    Target: Basis<State = Source::State>,
{
    pub fn between(source: &'a Source, target: &'a Target) -> Self {
        Self {
            source,
            target,
            terms: Vec::new(),
            checks: AssemblyChecks::rectangular(),
        }
    }

    pub fn terms(mut self, terms: impl IntoIterator<Item = OperatorTerm>) -> Self {
        self.terms.extend(terms);
        self
    }

    pub fn term(mut self, term: OperatorTerm) -> Self {
        self.terms.push(term);
        self
    }

    pub const fn checks(mut self, checks: AssemblyChecks) -> Self {
        self.checks = checks;
        self
    }

    pub fn build(self, format: MatrixFormat) -> Result<Operator> {
        let shape = (self.target.len(), self.source.len());
        if self.checks.particle_conservation
            && (!operator_sum_preserves_particle_sector(self.source, &self.terms)?
                || !operator_sum_preserves_particle_sector(self.target, &self.terms)?)
        {
            return Err(QmbedError::InvalidSector(
                "operator sum does not preserve the selected particle sector".into(),
            ));
        }
        // Local actions are generated one source column at a time. Preserve
        // that structure instead of hashing every `(row, column)` pair into a
        // global table and sorting it back into sparse-matrix order later.
        //
        // A reusable column buffer remains fully general: arbitrary terms may
        // still emit the same row repeatedly, including branching and
        // symmetry-reduced actions. Sorting and coalescing only the current
        // column bounds temporary memory by the local connectivity.
        let local_capacity = self.terms.iter().map(|term| term.couplings().len()).sum();
        let mut column_entries = Vec::<(usize, Complex64)>::with_capacity(local_capacity);
        let mut entries = Vec::<Triplet>::new();
        let mut csc_column_offsets = Vec::new();
        let mut csc_row_indices = Vec::new();
        let mut csc_values = Vec::new();
        if format == MatrixFormat::Csc {
            csc_column_offsets.reserve(shape.1 + 1);
            csc_column_offsets.push(0);
        }
        for column in 0..self.source.len() {
            column_entries.clear();
            let source_state = self.source.state(column)?;
            let source_orbit_size = self.source.transition_orbit_size(source_state)?;
            for term in &self.terms {
                for coupling in term.couplings() {
                    self.source.visit_preparsed_local_unreduced_transitions(
                        source_state,
                        term.operator(),
                        term.symbols(),
                        term.split(),
                        &coupling.sites,
                        |unreduced_target, local_amplitude| {
                            let Some((row, reduction_amplitude)) = self
                                .target
                                .index_transition(unreduced_target, source_orbit_size)?
                            else {
                                return Ok(());
                            };
                            column_entries.push((
                                row,
                                coupling.coefficient * local_amplitude * reduction_amplitude,
                            ));
                            Ok(())
                        },
                    )?;
                }
            }

            column_entries.sort_unstable_by_key(|(row, _)| *row);
            let mut position = 0;
            while position < column_entries.len() {
                let row = column_entries[position].0;
                let mut value = column_entries[position].1;
                position += 1;
                while position < column_entries.len() && column_entries[position].0 == row {
                    value += column_entries[position].1;
                    position += 1;
                }
                if value.norm() <= f64::EPSILON {
                    continue;
                }
                if format == MatrixFormat::Csc {
                    csc_row_indices.push(row);
                    csc_values.push(value);
                } else {
                    entries.push(Triplet { row, column, value });
                }
            }
            if format == MatrixFormat::Csc {
                csc_column_offsets.push(csc_row_indices.len());
            }
        }
        match format {
            MatrixFormat::Csc => {}
            MatrixFormat::Csr => entries.sort_by_key(|entry| (entry.row, entry.column)),
            _ => entries.sort_by_key(|entry| (entry.row, entry.column)),
        }
        if format == MatrixFormat::Dia
            && entries
                .iter()
                .map(|entry| entry.row as isize - entry.column as isize)
                .collect::<std::collections::HashSet<_>>()
                .len()
                > shape.0.min(shape.1).saturating_mul(2).max(1)
        {
            return Err(QmbedError::UnsupportedBackend(
                "the requested operator is not usefully diagonal-banded".into(),
            ));
        }
        let storage = match format {
            MatrixFormat::Dense => {
                let mut dense = vec![Complex64::new(0.0, 0.0); shape.0 * shape.1];
                for entry in &entries {
                    dense[entry.row * shape.1 + entry.column] = entry.value;
                }
                Storage::Dense(dense)
            }
            MatrixFormat::Csc => Storage::Csc {
                column_offsets: csc_column_offsets,
                row_indices: csc_row_indices,
                values: csc_values,
            },
            MatrixFormat::Csr => csr_storage(&entries, shape.0),
            MatrixFormat::Dia => dia_storage(&entries, shape.0),
            MatrixFormat::MatrixFree => Storage::MatrixFree(entries),
        };
        let operator = Operator {
            shape,
            format,
            real: storage_is_real(&storage),
            storage,
        };
        if self.checks.hermiticity && !operator.is_hermitian(1.0e-12) {
            return Err(QmbedError::NonHermitian);
        }
        Ok(operator)
    }

    /// Assemble static and driven terms through the same local-action path.
    pub fn build_dynamic(
        self,
        dynamic_terms: impl IntoIterator<Item = DynamicTerm>,
        format: MatrixFormat,
    ) -> Result<Hamiltonian<Dynamic>> {
        let Self {
            source,
            target,
            terms,
            checks,
        } = self;
        if source.len() != target.len() {
            return Err(QmbedError::DimensionMismatch(
                "a Hamiltonian must be square".into(),
            ));
        }
        let static_part = Self {
            source,
            target,
            terms,
            checks,
        }
        .build(format)?;
        let component_checks = AssemblyChecks {
            hermiticity: false,
            particle_conservation: checks.particle_conservation,
            symmetry_compatibility: checks.symmetry_compatibility,
        };
        let mut components = Vec::new();
        for dynamic_term in dynamic_terms {
            let (term, drive) = dynamic_term.into_parts();
            let operator = Self {
                source,
                target,
                terms: vec![term],
                checks: component_checks,
            }
            .build(format)?;
            components.push(DynamicComponent { operator, drive });
        }
        Hamiltonian::<Dynamic>::new(static_part, components)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct BraKetTransition<State> {
    pub bra: State,
    pub ket: State,
    pub matrix_element: Complex64,
}

/// Raw local transition table, including branching callbacks and fermionic signs.
pub fn bra_ket_transitions<B>(
    basis: &B,
    operator: &str,
    sites: &[usize],
    coefficient: impl Into<Complex64>,
    kets: impl IntoIterator<Item = B::State>,
) -> Result<Vec<BraKetTransition<B::State>>>
where
    B: Basis,
{
    let coefficient = coefficient.into();
    if !coefficient.re.is_finite() || !coefficient.im.is_finite() {
        return Err(QmbedError::InvalidCoupling(
            "transition-table coefficient must be finite".into(),
        ));
    }
    let mut transitions = Vec::new();
    for ket in kets {
        basis.visit_local_unreduced_transitions(ket, operator, sites, |bra, amplitude| {
            let matrix_element = coefficient * amplitude;
            if matrix_element.norm() > f64::EPSILON {
                transitions.push(BraKetTransition {
                    bra,
                    ket,
                    matrix_element,
                });
            }
            Ok(())
        })?;
    }
    Ok(transitions)
}

/// Apply parsed terms directly between sectors without materializing a matrix.
pub fn apply_sector_shift<Source, Target>(
    source: &Source,
    target: &Target,
    terms: &[OperatorTerm],
    input: &[Complex64],
    output: &mut [Complex64],
) -> Result<()>
where
    Source: Basis,
    Target: Basis<State = Source::State>,
{
    if input.len() != source.len() || output.len() != target.len() {
        return Err(QmbedError::DimensionMismatch(
            "sector-shift state dimensions do not match source and target bases".into(),
        ));
    }
    output.fill(Complex64::new(0.0, 0.0));
    for (column, input_value) in input.iter().copied().enumerate() {
        if input_value.norm() <= f64::EPSILON {
            continue;
        }
        let source_state = source.state(column)?;
        let source_orbit_size = source.transition_orbit_size(source_state)?;
        for term in terms {
            for coupling in term.couplings() {
                source.visit_local_unreduced_transitions(
                    source_state,
                    term.operator(),
                    &coupling.sites,
                    |unreduced_target, local_amplitude| {
                        let Some((row, reduction_amplitude)) =
                            target.index_transition(unreduced_target, source_orbit_size)?
                        else {
                            return Ok(());
                        };
                        output[row] += input_value
                            * coupling.coefficient
                            * local_amplitude
                            * reduction_amplitude;
                        Ok(())
                    },
                )?;
            }
        }
    }
    Ok(())
}

type DriveFunction = Arc<dyn Fn(f64) -> Complex64 + Send + Sync>;

/// A parsed local term multiplied by a scalar function of time.
#[derive(Clone)]
pub struct DynamicTerm {
    term: OperatorTerm,
    drive: DriveFunction,
}

impl std::fmt::Debug for DynamicTerm {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("DynamicTerm")
            .field("term", &self.term)
            .field("drive", &"<callable>")
            .finish()
    }
}

impl DynamicTerm {
    pub fn new<F>(term: OperatorTerm, drive: F) -> Self
    where
        F: Fn(f64) -> Complex64 + Send + Sync + 'static,
    {
        Self {
            term,
            drive: Arc::new(drive),
        }
    }

    pub fn coefficient_at(&self, time: f64) -> Result<Complex64> {
        finite_drive_value(time, (self.drive)(time))
    }

    fn into_parts(self) -> (OperatorTerm, DriveFunction) {
        (self.term, self.drive)
    }
}

#[derive(Clone)]
pub struct DynamicComponent {
    operator: Operator,
    drive: DriveFunction,
}

impl std::fmt::Debug for DynamicComponent {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("DynamicComponent")
            .field("operator", &self.operator)
            .field("drive", &"<callable>")
            .finish()
    }
}

impl DynamicComponent {
    pub fn new<F>(operator: Operator, drive: F) -> Self
    where
        F: Fn(f64) -> Complex64 + Send + Sync + 'static,
    {
        Self {
            operator,
            drive: Arc::new(drive),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Static;

#[derive(Clone, Copy, Debug)]
pub struct Dynamic;

/// Static or explicitly time-dependent Hamiltonian with a type-state marker.
#[derive(Clone, Debug)]
pub struct Hamiltonian<Kind = Static> {
    static_part: Operator,
    dynamic_parts: Vec<DynamicComponent>,
    marker: PhantomData<Kind>,
}

impl Hamiltonian<Static> {
    pub fn new(operator: Operator) -> Result<Self> {
        let shape = operator.shape();
        if shape.0 != shape.1 {
            return Err(QmbedError::DimensionMismatch(
                "a Hamiltonian must be square".into(),
            ));
        }
        Ok(Self {
            static_part: operator,
            dynamic_parts: Vec::new(),
            marker: PhantomData,
        })
    }

    pub fn operator(&self) -> &Operator {
        &self.static_part
    }

    pub fn transpose(&self) -> Result<Self> {
        Self::new(self.static_part.transpose()?)
    }

    pub fn conjugated(&self) -> Result<Self> {
        Self::new(self.static_part.conjugated()?)
    }

    pub fn adjoint(&self) -> Result<Self> {
        Self::new(self.static_part.adjoint()?)
    }

    pub fn scaled(&self, coefficient: impl Into<Complex64>) -> Result<Self> {
        Self::new(self.static_part.scaled(coefficient)?)
    }
}

impl Hamiltonian<Dynamic> {
    pub fn new(static_part: Operator, dynamic_parts: Vec<DynamicComponent>) -> Result<Self> {
        let shape = static_part.shape();
        if shape.0 != shape.1
            || dynamic_parts
                .iter()
                .any(|component| component.operator.shape() != shape)
        {
            return Err(QmbedError::DimensionMismatch(
                "all Hamiltonian components must share one square shape".into(),
            ));
        }
        Ok(Self {
            static_part,
            dynamic_parts,
            marker: PhantomData,
        })
    }

    pub fn evaluate(&self, time: f64, format: MatrixFormat) -> Result<Operator> {
        if !time.is_finite() {
            return Err(QmbedError::InvalidOptions(
                "evaluation time must be finite".into(),
            ));
        }
        let shape = self.static_part.shape();
        let mut entries = self.static_part.triplets();
        for component in &self.dynamic_parts {
            let coefficient = finite_drive_value(time, (component.drive)(time))?;
            entries.extend(
                component
                    .operator
                    .triplets()
                    .into_iter()
                    .map(|(row, column, value)| (row, column, coefficient * value)),
            );
        }
        Operator::from_triplets(shape.0, shape.1, entries, format)
    }

    pub fn dynamic_components(&self) -> usize {
        self.dynamic_parts.len()
    }

    pub fn static_part(&self) -> &Operator {
        &self.static_part
    }

    pub fn transpose(&self) -> Result<Self> {
        Self::new(
            self.static_part.transpose()?,
            self.dynamic_parts
                .iter()
                .map(|component| {
                    Ok(DynamicComponent {
                        operator: component.operator.transpose()?,
                        drive: component.drive.clone(),
                    })
                })
                .collect::<Result<_>>()?,
        )
    }

    pub fn conjugated(&self) -> Result<Self> {
        Self::new(
            self.static_part.conjugated()?,
            self.dynamic_parts
                .iter()
                .map(|component| {
                    let drive = component.drive.clone();
                    Ok(DynamicComponent {
                        operator: component.operator.conjugated()?,
                        drive: Arc::new(move |time| drive(time).conj()),
                    })
                })
                .collect::<Result<_>>()?,
        )
    }

    pub fn adjoint(&self) -> Result<Self> {
        Self::new(
            self.static_part.adjoint()?,
            self.dynamic_parts
                .iter()
                .map(|component| {
                    let drive = component.drive.clone();
                    Ok(DynamicComponent {
                        operator: component.operator.adjoint()?,
                        drive: Arc::new(move |time| drive(time).conj()),
                    })
                })
                .collect::<Result<_>>()?,
        )
    }

    pub fn scaled(&self, coefficient: impl Into<Complex64>) -> Result<Self> {
        let coefficient = coefficient.into();
        if !coefficient.re.is_finite() || !coefficient.im.is_finite() {
            return Err(QmbedError::InvalidOptions(
                "Hamiltonian scale must be finite".into(),
            ));
        }
        Self::new(
            self.static_part.scaled(coefficient)?,
            self.dynamic_parts
                .iter()
                .map(|component| DynamicComponent {
                    operator: component.operator.clone(),
                    drive: {
                        let drive = component.drive.clone();
                        Arc::new(move |time| coefficient * drive(time))
                    },
                })
                .collect(),
        )
    }
}

fn finite_drive_value(time: f64, value: Complex64) -> Result<Complex64> {
    if !time.is_finite() || !value.re.is_finite() || !value.im.is_finite() {
        return Err(QmbedError::InvalidOptions(
            "drive time and coefficient must be finite".into(),
        ));
    }
    Ok(value)
}

impl LinearOperator for Hamiltonian<Static> {
    fn shape(&self) -> (usize, usize) {
        self.static_part.shape()
    }

    fn format(&self) -> MatrixFormat {
        self.static_part.format()
    }

    fn apply(&self, input: &[Complex64], output: &mut [Complex64]) -> Result<()> {
        self.static_part.apply(input, output)
    }

    fn apply_transpose(&self, input: &[Complex64], output: &mut [Complex64]) -> Result<()> {
        self.static_part.apply_transpose(input, output)
    }

    fn apply_adjoint(&self, input: &[Complex64], output: &mut [Complex64]) -> Result<()> {
        self.static_part.apply_adjoint(input, output)
    }

    fn stored_triplets(&self) -> Result<Option<Vec<(usize, usize, Complex64)>>> {
        self.static_part.stored_triplets()
    }

    fn shifted_solver(&self, shift: f64) -> Result<Option<Box<dyn ShiftedLinearSolver>>> {
        self.static_part.shifted_solver(shift)
    }
}

impl TimeDependentOperator for Hamiltonian<Static> {
    fn shape(&self) -> (usize, usize) {
        self.static_part.shape()
    }

    fn apply_at(&self, time: f64, input: &[Complex64], output: &mut [Complex64]) -> Result<()> {
        if !time.is_finite() {
            return Err(QmbedError::InvalidOptions(
                "evaluation time must be finite".into(),
            ));
        }
        self.static_part.apply(input, output)
    }
}

impl TimeDependentOperator for Hamiltonian<Dynamic> {
    fn shape(&self) -> (usize, usize) {
        self.static_part.shape()
    }

    fn apply_at(&self, time: f64, input: &[Complex64], output: &mut [Complex64]) -> Result<()> {
        check_apply_shape(self.static_part.shape(), input, output)?;
        self.static_part.apply(input, output)?;
        let mut driven = vec![Complex64::new(0.0, 0.0); output.len()];
        for component in &self.dynamic_parts {
            let coefficient = finite_drive_value(time, (component.drive)(time))?;
            component.operator.apply(input, &mut driven)?;
            for (value, contribution) in output.iter_mut().zip(&driven) {
                *value += coefficient * *contribution;
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct QuantumComponent {
    name: String,
    operator: Operator,
    default: Option<Complex64>,
}

impl QuantumComponent {
    pub fn required(name: impl Into<String>, operator: Operator) -> Self {
        Self {
            name: name.into(),
            operator,
            default: None,
        }
    }

    pub fn with_default(
        name: impl Into<String>,
        operator: Operator,
        default: impl Into<Complex64>,
    ) -> Self {
        Self {
            name: name.into(),
            operator,
            default: Some(default.into()),
        }
    }

    /// Python-compatible parameter component: an omitted parameter equals one.
    pub fn parameter(name: impl Into<String>, operator: Operator) -> Self {
        Self::with_default(name, operator, Complex64::new(1.0, 0.0))
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn operator(&self) -> &Operator {
        &self.operator
    }

    pub const fn default(&self) -> Option<Complex64> {
        self.default
    }
}

/// Named linear combination of operator components.
#[derive(Clone, Debug)]
pub struct QuantumOperator {
    components: Vec<QuantumComponent>,
    shape: (usize, usize),
}

impl QuantumOperator {
    pub fn new(components: impl IntoIterator<Item = QuantumComponent>) -> Result<Self> {
        let components: Vec<_> = components.into_iter().collect();
        let first = components.first().ok_or_else(|| {
            QmbedError::InvalidOptions("QuantumOperator requires at least one component".into())
        })?;
        let shape = first.operator.shape();
        let mut names = std::collections::HashSet::new();
        for component in &components {
            if component.name.is_empty() || !names.insert(component.name.clone()) {
                return Err(QmbedError::InvalidOptions(
                    "component names must be nonempty and unique".into(),
                ));
            }
            if component.operator.shape() != shape {
                return Err(QmbedError::DimensionMismatch(
                    "all parameterized components must have equal shapes".into(),
                ));
            }
            if component
                .default
                .is_some_and(|value| !value.re.is_finite() || !value.im.is_finite())
            {
                return Err(QmbedError::InvalidOptions(
                    "component defaults must be finite".into(),
                ));
            }
        }
        Ok(Self { components, shape })
    }

    pub const fn shape(&self) -> (usize, usize) {
        self.shape
    }

    pub fn component_names(&self) -> impl Iterator<Item = &str> {
        self.components
            .iter()
            .map(|component| component.name.as_str())
    }

    pub fn components(&self) -> &[QuantumComponent] {
        &self.components
    }

    pub fn component(&self, name: &str) -> Result<&Operator> {
        self.components
            .iter()
            .find(|component| component.name == name)
            .map(|component| &component.operator)
            .ok_or_else(|| {
                QmbedError::InvalidOptions(format!("unknown operator component {name:?}"))
            })
    }

    /// Resolve named parameters into the stable component order returned by
    /// [`Self::component_names`].
    ///
    /// This is the allocation-light coefficient boundary used by dynamic
    /// frontends: names are validated once at the edge, while repeated
    /// operator applications consume a compact coefficient slice.
    pub fn resolve_coefficients(
        &self,
        parameters: &HashMap<String, Complex64>,
    ) -> Result<Vec<Complex64>> {
        if let Some(name) = parameters.keys().find(|name| {
            !self
                .components
                .iter()
                .any(|component| &component.name == *name)
        }) {
            return Err(QmbedError::InvalidOptions(format!(
                "unknown operator parameter {name:?}"
            )));
        }
        self.components
            .iter()
            .map(|component| {
                let coefficient = parameters
                    .get(&component.name)
                    .copied()
                    .or(component.default)
                    .ok_or_else(|| {
                        QmbedError::InvalidOptions(format!(
                            "missing required operator parameter {:?}",
                            component.name
                        ))
                    })?;
                if !coefficient.re.is_finite() || !coefficient.im.is_finite() {
                    return Err(QmbedError::InvalidOptions(format!(
                        "operator parameter {:?} must be finite",
                        component.name
                    )));
                }
                Ok(coefficient)
            })
            .collect()
    }

    /// Apply a compact ordered coefficient vector without materializing the
    /// linear combination.
    pub fn apply_coefficients(
        &self,
        coefficients: &[Complex64],
        input: &[Complex64],
        output: &mut [Complex64],
    ) -> Result<()> {
        if coefficients.len() != self.components.len() {
            return Err(QmbedError::DimensionMismatch(format!(
                "received {} operator coefficients for {} components",
                coefficients.len(),
                self.components.len()
            )));
        }
        check_apply_shape(self.shape, input, output)?;
        output.fill(Complex64::new(0.0, 0.0));
        let mut contribution = vec![Complex64::new(0.0, 0.0); output.len()];
        for (component, coefficient) in self.components.iter().zip(coefficients) {
            if !coefficient.re.is_finite() || !coefficient.im.is_finite() {
                return Err(QmbedError::InvalidOptions(format!(
                    "operator parameter {:?} must be finite",
                    component.name
                )));
            }
            component.operator.apply(input, &mut contribution)?;
            for (value, addition) in output.iter_mut().zip(&contribution) {
                *value += *coefficient * *addition;
            }
        }
        Ok(())
    }

    /// Apply one named member of the family without assembling a temporary
    /// sparse matrix.
    pub fn apply(
        &self,
        parameters: &HashMap<String, Complex64>,
        input: &[Complex64],
        output: &mut [Complex64],
    ) -> Result<()> {
        let coefficients = self.resolve_coefficients(parameters)?;
        self.apply_coefficients(&coefficients, input, output)
    }

    pub fn evaluate(
        &self,
        parameters: &HashMap<String, Complex64>,
        format: MatrixFormat,
    ) -> Result<Operator> {
        let coefficients = self.resolve_coefficients(parameters)?;
        let mut entries = Vec::new();
        for (component, coefficient) in self.components.iter().zip(coefficients) {
            entries.extend(
                component
                    .operator
                    .triplets()
                    .into_iter()
                    .map(|(row, column, value)| (row, column, coefficient * value)),
            );
        }
        Operator::from_triplets(self.shape.0, self.shape.1, entries, format)
    }

    pub fn scaled(&self, coefficient: impl Into<Complex64>) -> Result<Self> {
        let coefficient = coefficient.into();
        let mut components = Vec::with_capacity(self.components.len());
        for component in &self.components {
            components.push(QuantumComponent {
                name: component.name.clone(),
                operator: component.operator.scaled(coefficient)?,
                default: component.default,
            });
        }
        Self::new(components)
    }

    pub fn conjugated(&self) -> Result<Self> {
        let mut components = Vec::with_capacity(self.components.len());
        for component in &self.components {
            components.push(QuantumComponent {
                name: component.name.clone(),
                operator: component.operator.conjugated()?,
                default: component.default.map(|value| value.conj()),
            });
        }
        Self::new(components)
    }

    pub fn add(&self, right: &Self) -> Result<Self> {
        if self.shape != right.shape {
            return Err(QmbedError::DimensionMismatch(
                "parameterized operators must have equal shapes".into(),
            ));
        }
        let mut components = self.components.clone();
        for right_component in &right.components {
            if let Some(left_component) = components
                .iter_mut()
                .find(|component| component.name == right_component.name)
            {
                left_component.operator = left_component.operator.add(&right_component.operator)?;
                if left_component.default != right_component.default {
                    left_component.default = None;
                }
            } else {
                components.push(right_component.clone());
            }
        }
        Self::new(components)
    }

    pub fn adjoint(&self) -> Result<Self> {
        let mut components = Vec::with_capacity(self.components.len());
        for component in &self.components {
            components.push(QuantumComponent {
                name: component.name.clone(),
                operator: component.operator.adjoint()?,
                default: component.default.map(|value| value.conj()),
            });
        }
        Self::new(components)
    }

    pub fn transpose(&self) -> Result<Self> {
        let mut components = Vec::with_capacity(self.components.len());
        for component in &self.components {
            components.push(QuantumComponent {
                name: component.name.clone(),
                operator: component.operator.transpose()?,
                default: component.default,
            });
        }
        Self::new(components)
    }
}

/// Fixed matrix-free operator with a mutable diagonal correction.
#[derive(Clone, Debug)]
pub struct QuantumLinearOperator {
    operator: Operator,
    diagonal: Vec<Complex64>,
}

impl QuantumLinearOperator {
    pub fn new(operator: Operator, diagonal: Vec<Complex64>) -> Result<Self> {
        let shape = operator.shape();
        if shape.0 != shape.1 || diagonal.len() != shape.0 {
            return Err(QmbedError::DimensionMismatch(
                "QuantumLinearOperator needs a square operator and one diagonal value per row"
                    .into(),
            ));
        }
        if diagonal
            .iter()
            .any(|value| !value.re.is_finite() || !value.im.is_finite())
        {
            return Err(QmbedError::InvalidOptions(
                "QuantumLinearOperator diagonal values must be finite".into(),
            ));
        }
        Ok(Self { operator, diagonal })
    }

    pub fn from_operator(operator: Operator) -> Result<Self> {
        let dimension = operator.shape().0;
        Self::new(operator, vec![Complex64::new(0.0, 0.0); dimension])
    }

    pub fn operator(&self) -> &Operator {
        &self.operator
    }

    pub fn diagonal_correction(&self) -> &[Complex64] {
        &self.diagonal
    }

    pub fn set_diagonal(&mut self, diagonal: Vec<Complex64>) -> Result<()> {
        if diagonal.len() != self.diagonal.len()
            || diagonal
                .iter()
                .any(|value| !value.re.is_finite() || !value.im.is_finite())
        {
            return Err(QmbedError::DimensionMismatch(
                "replacement diagonal has the wrong length or non-finite values".into(),
            ));
        }
        self.diagonal = diagonal;
        Ok(())
    }

    pub fn materialize(&self, format: MatrixFormat) -> Result<Operator> {
        let dimension = self.diagonal.len();
        let entries = self.operator.triplets().into_iter().chain(
            self.diagonal
                .iter()
                .copied()
                .enumerate()
                .filter_map(|(index, value)| {
                    (value.norm() > f64::EPSILON).then_some((index, index, value))
                }),
        );
        Operator::from_triplets(dimension, dimension, entries, format)
    }

    pub fn adjoint(&self) -> Result<Self> {
        Self::new(
            self.operator.adjoint()?,
            self.diagonal.iter().map(|value| value.conj()).collect(),
        )
    }

    pub fn transpose(&self) -> Result<Self> {
        Self::new(self.operator.transpose()?, self.diagonal.clone())
    }

    pub fn conjugated(&self) -> Result<Self> {
        Self::new(
            self.operator.conjugated()?,
            self.diagonal.iter().map(|value| value.conj()).collect(),
        )
    }

    pub fn right_apply(&self, input: &[Complex64]) -> Result<Vec<Complex64>> {
        let mut output = vec![Complex64::new(0.0, 0.0); self.shape().1];
        self.apply_transpose(input, &mut output)?;
        Ok(output)
    }
}

impl LinearOperator for QuantumLinearOperator {
    fn shape(&self) -> (usize, usize) {
        self.operator.shape()
    }

    fn format(&self) -> MatrixFormat {
        MatrixFormat::MatrixFree
    }

    fn apply(&self, input: &[Complex64], output: &mut [Complex64]) -> Result<()> {
        self.operator.apply(input, output)?;
        for ((value, diagonal), input_value) in output.iter_mut().zip(&self.diagonal).zip(input) {
            *value += *diagonal * *input_value;
        }
        Ok(())
    }

    fn apply_transpose(&self, input: &[Complex64], output: &mut [Complex64]) -> Result<()> {
        self.operator.apply_transpose(input, output)?;
        for ((value, diagonal), input_value) in output.iter_mut().zip(&self.diagonal).zip(input) {
            *value += *diagonal * *input_value;
        }
        Ok(())
    }

    fn stored_triplets(&self) -> Result<Option<Vec<(usize, usize, Complex64)>>> {
        Ok(Some(self.materialize(MatrixFormat::Csc)?.triplets()))
    }

    fn shifted_solver(&self, shift: f64) -> Result<Option<Box<dyn ShiftedLinearSolver>>> {
        self.materialize(MatrixFormat::Csc)?.shifted_solver(shift)
    }
}

pub fn commutator(left: &Operator, right: &Operator) -> Result<Operator> {
    left.product(right)?.subtract(&right.product(left)?)
}

pub fn anticommutator(left: &Operator, right: &Operator) -> Result<Operator> {
    left.product(right)?.add(&right.product(left)?)
}

/// Immutable real grid multiplying an [`ExpOp`] exponent.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ExpGrid {
    pub start: f64,
    pub stop: f64,
    pub points: usize,
    pub endpoint: bool,
}

impl ExpGrid {
    pub fn new(start: f64, stop: f64, points: usize, endpoint: bool) -> Result<Self> {
        if !start.is_finite() || !stop.is_finite() || points == 0 {
            return Err(QmbedError::InvalidOptions(
                "exponential grid endpoints must be finite and points must be positive".into(),
            ));
        }
        Ok(Self {
            start,
            stop,
            points,
            endpoint,
        })
    }

    pub fn values(&self) -> Vec<f64> {
        if self.points == 1 {
            return vec![self.start];
        }
        let intervals = if self.endpoint {
            self.points - 1
        } else {
            self.points
        };
        let step = (self.stop - self.start) / intervals as f64;
        (0..self.points)
            .map(|index| self.start + index as f64 * step)
            .collect()
    }
}

/// Lazy state iterator over an exponential grid.
pub struct ExpOpGridIter {
    operator: ExpOp,
    input: Vec<Complex64>,
    scales: std::vec::IntoIter<f64>,
}

impl Iterator for ExpOpGridIter {
    type Item = Result<Vec<Complex64>>;

    fn next(&mut self) -> Option<Self::Item> {
        self.scales.next().map(|scale| {
            let mut operator = self.operator.clone();
            operator.set_exponent(scale * self.operator.exponent)?;
            let mut output = vec![Complex64::new(0.0, 0.0); self.input.len()];
            operator.apply(&self.input, &mut output)?;
            Ok(output)
        })
    }
}

/// Matrix-free exponential `exp(exponent * A)`.
#[derive(Clone)]
pub struct ExpOp {
    operator: Arc<dyn LinearOperator>,
    exponent: Complex64,
    krylov_dimension: usize,
    tolerance: f64,
    max_substeps: usize,
    plan: crate::solve::ExpmActionPlan,
}

impl std::fmt::Debug for ExpOp {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ExpOp")
            .field("shape", &self.operator.shape())
            .field("exponent", &self.exponent)
            .field("krylov_dimension", &self.krylov_dimension)
            .field("tolerance", &self.tolerance)
            .field("max_substeps", &self.max_substeps)
            .finish()
    }
}

impl ExpOp {
    pub fn new(
        operator: Arc<dyn LinearOperator>,
        exponent: Complex64,
        krylov_dimension: usize,
        tolerance: f64,
        max_substeps: usize,
    ) -> Result<Self> {
        let shape = operator.shape();
        if shape.0 != shape.1 {
            return Err(QmbedError::DimensionMismatch(
                "ExpOp requires a square operator".into(),
            ));
        }
        if !exponent.re.is_finite()
            || !exponent.im.is_finite()
            || krylov_dimension == 0
            || !tolerance.is_finite()
            || tolerance <= 0.0
            || max_substeps == 0
        {
            return Err(QmbedError::InvalidOptions(
                "invalid ExpOp coefficient or numerical controls".into(),
            ));
        }
        let plan = crate::solve::ExpmActionPlan::new(
            operator.as_ref(),
            exponent,
            krylov_dimension,
            tolerance,
            max_substeps,
        )?;
        Ok(Self {
            operator,
            exponent,
            krylov_dimension,
            tolerance,
            max_substeps,
            plan,
        })
    }

    pub const fn exponent(&self) -> Complex64 {
        self.exponent
    }

    pub fn generator(&self) -> &Arc<dyn LinearOperator> {
        &self.operator
    }

    pub fn set_exponent(&mut self, exponent: Complex64) -> Result<()> {
        if !exponent.re.is_finite() || !exponent.im.is_finite() {
            return Err(QmbedError::InvalidOptions(
                "ExpOp coefficient must be finite".into(),
            ));
        }
        self.plan = crate::solve::ExpmActionPlan::new(
            self.operator.as_ref(),
            exponent,
            self.krylov_dimension,
            self.tolerance,
            self.max_substeps,
        )?;
        self.exponent = exponent;
        Ok(())
    }

    pub fn sandwich(&self, state: &[Complex64]) -> Result<Complex64> {
        let mut output = vec![Complex64::new(0.0, 0.0); state.len()];
        self.apply(state, &mut output)?;
        Ok(state
            .iter()
            .zip(output)
            .map(|(left, right)| left.conj() * right)
            .sum())
    }

    pub fn matrix(&self, format: MatrixFormat) -> Result<Operator> {
        let shape = self.shape();
        let mut input = vec![Complex64::new(0.0, 0.0); shape.1];
        let mut output = vec![Complex64::new(0.0, 0.0); shape.0];
        let mut triplets = Vec::new();
        for column in 0..shape.1 {
            input.fill(Complex64::new(0.0, 0.0));
            input[column] = Complex64::new(1.0, 0.0);
            self.apply(&input, &mut output)?;
            for (row, value) in output.iter().copied().enumerate() {
                if value.norm() > f64::EPSILON {
                    triplets.push((row, column, value));
                }
            }
        }
        Operator::from_triplets(shape.0, shape.1, triplets, format)
    }

    pub fn iter_grid(&self, input: &[Complex64], grid: ExpGrid) -> Result<ExpOpGridIter> {
        if input.len() != self.shape().1 {
            return Err(QmbedError::DimensionMismatch(
                "ExpOp grid input must match the operator dimension".into(),
            ));
        }
        Ok(ExpOpGridIter {
            operator: self.clone(),
            input: input.to_vec(),
            scales: grid.values().into_iter(),
        })
    }

    pub fn apply_grid(&self, input: &[Complex64], grid: ExpGrid) -> Result<Vec<Vec<Complex64>>> {
        self.iter_grid(input, grid)?.collect()
    }

    pub fn transpose(&self) -> Result<Self> {
        Self::new(
            Arc::new(TransposedLinearOperator::new(self.operator.clone())),
            self.exponent,
            self.krylov_dimension,
            self.tolerance,
            self.max_substeps,
        )
    }

    pub fn conjugated(&self) -> Result<Self> {
        Self::new(
            Arc::new(ConjugatedLinearOperator::new(self.operator.clone())),
            self.exponent.conj(),
            self.krylov_dimension,
            self.tolerance,
            self.max_substeps,
        )
    }

    pub fn adjoint(&self) -> Result<Self> {
        Self::new(
            Arc::new(AdjointLinearOperator::new(self.operator.clone())),
            self.exponent.conj(),
            self.krylov_dimension,
            self.tolerance,
            self.max_substeps,
        )
    }

    pub fn right_apply(&self, input: &[Complex64]) -> Result<Vec<Complex64>> {
        if input.len() != self.shape().0 {
            return Err(QmbedError::DimensionMismatch(
                "ExpOp right-apply input must match the operator dimension".into(),
            ));
        }
        let mut output = vec![Complex64::new(0.0, 0.0); self.shape().1];
        self.apply_transpose(input, &mut output)?;
        Ok(output)
    }
}

impl LinearOperator for ExpOp {
    fn shape(&self) -> (usize, usize) {
        self.operator.shape()
    }

    fn format(&self) -> MatrixFormat {
        MatrixFormat::MatrixFree
    }

    fn apply(&self, input: &[Complex64], output: &mut [Complex64]) -> Result<()> {
        check_apply_shape(self.shape(), input, output)?;
        let result = self.plan.apply(self.operator.as_ref(), input)?;
        output.copy_from_slice(&result);
        Ok(())
    }

    fn apply_transpose(&self, input: &[Complex64], output: &mut [Complex64]) -> Result<()> {
        check_apply_shape((self.shape().1, self.shape().0), input, output)?;
        let transpose = TransposedLinearOperator::new(self.operator.clone());
        let result = crate::solve::expm_action_complex(
            &transpose,
            input,
            self.exponent,
            self.krylov_dimension,
            self.tolerance,
            self.max_substeps,
        )?;
        output.copy_from_slice(&result);
        Ok(())
    }
}

pub fn is_exp_op(value: &dyn std::any::Any) -> bool {
    value.is::<ExpOp>()
}

pub fn is_hamiltonian(value: &dyn std::any::Any) -> bool {
    value.is::<Hamiltonian<Static>>() || value.is::<Hamiltonian<Dynamic>>()
}

pub fn is_quantum_operator(value: &dyn std::any::Any) -> bool {
    value.is::<QuantumOperator>()
}

pub fn is_quantum_linear_operator(value: &dyn std::any::Any) -> bool {
    value.is::<QuantumLinearOperator>()
}
