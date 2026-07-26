use std::collections::BTreeMap;
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;

use num_complex::Complex64;
use zip::write::SimpleFileOptions;
use zip::{CompressionMethod, ZipArchive, ZipWriter};

use crate::basis::{Basis, ErasedState};
use crate::operator::{
    LinearOperator, MatrixFormat, Operator, QuantumComponent, QuantumOperator, materialize_dense,
};
use crate::{QmbedError, Result};

pub const ARCHIVE_VERSION: u8 = 1;
/// Current portable basis-manifest archive version.
pub const BASIS_ARCHIVE_VERSION: u8 = 1;

fn archive_error(error: impl std::fmt::Display) -> QmbedError {
    QmbedError::Archive(error.to_string())
}

fn npy_header(descriptor: &str, shape: &[usize]) -> Result<Vec<u8>> {
    let shape = if shape.len() == 1 {
        format!("({},)", shape[0])
    } else {
        format!(
            "({})",
            shape
                .iter()
                .map(usize::to_string)
                .collect::<Vec<_>>()
                .join(", ")
        )
    };
    let dictionary =
        format!("{{'descr': '{descriptor}', 'fortran_order': False, 'shape': {shape}, }}");
    let padding = (16 - ((10 + dictionary.len() + 1) % 16)) % 16;
    let header_length = dictionary
        .len()
        .checked_add(padding)
        .and_then(|value| value.checked_add(1))
        .ok_or_else(|| QmbedError::Archive("NPY header length overflow".into()))?;
    let header_length = u16::try_from(header_length)
        .map_err(|_| QmbedError::Archive("NPY v1 header is too long".into()))?;
    let mut output = Vec::with_capacity(10 + usize::from(header_length));
    output.extend_from_slice(b"\x93NUMPY");
    output.extend_from_slice(&[1, 0]);
    output.extend_from_slice(&header_length.to_le_bytes());
    output.extend_from_slice(dictionary.as_bytes());
    output.extend(std::iter::repeat_n(b' ', padding));
    output.push(b'\n');
    Ok(output)
}

fn write_u8_npy(writer: &mut impl Write, values: &[u8]) -> Result<()> {
    writer
        .write_all(&npy_header("|u1", &[values.len()])?)
        .map_err(archive_error)?;
    writer.write_all(values).map_err(archive_error)
}

fn write_u64_npy(writer: &mut impl Write, values: &[u64]) -> Result<()> {
    writer
        .write_all(&npy_header("<u8", &[values.len()])?)
        .map_err(archive_error)?;
    for value in values {
        writer
            .write_all(&value.to_le_bytes())
            .map_err(archive_error)?;
    }
    Ok(())
}

fn write_complex_npy(
    writer: &mut impl Write,
    rows: usize,
    columns: usize,
    values: &[Complex64],
) -> Result<()> {
    if values.len() != rows.saturating_mul(columns) {
        return Err(QmbedError::DimensionMismatch(
            "archive matrix values do not match its shape".into(),
        ));
    }
    writer
        .write_all(&npy_header("<c16", &[rows, columns])?)
        .map_err(archive_error)?;
    for value in values {
        writer
            .write_all(&value.re.to_le_bytes())
            .and_then(|()| writer.write_all(&value.im.to_le_bytes()))
            .map_err(archive_error)?;
    }
    Ok(())
}

fn parse_npy_header(reader: &mut impl Read) -> Result<(String, Vec<usize>)> {
    let mut magic = [0_u8; 6];
    reader.read_exact(&mut magic).map_err(archive_error)?;
    if &magic != b"\x93NUMPY" {
        return Err(QmbedError::Archive("invalid NPY magic".into()));
    }
    let mut version = [0_u8; 2];
    reader.read_exact(&mut version).map_err(archive_error)?;
    let header_length = match version[0] {
        1 => {
            let mut bytes = [0_u8; 2];
            reader.read_exact(&mut bytes).map_err(archive_error)?;
            usize::from(u16::from_le_bytes(bytes))
        }
        2 | 3 => {
            let mut bytes = [0_u8; 4];
            reader.read_exact(&mut bytes).map_err(archive_error)?;
            usize::try_from(u32::from_le_bytes(bytes))
                .map_err(|_| QmbedError::Archive("NPY header is too large".into()))?
        }
        _ => return Err(QmbedError::Archive("unsupported NPY version".into())),
    };
    let mut header = vec![0_u8; header_length];
    reader.read_exact(&mut header).map_err(archive_error)?;
    let header = std::str::from_utf8(&header).map_err(archive_error)?.trim();
    if header.contains("'fortran_order': True") || header.contains("\"fortran_order\": True") {
        return Err(QmbedError::Archive(
            "Fortran-ordered NPY matrices are not supported".into(),
        ));
    }
    let descriptor_marker = if header.contains("'descr':") {
        "'descr':"
    } else {
        "\"descr\":"
    };
    let descriptor_tail = header
        .split_once(descriptor_marker)
        .ok_or_else(|| QmbedError::Archive("NPY descriptor is missing".into()))?
        .1
        .trim_start();
    let quote = descriptor_tail
        .chars()
        .next()
        .filter(|character| *character == '\'' || *character == '"')
        .ok_or_else(|| QmbedError::Archive("invalid NPY descriptor".into()))?;
    let descriptor = descriptor_tail[1..]
        .split_once(quote)
        .ok_or_else(|| QmbedError::Archive("invalid NPY descriptor".into()))?
        .0
        .to_string();
    let shape_marker = if header.contains("'shape':") {
        "'shape':"
    } else {
        "\"shape\":"
    };
    let shape_tail = header
        .split_once(shape_marker)
        .ok_or_else(|| QmbedError::Archive("NPY shape is missing".into()))?
        .1;
    let shape_text = shape_tail
        .split_once('(')
        .and_then(|(_, tail)| tail.split_once(')'))
        .map(|(shape, _)| shape)
        .ok_or_else(|| QmbedError::Archive("invalid NPY shape".into()))?;
    let shape = shape_text
        .split(',')
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(|value| value.parse::<usize>().map_err(archive_error))
        .collect::<Result<Vec<_>>>()?;
    Ok((descriptor, shape))
}

fn read_version(reader: &mut impl Read) -> Result<u8> {
    let (descriptor, shape) = parse_npy_header(reader)?;
    if descriptor != "|u1" || shape != [1] {
        return Err(QmbedError::Archive("invalid archive-version array".into()));
    }
    let mut version = [0_u8; 1];
    reader.read_exact(&mut version).map_err(archive_error)?;
    Ok(version[0])
}

fn read_u64_vector(reader: &mut impl Read) -> Result<Vec<u64>> {
    let (descriptor, shape) = parse_npy_header(reader)?;
    if !matches!(descriptor.as_str(), "<u8" | "=u8" | "|u8") || shape.len() != 1 {
        return Err(QmbedError::Archive(
            "sparse index arrays must be one-dimensional uint64 arrays".into(),
        ));
    }
    let mut values = Vec::with_capacity(shape[0]);
    for _ in 0..shape[0] {
        let mut bytes = [0_u8; 8];
        reader.read_exact(&mut bytes).map_err(archive_error)?;
        values.push(u64::from_le_bytes(bytes));
    }
    Ok(values)
}

fn read_complex_matrix(reader: &mut impl Read) -> Result<(usize, usize, Vec<Complex64>)> {
    let (descriptor, shape) = parse_npy_header(reader)?;
    if !matches!(descriptor.as_str(), "<c16" | "=c16" | "|c16") || shape.len() != 2 {
        return Err(QmbedError::Archive(
            "matrix.npy must be a C-order complex128 matrix".into(),
        ));
    }
    let count = shape[0]
        .checked_mul(shape[1])
        .ok_or_else(|| QmbedError::Archive("archive matrix size overflow".into()))?;
    let mut values = Vec::with_capacity(count);
    for _ in 0..count {
        let mut real = [0_u8; 8];
        let mut imaginary = [0_u8; 8];
        reader.read_exact(&mut real).map_err(archive_error)?;
        reader.read_exact(&mut imaginary).map_err(archive_error)?;
        values.push(Complex64::new(
            f64::from_le_bytes(real),
            f64::from_le_bytes(imaginary),
        ));
    }
    Ok((shape[0], shape[1], values))
}

fn validate_metadata(key: &str, value: &str) -> Result<()> {
    if key.is_empty()
        || key
            .chars()
            .any(|character| !(character.is_ascii_alphanumeric() || "_.-".contains(character)))
    {
        return Err(QmbedError::Archive(
            "archive metadata keys may contain only ASCII letters, digits, `_`, `-`, and `.`"
                .into(),
        ));
    }
    if value
        .chars()
        .any(|character| matches!(character, '\t' | '\r' | '\n'))
    {
        return Err(QmbedError::Archive(
            "archive metadata values may not contain tabs or newlines".into(),
        ));
    }
    Ok(())
}

/// Portable, versioned manifest of ordered physical basis states.
///
/// A basis archive stores stable state identifiers, their logical bit width,
/// public row ordering, and scalar metadata.  It intentionally does not
/// serialize executable callbacks.  Built-in frontends may store their
/// language-neutral constructor request in metadata and rebuild local
/// operator semantics after loading, while every consumer can use the state
/// manifest to align vectors, projectors, and operator archives safely.
#[derive(Clone, Debug)]
pub struct BasisArchive {
    width_bits: usize,
    states: Vec<ErasedState>,
    metadata: BTreeMap<String, String>,
}

impl BasisArchive {
    /// Construct an archive from ordered, unique physical state identifiers.
    pub fn new(width_bits: usize, states: impl IntoIterator<Item = ErasedState>) -> Result<Self> {
        let states = states.into_iter().collect::<Vec<_>>();
        if width_bits == 0 || states.iter().any(|state| state.width_bits() != width_bits) {
            return Err(QmbedError::Archive(
                "basis states must share one positive logical width".into(),
            ));
        }
        let mut unique = std::collections::HashSet::with_capacity(states.len());
        if states.iter().any(|state| !unique.insert(*state)) {
            return Err(QmbedError::Archive(
                "basis archives require unique physical states".into(),
            ));
        }
        Ok(Self {
            width_bits,
            states,
            metadata: BTreeMap::new(),
        })
    }

    /// Snapshot any integer-labelled basis without changing its public order.
    pub fn from_basis<B>(basis: &B, width_bits: usize) -> Result<Self>
    where
        B: Basis,
        B::State: std::fmt::Display,
    {
        let states = (0..basis.len())
            .map(|index| {
                let state = basis.state(index)?;
                ErasedState::from_decimal(width_bits, &state.to_string())
            })
            .collect::<Result<Vec<_>>>()?;
        Self::new(width_bits, states)
    }

    /// Logical bit width of every archived state.
    pub const fn width_bits(&self) -> usize {
        self.width_bits
    }

    /// Ordered physical states, one per public basis row.
    pub fn states(&self) -> &[ErasedState] {
        &self.states
    }

    /// Attach a small language-neutral reconstruction hint or provenance item.
    pub fn insert_metadata(
        &mut self,
        key: impl Into<String>,
        value: impl Into<String>,
    ) -> Result<()> {
        let key = key.into();
        let value = value.into();
        validate_metadata(&key, &value)?;
        self.metadata.insert(key, value);
        Ok(())
    }

    /// Iterate over archived scalar metadata.
    pub fn metadata(&self) -> impl Iterator<Item = (&str, &str)> {
        self.metadata
            .iter()
            .map(|(key, value)| (key.as_str(), value.as_str()))
    }

    /// Read one archived metadata value.
    pub fn metadata_value(&self, key: &str) -> Option<&str> {
        self.metadata.get(key).map(String::as_str)
    }
}

/// Save a pickle-free basis manifest preserving logical width and row order.
pub fn save_basis_zip(path: impl AsRef<Path>, basis: &BasisArchive) -> Result<()> {
    let file = File::create(path).map_err(archive_error)?;
    let mut archive = ZipWriter::new(file);
    let options = SimpleFileOptions::default().compression_method(CompressionMethod::Deflated);
    archive
        .start_file("basis_archive_version.npy", options)
        .map_err(archive_error)?;
    write_u8_npy(&mut archive, &[BASIS_ARCHIVE_VERSION])?;
    archive
        .start_file("basis_manifest.tsv", options)
        .map_err(archive_error)?;
    archive
        .write_all(format!("width_bits\t{}\n", basis.width_bits).as_bytes())
        .map_err(archive_error)?;
    for state in &basis.states {
        archive
            .write_all(format!("state\t{}\n", state.to_decimal()).as_bytes())
            .map_err(archive_error)?;
    }
    archive
        .start_file("metadata.tsv", options)
        .map_err(archive_error)?;
    for (key, value) in &basis.metadata {
        archive
            .write_all(format!("{key}\t{value}\n").as_bytes())
            .map_err(archive_error)?;
    }
    archive.finish().map_err(archive_error)?;
    Ok(())
}

/// Load and validate a portable basis manifest.
pub fn load_basis_zip(path: impl AsRef<Path>) -> Result<BasisArchive> {
    let file = File::open(path).map_err(archive_error)?;
    let mut archive = ZipArchive::new(file).map_err(archive_error)?;
    {
        let mut version = archive
            .by_name("basis_archive_version.npy")
            .map_err(archive_error)?;
        let version = read_version(&mut version)?;
        if version != BASIS_ARCHIVE_VERSION {
            return Err(QmbedError::Archive(format!(
                "unsupported basis archive version {version}"
            )));
        }
    }
    let manifest = {
        let mut entry = archive
            .by_name("basis_manifest.tsv")
            .map_err(archive_error)?;
        let mut text = String::new();
        entry.read_to_string(&mut text).map_err(archive_error)?;
        text
    };
    let mut lines = manifest.lines();
    let width_bits = lines
        .next()
        .and_then(|line| line.strip_prefix("width_bits\t"))
        .ok_or_else(|| QmbedError::Archive("basis width is missing".into()))?
        .parse::<usize>()
        .map_err(archive_error)?;
    let states = lines
        .map(|line| {
            let value = line
                .strip_prefix("state\t")
                .ok_or_else(|| QmbedError::Archive("invalid basis state row".into()))?;
            ErasedState::from_decimal(width_bits, value)
        })
        .collect::<Result<Vec<_>>>()?;
    let mut basis = BasisArchive::new(width_bits, states)?;
    let has_metadata = archive.file_names().any(|name| name == "metadata.tsv");
    if has_metadata {
        let mut entry = archive.by_name("metadata.tsv").map_err(archive_error)?;
        let mut text = String::new();
        entry.read_to_string(&mut text).map_err(archive_error)?;
        for line in text.lines() {
            let (key, value) = line
                .split_once('\t')
                .ok_or_else(|| QmbedError::Archive("invalid archive metadata row".into()))?;
            basis.insert_metadata(key, value)?;
        }
    }
    Ok(basis)
}

/// Save a versioned, pickle-free NPZ archive readable as `np.load(path)["matrix"]`.
pub fn save_operator_npz(
    operator: &(impl LinearOperator + ?Sized),
    path: impl AsRef<Path>,
) -> Result<()> {
    let shape = operator.shape();
    let values = materialize_dense(operator)?;
    let file = File::create(path).map_err(archive_error)?;
    let mut archive = ZipWriter::new(file);
    let options = SimpleFileOptions::default().compression_method(CompressionMethod::Deflated);
    archive
        .start_file("archive_version.npy", options)
        .map_err(archive_error)?;
    write_u8_npy(&mut archive, &[ARCHIVE_VERSION])?;
    archive
        .start_file("matrix.npy", options)
        .map_err(archive_error)?;
    write_complex_npy(&mut archive, shape.0, shape.1, &values)?;
    archive.finish().map_err(archive_error)?;
    Ok(())
}

/// Load a pickle-free NPZ produced by Rust or NumPy `savez`/`savez_compressed`.
pub fn load_operator_npz(path: impl AsRef<Path>, format: MatrixFormat) -> Result<Operator> {
    let file = File::open(path).map_err(archive_error)?;
    let mut archive = ZipArchive::new(file).map_err(archive_error)?;
    if let Ok(mut version_entry) = archive.by_name("archive_version.npy") {
        let version = read_version(&mut version_entry)?;
        if version != ARCHIVE_VERSION {
            return Err(QmbedError::Archive(format!(
                "unsupported operator archive version {version}"
            )));
        }
    }
    let mut matrix = archive.by_name("matrix.npy").map_err(archive_error)?;
    let (rows, columns, values) = read_complex_matrix(&mut matrix)?;
    Operator::from_dense(rows, columns, values)?.converted(format)
}

#[derive(Clone, Debug)]
pub struct OperatorArchiveEntry {
    pub operator: Operator,
    pub default: Option<Complex64>,
}

#[derive(Clone, Debug, Default)]
pub struct OperatorArchive {
    entries: BTreeMap<String, OperatorArchiveEntry>,
    metadata: BTreeMap<String, String>,
}

impl OperatorArchive {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(
        &mut self,
        name: impl Into<String>,
        operator: Operator,
        default: Option<Complex64>,
    ) -> Result<()> {
        let name = name.into();
        if name.is_empty()
            || name
                .chars()
                .any(|character| !(character.is_ascii_alphanumeric() || "_.-".contains(character)))
        {
            return Err(QmbedError::Archive(
                "archive entry names may contain only ASCII letters, digits, `_`, `-`, and `.`"
                    .into(),
            ));
        }
        if self
            .entries
            .insert(name.clone(), OperatorArchiveEntry { operator, default })
            .is_some()
        {
            return Err(QmbedError::Archive(format!(
                "duplicate archive entry {name:?}"
            )));
        }
        Ok(())
    }

    pub fn get(&self, name: &str) -> Option<&OperatorArchiveEntry> {
        self.entries.get(name)
    }

    pub fn iter(&self) -> impl Iterator<Item = (&str, &OperatorArchiveEntry)> {
        self.entries
            .iter()
            .map(|(name, entry)| (name.as_str(), entry))
    }

    /// Attach small, language-neutral metadata to the archive.
    ///
    /// Metadata is deliberately scalar text rather than executable
    /// serialization. This keeps archives safe to inspect and lets frontends
    /// preserve contracts such as declared scalar dtype or a basis-schema
    /// version without embedding pickled language objects.
    pub fn insert_metadata(
        &mut self,
        key: impl Into<String>,
        value: impl Into<String>,
    ) -> Result<()> {
        let key = key.into();
        let value = value.into();
        validate_metadata(&key, &value)?;
        self.metadata.insert(key, value);
        Ok(())
    }

    pub fn metadata(&self) -> impl Iterator<Item = (&str, &str)> {
        self.metadata
            .iter()
            .map(|(key, value)| (key.as_str(), value.as_str()))
    }

    pub fn metadata_value(&self, key: &str) -> Option<&str> {
        self.metadata.get(key).map(String::as_str)
    }

    pub fn from_quantum_operator(operator: &QuantumOperator) -> Result<Self> {
        let mut archive = Self::new();
        for component in operator.components() {
            archive.insert(
                component.name(),
                component.operator().clone(),
                component.default(),
            )?;
        }
        Ok(archive)
    }

    pub fn into_quantum_operator(self) -> Result<QuantumOperator> {
        QuantumOperator::new(self.entries.into_iter().map(|(name, entry)| {
            if let Some(default) = entry.default {
                QuantumComponent::with_default(name, entry.operator, default)
            } else {
                QuantumComponent::required(name, entry.operator)
            }
        }))
    }
}

fn format_name(format: MatrixFormat) -> &'static str {
    match format {
        MatrixFormat::Dense => "dense",
        MatrixFormat::Csc => "csc",
        MatrixFormat::Csr => "csr",
        MatrixFormat::Dia => "dia",
        MatrixFormat::MatrixFree => "matrix_free",
    }
}

fn parse_format(value: &str) -> Result<MatrixFormat> {
    match value {
        "dense" => Ok(MatrixFormat::Dense),
        "csc" => Ok(MatrixFormat::Csc),
        "csr" => Ok(MatrixFormat::Csr),
        "dia" => Ok(MatrixFormat::Dia),
        "matrix_free" => Ok(MatrixFormat::MatrixFree),
        _ => Err(QmbedError::Archive(format!(
            "unknown archived matrix format {value:?}"
        ))),
    }
}

/// Save named dense/sparse components without executing serialization callbacks.
pub fn save_zip(path: impl AsRef<Path>, entries: &OperatorArchive) -> Result<()> {
    if entries.entries.is_empty() {
        return Err(QmbedError::Archive(
            "an operator archive must contain at least one entry".into(),
        ));
    }
    let file = File::create(path).map_err(archive_error)?;
    let mut archive = ZipWriter::new(file);
    let options = SimpleFileOptions::default().compression_method(CompressionMethod::Deflated);
    archive
        .start_file("archive_version.npy", options)
        .map_err(archive_error)?;
    write_u8_npy(&mut archive, &[ARCHIVE_VERSION])?;

    let mut manifest = String::new();
    for (index, (name, entry)) in entries.entries.iter().enumerate() {
        let shape = entry.operator.shape();
        let (present, real, imaginary) = entry
            .default
            .map_or((0, 0.0, 0.0), |value| (1, value.re, value.im));
        manifest.push_str(&format!(
            "{name}\t{}\t{}\t{}\t{present}\t{real:.17e}\t{imaginary:.17e}\n",
            format_name(entry.operator.format()),
            shape.0,
            shape.1,
        ));
        if entry.operator.format() == MatrixFormat::Dense {
            archive
                .start_file(format!("entries/{index}/matrix.npy"), options)
                .map_err(archive_error)?;
            write_complex_npy(&mut archive, shape.0, shape.1, &entry.operator.to_dense())?;
        } else {
            let triplets = entry.operator.triplets();
            let rows = triplets
                .iter()
                .map(|(row, _, _)| u64::try_from(*row).map_err(archive_error))
                .collect::<Result<Vec<_>>>()?;
            let columns = triplets
                .iter()
                .map(|(_, column, _)| u64::try_from(*column).map_err(archive_error))
                .collect::<Result<Vec<_>>>()?;
            let values: Vec<_> = triplets.iter().map(|(_, _, value)| *value).collect();
            archive
                .start_file(format!("entries/{index}/rows.npy"), options)
                .map_err(archive_error)?;
            write_u64_npy(&mut archive, &rows)?;
            archive
                .start_file(format!("entries/{index}/columns.npy"), options)
                .map_err(archive_error)?;
            write_u64_npy(&mut archive, &columns)?;
            archive
                .start_file(format!("entries/{index}/values.npy"), options)
                .map_err(archive_error)?;
            write_complex_npy(&mut archive, values.len(), 1, &values)?;
        }
    }
    archive
        .start_file("manifest.tsv", options)
        .map_err(archive_error)?;
    archive
        .write_all(manifest.as_bytes())
        .map_err(archive_error)?;
    archive
        .start_file("metadata.tsv", options)
        .map_err(archive_error)?;
    for (key, value) in &entries.metadata {
        archive
            .write_all(format!("{key}\t{value}\n").as_bytes())
            .map_err(archive_error)?;
    }
    archive.finish().map_err(archive_error)?;
    Ok(())
}

/// Load a named operator archive while preserving every requested storage format.
pub fn load_zip(path: impl AsRef<Path>) -> Result<OperatorArchive> {
    let file = File::open(path).map_err(archive_error)?;
    let mut archive = ZipArchive::new(file).map_err(archive_error)?;
    {
        let mut version_entry = archive
            .by_name("archive_version.npy")
            .map_err(archive_error)?;
        let version = read_version(&mut version_entry)?;
        if version != ARCHIVE_VERSION {
            return Err(QmbedError::Archive(format!(
                "unsupported operator archive version {version}"
            )));
        }
    }
    let manifest = {
        let mut entry = archive.by_name("manifest.tsv").map_err(archive_error)?;
        let mut text = String::new();
        entry.read_to_string(&mut text).map_err(archive_error)?;
        text
    };
    let mut result = OperatorArchive::new();
    let has_metadata = archive.file_names().any(|name| name == "metadata.tsv");
    if has_metadata {
        let mut entry = archive.by_name("metadata.tsv").map_err(archive_error)?;
        let mut text = String::new();
        entry.read_to_string(&mut text).map_err(archive_error)?;
        for line in text.lines() {
            let (key, value) = line
                .split_once('\t')
                .ok_or_else(|| QmbedError::Archive("invalid archive metadata row".into()))?;
            result.insert_metadata(key, value)?;
        }
    }
    for (index, line) in manifest.lines().enumerate() {
        let fields: Vec<_> = line.split('\t').collect();
        if fields.len() != 7 {
            return Err(QmbedError::Archive("invalid archive manifest row".into()));
        }
        let format = parse_format(fields[1])?;
        let rows = fields[2].parse::<usize>().map_err(archive_error)?;
        let columns = fields[3].parse::<usize>().map_err(archive_error)?;
        let default = match fields[4] {
            "0" => None,
            "1" => Some(Complex64::new(
                fields[5].parse::<f64>().map_err(archive_error)?,
                fields[6].parse::<f64>().map_err(archive_error)?,
            )),
            _ => return Err(QmbedError::Archive("invalid default marker".into())),
        };
        let operator = if format == MatrixFormat::Dense {
            let mut matrix = archive
                .by_name(&format!("entries/{index}/matrix.npy"))
                .map_err(archive_error)?;
            let (actual_rows, actual_columns, values) = read_complex_matrix(&mut matrix)?;
            if (actual_rows, actual_columns) != (rows, columns) {
                return Err(QmbedError::Archive(
                    "dense entry shape disagrees with the manifest".into(),
                ));
            }
            Operator::from_dense(rows, columns, values)?
        } else {
            let row_indices = {
                let mut entry = archive
                    .by_name(&format!("entries/{index}/rows.npy"))
                    .map_err(archive_error)?;
                read_u64_vector(&mut entry)?
            };
            let column_indices = {
                let mut entry = archive
                    .by_name(&format!("entries/{index}/columns.npy"))
                    .map_err(archive_error)?;
                read_u64_vector(&mut entry)?
            };
            let values = {
                let mut entry = archive
                    .by_name(&format!("entries/{index}/values.npy"))
                    .map_err(archive_error)?;
                let (value_rows, value_columns, values) = read_complex_matrix(&mut entry)?;
                if value_columns != 1 || value_rows != row_indices.len() {
                    return Err(QmbedError::Archive(
                        "sparse values shape disagrees with sparse indices".into(),
                    ));
                }
                values
            };
            if row_indices.len() != column_indices.len() || row_indices.len() != values.len() {
                return Err(QmbedError::Archive(
                    "sparse archive arrays have unequal lengths".into(),
                ));
            }
            let triplets = row_indices
                .into_iter()
                .zip(column_indices)
                .zip(values)
                .map(|((row, column), value)| {
                    Ok((
                        usize::try_from(row).map_err(archive_error)?,
                        usize::try_from(column).map_err(archive_error)?,
                        value,
                    ))
                })
                .collect::<Result<Vec<_>>>()?;
            Operator::from_triplets(rows, columns, triplets, format)?
        };
        result.insert(fields[0], operator, default)?;
    }
    Ok(result)
}
