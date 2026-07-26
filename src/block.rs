use std::sync::Arc;

use num_complex::Complex64;

use crate::operator::{
    LinearOperator, MatrixFormat, Operator, TimeDependentOperator, TimeOperator, check_apply_shape,
};
use crate::{QmbedError, Result};

fn block_offsets(shapes: impl IntoIterator<Item = (usize, usize)>) -> Result<Vec<usize>> {
    let mut offsets = vec![0_usize];
    for shape in shapes {
        if shape.0 != shape.1 {
            return Err(QmbedError::DimensionMismatch(
                "block-diagonal operators require square blocks".into(),
            ));
        }
        offsets.push(
            offsets
                .last()
                .copied()
                .unwrap_or_default()
                .checked_add(shape.0)
                .ok_or_else(|| QmbedError::UnsupportedBackend("block dimension overflow".into()))?,
        );
    }
    if offsets.len() == 1 {
        return Err(QmbedError::InvalidOptions(
            "at least one operator block is required".into(),
        ));
    }
    Ok(offsets)
}

#[derive(Clone)]
struct ProjectionLayout {
    projectors: Vec<Arc<dyn LinearOperator>>,
    offsets: Vec<usize>,
    full_dimension: usize,
}

impl ProjectionLayout {
    fn new(
        block_shapes: impl IntoIterator<Item = (usize, usize)>,
        projectors: impl IntoIterator<Item = Arc<dyn LinearOperator>>,
        tolerance: f64,
    ) -> Result<Self> {
        if !tolerance.is_finite() || tolerance <= 0.0 {
            return Err(QmbedError::InvalidOptions(
                "projector-isometry tolerance must be positive and finite".into(),
            ));
        }
        let block_shapes: Vec<_> = block_shapes.into_iter().collect();
        let offsets = block_offsets(block_shapes.iter().copied())?;
        let projectors: Vec<_> = projectors.into_iter().collect();
        if projectors.len() != block_shapes.len() {
            return Err(QmbedError::DimensionMismatch(format!(
                "{} blocks require the same number of projectors, got {}",
                block_shapes.len(),
                projectors.len()
            )));
        }
        let full_dimension = projectors
            .first()
            .map(|projector| projector.shape().0)
            .unwrap_or_default();
        if full_dimension == 0 {
            return Err(QmbedError::DimensionMismatch(
                "projectors must have a positive parent-space dimension".into(),
            ));
        }
        for (index, (block_shape, projector)) in
            block_shapes.iter().zip(projectors.iter()).enumerate()
        {
            let projector_shape = projector.shape();
            if projector_shape != (full_dimension, block_shape.0) {
                return Err(QmbedError::DimensionMismatch(format!(
                    "projector {index} has shape {projector_shape:?}, expected ({full_dimension}, {})",
                    block_shape.0
                )));
            }
        }

        // Validate the combined map P = [P_0 ... P_n] as one isometry.
        // This checks both normalization inside a block and mutual
        // orthogonality between different sectors without requiring stored
        // projector matrices.
        for (source_index, source) in projectors.iter().enumerate() {
            let source_dimension = block_shapes[source_index].0;
            let mut coordinate = vec![Complex64::new(0.0, 0.0); source_dimension];
            let mut parent = vec![Complex64::new(0.0, 0.0); full_dimension];
            for source_column in 0..source_dimension {
                coordinate.fill(Complex64::new(0.0, 0.0));
                coordinate[source_column] = Complex64::new(1.0, 0.0);
                source.apply(&coordinate, &mut parent)?;
                for (target_index, target) in projectors.iter().enumerate() {
                    let target_dimension = block_shapes[target_index].0;
                    let mut overlap = vec![Complex64::new(0.0, 0.0); target_dimension];
                    target.apply_adjoint(&parent, &mut overlap)?;
                    for (target_column, value) in overlap.into_iter().enumerate() {
                        let expected =
                            if source_index == target_index && source_column == target_column {
                                Complex64::new(1.0, 0.0)
                            } else {
                                Complex64::new(0.0, 0.0)
                            };
                        if (value - expected).norm() > tolerance {
                            return Err(QmbedError::InvalidOptions(format!(
                                "combined block projector is not an isometry within tolerance {tolerance}: \
                                 block {target_index} column {target_column} overlaps block \
                                 {source_index} column {source_column} by {value}"
                            )));
                        }
                    }
                }
            }
        }

        Ok(Self {
            projectors,
            offsets,
            full_dimension,
        })
    }

    fn block_dimension(&self) -> usize {
        self.offsets.last().copied().unwrap_or_default()
    }

    fn project(&self, parent: &[Complex64], blocks: &mut [Complex64]) -> Result<()> {
        if parent.len() != self.full_dimension || blocks.len() != self.block_dimension() {
            return Err(QmbedError::DimensionMismatch(format!(
                "projecting from parent dimension {} requires input length {} and block output length {}, got {} and {}",
                self.full_dimension,
                self.full_dimension,
                self.block_dimension(),
                parent.len(),
                blocks.len()
            )));
        }
        for (index, projector) in self.projectors.iter().enumerate() {
            let start = self.offsets[index];
            let end = self.offsets[index + 1];
            projector.apply_adjoint(parent, &mut blocks[start..end])?;
        }
        Ok(())
    }

    fn lift(&self, blocks: &[Complex64], parent: &mut [Complex64]) -> Result<()> {
        if blocks.len() != self.block_dimension() || parent.len() != self.full_dimension {
            return Err(QmbedError::DimensionMismatch(format!(
                "lifting from block dimension {} requires input length {} and parent output length {}, got {} and {}",
                self.block_dimension(),
                self.block_dimension(),
                self.full_dimension,
                blocks.len(),
                parent.len()
            )));
        }
        parent.fill(Complex64::new(0.0, 0.0));
        let mut contribution = vec![Complex64::new(0.0, 0.0); self.full_dimension];
        for (index, projector) in self.projectors.iter().enumerate() {
            let start = self.offsets[index];
            let end = self.offsets[index + 1];
            projector.apply(&blocks[start..end], &mut contribution)?;
            for (value, added) in parent.iter_mut().zip(contribution.iter()) {
                *value += *added;
            }
        }
        Ok(())
    }

    fn completeness_residual(&self) -> Result<f64> {
        let mut parent = vec![Complex64::new(0.0, 0.0); self.full_dimension];
        let mut blocks = vec![Complex64::new(0.0, 0.0); self.block_dimension()];
        let mut reconstructed = vec![Complex64::new(0.0, 0.0); self.full_dimension];
        let mut maximum = 0.0_f64;
        for column in 0..self.full_dimension {
            parent.fill(Complex64::new(0.0, 0.0));
            parent[column] = Complex64::new(1.0, 0.0);
            self.project(&parent, &mut blocks)?;
            self.lift(&blocks, &mut reconstructed)?;
            for (row, value) in reconstructed.iter().copied().enumerate() {
                let expected = if row == column {
                    Complex64::new(1.0, 0.0)
                } else {
                    Complex64::new(0.0, 0.0)
                };
                maximum = maximum.max((value - expected).norm());
            }
        }
        Ok(maximum)
    }
}

/// Delayed matrix-free direct sum of static blocks.
pub struct BlockOps {
    blocks: Vec<Arc<dyn LinearOperator>>,
    offsets: Vec<usize>,
}

impl BlockOps {
    pub fn new(blocks: impl IntoIterator<Item = Arc<dyn LinearOperator>>) -> Result<Self> {
        let blocks: Vec<_> = blocks.into_iter().collect();
        let offsets = block_offsets(blocks.iter().map(|block| block.shape()))?;
        Ok(Self { blocks, offsets })
    }

    pub fn blocks(&self) -> usize {
        self.blocks.len()
    }

    pub fn materialize(&self, format: MatrixFormat) -> Result<Operator> {
        block_diag_hamiltonian(self.blocks.iter().cloned(), format)
    }

    pub fn push(&mut self, block: Arc<dyn LinearOperator>) -> Result<()> {
        if block.shape().0 != block.shape().1 {
            return Err(QmbedError::DimensionMismatch(
                "block-diagonal operators require square blocks".into(),
            ));
        }
        let next = self
            .offsets
            .last()
            .copied()
            .unwrap_or_default()
            .checked_add(block.shape().0)
            .ok_or_else(|| QmbedError::UnsupportedBackend("block dimension overflow".into()))?;
        self.blocks.push(block);
        self.offsets.push(next);
        Ok(())
    }
}

impl LinearOperator for BlockOps {
    fn shape(&self) -> (usize, usize) {
        let dimension = self.offsets.last().copied().unwrap_or_default();
        (dimension, dimension)
    }

    fn format(&self) -> MatrixFormat {
        MatrixFormat::MatrixFree
    }

    fn apply(&self, input: &[Complex64], output: &mut [Complex64]) -> Result<()> {
        check_apply_shape(self.shape(), input, output)?;
        output.fill(Complex64::new(0.0, 0.0));
        for (index, block) in self.blocks.iter().enumerate() {
            let start = self.offsets[index];
            let end = self.offsets[index + 1];
            block.apply(&input[start..end], &mut output[start..end])?;
        }
        Ok(())
    }

    fn stored_triplets(&self) -> Result<Option<Vec<(usize, usize, Complex64)>>> {
        let mut entries = Vec::new();
        for (block_index, block) in self.blocks.iter().enumerate() {
            let Some(block_entries) = block.stored_triplets()? else {
                return Ok(None);
            };
            let offset = self.offsets[block_index];
            entries.extend(
                block_entries
                    .into_iter()
                    .map(|(row, column, value)| (offset + row, offset + column, value)),
            );
        }
        Ok(Some(entries))
    }
}

/// A direct sum of sector operators acting in a shared parent Hilbert space.
///
/// Each projector maps one sector coordinate vector into the parent space.
/// The combined projector is validated as an isometry, so this operator
/// represents `P (⊕ H_sector) P†` without materializing either the direct sum
/// or the parent-space matrix.
#[derive(Clone)]
pub struct ProjectedBlockOps {
    blocks: Vec<Arc<dyn LinearOperator>>,
    projection: ProjectionLayout,
}

impl ProjectedBlockOps {
    pub fn new(
        blocks: impl IntoIterator<Item = Arc<dyn LinearOperator>>,
        projectors: impl IntoIterator<Item = Arc<dyn LinearOperator>>,
        tolerance: f64,
    ) -> Result<Self> {
        let blocks: Vec<_> = blocks.into_iter().collect();
        let projection = ProjectionLayout::new(
            blocks.iter().map(|block| block.shape()),
            projectors,
            tolerance,
        )?;
        Ok(Self { blocks, projection })
    }

    pub fn blocks(&self) -> usize {
        self.blocks.len()
    }

    pub fn full_dimension(&self) -> usize {
        self.projection.full_dimension
    }

    pub fn block_dimension(&self) -> usize {
        self.projection.block_dimension()
    }

    pub fn project(&self, parent: &[Complex64], blocks: &mut [Complex64]) -> Result<()> {
        self.projection.project(parent, blocks)
    }

    pub fn lift(&self, blocks: &[Complex64], parent: &mut [Complex64]) -> Result<()> {
        self.projection.lift(blocks, parent)
    }

    /// Maximum entrywise residual of `P P† - I` in the parent basis.
    ///
    /// An isometric projector collection need not be complete: selected
    /// sectors can intentionally span only a subspace. Callers which require a
    /// full decomposition can enforce their own tolerance on this residual.
    pub fn completeness_residual(&self) -> Result<f64> {
        self.projection.completeness_residual()
    }

    pub fn materialize(&self, format: MatrixFormat) -> Result<Operator> {
        Operator::from_triplets(
            self.shape().0,
            self.shape().1,
            streamed_triplets(self)?,
            format,
        )
    }
}

impl LinearOperator for ProjectedBlockOps {
    fn shape(&self) -> (usize, usize) {
        (
            self.projection.full_dimension,
            self.projection.full_dimension,
        )
    }

    fn format(&self) -> MatrixFormat {
        MatrixFormat::MatrixFree
    }

    fn apply(&self, input: &[Complex64], output: &mut [Complex64]) -> Result<()> {
        check_apply_shape(self.shape(), input, output)?;
        let mut block_input = vec![Complex64::new(0.0, 0.0); self.projection.block_dimension()];
        let mut block_output = block_input.clone();
        self.projection.project(input, &mut block_input)?;
        for (index, block) in self.blocks.iter().enumerate() {
            let start = self.projection.offsets[index];
            let end = self.projection.offsets[index + 1];
            block.apply(&block_input[start..end], &mut block_output[start..end])?;
        }
        self.projection.lift(&block_output, output)
    }
}

fn streamed_triplets(
    operator: &(impl LinearOperator + ?Sized),
) -> Result<Vec<(usize, usize, Complex64)>> {
    if let Some(entries) = operator.stored_triplets()? {
        return Ok(entries);
    }
    let shape = operator.shape();
    let mut input = vec![Complex64::new(0.0, 0.0); shape.1];
    let mut output = vec![Complex64::new(0.0, 0.0); shape.0];
    let mut entries = Vec::new();
    for column in 0..shape.1 {
        input.fill(Complex64::new(0.0, 0.0));
        input[column] = Complex64::new(1.0, 0.0);
        operator.apply(&input, &mut output)?;
        for (row, value) in output.iter().copied().enumerate() {
            if value.norm() > f64::EPSILON {
                entries.push((row, column, value));
            }
        }
    }
    Ok(entries)
}

pub fn block_diag_hamiltonian(
    blocks: impl IntoIterator<Item = Arc<dyn LinearOperator>>,
    format: MatrixFormat,
) -> Result<Operator> {
    let blocks: Vec<_> = blocks.into_iter().collect();
    let offsets = block_offsets(blocks.iter().map(|block| block.shape()))?;
    let dimension = offsets.last().copied().unwrap_or_default();
    let mut triplets = Vec::new();
    for (block_index, block) in blocks.iter().enumerate() {
        let offset = offsets[block_index];
        triplets.extend(
            streamed_triplets(block.as_ref())?
                .into_iter()
                .map(|(row, column, value)| (offset + row, offset + column, value)),
        );
    }
    Operator::from_triplets(dimension, dimension, triplets, format)
}

/// Delayed direct sum of explicitly time-dependent blocks.
pub struct DynamicBlockOps {
    blocks: Vec<Arc<dyn TimeDependentOperator>>,
    offsets: Vec<usize>,
}

impl DynamicBlockOps {
    pub fn new(blocks: impl IntoIterator<Item = Arc<dyn TimeDependentOperator>>) -> Result<Self> {
        let blocks: Vec<_> = blocks.into_iter().collect();
        let offsets = block_offsets(blocks.iter().map(|block| block.shape()))?;
        Ok(Self { blocks, offsets })
    }

    pub fn blocks(&self) -> usize {
        self.blocks.len()
    }

    pub fn push(&mut self, block: Arc<dyn TimeDependentOperator>) -> Result<()> {
        if block.shape().0 != block.shape().1 {
            return Err(QmbedError::DimensionMismatch(
                "block-diagonal operators require square blocks".into(),
            ));
        }
        let next = self
            .offsets
            .last()
            .copied()
            .unwrap_or_default()
            .checked_add(block.shape().0)
            .ok_or_else(|| QmbedError::UnsupportedBackend("block dimension overflow".into()))?;
        self.blocks.push(block);
        self.offsets.push(next);
        Ok(())
    }

    pub fn materialize(&self, time: f64, format: MatrixFormat) -> Result<Operator> {
        if !time.is_finite() {
            return Err(QmbedError::InvalidOptions(
                "dynamic block materialization time must be finite".into(),
            ));
        }
        let dimension = self.offsets.last().copied().unwrap_or_default();
        let mut entries = Vec::new();
        for (block_index, block) in self.blocks.iter().enumerate() {
            let block_dimension = block.shape().0;
            let offset = self.offsets[block_index];
            let mut input = vec![Complex64::new(0.0, 0.0); block_dimension];
            let mut output = vec![Complex64::new(0.0, 0.0); block_dimension];
            for column in 0..block_dimension {
                input.fill(Complex64::new(0.0, 0.0));
                input[column] = Complex64::new(1.0, 0.0);
                block.apply_at(time, &input, &mut output)?;
                for (row, value) in output.iter().copied().enumerate() {
                    if value.norm() > f64::EPSILON {
                        entries.push((offset + row, offset + column, value));
                    }
                }
            }
        }
        Operator::from_triplets(dimension, dimension, entries, format)
    }
}

impl TimeDependentOperator for DynamicBlockOps {
    fn shape(&self) -> (usize, usize) {
        let dimension = self.offsets.last().copied().unwrap_or_default();
        (dimension, dimension)
    }

    fn apply_at(&self, time: f64, input: &[Complex64], output: &mut [Complex64]) -> Result<()> {
        check_apply_shape(self.shape(), input, output)?;
        output.fill(Complex64::new(0.0, 0.0));
        for (index, block) in self.blocks.iter().enumerate() {
            let start = self.offsets[index];
            let end = self.offsets[index + 1];
            block.apply_at(time, &input[start..end], &mut output[start..end])?;
        }
        Ok(())
    }
}

/// Time-dependent sector operators lifted into a shared parent Hilbert space.
///
/// This is the dynamic analogue of [`ProjectedBlockOps`] and evaluates
/// `P (⊕ H_sector(t)) P†` matrix-free at every requested time.
#[derive(Clone)]
pub struct ProjectedDynamicBlockOps {
    blocks: Vec<Arc<dyn TimeDependentOperator>>,
    projection: ProjectionLayout,
}

impl ProjectedDynamicBlockOps {
    pub fn new(
        blocks: impl IntoIterator<Item = Arc<dyn TimeDependentOperator>>,
        projectors: impl IntoIterator<Item = Arc<dyn LinearOperator>>,
        tolerance: f64,
    ) -> Result<Self> {
        let blocks: Vec<_> = blocks.into_iter().collect();
        let projection = ProjectionLayout::new(
            blocks.iter().map(|block| block.shape()),
            projectors,
            tolerance,
        )?;
        Ok(Self { blocks, projection })
    }

    pub fn blocks(&self) -> usize {
        self.blocks.len()
    }

    pub fn full_dimension(&self) -> usize {
        self.projection.full_dimension
    }

    pub fn block_dimension(&self) -> usize {
        self.projection.block_dimension()
    }

    pub fn project(&self, parent: &[Complex64], blocks: &mut [Complex64]) -> Result<()> {
        self.projection.project(parent, blocks)
    }

    pub fn lift(&self, blocks: &[Complex64], parent: &mut [Complex64]) -> Result<()> {
        self.projection.lift(blocks, parent)
    }

    pub fn completeness_residual(&self) -> Result<f64> {
        self.projection.completeness_residual()
    }

    pub fn materialize(&self, time: f64, format: MatrixFormat) -> Result<Operator> {
        TimeOperator::from_operator(Arc::new(self.clone())).evaluate(time, format)
    }
}

impl TimeDependentOperator for ProjectedDynamicBlockOps {
    fn shape(&self) -> (usize, usize) {
        (
            self.projection.full_dimension,
            self.projection.full_dimension,
        )
    }

    fn apply_at(&self, time: f64, input: &[Complex64], output: &mut [Complex64]) -> Result<()> {
        if !time.is_finite() {
            return Err(QmbedError::InvalidOptions(
                "dynamic block evaluation time must be finite".into(),
            ));
        }
        check_apply_shape(self.shape(), input, output)?;
        let mut block_input = vec![Complex64::new(0.0, 0.0); self.projection.block_dimension()];
        let mut block_output = block_input.clone();
        self.projection.project(input, &mut block_input)?;
        for (index, block) in self.blocks.iter().enumerate() {
            let start = self.projection.offsets[index];
            let end = self.projection.offsets[index + 1];
            block.apply_at(
                time,
                &block_input[start..end],
                &mut block_output[start..end],
            )?;
        }
        self.projection.lift(&block_output, output)
    }
}
