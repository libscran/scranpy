from dataclasses import dataclass
from typing import Optional, Sequence

from biocframe import BiocFrame
from numpy import bool_, float64, ndarray, zeros, uint8

from .. import cpphelpers as lib
from ..utils import factorize

__author__ = "ltla, jkanche"
__copyright__ = "ltla, jkanche"
__license__ = "MIT"


@dataclass
class CreateCrisprQcFilterOptions:
    """Optional arguments for :py:meth:`~scranpy.quality_control.create_crispr_qc_filter.create_crispr_qc_filter`.

    Attributes:
        block (Sequence, optional):
            Block assignment for each cell.
            This should be the same as that used in
            in :py:meth:`~scranpy.quality_control.rna.suggest_crispr_qc_filters`.
        verbose (bool, optional): Whether to print logs. Defaults to False.
    """

    block: Optional[Sequence] = None
    verbose: bool = False


def create_crispr_qc_filter(
    metrics: BiocFrame,
    thresholds: BiocFrame,
    options: CreateCrisprQcFilterOptions = CreateCrisprQcFilterOptions(),
) -> ndarray:
    """Defines a filtering vector based on the RNA-derived per-cell quality control (QC) metrics and thresholds.

    Args:
        metrics (BiocFrame): Data frame of metrics,
            see :py:meth:`~scranpy.quality_control.per_cell_crispr_qc_metrics.per_cell_crispr_qc_metrics`
            for the expected format.

        thresholds (BiocFrame): Data frame of filter thresholds,
            see :py:meth:`~scranpy.quality_control.suggest_crispr_qc_filters.suggest_crispr_qc_filters`
            for the expected format.

        options (CreateCrisprQcFilterOptions): Optional parameters.

    Returns:
        ndarray: A numpy boolean array filled with 1 for cells to filter.
    """

    if not isinstance(metrics, BiocFrame):
        raise TypeError("'metrics' is not a `BiocFrame` object.")

    if not isinstance(thresholds, BiocFrame):
        raise TypeError("'thresholds' is not a `BiocFrame` object.")

    num_blocks = 1
    block_offset = 0
    block_info = None

    if options.block is not None:
        block_info = factorize(options.block)
        block_offset = block_info.indices.ctypes.data
        num_blocks = len(block_info.levels)

    tmp_sums_in = metrics.column("sums").astype(float64, copy=False)
    tmp_max_proportions_in = metrics.column("detected").astype(float64, copy=False)
    tmp_max_count_thresh = thresholds.column("max_count").astype(float64, copy=False)
    output = zeros(metrics.shape[0], dtype=uint8)

    lib.create_crispr_qc_filter(
        metrics.shape[0],
        tmp_sums_in,
        tmp_max_proportions_in,
        num_blocks,
        block_offset,
        tmp_max_count_thresh,
        output,
    )

    return output.astype(bool_, copy=False)
