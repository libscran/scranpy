from dataclasses import dataclass
from typing import Optional, Sequence

from biocframe import BiocFrame
from numpy import bool_, float64, int32, ndarray, zeros, uint8

from .. import cpphelpers as lib
from ..utils import factorize
from .utils import create_pointer_array

__author__ = "ltla, jkanche"
__copyright__ = "ltla, jkanche"
__license__ = "MIT"


@dataclass
class CreateRnaQcFilterOptions:
    """Optional arguments for :py:meth:`~scranpy.quality_control.rna.create_rna_qc_filter`.

    Attributes:
        block (Sequence, optional):
            Block assignment for each cell.
            This should be the same as that used in
            in :py:meth:`~scranpy.quality_control.rna.suggest_rna_qc_filters`.
        verbose (bool, optional): Whether to print logs. Defaults to False.
    """

    block: Optional[Sequence] = None
    verbose: bool = False


def create_rna_qc_filter(
    metrics: BiocFrame,
    thresholds: BiocFrame,
    options: CreateRnaQcFilterOptions = CreateRnaQcFilterOptions(),
) -> ndarray:
    """Defines a filtering vector based on the RNA-derived per-cell quality control (QC) metrics and thresholds.

    Args:
        metrics (BiocFrame): Data frame of metrics,
            see :py:meth:`~scranpy.quality_control.rna.per_cell_rna_qc_metrics` for the expected format.

        thresholds (BiocFrame): Data frame of filter thresholds,
            see :py:meth:`~scranpy.quality_control.rna.suggest_rna_qc_filters` for the expected format.

        options (CreateRnaQcFilterOptions): Optional parameters.

    Returns:
        ndarray: A numpy boolean array filled with 1 for cells to filter.
    """

    if not isinstance(metrics, BiocFrame):
        raise TypeError("'metrics' is not a `BiocFrame` object.")

    if not isinstance(thresholds, BiocFrame):
        raise TypeError("'thresholds' is not a `BiocFrame` object.")

    subprop = metrics.column("subset_proportions")
    subpropthresh = thresholds.column("subset_proportions")
    num_subsets = subprop.shape[1]
    subset_in = []
    filter_in = []

    for i in range(num_subsets):
        cursub = subprop.column(i)
        subset_in.append(cursub.astype(float64, copy=False))
        curfilt = subpropthresh.column(i)
        filter_in.append(curfilt.astype(float64, copy=False))

    subset_in_ptr = create_pointer_array(subset_in)
    filter_in_ptr = create_pointer_array(filter_in)

    num_blocks = 1
    block_offset = 0
    block_info = None

    if options.block is not None:
        block_info = factorize(options.block)
        block_offset = block_info.indices.ctypes.data
        num_blocks = len(block_info.levels)

    tmp_sums_in = metrics.column("sums").astype(float64, copy=False)
    tmp_detected_in = metrics.column("detected").astype(int32, copy=False)
    tmp_sums_thresh = thresholds.column("sums").astype(float64, copy=False)
    tmp_detected_thresh = thresholds.column("detected").astype(float64, copy=False)
    output = zeros(metrics.shape[0], dtype=uint8)

    lib.create_rna_qc_filter(
        metrics.shape[0],
        num_subsets,
        tmp_sums_in,
        tmp_detected_in,
        subset_in_ptr.ctypes.data,
        num_blocks,
        block_offset,
        tmp_sums_thresh,
        tmp_detected_thresh,
        filter_in_ptr.ctypes.data,
        output,
    )

    return output.astype(bool_, copy=False)
