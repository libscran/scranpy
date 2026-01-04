from typing import Union, Sequence, Any, Optional

import numpy
import biocutils
import mattress
import biocframe

from . import _lib_scranpy as lib


def compute_crispr_qc_metrics(x: Any, num_threads: int = 1) -> biocframe.BiocFrame:
    """Compute quality control metrics from CRISPR count data.

    Args: 
        x:
            A matrix-like object containing CRISPR counts.
            Each row should correspond to a guide while each column should correspond to a cell.

        num_threads:
            Number of threads to use.

    Returns:
        A :py:class:`~biocframe.BiocFrame.BiocFrame` with number of rows equal to the number of cells (i.e., columns) in ``x``.
        It contains the following columns.

        - ``sum``, a double-precision NumPy array containing the sum of counts across all guides for each cell.
        - ``detected``, an integer NumPy array containing the number of guides with non-zero counts in each cell.
        - ``max_value``, a double-precision NumPy array containing the maximum count for each cell.
        - ``max_index``, an integer NumPy array containing the row index of the guide with the maximum count in each cell.

    References:
        The ``compute_crispr_qc_metrics`` function in the `scran_qc`_ C++ library, which describes the rationale behind these QC metrics.

    Examples:
        >>> import numpy
        >>> mat = numpy.reshape(numpy.random.poisson(lam=5, size=1000), (50, 20))
        >>> import scranpy
        >>> res = scranpy.compute_crispr_qc_metrics(mat)
        >>> print(res)
    """

    ptr = mattress.initialize(x)
    osum, odetected, omaxval, omaxind = lib.compute_crispr_qc_metrics(ptr.ptr, num_threads)
    return biocframe.BiocFrame({
        "sum": osum,
        "detected": odetected,
        "max_value": omaxval,
        "max_index": omaxind
    })


def suggest_crispr_qc_thresholds(
    metrics: biocframe.BiocFrame,
    block: Optional[Sequence] = None,
    num_mads: float = 3.0,
) -> biocutils.NamedList:
    """
    Suggest filter thresholds for the CRISPR-derived QC metrics, typically generated from :py:func:`~compute_crispr_qc_metrics`.

    Args:
        metrics:
            CRISPR-derived QC metrics from :py:func:`~compute_crispr_qc_metrics`.
            This should contain the ``sum``, ``detected``, ``max_value`` and ``max_index`` columns.

        block:
            Blocking factor specifying the block of origin (e.g., batch, sample) for each cell in ``metrics``.
            If supplied, a separate threshold is computed from the cells in each block.
            Alternatively ``None``, if all cells are from the same block.

        num_mads:
            Number of MADs from the median to define the threshold for outliers in each QC metric.

    Returns:
        If ``block = None``, a :py:class:`~biocutils.NamedList.NamedList` is returned, containing the following entries.

        - ``max_value``, a number specifying the lower threshold on the maximum count in each cell.

        If ``block`` is provided, the ``NamedList`` instead contains:

        - ``max_value``, a FloatList of length equal to the number of blocks (and named accordingly).
          Each entry represents the lower threshold on the maximum count in the corresponding block.
        - ``block_ids``, a list containing the unique levels of the blocking factor.
          This is in the same order as the blocks in ``detected`` and ``subset_sum``.

    References:
        The ``compute_crispr_qc_filters`` and ``compute_crispr_qc_filters_blocked`` functions in the `scran_qc`_ C++ library,
        which describes the rationale behind the suggested filters.

    Examples:
        >>> import numpy
        >>> mat = numpy.reshape(numpy.random.poisson(lam=5, size=1000), (50, 20))
        >>> import scranpy
        >>> res = scranpy.compute_crispr_qc_metrics(mat)
        >>> filt = scranpy.suggest_crispr_qc_thresholds(res)
        >>> print(filt)
    """

    if block is not None:
        blocklev, blockind = biocutils.factorize(block, sort_levels=True, dtype=numpy.uint32, fail_missing=True)
    else:
        blocklev = None
        blockind = None

    max_value, = lib.suggest_crispr_qc_thresholds(
        (metrics["sum"], metrics["detected"], metrics["max_value"], metrics["max_index"]),
        blockind,
        num_mads
    )

    output = biocutils.NamedList()

    if blockind is not None:
        output["max_value"] = biocutils.NamedList(max_value, blocklev)
        output["block_ids"] = blocklev
    else:
        output["max_value"] = max_value

    return output


def filter_crispr_qc_metrics(
    thresholds: biocutils.NamedList,
    metrics: biocframe.BiocFrame,
    block: Optional[Sequence] = None
) -> numpy.ndarray:  
    """
    Filter for high-quality cells based on CRISPR-derived QC metrics.

    Args:
        thresholds:
            Filter thresholds on the QC metrics, typically computed with :py:func:`~suggest_crispr_qc_thresholds`.

        metrics:
            CRISPR-derived QC metrics, typically computed with :py:func:`~compute_crispr_qc_metrics`.

        block:
            Blocking factor specifying the block of origin (e.g., batch, sample) for each cell in ``metrics``.
            The levels should be a subset of those used in :py:func:`~suggest_crispr_qc_thresholds`.

    Returns:
        A boolean NumPy vector of length equal to the number of cells in ``metrics``, containing truthy values for putative high-quality cells.

    References:
        The ``CrisprQcFilters`` and ``CrisprQcBlockedFilters`` functions in the `scran_qc`_ C++ library.

    Examples:
        >>> import numpy
        >>> mat = numpy.reshape(numpy.random.poisson(lam=5, size=1000), (50, 20))
        >>> import scranpy
        >>> res = scranpy.compute_crispr_qc_metrics(mat)
        >>> filt = scranpy.suggest_crispr_qc_thresholds(res)
        >>> keep = scranpy.filter_crispr_qc_metrics(filt, res)
        >>> keep.sum()
    """

    if "block_ids" in thresholds.get_names():
        if block is None:
            raise ValueError("'block' must be supplied if it was used in 'suggest_crispr_qc_thresholds'")
        blockind = biocutils.match(block, thresholds["block_ids"], dtype=numpy.uint32, fail_missing=True)
        max_value = numpy.array(thresholds["max_value"].as_list(), dtype=numpy.float64)
    else:
        if block is not None:
            raise ValueError("'block' cannot be supplied if it was not used in 'suggest_crispr_qc_thresholds'")
        blockind = None
        max_value = thresholds["max_value"]

    return lib.filter_crispr_qc_metrics(
        (max_value,),
        (metrics["sum"], metrics["detected"], metrics["max_value"], metrics["max_index"]),
        blockind
    )
