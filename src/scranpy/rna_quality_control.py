from typing import Union, Sequence, Any, Optional

import numpy
import biocutils
import mattress
import biocframe

from ._utils_qc import _sanitize_subsets
from . import lib_scranpy as lib


def compute_rna_qc_metrics(
    x: Any,
    subsets: Union[dict, Sequence, biocutils.NamedList],
    row_names: Optional[Sequence] = None,
    num_threads: int = 1
) -> biocframe.BiocFrame:
    """Compute quality control metrics from RNA count data.

    Args: 
        x:
            A matrix-like object containing RNA counts.
            Each row should correspond to a gene while each column should correspond to a cell.

        subsets:
            Subsets of genes corresponding to "control" features like mitochondrial genes.
            This may be either:

            - A list of sequences.
              Each sequence corresponds to a gene subset and can contain booleans, integers or strings.

              - For booleans, the sequence should be of length equal to the number of rows, and values should be truthy for rows that belong in the subset.
                If the sequence contains booleans, it should not contain any other type.
              - For integers, the value is the row index of a gene in the subset.
              - For strings, the value is the name of a gene in the subset.
                This should match at least one element in ``row_names``.

            - A dictionary where keys are the names of each gene subset and the values are arrays as described above.
            - A :py:class:`~biocutils.NamedList.NamedList` where each element is an array as described above, possibly with names.

        row_names:
            Sequence of strings of length equal to the number of rows of ``x``, containing the name of each gene.
            Duplicate names are allowed but only the first occurrence will be used.
            If ``None``, rows are assumed to be unnamed.

        num_threads:
            Number of threads to use.

    Returns:
        A :py:class:`~biocframe.BiocFrame.BiocFrame` with number of rows equal to the number of cells (i.e., columns) in ``x``.
        It contains the following columns:

        - ``sum``: a double-precision NumPy array containing the sum of counts across all genes for each cell.
        - ``detected``: an integer NumPy array containing the number of genes with non-zero expression in each cell.
        - ``subset_proportion``: a nested :py:class:`~biocframe.BiocFrame.BiocFrame` with one column per subset in ``subsets``.
          Each column is a double-precision NumPy array that contains the proportion of counts in the corresponding subset in each cell.

    References:
        The ``compute_rna_qc_metrics`` function in the `scran_qc <https://github.com/libscran/scran_qc>`_ C++ library, which describes the rationale behind these QC metrics.

    Examples:
        >>> import numpy
        >>> mat = numpy.reshape(numpy.random.poisson(lam=5, size=1000), (50, 20))
        >>> import scranpy
        >>> res = scranpy.compute_rna_qc_metrics(mat, { "mito": [ 1, 10, 20, 40 ] })
        >>> print(res)
    """

    ptr = mattress.initialize(x)
    subkeys, subvals = _sanitize_subsets(subsets, x.shape[0], row_names=row_names)
    osum, odetected, osubset_prop = lib.compute_rna_qc_metrics(ptr.ptr, subvals, num_threads)

    new_subset_prop = biocframe.BiocFrame(number_of_rows = x.shape[1])
    for i, k in enumerate(subkeys):
        new_subset_prop.set_column(k, osubset_prop[i], in_place=True)

    return biocframe.BiocFrame({
        "sum": osum,
        "detected": odetected,
        "subset_proportion": new_subset_prop
    })


def suggest_rna_qc_thresholds(
    metrics: biocframe.BiocFrame,
    block: Optional[Sequence] = None,
    num_mads: float = 3.0,
) -> biocutils.NamedList:
    """
    Suggest filter thresholds for the RNA-derived QC metrics, typically generated from :py:func:`~compute_rna_qc_metrics`.

    Args:
        metrics:
            RNA-derived QC metrics from :py:func:`~compute_rna_qc_metrics`.
            This should contain the ``sum``, ``detected`` and ``subset_proportion`` columns.

        block:
            Blocking factor specifying the block of origin (e.g., batch, sample) for each cell in ``metrics``.
            If supplied, a separate threshold is computed from the cells in each block.
            Alternatively ``None``, if all cells are from the same block.

        num_mads:
            Number of MADs from the median to define the threshold for outliers in each QC metric.

    Returns:
        If ``block = None``, a :py:class:`~biocutils.NamedList.NamedList` is returned, containing the following entries.

        - ``sum``, a number specifying the lower threshold on the sum of counts in each cell.
        - ``detected``, a number specifying the lower threshold on the number of detected genes.
        - ``subset_proportion``, a :py:class:`~biocutils.FloatList.FloatList` of length equal to the number of control subsets (and named accordingly).
          Each entry represents the upper bound on the proportion of counts in the corresponding control subset. 

        If ``block`` is provided, the NamedList instead contains:

        - ``sum``, a FloatList of length equal to the number of blocks (and named accordingly).
          Each entry represents the lower threshold on the sum of counts in the corresponding block.
        - ``detected``, a FloatList of length equal to the number of blocks (and named accordingly).
          Each entry represents the lower threshold on the number of detected genes in the corresponding block.
        - ``subset_proportion``, a NamedList of length equal to the number of control subsets.
          Each entry is another FloatList that contains the upper threshold on the proportion of counts for that subset in each block.
        - ``block_ids``, a list containing the unique levels of the blocking factor.
          This is in the same order as the blocks in ``detected`` and ``subset_sum``.

    References:
        The ``compute_rna_qc_filters`` and ``compute_rna_qc_filters_blocked`` functions in the `scran_qc`_ C++ library,
        which describes the rationale behind the suggested filters.

    Examples:
        >>> import numpy
        >>> mat = numpy.reshape(numpy.random.poisson(lam=5, size=1000), (50, 20))
        >>> import scranpy
        >>> res = scranpy.compute_rna_qc_metrics(mat, { "mito": [ 1, 10, 20, 40 ] })
        >>> filt = scranpy.suggest_rna_qc_thresholds(res)
        >>> print(filt)
    """

    if block is not None:
        blocklev, blockind = biocutils.factorize(block, sort_levels=True, dtype=numpy.uint32, fail_missing=True)
    else:
        blocklev = None
        blockind = None

    submet = metrics["subset_proportion"]
    sums, detected, subset_proportions = lib.suggest_rna_qc_thresholds(
        (
            metrics["sum"],
            metrics["detected"],
            [submet.get_column(y) for y in submet.get_column_names()]
        ),
        blockind,
        num_mads
    )

    output = biocutils.NamedList()

    if blockind is not None:
        output["sum"] = biocutils.FloatList(sums, blocklev)
        output["detected"] = biocutils.FloatList(detected, blocklev)
        for i, s in enumerate(subset_proportions):
            subset_proportions[i] = biocutils.FloatList(s, blocklev)
        output["subset_proportion"] = biocutils.NamedList(subset_proportions, submet.get_column_names())
        output["block_ids"] = blocklev
    else:
        output["sum"] = sums
        output["detected"] = detected
        output["subset_proportion"] = biocutils.FloatList(subset_proportions, submet.get_column_names())

    return output


def filter_rna_qc_metrics(
    thresholds: biocutils.NamedList,
    metrics: biocframe.BiocFrame,
    block: Optional[Sequence] = None
) -> numpy.ndarray:
    """Filter for high-quality cells based on RNA-derived QC metrics.

    Args:
        thresholds:
            Filter thresholds on the QC metrics, typically computed with :py:func:`~suggest_rna_qc_thresholds`.

        metrics:
            RNA-derived QC metrics, typically computed with :py:func:`~compute_rna_qc_metrics`.

        block:
            Blocking factor specifying the block of origin (e.g., batch, sample) for each cell in ``metrics``.
            The levels should be a subset of those used in :py:func:`~suggest_rna_qc_thresholds`.

    Returns:
        A boolean NumPy vector of length equal to the number of cells in ``metrics``, containing truthy values for putative high-quality cells.

    References:
        The ``RnaQcFilters`` and ``RnaQcBlockedFilters`` functions in the `scran_qc`_ C++ library.

    Examples:
        >>> import numpy
        >>> mat = numpy.reshape(numpy.random.poisson(lam=5, size=1000), (50, 20))
        >>> import scranpy
        >>> res = scranpy.compute_rna_qc_metrics(mat, { "mito": [ 1, 10, 20, 40 ] })
        >>> filt = scranpy.suggest_rna_qc_thresholds(res)
        >>> keep = scranpy.filter_rna_qc_metrics(filt, res)
        >>> keep.sum()
    """

    sthresh = thresholds["sum"]
    dthresh = thresholds["detected"]
    subthresh = thresholds["subset_proportion"]

    if "block_ids" in thresholds.get_names():
        if block is None:
            raise ValueError("'block' must be supplied if it was used in 'suggest_rna_qc_thresholds'")
        blockind = biocutils.match(block, thresholds["block_ids"], dtype=numpy.uint32, fail_missing=True)
        sfilt = numpy.array(sthresh.as_list(), dtype=numpy.float64)
        dfilt = numpy.array(dthresh.as_list(), dtype=numpy.float64)
        subfilt = [numpy.array(s.as_list(), dtype=numpy.float64) for s in subthresh.as_list()]
    else:
        if block is not None:
            raise ValueError("'block' cannot be supplied if it was not used in 'suggest_rna_qc_thresholds'")
        blockind = None
        sfilt = sthresh
        dfilt = dthresh
        subfilt = numpy.array(subthresh.as_list(), dtype=numpy.float64)

    smet = metrics["sum"]
    dmet = metrics["detected"]
    submet = metrics["subset_proportion"]
    if subthresh.get_names() != submet.get_column_names():
        raise ValueError("mismatch in the subset names between 'thresholds' and 'metrics'")

    return lib.filter_rna_qc_metrics(
        (sfilt, dfilt, subfilt),
        (smet, dmet, [submet.get_column(y) for y in submet.get_column_names()]),
        blockind
    )
