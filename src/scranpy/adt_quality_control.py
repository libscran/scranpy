from typing import Union, Sequence, Any, Optional
from dataclasses import dataclass
from collections.abc import Mapping

import numpy
import biocutils
import mattress

from ._utils_qc import _sanitize_subsets
from . import lib_scranpy as lib


@dataclass
class ComputeAdtQcMetricsResults:
    """Results of :py:func:`~compute_adt_qc_metrics`."""

    sum: numpy.ndarray 
    """Floating-point array of length equal to the number of cells, containing the sum of counts across all ADTs for each cell."""

    detected: numpy.ndarray 
    """Integer array of length equal to the number of cells, containing the number of detected ADTs in each cell."""

    subset_sum: biocutils.NamedList
    """List of length equal to the number of ``subsets`` in :py:func:`~compute_adt_qc_metrics`.
    Each element corresponds to a subset of ADTs and is a NumPy array of length equal to the number of cells.
    Each entry of the array contains the sum of counts for that subset in each cell."""

    def to_biocframe(self, flatten: bool = True): 
        """Convert the results into a :py:class:`~biocframe.BiocFrame.BiocFrame`.

        Args:
            flatten:
                Whether to flatten the subset sums into separate columns.
                If ``True``, each entry of :py:attr:`~subset_sum` is represented by a ``subset_sum_<NAME>`` column,
                where ``<NAME>`` is the the name of each entry (if available) or its index (otherwise).
                If ``False``, :py:attr:`~subset_sum` is represented by a nested :py:class:`~biocframe.BiocFrame.BiocFrame`.

        Returns:
            A :py:class:`~biocframe.BiocFrame.BiocFrame` where each row corresponds to a cell and each column is one of the metrics.
        """
        colnames = ["sum", "detected"]
        contents = {}
        for n in colnames:
            contents[n] = getattr(self, n)

        subnames = self.subset_sum.get_names()
        if subnames is not None:
            subnames = subnames.as_list()
        else:
            subnames = [str(i) for i in range(len(self.subset_sum))]

        import biocframe
        if flatten:
            for i, n in enumerate(subnames):
                nn = "subset_sum_" + n
                colnames.append(nn)
                contents[nn] = self.subset_sum[i]
        else:
            subcontents = {}
            for i, n in enumerate(subnames):
                subcontents[n] = self.subset_sum[i]
            colnames.append("subset_sum")
            contents["subset_sum"] = biocframe.BiocFrame(subcontents, column_names=subnames, number_of_rows=len(self.sum))

        return biocframe.BiocFrame(contents, column_names=colnames)


def compute_adt_qc_metrics(
    x: Any,
    subsets: Union[Mapping, Sequence],
    num_threads: int = 1
) -> ComputeAdtQcMetricsResults :
    """Compute quality control metrics from ADT count data.

    Args: 
        x:
            A matrix-like object containing ADT counts.

        subsets:
            Subsets of ADTs corresponding to control features like IgGs.
            This may be either:

            - A list of arrays.
              Each array corresponds to an ADT subset and can either contain boolean or integer values.
              For booleans, the array should be of length equal to the number of rows, and values should be truthy for rows that belong in the subset.
              For integers, each element of the array is treated the row index of an ADT in the subset.
            - A dictionary where keys are the names of each ADT subset and the values are arrays as described above.
            - A :py:class:`~biocutils.NamedList.NamedList` where each element is an array as described above, possibly with names.

        num_threads:
            Number of threads to use.

    Returns:
        QC metrics computed from the ADT count matrix for each cell.

    References:
        The ``compute_adt_qc_metrics`` function in the `scran_qc <https://libscran.github.io/scran_qc>`_ C++ library, which describes the rationale behind these QC metrics.
    """
    ptr = mattress.initialize(x)
    subkeys, subvals = _sanitize_subsets(subsets, x.shape[0])
    osum, odetected, osubset_sum = lib.compute_adt_qc_metrics(ptr.ptr, subvals, num_threads)
    osubset_sum = biocutils.NamedList(osubset_sum, subkeys)
    return ComputeAdtQcMetricsResults(osum, odetected, osubset_sum)


@dataclass
class SuggestAdtQcThresholdsResults:
    """Results of :py:func:`~suggest_adt_qc_thresholds`."""

    detected: Union[biocutils.NamedList, float]
    """Threshold on the number of detected ADTs.
    Cells with lower numbers of detected ADTs are considered to be of low quality.

    If ``block`` is provided in :py:func:`~suggest_adt_qc_thresholds`, a list is returned containing a separate threshold for each level of the factor.
    Otherwise, a single float is returned containing the threshold for all cells."""

    subset_sum: biocutils.NamedList
    """Thresholds on the sum of counts in each ADT subset.
    Each element of the list corresponds to a ADT subset. 
    Cells with higher sums than the threshold for any subset are considered to be of low quality. 

    If ``block`` is provided in :py:func:`~suggest_adt_qc_thresholds`, each entry of the returned list is another :py:class:`~biocutils.NamedList.NamedList`  containing a separate threshold for each level.
    Otherwise, each entry of the list is a single float containing the threshold for all cells."""

    block: Optional[list]
    """Levels of the blocking factor.
    Each entry corresponds to a element of :py:attr:`~detected`, etc., if ``block`` was provided in :py:func:`~suggest_adt_qc_thresholds`.
    This is set to ``None`` if no blocking was performed."""


def suggest_adt_qc_thresholds(
    metrics: ComputeAdtQcMetricsResults,
    block: Optional[Sequence] = None,
    min_detected_drop: float = 0.1,
    num_mads: float = 3.0,
) -> SuggestAdtQcThresholdsResults:
    """Suggest filter thresholds for the ADT-derived QC metrics, typically generated from :py:func:`~compute_adt_qc_metrics`.

    Args:
        metrics:
            ADT-derived QC metrics from :py:func:`~compute_adt_qc_metrics`.

        block:
            Blocking factor specifying the block of origin (e.g., batch, sample) for each cell in ``metrics``.
            If supplied, a separate threshold is computed from the cells in each block.
            Alternatively ``None``, if all cells are from the same block.

        min_detected_drop:
            Minimum proportional drop in the number of detected ADTs to consider a cell to be of low quality.
            Specifically, the filter threshold on ``metrics.detected`` must be no higher than the product of ``min_detected_drop`` and the median number of ADTs, regardless of ``num_mads``.

        num_mads:
            Number of MADs from the median to define the threshold for outliers in each QC metric.

    Returns:
        Suggested filters on the relevant QC metrics.

    References:
        The ``compute_adt_qc_filters`` and ``compute_adt_qc_filters_blocked`` functions in the `scran_qc <https://libscran.github.io/scran_qc>`_ C++ library, which describes the rationale behind the suggested filters.
    """
    if block is not None:
        blocklev, blockind = biocutils.factorize(block, sort_levels=True, dtype=numpy.uint32, fail_missing=True)
    else:
        blocklev = None
        blockind = None

    detected, subset_sums = lib.suggest_adt_qc_thresholds(
        (metrics.sum, metrics.detected, metrics.subset_sum.as_list()),
        blockind,
        min_detected_drop,
        num_mads
    )

    if blockind is not None:
        detected = biocutils.NamedList(detected, blocklev)
        for i, s in enumerate(subset_sums):
            subset_sums[i] = biocutils.NamedList(s, blocklev)

    subset_sums = biocutils.NamedList(subset_sums, metrics.subset_sum.get_names())
    return SuggestAdtQcThresholdsResults(detected, subset_sums, blocklev)


def filter_adt_qc_metrics(
    thresholds: SuggestAdtQcThresholdsResults,
    metrics: ComputeAdtQcMetricsResults,
    block: Optional[Sequence] = None
) -> numpy.ndarray:
    """Filter for high-quality cells based on ADT-derived QC metrics.

    Args:
        thresholds:
            Filter thresholds on the QC metrics, typically computed with :py:func:`~suggest_adt_qc_thresholds`.

        metrics:
            ADT-derived QC metrics, typically computed with :py:func:`~compute_adt_qc_metrics`.

        block:
            Blocking factor specifying the block of origin (e.g., batch, sample) for each cell in ``metrics``.
            The levels should be a subset of those used in :py:func:`~suggest_adt_qc_thresholds`.

    Returns:
        A NumPy vector of length equal to the number of cells in ``metrics``, containing truthy values for putative high-quality cells.
    """
    if thresholds.block is not None:
        if block is None:
            raise ValueError("'block' must be supplied if it was used in 'suggest_adt_qc_thresholds'")
        blockind = biocutils.match(block, thresholds.block, dtype=numpy.uint32, fail_missing=True)
        detected = numpy.array(thresholds.detected.as_list(), dtype=numpy.float64)
        subset_sum = [numpy.array(s.as_list(), dtype=numpy.float64) for s in thresholds.subset_sum.as_list()]
    else:
        if block is not None:
            raise ValueError("'block' cannot be supplied if it was not used in 'suggest_adt_qc_thresholds'")
        blockind = None
        detected = thresholds.detected
        subset_sum = numpy.array(thresholds.subset_sum.as_list(), dtype=numpy.float64)

    return lib.filter_adt_qc_metrics(
        (detected, subset_sum),
        (metrics.sum, metrics.detected, metrics.subset_sum.as_list()),
        blockind
    )

if biocutils.package_utils.is_package_installed("summarizedexperiment"):
    import summarizedexperiment


    def quick_adt_qc_se(
        x: summarizedexperiment.SummarizedExperiment,
        subsets: Union[Mapping, Sequence],
        more_suggest_args: dict = {},
        num_threads: int = 1,
        block: Optional[Sequence] = None,
        assay_type: Union[int, str] = "counts",
        output_prefix: Optional[str] = None, 
        meta_name: Optional[str] = "qc",
        flatten: bool = True
    ) -> summarizedexperiment.SummarizedExperiment: 
        """
        Quickly compute quality control (QC) metrics, thresholds and filters from ADT data in a :py:class:`~summarizedexperiment.SummarizedExperiment.SummarizedExperiment`.
        This calls :py:func:`~compute_adt_qc_metrics`, :py:func:`~suggest_adt_qc_thresholds`, and :py:func:`~filter_adt_qc_metrics`.

        Args:
            x:
                A :py:class:`~summarizedexperiment.SummarizedExperiment.SummarizedExperiment` or one of its subclasses.
                Rows correspond to antibody-derived tags (ADTs) and columns correspond to cells.

            subsets:
                Passed to :py:func:`~compute_adt_qc_metrics`.

            more_suggest_args:
                Named dictionary of arguments, to pass to :py:func:`~compute_adt_qc_metrics`.

            num_threads:
                Passed to :py:func:`~compute_adt_qc_metrics`.

            block:
                Blocking factor specifying the block of origin (e.g., batch, sample) for each cell in ``metrics``.
                If supplied, a separate threshold is computed from the cells in each block.
                Alternatively ``None``, if all cells are from the same block.

            assay_type:
                Index or name of the assay of ``x`` containing the ADT count matrix.

            output_prefix:
                Prefix to add to the column names of the column data containing the output QC statistics.
                If ``None``, no prefix is added.

            meta_name:
                Name of the metadata entry in which to store additional outputs like the filtering thresholds.
                If ``None``, additional outputs are not reported.

            flatten:
                Whether to flatten the subset proportions into separate columns of the column data.
                If ``False``, the subset proportions are stored in a nested :py:class:`~biocframe.BiocFrame.BiocFrame`.

        Returns:
            ``x``, with additional columns added to its column data.
            Each column contains per-cell values for one of the ADT-related QC metrics, see :py:func:`~compute_adt_qc_metrics` for details.
            The suggested thresholds are stored as a list in the metadata.
            The column data also contains a ``keep`` column, specifying which cells are to be retained.
        """

        metrics = compute_adt_qc_metrics(x.get_assay(assay_type), subsets, num_threads=num_threads)
        thresholds = suggest_adt_qc_thresholds(metrics, block=block, **more_suggest_args)
        keep = filter_adt_qc_metrics(thresholds, metrics, block=block)

        df = metrics.to_biocframe(flatten=flatten)
        df.set_column("keep", keep, in_place=True)
        if output_prefix is not None:
            df.set_column_names([output_prefix + n for n in df.get_column_names()], in_place=True)

        x = x.set_column_data(biocutils.combine_columns(x.get_column_data(), df))
        if meta_name is not None:
            import copy
            meta = copy.copy(x.get_metadata())
            meta[meta_name] = { "thresholds": thresholds }
            x = x.set_metadata(meta)

        return x
