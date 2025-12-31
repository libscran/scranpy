from typing import Union, Sequence, Mapping, Optional

import biocutils
import summarizedexperiment

from .adt_quality_control import *
from . import _utils_se as seutils


def quick_adt_qc_se(
    x: summarizedexperiment.SummarizedExperiment,
    subsets: Union[Mapping, Sequence],
    num_threads: int = 1,
    more_suggest_args: dict = {},
    block: Optional[Sequence] = None,
    assay_type: Union[int, str] = "counts",
    output_prefix: Optional[str] = None, 
    meta_name: Optional[str] = "qc",
    flatten: bool = True
) -> summarizedexperiment.SummarizedExperiment: 
    """
    Quickly compute quality control (QC) metrics, thresholds and filters from ADT data in a :py:class:`~summarizedexperiment.SummarizedExperiment.SummarizedExperiment`.
    This calls :py:func:`~scranpy.adt_quality_control.compute_adt_qc_metrics`,
    :py:func:`~scranpy.adt_quality_control.suggest_adt_qc_thresholds`,
    and :py:func:`~scranpy.adt_quality_control.filter_adt_qc_metrics`.

    Args:
        x:
            A :py:class:`~summarizedexperiment.SummarizedExperiment.SummarizedExperiment` or one of its subclasses.
            Rows correspond to antibody-derived tags (ADTs) and columns correspond to cells.

        subsets:
            List of subsets of control features, passed to :py:func:`~scranpy.adt_quality_control.compute_adt_qc_metrics`.

        num_threads:
            Number of threads, passed to :py:func:`~scranpy.adt_quality_control.compute_adt_qc_metrics`.

        more_suggest_args:
            Additional arguments to pass to :py:func:`~scranpy.adt_quality_control.suggest_adt_qc_thresholds`.

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
        A copy of ``x`` with additional columns added to its column data.
        Each column contains per-cell values for one of the ADT-related QC metrics, see :py:func:`~scranpy.adt_quality_control.compute_adt_qc_metrics` for details.
        The suggested thresholds are stored as a list in the metadata.
        The column data also contains a ``keep`` column, specifying which cells are to be retained.

    Examples:
        >>> import scranpy
        >>> sce = scranpy.get_test_adt_data_se().get_alternative_experiment("ADT")
        >>> is_igg = list(y.find("IgG") >= 0 for y in sce.get_row_names())
        >>> sce = scranpy.quick_adt_qc_se(sce, subsets={ "igg": is_igg })
        >>> print(sce.get_column_data()[:,["sum", "detected", "subset_sum_igg"]])
        >>> print(sce.get_metadata()["qc"]["thresholds"])
        >>> sce.get_column_data()["keep"].sum()
    """

    metrics = compute_adt_qc_metrics(x.get_assay(assay_type), subsets, row_names=x.get_row_names(), num_threads=num_threads)
    thresholds = suggest_adt_qc_thresholds(metrics, block=block, **more_suggest_args)
    keep = filter_adt_qc_metrics(thresholds, metrics, block=block)

    df = format_compute_adt_qc_metrics_result(metrics, flatten=flatten)
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


def format_compute_adt_qc_metrics_result(df: biocframe.BiocFrame, flatten: bool = True) -> biocframe.BiocFrame:
    """
    Pretty-format the results of :py:func:`~scranpy.adt_quality_control.compute_adt_qc_metrics`.

    Args:
        df:
            Result of :py:func:`~scranpy.adt_quality_control.compute_adt_qc_metrics`.

        flatten:
            Whether to flatten the nested BiocFrame of subset proportions.

    Returns:
        A BiocFrame containing per-cell QC statistics.
    """

    if not flatten:
        return df

    field = "subset_sum"
    values = df.get_column(field)
    values = values.set_column_names([field + "_" + n for n in values.get_column_names()])
    return biocutils.combine_columns(df.remove_column(field), values)
