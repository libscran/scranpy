from typing import Union, Sequence, Mapping, Optional

import biocutils
import summarizedexperiment

from .rna_quality_control import *
from . import _utils_se as seutils


def compute_rna_qc_metrics_with_altexps(
    x: summarizedexperiment.SummarizedExperiment,
    subsets: Union[Mapping, Sequence],
    altexp_proportions: Optional[Union[list, dict, biocutils.NamedList]] = None,
    num_threads: int = 1,
    assay_type: Union[int, str] = "counts"
) -> tuple:
    """
    Compute RNA-related QC metrics for the main and alternative experiments of a :py:class:`~singlecellexperiment.SingleCellExperiment.SingleCellExperiment`.
    This calls :py:func:`~scranpy.rna_quality_control.compute_rna_qc_metrics` on each experiment.

    Args:
        x:
            A :py:class:`~summarizedexperiment.SummarizedExperiment.SummarizedExperiment` object or one of its subclasses.
            Rows correspond to genes and columns correspond to cells.

        subsets:
            List of subsets of control genes, passed to :py:func:`~scranpy.rna_quality_control.compute_rna_qc_metrics` for the main experiment.

        altexp_proportions:
            Indices or names of alternative experiments for which to compute QC metrics.
            This is typically used to refer to alternative experiments holding spike-in data.
            An additional proportion is computed for each alternative experiment relative to the main experiment, 
            i.e., X/(X+Y) where X is the alternative experiment's total and Y is the main experiment's total.
            These proportions will be used for filtering in the same manner as the proportions computed from ``subsets``.

            If a list or unnamed :py:class:`~biocutils.NamedList.NamedList` is supplied,
            it should be contain the indices/names of alternative experiments for which to compute QC metrics.
            The assay to use from each alternative experiment is determined by ``assay_type``.

            If a dictionary or named :py:class:`~biocutils.NamedList.NamedList` is supplied,
            each name specifies an alternative experiment while each value is the index/name of the assay to use from that experiment.

            This option is Only relevant if ``x`` is a `~singlecellexperiment.SingleCellExperiment.SingleCellExperiment`.

        num_threads:
            Number of threads, passed to :py:func:`~scranpy.rna_quality_control.compute_rna_qc_metrics`.

        assay_type:
            Index or name of the assay of ``x`` containing the ADT count matrix.

    Returns:
        Tuple of QC metrics for the main experiment (first) and all of the alternative experiments specified by ``altexp_proportions`` (second).
        The latter is returned as a `biocutils.NamedList.NamedList`.
        The proportion of counts for each alternative experiment is also returned in the ``subsets`` of the main experiment.
    """

    metrics = compute_rna_qc_metrics(x.get_assay(assay_type), subsets, num_threads=num_threads)

    altexp_collected = biocutils.NamedList()
    if altexp_proportions is not None:
        altexp_proportions = seutils.sanitize_altexp_assays(altexp_proportions, x.get_alternative_experiment_names(), default_assay_type=assay_type)
        total_sum = metrics["sum"]

        for ae_name, ae_assay_type in altexp_proportions.items():
            ae_assay = x.get_alternative_experiment(ae_name).get_assay(ae_assay_type)
            ae_metrics = compute_rna_qc_metrics(ae_assay, subsets=[], num_threads=num_threads)
            altexp_collected[ae_name] = ae_metrics
            ae_sum = ae_metrics["sum"]
            metrics["subset_proportion"][ae_name] = ae_sum / (total_sum + ae_sum)

    return (metrics, altexp_collected)


def quick_rna_qc_se(
    x: summarizedexperiment.SummarizedExperiment,
    subsets: Union[Mapping, Sequence],
    altexp_proportions: Optional[Union[list, dict, biocutils.NamedList]] = None,
    num_threads: int = 1,
    more_suggest_args: dict = {},
    block: Optional[Sequence] = None,
    assay_type: Union[int, str] = "counts",
    output_prefix: Optional[str] = None, 
    meta_name: Optional[str] = "qc",
    flatten: bool = True
) -> summarizedexperiment.SummarizedExperiment: 
    """
    Quickly compute quality control (QC) metrics, thresholds and filters from RNA data in a :py:class:`~summarizedexperiment.SummarizedExperiment.SummarizedExperiment`.
    This calls :py:func:`~scranpy.rna_quality_control.compute_rna_qc_metrics`,
    :py:func:`~scranpy.rna_quality_control.suggest_rna_qc_thresholds`,
    and :py:func:`~scranpy.rna_quality_control.filter_rna_qc_metrics`.

    Args:
        x:
            A :py:class:`~summarizedexperiment.SummarizedExperiment.SummarizedExperiment` or one of its subclasses.
            Rows correspond to genes and columns correspond to cells.

        subsets:
            List of subsets of control genes, passed to :py:func:`~compute_rna_qc_metrics_with_altexps`.

        altexp_proportions:
            Indices or names of alternative experiments for which to compute QC metrics, see :py:func:`~compute_rna_qc_metrics_with_altexps` for details.

        num_threads:
            Number of threads, passed to :py:func:`~compute_rna_qc_metrics_with_altexps`.

        more_suggest_args:
            Additional arguments to pass to :py:func:`~scranpy.rna_quality_control.suggest_rna_qc_thresholds`.

        block:
            Blocking factor specifying the block of origin (e.g., batch, sample) for each cell in ``metrics``.
            If supplied, a separate threshold is computed from the cells in each block.
            Alternatively ``None``, if all cells are from the same block.

        assay_type:
            Index or name of the assay of ``x`` containing the RNA count matrix.

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
        Each column contains per-cell values for one of the RNA-related QC metrics, see :py:func:`~scranpy.rna_quality_control.compute_rna_qc_metrics` for details.
        The suggested thresholds are stored as a list in the metadata.
        The column data also contains a ``keep`` column, specifying which cells are to be retained.
    """

    main_metrics, ae_metrics = compute_rna_qc_metrics_with_altexps(x, subsets, altexp_proportions=altexp_proportions, num_threads=num_threads, assay_type=assay_type)
    thresholds = suggest_rna_qc_thresholds(main_metrics, block=block, **more_suggest_args)
    keep = filter_rna_qc_metrics(thresholds, main_metrics, block=block)

    df = format_compute_rna_qc_metrics_result(main_metrics, flatten=flatten)
    df.set_column("keep", keep, in_place=True)
    if output_prefix is not None:
        df.set_column_names([output_prefix + n for n in df.get_column_names()], in_place=True)
    x = x.set_column_data(biocutils.combine_columns(x.get_column_data(), df))

    if altexp_proportions is not None:
        for ae_name in ae_metrics.get_names():
            ae_df = format_compute_rna_qc_metrics_result(ae_metrics[ae_name], flatten=flatten)
            if output_prefix is not None:
                ae_df.set_column_names([output_prefix + n for n in ae_df.get_column_names()], in_place=True)
            ae_se = x.get_alternative_experiment(ae_name)
            ae_se = ae_se.set_column_data(biocutils.combine_columns(ae_se.get_column_data(), ae_df))
            x = x.set_alternative_experiment(ae_name, ae_se)

    if meta_name is not None:
        import copy
        meta = copy.copy(x.get_metadata())
        meta[meta_name] = { "thresholds": thresholds }
        x = x.set_metadata(meta)

    return x


def format_compute_rna_qc_metrics_result(df: biocframe.BiocFrame, flatten: bool = True) -> biocframe.BiocFrame:
    """
    Pretty-format the results of :py:func:`~scranpy.rna_quality_control.compute_rna_qc_metrics`.

    Args:
        df:
            Result of :py:func:`~scranpy.rna_quality_control.compute_rna_qc_metrics`.

        flatten:
            Whether to flatten the nested BiocFrame of subset proportions.

    Returns:
        A BiocFrame containing per-cell QC statistics.
    """

    if not flatten:
        return df

    field = "subset_proportion"
    values = df.get_column(field)
    values = values.set_column_names([field + "_" + n for n in values.get_column_names()])
    return biocutils.combine_columns(df, values)
