from typing import Optional, Sequence, Union

import summarizedexperiment

from ._crispr_quality_control import *



def quick_crispr_qc_se(
    x: summarizedexperiment.SummarizedExperiment,
    more_suggest_args: dict = {},
    num_threads: int = 1,
    block: Optional[Sequence] = None,
    assay_type: Union[int, str] = "counts",
    output_prefix: Optional[str] = None,
    meta_name: str = "qc"
) -> summarizedexperiment.SummarizedExperiment:
    """
    Quickly compute quality control (QC) metrics, thresholds and filters from CRISPR data in a :py:class:`~summarizedexperiment.SummarizedExperiment.SummarizedExperiment`.
    This calls :py:func:`~scranpy.compute_crispr_qc_metrics`, :py:func:`~scranpy.suggest_crispr_qc_thresholds`, and :py:func:`~scranpy.filter_crispr_qc_metrics`.

    Args:
        x:
            A :py:class:`~summarizedexperiment.SummarizedExperiment.SummarizedExperiment` or one of its subclasses.
            Rows correspond to antibody-derived tags (ADTs) and columns correspond to cells.

        subsets:
            Passed to :py:func:`~scranpy.compute_crispr_qc_metrics`.

        more_suggest_args:
            Additional arguments to pass to :py:func:`~scranpy.suggest_crispr_qc_thresholds`.

        num_threads:
            Passed to :py:func:`~scranpy.compute_crispr_qc_metrics`.

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

    Returns:
        ``x``, with additional columns added to its column data.
        Each column contains per-cell values for one of the ADT-related QC metrics, see :py:func:`~scranpy.compute_crispr_qc_metrics` for details.
        The suggested thresholds are stored as a list in the metadata.
        The column data also contains a ``keep`` column, specifying which cells are to be retained.

    Examples:
        >>> import scranpy
        >>> sce = scranpy.get_test_crispr_data_se().get_alternative_experiment("CRISPR Guide Capture")
        >>> sce = scranpy.quick_crispr_qc_se(sce)
        >>> print(sce.get_column_data()[:,["sum", "detected", "max_value", "max_index"]])
        >>> print(sce.get_metadata()["qc"]["thresholds"])
        >>> sce.get_column_data()["keep"].sum()
    """

    metrics = compute_crispr_qc_metrics(x.get_assay(assay_type), num_threads=num_threads)
    thresholds = suggest_crispr_qc_thresholds(metrics, block=block, **more_suggest_args)
    keep = filter_crispr_qc_metrics(thresholds, metrics, block=block)

    df = format_compute_crispr_qc_metrics_result(metrics)
    df.set_column("keep", keep, in_place=True)
    if output_prefix is not None:
        df.set_column_names([output_prefix + n for n in df.get_column_names()], in_place=True)

    x = x.set_column_data(biocutils.combine_columns(x.get_column_data(), df))
    if meta_name is not None:
        new_meta = x.get_metadata().set_value(meta_name, { "thresholds": thresholds })
        x = x.set_metadata(new_meta)

    return x


def format_compute_crispr_qc_metrics_result(df: biocframe.BiocFrame) -> biocframe.BiocFrame:
    """
    Pretty-format the results of :py:func:`~scranpy.compute_crispr_qc_metrics`.

    Args:
        df:
            Result of :py:func:`~scranpy.compute_crispr_qc_metrics`.

    Returns:
        A ``BiocFrame`` containing per-cell QC statistics.

    Examples:
        >>> import scranpy
        >>> sce = scranpy.get_test_crispr_data_se().get_alternative_experiment("CRISPR Guide Capture")
        >>> df = scranpy.compute_crispr_qc_metrics(sce.get_assay(0))
        >>> print(scranpy.format_compute_crispr_qc_metrics_result(df))
    """

    return df
