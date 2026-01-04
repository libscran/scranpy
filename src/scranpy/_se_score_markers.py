from typing import Optional, Sequence, Union

import biocutils
import biocframe
import summarizedexperiment

from ._score_markers import *


def score_markers_se(
    x: summarizedexperiment.SummarizedExperiment,
    groups: Sequence,
    block: Optional[Sequence] = None,
    num_threads: int = 1,
    more_marker_args: dict = {},
    assay_type: Union[str, int] = "logcounts",
    extra_columns: Optional[Union[Sequence, str, biocutils.NamedList]] = None,
    order_by: Optional[Union[bool, str]] = True 
) -> biocutils.NamedList:
    """
    Identify candidate marker genes based on effect sizes from pairwise comparisons between groups of cells.
    This calls :py:func:`~scranpy.score_markers.score_markers` on an assay of a :py:class:`~summarizedexperiment.SummarizedExperiment.SummarizedExperiment`,
    and then uses :py:func:`~format_score_markers_result` to reformat the results.

    Args:
        x:
            A :py:class:`~summarizedexperiment.SummarizedExperiment.SummarizedExperiment` object or one of its subclasses.
            Rows correspond to genes and columns correspond to cells.

        groups:
            Group assignment for each cell, passed to :py:func:`~scranpy.score_markers.score_markers`.

        block:
            Block assignment for each cell, passed to :py:func:`~scranpy.score_markers.score_markers`.

        num_threads:
            Number of threads for marker scoring, passed to :py:func:`~scranpy.score_markers.score_markers`.

        more_marker_args:
            Additional arguments to pass to :py:func:`~scranpy.score_markers.score_markers`.

        assay_type:
            Name or index of the assay of ``x`` to use for marker detection, usually containing log-normalized expression values.

        extra_columns:
            A :py:class:`~biocframe.BiocFrame.BiocFrame` with the same number of rows as ``x``, containing extra columns to add each DataFrame.
            Alternatively, a list of strings specifying the columns of the row data of ``x`` to be added.
            A single string is treated as a list of length 1.

        order_by:
            Name of the column to use for ordering the rows of each output :py:class:`~biocframe.BiocFrame.BiocFrame`.
            Alternatively ``True``, in which case a column is automatically chosen from the effect size summaries.
            If ``None`` or ``False``, no ordering is performed.

    Returns:
        A :py:class:`~biocutils.NamedList.NamedList` of :py:class:`~biocframe.BiocFrame.BiocFrame`s.
        Each ``BiocFrame`` corresponds to a unique group in ``groups``.
        Each row contains statistics for a gene in ``x``, with the following columns:

        - ``mean``, the mean expression in the current group.
        - ``detected``, the proportion of cells with detected expression in the current group.
        - ``<effect>_<summary>``, a summary statistic for an effect size.
          For example, ``cohens_d_mean`` contains the mean Cohen's d across comparisons involving the current group.

    Examples:
        >>> import scranpy
        >>> sce = scranpy.get_test_rna_data_se("cluster")
        >>> markers = scranpy.score_markers_se(sce, sce.get_column_data()["clusters"])
        >>> print(scranpy.preview_markers(markers["0"]))
    """

    res = score_markers(
        x.get_assay(assay_type),
        groups=groups,
        block=block,
        num_threads=num_threads,
        **more_marker_args
    )

    if extra_columns is not None and not isinstance(extra_columns, biocframe.BiocFrame):
        if isinstance(extra_columns, str):
            extra_columns = [extra_columns]
        extra_columns = x.get_row_data()[:,extra_columns]

    return format_score_markers_result(res, extra_columns=extra_columns, order_by=order_by, row_names=x.get_row_names())


def _find_order_by(df: biocframe.BiocFrame, order_by: Optional[Union[str, bool]]) -> Union[None, str]:
    if isinstance(order_by, bool):
        if order_by:
            # Find something decent to use for ordering.
            for summ in ["mean", "median", "min_rank", "min", "max"]:
                for eff in ["cohens_d", "auc", "delta_mean", "delta_detected"]:
                    candidate = eff + "_" + summ
                    if df.has_column(candidate):
                        return candidate
            return None
        else:
            return None
    else:
        # No-op if it was already NULL or a string.
        return order_by


def _order(x, decreasing):
    if decreasing:
        return numpy.argsort(-x)
    else:
        return numpy.argsort(x)


def format_score_markers_result(
    res,
    extra_columns: Optional[Union[str, Sequence, biocframe.BiocFrame]] = None,
    order_by: Optional[Union[str, bool]] = True,
    row_names: Optional[Sequence] = None
) -> biocutils.NamedList:
    """
    Reformat the output of :py:func:`~scranpy.score_markers.score_markers` into a list of per-group :py:class:`~biocframe.BiocFrame.BiocFrame` objects.

    Args:
        res:
            Results of :py:func:`~scranpy.score_markers.score_markers`.

        extra_columns:
            A :py:class:`~biocframe.BiocFrame.BiocFrame` with the same number of rows as ``x``, containing extra columns to add each ``BiocFrame``.

        order_by:
            Name of the column to use for ordering the rows of each output :py:class:`~biocframe.BiocFrame.BiocFrame`.
            Alternatively ``True``, in which case a column is automatically chosen from the effect size summaries.
            If ``None`` or ``False``, no ordering is performed.

        row_names:
            Sequence of strings containing the row names to add to each ``BiocFrame``.
            This should correspond to the gene names corresponding to the rows of ``x`` used in :py:func:`~scranpy.score_markers.score_markers`. 

    Returns:
        A :py:class:`~biocutils.NamedList.NamedList` of :py:class:`~biocframe.BiocFrame.BiocFrame` objects.
        Each ``BiocFrame`` corresponds to a unique group in ``groups``.
        Each row of each ``BiocFrame`` contains statistics for a gene in ``x``.
        Each ``BiocFrame`` contains the following columns:

        - ``mean``, the mean expression in the current group.
        - ``detected``, the proportion of cells with detected expression in the current group.
        - ``<effect>_<summary>``, a summary statistic for an effect size,
          e.g., ``cohens_d_mean`` contains the mean Cohen's d across comparisons involving the current group.

    Examples:
        >>> import scranpy
        >>> sce = scranpy.get_test_rna_data_se("cluster")
        >>> raw_markers = scranpy.score_markers(sce.get_assay("logcounts"), sce.get_column_data()["clusters"])
        >>> formatted = scranpy.format_score_markers_result(raw_markers, row_names=sce.get_row_names())
    """

    effect_sizes = ["cohens_d", "auc", "delta_mean", "delta_detected"]
    summaries = ["min", "mean", "median", "max", "quantile", "min_rank"]

    NR = res["nrow"]
    group_ids = [str(g) for g in res["group_ids"]] # stringify to match to the column names of various things.

    output = biocutils.NamedList([], [])
    has_order_by = False

    for i, group in enumerate(group_ids):
        current = biocframe.BiocFrame(number_of_rows=NR, row_names=row_names)
        if extra_columns is not None:
            current = biocutils.combine_columns(current, extra_columns)

        for grpstat in ["mean", "detected"]:
            if grpstat in res.get_names():
                current.set_column(grpstat, res[grpstat][:,i], in_place=True)

        for eff in effect_sizes:
            if eff not in res.get_names():
                continue
            eff_df = res[eff][group]
            for summ in summaries:
                if not eff_df.has_column(summ):
                    continue
                eff_summ = eff_df[summ]
                if isinstance(eff_summ, biocframe.BiocFrame):
                    eff_summ = eff_summ.set_column_names([eff + "_" + summ + "_" + y for y in eff_summ.get_column_names()])
                    current = biocutils.combine_columns(current, eff_summ)
                else:
                    current.set_column(eff + "_" + summ, eff_summ, in_place=True)

        if not has_order_by:
            order_by = _find_order_by(current, order_by)
            has_order_by = True
        if order_by is not None:
            dec = not order_by.endswith("_min_rank")
            ordering = _order(current[order_by], decreasing=dec)
            current = current[ordering,:]

        output[group] = current

    return output


def preview_markers(
    df: biocframe.BiocFrame,
    columns: Optional[Sequence] = ["mean", "detected", ('lfc', "delta_mean_mean")],
    rows: Optional[int] = 10,
    order_by: Optional[Union[str, bool]] = None,
    include_order_by: bool = True
) -> biocframe.BiocFrame:
    """
    Preview the top markers for a group in a pretty format.

    Args:
        df:
            :py:class:`~biocframe.BiocFrame.BiocFrame` containing the marker statistics for a single group.

        columns:
            Names of columns of ``df`` to retain in the preview.

            Alternatively, each entry may be a tuple of two strings.
            The first string is the name of the column in the output ``BiocFrame``, and the second string is the name of the column of ``df`` to retain.

        rows:
            Number of rows to show.
            If ``None``, all rows are returned.

        order_by:
            Name of the column to use for ordering the rows of the output :py:class:`~biocframe.BiocFrame.BiocFrame`s.
            Alternatively ``True``, in which case a column is automatically chosen from the effect size summaries.
            If ``None`` or ``False``, no ordering is performed.

        include_order_by:
            Whether to include the column named by ``order_by`` in the output ``BiocFrame``.
            This may also be a string, which is treated as ``True``;
            the value is used as the name of the column in the output ``BiocFrame``. 

    Returns:
        A :py:class:`~biocframe.BiocFrame.BiocFrame` containing important columns for the top markers.

    Examples:
        >>> import scranpy
        >>> sce = scranpy.get_test_rna_data_se("cluster")
        >>> markers = scranpy.score_markers_se(sce, sce.get_column_data()["clusters"])
        >>> print(scranpy.preview_markers(markers["1"]))
    """

    order_by = _find_order_by(df, order_by)

    all_cols = []
    if columns is not None:
        all_cols += columns

    if order_by is not None:
        if isinstance(include_order_by, bool):
            if include_order_by:
                all_cols.append(order_by)
        elif isinstance(include_order_by, str):
            all_cols.append((include_order_by, order_by))

    new_df = biocframe.BiocFrame(number_of_rows=df.shape[0], row_names=df.get_row_names())
    for cn in all_cols:
        if isinstance(cn, tuple):
            new_df.set_column(cn[0], df.get_column(cn[1]), in_place=True)
        else:
            new_df.set_column(cn, df.get_column(cn), in_place=True)

    if order_by is not None:
        dec = not order_by.endswith("_min_rank")
        ordering = _order(df[order_by], decreasing=dec)
        if rows is not None and rows < len(ordering):
            ordering = ordering[:rows]
        new_df = new_df[ordering,:]
    else:
        if rows is not None and rows < new_df.shape[0]:
            new_df = new_df[:rows,:]

    return new_df
