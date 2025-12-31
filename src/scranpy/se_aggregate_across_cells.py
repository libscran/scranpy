from typing import Union, Optional

import summarizedexperiment
import biocframe
import biocutils
from . import _utils_se as seutils
from .aggregate_across_cells import *


def aggregate_across_cells_se(
    x,
    factors: Union[list, dict, biocutils.NamedList],
    num_threads: int = 1,
    more_aggr_args: dict = {},
    assay_type: Union[str, int] = "counts",
    output_prefix: Optional[str] = "factor_",
    counts_name: str = "counts",
    meta_name: Optional[str] = "aggregated",
    include_coldata: bool = True,
    more_coldata_args: dict = {},
    altexps: Optional[Union[str, int, dict, Sequence, biocutils.NamedList]] = None,
    copy_altexps: bool = False
) -> summarizedexperiment.SummarizedExperiment:
    """
    Aggregate expression values across groups of cells for each gene, storing the result in a :py:class:`~summarizedexperiment.SummarizedExperiment.SummarizedExperiment`.
    This calls :py:func:`~aggregate_across_cells` along with :py:func:`~aggregate_column_data`.

    Args:
        x:
            A :py:class:`~summarizedexperiment.SummarizedExperiment.SummarizedExperiment` object or one of its subclasses.
            Rows correspond to genes and columns correspond to cells.

        factors:
            One or more grouping factors, see the argument of the same name in :py:func:`~aggregate_across_cells`.
            This may also be a :py:class:`~biocframe.BiocFrame.BiocFrame` with number of rows equal to the the number of columns in ``x``,
            where each column contains one grouping factor.

        num_threads:
            Passed to :py:func:`~aggregate_across_cells`.

        more_aggr_args:
            Further arguments to pass to :py:func:`~aggregate_across_cells`.

        assay_type:
            Name or index of the assay of ``x`` to be aggregated.

        output_prefix:
            Prefix to add to the names of the columns containing the factor combinations in the column data of the output object.
            If ``None``, no prefix is added.

        counts_name: 
            Name of the column in which to store the cell count for each factor combination in the column data of the output object.
            If ``None``, the cell counts are not reported.

        meta_name:
            Name of the metadata entry in which to store additional information like the combination indices in the output object.
            If ``None``, additional outputs are not reported.

        include_coldata:
            Whether to add the aggregated column data from ``x`` to the output.
            If ``True``, this adds the output of :py:func:`~aggregate_column_data` to the column data of the output object.

        more_coldata_args:
            Additional arguments to pass to :py:func:`~aggregate_column_data`.
            Only relevant if ``include_coldata = True``.

        altexps:
            List of integers or strings, containing the indices or names of alternative experiments of ``x`` to aggregate.
            The aggregated assay from each alternative experiment is determined by ``assay_type``.

            Alternatively, this may be a single integer or string containing the index or name of one alternative experiment to aggregate.
            Again, the aggregated assay from each alternative experiment is determined by ``assay_type``.

            Alternatively, this may be a dictionary where keys are strings and values are integers or strings.
            Each key should be the name of an alternative experiment while each value is the index/name of the assay to aggregate from that experiment.

            Alternatively, a :py:class:`~biocutils.NamedList.NamedList` of integers or strings.
            If named, this is treated as a dictionary, otherwise it is treated as a list.

            Only relevant if ``x`` is a :py:class:`~singlecellexperiment.SingleCellExperiment.SingleCellExperiment` or one of its subclasses.

        copy_altexps:
            Whether to copy the column data and metadata of the output ``SingleCellExperiment`` into each of its alternative experiments.
            Only relevant if ``x`` is a :py:class:`~singlecellexperiment.SingleCellExperiment.SingleCellExperiment` or one of its subclasses.

    Returns:
        A :py:class:`~summarizedexperiment.SummarizedExperiment.SummarizedExperiment` where each column corresponds to a factor combination.
        Each row corresponds to a gene in ``x``, and the row data is taken directly from ``x``.
        The assays contain the sum of counts (``sum``) and the number of detected cells (``detected``) in each combination for each gene.
        The column data contains:

        - The factor combinations, with column names prefixed by ``output_prefix``.
        - The cell count for each combination, named by ``counts_name``.
        - Additional column data from ``x`` if ``include_coldata = True``.
          This is aggregated with :py:func:`~aggregate_column_data`` on the combination indices.

        The metadata contains a ``meta_name`` entry, which is a list with an ``index`` integer vector of length equal to the number of cells in ``x``.
        Each entry of this vector is an index of the factor combination (i.e., column of the output object) to which the corresponding cell was assigned.

        If ``altexps`` is specified, a :py:class:`~singlecellexperiment.SingleCellExperiment.SingleCellExperiment` is returned instead.
        The same aggregation described above for the main experiment is applied to each alternative experiment.
        If ``copy_altexps = True``, the column data for each alternative experiment will contain a copy of the factor combinations and cell counts,
        and the ``metadata`` will contain a copy of the index vector.

    Examples:
        >>> import scranpy
        >>> sce = scranpy.get_test_rna_data_se("start")
        >>> aggr = scranpy.aggregate_across_cells_se(sce, factors=[sce.get_column_data()["level1class"]])
        >>> aggr.get_assay(0)[:5,]
        >>> print(aggr.get_column_data())
        >>> 
        >>> # We can also aggregate within alternative experiments.
        >>> aggr2 = scranpy.aggregate_across_cells_se(sce, factors=[sce.get_column_data()["level1class"]], altexps=["ERCC"])
        >>> aggr2.get_alternative_experiment("ERCC").get_assay(0)[:5,]
    """

    if isinstance(factors, biocframe.BiocFrame):
        factors = biocutils.NamedList.from_dict(factors.get_data())

    out = aggregate_across_cells(
        x.assay(assay_type),
        factors=factors,
        num_threads=num_threads,
        **more_aggr_args
    )

    CON = summarizedexperiment.SummarizedExperiment
    use_sce = altexps is not None and len(altexps) > 0
    if use_sce:
        import singlecellexperiment
        CON = singlecellexperiment.SingleCellExperiment
    se = CON({ "sum": out["sum"], "detected": out["detected"] }, row_data = x.get_row_data())

    common_cd = out["combinations"]
    if output_prefix is not None:
        common_cd.set_column_names([(output_prefix + y) for y in common_cd.get_column_names()], in_place=True)
    if counts_name is not None:
        common_cd.set_column(counts_name, out["counts"], in_place=True)

    full_cd = common_cd
    if include_coldata:
        aggr_cd = aggregate_column_data(x.get_column_data(), out["index"], number=common_cd.shape[0], **more_coldata_args)
        full_cd = biocutils.combine_columns(common_cd, aggr_cd)
    se.set_column_data(full_cd, in_place=True)

    if meta_name is not None:
        import copy
        meta = copy.copy(se.get_metadata())
        meta[meta_name] = { "index": out["index"] }
        se.set_metadata(meta, in_place=True)

    if use_sce:
        se.set_main_experiment_name(x.get_main_experiment_name(), in_place=True)
        altexps = seutils.sanitize_altexp_assays(altexps, x.get_alternative_experiment_names(), default_assay_type=assay_type)

        for ae, ae_assay in altexps.items():
            ae_se = aggregate_across_cells_se(
                x.alternative_experiment(ae),
                [out["index"]],
                num_threads=num_threads,
                more_aggr_args=more_aggr_args,
                assay_type=ae_assay,
                altexps=None,
                output_prefix=None,
                counts_name=None,
                meta_name=None,
                include_coldata=include_coldata
            )

            ae_cd = ae_se.get_column_data()
            ae_cd.remove_column(0, in_place=True)
            if copy_altexps:
                ae_cd = biocutils.combine_columns(common_cd, ae_cd)
            ae_se.set_column_data(ae_cd, in_place=True)

            if copy_altexps:
                ae_se.set_metadata(se.get_metadata(), in_place=True)
            se.set_alternative_experiment(ae, ae_se, in_place=True)

    return se


def aggregate_column_data(coldata: biocframe.BiocFrame, index: Sequence, number: int, only_simple: bool = True, placeholder = None) -> biocframe.BiocFrame:
    """
    Aggregate the column data from a :py:class:`~summarizedexperiment.SummarizedExperiment.SummarizedExperiment` for groups of cells.

    Args:
        coldata:
            A :py:class:`~biocframe.BiocFrame.BiocFrame` containing the column data for a SummarizedExperiment.
            Each row should correspond to a cell.

        index:
            Vector of length equal to the number of cells.
            Each entry should be the index of the factor combination to which each cell in ``coldata`` was assigned,
            e.g., the index vector produced by :py:func:`~scranpy.combine_factors.combine_factors`.

        number:
            Total number of unique factor combinations.
            All elements of ``index`` should be less than ``number``.

        only_simple:
            Whether to skip columns of ``coldata`` that are not lists, NumPy arrays, :py:class:`~biocutils.NamedList.NamedList`s or :py:class:`~biocutils.Factor.Factor`s.

        placeholder:
            Placeholder value to store in the output column when a factor combination does not have a single unique value. 

    Returns:
        A :py:class:`~biocframe.BiocFrame.BiocFrame` with number of rows equal to ``number``.
        Each "simple" column in ``coldata`` (i.e., list, NumPy array, NamedList or Factor) is represented by a column in the output BiocFrame.
        In each column, the ``j``-th entry is equal to the unique value of all rows where ``index == j``, or ``placeholder`` if there is not exactly one unique value.
        If ``only_simple = False``, any non-simple columns of ``coldata`` are represented in the output BiocFrame by a list of ``placeholder``values.
        Otherwise, if ``only_simple = True``, any non-simple columns of ``coldata`` are skipped.
    """

    collected = biocframe.BiocFrame(number_of_rows=number)

    for cn in coldata.get_column_names():
        curcol = coldata.get_column(cn)
        if not isinstance(curcol, list) and not isinstance(curcol, biocutils.NamedList) and not isinstance(curcol, numpy.ndarray) and not isinstance(curcol, biocutils.Factor):
            if not only_simple:
                collected.set_column(cn, [placeholder] * number, in_place=True)
            continue

        alloc = []
        for n in range(number):
            alloc.append(set())

        for i, val in enumerate(curcol):
            g = index[i]
            if alloc[g] is not None:
                try: 
                    alloc[g].add(val)
                except:
                    alloc[g] = None

        for n in range(number):
            if len(alloc[n]) == 1:
                alloc[n] = list(alloc[n])[0]
            else:
                alloc[n] = None

        if isinstance(curcol, biocutils.NamedList):
            alloc = type(curcol)(alloc)
        elif isinstance(curcol, biocutils.Factor):
            alloc = biocutils.Factor.from_sequence(alloc, levels=curcol.get_levels()) 

        collected.set_column(cn, alloc, in_place=True)

    return collected
