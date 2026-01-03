from typing import Optional, Union, Sequence

import summarizedexperiment
import biocframe
import biocutils
from . import _utils_general as gutils
from .aggregate_across_genes import *


def aggregate_across_genes_se(
    x: summarizedexperiment.SummarizedExperiment,
    sets: Sequence,
    num_threads: int = 1,
    more_aggr_args: dict = {},
    assay_type: Union[str, int] = "logcounts",
    output_name: Optional[str] = None
) -> summarizedexperiment.SummarizedExperiment:
    """
    Aggregate expression values across sets of genes for each cell.
    This calls :py:func:`~scranpy.aggregate_across_genes.aggregate_across_genes` on an assay from a :py:class:`~summarizedexperiment.SummarizedExperiment.SummarizedExperiment`.

    Args:
        x:
            A :py:class:`~summarizedexperiment.SummarizedExperiment.SummarizedExperiment` object or one of its subclasses.
            Rows correspond to genes and columns correspond to cells.

        sets:
            Sequence of gene sets, see :py:func:`~scranpy.aggregate_across_genes.aggregate_across_genes` for details.

        num_threads:
            Passed to :py:func:`~scranpy.aggregate_across_genes.aggregate_across_genes`.

        more_aggr_args:
            Further arguments to pass to :py:func:`~scranpy.aggregate_across_genes.aggregate_across_genes`.

        assay_type:
            Name or index of the assay of ``x`` to be aggregated across genes.

        output_name:
            String specifying the assay name of the aggregated values in the output object.
            If ``None``, it defaults to ``assay_type`` if that argument is a string, otherwise it is set to ``"aggregated"``.

    Returns:
        A :py:class:`~summarizedexperiment.SummarizedExperiment.SummarizedExperiment` with number of rows equal to the number of gene sets.
        The lone assay contains the aggregated values for each gene set for all cells.
        The column data is the same as that of ``x``.
        If ``sets`` is named, the names are used as the row names of the output.

    Examples:
        >>> import scranpy
        >>> sce = scranpy.get_test_rna_data_se("norm")
        >>> sets = { "foo": [ 0, 2, 5, 10 ], "bar": [ 1, 3, 11, 17, 23 ] }
        >>> aggregated = scranpy.aggregate_across_genes_se(sce, sets)
        >>> aggregated.get_assay(0)[:,:10]
    """

    vecs = aggregate_across_genes(
        x.get_assay(assay_type),
        sets,
        num_threads=num_threads,
        **more_aggr_args
    )

    output = numpy.ndarray((len(vecs), x.shape[1]))
    for i, val in enumerate(vecs):
        output[i,:] = val

    if output_name is None:
        if isinstance(assay_type, str):
            output_name = assay_type
        else:
            output_name = "aggregated"
    assays = {}
    assays[output_name] = output

    return summarizedexperiment.SummarizedExperiment(
        assays,
        column_data = x.get_column_data(),
        row_data = biocframe.BiocFrame(number_of_rows=len(sets), row_names=vecs.get_names())
    )
