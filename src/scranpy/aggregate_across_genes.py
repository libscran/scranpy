from typing import Any, Sequence, Union, Optional

import numpy
import mattress
import biocutils

from . import lib_scranpy as lib


def aggregate_across_genes(
    x: Any,
    sets: Sequence,
    average: bool = False,
    num_threads: int = 1
) -> biocutils.NamedList:
    """Aggregate expression values across genes, potentially with weights.
    This is typically used to summarize expression values for gene sets into a single per-cell score.

    Args:
        x:
            Matrix-like object where rows correspond to genes or genomic features and columns correspond to cells. 
            Values are expected to be log-expression values.

        sets:
            Sequence of integer arrays containing the row indices of genes in each set.
            Alternatively, each entry may be a tuple of length 2, containing an integer vector (row indices) and a numeric vector (weights).
            If this is a :py:class:`~biocutils.NamedList.NamedList`, the names will be preserved in the output.

        average:
            Whether to compute the average rather than the sum.

        num_threads: 
            Number of threads to be used for aggregation.

    Returns:
        List of length equal to that of ``sets``.
        Each entry is a numeric vector of length equal to the number of columns in ``x``,
        containing the (weighted) sum/mean of expression values for the corresponding set across all cells.

    References:
        The ``aggregate_across_genes`` function in the `scran_aggregate <https://libscran.github.io/scran_aggregate>`_ C++ library, which implements the aggregation.
    """ 
    new_sets = [] 
    for s in sets:
        if isinstance(s, tuple):
            new_sets.append((
                numpy.array(s[0], copy=None, order="A", dtype=numpy.uint32),
                numpy.array(s[1], copy=None, order="A", dtype=numpy.float64)
            ))
        else:
            new_sets.append(numpy.array(s, copy=None, order="A", dtype=numpy.uint32))

    mat = mattress.initialize(x)
    output = lib.aggregate_across_genes(
        mat.ptr,
        new_sets,
        average,
        num_threads
    )

    names = None
    if isinstance(sets, biocutils.NamedList):
        names = sets.get_names()
    return biocutils.NamedList(output, names)


if biocutils.package_utils.is_package_installed("summarizedexperiment"):
    import summarizedexperiment
    import biocframe
    from . import _utils_se as seutils


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
        This calls :py:func:`~aggregate_across_cells` on an assay from a :py:class:`~summarizedexperiment.SummarizedExperiment.SummarizedExperiment`.

        Args:
            x:
                A :py:class:`~summarizedexperiment.SummarizedExperiment.SummarizedExperiment` object or one of its subclasses.
                Rows correspond to genes and columns correspond to cells.

            sets:
                Sequence of integer arrays or tuples containing the row indices of genes in each set, see :py:func:`~aggregate_across_cells` for details.
                Each entry may also be a `~biocframe.BiocFrame.BiocFrame`, in which case the first and second columns are assumed to containg the row indices and weights, respectively.

            num_threads:
                Passed to :py:func:`~aggregate_across_cells`.

            more_aggr_args:
                Further arguments to pass to :py:func:`~aggregate_across_cells`.

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
        """

        sets = seutils.to_NamedList(sets)
        for i in range(len(sets)):
            current = sets[i]
            if isinstance(current, biocframe.BiocFrame):
                sets[i] = (current[0], current[1])

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
            row_data = biocframe.BiocFrame(number_of_rows=len(sets), row_names=sets.get_names())
        )
