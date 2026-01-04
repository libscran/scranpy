from typing import Optional, Sequence, Union

import numpy
import biocutils
import summarizedexperiment

from ._model_gene_variances import *
from ._choose_highly_variable_genes import *


def choose_rna_hvgs_se( 
    x: summarizedexperiment.SummarizedExperiment, 
    block: Optional[Sequence] = None,
    num_threads: int = 1,
    more_var_args: dict = {},
    top: int = 4000,
    more_choose_args: dict = {},
    assay_type: Union[str, int] = "logcounts",
    output_prefix: Optional[str] = None,
    include_per_block: bool = False 
) -> summarizedexperiment.SummarizedExperiment:
    """
    Model the mean-variance relationship across genes and choose highly variable genes (HVGs) based on the residuals of the fitted trend.

    Args:
        x:
            A :py:class:`~summarizedexperiment.SummarizedExperiment.SummarizedExperiment` object or any of its subclasses.
            Rows correspond to genes and columns correspond to cells.

        block:
            Block assignment for each cell in ``x``, passed to :py:func:`~scranpy.model_gene_variances.model_gene_variances`.

        num_threads:
            Number of threads, passed to :py:func:`~scranpy.model_gene_variances.model_gene_variances`.

        more_var_args:
            Additional arguments to pass to :py:func:`~scranpy.model_gene_variances.model_gene_variances`.

        top:
            Number of top genes to use as HVGs, passed to :py:func:`~scranpy.choose_highly_variable_genes.choose_highly_variable_genes`.

        more_choose_args:
            Additional arguments to pass to :py:func:`~scranpy.choose_highly_variable_genes.choose_highly_variable_genes`.

        assay_type:
            Name or index of the assay of ``x`` containing the log-expression data to use for computing variances.

        output_prefix:
            Prefix to add to the names of the columns of the row data in which to store the output statistics.

        include_per_block:
            Whether the per-block statistics should be stored in the row data of the output object.
            Only relevant if ``block`` is specified.

    Returns:
        A copy of ``x``, with per-gene variance modelling statistics added to its row data.
        An ``hvg`` column indicates the genes that were chosen as HVGs.
        If ``include_per_block = True`` and ``block`` is specified,
        the per-block statistics are stored as a nested :py:class:`~biocframe.BiocFrame.BiocFrame` in the ``per_block`` column.

    Examples:
        >>> import scranpy 
        >>> sce = scranpy.get_test_rna_data_se("norm")
        >>> sce = scranpy.choose_rna_hvgs_se(sce, more_var_args={ "use_min_width": True })
        >>> import biocutils
        >>> print(biocutils.table(sce.get_row_data()["hvg"]))
    """

    info = model_gene_variances(
        x.get_assay(assay_type),
        block=block,
        num_threads=num_threads,
        **more_var_args
    )

    hvg_index = choose_highly_variable_genes(
        info["statistics"]["residual"],
        top=top,
        larger=True,
        **more_choose_args
    )

    df = info["statistics"]
    df.set_row_names(x.get_row_names())

    if include_per_block and "per_block" in info.get_names():
        pbinfo = info["per_block"]
        pbout = biocframe.BiocFrame(number_of_rows=x.shape[0])
        for bname in pbinfo.get_names():
            pbout.set_column(bname, pbinfo[bname], in_place=True)
        df.set_column("per_block", pbout, in_place=True)

    keep = numpy.ndarray(x.shape[0], numpy.dtype("bool"))
    keep[:] = False
    keep[hvg_index] = True
    df.set_column("hvg", keep, in_place=True)

    if output_prefix is not None:
        df.set_column_names([output_prefix + y for y in df.get_column_names()], in_place=True)

    return x.set_row_data(biocutils.combine_columns(x.get_row_data(), df))
