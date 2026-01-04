from typing import Sequence, Optional, Union

import summarizedexperiment

from ._score_gene_set import *


def score_gene_set_se(
    x: summarizedexperiment.SummarizedExperiment,
    set: Sequence,
    block: Optional[Sequence] = None,
    num_threads: int = 1,
    more_score_args: dict = {},
    assay_type: Union[str, int] = "logcounts"
) -> biocutils.NamedList:
    """
    Compute a gene set activity score for each cell based on the expression values of the genes in the set.
    This calls :py:func:`~scranpy.score_gene_set.score_gene_set` on an assay of a :py:class:`~summarized_experiment.SummarizedExperiment.SummarizedExperiment`.

    Args:
        x:
            A :py:class:`~summarizedexperiment.SummarizedExperiment.SummarizedExperiment` object or one of its subclasses.
            Rows correspond to genes and columns correspond to cells.

        set:
            Names or indices of genes in the set, see :py:func:`~scranpy.score_gene_set.score_gene_set` for details.

        block:
            Block assignment for each cell, passed to :py:func:`~scranpy.score_gene_set.score_gene_set`.

        num_threads:
            Number of threads, passed to :py:func:`~scranpy.score_gene_set.score_gene_set`.

        more_score_args:
            Additional arguments to pass to :py:func:`~scranpy.score_gene_set.score_gene_set`.

        assay_type:
            Name or index of the assay of ``x`` from which to compute the gene set scores.

    Returns:
        A :py:class:`~biocutils.NamedList.NamedList` containing per-cell scores and per-gene weights,
        see :py:func:`~scranpy.score_gene_set.score_gene_set` for details.

    Examples:
        >>> import scranpy
        >>> sce = scranpy.get_test_rna_data_se("norm")
        >>> custom_set = [0, 1, 4, 5, 7]
        >>> custom_scores = scranpy.score_gene_set_se(sce, custom_set)
        >>> custom_scores["scores"]
    """

    return score_gene_set(
        x.get_assay(assay_type),
        set=set,
        block=block,
        num_threads=num_threads,
        **more_score_args
    )
