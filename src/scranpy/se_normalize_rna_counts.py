from typing import Optional, Sequence, Literal

import summarizedexperiment
import biocutils

from .center_size_factors import *
from .normalize_counts import *


def normalize_rna_counts_se(
    x: summarizedexperiment.SummarizedExperiment,
    size_factors: Optional[Sequence] = None,
    center: bool = True,
    block: Optional[Sequence] = None,
    mode: Literal["lowest", "per-block"] = "lowest",
    log: bool = True,
    pseudo_count: float = 1,
    assay_type: Union[str, int] = "counts",
    output_name: str = "logcounts",
    factor_name: Optional[str] = "size_factor"
) -> summarizedexperiment.SummarizedExperiment:
    """
    Compute (log-)normalized expression values after performing scaling normalization of an RNA count matrix.

    Args:
        x:
            A :py:class:`~summarizedexperiment.SummarizedExperiment.SummarizedExperiment` object or one of its subclasses.
            Rows correspond to genes and columns correspond to cells.

        size_factors:
            Size factor for each cell in ``x``, to be passed to :py:func:`~scranpy.normalize_counts.normalize_counts`.
            If ``None``, this defaults to the column sums of the count matrix in ``x``.

        center:
            Whether to center the ``size_factors`` by passing them to :py:func:`~scranpy.center_size_factors.center_size_factors`.

        block:
            Block assignment for each cell in ``x``, to be passed to :py:func:`~scranpy.center_size_factors.center_size_factors`.

        mode:
            How to scale the size factors across blocks, see :py:func:`~scranpy.center_size_factors.center_size_factors`.
            Only relevant if ``block`` is provided.

        log:
            Whether to log-transform the normalized expression values, see :py:func:`~scranpy.normalize_counts.normalize_counts`.

        pseudo_count:
            Pseudo-count for the log-transformation, see :py:func:`~scranpy.normalize_counts.normalize_counts`.
            Only relevant if ``log = True``.

        assay_type:
            Name or index specifying the assay of ``x`` that contains the count matrix to be normalized.

        output_name:
            Name of the assay in which to store the normalized matrix in the output object.

        factor_name:
            Name of the column of the column data in which to store the size factors in the output object.
            If ``None``, the size factors are not stored. 

    Returns:
        A copy of ``x``, with an additional assay containing the (log-)normalized matrix.
        Size factors are also stored in the column data.
    """

    y = x.get_assay(assay_type)

    if size_factors is None:
        size_factors = y.sum(axis=0)
    if center:
        size_factors = center_size_factors(size_factors, block=block, mode=mode)

    norm = normalize_counts(y, size_factors=size_factors, log=log, pseudo_count=pseudo_count)
    x = x.set_assay(output_name, norm)

    if factor_name is not None:
        df = x.get_column_data()
        df = df.set_column(factor_name, size_factors)
        x = x.set_column_data(df)

    return x
