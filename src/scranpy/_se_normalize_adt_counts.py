from typing import Optional, Union

import summarizedexperiment

from ._compute_clrm1_factors import *
from ._center_size_factors import *
from ._normalize_counts import *


def normalize_adt_counts_se(
    x: summarizedexperiment.SummarizedExperiment,
    size_factors: Optional[Sequence] = None,
    num_threads: int = 1,
    center: bool = True,
    block: Optional[Sequence] = None,
    mode: Literal["lowest", "per-block"] = "lowest",
    log: bool = True,
    pseudo_count: float = 1,
    assay_type: Union[int, str] = "counts",
    output_name: str = "logcounts",
    factor_name: Optional[str] = "size_factor"
) -> summarizedexperiment.SummarizedExperiment:
    """
    Compute (log-)normalized expression values after performing scaling normalization of an ADT count matrix.
    This calls :py:func:`~scranpy.compute_clrm1_factors.compute_clrm1_factors` to compute CLRm1 size factors,
    :py:func:`~scranpy.center_size_factors.center_size_factors` to center the size factors,
    and then :py:func:`~scranpy.normalize_counts.normalize_counts` to compute normalized log-expression values.

    Args:
        x:
            A :py:class:`~summarizedexperiment.SummarizedExperiment.SummarizedExperiment` object or one of its subclasses.
            Rows correspond to antibody-derived tags (ADTs) and columns correspond to cells.

        size_factors:
            Size factor for each cell in ``x``, to be passed to :py:func:`~scranpy.normalize_counts.normalize_counts`.
            If ``None``, size factors are computed with :py:func:`~scranpy.compute_clrm1_factors.compute_clrm1_factors`.

        num_threads:
            Number of threads, to be passed to :py:func:`~scranpy.normalize_counts.normalize_counts`.

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

    Examples:
        >>> import scranpy
        >>> sce = scranpy.get_test_adt_data_se("qc").get_alternative_experiment("ADT")
        >>> sce = scranpy.normalize_adt_counts_se(sce)
        >>> sce.get_assay_names()
        >>> sce.get_column_data()["size_factor"]
    """

    y = x.get_assay(assay_type)

    if size_factors is None:
        size_factors = compute_clrm1_factors(y, num_threads=num_threads)
    else:
        size_factors = numpy.asarray(size_factors, dtype=numpy.dtype("double"))
    if center:
        size_factors = center_size_factors(size_factors, block=block, mode=mode)

    y = normalize_counts(y, size_factors=size_factors, log=log, pseudo_count=pseudo_count)
    x = x.set_assay(output_name, y) 

    if factor_name is not None:
        df = x.get_column_data()
        df = df.set_column(factor_name, size_factors)
        x = x.set_column_data(df)

    return x
