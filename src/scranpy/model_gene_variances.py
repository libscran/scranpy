from typing import Optional, Any, Sequence, Literal, Tuple

import biocutils
import mattress
import biocframe
import numpy

from . import lib_scranpy as lib


def model_gene_variances(
    x: Any,
    block: Optional[Sequence] = None,
    block_average_policy: Literal["mean", "quantile"] = "mean",
    block_weight_policy: Literal["variable", "equal", "none"] = "variable",
    variable_block_weight: Tuple = (0, 1000),
    block_quantile: float = 0.8,
    mean_filter: bool = True,
    min_mean: float = 0.1, 
    transform: bool = True, 
    span: float = 0.3,
    use_min_width: bool = False,
    min_width: float = 1,
    min_window_count: int = 200,
    num_threads: int = 1
) -> biocframe.BiocFrame:
    """
    Compute the variance in (log-)expression values for each gene, and model the trend in the variances with respect to the mean.

    Args:
        x:
            A matrix-like object where rows correspond to genes or genomic features and columns correspond to cells.
            It is typically expected to contain log-expression values, e.g., from :py:func:`~scranpy.normalize_counts.normalize_counts`.

        block:
            Array of length equal to the number of columns of ``x``, containing the block of origin (e.g., batch, sample) for each cell.
            Alternatively ``None``, if all cells are from the same block.

        block_average_policy:
            Policy for averaging statistics across blocks.
            This can either use a (weighted) ``mean`` or a ``quantile``.

        block_weight_policy:
            Policy for weighting different blocks when computing the weighted mean across blocks for each statistic.
            Only used if ``block`` is provided and ``block_average_policy == "mean"``.

        variable_block_weight:
            Parameters for variable block weighting.
            This should be a tuple of length 2 where the first and second values are used as the lower and upper bounds, respectively, for the variable weight calculation.
            Only used if ``block`` is provided, ``block_average_policy == "mean"``, and ``block_weight_policy = "variable"``.

        block_quantile:
            Probability for computing the quantile across blocks.
            Defaults to 0.5, i.e., the median of per-block statistics.
            Only used if ``block`` is provided and ``block_average_policy == "quantile"``.

        mean_filter:
            Whether to filter on the means before trend fitting.

        min_mean:
            The minimum mean of genes to use in trend fitting.
            Only used if ``mean_filter = True``.

        transform:
            Whether a quarter-root transformation should be applied before trend fitting.

        span:
            Span of the LOWESS smoother for trend fitting, see :py:func:`~scranpy.fit_variance_trend.fit_variance_trend`.

        use_min_width:
            Whether a minimum width constraint should be applied during trend fitting, see :py:func:`~scranpy.fit_variance_trend.fit_variance_trend`.

        min_width:
            Minimum width of the smoothing window for trend fitting, see :py:func:`~scranpy.fit_variance_trend.fit_variance_trend`.

        min_window_count:
            Minimum number of observations in each smoothing window for trend fitting, see :py:func:`~scranpy.fit_variance_trend.fit_variance_trend`.

        num_threads:
            Number of threads to use.

    Returns:
        A `:py:class:`~biocutils.NamedList.NamedList` containing `statistics`.
        This is a :py:class:`~biocframe.BiocFrame.BiocFrame` with one row per gene and the following columns:

        - ``mean``: a double-precision NumPy array containing the mean (log-)expression for each gene.
        - ``variance``: a double-precision NumPy array containing the mean (log-)expression for each gene.
        - ``fitted``: a double-precision NumPy array containing the fitted value of the mean-variance trend for each gene.
        - ``residual``: a double-precision NumPy array containing the residual from the mean-variance trend for each gene.

        If ``block`` is supplied, the NamedList will also contain:

        - ``per_block``: a :py:class:`~biocutils.NamedList.NamedList` containing the per-block statistics.
          Each entry is a BiocFrame that contains the ``mean``, ``variance``, ``fitted`` and ``residual`` for each block.
        - ``block_ids``: a list containing the identities of the blocks.
          This corresponds to the entries of ``per_block``.

    References:
        The ``model_gene_variances`` function in the `scran_variances <https://libscran.github.io/scran_variances>`_ C++ library. 

    Examples:
        >>> import numpy
        >>> mu = numpy.random.rand(200) * 5
        >>> counts = numpy.ndarray((200, 10))
        >>> for c in range(10):
        >>>     counts[:,c] = numpy.random.poisson(lam=mu, size=200)
        >>> 
        >>> import scranpy
        >>> res = scranpy.model_gene_variances(counts)
    """

    if block is None:
        blocklev = [] 
        blockind = None
    else:
        blocklev, blockind = biocutils.factorize(block, sort_levels=True, dtype=numpy.uint32, fail_missing=True)

    ptr = mattress.initialize(x)
    res = lib.model_gene_variances(
        ptr.ptr,
        blockind,
        len(blocklev),
        block_average_policy,
        block_weight_policy,
        variable_block_weight,
        block_quantile,
        mean_filter,
        min_mean,
        transform,
        span,
        use_min_width,
        min_width,
        min_window_count,
        num_threads
    )

    output = biocutils.NamedList([ biocframe.BiocFrame(res["statistics"]) ], [ "statistics" ])

    if "per_block" in res:
        pb = biocutils.NamedList()
        for b, binfo in enumerate(res["per_block"]):
            bdf = biocframe.BiocFrame({ "mean": binfo[0], "variance": binfo[1], "fitted": binfo[2], "residual": binfo[3] })
            pb[str(blocklev[b])] = bdf
        output["per_block"] = pb
        output["block_ids"] = blocklev

    return output
