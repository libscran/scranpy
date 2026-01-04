from typing import Optional, Sequence

import numpy
import biocutils
import biocframe

from . import lib_scranpy as lib


def _fix_summary_quantiles(payload: dict, ngenes: int, qin: Optional[Sequence[float]]): 
    if qin is not None:
        combined = biocframe.BiocFrame(number_of_rows=ngenes)
        qout = payload["quantile"]
        for i, s in enumerate(qin):
            combined.set_column(str(s), qout[i], in_place=True)
        payload["quantile"] = combined 


def summarize_effects(
    effects: numpy.ndarray,
    compute_min: bool = True,
    compute_mean: bool = True,
    compute_median: bool = True,
    compute_max: bool = True,
    compute_quantiles: Optional[Sequence] = None,
    compute_min_rank: bool = True,
    num_threads: int = 1
) -> biocutils.NamedList: 
    """For each group, summarize the effect sizes for all pairwise comparisons
    to other groups. This yields a set of summary statistics that can be used
    to rank marker genes for each group.

    Args:
        effects:
            A 3-dimensional numeric containing the effect sizes from each pairwise comparison between groups.
            The extents of the first two dimensions should be equal to the number of groups, while the extent of the final dimension is equal to the number of genes. 
            The entry ``[i, j, k]`` should represent the effect size from the comparison of group ``j`` against group ``i`` for gene ``k``.
            See also the output of :py:func:`~scranpy.score_markers.score_markers` with ``all_pairwise = True``.

        compute_min:
            Whether to compute the minimum as a summary statistic for each effect size.

        compute_mean:
            Whether to compute the mean as a summary statistic for each effect size.

        compute_median:
            Whether to compute the median as a summary statistic for each effect size.

        compute_max:
            Whether to compute the maximum as a summary statistic for each effect size.

        compute_quantiles:
            Probabilities of quantiles to compute as summary statistics for each effect size.
            This should be in [0, 1] and sorted in order of increasing size.
            If ``None``, no quantiles are computed.

        compute_min_rank:
            Whether to compute the mininum rank as a summary statistic for each effect size.
            If ``None``, no quantiles are computed.

        num_threads:
            Number of threads to use.

    Returns:
        A :py:class:`~biocutils.NamedList.NamedList` of length equal to the number of groups (i.e., the extents of the first two dimensions of ``effects``).
        Each entry is a :py:class:`~biocframe.BiocFrame.BiocFrame` where each row corresponds to a gene in ``effects``.
        Each column contain a summary statistic of the effect sizes of the comparisons involving its corresponding group.

        - ``min``: double-precision NumPy array containing the minimum effect size across all pairwise comparisons to other groups.
          Only present if ``compute_min = true``.
        - ``median``: double-precision NumPy array containing the median effect size across all pairwise comparisons to other groups.
          Only present if ``compute_median = True``.
        - ``mean``: double-precision NumPy array containing the mean effect size from across all pairwise comparisons to other groups.
          Only present if ``compute_median = True``.
        - ``quantile``: nested :py:class:`~biocframe.BiocFrame.BiocFrame` containing the specified quantiles of the effect sizes across all pairwise comparisons to other groups.
          Only present if ``compute_quantiles`` is provided.
        - ``max``: double-precision NumPy array containing the maximum effect size across all pairwise comparisons to other groups.
          Only present if ``compute_max = true``.
        - ``min_rank``: integer array containing the minimum rank of each gene across all pairwise comparisons to other groups.
          Only present if ``compute_min_rank = true``.

    References:
        The ``summarize_effects`` function in the `scran_markers <https://libscran.github.io/scran_markers>`_ C++ library, for more details on the statistics.

    Examples:
        >>> import numpy
        >>> normed = numpy.random.rand(200, 100)
        >>> import scranpy
        >>> group = ["A", "B", "C", "D"] * 25
        >>> res = scranpy.score_markers(normed, group, all_pairwise=True)
        >>> summaries = scranpy.summarize_effects(res["cohens_d"])
        >>> print(summaries[0])
    """

    ngenes = effects.shape[2]
    if compute_quantiles is not None:
        compute_quantiles = numpy.array(compute_quantiles, dtype=numpy.dtype("double"))

    results = lib.summarize_effects(
        effects,
        compute_min,
        compute_mean,
        compute_median,
        compute_max,
        compute_quantiles,
        compute_min_rank,
        num_threads
    )

    output = biocutils.NamedList() 
    for val in results:
        _fix_summary_quantiles(val, ngenes, compute_quantiles)
        output.append(biocframe.BiocFrame(val, number_of_rows=ngenes))

    return output
