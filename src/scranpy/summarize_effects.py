from typing import Optional, Sequence
from dataclasses import dataclass

import numpy
import biocutils

from . import lib_scranpy as lib


def _fix_summary_quantiles(payload: dict, qin: Optional[Sequence[float]]): 
    if qin is not None:
        payload["quantiles"] = biocutils.NamedList(payload["quantiles"], [str(x) for x in qin])


@dataclass
class GroupwiseSummarizedEffects:
    """Summarized effect sizes for a single group, typically created by :py:func:`~summarize_effects` or :py:func:`~scranpy.score_markers.score_markers`."""

    min: Optional[numpy.ndarray]
    """
    Floating-point array of length equal to the number of genes.
    Each entry is the minimum effect size for that gene from all pairwise comparisons to other groups.
    Alternatively ``None``, if the minimum was not computed.
    """

    mean: Optional[numpy.ndarray]
    """
    Floating-point array of length equal to the number of genes.
    Each entry is the mean effect size for that gene from all pairwise comparisons to other groups.
    Alternatively ``None``, if the mean was not computed.
    """

    median: Optional[numpy.ndarray]
    """
    Floating-point array of length equal to the number of genes.
    Each entry is the median effect size for that gene from all pairwise comparisons to other groups.
    Alternatively ``None``, if the mean was not computed.
    """

    max: Optional[numpy.ndarray]
    """
    Floating-point array of length equal to the number of genes.
    Each entry is the maximum effect size for that gene from all pairwise comparisons to other groups.
    Alternatively ``None``, if the mean was not computed.
    """

    quantiles: Optional[biocutils.NamedList]
    """
    Named list of floating point arrays of length equal to the number of quantiles.
    Each entry of each array contains a quantile of the effect sizes across all pairwise comparisons for a gene.
    Alternatively ``None``, if the mean was not computed.
    """

    min_rank: Optional[numpy.ndarray]
    """
    Floating-point array of length equal to the number of genes.
    Each entry is the minimum rank of the gene from all pairwise comparisons to other groups.
    Alternatively ``None``, if the mean was not computed.
    """

    def to_biocframe(self):
        """Convert the results to a :py:class:`~biocframe.BiocFrame.BiocFrame`.

        Returns:
            A :py:class:`~biocframe.BiocFrame.BiocFrame` where each row is a gene and each column is a summary statistic.
        """
        cols = ["min", "mean", "median", "max", "min_rank"]
        contents = {}
        for n in cols:
            contents[n] = getattr(self, n)
        import biocframe
        return biocframe.BiocFrame(contents, column_names=cols)


def summarize_effects(
    effects: numpy.ndarray,
    compute_min: bool = True,
    compute_mean: bool = True,
    compute_median: bool = True,
    compute_max: bool = True,
    compute_quantiles: Optional[Sequence] = None,
    compute_min_rank: bool = True,
    num_threads: int = 1
) -> list[GroupwiseSummarizedEffects]: 
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
        List of length equal to the number of groups (i.e., the extents of the first two dimensions of ``effects``).
        Each entry contains the summary statistics of the effect sizes of the comparisons involving the corresponding group.

    References:
        The ``summarize_effects`` function in the `scran_markers <https://libscran.github.io/scran_markers>`_ C++ library, for more details on the statistics.
    """
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

    output = []
    for val in results:
        _fix_summary_quantiles(val, compute_quantiles)
        output.append(GroupwiseSummarizedEffects(**val))

    return output
