from typing import Any, Sequence, Union, Optional

import numpy
import mattress
import biocutils
import biocframe

from . import lib_scranpy as lib
from .combine_factors import combine_factors


def aggregate_across_cells(
    x: Any,
    factors: Union[dict, Sequence, biocutils.NamedList, biocframe.BiocFrame],
    num_threads: int = 1
) -> biocutils.NamedList:
    """
    Aggregate expression values across cells based on one or more grouping factors.
    This is primarily used to create pseudo-bulk profiles for each cluster/sample combination.

    Args:
        x: 
            A matrix-like object where rows correspond to genes or genomic features and columns correspond to cells.
            Values are expected to be counts.

        factors:
            One or more grouping factors, see :py:func:`~scranpy.combine_factors.combine_factors`.
            Each entry should be a sequence of length equal to the number of columns in ``x``.

        num_threads:
            Number of threads to use for aggregation.

    Returns:
        A :py:class:`~biocutils.named_list.NamedList` containing the following entries.

        - ``sum``: double-precision NumPy matrix where each row corresponds to a gene and each column corresponds to a unique combination of grouping levels.
          Each matrix entry contains the summed expression across all cells with that combination.
        - ``detected``: integer NumPy matrix where each row corresponds to a gene and each column corresponds to a unique combination of grouping levels.
          Each matrix entry contains the number of cells with detected expression in that combination.
        - ``combinations``: a :py:class:`~biocframe.BiocFrame.BiocFrame` containing all unique combinations of levels across ``factors``. 
          Each column corresponds to an entry of ``factors`` while each row corresponds to a combination.
          Specifically, the ``i``-th combination is defined as the ``i``-th elements of all columns.
          Combinations are in the same order as the columns of :py:attr:`~sum` and :py:attr:`~detected`.
        - ``counts``: an integer NumPy array containing the number of cells associated with each combination in ``combinations``.
        - ``index``: an Integer NumPy array of length equal to the number of cells.
          This specifies the combination ``combinations`` associated with each cell in ``x``.

    References:
        The ``aggregate_across_cells`` function in the `scran_aggregate <https://libscran.github.io/scran_aggregate>`_ C++ library. 

    Examples:
        >>> import numpy
        >>> mat = numpy.random.rand(100, 20)
        >>> import scranpy
        >>> clusters = ["A", "B", "C", "D"] * 5
        >>> blocks = [1] * 4 + [2] * 4 + [3] * 4 + [4] * 4 + [5] * 4 
        >>> aggr = scranpy.aggregate_across_cells(mat, { "clusters": clusters, "blocks": blocks })
        >>> aggr["sum"][:5,]
        >>> print(aggr["combinations"])
    """

    combout = combine_factors(factors)
    comblev = combout["levels"]
    combind = combout["index"]

    mat = mattress.initialize(x)
    outsum, outdet = lib.aggregate_across_cells(mat.ptr, combind, num_threads)

    counts = numpy.zeros(comblev.shape[0], dtype=numpy.uint32)
    for i in combind:
        counts[i] += 1

    output = biocutils.NamedList()
    output["sum"] = outsum
    output["detected"] = outdet
    output["combinations"] = comblev
    output["counts"] = counts
    output["index"] = combind

    return output
