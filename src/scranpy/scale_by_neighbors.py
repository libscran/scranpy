from typing import Sequence, Optional, Literal, Tuple

import knncolle
import numpy
import biocutils

from . import lib_scranpy as lib


def scale_by_neighbors(
    x: Sequence,
    num_neighbors: int = 20,
    block: Optional[Sequence] = None,
    block_weight_policy: Literal["variable", "equal", "none"] = "variable",
    variable_block_weight: Tuple = (0, 1000),
    num_threads: int = 1,
    weights: Optional[Sequence] = None,
    nn_parameters: knncolle.Parameters = knncolle.AnnoyParameters()
) -> biocutils.NamedList:
    """
    Scale multiple embeddings (usually derived from different modalities across the same set of cells) so that their within-population variances are comparable.
    Then, combine them into a single embedding matrix for combined downstream analysis.

    Args:
        x: 
            Sequence of of numeric matrices of principal components or other embeddings, one for each modality.
            For each entry, rows are dimensions and columns are cells.
            All entries should have the same number of columns but may have different numbers of rows.

        num_neighbors:
            Number of neighbors to use to define the scaling factor.

        num_threads
            Number of threads to use.

        nn_parameters:
            Algorithm for the nearest-neighbor search.

        weights:
            Array of length equal to ``x``, specifying the weights to apply to each modality.
            Each value represents a multiplier of the within-population variance of its modality, i.e., larger values increase the contribution of that modality in the combined output matrix.
            The default of ``None`` is equivalent to an all-1 vector, i.e., all modalities are scaled to have the same within-population variance.

    Returns:
        A :py:class:`~biocutils.NamedList.NamedList` containing the following entries.

        - ``scaling``: a double-precision NumPy array containing the scaling factor to be applied to each embedding in ``x``.
        - ``combined``: a double-precision NumPy matrix of scaled and concatenated embeddings.
          This is formed by scaling each embedding in ``x`` by its corresponding entry of ``scaling``, and then concatenating them together by row.
          Each row corresponds to a dimension and each column corresponds to a cell.

    References:
        https://libscran.github.io/mumosa, for the basis and caveats of this approach.

    Examples:
        >>> import numpy
        >>> rna_pcs = numpy.random.rand(25, 200)
        >>> adt_pcs = numpy.random.rand(15, 200)
        >>> other_pcs = numpy.random.rand(10, 200)
        >>> import scranpy
        >>> res = scranpy.scale_by_neighbors([rna_pcs, adt_pcs, other_pcs])
        >>> print(res["scaling"])
    """

    builder, _ = knncolle.define_builder(nn_parameters)

    if block is None:
        blockind = None
    else: 
        _, blockind = biocutils.factorize(block, fail_missing=True, dtype=numpy.uint32)

    scaling = lib.scale_by_neighbors(
        x[0].shape[1],
        x,
        num_neighbors,
        blockind,
        block_weight_policy,
        variable_block_weight,
        num_threads,
        builder.ptr
    )

    if weights is not None:
        if len(weights) != len(x):
            raise ValueError("'weights' should have the same length as 'x'")
        for i, w in enumerate(weights):
            scaling[i] *= w

    copies = []
    for i, m in enumerate(x):
        copies.append(m * scaling[i])
    combined = numpy.concatenate(copies, axis=0)

    return biocutils.NamedList([scaling, combined], ["scaling", "combined"])
