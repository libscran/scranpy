from typing import Optional, Sequence, Literal, Tuple, Any

import numpy
import biocutils
import mattress

from . import lib_scranpy as lib


def run_pca(
    x: Any,
    number: int = 25,
    scale: bool = False,
    block: Optional[Sequence] = None, 
    block_weight_policy: Literal["variable", "equal", "none"] = "variable",
    variable_block_weight: Tuple = (0, 1000),
    subset: Optional[Sequence] = None,
    components_from_residuals: bool = False,
    extra_work: int = 7,
    iterations: int = 1000,
    seed: int = 5489,
    realized: bool = True,
    num_threads: int =1
) -> biocutils.NamedList:
    """
    Run a PCA on the gene-by-cell log-expression matrix to obtain a low-dimensional representation for downstream analyses.

    Args:
        x:
            A matrix-like object where rows correspond to genes or genomic features and columns correspond to cells.
            Typically, the matrix is expected to contain log-expression values, and the rows should be filtered to relevant (e.g., highly variable) genes.

        number:
            Number of PCs to retain.

        scale:
            Whether to scale all genes to have the same variance.

        block:
           Array of length equal to the number of columns of ``x``, containing the block of origin (e.g., batch, sample) for each cell.
           Alternatively ``None``, if all cells are from the same block.

        block_weight_policy:
            Policy to use for weighting different blocks when computing the average for each statistic.
            Only used if ``block`` is provided.

        variable_block_weight:
            Parameters for variable block weighting.
            This should be a tuple of length 2 where the first and second values are used as the lower and upper bounds, respectively, for the variable weight calculation.
            Only used if ``block`` is provided and ``block_weight_policy = "variable"``.

        components_from_residuals:
            Whether to compute the PC scores from the residuals in the presence of a blocking factor.
            If ``False``, the residuals are only used to compute the rotation matrix, and the original expression values of the cells are projected onto this new space.
            Only used if ``block`` is provided.

        extra_work:
            Number of extra dimensions for the IRLBA workspace.

        iterations:
            Maximum number of restart iterations for IRLBA.

        seed:
            Seed for the initial random vector in IRLBA.

        realized:
            Whether to realize ``x`` into an optimal memory layout for IRLBA.
            This speeds up computation at the cost of increased memory usage.

        num_threads:
            Number of threads to use.

    Returns:
        A :py:class:`~biocutils.NamedList.NamedList` containing the following entries.

        - ``components``: a double-precision NumPy matrix of principal component (PC) scores.
          Rows are dimensions (i.e., PCs) and columns are cells.
        - ``rotation``: a double-precision NumPy matrix containing the rotation vectors.
          Rows are genes and columns are dimensions (i.e., PCs).
        - ``variance_explained``: a double-precision NumPy array containing the variance explained by each successive PC.
        - ``total_variance``: total variance in the input data.
          Guaranteed to be greater than the sum of ``variance_explained``.
        - ``center``: a double-precision NumPy array containing the mean for each gene across all cells.
          If ``block`` was supplied, this is instead a matrix containing the mean for each gene (column) in each block of cells (row).
        - ``block_ids``: list containing the levels of the blocking factor corresponding to each row of ``center``.
          Only reported if ``block`` was supplied.
        - ``scale``: a double-precision NumPy arrary containing the scaling for each gene. 
          Only reported if ``scale = True`` in the function call. 

    References:
        https://libscran.github.io/scran_pca, which describes the approach in more detail.
        In particular, the documentation for the ``blocked_pca`` function explains the blocking strategy.

    Examples:
        >>> import numpy
        >>> normed = numpy.random.rand(500, 100)
        >>> import scranpy
        >>> res = scranpy.run_pca(normed)
        >>> print(res["variance_explained"] / res["total_variance"])
    """

    if block is not None:
        blocklev, blockind = biocutils.factorize(block, sort_levels=True, dtype=numpy.uint32, fail_missing=True)
    else:
        blocklev = None
        blockind = None

    if subset is not None:
        subset = numpy.array(subset, dtype=numpy.dtype("uint32"))

    mat = mattress.initialize(x)
    output = lib.run_pca(
        mat.ptr,
        number,
        blockind,
        block_weight_policy,
        variable_block_weight,
        components_from_residuals,
        scale,
        subset,
        realized,
        extra_work,
        iterations,
        seed,
        num_threads
    )

    if not scale:
        del output["scale"]
    if block is not None:
        output["block_ids"] = blocklev

    return biocutils.NamedList.from_dict(output)
