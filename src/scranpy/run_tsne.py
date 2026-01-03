from typing import Optional, Union

import knncolle
import numpy
import warnings

from . import lib_scranpy as lib
from ._utils_neighbors import _check_neighbor_results


def run_tsne(
    x: Union[numpy.ndarray, knncolle.FindKnnResults, knncolle.Index],
    perplexity: float = 30,
    num_neighbors: Optional[int] = None,
    theta: float = 1,
    early_exaggeration_iterations: int = 250,
    exaggeration_factor: float = 12,
    momentum_switch_iterations: int = 250,
    start_momentum: float = 0.5,
    final_momentum: float = 0.8,
    eta: float = 200,
    max_depth: int = 7,
    leaf_approximation: bool = False,
    max_iterations: int = 500,
    seed: int = 42,
    num_threads: int = 1, 
    nn_parameters: knncolle.Parameters = knncolle.AnnoyParameters()
) -> numpy.ndarray:
    """Compute t-SNE coordinates to visualize similarities between cells.

    Args:
        x: 
            Numeric matrix where rows are dimensions and columns are cells, typically containing a low-dimensional representation from, e.g., :py:func:`~scranpy.run_pca.run_pca`.

            Alternatively, a :py:class:`~knncolle.find_knn.FindKnnResults` object containing existing neighbor search results.

            Alternatively, a :py:class:`~knncolle.Index.Index` object.

        perplexity:
            Perplexity to use in the t-SNE algorithm.
            Larger values cause the embedding to focus on global structure.

        num_neighbors:
            Number of neighbors in the nearest-neighbor graph.
            Typically derived from ``perplexity`` using :py:func:`~tsne_perplexity_to_neighbors`.
            If ``x`` is a :py:class:`~knncolle.find_knn.FindKnnResults` object and the number of neighbors is not equal to ``num_neighbors``, a warning is raised; 
            this can be suppressed by setting ``num_neighbors = None``.

        theta:
             Approximation level for the Barnes-Hut calculation of repulsive forces.
             Lower values increase accuracy at the cost of computational time.

        early_exaggeration_iterations:
             Number of iterations of the early exaggeration phase, where the conditional probabilities are multiplied by ``exaggeration_factor``.
             In this phase, the empty space between clusters is increased so that clusters can easily relocate to find a good global organization.
             Larger values improve convergence within this phase at the cost of reducing the remaining iterations in ``max_iterations``.

        exaggeration_factor:
             Exaggeration factor to scale the probabilities during the early exaggeration phase (see ``early_exaggeration_iterations``).
             Larger values increase the attraction between nearest neighbors to favor local structure during this phase.

        momentum_switch_iterations:
             Number of iterations to perform before switching from the starting momentum (``start_momentum``) to the final momentum (``final_momentum``).
             Greater momentum can improve convergence by increasing the step size and smoothing over local oscillations, at the risk of potentially skipping over relevant minima.

        start_momentum:
            Starting momentum in [0, 1) to be used in the iterations before the momentum switch at ``momentum_switch_iterations``.
            This is usually lower than ``final_momentum`` to avoid skipping over suitable local minima.

        final_momentum:
            Final momentum in [0, 1) to be used in the iterations after the momentum switch at ``momentum_switch_iterations``.
            This is usually higher than ``start_momentum`` to accelerate convergence to the local minima once the observations are moderately well-organized.

        eta:
            The learning rate, used to scale the updates to the coordinates at each iteration.
            Larger values can speed up convergence at the cost of potentially skipping over local minima.

        max_depth:
            Maximum depth of the Barnes-Hut quadtree.
            If neighboring observations cannot be separated before the maximum depth is reached, they will be assigned to the same leaf node.
            This effectively approximates each observation's coordinates with the center of mass of its leaf node.
            Smaller values (7-10) improve speed at the cost of accuracy.

        leaf_approximation:
            Whether to use the "leaf approximation" approach, which sacrifices some accuracy for greater speed.
            This replaces a observation with the center of mass of its leaf node when computing the repulsive forces to all other observations.
            Only effective when ``max_depth`` is small enough for multiple cells to be assigned to the same leaf node of the quadtree.

        max_iterations:
            Maximum number of iterations to perform.

        seed:
            Random seed to use for generating the initial coordinates.

        num_threads:
            Number of threads to use.

        nn_parameters:
            The algorithm to use for the nearest-neighbor search.
            Only used if ``x`` is not a pre-built nearest-neighbor search index or a list of existing nearest-neighbor search results.

    Returns:
        Array containing the coordinates of each cell in a 2-dimensional embedding.
        Each row corresponds to a cell and each column corresponds to a dimension. 

    References:
        https://libscran.github.io/qdtsne, for some more details on the approximations.

    Examples:
        >>> import numpy
        >>> pcs = numpy.random.rand(20, 500)
        >>> import scranpy
        >>> tout = scranpy.run_tsne(pcs)
        >>> print(tout[:5,:])
    """

    if num_neighbors is None:
        num_neighbors = tsne_perplexity_to_neighbors(perplexity)

    if isinstance(x, knncolle.FindKnnResults):
        nnidx = x.index
        nndist = x.distance
        _check_neighbor_results(nnidx, nndist)
        if num_neighbors is not None and nnidx.shape[1] != num_neighbors:
            warnings.warn("number of columns in 'index' is not consistent with 'num_neighbors'")

    else:
        if not isinstance(x, knncolle.Index):
            x = knncolle.build_index(nn_parameters, x.T)
        x = knncolle.find_knn(x, num_neighbors=num_neighbors, num_threads=num_threads)
        nnidx = x.index
        nndist = x.distance

    out = lib.run_tsne(
        nnidx,
        nndist,
        perplexity,
        theta,
        early_exaggeration_iterations,
        exaggeration_factor,
        momentum_switch_iterations,
        start_momentum,
        final_momentum,
        eta,
        max_depth,
        leaf_approximation,
        max_iterations,
        seed,
        num_threads
    )

    return out.transpose()


def tsne_perplexity_to_neighbors(perplexity: float) -> int:
    """Determine the number of nearest neighbors required to support a given
    perplexity in the t-SNE algorithm.

    Args:
        perplexity:
            Perplexity to use in :py:func:`~run_tsne`.

    Returns:
        The corresponding number of nearest neighbors.
    """
    return lib.perplexity_to_neighbors(perplexity)
