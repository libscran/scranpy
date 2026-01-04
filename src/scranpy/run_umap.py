from typing import Optional, Union, Literal

import knncolle
import numpy

from . import lib_scranpy as lib
from ._utils_neighbors import _check_neighbor_results


def run_umap(
    x: Union[numpy.ndarray, knncolle.FindKnnResults, knncolle.Index],
    num_dim: int = 2,
    parallel_optimization: bool = False,
    local_connectivity: float = 1,
    bandwidth: float = 1,
    mix_ratio: float = 1,
    spread: float = 1,
    min_dist: float = 0.1, 
    a: Optional[float] = None,
    b: Optional[float] = None,
    repulsion_strength: float = 1,
    initialize_method: Literal["spectral", "random", "none"] = "spectral",
    initial_coordinates: Optional[numpy.array] = None,
    initialize_random_on_spectral_fail: bool = True,
    initialize_spectral_scale: float = 10,
    initialize_spectral_jitter: bool = False,
    initialize_spectral_jitter_sd: float = 0.0001,
    initialize_random_scale: float = 10,
    initialize_seed: int = 9876543210,
    num_epochs: Optional[int] = None,
    learning_rate: float = 1,
    negative_sample_rate: float = 5,
    num_neighbors: int = 15, 
    optimize_seed: int = 1234567890, 
    num_threads: int = 1,
    nn_parameters: knncolle.Parameters = knncolle.AnnoyParameters()
) -> numpy.ndarray:
    """Compute UMAP coordinates to visualize similarities between cells.

    Args:
        x: 
            Numeric matrix where rows are dimensions and columns are cells, typically containing a low-dimensional representation from, e.g., :py:func:`~scranpy.run_pca.run_pca`.

            Alternatively, a :py:class:`~knncolle.find_knn.FindKnnResults` object containing existing neighbor search results.

            Alternatively, a :py:class:`~knncolle.Index.Index` object.

        num_dim:
            Number of dimensions in the UMAP embedding.

        local_connectivity:
            Number of nearest neighbors that are assumed to be always connected, with maximum membership confidence.
            Larger values increase the connectivity of the embedding and reduce the focus on local structure.
            This may be a fractional number of neighbors, in which case interpolation is performed when computing the membership confidence.

        bandwidth:
            Effective bandwidth of the kernel when converting the distance to a neighbor into a fuzzy set membership confidence.
            Larger values reduce the decay in confidence with respect to distance, increasing connectivity and favoring global structure. 

        mix_ratio
            Number between 0 and 1 specifying the mixing ratio when combining fuzzy sets.
            A mixing ratio of 1 will take the union of confidences, a ratio of 0 will take the intersection, and intermediate values will interpolate between them.
            Larger values favor connectivity and more global structure.

        spread:
            Scale of the coordinates of the final low-dimensional embedding.
            Ignored if ``a`` and ``b`` are provided.

        min_dist:
            Minimum distance between observations in the final low-dimensional embedding.
            Smaller values will increase local clustering while larger values favor a more even distribution of observations throughout the low-dimensional space.
            This is interpreted relative to ``spread``.
            Ignored if ``a`` and ``b`` are provided.

        a:
            The ``a`` parameter for the fuzzy set membership strength calculations.
            Larger values yield a sharper decay in membership strength with increasing distance between observations.
            If this or ``b`` are ``None``, a suitable value for this parameter is automatically determined from ``spread`` and ``min_dist``.

        b:
            The ``b`` parameter for the fuzzy set membership strength calculations.
            Larger values yield an earlier decay in membership strength with increasing distance between observations.
            If this or ``a`` are ``None``, a suitable value for this parameter is automatically determined from ``spread`` and ``min_dist``.

        repulsion_strength
            Modifier for the repulsive force.
            Larger values increase repulsion and favor local structure.

        initialize_method:
            How to initialize the embedding.

            - ``spectral``: spectral decomposition of the normalized graph Laplacian.
              Specifically, the initial coordinates are defined from the eigenvectors corresponding to the smallest non-zero eigenvalues.
              This fails in the presence of multiple graph components or if the approximate SVD fails to converge.
            - ``random``:  fills the embedding with random draws from a normal distribution.
            - ``none``: uses initial values from ``initial_coordinates``.

        initialize_random_on_spectral_fail:
            Whether to fall back to random sampling (i.e., ``random``) if spectral initialization fails due to the presence of multiple components in the graph.
            If False, the values in ``initial_coordinates`` will be used instead, i.e., same as ``none``.
            Only relevant if ``initialize_method = "spectral"`` and spectral initialization fails.

        initialize_spectral_scale:
            Maximum absolute magnitude of the coordinates after spectral initialization.
            The default is chosen to avoid outlier observations with large absolute distances that may interfere with optimization.
            Only relevant if ``initialize_method = "spectral"`` and spectral initialization does not fail.

        initialize_spectral_jitter:
            Whether to jitter coordinates after spectral initialization to separate duplicate observations (e.g., to avoid overplotting).
            This is done using normally-distributed noise of mean zero and standard deviation of ``initialize_spectral_jitter_sd``.
            Only relevant if ``initialize_method = "spectral"`` and spectral initialization does not fail.

        initialize_spectral_jitter_sd
            Standard deviation of the jitter to apply after spectral initialization.
            Only relevant if ``initialize_method = "spectral"`` and spectral initialization does not fail and ``initialize_spectral_jitter = True``.

        initialize.random.scale:
            Scale of the randomly generated initial coordinates.
            Coordinates are sampled from a uniform distribution from [-x, x) where x is ``initialize_random_scale``.
            Only relevant if ``initialize_method = "random"``,
            or ``initialize_method = "spectral"`` and spectral initialization fails and ``initialize_random_on_spectral_fail = True``.

        initialize_seed:
            Seed for the random number generation during initialization.
            Only relevant if ``initialize_method = "random"``,
            or ``initialize_method = "spectral"`` and ``initialize_spectral_jitter = True``;
            or ``initialize_method = "spectral"`` and spectral initialization fails and ``initialize_random_on_spectral_fail = True``.

        initial_coordinates:
            Double-precision matrix of initial coordinates with number of rows and columns equal to the number of observations and ``num_dim``, respectively.
            Only relevant if ``initialize_method = "none"``;
            or ``initialize_method = "spectral"`` and spectral initialization fails and ``initialize_random_on_spectral_fail = False``.

        num_epochs:
            Number of epochs to perform.
            If set to None, an appropriate number of epochs is chosen based on the number of points in ``x``.

        num_neighbors:
            Number of neighbors to use in the UMAP algorithm.
            Larger values cause the embedding to focus on global structure.
            Ignored if ``x`` is a :py:class:`~knncolle.find_knn.FindKnnResults` object.

        optimize_seed:
            Integer scalar specifying the seed to use. 

        num_threads:
            Number of threads to use.

        nn_parameters:
            The algorithm to use for the nearest-neighbor search.
            Only used if ``x`` is not a pre-built nearest-neighbor search index or a list of existing nearest-neighbor search results.

    Returns:
        Double-precision NumPy matrix containing the coordinates of each cell in a 2-dimensional embedding.
        Each row corresponds to a cell and each column corresponds to a dimension. 

    References:
        https://libscran.github.io/umappp, for the underlying implementation.

    Examples:
        >>> import numpy
        >>> pcs = numpy.random.rand(20, 500)
        >>> import scranpy
        >>> uout = scranpy.run_umap(pcs)
        >>> print(uout[:5,:])
    """

    if isinstance(x, knncolle.FindKnnResults):
        nnidx = x.index
        nndist = x.distance
        _check_neighbor_results(nnidx, nndist)
    else:
        if not isinstance(x, knncolle.Index):
            x = knncolle.build_index(nn_parameters, x.T)
        x = knncolle.find_knn(x, num_neighbors=num_neighbors, num_threads=num_threads)
        nnidx = x.index
        nndist = x.distance

    if initial_coordinates is not None:
        initial_coordinates = initial_coordinates.T

    out = lib.run_umap(
        nnidx,
        nndist,
        num_dim,
        local_connectivity,
        bandwidth,
        mix_ratio,
        spread,
        min_dist,
        a,
        b,
        repulsion_strength,
        initialize_method,
        initial_coordinates,
        initialize_random_on_spectral_fail,
        initialize_spectral_scale,
        initialize_spectral_jitter,
        initialize_spectral_jitter_sd,
        initialize_random_scale,
        initialize_seed,
        num_epochs,
        learning_rate,
        negative_sample_rate,
        optimize_seed,
        num_threads
    )

    return out.transpose()
