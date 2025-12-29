from typing import Literal

import numpy
import biocutils

from . import lib_scranpy as lib


def cluster_kmeans(
    x: numpy.ndarray,
    k: int,
    init_method: Literal["var-part", "kmeans++", "random"] = "var-part",
    refine_method: Literal["hartigan-wong", "lloyd"] = "hartigan-wong",
    var_part_optimize_partition: bool = True,
    var_part_size_adjustment: float = 1,
    lloyd_iterations: int = 100,
    hartigan_wong_iterations: int = 10,
    hartigan_wong_quick_transfer_iterations: int = 50,
    hartigan_wong_quit_quick_transfer_failure: bool = False,
    seed: int = 5489,
    num_threads: int = 1
) -> biocutils.NamedList:
    """
    Perform k-means clustering with a variety of different initialization and refinement algorithms.

    Args:
        x: 
            Input data matrix where rows are dimensions and columns are observations (i.e., cells).
        
        k: 
            Number of clusters.

        init_method:
            Initialization method for defining the initial centroid coordinates.
            Choices are variance partitioning (``var-part``), kmeans++ (``kmeans++``) or random initialization (``random``).

        refine_method:
            Method to use to refine the cluster assignments and centroid coordinates.
            Choices are Lloyd's algorithm (``lloyd``) or the Hartigan-Wong algorithm (``hartigan-wong``).

        var_part_optimize_partition:
            Whether each partition boundary should be optimized to reduce the sum of squares in the child partitions.
            Only used if ``init_method = "var-part"``.

        var_part_size_adjustment:
            Floating-point value between 0 and 1, specifying the adjustment to the cluster size when prioritizing the next cluster to partition.
            Setting this to 0 will ignore the cluster size while setting this to 1 will generally favor larger clusters.
            Only used if ``init_method = "var-part"``.

        lloyd_iterations:
            Maximmum number of iterations for the Lloyd algorithm.

        hartigan_wong_iterations:
            Maximmum number of iterations for the Hartigan-Wong algorithm.

        hartigan_wong_quick_transfer_iterations:
            Maximmum number of quick transfer iterations for the Hartigan-Wong algorithm.

        hartigan_wong_quit_quick_transfer_failure
            Whether to quit the Hartigan-Wong algorithm upon convergence failure during quick transfer iterations.

        seed:
            Seed to use for random or kmeans++ initialization.

        num.threads:
            Number of threads to use.

    Returns:
        A :py:class:`~biocutils.NamedList.NamedList` containing:

        - ``clusters``: an integer NumPy array containing the cluster assignment for each cell.
          Values are integers in [0, N) where N is the total number of clusters.
        - ``centers``: a double-precision NumPy matrix containing the coordinates of the cluster centroids.
          Dimensions are in the rows while centers are in the columns.
        - ``iterations``: integer specifying the number of refinement iterations that were performed.
        - ``status``: convergence status.
           Any non-zero value indicates a convergence failure though the exact meaning depends on the choice of ``refine_method``.
           - For Lloyd, a value of 2 indicates convergence failure.
           - For Hartigan-Wong, a value of 2 indicates convergence failure in the optimal transfer iterations.
             A value of 4 indicates convergence failure in the quick transfer iterations when ``hartigan_wong_quit_quick_transfer_failure = True``.

    References:
        https://ltla.github.io/CppKmeans, which describes the various initialization and refinement algorithms in more detail.

    Examples:
        >>> import numpy
        >>> pcs = numpy.random.rand(10, 200)
        >>> import scranpy
        >>> clust = scranpy.cluster_kmeans(pcs, k=3)
    """
    out = lib.cluster_kmeans(
        numpy.array(x, copy=None, dtype=numpy.float64, order="F"),
        k,
        init_method,
        refine_method,
        var_part_optimize_partition,
        var_part_size_adjustment,
        lloyd_iterations,
        hartigan_wong_iterations,
        hartigan_wong_quick_transfer_iterations,
        hartigan_wong_quit_quick_transfer_failure,
        seed, 
        num_threads
    )

    return biocutils.NamedList.from_dict(out)
