from typing import Literal, Optional, Union

import numpy
import biocutils

from . import _lib_scranpy as lib
from ._build_snn_graph import build_snn_graph


def cluster_graph(
    x: biocutils.NamedList,
    method: Literal["multilevel", "leiden", "walktrap"] = "multilevel",
    multilevel_resolution: float = 1, 
    leiden_resolution: float = 1, 
    leiden_objective: Literal["modularity", "cpm", "er"] = "modularity",
    walktrap_steps: int = 4,
    seed: int = 42
) -> biocutils.NamedList:
    """
    Identify clusters of cells using a variety of community detection methods from a graph where similar cells are connected.

    Args:
        x:
            Components of the graph to be clustered, typically produced by :py:func:`~build_snn_graph.build_snn_graph`.
            Each node of the graph should be a cell.

        method:
            Community detection algorithm to use.

        multilevel_resolution:
            Resolution of the clustering when ``method = "multilevel"``.
            Larger values result in finer clusters.

        leiden_resolution:
            Resolution of the clustering when ``method = "leiden"``.
            Larger values result in finer clusters.

        leiden_objective:
            Objective function to use when ``method = "leiden"``.

        walktrap_steps:
            Number of steps to use when ``method = "walktrap"``.

        seed:
            Random seed to use for ``method = "multilevel"`` or ``"leiden"``.

    Returns:
        A :py:class:`~biocutils.NamedList.NamedList` containing the following entries.

        - ``membership``: an integer NumPy array containing the cluster assignment for each vertex, i.e., cell.
          All values are in [0, N) where N is the total number of clusters.

        For ``method = "multilevel"``, the output also contains:

        - ``levels``: a list containing the clustering at each level of the algorithm.
          Each entry corresponds to one level and is an integer NumPy array that contains the cluster assignment for each cell at that level.
        - ``modularity``: a double-precision NumPy array containing the modularity at each level.
          This has length equal to that of ``levels``, and the largest value corresponds to the assignments reported in ``membership``.

        For ``method = "walktrap"``, the output also contains:

        - ``merges``: an integer NumPy matrix with two columns.
          Each row  corresponds to a merge step and specifies the pair of cells or clusters that were merged at that step.
        - ``modularity``: a double-precision NumPy array that contains the modularity score at each merge step.

        For ``method = "leiden"``, the output also contains:

        - ``quality``: quality of the clustering.
          This is the same as the modularity if ``leiden_objective = "modularity"``.

    References:
        https://igraph.org/c/html/latest/igraph-Community.html, for the underlying implementation of each clustering method.

        The various ``cluster_*`` functions in the `scran_graph_cluster <https://libscran.github.io/scran_graph_cluster>`_ C++ library. 

    Examples:
        >>> import numpy
        >>> pcs = numpy.random.rand(10, 200)
        >>> import scranpy
        >>> graph = scranpy.build_snn_graph(pcs)
        >>> clust = scranpy.cluster_graph(graph)
        >>> import biocutils
        >>> print(biocutils.table(clust["membership"]))
    """

    graph = (x["vertices"], x["edges"], x["weights"])

    if method == "multilevel":
        out = lib.cluster_multilevel(graph, multilevel_resolution, seed)
        return biocutils.NamedList.from_dict(out)

    elif method == "leiden":
        out = lib.cluster_leiden(graph, leiden_resolution, leiden_objective, seed)
        return biocutils.NamedList.from_dict(out)

    elif method == "walktrap":
        out = lib.cluster_walktrap(graph, walktrap_steps)
        return biocutils.NamedList.from_dict(out)

    else:
        raise NotImplementedError("unsupported community detection method '" + method + "'")
