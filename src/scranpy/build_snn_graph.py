from typing import Optional, Union, Literal

import knncolle
import biocutils
import numpy

from . import lib_scranpy as lib
from ._utils_neighbors import _check_indices



def build_snn_graph(
    x: Union[numpy.ndarray, knncolle.FindKnnResults, knncolle.Index],
    num_neighbors: int = 10,
    weight_scheme: Literal["ranked", "number", "jaccard"] = "ranked",
    num_threads: int = 1, 
    nn_parameters: knncolle.Parameters = knncolle.AnnoyParameters()
) -> biocutils.NamedList:
    """
    Build a shared nearest neighbor (SNN) graph where each node is a cell.
    Edges are formed between cells that share one or more nearest neighbors, weighted by the number or importance of those shared neighbors.

    Args:
        x: 
            Numeric matrix where rows are dimensions and columns are cells,
            typically containing a low-dimensional representation from, e.g., :py:func:`~scranpy.run_pca.run_pca`.

            Alternatively, a :py:class:`~knncolle.find_knn.FindKnnResults` object containing existing neighbor search results.
            The number of neighbors should be the same as ``num_neighbors``, otherwise a warning is raised.

            Alternatively, a :py:class:`~knncolle.Index.Index` object.

        num_neighbors:
            Number of neighbors in the nearest-neighbor graph.
            Larger values generally result in broader clusters during community detection.

        weight_scheme:
            Weighting scheme to use for the edges of the SNN graph, based on the number or ranking of the shared nearest neighbors.

        num_threads:
            Number of threads to use.

        nn_parameters:
            The algorithm to use for the nearest-neighbor search.
            Only used if ``x`` is not a pre-built nearest-neighbor search index or a list of existing nearest-neighbor search results.

    Results:
        A :py:class:`~biocutils.NamedList.NamedList` containing the components of a (possibly weighted) graph.

        - ``vertices``: integer specifying the number of vertices (i.e., cells) in the graph.
        - ``edges``: integer NumPy array containing the graph edges.
          Pairs of values represent the endpoints of an (undirected) edge, i.e., ``edges[0:2]`` form the first edge, ``edges[2:4]`` form the second edge and so on.
        - ``weights``: double-precision NumPy array of edge weights.
          This has length equal to half the length of ``edges``; the first weight corresponds to the first edge, and so on.

    References:
        The ``build_snn_graph`` function in the `scran_graph_cluster <https://libscran.github.io/scran_graph_cluster>`_ C++ library. 

    Examples:
        >>> import numpy
        >>> pcs = numpy.random.rand(10, 200)
        >>> import scranpy
        >>> graph = scranpy.build_snn_graph(pcs)
        >>> print(graph.get_names()) 
    """

    if isinstance(x, knncolle.FindKnnResults):
        nnidx = x.index
        _check_indices(nnidx, num_neighbors)
    else:
        if not isinstance(x, knncolle.Index):
            x = knncolle.build_index(nn_parameters, x.T)
        x = knncolle.find_knn(x, num_neighbors=num_neighbors, num_threads=num_threads)
        nnidx = x.index

    ncells, edges, weights = lib.build_snn_graph(nnidx, weight_scheme, num_threads)
    return biocutils.NamedList.from_dict({
        "vertices": ncells,
        "edges": edges,
        "weights": weights
    })
