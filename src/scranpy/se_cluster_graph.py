from typing import Optional, Union, Literal

import singlecellexperiment 
from . import _utils_se as seutils
from .cluster_graph import *
from .build_snn_graph import *


def cluster_graph_se(
    x: singlecellexperiment.SingleCellExperiment,
    num_neighbors: int = 10,
    num_threads: int = 1,
    more_build_args: dict = {},
    method: Literal["multilevel", "leiden", "walktrap"] = "multilevel",
    resolution: Optional[float] = None,
    more_cluster_args: dict = {},
    reddim_type: Union[int, str, tuple] = "PCA",
    output_name: str = "clusters",
    meta_name: Optional[str] = None,
    graph_name: Optional[str] = None 
) -> singlecellexperiment.SingleCellExperiment:
    """
    Construct a shared-nearest neighbor (SNN) graph from an existing low-dimensional embedding,
    and apply community detection algorithms to obtain clusters of cells.
    This calls :py:func:`~scranpy.build_snn_graph.build_snn_graph` followed by :py:func:`~scranpy.cluster_graph.cluster_graph`.

    Args:
        x:
            A :py:class:`~singlecellexperiment.SingleCellExperiment.SingleCellExperiment` or one of its subclasses.
            Rows correspond to genomic features and columns correspond to cells.

        num_neighbors:
            Number of neighbors for constructing the SNN graph, see :py:func:`~scranpy.build_snn_graph.build_snn_graph` for details.

        num_threads:
            Number of threads for the neighbor search when constructing the SNN graph, see :py:func:`~scranpy.build_snn_graph.build_snn_graph` for details.

        more_build_args:
            Additional arguments to be passed to :py:func:`~scranpy.build_snn_graph.build_snn_graph`.

        method:
            Community detection method to be used by :py:func:`~scranpy.cluster_graph.cluster_graph`.

        resolution:
            Clustering resolution to be used by :py:func:`~scranpy.cluster_graph.cluster_graph`.
            This is either passed as ``multilevel_resolution`` or ``leiden_resolution``, depending on ``method``.

        more_cluster_args:
            Additional arguments to be passed to :py:func:`~scranpy.cluster_graph.cluster_graph`.

        reddim_type:
            Name or index of the existing reduced dimension of ``x`` to be used for clustering.
            Alternatively a tuple, where the first element contains the name of an alternative experiment of ``x``,
            and the second element contains the name/index of an embedding in that alternative experiment.

        output_name:
            Name of the column of the column data in which to store the cluster assignments.

        meta_name:
            Name of the metadta entry in which to store extra clustering output.
            If ``None``, no extra clustering output is stored. 

        graph_name:
            Name of the metadta entry in which to store the SNN graph.
            If ``None``, the graph is not stored. 

    Returns:
        ``x`` is returned with the cluster assignment for each cell stored in its column data.
        Additional clustering output is stored in its metadata.

    Examples:
        >>> import scranpy
        >>> sce = scranpy.get_test_rna_data_se("pca")
        >>> sce = scranpy.cluster_graph_se(sce)
        >>> 
        >>> # Looking at the distribution of cells among the clusters:
        >>> import numpy
        >>> clust = sce.get_column_data()["clusters"]
        >>> clustids, counts = numpy.unique(clust, return_counts=True)
        >>> import biocframe
        >>> print(biocframe.BiocFrame({ "cluster": clustids, "counts": counts }))
    """

    graph_out = build_snn_graph(
        seutils.get_transposed_reddim(x, reddim_type),
        num_neighbors=num_neighbors,
        num_threads=num_threads,
        **more_build_args
    )

    x = _add_build_graph_results(x, graph_out, graph_name=graph_name)

    import copy
    more_cluster_args = copy.copy(more_cluster_args)
    if resolution is not None:
        more_cluster_args['multilevel_resolution'] = resolution
        more_cluster_args['leiden_resolution'] = resolution

    clust_out = cluster_graph(graph_out, method=method, **more_cluster_args)
    return _add_cluster_graph_results(x, clust_out, output_name, meta_name)


def _add_build_graph_results(x: singlecellexperiment.SingleCellExperiment, graph: biocutils.NamedList, graph_name: Optional[str]) -> singlecellexperiment.SingleCellExperiment:
    if graph_name is not None:
        import copy
        meta = copy.copy(x.get_metadata())
        meta[graph_name] = graph
        x = x.set_metadata(meta)
    return x


def _add_cluster_graph_results(x: singlecellexperiment.SingleCellExperiment, res: biocutils.NamedList, output_name: str, meta_name: Optional[str]) -> singlecellexperiment.SingleCellExperiment:
    df = x.get_column_data()
    df = df.set_column(output_name, res["membership"])
    x = x.set_column_data(df)

    if meta_name is not None:
        import copy
        meta = copy.copy(x.get_metadata())
        del res["membership"] # already stored in the column data. 
        meta[meta_name] = res
        x = x.set_metadata(meta)

    return x
