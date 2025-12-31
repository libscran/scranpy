from typing import Union, Optional

import singlecellexperiment

from .cluster_kmeans import *
from . import _utils_se as seutils


def cluster_kmeans_se(
    x: singlecellexperiment.SingleCellExperiment,
    k: int,
    num_threads: int = 1,
    more_kmeans_args: dict = {},
    reddim_type: Union[str, int, tuple] = "PCA",
    output_name: str = "clusters",
    meta_name: Optional[str] = None
) -> singlecellexperiment.SingleCellExperiment:
    """
    Perform k-means clustering on an existing low-dimensional embedding.
    This calls :py:func:`~scranpy.cluster_kmeans.cluster_kmeans` on reduced dimensions from a :py:class:`~singlecellexperiment.SingleCellExperiment.SingleCellExperiment`.

    Args:
        x:
            A :py:class:`~singlecellexperiment.SingleCellExperiment.SingleCellExperiment` object or one of its subclasses.
            Rows correspond to genomic features and columns correspond to cells.

        k:
            Number of clusters, to be passed to :py:func:`~scranpy.cluster_kmeans.cluster_kmeans`.

        num_threads:
            Number of threads, to be passed to :py:func:`~scranpy.cluster_kmeans.cluster_kmeans`.

        more_kmeans_args:
            Additional arguments to be passed to :py:func:`~scranpy.cluster_kmeans.cluster_kmeans`.

        reddim_type:
            Name or index of an existing embedding in ``x``, on which to perform the clustering.
            Alternatively a tuple, where the first element contains the name of an alternative experiment of ``x``,
            and the second element contains the name/index of an embedding in that alternative experiment.

        output_name:
            Name of the column of the column data in which to store the cluster assignments.

        meta_name:
            Name of the metadta entry in which to store extra clustering output.
            If ``None``, no extra clustering output is stored. 

    Returns:
        A copy of ``x``, with the cluster assignment for each cell stored in its column data.
        Additional clustering output is stored in its metadata.

    Examples:
        >>> import scranpy
        >>> sce = scranpy.get_test_rna_data_se("pca")
        >>> sce = scranpy.cluster_kmeans_se(sce, k=10)
        >>> 
        >>> # Looking at the distribution of cells among the clusters:
        >>> import numpy
        >>> clust = sce.get_column_data()["clusters"]
        >>> clustids, counts = numpy.unique(clust, return_counts=True)
        >>> import biocframe
        >>> print(biocframe.BiocFrame({ "cluster": clustids, "counts": counts }))
    """

    clout = cluster_kmeans(
        seutils.get_transposed_reddim(x, reddim_type),
        k=k,
        num_threads=num_threads,
        **more_kmeans_args
    )

    df = x.get_column_data()
    df = df.set_column(output_name, clout["clusters"])
    x = x.set_column_data(df)

    if meta_name is not None:
        import copy
        meta = copy.copy(x.get_metadata())
        del clout["clusters"] # already stored in the column data. 
        meta[meta_name] = clout
        x = x.set_metadata(meta)

    return x
