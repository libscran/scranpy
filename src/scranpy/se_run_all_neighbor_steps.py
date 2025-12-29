from typing import Optional, Union

import scranpy
import knncolle
import singlecellexperiment

from .run_all_neighbor_steps import *
from .se_run_tsne import _add_tsne_results
from .se_run_umap import _add_umap_results
from .se_cluster_graph import _add_build_graph_results, _add_cluster_graph_results

from . import _utils_se as seutils


def run_all_neighbor_steps_se(
    x: singlecellexperiment.SingleCellExperiment,
    umap_output_name: Optional[str] = "UMAP",
    more_umap_args: Optional[dict] = {},
    tsne_output_name: Optional[str] = "TSNE",
    more_tsne_args: Optional[dict] = {},
    build_graph_name: Optional[str] = None,
    more_build_graph_args: Optional[dict] = {},
    cluster_output_name: Optional[str] = "clusters",
    cluster_meta_name: Optional[str] = None,
    more_cluster_graph_args: Optional[dict] = {},
    nn_parameters: knncolle.Parameters = knncolle.AnnoyParameters(),
    num_threads: int = 3,
    more_neighbor_args: dict = {},
    reddim_type: Union[str, int] = "PCA"
) -> singlecellexperiment.SingleCellExperiment: 
    """
    Concurrently run all steps involving a nearest-neighbor search, using the same nearest-neighbor search index built from an existing embedding.
    This includes the t-SNE, UMAP and graph-based clustering.
    Internally, this uses :py:func:`~scranpy.run_all_neighbor_steps` on a reduced dimension entry in :py:class:`~singlecellexperiment.SingleCellExperiment.SingleCellExperiment`.

    Args:
        x:
            A :py:class:`~singlecellexperiment.SingleCellExperiment.SingleCellExperiment` object or one of its subclasses.
            Rows correspond to genomic features and columns correspond to cells.

        umap_output_name:
            Name of the reduced dimension entry in which to store the UMAP coordinates in the output object.
            If ``None``, the UMAP is not computed.

        more_umap_args:
            Additional arguments for UMAP, to pass to :py:func:`~scranpy.run_all_neighbor_steps.run_all_neighbor_steps` as `run_umap_options`.
            If ``None``, the UMAP is not computed.

        tsne_output_name:
            Name of the reduced dimension entry in which to store the t-SNE coordinates in the output object.
            If ``None``, the t-SNE is not computed.

        more_tsne_args:
            Additional arguments for t-SNE, to pass to :py:func:`~scranpy.run_all_neighbor_steps.run_all_neighbor_steps` as `run_tsne_options`.
            If ``None``, the t-SNE is not computed.

        build_graph_name:
            Name of the metadata entry in which to store the nearest neighbor graph.
            If ``None``, the graph is not stored.

        more_build_graph_args:
            Additional arguments for graph construction, to pass to :py:func:`~scranpy.run_all_neighbor_steps.run_all_neighbor_steps` as `more_build_graph_args`.

        cluster_output_name:
            Name of the column of the column data in which to store the cluster assignments.
            If ``None``, graph-based clustering is not performed.
        
        cluster_meta_name:
            Name of the metadata entry in which to store additional clustering outputs.
            If ``None``, additional outputs are not stored.

        more_cluster_graph_args:
            Additional arguments for graph-based clustering, to pass to :py:func:`~scranpy.run_all_neighbor_steps.run_all_neighbor_steps` as `more_cluster_graph_args`.
            If ``None``, the graph-based clustering is not performed.

        nn_parameters:
            Parameters for the nearest-neighbor search.

        num_threads:
            Number of threads to use, passed to :py:func:`~scranpy.run_all_neighbor_steps.run_all_neighbor_steps`.

        more_neighbor_args:
            Additional arguments to pass to :py:func:`~scranpy.run_all_neighbor_steps.run_all_neighbor_steps`.

        reddim_type:
            Name or index of the reduced dimensions of ``x`` on which to perform the nearest neighbor search.
            Alternatively a tuple, where the first element contains the name of an alternative experiment of ``x``,
            and the second element contains the name/index of an embedding in that alternative experiment.

    Returns:
        A copy of ``x``, with additional coordinates in its reduced dimensions and clustering output in its column data.
        Additional information may also be stored in its metadata.
    """

    if umap_output_name is None:
        more_umap_args = None
    if tsne_output_name is None:
        more_tsne_args = None
    if cluster_output_name is None:
        more_cluster_graph_args = None

    outputs = run_all_neighbor_steps(
        seutils.get_transposed_reddim(x, reddim_type),
        run_umap_options=more_umap_args,
        run_tsne_options=more_tsne_args,
        build_snn_graph_options=more_build_graph_args,
        cluster_graph_options=more_cluster_graph_args,
        nn_parameters=nn_parameters,
        num_threads=num_threads,
        **more_neighbor_args
    )

    if "run_tsne" in outputs.get_names():
        x = _add_tsne_results(x, tsne_output_name, outputs["run_tsne"].transpose())
    if "run_umap" in outputs.get_names():
        x = _add_umap_results(x, umap_output_name, outputs["run_umap"].transpose())
    if "cluster_graph" in outputs.get_names():
        x = _add_build_graph_results(x, outputs["build_snn_graph"], graph_name=build_graph_name)
        x = _add_cluster_graph_results(x, outputs["cluster_graph"], output_name=cluster_output_name, meta_name=cluster_meta_name)

    return x
