from .adt_quality_control import *
from .rna_quality_control import *
from .crispr_quality_control import *
from .normalize_counts import *
from .center_size_factors import *
from .compute_clrm1_factors import *
from .model_gene_variances import *
from .fit_variance_trend import *
from .choose_highly_variable_genes import *
from .run_pca import *
from .cluster_kmeans import *
from .run_all_neighbor_steps import *
from .score_markers import *
from .summarize_effects import *
from .correct_mnn import *
from .scale_by_neighbors import *

from typing import Any, Sequence, Optional, Union
from collections.abc import Mapping
from dataclasses import dataclass

import biocutils
import delayedarray
import numpy


@dataclass
class AnalyzeResults:
    """Results of :py:func:`~scranpy.analyze.analyse`."""

    rna_qc_metrics: Optional[ComputeRnaQcMetricsResults]
    """Results of :py:func:`~scranpy.rna_quality_control.compute_rna_qc_metrics`.
    If RNA data is not available, this is set to ``None`` instead."""

    rna_qc_thresholds: Optional[SuggestRnaQcThresholdsResults]
    """Results of :py:func:`~scranpy.rna_quality_control.suggest_rna_qc_thresholds`.
    If RNA data is not available, this is set to ``None`` instead."""

    rna_qc_filter: Optional[numpy.ndarray]
    """Results of :py:func:`~scranpy.rna_quality_control.filter_rna_qc_metrics`.
    If RNA data is not available, this is set to ``None`` instead."""

    adt_qc_metrics: Optional[ComputeAdtQcMetricsResults]
    """Results of :py:func:`~scranpy.adt_quality_control.compute_adt_qc_metrics`.
    If ADT data is not available, this is set to ``None`` instead."""

    adt_qc_thresholds: Optional[SuggestAdtQcThresholdsResults]
    """Results of :py:func:`~scranpy.adt_quality_control.suggest_adt_qc_thresholds`.
    If ADT data is not available, this is set to ``None`` instead."""

    adt_qc_filter: Optional[numpy.ndarray]
    """Results of :py:func:`~scranpy.adt_quality_control.filter_adt_qc_metrics`.
    If ADT data is not available, this is set to ``None`` instead."""

    crispr_qc_metrics: Optional[ComputeCrisprQcMetricsResults]
    """Results of :py:func:`~scranpy.crispr_quality_control.compute_crispr_qc_metrics`.
    If CRISPR data is not available, this is set to ``None`` instead."""

    crispr_qc_thresholds: Optional[SuggestCrisprQcThresholdsResults]
    """Results of :py:func:`~scranpy.crispr_quality_control.suggest_crispr_qc_thresholds`.
    If CRISPR data is not available, this is set to ``None`` instead."""

    crispr_qc_filter: Optional[numpy.ndarray]
    """Results of :py:func:`~scranpy.crispr_quality_control.filter_crispr_qc_metrics`.
    If CRISPR data is not available, this is set to ``None`` instead."""

    combined_qc_filter: numpy.ndarray
    """Array of booleans indicating which cells are of high quality and should be retained for downstream analyses."""

    rna_filtered: Optional[delayedarray.DelayedArray]
    """Matrix of RNA counts that has been filtered to only contain the high-quality cells in :py:attr:`~combined_qc_filter`.
    If RNA data is not available, this is set to ``None`` instead."""

    adt_filtered: Optional[delayedarray.DelayedArray]
    """Matrix of ADT counts that has been filtered to only contain the high-quality cells in :py:attr:`~combined_qc_filter`.
    If ADT data is not available, this is set to ``None`` instead."""

    crispr_filtered: Optional[delayedarray.DelayedArray]
    """Matrix of CRISPR counts that has been filtered to only contain the high-quality cells in :py:attr:`~combined_qc_filter`.
    If CRISPR data is not available, this is set to ``None`` instead."""

    rna_size_factors: Optional[numpy.ndarray]
    """Size factors for the RNA count matrix, derived from the sum of counts for each cell.
    Size factors are centered with :py:func:`~scranpy.center_size_factors.center_size_factors`.
    If RNA data is not available, this is set to ``None`` instead."""

    rna_normalized: Optional[delayedarray.DelayedArray]
    """Matrix of (log-)normalized expression values derived from RNA counts, as computed by :py:func:`~scranpy.normalize_counts.normalize_counts` using :py:attr:`~rna_size_factors`.
    If RNA data is not available, this is set to ``None`` instead."""

    adt_size_factors: Optional[numpy.ndarray]
    """Size factors for the ADT count matrix, computed using the CLRm1 method.
    Size factors are centered with :py:func:`~scranpy.center_size_factors.center_size_factors`.
    If ADT data is not available, this is set to ``None`` instead."""

    adt_normalized: Optional[delayedarray.DelayedArray]
    """Matrix of (log-)normalized expression values derived from ADT counts, as computed by :py:func:`~scranpy.normalize_counts.normalize_counts` using :py:attr:`~adt_size_factors`.
    If ADT data is not available, this is set to ``None`` instead."""

    crispr_size_factors: Optional[numpy.ndarray]
    """Size factors for the CRISPR count matrix, derived from the sum of counts for each cell.
    Size factors are centered with :py:func:`~scranpy.center_size_factors.center_size_factors`.
    If CRISPR data is not available, this is set to ``None`` instead."""

    crispr_normalized: Optional[delayedarray.DelayedArray]
    """Matrix of (log-)normalized expression values derived from CRISPR counts, as computed by :py:func:`~scranpy.normalize_counts.normalize_counts` using :py:attr:`~crispr_size_factors`.
    If CRISPR data is not available, this is set to ``None`` instead."""

    rna_gene_variances: Optional[ModelGeneVariancesResults]
    """Results of :py:func:`~scranpy.model_gene_variances.model_gene_variances`.
    If RNA data is not available, this is set to ``None`` instead."""

    rna_highly_variable_genes: Optional[numpy.ndarray]
    """Results of :py:func:`~scranpy.choose_highly_variable_genes.choose_highly_variable_genes`.
    If RNA data is not available, this is set to ``None`` instead."""

    rna_pca: Optional[RunPcaResults]
    """Results of calling :py:func:`~scranpy.run_pca.run_pca` on the RNA log-expression matrix.
    If RNA data is not available, this is set to ``None`` instead."""

    adt_pca: Optional[RunPcaResults]
    """Results of calling :py:func:`~scranpy.run_pca.run_pca` on the ADT log-expression matrix.
    If ADT data is not available, this is set to ``None`` instead."""

    crispr_pca: Optional[RunPcaResults]
    """Results of calling :py:func:`~scranpy.run_pca.run_pca` on the CRISPR log-expression matrix.
    If CRISPR data is not available, this is set to ``None`` instead."""

    combined_pca: Union[Literal["rna_pca", "adt_pca", "crispr_pca"], ScaleByNeighborsResults]
    """If only one modality is used for the downstream analysis, this is a string specifying the attribute containing the components to be used.
    If multiple modalities are to be combined for downstream analysis, this contains the results of :py:func:`~scranpy.scale_by_neighbors.scale_by_neighbors` on the PCs of those modalities."""

    batch_corrected: Optional[CorrectMnnResults]
    """Results of :py:func:`~scranpy.correct_mnn.correct_mnn`.
    If no blocking factor is supplied, this is set to ``None`` instead."""

    tsne: Optional[numpy.ndarray]
    """Results of :py:func:`~scranpy.run_tsne.run_tsne`.
    This is ``None`` if t-SNE was not performed."""

    umap: Optional[numpy.ndarray]
    """Results of :py:func:`~scranpy.run_umap.run_umap`. 
    This is ``None`` if UMAP was not performed."""

    snn_graph: Optional[GraphComponents]
    """Results of :py:func:`~scranpy.build_snn_graph.build_snn_graph`. 
    This is ``None`` if graph-based clustering was not performed."""

    graph_clusters: Optional[ClusterGraphResults]
    """Results of :py:func:`~scranpy.cluster_graph.cluster_graph`.
    This is ``None`` if graph-based clustering was not performed."""

    kmeans_clusters: Optional[ClusterGraphResults]
    """Results of :py:func:`~scranpy.cluster_kmeans.cluster_kmeans`.
    This is ``None`` if k-means clustering was not performed."""

    rna_markers: Optional[RunPcaResults]
    """Results of calling :py:func:`~scranpy.score_markers.score_markers` on the RNA log-expression matrix.
    If RNA data is not available, this is set to ``None`` instead.
    This will also be ``None`` if no suitable clusterings are available."""

    adt_markers: Optional[RunPcaResults]
    """Results of calling :py:func:`~scranpy.score_markers.score_markers` on the ADT log-expression matrix.
    If ADT data is not available, this is set to ``None`` instead.
    This will also be ``None`` if no suitable clusterings are available."""

    crispr_markers: Optional[RunPcaResults]
    """Results of calling :py:func:`~scranpy.score_markers.score_markers` on the CRISPR log-expression matrix.
    If CRISPR data is not available, this is set to ``None`` instead.
    This will also be ``None`` if no suitable clusterings are available."""


def analyze(
    rna_x: Optional[Any],
    adt_x: Optional[Any] = None,
    crispr_x: Optional[Any] = None,
    block: Optional[Sequence] = None,
    rna_subsets: Union[Mapping, Sequence] = [],
    adt_subsets: Union[Mapping, Sequence] = [],
    suggest_rna_qc_thresholds_options: dict = {},
    suggest_adt_qc_thresholds_options: dict = {},
    suggest_crispr_qc_thresholds_options: dict = {},
    filter_cells: bool = True,
    center_size_factors_options: dict = {},
    compute_clrm1_factors_options: dict = {},
    normalize_counts_options: dict = {},
    model_gene_variances_options: dict = {},
    choose_highly_variable_genes_options: dict = {},
    run_pca_options: dict = {},
    use_rna_pcs: bool = True,
    use_adt_pcs: bool = True,
    use_crispr_pcs: bool = True,
    scale_by_neighbors_options: dict = {},
    correct_mnn_options: dict= {},
    run_umap_options: Optional[dict] = {},
    run_tsne_options: Optional[dict] = {},
    build_snn_graph_options: Optional[dict] = {},
    cluster_graph_options: dict = {},
    run_all_neighbor_steps_options: dict = {},
    kmeans_clusters: Optional[int] = None,
    cluster_kmeans_options: dict = {},
    clusters_for_markers: list = ["graph", "kmeans"],
    score_markers_options: dict = {},
    nn_parameters: knncolle.Parameters = knncolle.AnnoyParameters(),
    num_threads: int = 3
) -> AnalyzeResults:
    """Run through a simple single-cell analysis pipeline, starting from a count matrix and ending with clusters, visualizations and markers.
    This also supports integration of multiple modalities and correction of batch effects.

    Args:
        rna_x:
            A matrix-like object containing RNA counts.
            This should have the same number of columns as the other ``x_*`` arguments.
            Alternatively ``None``, if no RNA counts are available.

        adt_x:
            A matrix-like object containing ADT counts.
            This should have the same number of columns as the other ``x_*`` arguments.
            Alternatively ``None``, if no ADT counts are available.

        crispr_x:
            A matrix-like object containing CRISPR counts.
            This should have the same number of columns as the other ``x_*`` arguments.
            Alternatively ``None``, if no CRISPR counts are available.

        block:
            Factor specifying the block of origin (e.g., batch, sample) for each cell in the ``x_*`` matrices.
            Alternatively ``None``, if all cells are from the same block.

        rna_subsets:
            Gene subsets for quality control, typically used for mitochondrial genes.
            Check out the ``subsets`` arguments in :py:func:`~scranpy.rna_quality_control.compute_rna_qc_metrics` for details.

        adt_subsets:
            ADT subsets for quality control, typically used for IgG controls.
            Check out the ``subsets`` arguments in :py:func:`~scranpy.adt_quality_control.compute_adt_qc_metrics` for details.

        suggest_rna_qc_thresholds_options:
            Arguments to pass to :py:func:`~scranpy.rna_quality_control.suggest_rna_qc_thresholds`.

        suggest_adt_qc_thresholds_options:
            Arguments to pass to :py:func:`~scranpy.adt_quality_control.suggest_adt_qc_thresholds`.

        suggest_crispr_qc_thresholds_options:
            Arguments to pass to :py:func:`~scranpy.crispr_quality_control.suggest_crispr_qc_thresholds`.

        filter_cells:
            Whether to filter the count matrices to only retain high-quality cells in all modalities.
            If ``False``, QC metrics and thresholds are still computed but are not used to filter the count matrices.

        center_size_factors_options:
            Arguments to pass to :py:func:`~scranpy.center_size_factors.center_size_factors`.

        compute_clrm1_factors_options:
            Arguments to pass to :py:func:`~scranpy.compute_clrm1_factors.compute_clrm1_factors`.

        normalized_counts_options:
            Arguments to pass to :py:func:`~scranpy.normalize_counts.normalize_counts`.

        model_gene_variances_options:
            Arguments to pass to :py:func:`~scranpy.model_gene_variances.model_gene_variances`.

        choose_highly_variable_genes_options:
            Arguments to pass to :py:func:`~scranpy.choose_highly_variable_genes.choose_highly_variable_genes`.

        run_pca_options:
            Arguments to pass to :py:func:`~scranpy.run_pca.run_pca`.

        use_rna_pcs:
            Whether to use the RNA-derived PCs for downstream steps (i.e., clustering, visualization).

        use_adt_pcs:
            Whether to use the ADT-derived PCs for downstream steps (i.e., clustering, visualization).

        use_crispr_pcs:
            Whether to use the CRISPR-derived PCs for downstream steps (i.e., clustering, visualization).

        scale_by_neighbors_options:
            Arguments to pass to :py:func:`~scranpy.scale_by_neighbors.scale_by_neighbors`.
            Only used if multiple modalities are available and their corresponding ``use_*`` arguments are ``True``.

        correct_mnn_options:
            Arguments to pass to :py:func:`~scranpy.correct_mnn.correct_mnn`.
            Only used if ``block`` is supplied.

        run_umap_options:
            Arguments to pass to :py:func:`~scranpy.run_umap.run_umap`.
            If ``None``, UMAP is not performed.

        run_tsne_options:
            Arguments to pass to :py:func:`~scranpy.run_tsne.run_tsne`.
            If ``None``, t-SNE is not performed.

        build_snn_graph_options:
            Arguments to pass to :py:func:`~scranpy.build_snn_graph.build_snn_graph`.
            Ignored if ``cluster_graph_options = None``.

        cluster_graph_options:
            Arguments to pass to :py:func:`~scranpy.cluster_graph.cluster_graph`.
            If ``None``, graph-based clustering is not performed.

        run_all_neighbor_steps_options:
            Arguments to pass to :py:func:`~scranpy.run_all_neighbor_steps.run_all_neighbor_steps`.

        kmeans_clusters:
            Number of clusters to use in k-means clustering.
            If ``None``, k-means clustering is not performed.

        cluster_kmeans_options:
            Arguments to pass to :py:func:`~scranpy.cluster_kmeans.cluster_kmeans`.
            Ignored if ``kmeans_clusters = None``.

        clusters_for_markers:
            List of clustering algorithms (either ``graph`` or ``kmeans``), specifying the clustering to be used for marker detection.
            The first available clustering will be chosen.

        score_markers_options:
            Arguments to pass to :py:func:`~scranpy.score_markers.score_markers`.
            Ignored if no suitable clusterings are available.

        nn_parameters:
            Algorithm to use for nearest-neighbor searches in the various steps.

        num_threads:
            Number of threads to use in each step.

    Returns:
        The results of the entire analysis, including the results from each step.
    """
    store = {}
    all_ncols = set() 
    if not rna_x is None:
        all_ncols.add(rna_x.shape[1])
    if not adt_x is None:
        all_ncols.add(adt_x.shape[1])
    if not crispr_x is None:
        all_ncols.add(crispr_x.shape[1])
    if len(all_ncols) > 1:
        raise ValueError("'x_*' have differing numbers of columns")
    if len(all_ncols) == 0:
        raise ValueError("at least one 'x_*' must be non-None")
    ncells = list(all_ncols)[0]

    ############ Quality control o(*°▽°*)o #############

    if not rna_x is None:
        store["rna_qc_metrics"] = compute_rna_qc_metrics(rna_x, subsets=rna_subsets, num_threads=num_threads)
        store["rna_qc_thresholds"] = suggest_rna_qc_thresholds(store["rna_qc_metrics"], block=block, **suggest_rna_qc_thresholds_options)
        store["rna_qc_filter"] = filter_rna_qc_metrics(store["rna_qc_thresholds"], store["rna_qc_metrics"], block=block)
    else:
        store["rna_qc_metrics"] = None
        store["rna_qc_thresholds"] = None
        store["rna_qc_filter"] = None

    if not adt_x is None:
        store["adt_qc_metrics"] = compute_adt_qc_metrics(adt_x, subsets=adt_subsets, num_threads=num_threads)
        store["adt_qc_thresholds"] = suggest_adt_qc_thresholds(store["adt_qc_metrics"], block=block, **suggest_adt_qc_thresholds_options)
        store["adt_qc_filter"] = filter_adt_qc_metrics(store["adt_qc_thresholds"], store["adt_qc_metrics"], block=block)
    else:
        store["adt_qc_metrics"] = None
        store["adt_qc_thresholds"] = None
        store["adt_qc_filter"] = None

    if not crispr_x is None:
        store["crispr_qc_metrics"] = compute_crispr_qc_metrics(crispr_x, num_threads=num_threads)
        store["crispr_qc_thresholds"] = suggest_crispr_qc_thresholds(store["crispr_qc_metrics"], block=block, **suggest_crispr_qc_thresholds_options)
        store["crispr_qc_filter"] = filter_crispr_qc_metrics(store["crispr_qc_thresholds"], store["crispr_qc_metrics"], block=block)
    else:
        store["crispr_qc_metrics"] = None
        store["crispr_qc_thresholds"] = None
        store["crispr_qc_filter"] = None

    # Combining all filters.
    combined_qc_filter = numpy.full((ncells,), True)
    for mod in ["rna", "adt", "crispr"]:
        modality_filter = store[mod + "_qc_filter"]
        if not modality_filter is None:
            combined_qc_filter = numpy.logical_and(combined_qc_filter, modality_filter)
    store["combined_qc_filter"] = combined_qc_filter

    if not rna_x is None:
        store["rna_filtered"] = delayedarray.DelayedArray(rna_x)
    else:
        store["rna_filtered"] = None
    if not adt_x is None:
        store["adt_filtered"] = delayedarray.DelayedArray(adt_x)
    else:
        store["adt_filtered"] = None
    if not crispr_x is None:
        store["crispr_filtered"] = delayedarray.DelayedArray(crispr_x)
    else:
        store["crispr_filtered"] = None

    if filter_cells:
        if not rna_x is None:
            store["rna_filtered"] = store["rna_filtered"][:,combined_qc_filter]
        if not adt_x is None:
            store["adt_filtered"] = store["adt_filtered"][:,combined_qc_filter]
        if not crispr_x is None:
            store["crispr_filtered"] = store["crispr_filtered"][:,combined_qc_filter]
        if not block is None:
            block = biocutils.subset_sequence(block, numpy.where(combined_qc_filter)[0])

    ############ Normalization ( ꈍᴗꈍ) #############

    def maybe_filter(x):
        if not filter_cells or combined_qc_filter.all():
            return x
        return x[combined_qc_filter]

    if not rna_x is None:
        store["rna_size_factors"] = center_size_factors(maybe_filter(store["rna_qc_metrics"].sum), block=block, **center_size_factors_options, in_place=False)
        store["rna_normalized"] = normalize_counts(store["rna_filtered"], store["rna_size_factors"], **normalize_counts_options)
    else:
        store["rna_size_factors"] = None
        store["rna_normalized"] = None

    if not adt_x is None:
        raw_adt_sf = compute_clrm1_factors(store["adt_filtered"], **compute_clrm1_factors_options, num_threads=num_threads)
        store["adt_size_factors"] = center_size_factors(raw_adt_sf, block=block, **center_size_factors_options)
        store["adt_normalized"] = normalize_counts(store["adt_filtered"], store["adt_size_factors"], **normalize_counts_options)
    else:
        store["adt_size_factors"] = None
        store["adt_normalized"] = None

    if not crispr_x is None:
        store["crispr_size_factors"] = center_size_factors(maybe_filter(store["crispr_qc_metrics"].sum), block=block, **center_size_factors_options, in_place=False)
        store["crispr_normalized"] = normalize_counts(store["crispr_filtered"], store["crispr_size_factors"], **normalize_counts_options)
    else:
        store["crispr_size_factors"] = None
        store["crispr_normalized"] = None

    ############ Variance modelling (～￣▽￣)～ #############

    if not rna_x is None:
        store["rna_gene_variances"] = model_gene_variances(store["rna_normalized"], block=block, **model_gene_variances_options, num_threads=num_threads)
        store["rna_highly_variable_genes"] = choose_highly_variable_genes(store["rna_gene_variances"].residual, **choose_highly_variable_genes_options)
    else:
        store["rna_gene_variances"] = None
        store["rna_highly_variable_genes"] = None

    ############ Principal components analysis \(>⩊<)/ #############

    if not rna_x is None:
        store["rna_pca"] = run_pca(store["rna_normalized"][store["rna_highly_variable_genes"],:], **run_pca_options, block=block, num_threads=num_threads)
    else:
        store["rna_pca"] = None

    if not adt_x is None:
        store["adt_pca"] = run_pca(store["adt_normalized"], **run_pca_options, block=block, num_threads=num_threads)
    else:
        store["adt_pca"] = None

    if not crispr_x is None:
        store["crispr_pca"] = run_pca(store["crispr_normalized"], **run_pca_options, block=block, num_threads=num_threads)
    else:
        store["crispr_pca"] = None

    ############ Combining modalities and batches („• ᴗ •„) #############

    embeddings = []
    if use_rna_pcs and not rna_x is None:
        embeddings.append("rna_pca")
    if use_adt_pcs and not adt_x is None:
        embeddings.append("adt_pca")
    if use_crispr_pcs and not crispr_x is None:
        embeddings.append("crispr_pca")

    if len(embeddings) == 0:
        raise ValueError("at least one 'use_*' must be True")
    if len(embeddings) == 1:
        store["combined_pca"] = embeddings[0]
    else:
        embeddings = [store[e].components for e in embeddings]
        store["combined_pca"] = scale_by_neighbors(embeddings, **scale_by_neighbors_options, nn_parameters=nn_parameters, num_threads=num_threads)

    if isinstance(store["combined_pca"], str):
        chosen_pcs = store[store["combined_pca"]].components
    else:
        chosen_pcs = store["combined_pca"].combined

    if block is None:
        store["batch_corrected"] = None
    else:
        store["batch_corrected"] = correct_mnn(chosen_pcs, block=block, **correct_mnn_options, nn_parameters=nn_parameters, num_threads=num_threads)
        chosen_pcs = store["batch_corrected"].corrected

    ############ Assorted neighbor-related stuff ⸜(⸝⸝⸝´꒳`⸝⸝⸝)⸝ #############

    all_out = run_all_neighbor_steps(
        chosen_pcs,
        run_umap_options=run_umap_options, 
        run_tsne_options=run_tsne_options, 
        build_snn_graph_options=build_snn_graph_options,
        cluster_graph_options=cluster_graph_options,
        nn_parameters=nn_parameters,
        **run_all_neighbor_steps_options,
        num_threads=num_threads
    )

    store["tsne"] = all_out.run_tsne
    store["umap"] = all_out.run_umap
    store["snn_graph"] = all_out.build_snn_graph
    store["graph_clusters"] = all_out.cluster_graph

    ############ Finding markers (˶˃ ᵕ ˂˶) #############

    if not kmeans_clusters is None:
        store["kmeans_clusters"] = cluster_kmeans(chosen_pcs, k=kmeans_clusters, **cluster_kmeans_options, num_threads=num_threads)
    else:
        store["kmeans_clusters"] = None

    chosen_clusters = None
    for c in clusters_for_markers:
        if c == "graph" and not store["graph_clusters"] is None:
            chosen_clusters = store["graph_clusters"].membership
            break
        elif c == "kmeans" and not store["kmeans_clusters"] is None:
            chosen_clusters = store["kmeans_clusters"].clusters
            break

    store["rna_markers"] = None
    store["adt_markers"] = None
    store["crispr_markers"] = None

    if not chosen_clusters is None:
        if not rna_x is None:
            store["rna_markers"] = score_markers(store["rna_normalized"], groups=chosen_clusters, num_threads=num_threads, block=block, **score_markers_options)
        if not adt_x is None:
            store["adt_markers"] = score_markers(store["adt_normalized"], groups=chosen_clusters, num_threads=num_threads, block=block, **score_markers_options)
        if not crispr_x is None:
            store["crispr_markers"] = score_markers(store["crispr_normalized"], groups=chosen_clusters, num_threads=num_threads, block=block, **score_markers_options)

    return AnalyzeResults(**store)
