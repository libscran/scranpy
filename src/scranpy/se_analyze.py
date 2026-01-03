from typing import Optional, Union, Literal, Sequence

import numpy
import delayedarray
import summarizedexperiment
import singlecellexperiment 
import biocutils
import knncolle

from .se_quick_rna_qc import quick_rna_qc_se
from .se_quick_adt_qc import quick_adt_qc_se
from .se_quick_crispr_qc import quick_crispr_qc_se

from .se_normalize_rna_counts import normalize_rna_counts_se
from .se_normalize_adt_counts import normalize_adt_counts_se
from .se_normalize_crispr_counts import normalize_crispr_counts_se

from .se_choose_rna_hvgs import choose_rna_hvgs_se
from .se_run_pca import run_pca_se
from .se_scale_by_neighbors import scale_by_neighbors_se
from .se_correct_mnn import correct_mnn_se

from .se_run_all_neighbor_steps import run_all_neighbor_steps_se
from .se_cluster_kmeans import cluster_kmeans_se

from .se_score_markers import score_markers_se


def analyze_se( 
    x: summarizedexperiment.SummarizedExperiment,
    rna_altexp: Optional[Union[int, str, bool]] = False,
    adt_altexp: Optional[Union[int, str, bool]] = None,
    crispr_altexp: Optional[Union[int, str, bool]] = None,
    rna_assay_type: Union[str, int] = "counts",
    adt_assay_type: Union[str, int] = "counts",
    crispr_assay_type: Union[str, int] = "counts",
    block: Optional[Sequence] = None,
    block_name: Optional[str] = "block",
    rna_qc_subsets: Union[dict, Sequence, biocutils.NamedList] = [],
    rna_qc_output_prefix: Optional[str] = None,
    more_rna_qc_args: dict = {},
    adt_qc_subsets: Union[dict, Sequence, biocutils.NamedList] = [],
    adt_qc_output_prefix: Optional[str] = None,
    more_adt_qc_args: dict = {},
    crispr_qc_output_prefix: Optional[str] = None,
    more_crispr_qc_args: dict = {},
    filter_cells: bool = True,
    rna_norm_output_name: str = "logcounts",
    more_rna_norm_args: dict = {},
    adt_norm_output_name: str = "logcounts",
    more_adt_norm_args: dict = {},
    crispr_norm_output_name: str = "logcounts",
    more_crispr_norm_args: dict = {},
    rna_hvg_output_prefix: Optional[str] = None,
    more_rna_hvg_args: dict = {},
    rna_pca_output_name: str = "PCA",
    more_rna_pca_args: dict = {},
    adt_pca_output_name: str = "PCA",
    more_adt_pca_args: dict = {},
    use_rna_pcs: bool = True,
    use_adt_pcs: bool = True,
    scale_output_name: str = "combined",
    more_scale_args: dict = {},
    mnn_output_name: str = "MNN",
    more_mnn_args: dict = {},
    more_umap_args: dict = {},
    more_tsne_args: dict = {},
    cluster_graph_output_name: str = "graph_cluster",
    more_build_graph_args: dict = {},
    more_cluster_graph_args: dict = {},
    more_neighbor_args: dict = {},
    kmeans_clusters: Optional[int] = None,
    kmeans_clusters_output_name = "kmeans_cluster",
    more_kmeans_args: dict = {},
    clusters_for_markers: Sequence[Literal["graph", "kmeans"]] = ["graph", "kmeans"],
    more_rna_marker_args: dict = {},
    more_adt_marker_args: dict = {},
    more_crispr_marker_args: dict = {},
    nn_parameters = knncolle.AnnoyParameters(),
    num_threads: int = 3
) -> biocutils.NamedList:
    """
    Execute a simple single-cell analysis pipeline, starting from a count matrix and ending with clusters, visualizations and markers.
    This is equivalent to:

    - Running :py:func:`~scranpy.se_quick_rna_qc.quick_rna_qc_se`,
      :py:func:`~scranpy.se_quick_adt_qc.quick_adt_qc_se`
      and/or :py:func:`~scranpy.se_quick_crispr_qc.quick_crispr_qc_se`,
      for quality control.
    - Subsetting ``x`` to only retain the high-quality cells in all modalities. 
    - Running :py:func:`~scranpy.se_normalize_rna_counts.normalize_rna_counts_se`,
      :py:func:`~scranpy.se_normalize_adt_counts.normalize_adt_counts_se`
      and/or :py:func:`~scranpy.se_normalize_crispr_counts.normalize_crispr_counts_se`,
      for normalization.
    - Running :py:func:`~scranpy.se_choose_rna_hvgs.choose_rna_hvgs_se` to identify highly variable genes.
    - Running :py:func:`~scranpy.se_run_pca.run_pca_se` on the RNA and/or ADT data.
    - Running :py:func:`~scranpy.se_scale_by_neighbors.scale_by_neighbors_se` if multiple modalities are present.
    - Running :py:func:`~scranpy.se_correct_mnn.correct_mnn_se` if multiple batches are present. 
    - Running :py:func:`~scranpy.se_run_all_neighbor_steps.run_all_neighbor_steps_se` to obtain t-SNE and UMAP coordinates, and to perform graph-based clustering.
    - Running :py:func:`~scranpy.se_cluster_kmeans.cluster_kmeans_se` to perform k-means clustering, if requested. 
    - Running :py:func:`~scranpy.se_score_markers.score_markers_se` to compute markers for each modality based on one of the clusterings. 

    Args:
        x:
            A :py:class:`~summarizedexperiment.SummarizedExperiment` object or one of its subclasses.
            Rows correspond to genomic features (genes, ADTs, or CRISPR guides) and columns correspond to cells.

        rna_altexp:
            Name or index of the alternative experiment of ``x`` containing the RNA data.
            Alternatively ``False``, in which case the main experiment is assumed to contain the RNA data.
            Alternatively ``None``, in which case it is assumed that no RNA data is available.

        adt_altexp:
            Name or index of the alternative experiment of ``x`` containing the ADT data.
            Alternatively ``False``, in which case the main experiment is assumed to contain the ADT data.
            Alternatively ``None``, in which case it is assumed that no ADT data is available.

        crispr_altexp:
            Name or index of the alternative experiment of ``x`` containing the CRISPR data.
            Alternatively ``False``, in which case the main experiment is assumed to contain the CRISPR data.
            Alternatively ``None``, in which case it is assumed that no CRISPR data is available.

        rna_assay_type:
            Name or index of the assay containing the RNA count data.
            Only used if ``rna_altexp`` is not ``None``.

        adt_assay_type:
            Name or index of the assay containing the ADT count data.
            Only used if ``rna_altexp`` is not ``None``.

        crispr_assay_type:
            Name or index of the assay containing the CRISPR count data.
            Only used if ``rna_altexp`` is not ``None``.

        block:
            Blocking factor, as a sequence containing the block of origin (e.g., batch, sample) for each cell in ``x``.
            Alternatively ``None``, if all cells are from the same block.

        block_name:
            Name of the column in which to store the blocking factor in the column data of the output ``SummarizedExperiment``.
            Only used if ``block`` is not ``None``.
            If ``None``, the blocking factor is not stored in the output.

        rna_qc_subsets:
            Passed to :py:func:`~scranpy.se_quick_rna_qc.quick_rna_qc_se` as the ``subsets`` argument.
            Only used if ``rna_altexp`` is not ``None``.

        rna_qc_output_prefix:
            Passed to :py:func:`~scranpy.se_quick_rna_qc.quick_rna_qc_se` as the ``output_prefix`` argument.
            Only used if ``rna_altexp`` is not ``None``.

        more_rna_qc_args:
            Additional arguments to pass to :py:func:`~scranpy.se_quick_rna_qc.quick_rna_qc_se`.
            Only used if ``rna_altexp`` is not ``None``.

        adt_qc_subsets:
            Passed to :py:func:`~scranpy.se_quick_adt_qc.quick_adt_qc_se` as the ``subsets`` argument.
            Only used if ``adt_altexp`` is not ``None``.

        adt_qc_output_prefix:
            Passed to :py:func:`~scranpy.se_quick_adt_qc.quick_adt_qc_se` as the ``output_prefix`` argument.
            Only used if ``adt_altexp`` is not ``None``.

        more_adt_qc_args:
            Additional arguments to pass to :py:func:`~scranpy.se_quick_adt_qc.quick_adt_qc_se`.
            Only used if ``adt_altexp`` is not ``None``.

        crispr_qc_output_prefix:
            Passed to :py:func:`~scranpy.se_quick_crispr_qc.quick_crispr_qc_se` as the ``output_prefix`` argument.
            Only used if ``crispr_altexp`` is not ``None``.

        more_crispr_qc_args:
            Additional arguments to pass to :py:func:`~scranpy.se_quick_crispr_qc.quick_crispr_qc_se`.
            Only used if ``crispr_altexp`` is not ``None``.

        filter_cells:
            Whether to filter ``x`` to only retain high-quality cells in all modalities.
            If ``False`` QC metrics and thresholds are still computed but are not used to filter the count matrices.

        rna_norm_output_name:
            Passed to :py:func:`~scranpy.se_normalize_rna_counts.normalize_rna_counts_se` as the ``output_name`` argument.
            Only used if ``rna_altexp`` is not ``None``.

        more_rna_norm_args:
            Additional arguments to pass to :py:func:`~scranpy.se_normalize_rna_counts.normalize_rna_counts_se`.
            Only used if ``rna_altexp`` is not ``None``.

        adt_norm_output_name:
            Passed to :py:func:`~scranpy.se_normalize_adt_counts.normalize_adt_counts_se` as the ``output_name`` argument.
            Only used if ``adt_altexp`` is not ``None``.

        more_adt_norm_args:
            Additional arguments to pass to :py:func:`~scranpy.se_normalize_adt_counts.normalize_adt_counts_se`.
            Only used if ``adt_altexp`` is not ``None``.

        crispr_norm_output_name:
            Passed to :py:func:`~scranpy.se_normalize_crispr_counts.normalize_crispr_counts_se` as the ``output_name`` argument.
            Only used if ``crispr_altexp`` is not ``None``.

        more_crispr_norm_args:
            Additional arguments to pass to :py:func:`~scranpy.se_normalize_crispr_counts.normalize_crispr_counts_se`.
            Only used if ``crispr_altexp`` is not ``None``.

        rna_hvg_output_prefix:
            Passed to :py:func:`~scranpy.se_choose_rna_hvgs.choose_rna_hvgs_se` as the ``output_prefix`` argument.
            Only used if ``rna_altexp`` is not ``None``.

        more_rna_hvg_args:
            Additional arguments to pass to :py:func:`~scranpy.se_choose_rna_hvgs.choose_rna_hvgs_se`.
            Only used if ``rna_altexp`` is not ``None``.

        rna_pca_output_name:
            Passed to :py:func:`~scranpy.se_run_pca.run_pca_se` as the ``output_name`` argument.
            Only used if ``rna_altexp`` is not ``None``.

        more_rna_pca_args:
            Additional arguments to pass to :py:func:`~scranpy.se_run_pca.run_pca_se`.
            Only used if ``rna_altexp`` is not ``None``.

        adt_pca_output_name:
            Passed to :py:func:`~scranpy.se_run_pca.run_pca_se` as the ``output_name`` argument.
            Only used if ``adt_altexp`` is not ``None``.

        more_adt_pca_args:
            Additional arguments to pass to :py:func:`~scranpy.se_run_pca.run_pca_se`.
            Only used if ``adt_altexp`` is not ``None``.

        use_rna_pcs:
            Whether to use the RNA-derived PCs for downstream steps (i.e., clustering, visualization).
            Only used if ``rna_altexp`` is not ``None``.

        use_adt_pcs:
            Whether to use the ADT-derived PCs for downstream steps (i.e., clustering, visualization).
            Only used if ``adt_altexp`` is not ``None``.

        scale_output_name:
            Passed to :py:func:`~scranpy.se_scale_by_neighbors.scale_by_neighbors_se` as the ``output_name`` argument.
            Only used if multiple modalities are available and their corresponding ``use_*_pcs`` arguments are ``True``.

        more_scale_args:
            Additional arguments to pass to :py:func:`~scranpy.se_scale_by_neighbors.scale_by_neighbors_se`.
            Only used if multiple modalities are available and their corresponding ``use_*_pcs`` arguments are ``True``.

        mnn_output_name:
            Passed to :py:func:`~scranpy.se_correct_mnn.correct_mnn_se` as the ``output_name`` argument.
            Only used if ``block`` is not ``None``.

        more_mnn_args:
            Additional arguments to pass to :py:func:`~scranpy.se_correct_mnn.correct_mnn_se`.
            Only used if ``block`` is not ``None``.

        more_tsne_args:
            Passed to :py:func:`~scranpy.se_run_all_neighbor_steps.run_all_neighbor_steps_se`.

        more_umap_args:
            Passed to :py:func:`~scranpy.se_run_all_neighbor_steps.run_all_neighbor_steps_se`.

        more_build_graph_args:
            Passed to :py:func:`~scranpy.se_run_all_neighbor_steps.run_all_neighbor_steps_se`.

        cluster_graph_output_name:
            Passed to :py:func:`~scranpy.se_run_all_neighbor_steps.run_all_neighbor_steps_se` as ``cluster_output_name``.

        more_cluster_graph_args:
            Passed to :py:func:`~scranpy.se_run_all_neighbor_steps.run_all_neighbor_steps_se`.

        more_neighbor_args:
            Passed to :py:func:`~scranpy.se_run_all_neighbor_steps.run_all_neighbor_steps_se`.

        kmeans_clusters:
            Passed to :py:func:`~scranpy.se_cluster_kmeans.cluster_kmeans_se` as the ``k`` argument.
            If ``None``, k-means clustering is not performed.

        kmeans_clusters_output_name:
            Passed to :py:func:`~scranpy.se_cluster_kmeans.cluster_kmeans_se` as the ``output_name`` argument.
            Ignored if ``kmeans_clusters`` is ``None``.

        more_kmeans_args:
            Additional arguments to pass to :py:func:`~scranpy.se_cluster_kmeans.cluster_kmeans_se`. 
            Ignored if ``kmeans_clusters`` is ``None``.

        clusters_for_markers:
            List of clustering algorithms, specifying the clusters to be used for marker detection.
            The first available clustering will be chosen.
            If no clustering is available from the list, markers will not be computed.

        more_rna_marker_args:
            Additional arguments to pass to :py:func:`~scranpy.se_score_markers.score_markers_se` for the RNA data.
            Ignored if no suitable clusterings are available or if ``rna_altexp`` is ``None``

        more_adt_marker_args:
            Additional arguments to pass to :py:func:`~scranpy.se_score_markers.score_markers_se` for the ADT data.
            Ignored if no suitable clusterings are available or if ``adt_altexp`` is ``None``

        more_crispr_marker_args:
            Additional arguments to pass to :py:func:`~scranpy.se_score_markers.score_markers_se` for the CRISPR data.
            Ignored if no suitable clusterings are available or if ``crispr_altexp`` is ``None``

        nn_parameters:
            Parameters for the nearest-neighbor search.

        num_threads:
            Number of threads to use in each step.

    Returns:
        A :py:class:`~biocutils.NamedList.NamedList` containing:

        - ``x``: a :py:class:`~singlecellexperiment.SingleCellExperiment.SingleCellExperiment` that is a copy of the input ``x``.
          It is also decorated with the results of each analysis step. 
        - ``markers``: a list of list of :py:class:`~biocframe.BiocFrame.BiocFrame` objects containing the marker statistics for each modality.
          Each inner list corresponds to a modality (RNA, ADT, etc.) while each ``BiocFrame`` corresponds to a group.

    Examples:
        >>> import scranpy
        >>> sce = scranpy.get_test_rna_data_se()
        >>> is_mito = [ s.startswith("mt-") for s in sce.get_row_names() ]
        >>> res = scranpy.analyze_se(
        >>>     sce, 
        >>>     rna_qc_subsets={ "mito": is_mito }
        >>> )
        >>> res["x"].get_assay_names()
        >>> res["x"].get_reduced_dimension_names()
        >>> print(res["x"].get_column_data())
        >>> import biocutils 
        >>> print(biocutils.table(res["x"].get_column_data()["graph_cluster"]))
        >>> print(scranpy.preview_markers(res["markers"]["rna"][0]))
    """

    ############ Quality control o(*°▽°*)o #############

    collected_filters = {}

    if rna_altexp is not None:
        tmp = _get_modality(x, rna_altexp)
        tmp = quick_rna_qc_se(
            tmp,
            output_prefix=rna_qc_output_prefix,
            subsets=rna_qc_subsets,
            block=block,
            num_threads=num_threads,
            assay_type=rna_assay_type,
            **more_rna_qc_args
        )
        collected_filters["rna"] = tmp.get_column_data()[_maybe_concat(rna_qc_output_prefix, "keep")]
        x = _set_modality(x, rna_altexp, tmp)

    if adt_altexp is not None:
        tmp = _get_modality(x, adt_altexp)
        tmp = quick_adt_qc_se(
            tmp,
            output_prefix=adt_qc_output_prefix,
            subsets=adt_qc_subsets,
            block=block,
            num_threads=num_threads,
            assay_type=adt_assay_type,
            **more_adt_qc_args
        )
        collected_filters["adt"] = tmp.get_column_data()[_maybe_concat(adt_qc_output_prefix, "keep")]
        x = _set_modality(x, adt_altexp, tmp)

    if crispr_altexp is not None:
        tmp = _get_modality(x, crispr_altexp)
        tmp = quick_crispr_qc_se(
            tmp,
            output_prefix=crispr_qc_output_prefix,
            num_threads=num_threads,
            block=block,
            assay_type=crispr_assay_type,
            **more_crispr_qc_args
        )
        collected_filters["crispr"] = tmp.get_column_data()[_maybe_concat(crispr_qc_output_prefix, "keep")]
        x = _set_modality(x, crispr_altexp, tmp)

    # Combining all filters.
    combined_qc_filter = numpy.ones(x.shape[1], dtype=numpy.bool)
    for keep in collected_filters.values():
        combined_qc_filter = numpy.logical_and(combined_qc_filter, keep)
    if len(collected_filters) > 1:
        df = x.get_column_data()
        df = df.set_column("combined_keep", combined_qc_filter)
        x = x.set_column_data(df)

    if filter_cells:
        x = _delayify_assays(x)
        x = x[:,combined_qc_filter]
        if block is not None:
            filter_indices = []
            for i, k in enumerate(combined_qc_filter):
                if k:
                    filter_indices.append(i)
            block = biocutils.subset(block, filter_indices)

    ############ Normalization ( ꈍᴗꈍ) #############

    if rna_altexp is not None:
        tmp = _get_modality(x, rna_altexp)
        tmp = normalize_rna_counts_se(
            tmp,
            assay_type=rna_assay_type,
            output_name=rna_norm_output_name,
            size_factors=tmp.get_column_data()[_maybe_concat(rna_qc_output_prefix, "sum")],
            block=block,
            **more_rna_norm_args
        )
        x = _set_modality(x, rna_altexp, tmp)

    if adt_altexp is not None:
        tmp = _get_modality(x, adt_altexp)
        tmp = normalize_adt_counts_se(
            tmp,
            assay_type=adt_assay_type,
            output_name=adt_norm_output_name,
            block=block,
            **more_adt_norm_args
        )
        x = _set_modality(x, adt_altexp, tmp)

    if crispr_altexp is not None:
        tmp = _get_modality(x, crispr_altexp)
        tmp = normalize_crispr_counts_se(
            tmp,
            assay_type=crispr_assay_type,
            output_name=crispr_norm_output_name,
            size_factors=tmp.get_column_data()[_maybe_concat(crispr_qc_output_prefix, "sum")],
            block=block,
            **more_crispr_norm_args
        )
        x = _set_modality(x, crispr_altexp, tmp)

    ############ Variance modelling (～￣▽￣)～ #############

    if rna_altexp is not None:
        tmp = _get_modality(x, rna_altexp)
        tmp = choose_rna_hvgs_se(
            tmp,
            output_prefix=rna_hvg_output_prefix,
            block=block,
            **more_rna_hvg_args
        )
        x = _set_modality(x, rna_altexp, tmp)

    ############ Principal components analysis \(>⩊<)/ #############

    if rna_altexp is not None:
        tmp = _get_modality(x, rna_altexp)
        tmp = run_pca_se(
            tmp,
            assay_type=rna_norm_output_name,
            output_name=rna_pca_output_name,
            features=tmp.get_row_data()[_maybe_concat(rna_hvg_output_prefix, "hvg")],
            block=block,
            **more_rna_pca_args
        )
        x = _set_modality(x, rna_altexp, tmp)

    if adt_altexp is not None:
        tmp = _get_modality(x, adt_altexp)
        tmp = run_pca_se(
            tmp,
            assay_type=adt_norm_output_name,
            output_name=adt_pca_output_name,
            features=None,
            block=block,
            **more_adt_pca_args
        )
        x = _set_modality(x, adt_altexp, tmp)

    ############ Combining modalities („• ᴗ •„) #############

    embeddings = []
    if use_rna_pcs and rna_altexp is not None:
        embeddings.append("rna")
    if use_adt_pcs and adt_altexp is not None:
        embeddings.append("adt")

    target_embedding = None
    if len(embeddings) == 0:
        raise ValueError("at least one 'use_*' must be true")

    elif len(embeddings) == 1:
        if embeddings[0] == "rna":
            target_embedding = _define_single_target_embedding(x, rna_altexp, rna_pca_output_name)
        elif embeddings[0] == "adt":
            target_embedding = _define_single_target_embedding(x, adt_altexp, adt_pca_output_name)

    else:
        main_reddims = []
        altexp_reddims = {}
        if "rna" in embeddings:
            _add_source_embedding_to_scale(x, rna_altexp, rna_pca_output_name, main_reddims, altexp_reddims)
        if "adt" in embeddings:
            _add_source_embedding_to_scale(x, adt_altexp, adt_pca_output_name, main_reddims, altexp_reddims)

        x = scale_by_neighbors_se( 
            x,
            output_name=scale_output_name,
            main_reddims=main_reddims,
            altexp_reddims=altexp_reddims,
            block=block,
            num_threads=num_threads,
            nn_parameters=nn_parameters,
            **more_scale_args
        )
        target_embedding = scale_output_name

    ############ Performing batch correction ⸜(｡˃ ᵕ ˂ )⸝ #############

    if block is not None:
        x = correct_mnn_se(
            x,
            output_name=mnn_output_name,
            reddim_type=target_embedding,
            block=block,
            num_threads=num_threads,
            nn_parameters=nn_parameters,
            **more_mnn_args
        )
        target_embedding = mnn_output_name

        if block_name is not None:
            df = x.get_column_data()
            df = df.set_column(block_name, block)
            x = x.set_column_data(df)

    ############ Assorted neighbor-related stuff ⸜(⸝⸝⸝´꒳`⸝⸝⸝)⸝ #############

    x = run_all_neighbor_steps_se(
        x,
        reddim_type=target_embedding,
        cluster_output_name=cluster_graph_output_name,
        more_umap_args=more_umap_args, 
        more_tsne_args=more_tsne_args, 
        more_build_graph_args=more_build_graph_args,
        more_cluster_graph_args=more_cluster_graph_args,
        more_neighbor_args=more_neighbor_args,
        nn_parameters=nn_parameters,
        num_threads=num_threads
    )

    ############ Maybe some k-means clustering ⸜(⸝⸝⸝´꒳`⸝⸝⸝)⸝ #############

    if kmeans_clusters is not None:
        x = cluster_kmeans_se(
            x,
            k=kmeans_clusters,
            reddim_type=target_embedding,
            output_name=kmeans_clusters_output_name,
            num_threads=num_threads,
            **more_kmeans_args
        )

    chosen_clusters = None
    for c in clusters_for_markers:
        if c == "graph":
            name = cluster_graph_output_name
        elif c == "kmeans":
            name = kmeans_clusters_output_name
        else:
            raise ValueError() 
        if name is not None and name in x.get_column_data().get_column_names():
            chosen_clusters = x.get_column_data()[name]
            break

    ############ Finding markers (˶˃ ᵕ ˂˶) #############

    markers = None

    if chosen_clusters is not None:
        markers = biocutils.NamedList([], [])

        if rna_altexp is not None:
            markers["rna"] = score_markers_se(
                _get_modality(x, rna_altexp),
                groups=chosen_clusters,
                assay_type=rna_norm_output_name,
                num_threads=num_threads,
                block=block,
                **more_rna_marker_args
            )

        if adt_altexp is not None:
            markers["adt"] = score_markers_se(
                _get_modality(x, adt_altexp),
                groups=chosen_clusters,
                assay_type=adt_norm_output_name,
                num_threads=num_threads,
                block=block,
                **more_adt_marker_args
            )

        if crispr_altexp is not None:
            markers["crispr"] = score_markers_se(
                _get_modality(x, crispr_altexp),
                groups=chosen_clusters,
                assay_type=crispr_norm_output_name,
                num_threads=num_threads,
                block=block,
                **more_crispr_marker_args
            )

    return biocutils.NamedList([x, markers], ["x", "markers"])


def _delayify_assays(x: summarizedexperiment.SummarizedExperiment) -> summarizedexperiment.SummarizedExperiment:
    for i in x.get_assay_names():
        x = x.set_assay(i, delayedarray.DelayedArray(x.get_assay(i)))

    if isinstance(x, singlecellexperiment.SingleCellExperiment):
        for i in x.get_alternative_experiment_names():
            x = x.set_alternative_experiment(i, _delayify_assays(x.get_alternative_experiment(i)))

    return x


def _maybe_concat(prefix: Optional[str], name: str) -> str:
    if prefix is None:
        return name
    else:
        return prefix + name


def _use_main_experiment(altexp: Union[int, str, bool]) -> bool:
    if not isinstance(altexp, bool):
        return False
    if altexp:
        raise ValueError("boolean '*_altexp' must be False to use the main experiment")
    return True


def _get_modality(x: summarizedexperiment.SummarizedExperiment, altexp: Union[int, str, bool]) -> summarizedexperiment.SummarizedExperiment:
    if _use_main_experiment(altexp):
        return x
    else:
        return x.get_alternative_experiment(altexp)


def _set_modality(
    x: summarizedexperiment.SummarizedExperiment,
    altexp: Union[int, str, bool],
    replacement: summarizedexperiment.SummarizedExperiment
) -> summarizedexperiment.SummarizedExperiment:
    if _use_main_experiment(altexp):
        return replacement
    else:
        return x.set_alternative_experiment(altexp, replacement)


def _define_single_target_embedding(
    x: summarizedexperiment.SummarizedExperiment,
    altexp: Union[int, str, bool],
    output_name: str
):
    if _use_main_experiment(altexp):
        return output_name
    if isinstance(altexp, int):
        altexp = x.get_alternative_experiment_names()[altexp]
    return (altexp, output_name)


def _add_source_embedding_to_scale(
    x: summarizedexperiment.SummarizedExperiment,
    altexp: Union[int, str, bool],
    output_name: str,
    main_reddim: list,
    altexp_reddim: dict 
):
    if _use_main_experiment(altexp):
        main_reddim.append(output_name)
    else:
        if isinstance(altexp, int):
            altexp = x.get_alternative_experiment_names()[altexp]
        if altexp not in altexp_reddim: 
            altexp_reddim[altexp] = []
        altexp_reddim[altexp].append(output_name)
