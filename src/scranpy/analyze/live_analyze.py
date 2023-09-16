from typing import Sequence, Optional
from mattress import tatamize
import numpy

from .. import dimensionality_reduction as dimred
from .. import feature_selection as feat
from .. import marker_detection as mark
from .. import normalization as norm
from .. import quality_control as qc
from .. import batch_correction as correct

from .AnalyzeOptions import AnalyzeOptions
from .AnalyzeResults import AnalyzeResults
from .run_neighbor_suite import run_neighbor_suite
from .update import update
from .._utils import MatrixTypes

__author__ = "ltla, jkanche"
__copyright__ = "ltla"
__license__ = "MIT"


def live_analyze(
    rna_matrix: Optional[MatrixTypes],
    rna_features: Optional[Sequence[str]],
    adt_matrix: Optional[MatrixTypes],
    adt_features: Optional[Sequence[str]],
    crispr_matrix: Optional[MatrixTypes],
    crispr_features: Optional[Sequence[str]],
    options: AnalyzeOptions = AnalyzeOptions(),
) -> AnalyzeResults:

    do_rna = rna_matrix is not None
    do_adt = adt_matrix is not None
    do_crispr = crispr_matrix is not None
    do_multiple = (do_rna + do_adt + do_crispr) > 1

    if do_rna:
        rna_ptr = tatamize(rna_matrix)
        if len(rna_features) != rna_ptr.nrow():
            raise ValueError(
                "length of 'rna_features' should equal the number of rows in 'rna_matrix'"
            )

    if do_adt:
        adt_ptr = tatamize(adt_matrix)
        if len(adt_features) != adt_ptr.nrow():
            raise ValueError(
                "length of 'adt_features' should equal the number of rows in 'adt_matrix'"
            )

    if do_crispr:
        crispr_ptr = tatamize(crispr_matrix)
        if len(crispr_features) != crispr_ptr.nrow():
            raise ValueError(
                "length of 'crispr_features' should equal the number of rows in 'crispr_matrix'"
            )

    # Start of the capture.
    results = AnalyzeResults()

    if do_rna:
        subsets = {}
        if isinstance(options.miscellaneous_options.mito_prefix, str):
            subsets["mito"] = qc.guess_mito_from_symbols(
                rna_features, options.miscellaneous_options.mito_prefix
            )
        results.rna_quality_control_subsets = subsets

        results.rna_quality_control_metrics = qc.per_cell_rna_qc_metrics(
            rna_ptr, options=update(options.per_cell_rna_qc_metrics_options, subsets=subsets)
        )

        results.rna_quality_control_thresholds = qc.suggest_rna_qc_filters(
            results.rna_quality_control_metrics,
            options=update(
                options.suggest_rna_qc_filters_options,
                block=options.miscellaneous_options.block,
            ),
        )

        results.rna_quality_control_filter = qc.create_rna_qc_filter(
            results.rna_quality_control_metrics,
            results.rna_quality_control_thresholds,
            options=update(
                options.create_rna_qc_filter_options,
                block=options.miscellaneous_options.block,
            ),
        )

    if do_adt:
        subsets = {}
        # TODO: store the ADT stuff in here.
        results.adt_quality_control_subsets = subsets

        results.adt_quality_control_metrics = qc.per_cell_adt_qc_metrics(
            adt_ptr, options=update(options.per_cell_adt_qc_metrics_options, subsets=subsets)
        )

        results.adt_quality_control_thresholds = qc.suggest_adt_qc_filters(
            results.adt_quality_control_metrics,
            options=update(
                options.suggest_adt_qc_filters_options,
                block=options.miscellaneous_options.block,
            ),
        )

        results.adt_quality_control_filter = qc.create_adt_qc_filter(
            results.adt_quality_control_metrics,
            results.adt_quality_control_thresholds,
            options=update(
                options.create_adt_qc_filter_options,
                block=options.miscellaneous_options.block,
            ),
        )

    if do_crispr:
        results.crispr_quality_control_metrics = qc.per_cell_crispr_qc_metrics(
            crispr_ptr, options=update(options.per_cell_crispr_qc_metrics_options, subsets=subsets)
        )

        results.crispr_quality_control_thresholds = qc.suggest_crispr_qc_filters(
            results.crispr_quality_control_metrics,
            options=update(
                options.suggest_crispr_qc_filters_options,
                block=options.miscellaneous_options.block,
            ),
        )

        results.crispr_quality_control_filter = qc.create_crispr_qc_filter(
            results.crispr_quality_control_metrics,
            results.crispr_quality_control_thresholds,
            options=update(
                options.create_crispr_qc_filter_options,
                block=options.miscellaneous_options.block,
            ),
        )

    all_filters = []
    if do_rna:
        all_filters.append(results.rna_quality_control_filter)
    if do_adt:
        all_filters.append(results.adt_quality_control_filter)
    if do_crispr:
        all_filters.append(results.crispr_quality_control_filter)

    if do_rna:
        rna_filtered = qc.filter_cells(rna_ptr, filter=(*all_filters,))
    if do_adt:
        adt_filtered = qc.filter_cells(adt_ptr, filter=(*all_filters,))
    if do_crispr:
        crispr_filtered = qc.filter_cells(crispr_ptr, filter=(*all_filters,))

    keep = numpy.logical_not(results.rna_quality_control_filter)
    if options.miscellaneous_options.block is not None:
        if isinstance(options.miscellaneous_options.block, numpy.ndarray):
            filtered_block = options.miscellaneous_options.block[keep]
        else:
            filtered_block = numpy.array(options.miscellaneous_options.block)[keep]
    else:
        filtered_block = None

    if do_rna:
        if options.rna_log_norm_counts_options.size_factors is None:
            raw_size_factors = results.rna_quality_control_metrics.column("sums")[keep]
        else:
            raw_size_factors = options.rna_log_norm_counts_options.size_factors

        rna_normed, final_size_factors = norm.log_norm_counts(
            rna_filtered,
            options=update(
                options.rna_log_norm_counts_options,
                size_factors=raw_size_factors,
                center_size_factors_options=update(
                    options.rna_log_norm_counts_options.center_size_factors_options,
                    block=filtered_block,
                ),
                with_size_factors=True,
            ),
        )

        results.rna_size_factors = final_size_factors

        results.gene_variances = feat.model_gene_variances(
            rna_normed,
            options=update(options.model_gene_variances_options, block=filtered_block),
        )

        results.hvgs = feat.choose_hvgs(
            results.gene_variances.column("residuals"),
            options=options.choose_hvgs_options,
        )

        results.rna_pca = dimred.run_pca(
            rna_normed,
            options=update(
                options.rna_run_pca_options, subset=results.hvgs, block=filtered_block
            ),
        )

    if do_adt:
        if options.adt_log_norm_counts_options.size_factors is None:
            raw_size_factors = results.adt_quality_control_metrics.column("sums")[keep]
        else:
            raw_size_factors = options.adt_log_norm_counts_options.size_factors

        adt_normed, final_size_factors = norm.log_norm_counts(
            adt_filtered,
            options=update(
                options.adt_log_norm_counts_options,
                size_factors=raw_size_factors,
                center_size_factors_options=update(
                    options.adt_log_norm_counts_options.center_size_factors_options,
                    block=filtered_block,
                ),
                with_size_factors=True,
            ),
        )

        results.adt_size_factors = final_size_factors

        results.adt_pca = dimred.run_pca(
            adt_normed,
            options=update(
                options.adt_run_pca_options, block=filtered_block
            ),
        )

    if do_crispr:
        if options.crispr_log_norm_counts_options.size_factors is None:
            raw_size_factors = results.crispr_quality_control_metrics.column("sums")[keep]
        else:
            raw_size_factors = options.crispr_log_norm_counts_options.size_factors

        crispr_normed, final_size_factors = norm.log_norm_counts(
            crispr_filtered,
            options=update(
                options.crispr_log_norm_counts_options,
                size_factors=raw_size_factors,
                center_size_factors_options=update(
                    options.crispr_log_norm_counts_options.center_size_factors_options,
                    block=filtered_block,
                ),
                with_size_factors=True,
            ),
        )

        results.crispr_size_factors = final_size_factors

        results.crispr_pca = dimred.run_pca(
            crispr_normed,
            options=update(
                options.crispr_run_pca_options, block=filtered_block
            ),
        )

    if do_multiple:
        all_embeddings = []
        if do_rna:
            all_embeddings.append(results.rna_pca.principal_components)
        if do_adt:
            all_embeddings.append(results.adt_pca.principal_components)
        if do_crispr:
            all_embeddings.append(results.crispr_pca.principal_components)

        results.combined_pcs = dimred.combine_embeddings(
            all_embeddings,
            options=options.combine_embeddings_options
        )
        lowdim = results.combined.pcs
    else:
        if do_rna:
            lowdim = results.rna_pca.principal_components
        elif do_adt:
            lowdim = results.adt_pca.principal_components
        elif do_crispr:
            lowdim = results.crispr_pca.principal_components

    if options.miscellaneous_options.block is not None:
        results.mnn = correct.mnn_correct(
            lowdim,
            filtered_block,
            options=options.mnn_correct_options,
        )
        lowdim = results.mnn.corrected

    get_tsne, get_umap, graph, remaining_threads = run_neighbor_suite(
        lowdim,
        build_neighbor_index_options=options.build_neighbor_index_options,
        find_nearest_neighbors_options=options.find_nearest_neighbors_options,
        run_umap_options=options.run_umap_options,
        run_tsne_options=options.run_tsne_options,
        build_snn_graph_options=options.build_snn_graph_options,
        num_threads=options.find_nearest_neighbors_options.num_threads,  # using this as the parallelization extent.
    )

    results.snn_graph = graph
    results.clusters = results.snn_graph.community_multilevel(
        resolution=options.miscellaneous_options.snn_graph_multilevel_resolution
    ).membership

    if do_rna:
        results.rna_markers = mark.score_markers(
            rna_normed,
            grouping=results.clusters,
            options=update(
                options.rna_score_markers_options,
                block=filtered_block,
                num_threads=remaining_threads,
            ),
        )

    if do_adt:
        results.adt_markers = mark.score_markers(
            adt_normed,
            grouping=results.clusters,
            options=update(
                options.adt_score_markers_options,
                block=filtered_block,
                num_threads=remaining_threads,
            ),
        )

    if do_crispr:
        results.crispr_markers = mark.score_markers(
            crispr_normed,
            grouping=results.clusters,
            options=update(
                options.crispr_score_markers_options,
                block=filtered_block,
                num_threads=remaining_threads,
            ),
        )

    results.tsne = get_tsne()
    results.umap = get_umap()
    return results
