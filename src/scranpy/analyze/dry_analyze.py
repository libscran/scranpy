# DO NOT MODIFY THIS FILE! This is automatically generated by './scripts/dryrun.py'
# from the source file 'src/scranpy/analyze/live_analyze.py', modify that instead and rerun the script.

from .AnalyzeOptions import AnalyzeOptions

def dry_analyze(options: AnalyzeOptions=AnalyzeOptions()) -> str:
    __commands = ['import scranpy', 'import copy', '']
    __commands.append('results = AnalyzeResults()')
    if do_rna:
        __commands.append('results.rna_quality_control_metrics = scranpy.quality_control.per_cell_rna_qc_metrics(rna_ptr, options=update(options.per_cell_rna_qc_metrics_options))')
        __commands.append('results.rna_quality_control_thresholds = scranpy.quality_control.suggest_rna_qc_filters(results.rna_quality_control_metrics, options=update(options.suggest_rna_qc_filters_options, block=options.miscellaneous_options.block))')
        __commands.append('results.rna_quality_control_filter = scranpy.quality_control.create_rna_qc_filter(results.rna_quality_control_metrics, results.rna_quality_control_thresholds, options=update(options.create_rna_qc_filter_options, block=options.miscellaneous_options.block))')
    if do_adt:
        __commands.append('results.adt_quality_control_metrics = scranpy.quality_control.per_cell_adt_qc_metrics(adt_ptr, options=update(options.per_cell_adt_qc_metrics_options))')
        __commands.append('results.adt_quality_control_thresholds = scranpy.quality_control.suggest_adt_qc_filters(results.adt_quality_control_metrics, options=update(options.suggest_adt_qc_filters_options, block=options.miscellaneous_options.block))')
        __commands.append('results.adt_quality_control_filter = scranpy.quality_control.create_adt_qc_filter(results.adt_quality_control_metrics, results.adt_quality_control_thresholds, options=update(options.create_adt_qc_filter_options, block=options.miscellaneous_options.block))')
    if do_crispr:
        __commands.append('results.crispr_quality_control_metrics = scranpy.quality_control.per_cell_crispr_qc_metrics(crispr_ptr, options=update(options.per_cell_crispr_qc_metrics_options, subsets=subsets))')
        __commands.append('results.crispr_quality_control_thresholds = scranpy.quality_control.suggest_crispr_qc_filters(results.crispr_quality_control_metrics, options=update(options.suggest_crispr_qc_filters_options, block=options.miscellaneous_options.block))')
        __commands.append('results.crispr_quality_control_filter = scranpy.quality_control.create_crispr_qc_filter(results.crispr_quality_control_metrics, results.crispr_quality_control_thresholds, options=update(options.create_crispr_qc_filter_options, block=options.miscellaneous_options.block))')
    if do_multiple:
        __commands.append('discard = False')
        if do_rna:
            __commands.append('discard = numpy.logical_or(discard, results.rna_quality_control_filter)')
        if do_adt:
            __commands.append('discard = numpy.logical_or(discard, results.adt_quality_control_filter)')
        if do_crispr:
            __commands.append('discard = numpy.logical_or(discard, results.crispr_quality_control_filter)')
    elif do_rna:
        __commands.append('discard = results.rna_quality_control_filter')
    elif do_adt:
        __commands.append('discard = results.adt_quality_control_filter')
    elif do_crispr:
        __commands.append('discard = results.crispr_quality_control_filter')
    if do_rna:
        __commands.append('rna_filtered = scranpy.quality_control.filter_cells(rna_ptr, filter=discard)')
    if do_adt:
        __commands.append('adt_filtered = scranpy.quality_control.filter_cells(adt_ptr, filter=discard)')
    if do_crispr:
        __commands.append('crispr_filtered = scranpy.quality_control.filter_cells(crispr_ptr, filter=discard)')
    __commands.append('keep = numpy.logical_not(discard)')
    __commands.append('results.quality_control_retained = keep')
    if options.miscellaneous_options.block is not None:
        if isinstance(options.miscellaneous_options.block, numpy.ndarray):
            __commands.append('filtered_block = options.miscellaneous_options.block[keep]')
        else:
            __commands.append('filtered_block = numpy.array(options.miscellaneous_options.block)[keep]')
    else:
        __commands.append('filtered_block = None')
    if do_rna:
        if options.rna_log_norm_counts_options.size_factors is None:
            __commands.append("raw_size_factors = results.rna_quality_control_metrics.column('sums')[keep]")
        else:
            __commands.append('raw_size_factors = options.rna_log_norm_counts_options.size_factors')
        __commands.append('(rna_normed, final_size_factors) = scranpy.normalization.log_norm_counts(rna_filtered, options=update(options.rna_log_norm_counts_options, size_factors=raw_size_factors, center_size_factors_options=update(options.rna_log_norm_counts_options.center_size_factors_options, block=filtered_block), with_size_factors=True))')
        __commands.append('results.rna_size_factors = final_size_factors')
        __commands.append('results.gene_variances = scranpy.feature_selection.model_gene_variances(rna_normed, options=update(options.model_gene_variances_options, block=filtered_block))')
        __commands.append("results.hvgs = scranpy.feature_selection.choose_hvgs(results.gene_variances.column('residuals'), options=options.choose_hvgs_options)")
        __commands.append('results.rna_pca = scranpy.dimensionality_reduction.run_pca(rna_normed, options=update(options.rna_run_pca_options, subset=results.hvgs, block=filtered_block))')
    if do_adt:
        if options.adt_log_norm_counts_options.size_factors is None:
            __commands.append("raw_size_factors = results.adt_quality_control_metrics.column('sums')[keep]")
        else:
            __commands.append('raw_size_factors = options.adt_log_norm_counts_options.size_factors')
        __commands.append('(adt_normed, final_size_factors) = scranpy.normalization.log_norm_counts(adt_filtered, options=update(options.adt_log_norm_counts_options, size_factors=raw_size_factors, center_size_factors_options=update(options.adt_log_norm_counts_options.center_size_factors_options, block=filtered_block), with_size_factors=True))')
        __commands.append('results.adt_size_factors = final_size_factors')
        __commands.append('results.adt_pca = scranpy.dimensionality_reduction.run_pca(adt_normed, options=update(options.adt_run_pca_options, block=filtered_block))')
    if do_crispr:
        if options.crispr_log_norm_counts_options.size_factors is None:
            __commands.append("raw_size_factors = results.crispr_quality_control_metrics.column('sums')[keep]")
        else:
            __commands.append('raw_size_factors = options.crispr_log_norm_counts_options.size_factors')
        __commands.append('(crispr_normed, final_size_factors) = scranpy.normalization.log_norm_counts(crispr_filtered, options=update(options.crispr_log_norm_counts_options, size_factors=raw_size_factors, center_size_factors_options=update(options.crispr_log_norm_counts_options.center_size_factors_options, block=filtered_block), with_size_factors=True))')
        __commands.append('results.crispr_size_factors = final_size_factors')
        __commands.append('results.crispr_pca = scranpy.dimensionality_reduction.run_pca(crispr_normed, options=update(options.crispr_run_pca_options, block=filtered_block))')
    if do_multiple:
        __commands.append('all_embeddings = []')
        if do_rna:
            __commands.append('all_embeddings.append(results.rna_pca.principal_components)')
        if do_adt:
            __commands.append('all_embeddings.append(results.adt_pca.principal_components)')
        if do_crispr:
            __commands.append('all_embeddings.append(results.crispr_pca.principal_components)')
        __commands.append('results.combined_pcs = scranpy.dimensionality_reduction.combine_embeddings(all_embeddings, options=options.combine_embeddings_options)')
        __commands.append('lowdim = results.combined.pcs')
    elif do_rna:
        __commands.append('lowdim = results.rna_pca.principal_components')
    elif do_adt:
        __commands.append('lowdim = results.adt_pca.principal_components')
    elif do_crispr:
        __commands.append('lowdim = results.crispr_pca.principal_components')
    if options.miscellaneous_options.block is not None:
        __commands.append('results.mnn = correct.mnn_correct(lowdim, filtered_block, options=options.mnn_correct_options)')
        __commands.append('lowdim = results.mnn.corrected')
    __commands.append('(get_tsne, get_umap, graph, remaining_threads) = scranpy.run_neighbor_suite(lowdim, build_neighbor_index_options=options.build_neighbor_index_options, find_nearest_neighbors_options=options.find_nearest_neighbors_options, run_umap_options=options.run_umap_options, run_tsne_options=options.run_tsne_options, build_snn_graph_options=options.build_snn_graph_options, num_threads=options.find_nearest_neighbors_options.num_threads)')
    __commands.append('results.snn_graph = graph')
    __commands.append('results.clusters = results.snn_graph.community_multilevel(resolution=options.miscellaneous_options.snn_graph_multilevel_resolution).membership')
    if do_rna:
        __commands.append('results.rna_markers = scranpy.marker_detection.score_markers(rna_normed, grouping=results.clusters, options=update(options.rna_score_markers_options, block=filtered_block, num_threads=remaining_threads))')
    if do_adt:
        __commands.append('results.adt_markers = scranpy.marker_detection.score_markers(adt_normed, grouping=results.clusters, options=update(options.adt_score_markers_options, block=filtered_block, num_threads=remaining_threads))')
    if do_crispr:
        __commands.append('results.crispr_markers = scranpy.marker_detection.score_markers(crispr_normed, grouping=results.clusters, options=update(options.crispr_score_markers_options, block=filtered_block, num_threads=remaining_threads))')
    __commands.append('results.tsne = get_tsne()')
    __commands.append('results.umap = get_umap()')
    return '\n'.join(__commands)
