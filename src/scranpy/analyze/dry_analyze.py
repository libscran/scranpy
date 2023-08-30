# DO NOT MODIFY THIS FILE! This is automatically generated by './scripts/dryrun.py'
# from the source file 'src/scranpy/analyze/live_analyze.py', modify that instead and rerun the script.

from .AnalyzeOptions import AnalyzeOptions


def dry_analyze(options: AnalyzeOptions = AnalyzeOptions()) -> str:
    __commands = ["import scranpy", "import copy", ""]
    __commands.append("results = AnalyzeResults()")
    __commands.append("subsets = {}")
    if isinstance(options.miscellaneous_options.mito_prefix, str):
        __commands.append("subsets['mito'] = scranpy.quality_control.guess_mito_from_symbols(features, options.miscellaneous_options.mito_prefix)")
    __commands.append('results.rna_quality_control_subsets = subsets')
    __commands.append('results.rna_quality_control_metrics = scranpy.quality_control.per_cell_rna_qc_metrics(matrix, options=update(options.per_cell_rna_qc_metrics_options, subsets=subsets))')
    __commands.append('results.rna_quality_control_thresholds = scranpy.quality_control.suggest_rna_qc_filters(results.rna_quality_control_metrics, options=update(options.suggest_rna_qc_filters_options, block=options.miscellaneous_options.block))')
    __commands.append('results.rna_quality_control_filter = scranpy.quality_control.create_rna_qc_filter(results.rna_quality_control_metrics, results.rna_quality_control_thresholds, options=update(options.create_rna_qc_filter_options, block=options.miscellaneous_options.block))')
    __commands.append('filtered = scranpy.quality_control.filter_cells(matrix, filter=results.rna_quality_control_filter)')
    __commands.append('keep = numpy.logical_not(results.rna_quality_control_filter)')
    if options.miscellaneous_options.block is not None:
        if isinstance(options.miscellaneous_options.block, numpy.ndarray):
            __commands.append('filtered_block = options.miscellaneous_options.block[keep]')
            __commands.append('filtered_block = numpy.array(options.miscellaneous_options.block)[keep]')
        else:
            filtered_block = numpy.array(options.miscellaneous_options.block)[keep]
        __commands.append('filtered_block = None')
    else:
        filtered_block = None
    if options.log_norm_counts_options.size_factors is None:
        __commands.append("results.size_factors = scranpy.normalization.center_size_factors(results.rna_quality_control_metrics.column('sums')[keep], options=update(options.center_size_factors_options, block=filtered_block))")
        __commands.append('results.size_factors = options.log_norm_counts_options.size_factors[keep]')
    else:
        results.size_factors = options.log_norm_counts_options.size_factors[keep]
    __commands.append('normed = scranpy.normalization.log_norm_counts(filtered, options=update(options.log_norm_counts_options, size_factors=results.size_factors))')
    __commands.append('results.gene_variances = scranpy.feature_selection.model_gene_variances(normed, options=update(options.model_gene_variances_options, block=filtered_block))')
    __commands.append("results.hvgs = scranpy.feature_selection.choose_hvgs(results.gene_variances.column('residuals'), options=options.choose_hvgs_options)")
    __commands.append('results.pca = scranpy.dimensionality_reduction.run_pca(normed, options=update(options.run_pca_options, subset=results.hvgs, block=filtered_block))')
    __commands.append('(get_tsne, get_umap, graph, remaining_threads) = scranpy.run_neighbor_suite(results.pca.principal_components, build_neighbor_index_options=options.build_neighbor_index_options, find_nearest_neighbors_options=options.find_nearest_neighbors_options, run_umap_options=options.run_umap_options, run_tsne_options=options.run_tsne_options, build_snn_graph_options=options.build_snn_graph_options, num_threads=options.find_nearest_neighbors_options.num_threads)')
    __commands.append('results.snn_graph = graph')
    __commands.append('results.clusters = results.snn_graph.community_multilevel(resolution=options.miscellaneous_options.snn_graph_multilevel_resolution).membership')
    __commands.append('results.markers = scranpy.marker_detection.score_markers(normed, grouping=results.clusters, options=update(options.score_markers_options, block=filtered_block, num_threads=remaining_threads))')
    __commands.append('results.tsne = get_tsne()')
    __commands.append('results.umap = get_umap()')
    return '\n'.join(__commands)
