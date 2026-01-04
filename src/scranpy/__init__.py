import sys

if sys.version_info[:2] >= (3, 8):
    # TODO: Import directly (no need for conditional) when `python_requires = >= 3.8`
    from importlib.metadata import PackageNotFoundError, version  # pragma: no cover
else:
    from importlib_metadata import PackageNotFoundError, version  # pragma: no cover

try:
    # Change here if project is renamed and does not equal the package name
    _dist_name = __name__
    __version__ = version(_dist_name)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"
finally:
    del version, PackageNotFoundError

from ._adt_quality_control import compute_adt_qc_metrics, suggest_adt_qc_thresholds, filter_adt_qc_metrics
from ._rna_quality_control import compute_rna_qc_metrics, suggest_rna_qc_thresholds, filter_rna_qc_metrics
from ._crispr_quality_control import compute_crispr_qc_metrics, suggest_crispr_qc_thresholds, filter_crispr_qc_metrics
from ._normalize_counts import normalize_counts
from ._center_size_factors import center_size_factors
from ._sanitize_size_factors import sanitize_size_factors
from ._compute_clrm1_factors import compute_clrm1_factors
from ._choose_pseudo_count import choose_pseudo_count
from ._model_gene_variances import model_gene_variances
from ._fit_variance_trend import fit_variance_trend
from ._choose_highly_variable_genes import choose_highly_variable_genes
from ._run_pca import run_pca
from ._run_tsne import run_tsne, tsne_perplexity_to_neighbors
from ._run_umap import run_umap
from ._build_snn_graph import build_snn_graph
from ._cluster_graph import cluster_graph
from ._cluster_kmeans import cluster_kmeans
from ._run_all_neighbor_steps import run_all_neighbor_steps
from ._score_markers import score_markers
from ._summarize_effects import summarize_effects
from ._aggregate_across_cells import aggregate_across_cells
from ._aggregate_across_genes import aggregate_across_genes
from ._combine_factors import combine_factors
from ._correct_mnn import correct_mnn
from ._subsample_by_neighbors import subsample_by_neighbors
from ._scale_by_neighbors import scale_by_neighbors
from ._score_gene_set import score_gene_set
from ._test_enrichment import test_enrichment


import biocutils
if biocutils.package_utils.is_package_installed("summarizedexperiment"):
    from ._se_quick_rna_qc import compute_rna_qc_metrics_with_altexps, quick_rna_qc_se, format_compute_rna_qc_metrics_result
    from ._se_quick_adt_qc import quick_adt_qc_se, format_compute_adt_qc_metrics_result
    from ._se_quick_crispr_qc import quick_crispr_qc_se, format_compute_crispr_qc_metrics_result
    from ._se_aggregate_across_cells import aggregate_across_cells_se, aggregate_column_data
    from ._se_aggregate_across_genes import aggregate_across_genes_se
    from ._se_choose_rna_hvgs import choose_rna_hvgs_se
    from ._se_normalize_rna_counts import normalize_rna_counts_se
    from ._se_normalize_adt_counts import normalize_adt_counts_se
    from ._se_normalize_crispr_counts import normalize_crispr_counts_se
    from ._se_score_gene_set import score_gene_set_se
    from ._se_score_markers import score_markers_se, format_score_markers_result, preview_markers


if biocutils.package_utils.is_package_installed("singlecellexperiment"):
    from ._se_cluster_graph import cluster_graph_se
    from ._se_cluster_kmeans import cluster_kmeans_se
    from ._se_correct_mnn import correct_mnn_se
    from ._se_run_pca import run_pca_se
    from ._se_run_tsne import run_tsne_se
    from ._se_run_umap import run_umap_se
    from ._se_run_all_neighbor_steps import run_all_neighbor_steps_se
    from ._se_scale_by_neighbors import scale_by_neighbors_se
    from ._se_get_test_data import get_test_rna_data_se, get_test_adt_data_se, get_test_crispr_data_se
    from ._se_analyze import analyze_se
    from ._analyze import analyze


__all__ = []
_toignore = set(["sys", "biocutils"]) 
for _name in dir():
    if _name.startswith("_") or _name in _toignore:
        continue
    __all__.append(_name)
