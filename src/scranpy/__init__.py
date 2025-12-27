import sys

if sys.version_info[:2] >= (3, 8):
    # TODO: Import directly (no need for conditional) when `python_requires = >= 3.8`
    from importlib.metadata import PackageNotFoundError, version  # pragma: no cover
else:
    from importlib_metadata import PackageNotFoundError, version  # pragma: no cover

try:
    # Change here if project is renamed and does not equal the package name
    dist_name = __name__
    __version__ = version(dist_name)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"
finally:
    del version, PackageNotFoundError

from .adt_quality_control import *
from .rna_quality_control import *
from .crispr_quality_control import *
from .normalize_counts import *
from .center_size_factors import *
from .sanitize_size_factors import *
from .compute_clrm1_factors import *
from .choose_pseudo_count import *
from .model_gene_variances import *
from .fit_variance_trend import *
from .choose_highly_variable_genes import *
from .run_pca import *
from .run_tsne import *
from .run_umap import *
from .build_snn_graph import *
from .cluster_graph import *
from .cluster_kmeans import *
from .run_all_neighbor_steps import *
from .score_markers import *
from .summarize_effects import *
from .aggregate_across_cells import *
from .aggregate_across_genes import *
from .combine_factors import *
from .correct_mnn import *
from .subsample_by_neighbors import *
from .scale_by_neighbors import *
from .score_gene_set import *
from .test_enrichment import *
from .analyze import *


import biocutils
if biocutils.package_utils.is_package_installed("summarizedexperiment"):
    from .se_quick_rna_qc import *
    from .se_quick_adt_qc import *
    from .se_quick_crispr_qc import *
    from .se_aggregate_across_cells import *
    from .se_aggregate_across_genes import *
    from .se_choose_rna_hvgs import *
    from .se_normalize_rna_counts import *
    from .se_normalize_adt_counts import *
    from .se_normalize_crispr_counts import *
    from .se_score_gene_set import *
    from .se_score_markers import *


if biocutils.package_utils.is_package_installed("singlecellexperiment"):
    from .se_cluster_graph import *
    from .se_cluster_kmeans import *
    from .se_correct_mnn import *
    from .se_run_pca import *
    from .se_run_tsne import *
    from .se_run_umap import *
    from .se_run_all_neighbor_steps import *
    from .se_scale_by_neighbors import *
