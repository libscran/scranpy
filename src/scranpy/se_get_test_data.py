from typing import Literal

import singlecellexperiment
import numpy
import delayedarray

from .se_quick_rna_qc import quick_rna_qc_se 
from .se_quick_adt_qc import quick_adt_qc_se 
from .se_quick_crispr_qc import quick_crispr_qc_se 
from .se_normalize_rna_counts import normalize_rna_counts_se 
from .se_normalize_adt_counts import normalize_adt_counts_se 
from .se_normalize_crispr_counts import normalize_crispr_counts_se 
from .se_choose_rna_hvgs import choose_rna_hvgs_se
from .se_run_pca import run_pca_se 
from .se_cluster_graph import cluster_graph_se


cache_rna = {}
cache_adt = {}
cache_crispr = {}


def _strings_with_prefix(vec, prefix):
    indices = []
    for i, y in enumerate(vec):
        if y.startswith(prefix):
            indices.append(i)
    return indices


def get_test_rna_data_se(at: Literal["start", "qc", "norm", "hvg", "pca", "cluster"] = "start") -> singlecellexperiment.SingleCellExperiment:
    """
    Get a single-cell RNA-seq dataset with varying levels of processing.
    This contains cells from the mouse brain, obtained with ``scrnaseq.fetch_dataset("zeisel-brain-2015", "2023-12-14")``.
    The main experiment contains RNA counts and the alternative experiments contain ERCC and repeat element counts.

    Args:
        at: 
            The desired level of processing.
            For ``start``, no processing is performed.
            Otherwise, the dataset is returned after quality control (``qc``), normalization (``norm``), feature selection (``hvg``), PCA (``pca``) or clustering (``cluster``).

    Returns:
        The dataset as a :py:class:`~singlecellexperiment.SingleCellExperiment.SingleCellExperiment`.
        This uses caching to avoid recomputation.

    Examples:
        >>> import scranpy
        >>> sce = scranpy.get_test_rna_data_se()
    """

    global cache_rna

    if "start" not in cache_rna:
        import scrnaseq
        cache_rna["start"] = scrnaseq.fetch_dataset("zeisel-brain-2015", "2023-12-14", realize_assays=True)
    sce = cache_rna["start"]
    if at == "start":
        return sce

    if "qc" not in cache_rna:
        sce = quick_rna_qc_se(sce, subsets={ "mito": _strings_with_prefix(sce.get_row_names(), "mt-") }, altexp_proportions=["ERCC"])
        sce = sce[:,sce.get_column_data()["keep"]]
        cache_rna["qc"] = sce
    sce = cache_rna["qc"]
    if at == "qc":
        return sce

    if "norm" not in cache_rna:
        sce = normalize_rna_counts_se(sce, size_factors=sce.get_column_data()["sum"])
        cache_rna["norm"] = sce
    sce = cache_rna["norm"]
    if at == "norm":
        return sce

    if "hvg" not in cache_rna:
        sce = choose_rna_hvgs_se(sce, more_var_args={ "use_min_width": True })
        cache_rna["hvg"] = sce
    sce = cache_rna["hvg"]
    if at == "hvg":
        return sce

    if "pca" not in cache_rna:
        sce = run_pca_se(sce, features=sce.get_row_data()["hvg"])
        cache_rna["pca"] = sce
    sce = cache_rna["pca"]
    if at == "pca":
        return sce

    if "cluster" not in cache_rna:
        sce = cluster_graph_se(sce)
        cache_rna["cluster"] = sce
    sce = cache_rna["cluster"]
    if at == "cluster":
        return sce


def get_test_adt_data_se(at: Literal["start", "qc", "norm", "hvg", "pca"] = "start") -> singlecellexperiment.SingleCellExperiment:
    """
    Get a CITE-seq dataset with varying levels of processing.
    This contains human PBMCs obtained with ``scrnaseq.fetch_dataset("kotliarov-pbmc-2020", "2024-04-18")``.
    The main experiment contains RNA counts and the alternative experiment contains ADT counts.

    Args:
        at: 
            The desired level of processing.
            For ``start``, no processing is performed.
            Otherwise, the dataset is returned after quality control (``qc``), normalization (``norm``), feature selection (``hvg``) or PCA (``pca``). 

    Returns:
        The dataset as a :py:class:`~singlecellexperiment.SingleCellExperiment.SingleCellExperiment`.
        This uses caching to avoid recomputation.

    Examples:
        >>> import scranpy
        >>> sce = scranpy.get_test_adt_data_se()
    """

    global cache_adt

    if "start" not in cache_adt:
        import scrnaseq 
        raw_sce = scrnaseq.fetch_dataset("kotliarov-pbmc-2020", "2024-04-18")
        raw_sce = raw_sce[:,:5000] # Cutting it down a bit for speed.
        raw_sce.set_assay(0, delayedarray.to_scipy_sparse_matrix(raw_sce.get_assay(0)), in_place=True)
        raw_ae = raw_sce.get_alternative_experiment("ADT")
        raw_ae.set_assay(0, delayedarray.to_dense_array(raw_ae.get_assay(0)), in_place=True)
        raw_sce.set_alternative_experiment("ADT", raw_ae, in_place=True)
        cache_adt["start"] = raw_sce
    sce = cache_adt["start"]
    if at == "start":
        return sce

    if "qc" not in cache_adt:
        sce = quick_rna_qc_se(sce, subsets={ "mito": _strings_with_prefix(sce.get_row_names(), "MT-") })
        alt_se = sce.get_alternative_experiment("ADT")
        alt_se = quick_adt_qc_se(alt_se, subsets={ "igg": alt_se.get_row_data()["isotype"] })
        sce = sce.set_alternative_experiment("ADT", alt_se)
        sce = sce[:,numpy.logical_and(sce.get_column_data()["keep"], alt_se.get_column_data()["keep"])]
        cache_adt["qc"] = sce
    sce = cache_adt["qc"]
    if at == "qc":
        return sce

    if "norm" not in cache_adt:
        sce = normalize_rna_counts_se(sce, size_factors=sce.get_column_data()["sum"])
        sce = sce.set_alternative_experiment("ADT", normalize_adt_counts_se(sce.get_alternative_experiment("ADT")))
        cache_adt["norm"] = sce
    sce = cache_adt["norm"]
    if at == "norm":
        return sce

    if "hvg" not in cache_adt:
        sce = choose_rna_hvgs_se(sce)
        cache_adt["hvg"] = sce
    sce = cache_adt["hvg"]
    if at == "hvg":
        return sce

    if "pca" not in cache_adt:
        sce = run_pca_se(sce, features=sce.get_row_data()["hvg"])
        sce = sce.set_alternative_experiment("ADT", run_pca_se(sce.get_alternative_experiment("ADT"), features=None))
        cache_adt["pca"] = sce
    sce = cache_adt["pca"]
    if at == "pca":
        return sce


def get_test_crispr_data_se(at: Literal["start", "qc"] = "start") -> singlecellexperiment.SingleCellExperiment:
    """
    Get a Perturb-seq dataset with varying levels of processing.
    This contains a pancreatic beta cell line obtained with ``scrnaseq.fetch_dataset("cao-pancreas-2025", "2025-10-10", "rqc")``.
    The main experiment contains RNA counts and the alternative experiment contains CRISPR guide counts.

    Args:
        at: 
            The desired level of processing.
            For ``start``, no processing is performed.
            Otherwise, the dataset is returned after quality control (``qc``), normalization (``norm``), feature selection (``hvg``) or PCA (``pca``). 

    Returns:
        The dataset as a :py:class:`~singlecellexperiment.SingleCellExperiment.SingleCellExperiment`.
        This uses caching to avoid recomputation.

    Examples:
        >>> import scranpy
        >>> sce = scranpy.get_test_crispr_data_se()
    """

    global cache_crispr

    if "start" not in cache_crispr:
        import scrnaseq
        raw_sce = scrnaseq.fetch_dataset("cao-pancreas-2025", "2025-10-10", "rqc")
        raw_sce = raw_sce[:,:5000] # Cutting it down a bit for speed.
        raw_sce.set_assay(0, delayedarray.to_scipy_sparse_matrix(raw_sce.get_assay(0)), in_place=True)
        raw_ae = raw_sce.get_alternative_experiment("CRISPR Guide Capture")
        raw_ae.set_assay(0, delayedarray.to_scipy_sparse_matrix(raw_ae.get_assay(0)), in_place=True)
        raw_sce.set_alternative_experiment("CRISPR Guide Capture", raw_ae, in_place=True)
        cache_crispr["start"] = raw_sce
    sce = cache_crispr["start"]
    if at == "start":
        return sce

    if "qc" not in cache_crispr:
        sce = quick_rna_qc_se(sce, subsets={ "mito": _strings_with_prefix(sce.get_row_data()["Symbol"], "MT-") })
        alt_se = sce.get_alternative_experiment("CRISPR Guide Capture")
        alt_se = quick_crispr_qc_se(alt_se)
        sce = sce.set_alternative_experiment("CRISPR Guide Capture", alt_se)
        sce = sce[:,numpy.logical_and(sce.get_column_data()["keep"], alt_se.get_column_data()["keep"])]
        cache_crispr["qc"] = sce
    sce = cache_crispr["qc"]
    if at == "qc":
        return sce
