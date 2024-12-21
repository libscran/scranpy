#include "pybind11/pybind11.h"

void init_adt_quality_control(pybind11::module&);
void init_rna_quality_control(pybind11::module&);
void init_crispr_quality_control(pybind11::module&);
void init_normalize_counts(pybind11::module&);
void init_center_size_factors(pybind11::module&);
void init_sanitize_size_factors(pybind11::module&);
void init_compute_clrm1_factors(pybind11::module&);
void init_choose_pseudo_count(pybind11::module&);
void init_model_gene_variances(pybind11::module&);
void init_fit_variance_trend(pybind11::module&);
void init_choose_highly_variable_genes(pybind11::module&);
void init_run_pca(pybind11::module&);

PYBIND11_MODULE(lib_scranpy, m) {
    init_adt_quality_control(m);
    init_rna_quality_control(m);
    init_crispr_quality_control(m);
    init_normalize_counts(m);
    init_center_size_factors(m);
    init_sanitize_size_factors(m);
    init_compute_clrm1_factors(m);
    init_choose_pseudo_count(m);
    init_model_gene_variances(m);
    init_fit_variance_trend(m);
    init_choose_highly_variable_genes(m);
    init_run_pca(m);
}
