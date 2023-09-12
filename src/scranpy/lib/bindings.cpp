/* DO NOT MODIFY: this is automatically generated by the cpptypes */

#include <cstring>
#include <stdexcept>
#include <cstdint>

#ifdef _WIN32
#define PYAPI __declspec(dllexport)
#else
#define PYAPI
#endif

static char* copy_error_message(const char* original) {
    auto n = std::strlen(original);
    auto copy = new char[n + 1];
    std::strcpy(copy, original);
    return copy;
}

void aggregate_across_cells(void*, const int32_t*, int32_t, uint8_t, double*, uint8_t, int32_t*, int32_t);

void* build_neighbor_index(int32_t, int32_t, const double*, uint8_t);

void* build_snn_graph_from_nn_index(const void*, int32_t, const char*, int32_t);

void* build_snn_graph_from_nn_results(const void*, const char*, int32_t);

void center_size_factors(int32_t, double*, uint8_t, uint8_t, uint8_t, const int32_t*);

void choose_hvgs(int32_t, const double*, int32_t, uint8_t*);

void* clone_tsne_status(const void*);

void* clone_umap_status(const void*, double*);

void* combine_factors(int32_t, int32_t, const uintptr_t*, int32_t*);

void create_adt_qc_filter(int, int, const int32_t*, const uintptr_t*, int, const int32_t*, const double*, const uintptr_t*, uint8_t*);

void create_rna_qc_filter(int, int, const double*, const int32_t*, const uintptr_t*, int, const int32_t*, const double*, const double*, const uintptr_t*, uint8_t*);

void downsample_by_neighbors(void*, int32_t*, int32_t);

const double* fetch_multibatch_pca_coordinates(const void*);

int32_t fetch_multibatch_pca_num_dims(const void*);

double fetch_multibatch_pca_total_variance(const void*);

const double* fetch_multibatch_pca_variance_explained(const void*);

int32_t fetch_neighbor_index_ndim(const void*);

int32_t fetch_neighbor_index_nobs(const void*);

int32_t fetch_neighbor_results_k(const void*);

int32_t fetch_neighbor_results_nobs(const void*);

void fetch_neighbor_results_single(const void*, int32_t, int32_t*, double*);

const double* fetch_residual_pca_coordinates(const void*);

int32_t fetch_residual_pca_num_dims(const void*);

double fetch_residual_pca_total_variance(const void*);

const double* fetch_residual_pca_variance_explained(const void*);

const double* fetch_simple_pca_coordinates(const void*);

int32_t fetch_simple_pca_num_dims(const void*);

double fetch_simple_pca_total_variance(const void*);

const double* fetch_simple_pca_variance_explained(const void*);

int32_t fetch_snn_graph_edges(const void*);

const int* fetch_snn_graph_indices(const void*);

const double* fetch_snn_graph_weights(const void*);

int32_t fetch_tsne_status_iteration(const void*);

int32_t fetch_tsne_status_nobs(const void*);

int32_t fetch_umap_status_epoch(const void*);

int32_t fetch_umap_status_nobs(const void*);

int32_t fetch_umap_status_num_epochs(const void*);

void* filter_cells(const void*, const uint8_t*, uint8_t);

void* find_nearest_neighbors(const void*, int32_t, int32_t);

void free_combined_factors(void*);

void free_multibatch_pca(void*);

void free_neighbor_index(void*);

void free_neighbor_results(void*);

void free_residual_pca(void*);

void free_simple_pca(void*);

void free_snn_graph(void*);

void free_tsne_status(void*);

void free_umap_status(void*);

void get_combined_factors_count(void*, int32_t*);

void get_combined_factors_level(void*, int32_t, int32_t*);

int32_t get_combined_factors_size(void*);

void* initialize_tsne(const void*, double, int32_t);

void* initialize_umap(const void*, int32_t, double, double*, int32_t);

void* log_norm_counts(const void*, const double*);

void mnn_correct(int32_t, int32_t, const double*, int32_t, const int32_t*, int32_t, double, int32_t, int32_t, uint8_t, const int32_t*, const char*, uint8_t, double*, int32_t*, int32_t*);

void model_gene_variances(const void*, double*, double*, double*, double*, double, int32_t);

void model_gene_variances_blocked(const void*, double*, double*, double*, double*, int32_t, const int32_t*, uintptr_t*, uintptr_t*, uintptr_t*, uintptr_t*, double, int32_t);

void per_cell_adt_qc_metrics(const void*, int32_t, const uintptr_t*, double*, int32_t*, uintptr_t*, int32_t);

void per_cell_rna_qc_metrics(const void*, int32_t, const uintptr_t*, double*, int32_t*, uintptr_t*, int32_t);

int32_t perplexity_to_k(double);

void randomize_tsne_start(size_t, double*, int32_t);

void* run_multibatch_pca(const void*, const int32_t*, uint8_t, uint8_t, int32_t, uint8_t, const uint8_t*, uint8_t, int32_t);

void* run_residual_pca(const void*, const int32_t*, uint8_t, int32_t, uint8_t, const uint8_t*, uint8_t, int32_t);

void* run_simple_pca(const void*, int32_t, uint8_t, const uint8_t*, uint8_t, int32_t);

void run_tsne(void*, int32_t, double*);

void run_umap(void*, int32_t);

void score_markers(const void*, int32_t, const int32_t*, int32_t, const int32_t*, uint8_t, double, uintptr_t*, uintptr_t*, uintptr_t*, uintptr_t*, uintptr_t*, uintptr_t*, int32_t);

void serialize_neighbor_results(const void*, int32_t*, double*);

void suggest_adt_qc_filters(int32_t, int32_t, int32_t*, uintptr_t*, int32_t, const int32_t*, double*, uintptr_t*, double);

void suggest_rna_qc_filters(int32_t, int32_t, double*, int32_t*, uintptr_t*, int32_t, const int32_t*, double*, double*, uintptr_t*, double);

void* unserialize_neighbor_results(int32_t, int32_t, int32_t*, double*);

extern "C" {

PYAPI void free_error_message(char** msg) {
    delete [] *msg;
}

PYAPI void py_aggregate_across_cells(void* mat, const int32_t* groups, int32_t ngroups, uint8_t do_sums, double* output_sums, uint8_t do_detected, int32_t* output_detected, int32_t nthreads, int32_t* errcode, char** errmsg) {
    try {
        aggregate_across_cells(mat, groups, ngroups, do_sums, output_sums, do_detected, output_detected, nthreads);
    } catch(std::exception& e) {
        *errcode = 1;
        *errmsg = copy_error_message(e.what());
    } catch(...) {
        *errcode = 1;
        *errmsg = copy_error_message("unknown C++ exception");
    }
}

PYAPI void* py_build_neighbor_index(int32_t ndim, int32_t nobs, const double* ptr, uint8_t approximate, int32_t* errcode, char** errmsg) {
    void* output = NULL;
    try {
        output = build_neighbor_index(ndim, nobs, ptr, approximate);
    } catch(std::exception& e) {
        *errcode = 1;
        *errmsg = copy_error_message(e.what());
    } catch(...) {
        *errcode = 1;
        *errmsg = copy_error_message("unknown C++ exception");
    }
    return output;
}

PYAPI void* py_build_snn_graph_from_nn_index(const void* x, int32_t num_neighbors, const char* weight_scheme, int32_t num_threads, int32_t* errcode, char** errmsg) {
    void* output = NULL;
    try {
        output = build_snn_graph_from_nn_index(x, num_neighbors, weight_scheme, num_threads);
    } catch(std::exception& e) {
        *errcode = 1;
        *errmsg = copy_error_message(e.what());
    } catch(...) {
        *errcode = 1;
        *errmsg = copy_error_message("unknown C++ exception");
    }
    return output;
}

PYAPI void* py_build_snn_graph_from_nn_results(const void* x, const char* weight_scheme, int32_t num_threads, int32_t* errcode, char** errmsg) {
    void* output = NULL;
    try {
        output = build_snn_graph_from_nn_results(x, weight_scheme, num_threads);
    } catch(std::exception& e) {
        *errcode = 1;
        *errmsg = copy_error_message(e.what());
    } catch(...) {
        *errcode = 1;
        *errmsg = copy_error_message("unknown C++ exception");
    }
    return output;
}

PYAPI void py_center_size_factors(int32_t num, double* size_factors, uint8_t allow_zeros, uint8_t allow_non_finite, uint8_t use_block, const int32_t* block, int32_t* errcode, char** errmsg) {
    try {
        center_size_factors(num, size_factors, allow_zeros, allow_non_finite, use_block, block);
    } catch(std::exception& e) {
        *errcode = 1;
        *errmsg = copy_error_message(e.what());
    } catch(...) {
        *errcode = 1;
        *errmsg = copy_error_message("unknown C++ exception");
    }
}

PYAPI void py_choose_hvgs(int32_t len, const double* stat, int32_t top, uint8_t* output, int32_t* errcode, char** errmsg) {
    try {
        choose_hvgs(len, stat, top, output);
    } catch(std::exception& e) {
        *errcode = 1;
        *errmsg = copy_error_message(e.what());
    } catch(...) {
        *errcode = 1;
        *errmsg = copy_error_message("unknown C++ exception");
    }
}

PYAPI void* py_clone_tsne_status(const void* ptr, int32_t* errcode, char** errmsg) {
    void* output = NULL;
    try {
        output = clone_tsne_status(ptr);
    } catch(std::exception& e) {
        *errcode = 1;
        *errmsg = copy_error_message(e.what());
    } catch(...) {
        *errcode = 1;
        *errmsg = copy_error_message("unknown C++ exception");
    }
    return output;
}

PYAPI void* py_clone_umap_status(const void* ptr, double* cloned, int32_t* errcode, char** errmsg) {
    void* output = NULL;
    try {
        output = clone_umap_status(ptr, cloned);
    } catch(std::exception& e) {
        *errcode = 1;
        *errmsg = copy_error_message(e.what());
    } catch(...) {
        *errcode = 1;
        *errmsg = copy_error_message("unknown C++ exception");
    }
    return output;
}

PYAPI void* py_combine_factors(int32_t length, int32_t number, const uintptr_t* inputs, int32_t* output_combined, int32_t* errcode, char** errmsg) {
    void* output = NULL;
    try {
        output = combine_factors(length, number, inputs, output_combined);
    } catch(std::exception& e) {
        *errcode = 1;
        *errmsg = copy_error_message(e.what());
    } catch(...) {
        *errcode = 1;
        *errmsg = copy_error_message("unknown C++ exception");
    }
    return output;
}

PYAPI void py_create_adt_qc_filter(int num_cells, int num_subsets, const int32_t* detected, const uintptr_t* subset_proportions, int num_blocks, const int32_t* block, const double* detected_thresholds, const uintptr_t* subset_proportions_thresholds, uint8_t* output, int32_t* errcode, char** errmsg) {
    try {
        create_adt_qc_filter(num_cells, num_subsets, detected, subset_proportions, num_blocks, block, detected_thresholds, subset_proportions_thresholds, output);
    } catch(std::exception& e) {
        *errcode = 1;
        *errmsg = copy_error_message(e.what());
    } catch(...) {
        *errcode = 1;
        *errmsg = copy_error_message("unknown C++ exception");
    }
}

PYAPI void py_create_rna_qc_filter(int num_cells, int num_subsets, const double* sums, const int32_t* detected, const uintptr_t* subset_proportions, int num_blocks, const int32_t* block, const double* sums_thresholds, const double* detected_thresholds, const uintptr_t* subset_proportions_thresholds, uint8_t* output, int32_t* errcode, char** errmsg) {
    try {
        create_rna_qc_filter(num_cells, num_subsets, sums, detected, subset_proportions, num_blocks, block, sums_thresholds, detected_thresholds, subset_proportions_thresholds, output);
    } catch(std::exception& e) {
        *errcode = 1;
        *errmsg = copy_error_message(e.what());
    } catch(...) {
        *errcode = 1;
        *errmsg = copy_error_message("unknown C++ exception");
    }
}

PYAPI void py_downsample_by_neighbors(void* ptr, int32_t* output, int32_t num_threads, int32_t* errcode, char** errmsg) {
    try {
        downsample_by_neighbors(ptr, output, num_threads);
    } catch(std::exception& e) {
        *errcode = 1;
        *errmsg = copy_error_message(e.what());
    } catch(...) {
        *errcode = 1;
        *errmsg = copy_error_message("unknown C++ exception");
    }
}

PYAPI const double* py_fetch_multibatch_pca_coordinates(const void* x, int32_t* errcode, char** errmsg) {
    const double* output = NULL;
    try {
        output = fetch_multibatch_pca_coordinates(x);
    } catch(std::exception& e) {
        *errcode = 1;
        *errmsg = copy_error_message(e.what());
    } catch(...) {
        *errcode = 1;
        *errmsg = copy_error_message("unknown C++ exception");
    }
    return output;
}

PYAPI int32_t py_fetch_multibatch_pca_num_dims(const void* x, int32_t* errcode, char** errmsg) {
    int32_t output = 0;
    try {
        output = fetch_multibatch_pca_num_dims(x);
    } catch(std::exception& e) {
        *errcode = 1;
        *errmsg = copy_error_message(e.what());
    } catch(...) {
        *errcode = 1;
        *errmsg = copy_error_message("unknown C++ exception");
    }
    return output;
}

PYAPI double py_fetch_multibatch_pca_total_variance(const void* x, int32_t* errcode, char** errmsg) {
    double output = 0;
    try {
        output = fetch_multibatch_pca_total_variance(x);
    } catch(std::exception& e) {
        *errcode = 1;
        *errmsg = copy_error_message(e.what());
    } catch(...) {
        *errcode = 1;
        *errmsg = copy_error_message("unknown C++ exception");
    }
    return output;
}

PYAPI const double* py_fetch_multibatch_pca_variance_explained(const void* x, int32_t* errcode, char** errmsg) {
    const double* output = NULL;
    try {
        output = fetch_multibatch_pca_variance_explained(x);
    } catch(std::exception& e) {
        *errcode = 1;
        *errmsg = copy_error_message(e.what());
    } catch(...) {
        *errcode = 1;
        *errmsg = copy_error_message("unknown C++ exception");
    }
    return output;
}

PYAPI int32_t py_fetch_neighbor_index_ndim(const void* ptr, int32_t* errcode, char** errmsg) {
    int32_t output = 0;
    try {
        output = fetch_neighbor_index_ndim(ptr);
    } catch(std::exception& e) {
        *errcode = 1;
        *errmsg = copy_error_message(e.what());
    } catch(...) {
        *errcode = 1;
        *errmsg = copy_error_message("unknown C++ exception");
    }
    return output;
}

PYAPI int32_t py_fetch_neighbor_index_nobs(const void* ptr, int32_t* errcode, char** errmsg) {
    int32_t output = 0;
    try {
        output = fetch_neighbor_index_nobs(ptr);
    } catch(std::exception& e) {
        *errcode = 1;
        *errmsg = copy_error_message(e.what());
    } catch(...) {
        *errcode = 1;
        *errmsg = copy_error_message("unknown C++ exception");
    }
    return output;
}

PYAPI int32_t py_fetch_neighbor_results_k(const void* ptr0, int32_t* errcode, char** errmsg) {
    int32_t output = 0;
    try {
        output = fetch_neighbor_results_k(ptr0);
    } catch(std::exception& e) {
        *errcode = 1;
        *errmsg = copy_error_message(e.what());
    } catch(...) {
        *errcode = 1;
        *errmsg = copy_error_message("unknown C++ exception");
    }
    return output;
}

PYAPI int32_t py_fetch_neighbor_results_nobs(const void* ptr, int32_t* errcode, char** errmsg) {
    int32_t output = 0;
    try {
        output = fetch_neighbor_results_nobs(ptr);
    } catch(std::exception& e) {
        *errcode = 1;
        *errmsg = copy_error_message(e.what());
    } catch(...) {
        *errcode = 1;
        *errmsg = copy_error_message("unknown C++ exception");
    }
    return output;
}

PYAPI void py_fetch_neighbor_results_single(const void* ptr0, int32_t i, int32_t* outdex, double* outdist, int32_t* errcode, char** errmsg) {
    try {
        fetch_neighbor_results_single(ptr0, i, outdex, outdist);
    } catch(std::exception& e) {
        *errcode = 1;
        *errmsg = copy_error_message(e.what());
    } catch(...) {
        *errcode = 1;
        *errmsg = copy_error_message("unknown C++ exception");
    }
}

PYAPI const double* py_fetch_residual_pca_coordinates(const void* x, int32_t* errcode, char** errmsg) {
    const double* output = NULL;
    try {
        output = fetch_residual_pca_coordinates(x);
    } catch(std::exception& e) {
        *errcode = 1;
        *errmsg = copy_error_message(e.what());
    } catch(...) {
        *errcode = 1;
        *errmsg = copy_error_message("unknown C++ exception");
    }
    return output;
}

PYAPI int32_t py_fetch_residual_pca_num_dims(const void* x, int32_t* errcode, char** errmsg) {
    int32_t output = 0;
    try {
        output = fetch_residual_pca_num_dims(x);
    } catch(std::exception& e) {
        *errcode = 1;
        *errmsg = copy_error_message(e.what());
    } catch(...) {
        *errcode = 1;
        *errmsg = copy_error_message("unknown C++ exception");
    }
    return output;
}

PYAPI double py_fetch_residual_pca_total_variance(const void* x, int32_t* errcode, char** errmsg) {
    double output = 0;
    try {
        output = fetch_residual_pca_total_variance(x);
    } catch(std::exception& e) {
        *errcode = 1;
        *errmsg = copy_error_message(e.what());
    } catch(...) {
        *errcode = 1;
        *errmsg = copy_error_message("unknown C++ exception");
    }
    return output;
}

PYAPI const double* py_fetch_residual_pca_variance_explained(const void* x, int32_t* errcode, char** errmsg) {
    const double* output = NULL;
    try {
        output = fetch_residual_pca_variance_explained(x);
    } catch(std::exception& e) {
        *errcode = 1;
        *errmsg = copy_error_message(e.what());
    } catch(...) {
        *errcode = 1;
        *errmsg = copy_error_message("unknown C++ exception");
    }
    return output;
}

PYAPI const double* py_fetch_simple_pca_coordinates(const void* x, int32_t* errcode, char** errmsg) {
    const double* output = NULL;
    try {
        output = fetch_simple_pca_coordinates(x);
    } catch(std::exception& e) {
        *errcode = 1;
        *errmsg = copy_error_message(e.what());
    } catch(...) {
        *errcode = 1;
        *errmsg = copy_error_message("unknown C++ exception");
    }
    return output;
}

PYAPI int32_t py_fetch_simple_pca_num_dims(const void* x, int32_t* errcode, char** errmsg) {
    int32_t output = 0;
    try {
        output = fetch_simple_pca_num_dims(x);
    } catch(std::exception& e) {
        *errcode = 1;
        *errmsg = copy_error_message(e.what());
    } catch(...) {
        *errcode = 1;
        *errmsg = copy_error_message("unknown C++ exception");
    }
    return output;
}

PYAPI double py_fetch_simple_pca_total_variance(const void* x, int32_t* errcode, char** errmsg) {
    double output = 0;
    try {
        output = fetch_simple_pca_total_variance(x);
    } catch(std::exception& e) {
        *errcode = 1;
        *errmsg = copy_error_message(e.what());
    } catch(...) {
        *errcode = 1;
        *errmsg = copy_error_message("unknown C++ exception");
    }
    return output;
}

PYAPI const double* py_fetch_simple_pca_variance_explained(const void* x, int32_t* errcode, char** errmsg) {
    const double* output = NULL;
    try {
        output = fetch_simple_pca_variance_explained(x);
    } catch(std::exception& e) {
        *errcode = 1;
        *errmsg = copy_error_message(e.what());
    } catch(...) {
        *errcode = 1;
        *errmsg = copy_error_message("unknown C++ exception");
    }
    return output;
}

PYAPI int32_t py_fetch_snn_graph_edges(const void* ptr, int32_t* errcode, char** errmsg) {
    int32_t output = 0;
    try {
        output = fetch_snn_graph_edges(ptr);
    } catch(std::exception& e) {
        *errcode = 1;
        *errmsg = copy_error_message(e.what());
    } catch(...) {
        *errcode = 1;
        *errmsg = copy_error_message("unknown C++ exception");
    }
    return output;
}

PYAPI const int* py_fetch_snn_graph_indices(const void* ptr, int32_t* errcode, char** errmsg) {
    const int* output = NULL;
    try {
        output = fetch_snn_graph_indices(ptr);
    } catch(std::exception& e) {
        *errcode = 1;
        *errmsg = copy_error_message(e.what());
    } catch(...) {
        *errcode = 1;
        *errmsg = copy_error_message("unknown C++ exception");
    }
    return output;
}

PYAPI const double* py_fetch_snn_graph_weights(const void* ptr, int32_t* errcode, char** errmsg) {
    const double* output = NULL;
    try {
        output = fetch_snn_graph_weights(ptr);
    } catch(std::exception& e) {
        *errcode = 1;
        *errmsg = copy_error_message(e.what());
    } catch(...) {
        *errcode = 1;
        *errmsg = copy_error_message("unknown C++ exception");
    }
    return output;
}

PYAPI int32_t py_fetch_tsne_status_iteration(const void* ptr, int32_t* errcode, char** errmsg) {
    int32_t output = 0;
    try {
        output = fetch_tsne_status_iteration(ptr);
    } catch(std::exception& e) {
        *errcode = 1;
        *errmsg = copy_error_message(e.what());
    } catch(...) {
        *errcode = 1;
        *errmsg = copy_error_message("unknown C++ exception");
    }
    return output;
}

PYAPI int32_t py_fetch_tsne_status_nobs(const void* ptr, int32_t* errcode, char** errmsg) {
    int32_t output = 0;
    try {
        output = fetch_tsne_status_nobs(ptr);
    } catch(std::exception& e) {
        *errcode = 1;
        *errmsg = copy_error_message(e.what());
    } catch(...) {
        *errcode = 1;
        *errmsg = copy_error_message("unknown C++ exception");
    }
    return output;
}

PYAPI int32_t py_fetch_umap_status_epoch(const void* ptr, int32_t* errcode, char** errmsg) {
    int32_t output = 0;
    try {
        output = fetch_umap_status_epoch(ptr);
    } catch(std::exception& e) {
        *errcode = 1;
        *errmsg = copy_error_message(e.what());
    } catch(...) {
        *errcode = 1;
        *errmsg = copy_error_message("unknown C++ exception");
    }
    return output;
}

PYAPI int32_t py_fetch_umap_status_nobs(const void* ptr, int32_t* errcode, char** errmsg) {
    int32_t output = 0;
    try {
        output = fetch_umap_status_nobs(ptr);
    } catch(std::exception& e) {
        *errcode = 1;
        *errmsg = copy_error_message(e.what());
    } catch(...) {
        *errcode = 1;
        *errmsg = copy_error_message("unknown C++ exception");
    }
    return output;
}

PYAPI int32_t py_fetch_umap_status_num_epochs(const void* ptr, int32_t* errcode, char** errmsg) {
    int32_t output = 0;
    try {
        output = fetch_umap_status_num_epochs(ptr);
    } catch(std::exception& e) {
        *errcode = 1;
        *errmsg = copy_error_message(e.what());
    } catch(...) {
        *errcode = 1;
        *errmsg = copy_error_message("unknown C++ exception");
    }
    return output;
}

PYAPI void* py_filter_cells(const void* mat0, const uint8_t* filter, uint8_t discard, int32_t* errcode, char** errmsg) {
    void* output = NULL;
    try {
        output = filter_cells(mat0, filter, discard);
    } catch(std::exception& e) {
        *errcode = 1;
        *errmsg = copy_error_message(e.what());
    } catch(...) {
        *errcode = 1;
        *errmsg = copy_error_message("unknown C++ exception");
    }
    return output;
}

PYAPI void* py_find_nearest_neighbors(const void* index, int32_t k, int32_t nthreads, int32_t* errcode, char** errmsg) {
    void* output = NULL;
    try {
        output = find_nearest_neighbors(index, k, nthreads);
    } catch(std::exception& e) {
        *errcode = 1;
        *errmsg = copy_error_message(e.what());
    } catch(...) {
        *errcode = 1;
        *errmsg = copy_error_message("unknown C++ exception");
    }
    return output;
}

PYAPI void py_free_combined_factors(void* ptr, int32_t* errcode, char** errmsg) {
    try {
        free_combined_factors(ptr);
    } catch(std::exception& e) {
        *errcode = 1;
        *errmsg = copy_error_message(e.what());
    } catch(...) {
        *errcode = 1;
        *errmsg = copy_error_message("unknown C++ exception");
    }
}

PYAPI void py_free_multibatch_pca(void* x, int32_t* errcode, char** errmsg) {
    try {
        free_multibatch_pca(x);
    } catch(std::exception& e) {
        *errcode = 1;
        *errmsg = copy_error_message(e.what());
    } catch(...) {
        *errcode = 1;
        *errmsg = copy_error_message("unknown C++ exception");
    }
}

PYAPI void py_free_neighbor_index(void* ptr, int32_t* errcode, char** errmsg) {
    try {
        free_neighbor_index(ptr);
    } catch(std::exception& e) {
        *errcode = 1;
        *errmsg = copy_error_message(e.what());
    } catch(...) {
        *errcode = 1;
        *errmsg = copy_error_message("unknown C++ exception");
    }
}

PYAPI void py_free_neighbor_results(void* ptr, int32_t* errcode, char** errmsg) {
    try {
        free_neighbor_results(ptr);
    } catch(std::exception& e) {
        *errcode = 1;
        *errmsg = copy_error_message(e.what());
    } catch(...) {
        *errcode = 1;
        *errmsg = copy_error_message("unknown C++ exception");
    }
}

PYAPI void py_free_residual_pca(void* x, int32_t* errcode, char** errmsg) {
    try {
        free_residual_pca(x);
    } catch(std::exception& e) {
        *errcode = 1;
        *errmsg = copy_error_message(e.what());
    } catch(...) {
        *errcode = 1;
        *errmsg = copy_error_message("unknown C++ exception");
    }
}

PYAPI void py_free_simple_pca(void* x, int32_t* errcode, char** errmsg) {
    try {
        free_simple_pca(x);
    } catch(std::exception& e) {
        *errcode = 1;
        *errmsg = copy_error_message(e.what());
    } catch(...) {
        *errcode = 1;
        *errmsg = copy_error_message("unknown C++ exception");
    }
}

PYAPI void py_free_snn_graph(void* ptr, int32_t* errcode, char** errmsg) {
    try {
        free_snn_graph(ptr);
    } catch(std::exception& e) {
        *errcode = 1;
        *errmsg = copy_error_message(e.what());
    } catch(...) {
        *errcode = 1;
        *errmsg = copy_error_message("unknown C++ exception");
    }
}

PYAPI void py_free_tsne_status(void* ptr, int32_t* errcode, char** errmsg) {
    try {
        free_tsne_status(ptr);
    } catch(std::exception& e) {
        *errcode = 1;
        *errmsg = copy_error_message(e.what());
    } catch(...) {
        *errcode = 1;
        *errmsg = copy_error_message("unknown C++ exception");
    }
}

PYAPI void py_free_umap_status(void* ptr, int32_t* errcode, char** errmsg) {
    try {
        free_umap_status(ptr);
    } catch(std::exception& e) {
        *errcode = 1;
        *errmsg = copy_error_message(e.what());
    } catch(...) {
        *errcode = 1;
        *errmsg = copy_error_message("unknown C++ exception");
    }
}

PYAPI void py_get_combined_factors_count(void* ptr, int32_t* output, int32_t* errcode, char** errmsg) {
    try {
        get_combined_factors_count(ptr, output);
    } catch(std::exception& e) {
        *errcode = 1;
        *errmsg = copy_error_message(e.what());
    } catch(...) {
        *errcode = 1;
        *errmsg = copy_error_message("unknown C++ exception");
    }
}

PYAPI void py_get_combined_factors_level(void* ptr, int32_t i, int32_t* output, int32_t* errcode, char** errmsg) {
    try {
        get_combined_factors_level(ptr, i, output);
    } catch(std::exception& e) {
        *errcode = 1;
        *errmsg = copy_error_message(e.what());
    } catch(...) {
        *errcode = 1;
        *errmsg = copy_error_message("unknown C++ exception");
    }
}

PYAPI int32_t py_get_combined_factors_size(void* ptr, int32_t* errcode, char** errmsg) {
    int32_t output = 0;
    try {
        output = get_combined_factors_size(ptr);
    } catch(std::exception& e) {
        *errcode = 1;
        *errmsg = copy_error_message(e.what());
    } catch(...) {
        *errcode = 1;
        *errmsg = copy_error_message("unknown C++ exception");
    }
    return output;
}

PYAPI void* py_initialize_tsne(const void* neighbors, double perplexity, int32_t nthreads, int32_t* errcode, char** errmsg) {
    void* output = NULL;
    try {
        output = initialize_tsne(neighbors, perplexity, nthreads);
    } catch(std::exception& e) {
        *errcode = 1;
        *errmsg = copy_error_message(e.what());
    } catch(...) {
        *errcode = 1;
        *errmsg = copy_error_message("unknown C++ exception");
    }
    return output;
}

PYAPI void* py_initialize_umap(const void* neighbors, int32_t num_epochs, double min_dist, double* Y, int32_t nthreads, int32_t* errcode, char** errmsg) {
    void* output = NULL;
    try {
        output = initialize_umap(neighbors, num_epochs, min_dist, Y, nthreads);
    } catch(std::exception& e) {
        *errcode = 1;
        *errmsg = copy_error_message(e.what());
    } catch(...) {
        *errcode = 1;
        *errmsg = copy_error_message("unknown C++ exception");
    }
    return output;
}

PYAPI void* py_log_norm_counts(const void* mat0, const double* size_factors, int32_t* errcode, char** errmsg) {
    void* output = NULL;
    try {
        output = log_norm_counts(mat0, size_factors);
    } catch(std::exception& e) {
        *errcode = 1;
        *errmsg = copy_error_message(e.what());
    } catch(...) {
        *errcode = 1;
        *errmsg = copy_error_message("unknown C++ exception");
    }
    return output;
}

PYAPI void py_mnn_correct(int32_t ndim, int32_t ncells, const double* x, int32_t nbatches, const int32_t* batch, int32_t k, double nmads, int32_t nthreads, int32_t mass_cap, uint8_t use_order, const int32_t* order, const char* ref_policy, uint8_t approximate, double* corrected_output, int32_t* merge_order_output, int32_t* num_pairs_output, int32_t* errcode, char** errmsg) {
    try {
        mnn_correct(ndim, ncells, x, nbatches, batch, k, nmads, nthreads, mass_cap, use_order, order, ref_policy, approximate, corrected_output, merge_order_output, num_pairs_output);
    } catch(std::exception& e) {
        *errcode = 1;
        *errmsg = copy_error_message(e.what());
    } catch(...) {
        *errcode = 1;
        *errmsg = copy_error_message("unknown C++ exception");
    }
}

PYAPI void py_model_gene_variances(const void* mat, double* means, double* variances, double* fitted, double* residuals, double span, int32_t num_threads, int32_t* errcode, char** errmsg) {
    try {
        model_gene_variances(mat, means, variances, fitted, residuals, span, num_threads);
    } catch(std::exception& e) {
        *errcode = 1;
        *errmsg = copy_error_message(e.what());
    } catch(...) {
        *errcode = 1;
        *errmsg = copy_error_message("unknown C++ exception");
    }
}

PYAPI void py_model_gene_variances_blocked(const void* mat, double* ave_means, double* ave_detected, double* ave_fitted, double* ave_residuals, int32_t num_blocks, const int32_t* block, uintptr_t* block_means, uintptr_t* block_variances, uintptr_t* block_fitted, uintptr_t* block_residuals, double span, int32_t num_threads, int32_t* errcode, char** errmsg) {
    try {
        model_gene_variances_blocked(mat, ave_means, ave_detected, ave_fitted, ave_residuals, num_blocks, block, block_means, block_variances, block_fitted, block_residuals, span, num_threads);
    } catch(std::exception& e) {
        *errcode = 1;
        *errmsg = copy_error_message(e.what());
    } catch(...) {
        *errcode = 1;
        *errmsg = copy_error_message("unknown C++ exception");
    }
}

PYAPI void py_per_cell_adt_qc_metrics(const void* mat, int32_t num_subsets, const uintptr_t* subset_ptrs, double* sum_output, int32_t* detected_output, uintptr_t* subset_output, int32_t num_threads, int32_t* errcode, char** errmsg) {
    try {
        per_cell_adt_qc_metrics(mat, num_subsets, subset_ptrs, sum_output, detected_output, subset_output, num_threads);
    } catch(std::exception& e) {
        *errcode = 1;
        *errmsg = copy_error_message(e.what());
    } catch(...) {
        *errcode = 1;
        *errmsg = copy_error_message("unknown C++ exception");
    }
}

PYAPI void py_per_cell_rna_qc_metrics(const void* mat, int32_t num_subsets, const uintptr_t* subset_ptrs, double* sum_output, int32_t* detected_output, uintptr_t* subset_output, int32_t num_threads, int32_t* errcode, char** errmsg) {
    try {
        per_cell_rna_qc_metrics(mat, num_subsets, subset_ptrs, sum_output, detected_output, subset_output, num_threads);
    } catch(std::exception& e) {
        *errcode = 1;
        *errmsg = copy_error_message(e.what());
    } catch(...) {
        *errcode = 1;
        *errmsg = copy_error_message("unknown C++ exception");
    }
}

PYAPI int32_t py_perplexity_to_k(double perplexity, int32_t* errcode, char** errmsg) {
    int32_t output = 0;
    try {
        output = perplexity_to_k(perplexity);
    } catch(std::exception& e) {
        *errcode = 1;
        *errmsg = copy_error_message(e.what());
    } catch(...) {
        *errcode = 1;
        *errmsg = copy_error_message("unknown C++ exception");
    }
    return output;
}

PYAPI void py_randomize_tsne_start(size_t n, double* Y, int32_t seed, int32_t* errcode, char** errmsg) {
    try {
        randomize_tsne_start(n, Y, seed);
    } catch(std::exception& e) {
        *errcode = 1;
        *errmsg = copy_error_message(e.what());
    } catch(...) {
        *errcode = 1;
        *errmsg = copy_error_message("unknown C++ exception");
    }
}

PYAPI void* py_run_multibatch_pca(const void* mat, const int32_t* block, uint8_t use_residuals, uint8_t equal_weights, int32_t number, uint8_t use_subset, const uint8_t* subset, uint8_t scale, int32_t num_threads, int32_t* errcode, char** errmsg) {
    void* output = NULL;
    try {
        output = run_multibatch_pca(mat, block, use_residuals, equal_weights, number, use_subset, subset, scale, num_threads);
    } catch(std::exception& e) {
        *errcode = 1;
        *errmsg = copy_error_message(e.what());
    } catch(...) {
        *errcode = 1;
        *errmsg = copy_error_message("unknown C++ exception");
    }
    return output;
}

PYAPI void* py_run_residual_pca(const void* mat, const int32_t* block, uint8_t equal_weights, int32_t number, uint8_t use_subset, const uint8_t* subset, uint8_t scale, int32_t num_threads, int32_t* errcode, char** errmsg) {
    void* output = NULL;
    try {
        output = run_residual_pca(mat, block, equal_weights, number, use_subset, subset, scale, num_threads);
    } catch(std::exception& e) {
        *errcode = 1;
        *errmsg = copy_error_message(e.what());
    } catch(...) {
        *errcode = 1;
        *errmsg = copy_error_message("unknown C++ exception");
    }
    return output;
}

PYAPI void* py_run_simple_pca(const void* mat, int32_t number, uint8_t use_subset, const uint8_t* subset, uint8_t scale, int32_t num_threads, int32_t* errcode, char** errmsg) {
    void* output = NULL;
    try {
        output = run_simple_pca(mat, number, use_subset, subset, scale, num_threads);
    } catch(std::exception& e) {
        *errcode = 1;
        *errmsg = copy_error_message(e.what());
    } catch(...) {
        *errcode = 1;
        *errmsg = copy_error_message("unknown C++ exception");
    }
    return output;
}

PYAPI void py_run_tsne(void* status, int32_t maxiter, double* Y, int32_t* errcode, char** errmsg) {
    try {
        run_tsne(status, maxiter, Y);
    } catch(std::exception& e) {
        *errcode = 1;
        *errmsg = copy_error_message(e.what());
    } catch(...) {
        *errcode = 1;
        *errmsg = copy_error_message("unknown C++ exception");
    }
}

PYAPI void py_run_umap(void* status, int32_t max_epoch, int32_t* errcode, char** errmsg) {
    try {
        run_umap(status, max_epoch);
    } catch(std::exception& e) {
        *errcode = 1;
        *errmsg = copy_error_message(e.what());
    } catch(...) {
        *errcode = 1;
        *errmsg = copy_error_message("unknown C++ exception");
    }
}

PYAPI void py_score_markers(const void* mat, int32_t num_clusters, const int32_t* clusters, int32_t num_blocks, const int32_t* block, uint8_t do_auc, double threshold, uintptr_t* raw_means, uintptr_t* raw_detected, uintptr_t* raw_cohen, uintptr_t* raw_auc, uintptr_t* raw_lfc, uintptr_t* raw_delta_detected, int32_t num_threads, int32_t* errcode, char** errmsg) {
    try {
        score_markers(mat, num_clusters, clusters, num_blocks, block, do_auc, threshold, raw_means, raw_detected, raw_cohen, raw_auc, raw_lfc, raw_delta_detected, num_threads);
    } catch(std::exception& e) {
        *errcode = 1;
        *errmsg = copy_error_message(e.what());
    } catch(...) {
        *errcode = 1;
        *errmsg = copy_error_message("unknown C++ exception");
    }
}

PYAPI void py_serialize_neighbor_results(const void* ptr0, int32_t* outdex, double* outdist, int32_t* errcode, char** errmsg) {
    try {
        serialize_neighbor_results(ptr0, outdex, outdist);
    } catch(std::exception& e) {
        *errcode = 1;
        *errmsg = copy_error_message(e.what());
    } catch(...) {
        *errcode = 1;
        *errmsg = copy_error_message("unknown C++ exception");
    }
}

PYAPI void py_suggest_adt_qc_filters(int32_t num_cells, int32_t num_subsets, int32_t* detected, uintptr_t* subset_proportions, int32_t num_blocks, const int32_t* block, double* detected_out, uintptr_t* subset_proportions_out, double nmads, int32_t* errcode, char** errmsg) {
    try {
        suggest_adt_qc_filters(num_cells, num_subsets, detected, subset_proportions, num_blocks, block, detected_out, subset_proportions_out, nmads);
    } catch(std::exception& e) {
        *errcode = 1;
        *errmsg = copy_error_message(e.what());
    } catch(...) {
        *errcode = 1;
        *errmsg = copy_error_message("unknown C++ exception");
    }
}

PYAPI void py_suggest_rna_qc_filters(int32_t num_cells, int32_t num_subsets, double* sums, int32_t* detected, uintptr_t* subset_proportions, int32_t num_blocks, const int32_t* block, double* sums_out, double* detected_out, uintptr_t* subset_proportions_out, double nmads, int32_t* errcode, char** errmsg) {
    try {
        suggest_rna_qc_filters(num_cells, num_subsets, sums, detected, subset_proportions, num_blocks, block, sums_out, detected_out, subset_proportions_out, nmads);
    } catch(std::exception& e) {
        *errcode = 1;
        *errmsg = copy_error_message(e.what());
    } catch(...) {
        *errcode = 1;
        *errmsg = copy_error_message("unknown C++ exception");
    }
}

PYAPI void* py_unserialize_neighbor_results(int32_t nobs, int32_t k, int32_t* indices, double* distances, int32_t* errcode, char** errmsg) {
    void* output = NULL;
    try {
        output = unserialize_neighbor_results(nobs, k, indices, distances);
    } catch(std::exception& e) {
        *errcode = 1;
        *errmsg = copy_error_message(e.what());
    } catch(...) {
        *errcode = 1;
        *errmsg = copy_error_message("unknown C++ exception");
    }
    return output;
}

}
