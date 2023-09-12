# DO NOT MODIFY: this is automatically generated by the cpptypes

import os
import ctypes as ct
import numpy as np

def _catch_errors(f):
    def wrapper(*args):
        errcode = ct.c_int32(0)
        errmsg = ct.c_char_p(0)
        output = f(*args, ct.byref(errcode), ct.byref(errmsg))
        if errcode.value != 0:
            msg = errmsg.value.decode('ascii')
            lib.free_error_message(errmsg)
            raise RuntimeError(msg)
        return output
    return wrapper

# TODO: surely there's a better way than whatever this is.
dirname = os.path.dirname(os.path.abspath(__file__))
contents = os.listdir(dirname)
lib = None
for x in contents:
    if x.startswith('core') and not x.endswith("py"):
        lib = ct.CDLL(os.path.join(dirname, x))
        break

if lib is None:
    raise ImportError("failed to find the core.* module")

lib.free_error_message.argtypes = [ ct.POINTER(ct.c_char_p) ]

def _np2ct(x, expected, contiguous=True):
    if not isinstance(x, np.ndarray):
        raise ValueError('expected a NumPy array')
    if x.dtype != expected:
        raise ValueError('expected a NumPy array of type ' + str(expected) + ', got ' + str(x.dtype))
    if contiguous:
        if not x.flags.c_contiguous and not x.flags.f_contiguous:
            raise ValueError('only contiguous NumPy arrays are supported')
    return x.ctypes.data

lib.py_aggregate_across_cells.restype = None
lib.py_aggregate_across_cells.argtypes = [
    ct.c_void_p,
    ct.c_void_p,
    ct.c_int32,
    ct.c_uint8,
    ct.c_void_p,
    ct.c_uint8,
    ct.c_void_p,
    ct.c_int32,
    ct.POINTER(ct.c_int32),
    ct.POINTER(ct.c_char_p)
]

lib.py_build_neighbor_index.restype = ct.c_void_p
lib.py_build_neighbor_index.argtypes = [
    ct.c_int32,
    ct.c_int32,
    ct.c_void_p,
    ct.c_uint8,
    ct.POINTER(ct.c_int32),
    ct.POINTER(ct.c_char_p)
]

lib.py_build_snn_graph_from_nn_index.restype = ct.c_void_p
lib.py_build_snn_graph_from_nn_index.argtypes = [
    ct.c_void_p,
    ct.c_int32,
    ct.c_char_p,
    ct.c_int32,
    ct.POINTER(ct.c_int32),
    ct.POINTER(ct.c_char_p)
]

lib.py_build_snn_graph_from_nn_results.restype = ct.c_void_p
lib.py_build_snn_graph_from_nn_results.argtypes = [
    ct.c_void_p,
    ct.c_char_p,
    ct.c_int32,
    ct.POINTER(ct.c_int32),
    ct.POINTER(ct.c_char_p)
]

lib.py_center_size_factors.restype = None
lib.py_center_size_factors.argtypes = [
    ct.c_int32,
    ct.c_void_p,
    ct.c_uint8,
    ct.c_uint8,
    ct.c_uint8,
    ct.c_void_p,
    ct.POINTER(ct.c_int32),
    ct.POINTER(ct.c_char_p)
]

lib.py_choose_hvgs.restype = None
lib.py_choose_hvgs.argtypes = [
    ct.c_int32,
    ct.c_void_p,
    ct.c_int32,
    ct.c_void_p,
    ct.POINTER(ct.c_int32),
    ct.POINTER(ct.c_char_p)
]

lib.py_clone_tsne_status.restype = ct.c_void_p
lib.py_clone_tsne_status.argtypes = [
    ct.c_void_p,
    ct.POINTER(ct.c_int32),
    ct.POINTER(ct.c_char_p)
]

lib.py_clone_umap_status.restype = ct.c_void_p
lib.py_clone_umap_status.argtypes = [
    ct.c_void_p,
    ct.c_void_p,
    ct.POINTER(ct.c_int32),
    ct.POINTER(ct.c_char_p)
]

lib.py_combine_factors.restype = ct.c_void_p
lib.py_combine_factors.argtypes = [
    ct.c_int32,
    ct.c_int32,
    ct.c_void_p,
    ct.c_void_p,
    ct.POINTER(ct.c_int32),
    ct.POINTER(ct.c_char_p)
]

lib.py_create_adt_qc_filter.restype = None
lib.py_create_adt_qc_filter.argtypes = [
    ct.c_int,
    ct.c_int,
    ct.c_void_p,
    ct.c_void_p,
    ct.c_int,
    ct.c_void_p,
    ct.c_void_p,
    ct.c_void_p,
    ct.c_void_p,
    ct.POINTER(ct.c_int32),
    ct.POINTER(ct.c_char_p)
]

lib.py_create_rna_qc_filter.restype = None
lib.py_create_rna_qc_filter.argtypes = [
    ct.c_int,
    ct.c_int,
    ct.c_void_p,
    ct.c_void_p,
    ct.c_void_p,
    ct.c_int,
    ct.c_void_p,
    ct.c_void_p,
    ct.c_void_p,
    ct.c_void_p,
    ct.c_void_p,
    ct.POINTER(ct.c_int32),
    ct.POINTER(ct.c_char_p)
]

lib.py_downsample_by_neighbors.restype = None
lib.py_downsample_by_neighbors.argtypes = [
    ct.c_void_p,
    ct.c_void_p,
    ct.c_int32,
    ct.POINTER(ct.c_int32),
    ct.POINTER(ct.c_char_p)
]

lib.py_fetch_multibatch_pca_coordinates.restype = ct.POINTER(ct.c_double)
lib.py_fetch_multibatch_pca_coordinates.argtypes = [
    ct.c_void_p,
    ct.POINTER(ct.c_int32),
    ct.POINTER(ct.c_char_p)
]

lib.py_fetch_multibatch_pca_num_dims.restype = ct.c_int32
lib.py_fetch_multibatch_pca_num_dims.argtypes = [
    ct.c_void_p,
    ct.POINTER(ct.c_int32),
    ct.POINTER(ct.c_char_p)
]

lib.py_fetch_multibatch_pca_total_variance.restype = ct.c_double
lib.py_fetch_multibatch_pca_total_variance.argtypes = [
    ct.c_void_p,
    ct.POINTER(ct.c_int32),
    ct.POINTER(ct.c_char_p)
]

lib.py_fetch_multibatch_pca_variance_explained.restype = ct.POINTER(ct.c_double)
lib.py_fetch_multibatch_pca_variance_explained.argtypes = [
    ct.c_void_p,
    ct.POINTER(ct.c_int32),
    ct.POINTER(ct.c_char_p)
]

lib.py_fetch_neighbor_index_ndim.restype = ct.c_int32
lib.py_fetch_neighbor_index_ndim.argtypes = [
    ct.c_void_p,
    ct.POINTER(ct.c_int32),
    ct.POINTER(ct.c_char_p)
]

lib.py_fetch_neighbor_index_nobs.restype = ct.c_int32
lib.py_fetch_neighbor_index_nobs.argtypes = [
    ct.c_void_p,
    ct.POINTER(ct.c_int32),
    ct.POINTER(ct.c_char_p)
]

lib.py_fetch_neighbor_results_k.restype = ct.c_int32
lib.py_fetch_neighbor_results_k.argtypes = [
    ct.c_void_p,
    ct.POINTER(ct.c_int32),
    ct.POINTER(ct.c_char_p)
]

lib.py_fetch_neighbor_results_nobs.restype = ct.c_int32
lib.py_fetch_neighbor_results_nobs.argtypes = [
    ct.c_void_p,
    ct.POINTER(ct.c_int32),
    ct.POINTER(ct.c_char_p)
]

lib.py_fetch_neighbor_results_single.restype = None
lib.py_fetch_neighbor_results_single.argtypes = [
    ct.c_void_p,
    ct.c_int32,
    ct.c_void_p,
    ct.c_void_p,
    ct.POINTER(ct.c_int32),
    ct.POINTER(ct.c_char_p)
]

lib.py_fetch_residual_pca_coordinates.restype = ct.POINTER(ct.c_double)
lib.py_fetch_residual_pca_coordinates.argtypes = [
    ct.c_void_p,
    ct.POINTER(ct.c_int32),
    ct.POINTER(ct.c_char_p)
]

lib.py_fetch_residual_pca_num_dims.restype = ct.c_int32
lib.py_fetch_residual_pca_num_dims.argtypes = [
    ct.c_void_p,
    ct.POINTER(ct.c_int32),
    ct.POINTER(ct.c_char_p)
]

lib.py_fetch_residual_pca_total_variance.restype = ct.c_double
lib.py_fetch_residual_pca_total_variance.argtypes = [
    ct.c_void_p,
    ct.POINTER(ct.c_int32),
    ct.POINTER(ct.c_char_p)
]

lib.py_fetch_residual_pca_variance_explained.restype = ct.POINTER(ct.c_double)
lib.py_fetch_residual_pca_variance_explained.argtypes = [
    ct.c_void_p,
    ct.POINTER(ct.c_int32),
    ct.POINTER(ct.c_char_p)
]

lib.py_fetch_simple_pca_coordinates.restype = ct.POINTER(ct.c_double)
lib.py_fetch_simple_pca_coordinates.argtypes = [
    ct.c_void_p,
    ct.POINTER(ct.c_int32),
    ct.POINTER(ct.c_char_p)
]

lib.py_fetch_simple_pca_num_dims.restype = ct.c_int32
lib.py_fetch_simple_pca_num_dims.argtypes = [
    ct.c_void_p,
    ct.POINTER(ct.c_int32),
    ct.POINTER(ct.c_char_p)
]

lib.py_fetch_simple_pca_total_variance.restype = ct.c_double
lib.py_fetch_simple_pca_total_variance.argtypes = [
    ct.c_void_p,
    ct.POINTER(ct.c_int32),
    ct.POINTER(ct.c_char_p)
]

lib.py_fetch_simple_pca_variance_explained.restype = ct.POINTER(ct.c_double)
lib.py_fetch_simple_pca_variance_explained.argtypes = [
    ct.c_void_p,
    ct.POINTER(ct.c_int32),
    ct.POINTER(ct.c_char_p)
]

lib.py_fetch_snn_graph_edges.restype = ct.c_int32
lib.py_fetch_snn_graph_edges.argtypes = [
    ct.c_void_p,
    ct.POINTER(ct.c_int32),
    ct.POINTER(ct.c_char_p)
]

lib.py_fetch_snn_graph_indices.restype = ct.POINTER(ct.c_int)
lib.py_fetch_snn_graph_indices.argtypes = [
    ct.c_void_p,
    ct.POINTER(ct.c_int32),
    ct.POINTER(ct.c_char_p)
]

lib.py_fetch_snn_graph_weights.restype = ct.POINTER(ct.c_double)
lib.py_fetch_snn_graph_weights.argtypes = [
    ct.c_void_p,
    ct.POINTER(ct.c_int32),
    ct.POINTER(ct.c_char_p)
]

lib.py_fetch_tsne_status_iteration.restype = ct.c_int32
lib.py_fetch_tsne_status_iteration.argtypes = [
    ct.c_void_p,
    ct.POINTER(ct.c_int32),
    ct.POINTER(ct.c_char_p)
]

lib.py_fetch_tsne_status_nobs.restype = ct.c_int32
lib.py_fetch_tsne_status_nobs.argtypes = [
    ct.c_void_p,
    ct.POINTER(ct.c_int32),
    ct.POINTER(ct.c_char_p)
]

lib.py_fetch_umap_status_epoch.restype = ct.c_int32
lib.py_fetch_umap_status_epoch.argtypes = [
    ct.c_void_p,
    ct.POINTER(ct.c_int32),
    ct.POINTER(ct.c_char_p)
]

lib.py_fetch_umap_status_nobs.restype = ct.c_int32
lib.py_fetch_umap_status_nobs.argtypes = [
    ct.c_void_p,
    ct.POINTER(ct.c_int32),
    ct.POINTER(ct.c_char_p)
]

lib.py_fetch_umap_status_num_epochs.restype = ct.c_int32
lib.py_fetch_umap_status_num_epochs.argtypes = [
    ct.c_void_p,
    ct.POINTER(ct.c_int32),
    ct.POINTER(ct.c_char_p)
]

lib.py_filter_cells.restype = ct.c_void_p
lib.py_filter_cells.argtypes = [
    ct.c_void_p,
    ct.c_void_p,
    ct.c_uint8,
    ct.POINTER(ct.c_int32),
    ct.POINTER(ct.c_char_p)
]

lib.py_find_nearest_neighbors.restype = ct.c_void_p
lib.py_find_nearest_neighbors.argtypes = [
    ct.c_void_p,
    ct.c_int32,
    ct.c_int32,
    ct.POINTER(ct.c_int32),
    ct.POINTER(ct.c_char_p)
]

lib.py_free_combined_factors.restype = None
lib.py_free_combined_factors.argtypes = [
    ct.c_void_p,
    ct.POINTER(ct.c_int32),
    ct.POINTER(ct.c_char_p)
]

lib.py_free_multibatch_pca.restype = None
lib.py_free_multibatch_pca.argtypes = [
    ct.c_void_p,
    ct.POINTER(ct.c_int32),
    ct.POINTER(ct.c_char_p)
]

lib.py_free_neighbor_index.restype = None
lib.py_free_neighbor_index.argtypes = [
    ct.c_void_p,
    ct.POINTER(ct.c_int32),
    ct.POINTER(ct.c_char_p)
]

lib.py_free_neighbor_results.restype = None
lib.py_free_neighbor_results.argtypes = [
    ct.c_void_p,
    ct.POINTER(ct.c_int32),
    ct.POINTER(ct.c_char_p)
]

lib.py_free_residual_pca.restype = None
lib.py_free_residual_pca.argtypes = [
    ct.c_void_p,
    ct.POINTER(ct.c_int32),
    ct.POINTER(ct.c_char_p)
]

lib.py_free_simple_pca.restype = None
lib.py_free_simple_pca.argtypes = [
    ct.c_void_p,
    ct.POINTER(ct.c_int32),
    ct.POINTER(ct.c_char_p)
]

lib.py_free_snn_graph.restype = None
lib.py_free_snn_graph.argtypes = [
    ct.c_void_p,
    ct.POINTER(ct.c_int32),
    ct.POINTER(ct.c_char_p)
]

lib.py_free_tsne_status.restype = None
lib.py_free_tsne_status.argtypes = [
    ct.c_void_p,
    ct.POINTER(ct.c_int32),
    ct.POINTER(ct.c_char_p)
]

lib.py_free_umap_status.restype = None
lib.py_free_umap_status.argtypes = [
    ct.c_void_p,
    ct.POINTER(ct.c_int32),
    ct.POINTER(ct.c_char_p)
]

lib.py_get_combined_factors_count.restype = None
lib.py_get_combined_factors_count.argtypes = [
    ct.c_void_p,
    ct.c_void_p,
    ct.POINTER(ct.c_int32),
    ct.POINTER(ct.c_char_p)
]

lib.py_get_combined_factors_level.restype = None
lib.py_get_combined_factors_level.argtypes = [
    ct.c_void_p,
    ct.c_int32,
    ct.c_void_p,
    ct.POINTER(ct.c_int32),
    ct.POINTER(ct.c_char_p)
]

lib.py_get_combined_factors_size.restype = ct.c_int32
lib.py_get_combined_factors_size.argtypes = [
    ct.c_void_p,
    ct.POINTER(ct.c_int32),
    ct.POINTER(ct.c_char_p)
]

lib.py_initialize_tsne.restype = ct.c_void_p
lib.py_initialize_tsne.argtypes = [
    ct.c_void_p,
    ct.c_double,
    ct.c_int32,
    ct.POINTER(ct.c_int32),
    ct.POINTER(ct.c_char_p)
]

lib.py_initialize_umap.restype = ct.c_void_p
lib.py_initialize_umap.argtypes = [
    ct.c_void_p,
    ct.c_int32,
    ct.c_double,
    ct.c_void_p,
    ct.c_int32,
    ct.POINTER(ct.c_int32),
    ct.POINTER(ct.c_char_p)
]

lib.py_log_norm_counts.restype = ct.c_void_p
lib.py_log_norm_counts.argtypes = [
    ct.c_void_p,
    ct.c_void_p,
    ct.POINTER(ct.c_int32),
    ct.POINTER(ct.c_char_p)
]

lib.py_mnn_correct.restype = None
lib.py_mnn_correct.argtypes = [
    ct.c_int32,
    ct.c_int32,
    ct.c_void_p,
    ct.c_int32,
    ct.c_void_p,
    ct.c_int32,
    ct.c_double,
    ct.c_int32,
    ct.c_int32,
    ct.c_uint8,
    ct.c_void_p,
    ct.c_char_p,
    ct.c_uint8,
    ct.c_void_p,
    ct.c_void_p,
    ct.c_void_p,
    ct.POINTER(ct.c_int32),
    ct.POINTER(ct.c_char_p)
]

lib.py_model_gene_variances.restype = None
lib.py_model_gene_variances.argtypes = [
    ct.c_void_p,
    ct.c_void_p,
    ct.c_void_p,
    ct.c_void_p,
    ct.c_void_p,
    ct.c_double,
    ct.c_int32,
    ct.POINTER(ct.c_int32),
    ct.POINTER(ct.c_char_p)
]

lib.py_model_gene_variances_blocked.restype = None
lib.py_model_gene_variances_blocked.argtypes = [
    ct.c_void_p,
    ct.c_void_p,
    ct.c_void_p,
    ct.c_void_p,
    ct.c_void_p,
    ct.c_int32,
    ct.c_void_p,
    ct.c_void_p,
    ct.c_void_p,
    ct.c_void_p,
    ct.c_void_p,
    ct.c_double,
    ct.c_int32,
    ct.POINTER(ct.c_int32),
    ct.POINTER(ct.c_char_p)
]

lib.py_per_cell_adt_qc_metrics.restype = None
lib.py_per_cell_adt_qc_metrics.argtypes = [
    ct.c_void_p,
    ct.c_int32,
    ct.c_void_p,
    ct.c_void_p,
    ct.c_void_p,
    ct.c_void_p,
    ct.c_int32,
    ct.POINTER(ct.c_int32),
    ct.POINTER(ct.c_char_p)
]

lib.py_per_cell_rna_qc_metrics.restype = None
lib.py_per_cell_rna_qc_metrics.argtypes = [
    ct.c_void_p,
    ct.c_int32,
    ct.c_void_p,
    ct.c_void_p,
    ct.c_void_p,
    ct.c_void_p,
    ct.c_int32,
    ct.POINTER(ct.c_int32),
    ct.POINTER(ct.c_char_p)
]

lib.py_perplexity_to_k.restype = ct.c_int32
lib.py_perplexity_to_k.argtypes = [
    ct.c_double,
    ct.POINTER(ct.c_int32),
    ct.POINTER(ct.c_char_p)
]

lib.py_randomize_tsne_start.restype = None
lib.py_randomize_tsne_start.argtypes = [
    ct.c_size_t,
    ct.c_void_p,
    ct.c_int32,
    ct.POINTER(ct.c_int32),
    ct.POINTER(ct.c_char_p)
]

lib.py_run_multibatch_pca.restype = ct.c_void_p
lib.py_run_multibatch_pca.argtypes = [
    ct.c_void_p,
    ct.c_void_p,
    ct.c_uint8,
    ct.c_uint8,
    ct.c_int32,
    ct.c_uint8,
    ct.c_void_p,
    ct.c_uint8,
    ct.c_int32,
    ct.POINTER(ct.c_int32),
    ct.POINTER(ct.c_char_p)
]

lib.py_run_residual_pca.restype = ct.c_void_p
lib.py_run_residual_pca.argtypes = [
    ct.c_void_p,
    ct.c_void_p,
    ct.c_uint8,
    ct.c_int32,
    ct.c_uint8,
    ct.c_void_p,
    ct.c_uint8,
    ct.c_int32,
    ct.POINTER(ct.c_int32),
    ct.POINTER(ct.c_char_p)
]

lib.py_run_simple_pca.restype = ct.c_void_p
lib.py_run_simple_pca.argtypes = [
    ct.c_void_p,
    ct.c_int32,
    ct.c_uint8,
    ct.c_void_p,
    ct.c_uint8,
    ct.c_int32,
    ct.POINTER(ct.c_int32),
    ct.POINTER(ct.c_char_p)
]

lib.py_run_tsne.restype = None
lib.py_run_tsne.argtypes = [
    ct.c_void_p,
    ct.c_int32,
    ct.c_void_p,
    ct.POINTER(ct.c_int32),
    ct.POINTER(ct.c_char_p)
]

lib.py_run_umap.restype = None
lib.py_run_umap.argtypes = [
    ct.c_void_p,
    ct.c_int32,
    ct.POINTER(ct.c_int32),
    ct.POINTER(ct.c_char_p)
]

lib.py_score_markers.restype = None
lib.py_score_markers.argtypes = [
    ct.c_void_p,
    ct.c_int32,
    ct.c_void_p,
    ct.c_int32,
    ct.c_void_p,
    ct.c_uint8,
    ct.c_double,
    ct.c_void_p,
    ct.c_void_p,
    ct.c_void_p,
    ct.c_void_p,
    ct.c_void_p,
    ct.c_void_p,
    ct.c_int32,
    ct.POINTER(ct.c_int32),
    ct.POINTER(ct.c_char_p)
]

lib.py_serialize_neighbor_results.restype = None
lib.py_serialize_neighbor_results.argtypes = [
    ct.c_void_p,
    ct.c_void_p,
    ct.c_void_p,
    ct.POINTER(ct.c_int32),
    ct.POINTER(ct.c_char_p)
]

lib.py_suggest_adt_qc_filters.restype = None
lib.py_suggest_adt_qc_filters.argtypes = [
    ct.c_int32,
    ct.c_int32,
    ct.c_void_p,
    ct.c_void_p,
    ct.c_int32,
    ct.c_void_p,
    ct.c_void_p,
    ct.c_void_p,
    ct.c_double,
    ct.POINTER(ct.c_int32),
    ct.POINTER(ct.c_char_p)
]

lib.py_suggest_rna_qc_filters.restype = None
lib.py_suggest_rna_qc_filters.argtypes = [
    ct.c_int32,
    ct.c_int32,
    ct.c_void_p,
    ct.c_void_p,
    ct.c_void_p,
    ct.c_int32,
    ct.c_void_p,
    ct.c_void_p,
    ct.c_void_p,
    ct.c_void_p,
    ct.c_double,
    ct.POINTER(ct.c_int32),
    ct.POINTER(ct.c_char_p)
]

lib.py_unserialize_neighbor_results.restype = ct.c_void_p
lib.py_unserialize_neighbor_results.argtypes = [
    ct.c_int32,
    ct.c_int32,
    ct.c_void_p,
    ct.c_void_p,
    ct.POINTER(ct.c_int32),
    ct.POINTER(ct.c_char_p)
]

def aggregate_across_cells(mat, groups, ngroups, do_sums, output_sums, do_detected, output_detected, nthreads):
    return _catch_errors(lib.py_aggregate_across_cells)(mat, _np2ct(groups, np.int32), ngroups, do_sums, output_sums, do_detected, output_detected, nthreads)

def build_neighbor_index(ndim, nobs, ptr, approximate):
    return _catch_errors(lib.py_build_neighbor_index)(ndim, nobs, _np2ct(ptr, np.float64), approximate)

def build_snn_graph_from_nn_index(x, num_neighbors, weight_scheme, num_threads):
    return _catch_errors(lib.py_build_snn_graph_from_nn_index)(x, num_neighbors, weight_scheme, num_threads)

def build_snn_graph_from_nn_results(x, weight_scheme, num_threads):
    return _catch_errors(lib.py_build_snn_graph_from_nn_results)(x, weight_scheme, num_threads)

def center_size_factors(num, size_factors, allow_zeros, allow_non_finite, use_block, block):
    return _catch_errors(lib.py_center_size_factors)(num, _np2ct(size_factors, np.float64), allow_zeros, allow_non_finite, use_block, block)

def choose_hvgs(len, stat, top, output):
    return _catch_errors(lib.py_choose_hvgs)(len, _np2ct(stat, np.float64), top, _np2ct(output, np.uint8))

def clone_tsne_status(ptr):
    return _catch_errors(lib.py_clone_tsne_status)(ptr)

def clone_umap_status(ptr, cloned):
    return _catch_errors(lib.py_clone_umap_status)(ptr, _np2ct(cloned, np.float64))

def combine_factors(length, number, inputs, output_combined):
    return _catch_errors(lib.py_combine_factors)(length, number, inputs, _np2ct(output_combined, np.int32))

def create_adt_qc_filter(num_cells, num_subsets, detected, subset_proportions, num_blocks, block, detected_thresholds, subset_proportions_thresholds, output):
    return _catch_errors(lib.py_create_adt_qc_filter)(num_cells, num_subsets, _np2ct(detected, np.int32), subset_proportions, num_blocks, block, _np2ct(detected_thresholds, np.float64), subset_proportions_thresholds, _np2ct(output, np.uint8))

def create_rna_qc_filter(num_cells, num_subsets, sums, detected, subset_proportions, num_blocks, block, sums_thresholds, detected_thresholds, subset_proportions_thresholds, output):
    return _catch_errors(lib.py_create_rna_qc_filter)(num_cells, num_subsets, _np2ct(sums, np.float64), _np2ct(detected, np.int32), subset_proportions, num_blocks, block, _np2ct(sums_thresholds, np.float64), _np2ct(detected_thresholds, np.float64), subset_proportions_thresholds, _np2ct(output, np.uint8))

def downsample_by_neighbors(ptr, output, num_threads):
    return _catch_errors(lib.py_downsample_by_neighbors)(ptr, _np2ct(output, np.int32), num_threads)

def fetch_multibatch_pca_coordinates(x):
    return _catch_errors(lib.py_fetch_multibatch_pca_coordinates)(x)

def fetch_multibatch_pca_num_dims(x):
    return _catch_errors(lib.py_fetch_multibatch_pca_num_dims)(x)

def fetch_multibatch_pca_total_variance(x):
    return _catch_errors(lib.py_fetch_multibatch_pca_total_variance)(x)

def fetch_multibatch_pca_variance_explained(x):
    return _catch_errors(lib.py_fetch_multibatch_pca_variance_explained)(x)

def fetch_neighbor_index_ndim(ptr):
    return _catch_errors(lib.py_fetch_neighbor_index_ndim)(ptr)

def fetch_neighbor_index_nobs(ptr):
    return _catch_errors(lib.py_fetch_neighbor_index_nobs)(ptr)

def fetch_neighbor_results_k(ptr0):
    return _catch_errors(lib.py_fetch_neighbor_results_k)(ptr0)

def fetch_neighbor_results_nobs(ptr):
    return _catch_errors(lib.py_fetch_neighbor_results_nobs)(ptr)

def fetch_neighbor_results_single(ptr0, i, outdex, outdist):
    return _catch_errors(lib.py_fetch_neighbor_results_single)(ptr0, i, _np2ct(outdex, np.int32), _np2ct(outdist, np.float64))

def fetch_residual_pca_coordinates(x):
    return _catch_errors(lib.py_fetch_residual_pca_coordinates)(x)

def fetch_residual_pca_num_dims(x):
    return _catch_errors(lib.py_fetch_residual_pca_num_dims)(x)

def fetch_residual_pca_total_variance(x):
    return _catch_errors(lib.py_fetch_residual_pca_total_variance)(x)

def fetch_residual_pca_variance_explained(x):
    return _catch_errors(lib.py_fetch_residual_pca_variance_explained)(x)

def fetch_simple_pca_coordinates(x):
    return _catch_errors(lib.py_fetch_simple_pca_coordinates)(x)

def fetch_simple_pca_num_dims(x):
    return _catch_errors(lib.py_fetch_simple_pca_num_dims)(x)

def fetch_simple_pca_total_variance(x):
    return _catch_errors(lib.py_fetch_simple_pca_total_variance)(x)

def fetch_simple_pca_variance_explained(x):
    return _catch_errors(lib.py_fetch_simple_pca_variance_explained)(x)

def fetch_snn_graph_edges(ptr):
    return _catch_errors(lib.py_fetch_snn_graph_edges)(ptr)

def fetch_snn_graph_indices(ptr):
    return _catch_errors(lib.py_fetch_snn_graph_indices)(ptr)

def fetch_snn_graph_weights(ptr):
    return _catch_errors(lib.py_fetch_snn_graph_weights)(ptr)

def fetch_tsne_status_iteration(ptr):
    return _catch_errors(lib.py_fetch_tsne_status_iteration)(ptr)

def fetch_tsne_status_nobs(ptr):
    return _catch_errors(lib.py_fetch_tsne_status_nobs)(ptr)

def fetch_umap_status_epoch(ptr):
    return _catch_errors(lib.py_fetch_umap_status_epoch)(ptr)

def fetch_umap_status_nobs(ptr):
    return _catch_errors(lib.py_fetch_umap_status_nobs)(ptr)

def fetch_umap_status_num_epochs(ptr):
    return _catch_errors(lib.py_fetch_umap_status_num_epochs)(ptr)

def filter_cells(mat0, filter, discard):
    return _catch_errors(lib.py_filter_cells)(mat0, _np2ct(filter, np.uint8), discard)

def find_nearest_neighbors(index, k, nthreads):
    return _catch_errors(lib.py_find_nearest_neighbors)(index, k, nthreads)

def free_combined_factors(ptr):
    return _catch_errors(lib.py_free_combined_factors)(ptr)

def free_multibatch_pca(x):
    return _catch_errors(lib.py_free_multibatch_pca)(x)

def free_neighbor_index(ptr):
    return _catch_errors(lib.py_free_neighbor_index)(ptr)

def free_neighbor_results(ptr):
    return _catch_errors(lib.py_free_neighbor_results)(ptr)

def free_residual_pca(x):
    return _catch_errors(lib.py_free_residual_pca)(x)

def free_simple_pca(x):
    return _catch_errors(lib.py_free_simple_pca)(x)

def free_snn_graph(ptr):
    return _catch_errors(lib.py_free_snn_graph)(ptr)

def free_tsne_status(ptr):
    return _catch_errors(lib.py_free_tsne_status)(ptr)

def free_umap_status(ptr):
    return _catch_errors(lib.py_free_umap_status)(ptr)

def get_combined_factors_count(ptr, output):
    return _catch_errors(lib.py_get_combined_factors_count)(ptr, _np2ct(output, np.int32))

def get_combined_factors_level(ptr, i, output):
    return _catch_errors(lib.py_get_combined_factors_level)(ptr, i, _np2ct(output, np.int32))

def get_combined_factors_size(ptr):
    return _catch_errors(lib.py_get_combined_factors_size)(ptr)

def initialize_tsne(neighbors, perplexity, nthreads):
    return _catch_errors(lib.py_initialize_tsne)(neighbors, perplexity, nthreads)

def initialize_umap(neighbors, num_epochs, min_dist, Y, nthreads):
    return _catch_errors(lib.py_initialize_umap)(neighbors, num_epochs, min_dist, _np2ct(Y, np.float64), nthreads)

def log_norm_counts(mat0, size_factors):
    return _catch_errors(lib.py_log_norm_counts)(mat0, _np2ct(size_factors, np.float64))

def mnn_correct(ndim, ncells, x, nbatches, batch, k, nmads, nthreads, mass_cap, use_order, order, ref_policy, approximate, corrected_output, merge_order_output, num_pairs_output):
    return _catch_errors(lib.py_mnn_correct)(ndim, ncells, _np2ct(x, np.float64), nbatches, _np2ct(batch, np.int32), k, nmads, nthreads, mass_cap, use_order, order, ref_policy, approximate, _np2ct(corrected_output, np.float64), _np2ct(merge_order_output, np.int32), _np2ct(num_pairs_output, np.int32))

def model_gene_variances(mat, means, variances, fitted, residuals, span, num_threads):
    return _catch_errors(lib.py_model_gene_variances)(mat, _np2ct(means, np.float64), _np2ct(variances, np.float64), _np2ct(fitted, np.float64), _np2ct(residuals, np.float64), span, num_threads)

def model_gene_variances_blocked(mat, ave_means, ave_detected, ave_fitted, ave_residuals, num_blocks, block, block_means, block_variances, block_fitted, block_residuals, span, num_threads):
    return _catch_errors(lib.py_model_gene_variances_blocked)(mat, _np2ct(ave_means, np.float64), _np2ct(ave_detected, np.float64), _np2ct(ave_fitted, np.float64), _np2ct(ave_residuals, np.float64), num_blocks, block, block_means, block_variances, block_fitted, block_residuals, span, num_threads)

def per_cell_adt_qc_metrics(mat, num_subsets, subset_ptrs, sum_output, detected_output, subset_output, num_threads):
    return _catch_errors(lib.py_per_cell_adt_qc_metrics)(mat, num_subsets, subset_ptrs, _np2ct(sum_output, np.float64), _np2ct(detected_output, np.int32), subset_output, num_threads)

def per_cell_rna_qc_metrics(mat, num_subsets, subset_ptrs, sum_output, detected_output, subset_output, num_threads):
    return _catch_errors(lib.py_per_cell_rna_qc_metrics)(mat, num_subsets, subset_ptrs, _np2ct(sum_output, np.float64), _np2ct(detected_output, np.int32), subset_output, num_threads)

def perplexity_to_k(perplexity):
    return _catch_errors(lib.py_perplexity_to_k)(perplexity)

def randomize_tsne_start(n, Y, seed):
    return _catch_errors(lib.py_randomize_tsne_start)(n, _np2ct(Y, np.float64), seed)

def run_multibatch_pca(mat, block, use_residuals, equal_weights, number, use_subset, subset, scale, num_threads):
    return _catch_errors(lib.py_run_multibatch_pca)(mat, _np2ct(block, np.int32), use_residuals, equal_weights, number, use_subset, subset, scale, num_threads)

def run_residual_pca(mat, block, equal_weights, number, use_subset, subset, scale, num_threads):
    return _catch_errors(lib.py_run_residual_pca)(mat, _np2ct(block, np.int32), equal_weights, number, use_subset, subset, scale, num_threads)

def run_simple_pca(mat, number, use_subset, subset, scale, num_threads):
    return _catch_errors(lib.py_run_simple_pca)(mat, number, use_subset, subset, scale, num_threads)

def run_tsne(status, maxiter, Y):
    return _catch_errors(lib.py_run_tsne)(status, maxiter, _np2ct(Y, np.float64))

def run_umap(status, max_epoch):
    return _catch_errors(lib.py_run_umap)(status, max_epoch)

def score_markers(mat, num_clusters, clusters, num_blocks, block, do_auc, threshold, raw_means, raw_detected, raw_cohen, raw_auc, raw_lfc, raw_delta_detected, num_threads):
    return _catch_errors(lib.py_score_markers)(mat, num_clusters, _np2ct(clusters, np.int32), num_blocks, block, do_auc, threshold, raw_means, raw_detected, raw_cohen, raw_auc, raw_lfc, raw_delta_detected, num_threads)

def serialize_neighbor_results(ptr0, outdex, outdist):
    return _catch_errors(lib.py_serialize_neighbor_results)(ptr0, _np2ct(outdex, np.int32), _np2ct(outdist, np.float64))

def suggest_adt_qc_filters(num_cells, num_subsets, detected, subset_proportions, num_blocks, block, detected_out, subset_proportions_out, nmads):
    return _catch_errors(lib.py_suggest_adt_qc_filters)(num_cells, num_subsets, _np2ct(detected, np.int32), subset_proportions, num_blocks, block, _np2ct(detected_out, np.float64), subset_proportions_out, nmads)

def suggest_rna_qc_filters(num_cells, num_subsets, sums, detected, subset_proportions, num_blocks, block, sums_out, detected_out, subset_proportions_out, nmads):
    return _catch_errors(lib.py_suggest_rna_qc_filters)(num_cells, num_subsets, _np2ct(sums, np.float64), _np2ct(detected, np.int32), subset_proportions, num_blocks, block, _np2ct(sums_out, np.float64), _np2ct(detected_out, np.float64), subset_proportions_out, nmads)

def unserialize_neighbor_results(nobs, k, indices, distances):
    return _catch_errors(lib.py_unserialize_neighbor_results)(nobs, k, _np2ct(indices, np.int32), _np2ct(distances, np.float64))
