#include <vector>
#include <stdexcept>
#include <memory>
#include <cstdint>

#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"
#include "kmeans/kmeans.hpp"

#include "utils.h"

pybind11::tuple cluster_kmeans(
    const pybind11::array& data,
    int num_clusters,
    std::string init_method,
    std::string refine_method,
    bool var_part_optimize_partition,
    double var_part_size_adjustment,
    int lloyd_iterations,
    int hartigan_wong_iterations,
    int hartigan_wong_quick_transfer_iterations,
    bool hartigan_wong_quit_quick_transfer_failure,
    int seed,
    int nthreads
) {
    auto dbuffer = data.request();
    if ((data.flags() & pybind11::array::f_style) == 0) {
        throw std::runtime_error("expected a column-major matrix for the coordinates");
    }
    if (!data.dtype().is(pybind11::dtype::of<double>())) {
        throw std::runtime_error("unexpected dtype for input matrix");
    }
    const auto ndims = dbuffer.shape[0];
    const auto nobs = dbuffer.shape[1];
    const double* dptr = get_numpy_array_data<double>(data);

    auto centers = create_numpy_matrix<double>(ndims, num_clusters);
    auto clusters = create_numpy_array<std::uint32_t>(nobs);
    auto center_ptr = static_cast<double*>(centers.request().ptr);
    auto cluster_ptr = static_cast<std::uint32_t*>(clusters.request().ptr);

    std::unique_ptr<kmeans::Initialize<std::uint32_t, double, std::uint32_t, double> > iptr;
    if (init_method == "random") {
        auto ptr = new kmeans::InitializeRandom<std::uint32_t, double, std::uint32_t, double>;
        ptr->get_options().seed = seed;
        iptr.reset(ptr);
    } else if (init_method == "kmeans++") {
        auto ptr = new kmeans::InitializeKmeanspp<std::uint32_t, double, std::uint32_t, double>;
        ptr->get_options().num_threads = nthreads;
        ptr->get_options().seed = seed;
        iptr.reset(ptr);;
    } else if (init_method == "var-part") {
        auto ptr = new kmeans::InitializeVariancePartition<std::uint32_t, double, std::uint32_t, double>;
        ptr->get_options().optimize_partition = var_part_optimize_partition;
        ptr->get_options().size_adjustment = var_part_size_adjustment;
        iptr.reset(ptr);
    } else {
        throw std::runtime_error("unknown init_method '" + init_method + "'");
    }

    std::unique_ptr<kmeans::Refine<std::uint32_t, double, std::uint32_t, double> > rptr;
    if (refine_method == "lloyd") {
        auto ptr = new kmeans::RefineLloyd<std::uint32_t, double, std::uint32_t, double>;
        ptr->get_options().max_iterations = lloyd_iterations;
        ptr->get_options().num_threads = nthreads;
        rptr.reset(ptr);
    } else if (refine_method == "hartigan-wong") {
        auto ptr = new kmeans::RefineHartiganWong<std::uint32_t, double, std::uint32_t, double>;
        ptr->get_options().max_iterations = hartigan_wong_iterations;
        ptr->get_options().max_quick_transfer_iterations = hartigan_wong_quick_transfer_iterations;
        ptr->get_options().quit_on_quick_transfer_convergence_failure = hartigan_wong_quit_quick_transfer_failure;
        ptr->get_options().num_threads = nthreads;
        rptr.reset(ptr);
    }

    auto out = kmeans::compute(
        kmeans::SimpleMatrix<std::uint32_t, double>(ndims, nobs, dptr),
        *iptr,
        *rptr,
        static_cast<std::uint32_t>(num_clusters),
        center_ptr,
        cluster_ptr
    );

    const auto actual_k = kmeans::remove_unused_centers(ndim, nobs, cluster_ptr, nclusters, center_ptr, out.sizes);
    if (actual_k != nclusters) {
        auto new_centers = create_numpy_matrix<double>(ndims, actual_k);
        std::copy_n(centers.begin(), new_centers.size(), new_centers.begin());
        centers = std::move(new_centers);
    }

    pybind11::tuple output(4); 
    output[0] = std::move(clusters);
    output[1] = std::move(centers);
    output[2] = out.iterations;
    output[3] = out.status;

    return output;
}

void init_cluster_kmeans(pybind11::module& m) {
    m.def("cluster_kmeans", &cluster_kmeans);
}
