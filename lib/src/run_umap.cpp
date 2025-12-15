#include <optional>
#include <cstdint>
#include <vector>
#include <stdexcept>
#include <algorithm>

#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"
#include "pybind11/stl.h"
#include "umappp/umappp.hpp"

#include "neighbors.h"

pybind11::array run_umap(
    const pybind11::array& nnidx,
    const pybind11::array& nndist, 
    int ndim,
    double local_connectivity,
    double bandwidth,
    double mix_ratio,
    double spread,
    double min_dist,
    std::optional<double> a,
    std::optional<double> b,
    double repulsion_strength,
    std::string initialize_method,
    std::optional<pybind11::array> initial_coordinates,
    bool initialize_random_on_spectral_fail,
    double initialize_spectral_scale,
    bool initialize_spectral_jitter,
    double initialize_spectral_jitter_sd,
    double initialize_random_scale,
    std::uint64_t initialize_seed,
    std::optional<int> num_epochs,
    double learning_rate,
    double negative_sample_rate,
    std::uint64_t optimize_seed,
    int num_threads,
    bool parallel_optimization
) {
    auto neighbors = unpack_neighbors<std::uint32_t, float>(nnidx, nndist);
    const auto nobs = neighbors.size();

    umappp::Options opt;
    opt.local_connectivity = local_connectivity;
    opt.bandwidth = bandwidth;
    opt.mix_ratio = mix_ratio;
    opt.spread = spread;
    opt.min_dist = min_dist;
    opt.a = a;
    opt.b = b;
    opt.repulsion_strength = repulsion_strength;

    if (initialize_method == "spectral") {
        opt.initialize_method = umappp::InitializeMethod::SPECTRAL;
    } else if (initialize_method == "random") {
        opt.initialize_method = umappp::InitializeMethod::RANDOM;
    } else if (initialize_method == "none") {
        opt.initialize_method = umappp::InitializeMethod::NONE;
    } else {
        throw std::runtime_error("unknown value for 'initialize_method'");
    }

    std::vector<float> embedding(sanisizer::product<typename std::vector<float>::size_type>(ndim, nobs));
    if (initial_coordinates.has_value()) {
        auto init_buffer = initial_coordinates.request();
        if ((init_buffer.flags() & pybind11::array::f_style) == 0) {
            throw std::runtime_error("expected a column-major matrix for the initial coordinates");
        }
        const auto& init_dtype = initial_coordinates.dtype(); // the usual is() doesn't work in a separate process.
        if (init_dtype.kind() != 'f' || init_dtype.itemsize() != 8) {
            throw std::runtime_error("unexpected dtype for array of initial coordinates");
        }
        auto iptr = get_numpy_array_data<double>(initial_coordinates);
        std::copy(iptr, embedding.size(), embedding.data());
    } else if (initialize_method == "none" || !initialize_random_on_spectral_fail) {
        throw std::runtime_error("expected initial coordinates to be supplied");
    }

    opt.initialize_random_on_spectral_fail = initialize_random_on_spectral_fail;
    opt.initialize_spectral_scale = initialize_spectral_scale;
    opt.initialize_spectral_jitter = initialize_spectral_jitter;
    opt.initialize_spectral_jitter_sd = initialize_spectral_jitter_sd;
    opt.initialize_random_scale = initialize_random_scale;
    opt.initialize_seed = initialize_seed;
    opt.num_epochs = num_epochs;

    opt.learning_rate = learning_rate;
    opt.negative_sample_rate = negative_sample_rate;
    opt.optimize_seed = optimize_seed;
    opt.num_threads = num_threads;
    opt.parallel_optimization = parallel_optimization;

    auto status = umappp::initialize(std::move(neighbors), ndim, embedding.data(), opt);
    status.run(embedding.data());

    auto output = create_numpy_array<double>(ndim, nobs);
    std::copy(embedding.begin(), embedding.end(), static_cast<double*>(output.request().ptr));
    return output;
}

void init_run_umap(pybind11::module& m) {
    m.def("run_umap", &run_umap);
}
