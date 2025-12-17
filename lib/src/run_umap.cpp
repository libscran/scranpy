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
    NeighborIndexArray nnidx,
    NeighborDistanceArray nndist,
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
    std::optional<DoubleArray> initial_coordinates,
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
    int num_threads
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
        const auto& init_shape = initial_coordinates->request().shape;
        if (!sanisizer::is_equal(init_shape[0], ndim) || !sanisizer::is_equal(init_shape[1], nobs)) {
            throw std::runtime_error("shape of the initial coordinates is not consistent with that of the output embeddings");
        }
        auto iptr = get_numpy_array_data<double>(*initial_coordinates);
        std::copy_n(iptr, embedding.size(), embedding.data());
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

    auto status = umappp::initialize(std::move(neighbors), ndim, embedding.data(), opt);
    status.run(embedding.data());

    auto output = create_numpy_matrix<double>(ndim, nobs);
    std::copy(embedding.begin(), embedding.end(), static_cast<double*>(output.request().ptr));
    return output;
}

void init_run_umap(pybind11::module& m) {
    m.def("run_umap", &run_umap);
}
