#include <cstdint>

#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"
#include "pybind11/stl.h"
#include "qdtsne/qdtsne.hpp"

#include "neighbors.h"

pybind11::array run_tsne(
    pybind11::array_t<std::uint32_t, pybind11::array::f_style | pybind11::array::forcecast> nnidx,
    pybind11::array_t<double, pybind11::array::f_style | pybind11::array::forcecast> nndist,
    double perplexity,
    double theta,
    int early_exaggeration_iterations,
    double exaggeration_factor,
    int momentum_switch_iterations,
    double start_momentum,
    double final_momentum,
    double eta,
    int max_depth,
    int leaf_approx,
    int max_iter,
    int seed,
    int num_threads
) {
    qdtsne::Options opt;
    opt.perplexity = perplexity;
    opt.infer_perplexity = false; // rely on the perplexity supplied by the user.
    opt.theta = theta;
    opt.early_exaggeration_iterations = early_exaggeration_iterations;
    opt.exaggeration_factor = exaggeration_factor;
    opt.momentum_switch_iterations = momentum_switch_iterations;
    opt.start_momentum = start_momentum;
    opt.final_momentum = final_momentum;
    opt.leaf_approximation = leaf_approx;
    opt.max_depth = max_depth;
    opt.max_iterations = max_iter;
    opt.num_threads = num_threads;

    auto neighbors = unpack_neighbors<std::uint32_t, double>(nnidx, nndist);
    const auto nobs = neighbors.size();
    auto status = qdtsne::initialize<2>(std::move(neighbors), opt);

    auto output = create_numpy_matrix<double>(2, nobs);
    auto optr = static_cast<double*>(output.request().ptr);
    qdtsne::initialize_random<2>(optr, nobs, seed);
    status.run(optr);

    return output;
}

int perplexity_to_neighbors(double p) {
    return qdtsne::perplexity_to_k(p);
}

void init_run_tsne(pybind11::module& m) {
    m.def("run_tsne", &run_tsne);
    m.def("perplexity_to_neighbors", &perplexity_to_neighbors);
}
