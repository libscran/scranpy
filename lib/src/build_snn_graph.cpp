#include <vector>
#include <stdexcept>
#include <string>
#include <cstdint>

#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"
#include "pybind11/stl.h"
#include "scran_graph_cluster/scran_graph_cluster.hpp"
#include "tatami/tatami.hpp"

#include "utils.h"
#include "neighbors.h"

pybind11::tuple build_snn_graph(NeighborIndexArray neighbors, std::string scheme, int num_threads) {
    const auto& ibuffer = neighbors.request();
    const auto ncells = ibuffer.shape[0], nneighbors = ibuffer.shape[1];
    const auto iptr = get_numpy_array_data<std::uint32_t>(neighbors);

    scran_graph_cluster::BuildSnnGraphOptions opt;
    opt.num_threads = num_threads;
    if (scheme == "ranked") {
        opt.weighting_scheme = scran_graph_cluster::SnnWeightScheme::RANKED;
    } else if (scheme == "number") {
        opt.weighting_scheme = scran_graph_cluster::SnnWeightScheme::NUMBER;
    } else if (scheme == "jaccard") {
        opt.weighting_scheme = scran_graph_cluster::SnnWeightScheme::JACCARD;
    } else {
        throw std::runtime_error("unknown weighting scheme '" + scheme + "'");
    }

    scran_graph_cluster::BuildSnnGraphResults<igraph_int_t, igraph_real_t> raw;
    scran_graph_cluster::build_snn_graph(
        ncells,
        [&](std::uint32_t i) -> tatami::ArrayView<std::uint32_t> {
            return tatami::ArrayView<uint32_t>(iptr + sanisizer::product_unsafe<std::size_t>(nneighbors, i), nneighbors);
        },
        [](std::uint32_t i) -> std::uint32_t { return i; },
        opt,
        raw 
    );

    pybind11::tuple output(3);
    output[0] = ncells;
    output[1] = create_numpy_vector<igraph_int_t>(raw.edges.size(), raw.edges.data());
    output[2] = create_numpy_vector<igraph_real_t>(raw.weights.size(), raw.weights.data());
    return output;
}

void init_build_snn_graph(pybind11::module& m) {
    m.def("build_snn_graph", &build_snn_graph);
}
