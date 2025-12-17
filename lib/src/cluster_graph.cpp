#include <vector>
#include <stdexcept>
#include <optional>

#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"
#include "pybind11/stl.h"
#include "scran_graph_cluster/scran_graph_cluster.hpp"
#include "sanisizer/sanisizer.hpp"
#include "igraph.h"
#include "raiigraph/raiigraph.hpp"

#include "utils.h"

typedef pybind11::array_t<igraph_int_t, pybind11::array::f_style | pybind11::array::forcecast> EdgeArray;

typedef pybind11::array_t<igraph_real_t, pybind11::array::f_style | pybind11::array::forcecast> WeightArray;

static std::pair<raiigraph::Graph, std::optional<igraph_vector_t> > formulate_graph(const pybind11::tuple& graph, std::optional<WeightArray>& weight_buffer) {
    if (graph.size() != 3) {
        throw std::runtime_error("graph should be represented by a 3-tuple");
    }
    auto vertices = graph[0].template cast<std::size_t>();
    const auto edges = graph[1].template cast<EdgeArray>();

    std::optional<igraph_vector_t> weight_view;
    if (!pybind11::isinstance<pybind11::none>(graph[2])) {
        // Store it in a buffer so that the view remains valid if a forcecast was required.
        weight_buffer = graph[2].template cast<I<decltype(weight_buffer)> >();
        weight_view = igraph_vector_view(
            get_numpy_array_data<igraph_real_t>(*weight_buffer),
            sanisizer::cast<igraph_int_t>(weight_buffer->size())
        );
    }

    return std::make_pair(
        scran_graph_cluster::edges_to_graph(edges.size(), get_numpy_array_data<igraph_int_t>(edges), vertices, false),
        std::move(weight_view)
    );
}

static const igraph_vector_t* get_weight_ptr(const std::optional<igraph_vector_t>& x) {
    if (x.has_value()) {
        return &(*x);
    } else {
        return NULL;
    }
}

pybind11::dict cluster_multilevel(const pybind11::tuple& graph, double resolution, int seed) {
    std::optional<WeightArray> weight_buffer;
    auto gpair = formulate_graph(graph, weight_buffer);

    scran_graph_cluster::ClusterMultilevelOptions opt;
    opt.resolution = resolution;
    opt.seed = seed;
    scran_graph_cluster::ClusterMultilevelResults res;
    scran_graph_cluster::cluster_multilevel(gpair.first.get(), get_weight_ptr(gpair.second), opt, res);

    const auto nlevels = res.levels.nrow();
    pybind11::tuple levels(nlevels);
    for (I<decltype(nlevels)> l = 0; l < nlevels; ++l) {
        auto incol = res.levels.row(l);
        auto current = sanisizer::create<pybind11::array_t<igraph_int_t> >(incol.size());
        std::copy(incol.begin(), incol.end(), static_cast<igraph_int_t*>(current.request().ptr));
        levels[l] = std::move(current);
    }

    pybind11::dict output;
    output["membership"] = create_numpy_vector<igraph_int_t>(res.membership.size(), res.membership.data());
    output["levels"] = std::move(levels);
    output["modularity"] = create_numpy_vector<igraph_real_t>(res.modularity.size(), res.modularity.data());

    return output;
}

pybind11::dict cluster_leiden(const pybind11::tuple& graph, double resolution, std::string objective, int seed) {
    std::optional<WeightArray> weight_buffer;
    auto gpair = formulate_graph(graph, weight_buffer);

    scran_graph_cluster::ClusterLeidenOptions opt;
    opt.resolution = resolution;
    opt.seed = seed;
    opt.report_quality = true;

    if (objective == "modularity") {
        opt.objective = IGRAPH_LEIDEN_OBJECTIVE_MODULARITY;
    } else if (objective == "cpm") {
        opt.objective = IGRAPH_LEIDEN_OBJECTIVE_CPM;
    } else if (objective == "er") {
        opt.objective = IGRAPH_LEIDEN_OBJECTIVE_ER;
    } else {
        throw std::runtime_error("unknown Leiden objective '" + objective + "'");
    }

    scran_graph_cluster::ClusterLeidenResults res;
    scran_graph_cluster::cluster_leiden(gpair.first.get(), get_weight_ptr(gpair.second), opt, res);

    pybind11::dict output;
    output["membership"] = create_numpy_vector<igraph_int_t>(res.membership.size(), res.membership.data());
    output["quality"] = res.quality;

    return output;
}

pybind11::dict cluster_walktrap(const pybind11::tuple& graph, int steps) {
    std::optional<WeightArray> weight_buffer;
    auto gpair = formulate_graph(graph, weight_buffer);

    scran_graph_cluster::ClusterWalktrapOptions opt;
    opt.steps = steps;
    scran_graph_cluster::ClusterWalktrapResults res;
    scran_graph_cluster::cluster_walktrap(gpair.first.get(), get_weight_ptr(gpair.second), opt, res);

    const auto merge_nrow = res.merges.nrow(), merge_ncol = res.merges.ncol();
    auto merges = create_numpy_matrix<igraph_int_t>(merge_nrow, merge_ncol);
    for (I<decltype(merge_ncol)> m = 0; m < merge_ncol; ++m) {
        auto incol = res.merges.column(m);
        auto outptr = static_cast<igraph_int_t*>(merges.request().ptr) + sanisizer::product_unsafe<std::size_t>(m, merge_nrow);
        std::copy(incol.begin(), incol.end(), outptr);
    }

    pybind11::dict output;
    output["membership"] = create_numpy_vector<igraph_int_t>(res.membership.size(), res.membership.data());
    output["merges"] = std::move(merges);
    output["modularity"] = create_numpy_vector<igraph_real_t>(res.modularity.size(), res.modularity.data());

    return output;
}

void init_cluster_graph(pybind11::module& m) {
    m.def("cluster_multilevel", &cluster_multilevel);
    m.def("cluster_leiden", &cluster_leiden);
    m.def("cluster_walktrap", &cluster_walktrap);
}
