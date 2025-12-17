#include <vector>
#include <stdexcept>
#include <cstdint>
#include <string>
#include <optional>

#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"
#include "pybind11/stl.h"
#include "gsdecon/gsdecon.hpp"
#include "mattress.h"

#include "utils.h"
#include "block.h"

pybind11::tuple score_gene_set(
    std::uintptr_t x,
    int rank,
    std::optional<UnsignedArray> maybe_block,
    std::string block_weight_policy,
    const pybind11::tuple& variable_block_weight,
    bool scale,
    bool realized,
    int irlba_work,
    int irlba_iterations,
    int irlba_seed,
    int num_threads
) {
    const auto& matrix = *(mattress::cast(x)->ptr);

    gsdecon::Options opt;
    opt.rank = rank;
    opt.scale = scale;
    opt.block_weight_policy = parse_block_weight_policy(block_weight_policy);
    opt.variable_block_weight_parameters = parse_variable_block_weight(variable_block_weight);
    opt.realize_matrix = realized;
    opt.irlba_options.extra_work = irlba_work;
    opt.irlba_options.max_iterations = irlba_iterations;
    opt.irlba_options.seed = irlba_seed;
    opt.num_threads = num_threads;

    const auto NR = matrix.nrow();
    const auto NC = matrix.ncol();
    auto scores = sanisizer::create<pybind11::array_t<double> >(NC);
    auto weights = sanisizer::create<pybind11::array_t<double> >(NR);
    gsdecon::Buffers<double> buffers;
    buffers.scores = static_cast<double*>(scores.request().ptr);
    buffers.weights = static_cast<double*>(weights.request().ptr);

    if (maybe_block.has_value()) {
        const auto& block = *maybe_block;
        if (static_cast<size_t>(block.size()) != NC) {
            throw std::runtime_error("'block' must be the same length as the number of cells");
        }
        gsdecon::compute_blocked(matrix, get_numpy_array_data<std::uint32_t>(block), opt, buffers);
    } else {
        gsdecon::compute(matrix, opt, buffers);
    }

    pybind11::tuple output(2);
    output[0] = scores;
    output[1] = weights;
    return output;
}

void init_score_gene_set(pybind11::module& m) {
    m.def("score_gene_set", &score_gene_set);
}
