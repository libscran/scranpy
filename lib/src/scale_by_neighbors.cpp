#include <vector>
#include <stdexcept>
#include <string>
#include <optional>
#include <cstdint>
#include <cstddef>
#include <memory>

#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"
#include "mumosa/mumosa.hpp"

#include "utils.h"

static std::pair<const double*, pybind11::ssize_t> check_embedding_matrix(const pybind11::array& x, const pybind11::ssize_t num_cells) {
    const auto& xbuffer = x.buffer();
    if (xbuffer.shape.size() != 2) {
        throw std::runtime_error("expected a 2-dimensional array for entries of 'embedding'");
    }
    if (!sanisizer::is_equal(xbuffer.shape[1], num_cells) ) {
        throw std::runtime_error("number of columns in each entry of 'embedding' should equal the nuumber of cells");
    }

    if ((x.flags() & pybind11::array::f_style) == 0) {
        throw std::runtime_error("expected Fortran-style storage for entries of 'embedding'");
    }
    if (!x.dtype().is(pybind11::dtype::of<double>())) {
        throw std::runtime_error("unexpected dtype for 'x'");
    }

    return std::make_pair(
        get_numpy_array_data<double>(x),
        xbuffer.shape[0]
    );
}

pybind11::array scale_by_neighbors(
    pybind11::ssize_t num_cells,
    const pybind11::list& embedding,
    int num_neighbors,
    std::optional<pybind11::array> block,
    std::string block_weight_policy,
    Rcpp::NumericVector variable_block_weight,
    int num_threads,
    std::uintptr_t nn_builder
) {
    const auto nmod = embedding.size();
    std::vector<std::pair<double, double> > values;
    values.reserve(nmod);
    const auto& builder = knncolle_py::cast_builder(builder_ptr)->ptr;

    if (block.has_value()) {
        mumosa::BlockedOptions opt;
        opt.num_neighbors = num_neighbors;
        opt.num_threads = num_threads;
        opt.block_weight_policy = parse_block_weight_policy(block_weight_policy);
        opt.variable_block_weight_parameters = parse_variable_block_weight(variable_block_weight);

        const auto ptr = check_numpy_array<std::uint32_t>(*block);
        if (!sanisizer::is_equal(num_cells, blocks->size())) {
            throw std::runtime_error("length of 'block' should equal the number of cells");
        }

        mumosa::BlockedIndicesFactory<knncolle_py::Index, std::uint32_t> factory(
            sanisizer::cast<knncolle_py::Index>(num_cells),
            ptr
        );
        auto buff = factory.create_buffers<double>();
        auto work = mumosa::create_workspace<double>(factory.sizes(), opt);

        std::vector<std::shared_ptr<const BiocNeighbors::Prebuilt> > prebuilts;
        for (I<decltype(nmod)> x = 0; x < nmod; ++x) {
            auto current = embedding[x].template cast<pybind11::array>();
            auto info = check_embedding_matrix(current, num_cells);
            factory.build(sanisizer::cast<std::size_t>(info.second), info.first, *builder, prebuilts, buff);
            values.push_back(mumosa::compute_distance_blocked(prebuilts, work, opt));
        }

    } else {
        auto dist = sanisizer::create<std::vector<double> >(num_cells); 
        mumosa::Options opt;
        opt.num_neighbors = num_neighbors;
        opt.num_threads = num_threads;

        for (I<decltype(nmod)> x = 0; x < nmod; ++x) {
            auto current = embedding[x].template cast<pybind11::array>();
            auto info = check_embedding_matrix(current, num_cells);
            const auto prebuilt = builder->build_unique(
                knncolle::SimpleMatrix(
                    sanisizer::cast<std::size_t>(info.second),
                    sanisizer::cast<knncolle_py::Index>(num_cells),
                    info.first
                )
            );
            values.push_back(mumosa::compute_distance<std::uint32_t, double>(*prebuilt, dist.data(), opt));
        }
    }

    auto output = mumosa::compute_scale<double>(values);
    return Rcpp::NumericVector(output.begin(), output.end());
}

void init_scale_by_neighbors(pybind11::module& m) {
    m.def("scale_by_neighbors", &scale_by_neighbors);
}
