#include <vector>
#include <stdexcept>
#include <string>
#include <optional>
#include <cstdint>
#include <cstddef>
#include <memory>

#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"
#include "pybind11/stl.h"
#include "mumosa/mumosa.hpp"
#include "knncolle_py.h"

#include "utils.h"
#include "block.h"

pybind11::array scale_by_neighbors(
    pybind11::ssize_t num_cells,
    const pybind11::list& embedding,
    int num_neighbors,
    std::optional<pybind11::array_t<std::uint32_t, pybind11::array::f_style | pybind11::array::forcecast> > block,
    std::string block_weight_policy,
    pybind11::tuple variable_block_weight,
    int num_threads,
    std::uintptr_t nn_builder
) {
    const auto nmod = embedding.size();
    std::vector<std::pair<double, double> > values;
    values.reserve(nmod);
    const auto& builder = knncolle_py::cast_builder(nn_builder)->ptr;
    sanisizer::cast<knncolle_py::Index>(num_cells);

    if (block.has_value()) {
        mumosa::BlockedOptions opt;
        opt.num_neighbors = num_neighbors;
        opt.num_threads = num_threads;
        opt.block_weight_policy = parse_block_weight_policy(block_weight_policy);
        opt.variable_block_weight_parameters = parse_variable_block_weight(variable_block_weight);

        if (!sanisizer::is_equal(num_cells, block->size())) {
            throw std::runtime_error("length of 'block' should equal the number of cells");
        }
        const auto ptr = get_numpy_array_data<std::uint32_t>(*block);

        mumosa::BlockedIndicesFactory<knncolle_py::Index, std::uint32_t> factory(num_cells, ptr);
        auto buff = factory.create_buffers<double>();
        auto work = mumosa::create_workspace<double>(factory.sizes(), opt);

        std::vector<std::shared_ptr<const knncolle::Prebuilt<knncolle_py::Index, knncolle_py::MatrixValue, knncolle_py::Distance> > > prebuilts;
        for (I<decltype(nmod)> x = 0; x < nmod; ++x) {
            const auto current = embedding[x].template cast<pybind11::array_t<double, pybind11::array::f_style | pybind11::array::forcecast> >();
            const auto& curbuffer = current.request();
            factory.build(
                sanisizer::cast<std::size_t>(curbuffer.shape[0]),
                static_cast<const double*>(curbuffer.ptr),
                *builder,
                prebuilts,
                buff
            );
            values.push_back(mumosa::compute_distance_blocked(prebuilts, work, opt));
        }

    } else {
        auto dist = sanisizer::create<std::vector<double> >(num_cells); 
        mumosa::Options opt;
        opt.num_neighbors = num_neighbors;
        opt.num_threads = num_threads;

        for (I<decltype(nmod)> x = 0; x < nmod; ++x) {
            const auto current = embedding[x].template cast<pybind11::array_t<double, pybind11::array::f_style | pybind11::array::forcecast> >();
            const auto& curbuffer = current.request();
            const auto prebuilt = builder->build_unique(
                knncolle::SimpleMatrix(
                    sanisizer::cast<std::size_t>(curbuffer.shape[0]),
                    static_cast<std::uint32_t>(num_cells),
                    static_cast<const double*>(curbuffer.ptr)
                )
            );
            values.push_back(mumosa::compute_distance<std::uint32_t, double>(*prebuilt, dist.data(), opt));
        }
    }

    auto output = mumosa::compute_scale<double>(values);
    return create_numpy_vector<double>(output.size(), output.data());
}

void init_scale_by_neighbors(pybind11::module& m) {
    m.def("scale_by_neighbors", &scale_by_neighbors);
}
