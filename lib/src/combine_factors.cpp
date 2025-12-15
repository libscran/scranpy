#include <vector>
#include <algorithm>
#include <stdexcept>
#include <cstdint>

#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"
#include "pybind11/stl.h"
#include "scran_aggregate/scran_aggregate.hpp"

#include "utils.h"

static pybind11::tuple convert_to_index_list(const std::vector<std::vector<std::uint32_t> >& levels) {
    const auto num_fac = levels.size();
    auto combos = sanisizer::create<pybind11::tuple>(num_fac);
    for (I<decltype(num_fac)> f = 0; f < num_fac; ++f) {
        const auto& current = levels[f];
        combos[f] = create_numpy_vector<std::uint32_t>(current.size(), current.data());
    }
    return combos;
}

pybind11::tuple combine_factors(const pybind11::tuple& factors, bool keep_unused, const pybind11::array& num_levels) {
    const auto num_fac = factors.size();
    if (num_fac == 0) {
        throw std::runtime_error("'factors' must have length greater than zero");
    }

    std::vector<pybind11::array> ibuffers;
    ibuffers.reserve(num_fac);
    for (I<decltype(num_fac)> f = 0; f < num_fac; ++f) {
        ibuffers.emplace_back(factors[f].template cast<pybind11::array>());
    }

    const auto ngenes = ibuffers.front().size();
    for (I<decltype(num_fac)> f = 1; f < num_fac; ++f) {
        if (!sanisizer::is_equal(ibuffers[f].size(), ngenes)) {
            throw std::runtime_error("all elements of 'factors' must have the same length");
        }
    }

    pybind11::tuple output(2);

    if (keep_unused) {
        if (!sanisizer::is_equal(num_levels.size(), num_fac)) {
            throw std::runtime_error("'num_levels' and 'factors' must have the same length");
        }
        auto lptr = check_numpy_array<std::uint32_t>(num_levels);

        std::vector<std::pair<const std::uint32_t*, std::uint32_t> > buffers;
        buffers.reserve(num_fac);
        for (I<decltype(num_fac)> f = 0; f < num_fac; ++f) {
            buffers.emplace_back(check_numpy_array<std::uint32_t>(ibuffers[f]), lptr[f]);
        }

        auto oindices = sanisizer::create<pybind11::array_t<std::uint32_t> >(ngenes);
        auto res = scran_aggregate::combine_factors_unused(ngenes, buffers, static_cast<std::uint32_t*>(oindices.request().ptr));
        output[0] = std::move(oindices);
        output[1] = convert_to_index_list(res);

    } else {
        std::vector<const std::uint32_t*> buffers;
        buffers.reserve(num_fac);
        for (I<decltype(num_fac)> f = 0; f < num_fac; ++f) {
            buffers.emplace_back(check_numpy_array<std::uint32_t>(ibuffers[f]));
        }

        auto oindices = sanisizer::create<pybind11::array_t<std::uint32_t> >(ngenes);
        auto res = scran_aggregate::combine_factors(ngenes, buffers, static_cast<std::uint32_t*>(oindices.request().ptr));
        output[0] = std::move(oindices);
        output[1] = convert_to_index_list(res);
    }

    return output;
}

void init_combine_factors(pybind11::module& m) {
    m.def("combine_factors", &combine_factors);
}
