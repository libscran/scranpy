#include <vector>
#include <stdexcept>
#include <cstdint>

#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"
#include "pybind11/stl.h"
#include "nenesub/nenesub.hpp"
#include "tatami/tatami.hpp"

#include "utils.h"

pybind11::array subsample_by_neighbors(const pybind11::array& indices, const pybind11::array& distances, int min_remaining) {
    auto ibuffer = indices.request();
    const auto nobs = ibuffer.shape[0], nneighbors = ibuffer.shape[1];
    if ((indices.flags() & pybind11::array::c_style) == 0) {
        throw std::runtime_error("expected a row-major matrix for the indices");
    }
    if (!indices.dtype().is(pybind11::dtype::of<std::uint32_t>())) {
        throw std::runtime_error("unexpected dtype for array of neighbor indices");
    }
    const auto iptr = get_numpy_array_data<std::uint32_t>(indices);

    auto dbuffer = distances.request();
    if ((distances.flags() & pybind11::array::c_style) == 0) {
        throw std::runtime_error("expected a row-major matrix for the distances");
    }
    if (!distances.dtype().is(pybind11::dtype::of<double>())) {
        throw std::runtime_error("unexpected dtype for array of neighbor distances");
    }
    if (!sanisizer::is_equal(nobs, dbuffer.shape[0]) || !sanisizer::is_equal(nneighbors, dbuffer.shape[1])) {
        throw std::runtime_error("neighbor indices and distances should have the same shape");
    }
    const auto dptr = get_numpy_array_data<double>(distances);

    if (sanisizer::is_greater_than(min_remaining, nneighbors)) {
        throw std::runtime_error("'min_remaining' should not be greater than the number of neighbors");
    }

    nenesub::Options opt;
    opt.min_remaining = min_remaining;
    std::vector<std::uint32_t> selected;
    nenesub::compute(
        sanisizer::cast<std::uint32_t>(nobs),
        /* get_neighbors = */ [&](std::uint32_t i) -> tatami::ArrayView<std::uint32_t> {
            return tatami::ArrayView<std::uint32_t>(iptr + sanisizer::product_unsafe<std::size_t>(nneighbors, i), nneighbors);
        },
        /* get_index = */ [](const tatami::ArrayView<std::uint32_t>& neighbors, std::uint32_t i) -> std::uint32_t {
            return neighbors[i];
        },
        /* get_max_distance = */ [&](std::uint32_t i) -> double {
            return dptr[sanisizer::nd_offset<std::size_t>(nneighbors - 1, nneighbors, i)];
        },
        opt, 
        selected
    );

    return create_numpy_vector<std::uint32_t>(selected.size(), selected.data());
}

void init_subsample_by_neighbors(pybind11::module& m) {
    m.def("subsample_by_neighbors", &subsample_by_neighbors);
}
