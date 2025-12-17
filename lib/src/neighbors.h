#ifndef UTILS_NEIGHBORS_H
#define UTILS_NEIGHBORS_H

#include <vector>
#include <stdexcept>
#include <cstdint>

#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"

#include "utils.h"

typedef pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast> NeighborIndexArray;

typedef pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast> NeighborDistanceArray;

template<typename Index_, class Distance_>
std::vector<std::vector<std::pair<Index_, Distance_> > > unpack_neighbors(const NeighborIndexArray& nnidx, const NeighborDistanceArray& nndist) {
    auto ibuffer = nnidx.request();
    const auto nobs = ibuffer.shape[0], nneighbors = ibuffer.shape[1];
    const auto iptr = get_numpy_array_data<std::uint32_t>(nnidx);

    auto dbuffer = nndist.request();
    if (!sanisizer::is_equal(nobs, dbuffer.shape[0]) || !sanisizer::is_equal(nneighbors, dbuffer.shape[1])) {
        throw std::runtime_error("neighbor indices and distances should have the same shape");
    }
    const auto dptr = get_numpy_array_data<double>(nndist);

    auto neighbors = sanisizer::create<std::vector<std::vector<std::pair<Index_, Distance_> > > >(nobs);
    for (I<decltype(nobs)> i = 0; i < nobs; ++ i) {
        auto& current = neighbors[i];
        current.reserve(nneighbors);
        for (I<decltype(nneighbors)> k = 0; k < nneighbors; ++k) {
            const auto offset = sanisizer::nd_offset<std::size_t>(k, nneighbors, i);
            current.emplace_back(iptr[offset], dptr[offset]);
        }
    }

    return neighbors;
}

#endif
