#ifndef UTILS_QC_H
#define UTILS_QC_H

#include <vector>
#include <stdexcept>
#include <optional>

#include "pybind11/pybind11.h"
#include "sanisizer/sanisizer.hpp"

#include "utils.h"

typedef pybind11::array_t<bool, pybind11::array::f_style | pybind11::array::forcecast> BoolArray;

template<typename Ngenes_>
std::vector<BoolArray> configure_qc_subsets(Ngenes_ ngenes, const pybind11::list& subsets) {
    const auto nsub = subsets.size();
    std::vector<BoolArray> in_subsets;
    in_subsets.reserve(nsub);

    for (I<decltype(nsub)> s = 0; s < nsub; ++s) {
        auto cursub = subsets[s].template cast<BoolArray>();
        if (!sanisizer::is_equal(ngenes, cursub.size())) {
            throw std::runtime_error("each entry of 'subsets' should have the same length as 'x.shape[0]'");
        }
        in_subsets.emplace_back(std::move(cursub));
    }

    return in_subsets;
}

template<typename Ncells_, typename Nsub_>
inline pybind11::list prepare_subset_metrics(Ncells_ ncells, Nsub_ nsub, std::vector<double*>& ptrs) {
    auto out_subsets = sanisizer::create<pybind11::list>(nsub);
    ptrs.clear();
    ptrs.reserve(nsub);

    for (I<decltype(nsub)> s = 0; s < nsub; ++s) {
        auto sub = sanisizer::create<pybind11::array_t<double> >(ncells);
        ptrs.push_back(static_cast<double*>(sub.request().ptr));
        out_subsets[s] = std::move(sub);
    }

    return out_subsets;
}

template<typename Ncells_>
void check_subset_metrics(Ncells_ ncells, const pybind11::list& input, std::vector<DoubleArray>& store) {
    const auto nsub = input.size();
    store.clear();
    store.reserve(nsub);

    for (I<decltype(nsub)> s = 0; s < nsub; ++s) {
        auto cursub = input[s].template cast<DoubleArray>();
        if (!sanisizer::is_equal(cursub.size(), ncells)) {
            throw std::runtime_error("all 'metrics' vectors should have the same length");
        }
        store.emplace_back(std::move(cursub));
    }
}

inline pybind11::list create_subset_filters(const std::vector<std::vector<double> >& input) {
    const auto nsub = input.size();
    auto subs = sanisizer::create<pybind11::list>(nsub);
    for (I<decltype(nsub)> s = 0; s < nsub; ++s) {
        const auto& cursub = input[s];
        subs[s] = pybind11::array_t<double>(cursub.size(), cursub.data());
    }
    return subs;
}

template<typename Nblocks_>
void copy_filters_blocked(std::optional<Nblocks_>& nblocks, const DoubleArray& input, std::vector<double>& store) {
    if (nblocks.has_value()) {
        if (!sanisizer::is_equal(input.size(), *nblocks)) {
            throw std::runtime_error("each array of thresholds in 'filters' should have length equal to the number of blocks");
        }
    } else {
        nblocks = sanisizer::cast<Nblocks_>(input.size());
    }
    auto ptr = get_numpy_array_data<double>(input);
    store.insert(store.end(), ptr, ptr + *nblocks);
}

template<typename Nsubs_, typename Nblocks_>
void copy_subset_filters_blocked(Nsubs_ nsub, std::optional<Nblocks_>& nblocks, const pybind11::list& subsets, std::vector<std::vector<double> >& store) {
    if (!sanisizer::is_equal(subsets.size(), nsub)) {
        throw std::runtime_error("'filters.subset_*' should have the same length as the number of subsets in 'metrics'");
    }

    sanisizer::resize(store, nsub);
    for (I<decltype(nsub)> s = 0; s < nsub; ++s) {
        const auto cursub = subsets[s].template cast<DoubleArray>();
        copy_filters_blocked(nblocks, cursub, store[s]);
    }
}

template<typename Nsubs_>
void copy_subset_filters_unblocked(Nsubs_ nsub, const DoubleArray& subsets, std::vector<double>& store) {
    if (!sanisizer::is_equal(subsets.size(), nsub)) {
        throw std::runtime_error("'filters.subset_*' should have the same length as the number of subsets in 'metrics'");
    }
    auto subptr = get_numpy_array_data<double>(subsets);
    store.insert(store.end(), subptr, subptr + nsub);
}

#endif
