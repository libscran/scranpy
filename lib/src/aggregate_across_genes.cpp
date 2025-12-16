#include <vector>
#include <stdexcept>
#include <cstdint>
#include <cstddef>

#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"
#include "pybind11/stl.h"
#include "scran_aggregate/aggregate_across_genes.hpp"
#include "tatami_stats/tatami_stats.hpp"
#include "mattress.h"

#include "utils.h"

pybind11::list aggregate_across_genes(std::uintptr_t x, const pybind11::list& sets, bool average, int nthreads) {
    const auto& mat = mattress::cast(x)->ptr;
    const auto NR = mat->nrow();
    const auto NC = mat->ncol();

    const auto nsets = sets.size();
    std::vector<std::tuple<std::size_t, const std::uint32_t*, const double*> > converted_sets;
    converted_sets.reserve(nsets);

    // Hold arrays here so that pointers to the buffers remain valid if a forcecast was required. 
    std::vector<pybind11::array_t<std::uint32_t, pybind11::array::f_style | pybind11::array::forcecast> > collected_indices;
    collected_indices.reserve(nsets);
    std::vector<pybind11::array_t<double, pybind11::array::f_style | pybind11::array::forcecast> > collected_weights;
    collected_weights.reserve(nsets);

    for (I<decltype(nsets)> s = 0; s < nsets; ++s) {
        const auto& current = sets[s];

        if (pybind11::isinstance<pybind11::array>(current)) {
            collected_indices.emplace_back(current.template cast<pybind11::array_t<std::uint32_t, pybind11::array::f_style | pybind11::array::forcecast> >());
            converted_sets.emplace_back(
                collected_indices.back().size(),
                get_numpy_array_data<std::uint32_t>(collected_indices.back()),
                static_cast<double*>(NULL)
            );

        } else if (pybind11::isinstance<pybind11::tuple>(current)) {
            const auto weighted = current.template cast<pybind11::tuple>();
            if (weighted.size() != 2) {
                throw std::runtime_error("tuple entries of 'sets' should be of length 2");
            }

            collected_indices.emplace_back(weighted[0].template cast<pybind11::array_t<std::uint32_t, pybind11::array::f_style | pybind11::array::forcecast> >());
            collected_weights.emplace_back(weighted[1].template cast<pybind11::array_t<double, pybind11::array::f_style | pybind11::array::forcecast> >());
            if (!sanisizer::is_equal(collected_indices.back().size(), collected_weights.back().size())) {
                throw std::runtime_error("tuple entries of 'sets' should have vectors of equal length");
            }

            converted_sets.emplace_back(
                collected_indices.back().size(),
                get_numpy_array_data<std::uint32_t>(collected_indices.back()),
                get_numpy_array_data<double>(collected_weights.back())
            );

        } else {
            throw std::runtime_error("unsupported type of 'sets' entry");
        }
    }

    scran_aggregate::AggregateAcrossGenesBuffers<double> buffers;
    buffers.sum.reserve(nsets);
    auto output = sanisizer::create<pybind11::list>(nsets);
    for (I<decltype(nsets)> s = 0; s < nsets; ++s) {
        auto current = sanisizer::create<pybind11::array_t<double> >(NC);
        output[s] = current;
        buffers.sum.push_back(static_cast<double*>(current.request().ptr));
    }

    scran_aggregate::AggregateAcrossGenesOptions opt;
    opt.average = average;
    opt.num_threads = nthreads;
    scran_aggregate::aggregate_across_genes(*mat, converted_sets, buffers, opt);

    return output;
}

void init_aggregate_across_genes(pybind11::module& m) {
    m.def("aggregate_across_genes", &aggregate_across_genes);
}
