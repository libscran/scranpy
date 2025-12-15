#include <vector>
#include <stdexcept>
#include <cstdint>
#include <cstddef>

#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"
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
    for (I<decltype(nsets)> s = 0; s < nsets; ++s) {
        const auto& current = sets[s];

        if (pybind11::isinstance<pybind11::array>(current)) {
            const auto& idx = current.template cast<pybind11::array>();
            converted_sets.emplace_back(idx.size(), check_numpy_array<std::uint32_t>(idx), static_cast<double*>(NULL));

        } else if (pybind11::isinstance<pybind11::tuple>(current)) {
            const auto& weighted = current.template cast<pybind11::tuple>();
            if (weighted.size() != 2) {
                throw std::runtime_error("tuple entries of 'sets' should be of length 2");
            }

            const auto& idx = weighted[0].template cast<pybind11::array>();
            const auto& wt = weighted[1].template cast<pybind11::array>();
            if (!sanisizer::is_equal(idx.size(), wt.size())) {
                throw std::runtime_error("tuple entries of 'sets' should have vectors of equal length");
            }

            converted_sets.emplace_back(idx.size(), check_numpy_array<uint32_t>(idx), check_numpy_array<double>(wt));

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
