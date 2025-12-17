#include <stdexcept>
#include <cstdint>

#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"
#include "pybind11/stl.h"
#include "scran_aggregate/scran_aggregate.hpp"
#include "tatami_stats/tatami_stats.hpp"

#include "mattress.h"
#include "utils.h"

pybind11::tuple aggregate_across_cells(std::uintptr_t x, UnsignedArray groups, int num_threads) {
    const auto& mat = mattress::cast(x)->ptr;
    const auto NC = mat->ncol();
    const auto NR = mat->nrow();
    if (!sanisizer::is_equal(groups.size(), NC)) {
        throw std::runtime_error("length of 'groups' should be equal to the number of columns in 'x'");
    }

    const auto gptr = get_numpy_array_data<std::uint32_t>(groups);
    const auto ncombos = tatami_stats::total_groups(gptr, NC);
    auto sums = create_numpy_matrix<double>(NR, ncombos);
    auto detected = create_numpy_matrix<std::uint32_t>(NR, ncombos);

    scran_aggregate::AggregateAcrossCellsBuffers<double, std::uint32_t> buffers;
    {
        buffers.sums.reserve(ncombos);
        buffers.detected.reserve(ncombos);
        auto osum = static_cast<double*>(sums.request().ptr);
        auto odet = static_cast<std::uint32_t*>(detected.request().ptr);
        for (I<decltype(ncombos)> i = 0; i < ncombos; ++i) {
            const auto offset = sanisizer::product_unsafe<std::size_t>(NR, i);
            buffers.sums.push_back(osum + offset);
            buffers.detected.push_back(odet + offset);
        }
    }

    scran_aggregate::AggregateAcrossCellsOptions opt;
    opt.num_threads = num_threads;
    scran_aggregate::aggregate_across_cells(*mat, gptr, buffers, opt);

    pybind11::tuple output(2);
    output[0] = std::move(sums);
    output[1] = std::move(detected);
    return output;
}

void init_aggregate_across_cells(pybind11::module& m) {
    m.def("aggregate_across_cells", &aggregate_across_cells);
}
