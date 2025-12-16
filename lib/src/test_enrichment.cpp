#include <vector>
#include <cstdint>
#include <stdexcept>
#include <cstddef>

#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"
#include "pybind11/stl.h"
#include "phyper/phyper.hpp"
#include "subpar/subpar.hpp"
#include "sanisizer/sanisizer.hpp"

#include "utils.h"

pybind11::array test_enrichment(
    pybind11::array_t<std::uint32_t, pybind11::array::f_style | pybind11::array::forcecast> overlap,
    std::uint32_t num_interest,
    pybind11::array_t<std::uint32_t, pybind11::array::f_style | pybind11::array::forcecast> set_sizes,
    std::uint32_t universe,
    bool log,
    int num_threads
) {
    const auto nsets = overlap.size();
    if (!sanisizer::is_equal(nsets, set_sizes.size())) {
        throw std::runtime_error("'overlap' and 'set_sizes' should have the same length");
    }

    phyper::Options opt;
    opt.upper_tail = true;
    opt.log = log;

    auto output = sanisizer::create<pybind11::array_t<double> >(nsets);
    double* optr = static_cast<double*>(output.request().ptr); // avoid any python references inside the parallel section.
    auto olptr = get_numpy_array_data<std::uint32_t>(overlap);
    auto ssptr = get_numpy_array_data<std::uint32_t>(set_sizes);

    subpar::parallelize(num_threads, nsets, [&](int, std::size_t start, std::size_t length) {
        for (std::size_t s = start, end = start + length; s < end; ++s) {
            optr[s] = phyper::compute(
                olptr[s],
                ssptr[s],
                universe - ssptr[s],
                num_interest,
                opt
            );
        }
    });

    return output;
}

void init_test_enrichment(pybind11::module& m) {
    m.def("test_enrichment", &test_enrichment);
}
