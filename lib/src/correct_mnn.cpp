#include <vector>
#include <cstdint>
#include <stdexcept>

#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "pybind11/numpy.h"
#include "mnncorrect/mnncorrect.hpp"
#include "knncolle_py.h"

#include "utils.h"

pybind11::tuple correct_mnn(
    const pybind11::array& x, 
    const pybind11::array& block, 
    int num_neighbors, 
    int num_steps, 
    int num_threads,
    std::string merge_policy, 
    std::uintptr_t builder_ptr
) {
    mnncorrect::Options<std::uint32_t, double> opts;
    opts.num_neighbors = num_neighbors;
    opts.num_steps = num_steps;
    opts.num_threads = num_threads;

    if (merge_policy == "input") {
        opts.merge_policy = mnncorrect::MergePolicy::INPUT;
    } else if (merge_policy == "max-variance" || merge_policy == "variance") {
        opts.merge_policy = mnncorrect::MergePolicy::VARIANCE;
    } else if (merge_policy == "max-rss" || merge_policy == "rss") {
        opts.merge_policy = mnncorrect::MergePolicy::RSS;
    } else if (merge_policy == "max-size" || merge_policy == "size") {
        opts.merge_policy = mnncorrect::MergePolicy::SIZE;
    } else {
        throw std::runtime_error("unknown merge policy");
    }

    const auto& builder = knncolle_py::cast_builder(builder_ptr)->ptr;
    typedef std::shared_ptr<knncolle::Builder<std::uint32_t, double, double> > BuilderPointer;
    opts.builder = BuilderPointer(BuilderPointer{}, builder.get()); // make a no-op shared pointer.

    auto xbuffer = x.request();
    if (xbuffer.shape.size() != 2) {
        throw std::runtime_error("expected a 2-dimensional array for 'x'");
    }
    if ((x.flags() & pybind11::array::f_style) == 0) {
        throw std::runtime_error("expected Fortran-style storage for 'x'");
    }
    if (!x.dtype().is(pybind11::dtype::of<double>())) {
        throw std::runtime_error("unexpected dtype for 'x'");
    }

    const auto ndim = xbuffer.shape[0];
    const auto nobs = sanisizer::cast<std::uint32_t>(xbuffer.shape[1]);
    if (!sanisizer::is_equal(nobs, block.size())) {
        throw std::runtime_error("length of 'block' should equal the number of columns in 'x'");
    }

    auto corrected = create_numpy_matrix<double>(ndim, nobs);
    mnncorrect::compute(
        ndim,
        nobs,
        static_cast<const double*>(xbuffer.ptr),
        check_numpy_array<std::uint32_t>(block),
        static_cast<double*>(corrected.request().ptr),
        opts
    );

    pybind11::dict output;
    output["corrected"] = std::move(corrected);
    return output;
}

void init_correct_mnn(pybind11::module& m) {
    m.def("correct_mnn", &correct_mnn);
}
