#include <vector>
#include <stdexcept>
#include <cstdint>
#include <cstddef>
#include <optional>

#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"
#include "pybind11/stl.h"
#include "scran_markers/scran_markers.hpp"

#include "utils.h"
#include "markers.h"

pybind11::list summarize_effects(
    pybind11::array_t<double, pybind11::array::f_style | pybind11::array::forcecast> effects,
    bool compute_min,
    bool compute_mean,
    bool compute_median,
    bool compute_max,
    std::optional<pybind11::array_t<double, pybind11::array::f_style | pybind11::array::forcecast> > compute_quantiles,
    bool compute_min_rank,
    int num_threads
) {
    auto ebuffer = effects.request();
    if (ebuffer.shape.size() != 3) {
        throw std::runtime_error("expected a 3-dimensional array for the effects");
    }
    const auto num_groups = ebuffer.shape[0];
    if (!sanisizer::is_equal(num_groups, ebuffer.shape[1])) {
        throw std::runtime_error("first two dimensions of the effects array should have the same extent");
    }
    const auto num_genes = ebuffer.shape[2];
    const double* eptr = get_numpy_array_data<double>(effects);

    scran_markers::SummarizeEffectsOptions opt;
    opt.num_threads = num_threads;
    const std::size_t num_quantiles = setup_quantile_options(compute_quantiles, opt.compute_quantiles);

    std::vector<pybind11::array_t<double> > min, mean, median, max;
    std::vector<std::vector<pybind11::array_t<double> > > quantiles;
    std::vector<pybind11::array_t<std::uint32_t> > min_rank;

    std::vector<scran_markers::SummaryBuffers<double, std::uint32_t> > groupwise;
    initialize_summary_buffers(
        num_groups,
        num_genes,
        groupwise,
        compute_min,
        min,
        compute_mean,
        mean,
        compute_median,
        median,
        compute_max,
        max,
        num_quantiles,
        quantiles,
        compute_min_rank,
        min_rank 
    );

    scran_markers::summarize_effects(num_genes, num_groups, eptr, groupwise, opt);

    return format_summary_output(
        num_groups,
        compute_min,
        min,
        compute_mean,
        mean,
        compute_median,
        median,
        compute_max,
        max,
        opt.compute_quantiles.has_value(),
        quantiles,
        compute_min_rank,
        min_rank
    );
}

void init_summarize_effects(pybind11::module& m) {
    m.def("summarize_effects", &summarize_effects);
}
