#include <vector>
#include <stdexcept>
#include <cstdint>
#include <cstddef>
#include <optional>
#include <string>

#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"
#include "pybind11/stl.h"
#include "scran_variances/scran_variances.hpp"
#include "sanisizer/sanisizer.hpp"
#include "tatami/tatami.hpp"

#include "mattress.h"
#include "utils.h"
#include "block.h"

pybind11::dict model_gene_variances(
    std::uintptr_t x,
    std::optional<pybind11::array_t<std::uint32_t, pybind11::array::f_style | pybind11::array::forcecast> > maybe_block,
    std::size_t nblocks,
    std::string block_average_policy,
    std::string block_weight_policy,
    pybind11::tuple variable_block_weight,
    double block_quantile,
    bool mean_filter,
    double min_mean,
    bool transform,
    double span,
    bool use_min_width,
    double min_width,
    int min_window_count,
    int num_threads
) {
    scran_variances::ModelGeneVariancesOptions opt;
    opt.fit_variance_trend_options.mean_filter = mean_filter;
    opt.fit_variance_trend_options.minimum_mean = min_mean;
    opt.fit_variance_trend_options.transform = transform;
    opt.fit_variance_trend_options.span = span;
    opt.fit_variance_trend_options.use_minimum_width = use_min_width;
    opt.fit_variance_trend_options.minimum_width = min_width;
    opt.fit_variance_trend_options.minimum_window_count = min_window_count;
    opt.num_threads = num_threads;

    if (block_average_policy == "mean") {
        opt.block_average_policy = scran_variances::BlockAveragePolicy::MEAN;
    } else if (block_average_policy == "quantile") {
        opt.block_average_policy = scran_variances::BlockAveragePolicy::QUANTILE;
    } else {
        throw std::runtime_error("block average policy should be either 'mean' or 'quantile'");
    }

    opt.block_weight_policy = parse_block_weight_policy(block_weight_policy);
    opt.variable_block_weight_parameters = parse_variable_block_weight(variable_block_weight);
    opt.block_quantile = block_quantile;

    const auto& mat = mattress::cast(x)->ptr;
    const auto nc = mat->ncol();
    const auto nr = mat->nrow();

    auto means = tatami::create_container_of_Index_size<pybind11::array_t<double> >(nr);
    auto variances = tatami::create_container_of_Index_size<pybind11::array_t<double> >(nr);
    auto fitted = tatami::create_container_of_Index_size<pybind11::array_t<double> >(nr);
    auto residuals = tatami::create_container_of_Index_size<pybind11::array_t<double> >(nr);
    scran_variances::ModelGeneVariancesBuffers<double> buffers;
    buffers.means = static_cast<double*>(means.request().ptr);
    buffers.variances = static_cast<double*>(variances.request().ptr);
    buffers.fitted = static_cast<double*>(fitted.request().ptr);
    buffers.residuals = static_cast<double*>(residuals.request().ptr);

    pybind11::dict output;

    if (maybe_block.has_value()) {
        const auto& block = *maybe_block;
        if (!sanisizer::is_equal(block.size(), nc)) {
            throw std::runtime_error("'block' must be the same length as the number of cells");
        }
        auto bptr = get_numpy_array_data<std::uint32_t>(block);

        scran_variances::ModelGeneVariancesBlockedBuffers<double> bbuffers;
        bbuffers.average = buffers;
        sanisizer::resize(bbuffers.per_block, nblocks);

        std::vector<pybind11::array_t<double> > block_mean, block_var, block_fit, block_res;
        block_mean.reserve(nblocks);
        block_var.reserve(nblocks);
        block_fit.reserve(nblocks);
        block_res.reserve(nblocks);

        tatami::can_cast_Index_to_container_size<pybind11::array_t<double> >(nr);
        for (I<decltype(nblocks)> b = 0; b < nblocks; ++b) {
            block_mean.emplace_back(nr);
            bbuffers.per_block[b].means = static_cast<double*>(block_mean.back().request().ptr);
            block_var.emplace_back(nr);
            bbuffers.per_block[b].variances = static_cast<double*>(block_var.back().request().ptr);
            block_fit.emplace_back(nr);
            bbuffers.per_block[b].fitted = static_cast<double*>(block_fit.back().request().ptr);
            block_res.emplace_back(nr);
            bbuffers.per_block[b].residuals = static_cast<double*>(block_res.back().request().ptr);
        }

        scran_variances::model_gene_variances_blocked(*mat, bptr, bbuffers, opt);

        auto pb = sanisizer::create<pybind11::tuple>(nblocks);
        for (I<decltype(nblocks)> b = 0; b < nblocks; ++b) {
            pybind11::tuple current(4);
            current[0] = std::move(block_mean[b]);
            current[1] = std::move(block_var[b]);
            current[2] = std::move(block_fit[b]);
            current[3] = std::move(block_res[b]);
            pb[b] = current;
        }
        output["per_block"] = std::move(pb);

    } else {
        scran_variances::model_gene_variances(*mat, buffers, opt);
        output["per_block"] = pybind11::none();
    }

    output["mean"] = std::move(means);
    output["variance"] = std::move(variances);
    output["fitted"] = std::move(fitted);
    output["residual"] = std::move(residuals);
    return output;
}

void init_model_gene_variances(pybind11::module& m) {
    m.def("model_gene_variances", &model_gene_variances);
}
