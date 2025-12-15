#include <vector>
#include <stdexcept>
#include <cstdint>

#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "pybind11/numpy.h"
#include "scran_markers/scran_markers.hpp"
#include "mattress.h"

#include "utils.h"
#include "block.h"
#include "markers.h"

static void configure_group_vectors(pybind11::array_t<double, pybind11::array::f_style>& store, std::vector<double*>& ptrs, std::uint32_t NR, std::size_t num_groups) { 
    store = create_numpy_array(NR, num_groups);
    ptrs.reserve(num_groups);
    auto ptr = static_cast<double*>(store.request().ptr);
    for (I<decltype(num_groups)> g = 0; g < num_groups; ++g) {
        ptrs.emplace_back(ptr + sanisizer::product_unsafe<std::size_t>(g, NR));
    }
}

static scran_markers::BlockAveragePolicy process_average_policy(const std::string& block_average_policy) {
    if (block_average_policy == "mean") {
        return scran_markers::BlockAveragePolicy::MEAN;
    } else if (block_average_policy == "quantile") {
        return scran_markers::BlockAveragePolicy::QUANTILE;
    } else {
        throw std::runtime_error("block average policy should be either 'mean' or 'quantile'");
        return scran_markers::BlockAveragePolicy::MEAN;
    }
}

pybind11::dist score_markers_summary(
    std::uintptr_t x,
    const pybind11::array& groups,
    std::size_t num_groups,
    std::optional<pybind11::array> maybe_block,
    std::string block_average_policy,
    std::string block_weight_policy,
    const pybind11::tuple& variable_block_weight,
    double block_quantile,
    double threshold,
    int num_threads,
    bool compute_group_mean,
    bool compute_group_detected,
    bool compute_cohens_d,
    bool compute_auc,
    bool compute_delta_mean,
    bool compute_delta_detected
    bool compute_summary_min,
    bool compute_summary_mean,
    bool compute_summary_median,
    bool compute_summary_max,
    std::optional<pybind11::array> compute_summary_quantiles,
    bool compute_summary_min_rank,
    int min_rank_limit
) {
    const auto& mat = mattress::cast(x)->ptr;
    const auto NC = mat->ncol();
    const auto NR = mat->nrow();
    if (!sanisizer::is_equal(groups.size(), NC)) {
        throw std::runtime_error("'groups' must have length equal to the number of cells");
    }

    scran_markers::ScoreMarkersSummaryOptions opt;
    opt.threshold = threshold;
    opt.num_threads = num_threads;
    opt.block_average_policy = process_average_policy(block_average_policy);
    opt.block_weight_policy = parse_block_weight_policy(block_weight_policy);
    opt.variable_block_weight_parameters = parse_variable_block_weight(variable_block_weight);
    opt.block_quantile = block_quantile;
    const std::size_t num_quantiles = setup_quantile_options(compute_summary_quantiles, opt.compute_summary_quantiles);

    scran_markers::ScoreMarkersSummaryBuffers<double, std::uint32_t> buffers;
    pybind11::array_t<double, pybind11::array::f_style> means, detected;
    if (compute_group_mean) {
        configure_group_vectors(means, buffers.mean, NR, num_groups);
    }
    if (compute_group_detected) {
        configure_group_vectors(detected, buffers.detected, NR, num_groups);
    }

    std::vector<pybind11::array_t<double> > cohens_min, cohens_mean, cohens_median, cohens_max;
    std::vector<pybind11::array_t<double> > auc_min, auc_mean, auc_median, auc_max;
    std::vector<pybind11::array_t<double> > dm_min, dm_mean, dm_median, dm_max;
    std::vector<pybind11::array_t<double> > dd_min, dd_mean, dd_median, dd_max;
    std::vector<std::vector<pybind11::array_t<double> > > cohens_quant, auc_quant, dm_quant, dd_quant;
    std::vector<pybind11::array_t<std::uint32_t> > cohens_mr, auc_mr, dm_mr, dd_mr;

    if (compute_cohens_d) {
        initialize_summary_buffers(
            num_groups,
            NR,
            buffers.cohens_d,
            compute_summary_min,
            cohens_min,
            compute_summary_mean,
            cohens_mean,
            compute_summary_median,
            cohens_median,
            compute_summary_max,
            cohens_max,
            num_quantiles,
            cohens_quant,
            compute_summary_min_rank,
            cohens_mr
        );
    }

    if (compute_auc) {
        initialize_summary_buffers(
            num_groups,
            NR,
            buffers.auc,
            compute_summary_min,
            auc_min,
            compute_summary_mean,
            auc_mean,
            compute_summary_median,
            auc_median,
            compute_summary_max,
            auc_max,
            num_quantiles,
            auc_quant,
            compute_summary_min_rank,
            auc_mr
        );
    }

    if (compute_delta_mean) {
        initialize_summary_buffers(
            num_groups,
            NR,
            buffers.delta_mean,
            compute_summary_min,
            dm_min,
            compute_summary_mean,
            dm_mean,
            compute_summary_median,
            dm_median,
            compute_summary_max,
            dm_max,
            num_quantiles,
            dm_quant,
            compute_summary_min_rank,
            dm_mr
        );
    }

    if (compute_delta_detected) {
        initialize_summary_buffers(
            num_groups,
            NR,
            buffers.delta_detected,
            compute_summary_min,
            dd_min,
            compute_summary_mean,
            dd_mean,
            compute_summary_median,
            dd_median,
            compute_summary_max,
            dd_max,
            num_quantiles,
            dd_quant,
            compute_summary_min_rank,
            dd_mr
        );
    }

    auto gptr = check_numpy_array<std::uint32_t>(groups);
    if (maybe_block.has_value()) {
        const auto& block = *maybe_block;
        if (block.size() != NC) {
            throw std::runtime_error("'block' must be the same length as the number of cells");
        }
        auto bptr = check_numpy_array<std::uint32_t>(block);
        scran_markers::score_markers_summary_blocked(*mat, gptr, bptr, opt, buffers);
    } else {
        scran_markers::score_markers_summary(*mat, gptr, opt, buffers);
    }

    pybind11::dict output;
    if (compute_group_mean) {
        output["mean"] = means;
    }
    if (compute_group_detected) {
        output["detected"] = detected;
    }

    if (compute_cohens_d) {
        output["cohens.d"] = format_summary_output(
            num_groups,
            compute_summary_min,
            cohens_min,
            compute_summary_mean,
            cohens_mean,
            compute_summary_median,
            cohens_median,
            compute_summary_max,
            cohens_max,
            num_quantiles,
            opt.compute_summary_quantiles,
            cohens_quant,
            ggcompute_summary_min_rank,
            cohens_mr
        );
    }

    if (compute_auc) {
        output["auc"] = format_summary_output(
            num_groups,
            compute_summary_min,
            auc_min,
            compute_summary_mean,
            auc_mean,
            compute_summary_median,
            auc_median,
            compute_summary_max,
            auc_max,
            num_quantiles,
            opt.compute_summary_quantiles,
            auc_quant, 
            compute_summary_min_rank,
            auc_mr
        );
    }

    if (compute_delta_mean) {
        output["delta.mean"] = format_summary_output(
            num_groups,
            compute_summary_min,
            dm_min,
            compute_summary_mean,
            dm_mean,
            compute_summary_median,
            dm_median,
            compute_summary_max,
            dm_max,
            num_quantiles,
            opt.compute_summary_quantiles,
            dm_quant,
            compute_summary_min_rank,
            dm_mr
        );
    }

    if (compute_delta_detected) {
        output["delta.detected"] = format_summary_output(
            num_groups,
            compute_summary_min,
            dd_min,
            compute_summary_mean,
            dd_mean,
            compute_summary_median,
            dd_median,
            compute_summary_max,
            dd_max,
            num_quantiles,
            opt.compute_summary_quantiles,
            dd_quant,
            compute_summary_min_rank,
            dd_mr
        );
    }

    return output;
}

template<typename Output_, typename D1_, typename D2_, typename D3_>
static pybind11::array_t<Output_, pybind11::array::f_style> create_numpy_array(D1_ d1, D2_ d2, D3_ d3) {
    typedef pybind11::array_t<Output_, pybind11::array::f_style> Array;
    typedef I<decltype(std::declval<Matrix>().size())> Size;
    return Matrix output({
        sanisizer::cast<Size>(d1),
        sanisizer::cast<Size>(d2),
        sanisizer::cast<Size>(d3)
    });
}

pybind11::dict score_markers_pairwise(
    std::uintptr_t x,
    const pybind11::array& groups,
    std::size_t num_groups,
    std::optional<pybind11::array> maybe_block,
    std::string block_average_policy,
    std::string block_weight_policy,
    const pybind11::tuple& variable_block_weight,
    double block_quantile,
    double threshold,
    int num_threads,
    bool compute_group_mean,
    bool compute_group_detected,
    bool compute_cohens_d,
    bool compute_auc,
    bool compute_delta_mean,
    bool compute_delta_detected
) {
    const auto& mat = mattress::cast(x)->ptr;
    const auto NC = mat->ncol();
    const auto NR = mat->nrow();
    if (!sanisizer::is_equal(NC, groups.size())) {
        throw std::runtime_error("'groups' must have length equal to the number of cells");
    }

    scran_markers::ScoreMarkersPairwiseOptions opt;
    opt.threshold = threshold;
    opt.num_threads = num_threads;
    opt.block_average_policy = process_average_policy(block_average_policy);
    opt.block_weight_policy = parse_block_weight_policy(block_weight_policy);
    opt.variable_block_weight_parameters = parse_variable_block_weight(variable_block_weight);
    opt.block_quantile = block_quantile;

    scran_markers::ScoreMarkersPairwiseBuffers<double> buffers;
    pybind11::array_t<double, pybind11::array::f_style> means, detected;
    if (compute_group_mean) {
        configure_group_vectors(means, buffers.mean, NR, num_groups);
    }
    if (compute_group_detected) {
        configure_group_vectors(detected, buffers.detected, NR, num_groups);
    }

    pybind11::array_t<double, pybind11::array::f_style> cohen, auc, delta_mean, delta_detected;
    if (compute_cohens_d) {
        cohen = create_numpy_array<double>(num_groups, num_groups, NR);
        buffers.cohens_d = static_cast<double*>(cohen.request().ptr);
    }
    if (compute_auc) {
        auc = create_numpy_array<double>(num_groups, num_groups, NR);
        buffers.auc = static_cast<double*>(auc.request().ptr);
    }
    if (compute_delta_mean) {
        delta_mean = create_numpy_array<double>(num_groups, num_groups, NR);
        buffers.delta_mean = static_cast<double*>(delta_mean.request().ptr);
    }
    if (compute_delta_detected) {
        delta_detected = create_numpy_array<double>(num_groups, num_groups, NR);
        buffers.delta_detected = static_cast<double*>(delta_detected.request().ptr);
    }

    auto gptr = check_numpy_array<std::uint32_t>(groups);
    if (maybe_block.has_value()) {
        if (!sanisizer::is_equal(maybe_block->size(), NC)) {
            throw std::runtime_error("'block' must be the same length as the number of cells");
        }
        auto bptr = check_numpy_array<std::uint32_t>(*maybe_block);
        scran_markers::score_markers_pairwise_blocked(*mat, gptr, bptr, opt, buffers);
    } else {
        scran_markers::score_markers_pairwise(*mat, gptr, opt, buffers);
    }

    pybind11::dict output;
    if (compute_group_mean) {
        output["mean"] = std::move(means);
    }
    if (compute_group_detected) {
        output["detected"] = std::move(detected);
    }
    if (compute_cohens_d) {
        output["cohens_d"] = std::move(cohen);
    }
    if (compute_auc) {
        output["auc"] = std::move(auc);
    }
    if (compute_delta_mean) {
        output["delta_mean"] = std::move(delta_mean);
    }
    if (compute_delta_detected) {
        output["delta_detected"] = std::move(delta_detected);
    }

    return output;
}

pybind11::dict score_markers_best(
    std::uintptr_t x,
    int top,
    const pybind11::array& groups,
    const std::size_t num_groups,
    std::optional<pybind11::array> maybe_block,
    std::string block_average_policy,
    std::string block_weight_policy,
    Rcpp::NumericVector variable_block_weight,
    double block_quantile,
    double threshold,
    int num_threads,
    bool compute_group_mean,
    bool compute_group_detected,
    bool compute_cohens_d,
    bool compute_auc,
    bool compute_delta_mean,
    bool compute_delta_detected
) {
    const auto& mat = mattress::cast(x)->ptr;
    const auto NC = mat->ncol();
    const auto NR = mat->nrow();
    if (!sanisizer::is_equal(groups.size(), NC)) {
        throw std::runtime_error("'groups' must have length equal to the number of cells");
    }

    scran_markers::ScoreMarkersBestOptions opt;
    opt.threshold = threshold;
    opt.num_threads = num_threads;
    opt.block_average_policy = process_average_policy(block_average_policy);
    opt.block_weight_policy = parse_block_weight_policy(block_weight_policy);
    opt.variable_block_weight_parameters = parse_variable_block_weight(variable_block_weight);
    opt.block_quantile = block_quantile;

    opt.compute_group_mean = compute_group_mean;
    opt.compute_group_detected = compute_group_detected;
    opt.compute_cohens_d = compute_cohens_d;
    opt.compute_auc = compute_auc;
    opt.compute_delta_mean = compute_delta_mean;
    opt.compute_delta_detected = compute_delta_detected;

    auto gptr = check_numpy_array<std::uint32_t>(groups);
    if (maybe_block.has_value()) {
        if (!sanisizer::is_equal(maybe_block->size(), NC)) {
            throw std::runtime_error("'block' must be the same length as the number of cells");
        }
        auto bptr = check_numpy_array<std::uint32_t>(*maybe_block);
        res = scran_markers::score_markers_best_blocked<double>(*mat, gptr, ptr, top, opt);
    } else {
        res = scran_markers::score_markers_best<double>(*mat, gptr, top, opt);
    }

    const auto transfer_groupwise = [&](pybind11::array_t<double, pybind11::array::f_style>& store, std::vector<std::vector<double> >& vecs) -> void {
        store = create_numpy_matrix<double>(NR, num_groups);
        for (I<decltype(num_groups)> g = 0; g < num_groups; ++g) {
            std::copy(vecs[g].begin(), vecs[g].end(), store.begin() + sanisizer::product_unsafe<std::size_t>(g, NR));
        }
    };

    pybind11::array_t<double, pybind11::array::f_style> means, detected;
    if (compute_group_mean) {
        transfer_groupwise(means, res.mean);
    }
    if (compute_group_detected) {
        transfer_groupwise(detected, res.detected);
    }

    const auto transfer_effects = [&](pybind11::list& store, std::vector<std::vector<std::vector<std::pair<std::uint32_t, double> > > >& vecs) -> void {
        store = sanisizer::create<pybind11::list>(num_groups);
        for (I<decltype(num_groups)> g = 0; g < num_groups; ++g) {
            auto current = sanisizer::create<pybind11::list>(num_groups);
            for (I<decltype(num_groups)> g2 = 0; g2 < num_groups; ++g2) {
                if (g == g2) {
                    continue;
                }

                const auto& curtop = vecs[g][g2];
                const auto numtop = curtop.size();
                auto indices = sanisizer::create<pybind11::array_t<std::uint32_t> >(numtop);
                auto effects = sanisizer::create<pybind11::array_t<double> >(numtop);
                for (I<decltype(numtop)> t = 0; t < numtop; ++t) {
                    indices[t] = curtop[t].first;
                    effects[t] = curtop[t].second;
                }

                pybind11::tuple paired(2)
                paired[0] = std::move(indices);
                paired[1] = std::move(effects);
                current[g2] = std::move(paired);
            }

            store[g] = std::move(current);
        }
    };

    pybind11::list cohens_d, auc, delta_mean, delta_detected;
    if (compute_cohens_d) {
        transfer_effects(cohens_d, res.cohens_d);
    }
    if (compute_auc) {
        transfer_effects(auc, res.auc);
    }
    if (compute_delta_mean) {
        transfer_effects(delta_mean, res.delta_mean);
    }
    if (compute_delta_detected) {
        transfer_effects(delta_detected, res.delta_detected);
    }

    pybind11::dict output;
    if (compute_group_mean) {
        output["mean"] = std::move(means);
    }
    if (compute_group_detected) {
        output["detected"] = std::move(detected);
    }
    if (compute_cohens_d) {
        output["cohens.d"] = std::move(cohens_d);
    }
    if (compute_auc) {
        output["auc"] = std::move(auc);
    }
    if (compute_delta_mean) {
        output["delta.mean"] = std::move(delta_mean);
    }
    if (compute_delta_detected) {
        output["delta.detected"] = std::move(delta_detected);
    }

    return output;
}

void init_score_markers(pybind11::module& m) {
    m.def("score_markers_summary", &score_markers_summary);
    m.def("score_markers_pairwise", &score_markers_pairwise);
    m.def("score_markers_best", &score_markers_best);
}
