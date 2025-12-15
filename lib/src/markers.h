#ifndef MARKERS_H
#define MARKERS_H

inline void initialize_summary_buffers(
    const std::size_t num_groups,
    const std::uint32_t num_genes,
    std::vector<scran_markers::SummaryBuffers<double, std::uint32_t> >& ptrs,
    const bool compute_min,
    std::vector<pybind11::array_t<double> >& min,
    const bool compute_mean,
    std::vector<pybind11::array_t<double> >& mean,
    const bool compute_median,
    std::vector<pybind11::array_t<double> >& median,
    const bool compute_max,
    std::vector<pybind11::array_t<double> >& max,
    const std::size_t num_quantiles,
    std::vector<std::vector<pybind11::array_t<double> > >& quantiles,
    const bool compute_min_rank,
    std::vector<pybind11::array_t<std::uint32_t> >& min_rank
) {
    sanisizer::resize(ptrs, num_groups);

    if (compute_min) {
        min.reserve(num_groups);
    }
    if (compute_mean) {
        mean.reserve(num_groups);
    }
    if (compute_median) {
        median.reserve(num_groups);
    }
    if (compute_max) {
        max.reserve(num_groups);
    }
    if (num_quantiles) {
        quantiles.reserve(num_groups);
    }
    if (compute_min_rank) {
        min_rank.reserve(num_groups);
    }

    sanisizer::cast<I<decltype(std::declval<pybind11::array_t<double> >().size())> >(num_genes);
    sanisizer::cast<I<decltype(std::declval<pybind11::array_t<std::uint32_t> >().size())> >(num_genes);
    for (I<decltype(num_groups)> g = 0; g < num_groups; ++g) {
        auto& curptr = ptrs[g];

        if (compute_min) {
            min.emplace_back(num_genes);
            curptr.min = static_cast<double*>(min.back().request().ptr);
        }

        if (compute_mean) {
            mean.emplace_back(num_genes);
            curptr.mean = static_cast<double*>(mean.back().request().ptr);
        }

        if (compute_median) {
            median.emplace_back(num_genes);
            curptr.median = static_cast<double*>(median.back().request().ptr);
        }

        if (compute_max) {
            max.emplace_back(num_genes);
            curptr.max = static_cast<double*>(max.back().request().ptr);
        }

        if (num_quantiles) {
            quantiles.emplace_back();
            quantiles.back().reserve(num_quantiles);
            curptr.quantiles.emplace();
            curptr.quantiles->reserve(num_quantiles);
            for (I<decltype(num_quantiles)> q = 0; q < num_quantiles; ++q) {
                quantiles.back().emplace_back(num_genes);
                curptr.quantiles->push_back(static_cast<double*>(quantiles.back().back().request().ptr));
            }
        }

        if (compute_min_rank) {
            min_rank.emplace_back(num_genes);
            curptr.min_rank = static_cast<std::uint32_t*>(min_rank.back().request().ptr);
        }
    }
}

inline std::size_t setup_quantile_options(const std::optional<pybind11::array>& input, std::optional<std::vector<double> >& output) {
    if (!input.has_value()) {
        return 0;
    }
    auto iptr = check_numpy_array<double>(*input);
    output.emplace(iptr, iptr + input->size());
    return sanisizer::cast<std::size_t>(output->size());
}

inline pybind11::list format_summary_output(
    const std::size_t num_groups,
    const bool compute_min,
    std::vector<pybind11::array_t<double> >& min,
    const bool compute_mean,
    std::vector<pybind11::array_t<double> >& mean,
    const bool compute_median,
    std::vector<pybind11::array_t<double> >& median,
    const bool compute_max,
    std::vector<pybind11::array_t<double> >& max,
    const std::size_t num_quantiles,
    const std::optional<std::vector<double> >& input_quantiles,
    std::vector<std::vector<pybind11::array_t<double> > >& output_quantiles,
    const bool compute_min_rank,
    std::vector<pybind11::array_t<std::uint32_t> >& min_rank
) { 
    auto output = sanisizer::create<pybind11::list>(num_groups);
    for (I<decltype(num_groups)> g = 0; g < num_groups; ++g) {
        pybind11::dict current;

        if (compute_min) {
            current["min"] = std::move(min[g]);
        }

        if (compute_mean) {
            current["mean"] = std::move(mean[g]);
        }

        if (compute_median) {
            current["median"] = std::move(median[g]);
        }

        if (compute_max) {
            current["max"] = std::move(max[g]);
        }

        if (input_quantiles.has_value()) {
            const auto& iquantiles = *input_quantiles;
            const auto& oquantiles = output_quantiles[g];
            for (I<decltype(num_quantiles)> q = 0; q < num_quantiles; ++q) {
                auto qname = "quantile." + std::to_string(iquantiles[q] * 100);
                current[qname.c_str()] = std::move(oquantiles[q]);
            }
        }

        if (compute_min_rank) {
            current["min.rank"] = std::move(min_rank[g]);
        }

        output[g] = current;
    }
    return output;
}

#endif
