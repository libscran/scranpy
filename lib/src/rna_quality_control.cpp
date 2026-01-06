#include <vector>
#include <stdexcept>
#include <cstdint>
#include <optional>

#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"
#include "pybind11/stl.h"
#include "scran_qc/scran_qc.hpp"
#include "sanisizer/sanisizer.hpp"

#include "mattress.h"
#include "utils.h"
#include "qc.h"

pybind11::tuple compute_rna_qc_metrics(std::uintptr_t x, pybind11::list subsets, int num_threads) {
    const auto& mat = mattress::cast(x)->ptr;
    const auto nc = mat->ncol();
    const auto nr = mat->nrow();

    const auto nsub = subsets.size();
    const auto in_subsets = configure_qc_subsets(nr, subsets);
    std::vector<const bool*> subptrs;
    subptrs.reserve(nsub);
    for (const auto& sub : in_subsets) {
        subptrs.push_back(get_numpy_array_data<bool>(sub));
    }

    // Creating output containers.
    scran_qc::ComputeRnaQcMetricsBuffers<double, std::uint32_t> buffers;
    auto sum = tatami::create_container_of_Index_size<pybind11::array_t<double> >(nc);
    buffers.sum = static_cast<double*>(sum.request().ptr);
    auto detected = tatami::create_container_of_Index_size<pybind11::array_t<uint32_t> >(nc);
    buffers.detected = static_cast<uint32_t*>(detected.request().ptr);
    pybind11::list out_subsets = prepare_subset_metrics(nc, nsub, buffers.subset_proportion);

    // Running QC code.
    scran_qc::ComputeRnaQcMetricsOptions opt;
    opt.num_threads = num_threads;
    scran_qc::compute_rna_qc_metrics(*mat, subptrs, buffers, opt);

    pybind11::tuple output(3);
    output[0] = std::move(sum);
    output[1] = std::move(detected);
    output[2] = std::move(out_subsets);
    return output;
}

class ConvertedRnaQcMetrics {
public:
    ConvertedRnaQcMetrics(pybind11::tuple metrics) {
        if (metrics.size() != 3) {
            throw std::runtime_error("'metrics' should have the same format as the output of 'compute_rna_qc_metrics'");
        }

        sum = metrics[0].template cast<I<decltype(sum)> >();
        const auto ncells = sum.size();

        detected = metrics[1].template cast<I<decltype(detected)> >();
        if (!sanisizer::is_equal(ncells, detected.size())) {
            throw std::runtime_error("all 'metrics' vectors should have the same length");
        }

        auto tmp = metrics[2].template cast<pybind11::list>();
        check_subset_metrics(ncells, tmp, subsets);
    }

private:
    DoubleArray sum;
    UnsignedArray detected;
    std::vector<DoubleArray> subsets;

public:
    auto size() const {
        return sum.size();
    }

    auto num_subsets() const {
        return subsets.size();
    }

    auto to_buffer() const {
        scran_qc::ComputeRnaQcMetricsBuffers<const double, const std::uint32_t, const double> buffers;
        buffers.sum = get_numpy_array_data<double>(sum);
        buffers.detected = get_numpy_array_data<std::uint32_t>(detected);
        buffers.subset_proportion.reserve(subsets.size());
        for (auto& s : subsets) {
            buffers.subset_proportion.push_back(get_numpy_array_data<double>(s));
        }
        return buffers;
    }
};

pybind11::tuple suggest_rna_qc_thresholds(pybind11::tuple metrics, std::optional<UnsignedArray> maybe_block, double num_mads) {
    ConvertedRnaQcMetrics all_metrics(metrics);
    auto buffers = all_metrics.to_buffer();
    const auto ncells = all_metrics.size();

    scran_qc::ComputeRnaQcFiltersOptions opt;
    opt.sum_num_mads = num_mads;
    opt.detected_num_mads = num_mads;
    opt.subset_proportion_num_mads = num_mads;

    pybind11::tuple output(3);

    if (maybe_block.has_value()) {
        const auto& block = *maybe_block;
        if (!sanisizer::is_equal(block.size(), ncells)) {
            throw std::runtime_error("'block' must be the same length as the number of cells");
        }
        auto bptr = get_numpy_array_data<std::uint32_t>(block);

        auto filt = scran_qc::compute_rna_qc_filters_blocked(ncells, buffers, bptr, opt);
        const auto& sout = filt.get_sum();
        output[0] = create_numpy_vector<double>(sout.size(), sout.data());
        const auto& dout = filt.get_detected();
        output[1] = create_numpy_vector<double>(dout.size(), dout.data());
        const auto& ssout = filt.get_subset_proportion();
        output[2] = create_subset_filters(ssout);

    } else {
        auto filt = scran_qc::compute_rna_qc_filters(ncells, buffers, opt);
        output[0] = filt.get_sum();
        output[1] = filt.get_detected();
        const auto& ssout = filt.get_subset_proportion();
        output[2] = create_numpy_vector<double>(ssout.size(), ssout.data());
    }

    return output;
}

pybind11::array filter_rna_qc_metrics(pybind11::tuple filters, pybind11::tuple metrics, std::optional<UnsignedArray> maybe_block) {
    ConvertedRnaQcMetrics all_metrics(metrics);
    auto mbuffers = all_metrics.to_buffer();
    const auto ncells = all_metrics.size();
    const auto nsubs = all_metrics.num_subsets();

    if (filters.size() != 3) {
        throw std::runtime_error("'filters' should have the same format as the output of 'suggestRnaQcFilters'");
    }

    pybind11::array_t<bool> keep(ncells);
    bool* kptr = static_cast<bool*>(keep.request().ptr);

    if (maybe_block.has_value()) {
        const auto& block = *maybe_block;
        if (!sanisizer::is_equal(block.size(), ncells)) {
            throw std::runtime_error("'block' must be the same length as the number of cells");
        }
        auto bptr = get_numpy_array_data<std::uint32_t>(block);

        scran_qc::RnaQcBlockedFilters filt;
        std::optional<std::size_t> nblocks;
        const auto sum = filters[0].template cast<DoubleArray>();
        copy_filters_blocked(nblocks, sum, filt.get_sum());
        const auto detected = filters[1].template cast<DoubleArray>();
        copy_filters_blocked(nblocks, detected, filt.get_detected());
        const auto subsets = filters[2].template cast<pybind11::list>();
        copy_subset_filters_blocked(nsubs, nblocks, subsets, filt.get_subset_proportion());

        filt.filter(ncells, mbuffers, bptr, kptr);

    } else {
        scran_qc::RnaQcFilters filt;
        filt.get_sum() = filters[0].template cast<double>();
        filt.get_detected() = filters[1].template cast<double>();
        const auto subsets = filters[2].template cast<DoubleArray>();
        copy_subset_filters_unblocked(nsubs, subsets, filt.get_subset_proportion());
        filt.filter(ncells, mbuffers, kptr);
    }

    return keep;
}

void init_rna_quality_control(pybind11::module& m) {
    m.def("compute_rna_qc_metrics", &compute_rna_qc_metrics);
    m.def("suggest_rna_qc_thresholds", &suggest_rna_qc_thresholds);
    m.def("filter_rna_qc_metrics", &filter_rna_qc_metrics);
}
