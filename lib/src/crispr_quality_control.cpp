#include <vector>
#include <stdexcept>
#include <cstdint>
#include <optional>

#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"
#include "pybind11/stl.h"
#include "scran_qc/scran_qc.hpp"
#include "sanisizer/sanisizer.hpp"
#include "tatami/tatami.hpp"

#include "mattress.h"
#include "utils.h"
#include "qc.h"

pybind11::tuple compute_crispr_qc_metrics(std::uintptr_t x, int num_threads) {
    const auto& mat = mattress::cast(x)->ptr;
    const auto nc = mat->ncol();

    // Creating output containers.
    scran_qc::ComputeCrisprQcMetricsBuffers<double, std::uint32_t, double, std::uint32_t> buffers;
    auto sum = tatami::create_container_of_Index_size<pybind11::array_t<double> >(nc);
    buffers.sum = static_cast<double*>(sum.request().ptr);
    auto detected = tatami::create_container_of_Index_size<pybind11::array_t<std::uint32_t> >(nc);
    buffers.detected = static_cast<std::uint32_t*>(detected.request().ptr);
    auto max_value = tatami::create_container_of_Index_size<pybind11::array_t<double> >(nc);
    buffers.max_value = static_cast<double*>(max_value.request().ptr);
    auto max_index = tatami::create_container_of_Index_size<pybind11::array_t<std::uint32_t> >(nc);
    buffers.max_index = static_cast<std::uint32_t*>(max_index.request().ptr);

    // Running QC code.
    scran_qc::ComputeCrisprQcMetricsOptions opt;
    opt.num_threads = num_threads;
    scran_qc::compute_crispr_qc_metrics(*mat, buffers, opt);

    pybind11::tuple output(4);
    output[0] = std::move(sum);
    output[1] = std::move(detected);
    output[2] = std::move(max_value);
    output[3] = std::move(max_index);
    return output;
}

class ConvertedCrisprQcMetrics {
public:
    ConvertedCrisprQcMetrics(pybind11::tuple metrics) {
        if (metrics.size() != 4) {
            throw std::runtime_error("'metrics' should have the same format as the output of 'compute_crispr_qc_metrics'");
        }

        sum = metrics[0].template cast<I<decltype(sum)> >();
        const auto ncells = sum.size();

        detected = metrics[1].template cast<I<decltype(detected)> >();
        if (!sanisizer::is_equal(ncells, detected.size())) {
            throw std::runtime_error("all 'metrics' vectors should have the same length");
        }

        max_value = metrics[2].template cast<I<decltype(max_value)> >();
        if (!sanisizer::is_equal(ncells, max_value.size())) {
            throw std::runtime_error("all 'metrics' vectors should have the same length");
        }

        max_index = metrics[3].template cast<I<decltype(max_index)> >();
        if (!sanisizer::is_equal(ncells, max_index.size())) {
            throw std::runtime_error("all 'metrics' vectors should have the same length");
        }
    }

private:
    DoubleArray sum;
    UnsignedArray detected;
    DoubleArray max_value;
    UnsignedArray max_index;

public:
    auto size() const {
        return sum.size();
    }

    auto to_buffer() const {
        scran_qc::ComputeCrisprQcMetricsBuffers<const double, const std::uint32_t, const double, const std::uint32_t> buffers;
        buffers.sum = get_numpy_array_data<double>(sum);
        buffers.detected = get_numpy_array_data<std::uint32_t>(detected);
        buffers.max_value = get_numpy_array_data<double>(max_value);
        buffers.max_index = get_numpy_array_data<std::uint32_t>(max_index);
        return buffers;
    }
};

pybind11::tuple suggest_crispr_qc_thresholds(
    pybind11::tuple metrics,
    std::optional<UnsignedArray> maybe_block,
    double num_mads
) {
    ConvertedCrisprQcMetrics all_metrics(metrics);
    auto buffers = all_metrics.to_buffer();
    const auto ncells = all_metrics.size();

    scran_qc::ComputeCrisprQcFiltersOptions opt;
    opt.max_value_num_mads = num_mads;

    pybind11::tuple output(1);

    if (maybe_block.has_value()) {
        const auto& block = *maybe_block;
        if (!sanisizer::is_equal(block.size(), ncells)) {
            throw std::runtime_error("'block' must be the same length as the number of cells");
        }
        const auto bptr = get_numpy_array_data<std::uint32_t>(block);

        auto filt = scran_qc::compute_crispr_qc_filters_blocked(ncells, buffers, bptr, opt);
        const auto& mout = filt.get_max_value();
        output[0] = create_numpy_vector<double>(mout.size(), mout.data());

    } else {
        auto filt = scran_qc::compute_crispr_qc_filters(ncells, buffers, opt);
        output[0] = filt.get_max_value();
    }

    return output;
}

pybind11::array filter_crispr_qc_metrics(
    pybind11::tuple filters,
    pybind11::tuple metrics,
    std::optional<UnsignedArray> maybe_block
) {
    ConvertedCrisprQcMetrics all_metrics(metrics);
    auto mbuffers = all_metrics.to_buffer();
    const auto ncells = all_metrics.size();

    if (filters.size() != 1) {
        throw std::runtime_error("'filters' should have the same format as the output of 'suggest_crispr_qc_thresholds'");
    }

    auto keep = sanisizer::create<pybind11::array_t<bool> >(ncells);
    bool* kptr = static_cast<bool*>(keep.request().ptr);

    if (maybe_block.has_value()) {
        const auto& block = *maybe_block;
        if (!sanisizer::is_equal(block.size(), ncells)) {
            throw std::runtime_error("'block' must be the same length as the number of cells");
        }
        auto bptr = get_numpy_array_data<std::uint32_t>(block);

        scran_qc::CrisprQcBlockedFilters filt;
        const auto max_value = filters[0].template cast<DoubleArray>();
        const auto nblocks = max_value.size();
        copy_filters_blocked(nblocks, max_value, filt.get_max_value());

        filt.filter(ncells, mbuffers, bptr, kptr);

    } else {
        scran_qc::CrisprQcFilters filt;
        filt.get_max_value() = filters[0].template cast<double>();
        filt.filter(ncells, mbuffers, kptr);
    }

    return keep;
}

void init_crispr_quality_control(pybind11::module& m) {
    m.def("compute_crispr_qc_metrics", &compute_crispr_qc_metrics);
    m.def("suggest_crispr_qc_thresholds", &suggest_crispr_qc_thresholds);
    m.def("filter_crispr_qc_metrics", &filter_crispr_qc_metrics);
}
