#include <vector>
#include <algorithm>
#include <string>
#include <stdexcept>
#include <cstdint>

#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"
#include "pybind11/stl.h"
#include "scran_pca/scran_pca.hpp"
#include "Eigen/Dense"
#include "tatami/tatami.hpp"

#include "mattress.h"
#include "utils.h"
#include "block.h"

static pybind11::array transfer(const Eigen::MatrixXd& x) {
    auto output = create_numpy_matrix<double>(x.rows(), x.cols());
    std::copy_n(x.data(), output.size(), static_cast<double*>(output.request().ptr));
    return output;
}

static pybind11::array transfer(const Eigen::VectorXd& x) {
    return create_numpy_vector<double>(x.size(), x.data());
}

pybind11::dict run_pca(
    std::uintptr_t x,
    int number,
    std::optional<UnsignedArray> maybe_block, 
    std::string block_weight_policy,
    const pybind11::tuple& variable_block_weight,
    bool components_from_residuals,
    bool scale,
    std::optional<UnsignedArray> subset,
    bool realized,
    int irlba_work,
    int irlba_iterations,
    int irlba_seed,
    int num_threads
) {
    const auto& mat = mattress::cast(x)->ptr;

    irlba::Options iopt;
    iopt.extra_work = irlba_work;
    iopt.max_iterations = irlba_iterations;
    iopt.seed = irlba_seed;
    iopt.cap_number = true;

    const auto fill_common_options = [&](auto& opt) -> void {
        opt.number = number;
        opt.scale = scale;
        opt.realize_matrix = realized;
        opt.irlba_options = iopt;
        opt.num_threads = num_threads;
    };

    pybind11::dict output;
    const auto deposit_outputs = [&](const auto& out) -> pybind11::dict {
        pybind11::dict output;
        output["components"] = transfer(out.components);
        output["rotation"] = transfer(out.rotation);
        output["variance_explained"] = transfer(out.variance_explained);
        output["total_variance"] = out.total_variance;
        output["center"] = transfer(out.center);
        output["scale"] = transfer(out.scale);
        return output;
    };

    if (maybe_block.has_value()) {
        if (!sanisizer::is_equal(maybe_block->size(), mat->ncol())) {
            throw std::runtime_error("'block' must be the same length as the number of cells");
        }
        const auto ptr = get_numpy_array_data<std::uint32_t>(*maybe_block);

        const auto fill_block_options = [&](auto& opt) -> void {
            fill_common_options(opt);
            opt.block_weight_policy = parse_block_weight_policy(block_weight_policy);
            opt.variable_block_weight_parameters = parse_variable_block_weight(variable_block_weight);
            opt.components_from_residuals = components_from_residuals;
        };

        if (!subset.has_value()) {
            scran_pca::BlockedPcaOptions opt;
            fill_block_options(opt);
            auto res = scran_pca::blocked_pca(*mat, ptr, opt);
            output = deposit_outputs(res);

        } else {
            scran_pca::SubsetPcaBlockedOptions opt;
            fill_block_options(opt);
            const auto subptr = get_numpy_array_data<std::uint32_t>(*subset);
            const auto subsize = subset->size();
            auto res = scran_pca::subset_pca_blocked(*mat, tatami::ArrayView<std::uint32_t>(subptr, subsize), ptr, opt);
            output = deposit_outputs(res);
        }

    } else {
        if (!subset.has_value()) {
            scran_pca::SimplePcaOptions opt;
            fill_common_options(opt);
            auto res = scran_pca::simple_pca(*mat, opt);
            output = deposit_outputs(res);

        } else {
            scran_pca::SubsetPcaOptions opt;
            fill_common_options(opt);
            const auto subptr = get_numpy_array_data<std::uint32_t>(*subset);
            const auto subsize = subset->size();
            auto res = scran_pca::subset_pca(*mat, tatami::ArrayView<std::uint32_t>(subptr, subsize), opt);
            output = deposit_outputs(res);
        }
    }

    return output;
}

void init_run_pca(pybind11::module& m) {
    m.def("run_pca", &run_pca);
}
