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
    auto output = create_numpy_matrix(x.rows(), x.cols());
    std::copy_n(x.data(), output.size(), static_cast<double*>(output.request().ptr));
    return output;
}

static pybind11::array transfer(const Eigen::VectorXd& x) {
    return create_numpy_array<double>(x.size(), x.data());
}

pybind11::tuple run_pca(
    uintptr_t x,
    int number,
    std::optional<pybind11::array> maybe_block, 
    std::string block_weight_policy,
    const pybind11::tuple& variable_block_weight,
    bool components_from_residuals,
    bool scale,
    std::optional<pybind11::array> subset,
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

    pybind11::tuple output;
    const auto deposit_outputs = [&](const auto& out) -> pybind11::tuple {
        pybind11::tuple output(6);
        output[0] = transfer(out.components);
        output[1] = transfer(out.rotation);
        output[2] = transfer(out.variance_explained);
        output[3] = out.total_variance;
        output[4] = transfer(out.center);
        output[5] = transfer(out.scale);
        return output;
    };

    if (ptr) {
        if (!sanisizer::is_equal(block_info.size(), mat->ptr->ncol())) {
            throw std::runtime_error("'block' must be the same length as the number of cells");
        }

        const auto fill_block_options = [&](auto& opt) -> void {
            fill_common_options(opt);
            opt.block_weight_policy = parse_block_weight_policy(block_weight_policy);
            opt.variable_block_weight_parameters = parse_variable_block_weight(variable_block_weight);
            opt.components_from_residuals = components_from_residuals;
        };

        if (!subset.has_value()) {
            scran_pca::BlockedPcaOptions opt;
            fill_block_options(opt);
            auto res = scran_pca::blocked_pca(*(mat->ptr), ptr, opt);
            output = deposit_outputs(res);

        } else {
            scran_pca::SubsetPcaBlockedOptions opt;
            fill_block_options(opt);
            const auto subptr = check_numpy_array<std::uint32_t>(*subset);
            const auto subsize = subset->size();
            auto res = scran_pca::subset_pca_blocked(*(mat->ptr), tatami::ArrayView<std::uint32_t>(subptr, subsize), ptr, opt);
            output = deposit_outputs(res);
        }

    } else {
        if (!subset.has_value()) {
            scran_pca::SimplePcaOptions opt;
            fill_common_options(opt);
            auto res = scran_pca::simple_pca(*(mat->ptr), opt);
            output = deposit_outputs(res);

        } else {
            scran_pca::SubsetPcaOptions opt;
            fill_common_options(opt);
            const auto subptr = check_numpy_array<std::uint32_t>(*subset);
            const auto subsize = subset->size();
            auto res = scran_pca::subset_pca(*(mat->ptr), tatami::ArrayView<std::uint32_t>(subptr, subsize), opt);
            output = deposit_outputs(res);
        }
    }

    return output;
}

void init_run_pca(pybind11::module& m) {
    m.def("run_pca", &run_pca);
}
