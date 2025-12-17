#ifndef UTILS_H
#define UTILS_H

#include <type_traits>
#include <stdexcept>
#include <cstdint>

#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"
#include "sanisizer/sanisizer.hpp"

template<typename Input_>
using I = std::remove_reference_t<std::remove_cv_t<Input_> >;

typedef pybind11::array_t<double, pybind11::array::f_style | pybind11::array::forcecast> DoubleArray;

typedef pybind11::array_t<std::uint32_t, pybind11::array::f_style | pybind11::array::forcecast> UnsignedArray;

template<typename Output_, typename Size_, typename Pointer_>
pybind11::array_t<Output_> create_numpy_vector(Size_ size, Pointer_ ptr) {
    typedef pybind11::array_t<Output_> Vector;
    return pybind11::array_t<Output_>(
        sanisizer::cast<I<decltype(std::declval<Vector>().size())> >(size),
        ptr
    );
}

template<typename Output_, typename Rows_, typename Cols_>
pybind11::array_t<Output_, pybind11::array::f_style> create_numpy_matrix(Rows_ rows, Cols_ cols) {
    typedef pybind11::array_t<Output_, pybind11::array::f_style> Matrix;
    typedef I<decltype(std::declval<Matrix>().size())> Size;
    return Matrix({
        sanisizer::cast<Size>(rows),
        sanisizer::cast<Size>(cols)
    });
}

template<typename Expected_>
const Expected_* get_numpy_array_data(const pybind11::array& x) {
    return static_cast<const Expected_*>(x.request().ptr);
}

template<typename Expected_>
const Expected_* check_contiguous_numpy_array(const pybind11::array& x) {
    auto flag = x.flags();
    if (!(flag & pybind11::array::c_style) || !(flag & pybind11::array::f_style)) {
        throw std::runtime_error("NumPy array contents should be contiguous");
    }
    return get_numpy_array_data<Expected_>(x);
}

template<typename Expected_>
const Expected_* check_numpy_array(const pybind11::array& x) {
    if (!x.dtype().is(pybind11::dtype::of<Expected_>())) {
        throw std::runtime_error("unexpected dtype for NumPy array");
    }
    return check_contiguous_numpy_array<Expected_>(x);
}

#endif
