#ifndef UTILS_H
#define UTILS_H

#include <type_traits>
#include <stdexcept>

#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"
#include "sanisizer/sanisizer.hpp"

template<typename Input_>
using I = std::remove_reference_t<std::remove_cv_t<Input_> >;

template<typename Output_, typename Size_, typename Pointer_>
pybind11::array_t<Output_> create_numpy_array(Size_ size, Pointer_ ptr) {
    typedef pybind11::array_t<Output_> Array;
    return pybind11::array_t<Output_>(
        sanisizer::cast<I<decltype(std::declval<Array>().size())> >(size),
        ptr
    );
}

template<typename Output_, typename Rows_, typename Cols_>
pybind11::array_t<Output_, pybind11::array::f_style> create_numpy_matrix(Rows_ rows, Cols_ cols) {
    typedef pybind11::array_t<Output_, pybind11::array::f_style> Matrix;
    typedef I<decltype(std::declval<Matrix>().size())> Size;
    return Matrix output({
        sanisizer::cast<Size>(rows),
        sanisizer::cast<Size>(cols)
    });
}

// As a general rule, we avoid using pybind11::array_t as function arguments,
// because pybind11 might auto-cast and create an allocation that we then
// create a view on; on function exit, our view would be a dangling reference
// once the allocation gets destructed. So, we accept instead a pybind11::array
// and make sure it has our desired type and contiguous storage.

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
