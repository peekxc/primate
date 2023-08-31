
#ifndef _LINEAR_OPERATOR_H
#define _LINEAR_OPERATOR_H

#include <type_traits>
#include <concepts>
#include <algorithm>
#include <functional>

using std::pair; 
using std::vector; 

template <typename T>
struct TypeString;

template <>
struct TypeString< float > {
  static constexpr const char* value = "float32";
};

template < typename T, typename F = typename T::value_type >
concept LinearOperator = requires(T op, const F* input, F* output) {
  { op.matvec(input, output) };
  { op.shape() } -> std::convertible_to< std::pair< size_t, size_t > >;
};

template < typename T, typename F = typename T::value_type >
concept AdjointOperator = requires(T op, const F* input, F* output) {
  { op.rmatvec(input, output) };
} && LinearOperator< T, F >;


// TODO: aslinearoperator on a sparse matrix and buffer_info by default 
// Should match: https://github.com/scipy/scipy/blob/v1.11.2/scipy/sparse/linalg/_interface.py
// via LO_facade : https://vector-of-bool.github.io/2020/06/13/cpp20-iter-facade.html
// Sparse and dense both yield: ['A', 'H', 'T', '_MatrixLinearOperator__adj', '__add__', '__call__', '__eq__', '__ge__', '__gt__', '__hash__', '__le__', '__lt__', '__matmul__', '__mul__', '__ne__', '__neg__', '__new__', '__pow__', '__reduce__', '__reduce_ex__', '__rmatmul__', '__rmul__', '__sizeof__', '_adjoint', '_init_dtype', '_matmat', '_matvec', '_rmatmat', '_rmatvec', '_transpose', 'adjoint', 'args', 'dot', 'dtype', 'matmat', 'matvec', 'ndim', 'rmatmat', 'rmatvec', 'shape', 'transpose'
// From SciPy docs: "A subclass must implement either ``_matvec`` (or) ``_matmat``(or both), 
// and have attributes/properties ``shape`` (pair of integers) and ``dtype`` (may be None).
// (If only matvec is given, matmat is deduced)
// Optionally, a subclass may implement ``_rmatvec`` (or) ``_adjoint`` (or both), preferring ``_adjoint``
// #include <pybind11/pybind11.h>
// #include <pybind11/numpy.h>
// #include <pybind11/stl.h>

// namespace py = pybind11;
// template < typename T, typename F >
// concept PyLinearOperator = requires(T op, F* input) {
//   { op.matvec(input) } -> std::convertible_to< py::array_t< F > >; // TODO: swap to buffer protocol + memoryviews
//   { op.shape() } -> std::convertible_to< std::pair< size_t, size_t > >;
//   { op.dtype() } -> std::convertible_to< py::dtype >;
// };


#endif