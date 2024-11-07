
// #define USE_NANOBIND 1 // To wrap with NANOBIND

#include <type_traits> // result_of
#include <cmath> // constants
#include <iostream>
#include <stdio.h>
#include <functional>

#include "lanczos.h"
#include "eigen_operators.h"
#include "pylinop.h"
// #include "spectral_functions.h"

// template< std::floating_point F, typename WrapperType > 
// auto matmat(const MatrixFunction< F, WrapperType >& M, const py_array< F >& X) -> py_array< F >{
//   if (size_t(X.ndim() == 1)){
//     if (size_t(M.shape().second) != size_t(X.size())){ throw std::invalid_argument("Input dimension mismatch; vector inputs must match shape of the operator."); }
//     auto Y = static_cast< DenseMatrix< F > >(DenseMatrix< F >::Zero(M.shape().second, 1));
//     M.matmat(X.data(), Y.data(), 1);
//     return py::cast(Y);
//   } else if (X.ndim() == 2){
//     if (size_t(M.shape().second) != size_t(X.shape(0))){ throw std::invalid_argument("Input dimension mismatch; vector inputs must match shape of the operator."); }
//     const auto k = X.shape(1);
//     auto Y = static_cast< DenseMatrix< F > >(DenseMatrix< F >::Zero(M.shape().second, k));
//     M.matmat(X.data(), Y.data(), k);
//     return py::cast(Y);
//   } else {
//     throw std::invalid_argument("Input dimension mismatch; input must be 1 or 2-dimensional.");
//   }
// }

#ifdef USE_NANOBIND

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/eigen/sparse.h>
namespace nb = nanobind;

template< typename F >
using nd_array = nb::ndarray< F >;

template< std::floating_point F, class Matrix, LinearOperator Wrapper >
void _lanczos_wrapper(nb::module_& m){
  m.def("lanczos", []( // keep wrap pass by value!
    const Matrix& A, 
    nd_array< F > v, const int lanczos_degree, const F lanczos_rtol, const int orth,
    nd_array< F >& alpha, nd_array< F >& beta, nd_array< F >& Q 
  ){ 
    const auto op = Wrapper(A);
    const size_t ncv = static_cast< size_t >(Q.shape(1));
    lanczos_recurrence(
      op, v.data(), lanczos_degree, lanczos_rtol, orth, 
      alpha.data(), beta.data(), Q.data(), ncv
    );
  });
} 

NB_MODULE(_lanczos, m) {

  _lanczos_wrapper< float, DenseMatrix< float >, DenseEigenLinearOperator< float > >(m);
  _lanczos_wrapper< double, DenseMatrix< double >, DenseEigenLinearOperator< double > >(m);

  _lanczos_wrapper< float, Eigen::SparseMatrix< float >, SparseEigenLinearOperator< float, false > >(m);
  _lanczos_wrapper< double, Eigen::SparseMatrix< double >, SparseEigenLinearOperator< double, false > >(m);
  
  // _lanczos_wrapper< float, py::object, PyLinearOperator< float > >(m);
  // _lanczos_wrapper< double, py::object, PyLinearOperator< double > >(m);
};

#else 

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
namespace py = pybind11;

// NOTE: all matrices should be cast to Fortran ordering for compatibility with Eigen
template< typename F >
using py_array = py::array_t< F, py::array::f_style | py::array::forcecast >;

// Template function for generating module definitions for a given Operator / precision 
template< std::floating_point F, class Matrix, LinearOperator Wrapper >
void _lanczos_wrapper(py::module& m){
  m.def("lanczos", []( // keep wrap pass by value!
    const Matrix& A, 
    py_array< F > v, const int lanczos_degree, const F lanczos_rtol, const int orth,
    py_array< F >& alpha, py_array< F >& beta, py_array< F >& Q 
  ){ 
    const auto op = Wrapper(A);
    const size_t ncv = static_cast< size_t >(Q.shape(1));
    lanczos_recurrence(
      op, v.mutable_data(), lanczos_degree, lanczos_rtol, orth, 
      alpha.mutable_data(), beta.mutable_data(), Q.mutable_data(), ncv
    );
  });
} 

PYBIND11_MODULE(_lanczos, m) {

  _lanczos_wrapper< float, DenseMatrix< float >, DenseEigenLinearOperator< float > >(m);
  _lanczos_wrapper< double, DenseMatrix< double >, DenseEigenLinearOperator< double > >(m);

  _lanczos_wrapper< float, Eigen::SparseMatrix< float >, SparseEigenLinearOperator< float, false > >(m);
  _lanczos_wrapper< double, Eigen::SparseMatrix< double >, SparseEigenLinearOperator< double, false > >(m);
  
  _lanczos_wrapper< float, py::object, PyLinearOperator< float > >(m);
  _lanczos_wrapper< double, py::object, PyLinearOperator< double > >(m);
};

#endif