#include <type_traits> // result_of
#include <cmath> // constants
// #include <iostream>
#include <functional>

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

#include "_operators/linear_operator.h"
#include "_random_generator/vector_generator.h"
#include "_lanczos/lanczos.h"
#include "eigen_core.h"
#include "spectral_functions.h"
#include "eigen_operators.h"
#include "pylinop.h"

namespace py = pybind11;

// NOTE: all matrices should be cast to Fortran ordering for compatibility with Eigen
template< typename F >
using py_array = py::array_t< F, py::array::f_style | py::array::forcecast >;


template< std::floating_point F, LinearOperator Matrix >
auto py_matvec(const Matrix& A, const py_array< F >& x) -> py_array< F > {
  using ArrayF = Array< F >;
  using VectorF = Vector< F >;
  if (size_t(A.shape().second) != size_t(x.size())){
    throw std::invalid_argument("Input dimension mismatch; vector inputs must match shape of the operator.");
  }
  auto output = static_cast< ArrayF >(VectorF::Zero(A.shape().first));
  A.matvec(x.data(), output.data());
  return py::cast(output);
}

template< std::floating_point F, LinearOperator Matrix >
void py_matvec_inplace(const Matrix& A, const py_array< F >& x, py_array< F >& y) {
  if (size_t(A.shape().second) != size_t(x.size()) || size_t(A.shape().first) != size_t(y.size())){
    throw std::invalid_argument("Input/output dimension mismatch; vector inputs must match shape of the operator.");
  }
  A.matvec(x.data(), y.mutable_data());
}

template< std::floating_point F, LinearOperator Matrix > 
auto py_matmat(const Matrix& M, const py_array< F >& X) -> py_array< F >{
  if constexpr (SupportsMatrixMult< Matrix >){
    if (size_t(X.ndim() == 1)){
      if (size_t(M.shape().second) != size_t(X.size())){ throw std::invalid_argument("Input dimension mismatch; vector inputs must match shape of the operator."); }
      auto Y = static_cast< DenseMatrix< F > >(DenseMatrix< F >::Zero(M.shape().second, 1));
      M.matmat(X.data(), Y.data(), 1);
      return py::cast(Y);
    } else if (X.ndim() == 2){
      if (size_t(M.shape().second) != size_t(X.shape(0))){ throw std::invalid_argument("Input dimension mismatch; vector inputs must match shape of the operator."); }
      const auto k = X.shape(1);
      auto Y = static_cast< DenseMatrix< F > >(DenseMatrix< F >::Zero(M.shape().second, k));
      M.matmat(X.data(), Y.data(), k);
      return py::cast(Y);
    } else {
      throw std::invalid_argument("Input dimension mismatch; input must be 1 or 2-dimensional.");
    }
  } else {
    return X;
  }
}

// template< std::floating_point F, typename WrapperFunc >
// requires std::invocable< WrapperFunc, const Matrix* >
// template< std::floating_point F, class Op, LinearOperator Matrix >
// void _operators_wrapper(py::module& m, std::string prefix){
//   // using WrapperType = decltype(wrap(static_cast< const Matrix* >(nullptr)));
//   prefix += std::string("Matrix");
//   const char* cls_name = prefix.c_str();
//   py::class_< Matrix >(m, cls_name)
//     .def(py::init< const Op >())
//     .def_property_readonly("shape", &Matrix::shape)
//     .def_property_readonly("dtype", [](const Matrix& M) {
//       return pybind11::dtype(pybind11::format_descriptor< F >::format());
//     })
//     .def("matvec", &py_matvec< F, Matrix >)
//     .def("matvec", &py_matvec_inplace< F, Matrix >)
//     .def("matmat", &py_matmat< F, Matrix >)
//     .def("__matmul__", &py_matmat< F, Matrix >) // see: https://peps.python.org/pep-0465/
//     ;
// }

// Template function for generating module definitions for a given Operator / precision 
// template< std::floating_point F, LinearOperator Matrix >
// void _matrix_function_wrapper(py::module& m, std::string prefix){
//   prefix += std::string("_MatrixFunction");
//   const char* cls_name = prefix.c_str();
//   using OP_t = MatrixFunction< F, Matrix >;
//   py::class_< OP_t >(m, cls_name)
//     .def(py::init([](const Matrix* A, const int deg, const F rtol, const int orth, const int ncv, const py::kwargs& kwargs) {
//       const auto sf = param_spectral_func< F >(kwargs);
//       return std::unique_ptr< OP_t >(new OP_t(A, sf, deg, rtol, orth, ncv));
//     }))
//     .def_property_readonly("shape", &OP_t::shape)
//     .def_property_readonly("dtype", [](const OP_t& M) -> py::dtype {
//       auto dtype = pybind11::dtype(pybind11::format_descriptor< F >::format());
//       return dtype; 
//     })
//     .def_readonly("deg", &OP_t::deg)
//     .def_readwrite("rtol", &OP_t::rtol)
//     .def_readwrite("orth", &OP_t::orth)
//     .def("matvec", &py_matvec< F, OP_t >)
//     .def("matvec", &py_matvec_inplace< F, OP_t >)
//     .def("matmat", &py_matmat< F, OP_t >)
//     .def("__matmul__", &py_matmat< F, OP_t >)
//     ; 
// }


template< std::floating_point F, class Matrix, LinearOperator Wrapper >
// requires std::invocable< Wrapper, const Matrix* >
void _matrix_function_wrapper(py::module& m, std::string prefix){
  prefix += std::string("_MatrixFunction");
  const char* cls_name = prefix.c_str();
  
  // using Wrapped_t = decltype(wrap(static_cast< const Matrix* >(nullptr)));
  using OP_t = MatrixFunction< F, Wrapper >;
  py::class_< OP_t >(m, cls_name)
    .def(py::init([](const Matrix& A, const int deg, const F rtol, const int orth, const int ncv, const py::kwargs& kwargs) {
      const auto sf = param_spectral_func< F >(kwargs);
      // const auto op = Wrapper(*A); // we want a new instance here!
      // std::cout << op.shape().first << ", ";
      const Wrapper* op = new Wrapper(A); // todo: add custom destructor to delete this
      return std::make_unique< OP_t >(OP_t(*op, sf, deg, rtol, orth, ncv));
    }))
    .def_property_readonly("shape", &OP_t::shape)
    .def_property_readonly("dtype", [](const OP_t& M) -> py::dtype {
      auto dtype = pybind11::dtype(pybind11::format_descriptor< F >::format());
      return dtype; 
    })
    .def_readonly("deg", &OP_t::deg)
    .def_readwrite("rtol", &OP_t::rtol)
    .def_readwrite("orth", &OP_t::orth)
    .def("matvec", &py_matvec< F, OP_t >)
    .def("matvec", &py_matvec_inplace< F, OP_t >)
    .def("matmat", &py_matmat< F, OP_t >)
    .def("__matmul__", &py_matmat< F, OP_t >)
    ; 
}

PYBIND11_MODULE(_operators, m) {
  m.doc() = "operators module";
  _matrix_function_wrapper< float, DenseMatrix< float >, DenseEigenLinearOperator< float > >(m, "DenseF");
  _matrix_function_wrapper< double, DenseMatrix< double >, DenseEigenLinearOperator< double > >(m, "DenseD");

  _matrix_function_wrapper< float, Eigen::SparseMatrix< float >, SparseEigenLinearOperator< float > >(m, "SparseF");
  _matrix_function_wrapper< double, Eigen::SparseMatrix< double >, SparseEigenLinearOperator< double > >(m, "SparseD");
  
  _matrix_function_wrapper< float, py::object, PyLinearOperator< float > >(m, "GenericF");
  _matrix_function_wrapper< double, py::object, PyLinearOperator< double > >(m, "GenericD");
  // _operators_wrapper< float, DenseMatrix< float >, DenseEigenLinearOperator< float > >(m, "Dense");
  // _matrix_function_wrapper< float, DenseEigenLinearOperator< float > >(m, "Dense");

  // _operators_wrapper< float, Eigen::SparseMatrix< float >, SparseEigenLinearOperator< float > >(m, "Sparse");
  // _matrix_function_wrapper< float, SparseEigenLinearOperator< float > >(m, "Sparse");

  // Handle generic py::objects separately
  // PyLinearOperator
  // _operators_wrapper< float, py::object, PyLinearOperator< float > >(m, "LinOp");
  // _matrix_function_wrapper< float, PyLinearOperator< float > >(m, "LinOp");
  
  // py::class_< MatrixFunction< F, Matrix > >(m, cls_name)
  //   .def(py::init([](const Matrix* A, const int deg, const F rtol, const int orth, const int ncv, const py::kwargs& kwargs) {
  //     const auto sf = param_spectral_func< F >(kwargs);
  //     return std::unique_ptr< OP_t >(new OP_t(A, sf, deg, rtol, orth, ncv));
  //   }))
  //   .def_property_readonly("shape", &OP_t::shape)
  //   .def_property_readonly("dtype", [](const OP_t& M) -> py::dtype {
  //     auto dtype = pybind11::dtype(pybind11::format_descriptor< F >::format());
  //     return dtype; 
  //   })
  //   .def_readonly("deg", &OP_t::deg)
  //   .def_readwrite("rtol", &OP_t::rtol)
  //   .def_readwrite("orth", &OP_t::orth)
  //   .def("matvec", &py_matvec< F, OP_t >)
  //   .def("matvec", &py_matvec_inplace< F, OP_t >)
  //   .def("matmat", &py_matmat< F, OP_t >)
  //   .def("__matmul__", &py_matmat< F, OP_t >)
  //   ; 
};

// Extra to not lose: 
// template< std::floating_point F, class Matrix, typename WrapperFunc >
// requires std::invocable< WrapperFunc, const Matrix* >
// using WrapperType = decltype(wrap(static_cast< const Matrix* >(nullptr)));