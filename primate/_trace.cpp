#include <pybind11/pybind11.h>
#include <binders/pb11_trace_bind.h>
#include "pylinops.h"
#include "eigen_operators.h"

namespace py = pybind11;

template< std::floating_point F >
auto eigen_sparse_wrapper(const Eigen::SparseMatrix< F >* A){
  return SparseEigenLinearOperator< F >(*A);
}

// TODO: Support Adjoint and Affine Operators out of the box
template< std::floating_point F >
auto eigen_sparse_affine_wrapper(const Eigen::SparseMatrix< F >* A){
  auto B = Eigen::SparseMatrix< F >(A->rows(), A->cols());
  B.setIdentity();
  return SparseEigenAffineOperator< F >(*A, B);
}

template< std::floating_point F >
auto eigen_dense_wrapper(const Eigen::Matrix< F, Eigen::Dynamic, Eigen::Dynamic >* A){
  return DenseEigenLinearOperator< F >(*A);
}

template< std::floating_point F >
auto linearoperator_wrapper(const py::object* A){
  return PyLinearOperator< F >(*A);
}

// Turns out using py::call_guard<py::gil_scoped_release>() just causes everthing to crash immediately
PYBIND11_MODULE(_trace, m) {
  m.doc() = "trace estimator module";

  // Sparse exports
  _trace_wrapper< false, float, Eigen::SparseMatrix< float > >(m, eigen_sparse_wrapper< float >); 
  _trace_wrapper< false, double, Eigen::SparseMatrix< double > >(m, eigen_sparse_wrapper< double >); 
  
  // Dense exports
  _trace_wrapper< false, float, Eigen::MatrixXf >(m, eigen_dense_wrapper< float >);
  _trace_wrapper< false, double, Eigen::MatrixXd >(m, eigen_dense_wrapper< double >);

  // LinearOperator exports
  _trace_wrapper< false, float, py::object >(m, linearoperator_wrapper< float >);
  _trace_wrapper< false, double, py::object >(m, linearoperator_wrapper< double >);
};