#include <pybind11/pybind11.h>
// #include <binders/pb11_trace_bind.h>
// #include "pylinops.h"         //
#include "eigen_operators.h"  // eigen_< mat >_wrappers
#include "_random_generator/vector_generator.h"
#include "_lanczos/lanczos.h"
namespace py = pybind11;


// auto rbg = ThreadedRNG64< RNE >(num_threads, seed);
// auto* data = out.mutable_data();
// auto array_sz = static_cast< size_t >(out.size());
// generate_array< 0, F >(rbg, data, array_sz, num_threads); 
// void stochastic_lanczos_quadrature(){
//   lanczos_quadrature();
//   return;
// }


// Turns out using py::call_guard<py::gil_scoped_release>() just causes everthing to crash immediately
PYBIND11_MODULE(_trace, m) {
  m.doc() = "trace estimator module";

  // // Sparse exports
  // _trace_wrapper< false, float, Eigen::SparseMatrix< float > >(m, eigen_sparse_wrapper< float >); 
  // _trace_wrapper< false, double, Eigen::SparseMatrix< double > >(m, eigen_sparse_wrapper< double >); 
  
  // // Dense exports
  // _trace_wrapper< false, float, Eigen::MatrixXf >(m, eigen_dense_wrapper< float >);
  // _trace_wrapper< false, double, Eigen::MatrixXd >(m, eigen_dense_wrapper< double >);

  // // LinearOperator exports
  // _trace_wrapper< false, float, py::object >(m, linearoperator_wrapper< float >);
  // _trace_wrapper< false, double, py::object >(m, linearoperator_wrapper< double >);
};