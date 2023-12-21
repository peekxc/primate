#include <type_traits> // result_of
#include <cmath> // constants
#include <iostream>
#include <stdio.h>
#include <functional>

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

#include "_random_generator/vector_generator.h"
#include "_random_generator/threadedrng64.h"
#include "_lanczos/lanczos.h"
#include "_trace/hutch.h"
#include "eigen_operators.h"
#include "pylinop.h"
#include "spectral_functions.h"

namespace py = pybind11;

// NOTE: all matrices should be cast to Fortran ordering for compatibility with Eigen
template< typename F >
using py_array = py::array_t< F, py::array::f_style | py::array::forcecast >;

// #define LANCZOS_PARAMS \
//   py_array< float >& v, \
//   const int num_steps, const float lanczos_tol, const int orthogonalize, \
//   py_array< float >& alpha, py_array< float >& beta, py::array_t< float, py::array::f_style >& Q 

// // These arugments must match the names of their corresponding parameters 
// #define LANCZOS_ARGS \
//   v, num_steps, lanczos_tol, orthogonalize\
//   orthogonalize, lanczos_degree, lanczos_tol, min_num_samples, max_num_samples, \
//   alpha, beta, Q

// #define TRACE_PYBIND_PARAMS \
//   py::arg("parameters"), py::arg("num_inqueries"), \
//   py::arg("orthogonalize"), py::arg("lanczos_degree"), py::arg("lanczos_tol"), py::arg("min_num_samples"), py::arg("max_num_samples"), \
//   py::arg("error_atol"), py::arg("error_rtol"), py::arg("confidence"), py::arg("outlier"), \
//   py::arg("distr"), py::arg("engine_id"), py::arg("seed"), \
//   py::arg("num_threads"), \
//   py::arg("trace").noconvert(), py::arg("error").noconvert(), py::arg("samples").noconvert(), \
//   py::arg("processed_samples_indices").noconvert(), py::arg("num_samples_used").noconvert(), py::arg("num_outliers").noconvert(), py::arg("converged").noconvert(), py::arg("alg_wall_time")

template< std::floating_point F, typename WrapperType > 
auto matmat(const MatrixFunction< F, WrapperType >& M, const py_array< F >& X) -> py_array< F >{
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
}

// Template function for generating module definitions for a given Operator / precision 
template< std::floating_point F, class Matrix, LinearOperator Wrapper >
void _lanczos_wrapper(py::module& m){
  // using ArrayF = Eigen::Array< F, Dynamic, 1 >;
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
  m.def("quadrature", [](py_array< F > a, py_array< F > b, const int k) -> py_array< F > {
    auto output = DenseMatrix< F >(k, 2); // [nodes, weights]
    auto solver = Eigen::SelfAdjointEigenSolver< DenseMatrix< F > >(k);
    lanczos_quadrature(a.data(), b.data(), k, solver, output.col(0).data(), output.col(1).data());
    return py::cast(output); 
  });
  // m.def("function_approx", [](
  //   const Matrix& A, 
  //   py_array< F > v, 
  //   const int lanczos_degree, const F lanczos_rtol, const int orth,
  //   const py::kwargs& kwargs
  // ){
  //   const auto op = Wrapper(A);
  //   const auto sf = param_spectral_func< F >(kwargs);
  //   F* v_inp = v.mutable_data();
  //   auto y_out = static_cast< ArrayF >(ArrayF::Zero(op.shape().second));
  //   matrix_approx(op, sf, v_inp, lanczos_degree, lanczos_rtol, orth, y_out.data());
  //   return py::cast(y_out);
  // });
  m.def("stochastic_quadrature", [](
    const Matrix& A, 
    const int nv, const int dist, const int engine_id, const int seed,
    const int lanczos_degree, const F lanczos_rtol, const int orth, const int ncv,
    const int num_threads
  ) -> py_array< F > {
    const auto op = Wrapper(A);
    auto rbg = ThreadedRNG64(num_threads, engine_id, seed);
    auto quad_nw = static_cast< DenseMatrix< F > >(DenseMatrix< F >::Zero(lanczos_degree * nv, 2));
    sl_quadrature(op, rbg, nv, dist, engine_id, seed, lanczos_degree, lanczos_rtol, orth, ncv, num_threads, quad_nw.data());
    return py::cast(quad_nw);
  });
} 

PYBIND11_MODULE(_lanczos, m) {

  _lanczos_wrapper< float, DenseMatrix< float >, DenseEigenLinearOperator< float > >(m);
  _lanczos_wrapper< double, DenseMatrix< double >, DenseEigenLinearOperator< double > >(m);

  _lanczos_wrapper< float, Eigen::SparseMatrix< float >, SparseEigenLinearOperator< float > >(m);
  _lanczos_wrapper< double, Eigen::SparseMatrix< double >, SparseEigenLinearOperator< double > >(m);
  
  _lanczos_wrapper< float, py::object, PyLinearOperator< float > >(m);
  _lanczos_wrapper< double, py::object, PyLinearOperator< double > >(m);
};



// Given an input vector 'z', yields the vector y = Qz where T(alpha, beta) = Q^T A Q is the tridiagonal matrix spanning K(A, q)
// template< std::floating_point F, LinearOperator Matrix >
// void lanczos_action(  
//   const Matrix& A,            // Symmetric linear operator 
//   F* q,                       // vector to expand the Krylov space K(A, q)
//   const int k,                // Dimension of the Krylov subspace to capture
//   const int ncv,              // Number of Lanczos vectors
//   const F* alpha,             // Diagonal elements of T of size A.shape[1]+1
//   const F* beta,              // Subdiagonal elements of T of size A.shape[1]+1
//   const F* z,                 // Input vector to multiply Q by 
//   F* V,                       // Storage for the Lanczos vectors (column-major)
//   F* y                        // Output vector y = Q z
// ){
//   using ColVectorF = Eigen::Matrix< F, Dynamic, 1 >;

//   // Constants `
//   const auto A_shape = A.shape();
//   const size_t n = A_shape.first;
//   const size_t m = A_shape.second;

//   // Allocation / views
//   Eigen::Map< DenseMatrix< F > > Q(V, n, ncv);  // Lanczos vectors 
//   Eigen::Map< ColVectorF > v(q, m, 1);          // map initial vector (no-op)
//   Eigen::Map< ColVectorF > w(z, k, 1);          // map input vector to multiply (no-op)
//   Eigen::Map< ColVectorF > x(y, n, 1);          // map output vector to store (no-op)
//   Q.col(0) = v.normalized();                    // load normalized v0 into Q  

//   // Set the output to zero to begin
//   x.setZero(m);

//   // In the following, beta[j] means beta[j-1] in the Demmel text
//   std::array< int, 3 > pos = { static_cast< int >(ncv - 1), 0, 1 };
//   for (int j = 0; j < k; ++j){
//     auto [p,c,n] = pos;                  // previous, current, next
//     x += w[j] * Q.col(c);                // x = Q_k w <=> x <- w[0] * q_0 + w[1] * q_1 + ... + w[k-1] * q_{k-1}

//     // Apply the three-term recurrence
//     A.matvec(Q.col(c).data(), v.data()); // v = A q_c
//     Q.col(n) = v - beta[j] * Q.col(p);   // q_n = v - b q_p
//     Q.col(n) -= alpha[j] * Q.col(c);     // subtract projected components

//     // Re-orthogonalize q_n against previous ncv-1 lanczos vectors
//     if (ncv > 2) {
//       auto qn = Eigen::Ref< ColVector< F > >(Q.col(n));          
//       const auto Q_ref = Eigen::Ref< const DenseMatrix< F > >(Q); 
//       orth_vector(qn, Q_ref, c, ncv-1, true);
//     }
//     Q.col(n) /= beta[j+1]; // normalize such that Q stays orthonormal
    
//     // Cyclic left-rotate to update the working column indices
//     std::rotate(pos.begin(), pos.begin() + 1, pos.end());
//     pos[2] = mod(j+2, ncv);
//   }
// }

//   // Number of Lanczos vectors to keep in memory
//   const size_t ncv = 
//     (orthogonalize == 0 || orthogonalize == 1) ? 2 :         // Minimum orthogonalization
//     (orthogonalize < 0 || orthogonalize > k) ? size_t(k) :   // Full reorthogonalization
//     static_cast< size_t >(orthogonalize);                    // Partial orthogonalization (0 < orthogonalize < m)