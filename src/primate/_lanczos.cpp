#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "_random_generator/threadedrng64.h"
#include "_lanczos/lanczos.h"
#include "eigen_operators.h"
#include <iostream>
#include <stdio.h>

namespace py = pybind11;

template< typename F >
using py_array = py::array_t< F, py::array::c_style | py::array::forcecast >;

#define LANCZOS_PARAMS \
  py_array< float >& v, \
  const int num_steps, const float lanczos_tol, const int orthogonalize, \
  py_array< float >& alpha, py_array< float >& beta, py::array_t< float, py::array::f_style >& Q 

// These arugments must match the names of their corresponding parameters 
#define LANCZOS_ARGS \
  v, num_steps, lanczos_tol, orthogonalize\
  orthogonalize, lanczos_degree, lanczos_tol, min_num_samples, max_num_samples, \
  alpha, beta, Q

// #define TRACE_PYBIND_PARAMS \
//   py::arg("parameters"), py::arg("num_inqueries"), \
//   py::arg("orthogonalize"), py::arg("lanczos_degree"), py::arg("lanczos_tol"), py::arg("min_num_samples"), py::arg("max_num_samples"), \
//   py::arg("error_atol"), py::arg("error_rtol"), py::arg("confidence"), py::arg("outlier"), \
//   py::arg("distr"), py::arg("engine_id"), py::arg("seed"), \
//   py::arg("num_threads"), \
//   py::arg("trace").noconvert(), py::arg("error").noconvert(), py::arg("samples").noconvert(), \
//   py::arg("processed_samples_indices").noconvert(), py::arg("num_samples_used").noconvert(), py::arg("num_outliers").noconvert(), py::arg("converged").noconvert(), py::arg("alg_wall_time")


// void slq(const Eigen::SparseMatrix< float >& mat, LANCZOS_PARAMS){

// }

// void slq_trace(){

// }

PYBIND11_MODULE(_lanczos, m) {
  m.def("lanczos", [](const Eigen::SparseMatrix< float >& mat, LANCZOS_PARAMS){
    const auto lo = SparseEigenLinearOperator(mat);
    const size_t ncv = static_cast< size_t >(Q.shape(1));
    lanczos_recurrence(
      lo, v.mutable_data(), num_steps, lanczos_tol, orthogonalize, 
      alpha.mutable_data(), beta.mutable_data(), Q.mutable_data(), ncv
    );
  });
  m.def("quadrature", [](py_array< float > a, py_array< float > b, const int k) -> py_array< float > {             
    auto output = DenseMatrix< float >(k, 2); // [nodes, weights]
    lanczos_quadrature(a.data(), b.data(), k, output.col(0).data(), output.col(1).data());
    return py::cast(output); 
  });
  m.def("stochastic_quadrature", [](
    const Eigen::SparseMatrix< float >& mat, 
    const int nv, const int dist, 
    const int lanczos_degree, const int lanczos_rtol, const int orth, const int ncv,
    const int num_threads, const int seed
  ){      
    const auto lo = SparseEigenLinearOperator(mat);
    auto rbg = ThreadedRNG64(num_threads, seed);
    const auto f = [](int i, float* q, float* Q, float* nodes, float* weights){
      printf("iter %0d, tid %0d\n", i, omp_get_thread_num());
    };
    slq< float >(lo, f, nv, rademacher, rbg, lanczos_degree, lanczos_rtol, orth, ncv, num_threads, seed);
  });
}


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


// template< std::floating_point F, LinearOperator Matrix >
// auto lanczos(
//   const Matrix& A,            // Symmetric linear operator 
//   F* q,                       // vector to expand the Krylov space K(A, q)
//   const int k,                // Dimension of the Krylov subspace to capture
//   const int orth,             // Number of additional vectors to orthogonalize againt 
//   const F lanczos_tol,        // Tolerance of residual error for early-stopping the iteration.
//   F* alpha,                   // Output diagonal elements of T of size A.shape[1]+1
//   F* beta,                    // Output subdiagonal elements of T of size A.shape[1]+1
//   F* lanczos_vectors,         // Output Lanczos vector (may be unused by caller)  
//   const size_t ncv            // Number of Lanczos vectors allocated
// ) -> void {                  
//   const auto A_shape = A.shape();
//   const size_t n = A_shape.first;
//   const size_t m = A_shape.second;
  
//   // Number of Lanczos vectors to keep in memory
//   const size_t ncv_actual = 
//     (orth == 0 || orth == 1) ? 2 :         // Minimum orthogonalization
//     (orthogonalize < 0 || orthogonalize > k) ? size_t(k) :   // Full reorthogonalization
//     static_cast< size_t >(orthogonalize);                    // Partial orthogonalization (0 < orthogonalize < m)
  
//   // Allocate lanczos vectors and apply the recurrence
//   // auto Q = (DenseMatrix< F >) DenseMatrix< F >::Zero(n, ncv); // Lanczos vectors
//   // auto Q_ref = Eigen::Ref< DenseMatrix < F > >(Q);
//   // Eigen::Map< DenseMatrix< F > > Q_ref; 
//   lanczos_recurrence(A, q, k, ncv, lanczos_tol, alpha, beta, lanczos_vectors);
// }

// template< std::floating_point F, LinearOperator Matrix >
// auto lanczos_Q(
//   const Matrix& A,            // Symmetric linear operator 
//   F* q,                       // vector to expand the Krylov space K(A, q)
//   const int k,                // Dimension of the Krylov subspace to capture
//   const F lanczos_tol,        // Tolerance of residual error for early-stopping the iteration.
//   const int orthogonalize,    // Number of lanczos vectors to keep numerically orthonormal in-memory
//   F* alpha,                   // Output diagonal elements of T of size A.shape[1]+1
//   F* beta,                    // Output subdiagonal elements of T of size A.shape[1]+1
//   Ref< DenseMatrix < F > > Q  // Output Lanczos vectors
// ) -> void {                  
//   const auto A_shape = A.shape();
//   const size_t n = A_shape.first;
//   const size_t m = A_shape.second;
  
//   // Number of Lanczos vectors to keep in memory
//   const size_t ncv = 
//     (orthogonalize == 0 || orthogonalize == 1) ? 2 :         // Minimum orthogonalization
//     (orthogonalize < 0 || orthogonalize > k) ? size_t(k) :   // Full reorthogonalization
//     static_cast< size_t >(orthogonalize);                    // Partial orthogonalization (0 < orthogonalize < m)
  
//   lanczos_recurrence(A, q, k, lanczos_tol, alpha, beta, Q);
// }


// template< std::floating_point F >
// void eigsh_tridiagonal(py_array< float >& alpha, py_array< float >& beta, bool eigenvectors = false){
//   auto t_solver =  Eigen::SelfAdjointEigenSolver< DenseMatrix< F > >(); 

//   auto Eigen::DecompositionOptions::EigenvaluesOnly; // ComputeEigenvectors


// }