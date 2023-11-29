#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include "_random_generator/vector_generator.h"
#include "_lanczos/lanczos.h"
#include "eigen_operators.h"
#include <iostream>
#include <stdio.h>
#include <any>

namespace py = pybind11;

// NOTE: all matrices should be cast to Fortran ordering for compatibility with Eigen
template< typename F >
using py_array = py::array_t< F, py::array::f_style | py::array::forcecast >;

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

template< std::floating_point F > 
auto param_spectral_func(const py::kwargs& kwargs) -> std::function< F(F) >{
  auto kwargs_map = kwargs.cast< std::unordered_map< std::string, py::object > >();
  std::function< F(F) > f = std::identity();
  if (kwargs_map.contains("function")){
    std::string matrix_func = kwargs_map["function"].cast< std::string >(); // py::function
    if (matrix_func == "identity"){
      f = std::identity(); 
    } else if (matrix_func == "abs"){
      f = [](F eigenvalue) -> F { return std::abs(eigenvalue); }; 
    } else if (matrix_func == "sqrt"){
      f = [](F eigenvalue) -> F { return std::sqrt(std::abs(eigenvalue)); }; 
    } else if (matrix_func == "log"){
      f = [](F eigenvalue) -> F { return std::log(eigenvalue); }; 
    } else if (matrix_func == "inv"){
      f = [](F eigenvalue) -> F {  return 1.0/eigenvalue; };
    } else if (matrix_func == "exp"){
      F t = kwargs_map.contains("t") ? kwargs_map["t"].cast< F >() : 0.0;
      f = [t](F eigenvalue) -> F {  return std::exp(t*eigenvalue); };  
    } else if (matrix_func == "smoothstep"){
      F a = kwargs_map.contains("a") ? kwargs_map["a"].cast< F >() : 0.0;
      F b = kwargs_map.contains("b") ? kwargs_map["b"].cast< F >() : 1.0;
      const F d = (b-a);
      f = [a, d](F eigenvalue) -> F { 
        return std::min(std::max((eigenvalue-a)/d, F(0.0)), F(1.0)); 
      }; 
    } else if (matrix_func == "gaussian"){
      F mu = kwargs_map.contains("mu") ? kwargs_map["mu"].cast< F >() : 0.0;
      F sigma = kwargs_map.contains("sigma") ? kwargs_map["sigma"].cast< F >() : 1.0;
      f = [mu, sigma](F eigenvalue) -> F {  
        auto x = (mu - eigenvalue) / sigma;
        return (0.5 * M_SQRT1_2 * M_2_SQRTPI / sigma) * exp(-0.5 * x * x); 
      }; 
    } else if (matrix_func == "numrank"){
      F threshold = kwargs_map.contains("threshold") ? kwargs_map["threshold"].cast< F >() : 0.000001;
      f = [threshold](F eigenvalue) -> F {  
        return std::abs(eigenvalue) > threshold ? F(1.0) : F(0.0);
      };  
    } else if (matrix_func == "generic"){
      if (kwargs_map.contains("matrix_func")){
        py::function g = kwargs_map["matrix_func"].cast< py::function >();
        f = [&g](F val) -> F { return g(val).template cast< F >(); };
      } else {
        f = std::identity();
      }
    } else {
      throw std::invalid_argument("Invalid matrix function supplied");
    }
  } else {
    throw std::invalid_argument("No matrix function supplied.");
  }
  return f; 
}

// Template function for generating module definitions for a given Operator / precision 
template< std::floating_point F, class Matrix, typename WrapperFunc >
void _lanczos_wrapper(py::module& m, WrapperFunc wrap = std::identity()){
  using ArrayF = Eigen::Array< F, Dynamic, 1 >;
  m.def("lanczos", [wrap]( // keep wrap pass by value!
    const Matrix* A, 
    py_array< F > v, const int lanczos_degree, const F lanczos_rtol, const int orth,
    py_array< F >& alpha, py_array< F >& beta, py_array< F >& Q 
  ){ 
    const auto op = wrap(A);
    const size_t ncv = static_cast< size_t >(Q.shape(1));
    lanczos_recurrence(
      op, v.mutable_data(), lanczos_degree, lanczos_rtol, orth, 
      alpha.mutable_data(), beta.mutable_data(), Q.mutable_data(), ncv
    );
  });
  m.def("quadrature", [](
    py_array< F > a, py_array< F > b, 
    const int k
  ) -> py_array< F > {             
    auto output = DenseMatrix< F >(k, 2); // [nodes, weights]
    lanczos_quadrature(a.data(), b.data(), k, output.col(0).data(), output.col(1).data());
    return py::cast(output); 
  });
  m.def("stochastic_quadrature", [wrap](
    const Matrix* A, 
    const int nv, const int dist, const int engine_id, const int seed,
    const int lanczos_degree, const F lanczos_rtol, const int orth, const int ncv,
    const int num_threads
  ) -> py_array< F > {
    const auto op = wrap(A);
    // auto rbg = ThreadedRNG64(num_threads, seed);
    // auto rbg = param_rng< 0 >(seed);
    // auto rbg = param_rng(0, seed, num_threads);
    auto rbg = ThreadedRNG64< pcg64 >(num_threads, seed);
    auto quad_nw = static_cast< DenseMatrix< F > >(DenseMatrix< F >::Zero(lanczos_degree * nv, 2));
    sl_quadrature(op, rbg, nv, dist, engine_id, seed, lanczos_degree, lanczos_rtol, orth, ncv, num_threads, quad_nw.data());
    return py::cast(quad_nw);
  });
  m.def("stochastic_trace", [wrap](
    const Matrix* A, 
    const int nv, const int dist, const int engine_id, const int seed,
    const int lanczos_degree, const F lanczos_rtol, const int orth, const int ncv,
    const F atol, const F rtol, 
    const int num_threads, 
    const py::kwargs& kwargs
  ) -> py_array< F > {
    const auto op = wrap(A);
    const auto sf = param_spectral_func< F >(kwargs);
    // std::any rbg = param_rng(0, seed, num_threads);
    // auto rbg = ThreadedRNG64< pcg64 >(num_threads, seed); // TODO: figure out either polymorphism approach or type-erasure approach
    auto rbg = ThreadedRNG64< pcg64 >(num_threads, seed);
    auto estimates = static_cast< ArrayF >(ArrayF::Zero(nv));
    sl_trace(op, sf, rbg, nv, dist, engine_id, seed, lanczos_degree, lanczos_rtol, orth, ncv, atol, rtol, num_threads, estimates.data());
    return py::cast(estimates);
  });
}

PYBIND11_MODULE(_lanczos, m) {
  // m.def("lanczos", _lanczos_wrapper< float, Eigen::MatrixXf, eigen_dense_wrapper< float > >);
  _lanczos_wrapper< float, Eigen::MatrixXf >(m, eigen_dense_wrapper< float >);
  _lanczos_wrapper< float, Eigen::SparseMatrix< float > >(m, eigen_sparse_wrapper< float >);
};

  // m.def("lanczos", _lanczos_wrapper< float, Eigen::MatrixXf, eigen_dense_wrapper< float > >);
  // m.def("lanczos", _lanczos_wrapper< float, Eigen::MatrixXf, eigen_dense_wrapper< float > >);

  
  // [](const Eigen::SparseMatrix< float >& mat, LANCZOS_PARAMS){
  //   const auto lo = SparseEigenLinearOperator(mat);
  //   const size_t ncv = static_cast< size_t >(Q.shape(1));
  //   lanczos_recurrence(
  //     lo, v.mutable_data(), num_steps, lanczos_tol, orthogonalize, 
  //     alpha.mutable_data(), beta.mutable_data(), Q.mutable_data(), ncv
  //   );
  // });
  // m.def("quadrature", [](py_array< float > a, py_array< float > b, const int k) -> py_array< float > {             
  //   auto output = DenseMatrix< float >(k, 2); // [nodes, weights]
  //   lanczos_quadrature(a.data(), b.data(), k, output.col(0).data(), output.col(1).data());
  //   return py::cast(output); 
  // });
  // m.def("stochastic_quadrature", [](
  //   const Eigen::SparseMatrix< float >& mat, 
  //   const int nv, 
  //   const int dist, const int engine_id, const int seed,
  //   const int lanczos_degree, const float lanczos_rtol, const int orth, const int ncv,
  //   const int num_threads
  // ) -> py_array< float > {
  //   const auto lo = SparseEigenLinearOperator(mat);
  //   auto rbg = ThreadedRNG64(num_threads, seed);
  //   auto quad_nw = static_cast< DenseMatrix< float > >(DenseMatrix< float >::Zero(lanczos_degree * nv, 2)); 
  //   return py::cast(quad_nw);
  // });
  // m.def("stochastic_trace", [](
  //   const Eigen::SparseMatrix< float >& mat, 
  //   const int nv, const int dist, 
  //   const int lanczos_degree, const float lanczos_rtol, const int orth, const int ncv,
  //   const int num_threads, const int seed, 
  //   const py::kwargs& kwargs
  // ) -> py_array< float > {
  //   // auto estimates = static_cast< DenseMatrix< float > >(ArrayF::Zero(nv));  
  //   const auto sf = parameterize_spectral_func< float >(kwargs);
  //   auto rbg = ThreadedRNG64(num_threads, seed);
  //   auto estimates = static_cast< ArrayF >(ArrayF::Zero(nv));
  //   slq_trace(mat, sf, nv, dist, lanczos_degree, lanczos_rtol, orth, ncv, num_threads, seed, estimates.data());
  //   return py::cast(estimates);
  // });


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