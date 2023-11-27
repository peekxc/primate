#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
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

template< std::floating_point F > 
auto parameterize_spectral_func(const py::kwargs& kwargs) -> std::function< F(F) >{
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


template< std::floating_point F > 
auto parameterize_rng(const int engine_id) {
  // "splitmix64", "xoshiro256**", "lcg64", "pcg64", "mt64"
  if (engine_id == 0){
    return ThreadedRNG64< SplitMix64 >(0, seed);
  } else if (engine_id == 1){
    return ThreadedRNG64< Xoshiro256StarStar >(0, seed);
  } else if (engine_id == 2){
    return ThreadedRNG64< knuth_lcg >(0, seed);
  } else if (engine_id == 3){
    return ThreadedRNG64< pcg64 >(0, seed);
  } else if (engine_id == 4){
    return ThreadedRNG64< std::mt19937_64 >(0, seed);
  } else {
    throw std::invalid_argument("Invalid random number engine id.");
  }
}


template< std::floating_point F >
void slq_trace(
  const Eigen::SparseMatrix< F >& mat, 
  const std::function< F(F) > sf, 
  const int nv, const int dist, 
  const int lanczos_degree, const F lanczos_rtol, const int orth, const int ncv,
  const int num_threads, const int seed, 
  F* estimates
){      
  using VectorF = Eigen::Array< F, Dynamic, 1>;
  const auto lo = SparseEigenLinearOperator(mat);
  auto rbg = ThreadedRNG64(num_threads, seed);

  // Parameterize the trace function
  const auto trace_f = [lanczos_degree, &sf, &estimates](int i, F* q, F* Q, F* nodes, F* weights){
    // printf("iter %0d, tid %0d, q[0] = %.4g, nodes[0] = %.4g\n", i, omp_get_thread_num(), q[0], nodes[0]);
    Eigen::Map< VectorF > nodes_v(nodes, lanczos_degree, 1);     // no-op
    Eigen::Map< VectorF > weights_v(weights, lanczos_degree, 1); // no-op
    nodes_v.unaryExpr(sf);
    estimates[i] = (nodes_v * weights_v).sum();
  };
  
  // Execute the stochastic Lanczos quadrature with the trace function 
  slq< float >(lo, trace_f, nv, rademacher, rbg, lanczos_degree, lanczos_rtol, orth, ncv, num_threads, seed);
}

PYBIND11_MODULE(_lanczos, m) {
  using ArrayF = Eigen::Array< float, Dynamic, 1 >;
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
    const int lanczos_degree, const float lanczos_rtol, const int orth, const int ncv,
    const int num_threads, const int seed){
    std::vector< DenseMatrix < float > > quad_nw; // quadrature nodes + weights
    auto estimates = static_cast< DenseMatrix >(ArrayF::Zero(nv));
  });
  m.def("stochastic_trace", [](
    const Eigen::SparseMatrix< float >& mat, 
    const int nv, const int dist, 
    const int lanczos_degree, const float lanczos_rtol, const int orth, const int ncv,
    const int num_threads, const int seed, 
    const py::kwargs& kwargs
  ) -> py_array< float > {
    const auto sf = parameterize_spectral_func< float >(kwargs);
    auto estimates = static_cast< ArrayF >(ArrayF::Zero(nv));
    slq_trace(mat, sf, nv, dist, lanczos_degree, lanczos_rtol, orth, ncv, num_threads, seed, estimates.data());
    return py::cast(estimates);
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