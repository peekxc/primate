#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <_linear_operator/linear_operator.h>
#include <Eigen/Core>
#include <Eigen/QR>

#include "eigen_operators.h"
#include <iostream>

namespace py = pybind11;

template< typename F >
using py_array = py::array_t< F, py::array::c_style | py::array::forcecast >;
using namespace Eigen; 

template< typename F >
using ColVector = Eigen::Matrix< F, Dynamic, 1 >;

template< typename F >
using RowVector = Eigen::Matrix< F, 1, Dynamic >;

template< typename F >
using DenseMatrix = Eigen::Matrix< F, Dynamic, Dynamic >;

// Emulate python modulus behavior since C++ '%' is not a true modulus
constexpr inline auto mod(int a, int b) noexcept -> int {
  return (b + (a % b)) % b; 
}

// Orthogonalizes v with respect to columns in U via modified gram schmidt
// Cyclically projects v onto the columns U[:,i:(i+p)] = u_i, u_{i+1}, ..., u_{i+p}, removing from v the components 
// of the vector projections. If any index i, ..., i+p exceeds the number of columns of U, the indices are cycled. 
// Both near-zero projections and columns of U with near-zero norm are ignored to avoid collapsing v to the trivial vector.
// Eigen::Ref use based on: https://stackoverflow.com/questions/21132538/correct-usage-of-the-eigenref-class
template< std::floating_point F >
void orth_vector(
  Ref< ColVector< F > > v,                  // input/output vector
  const Ref< const DenseMatrix< F > >& U,   // matrix of vectors to project onto
  const int start_idx,                      // starting column index
  const int p,                              // number of projections
  const bool reverse = false                // whether to cycle through the columns of U backwards
) {
  const int n = (int) U.rows(); 
  const int m = (int) U.cols(); 
  const auto tol = 2 * std::numeric_limits< F >::epsilon() * std::sqrt(n); // numerical tolerance for orthogonality

  // Successively subtracting the projection of v onto the first p columns starting at start_idx
  // If projection or the target vector is near-zero, ignore and continue, as numerical orthogonality is already met
  const int diff = reverse ? -1 : 1; 
  for (int i = mod(start_idx, m), c = 0; c < p; ++c, i = mod(i + diff, m)){
    const auto u_norm = U.col(i).squaredNorm();     // norm of u_i
    const auto proj_len = v.dot(U.col(i));          // < v, u_i > 
    if (std::min(std::abs(proj_len), u_norm) > tol){
      v -= (proj_len / u_norm) * U.col(i);
    }
  }
}

// Paige's A1 variant of the Lanczos method
// Computes the first k elements (a,b) := (alpha,beta) of the tridiagonal matrix T(a,b) where T = Q^T A Q
template< std::floating_point F, LinearOperator Matrix >
void lanczos_recurrence(
  const Matrix& A,            // Symmetric linear operator 
  F* q,                       // vector to expand the Krylov space K(A, q)
  const int k,                // Dimension of the Krylov subspace to capture
  const F lanczos_tol,        // Tolerance of residual error for early-stopping the iteration.
  const int orth,             // Number of additional vectors to orthogonalize againt 
  F* alpha,                   // Output diagonal elements of T of size A.shape[1]+1
  F* beta,                    // Output subdiagonal elements of T of size A.shape[1]+1
  F* V,                       // Output matrix for Lanczos vectors (column-major)
  const size_t ncv            // Number of Lanczos vectors pre-allocated (must be at least 2)
){
  using ColVectorF = Eigen::Matrix< F, Dynamic, 1 >;

  // Constants `
  const auto A_shape = A.shape();
  const size_t n = A_shape.first;
  const size_t m = A_shape.second;
  const F residual_tol = std::sqrt(n) * lanczos_tol;

  // Allocation / views
  Eigen::Map< DenseMatrix< F > > Q(V, n, ncv);  // Lanczos vectors 
  Eigen::Map< ColVectorF > v(q, m, 1);          // map initial vector (no-op)
  Q.col(0) = v.normalized();                    // load normalized v0 into Q  

  // In the following, beta[j] means beta[j-1] in the Demmel text
  const auto Q_ref = Eigen::Ref< const DenseMatrix< F > >(Q); 
  std::array< int, 3 > pos = { static_cast< int >(ncv - 1), 0, 1 };
  for (int j = 0; j < k; ++j) {

    // Apply the three-term recurrence
    auto [p,c,n] = pos;                  // previous, current, next
    A.matvec(Q.col(c).data(), v.data()); // v = A q_c
    v -= beta[j] * Q.col(p);            // q_n = v - b q_p
    alpha[j] = Q.col(c).dot(v);   // projection size of < qc, qn > 
    v -= alpha[j] * Q.col(c);     // subtract projected components

    // Re-orthogonalize q_n against previous ncv-1 lanczos vectors
    if (orth > 0) {
      auto qn = Eigen::Ref< ColVector< F > >(v);          
      orth_vector(qn, Q_ref, c, orth, true);
    }

    // Early-stop criterion is when K_j(A, v) is near invariant subspace.
    beta[j+1] = v.norm();
    if (beta[j+1] < residual_tol || (j+1) == k) { // additional break to prevent overriding qn
      break;
    }
    Q.col(n) = v / beta[j+1]; // normalize such that Q stays orthonormal

    // Cyclic left-rotate to update the working column indices
    std::rotate(pos.begin(), pos.begin() + 1, pos.end());
    pos[2] = mod(j+2, ncv);
  }
}

// Given an input vector 'z', yields the vector y = Qz where T(alpha, beta) = Q^T A Q is the tridiagonal matrix spanning K(A, q)
template< std::floating_point F, LinearOperator Matrix >
void lanczos_action(  
  const Matrix& A,            // Symmetric linear operator 
  F* q,                       // vector to expand the Krylov space K(A, q)
  const int k,                // Dimension of the Krylov subspace to capture
  const int ncv,              // Number of Lanczos vectors
  const F* alpha,             // Diagonal elements of T of size A.shape[1]+1
  const F* beta,              // Subdiagonal elements of T of size A.shape[1]+1
  const F* z,                 // Input vector to multiply Q by 
  F* V,                       // Storage for the Lanczos vectors (column-major)
  F* y                        // Output vector y = Q z
){
  using ColVectorF = Eigen::Matrix< F, Dynamic, 1 >;

  // Constants `
  const auto A_shape = A.shape();
  const size_t n = A_shape.first;
  const size_t m = A_shape.second;

  // Allocation / views
  Eigen::Map< DenseMatrix< F > > Q(V, n, ncv);  // Lanczos vectors 
  Eigen::Map< ColVectorF > v(q, m, 1);          // map initial vector (no-op)
  Eigen::Map< ColVectorF > w(z, k, 1);          // map input vector to multiply (no-op)
  Eigen::Map< ColVectorF > x(y, n, 1);          // map output vector to store (no-op)
  Q.col(0) = v.normalized();                    // load normalized v0 into Q  

  // Set the output to zero to begin
  x.setZero(m);

  // In the following, beta[j] means beta[j-1] in the Demmel text
  std::array< int, 3 > pos = { static_cast< int >(ncv - 1), 0, 1 };
  for (int j = 0; j < k; ++j){
    auto [p,c,n] = pos;                   // previous, current, next
    x += w[j] * Q.col(c);                 // x = Q_k w <=> x <- w[0] * q_0 + w[1] * q_1 + ... + w[k-1] * q_{k-1}

    // Apply the three-term recurrence
    A.matvec(Q.col(c).data(), v.data()); // v = A q_c
    Q.col(n) = v - beta[j] * Q.col(p);   // q_n = v - b q_p
    Q.col(n) -= alpha[j] * Q.col(c);     // subtract projected components

    // Re-orthogonalize q_n against previous ncv-1 lanczos vectors
    if (ncv > 2) {
      auto qn = Eigen::Ref< ColVector< F > >(Q.col(n));          
      const auto Q_ref = Eigen::Ref< const DenseMatrix< F > >(Q); 
      orth_vector(qn, Q_ref, c, ncv-1, true);
    }
    Q.col(n) /= beta[j+1]; // normalize such that Q stays orthonormal
    
    // Cyclic left-rotate to update the working column indices
    std::rotate(pos.begin(), pos.begin() + 1, pos.end());
    pos[2] = mod(j+2, ncv);
  }
}


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

// Modified Gram-Schmidt in-place
// G. W. Stewart, "Matrix Algorithms, Volume 1", SIAM, 1998.
template< std::floating_point F >
void modified_gram_schmidt(Ref< DenseMatrix< F > > U, const int s = 0){
  // const int n = static_cast< const int >(U.rows());
  const int m = static_cast< const int >(U.cols());
  // auto i = mod(s, m);
  auto R = DenseMatrix< F >(m, m);
  for (int k = 0; k < m; ++k){
    for (int i = 0; i < k; ++i){
      R(i,k) = U.col(i).dot(U.col(k));
      U.col(k) -= R(i,k) * U.col(i);
    }
    R(k,k) = U.col(k).norm();
    U.col(k) /= R(k,k);
  }
}

PYBIND11_MODULE(_lanczos, m) {
  // m.def("orthogonalize", &orthogonalize< float >);
  m.def("orth_vector", &orth_vector< float >);
  m.def("lanczos", [](
    const Eigen::SparseMatrix< float >& mat, 
    py_array< float >& v, 
    const int num_steps, 
    const float lanczos_tol, 
    const int orthogonalize, 
    py_array< float >& alpha, 
    py_array< float >& beta, 
    py::array_t< float, py::array::f_style >& Q
  ){
    const auto lo = SparseEigenLinearOperator(mat);
    // lanczos< float >(lo, v.mutable_data(), num_steps, lanczos_tol, orthogonalize, alpha.mutable_data(), beta.mutable_data(), Q.mutable_data());
    const size_t ncv = static_cast< size_t >(Q.shape(1));
    lanczos_recurrence(
      lo, v.mutable_data(), num_steps, lanczos_tol, orthogonalize, 
      alpha.mutable_data(), beta.mutable_data(), Q.mutable_data(), ncv
    );
  });
  m.def("mgs", &modified_gram_schmidt< float >);
}

  // const Matrix& A,            // Symmetric linear operator 
  // F* q,                       // vector to expand the Krylov space K(A, q)
  // const int k,                // Dimension of the Krylov subspace to capture
  // const F lanczos_tol,        // Tolerance of residual error for early-stopping the iteration.
  // const int orth,             // Number of additional vectors to orthogonalize againt 
  // F* alpha,                   // Output diagonal elements of T of size A.shape[1]+1
  // F* beta,                    // Output subdiagonal elements of T of size A.shape[1]+1
  // F* V,                       // Output matrix for Lanczos vectors (column-major)
  // const size_t ncv            // Number of Lanczos vectors pre-allocated (must be at least 2)
