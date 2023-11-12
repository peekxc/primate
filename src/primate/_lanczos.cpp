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
void lanczos(
  const Matrix& A,            // Symmetric linear operator 
  F* q,                       // vector to expand the Krylov space K(A, q)
  const int k,                // Dimension of the Krylov subspace to capture
  const F lanczos_tol,        // Tolerance of residual error for early-stopping the iteration.
  const int orthogonalize,    // Number of lanczos vectors to keep numerically orthonormal in-memory
  F* alpha,                   // Output diagonal elements of T of size A.shape[1]+1
  F* beta                     // Output subdiagonal elements of T of size A.shape[1]+1
){
  using ColVectorF = Eigen::Matrix< F, Dynamic, 1 >;
  
  // Number of Lanczos vectors to keep in memory
  const size_t ncv = 
    (orthogonalize == 0 || orthogonalize == 1) ? 2 :         // Minimum orthogonalization
    (orthogonalize < 0 || orthogonalize > k) ? size_t(k) :   // Full reorthogonalization
    static_cast< size_t >(orthogonalize);                    // Partial orthogonalization (0 < orthogonalize < m)

  // Constants `
  const auto A_shape = A.shape();
  const size_t n = A_shape.first;
  const size_t m = A_shape.second;
  const F residual_tol = std::sqrt(n) * lanczos_tol;

  // Allocation / views
  auto Q = (DenseMatrix< F >) DenseMatrix< F >::Zero(n, ncv);   // Lanczos vectors (zero-initialized to prevent branching below)
  Eigen::Map< ColVectorF > v(q, m, 1);                          // map initial vector (no-op)
  Q.col(0) = v.normalized();                                    // load normalized v0 into Q  

  // In the following, beta[j] means beta[j-1] in the Demmel text
  std::array< int, 3 > pos = { static_cast< int >(ncv - 1), 0, 1 };
  for (int j = 0; j < k; ++j) {

    // Apply the three-term recurrence
    auto [p,c,n] = pos;                  // previous, current, next
    A.matvec(Q.col(c).data(), v.data()); // v = A q_c
    Q.col(n) = v - beta[j] * Q.col(p);   // q_n = v - b q_p
    alpha[j] = Q.col(c).dot(Q.col(n));   // projection size of < qc, qn > 
    Q.col(n) -= alpha[j] * Q.col(c);     // subtract projected components

    // Re-orthogonalize against previous ncv-1 lanczos vectors
    if (ncv > 2) {
      auto qn = Eigen::Ref< ColVector< F > >(Q.col(n));          
      const auto Q_ref = Eigen::Ref< const DenseMatrix< F > >(Q); 
      orth_vector(qn, Q_ref, c, ncv-1, true);
    }

    // Early-stop criterion is when K_j(A, v) is near invariant subspace.
    beta[j+1] = Q.col(n).norm();
    if (beta[j+1] < residual_tol) {
      break;
    }
    Q.col(n) /= beta[j+1]; // normalize such that Q stays orthonormal
    
    // Cyclic left-rotate to update the working column indices
    std::rotate(pos.begin(), pos.begin() + 1, pos.end());
    pos[2] = mod(j+2, ncv);
  }
}


// Lanczos FA-1 



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
  m.def("lanczos", [](const Eigen::SparseMatrix< float >& mat, py_array< float >& v, const int num_steps, const float lanczos_tol, const int orthogonalize, py_array< float >& alpha, py_array< float >& beta){
    const auto lo = SparseEigenLinearOperator(mat);
    lanczos< float >(lo, v.mutable_data(), num_steps, lanczos_tol, orthogonalize, alpha.mutable_data(), beta.mutable_data());
  });
  m.def("mgs", &modified_gram_schmidt< float >);
}
