#ifndef _ORTHOGONALIZE_ORTHOGONALIZE_H
#define _ORTHOGONALIZE_ORTHOGONALIZE_H

#include <concepts> // std::floating_point
// #include <cmath> // std::isnan
#include "eigen_core.h" // DenseMatrix, EigenCore
#include <Eigen/QR>


// Emulate python modulus behavior since C++ '%' is not a true modulus
constexpr inline auto mod(int a, int b) noexcept -> int {
  return (b + (a % b)) % b; 
}

template< std::floating_point F >
inline auto orth_poly_weight(const F x, const F mu_rec_sqrt, const F* a, const F* b, F* tbl, const int n) noexcept -> F { 
  // const auto mu_rec_sqrt = 1.0 / std::sqrt(mu)
  tbl[0] = mu_rec_sqrt;
  tbl[1] = (x - a[0]) * mu_rec_sqrt / b[1];
  F w = std::pow(tbl[0], 2) + std::pow(tbl[1], 2);
  for (int i = 2; i < n; ++i){ 
    tbl[i] = (((x - a[i-1]) * tbl[i-1]) - b[i-1] * tbl[i-2]) / b[i];
    w += std::pow(tbl[i], 2);
  }
  return 1.0 / w;
}

// Orthogonalizes v with respect to columns in U via modified gram schmidt
// Cyclically projects v onto the columns U[:,i:(i+p)] = u_i, u_{i+1}, ..., u_{i+p}, removing from v the components 
// of the vector projections. If any index i, ..., i+p exceeds the number of columns of U, the indices are cycled. 
// Both near-zero projections and columns of U with near-zero norm are ignored to avoid collapsing v to the trivial vector.
// Eigen::Ref use based on: https://stackoverflow.com/questions/21132538/correct-usage-of-the-eigenref-class
template< std::floating_point F >
void orth_vector(
  Ref< Vector< F > > v,                   // input/output vector
  const Ref< const DenseMatrix< F > >& U,    // matrix of vectors to project onto
  const int start_idx,                       // starting column index
  const int p,                               // number of projections
  const bool reverse = false                 // whether to cycle through the columns of U backwards
) {
  const int n = (int) U.rows(); 
  const int m = (int) U.cols(); 
  const auto tol = 2 * std::numeric_limits< F >::epsilon() * std::sqrt(n); // numerical tolerance for orthogonality

  // Successively subtracting the projection of v onto the first p columns starting at start_idx
  // If projection or the target vector is near-zero, ignore and continue, as numerical orthogonality is already met
  const int diff = reverse ? -1 : 1; 
  for (int i = mod(start_idx, m), c = 0; c < p; ++c, i = mod(i + diff, m)){
    const auto u_norm = U.col(i).squaredNorm();      // norm of u_i
    const auto s_proj = v.dot(U.col(i)); // < v, u_i > 
    // should protect against nan and 0 vectors, even is isnan(u_norm) is true
    if (u_norm > tol && std::abs(s_proj) > tol){ 
      v -= (s_proj / u_norm) * U.col(i);
    }
  }
}


// Modified Gram-Schmidt in-place
// G. W. Stewart, "Matrix Algorithms, Volume 1", SIAM, 1998.
template< std::floating_point F >
void modified_gram_schmidt(Eigen::Ref< DenseMatrix< F > > U, [[maybe_unused]] const int s = 0){
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

#endif