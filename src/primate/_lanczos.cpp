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

// Emulate python modulus behavior since % is not a true modulus
constexpr inline auto mod(int a, int b) noexcept -> int {
  return (b + (a % b)) % b; 
}


// Orthogonalize a vector v against a matrix V
template< std::floating_point F >
void gram_schmidt_process(
  const Eigen::Matrix< F, Dynamic, Dynamic >& V,  // Column vectors to orthogonalize v against. Need not necessarily be orthogonal. 
  const size_t j,                                 // Column index in V indicating vectors to orthogonalize against
  const size_t num_ortho,                         // Number of vectors to be orthogonalized starting from j
  F* v)
{
  // No orthogonalization is performed
  const size_t nv = static_cast< size_t >(V.cols());
  if ((num_ortho == 0) || (nv < 2)) { return; }
  const size_t J = j % nv; 
  
  // Figure out actual number of vectors to orthogonalize against
  size_t num_steps = (num_ortho < 0 || num_ortho > nv) ? nv : num_ortho;
  num_steps = std::max(num_steps, static_cast< size_t >(V.rows()));

  // Setup the column-vector like translation for input vector v
  auto v_col = Eigen::Map< Eigen::Matrix< F, Dynamic, 1 > >(v, V.rows(), 1); // no-op

  // Iterate over vectors
  const auto epsilon = std::numeric_limits< F >::epsilon();
  const auto sqrt_n = std::sqrt(nv); 
  for (size_t step = 0; step < num_steps; ++step) {
    // i is the index of a column vector in V to orthogonalize v against it
    // Wrap around negative indices from the end of column index
    auto i = J >= step ? J - step : nv - step - J;
    // i = i >= step ? i - step : i - step + nv;
    // i = (J + step >= nv) ? (J - num_ortho + step) : (j);

    auto norm = V.col(i).norm(); 
    if (norm < epsilon * sqrt_n) { continue; }

    // Project v onto the i'th column of V
    auto inner_prod = V.col(i).dot(v_col);
    auto scale = inner_prod / std::pow(norm, 2);

    // If scale is is 1, it is possible that vector v and j-th vector are identical (or close).
    if (std::abs(scale - 1.0) <= 2.0 * epsilon) {
      auto norm_v = std::pow(v_col.norm(), 2);
      auto distance = std::sqrt(norm_v - 2.0*inner_prod + norm_v); // distance between the j-th vector and vector v

      // If distance is zero, do not reorthogonalize i-th against the j-th vector.
      if (distance < 2.0 * epsilon * sqrt_n) {
        continue;
      }
    }
    v_col -= scale * V.col(i);
  }
}

// Orthogonalizes v with respect to U via modified gram schmidt
// Projects v onto the columns U[:,i:(i+p)] = u_i, u_{i+1}, ..., u_{i+p}, removing from v the components 
// in the direction of the vector projections. If any index i, ..., i+p exceeds the number of columns of U, the 
// indices are cycled. Near-zero projections are ignored, as are columns of U with near-zero norms.
// https://stackoverflow.com/questions/21132538/correct-usage-of-the-eigenref-class
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
    const auto u_norm = U.col(i).squaredNorm();
    const auto proj_len = v.dot(U.col(i));
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
  m.def("lanczos", [](const Eigen::SparseMatrix< float >& mat, py_array< float >& v, const int num_steps, const float lanczos_tol, const int orthogonalize, py_array< float >& alpha, py_array< float >& beta){
    const auto lo = SparseEigenLinearOperator(mat);
    lanczos< float >(lo, v.mutable_data(), num_steps, lanczos_tol, orthogonalize, alpha.mutable_data(), beta.mutable_data());
  });
  m.def("mgs", &modified_gram_schmidt< float >);
  m.def("orth_vector", &orth_vector< float >);
}
