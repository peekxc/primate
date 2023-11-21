#include <concepts> 
#include "_linear_operator/linear_operator.h" // LinearOperator
#include "_orthogonalize/orthogonalize.h" // orth_vector, mod
#include <Eigen/Eigenvalues> 

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
    auto [p,c,n] = pos;                   // previous, current, next
    A.matvec(Q.col(c).data(), v.data());  // v = A q_c
    v -= beta[j] * Q.col(p);              // q_n = v - b q_p
    alpha[j] = Q.col(c).dot(v);           // projection size of < qc, qn > 
    v -= alpha[j] * Q.col(c);             // subtract projected components

    // Re-orthogonalize q_n against previous ncv-1 lanczos vectors
    if (orth > 0) {
      auto qn = Eigen::Ref< ColVector< F > >(v);          
      orth_vector(qn, Q_ref, c, orth, true);
    }

    // Early-stop criterion is when K_j(A, v) is near invariant subspace.
    beta[j+1] = v.norm();
    if (beta[j+1] < residual_tol || (j+1) == k) { // additional break prevents overriding qn
      break;
    }
    Q.col(n) = v / beta[j+1]; // normalize such that Q stays orthonormal

    // Cyclic left-rotate to update the working column indices
    std::rotate(pos.begin(), pos.begin() + 1, pos.end());
    pos[2] = mod(j+2, ncv);
  }
}

// Uses the Lanczos method to obtain Gaussian quadrature estimates of the spectrum of an arbitrary operator
template< std::floating_point F, LinearOperator Matrix >
auto lanczos_quadrature(
  const Matrix& A,            // Symmetric linear operator 
  F* q,                       // vector to expand the Krylov space K(A, q)
  const int k,                // Dimension of the Krylov subspace to capture
  const F lanczos_tol,        // Tolerance of residual error for early-stopping the iteration.
  const int orth,             // Number of additional vectors to orthogonalize againt 
  F* alpha,                   // Output diagonal elements of T of size A.shape[1]+1
  F* beta,                    // Output subdiagonal elements of T of size A.shape[1]+1
  F* V,                       // Output matrix for Lanczos vectors (column-major)
  const size_t ncv            // Number of Lanczos vectors to allocate
) -> ColVector< F >{
  using ColVectorF = ColVector< F >; 

  // Constants
  const auto A_shape = A.shape();
  const size_t n = A_shape.first;
  const size_t m = A_shape.second;
  
  // Perform the lanczos iteration (populates alpha, beta)
  lanczos_recurrence(A, q, k, lanczos_tol, orth, alpha, beta, V, ncv); 

  // Use Eigen to obtain eigenvalues + eigenvectors of tridiagonal
  // rw, V = eigh_tridiagonal(a,b, eigvals_only=False)
  auto solver = Eigen::SelfAdjointEigenSolver< DenseMatrix< F > >(n);
  Eigen::Map< ColVectorF > a(alpha, k, 1);    // diagonal elements
  Eigen::Map< ColVectorF > b(beta+1, k, 1);   // subdiagonal elements (offset by 1!)

  // Compute the eigen-decomposition from the tridiagonal matrix
  solver.computeFromTridiagonal(a, b, Eigen::DecompositionOptions::ComputeEigenvectors);
  
  // Retrieve the eigenvalues
  auto ew = static_cast< ColVectorF >(solver.eigenvalues());
  // auto V = solver.get_Mat();
  // auto rw = solver.get_eigenvalues(); // Rayleigh-Ritz values
  
  // // Get the quadrature nodes + weights 
  // auto weights = (RowVector< F >) V.row(0).pow(2);
  return ew;
};