#include <concepts> 
#include <functional> // function
#include <algorithm>  // max
#include "_linear_operator/linear_operator.h" // LinearOperator
#include "_orthogonalize/orthogonalize.h"   // orth_vector, mod
#include "_random_generator/vector_generator.h" // ThreadSafeRBG, generate_array
#include <Eigen/Core> 
#include <Eigen/Eigenvalues> 

using std::function; 

// Paige's A1 variant of the Lanczos method
// Computes the first k elements (a,b) := (alpha,beta) of the tridiagonal matrix T(a,b) where T = Q^T A Q
// Precondition: orth < ncv <= k and ncv >= 2.
template< std::floating_point F, LinearOperator Matrix >
void lanczos_recurrence(
  const Matrix& A,            // Symmetric linear operator 
  F* q,                       // vector to expand the Krylov space K(A, q)
  const int k,                // Dimension of the Krylov subspace to capture
  const F lanczos_rtol,       // Tolerance of residual error for early-stopping the iteration.
  const int orth,             // Number of *additional* vectors to orthogonalize against 
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
  const F residual_tol = std::sqrt(n) * lanczos_rtol;

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

// const Matrix& A,            // Symmetric linear operator 
// F* q,                       // vector to expand the Krylov space K(A, q)
// const int k,                // Dimension of the Krylov subspace to capture
// const F lanczos_tol,        // Tolerance of residual error for early-stopping the iteration.
// const int orth,             // Number of additional vectors to orthogonalize againt 

// Uses the Lanczos method to obtain Gaussian quadrature estimates of the spectrum of an arbitrary operator
template< std::floating_point F >
auto lanczos_quadrature(
  const F* alpha,                   // Input diagonal elements of T of size k
  const F* beta,                    // Output subdiagonal elements of T of size k, whose non-zeros start at index 1
  const int k,                      // Size of input / output elements 
  F* nodes,                         // Output nodes of the quadrature
  F* weights                        // Output weights of the quadrature
) -> void {
  using VectorF = Eigen::Array< F, Dynamic, 1>;

  // Use Eigen to obtain eigenvalues + eigenvectors of tridiagonal
  auto solver = Eigen::SelfAdjointEigenSolver< DenseMatrix< F > >(k);
  Eigen::Map< const VectorF > a(alpha, k);        // diagonal elements
  Eigen::Map< const VectorF > b(beta+1, k-1);     // subdiagonal elements (offset by 1!)

  // Compute the eigen-decomposition from the tridiagonal matrix
  solver.computeFromTridiagonal(a, b, Eigen::DecompositionOptions::ComputeEigenvectors);
  
  // Retrieve the Rayleigh-Ritz values (nodes)
  auto theta = static_cast< VectorF >(solver.eigenvalues());
  
  // Retrieve the first components of the eigenvectors (weights)
  // NOTE: one idea to reduce memory is to use the matching moment idea
  // - Take T - \lambda I for some eigenvalue \lambda; then dim(null(T- \lambda I)) = 1, thus 
  // - we can solve for (T - \lambda I)x = 0 for x repeatedly, for all \lambda, and take x[0] to be the quadrature weight
  auto tau = static_cast< VectorF >(solver.eigenvectors().row(0));
  tau *= tau;
  
  // Copy the quadrature nodes and weights to the output
  std::copy(theta.begin(), theta.end(), nodes);
  std::copy(tau.begin(), tau.end(), weights);
};

// Stochastic Lanczos quadrature method
// const std::function<F(F)> mf,     // function to apply the eigenvalues of T = Q^T A Q
template< std::floating_point F, LinearOperator Matrix, ThreadSafeRBG RBG, typename Lambda >
void slq (
  const Matrix& A,                  // Any linear operator supporting .matvec() and .shape() methods
  const Lambda& f,                  // Thread-safe function with signature f(int i, F* nodes, F* weights)
  const function< bool() >& stop_check,   // Function to check for convergence or early-stopping (takes no arguments)
  const int nv,                     // Number of sample vectors to generate
  const Distribution dist,          // Isotropic distribution used to generate random vectors
  RBG& rng,                         // Random bit generator
  const int lanczos_degree,         // Polynomial degree of the Krylov expansion
  const F lanczos_rtol,             // residual tolerance to consider subspace A-invariant
  const int orth,                   // Number of vectors to re-orthogonalize against <= lanczos_degree
  const int ncv,                    // Number of Lanczos vectors to keep in memory (per-thread)
  const int num_threads,            // Number of threads used to parallelize the computation   
  const int seed                    // Seed for random number generator for determinism
){   
  using ArrayF = Eigen::Array< F, Dynamic, 1 >;

  // Constants
  const auto A_shape = A.shape();
  const size_t n = A_shape.first;
  const size_t m = A_shape.second;

  // Set the number of threads + initialize multi-threaded RNG
  omp_set_num_threads(num_threads);
  rng.initialize(num_threads, seed);

  // Using square-root of max possible chunk size for parallel schedules
  unsigned int chunk_size = std::max(int(sqrt(nv / num_threads)), 1);
  
  // Monte-Carlo ensemble sampling
  int i;
  volatile bool stop_flag = false; // early-stop flag for convergence checking
  #pragma omp parallel shared(stop_flag)
  {
    int tid = omp_get_thread_num(); // thread-id 

    // Pre-allocate memory needed for Lanczos iterations
    auto q_norm = static_cast< F >(0.0);
    auto q = static_cast< ArrayF >(ArrayF::Zero(m)); 
    auto Q = static_cast< DenseMatrix< F > >(DenseMatrix< F >::Zero(n, ncv));
    auto alpha = static_cast< ArrayF >(ArrayF::Zero(lanczos_degree));
    auto beta = static_cast< ArrayF >(ArrayF::Zero(lanczos_degree));
    auto nodes = static_cast< ArrayF >(ArrayF::Zero(lanczos_degree));
    auto weights = static_cast< ArrayF >(ArrayF::Zero(lanczos_degree));
    
    // Run in parallel 
    #pragma omp for schedule(dynamic, chunk_size)
    for (i = 0; i < nv; ++i){
      if (stop_flag){ continue; }

      // Generate isotropic vector (w/ unit norm)
      generate_isotropic< F >(dist, m, rng, tid, q.data(), q_norm);
      
      // Perform a lanczos iteration (populates alpha, beta)
      lanczos_recurrence< F >(A, q.data(), lanczos_degree, lanczos_rtol, orth, alpha.data(), beta.data(), Q.data(), ncv); 

      // Obtain nodes + weights via quadrature algorithm
      lanczos_quadrature< F >(alpha.data(), beta.data(), lanczos_degree, nodes.data(), weights.data());

      // Run the user-supplied function (parallel section!)
      f(i, q.data(), Q.data(), nodes.data(), weights.data());
      
      // If supplied, check early-stopping condition
      #pragma omp critical
      {
        stop_flag = stop_check();
      }
    }
  }
}

template< std::floating_point F, LinearOperator Matrix, ThreadSafeRBG RBG >
void sl_trace(
  const Matrix& mat, const std::function< F(F) > sf, 
  RBG& rbg, const int nv, const int dist, const int engine_id, const int seed,
  const int lanczos_degree, const float lanczos_rtol, const int orth, const int ncv,
  const int num_threads,
  F* estimates
){      
  using VectorF = Eigen::Array< F, Dynamic, 1>;

  // Parameterize the trace function (run in parallel)
  const auto trace_f = [lanczos_degree, &sf, &estimates](int i, [[maybe_unused]] F* q, [[maybe_unused]] F* Q, F* nodes, F* weights){
    Eigen::Map< VectorF > nodes_v(nodes, lanczos_degree, 1);     // no-op
    Eigen::Map< VectorF > weights_v(weights, lanczos_degree, 1); // no-op
    nodes_v.unaryExpr(sf);
    estimates[i] = (nodes_v * weights_v).sum();
  };
  
  // Parameterize when to stop (run in critical section)
  const auto early_stop = []() -> bool {
    return false; 
  };

  // Execute the stochastic Lanczos quadrature with the trace function 
  slq< float >(mat, trace_f, early_stop, nv, static_cast< Distribution >(dist), rbg, lanczos_degree, lanczos_rtol, orth, ncv, num_threads, seed);
}

template< std::floating_point F, LinearOperator Matrix, ThreadSafeRBG RBG > 
void sl_quadrature(
  const Matrix& mat, 
  RBG& rbg, const int nv, const int dist, const int engine_id, const int seed,
  const int lanczos_degree, const F lanczos_rtol, const int orth, const int ncv,
  const int num_threads, 
  F* quad_nw
){
  using VectorF = Eigen::Array< F, Dynamic, 1>;

  // Parameterize the quadrature function
  Eigen::Map< DenseMatrix< F >> quad_nw_map(quad_nw, lanczos_degree * nv, 2);
  const auto quad_f = [lanczos_degree, &quad_nw_map](int i, [[maybe_unused]] F* q, [[maybe_unused]] F* Q, F* nodes, F* weights){
    // printf("iter %0d, tid %0d, q[0] = %.4g, nodes[0] = %.4g\n", i, omp_get_thread_num(), q[0], nodes[0]);
    Eigen::Map< VectorF > nodes_v(nodes, lanczos_degree, 1);        // no-op
    Eigen::Map< VectorF > weights_v(weights, lanczos_degree, 1);    // no-op
    quad_nw_map.block(i*lanczos_degree, 0, lanczos_degree, 1) = nodes_v;
    quad_nw_map.block(i*lanczos_degree, 1, lanczos_degree, 1) = weights_v;
  };

  // Parameterize when to stop (run in critical section)
  const auto early_stop = []() -> bool {
    return false; 
  };

  // Execute the stochastic Lanczos quadrature
  slq< float >(mat, quad_f, early_stop, nv, static_cast< Distribution >(dist), rbg, lanczos_degree, lanczos_rtol, orth, ncv, num_threads, seed);
}