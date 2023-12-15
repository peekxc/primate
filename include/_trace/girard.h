#include <concepts> 
#include <functional> // function
#include <algorithm>  // max
#include "omp_support.h" // conditionally enables openmp pragmas
#include "_operators/linear_operator.h" // LinearOperator
#include "_orthogonalize/orthogonalize.h"   // orth_vector, mod
#include "_random_generator/vector_generator.h" // ThreadSafeRBG, generate_array
#include "eigen_core.h"
#include <Eigen/Eigenvalues> 

template< std::floating_point F, LinearOperator Matrix, ThreadSafeRBG RBG, typename Lambda >
void monte_carlo_quad(
  const Matrix& A,                              // Any linear operator supporting .matvec() and .shape() methods
  const Lambda& f,                              // Thread-safe function with signature f(int i, F sample, F* q)
  const std::function< bool(int) >& stop_check, // Function to check for convergence or early-stopping (takes no arguments)
  const int nv,                                 // Number of sample vectors to generate
  const Distribution dist,                      // Isotropic distribution used to generate random vectors
  RBG& rng,                                     // Random bit generator
  const int num_threads,                        // Number of threads used to parallelize the computation   
  const int seed                                // Seed for random number generator for determinism
){
  using ArrayF = Eigen::Array< F, Dynamic, 1 >;
  
  // Constants
  const auto A_shape = A.shape();
  const size_t n = A_shape.first;
  const size_t m = A_shape.second;

  // Set the number of threads + initialize multi-threaded RNG
  const auto nt = num_threads <= 0 ? omp_get_max_threads() : num_threads;
  omp_set_num_threads(nt);
  rng.initialize(nt, seed);

  // Using square-root of max possible chunk size for parallel schedules
  unsigned int chunk_size = std::max(int(sqrt(nv / nt)), 1);
  
  // Monte-Carlo ensemble sampling
  int i;
  volatile bool stop_flag = false; // early-stop flag for convergence checking
  #pragma omp parallel shared(stop_flag)
  {
    int tid = omp_get_thread_num(); // thread-id 
    #pragma omp for schedule(dynamic, chunk_size)
    for (i = 0; i < nv; ++i){
      if (stop_flag){ continue; }

      // Generate isotropic vector (w/ unit norm)
      generate_isotropic< F >(dist, m, rng, tid, q.data(), q_norm);
      
      // Apply quadratic form
      F sample = 0.0;  
      if constexpr (QuadOperator< Matrix >){
        sample = A.quad(q.data()); // x^T A x
      } else {
        // ArrayF = ArrayF
        // sample = 
      }
      
      // Run the user-supplied function (parallel section!)
      f(i, sample, q.data());
      
      // If supplied, check early-stopping condition
      #pragma omp critical
      {
        stop_flag = stop_check(i);
      }
    }
  }
}


template< std::floating_point F, LinearOperator Matrix, ThreadSafeRBG RBG >
void girard(
  const Matrix& mat,
  RBG& rbg, const int nv, const int dist, const int engine_id, const int seed,
  const F atol, const F rtol, 
  const int num_threads,
  const bool use_CLT, 
  F* estimates
){  
  using VectorF = Eigen::Array< F, Dynamic, 1>;

  // Save the sample estimates
  const auto save_sample = [&estimates](int i, F sample, [[maybe_unused]] F* q){
    estimates[i] = sample;
  };
  
  // Type-erased function since the call is cheap
  std::function< bool (int) > early_stop; 
  if (atol == 0.0 && rtol == 0.0){
    early_stop = [](int i) -> bool { return false; };
  } else if (use_CLT){
    // Parameterize when to stop using either the CLT over the given confidence level or 
    // This runs in critical section of the SLQ, so we can depend sequential execution (but i will vary!)
    // See: https://math.stackexchange.com/questions/102978/incremental-computation-of-standard-deviation
    F mu_est = 0.0, vr_est = 0.0; 
    F mu_pre = 0.0, vr_pre = 0.0; 
    int n = 0; // number of estimates computed
    const auto z = std::sqrt(2.0) * erf_inv(0.95);
    early_stop = [&estimates, &mu_est, &vr_est, &mu_pre, &vr_pre, &n, z, atol, rtol](int i) -> bool {
      ++n; 
      const F denom = (1.0 / F(n));
      const F L = n > 2 ? F(n-2) / F(n-1) : 0.0;
      mu_est = denom * (estimates[i] + (n - 1) * mu_pre);
      mu_pre = n == 1 ? mu_est : mu_pre;
      vr_est = L * vr_pre + denom * std::pow(estimates[i] - mu_pre, 2); // update sample variance
      mu_pre = mu_est;
      vr_pre = vr_est;
      if (n < 3){
        return false; 
      } else {
        const auto sd_est = std::sqrt(vr_est);
        const auto margin_of_error = z * sd_est / std::sqrt(F(n)); // todo: remove sqrt's 
        // std::cout << "n: " << n << ", mu: " << mu_est << ", ci: [" << mu_est - margin_of_error << ", " << mu_est + margin_of_error << "]";
        // std::cout << "margin/atol: " << margin_of_error << ", " << atol << "\n";
        return margin_of_error <= atol || std::abs(sd_est / mu_est) <= rtol;
      }
    };
  } else {
    // Use traditional iteration checking, akin to scipy.integrate.quadrature
    F mu_est = 0.0, mu_pre = 0.0;
    int n = 0; 
    early_stop = [&n, &estimates, &mu_est, &mu_pre, atol, rtol](int i) -> bool {
      ++n; 
      const F denom = (1.0 / F(n));
      mu_est = denom * (estimates[i] + (n - 1) * mu_pre);
      const bool atol_check = std::abs(mu_est - mu_pre) <= atol;
      const bool rtol_check = (std::abs(mu_est - mu_pre) / mu_est) <= rtol; 
      mu_pre = mu_est; 
      return atol_check || rtol_check;
    };
  }

  monte_carlo_quad(A, save_sample, early_stop, nv, dist, rbg, num_threads, seed);
}
  