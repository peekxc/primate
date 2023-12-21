#ifndef _TRACE_HUTCH_H
#define _TRACE_HUTCH_H

#include <concepts> 
#include <functional> // function
#include <algorithm>  // max
#include <cmath> // constants, isnan
#include "omp_support.h" // conditionally enables openmp pragmas
#include "_operators/linear_operator.h" // LinearOperator
#include "_orthogonalize/orthogonalize.h"   // orth_vector, mod
#include "_random_generator/vector_generator.h" // ThreadSafeRBG, generate_array, Distribution
#include "_lanczos/lanczos.h"
#include "eigen_core.h"
#include <Eigen/Eigenvalues> 
#include <iostream> 


template < int Iterate = 3 >
double erf_inv(double x) noexcept {
  // Strategy: solve f(y) = x - erf(y) = 0 for given x with Newton's method.
  // f'(y) = -erf'(y) = -2/sqrt(pi) e^(-y^2)
  // Has quadratic convergence, achieving machine precision with ~three iterations.
  if constexpr(Iterate == 0){
    // Specialization to get initial estimate; accurate to about 1e-3.
    // Based on https://stackoverflow.com/questions/27229371/inverse-error-function-in-c
    const double a = std::log((1 - x) * (1 + x));
    const double b = std::fma(0.5, a, 4.120666747961526);
    const double c = 6.47272819164 * a;
    return std::copysign(std::sqrt(-b + std::sqrt(std::fma(b, b, -c))), x);
  } else {
    const double x0 = erf_inv< Iterate - 1 >(x); // compile-time recurse
    const double fx0 = x - std::erf(x0);
    const double pi = std::acos(-1);
    double fpx0 = -2.0 / std::sqrt(pi) * std::exp(-x0 * x0);
    return x0 - fx0 / fpx0; // = x1
  } 
}

// Work-around to avoid copying for the multi-threaded build
template< LinearOperator Matrix > 
auto get_matrix(const Matrix& A){
  // const auto M = is_instance< Matrix, MatrixFunction >{} ? 
  if constexpr(HasOp< Matrix >){
    return MatrixFunction(A.op, A.f, A.deg, A.rtol, A.orth, A.ncv);
  } else {
    return A; 
  } 
}
   
// std::function< bool(int) >
template< std::floating_point F, LinearOperator Matrix, ThreadSafeRBG RBG, typename Lambda, typename Lambda2 >
void monte_carlo_quad(
  const Matrix& A,                              // Any linear operator supporting .matvec() and .shape() methods
  const Lambda& f_cb,                           // Thread-safe callback function with signature f(int i, F sample, F* q)
  const Lambda2& stop_check, // Function to check for convergence or early-stopping (takes no arguments)
  const int nv,                                 // Number of sample vectors to generate
  const Distribution dist,                      // Isotropic distribution used to generate random vectors
  RBG& rng,                                     // Random bit generator
  const int num_threads,                        // Number of threads used to parallelize the computation   
  const int seed                                // Seed for random number generator for determinism
){
  using VectorF = Eigen::Matrix< F, Dynamic, 1 >;
  // using ArrayF = Eigen::Array< F, Dynamic, 1 >;
  
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
  #pragma omp parallel shared(A, stop_flag)
  {
    int tid = omp_get_thread_num(); // thread-id 
    
    auto q_norm = static_cast< F >(0.0);
    auto q = static_cast< VectorF >(VectorF::Zero(m));
    
    // Parameterize the matrix function
    // TODO: this is a somewhat poorly designed way to get around copying matrices / matrix functions, 
    // as the former is read-only and can be shared amongst threads, but the latter needs thread-specific storage
    const auto M = get_matrix(A); 

    #pragma omp for schedule(dynamic, chunk_size)
    for (i = 0; i < nv; ++i){
      if (stop_flag){ continue; }

      // Generate isotropic vector (w/ unit norm)
      generate_isotropic< F >(dist, m, rng, tid, q.data(), q_norm);
      
      // Apply quadratic form, using quad() if available
      F sample = 0.0;  
      if constexpr (QuadOperator< Matrix >){
        sample = M.quad(q.data()); // x^T A x
      } else {
        auto y = static_cast< VectorF >(VectorF::Zero(n));
        A.matvec(q.data(), y.data()); 
        sample = y.dot(q);
      }
      
      // Run the user-supplied function (parallel section!)
      f_cb(i, sample, q.data());
      
      // If supplied, check early-stopping condition
      #pragma omp critical
      {
        stop_flag = stop_check(i);
      }
    } // omp for
  } // omp parallel 
}

template< std::floating_point F, LinearOperator Matrix, ThreadSafeRBG RBG >
auto hutch(
  const Matrix& A,
  RBG& rbg, const int nv, const int dist, const int engine_id, const int seed,
  const F atol, const F rtol, 
  const int num_threads,
  const bool use_CLT, 
  F* estimates
) -> F {  
  using ArrayF = Eigen::Array< F, Dynamic, 1 >;
  
  // Save the sample estimates
  const auto save_sample = [&estimates](int i, F sample, [[maybe_unused]] F* q){
    estimates[i] = sample;
  };
  
  // Type-erased function since the call is cheap
  // std::function< bool (int) > early_stop; 
  if (atol == 0.0 && rtol == 0.0){
    const auto early_stop = [](int i) -> bool { return false; };
    monte_carlo_quad< F >(A, save_sample, early_stop, nv, static_cast< Distribution >(dist), rbg, num_threads, seed);
    F mu_est = 0.0;

    Eigen::Map< ArrayF > est(estimates, nv);
    est *= A.shape().first;
    mu_est = est.sum() / nv;  
    return mu_est;
  } else if (use_CLT){
    // Parameterize when to stop using either the CLT over the given confidence level or 
    // This runs in critical section of the SLQ, so we can depend sequential execution (but i will vary!)
    // See: https://math.stackexchange.com/questions/102978/incremental-computation-of-standard-deviation
    F mu_est = 0.0, vr_est = 0.0; 
    F mu_pre = 0.0, vr_pre = 0.0; 
    int n_samples = 0; // number of estimates computed
    const auto z = std::sqrt(2.0) * erf_inv< 3 >(double(0.95));
    const auto early_stop = [&estimates, &mu_est, &vr_est, &mu_pre, &vr_pre, &n_samples, z, atol, rtol](int i) -> bool {
      if (isnan(estimates[i])){ return false; }
      ++n_samples; 
      const F denom = (1.0 / F(n_samples));
      const F L = n_samples > 2 ? F(n_samples-2) / F(n_samples-1) : 0.0;
      mu_est = denom * (estimates[i] + (n_samples - 1) * mu_pre);
      mu_pre = n_samples == 1 ? mu_est : mu_pre;
      vr_est = L * vr_pre + denom * std::pow(estimates[i] - mu_pre, 2); // update sample variance
      mu_pre = mu_est;
      vr_pre = vr_est;
      if (n_samples < 3){
        return false; 
      } else {
        const auto sd_est = std::sqrt(vr_est);
        const auto margin_of_error = z * sd_est / std::sqrt(F(n_samples)); // todo: remove sqrt's 
        // std::cout << "n: " << n << ", mu: " << mu_est << ", ci: [" << mu_est - margin_of_error << ", " << mu_est + margin_of_error << "]";
        // std::cout << "margin/atol: " << margin_of_error << ", " << atol << "\n";
        return margin_of_error <= atol || std::abs(sd_est / mu_est) <= rtol;
      }
    };
    monte_carlo_quad< F >(A, save_sample, early_stop, nv, static_cast< Distribution >(dist), rbg, num_threads, seed);
    Eigen::Map< ArrayF > est(estimates, nv);
    est *= A.shape().first;
    return mu_est * A.shape().first;
  } else {
    // Use traditional iteration checking, akin to scipy.integrate.quadrature
    F mu_est = 0.0, mu_pre = 0.0;
    int n_samples = 0; 
    const auto early_stop = [&estimates, &n_samples, &mu_est, &mu_pre, atol, rtol](int ii) -> bool { // &estimates, &n_samples, &mu_est, &mu_pre, atol, rtol
      if (isnan(estimates[ii])){ return false; }
      ++n_samples; 
      const F denom = (1.0 / F(n_samples));
      mu_est = denom * (estimates[ii] + (n_samples - 1) * mu_pre);
      const bool atol_check = std::abs(mu_est - mu_pre) <= atol;
      const bool rtol_check = (std::abs(mu_est - mu_pre) / mu_est) <= rtol; 
      mu_pre = mu_est;
      // std::cout << std::fixed << std::showpoint;
      // std::cout << std::setprecision(10);
      // std::cout << "n: " << n_samples << ", mu: " << mu_est << ", atol: " << std::abs(mu_est - mu_pre) << ", rtol: " << (std::abs(mu_est - mu_pre) / mu_est) << std::endl; 
      return atol_check || rtol_check;
    };
    monte_carlo_quad< F >(A, save_sample, early_stop, nv, static_cast< Distribution >(dist), rbg, num_threads, seed);
    Eigen::Map< ArrayF > est(estimates, nv);
    est *= A.shape().first;
    return mu_est * A.shape().first; 
  }
  // Don't pull down monte_carlo_quad; the early-stop defs need to be local scope for some reason!
}

#endif