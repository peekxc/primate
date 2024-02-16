#ifndef _TRACE_HUTCH_H
#define _TRACE_HUTCH_H

#include <concepts> 
#include <functional> // function
#include <algorithm>  // max
#include <cmath> // constants, isnan
#include "omp_support.h" // conditionally enables openmp pragmas
// #include "_operators/linear_operator.h" // LinearOperator
// #include "_orthogonalize/orthogonalize.h"   // orth_vector, mod
#include "_random_generator/vector_generator.h" // ThreadSafeRBG, generate_array, Distribution
#include "_lanczos/lanczos.h"
#include "eigen_core.h"
#include <Eigen/Eigenvalues> 
#include <iostream> 
#include <chrono>

using us = std::chrono::microseconds;
using dur_seconds = std::chrono::duration< double >;
using hr_clock = std::chrono::high_resolution_clock;

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
    return MatrixFunction(A.op, A.f, A.deg, A.rtol, A.orth, A.ncv, A.wgt_method);
  } else {
    return A; 
  } 
}

auto get_num_threads(int nt) -> int {
  const auto max_threads = omp_get_max_threads();
  nt = nt <= 0 ? max_threads : nt; 
  return std::max(1, int(nt));
}   

// std::function< bool(int) >
template< std::floating_point F, LinearOperator Matrix, ThreadSafeRBG RBG, typename Lambda, typename Lambda2 >
void monte_carlo_quad(
  const Matrix& A,                              // Any linear operator supporting .matvec() and .shape() methods
  const Lambda& f_cb,                           // Thread-safe callback function with signature f(int i, F sample, F* q)
  const Lambda2& stop_check,                    // Function to check for convergence or early-stopping (takes no arguments)
  const int nv,                                 // Number of sample vectors to generate
  const Distribution dist,                      // Isotropic distribution used to generate random vectors
  RBG& rng,                                     // Random bit generator
  const int num_threads,                        // Number of threads used to parallelize the computation   
  const int seed,                               // Seed for random number generator for determinism
  size_t& wall_time                             // Wall clock time
){
  using VectorF = Eigen::Matrix< F, Dynamic, 1 >;
  // using ArrayF = Eigen::Array< F, Dynamic, 1 >;
  
  // Constants
  const auto A_shape = A.shape();
  const size_t n = A_shape.first;
  const size_t m = A_shape.second;

  // Set the number of threads + initialize multi-threaded RNG
  const auto nt = get_num_threads(num_threads);
  omp_set_num_threads(nt);
  rng.initialize(nt, seed);

  // Using square-root of max possible chunk size for parallel schedules
  unsigned int chunk_size = std::max(int(sqrt(nv / nt)), 1);
  
  // Monte-Carlo ensemble sampling
  int i;
  volatile bool stop_flag = false; // early-stop flag for convergence checking
  const auto t_start = hr_clock::now();
  #pragma omp parallel shared(A, stop_flag)
  {
    int tid = omp_get_thread_num(); // thread-id 
    
    auto q_norm = static_cast< F >(0.0);
    auto q = static_cast< VectorF >(VectorF::Zero(m));
    
    // Parameterize the matrix function
    // TODO: this is not a great way to avoid copying matrices / matrix functions, but is needed as 
    // the former is read-only and can be shared amongst threads, but the latter needs thread-specific storage
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
        // std::cout << "quad output: " << sample << std::endl;
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
  const auto t_end = hr_clock::now();
  wall_time = duration_cast< us >(dur_seconds(t_end - t_start)).count();
}

template< std::floating_point F, LinearOperator Matrix, ThreadSafeRBG RBG >
auto hutch(
  const Matrix& A,
  RBG& rbg, const int nv, const int dist, const int engine_id, const int seed,
  const F atol, const F rtol, 
  const int num_threads,
  const bool use_CLT, 
  const F* t_scores, 
  const F z, 
  F* estimates, 
  size_t& wall_time
) -> F {  
  using ArrayF = Eigen::Array< F, Dynamic, 1 >;
  
  // Save the sample estimates
  const auto save_sample = [&estimates](int i, F sample, [[maybe_unused]] F* q){
    estimates[i] = sample;
  };
  
  // Type-erased function since the call is cheap
  // std::function< bool (int) > early_stop; // apparently this will cause problems!
  if (atol == 0.0 && rtol == 0.0){
    const auto early_stop = [](int i) -> bool { return false; };
    monte_carlo_quad< F >(A, save_sample, early_stop, nv, static_cast< Distribution >(dist), rbg, num_threads, seed, wall_time);
    Eigen::Map< ArrayF > est(estimates, nv);
    est *= A.shape().first;
    // std::cout << "est sum: " << est.sum() << ", nv: " << nv << std::endl; 
    F mu_est = est.sum() / nv;  
    return mu_est;
  } else if (use_CLT){
    // Parameterize when to stop using either the CLT over the given confidence level or 
    // This runs in critical section of the SLQ, so we can depend sequential execution (but i will vary!)
    // See: https://math.stackexchange.com/questions/102978/incremental-computation-of-standard-deviation
    F mu_est = 0.0, vr_est = 0.0; 
    F mu_pre = 0.0, vr_pre = 0.0; 
    int n_samples = 0; // number of estimates computed
    // const auto z = std::sqrt(2.0) * erf_inv< 3 >(double(0.95));
    const auto early_stop = [&estimates, &mu_est, &vr_est, &mu_pre, &vr_pre, &n_samples, &t_scores, z, atol, rtol](int i) -> bool {
      if (std::isnan(estimates[i])){ return false; }
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
        const auto score = n_samples < 30 ? t_scores[n_samples] : z;
        const auto margin_of_error = score * sd_est / std::sqrt(F(n_samples)); // todo: remove sqrt's 
        // std::cout << "n: " << n << ", mu: " << mu_est << ", ci: [" << mu_est - margin_of_error << ", " << mu_est + margin_of_error << "]";
        // std::cout << "margin/atol: " << margin_of_error << ", " << atol << "\n";
        return margin_of_error <= atol || std::abs(sd_est / mu_est) <= rtol;
      }
    };
    monte_carlo_quad< F >(A, save_sample, early_stop, nv, static_cast< Distribution >(dist), rbg, num_threads, seed, wall_time);
    Eigen::Map< ArrayF > est(estimates, nv);
    est *= A.shape().first;
    return mu_est * A.shape().first;
  } else {
    // Use traditional iteration checking, akin to scipy.integrate.quadrature
    F mu_est = 0.0, mu_pre = 0.0;
    int n_samples = 0; 
    const auto early_stop = [&estimates, &n_samples, &mu_est, &mu_pre, atol, rtol](int ii) -> bool { // &estimates, &n_samples, &mu_est, &mu_pre, atol, rtol
      if (std::isnan(estimates[ii])){ return false; }
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
    monte_carlo_quad< F >(A, save_sample, early_stop, nv, static_cast< Distribution >(dist), rbg, num_threads, seed, wall_time);
    Eigen::Map< ArrayF > est(estimates, nv);
    est *= A.shape().first;
    return mu_est * A.shape().first; 
  }
  // Don't pull down monte_carlo_quad; the early-stop defs need to be local scope for some reason!
}


// Stochastic Lanczos quadrature method
// std::function<F(int,F*,F*)>
template< std::floating_point F, LinearOperator Matrix, ThreadSafeRBG RBG, typename Lambda >
void slq (
  const Matrix& A,                            // Any linear operator supporting .matvec() and .shape() methods
  const Lambda& f,                            // Thread-safe function with signature f(int i, F* nodes, F* weights)
  const std::function< bool(int) >& stop_check, // Function to check for convergence or early-stopping (takes no arguments)
  const int nv,                               // Number of sample vectors to generate
  const Distribution dist,                    // Isotropic distribution used to generate random vectors
  RBG& rng,                                   // Random bit generator
  const int lanczos_degree,                   // Polynomial degree of the Krylov expansion
  const F lanczos_rtol,                       // residual tolerance to consider subspace A-invariant
  const int orth,                             // Number of vectors to re-orthogonalize against <= lanczos_degree
  const int ncv,                              // Number of Lanczos vectors to keep in memory (per-thread)
  const int num_threads,                      // Number of threads used to parallelize the computation   
  const int seed                              // Seed for random number generator for determinism
){   
  using ArrayF = Eigen::Array< F, Dynamic, 1 >;
  using EigenSolver = Eigen::SelfAdjointEigenSolver< DenseMatrix< F > >; 
  if (ncv < 2){ throw std::invalid_argument("Invalid number of lanczos vectors supplied; must be >= 2."); }
  if (ncv < orth+2){ throw std::invalid_argument("Invalid number of lanczos vectors supplied; must be >= 2+orth."); }

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

    // Pre-allocate memory needed for Lanczos iterations
    auto q_norm = static_cast< F >(0.0);
    auto q = static_cast< ArrayF >(ArrayF::Zero(m)); 
    auto Q = static_cast< DenseMatrix< F > >(DenseMatrix< F >::Zero(n, ncv));
    auto alpha = static_cast< ArrayF >(ArrayF::Zero(lanczos_degree+1));
    auto beta = static_cast< ArrayF >(ArrayF::Zero(lanczos_degree+1));
    auto nodes = static_cast< ArrayF >(ArrayF::Zero(lanczos_degree));
    auto weights = static_cast< ArrayF >(ArrayF::Zero(lanczos_degree));
    auto solver = EigenSolver(lanczos_degree);
    
    // Run in parallel 
    #pragma omp for schedule(dynamic, chunk_size)
    for (i = 0; i < nv; ++i){
      if (stop_flag){ continue; }

      // Generate isotropic vector (w/ unit norm)
      generate_isotropic< F >(dist, m, rng, tid, q.data(), q_norm);
      
      // Perform a lanczos iteration (populates alpha, beta)
      lanczos_recurrence< F >(A, q.data(), lanczos_degree, lanczos_rtol, orth, alpha.data(), beta.data(), Q.data(), ncv); 

      // Obtain nodes + weights via quadrature algorithm
      lanczos_quadrature< F >(alpha.data(), beta.data(), lanczos_degree, solver, nodes.data(), weights.data());

      // Run the user-supplied function (parallel section!)
      f(i, q.data(), Q.data(), nodes.data(), weights.data());
      
      // If supplied, check early-stopping condition
      #pragma omp critical
      {
        stop_flag = stop_check(i);
      }
    }
  }
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
  constexpr auto early_stop = [](int i) -> bool {
    return false; 
  };

  // Execute the stochastic Lanczos quadrature
  slq< F >(mat, quad_f, early_stop, nv, static_cast< Distribution >(dist), rbg, lanczos_degree, lanczos_rtol, orth, ncv, num_threads, seed);
}

#endif