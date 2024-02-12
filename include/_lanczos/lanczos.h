#ifndef _LANCZOS_LANCZOS_H
#define _LANCZOS_LANCZOS_H

#include <concepts> 
#include <functional> // function
#include <algorithm>  // max
#include "_operators/linear_operator.h" // LinearOperator
#include "_orthogonalize/orthogonalize.h"   // orth_vector, mod
// #include "_random_generator/vector_generator.h" // ThreadSafeRBG, generate_array

#include "omp_support.h" // conditionally enables openmp pragmas
#include "eigen_core.h"
#include <Eigen/Eigenvalues>
#include <iostream>

using std::function; 

template< typename T >
using AdjSolver = Eigen::SelfAdjointEigenSolver< T >;

// Krylov dimension 'deg' should be at least 1 and at most dimension of the operator
// Precondition: None
constexpr int param_deg(const int deg, const std::pair< size_t, size_t > dim){
  return std::max(1, std::min(deg, int(dim.first)));
} 

// Need to allocate at least 2 Lanczos vectors, should never need more than 
// Precondition: deg = param_deg(deg)
constexpr int param_ncv(const int ncv, const int deg, const std::pair< size_t, size_t > dim){
  return std::min(std::max(ncv, 2), std::min(deg, int(dim.first)));
} 

// Orth should be strictly less than ncv, as there are only ncv Lanczos vectors in memory
// If negative or larger than the Krylov dimension, orthogonalize against the maximal number of distinct Lanczos vectors 
// Precondition: deg = param_deg(deg) and ncv = param_ncv(ncv)
constexpr int param_orth(const int orth, const int deg, const int ncv, const std::pair< size_t, size_t > dim){
  if (orth < 0 || orth > deg){ return std::min(deg, ncv - 1); }
  return std::min(orth, ncv - 1); // should only orthogonalize against in-memory Lanczos vectors
} 

// Paige's A27 variant of the Lanczos method
// Computes the first k elements (a,b) := (alpha,beta) of the tridiagonal matrix T(a,b) where T = Q^T A Q
// Precondition: orth < ncv <= deg and ncv >= 2.
template< std::floating_point F, LinearOperator Matrix >
void lanczos_recurrence(
  const Matrix& A,            // Symmetric linear operator 
  F* q,                       // vector to expand the Krylov space K(A, q)
  const int deg,              // Dimension of the Krylov subspace to capture
  const F rtol,               // Tolerance of residual error for early-stopping the iteration.
  const int orth,             // Number of *additional* vectors to orthogonalize against 
  F* alpha,                   // Output diagonal elements of T of size A.shape[1]+1
  F* beta,                    // Output subdiagonal elements of T of size A.shape[1]+1; should be 0; 
  F* V,                       // Output matrix for Lanczos vectors (column-major)
  const size_t ncv            // Number of Lanczos vectors pre-allocated (must be at least 2)
){
  using VectorF = Eigen::Matrix< F, Dynamic, 1 >;

  // Constants `
  const auto A_shape = A.shape();
  const size_t n = A_shape.first;
  const size_t m = A_shape.second;
  const F residual_tol = std::sqrt(n) * rtol;

  // Setup views
  Eigen::Map< DenseMatrix< F > > Q(V, n, ncv);                // Lanczos vectors 
  Eigen::Map< VectorF > v(q, m, 1);                           // map initial vector (no-op)
  const auto Q_ref = Eigen::Ref< const DenseMatrix< F > >(Q); // const view 

  // Setup for first iteration
  std::array< int, 3 > pos = { int(ncv) - 1, 0, 1 };          // Indices for the recurrence
  Q.col(pos[0]) = static_cast< VectorF >(VectorF::Zero(n));   // Ensure previous is 0
  Q.col(0) = v.normalized();                                  // Load unit-norm v as q0
  beta[0] = 0.0;                                              // Ensure beta_0 is 0

  for (int j = 0; j < deg; ++j) {

    // Apply the three-term recurrence
    auto [p,c,n] = pos;                   // previous, current, next
    A.matvec(Q.col(c).data(), v.data());  // v = A q_c
    v -= beta[j] * Q.col(p);              // q_n = v - b q_p
    alpha[j] = Q.col(c).dot(v);           // projection size of < qc, qn > 
    v -= alpha[j] * Q.col(c);             // subtract projected components

    // Re-orthogonalize q_n against previous orth lanczos vectors, up to ncv-1
    if (orth > 0) {
      auto qn = Eigen::Ref< Vector< F > >(v);          
      orth_vector(qn, Q_ref, c, orth, true);
    }

    // Early-stop criterion is when K_j(A, v) is near invariant subspace.
    beta[j+1] = v.norm();
    if (beta[j+1] < residual_tol || (j+1) == deg) { // additional break prevents overriding qn
      break;
    }
    Q.col(n) = v / beta[j+1]; // normalize such that Q stays orthonormal

    // Cyclic left-rotate to update the working column indices
    std::rotate(pos.begin(), pos.begin() + 1, pos.end());
    pos[2] = mod(j+2, ncv);
  }
}

enum weight_method { golub_welsch = 0, fttr = 1 };

// NOTE: one idea to reduce memory is to use the matching moment idea
// - Take T - \lambda I for some eigenvalue \lambda; then dim(null(T- \lambda I)) = 1, thus 
// - we can solve for (T - \lambda I)x = 0 for x repeatedly, for all \lambda, and take x[0] to be the quadrature weight

// FTTR 
// Uses the Lanczos method to obtain Gaussian quadrature estimates of the spectrum of an arbitrary operator
template< std::floating_point F >
auto lanczos_quadrature(
  const F* alpha,                           // Input diagonal elements of T of size k
  const F* beta,                            // Output subdiagonal elements of T of size k, whose non-zeros start at index 1
  const int k,                              // Size of input / output elements 
  AdjSolver< DenseMatrix< F > >& solver,    // Solver to use. Assumes workspace has been allocated
  F* nodes,                                 // Output nodes of the quadrature
  F* weights,                               // Output weights of the quadrature
  const weight_method method = golub_welsch  // The method of computing the weights 
) -> void {
  using VectorF = Eigen::Array< F, Dynamic, 1>;
  assert(beta[0] == 0.0);
  
  Eigen::Map< const VectorF > a(alpha, k);        // diagonal elements
  Eigen::Map< const VectorF > b(beta+1, k-1);     // subdiagonal elements (offset by 1!)

  // Golub-Welsch approach: just compute eigen-decomposition from T using QR steps
  if (method == golub_welsch){
    // std::cout << " GW ";
    solver.computeFromTridiagonal(a, b, Eigen::DecompositionOptions::ComputeEigenvectors);
    auto theta = static_cast< VectorF >(solver.eigenvalues()); // Rayleigh-Ritz values == nodes
    auto tau = static_cast< VectorF >(solver.eigenvectors().row(0));
    tau *= tau;
    std::copy(theta.begin(), theta.end(), nodes);
    std::copy(tau.begin(), tau.end(), weights);
  } else {
    // std::cout << " FTTR ";
  // Uses the Foward Three Term Recurrence (FTTR) approach 
    solver.computeFromTridiagonal(a, b, Eigen::DecompositionOptions::EigenvaluesOnly);
    auto theta = static_cast< VectorF >(solver.eigenvalues()); // Rayleigh-Ritz values == nodes
    std::copy(theta.begin(), theta.end(), nodes);

    // Compute weights via FTTR
    FTTR_weights(theta.data(), alpha, beta, k, weights);
  }
};

template< std::floating_point F, LinearOperator Matrix > 
struct MatrixFunction {
  // static const bool owning = false;
  using value_type = F;
  using VectorF = Eigen::Matrix< F, Dynamic, 1 >;
  using ArrayF = Eigen::Array< F, Dynamic, 1 >;
  using EigenSolver = Eigen::SelfAdjointEigenSolver< DenseMatrix< F > >; 

  // Use non-owning class here rather than copy-constructor to make parallizing easier
  // See: https://stackoverflow.com/questions/35770357/storing-const-reference-to-an-object-in-class
  // Also see: https://zpjiang.me/2020/01/20/const-reference-vs-pointer/
  const Matrix& op; 
  // const Matrix op; 

  // Fields
  std::function< F(F) > f; 
  const int deg;
  const int ncv; 
  F rtol; 
  int orth;
  weight_method wgt_method; 
  std::function< void(F*) > transform;

  MatrixFunction(const Matrix& A, const std::function< F(F) > fun, int lanczos_degree, F lanczos_rtol, int _orth, int _ncv, weight_method _method = golub_welsch) 
  : op(A), f(fun), 
    deg(param_deg(lanczos_degree, A.shape())), 
    ncv(param_ncv(_ncv, deg, A.shape())), 
    rtol(lanczos_rtol), 
    orth(param_orth(_orth, deg, ncv, A.shape())), 
    wgt_method(_method) 
  {
    // Pre-allocate all but Q memory needed for Lanczos iterations
    alpha = static_cast< ArrayF >(ArrayF::Zero(deg+1));
    beta = static_cast< ArrayF >(ArrayF::Zero(deg+1));
    nodes = static_cast< ArrayF >(ArrayF::Zero(deg));
    weights = static_cast< ArrayF >(ArrayF::Zero(deg));
    solver = EigenSolver(deg);
    transform = [](F* v){ return; };
  };

  MatrixFunction(Matrix&& A, const std::function< F(F) > fun, int lanczos_degree, F lanczos_rtol, int _orth, int _ncv, weight_method _method = golub_welsch) 
  : op(std::move(A)), f(fun), 
    deg(param_deg(lanczos_degree, A.shape())), 
    ncv(param_ncv(_ncv, deg, A.shape())), 
    rtol(lanczos_rtol), 
    orth(param_orth(_orth, deg, ncv, A.shape())), 
    wgt_method(_method) {
    // Pre-allocate all but Q memory needed for Lanczos iterations
    alpha = static_cast< ArrayF >(ArrayF::Zero(deg+1));
    beta = static_cast< ArrayF >(ArrayF::Zero(deg+1));
    nodes = static_cast< ArrayF >(ArrayF::Zero(deg));
    weights = static_cast< ArrayF >(ArrayF::Zero(deg));
    solver = EigenSolver(deg);
    transform = [](F* v){ return; };
  };
 
  // Approximates v |-> f(A)v via a limited degree Lanczos iteration
  void matvec(const F* v, F* y) const noexcept {
    // if (op == nullptr){ return; }
    // Apply configured transform to the input vector
    transform(y);

    // By default, Q is not allocated in constructor, as quad may used less memory
    // For all calls after the first matvec(), Eigen promises this is a no-op
    // Note we *need* Q to have exactly deg columns for the matvec approx
    if (Q.cols() < deg){ Q.resize(op.shape().first, deg); }
  
    // Inputs / outputs 
    Eigen::Map< const VectorF > v_map(v, op.shape().second);
    Eigen::Map< VectorF > y_map(y, op.shape().first);
    
    // Lanczos iteration: save v norm 
    const F v_scale = v_map.norm(); 
    VectorF v_copy = v_map; // save copy
    // std::cout << v_copy << std::endl;
    lanczos_recurrence< F >(op, v_copy.data(), deg, rtol, orth, alpha.data(), beta.data(), Q.data(), deg); 
    
    // Note: Maps are used here to offset the pointers; they should be no-ops anyways
    Eigen::Map< ArrayF > a(alpha.data(), deg);      // diagonal elements
    Eigen::Map< ArrayF > b(beta.data()+1, deg-1);   // subdiagonal elements (offset by 1!)
    solver.computeFromTridiagonal(a, b, Eigen::DecompositionOptions::ComputeEigenvectors);

    // Apply the spectral function (in-place) to Rayleigh-Ritz values (nodes)
    auto theta = static_cast< ArrayF >(solver.eigenvalues());
    // theta.unaryExpr(f); // this doesn't always work for some reason
    std::transform(theta.begin(), theta.end(), theta.begin(), f);
    
    // The approximation v |-> f(A)v 
    const auto V = static_cast< DenseMatrix< F > >(solver.eigenvectors()); // maybe dont cast here 
    auto v_mod = static_cast< ArrayF >(V.row(0).array());
    v_mod *= theta;
    y_map = Q * static_cast< VectorF >(V * v_mod.matrix()); // equivalent to Q V diag(sf(theta)) V^T e_1
    y_map.array() *= v_scale; // re-scale
  }   

  void matmat(const F* X, F* Y, const size_t k) const noexcept {
    // if (op == nullptr){ return; }
    Eigen::Map< const DenseMatrix< F > > XM(X, op.shape().second, k);
    Eigen::Map< DenseMatrix< F > > YM(Y, op.shape().first, k);
    for (size_t j = 0; j < k; ++j){
      matvec(XM.col(j).data(), YM.col(j).data());
    }
  }

  // Approximates v^T A v
  auto quad(const F* v) const noexcept -> F {
    // if (op == nullptr){ return; }

    // Only allocate NCV columns as necessary -- no-op after first resizing
    if (Q.cols() < ncv){ Q.resize(op.shape().first, ncv); }

    // Save copy of v + its norm 
    Eigen::Map< const VectorF > v_map(v, op.shape().second);
    VectorF v_copy = v_map; // save copy; needs to be norm-1 for Lanczos + quad to work
    // const F v_scale = v_map.norm(); 
        
    // Apply configured transform to the input vector
    transform(v_copy.data());
    const F v_scale = v_copy.norm(); 
    v_copy.normalize();

    // Execute lanczos method + Golub-Welsch algorithm
    lanczos_recurrence< F >(op, v_copy.data(), deg, rtol, orth, alpha.data(), beta.data(), Q.data(), ncv);   
    lanczos_quadrature< F >(alpha.data(), beta.data(), deg, solver, nodes.data(), weights.data(), wgt_method);
  
    // Apply f to the nodes and sum
    std::transform(nodes.begin(), nodes.end(), nodes.begin(), f);
    // std::cout << "f(nodes): " << nodes << std::endl;
    // std::cout << "quad: " << F((nodes * weights).sum()) << std::endl; 
    // std::cout << "vscale**2 " << std::pow(v_scale, 2) << std::endl; 
    return std::pow(v_scale, 2) * (nodes * weights).sum();
  }

  // Returns (rows, columns)
  auto shape() const noexcept -> std::pair< size_t, size_t > {
    // if (op == nullptr){ return std::make_pair< size_t, size_t >(0, 0); }
    return op.shape();
  }

  // private: // Internal state to re-use
  mutable DenseMatrix< F > Q;
  mutable ArrayF alpha;
  mutable ArrayF beta;
  mutable ArrayF nodes;
  mutable ArrayF weights;
  mutable EigenSolver solver;  
};

// Approximates the action v |-> f(A)v via the Lanczos method
template< std::floating_point F, LinearOperator Matrix > 
void matrix_approx(
  const Matrix& A,                            // LinearOperator 
  std::function< F(F) > sf,             // the spectral function 
  const F* v,                                 // the input vector
  const int lanczos_degree,                   // Polynomial degree of the Krylov expansion
  const F lanczos_rtol,                       // residual tolerance to consider subspace A-invariant
  const int orth,                             // Number of vectors to re-orthogonalize against <= lanczos_degree
  F* y                                        // Output vector
){
  MatrixFunction< F, Matrix >(A, sf, lanczos_degree, lanczos_rtol, orth, lanczos_degree).matvec(v, y);
};

#endif 