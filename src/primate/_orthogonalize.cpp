#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "_orthogonalize/orthogonalize.h"
#include <iomanip>
#include <iostream>
#include <vector>
#include <pybind11/eigen.h>

namespace py = pybind11;

// Note we enforce fortran style ordering here
template< std::floating_point F > 
using py_array = py::array_t< F, py::array::f_style | py::array::forcecast >;

using std::vector; 

// template< std::floating_point F >
// auto orth_poly(F x, int i, F mu_sqrt, const F* a, const F* b, const int n) noexcept -> F {
//   if (i < 0){ 
//     return 0.0; 
//   } else if (i == 0){
//     return 1 / mu_sqrt;
//   } else if (i == 1){
//     return (x - a[0]) * (1 / mu_sqrt) / b[1];
//   } else if (i < n){
//     F z = (x - a[i-1]) * orth_poly(x, i - 1, mu_sqrt, a, b, n);
//     z -= b[i-1] * orth_poly(x, i - 2, mu_sqrt, a, b, n);
//     z /= b[i];
//     return z; 
//   } else {
//     return 0; 
//   }
// }


// Compute weights via FTTR 
// Debug update: Algorithm is correct, just unstable for lower precision types! 
// Returns a vector p(x) of orthogonal polynomials representing the recurrence 
// Jp(x) = x p(x) - \gamma_n p_n(x) e_n
// where J is the Jacobi / tridiagonal matrix
// Parameters: 
//   x: value to evaluate polynomial at. Typically an eigenvalue. 
//   mu_0: sum of the eigenvalues of J.
//   a: diagonal values of J. 
//   b: off-diagonal values of J
//   z: output vector. 
// Weights should be populated with mu_0 * U[0,:]**2
template< std::floating_point F >
auto fttr(const Array< F >& ew, const Array< F >& a, const Array< F >& b) -> Array< F > {
  assert(ew.size() == a.size());
  const F* theta = ew.data();
  const size_t k = ew.size();
  const F mu_0 = ew.abs().sum();
  const F mu_sqrt_rec = 1.0 / std::sqrt(mu_0);
  // std::cout << std::fixed << std::showpoint;
  // std::cout << std::setprecision(15);
  // std::cout << "---- TEST --- ( " << sizeof(F) << ")" << std::endl;
  // std::cout << "a: " << a.matrix().transpose() << std::endl;
  // std::cout << "b: " << b.matrix().transpose() << std::endl;
  // std::cout << "ew: " << ew.matrix().transpose() << std::endl;
  // std::cout << "mu_0: " << mu_0 << std::endl;
  Array< F > p(a.size());
  Array< F > weights(a.size());
  for (size_t i = 0; i < k; ++i){
    poly(theta[i], mu_sqrt_rec, a.data(), b.data(), p.data(), a.size());
    F weight = 1.0 / p.square().sum();
    // std::cout << i << ": (x: " << theta[i] << ", w: " << weight << ") p: " << p.matrix().transpose() << std::endl;
    weights[i] = weight / mu_0; 
  }
  return weights; 
}

template< std::floating_point F >
void _orthogonalize(py::module &m){
  m.def("mgs", &modified_gram_schmidt< F >);
  m.def("orth_vector", &orth_vector< F >);
  // m.def("fttr", &fttr< F >);
}

PYBIND11_MODULE(_orthogonalize, m) {
  m.doc() = "orthogonalization module"; 
  _orthogonalize< float >(m);
  _orthogonalize< double >(m);
}