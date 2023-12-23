#ifndef EIGEN_OPERATORS_H
#define EIGEN_OPERATORS_H

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <eigen_core.h> // Vector, Array, DenseMatrix
#include <Eigen/SparseCore> // SparseMatrix, Matrix
#include <chrono>
namespace py = pybind11;

using us = std::chrono::microseconds;
using dur_seconds = std::chrono::duration< double >;
using hr_clock = std::chrono::high_resolution_clock;

// ## TODO: use CRTP to craft a set of template classes

// Storing const should be safe for parallel execution, right?
template< std::floating_point F >
struct DenseEigenLinearOperator {
  using value_type = F;
  const DenseMatrix< F > A;  
  mutable size_t matvec_time; 
  DenseEigenLinearOperator(const DenseMatrix< F > _mat) : A(_mat), matvec_time(0.0){}

  void matvec(const F* inp, F* out) const noexcept {
    auto ts = hr_clock::now();
    auto input = Eigen::Map< const Vector< F > >(inp, A.cols(), 1); // this should be a no-op
    auto output = Eigen::Map< Vector< F > >(out, A.rows(), 1); // this should be a no-op
    output = A * input; 
    matvec_time += duration_cast< us >(dur_seconds(hr_clock::now() - ts)).count();
  }

  void rmatvec(const F* inp, F* out) const noexcept {
    auto ts = hr_clock::now();
    auto input = Eigen::Map< const Vector< F > >(inp, A.rows(), 1); // this should be a no-op
    auto output = Eigen::Map< Vector< F > >(out, A.cols(), 1); // this should be a no-op
    output = A.adjoint() * input; 
    matvec_time += duration_cast< us >(dur_seconds(hr_clock::now() - ts)).count();
  }

  void matmat(const F* X, F* Y, const size_t k) const noexcept {
    auto ts = hr_clock::now();
    Eigen::Map< const DenseMatrix< F > > XM(X, A.cols(), k);
    Eigen::Map< DenseMatrix< F > > YM(Y, A.rows(), k);
    YM = A * XM;
    matvec_time += duration_cast< us >(dur_seconds(hr_clock::now() - ts)).count();
  }

  auto shape() const noexcept -> std::pair< size_t, size_t > {
    return std::make_pair((size_t) A.rows(), (size_t) A.cols());
  }

  // auto quad(const F* inp) const noexcept -> F {
  //   auto input = Eigen::Map< const Vector< F > >(inp, A.rows(), 1); //  no-op
  // }
};

// TODO: store only lower/upper part for symmetric? http://www.eigen.tuxfamily.org/dox/group__TutorialSparse.html
template< std::floating_point F >
struct SparseEigenLinearOperator {
  using value_type = F;
  const Eigen::SparseMatrix< F > A;  
  mutable size_t matvec_time; 

  SparseEigenLinearOperator(const Eigen::SparseMatrix< F > _mat) : A(_mat){}

  void matvec(const F* inp, F* out) const noexcept {
    auto ts = hr_clock::now();
    auto input = Eigen::Map< const Eigen::Matrix< F, Eigen::Dynamic, 1 > >(inp, A.cols(), 1); // this should be a no-op
    auto output = Eigen::Map< Eigen::Matrix< F, Eigen::Dynamic, 1 > >(out, A.rows(), 1); // this should be a no-op
    output = A * input; 
    matvec_time += duration_cast< us >(dur_seconds(hr_clock::now() - ts)).count();
  }

  void rmatvec(const F* inp, F* out) const noexcept {
    auto ts = hr_clock::now();
    auto input = Eigen::Map< const Eigen::Matrix< F, Eigen::Dynamic, 1 > >(inp, A.rows(), 1); // this should be a no-op
    auto output = Eigen::Map< Eigen::Matrix< F, Eigen::Dynamic, 1 > >(out, A.cols(), 1); // this should be a no-op
    output = A.adjoint() * input; 
    matvec_time += duration_cast< us >(dur_seconds(hr_clock::now() - ts)).count();
  }

  void matmat(const F* X, F* Y, const size_t k) const noexcept {
    auto ts = hr_clock::now();
    Eigen::Map< const DenseMatrix< F > > XM(X, A.cols(), k);
    Eigen::Map< DenseMatrix< F > > YM(Y, A.rows(), k);
    YM = A * XM;
    matvec_time += duration_cast< us >(dur_seconds(hr_clock::now() - ts)).count();
  }

  auto shape() const noexcept -> std::pair< size_t, size_t > {
    return std::make_pair((size_t) A.rows(), (size_t) A.cols());
  }
};

template< std::floating_point F >
struct SparseEigenAffineOperator {
  using value_type = F;
  static constexpr bool relation_known = false; 
  const Eigen::SparseMatrix< F > A;  
  const Eigen::SparseMatrix< F > B;  
  mutable F _param; 
  mutable size_t n_matvecs; 

  SparseEigenAffineOperator(
    const Eigen::SparseMatrix< F >& _A,
    const Eigen::SparseMatrix< F >& _B 
  ) : A(_A), B(_B), n_matvecs(0) {
    _param = 0.0f;
  }

  // Uses Eigen basically as a no-overhead call to BLAS: https://eigen.tuxfamily.org/dox/group__TutorialMapClass.html
  void matvec(const F* inp, F* out) const noexcept {
    auto input = Eigen::Map< const Eigen::Matrix< F, Eigen::Dynamic, 1 > >(inp, A.cols(), 1); // this should be a no-op
    auto output = Eigen::Map< Eigen::Matrix< F, Eigen::Dynamic, 1 > >(out, A.rows(), 1);      // this should be a no-op
    output = (A + _param * B) * input; // This is 100x faster than copying!
    n_matvecs++;
  }

  auto shape() const noexcept -> std::pair< size_t, size_t > {
    return std::make_pair((size_t) A.rows(), (size_t) A.cols());
  }

  void set_parameter(F t) const {
    _param = t; 
  }
};

// template< std::floating_point F >
// auto eigen_sparse_wrapper(const Eigen::SparseMatrix< F >* A){
//   return SparseEigenLinearOperator< F >(*A);
// }

// // TODO: Support Adjoint and Affine Operators out of the box
// template< std::floating_point F >
// auto eigen_sparse_affine_wrapper(const Eigen::SparseMatrix< F >* A){
//   auto B = Eigen::SparseMatrix< F >(A->rows(), A->cols());
//   B.setIdentity();
//   return SparseEigenAffineOperator< F >(*A, B);
// }

// template< std::floating_point F >
// auto eigen_dense_wrapper(const Eigen::Matrix< F, Eigen::Dynamic, Eigen::Dynamic >* A){
//   return DenseEigenLinearOperator< F >(*A);
// }

// template< std::floating_point F >
// auto linearoperator_wrapper(const py::object* A){
//   return PyLinearOperator< F >(*A);
// }

#endif 