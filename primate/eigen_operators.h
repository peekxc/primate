#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <Eigen/SparseCore> // SparseMatrix, Matrix
namespace py = pybind11;

// TODO: store only lower/upper part for symmetric? http://www.eigen.tuxfamily.org/dox/group__TutorialSparse.html
template< std::floating_point F >
struct SparseEigenLinearOperator {
  using value_type = F;
  const Eigen::SparseMatrix< F > A;  
  SparseEigenLinearOperator(const Eigen::SparseMatrix< F >& _mat) : A(_mat){}

  void matvec(const F* inp, F* out) const noexcept {
    auto input = Eigen::Map< const Eigen::Matrix< F, Eigen::Dynamic, 1 > >(inp, A.cols(), 1); // this should be a no-op
    auto output = Eigen::Map< Eigen::Matrix< F, Eigen::Dynamic, 1 > >(out, A.rows(), 1); // this should be a no-op
    output = A * input; 
  }

  void rmatvec(const F* inp, F* out) const noexcept {
    auto input = Eigen::Map< const Eigen::Matrix< F, Eigen::Dynamic, 1 > >(inp, A.rows(), 1); // this should be a no-op
    auto output = Eigen::Map< Eigen::Matrix< F, Eigen::Dynamic, 1 > >(out, A.cols(), 1); // this should be a no-op
    output = A.adjoint() * input; 
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