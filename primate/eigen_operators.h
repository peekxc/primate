#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <Eigen/SparseCore> // SparseMatrix, Matrix
namespace py = pybind11;

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

  auto shape() const noexcept -> std::pair< size_t, size_t > {
    return std::make_pair((size_t) A.rows(), (size_t) A.cols());
  }
};

template< std::floating_point F >
struct SparseEigenAffineOperator {
  using value_type = F;
  
  const Eigen::SparseMatrix< F > A;  
  const Eigen::SparseMatrix< F > B;  
  mutable std::vector< F > _params; 
  mutable size_t n_matvecs; 

  SparseEigenAffineOperator(
    const Eigen::SparseMatrix< F >& _A,
    const Eigen::SparseMatrix< F >& _B,
    const size_t num_params 
  ) : A(_A), B(_B), n_matvecs(0) {
    _params = std::vector< F >(num_params, 0.0f);
  }

  // Uses Eigen basically as a no-overhead call to BLAS: https://eigen.tuxfamily.org/dox/group__TutorialMapClass.html
  void matvec(const F* inp, F* out) const noexcept {
    auto input = Eigen::Map< const Eigen::Matrix< F, Eigen::Dynamic, 1 > >(inp, A.cols(), 1); // this should be a no-op
    auto output = Eigen::Map< Eigen::Matrix< F, Eigen::Dynamic, 1 > >(out, A.rows(), 1);      // this should be a no-op
    output = A * input; // This is 100x faster than copying!
    n_matvecs++;
  }

  auto shape() const noexcept -> std::pair< size_t, size_t > {
    return std::make_pair((size_t) A.rows(), (size_t) A.cols());
  }

  void set_parameters(F* params) const {
    std::copy(params, params + _params.size(), _params.begin());
  }

  auto get_num_parameters() const -> size_t {
    return _params.size();
  }
};