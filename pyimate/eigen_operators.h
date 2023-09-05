#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <Eigen/SparseCore> // SparseMatrix

namespace py = pybind11;
// using py_arr_f = py::array_t< float >;

template< std::floating_point F >
struct SparseEigenLinearOperator {
  using value_type = F;
  const Eigen::SparseMatrix< F > mat;  
  SparseEigenLinearOperator(const Eigen::SparseMatrix< F >& _mat) : mat(_mat){}

  void matvec(const F* inp, F* out){
    auto input = Eigen::VectorXf(mat.cols());
    std::copy(inp, inp + input.size(), input.begin());  
    auto output = mat * input;
    std::copy(output.begin(), output.end(), out);
  }

  auto shape() -> std::pair< size_t, size_t > {
    return std::make_pair((size_t) mat.rows(), (size_t) mat.cols());
  }
};

template< std::floating_point F >
struct SparseEigenAffineOperator {
  using value_type = F;
  
  const Eigen::SparseMatrix< F > A;  
  const Eigen::SparseMatrix< F > B;  
  std::vector< F > _params; 

  SparseEigenAffineOperator(
    const Eigen::SparseMatrix< F >& _A,
    const Eigen::SparseMatrix< F >& _B,
    const size_t num_params 
  ) : A(_A), B(_B) {
    _params = std::vector< F >(num_params, 0.0f);
  }

  // Or possibly: https://eigen.tuxfamily.org/dox/group__TutorialMapClass.html
  // TODO: use aliasing to avoid the copies, see: https://eigen.tuxfamily.org/dox/group__TutorialMatrixArithmetic.html
  void matvec(const F* inp, F* out){
    auto input = Eigen::VectorXf(A.cols());
    std::copy(inp, inp + A.cols(), input.begin());  
    auto output = A * input;
    std::copy(output.begin(), output.end(), out);
    // output.noalias() += mat * input;
  }

  auto shape() -> std::pair< size_t, size_t > {
    return std::make_pair((size_t) A.rows(), (size_t) A.cols());
  }

  void set_parameters(F* params){
    std::copy(params, params + _params.size(), _params.begin());
  }

  auto get_num_parameters() -> size_t {
    return _params.size();
  }

};