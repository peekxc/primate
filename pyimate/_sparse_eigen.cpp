#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <Eigen/SparseCore> // SparseMatrix
#include <_diagonalization/lanczos_tridiagonalization.h>

namespace py = pybind11;
using py_arr_f = py::array_t< float >;

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

IndexType lanczos_tridiagonalize_py(const Eigen::SparseMatrix< float >* matrix, const py_arr_f& v, const float lanczos_tol, const int orthogonalize, py_arr_f& alpha, py_arr_f& beta){
  auto lo = SparseEigenLinearOperator(*matrix);
  std::pair< size_t, size_t > shape = lo.shape();
  auto n = static_cast< long int >(shape.first);
  auto m = static_cast< int >(shape.second);
  if (n != m){ throw std::invalid_argument("The Lanczos iterations only works with square operators!"); }
  auto alpha_out = static_cast< float* >(alpha.request().ptr);
  auto beta_out = static_cast< float* >(beta.request().ptr);
  if (alpha.shape(0) < m || beta.shape(0) < m){
    throw std::invalid_argument("Ouputs arrays 'alpha' / 'beta' must ");
  }
  lanczos_tridiagonalization< SparseEigenLinearOperator< float > >(&lo, v.data(), n, m, lanczos_tol, orthogonalize, alpha_out, beta_out);
  return 0; 
}

PYBIND11_MODULE(_sparse_eigen, m) {
  m.def("lanczos_tridiagonalize", &lanczos_tridiagonalize_py);
  // py::class_< Eigen::SparseMatrix< double > >(m, "SparseMatrix")
  //   .def(py::init< const py::object& >()) // or (2, 1)? py::keep_alive< 1, 2 >()
  //   .def("matvec", (py::array_t< float >(PyLinearOperator< float >::*)(const py::array_t< float >&) const) &PyLinearOperator< float >::matvec)
  //   .def_property_readonly("shape", &PyLinearOperator< float >::shape)
  //   .def_property_readonly("dtype", &PyLinearOperator< float >::dtype)
  //   .def("lanczos", &lanczos_tridiagonalize_py)
  //   ;
}