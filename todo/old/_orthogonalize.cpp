#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

// #include <_definitions/types.h> // DataType, IndexType, etc 
// #include <_c_trace_estimator/c_orthogonalization.h>

#include "_definitions/types.h"
#include "_orthogonalization/orthogonalization.h"

namespace py = pybind11;

// Note we enforce fortran style ordering here
template< std::floating_point F > 
using py_array = py::array_t< F, py::array::f_style | py::array::forcecast >;

template< std::floating_point F >
void gram_schmidt (
  const py_array< F >& V,  // note this should be in Fortran / column-major order
  const LongIndexType vector_size, 
  const IndexType num_vectors, 
  const IndexType last_vector, 
  const FlagType num_ortho, 
  py_array< F >& v
){
  const F* V_data = static_cast< const F* >(V.request().ptr);
  F* v_data = static_cast< F* >(v.request().ptr);
  cOrthogonalization< F >::gram_schmidt_process(V_data, vector_size, num_vectors, last_vector, num_ortho, v_data);
}

template< std::floating_point F >
void orthogonalize_vectors(py_array< F >& V){
  F* V_data = static_cast< F* >(V.request().ptr);
  LongIndexType num_vectors = static_cast< LongIndexType >(V.shape(0));
  IndexType vector_size = static_cast< IndexType >(V.shape(1));
  vector_size = vector_size = 1;
  cOrthogonalization< F >::orthogonalize_vectors(V_data, vector_size, num_vectors);
}

template< std::floating_point F >
void _orthogonalize(py::module &m){
  m.def("gram_schmidt", &gram_schmidt< F >);
  m.def("orthogonalize_vectors", &orthogonalize_vectors< F >);
}

PYBIND11_MODULE(_orthogonalize, m) {
  m.doc() = "orthogonalization module"; 
  _orthogonalize< float >(m);
  _orthogonalize< double >(m);
  // _orthogonalize< long double >(m);
}