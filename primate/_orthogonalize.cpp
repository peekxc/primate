#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

// #include <_definitions/types.h> // DataType, IndexType, etc 
// #include <_c_trace_estimator/c_orthogonalization.h>

#include "_definitions/types.h"
#include "_orthogonalization/orthogonalization.h"

namespace py = pybind11;
using py_arr_f = py::array_t< float, py::array::c_style | py::array::forcecast >;

void gram_schmidt (
  const py_arr_f& V, 
  const LongIndexType vector_size, 
  const IndexType num_vectors, 
  const IndexType last_vector, 
  const FlagType num_ortho, 
  py_arr_f& v
){
  const float* V_data = static_cast< const float* >(V.request().ptr);
  float* v_data = static_cast< float* >(v.request().ptr);
  cOrthogonalization< float >::gram_schmidt_process(V_data, vector_size, num_vectors, last_vector, num_ortho, v_data);
}

void orthogonalize_vectors (py_arr_f& V){
  float* V_data = static_cast< float* >(V.request().ptr);
  LongIndexType num_vectors = static_cast< LongIndexType >(V.shape(0));
  IndexType vector_size = static_cast< IndexType >(V.shape(1));
  cOrthogonalization< float >::orthogonalize_vectors(V_data, vector_size, num_vectors);
}

PYBIND11_MODULE(_orthogonalize, m) {
  m.doc() = "orthogonalization module"; 
  m.def("gram_schmidt", &gram_schmidt);
  m.def("orthogonalize_vectors", &orthogonalize_vectors);
}