
#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
namespace py = pybind11;

#include <vector> 
#include <functional>
#include <algorithm>

// #include "ccallback.h" // TODO: add in future when it's easier to work with in SciPy


auto do_call(std::function< float(float)> f) -> float {
  std::vector< float > my_vec(5, 0);
  my_vec[0] = 1; 
  my_vec[1] = 2; 
  my_vec[2] = 3; 
  my_vec[3] = 4;
  my_vec[4] = 5;  
  std::transform(my_vec.begin(), my_vec.end(), my_vec.begin(), [&f](float value){
    return f(value);
  });
  return std::accumulate(my_vec.begin(), my_vec.end(), 0.0);
  // return py::cast(my_vec);
  // return 0;
}

auto do_call_pure(const std::function<float(float)>& f) -> float {
  using fn_type = float (*)(float);
  auto result = f.target<fn_type>();
  if (!result) {
    throw std::invalid_argument("Failed to convert to function ptr");
  } 
  // else if (*result == ){
  // } else {
  //   throw std::invalid_argument("Failed to convert to function ptr");
  // }
  const auto f_wrapped = [&result](float val) -> float { return (*result)(val); };
  std::vector< float > my_vec(5, 0);
  my_vec[0] = 1; 
  my_vec[1] = 2; 
  my_vec[2] = 3; 
  my_vec[3] = 4;
  my_vec[4] = 5; 
  std::transform(my_vec.begin(), my_vec.end(), my_vec.begin(), f_wrapped);
  return std::accumulate(my_vec.begin(), my_vec.end(), 0.0);
}

#include <cmath>

float add_one(float a) { return a + 1; }

float receive_capsule(py::capsule cap){
  const void* f = cap.get_pointer();
  auto fc = (float (*)(float))(f);
  return (*fc)(1);
  // return py::capsule(reinterpret_cast< const void* >(&add_one), "float (*)(float)", nullptr);
}

PYBIND11_MODULE(_call, m) {
  m.def("call", &do_call);
  m.def("call_cptr", &do_call_pure);
  m.def("receive_capsule", &receive_capsule);
  m.def("add_one", &add_one);
}
// typedef struct {
//   PyObject *extra_arguments;
//   PyObject *extra_keywords;
// } PythonCallbackData;

// Based on Py_GeometricTransform
// void PyObject *Py_GeometricTransform(PyObject *obj, PyObject *args) {
//   PyObject *fnc = NULL, *extra_arguments = NULL, *extra_keywords = NULL;
//   int mode, order, nprepad;
//   double cval;
//   void *func = NULL, *data = NULL;
//   PythonCallbackData cbdata;
//   ccallback_t callback;
//   static ccallback_signature_t callback_signatures[] = {
//     {"int (intptr_t *, double *, int, int, void *)"},
//     {"int (npy_intp *, double *, int, int, void *)"},
//     #if NPY_SIZEOF_INTP == NPY_SIZEOF_SHORT
//     {"int (short *, double *, int, int, void *)"},
//     #endif
//     #if NPY_SIZEOF_INTP == NPY_SIZEOF_INT
//     {"int (int *, double *, int, int, void *)"},
//     #endif
//     #if NPY_SIZEOF_INTP == NPY_SIZEOF_LONG
//     {"int (long *, double *, int, int, void *)"},
//     #endif
//     #if NPY_SIZEOF_INTP == NPY_SIZEOF_LONGLONG
//     {"int (long long *, double *, int, int, void *)"},
//     #endif
//     {NULL}
//   };

//   callback.py_function = NULL;
//   callback.c_function = NULL;

//   if (fnc != Py_None) {
//     if (!PyTuple_Check(extra_arguments)) {
//       PyErr_SetString(PyExc_RuntimeError, "extra_arguments must be a tuple");
//       goto exit;
//     }
//     if (!PyDict_Check(extra_keywords)) {
//       PyErr_SetString(PyExc_RuntimeError, "extra_keywords must be a dictionary");
//       goto exit;
//     }
//     if (PyCapsule_CheckExact(fnc) && PyCapsule_GetName(fnc) == NULL) {
//       func = PyCapsule_GetPointer(fnc, NULL);
//       data = PyCapsule_GetContext(fnc);
//     } else {
//       int ret;
//       ret = ccallback_prepare(&callback, callback_signatures, fnc, CCALLBACK_DEFAULTS);
//       if (ret == -1) {
//         goto exit;
//       }

//       if (callback.py_function != NULL) {
//         cbdata.extra_arguments = extra_arguments;
//         cbdata.extra_keywords = extra_keywords;
//         callback.info_p = (void*)&cbdata;
//         func = Py_Map;
//         data = (void*)&callback;
//       }
//       else {
//         func = callback.c_function;
//         data = callback.user_data;
//       }
//     }
//   }
//   if (callback.py_function != NULL || callback.c_function != NULL) {
//     ccallback_release(&callback);
//   }
//   return PyErr_Occurred() ? NULL : Py_BuildValue("");
// }

