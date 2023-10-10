#define PY_MAJOR_VERSION 3
#undef ENABLE_PYTHON_MODULE
#include <pythonic/core.hpp>
#include <pythonic/python/core.hpp>
#include <pythonic/types/bool.hpp>
#include <pythonic/types/int.hpp>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <pythonic/include/types/int.hpp>
#include <pythonic/include/types/numpy_texpr.hpp>
#include <pythonic/include/types/float.hpp>
#include <pythonic/include/types/ndarray.hpp>
#include <pythonic/types/int.hpp>
#include <pythonic/types/numpy_texpr.hpp>
#include <pythonic/types/ndarray.hpp>
#include <pythonic/types/float.hpp>
#include <pythonic/include/numpy/ndarray/fill.hpp>
#include <pythonic/include/numpy/ndarray/reshape.hpp>
#include <pythonic/include/operator_/add.hpp>
#include <pythonic/include/operator_/iadd.hpp>
#include <pythonic/include/operator_/mul.hpp>
#include <pythonic/numpy/ndarray/fill.hpp>
#include <pythonic/numpy/ndarray/reshape.hpp>
#include <pythonic/operator_/add.hpp>
#include <pythonic/operator_/iadd.hpp>
#include <pythonic/operator_/mul.hpp>
namespace 
{
  namespace __pythran_up_laplacian
  {
    struct up_laplacian_matvec
    {
      typedef void callable;
      ;
      template <typename argument_type0 , typename argument_type1 , typename argument_type2 , typename argument_type3 >
      struct type
      {
        typedef typename std::remove_cv<typename std::remove_reference<argument_type1>::type>::type __type0;
        typedef __type0 __type1;
        typedef __type1 __type2;
        typedef typename std::remove_cv<typename std::remove_reference<argument_type2>::type>::type __type3;
        typedef __type3 __type4;
        typedef typename std::remove_cv<typename std::remove_reference<decltype(pythonic::numpy::ndarray::functor::reshape{})>::type>::type __type5;
        typedef typename std::remove_cv<typename std::remove_reference<argument_type0>::type>::type __type6;
        typedef __type6 __type7;
        typedef long __type8;
        typedef decltype(std::declval<__type5>()(std::declval<__type7>(), std::declval<__type8>())) __type9;
        typedef decltype(pythonic::operator_::mul(std::declval<__type4>(), std::declval<__type9>())) __type10;
        typedef __type10 __type11;
        typedef typename pythonic::assignable<__type1>::type __type12;
        typedef typename __combined<__type12,__type10>::type __type13;
        typedef __type13 __type14;
        typedef typename pythonic::returnable<__type14>::type __type15;
        typedef __type2 __ptype0;
        typedef __type11 __ptype1;
        typedef __type15 result_type;
      }  
      ;
      template <typename argument_type0 , typename argument_type1 , typename argument_type2 , typename argument_type3 >
      inline
      typename type<argument_type0, argument_type1, argument_type2, argument_type3>::result_type operator()(argument_type0 x, argument_type1 v, argument_type2 degree, argument_type3 simplices) const
      ;
    }  ;
    template <typename argument_type0 , typename argument_type1 , typename argument_type2 , typename argument_type3 >
    inline
    typename up_laplacian_matvec::type<argument_type0, argument_type1, argument_type2, argument_type3>::result_type up_laplacian_matvec::operator()(argument_type0 x, argument_type1 v, argument_type2 degree, argument_type3 simplices) const
    {
      typedef typename std::remove_cv<typename std::remove_reference<argument_type1>::type>::type __type0;
      typedef __type0 __type1;
      typedef typename pythonic::assignable<__type1>::type __type2;
      typedef typename std::remove_cv<typename std::remove_reference<argument_type2>::type>::type __type3;
      typedef __type3 __type4;
      typedef typename std::remove_cv<typename std::remove_reference<decltype(pythonic::numpy::ndarray::functor::reshape{})>::type>::type __type5;
      typedef typename std::remove_cv<typename std::remove_reference<argument_type0>::type>::type __type6;
      typedef __type6 __type7;
      typedef long __type8;
      typedef decltype(std::declval<__type5>()(std::declval<__type7>(), std::declval<__type8>())) __type9;
      typedef decltype(pythonic::operator_::mul(std::declval<__type4>(), std::declval<__type9>())) __type10;
      typedef typename __combined<__type2,__type10>::type __type11;
      typedef typename pythonic::assignable<__type11>::type __type12;
      __type12 v_ = v;
      pythonic::numpy::ndarray::functor::fill{}(v_, 0L);
      v_ += pythonic::operator_::mul(degree, pythonic::numpy::ndarray::functor::reshape{}(x, -1L));
      return v_;
    }
  }
}
#include <pythonic/python/exception_handler.hpp>
#ifdef ENABLE_PYTHON_MODULE
inline
typename __pythran_up_laplacian::up_laplacian_matvec::type<pythonic::types::ndarray<double,pythonic::types::pshape<long>>, pythonic::types::ndarray<double,pythonic::types::pshape<long>>, pythonic::types::ndarray<double,pythonic::types::pshape<long>>, pythonic::types::ndarray<long,pythonic::types::pshape<long,long>>>::result_type up_laplacian_matvec0(pythonic::types::ndarray<double,pythonic::types::pshape<long>>&& x, pythonic::types::ndarray<double,pythonic::types::pshape<long>>&& v, pythonic::types::ndarray<double,pythonic::types::pshape<long>>&& degree, pythonic::types::ndarray<long,pythonic::types::pshape<long,long>>&& simplices) 
{
  
                            PyThreadState *_save = PyEval_SaveThread();
                            try {
                                auto res = __pythran_up_laplacian::up_laplacian_matvec()(x, v, degree, simplices);
                                PyEval_RestoreThread(_save);
                                return res;
                            }
                            catch(...) {
                                PyEval_RestoreThread(_save);
                                throw;
                            }
                            ;
}
inline
typename __pythran_up_laplacian::up_laplacian_matvec::type<pythonic::types::ndarray<double,pythonic::types::pshape<long>>, pythonic::types::ndarray<double,pythonic::types::pshape<long>>, pythonic::types::ndarray<double,pythonic::types::pshape<long>>, pythonic::types::numpy_texpr<pythonic::types::ndarray<long,pythonic::types::pshape<long,long>>>>::result_type up_laplacian_matvec1(pythonic::types::ndarray<double,pythonic::types::pshape<long>>&& x, pythonic::types::ndarray<double,pythonic::types::pshape<long>>&& v, pythonic::types::ndarray<double,pythonic::types::pshape<long>>&& degree, pythonic::types::numpy_texpr<pythonic::types::ndarray<long,pythonic::types::pshape<long,long>>>&& simplices) 
{
  
                            PyThreadState *_save = PyEval_SaveThread();
                            try {
                                auto res = __pythran_up_laplacian::up_laplacian_matvec()(x, v, degree, simplices);
                                PyEval_RestoreThread(_save);
                                return res;
                            }
                            catch(...) {
                                PyEval_RestoreThread(_save);
                                throw;
                            }
                            ;
}

static PyObject *
__pythran_wrap_up_laplacian_matvec0(PyObject *self, PyObject *args, PyObject *kw)
{
    PyObject* args_obj[4+1];
    
    char const* keywords[] = {"x", "v", "degree", "simplices",  nullptr};
    if(! PyArg_ParseTupleAndKeywords(args, kw, "OOOO",
                                     (char**)keywords , &args_obj[0], &args_obj[1], &args_obj[2], &args_obj[3]))
        return nullptr;
    if(is_convertible<pythonic::types::ndarray<double,pythonic::types::pshape<long>>>(args_obj[0]) && is_convertible<pythonic::types::ndarray<double,pythonic::types::pshape<long>>>(args_obj[1]) && is_convertible<pythonic::types::ndarray<double,pythonic::types::pshape<long>>>(args_obj[2]) && is_convertible<pythonic::types::ndarray<long,pythonic::types::pshape<long,long>>>(args_obj[3]))
        return to_python(up_laplacian_matvec0(from_python<pythonic::types::ndarray<double,pythonic::types::pshape<long>>>(args_obj[0]), from_python<pythonic::types::ndarray<double,pythonic::types::pshape<long>>>(args_obj[1]), from_python<pythonic::types::ndarray<double,pythonic::types::pshape<long>>>(args_obj[2]), from_python<pythonic::types::ndarray<long,pythonic::types::pshape<long,long>>>(args_obj[3])));
    else {
        return nullptr;
    }
}

static PyObject *
__pythran_wrap_up_laplacian_matvec1(PyObject *self, PyObject *args, PyObject *kw)
{
    PyObject* args_obj[4+1];
    
    char const* keywords[] = {"x", "v", "degree", "simplices",  nullptr};
    if(! PyArg_ParseTupleAndKeywords(args, kw, "OOOO",
                                     (char**)keywords , &args_obj[0], &args_obj[1], &args_obj[2], &args_obj[3]))
        return nullptr;
    if(is_convertible<pythonic::types::ndarray<double,pythonic::types::pshape<long>>>(args_obj[0]) && is_convertible<pythonic::types::ndarray<double,pythonic::types::pshape<long>>>(args_obj[1]) && is_convertible<pythonic::types::ndarray<double,pythonic::types::pshape<long>>>(args_obj[2]) && is_convertible<pythonic::types::numpy_texpr<pythonic::types::ndarray<long,pythonic::types::pshape<long,long>>>>(args_obj[3]))
        return to_python(up_laplacian_matvec1(from_python<pythonic::types::ndarray<double,pythonic::types::pshape<long>>>(args_obj[0]), from_python<pythonic::types::ndarray<double,pythonic::types::pshape<long>>>(args_obj[1]), from_python<pythonic::types::ndarray<double,pythonic::types::pshape<long>>>(args_obj[2]), from_python<pythonic::types::numpy_texpr<pythonic::types::ndarray<long,pythonic::types::pshape<long,long>>>>(args_obj[3])));
    else {
        return nullptr;
    }
}

            static PyObject *
            __pythran_wrapall_up_laplacian_matvec(PyObject *self, PyObject *args, PyObject *kw)
            {
                return pythonic::handle_python_exception([self, args, kw]()
                -> PyObject* {

if(PyObject* obj = __pythran_wrap_up_laplacian_matvec0(self, args, kw))
    return obj;
PyErr_Clear();


if(PyObject* obj = __pythran_wrap_up_laplacian_matvec1(self, args, kw))
    return obj;
PyErr_Clear();

                return pythonic::python::raise_invalid_argument(
                               "up_laplacian_matvec", "\n""    - up_laplacian_matvec(float[:], float[:], float[:], int[:,:])", args, kw);
                });
            }


static PyMethodDef Methods[] = {
    {
    "up_laplacian_matvec",
    (PyCFunction)__pythran_wrapall_up_laplacian_matvec,
    METH_VARARGS | METH_KEYWORDS,
    "Supported prototypes:\n""\n""    - up_laplacian_matvec(float[:], float[:], float[:], int[:,:])"},
    {NULL, NULL, 0, NULL}
};


#if PY_MAJOR_VERSION >= 3
  static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "up_laplacian",            /* m_name */
    "",         /* m_doc */
    -1,                  /* m_size */
    Methods,             /* m_methods */
    NULL,                /* m_reload */
    NULL,                /* m_traverse */
    NULL,                /* m_clear */
    NULL,                /* m_free */
  };
#define PYTHRAN_RETURN return theModule
#define PYTHRAN_MODULE_INIT(s) PyInit_##s
#else
#define PYTHRAN_RETURN return
#define PYTHRAN_MODULE_INIT(s) init##s
#endif
PyMODINIT_FUNC
PYTHRAN_MODULE_INIT(up_laplacian)(void)
#ifndef _WIN32
__attribute__ ((visibility("default")))
#if defined(GNUC) && !defined(__clang__)
__attribute__ ((externally_visible))
#endif
#endif
;
PyMODINIT_FUNC
PYTHRAN_MODULE_INIT(up_laplacian)(void) {
    import_array()
    #if PY_MAJOR_VERSION >= 3
    PyObject* theModule = PyModule_Create(&moduledef);
    #else
    PyObject* theModule = Py_InitModule3("up_laplacian",
                                         Methods,
                                         ""
    );
    #endif
    if(! theModule)
        PYTHRAN_RETURN;
    PyObject * theDoc = Py_BuildValue("(ss)",
                                      "0.14.0",
                                      "cf96d8b5cf91b5dc1a55a8da53d05122d113a93d9a6a38d9689bedf8898985a3");
    if(! theDoc)
        PYTHRAN_RETURN;
    PyModule_AddObject(theModule,
                       "__pythran__",
                       theDoc);


    PYTHRAN_RETURN;
}

#endif