#include <pythonic/core.hpp>
#include <pythonic/python/core.hpp>
#include <pythonic/types/bool.hpp>
#include <pythonic/types/int.hpp>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <pythonic/include/types/float.hpp>
#include <pythonic/include/types/ndarray.hpp>
#include <pythonic/types/ndarray.hpp>
#include <pythonic/types/float.hpp>
#include <pythonic/include/builtins/assert.hpp>
#include <pythonic/include/builtins/getattr.hpp>
#include <pythonic/include/builtins/len.hpp>
#include <pythonic/include/builtins/pythran/static_list.hpp>
#include <pythonic/include/builtins/range.hpp>
#include <pythonic/include/builtins/tuple.hpp>
#include <pythonic/include/numpy/append.hpp>
#include <pythonic/include/numpy/copy.hpp>
#include <pythonic/include/numpy/linalg/norm.hpp>
#include <pythonic/include/numpy/zeros.hpp>
#include <pythonic/include/operator_/add.hpp>
#include <pythonic/include/operator_/div.hpp>
#include <pythonic/include/operator_/eq.hpp>
#include <pythonic/include/operator_/iadd.hpp>
#include <pythonic/include/operator_/idiv.hpp>
#include <pythonic/include/operator_/matmul.hpp>
#include <pythonic/include/operator_/mul.hpp>
#include <pythonic/include/operator_/sub.hpp>
#include <pythonic/include/types/str.hpp>
#include <pythonic/builtins/assert.hpp>
#include <pythonic/builtins/getattr.hpp>
#include <pythonic/builtins/len.hpp>
#include <pythonic/builtins/pythran/static_list.hpp>
#include <pythonic/builtins/range.hpp>
#include <pythonic/builtins/tuple.hpp>
#include <pythonic/numpy/append.hpp>
#include <pythonic/numpy/copy.hpp>
#include <pythonic/numpy/linalg/norm.hpp>
#include <pythonic/numpy/zeros.hpp>
#include <pythonic/operator_/add.hpp>
#include <pythonic/operator_/div.hpp>
#include <pythonic/operator_/eq.hpp>
#include <pythonic/operator_/iadd.hpp>
#include <pythonic/operator_/idiv.hpp>
#include <pythonic/operator_/matmul.hpp>
#include <pythonic/operator_/mul.hpp>
#include <pythonic/operator_/sub.hpp>
#include <pythonic/types/str.hpp>
namespace 
{
  namespace __pythran_lanczos_pythran
  {
    struct lanczos_action
    {
      typedef void callable;
      ;
      template <typename argument_type0 , typename argument_type1 , typename argument_type2 , typename argument_type3 , typename argument_type4 >
      struct type
      {
        typedef typename std::remove_cv<typename std::remove_reference<argument_type1>::type>::type __type0;
        typedef __type0 __type1;
        typedef decltype(pythonic::builtins::getattr(pythonic::types::attr::SHAPE{}, std::declval<__type1>())) __type2;
        typedef decltype(pythonic::types::as_const(std::declval<__type2>())) __type3;
        typedef typename std::tuple_element<0,typename std::remove_reference<__type3>::type>::type __type4;
        typedef typename std::remove_cv<typename std::remove_reference<decltype(pythonic::builtins::functor::len{})>::type>::type __type5;
        typedef typename std::remove_cv<typename std::remove_reference<argument_type0>::type>::type __type6;
        typedef __type6 __type7;
        typedef decltype(std::declval<__type5>()(std::declval<__type7>())) __type8;
        typedef decltype(pythonic::types::make_tuple(std::declval<__type4>(), std::declval<__type8>())) __type9;
        typedef typename pythonic::assignable<__type9>::type __type10;
        typedef __type10 __type11;
        typedef decltype(pythonic::types::as_const(std::declval<__type11>())) __type12;
        typedef typename std::tuple_element<0,typename std::remove_reference<__type12>::type>::type __type13;
        typedef typename pythonic::assignable<__type13>::type __type14;
        typedef __type14 __type15;
        typedef typename pythonic::returnable<__type15>::type __type16;
        typedef __type16 result_type;
      }  
      ;
      template <typename argument_type0 , typename argument_type1 , typename argument_type2 , typename argument_type3 , typename argument_type4 >
      inline
      typename type<argument_type0, argument_type1, argument_type2, argument_type3, argument_type4>::result_type operator()(argument_type0 z, argument_type1 A, argument_type2 alpha, argument_type3 beta, argument_type4 v) const
      ;
    }  ;
    template <typename argument_type0 , typename argument_type1 , typename argument_type2 , typename argument_type3 , typename argument_type4 >
    inline
    typename lanczos_action::type<argument_type0, argument_type1, argument_type2, argument_type3, argument_type4>::result_type lanczos_action::operator()(argument_type0 z, argument_type1 A, argument_type2 alpha, argument_type3 beta, argument_type4 v) const
    {
      typedef typename std::remove_cv<typename std::remove_reference<decltype(pythonic::numpy::functor::zeros{})>::type>::type __type0;
      typedef typename std::remove_cv<typename std::remove_reference<argument_type1>::type>::type __type1;
      typedef __type1 __type2;
      typedef decltype(pythonic::builtins::getattr(pythonic::types::attr::SHAPE{}, std::declval<__type2>())) __type3;
      typedef decltype(pythonic::types::as_const(std::declval<__type3>())) __type4;
      typedef typename std::tuple_element<0,typename std::remove_reference<__type4>::type>::type __type5;
      typedef typename std::remove_cv<typename std::remove_reference<decltype(pythonic::builtins::functor::len{})>::type>::type __type6;
      typedef typename std::remove_cv<typename std::remove_reference<argument_type0>::type>::type __type7;
      typedef __type7 __type8;
      typedef decltype(std::declval<__type6>()(std::declval<__type8>())) __type9;
      typedef decltype(pythonic::types::make_tuple(std::declval<__type5>(), std::declval<__type9>())) __type10;
      typedef typename pythonic::assignable<__type10>::type __type11;
      typedef __type11 __type12;
      typedef decltype(pythonic::types::as_const(std::declval<__type12>())) __type13;
      typedef typename std::tuple_element<0,typename std::remove_reference<__type13>::type>::type __type14;
      typedef typename pythonic::assignable<__type14>::type __type15;
      typedef __type15 __type16;
      typedef decltype(std::declval<__type0>()(std::declval<__type16>())) __type17;
      typedef typename pythonic::assignable<__type17>::type __type18;
      typedef decltype(pythonic::types::as_const(std::declval<__type8>())) __type20;
      typedef typename std::remove_cv<typename std::remove_reference<decltype(pythonic::builtins::functor::range{})>::type>::type __type21;
      typedef long __type22;
      typedef typename std::tuple_element<1,typename std::remove_reference<__type13>::type>::type __type23;
      typedef typename pythonic::lazy<__type23>::type __type24;
      typedef __type24 __type25;
      typedef decltype(pythonic::operator_::add(std::declval<__type25>(), std::declval<__type22>())) __type26;
      typedef decltype(std::declval<__type21>()(std::declval<__type22>(), std::declval<__type26>())) __type27;
      typedef typename std::remove_cv<typename std::iterator_traits<typename std::remove_reference<__type27>::type::iterator>::value_type>::type __type28;
      typedef __type28 __type29;
      typedef decltype(pythonic::operator_::sub(std::declval<__type29>(), std::declval<__type22>())) __type30;
      typedef decltype(std::declval<__type20>()[std::declval<__type30>()]) __type31;
      typedef typename std::remove_cv<typename std::remove_reference<decltype(pythonic::numpy::functor::copy{})>::type>::type __type34;
      typedef typename std::remove_cv<typename std::remove_reference<argument_type4>::type>::type __type35;
      typedef __type35 __type36;
      typedef decltype(std::declval<__type34>()(std::declval<__type36>())) __type37;
      typedef decltype(pythonic::types::make_tuple(std::declval<__type17>(), std::declval<__type37>(), std::declval<__type17>())) __type40;
      typedef typename pythonic::assignable<__type40>::type __type41;
      typedef __type41 __type42;
      typedef decltype(pythonic::types::as_const(std::declval<__type42>())) __type43;
      typedef typename std::tuple_element<1,typename std::remove_reference<__type43>::type>::type __type44;
      typedef typename pythonic::assignable<__type44>::type __type45;
      typedef typename std::remove_cv<typename std::remove_reference<decltype(pythonic::numpy::linalg::functor::norm{})>::type>::type __type46;
      typedef decltype(std::declval<__type46>()(std::declval<__type36>())) __type48;
      typedef typename __combined<__type45,__type48>::type __type50;
      typedef __type50 __type51;
      typedef decltype(pythonic::operator_::functor::matmul()(std::declval<__type2>(), std::declval<__type51>())) __type52;
      typedef typename std::remove_cv<typename std::remove_reference<decltype(pythonic::numpy::functor::append{})>::type>::type __type53;
      typedef typename std::remove_cv<typename std::remove_reference<decltype(pythonic::builtins::pythran::functor::static_list{})>::type>::type __type54;
      typedef decltype(pythonic::types::make_tuple(std::declval<__type22>())) __type55;
      typedef decltype(std::declval<__type54>()(std::declval<__type55>())) __type56;
      typedef typename std::remove_cv<typename std::remove_reference<argument_type2>::type>::type __type57;
      typedef __type57 __type58;
      typedef decltype(std::declval<__type53>()(std::declval<__type56>(), std::declval<__type58>())) __type59;
      typedef decltype(pythonic::types::make_tuple(std::declval<__type22>(), std::declval<__type22>())) __type60;
      typedef decltype(std::declval<__type54>()(std::declval<__type60>())) __type61;
      typedef typename std::remove_cv<typename std::remove_reference<argument_type3>::type>::type __type62;
      typedef __type62 __type63;
      typedef decltype(std::declval<__type53>()(std::declval<__type61>(), std::declval<__type63>())) __type64;
      typedef decltype(pythonic::types::make_tuple(std::declval<__type59>(), std::declval<__type64>())) __type65;
      typedef typename pythonic::assignable<__type65>::type __type66;
      typedef __type66 __type67;
      typedef decltype(pythonic::types::as_const(std::declval<__type67>())) __type68;
      typedef typename std::tuple_element<1,typename std::remove_reference<__type68>::type>::type __type69;
      typedef typename pythonic::assignable<__type69>::type __type70;
      typedef __type70 __type71;
      typedef decltype(pythonic::types::as_const(std::declval<__type71>())) __type72;
      typedef decltype(std::declval<__type72>()[std::declval<__type29>()]) __type74;
      typedef typename std::tuple_element<0,typename std::remove_reference<__type43>::type>::type __type75;
      typedef typename pythonic::assignable<__type75>::type __type76;
      typedef __type76 __type77;
      typedef decltype(pythonic::operator_::mul(std::declval<__type74>(), std::declval<__type77>())) __type78;
      typedef decltype(pythonic::operator_::sub(std::declval<__type52>(), std::declval<__type78>())) __type79;
      typedef typename std::tuple_element<0,typename std::remove_reference<__type68>::type>::type __type80;
      typedef typename pythonic::assignable<__type80>::type __type81;
      typedef __type81 __type82;
      typedef decltype(pythonic::types::as_const(std::declval<__type82>())) __type83;
      typedef decltype(std::declval<__type83>()[std::declval<__type29>()]) __type85;
      typedef decltype(pythonic::operator_::mul(std::declval<__type85>(), std::declval<__type51>())) __type87;
      typedef decltype(pythonic::operator_::sub(std::declval<__type79>(), std::declval<__type87>())) __type88;
      typedef typename pythonic::assignable<__type88>::type __type89;
      typedef typename __combined<__type89,__type51>::type __type91;
      typedef __type91 __type92;
      typedef typename pythonic::assignable<__type92>::type __type93;
      typedef typename __combined<__type45,__type48,__type93>::type __type94;
      typedef typename pythonic::assignable<__type51>::type __type95;
      typedef typename __combined<__type76,__type95,__type92>::type __type96;
      typedef typename __combined<__type89,__type51,__type92>::type __type97;
      typedef __type94 __type98;
      typedef decltype(pythonic::operator_::mul(std::declval<__type31>(), std::declval<__type98>())) __type99;
      typedef typename __combined<__type18,__type99>::type __type100;
      typedef typename pythonic::assignable<__type100>::type __type101;
      typedef typename pythonic::assignable<__type96>::type __type102;
      typedef typename pythonic::assignable<__type94>::type __type103;
      typedef typename pythonic::assignable<__type97>::type __type104;
      typename pythonic::assignable_noescape<decltype(pythonic::types::make_tuple(std::get<0>(pythonic::types::as_const(pythonic::builtins::getattr(pythonic::types::attr::SHAPE{}, A))), pythonic::builtins::functor::len{}(z)))>::type __tuple0 = pythonic::types::make_tuple(std::get<0>(pythonic::types::as_const(pythonic::builtins::getattr(pythonic::types::attr::SHAPE{}, A))), pythonic::builtins::functor::len{}(z));
      typename pythonic::assignable_noescape<decltype(std::get<0>(pythonic::types::as_const(__tuple0)))>::type n = std::get<0>(pythonic::types::as_const(__tuple0));
      typename pythonic::lazy<decltype(std::get<1>(pythonic::types::as_const(__tuple0)))>::type k = std::get<1>(pythonic::types::as_const(__tuple0));
      pythonic::pythran_assert(pythonic::operator_::eq(pythonic::builtins::functor::len{}(v), n), pythonic::types::str("v dimension mismatch"));
      __type101 y = pythonic::numpy::functor::zeros{}(n);
      typename pythonic::assignable_noescape<decltype(pythonic::types::make_tuple(pythonic::numpy::functor::append{}(pythonic::builtins::pythran::functor::static_list{}(pythonic::types::make_tuple(0L)), alpha), pythonic::numpy::functor::append{}(pythonic::builtins::pythran::functor::static_list{}(pythonic::types::make_tuple(0L, 0L)), beta)))>::type __tuple1 = pythonic::types::make_tuple(pythonic::numpy::functor::append{}(pythonic::builtins::pythran::functor::static_list{}(pythonic::types::make_tuple(0L)), alpha), pythonic::numpy::functor::append{}(pythonic::builtins::pythran::functor::static_list{}(pythonic::types::make_tuple(0L, 0L)), beta));
      typename pythonic::assignable_noescape<decltype(std::get<0>(pythonic::types::as_const(__tuple1)))>::type av = std::get<0>(pythonic::types::as_const(__tuple1));
      typename pythonic::assignable_noescape<decltype(std::get<1>(pythonic::types::as_const(__tuple1)))>::type bv = std::get<1>(pythonic::types::as_const(__tuple1));
      typename pythonic::assignable_noescape<decltype(pythonic::types::make_tuple(pythonic::numpy::functor::zeros{}(n), pythonic::numpy::functor::copy{}(v), pythonic::numpy::functor::zeros{}(n)))>::type __tuple2 = pythonic::types::make_tuple(pythonic::numpy::functor::zeros{}(n), pythonic::numpy::functor::copy{}(v), pythonic::numpy::functor::zeros{}(n));
      __type102 qp = std::get<0>(pythonic::types::as_const(__tuple2));
      __type103 qc = std::get<1>(pythonic::types::as_const(__tuple2));
      pythonic::operator_::idiv(qc, pythonic::numpy::linalg::functor::norm{}(v));
      {
        long  __target4990541184 = pythonic::operator_::add(k, 1L);
        for (long  i=1L; i < __target4990541184; i += 1L)
        {
          __type104 qn_ = pythonic::operator_::sub(pythonic::operator_::sub(pythonic::operator_::functor::matmul()(A, qc), pythonic::operator_::mul(pythonic::types::as_const(bv).fast(i), qp)), pythonic::operator_::mul(pythonic::types::as_const(av).fast(i), qc));
          y += pythonic::operator_::mul(pythonic::types::as_const(z).fast(pythonic::operator_::sub(i, 1L)), qc);
          qp = qc;
          qc = qn_;
        }
      }
      return n;
    }
  }
}
#include <pythonic/python/exception_handler.hpp>
#ifdef ENABLE_PYTHON_MODULE
inline
typename __pythran_lanczos_pythran::lanczos_action::type<pythonic::types::ndarray<double,pythonic::types::pshape<long>>, pythonic::types::ndarray<double,pythonic::types::pshape<long,long>>, pythonic::types::ndarray<double,pythonic::types::pshape<long>>, pythonic::types::ndarray<double,pythonic::types::pshape<long>>, pythonic::types::ndarray<double,pythonic::types::pshape<long>>>::result_type lanczos_action0(pythonic::types::ndarray<double,pythonic::types::pshape<long>>&& z, pythonic::types::ndarray<double,pythonic::types::pshape<long,long>>&& A, pythonic::types::ndarray<double,pythonic::types::pshape<long>>&& alpha, pythonic::types::ndarray<double,pythonic::types::pshape<long>>&& beta, pythonic::types::ndarray<double,pythonic::types::pshape<long>>&& v) 
{
  
                            PyThreadState *_save = PyEval_SaveThread();
                            try {
                                auto res = __pythran_lanczos_pythran::lanczos_action()(z, A, alpha, beta, v);
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
__pythran_wrap_lanczos_action0(PyObject *self, PyObject *args, PyObject *kw)
{
    PyObject* args_obj[5+1];
    
    char const* keywords[] = {"z", "A", "alpha", "beta", "v",  nullptr};
    if(! PyArg_ParseTupleAndKeywords(args, kw, "OOOOO",
                                     (char**)keywords , &args_obj[0], &args_obj[1], &args_obj[2], &args_obj[3], &args_obj[4]))
        return nullptr;
    if(is_convertible<pythonic::types::ndarray<double,pythonic::types::pshape<long>>>(args_obj[0]) && is_convertible<pythonic::types::ndarray<double,pythonic::types::pshape<long,long>>>(args_obj[1]) && is_convertible<pythonic::types::ndarray<double,pythonic::types::pshape<long>>>(args_obj[2]) && is_convertible<pythonic::types::ndarray<double,pythonic::types::pshape<long>>>(args_obj[3]) && is_convertible<pythonic::types::ndarray<double,pythonic::types::pshape<long>>>(args_obj[4]))
        return to_python(lanczos_action0(from_python<pythonic::types::ndarray<double,pythonic::types::pshape<long>>>(args_obj[0]), from_python<pythonic::types::ndarray<double,pythonic::types::pshape<long,long>>>(args_obj[1]), from_python<pythonic::types::ndarray<double,pythonic::types::pshape<long>>>(args_obj[2]), from_python<pythonic::types::ndarray<double,pythonic::types::pshape<long>>>(args_obj[3]), from_python<pythonic::types::ndarray<double,pythonic::types::pshape<long>>>(args_obj[4])));
    else {
        return nullptr;
    }
}

            static PyObject *
            __pythran_wrapall_lanczos_action(PyObject *self, PyObject *args, PyObject *kw)
            {
                return pythonic::handle_python_exception([self, args, kw]()
                -> PyObject* {

if(PyObject* obj = __pythran_wrap_lanczos_action0(self, args, kw))
    return obj;
PyErr_Clear();

                return pythonic::python::raise_invalid_argument(
                               "lanczos_action", "\n""    - lanczos_action(float[:], float[:,:], float[:], float[:], float[:])", args, kw);
                });
            }


static PyMethodDef Methods[] = {
    {
    "lanczos_action",
    (PyCFunction)__pythran_wrapall_lanczos_action,
    METH_VARARGS | METH_KEYWORDS,
    "Yields the matrix y = Qz where T(alpha, beta) = Q^T A Q is the tridiagonal matrix spanning K(A, v)\n""\n""    Supported prototypes:\n""\n""    - lanczos_action(float[:], float[:,:], float[:], float[:], float[:])"},
    {NULL, NULL, 0, NULL}
};


#if PY_MAJOR_VERSION >= 3
  static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "lanczos_pythran",            /* m_name */
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
PYTHRAN_MODULE_INIT(lanczos_pythran)(void)
#ifndef _WIN32
__attribute__ ((visibility("default")))
#if defined(GNUC) && !defined(__clang__)
__attribute__ ((externally_visible))
#endif
#endif
;
PyMODINIT_FUNC
PYTHRAN_MODULE_INIT(lanczos_pythran)(void) {
    import_array()
    #if PY_MAJOR_VERSION >= 3
    PyObject* theModule = PyModule_Create(&moduledef);
    #else
    PyObject* theModule = Py_InitModule3("lanczos_pythran",
                                         Methods,
                                         ""
    );
    #endif
    if(! theModule)
        PYTHRAN_RETURN;
    PyObject * theDoc = Py_BuildValue("(ss)",
                                      "0.14.0",
                                      "c28d41579692fad8840cab46324a3830d3b838d93e076ca3a40f4a67a8887278");
    if(! theDoc)
        PYTHRAN_RETURN;
    PyModule_AddObject(theModule,
                       "__pythran__",
                       theDoc);


    PYTHRAN_RETURN;
}

#endif