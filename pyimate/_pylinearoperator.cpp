#include <type_traits>
#include <concepts>
#include <algorithm>
#include <functional>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <_definitions/definitions.h>
#include <_definitions/types.h>
#include <_diagonalization/diagonalization.h>
#include <_diagonalization/lanczos_tridiagonalization.h>
#include <_diagonalization/golub_kahn_bidiagonalization.h>
#include <_linear_operator/linear_operator.h>
#include <_c_basic_algebra/c_matrix_operations.h>

#include "pylinops.h"
namespace py = pybind11;

