from primate import _call
def identity(x): return x
def square(x): return x ** 2

_call.call(identity)
_call.call(square)
_call.call_cptr(square)
_call.call_cptr(_call.add_one) # works! 





import pyimate
import ctypes
import numpy as np
from scipy import LowLevelCallable

## Doesn't work unfortunately
# import cppimport
# somecode = cppimport.imp_from_filepath("/Users/mpiekenbrock/pyimate/examples/somecode.cpp")

from scipy._lib._ccallback import _ccallback_c
square = pyimate.functions._ridge.square
capsule_obj = _ccallback_c.get_raw_capsule(square.__self__, "double (double)", 0)
square_callable = LowLevelCallable(capsule_obj)

integrate.quad(square_callable, 0, 4, epsabs=1.49e-14)

from scipy import integrate
square = pyimate.functions._ridge.square
integrate.quad(square, 0, 4, epsabs=1.49e-14)


import timeit
timeit.timeit(lambda: integrate.quad(lambda x: x**2, 0, 4, epsabs=1.49e-14))
timeit.timeit(lambda: integrate.quad(square, 0, 4, epsabs=1.49e-14))

import ctypes
lib = ctypes.CDLL(pyimate.functions._ridge.__spec__.origin)
lib.square.restype = ctypes.c_int
lib.func.argtypes = (ctypes.c_int,ctypes.c_double)


LowLevelCallable(square)
LowLevelCallable(pyimate.functions._ridge.square, signature="int(int)")




_ccallback_c.get_capsule_signature


_ccallback_c.get_capsule_signature(capsule)
# LowLevelCallable(pyimate.functions._ridge)

pyimate.functions._ridge.__name__

# ctypes.LibraryLoader

from pyimate import trace


def add_numbers_wrapper(args, user_data):
  func_ptr = trace._trace.apply_smoothstep
  return func_ptr(int(args[0]), int(args[1]))

# Create the LowLevelCallable
add_numbers_callable = LowLevelCallable(trace._trace.apply_smoothstep)

# # Load the shared library
# example_module = ctypes.CDLL("./example_module.so")

# # Define the PyCapsule cleanup function
# def cleanup(capsule):
#     print("Capsule is being deleted")
#     # Additional cleanup code can be added here
#     pass

# # Create a PyCapsule with the function pointer and cleanup function
# add_numbers_capsule = ctypes.pythonapi.PyCapsule_New(
#   ctypes.c_void_p(example_module.add_numbers), "example_module.add_numbers", cleanup
# )

# # Low-level wrapper function for the PyCapsule
# def add_numbers_wrapper(args, user_data):
#     return ctypes.c_double(example_module.add_numbers(int(args[0]), int(args[1])))

# # Create the LowLevelCallable
# add_numbers_callable = LowLevelCallable(add_numbers_wrapper)

# Test using the SciPy quad function
from scipy.integrate import quad

result, _ = quad(add_numbers_callable, 0, 10)

print("Result:", result)


f = _call.add_one
str(f) # '<built-in method add_one of PyCapsule object at 0x13642b570>'
f.__self__ # <capsule object NULL at 0x13642b570>

_call.receive_capsule(f.__self__)

import os 
import pybind11
from scipy import LowLevelCallable
import cffi
# from scipy._lib._ccallback import _ccallback_c

ffi = cffi.FFI()
ffi.cdef("""
float add_one(float a);
""")
lib = ffi.dlopen(os.path.abspath(_call.__file__))
lib.add_one

LowLevelCallable(_call.add_one)