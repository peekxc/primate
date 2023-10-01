import numpy as np
from typing import * 
import _random_gen
from math import prod 
from numbers import Integral

_engines = ["splitmix64", "xoshiro256**", "lcg64", "pcg64", "mt64"]
_engine_prefixes = ["sx", "xs", "lcg", "pcg", "mt"]

def rademacher(size: Union[int, tuple], engine: str = "splitmix64", num_threads: int = 1, seed: int = -1, dtype=np.float32):
  """Generates random vectors from the rademacher distribution. 
  
  Parameters
  ----------
  size : int or tuple
      the output shape to generate
  """
  assert engine in _engine_prefixes or engine in _engines, f"Invalid pseudo random number engine supplied '{str(engine)}'"
  assert dtype == np.float32 or dtype == np.float64, "Only 32- or 64-bit floating point numbers are supported."
  engine_id = _engine_prefixes.index(engine) if engine in _engine_prefixes else _engines.index(engine)
  out = np.empty(size, dtype=dtype) if isinstance(size, Integral) else np.empty(prod(size), dtype=dtype)
  engine_f = getattr(_random_gen, 'rademacher_'+_engine_prefixes[engine_id])
  engine_f(out, num_threads, seed)
  return out.reshape(size)

def normal(size: Union[int, tuple], engine: str = "splitmix64", num_threads: int = 1, seed: int = -1, dtype = np.float32):
  assert engine in _engine_prefixes or engine in _engines, f"Invalid pseudo random number engine supplied '{str(engine)}'"
  assert dtype == np.float32 or dtype == np.float64, "Only 32- or 64-bit floating point numbers are supported."
  engine_id = _engine_prefixes.index(engine) if engine in _engine_prefixes else _engines.index(engine)
  out = np.empty(size, dtype=dtype) if isinstance(size, Integral) else np.empty(prod(size), dtype=dtype)
  engine_f = getattr(_random_gen, 'normal_'+_engine_prefixes[engine_id])
  engine_f(out, num_threads, seed)
  return out.reshape(size)

# def rayleigh(n: int, engine: str = "splitmix64", num_threads: int = 1, dtype = np.float32):
#   assert engine in _engine_prefixes or engine in _engines, f"Invalid pseudo random number engine supplied '{str(engine)}'"
#   assert dtype == np.float32 or dtype == np.float64, "Only 32- or 64-bit floating point numbers are supported."
#   engine_id = _engine_prefixes.index(engine) if engine in _engine_prefixes else _engines.index(engine)
#   out = np.empty(n, dtype=dtype)
#   engine_f = getattr(_random_gen, 'rayleigh_'+_engine_prefixes[engine_id])
#   engine_f(out, num_threads)
#   return out
