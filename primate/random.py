import numpy as np
from typing import * 
import _random_gen

_engines = ["splitmix64", "xoshiro256**", "lcg64", "pcg64", "mt64"]
_engine_prefixes = ["sx", "xs", "lcg", "pcg", "mt"]

def rademacher(n: int, engine: str = "splitmix64", num_threads: int = 1, dtype=np.float32):
  assert engine in _engine_prefixes or engine in _engines, f"Invalid pseudo random number engine supplied '{str(engine)}'"
  assert dtype == np.float32 or dtype == np.float64, "Only 32- or 64-bit floating point numbers are supported."
  engine_id = _engine_prefixes.index(engine) if engine in _engine_prefixes else _engines.index(engine)
  out = np.empty(n, dtype=dtype)
  engine_f = getattr(_random_gen, 'rademacher_'+_engine_prefixes[engine_id])
  engine_f(out, num_threads)
  return out

def normal(n: int, engine: str = "splitmix64", num_threads: int = 1, dtype = np.float32):
  assert engine in _engine_prefixes or engine in _engines, f"Invalid pseudo random number engine supplied '{str(engine)}'"
  assert dtype == np.float32 or dtype == np.float64, "Only 32- or 64-bit floating point numbers are supported."
  engine_id = _engine_prefixes.index(engine) if engine in _engine_prefixes else _engines.index(engine)
  out = np.empty(n, dtype=dtype)
  engine_f = getattr(_random_gen, 'normal_'+_engine_prefixes[engine_id])
  engine_f(out, num_threads)
  return out

def rayleigh(n: int, engine: str = "splitmix64", num_threads: int = 1, dtype = np.float32):
  assert engine in _engine_prefixes or engine in _engines, f"Invalid pseudo random number engine supplied '{str(engine)}'"
  assert dtype == np.float32 or dtype == np.float64, "Only 32- or 64-bit floating point numbers are supported."
  engine_id = _engine_prefixes.index(engine) if engine in _engine_prefixes else _engines.index(engine)
  out = np.empty(n, dtype=dtype)
  engine_f = getattr(_random_gen, 'rayleigh_'+_engine_prefixes[engine_id])
  engine_f(out, num_threads)
  return out
