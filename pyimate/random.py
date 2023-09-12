import numpy as np
from typing import * 
import _random_generator

def rademacher(n: int, engine: str = "splitmix64", num_threads: int = 1, dtype=np.float32):
  out = np.empty(n, dtype=dtype)
  _random_generator.rademacher_sx(out, 1)
  return out

