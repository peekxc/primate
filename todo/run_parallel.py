from timeit import timeit
import numpy as np
from pyimate import _random_generator
z = np.zeros(150000, np.float32)

timeit(lambda: _random_generator.rademacher_xs(z, 1), number=1000)
timeit(lambda: _random_generator.rademacher_xs(z, 2), number=1000)