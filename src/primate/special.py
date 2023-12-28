from typing import *
import numpy as np 

## Natively support matrix functions
_builtin_matrix_functions = ["identity", "abs", "sqrt", "log", "inv", "exp", "smoothstep", "numrank", "gaussian"]

def soft_sign(x: np.ndarray = None, q: int = 1):
  """Soft-sign function.
  
  This function computes a continuous version of the sign function (centered at 0) which is uniformly close to the 
  sign function for sufficiently large q, and converges to sgn(x) as q -> +infty for all x in [-1, 1].  

  The soft-sign function is often used in principal component regression and norm estimation algorithms, see 
  equation (60) of "Stability of the Lanczos Method for Matrix Function Approximation"
  """
  I = np.arange(q+1)
  J = np.append([1], np.cumprod([(2*j-1)/(2*j) for j in np.arange(1, q+1)]))
  def _sign(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    x = np.clip(x,-1.0,+1.0)
    x = np.atleast_2d(x).T
    sx = np.ravel(np.sum(x * (1-x**2)**I * J, axis=1))
    return sx if len(sx) > 1 else np.take(sx,0)
  return _sign(x) if x is not None else _sign

# _builtin_matrix_functions = ["identity", "abs", "sqrt", "log", "inv", "exp", "smoothstep", "numrank", "gaussian"]
def figure_fun(fun: Union[str, Callable], bounds: tuple = (-1, 1), *args, **kwargs):
  assert isinstance(fun, str) or isinstance(fun, Callable), "'fun' must be string or callable."
  from bokeh.plotting import figure
  dom = np.linspace(bounds[0], bounds[1], 250, endpoint=True)
  p = figure(width=250, height=250, title=f"fun = {fun}")
  out = None
  if isinstance(fun, str):
    if fun == "identity":
      out = dom
    elif fun == "abs":
      out = np.abs(dom)
    elif fun == "sqrt":
      out = np.sqrt(np.abs(dom))
    elif fun == "log":
      out = np.log(dom)
    elif fun == "inv":
      out = np.reciprocal(dom)
    elif fun == "exp":
      out = np.exp(dom)
    elif fun == "smoothstep":
      out = soft_sign(dom, *args, **kwargs)
    elif fun == "rank":
      out = np.sign(dom)
    else:
      raise ValueError(f"Invalid function '{fun}' supplied. Must be one of {str(_builtin_matrix_functions)}")
  else: 
    out = fun(dom, *args, **kwargs)
  p.line(dom, out)
  return p 
    


