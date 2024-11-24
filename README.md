[![](https://img.shields.io/badge/docs-quarto-blue.svg?logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADAAAAAwCAYAAABXAvmHAAADDklEQVR4nOyZXYhNXRjHf+8bw4TyUYYpxUQm8jG+wo0rX0XiYkoaLpRcUEzNXEiZuZjQhCsKd27UiEa5RCSSbzJGNBMZIo1EYTSGVj1pOufZ+6y1zlpz5mh+dZrp2Xuv9f/vvdfa63nW/xQ5QwYKTdEbGJYZKKuYnW+b5qZMBH4Cn4ASYBzwEejLp+EPnW1Zsf8yA3UtL33argI2AWuBOSK6AWgEFgL3gB/AE+Aa0ArcBn67dNJcPSMrlvUEHCgDtgM1QKXF+SOBJfKrBzqBU8BpeVJe+IyBCcBRoANoshSvUQEcArqAw8AYn0ZcDcwCHgJ7gVE+HSqUyhN5Dix2vdjFwEbgBjDFtRNLyoGrwHqXi2wNbAEuAOP9tFkzGjgPbLC9wMbAGhloA8VwoEX6zUkuA0tlyisNo80aMw2flTGXSpoB8404CYwIq82asfLalqSdlGZgGzA3vC4nZgK7005IMmDu/r44mpypS3sLkgxUA9nf7cJgvvg7kw4mGaiNp8eLPdq6jQQD02S9MpiYCizQDmgGVsbX44WqSzMwP74WL6q0oGZgsAzeTKZrQc3ApPhavFB1aQnN5kBf33fytx1YFKC9Xi2oGXgaoLP+fAPuB27zL5qBXZJ15YvJfa8Dk4EdAdozN6I5M6gZqAn0HWgQA+Xyf750aAa0QfwmQGcxUHVpBtrja/HimRbUDNyJr8WLu1pQM2AS9574epzoA65oBzQDn4Ez8TU5ccllDCCFprzqmIE5mHQgyYCZss7F0+PEZamjqqTlxI3ArzianDiQdjDNgJlOj4fX48QR4FbaCbnqQvVJo38AuCgJfSq5DPRIrfJxOF1WtAFbbfYPbEqL34F1wIMw2nJyE1gBfLE52ba42wUsl/k4Jmbxtwrotr3ApbzeIyX2BtkuCslXYD+wWpbN1rhucPTK9FoZ8Gm0SgmzyWcJ47vN+loGdxVwTPa7XDCv5AlgnjzVV546shOa951OGeUj+dVK8WmZFGTL+6Wm3bKZ9xZ4Ieml11boP0nR79QPGSg0fwIAAP//ppR6oEaYviYAAAAASUVORK5CYII=)](https://peekxc.github.io/primate/)
[![Cirrus CI - build status](https://api.cirrus-ci.com/github/peekxc/primate.svg?branch=main)](https://cirrus-ci.com/github/peekxc/primate)
[![Coverage Status](https://badgen.net/coveralls/c/github/peekxc/primate/main)](https://coveralls.io/github/peekxc/primate?branch=main)
[![Python versions](https://badgen.net/badge/python/3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12%20%7C%203.13/blue)](https://github.com/peekxc/primate/actions)
[![PyPI Version](https://badgen.net/pypi/v/scikit-primate)](https://pypi.org/project/scikit-primate/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

<!-- [![Cirrus CI - Specific Branch Build Status](https://img.shields.io/cirrus/github/peekxc/primate/pythran_overhaul?task=build_and_test&style=flat&logo=pytest&logoColor=white&label=build)](https://cirrus-ci.com/github/peekxc/primate/pythran_overhaul) -->
<!-- [![build_macos](https://img.shields.io/github/actions/workflow/status/peekxc/primate/build_macos.yml?logo=apple&logoColor=white)](https://github.com/peekxc/primate/actions/workflows/wheels.yml) 
[![build_windows](https://img.shields.io/github/actions/workflow/status/peekxc/primate/build_windows.yml?logo=windows&logoColor=white)](https://github.com/peekxc/primate/actions/workflows/wheels.yml) 
[![build_linux](https://img.shields.io/github/actions/workflow/status/peekxc/primate/build_linux.yml?logo=linux&logoColor=white)](https://github.com/peekxc/primate/actions/workflows/wheels.yml) -->
<!-- [![coverage_badge](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/peekxc/ef42349965f40edf4232737026690c5f/raw/coverage_info.json)](https://coveralls.io/github/peekxc/simplextree-py)  -->
<!-- [![PyPI Version](https://img.shields.io/pypi/v/simplextree)](https://pypi.org/project/simplextree) -->
<!-- https://badgen.net/github/checks/peekxc/primate/pythran_overhaul?label=tests -->
<!-- https://badgen.net/pypi/python/scikit-primate -->

`primate`, short for **Pr**obabilistic **I**mplicit **Ma**trix **T**race **E**stimator, is a Python package that provides estimators of quantities from matrices, linear operators, and [matrix functions](https://en.wikipedia.org/wiki/Analytic_function_of_a_matrix):

$$ f(A) \triangleq U f(\Lambda) U^{\intercal}, \quad \quad f : [a,b] \to \mathbb{R}$$

This definition is quite general in that different parameterizations of $f$ produce a variety of spectral quantities, including the matrix inverse $A^{-1}$, the matrix exponential $\mathrm{exp}(A)$, the matrix logarithm $\mathrm{log}(A)$, and so on. 

Composing these with _trace_ and _diagonal_ estimators yields approximations for the [numerical rank](https://doi.org/10.1016/j.amc.2007.06.005), the [log-determinant](https://en.wikipedia.org/wiki/Determinant#Trace), the [Schatten norms](https://en.wikipedia.org/wiki/Schatten_norm), the [eigencount](https://doi.org/10.1002/nla.2048), the [Estrada index](https://en.wikipedia.org/wiki/Estrada_index), the [Heat Kernel Signature](https://en.wikipedia.org/wiki/Heat_kernel_signature), and so on. 

<!-- `primate` also exports functionality for estimating the [spectral density](https://doi.org/10.1137/130934283) and for computing Gaussian quadrature rules from Jacobi matrices.  -->

Notable features of `primate` include:

- Efficient methods for _trace_, _diagonal_, and _matrix function_ approximation
- Support for _arbitrary_ matrix types, e.g. NumPy arrays, sparse matrices, or [LinearOperator](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.LinearOperator.html#scipy-sparse-linalg-linearoperator)'s
- Support for _arbitrary_ matrix functions, i.e. `Callable`'s (Python) and `invocable`'s[^3] (C++)
- Matrix-free interface to the _Lanczos_ and _Golub-Welsch_ methods 
- Various composable stopping criteria for easy and adaptive convergence checking

`primate` was partially inspired by the [`imate` package](https://github.com/ameli/imate)---for a comparison of the two, see [here](https://peekxc.github.io/primate/imate_compare.html).

## Installation

`primate` is a standard PEP-517 package, and thus can be installed via pip:

```{bash}
python -m pip install scikit-primate
```

Assuming your platform is supported, no compilation is needed. 

See the [installation page](https://peekxc.github.io/primate/basic/install.html) for details.

## Applications 

Applications of matrix functions include [characterizing folding in proteins](https://en.wikipedia.org/wiki/Estrada_index), [principal component regression](https://en.wikipedia.org/wiki/Principal_component_regression), [spectral clustering](https://en.wikipedia.org/wiki/Spectral_clustering),  [Gaussian process likelihood estimation](https://en.wikipedia.org/wiki/Gaussian_process), [counting triangles in distributed-memory networks](https://doi.org/10.1137/23M1548323), [characterizing graph similarity](https://doi.org/10.1016/j.patcog.2008.12.029), and [deep neural loss landscape analysis](https://proceedings.mlr.press/v97/ghorbani19b).

If you have a particular application, feel free to make a computational notebook to illustrate it as a use-case!

[^1]: Musco, Cameron, Christopher Musco, and Aaron Sidford. (2018) "Stability of the Lanczos method for matrix function approximation."
[^2]: Ubaru, S., Chen, J., & Saad, Y. (2017). Fast estimation of tr(f(A)) via stochastic Lanczos quadrature.
[^3]: This includes [std::function](https://en.cppreference.com/w/cpp/utility/functional/function)'s, C-style function pointers, [functors](https://stackoverflow.com/questions/356950/what-are-c-functors-and-their-uses), and [lambda expressions](https://en.cppreference.com/w/cpp/language/lambda).

