from array import array
from numbers import Number
from typing import Callable, Optional, Union

import numpy as np
from scipy.sparse.linalg import LinearOperator

from .fttr import fttr
from .lanczos import lanczos
from .tridiag import eigh_tridiag, eigvalsh_tridiag


def lanczos_quadrature(
	d: np.ndarray,
	e: np.ndarray,
	deg: Optional[int] = None,
	quad: str = "gw",  # The method of computing the weights
	nodes: Optional[np.ndarray] = None,  # Output nodes of the quadrature
	weights: Optional[np.ndarray] = None,  # Output weights of the quadrature
	**kwargs: dict,
):
	r"""Compute the Gaussian quadrature rule of a tridiagonal Jacobi matrix.

	This function computes the degree-$d$ (`deg`) Gaussian quadrature rule for a symmetric Jacobi matrix $J$,
	which associates `nodes` to the eigenvalues of $J$ and `weights` to the squares of the first components
	of the eigenvectors of $J$. The resulting rule is a weighted sum approximating the definite integral:

	$$ \int_{a}^{b} f(x) \omega(x) dx \approx \sum\limits_{i=1}^d f(x_i) w_i $$

	where $\omega(x)$ denotes the weight function and $f(x)$ represents the function being approximated.
	The limits $a,b$ and weight function $\omega(x)$ depend on how `J` was constructed. When $J$ arises from the Lanczos
	method on a symmetric matrix $A \in \mathbb{R}^{n \times n}$, the estimated quantity corresponds to a spectral sum:

	$$ \int_{a}^{b} f(x) \psi(x; A, v) dx $$

	where $psi(x)$ is the eigenvector spectral density associated to the pair $(A,v)$:

	$$ \psi(x; A, v) = \sum\limits_{i=1}^n \lvert u_i^T v \rvert^2 \delta(x - \lambda_i), \quad A = U \Lambda U^T $$

	In this sense, by applying $f$ to the nodes $x_i$, the corresponding quadrature rule can approximate any spectral sum.

	Parameters:
		d: array of `n` diagonal elements.
		e: array of `n` or `n-1` off-diagonal elements. See details.
		deg: degree of the quadrature rule to compute.
		quad: method used to compute the rule. Either Golub Welsch or FTTR is supported.
		nodes: output array to store the `n` nodes of the quadrature (optional).
		weights: output array to store the `n` weights of the quadrature (optional).

	Notes:
		To compute the weights of the quadrature, `quad` can be set to either 'golub_welsch' or 'fttr'. The former uses a LAPACK call to
		the method of relatively robust representations (RRR), which builds local LDL decompositions around clusters of eigenvalues,
		while the latter (FTTR) uses the explicit recurrence expression for orthogonal polynomials. Though both require
		$O(\mathrm{deg}^2)$ time to execute, the former requires $O(\mathrm{deg}^2)$ space but is highly accurate, while the latter uses
		only $O(1)$ space at the cost of backward stability. If `deg` is large, `fttr` is preferred for performance, though pilot testing
		should be done to ensure that instability does not cause a large bias in the approximation.
	"""
	deg = len(d) if deg is None else int(min(deg, len(d)))
	e = np.append([0], e) if len(e) == (len(d) - 1) else e
	assert len(d) == len(e) and np.isclose(e[0], 0.0), "Subdiagonal first element 'e[0]' must be close to zero"

	if quad in {"gw", "golub_welsch"}:
		## Golub-Welsch approach: just compute eigen-decomposition from T using QR steps
		theta, ev = eigh_tridiag(d[:deg], e[:deg], **kwargs)
		tau = np.square(ev[0, :])
	elif quad == "fttr":
		## Uses the Forward Three Term Recurrence (FTTR) approach
		theta = eigvalsh_tridiag(d, e, **kwargs)
		tau = np.zeros(len(theta), dtype=theta.dtype)
		fttr(theta, d, e, deg, tau)
	else:
		raise ValueError(f"Invalid quadrature method '{quad}' supplied")
	if nodes is not None and weights is not None:
		assert len(nodes) == deg and len(weights) == deg, "`nodes` and `weights` output arrays must be `deg` in length."
		np.copyto(nodes, theta)
		np.copyto(weights, tau)
	return theta, tau
