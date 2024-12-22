from typing import Optional

import numpy as np

from .fttr import fttr
from .tridiag import eigh_tridiag, eigvalsh_tridiag


def quadrature(
	d: np.ndarray,
	e: np.ndarray,
	deg: Optional[int] = None,
	quad: str = "gw",  # The method of computing the weights
	nodes: Optional[np.ndarray] = None,  # Output nodes of the quadrature
	weights: Optional[np.ndarray] = None,  # Output weights of the quadrature
	**kwargs,
) -> tuple:
	r"""Compute the Gaussian quadrature rule of a tridiagonal Jacobi matrix.

	This function computes the fixed degree Gaussian quadrature rule for a symmetric Jacobi matrix $J$,
	which associates `nodes` $x_i$ to the eigenvalues of $J$ and `weights` $w_i$ to the squares of the first components
	of their corresponding normalized eigenvectors. The resulting rule is a weighted sum approximating the definite integral:

	$$ \int_{a}^{b} f(x) \omega(x) dx \approx \sum\limits_{i=1}^d f(x_i) \cdot w_i $$

	where $\omega(x)$ denotes the weight function and $f(x)$ represents the function being approximated.
	When `J` is constructed by the Lanczos method on a symmetric matrix $A \in \mathbb{R}^{n \times n}$, the
	rule can be used to approximate the weighted integral:

	$$ \int_{a}^{b} f(x) \psi(x; A, v) dx \approx \sum\limits_{i=1}^n f(\lambda_i)$$

	where $\psi(x)$ is the eigenvector spectral density associated to the pair $(A,v)$:

	$$ \psi(x; A, v) = \sum\limits_{i=1}^n \lvert u_i^T v \rvert^2 \delta(x - \lambda_i), \quad A = U \Lambda U^T $$

	For more details on this, see the references.

	Parameters:
		d: array of `n` diagonal elements.
		e: array of `n` or `n-1` off-diagonal elements. See details.
		deg: degree of the quadrature rule to compute.
		quad: method used to compute the rule. Either Golub Welsch or FTTR is supported.
		nodes: output array to store the `deg` nodes of the quadrature (optional).
		weights: output array to store the `deg` weights of the quadrature (optional).

	Returns:
		tuple (nodes, weights) of the degree-`deg` Gaussian quadrature rule.

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
