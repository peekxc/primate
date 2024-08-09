import numpy as np

# def tqli(d, e, z=None):
#   """
#   QL algorithm with implicit shifts to determine the eigenvalues and eigenvectors of a real, symmetric, tridiagonal matrix.

#   Parameters:
#       d : array_like
#           Diagonal elements of the tridiagonal matrix.
#       e : array_like
#           Subdiagonal elements of the tridiagonal matrix.
#       z : array_like, optional
#           Matrix containing the eigenvectors. If None, only eigenvalues are computed.

#   Returns:
#       w : ndarray
#           The eigenvalues in ascending order.
#       z : ndarray, optional
#           The normalized eigenvectors if `z` is provided.
#   """
#   n = len(d)
#   e = np.array([0] + list(e) + [0]) #np.concatenate(np.concatenate(([0], e[:-1])), [0]) # Convenient to renumber the elements of e
#   for l in range(n):
#     ii = 0
#     while True:
#       # Look for a single small subdiagonal element to split the matrix
#       for m in range(l, n - 1):
#         if (abs(e[m]) + abs(d[m])) == abs(d[m]) or d[m] == 0:
#           break
#       else:
#         m = n - 1

#       if m != l:
#         # assert ii < 130, f"Too many iterations in tqli ({ii})"
#         if ii > 130: 
#           break # just give up
#         ii += 1
#         g = (d[l + 1] - d[l]) / (2 * e[l])  # Form shift
#         r = np.hypot(g, 1)
#         g = d[m] - d[l] + e[l] / (g + np.sign(g) * r)  # This is dm - ks
#         s, c, p = 1, 1, 0
#         for i in range(m - 1, l, -1):
#           f, b = s * e[i], c * e[i]
#           r = np.hypot(f, g)
#           e[i + 1] = r
#           if r == 0:
#             # Recover from underflow
#             d[i + 1] -= p
#             e[m] = 0
#             break
#           s, c, g = f / r, g / r, d[i + 1] - p
#           r = (d[i] - g) * s + 2 * c * b
#           p = s * r
#           d[i + 1] = g + p
#           g = c * r - b
#         print(f"d: {d}, e: {e}")
#         d[l] -= p
#         e[l], e[m] = g, 0
#       else:
#         break
#   return d

# import numpy as np

# # def sign(a,b):
# #   return -abs(a) if b < 0 else abs(a)

# # def tqli(d, e):
# #   n = len(d)
# #   for i in range(1, n):
# #       e[i-1] = e[i]
# #   e[n-1] = 0.0
# #   for l in range(n):
# #       iter = 0
# #       while True:
# #           for m in range(l, n-1):
# #               dd = abs(d[m]) + abs(d[m+1])
# #               if abs(e[m]) + dd == dd:
# #                   break
# #           if m != l:
# #               assert l !=
# #               if iter >= 30:
# #                   raise ValueError("No convergence in TLQI.")
# #               g = (d[l+1] - d[l]) / (2.0 * e[l])
# #               r = np.sqrt(g * g + 1.0)
# #               g = d[m] - d[l] + e[l] / (g + sign(r, g))
# #               s = c = 1.0
# #               p = 0.0
# #               for i in range(m-1, l-1, -1):
# #                   f = s * e[i]
# #                   b = c * e[i]
# #                   if abs(f) >= abs(g):
# #                       c = g / f
# #                       r = np.sqrt(c * c + 1.0)
# #                       e[i+1] = f * r
# #                       c *= (s := 1.0/r)
# #                   else:
# #                       s = f / g
# #                       r = np.sqrt(s * s + 1.0)
# #                       e[i+1] = g * r
# #                       s *= (c := 1.0/r)
# #                   g = d[i+1] - p
# #                   r = (d[i] - g) * s + 2.0 * c * b
# #                   p = s * r
# #                   d[i+1] = g + p
# #                   g = c * r - b
# #               d[l] = d[l] - p
# #               e[l] = g
# #               e[m] = 0.0
# #           else:
# #               break


# def tridiag_eigvalsh(d, s, max_iterations: int = 30):
#     """
#     Compute the eigenvalues of a symmetric tridiagonal matrix using QR algorithm.

#     Parameters:
#         d : array_like
#             The diagonal elements of the tridiagonal matrix.
#         s : array_like
#             The subdiagonal elements of the tridiagonal matrix.
#         max_iterations : int, optional
#             The maximum number of iterations. Default is 30.

#     Returns:
#         eigenvalues : ndarray
#             The eigenvalues of the matrix.
#     """
#     n = len(d)
#     consider_as_zero = np.finfo(d.dtype).tiny
#     precision = 2 * np.finfo(d.dtype).eps
#     iter_count = 0
#     while n > 0:
#       for i in range(n - 1):
#         if abs(s[i]) <= precision * (abs(d[i]) + abs(d[i + 1])) or abs(s[i]) <= consider_as_zero:
#           s[i] = 0
#       end = np.max(np.flatnonzero(s == 0))
#       if end == 0:
#         break
#       iter_count += 1
#       if iter_count > max_iterations * n:
#         break
#       start = np.min(np.flatnonzero(s == 0))
#       d, s = tridiag_qr_step(d, s, start, end)
#     return d

# def tridiag_eigvalsh2(diag, subdiag, max_iterations: int = 30):
#     """
#     Compute the eigenvalues and optionally eigenvectors of a symmetric tridiagonal matrix using QR algorithm.

#     Parameters:
#         d : array_like
#             The diagonal elements of the tridiagonal matrix.
#         s : array_like
#             The subdiagonal elements of the tridiagonal matrix. Should have a zero appended to the end to match lenth of diagonal.
#         max_iterations : int, optional
#             The maximum number of iterations. Default is 1000.

#     Returns:
#         eigenvalues : ndarray
#             The eigenvalues of the matrix.
#     """
#     n = len(diag)
#     consider_as_zero = np.finfo(diag.dtype).tiny
#     precision = 2 * np.finfo(diag.dtype).eps
#     start = 0
#     iter_count = 0
#     while n > 0:
#       for i in range(start, n - 1):
#         if abs(subdiag[i]) <= precision * (abs(diag[i]) + abs(diag[i+1])) or abs(subdiag[i]) <= consider_as_zero:
#           subdiag[i] = 0
#       end = n - 1
#       while end > 0 and subdiag[end - 1] == 0:
#         end -= 1
#       if end <= 0:
#         break
#       iter_count += 1
#       if iter_count > max_iterations * n:
#         raise ValueError("No convergence")
#       start = end - 1
#       while start > 0 and subdiag[start - 1] != 0:
#           start -= 1
#       d, s = tridiag_qr_step(diag, subdiag, start, end)
#     return d

# # def tridiag_eigvalsh(d, s, max_iterations: int = 30):
# #     """
# #     Compute the eigenvalues of a symmetric tridiagonal matrix using QR algorithm.

# #     Parameters:
# #         d : array_like
# #             The diagonal elements of the tridiagonal matrix.
# #         s : array_like
# #             The subdiagonal elements of the tridiagonal matrix.
# #         max_iterations : int, optional
# #             The maximum number of iterations. Default is 30.

# #     Returns:
# #         eigenvalues : ndarray
# #             The eigenvalues of the matrix.
# #     """
# #     n = len(d)
# #     consider_as_zero = np.finfo(d.dtype).tiny
# #     precision = 2 * np.finfo(d.dtype).eps
# #     iter_count = 0
# #     while n > 0:
# #       mask = np.abs(s[:-1]) > precision * (np.abs(d[:-1]) + np.abs(d[1:]))  # Mask for non-zero subdiagonals
# #       s[:-1] = np.where(mask, s[:-1], 0)            # Set small subdiagonals to zero
# #       end = np.where(s > consider_as_zero)[0][-1] + 1 if np.any(s > consider_as_zero) else 0  # Find the last non-zero subdiagonal index
# #       if end == 0 or (iter_count + 1) > max_iterations * n:
# #         break
# #       iter_count += 1
# #       start = np.where(s[:end] != 0)[0][0]  # Find the first non-zero subdiagonal index
# #       d, s = tridiag_qr_step(d, s, start, end)
# #     return d

# def tridiag_qr_step(diag, subdiag, start, end, eivec=None):
#     """
#     Perform one step of the tridiagonal QR algorithm.

#     Parameters:
#         diag : array_like
#             The diagonal elements of the tridiagonal matrix.
#         subdiag : array_like
#             The subdiagonal elements of the tridiagonal matrix.
#         start : int
#             Start index of the unreduced block.
#         end : int
#             End index of the unreduced block.
#         eivec : array_like, optional
#             The matrix to store the eigenvectors. If provided, the eigenvectors are computed.

#     Returns:
#         diag : ndarray
#             The updated diagonal elements.
#         subdiag : ndarray
#             The updated subdiagonal elements.
#         eivec : ndarray, optional
#             The updated eigenvectors if `eivec` is provided.
#     """
#     for i in range(end - 1, start - 1, -1):
#       d, e = diag[i], subdiag[i]
#       if e != 0:
#         ## Compute Givens rotation
#         if abs(d) < abs(diag[i - 1]):
#           t = -d / diag[i - 1]
#           r = np.sqrt(1 + t * t)
#           c = 1 / r
#           s = c * t
#         else:
#           t = -diag[i - 1] / d
#           r = np.sqrt(1 + t * t)
#           s = 1 / r
#           c = s * t

#         ## Apply Givens rotation to the diagonal and subdiagonal
#         diag[i] = r * d - s * diag[i - 1]
#         diag[i - 1] = c * diag[i - 1] - s * r * d
#         subdiag[i] = 0
#     return diag, subdiag

# def compute_from_tridiagonal(diag, subdiag, max_iterations=1000):
#     """
#     Compute the eigenvalues and optionally eigenvectors of a symmetric tridiagonal matrix using QR algorithm.

#     Parameters:
#         diag : array_like
#             The diagonal elements of the tridiagonal matrix.
#         subdiag : array_like
#             The subdiagonal elements of the tridiagonal matrix.
#         max_iterations : int, optional
#             The maximum number of iterations. Default is 1000.

#     Returns:
#         eigenvalues : ndarray
#             The eigenvalues of the matrix.
#         eigenvectors : ndarray, optional
#             The eigenvectors of the matrix, if computed.
#     """
#     n = len(diag)

#     consider_as_zero = np.finfo(diag.dtype).tiny
#     precision = 2 * np.finfo(diag.dtype).eps

#     start = 0
#     iter_count = 0

#     while n > 0:
#         for i in range(start, n - 1):
#             if abs(subdiag[i]) <= precision * (abs(diag[i]) + abs(diag[i + 1])) or abs(subdiag[i]) <= consider_as_zero:
#                 subdiag[i] = 0

#         end = n - 1
#         while end > 0 and subdiag[end - 1] == 0:
#             end -= 1

#         if end <= 0:
#             break

#         iter_count += 1
#         if iter_count > max_iterations * n:
#             raise ValueError("No convergence")

#         start = end - 1
#         while start > 0 and subdiag[start - 1] != 0:
#             start -= 1

#         diag, subdiag = tridiagonal_qr_step(diag, subdiag, start, end)

#     # Sort eigenvalues and corresponding vectors
#     sort_indices = np.argsort(diag)
#     eigenvalues = diag[sort_indices]
#     return eigenvalues


# def tridiagonal_qr_step(diag, subdiag, start, end, eivec=None):
#     """
#     Perform one step of the tridiagonal QR algorithm.

#     Parameters:
#         diag : array_like
#             The diagonal elements of the tridiagonal matrix.
#         subdiag : array_like
#             The subdiagonal elements of the tridiagonal matrix.
#         start : int
#             Start index of the unreduced block.
#         end : int
#             End index of the unreduced block.
#         eivec : array_like, optional
#             The matrix to store the eigenvectors. If provided, the eigenvectors are computed.

#     Returns:
#         diag : ndarray
#             The updated diagonal elements.
#         subdiag : ndarray
#             The updated subdiagonal elements.
#         eivec : ndarray, optional
#             The updated eigenvectors if `eivec` is provided.
#     """
#     for i in range(end - 1, start - 1, -1):
#         d, e = diag[i], subdiag[i]

#         if e != 0:
#             # Compute Givens rotation
#             if abs(d) < abs(diag[i - 1]):
#                 t = -d / diag[i - 1]
#                 r = np.sqrt(1 + t * t)
#                 c = 1 / r
#                 s = c * t
#             else:
#                 t = -diag[i - 1] / d
#                 r = np.sqrt(1 + t * t)
#                 s = 1 / r
#                 c = s * t

#             # Apply Givens rotation to the diagonal and subdiagonal
#             diag[i] = r * d - s * diag[i - 1]
#             diag[i - 1] = c * diag[i - 1] - s * r * d
#             subdiag[i] = 0
#     return diag, subdiag
# print(f"EW 2: {np.sort(tridiag_eigvalsh(a.copy(),np.append([0], b)))}")
# print(f"EW 3: {np.sort(tridiag_eigvalsh2(a.copy(),np.append([0], b)))}")
# print(f"EW 4: {compute_from_tridiagonal(a.copy(),np.append([0], b))}")


import numba as nb

@nb.jit(nopython=True)
def sign(a, b):
	# return 1 if b > 0 else (-1 if a < 0 else 1)
  return int(b > 1) - int(a < 0) + 1

## Based on: https://github.com/rappoccio/PHY410/blob/f6a183eb48807841f6d35be45b4aa845d905a04c/cpt_python/cpt.py#L438
## This uses Givens rotations instead of divide-and-conquer to find the eigenvalues of a tridiagonal matrix 
## Uses O(1) space, thus is preferred over Golub-Welsch if only ritz values are needed and space is an issue. 
## However this is much slower and less stable than divide-and-conquer, if can fit in memory. 
@nb.jit(nopython=True)
def tqli(d: np.ndarray, e: np.ndarray, max_iter: int = 30):
  """Tridiagonal QL Implicit algorithm w/ shifts to determine the eigenvalues of a real symmetric tridiagonal matrix
  
  Note that with shifting, the eigenvalues no longer necessarily appear on the diagonal in order of increasing absolute magnitude.

  Based on pseudocode from the Chapter on "Eigenvalues and Eigenvectors of a Tridiagonal Matrix" in 
  NUMERICAL RECIPES IN FORTRAN 77: THE ART OF SCIENTIFIC COMPUTING.

  Parameters:
    d = Diagonal elements of the tridiagonal matrix.
    e = Subdiagonal elements of the tridiagonal matrix.
    max_iter = Number of iterations to align the eigenspace(s). 

  Returns:
    w = The eigenvalues of the tridiagonal matrix T(d,e).
  """
  assert len(d) == len(e), "Diagonal and subdiagonal should have same length (subdiagonal should be prefixed with 0)"
  n = len(d)
  for i in range(1, n):
    e[i-1] = e[i]
  e[n-1] = 0.0
  for l in range(n):
    ii = 0
    m = l
    while True:
      for ml in range(l, n-1):
        ## Look for a single small subdiagonal element to split the matrix
        m = ml 
        dd = abs(d[m]) + abs(d[m+1])
        if (abs(e[m]) + dd) == dd:
          break
        else:
          m += 1
      if m != l:
        if ii > max_iter:
          ## Could throw here, but since this is used downstream in randomized algorithm we take as-is
          break
        ii += 1
        g = (d[l+1] - d[l]) / (2.0 * e[l])        # shift 
        r = np.hypot(g, 1.0)                      # pythag
        g = d[m] - d[l] + e[l] / (g + sign(r, g)) # dm - ks 
        s, c, p = 1.0, 1.0, 0.0
        for i in range(m-1, l-1, -1):
          ## Plane rotation followed by Givens to restore tridiagonal 
          f, b = s * e[i], c * e[i]
          e[i+1] = r = np.hypot(f, g)
          if r == 0.0: ## Recover from underflow
            d[i+1] -= p
            e[m] = 0.0
            break
          s, c = f / r, g / r
          g = d[i+1] - p
          r = (d[i] - g) * s + 2.0 * c * b
          p = s * r
          d[i+1] = g + p
          g = c * r - b
          # for k in range(n):
          #   f = z[k][i+1]
          #   z[k][i+1] = s * z[k][i] + c * f
          #   z[k][i] = c * z[k][i] - s * f
        if r == 0.0 and i >= l:
          continue
        d[l] -= p
        e[l] = g
        e[m] = 0.0
      if m == l:
        break
