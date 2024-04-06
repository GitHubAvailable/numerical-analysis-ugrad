# -*- coding: utf-8 -*-

"""
q3b.py - Find the largest eigenvalue (in absolute value)
of a matrix using inverse power method.
"""

import time
import numpy as np

TOLERANCE = 10 ** (-6)
ROW = 0
COLUMN = 1

def _is_square(m : np.ndarray) -> bool:
    """Return if the given matrix is a square matrix."""
    if not m.size:
        # `m` is empty.
        return True
    
    if m.ndim < 2:
        # `m` is less than 2D.
        return False
    
    row_len = len(m)
    col_len = len(m[0])

    if row_len != col_len or row_len * col_len != m.size:
        # `m` is rectangular or does not have consistent column width.
        return False
    
    return True

def _is_postive_integer(x : int) -> bool:
    """Return if input is positive integer."""
    return type(x) == int and x > 0

def get_dominint_eig(matrix : np.ndarray) -> float:
    """
    Return the largest eigenvalue (in absolute value)
    of a matrix.
    """
    # Get eigenvalues.
    eigs = np.linalg.eigvals(matrix)

    # Get the minimum and maximum eigenvalues.
    min_eig = np.min(eigs)
    max_eig = np.max(eigs)

    if abs(min_eig) > abs(max_eig):
        return min_eig
    return max_eig

def rayleigh_quotient(matrix : np.ndarray, 
                      x : np.ndarray) -> float:
    """
    Return the Rayleigh quotient for given matrix and nonzero vector `x`.
    
    Parameters
    ----------
    matrix : np.ndarray, the input matrix to calculate 
        the Rayleigh quotient.
    x : np.ndarray, a nonzero vector which the Rayleigh quotient 
        will be calculated for.
    
    Returns
    -------
    float
        The Rayleigh quotient for given matrix and nonzero vector `x`.
    
    Raises
    ------
    ValueError : any parameter is invalid.
    ZeroDivisionError : `x` is a zero vector.
    """
    # Check input validity.
    if not matrix.size:
        raise ValueError(f"input matrix cannot be empty")
    if not _is_square(matrix):
        raise ValueError(f"`matrix` must be square matrix")
    if matrix.ndim > 2:
        raise ValueError(f"matrix (2D array) expected"
                         + f"but {matrix.ndim}D array received")
    if x.shape[ROW] != matrix.shape[COLUMN]:
        raise ValueError(f"size mismatch, matrix has shape {matrix.shape},"
                         + f"but `x` has shape {x.shape}")
    if x.shape[COLUMN] != 1:
        raise ValueError(f"`x` must be a vector")
    if x.ndim > 2:
        raise ValueError(f"`x` ({len(matrix)}, 1) 2D arrap expected"
                         + f"but {x.ndim}D array received")
    if not x.any():
        raise ZeroDivisionError(f"`x` cannot be zero vector")
    
    return (x.T @ matrix @ x) / (x.T @ x)
    

def find_dom_eigen(matrix : np.ndarray) -> float:
    """
    Return the largest largest eigenvalue (in absolute value) 
    of the input matrix found by inverse power method.
    
    Parameter
    ---------
    matrix : np.ndarray, the matrix whose eigenvalue is to be found.
    
    Returns
    -------
    val : float
        The largest eigenvalue (in absolute value) of the input matrix.
    
    Raises
    ------
    ValueError : if the input is not a square matrix.
    """
    # Check input validity.
    if not matrix.size:
        raise ValueError(f"input matrix cannot be empty")
    if not _is_square(matrix):
        raise ValueError(f"input must be square matrix")
    if matrix.ndim > 2:
        raise ValueError(f"matrix (2D array) expected"
                         + f"but {matrix.ndim}D array received")
    
    # Get side-length of the matrix.
    n = len(matrix)

    # Get identity of size n.
    identity = np.identity(n)

    # Start timing.
    start = time.time()

    # Setup initial guess.
    x = np.ones((n, 1))
    val = rayleigh_quotient(matrix, x)

    # Setup iteration count.
    iter_used = 0

    # Setup stop flag.
    diff = 1 # Ensure can start 1 iteration

    while diff >= TOLERANCE:
        # Find next guess.
        v = np.linalg.solve(matrix - val * identity, x)
        v /= np.linalg.norm(v, 2)

        # Calculate change.
        new_val = rayleigh_quotient(matrix, v)
        diff = abs(new_val - val)

        # Update values.
        x = v
        val = new_val

        # Update iteration used.
        iter_used += 1
    
    # End timing.
    end = time.time()
    
    print(f"{iter_used} iterations, used {end - start}s")

    return val

def get_rand_matrix(row_len : int, 
                    col_len : int | None = None, 
                    as_float : bool = False) -> np.ndarray:
    """
    Generate an random n x m matrix of integers.

    Parameters
    ----------
    row_len : int, row length of the matrix
        Positive integer, the row length `m` of the matrix.
    col_len : int, column length of the matrix
        Positive integer, optional, the column length `n` of the matrix.
        Default is `row_len`.
    as_float : bool, whether converting output to type `float64`
        Boolean, optional, whether output entries `float64`.
        Default is False
        
    Returns
    -------
    matrix : ndarray
        An `m`x`n` array with randomly filled elements.
    
    Raises
    ------
    ValueError : if the input parameter is invalid.
    """
    if not col_len:
        # On default `row_len` x `row_len`.
        col_len = row_len
    
    # Check parameters.
    if not _is_postive_integer(row_len):
        raise ValueError(f"'row_len' ({row_len}) must be a positive integer")
    if not _is_postive_integer(col_len):
        raise ValueError(f"'col_len' ({col_len}) must be a positive integer")
    
    # Number of elements needed.
    space: int = row_len * col_len
    
    # Generate the random matrix of integers.
    matrix = np.random.randint(space, 
                               size=(row_len, col_len))
    if as_float:
        return matrix.astype(float)
    return matrix

def main():
    # Set size of `A`.
    N = 100

    # Generate `A`.
    a = get_rand_matrix(N, as_float=True)
    print(f"a = {a}")

    # Find largest eigenvalue (in absolute value).
    val = find_dom_eigen(a)
    print(f"lambda_1 = {val}")

    # Error check.
    ext_val = get_dominint_eig(a)
    print(f"Expected: {ext_val}")
    print(f"Error = {abs(ext_val - val)}")

if __name__ == "__main__":
    main()