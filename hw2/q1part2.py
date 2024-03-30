# -*- coding: utf-8 -*-

"""
q1part2.py - LU factorize a square matrix of n x n
"""

import numpy as np

def _is_postive_integer(x : int) -> bool:
    """Return if input is positive integer."""
    return type(x) == int and x > 0

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

def lu_factorize(matrix : np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Return the LU factorization of input square matrix. Note that due to 
    round off errors, the factorized matrices may not multiply exactly to 
    the original matrix.

    Parameter
    ---------
    matrix: np.ndarray, the matrix to be LU factorized.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        a tuple contains the LU factorization of the input matrix.
        `[0]` is L, and `[1]` is U.
    
    Raises
    ------
    ValueError : if the input is not a square, non-empty matrix of numbers.
    ZeroDivisionError : find 0 during LU factorizing, input matrix not 
        LU factorizable
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

    # Create L and U
    l: np.ndarray = np.identity(n, dtype=float)
    u: np.ndarray = np.zeros((n, n), dtype=float)

    # LU factorize the input matrix.
    # Calculate first row of `u`.
    u[0, :] = matrix[0, :]
    # Calculate first column of `l`.
    if matrix[0, 0] == 0:
        raise ZeroDivisionError(f"Found 0 at u[{iter}, {iter}], "
                                + f"`matrix` is not LU diagonalizable")
    l[1:, 0] = matrix[1:, 0] / matrix[0, 0]

    # Calculate remaining entries for `l` and `u`.
    for iter in range(1, n):
        # Calculate a row of entries for `u`.
        for col in range(iter, n):
            u[iter, col] = matrix[iter, col] \
                           - np.dot(l[iter, :iter], u[:iter, col])
        
        # Calculate a column of entries for `l`.
        if u[iter, iter] == 0 and iter != n - 1:
            # Matrix cannot be LU factorized
            raise ZeroDivisionError(f"Found 0 at u[{iter}, {iter}], "
                                    + f"`matrix` is not LU diagonalizable")
        for row in range(iter + 1, n): # `l` has `1` on diagonal
            l[row, iter] = (matrix[row, iter] \
                            - np.dot(l[row, :iter], u[:iter, iter])) \
                            / u[iter, iter]
    return l, u

def get_rand_matrix(row_len : int, 
                    col_len : int | None = None) -> np.ndarray:
    """
    Generate an random n x m matrix of integers.

    Parameters
    ----------
    row_len : int, row length of the matrix
        Positive integer, the row length `m` of the matrix.
    col_len : int, column length of the matrix
        Positive integer, optional, the column length `n` of the matrix.
        Default is `row_len`.
        
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
    return matrix

def main():
    # Set side-length of the matrix to be factorized.
    N = 100

    # Get `N` x `N` matrix.
    a = get_rand_matrix(N)
    
    # LU factorize `m`.
    res = lu_factorize(a)
    l = res[0]
    u = res[1]

    print(f"A = {a}")
    print(f"L = {l}")
    print(f"U = {u}")

    # Compare the product with the original matrix.
    print("Errors:")
    print(f"{np.dot(l, u) - a}")

if __name__ == "__main__":
    main()