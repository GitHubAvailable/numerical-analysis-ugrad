# -*- coding: utf-8 -*-

"""
q2part1.py - Find the inverse of random lower triangular matrix.
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

def invert_lower_trig(matrix : np.ndarray) -> np.ndarray:
    """
    Return the inverse of an lower triangular square matrix.

    Parameter
    ---------
    matrix: np.ndarray, the lower triangular matrix to be inverted.

    Returns
    -------
    inverse : np.ndarray
        The inverse of the input lower triangular matrix.
    
    Raises
    ------
    ValueError : if the input is not a square, non-empty matrix of numbers.
    ZeroDivisionError : find 0 on diagonal, input matrix not invertible
    """
    # Check input validity.
    if not matrix.size:
        raise ValueError(f"input matrix cannot be empty")
    if not _is_square(matrix):
        raise ValueError(f"input must be square matrix")
    if matrix.ndim > 2:
        raise ValueError(f"matrix (2D array) expected"
                         + f"but {matrix.ndim}D array received")
    if not np.array_equal(np.tril(matrix), matrix):
        raise ValueError(f"input matrix must be lower triangular")

    # Get side-length of the matrix.
    n = len(matrix)

    # Check if the last entry on diagonal is 0 (since 
    # loop will not check this entry)
    if matrix[-1, -1] == 0:
        raise ZeroDivisionError(f"Found 0 at u[{n - 1}, {n - 1}], "
                                + f"`matrix` is not invertible")

    # Create inverse matrix.
    inverse = np.identity(n)
    
    # Calculate the inverse matrix.
    for col in range(n):
        for row in range(col + 1, n):
            # Calculate `M_{ij} / M_jj`
            if matrix[col, col] == 0:
                raise ZeroDivisionError(f"Found 0 at u[{col}, {col}], "
                                      + f"`matrix` is not invertible")
            coef = matrix[row, col] / matrix[col, col]

            # Perform row operation
            inverse[row, :col + 1] = inverse[row, :col + 1] \
                                     - coef * inverse[col, : col + 1]
    
    # Scale each row by `1 / matrix[row, row]`
    for row in range(n):
        inverse[row, :row + 1] /= matrix[row, row]
    
    return inverse

def get_lower_triangular(row_len : int, 
                         col_len : int | None = None) -> np.ndarray:
    """
    Generate an random n x m lower triangular matrix of integers.

    Parameter
    ---------
    row_len : int, row length of the matrix
        Positive integer, the row length `m` of the matrix.
    col_len : int, column length of the matrix
        Positive integer, optional, the column length `n` of the matrix.
        Default is `row_len`.
        
    Returns
    -------
    np.ndarray
        An `m`x`n` lower triangular matrix with randomly filled elements.
    
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
    return np.tril(matrix)

def main():
    # Set side-length of the matrix to be inverted.
    N = 5

    # Get `N` x `N` matrix.
    a = get_lower_triangular(N)
    
    # LU factorize `m`.
    a_inv = invert_lower_trig(a)

    print(f"A = {a}")
    print(f"A^(-1) = {a_inv}")

    # Compare the `A * A^{-1}` with the identity matrix.
    print("Errors:")
    print(f"{np.dot(a, a_inv) - np.identity(N)}")

if __name__ == "__main__":
    main()