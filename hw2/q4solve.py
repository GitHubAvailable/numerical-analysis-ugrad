# -*- coding: utf-8 -*-

"""
q4solve.py - LU factorize and solve an n x n matrix equation by pivoting.
"""

import numpy as np

TOLERANCE = 10 ** (-6) # value `0.` below this value considered `0`.

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

def _pivot_convert(matrix : np.ndarray, 
                   b : np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Return the matrix U of the LU factorization of input square matrix 
    and the new right hand side of equation Ux = b. 
    Note that due to round off errors, the factorized matrices 
    may not multiply exactly to the original matrix.

    Parameters
    ----------
    matrix: np.ndarray, the matrix to be LU factorized.
    b : np.ndarray, column vector(s) serve as the right hand side 
        of the equation Ax = b.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        a tuple contains the matrix U of the LU factorization of 
        the input matrix as well as the right hand side
        of the equation Ux = b.`[0]` is U, and `[1]` is b.
    
    Raises
    ------
    ValueError : if the input is not a square, non-empty matrix of numbers, 
        or the size of `matrix` and `b` do not match.
    ZeroDivisionError : find 0 after pivoting, input matrix not 
        LU factorizable
    """
    # Check input validity.
    if not matrix.size:
        raise ValueError(f"input matrix cannot be empty")
    if not b.size:
        raise ValueError(f"right hand side vector `b` can't be empty")
    if not _is_square(matrix):
        raise ValueError(f"input must be square matrix")
    if matrix.ndim > 2:
        raise ValueError(f"matrix (2D array) expected"
                         + f"but {matrix.ndim}D array received")
    if len(matrix) != len(b):
        raise ValueError(f"size mismatch, matrix has {len(matrix)}, "
                         + f"but `b` has {len(b)}")

    # Get side-length of the matrix.
    n = len(matrix)

    # Create container for `U`.
    u: np.ndarray = matrix.copy().astype(float)
    b_new: np.ndarray = b.copy().astype(float)

    for col in range(n - 1):
        # Find row with the largest entry in the column
        max_row_index = col # row number of entry with largest absolute value
        max_abs = abs(u[col, col])

        # Find the row number with the largest element in absolute value.
        for row in range(col + 1, n):
            cur_abs = abs(u[row, col])
            if cur_abs <= max_abs:
                continue
            
            max_row_index = row
            max_abs = cur_abs
        
        # Swap `max_row` with current row.
        u[[col, max_row_index], col:] = u[[max_row_index, col], col:]
        b_new[[col, max_row_index], :] = b_new[[max_row_index, col], :]
        
        if max_abs < TOLERANCE and col < n - 1:
            # Column already 0, not LU factorizable
            raise ZeroDivisionError(f"Found {u[col, col]} at "
                                    + f"u[{col}, {col}] after "
                                    + f"pivoting, `matrix` is "
                                    + f"not LU diagonalizable")
        
        # Perform Gauss elimination on column `col`.
        for row in range(col + 1, n):
            coef = u[row, col] / u[col, col]
            # Simplified for `u` as entry in previous columns are 0
            u[row, col:] -= coef * u[col, col:]
            b_new[row, :] -= coef * b_new[col, :]
    
    return u, b_new

def solve_upper_triangular(u : np.ndarray, 
                           b : np.ndarray) -> np.ndarray:
    """
    Solve the matrix equation A * x = b where A is square matrix
    and return x. Note that due to round off errors, the factorized matrices 
    may not multiply exactly to the original matrix.

    Parameters
    ----------
    matrix : np.ndarray, the upper triangular square matrix `A` of 
        the equation.
    b : np.ndarray, a column vector serves as the right hand side 
        of the equation Ax = b.
    
    Returns
    -------
    x : np.ndarray, the solution that satisfies equation `Ax = b`.

    Raises
    ------
    ValueError : if the size of `matrix` and `b` do not match.
    RuntimeError : there is no solution or infinitely many solutions.
    """
    # Check input validity.
    if not u.size:
        raise ValueError(f"input matrix cannot be empty")
    if not b.size:
        raise ValueError(f"right hand side vector `b` can't be empty")
    if not _is_square(u):
        raise ValueError(f"input must be square matrix")
    if u.ndim > 2:
        raise ValueError(f"matrix (2D array) expected"
                         + f"but {u.ndim}D array received")
    if len(b.shape) < 2 or b.shape[1] > 1:
        raise ValueError(f"`b` must be a column vector, but get shape"
                         + f"{b.shapes}")
    if len(u) != len(b):
        raise ValueError(f"size mismatch, matrix has {len(u)} columns, "
                         + f"but `b` has {len(b)} rows")

    # Get side-length of the matrix.
    n = len(u)

    # Create container for `x`.
    x = np.zeros((n, 1)).astype(float)

    # Solve the last entry.
    if abs(u[n - 1, n - 1]) < TOLERANCE:
        msg = f"find 0 at u[{n - 1}, {n - 1}], " \
              + f"{b[n - 1, 0]} at row {n - 1} of b."
        if abs(b[n - 1, 0]) < TOLERANCE:
            raise RuntimeError(msg + "There are infinitely many solutions")
        raise RuntimeError(msg + "There is no solution.")
    x[n - 1, 0] = b[n - 1, 0] / u[n - 1, n - 1]

    # Solve the system via backwards substitution.
    for row in range(n - 2, -1, -1):
        # Find the coefficeint of `x_i` at current row.
        multiple = u[row, row]

        if abs(multiple) < TOLERANCE:
            msg = f"find 0 at u[{row}, {row}], " \
                  + f"{b[row, 0]} at row {row} of b."
            if abs(b[row, 0]) < TOLERANCE:
                raise RuntimeError(msg + "There are infinitely many solutions")
            raise RuntimeError(msg + "There is no solution.")

        # Solve entry `x_i`.
        x[row, 0] = (b[row, 0] - np.dot(u[row, row + 1:], x[row + 1:, 0])) \
                     / multiple
    
    return x

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
    N = 5

    for trial in range(4):
        print(f"--------- Example {trial + 1} ----------")
        # Get `N` x `N` matrix.
        a = get_rand_matrix(N)
        print(f"A = {a}")

        # Get vector `b`.
        b = get_rand_matrix(N, 1)
        print(f"b = {b}")

        # Pivot and LU factorize to get `u` and `b`.
        res = _pivot_convert(a, b)
        u = res[0]
        b_new = res[1]

        print(f"u = {u}")
        print(f"b_new = {b_new}")

        # Solve equation `Ux = b_new`.
        x = solve_upper_triangular(u, b_new)

        print(f"x = {x}")

        # Verify the correctness of the solution.
        print("Errors: ")
        print(f"{np.dot(a, x) - b}")

if __name__ == "__main__":
    main()