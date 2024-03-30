# -*- coding: utf-8 -*-

"""
q2.py - Minimize |Ax - b| using LU factorization
"""

import numpy as np

# value `0.`
# Any positive value below is considered `0`.
TOLERANCE = 10 ** (-6)

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
        raise ZeroDivisionError(f"Found 0 at u[0, 0], "
                                + f"`matrix` is not LU diagonalizable")
    l[1:, 0] = matrix[1:, 0] / matrix[0, 0]

    # Calculate remaining entries for `l` and `u`.
    for i in range(1, n):
        # Calculate a row of entries for `u`.
        for col in range(i, n):
            u[i, col] = matrix[i, col] \
                           - np.dot(l[i, :i], u[:i, col])
        
        # Calculate a column of entries for `l`.
        if u[i, i] == 0 and i != n - 1:
            # Matrix cannot be LU factorized
            raise ZeroDivisionError(f"Found 0 at u[{i}, {i}], "
                                    + f"`matrix` is not LU diagonalizable")
        for row in range(i + 1, n): # `l` has `1` on diagonal
            l[row, i] = (matrix[row, i] \
                            - np.dot(l[row, :i], u[:i, i])) \
                            / u[i, i]
    return l, u

def solve_lower_triangular(l: np.ndarray, 
                           b: np.ndarray) -> np.ndarray:
    """
    Solve the matrix equation L * y = b where L is an lower triangular 
    and return y. 
    Note that due to round off errors, the factorized matrices 
    may not multiply exactly to the original matrix.
    Note that `L` is not necessarily unit lower triangular

    Parameters
    ----------
    matrix : np.ndarray, the upper triangular square matrix `L` of 
        the equation.
    b : np.ndarray, a column vector serves as the right hand side 
        of the equation Ly = b.
    
    Returns
    -------
    y : np.ndarray, the solution that satisfies equation `Ly = b`.

    Raises
    ------
    ValueError : if the size of `matrix` and `b` do not match.
    RuntimeError : there is no solution or infinitely many solutions.
    """
    # Check input validity.
    if not l.size:
        raise ValueError(f"input matrix cannot be empty")
    if not b.size:
        raise ValueError(f"right hand side vector `b` can't be empty")
    if not _is_square(l):
        raise ValueError(f"input must be square matrix")
    if l.ndim > 2:
        raise ValueError(f"matrix (2D array) expected"
                         + f"but {l.ndim}D array received")
    if len(b.shape) < 2 or b.shape[1] > 1:
        raise ValueError(f"`b` must be a column vector, but get shape"
                         + f"{b.shapes}")
    if len(l) != len(b):
        raise ValueError(f"size mismatch, matrix has {len(l)} columns, "
                         + f"but `b` has {len(b)} rows")
    
    # Get side-length of the matrix.
    n = len(l)

    # Create container for `y`.
    y = np.zeros((n, 1)).astype(float)

    # Solve the first entry.
    if abs(l[0, 0]) < TOLERANCE:
        msg = f"find 0 at l[{0}, {0}], " \
              + f"{b[0, 0]} at row {0} of b."
        if abs(b[n - 1, 0]) < TOLERANCE:
            raise RuntimeError(msg + "There are infinitely many solutions")
        raise RuntimeError(msg + "There is no solution.")
    y[0, 0] = b[0, 0] / l[0, 0]

    # Solve the system via forwards substitution.
    for row in range(1, n):
        # Find the coefficeint of `y_i` at current row.
        multiple = l[row, row]

        if abs(multiple) < TOLERANCE:
            msg = f"find 0 at l[{row}, {row}], " \
                  + f"{b[row, 0]} at row {row} of b."
            if abs(b[row, 0]) < TOLERANCE:
                raise RuntimeError(msg + "There are infinitely many solutions")
            raise RuntimeError(msg + "There is no solution.")

        # Solve entry `y_i`.
        y[row, 0] = (b[row, 0] - np.dot(l[row, 0:row], y[0:row, 0])) \
                     / multiple
    
    return y

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
    # Set size of `A` and `b`.
    M = 100
    N = 10

    # Generate `A`.
    a = get_rand_matrix(M, N)
    print(f"A = {a}")
    
    # Generate `b`.
    b = get_rand_matrix(M, 1)
    print(f"b = {b}")

    # Calculate `A^T A` and `A^T b`.
    at = a.transpose()
    ata = np.dot(at, a)
    print(f"A^T A = {ata}")
    ab = np.dot(at, b)
    print(f"Ab = {ab}")

    # LU factorize `A^T A`.
    factors = lu_factorize(ata)
    l: np.ndarray = factors[0]
    u: np.ndarray = factors[1]

    print(f"l = {l}")
    print(f"u = {u}")

    # Solve Ly = Ab.
    y: np.ndarray = solve_lower_triangular(l, ab)
    print(f"y = {y}")

    # Solve Ux = y.
    x: np.ndarray = solve_upper_triangular(u, y)
    print(f"x = {x}")

    # Error Check.
    print(f"e = {np.dot(a, x) - b}")

if __name__ == "__main__":
    main()