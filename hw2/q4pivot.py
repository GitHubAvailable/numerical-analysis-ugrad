# -*- coding: utf-8 -*-

"""
q4pivot.py - Pivot and LU factorize an matrix of n x n.
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

def swap_entries(m : np.ndarray, 
                 pos1 : tuple[int, int], 
                 pos2 : tuple[int, int]):
    """Swap two entries of a matrix."""
    tmp = m[pos1[0], pos1[1]]
    m[pos1[0], pos1[1]] = m[pos2[0], pos2[1]]
    m[pos2[0], pos2[1]] = tmp

# Only for display step purpose, do not call if `m` is not identity.
def _swap_row_display(m : np.ndarray, 
                      row1 : int, 
                      row2 : int):
    """Display the swapping row operation matrix and then rollback."""
    swap_entries(m, (row1, row1), (row1, row2))
    swap_entries(m, (row2, row2), (row2, row1))

    print(f"P_{row1 + 1} = {m}")

    # Rollback.
    swap_entries(m, (row1, row1), (row1, row2))
    swap_entries(m, (row2, row2), (row2, row1))

def _pivot_factorize(matrix : np.ndarray, 
                     show_step : bool = False) -> np.ndarray:
    """
    Return the matrix U of the LU factorization of input square matrix 
    while displaying steps if needed. 
    Note that due to round off errors, the factorized matrices 
    may not multiply exactly to the original matrix.

    Parameter
    ---------
    matrix: np.ndarray, the matrix to be LU factorized.
    show_step: bool, whether row operation matrices needs to be printed.
        Boolean, optional, true will print all row operation matrices used.
        Default is `False`.

    Returns
    -------
    np.ndarray
        The matrix U of the LU factorization of the pivoted input matrix.
    
    Raises
    ------
    ValueError : if the input is not a square, non-empty matrix of numbers.
    ZeroDivisionError : find 0 after pivoting, input matrix not 
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

    # Create container for row operation matrix and `U`.
    m: np.ndarray = np.identity(n) # row operation matrix
    u: np.ndarray = matrix.copy().astype(float)

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
        if show_step:
            # Show P_i
            _swap_row_display(m, col, max_row_index)
        
        if max_abs < TOLERANCE and col < n - 1:
            # Column already 0, not LU factorizable
            raise ZeroDivisionError(f"Found {u[col, col]} at "
                                    + f"u[{col}, {col}] after "
                                    + f"pivoting, `matrix` is "
                                    + f"not LU diagonalizable")
        
        # Perform Gauss elimination on column `col`.
        for row in range(col + 1, n):
            coef = u[row, col] / u[col, col]
            u[row, col:] -= coef * u[col, col:]

            if not show_step:
                continue
            m[row, col] = -coef
        
        if not show_step:
            continue

        # Show L_i
        print(f"L_{col + 1} = {m}")

        # Reset `m`.
        # m[a:, b] is selected as row vector.
        m[col + 1:, col] = np.zeros((n - col - 1))
    
    return u

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
    N = 3

    # Get `N` x `N` matrix.
    a = get_rand_matrix(N)
    print(f"A = {a}")
    
    # LU factorize `m` with pivoting.
    u = _pivot_factorize(a, show_step=True)

    print(f"U = {u}")

if __name__ == "__main__":
    main()