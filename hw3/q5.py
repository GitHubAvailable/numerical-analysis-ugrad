# -*- coding: utf-8 -*-

"""
q5.py - Schmidt normalize a group of vectors.
"""

import numpy as np

def _is_postive_integer(x : int) -> bool:
    """Return if input is positive integer."""
    return type(x) == int and x > 0

def schimit_normalize(vectors: np.ndarray) -> np.ndarray:
    """
    Performing Schmidt normalization and return a matrix of 
    normalized unit column vectors.

    Parameters
    ----------
    vectors : np.ndarray
    A matrix that contains the group of vectors to be
        Schimit-Normalized.
    
    Return
    ------
    basis : np.ndarray
    A matrix with each column being the 
        Schmidt normalized vector of that column in `A`.

    Raises
    ------
    ValueError : invalid input, there are more columns than rows, or 
        found zero vector during normalization.
    """
    # Check input validity.
    if not vectors.size:
        raise ValueError(f"input matrix cannot be empty")
    if vectors.ndim > 2:
        raise ValueError(f"matrix (2D array) expected"
                         + f"but {vectors.ndim}D array received")
    if vectors.shape[0] < vectors.shape[1]:
        raise ValueError(f"columns cannot be more than rows, "
                         + f"but received {vectors.shape[1]} "
                         + f"{vectors.shape[0]}D vectors")
    
    # Get the size of `vectors`.
    dim = vectors.shape[0]
    n = vectors.shape[1]

    # Create container for normalized vectors.
    basis = np.zeros((dim, n))

    # Normalize the first vector.
    if not vectors[:, 0].any():
        raise RuntimeError(f"Found zero vector at column 0")
    basis[:, 0] = vectors[:, 0] / np.linalg.norm(vectors[:, 0], ord=2)

    # Normalize the remaining vectors.
    for col in range(n):
        # Get current column vector.
        cur_vec = vectors[:, col]

        # Check if `cur_vec` is zero vector.
        if not cur_vec.any():
            raise RuntimeError(f"Found zero vector at column {col}")

        # Subtraction projection in other normalized vectors
        for index in range(col):
            cur_vec -= np.dot(cur_vec, basis[:, index]) * basis[:, index]

            # Check if `cur_vec` becomes zero vector.
            if not cur_vec.any():
                raise RuntimeError(f"Found zero vector when "
                                   + f" normalizing column {col}")
        
        # Store normalized `cur_vec`.
        basis[:, col] = cur_vec / np.linalg.norm(cur_vec, ord=2)

    return basis

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
    # Set number and dimension of vectors
    N = 100
    DIM = 200

    # Get group of vectors.
    vectors = get_rand_matrix(DIM, N, as_float=True)
    print(f"V = {vectors}")

    # Schmidt Normalize `vectors`.
    basis = schimit_normalize(vectors)
    print(f"B = {basis}")

    # Error Check
    print("Error Check:")
    print(f"B^T B = {np.dot(basis.transpose(), basis)}")

if __name__ == "__main__":
    main()