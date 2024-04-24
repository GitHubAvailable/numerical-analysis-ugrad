# -*- coding: utf-8 -*-

"""
q2.py - Calculate the best 2-norm approximation of specified functions
at certain intervals.
"""

import abc
import matplotlib.pyplot as plt
import numpy as np
from typing import Any, Callable, Sequence, overload
from scipy.integrate import quad


class Function(abc.ABC):
    """
    A class that represents an transform of a function under 
    a specific basis.
    """
    @abc.abstractmethod
    def eval(self, x : Any) -> Any:
        """
        Return the value of the function at specified index 
        for input `x`.
        
        Parameter
        ---------
        x : any, the input to the function.
        
        Returns
        -------
        Any
            The value of the specified function for input `x`.
        """
        pass


class Polynomial(Function):
    """
    A class that represent a polynomial of real coefficients.
    """
    def __init__(self, 
                 coefficients : Sequence[float]) -> None:
        """
        Create a polynomial of order `n` with corresponding coefficients.

        Parameters
        ----------
        coefficients : Sequence[float], the coefficients of each order
            of the polynomial in increasing order.
        """
        self.coefficients = coefficients
    
    @property
    def n(self):
        return len(self.coefficients)
    
    def eval(self, x : float | np.ndarray) -> float | np.ndarray:
        if not any(x):
            # Return np.ndarray if `x` is np.ndarray.
            return 0 * x + self.coefficients[0]
        
        # Initiate container to store result.
        y = 0 * x + self.coefficients[0] # avoid 0 ** 0 in for loop

        # Evaluate the polynomial.
        for order in range(1, len(self.coefficients)):
            y += self.coefficients[order] * np.float_power(x, order)
        return y
    
    def __str__(self) -> str:
        return f"p_{self.n}(x) = " \
               + ' + '.join([
                   f"{coef}x^{i}" for i, coef in enumerate(self.coefficients)
               ])


def func1(x : float) -> float:
    """Return value of `exp(x)`."""
    return np.exp(x)

def func2(x : float) -> float:
    """Return value of `sin(x)`."""
    return np.sin(x)

def norm2_approx(func : Callable[[float], float], 
                 n : int, 
                 start : float, 
                 end : float) -> Polynomial:
    """
    Calculate and return the 2-norm best approximation polynomial 
    of the specfied function.
    
    Parameters
    ----------
    func : Callable[[float], float], the function to be approximated.
    n : int, nonnegative, the order of the polynomial to be found.
    start : float, the lower bound of the interval, must be finite 
        and smaller than end.
    end : float, the upper bound of the interval, must be finite.

    Returns
    -------
    Polynomial
        The 2-norm best approximation polynomial of 
        the function `func.`
    
    Raises
    ------
    ValueError : one of the input is invalid, `start` is greater 
        than or equal to `end` or one of them is infinite.
    """
    # Check input validity.
    if n < 0:
        raise ValueError(f"`n` must be nonnegative, but received {n}")
    if start >= end:
        raise ValueError(f"`start` must be less than `end`, but "
                         + f"received `start` ({start}) and `end` ({end})")
    if abs(start) in (np.inf, np.nan):
        raise ValueError(f"`start` must be finite, but received {start}")
    if abs(end) in (np.inf, np.nan):
        raise ValueError(f"`start` must be finite, but received {start}")
    
    # Create container.
    b = np.zeros(n + 1) # b_0, b_1, ... , b_{n}

    # Calculate `b_j`.
    for index in range(n + 1):
        # Evaluate integral f(x)x^k from `start` to `end`.
        b[index] = quad(lambda x: func(x) * x ** index, start, end)[0]
    
    # Calculate matrix M.
    m_ij = lambda i, j: (end ** (i + j + 1) - start ** (i + j + 1)) \
                        / (i + j + 1)
    m: np.ndarray = np.fromfunction(m_ij, (n + 1, n + 1), dtype=float)

    # Use results of equation `mx = b` as coefficients of polynomial.
    return Polynomial(np.linalg.solve(m, b))

def _print_table(x : np.ndarray, 
                 y : np.ndarray, 
                 approx : Sequence[np.ndarray], 
                 *labels) -> None:
    """
    Print calculation result in a Markdown table with specified 
    column label, fill `Col <number>` for missing labels and 
    empty string for missing data entries.
    
    Parameters
    ----------
    x : np.ndarray, the input values.
    y : np.ndarray, the exact output values.
    approx : Sequence[np.ndarray], less than equal to 2D, 
        the approximated values, with each index reprenenting 
        the data returned by one approximation function.
    labels : the name of each column to be printed.
    """
    # Record number of columns.
    n = 2 + len(approx)

    # Get lengths.
    data_len = (len(x), len(y)) + tuple([len(ap) for ap in approx])

    # Build header and fill default col names if needed.
    header = "| " + " | ".join(
        [f'{label : ^20}' for label in labels] \
        + [f'{"Col" + f"{i + 1}" : ^20}' for i in range(len(labels), n)]
    ) + " |"

    # Print header.
    print(header)
    print("|" + "|".join([f":{'-' * 20}:" for _ in range(n)]) + "|")

    # Print data.
    for index in range(max(data_len)):
        # Build row
        row = "| " + " | ".join([
            f"{x[index] if index < data_len[0] else '' : ^20}", 
            f"{y[index] if index < data_len[1] else '' : ^20}"
        ] + [
            f"{ap[index] if index < data_len[i + 2] else '' : ^20}"
            for i, ap in enumerate(approx)
        ]) + " |"
        print(row)

def _plot(x : np.ndarray, 
          y : np.ndarray, 
          approx : Sequence[np.ndarray], 
          title : str = "", 
          *labels) -> None:
    """
    Print calculation result in a Markdown table with specified 
    column label, fill `Col <number>` for missing labels and 
    empty string for missing data entries.
    
    Parameters
    ----------
    x : np.ndarray, the input values.
    y : np.ndarray, the exact output values.
    approx : Sequence[np.ndarray], less than equal to 2D, 
        the approximated values, with each index reprenenting 
        the data returned by one approximation function.
    title : str, the titile of the graph
    labels : the name of each curve to be printed, 
        default is `f(x)` for exact and `approx <i>` 
        for approximated.
    """
    # Recond the length of `labels`.
    n = len(labels)

    plt.figure() # Create a new figure.

    # Plot exact value.
    plt.plot(x, y, "-", label=(labels[0] if 0 < n else "f(x)"))

    # Plot approximated values.
    for index, ap in enumerate(approx):
        plt.plot(x, ap, "--", 
                 label=(labels[index + 1] if index + 1 < n \
                        else f"approx {index}"))
    # Configure graph settings.
    plt.xlabel("x")
    plt.ylabel("y")
    plt.ylim(bottom=0)
    plt.xlim((0, 1))
    plt.legend()
    plt.grid(True)
    plt.title(title)

    plt.show()


def main():
    # Define constants.
    START_INDEX = 0
    END_INDEX = 1
    INTERVAL = [0, 1]
    SAMPLE_SIZE = 100
    N = [3, 5, 10]

    # Generate test datapoints.
    x = np.linspace(INTERVAL[START_INDEX], INTERVAL[END_INDEX], 
                    num=SAMPLE_SIZE)
    
    print(f"Generated {SAMPLE_SIZE} input `x`.")

    # Get exact value of `func1(x)`.
    func1_vec = np.vectorize(func1)
    y1 = func1_vec(x)

    # Get 2-norm best approximation polynomials.
    func1_approx = [norm2_approx(func1, order, 
                    INTERVAL[START_INDEX], INTERVAL[END_INDEX])
                    for order in N]
    
    print("f_1(x) = exp(x)")
    for p in func1_approx:
        print(p)
    
    # Evaluate polynomials at `x`.
    y1_approx = [p.eval(x) for p in func1_approx]

    _print_table(x, y1, y1_approx, 
                 "x", "exp(x)", "p_3(x)", "p_5(x)", "p_10(x)")
    
    print()

    # Get exact value of `func2(x)`.
    func2_vec = np.vectorize(func2)
    y2 = func2_vec(x)

    # Get 2-norm best approximation polynomials.
    func2_approx = [norm2_approx(func2, order, 
                    INTERVAL[START_INDEX], INTERVAL[END_INDEX])
                    for order in N]
    
    print("f_2(x) = sin(x)")
    for p in func2_approx:
        print(p)
    
    # Evaluate polynomials at `x`.
    y2_approx = [p.eval(x) for p in func2_approx]

    _print_table(x, y2, y2_approx, 
                 "x", "sin(x)", "p_3(x)", "p_5(x)", "p_10(x)")
    
    # Plot graphs.
    _plot(x, y1, y1_approx, 
          r"Comparison of $\exp(x)$ and its Best 2-norm Approximation"
          + r" $p_i(x)$", 
          r"$\exp(x)$", r"$p_3(x)$", r"$p_5(x)$", r"$p_{10}(x)$")
    _plot(x, y2, y2_approx, 
          r"Comparison of $\sin(x)$ and its Best 2-norm Approximation"
          + r" $p_i(x)$", 
          r"$\sin(x)$", r"$p_3(x)$", r"$p_5(x)$", r"$p_{10}(x)$")

if __name__ == "__main__":
    main()