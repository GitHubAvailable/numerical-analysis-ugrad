# -*- coding: utf-8 -*-

"""
q4.py - Construct and analyze the discrete Fourier transform of 
the specified function.
"""

import abc
import numpy as np
import matplotlib.pyplot as plt
from typing import Any, Callable, List, Sequence


class FunctionTransform(abc.ABC):
    """
    A class that represents an transform of a function under 
    a specific basis.
    """
    def __init__(self, 
                 func : Callable[[Any], Any]) -> None:
        self.__func = func

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
    
    def eval_inputs(self, 
                    inputs : Sequence, 
                    default : Any) -> List[Any]:
        """
        Return the function value of the function for each input `x`.
        
        Parameters
        ----------
        inputs : Sequence, a list of input to the function.
        default : placeholder value, used when the function cannot be 
            evaluated at `x`. 

        Returns
        -------
        List[Any]
            The list of values of the function evaluated for 
            each input `x`.
        """
        vals = []

        for x in inputs:
            try:
                val = self.eval(x)
            except Exception as e:
                print(e)
                print(f"Warning : evaluation at {x} failed, "
                      + f"{default} inserted!")
                val = default

            vals.append(val)
        
        return vals


class DiscreteFourierTransform(FunctionTransform):
    """
    A class that represents the discrete Fourier transform of 
    a function on [0, l].
    """
    def __init__(self, 
                 func : Callable[[float], float], 
                 end : float, 
                 l : int) -> None:
        """
        Construct a discrete Fourier transform using given datapoints.

        Parameters
        ----------
        func : Callable[[float], float], the function to be transformed.
        end : float, nonzero, the end of the interval.
        l : int, positive, order of trignometric functions used (e.g.,
            `n` of `cos(n * pi)`).
        
        Raises
        ------
        ValueError : `end` is zero, `l` is not a positive integer.
        """
        # Check input validity.
        if not end:
            raise ValueError(f"`end` must be nonzero")
        if type(l) != int:
            raise ValueError(f"`l` must be an positive integer, "
                             + f"but received {l}")
        if l <= 0:
            raise ValueError(f"`l` must be positive, but received {l}")
        
        super().__init__(func)
        self.__end = end
        self.__l = l
        self.__coss, self.__sins = self.__getCoeff(func, end, l)
    
    def eval(self, x : float) -> float:
        res = 0

        # Sum up each terms.
        for k in range(0, self.__l + 1):
            res += self.__coss[k] * np.cos(k * np.pi * x / self.__end)
            res += self.__sins[k] * np.sin(k * np.pi * x / self.__end)
        return res
    
    def eval_inputs(self, 
                    inputs : Sequence, 
                    default : float = np.nan) -> List[float]:
        return super().eval_inputs(inputs, default)
    
    def printCoefficients(self):
        """Print all discrete Fourier coefficients in a table."""
        # Print header.
        print(f"| {'k' : >4} | {'a_k (cos)' : ^25} | {'b_k (sin)' : ^25} |")
        print(f"|:{'-' * 4}:|:{'-' * 25}:|:{'-' * 25}:|")

        # Print coefficients.
        for k in range(0, self.__l + 1):
            print(f"| {k : >4} | {self.__coss[k] : >25} |"
                  + f" {self.__sins[k] : >25} |")
    
    @staticmethod
    def __getCoeff(func : Callable[[float], float], 
                   end : float, 
                   l : int) -> np.ndarray:
        """Calculate the discrete Fourier coefficients using given data."""
        # Generate datapoints.
        x = np.linspace(0, end, 2 * l, endpoint=False)
        vec_func = np.vectorize(func)
        y = vec_func(x)

        # Record length of data.
        l = len(x)

        # Initialize coefficient containers.
        coss = [0.5 / l * np.sum(y)] # calculate 0.5 * A_0
        sins = [0] # add 0 here to match the length with coss

        # Calculate coefficients 1,..., l - 1.
        for k in range(1, l):
            # Store key results to avoid repeated calculation.
            param: np.ndarray = k * np.pi * x / end

            # Add coefficients
            coss.append(1 / l * sum(y * np.cos(param)))
            sins.append(1 / l * sum(y * np.sin(param)))
        
        # Calculate coefficients of `l`th order.
        coss.append(1 / l * sum(y * np.cos(2 * np.pi * np.arange(0, l) / l)))
        sins.append(0) # add 0 here to match the length with coss

        return coss, sins


def func(x : float) -> float:
    """Return the value of `np.log(x + 1)`."""
    return np.log(x + 1)

def _even_ext_func(x : float) -> float:
    """
    Return the value of `np.log(x + 1)` for `[0, 2 * np.pi]` and return
    the value of `np.log(4 * np.pi - x + 1)` for `[2 * np.pi, 4 * np.pi]`.
    """
    if x > 2 * np.pi and x < 4 * np.pi:
        return np.log(4 * np.pi - x + 1)
    return np.log(x + 1)

def print_results(col_a : np.array, 
                  col_b : np.array, 
                  label_a : str = "", 
                  label_b : str = "", 
                  default : Any = "") -> None:
    """
    Print results in `| col_a[i] | col_b[i] |` format with 
    column label .
    
    Parameters
    ----------
    col_a : np.array, the first column of data.
    col_b : np.array, the second column of data.
    label_a : str, optional, the label for the first column, default is `""`.
    label_b : str, optional, the label for the second column, default is `""`.

    Raise
    -----
    ValueError : invalid input for `max_a` or `max_b`.
    """

    # Print header.
    # NOTE: the width of each cell is currently required to be literal int
    print(f"|{label_a : ^25}|{label_b : ^25}|")
    print(f"|{':' + '-' * 23 + ':'}|"
          + f"{':' + '-' * 23 + ':'}|")
    
    # Get the length of column `a`, column `b`.
    len_a = len(col_a)
    len_b = len(col_b)

    # Print data.
    for index in range(max(len_a, len_b)):
        # Initialize container for row data.
        row = [default] * 2

        # Retrieve entries, if possible.
        if index < len_a:
            row[0] = col_a[index]
        if index < len_b:
            row[1] = col_b[index]
        
        # Print row.
        print(f"|{row[0] : ^25}|{row[1] : ^25}|")

def _plot(x : np.ndarray, 
          y1 : np.ndarray, 
          y2 : np.ndarray, 
          y3 : np.ndarray, 
          label1 : str = "y1", 
          label2 : str = "y2", 
          label3 : str = "y3", 
          title : str = "") -> None:
    """Plot `y1` and `y2` agaist `x` with specified curve labels and graph title."""
    plt.figure() # create a new figure.

    # Plot curves.
    plt.plot(x, y1, "-", label=label1)
    plt.plot(x, y2, "-.", label=label2)
    plt.plot(x, y3, "-.", label=label3)

    # Configure graph settings.
    plt.xlabel("x")
    plt.ylabel("y")
    plt.ylim(bottom=0)
    plt.xlim((0, 2 * np.pi))
    plt.legend()
    plt.grid(True)
    plt.title(title)

    # Show figure.
    plt.show()

def main():
    # Define constants
    L1 = 16 # Number of points used.
    L2 = 32
    END = 2 * np.pi
    END_EXT = 4 * np.pi

    # Create test input `x`.
    x = np.linspace(0, END, 100)

    # Get the vectorized original function.
    vec_func = np.vectorize(func)

    # Get precise value of `y`.
    y = vec_func(x)

    print("Actual Value:")
    print_results(x, y, "x", "log(x + 1)")

    # Create discrete Fourier transform for the original function.
    trans16 = DiscreteFourierTransform(func, END, L1)
    trans32 = DiscreteFourierTransform(func, END, L2)

    # Print coefficients.
    print("l = 16:")
    trans16.printCoefficients()
    print("l = 32")
    trans32.printCoefficients()

    # Evaluate transform functions at `x`.
    y16 = np.array(trans16.eval_inputs(x), dtype=float)
    y32 = np.array(trans32.eval_inputs(x), dtype=float)

    print("Original Function:")
    print_results(x, y16, "x", "y (l = 16)")
    print()
    print_results(x, y32, "x", "y (l = 32)")

    # Create discrete Fourier transform for the extended function.
    trans_ext16 = DiscreteFourierTransform(_even_ext_func, END_EXT, L1)
    trans_ext32 = DiscreteFourierTransform(_even_ext_func, END_EXT, L2)

    # Print coefficients.
    print("l = 16:")
    trans16.printCoefficients()
    print("l = 32")
    trans32.printCoefficients()

    # Evaluate transform functions at `x`.
    y_ext16 = np.array(trans_ext16.eval_inputs(x), dtype=float)
    y_ext32 = np.array(trans_ext32.eval_inputs(x), dtype=float)

    print("Extended Function:")
    print_results(x, y_ext16, "x", "y (l = 16)")
    print()
    print_results(x, y_ext32, "x", "y (l = 32)")

    # Calculate max error.
    max_err_y16 = max(abs(y16 - y))
    max_err_y32 = max(abs(y32 - y))
    max_err_y_ext16 = max(abs(y_ext16 - y))
    max_err_y_ext32 = max(abs(y_ext32 - y))

    print("Max Error:")
    print(f"l = 16: {max_err_y16} -> {max_err_y_ext16}")
    print(f"l = 32: {max_err_y32} -> {max_err_y_ext32}")

    # Plot graphs.
    _plot(x, y, y16, y32, 
          "log(x + 1)", "l = 16", "l = 32", 
          r"Comparison of DFT of $\log(x + 1)$ on $[0, 2\pi]$"
          + r" with Different $l$")
    _plot(x, y, y_ext16, y_ext32, 
          "log(x + 1)", "l = 16", "l = 32", 
          r"Comparison of DFT of Even Extension on $[0, 2\pi]$ "
          + r"with Different $l$")

if __name__ == "__main__":
    main()