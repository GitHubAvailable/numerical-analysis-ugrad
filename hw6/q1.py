# -*- coding: utf-8 -*-

"""
q1.py - Numerically intergrate specific functions by composite
Trapezium rule or Simpson's rule.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Any, Callable
import warnings

def func1(x : np.ndarray) -> np.ndarray:
    """Return value of `exp(x ** 3) + sin(x ** 2) + x`."""
    # Find intermediate values.
    sq = np.float_power(x, 2)
    cube = np.float_power(x, 3)
    
    return np.exp(cube) + np.sin(sq) + x

def func2(x : np.ndarray) -> np.ndarray:
    """Return value of `cos(x ** 5 + 4x ** 4 + 3x ** 3 + 2x ** 2 + x + 1)`."""
    # Find intermediate values.
    sq = np.float_power(x, 2)
    cube = np.float_power(x, 3)
    fourth = np.float_power(x, 4)
    fifth = np.float_power(x, 5)

    return np.cos(fifth + 4 * fourth + 3 * cube + 2 * sq + x + 1)

def composite_trapezium(f : Callable[[float], float], 
                        start : float, 
                        end : float, 
                        step : float) -> float:
    """
    Return the numerical integral of `f` from `start` to `end` calculated by 
    composite Trapezium rule.
    
    Parameters
    ----------
    f : Callable[[float], float], the function to be integrated.
    start : float, the lower limit of the integral.
    end : float, the upper limit of the integral.
    step : float, nonzero number, the step size between datapoints where 
        the function will be evaluated. The whole range will 
        start from `start` to `end` or the point where it is less 
        than the end but the absolute difference between them is 
        less than `step`.
    
    Returns
    -------
    float
        The integral calculated by composite Trapezium rule.
    
    Raises
    ------
    ValueError : the interval is invalid or the step is greater than 
        the length of the interval.
    """
    # Check input validity.
    if start == end:
        return 0
    if step == 0:
        raise ValueError(f"`step` cannot be 0")
    if abs(step) > abs(end - start):
        raise ValueError(f"`abs(step)` cannot be greater than interval length,"
                         + f" expected abs({end} - {start}) = "
                         + f"{abs(end - start)}, but received {step}")
    if step * (end - start) < 0:
        raise ValueError(f"`step` should be the same sign as `end - start`, "
                         + f"`end - start` = {end - start}, but `step`"
                         + f" is {step}")
    
    # Create containers for `x_i`.
    x = arange(start, end, step)

    # Record length of `x`.
    l = len(x)

    # Calculate coefficient `h`.
    h = (end - start) / (l - 1)

    vect_f = np.vectorize(f) # vectorize function
    y: np.ndarray = vect_f(x) # evaluation function at `x`

    return h * (np.sum(y) - 0.5 * (y[0] + y[-1]))

def composite_simpson(f : Callable[[float], float], 
                      start : float, 
                      end : float, 
                      step : float) -> float:
    """
    Return the numerical integral of `f` from `start` to `end` calculated by 
    composite Simpson's rule.
    
    Parameters
    ----------
    f : Callable[[float], float], the function to be integrated.
    start : float, the lower limit of the integral.
    end : float, the upper limit of the integral.
    step : float, nonzero number, the step size (one end of subinterval to 
        the midpoint) between datapoints where the function 
        will be evaluated. The whole range will start from `start` to 
        `end` or the point where it is less than the end 
        but the absolute difference between them is less than `step`.
    
    Returns
    -------
    np.float
        The integral calculated by composite Simpson's rule.
    
    Raises
    ------
    ValueError : the interval is invalid or the step is greater than 
        the length of the interval.
    """
    # Check input validity.
    if start == end:
        return 0
    if step == 0:
        raise ValueError(f"`step` cannot be 0")
    if abs(step) > abs(end - start) / 2:
        raise ValueError(f"`abs(step)` cannot be greater than half "
                         + f"interval length, expected abs(0.5"
                         + f"* ({end} - {start})) = {0.5 * abs(end - start)},"
                         + f" but received {step}")
    if step * (end - start) < 0:
        raise ValueError(f"`step` should be the same sign as `end - start`, "
                         + f"`end - start` = {end - start}, but `step`"
                         + f" is {step}")
    
    # Create containers for `x_i`.
    x = arange(start, end, step)

    # Record length of `x`.
    l = len(x)

    if not l % 2:
        # Simpson method requires `2m + 1` datapoints.
        warnings.warn(f"{l} datapoints generated, odd number of "
                      + f"datapoints required, last one dropped", RuntimeWarning)
        l -= 1

    # Calculate coefficient `h`.
    h = (end - start) / (l - 1) # `l - 1` is `M`

    vect_f = np.vectorize(f) # vectorize function
    y: np.ndarray = vect_f(x) # evaluation function at `x`

    num = 0 # create container for adding weighted `f(x)`.
    
    # Calculate weight sum of `f(x)`.
    for index in range(0, l):
        if index % 2:
            num += 4 * y[index]
            continue
        num += 2 * y[index]
    
    return h / 3 * (num - y[0] - y[l - 1])

def arange(start : float, 
           end : float, 
           step : float, 
           dtype : type = float) -> np.ndarray:
    """
    Return an array of equally spaced numbers including upper bound, 
    if possible.

    Parameters
    ----------
    start : float, the lower bound of the interval.
    end : float, the upper bound of the interval.
    step : float, the step length of the interval.

    Returns
    -------
    np.ndarray
        An numpy array of equally spaced numbers in [start, end].
    """
    # Check input validity.
    if step == 0:
        raise ValueError(f"`step` cannot be 0")
    if abs(step) > abs(end - start):
        raise ValueError(f"`abs(step)` cannot be greater than interval length,"
                         + f" expected abs({end} - {start}) = "
                         + f"{abs(end - start)}, but received {step}")
    
    if (end - start) % step == 0:
        # Need to include bound.
        return np.arange(start, end + step, step, dtype)
    
    return np.arange(start, end, step, dtype)

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
    print(f"|{label_a : ^10}|{label_b : ^25}|")
    print(f"|{':' + '-' * 8 + ':'}|"
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
        print(f"| {row[0] : <9}|{row[1] : ^25}|")

def main():
    LOW = 0
    HIGH = 1

    # Initialize bounds.
    BOUND_1 = (-1, 1)
    BOUND_2 = (0, 2)

    # Initialize `h`.
    steps = np.array([2 ** (-i) for i in range(7)])
    print(f"Steps: {steps}")

    # Get vectorized integral functions.
    ctrape_vec = np.vectorize(composite_trapezium)
    csimp_vec = np.vectorize(composite_simpson)

    # Integrate `func1`.
    trape1 = ctrape_vec(func1, BOUND_1[LOW], BOUND_1[HIGH], steps)
    simp1 = csimp_vec(func1, BOUND_1[LOW], BOUND_1[HIGH], steps)

    print(f"Trapezium Rule for function 1: ")
    print_results(steps, trape1, "Step", "Result")
    print(f"Simpson's Rule for function 1: ")
    print_results(steps, simp1, "Step", "Result")

    # Integrate `func2`.
    trape2 = ctrape_vec(func2, BOUND_2[LOW], BOUND_2[HIGH], steps)
    simp2 = csimp_vec(func2, BOUND_2[LOW], BOUND_2[HIGH], steps)

    print(f"Trapezium Rule for function 2: ")
    print_results(steps, trape2, "Step", "Result")
    print(f"Simpson's Rule for function 2: ")
    print_results(steps, simp2, "Step", "Result")

    # Plot graphs.
    fig1 = plt.figure()
    plt.plot(steps, trape1, "-o", label=r"Composite Trapezium Rule")
    plt.plot(steps, simp1, "-o", label=r"Composite Simpson's Rule")

    plt.xscale("log", base=2)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    plt.title(r"$\int^{1}_{-1} \exp(x^3) + \sin(x^2) + x dx$ "
              + r"over Different Step Size")

    plt.show()

    fig2 = plt.figure()
    plt.plot(steps, trape2, "-o", label=r"Composite Trapezium Rule")
    plt.plot(steps, simp2, "-o", label=r"Composite Simpson's Rule")

    plt.xscale("log", base=2)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    plt.title(r"$\int^{2}_{0} \cos(x^5 + 4x^4 + 3x^3 + 2x^2 + x + 1) dx$ "
              + r"over Different Step Size")

    plt.show()

if __name__ == "__main__":
    main()