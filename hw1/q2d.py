# -*- coding: utf-8 -*-

"""
q2d.py - plot the error of an estimation of a function over an interval
"""

import math
import numpy as np
from typing import Callable
from matplotlib import pyplot as plt

def _d_f(x: float) -> float:
    return math.cos(x)

def _d_f_estimate(x: float, 
                  h: float) -> float:
    return (math.sin(x + h) - math.sin(x - h)) / (2 * h)

def get_errors(func: Callable[[float], float], 
               estimate: Callable[[float, float], float], 
               h_range: tuple[float, float], 
               target_x: float, 
               *, 
               sample_size: int = 50, 
               step: float | None = None, 
               is_log: bool = False, 
               take_abs: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate and return the error for evenly generated step size h.

    Parameters
    ----------
    func : Callable[[float], float]
        The original function that provide the desired value.
    estimate : Callable[[float], float]
        The function that provides an estimated value of the original function.
    h_range : tuple[float, float]
        The half-open interval where the value of `h` will be evenly generated 
        (i.e., `[h_range[0], h_range[1])`).
    target_x : float
        The x-coordinate where the estimation and the desired 
        value are compared.
    sample_size : int, optional
        The number of points needed, will be neglected
        when `step` is specified. Default is 50.
    step: float, optional
        The step size between each `h` value, only available for linear space. 
        Specifying will override `sample_size`.
    is_log : bool, optional
        True if the data need to be taken on logarithmic scale. 
        Default is `False`.
    take_abs : bool, optional
        True if the errors need to be in absolute value. Default is `True`.
    
    Returns
    -------
    errors : ndarray
        `num` samples, equally spaced on a linear or logrithmic scale.
    """
    # Generate array for `h`.
    h_vals: np.ndarray
    if is_log:
        # Logrithmic scale.
        if sample_size <= 1 or type(sample_size) != int:
            raise ValueError(f"'sample_size' must be positive integer!"
                             + f" But got {sample_size}")
        h_vals = np.logspace(h_range[0], 
                             h_range[1], 
                             num=sample_size, 
                             endpoint=False)
    elif step:
        # Linear scale, step specified.
        h_vals = np.arange(h_range[0], h_range[1], step=step)
    else:
        # Linear scale, step unspecified.
        if sample_size <= 1 or type(sample_size) != int:
            raise ValueError(f"'sample_size' must be positive integer!"
                             + f" But got {sample_size}")
        h_vals = np.linspace(h_range[0], 
                             h_range[1], 
                             num=sample_size, 
                             endpoint=False)
    
    # Get the desired x_value.
    target_val: float = func(target_x)

    # Prepare vectorized functions.
    err_func = np.vectorize(estimate)

    # Get error for each h value.
    err_vals: np.ndarray = err_func(target_x, h_vals) - target_val
    
    if take_abs:
        return h_vals, np.abs(err_vals)
    return h_vals, err_vals

def main():
    # Define constants
    MACHINE_ERROR_32 = 10 ** (-6)
    MACHINE_ERROR_64 = 10 ** (-16)

    # Set parameters.
    target_x = np.pi / 4
    h_range = (-10, 0)
    sample_size = 200
    predicted_h = ((3 * MACHINE_ERROR_64) / (math.cos(target_x))) ** (1 / 3)

    data = get_errors(_d_f, _d_f_estimate, h_range, target_x, 
                      sample_size=sample_size, is_log=True)
    
    # Extract data.
    h_vals = data[0]
    err_vals = data[1]
    
    # Set figure display.
    plt.xscale('log')
    plt.yscale('log')
    plt.xticks(np.logspace(h_range[0], h_range[1], num=11))
    plt.yticks(np.logspace(-13, -1, num=13))
    plt.grid(True)

    # Set texts.
    plt.xlabel(r"$h$")
    plt.ylabel("Error")
    plt.title(r"Error $v$. $h$ at $x=\frac{\pi}{4}$")
    plt.text(10 ** (-8), 10 ** (-5), 
             r"Error = $\left|\frac{sin\left(\frac{\pi}{4} + h\right) - sin\left(\frac{\pi}{4} - h\right)}{2h} - cos\left(\frac{\pi}{4}\right)\right|$")
    
    # Plot data.
    plt.plot(h_vals, err_vals)

    # Plot predicted error.
    f_estimate = np.vectorize(_d_f_estimate)
    estimate_err = abs(f_estimate(target_x, predicted_h) - _d_f(target_x))
    plt.plot(predicted_h, estimate_err, 
             'ro', markersize=4)
    print(estimate_err)
    plt.text(10 ** (-5), 10 ** (-12), 
             r"$(7.514 \times 10^{-6}, 6.334 \times 10^{-12})$")

    plt.show()

if __name__ == "__main__":
    main()