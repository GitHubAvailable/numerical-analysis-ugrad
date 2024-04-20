# -*- coding: utf-8 -*-

"""
q4.py - Numerically intergrate specific functions by Trapezium 
rule or Simpson's rule.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes as axes
from typing import Callable

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

def trapezium(f : Callable[[float], float], 
              start : float, 
              end : float) -> float:
    """
    Return the numerical integral of `f` from `start` to `end` calculated by 
    Trapezium rule.
    
    Parameters
    ----------
    f : Callable[[float], float], the function to be integrated.
    start : float, the lower limit of the integral.
    end : float, the upper limit of the integral.
    
    Returns
    -------
    float
        The integral calculated by Trapezium rule.
    """
    if start == end:
        return 0
    
    return 0.5 * (end - start) * (f(start) + f(end))

def simpson(f : Callable[[float], float], 
            start : float, 
            end : float) -> float:
    """
    Return the numerical integral of `f` from `start` to `end` calculated by 
    Simpson's rule.
    
    Parameters
    ----------
    f : Callable[[float], float], the function to be integrated.
    start : float, the lower limit of the integral.
    end : float, the upper limit of the integral.
    
    Returns
    -------
    float
        The integral calculated by Simpson's rule.
    """
    if start == end:
        return 0
    
    # Get mid point of the interval.
    mid = start + (end - start) / 2
    
    return (end - start) / 6 * (f(start) + 4 * f(mid) + f(end))

def main():
    # Initialize `h`.
    h = np.array([2 ** (-i) for i in range(7)])
    print(f"h: {h}")

    # Get vectorized integral functions.
    trape_vec = np.vectorize(trapezium)
    simp_vec = np.vectorize(simpson)

    # Integrate `func1`.
    trape1 = trape_vec(func1, -h, h)
    simp1 = simp_vec(func1, -h, h)

    print(f"Trapezium Rule for function 1: {trape1}")
    print(f"Simpson's Rule for function 1: {simp1}")

    # Integrate `func2`.
    trape2 = trape_vec(func2, 0, h)
    simp2 = simp_vec(func2, 0, h)

    print(f"Trapezium Rule for function 2: {trape2}")
    print(f"Simpson's Rule for function 2: {simp2}")

    # Plot graphs.
    fig1 = plt.figure()
    plt.plot(h, trape1, "-o", label = r"Trapezium Rule")
    plt.plot(h, simp1, "-o", label=r"Simpson's Rule")

    plt.xscale("log", base=2)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.title(r"$\int^{h}_{-h} \exp(x^3) + \sin(x^2) + x dx$ "
              + r"over Different $h$")

    plt.show()

    fig2 = plt.figure()
    plt.plot(h, trape2, "-o", label = r"Trapezium Rule")
    plt.plot(h, simp2, "-o", label=r"Simpson's Rule")

    plt.xscale("log", base=2)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.title(r"$\int^{h}_{0} \cos(x^5 + 4x^4 + 3x^3 + 2x^2 + x + 1) dx$ "
              + r"over Different $h$")

    plt.show()

if __name__ == "__main__":
    main()