# -*- coding: utf-8 -*-

"""
q4.py - use Newton's method to find all roots of f(x) in given interval
"""

import math
import warnings

def _f(x: float) -> float:
    """
    Return the value of sin(x) / x (return 1 if x is 0).

    Parameter
    ---------
    x : float, the input value

    Returns
    -------
    float
        the value of sin(x) / x (return 1 if x is 0).
    """
    if x == 0:
        return 1
    
    return math.sin(x) / x

def _f_derivative(x: float) -> float:
    """
    Return the derivative of sin(x) / x evaluated at x (return 0 if x is 0).

    Parameter
    ---------
    x : float, the input value

    Returns
    -------
    float
        the derivative of sin(x) / x evaluated at x (return 0 if x is 0).
    """
    if x == 0:
        return 0
    
    return (math.cos(x) * x - math.sin(x)) / (x ** 2)

def find_root(guesses: list[float], 
              lower: float = -float("inf"), 
              upper: float = float("inf"), 
              *, 
              tolerance: float = 10 ** (-6), 
              omit_invalid: bool = True) -> list[float]:
    """
    Find the approximated root of `f` in specify closed interval 
    using Newton's method for each inital guess.

    Parameter
    ---------
    x : list of float values, a list of inital guesses
    lower : float, the lower bound of the interval, default ``-inf``
    upper : float, the upper bound of the interval, default ``inf``
    tolerance : float, tolerance of root, default ``10 ** (-6)``
    omit_invalid : bool, discard invalid solution, default ``True``

    Returns
    -------
    list[float]
        the approximated root for each input x with their 
        order the same as their inital guesses (i.e., the solution
        of the first inital guess is the first in the solution list), 
        omit solutions outide of interval ``[lower, upper]`` when 
        `omit_invalid` is `True`.
    """
    solutions: list[float] = []

    # Discard iteration when not a good guess.
    quit_iter = False

    for guess in guesses:
        x_curr = guess

        # Check if can perform iteration.
        if abs(_f_derivative(x_curr)) < tolerance: 
            if abs(_f(x_curr)) < tolerance:
                warnings.warn(f"Initial guess {x_curr} dropped because"
                              + f"f'({x_curr}) is 0.", RuntimeWarning)
            solutions.append(x_new)    
            continue
            

        # Perform first iteration using Newton's method.
        x_new = x_curr - _f(x_curr) / _f_derivative(x_curr)

        # Iterate until `x_curr - x_new` within tolerance.
        while abs(x_new - x_curr) >= tolerance:
            x_curr = x_new

            if abs(_f_derivative(x_curr)) < tolerance: 
                if abs(_f(x_curr)) < tolerance:
                    warnings.warn(f"Initial guess {guess} dropped because"
                                  + f"f'({x_curr}) is 0.", RuntimeWarning)
                solutions.append(x_new)
                quit_iter = True    
                break

            if quit_iter:
                continue

            x_new = x_curr - _f(x_curr) / _f_derivative(x_curr)
        
        # Check if the solution is need to be recorded.
        if omit_invalid and (x_new < lower or x_new > upper):
            continue

        solutions.append(x_new)
    
    return solutions
    

def main():
    # Setup tolerance.
    tolerance = 10 ** (-7)
    # Define inital guess.
    guesses = [-10, -5, -1, 1, 5, 10]

    print(f"The roots are {find_root(guesses, -10, 10, tolerance=tolerance)}.")

if __name__ == "__main__":
    main()