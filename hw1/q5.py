# -*- coding: utf-8 -*-

"""
q5.py - use Secant method to minimize given functions in given interval
"""

import math
import warnings
from typing import Callable # for function type annotation


def _phi_a(x: float) -> float:
    """
    Return the value of sin(x) evaluated at x.

    Parameter
    ---------
    x : float, the input value

    Returns
    -------
    float
        the value of sin(x) evaluated at x.
    """
    return math.sin(x)


def _d_phi_a(x: float) -> float:
    """
    Return the derivative of sin(x) evaluated at x.

    Parameter
    ---------
    x : float, the input value

    Returns
    -------
    float
        the derivative of -sin(x) / x evaluated at x.
    """
    return math.cos(x)


def _phi_b(x: float) -> float:
    """
    Return the value of -sin(x) / x (return 1 if x is 0).

    Parameter
    ---------
    x : float, the input value

    Returns
    -------
    float
        the value of -sin(x) / x (return 1 if x is 0).
    """
    if (x == 0):
        return -1
    return -(math.sin(x)) / x


def _d_phi_b(x: float) -> float:
    """
    Return the derivative of -sin(x) / x evaluated at x (return 0 if x is 0).

    Parameter
    ---------
    x : float, the input value

    Returns
    -------
    float
        the derivative of -sin(x) / x evaluated at x (return 0 if x is 0).
    """
    if x == 0:
        return 0
    
    return -(math.cos(x) * x - math.sin(x)) / (x ** 2)


def find_min(df: Callable[[float], float], 
             guess_0: float, 
             guess_1: float, 
             *, 
             tolerance: float = 10 ** (-6), 
             f_tolerance: float = 10 ** (-6), 
             lower: float = -float("inf"), 
             upper: float = float("inf")) -> (float | None):
    """
    Find the local minimum of `f(x)` in a close interval 
    using Secant method with given inital guesses.

    Parameters
    ----------
    df : Callable[[float], float], the first derivative of function `f(x)`
    guess_0 : float, the first inital guess of the solution
    guess_1 : float, the second initial guess of the solution
    tolerance : float, the tolerance of `|x_{k + 1} - x_k|`, default 10 ** (-6)
    f_tolerance : float, the tolerance of `|f'(x_{k + 1})|`, 
    default 10 ** (-6)
    lower : float, the lower bound of the interval, default ``-inf``
    upper : float, the upper bound of the interval, default ``inf``

    Returns
    -------
    float
        the local minimum of `f(x)` using Secant method with 
        given inital guesses.
    
    Raises
    ------
    ValueError : if input parameters invalid.
    RuntimeWarning : if whether the result is a minimum cannot be verified.
    """
    # Check input validity.
    if lower >= upper:
        raise ValueError(f"Lower bound {lower} must be "
                         + f"less than upper bound ({upper})")
    if guess_0 < lower or guess_0 > upper:
        raise ValueError(f"{guess_0} not in [{lower}, {upper}]")
    if guess_1 < lower or guess_1 > upper:
        raise ValueError(f"{guess_1} not in [{lower}, {upper}]")
    
    # Setup initial guesses
    x_0: float = guess_0
    x_1: float = guess_1

    while abs(x_1 - x_0) >= tolerance:
        # Check if denominator is 0 but `x_1 != x_0`.
        diff_f: float = df(x_1) - df(x_0)
        if diff_f == 0:
            raise ValueError(f"Denominator df({x_1}) - df({x_0}) = 0")
        
        # Perform an iteration
        x_2: float = x_1 - df(x_1) * (x_1 - x_0) / (df(x_1) - df(x_0))
        
        # Check if `x_2` in domain
        if x_2 < lower or x_2 > upper:
            return None

        x_0 = x_1
        x_1 = x_2
    
    # When `x_1 == x_0`, return value only if `x_1` is root.
    if x_1 == x_0:
        if df(x_1) != 0:
            return None
        
        warnings.warn(f"Approximated root {x_1} may "
                      + "not be a minimum", RuntimeWarning)
        return x_1
    
    # Determine if `x_1` is minimum of `f(x)` using secant as slope.
    if abs(df(x_1)) < f_tolerance and (df(x_1) - df(x_0)) / (x_1 - x_0) >= 0:
        return x_1
    
    return None


def _print_output(f: Callable[[float], float], 
                  df: Callable[[float], float], 
                  guesses : list[(float, float)], 
                  *, 
                  lower : float = -float("inf"), 
                  upper : float = -float("inf"), 
                  tolerance : float = 10 ** (-6), 
                  f_tolerance : float = 10 ** (-6)) -> None:
    """Find and print local and global minimum found by inital guesses.
    
    Parameters
    ----------
    f : Callable[[float], float], the original function `f(x)`
    df : Callable[[float], float], the first derivative of function `f(x)`
    guesses : list[(float, float)], a list of pair of inital guesses
    tolerance : float, the tolerance of `|x_{k + 1} - x_k|`, default 10 ** (-6)
    f_tolerance : float, the tolerance of `|f'(x_{k + 1})|`, 
    default 10 ** (-6)
    lower : float, the lower bound of the interval, default ``-inf``
    upper : float, the upper bound of the interval, default ``inf``
    """
    # Setup absolute minimum.
    abs_mins: set[float] = set()
    min_val: float = float("inf")
    
    local_mins: set[float] = set()

    # Find minimize for each pair ofs inital guesses.
    for guess in guesses:
        root = find_min(df, guess[0], guess[1], 
                        tolerance=tolerance, f_tolerance=f_tolerance, 
                        lower=lower, upper=upper)

        if type(root) != float:
            # No valid root found.
            continue
        local_mins.add(root)

        # Update absolute minimum.
        root_val = f(root)
        if root_val > min_val:
            continue
        if root_val < min_val:
            min_val = root_val
            abs_mins.clear()
        abs_mins.add(root)
            
            
    
    if len(local_mins) > 0:
        print(f"Local minimum: {local_mins}; global minimum: {abs_mins}.")
    else:
        print("No minimum found.")
    

def main():
    # Setup parameters.
    tolerance = 10 ** (-8)
    lower = -10
    upper = 10

    # Make inital guesses for part (a) and (b).
    a_guesses = [(-9, -8.5), 
                 (-3, -2.5), 
                 (3.5, 4)]

    b_guesses = [(-7.5, -7), 
                 (-1.5, -1), 
                 (7, 7.5)]

    # Minimize part (a).
    print("(a)", end=": ")
    _print_output(_phi_a, _d_phi_a, a_guesses, lower=lower, upper=upper, 
                  tolerance=tolerance, f_tolerance=tolerance)
    # Minimize part (b).
    print("(b)", end=": ")
    _print_output(_phi_b, _d_phi_b, b_guesses, lower=lower, upper=upper, 
                  tolerance=tolerance, f_tolerance=tolerance)


if __name__ == "__main__":
    main()