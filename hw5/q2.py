# -*- coding: utf-8 -*-

"""
q2.py - Find and plot Hermite interpolation of specific functions.
"""

import abc
import numpy as np
import matplotlib.pyplot as plt


class FunctionGroup(abc.ABC): # define abstract function class
    """
    A class that represents a group of similar functions
    that can be called by index.
    """

    @abc.abstractmethod
    def eval(self,base_index : int, x : any) -> any:
        """
        Return the value of the function at specified index 
        for input `x`.
        
        Parameters
        ----------
        base_index : int, the index of the specified function 
            to be called.
        x : any, the input to the function.
        
        Returns
        -------
        any
            The value of the specified function for input `x`.
        """
        pass


class LagrangianGroup(FunctionGroup):
    """
    A class that represents a group of Lagrangian functions 
    `L_1(x), L_2(x), ..., L_n(x)` generated from input array
    `[x_1, x_2, ..., x_n]`.
    """

    def __init__(self, x : list[float]) -> None:
        """
        Create an LagrangianGroup using specified data.

        Parameters
        ----------
        x : list[float], the array containing `x_1, x_2, ..., x_n`, 
            note that `x` must have at least 2 elements.
        
        Raises
        ------
        ValueError : if `x` has less than 2 numbers.
        """
        # Check validity of parameters.
        if len(x) < 2:
            raise ValueError(f"`x` must have at least 2 elements"
                             + f"but {len(x)} elements received")
        
        self.__x_arr = x
        self.__bases = LagrangianGroup.__create_bases(x)

    @property
    def x_arr(self) -> list[float]:
        return self.__x_arr.copy()
    
    @property
    def bases(self) -> list[float]:
        return self.__bases.copy()

    @staticmethod
    def __create_bases(x_arr : list[float]) -> list[float]:
        """Create `bases` with given input."""
        # Initialize the container.
        bases: list[float] = []

        for curr_index in range(len(x_arr)):
            base: float = 1 # represents the denominator of Lagrangian

            for index in range(len(x_arr)):
                if (curr_index == index):
                    # Skip `curr_index`.
                    continue
                
                base *= x_arr[curr_index] - x_arr[index]
            
            bases.append(base)
        
        return bases

    def eval(self, base_index : int, x : float) -> float:
        if x == self.x_arr[base_index]:
            # L_k(x_k) = 1.
            return 1
        
        if x in self.x_arr:
            # L_k(x_i) = 0 for `i` not equal to `j``.
            return 0
        
        prod: float = 1 # initialize the numerator

        for index in range(len(self.x_arr)):
            if index == base_index:
                # Skip the base.
                continue

            prod *= x - self.x_arr[index]
        
        return prod / self.bases[base_index]
        

class Hermite():
    """A class that represents a Hermite Interpolation function."""
    SHIFT = 10 ** (-4) # a small shift used to approximate derivative

    def __init__(self, 
                 x : list[float], 
                 y : list[float], 
                 dydx : list[float]) -> None:
        """
        Create a Hermite interpolation function with given 
        data points and derivatives.
        
        Parameters
        ----------
        x : list[float], the array of x-coordinate of datapoints, 
            must have at least 2 elements.
        y : list[float], the array of y-coordinate 
            of corresponding datapoints, must have the same length 
            as `x`.
        dydx : list[float], the array of first derivatives of 
            corresponding datapoints, must have the same length 
            as `x`.
        
        Raises
        ------
        ValueError : if input parameters are invalid or differ in length.
        """

        # Check validity of parameters.
        if len(x) < 2:
            raise ValueError(f"`x` must have at least 2 elements"
                             + f"but {len(x)} elements received")
        x_len: int = len(x)
        if len(y) != x_len:
            raise ValueError(f"`y` must have the same length as x, "
                             + f"{x_len} elements expected,"
                             + f"but {len(y)} elements received")
        if len(dydx) != x_len:
            raise ValueError(f"`y` must have the same length as x, "
                             + f"{x_len} elements expected,"
                             + f"but {len(dydx)} elements received")
        
        self.__x_arr = x
        self.__y_arr = y
        self.__dydx_arr = dydx
        self.__lagrange = LagrangianGroup(x)

    def h(self, index : int, x : float) -> float:
        """
        Return the value of `H_k(x)`.
        
        Parameters
        ----------
        index : int, the index k of the function `H_k` to be called.
        x : float, the input value.
        
        Returns
        -------
        float
            The value of `H_k` evaluated at `x`.
        """
        lagrange: float = self.__lagrange.eval(index, x)
        # Calculate slope.
        slope: float = (self.__lagrange.eval(index, x - Hermite.SHIFT) \
                        + self.__lagrange.eval(index, x + Hermite.SHIFT)) \
                        / (2 * Hermite.SHIFT)

        return (lagrange ** 2) * (1 - 2 * slope * (x - self.__x_arr[index]))

    def k(self, index : int, x : float) -> float:
        """
        Return the value of `K_k(x)`.
        
        Parameters
        ----------
        index : int, the index k of the function `K_k` to be called.
        x : float, the input value.
        
        Returns
        -------
        float
            The value of `K_k` evaluated at `x`.
        """
        lagrange: float = self.__lagrange.eval(index, x)
        return (lagrange ** 2) * (x - self.__x_arr[index])
    
    def eval(self, x : float) -> float:
        """
        Return the value of Hermite interpolation function at `x`.
        
        Parameters
        ----------
        x : float, the input to the function.
        
        Returns
        -------
        float
            The value of Hermite interpolation function for input `x`.
        """
        res : float = 0

        # Add each term of the interpolation function.
        for index in range(len(self.__x_arr)):
            res += self.h(index, x) * self.__y_arr[index] \
                   + self.k(index, x) * self.__dydx_arr[index]
        
        return res
    
    def eval_list(self, 
                  x : np.ndarray, 
                  default : float = np.nan) -> np.ndarray:
        """
        Return the value of Hermite interpolation function 
            for each input `x`.
        
        Parameters
        ----------
        x : np.ndarray, a list of input to the function.
        default : placeholder value, used when the interpolation cannot be 
            evaluated at `x`, default is `np.nan`. 

        Returns
        -------
        np.ndarray
            The list values of Hermite interpolation function for 
            each input `x`.
        """
        p_x: list[float] = []

        for x_i in x:
            try:
                val = self.eval(x_i)
            except Exception as e:
                print(e)
                print(f"Warning : evaluation at {x_i} failed, "
                      + f"nan inserted!")
                val = default
            p_x.append(val)
        
        return np.array(p_x)


def tanh(x : np.ndarray) -> np.ndarray:
    """Return `tanh(x)` of input `x`.
    
    Parameter
    ---------
    x : float, the input `x` where the function will be evaluated.
    
    Returns
    -------
    float
        `tanh(x)` evaluated at `x`.
    """
    return np.tanh(x)

def dtanh_dx(x : np.ndarray) -> np.ndarray:
    """Return first derivative of `tanh(x)` at input `x`.
    
    Parameter
    ---------
    x : float, the input `x` where the derivative will be evaluated.
    
    Returns
    -------
    float
        derivative of `tanh(x)` (i.e. `(sech(x) ** 2)`) evaluated at `x`.
    """
    return 1. / np.square(np.cosh(x))

def main():
    # Define range
    START = 0
    STOP = 20
    LEN = STOP - START + 1

    # Initialize data.
    x = np.linspace(START, STOP, LEN)
    y = tanh(x)
    dydx = dtanh_dx(x)

    print(f"x : {x}")
    print(f"tanh(x) : {y}")
    print(f"dtanhx/dx : {dydx}")

    # Create Hermit interpolation.
    hermite = Hermite(x, y, dydx)
    approx_y = hermite.eval_list(x)

    print(f"p(x) : {approx_y}")

    # Error check.
    errs = y - approx_y
    print(f"Errors : {errs}")

    # Plot graph.
    points = np.linspace(START, STOP, 75)
    plt.plot(points, tanh(points), label=r"$\tanh(x)$")
    plt.plot(x, approx_y, "-o", markersize=4, label=r"$p_{41}(x)$")

    plt.title(r"Comparison of $\tanh(x)$ and $p_{41}(x)$ Found by "
              + r"Hermite Interpolation")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)

    plt.show()

if __name__ == "__main__":
    main()