from typing import Callable
import sympy
from typing import Union
from ..utils  import parse_function
import numpy as np
import matplotlib.pyplot as plt

def central_difference(func, x0: float, h: float = 1e-6, order: int = 1):
    """
        Calculates the numerical derivative of function f(x) at point x0
        using the central difference formula (recursively for any order n).

        This method is of second-order accuracy (O(h^2)).

        Args:
            func: The function to be differentiated (callable).
            x0: The point at which to calculate the derivative.
            h: The step size (must be small and positive). Defaults to 1e-6.
            order: The order of the derivative (integer >= 1). Defaults to 1.

        Returns:
            float: The numerical value of the n-th order derivative.

        Raises:
            ValueError: If h is non-positive (h <= 0) or order is less than 1.
        """
    if h <= 0:
        raise ValueError("h must be > 0")
    if order < 1:
        raise ValueError("order must be >= 1")


    if order == 1:
        return (func(x0 + h) - func(x0 - h)) / (2 * h)

    else:
        def derivative_of_prev_order(x):
            return central_difference(func, x, h, order=order - 1)

        return central_difference(derivative_of_prev_order, x0, h, order=1)




def forward_difference(
    func: Callable,
    x0: float,
    h: float = 1e-6
) -> float:
    """
    Calculates the numerical derivative of function f(x) at point x0
    using the forward difference formula.

    This method is of first-order accuracy (O(h)). It is generally less
    accurate than the central difference method.

    Args:
        func (Callable): The function to be differentiated.
        x0 (float): The point at which to calculate the derivative.
        h (float): The step size (must be small and positive). Defaults to 1e-6.

    Returns:
        float: The numerical value of the first-order derivative.

    Raises:
        ValueError: If h is non-positive (h <= 0).
    """
    if h <= 0:
        raise ValueError("Step size 'h' must be greater than zero.")

    return (func(x0 + h) - func(x0)) / h


def symbolic_derivative(func_str: str, order: int = 1) -> sympy.Expr:
    """
    Calculates the symbolic (exact) derivative of the function f(x) using SymPy.

    This function finds the n-th order derivative analytically.

    Args:
        func_str (str): The function string (e.g., "x**3 + sin(x)").
        order (int): The order of the derivative (n-th order). Defaults to 1.

    Returns:
        sympy.Expr: The symbolic expression for the n-th order derivative.

    Raises:
        ValueError: If parsing fails or order is less than 1.
    """
    if order < 1:
        raise ValueError("Derivative order must be 1 or greater.")


    symbolic_f, _ = parse_function(func_str)

    x = symbolic_f.free_symbols.pop() if symbolic_f.free_symbols else sympy.symbols('x')

    derivative_expr = sympy.diff(symbolic_f, x, order)

    return derivative_expr


def visualize_derivative(
        func_str: str,
        x0: float,
        h_step: float = 0.5,
        h_approx: float = 0.1
):
    """
    Visualizes the function, its exact tangent line, and the secant line (Central Difference)
    to demonstrate the derivative approximation.

    Args:
        func_str (str): The function string (e.g., "x**3").
        x0 (float): The point at which the derivative is calculated.
        h_step (float): The radius of the display window around x0.
        h_approx (float): The step size (h) used for the numerical approximation (secant line).
    """


    try:
        symbolic_f, numpy_func = parse_function(func_str)
    except Exception as e:
        print(f"Function parsing error: {e}")
        return

    symbolic_deriv = symbolic_derivative(func_str, order=1)

    x_sym = sympy.Symbol('x')
    f_prime_x0 = symbolic_deriv.subs(x_sym, x0).evalf()
    f_x0 = numpy_func(x0)


    # Central Difference approximation
    f_prime_numerical = central_difference(numpy_func, x0, h=h_approx, order=1)


    x_range = np.linspace(x0 - h_step, x0 + h_step, 200)
    y_range = numpy_func(x_range)



    y_tangent = f_x0 + f_prime_x0 * (x_range - x0)

    y_secant = f_x0 + f_prime_numerical * (x_range - x0)

    plt.figure(figsize=(12, 7))

    plt.plot(x_range, y_range, label=f'$f(x) = {func_str}$', color='blue', linewidth=3)

    plt.plot(x_range, y_tangent, label=f'Exact Tangent ($f\'({x0}) = {f_prime_x0:.3f}$)',
             color='green', linestyle='-', linewidth=1.5)

    plt.plot(x_range, y_secant, label=f'Secant Line (CD, $h={h_approx}$)',
             color='red', linestyle='--', linewidth=1.5)

    plt.scatter([x0 - h_approx, x0 + h_approx],
                [numpy_func(x0 - h_approx), numpy_func(x0 + h_approx)],
                color='red', s=50, zorder=5)

    plt.scatter([x0], [f_x0], color='black', s=80, zorder=6, label=f'Point $x_0 = {x0}$')

    plt.title(f'Visualization of the Derivative Approximation at $x = {x0}$')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.show()


