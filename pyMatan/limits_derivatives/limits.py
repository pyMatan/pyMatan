import sympy
from typing import Union
from ..utils import parse_function
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable

def numerical_limit(func, x0:float, side: str = "both", h:float = 1e-6):
    """
        Calculates the numerical limit of the function f(x) as x approaches x0.

        This function approximates the limit by evaluating the function at points
        (x0 - h) and/or (x0 + h).

        Args:
            func: The function for which the limit is being sought (callable).
            x0: The point that x approaches.
            side: The side of approach ('left', 'right', or 'both').
                  Defaults to 'both'.
            h: The small step size used for approximation (h > 0).
               Defaults to 1e-6.

        Returns:
            float or None: The numerical value of the limit.
                           Returns None if side is 'both' and the left and
                           right limits do not agree within a tolerance.

        Raises:
            ValueError: If h is non-positive (h <= 0) or if the 'side'
                        parameter is invalid.
    """

    if h <= 0:
        raise ValueError("h should be >= 0")

    if side == "both":
        left_limit = func(x0 - h)
        right_limit = func(x0 + h)

        if abs(left_limit - right_limit) < h * 100:
            return (left_limit + right_limit) / 2
        else:
            return None

    elif side == "left":
        return func(x0 - h)

    elif side == "right":
        return func(x0 + h)

    else:
        raise ValueError("Side should be either 'left' or 'right'")






def symbolic_limit(
        func_str: str,
        x0: Union[int, float, str, sympy.Symbol],
        side: str = "both"
) -> Union[sympy.Expr, float]:
    """
    Calculates the symbolic (exact) limit of the function f(x) as x approaches x0
    using SymPy.

    This function relies on parse_function to convert the input string into
    a SymPy symbolic expression.

    Args:
        func_str (str): The function for which the limit is being sought (e.g., "sin(x)/x").
        x0 (Union[int, float, str, sympy.Symbol]): The point that x approaches.
                                                   Can be a number or a string like "oo" (infinity).
        side (str): The side of approach: 'both' (default), 'left' ('-'), or 'right' ('+').

    Returns:
        Union[sympy.Expr, float]: The exact value of the limit. Can be a number,
                                  a symbol (like oo for infinity), or a symbolic expression.

    Raises:
        ValueError: If parsing fails.
    """


    symbolic_f, _ = parse_function(func_str)

    x = symbolic_f.free_symbols.pop() if symbolic_f.free_symbols else sympy.symbols('x')

    direction = '+'
    if side == 'left':
        direction = '-'
    elif side == 'right':
        direction = '+'
    elif side == 'both':
        pass
    else:
        raise ValueError("Side must be 'left', 'right', or 'both'.")

    if side == 'both':
        limit_left = sympy.limit(symbolic_f, x, x0, dir='-')
        limit_right = sympy.limit(symbolic_f, x, x0, dir='+')

        if sympy.Eq(limit_left, limit_right):
            return limit_left
        else:
            return sympy.nan
    else:
        return sympy.limit(symbolic_f, x, x0, dir=direction)


def visualize_limit(
        func_str: str,
        x0: float,
        h_window: float = 0.5,
        n_points: int = 100,
        h_approx: float = 1e-6
):
    """
        Visualizes the function around the point x0 and displays the specific points
        used for the numerical approximation of the limit.

        Args:
            func_str (str): The function string (e.g., "sin(x)/x").
            x0 (float): The point the limit approaches.
            h_window (float): The radius of the display window around x0 (controls the x-axis range).
                              Defaults to 0.5.
            n_points (int): The number of points used to plot a smooth curve.
                            Defaults to 100.
            h_approx (float): The small step size (h) used to calculate the numerical
                              approximation points (x0 - h and x0 + h).
                              Defaults to 1e-6.
        """
    try:
        _, numpy_func = parse_function(func_str)
    except ValueError as e:
        print(f"Function parsing error: {e}")
        return

    x_min = x0 - h_window
    x_max = x0 + h_window

    x_smooth = np.linspace(x_min, x_max, n_points)

    y_smooth = numpy_func(x_smooth)

    y_smooth[np.abs(y_smooth) > 100] = np.nan

    x_approx = [x0 - h_approx, x0 + h_approx]
    y_approx = [numpy_func(x0 - h_approx), numpy_func(x0 + h_approx)]

    plt.figure(figsize=(10, 6))

    plt.plot(x_smooth, y_smooth, label=f'$f(x) = {func_str}$', color='blue', linewidth=2)

    plt.scatter(x_approx, y_approx, color='red', s=100, zorder=5,
                label=f'Numerical Approximation $x_0 \pm h$ (h={h_approx:.1e})')

    plt.axvline(x=x0, color='gray', linestyle='--', linewidth=1, label=f'Limit Point $x_0 = {x0}$')

    plt.title(f'Visualization of the Function Limit $f(x)$ as $x \\to {x0}$')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.show()




