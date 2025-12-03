from typing import Callable, Union, Iterable
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

from ..utils import parse_function
from ..limits_derivatives.derivatives import symbolic_derivative
from ..limits_derivatives.limits import symbolic_limit


def d_alembert_test(a_n: Callable[[int], float], n_start: int = 50, steps: int = 200) -> float:
    """
        Applies the D'Alembert ratio test to estimate the limit:

            L = lim |a_{n+1} / a_n| as n → ∞.

        The function computes this numerically by sampling ratio values
        for large n and averaging the last few results.

        Args:
            a_n: Function giving the n-th term of the series.
            n_start: Starting index from which we begin sampling.
            steps: How many ratios to compute.

        Returns:
            float: Approximate ratio limit (L).

        Raises:
            ValueError: If n_start < 1 or steps < 1.
    """

    if n_start < 1 or steps < 1:
        raise ValueError("n_start and steps must be >= 1.")

    ratios = []
    for n in range(n_start, n_start + steps):
        an = a_n(n)
        an1 = a_n(n + 1)
        if an == 0:
            continue
        ratios.append(abs(an1 / an))

    if len(ratios) == 0:
        return float("nan")

    return float(np.mean(ratios[-20:]))

def cauchy_root_test(a_n: Callable[[int], float], n_start: int = 50, steps: int = 200) -> float:
    """
        Applies the Cauchy root test to estimate:

            L = lim ( |a_n|^(1/n) ) as n → ∞.

        Works numerically by sampling large-n values of the root expression.

        Args:
            a_n: Function giving the n-th term of the series.
            n_start: First n from which sampling begins.
            steps: How many samples to calculate.

        Returns:
            float: Approximate value of L.

        Raises:
            ValueError: If parameters are less than 1.
    """

    if n_start < 1 or steps < 1:
        raise ValueError("n_start and steps must be >= 1.")

    values = []
    for n in range(n_start, n_start + steps):
        an = abs(a_n(n))
        if an <= 0:
            continue
        root_val = an ** (1.0 / n)
        values.append(root_val)

    if not values:
        return float("nan")

    return float(np.mean(values[-20:]))


def integral_test(func_str: str, a: int = 1, upper: float = 2000.0, n: int = 50000) -> float:
    """
        Applies the integral test by numerically approximating:

            ∫ from a to large_value of f(x) dx

        using a simple Riemann/trapezoid computation. This is not meant to
        replace student's integration module but works fine as a quick check.

        Args:
            func_str: The function as a string, like "1/x**2".
            a: Lower bound (must be >= 1).
            upper: Upper limit used as a stand-in for +∞.
            n: Number of subdivisions.

        Returns:
            float: Approximate value of the integral used for convergence check.

        Raises:
            ValueError: For invalid arguments.
    """

    if a < 1:
        raise ValueError("Lower bound a must be >= 1.")
    if upper <= a:
        raise ValueError("Upper must be > a.")
    if n < 10:
        raise ValueError("n should be reasonably large (>= 10).")

    _, f = parse_function(func_str)
    xs = np.linspace(a, upper, n)
    ys = f(xs)

    return float(np.trapz(ys, xs))


def leibniz_test(a_n: Callable[[int], float], check_terms: int = 50) -> bool:
    """
        Applies the Leibniz test for alternating series:

            (-1)^n * a_n

        Requirements:
          * a_n → 0
          * a_n is eventually decreasing

        Args:
            a_n: Function returning the n-th positive term.
            check_terms: How many ending terms to inspect.

        Returns:
            bool: True if the test conditions seem satisfied.
    """

    values = [abs(a_n(n)) for n in range(1, check_terms + 1)]


    if values[-1] > values[0]:
        return False


    for i in range(1, len(values)):
        if values[i] > values[i - 1]:
            return False

    return True


def partial_sum(a_n: Callable[[int], float], N: int) -> float:
    """
        Computes the partial sum:

            S_N = a_1 + a_2 + ... + a_N.

        Very straightforward helper when testing numeric convergence.

        Args:
            a_n: Function giving the n-th term of the series.
            N: Number of terms to sum (must be >= 1).

        Returns:
            float: The partial sum value.

        Raises:
            TypeError: If a_n is not callable.
            ValueError: If N < 1.
    """

    if not callable(a_n):
        raise TypeError("a_n must be callable.")

    if N < 1:
        raise ValueError("N must be >= 1.")

    total = 0.0
    for n in range(1, N + 1):
        total += a_n(n)

    return total



def taylor_polynomial(func_str: str, x0: float = 0.0, degree: int = 5) -> sp.Expr:
    """
        Generates the Taylor polynomial of a given degree for f(x) at x = x0.

        Uses symbolic derivatives from SymPy for accuracy.

        Args:
            func_str: Function as a string (like "sin(x)").
            x0: Expansion point.
            degree: Polynomial degree (>= 0).

        Returns:
            sympy.Expr: The Taylor polynomial expression.

        Raises:
            ValueError: If degree < 0.
    """

    if degree < 0:
        raise ValueError("degree must be >= 0.")

    symbolic_f, _ = parse_function(func_str)
    x = list(symbolic_f.free_symbols)[0]

    poly = 0
    for k in range(degree + 1):
        deriv = sp.diff(symbolic_f, x, k)
        term = deriv.subs(x, x0) / sp.factorial(k) * (x - x0) ** k
        poly += term

    return sp.simplify(poly)



def taylor_remainder_lagrange(func_str: str, x: float, x0: float, degree: int) -> sp.Expr:
    """
        Computes the Lagrange-form remainder:

            R_n(x) = f^{(n+1)}(ξ) / (n+1)! * (x - x0)^{n+1}

        Since ξ is unknown, this function returns a symbolic expression
        with f^{(n+1)}(x) substituted instead, which still gives an idea
        of error behavior.

        Args:
            func_str: Original function as a string.
            x: Point where remainder is evaluated.
            x0: Expansion point.
            degree: Degree n of the polynomial.

        Returns:
            sympy.Expr: The symbolic remainder estimate.
    """

    if degree < 0:
        raise ValueError("degree must be >= 0.")

    symbolic_f, _ = parse_function(func_str)
    var = list(symbolic_f.free_symbols)[0]

    deriv = sp.diff(symbolic_f, var, degree + 1)
    remainder = deriv.subs(var, x) / sp.factorial(degree + 1) * (x - x0) ** (degree + 1)

    return sp.simplify(remainder)


def fourier_coefficients(func_str: str, L: float, n_terms: int = 10):
    """
        Computes Fourier coefficients a0, a_n, b_n for f(x) on [-L, L].

        Uses symbolic integration from SymPy.

        Args:
            func_str: Function as a string.
            L: Half-period of the interval.
            n_terms: Number of harmonic terms.

        Returns:
            tuple: (a0, [a_n], [b_n]) all symbolic expressions.

        Raises:
            ValueError: For invalid L or n_terms.
    """

    if L <= 0:
        raise ValueError("L must be > 0.")
    if n_terms < 1:
        raise ValueError("n_terms must be >= 1.")

    symbolic_f, _ = parse_function(func_str)
    x = list(symbolic_f.free_symbols)[0]

    a0 = (1 / L) * sp.integrate(symbolic_f, (x, -L, L))

    a_list = []
    b_list = []

    for n in range(1, n_terms + 1):
        a_n = (1 / L) * sp.integrate(symbolic_f * sp.cos(n * sp.pi * x / L), (x, -L, L))
        b_n = (1 / L) * sp.integrate(symbolic_f * sp.sin(n * sp.pi * x / L), (x, -L, L))
        a_list.append(sp.simplify(a_n))
        b_list.append(sp.simplify(b_n))

    return a0, a_list, b_list


def fourier_partial_sum(func_str: str, L: float, n_terms: int = 10) -> sp.Expr:
    """
        Builds the symbolic partial Fourier sum:

            S_n(x) = a0/2 + Σ [ a_n cos(nxπ/L) + b_n sin(nxπ/L) ].

        Args:
            func_str: Function string.
            L: Half-period.
            n_terms: Number of terms.

        Returns:
            sympy.Expr: The symbolic expression for S_n(x).
    """

    a0, a_list, b_list = fourier_coefficients(func_str, L, n_terms)
    symbolic_f, _ = parse_function(func_str)
    x = list(symbolic_f.free_symbols)[0]

    S = a0 / 2

    for n in range(1, n_terms + 1):
        S += a_list[n - 1] * sp.cos(n * sp.pi * x / L)
        S += b_list[n - 1] * sp.sin(n * sp.pi * x / L)

    return sp.simplify(S)



def plot_taylor_approximations(func_str: str, x0: float = 0.0,degrees=(1, 2, 3, 5), x_min=-2, x_max=2):
    """
        Plots the original function and several Taylor polynomials of
        increasing degree so you can visually compare how good they are.

        Args:
            func_str: Function as a string.
            x0: Expansion point.
            degrees: Iterable of polynomial degrees to plot.
            x_min: Minimum x-value on graph.
            x_max: Maximum x-value.

        Returns:
            None: Shows a Matplotlib plot.

        Raises:
            ValueError: If x_min >= x_max.
    """

    if x_min >= x_max:
        raise ValueError("x_min must be < x_max.")

    symbolic_f, num_f = parse_function(func_str)
    x = np.linspace(x_min, x_max, 400)

    plt.figure(figsize=(10, 6))
    plt.plot(x, num_f(x), label=f"f(x) = {func_str}", linewidth=2)

    for d in degrees:
        poly = taylor_polynomial(func_str, x0, d)
        poly_func = sp.lambdify(list(symbolic_f.free_symbols)[0], poly, "numpy")
        plt.plot(x, poly_func(x), label=f"Degree {d}")

    plt.title("Taylor Approximations")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    plt.show()
