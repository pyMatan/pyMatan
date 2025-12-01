import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from scipy.special import roots_legendre
from scipy import integrate

class IntegralSolver:
    def __init__(self, function_str: str):
        """
        Initializes the solver with a string representation of the function.

        Args:
            function_str (str): The function to integrate, e.g., 'x**2 + sin(x)'.

        Raises:
            ValueError: If the function string cannot be parsed.
        """
        self.function_str = function_str
        self.x = sp.symbols('x')
        try:
            self.expr = sp.sympify(function_str)
            self.f = sp.lambdify(self.x, self.expr, 'numpy')
        except Exception as e:
            raise ValueError(f"Parsing error: {e}")

    def midpoint(self, a, b, n):
        """
        Calculates the integral using the Midpoint rule.

        Args:
            a (float): Lower limit of integration.
            b (float): Upper limit of integration.
            n (int): Number of intervals.

        Returns:
            float: Approximate value of the integral.
        """
        h = (b - a) / n
        x = np.linspace(a + h/2, b - h/2, n)
        return h * np.sum(self.f(x))

    def trapezoidal(self, a, b, n):
        """
        Calculates the integral using the Trapezoidal rule.

        Args:
            a (float): Lower limit of integration.
            b (float): Upper limit of integration.
            n (int): Number of intervals.

        Returns:
            float: Approximate value of the integral.
        """
        h = (b - a) / n
        x = np.linspace(a, b, n + 1)
        y = self.f(x)
        return h * (y[0]/2 + np.sum(y[1:-1]) + y[-1]/2)

    def simpson(self, a, b, n):
        """
        Calculates the integral using Simpson's rule.

        Args:
            a (float): Lower limit of integration.
            b (float): Upper limit of integration.
            n (int): Number of intervals (must be even).

        Returns:
            float: Approximate value of the integral.

        Raises:
            ValueError: If n is not an even number.
        """
        if n % 2 != 0:
            raise ValueError("n must be an even number for Simpson's rule.")
        h = (b - a) / n
        x = np.linspace(a, b, n + 1)
        y = self.f(x)
        return h/3 * (y[0] + 4*np.sum(y[1:-1:2]) + 2*np.sum(y[2:-2:2]) + y[-1])

    def gauss_legendre(self, a, b, n):
        """
        Calculates the integral using Gauss-Legendre quadrature.

        Args:
            a (float): Lower limit of integration.
            b (float): Upper limit of integration.
            n (int): Number of sample points (degree of the polynomial).

        Returns:
            float: Approximate value of the integral.
        """
        roots, weights = roots_legendre(n)
        # Map roots from [-1, 1] to [a, b]
        shift = 0.5 * (b + a)
        scale = 0.5 * (b - a)
        x_mapped = scale * roots + shift
        return scale * np.sum(weights * self.f(x_mapped))

    def runge_estimate(self, method_func, a, b, n, p):
        """
        Estimates the integration error using Runge's rule.

        Args:
            method_func (callable): The integration method to use (e.g., self.trapezoidal).
            a (float): Lower limit.
            b (float): Upper limit.
            n (int): Number of intervals.
            p (int): Order of accuracy of the method (2 for Trapezoidal, 4 for Simpson).

        Returns:
            tuple: (Integral value at 2n, Estimated error).
        """
        I1 = method_func(a, b, n)
        I2 = method_func(a, b, n * 2)
        error = abs(I2 - I1) / (2**p - 1)
        return I2, error

    def adaptive(self, method_name, a, b, tol=1e-6):
        """
        Performs adaptive integration, doubling n until the error is below tolerance.

        Args:
            method_name (str): Name of the method ('trapezoidal', 'simpson', 'midpoint').
            a (float): Lower limit.
            b (float): Upper limit.
            tol (float): Desired error tolerance.

        Returns:
            tuple: (Approximate integral, Final number of intervals, Estimated error).
        """
        method_map = {
            'trapezoidal': (self.trapezoidal, 2),
            'midpoint': (self.midpoint, 2),
            'simpson': (self.simpson, 4)
        }
        
        if method_name not in method_map:
            raise ValueError(f"Unknown method '{method_name}' for adaptation.")
            
        func, p = method_map[method_name]
        n = 2 if method_name == 'simpson' else 1
        
        integral = 0.0
        error = 1.0 

        for _ in range(20):
            integral, error = self.runge_estimate(func, a, b, n, p)
            if error < tol:
                return integral, n * 2, error
            n *= 2
            
        print("Warning: Iteration limit reached without achieving desired tolerance.")
        return integral, n, error

    def improper_integral(self, a, b):
        """
        Calculates an improper integral using SciPy's quad function.
        Suitable for infinite limits (np.inf).

        Args:
            a (float): Lower limit (can be -np.inf).
            b (float): Upper limit (can be np.inf).

        Returns:
            tuple: (Integral value, Absolute error estimate from SciPy).
        """
        res, err = integrate.quad(lambda t: self.f(t), a, b)
        return res, err

    def symbolic_solve(self, a=None, b=None):
        """
        Computes the integral symbolically using SymPy.

        Args:
            a (float, optional): Lower limit. Defaults to None.
            b (float, optional): Upper limit. Defaults to None.

        Returns:
            SymPy expression or tuple:
                - If limits are None: Returns the indefinite integral (antiderivative).
                - If limits are provided: Returns (Exact symbolic value, Numeric evaluation).
        """
        if a is not None and b is not None:
            res = sp.integrate(self.expr, (self.x, a, b))
            return res, res.evalf()
        else:
            return sp.integrate(self.expr, self.x)

    def plot_convergence(self, a, b, exact_val, method_names=['trapezoidal', 'simpson'], max_n=50):
        """
        Plots the error convergence of different methods as n increases.

        Args:
            a (float): Lower limit.
            b (float): Upper limit.
            exact_val (float): The true value of the integral for error calculation.
            method_names (list): List of method names to plot. Defaults to ['trapezoidal', 'simpson'].
            max_n (int): Maximum number of intervals to test.

        Returns:
            None: Displays a matplotlib plot.
        """
        ns = range(2, max_n + 1, 2)
        plt.figure(figsize=(10, 6))

        for method_name in method_names:
            errors = []
            func = getattr(self, method_name)
            for n in ns:
                try:
                    val = func(a, b, n)
                    errors.append(abs(val - exact_val))
                except ValueError:
                    errors.append(None)
            
            plt.plot(ns, errors, label=method_name, marker='o')

        plt.yscale('log')
        plt.title(f'Convergence of methods for {self.function_str}')
        plt.xlabel('Number of intervals (n)')
        plt.ylabel('Absolute Error (log scale)')
        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.legend()
        plt.show()