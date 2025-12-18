# pyMatan.integration — Integral Solver Module

This module provides a comprehensive suite of tools for calculating integrals. It combines classic numerical methods, adaptive algorithms with error control, and symbolic computation capabilities powered by the SymPy library.

---

# Table of Contents
- [1. midpoint](#1-midpoint)
- [2. trapezoidal](#2-trapezoidal)
- [3. simpson](#3-simpson)
- [4. gauss_legendre](#4-gauss_legendre)
- [5. adaptive](#5-adaptive)
- [6. improper_integral](#6-improper_integral)
- [7. symbolic_solve](#7-symbolic_solve)
- [8. plot_convergence](#8-plot_convergence)

---

# 1. midpoint

### midpoint(a, b, n)
Calculates a definite integral using the Midpoint Rule. The function is evaluated at the center of each sub-interval. This method provides an accuracy of order $O(h^2)$.



### Parameters
| Parameter | Type | Description |
|:---|:---|:---|
| a | float | Lower limit of integration. |
| b | float | Upper limit of integration. |
| n | int | Number of intervals (subdivisions). |

---

# 2. trapezoidal

### trapezoidal(a, b, n)
Implements the Trapezoidal Rule, where the curve is approximated by linear segments. The area under the curve is calculated as the sum of the areas of trapezoids formed by the function values at the endpoints of the intervals.



---

# 3. simpson

### simpson(a, b, n)
Calculates the integral using Simpson's Rule (parabolic method). Instead of straight lines, the function is approximated by quadratic polynomials over each pair of intervals, yielding an accuracy of order $O(h^4)$.



> Note: The number of intervals n must be even.

---

# 4. gauss_legendre

### gauss_legendre(a, b, n)
Uses the Gauss-Legendre Quadrature formula. This method strategically selects optimal integration nodes (roots of Legendre polynomials) and weights, allowing for extremely high precision with a minimal number of function evaluations.

---

# 5. adaptive

### adaptive(method_name, a, b, tol=1e-6)
Performs integration with automatic step size selection. The algorithm doubles the number of intervals n until the estimated error (calculated using **Runge's Rule**) falls below the specified tolerance tol.

### Returns
| Type | Description |
|:---|:---|
| tuple | (Approximate_integral, Final_number_of_intervals, Estimated_error) |

---

# 6. improper_integral

### improper_integral(a, b)
Designed to calculate improper integrals (e.g., when integration limits are $\infty$ or $-\infty$, or when the function has singularities). It utilizes the robust scipy.integrate.quad engine.

### Example
```python
solver = IntegralSolver("exp(-x**2)")
val, err = solver.improper_integral(-np.inf, np.inf)
# Result will be close to sqrt(pi)
```
# 7. symbolic_solve

### symbolic_solve(a=None, b=None)
Calculates the analytical integral using SymPy. Unlike numerical methods, this function finds the exact mathematical expression. It can compute both indefinite integrals (antiderivatives) and definite integrals.

---

### Parameters
| Parameter | Type | Description |
|:---|:---|:---|
| a | float (optional) | Lower limit of integration. |
| b | float (optional) | Upper limit of integration. |

---

### Returns
| Context | Type | Description |
|:---|:---|:---|
| Without limits | sympy.Expr | The indefinite integral (e.g., $x^2 \rightarrow \frac{x^3}{3}$). |
| With limits | tuple | A pair: (Symbolic_Value, Numeric_Evaluation). |

---

### Example
```python
# Indefinite integral
antiderivative = solver.symbolic_solve()
print(f"Antiderivative: {antiderivative}")

# Definite integral
exact_expr, numeric_val = solver.symbolic_solve(a=0, b=1)
print(f"Exact value: {exact_expr} (≈ {numeric_val})")
```
# 8. plot_convergence

### plot_convergence(a, b, exact_val, method_names=['trapezoidal', 'simpson'], max_n=50)
Generates a diagnostic plot to visualize the error reduction of different numerical methods as the number of intervals ($n$) increases. This tool is essential for analyzing the efficiency and "speed" of a method (its order of accuracy). It helps to visually confirm why higher-order methods like Simpson's rule converge much faster than the Trapezoidal rule.



---

### Parameters

| Parameter | Type | Description |
|:---|:---|:---|
| a | float | Lower limit of the interval. |
| b | float | Upper limit of the interval. |
| exact_val | float | The true value of the integral (used as a reference to calculate the absolute error). |
| method_names | list | List of method names to compare. Default: ['trapezoidal', 'simpson']. |
| max_n | int | The maximum number of intervals to test. Default: 50. |

---

### Key Features

* Logarithmic Scaling: The y-axis uses a log scale, which is the standard way to visualize convergence rates ($O(h^p)$).
* Automatic Error Calculation: Calculates $|Value_{approx} - Value_{exact}|$ for every step of $n$.
* Multi-Method Support: Allows plotting multiple methods on a single chart for direct performance benchmarking.
* Robustness: Includes error handling (try/except) for methods that might fail at specific $n$ values (e.g., Simpson's rule requiring even $n$).

---

### Example

```python
import numpy as np
from pyMatan.integration import IntegralSolver

# Initialize with a test function
solver = IntegralSolver("exp(x)")

# Get the exact value for comparison (e^1 - e^0)
_, exact = solver.symbolic_solve(0, 1)

# Visualize the convergence of three different methods
solver.plot_convergence(
    a=0, 
    b=1, 
    exact_val=float(exact), 
    method_names=['midpoint', 'trapezoidal', 'simpson'], 
    max_n=60
)
```
