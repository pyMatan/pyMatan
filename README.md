# pyMatan
# pyMatan: Numerical and Symbolic Calculus Library


[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue)]()

## Developers
Serhii Marynokha,
Konstantin Bachynskyi,
Maksym Pugach
## License

This project is licensed under the **MIT License** - see the `LICENSE` file for details.

## Project Overview

`pyMatan` (Python Mathematical Analysis) is a versatile Python library designed to simplify the process of solving fundamental problems in mathematical analysis, including **limits, derivatives, integrals, and series**.

By strategically combining the high-performance numerical capabilities of **NumPy/SciPy** with the powerful symbolic manipulation of **SymPy**, `pyMatan` provides a robust, easy-to-use, and highly accurate toolset for students, engineers, and researchers.

This project was developed as a Computational Work (Розрахнукова Робота) for **[Your Course Name, e.g., Numerical Methods and Computational Analysis]** at **[Your University/Department]**.

## Key Features

The library is structured into three main modules, covering both **Numerical** (Num) and **Symbolic** (Sym) analysis:

| Module | Core Functionality | Numerical Tools (NumPy/SciPy) | Symbolic Tools (SymPy) |
| :--- | :--- | :--- | :--- |
| **`limits_derivatives`** | Calculation and Visualization of Limits and Derivatives. | Finite Difference Methods (Forward, Central). | SymPy's `limit`, `diff`. |
| **`integrals`** | Numerical Quadrature, Adaptive Integration, and Symbolic Integration. | Trapezoidal, Simpson's, Midpoint, Gauss-Legendre, Adaptive Integration, Runge Error Estimation. | SymPy's `integrate`, Handling Improper Integrals. |
| **`series`** | Convergence Tests, Taylor/Maclaurin Series, Fourier Series. | Calculation of Partial Sums, Visualization of Approximations. | D'Alembert, Cauchy, Integral Test, Alternating Series Test. |

## Installation

### Prerequisites

* Python 3.8+
* Required packages: `numpy`, `scipy`, `sympy`, `matplotlib`.

### From Source

For this project phase, clone the repository and install the package in "editable" mode:

```bash
git clone "https://github.com/pyMatan"
cd pyMatan_project
pip install -e .
```
The `-e` flag allows you to run the package functions directly while still being able to edit the source code.

## Usage Examples

1. Calculating Integrals (Kostiantyn)
The IntegralSolver class (in integrals.py) provides a unified interface for both numerical and symbolic integration, error estimation, and convergence plotting.Example: Gauss-Legendre Quadrature and Symbolic VerificationGauss-Legendre is a high-precision method capable of integrating polynomials of degree $2n-1$ exactly, using only $n$ points. We verify the numerical result using SymPy's symbolic solver.
```python
from pyMatan.integrals.integrals import IntegralSolver
import numpy as np

# Function: f(x) = 5*x^4 + 3*x^2 + 1. Integral from 0 to 1 should be 3.
solver = IntegralSolver(function_str='5*x**4 + 3*x**2 + 1')

a = 0  # Lower limit
b = 1  # Upper limit
n = 3  # Number of points (exact for polynomials up to degree 5)

# 1. Calculate the numerical integral
result_gauss = solver.gauss_legendre(a, b, n)

# 2. Get the exact symbolic result for comparison
exact_sym, exact_num = solver.symbolic_solve(a=a, b=b)

print(f"Function: {solver.function_str}")
print(f"Limits: [{a}, {b}]")
print(f"Gauss-Legendre Result (n={n}): {result_gauss}")
print(f"Exact Value (Symbolic): {exact_num}")
# Expected Exact Value: 3.0
```

2. Calculating a Numerical Limit (Serhii)
Numerically estimate a limit by approaching the point from the right.
This function is vital for verifying symbolic limits or handling functions where symbolic manipulation is complex.Example: The First Remarkable LimitCalculating the limit x->0 sin(x)/x.
```python
from pyMatan.limits_derivatives.limits import numerical_limit_right
import numpy as np

# Example function: f(x) = (sin(x) / x)
def f(x):
    # This check prevents division by zero at x=0
    return np.sin(x) / x

x0 = 0
tolerance = 1e-6

limit_value = numerical_limit_right(f, x0, tolerance)

print(f"Function: f(x) = sin(x)/x")
print(f"Numerical limit of f(x) as x -> 0+ is: {limit_value}")
# Expected: close to 1.0
```

Project Structure
The project adheres to the standard Python package structure, ensuring clean separation of concerns and clear responsibilities for each module.

pyMatan_project/ <br>
├── pyMatan/  <br>
│   ├── integrals/ <br>           
│   │   ├── __init__.py <br>
│   │   └── integrals.py <br>
│   ├── series/ <br>
│   │   ├── __init__.py <br>
│   │   └── series.py <br>
│   ├── limits_derivatives/ <br>
│   │   ├── __init__.py <br>
│   │   ├── limits.py <br>
│   │   └── derivatives.py <br>
│   
├── setup.py <br>              
├── testing <br>
|   ├──notebook_example.ipynb <br>
|   ├──test_pyMatan.py <br>
├── README.md <br>
└── LICENSE<br>

## Contributors

This project was a collaborative effort by the following students:

| Student | Module | Key Functions |
| :--- | :--- | :--- |
| **Kostiantyn** | `integrals` | Trapezoidal, Simpson, Gauss, Adaptive Integration, Runge Error. |
| **Maksym** | `series` | D'Alembet, Cauchy, Integral Test, Taylor Polynomials, Fourier Series. |
| **Serhii** | `limits_derivatives` | Numerical & Symbolic Limits/Derivatives, Project Setup, Testing. |
