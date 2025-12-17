# pyMatan.limits_derivatives — Derivatives and Limits Tools

This module provides tools for calculating derivatives and limits. It combines numerical methods (approximation via small increments) and symbolic computations (via SymPy) with integrated visualization using Matplotlib.

---

# Table of Contents
- [1. central_difference](#1-central_difference)
- [2. forward_difference](#2-forward_difference)
- [3. symbolic_derivative](#3-symbolic_derivative)
- [4. visualize_derivative](#4-visualize_derivative)
- [5. numerical_limit](#5-numerical_limit)
- [6. symbolic_limit](#6-symbolic_limit)
- [7. visualize_limit](#7-visualize_limit)

---

# 1. `central_difference`

### **central_difference(func, x0, h=1e-6, order=1)**
Calculates the numerical derivative of function $f(x)$ at point $x_0$ using the central difference formula.
This method has second-order accuracy $O(h^2)$ and is implemented recursively to support any derivative order.

---

### **Parameters**
| Parameter | Type | Description |
|----------|------|-------------|
| `func` | callable | The function to be differentiated. |
| `x0` | float | The point at which to calculate the derivative. |
| `h` | float | The step size (must be positive). Default is 1e-6. |
| `order` | int | The order of the derivative (integer >= 1). |

---

### **Returns**
| Type | Description |
|------|-------------|
| `float` | The numerical value of the n-th order derivative. |

---

### **Example**
```python
from pyMatan.derivatives import central_difference

def f(x): return x**2
central_difference(f, x0=2.0, order=1)
# → 4.0 (approx)
---

# 2. `forward_difference`

### **forward_difference(func, x0, h=1e-6)**
Calculates the numerical derivative of function $f(x)$ at point $x_0$ using the forward difference formula. This method approximates the slope of the tangent line by calculating the slope of the secant line through $x_0$ and $x_0 + h$.



---

### **Parameters**

| Parameter | Type | Description |
|:---|:---|:---|
| `func` | callable | The function to be differentiated (must accept a float). |
| `x0` | float | The point at which to calculate the derivative. |
| `h` | float | The step size (must be small and positive). Defaults to 1e-6. |

---

### **Returns**

| Type | Description |
|:---|:---|
| `float` | The numerical value of the first-order derivative. |

---

### **Example**

```python
from pyMatan.derivatives import forward_difference

# Define a function f(x) = x^2
def f(x): 
    return x**2

# Calculate derivative at x = 3 (Exact analytical value is 6.0)
result = forward_difference(f, x0=3.0, h=1e-5)
print(result)
# → 6.000010000027034

---
# 3. `symbolic_derivative`

### **symbolic_derivative(func_str, order=1)**
Calculates the symbolic (exact) derivative of the function $f(x)$ using analytical rules. Unlike numerical methods, this function returns a mathematical expression representing the $n$-th order derivative.

---

### **Parameters**

| Parameter | Type | Description |
|:---|:---|:---|
| `func_str` | str | The function string to differentiate (e.g., `"x**3 + sin(x)"`). |
| `order` | int | The order of the derivative (e.g., 1 for $f'$, 2 for $f''$). Defaults to 1. |

---

### **Returns**

| Type | Description |
|:---|:---|
| `sympy.Expr` | The symbolic expression for the $n$-th order derivative. |

---

### **Example**

```python
from pyMatan.derivatives import symbolic_derivative

# Calculate the 2nd order derivative of x^4
derivative = symbolic_derivative("x**4", order=2)
print(derivative)
# → 12*x**2

# Calculate the derivative of sin(x)
derivative_sin = symbolic_derivative("sin(x)", order=1)
print(derivative_sin)
# → cos(x)

---
# 4. `visualize_derivative`

### **visualize_derivative(func_str, x0, h_step=0.5, h_approx=0.1)**
Generates a comprehensive plot to visualize the derivative at a specific point. It compares the **exact tangent line** (derived symbolically) with a **secant line** (derived numerically via central difference), helping to illustrate how the approximation improves as $h$ decreases.



---

### **Parameters**

| Parameter | Type | Description |
|:---|:---|:---|
| `func_str` | str | The function string to plot (e.g., `"x**3"`). |
| `x0` | float | The point at which the derivative and tangent are calculated. |
| `h_step` | float | The radius of the viewing window around $x_0$ for the x-axis. Defaults to 0.5. |
| `h_approx` | float | The step size $h$ used specifically for the numerical secant line. Defaults to 0.1. |

---

### **Returns**

| Type | Description |
|:---|:---|
| `None` | Displays a Matplotlib figure. |

---

### **Example**

```python
from pyMatan.derivatives import visualize_derivative

# Visualize the derivative of x^3 at x=1
# This will show the blue curve, green tangent, and red secant lines.
visualize_derivative("x**3", x0=1.0, h_step=1.0, h_approx=0.5)

---
# 5. `numerical_limit`

### **numerical_limit(func, x0, side="both", h=1e-6)**
Calculates an approximation of the limit of a function $f(x)$ as $x$ approaches $x_0$. It evaluates the function at points very close to $x_0$ from the left, right, or both sides to estimate the value the function is approaching.



---

### **Parameters**

| Parameter | Type | Description |
|:---|:---|:---|
| `func` | callable | The function for which the limit is being sought. |
| `x0` | float | The point that $x$ approaches. |
| `side` | str | The direction of approach: `"left"`, `"right"`, or `"both"`. Defaults to `"both"`. |
| `h` | float | The small step size used for evaluation (must be $> 0$). Defaults to 1e-6. |

---

### **Returns**

| Type | Description |
|:---|:---|
| `float` \| `None` | The numerical approximation of the limit. Returns `None` if the side is `"both"` and the left and right limits do not converge within tolerance. |

---

### **Example**

```python
from pyMatan.limits import numerical_limit

# Function with a limit at x=0
def f(x): 
    return (x**2 - 1) / (x - 1)

# Approximate the limit as x approaches 1 (Exact value is 2)
res = numerical_limit(f, x0=1.0, side="both")
print(res)
# → 2.0

---
# 6. `symbolic_limit`

### **symbolic_limit(func_str, x0, side="both")**
Calculates the exact analytical limit of the function $f(x)$ as $x$ approaches $x_0$ using **SymPy**. This function can handle indeterminate forms (like $0/0$ or $\infty/\infty$) and points of discontinuity where numerical methods might fail.

---

### **Parameters**

| Parameter | Type | Description |
|:---|:---|:---|
| `func_str` | str | The function string (e.g., `"sin(x)/x"`). |
| `x0` | Union[int, float, str] | The point $x$ approaches. Supports `"oo"` for positive infinity and `"-oo"` for negative infinity. |
| `side` | str | Direction of approach: `"both"` (default), `"left"`, or `"right"`. |

---

### **Returns**

| Type | Description |
|:---|:---|
| `sympy.Expr` \| `float` | The exact value of the limit. May return `sympy.nan` if the two-sided limit does not exist. |

---

### **Example**

```python
from pyMatan.limits import symbolic_limit

# Calculate the famous limit: lim_{x->0} sin(x)/x
res = symbolic_limit("sin(x)/x", x0=0)
print(res)
# → 1

# Calculate limit at infinity: lim_{x->oo} 1/x
inf_res = symbolic_limit("1/x", x0="oo")
print(inf_res)
# → 0

---
# 7. `visualize_limit`

### **visualize_limit(func_str, x0, h_window=0.5, n_points=100, h_approx=1e-6)**
Provides a graphical representation of a function's behavior as it approaches a specific point $x_0$. This tool is particularly useful for identifying jump discontinuities, vertical asymptotes, or removable singularities. It highlights the points $(x_0 - h)$ and $(x_0 + h)$ used in numerical estimation to show how the limit is derived.



---

### **Parameters**

| Parameter | Type | Description |
|:---|:---|:---|
| `func_str` | str | The function string to visualize (e.g., `"sin(x)/x"`). |
| `x0` | float | The limit point to investigate. |
| `h_window` | float | The range shown on the x-axis around $x_0$ (distance from center to edge). Defaults to 0.5. |
| `n_points` | int | The number of points sampled to draw the smooth curve. Defaults to 100. |
| `h_approx` | float | The step size $h$ used to place the red "approximation" markers on the graph. Defaults to 1e-6. |

---

### **Returns**

| Type | Description |
|:---|:---|
| `None` | Displays a Matplotlib figure. |

---

### **Example**

```python
from pyMatan.limits import visualize_limit

# Visualize the behavior of f(x) = (x^2 - 1)/(x - 1) near x = 1
# Even though the function is undefined at 1, the limit is visible.
visualize_limit("(x**2 - 1)/(x - 1)", x0=1.0, h_window=2.0)
