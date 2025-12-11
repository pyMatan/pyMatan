markdown
# pyMatan.series — Series and Expansion Tools

This module provides numerical and symbolic tools for analyzing infinite series,
Taylor expansions, and Fourier series.  
Functions follow a design similar to scientific libraries (NumPy, SymPy, SciPy).

---

# Table of Contents
- [1. d_alembert_test](#1-d_alembert_test)
- [2. cauchy_root_test](#2-cauchy_root_test)
- [3. integral_test](#3-integral_test)
- [4. leibniz_test](#4-leibniz_test)
- [5. partial_sum](#5-partial_sum)
- [6. taylor_polynomial](#6-taylor_polynomial)
- [7. taylor_remainder_lagrange](#7-taylor_remainder_lagrange)
- [8. fourier_coefficients](#8-fourier_coefficients)
- [9. fourier_partial_sum](#9-fourier_partial_sum)
- [10. plot_taylor_approximations](#10-plot_taylor_approximations)

---

# 1. `d_alembert_test`

### **d_alembert_test(a_n, n_start=50, steps=200)**  
Estimate the limit in D’Alembert’s ratio test:
```markdown
L = \lim_{n\to\infty} \left|\frac{a_{n+1}}{a_n}\right|
```
Used to determine absolute convergence of a series.

---

### **Parameters**
| Parameter | Type | Description |
|----------|------|-------------|
| `a_n` | callable | Function returning series term \(a_n\). |
| `n_start` | int, default 50 | Starting index for ratio sampling. |
| `steps` | int, default 200 | Number of ratios to compute. |

---

### **Returns**
| Type | Description |
|------|-------------|
| `float` | Estimated limit \(L\). Returns `nan` if invalid. |

---

### **See also**
- `cauchy_root_test`
- `partial_sum`
- `leibniz_test`

---

### **Example**
python
from pyMatan.series import d_alembert_test

def a(n): return (1/3)**n
d_alembert_test(a)
# → 0.3333 (approx)


---

# 2. `cauchy_root_test`

### **cauchy_root_test(a_n, n_start=50, steps=200)**

Estimates the limit:

[
L = \lim_{n\to\infty} |a_n|^{1/n}
]

Useful for detecting **exponential-type decay**.

---

### **Parameters**

| Parameter | Type     | Description                  |
| --------- | -------- | ---------------------------- |
| `a_n`     | callable | Function producing (a_n).    |
| `n_start` | int      | Starting index for sampling. |
| `steps`   | int      | Number of samples.           |

---

### **Returns**

`float` — Approximation of root-limit.

---

### **Example**

python
from pyMatan.series import cauchy_root_test

cauchy_root_test(lambda n: (1/2)**n)
# → ~0.5


---

# 3. `integral_test`

### **integral_test(func_str, a=1, upper=2000.0, n=50000)**

Numerically evaluates an improper integral to determine series convergence.

Approximates:

[
\int_a^{\infty} f(x),dx
]

using trapezoidal integration.

---

### **Parameters**

| Parameter  | Type  | Description                               |
| ---------- | ----- | ----------------------------------------- |
| `func_str` | str   | Function in string form, e.g. `"1/x**2"`. |
| `a`        | int   | Lower bound (≥ 1).                        |
| `upper`    | float | Upper cutoff approximating infinity.      |
| `n`        | int   | Number of evaluation points.              |

---

### **Returns**

`float` — Approximate integral value.

---

### **Example**

```python
integral_test("1/x**2", a=1, upper=500)
# → ~1.0
```

---

# 4. `leibniz_test`

### **leibniz_test(a_n, check_terms=50)**

Checks convergence of alternating series:

[
\sum (-1)^{n+1} a_n
]

Conditions tested:

1. (a_n > 0)
2. (a_n) is decreasing
3. (a_n \to 0)

---

### **Parameters**

| Parameter     | Type     | Description                             |
| ------------- | -------- | --------------------------------------- |
| `a_n`         | callable | Function returning positive term (a_n). |
| `check_terms` | int      | Number of terms inspected.              |

---

### **Returns**

`bool` — Whether the alternating test appears satisfied.

---

### **Example**

```python
leibniz_test(lambda n: 1/n)
# → True
```

---

# 5. `partial_sum`

### **partial_sum(a_n, N)**

Computes:
```
S_N = \sum_{n=1}^N a_n

```
---

### **Parameters**

| Parameter | Type     | Description                     |
| --------- | -------- | ------------------------------- |
| `a_n`     | callable | Function returning series term. |
| `N`       | int      | Number of terms to sum.         |

---

### **Returns**

`float` — Partial sum value.

---

### **Example**

```python
partial_sum(lambda n: n, 5)
# → 15
```

---

# 6. `taylor_polynomial`

### **taylor_polynomial(func_str, x0=0.0, degree=5)**

Generates symbolic Taylor polynomial around point `x0`.

[
P_n(x) = \sum_{k=0}^n \frac{f^{(k)}(x_0)}{k!}(x-x_0)^k
]

---

### **Parameters**

| Parameter  | Type  | Description                          |
| ---------- | ----- | ------------------------------------ |
| `func_str` | str   | Function as string, e.g. `"sin(x)"`. |
| `x0`       | float | Expansion point.                     |
| `degree`   | int   | Degree of polynomial.                |

---

### **Returns**

`sympy.Expr` — Symbolic polynomial.

---

### **Example**

```python
taylor_polynomial("sin(x)", degree=5)
```

---

# 7. `taylor_remainder_lagrange`

### **taylor_remainder_lagrange(func_str, x, x0, degree)**

Computes symbolic Lagrange-form remainder:

[
R_n(x)=\frac{f^{(n+1)}(\xi)}{(n+1)!}(x-x_0)^{n+1}
]

The value uses (f^{(n+1)}(x)) as an estimate.

---

### **Parameters**

| Parameter  | Type  | Description                 |
| ---------- | ----- | --------------------------- |
| `func_str` | str   | Function.                   |
| `x`        | float | Evaluation point.           |
| `x0`       | float | Center of Taylor expansion. |
| `degree`   | int   | n for remainder.            |

---

### **Returns**

`sympy.Expr` — Remainder estimate.

---

### **Example**

```python
taylor_remainder_lagrange("exp(x)", 1, 0, 2)
```

---

# 8. `fourier_coefficients`

### **fourier_coefficients(func_str, L, n_terms=10)**

Computes symbolic Fourier coefficients for function on interval ([-L, L]):

[
a_0,\ a_n,\ b_n
]

---

### **Parameters**

| Parameter  | Type  | Description              |
| ---------- | ----- | ------------------------ |
| `func_str` | str   | Function string.         |
| `L`        | float | Half-length of interval. |
| `n_terms`  | int   | Number of harmonics.     |

---

### **Returns**

Tuple `(a0, a_list, b_list)`.

---

### **Example**

```python
fourier_coefficients("x", L=sp.pi, n_terms=5)
```

---

# 9. `fourier_partial_sum`

### **fourier_partial_sum(func_str, L, n_terms)**

Builds symbolic Fourier partial sum:

[
S_n(x)=\frac{a_0}{2}+\sum_{k=1}^n\left(a_k\cos\frac{k\pi x}{L}+b_k\sin\frac{k\pi x}{L}\right)
]

---

### **Returns**

`sympy.Expr`

---

### **Example**

```python
S5 = fourier_partial_sum("x", L=np.pi, n_terms=5)
```

---

# 10. `plot_taylor_approximations`

### **plot_taylor_approximations(func_str, x0=0, degrees=(1,2,3), x_min=-2, x_max=2)**

Plots a function and several of its Taylor polynomial approximations.

Useful for visualizing convergence.

---

### **Parameters**

| Parameter  | Type          | Description         |
| ---------- | ------------- | ------------------- |
| `func_str` | str           | Function to plot.   |
| `x0`       | float         | Expansion point.    |
| `degrees`  | iterable[int] | Polynomial degrees. |
| `x_min`    | float         | Left boundary.      |
| `x_max`    | float         | Right boundary.     |

---

### **Returns**

`None`

---

### **Example**

```python
plot_taylor_approximations("sin(x)", 0, degrees=(1,3,5))
```

---

# End of `series.md`
