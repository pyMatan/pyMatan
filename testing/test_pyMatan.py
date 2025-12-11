import sympy
import numpy as np
import matplotlib.pyplot as plt



from pyMatan.limits_derivatives.derivatives import (
    central_difference,
    forward_difference,
    symbolic_derivative,
    visualize_derivative
)
from pyMatan.limits_derivatives.limits import (
    numerical_limit,
    symbolic_limit,
    visualize_limit
)

def assert_close(actual, expected, tol=1e-5):
    """Checks if values are sufficiently close."""
    assert np.isclose(actual, expected, atol=tol), f"Error: Got {actual}, expected {expected} (within {tol})"

print("Imports done.")
# --- TEST 1: Simple Polynomial Function ---
func_str_1 = "x**3 + 2*x"
order_1 = 1
# The exact derivative of x^3 + 2x is 3*x^2 + 2
exact_deriv_1 = sympy.parse_expr("3*x**2 + 2")

result_1 = symbolic_derivative(func_str_1, order=order_1)
assert result_1 == exact_deriv_1, f"Error: Received {result_1}, expected {exact_deriv_1}"
print(f"Test 1 (f(x)={func_str_1}, 1st order): {result_1}")


# --- TEST 2: Higher Order Derivative ---
func_str_2 = "sin(x)"
order_2 = 2  # The second derivative of sin(x) is -sin(x)
exact_deriv_2 = sympy.parse_expr("-sin(x)")

result_2 = symbolic_derivative(func_str_2, order=order_2)
assert result_2 == exact_deriv_2, f"Error: Received {result_2}, expected {exact_deriv_2}"
print(f"Test 2 (f(x)={func_str_2}, 2nd order): {result_2}")

print("\nsymbolic_derivative testing completed successfully.")

# Exact value for reference
x0_test = 2.0
h_small = 1e-4
exact_value = 12.0

# Function for numerical methods
def func_x3(x):
    return x**3

# --- TEST 4: Central Difference (1st Order) ---
# Central Difference has second-order accuracy (O(h^2)), so we expect high precision.
cd_result = central_difference(func_x3, x0_test, h=h_small, order=1)
# High tolerance (1e-8)
assert_close(cd_result, exact_value, tol=1e-8)
print(f"Test 4 (Central Difference, 1st order): Result = {cd_result}, Exact = {exact_value}")

# --- TEST 5: Forward Difference (1st Order) ---
# Forward Difference has first-order accuracy (O(h)), so we expect lower precision.
fd_result = forward_difference(func_x3, x0_test, h=h_small)
# Lower tolerance (1e-3)
assert_close(fd_result, exact_value, tol=1e-3)
print(f"Test 5 (Forward Difference, 1st order): Result = {fd_result}, Exact = {exact_value}")

# --- TEST 6: Central Difference, 2nd Order ---
# The exact 2nd derivative f''(x) = 6x. At x0=2.0, f''(2.0) = 12.0.
exact_value_2nd = 12.0
cd_2nd_result = central_difference(func_x3, x0_test, h=h_small, order=2)
# High tolerance (1e-6)
assert_close(cd_2nd_result, exact_value_2nd, tol=1e-6)
print(f"Test 6 (Central Difference, 2nd order): Result = {cd_2nd_result}, Exact = {exact_value_2nd}")

# --- TEST 7: Input Validation (h <= 0) ---
try:
    central_difference(func_x3, 1.0, h=0)
except ValueError as e:
    print(f"Test 7 (Input Error): Caught expected error: {e}")

print("\nNumerical derivative testing completed successfully.")


# --- TEST 8: Visualization of a Polynomial with moderate h ---
print("Visualization 1: Polynomial Function (x^3 - 4x + 2)")
# x0 = 1.5, h_step = 2.0 (window width), h_approx = 0.5 (secant step)
visualize_derivative("x**3 - 4*x + 2", x0=1.5, h_step=2.0, h_approx=0.5)


# --- TEST 9: Visualization of a Trigonometric Function with small h ---
print("\nVisualization 2: Trigonometric Function (sin(x)) with small h")
# Small h_approx (0.1) should show the secant line closely matching the tangent line.
visualize_derivative("sin(x)", x0=np.pi/4, h_step=1.0, h_approx=0.1)


# --- TEST 10: Visualization showing approximation error with a large h ---
print("\nVisualization 3: Shows poor approximation with a large h")
# Large h_approx (1.0) visibly shows the error between the secant and tangent lines.
visualize_derivative("sin(x)", x0=np.pi/4, h_step=1.0, h_approx=1.0)


print("\nVisualization testing completed successfully.")


# --- TEST 1: Finite Limit (Indeterminate Form 0/0) ---
func_str_1 = "sin(x) / x"
x0_1 = 0
exact_1 = 1
result_1 = symbolic_limit(func_str_1, x0_1, side="both")
assert result_1 == exact_1, f"Error (Test 1): sin(x)/x. Got {result_1}, expected {exact_1}"
print(f"Test 1 (Indeterminate 0/0): Result = {result_1}")


# --- TEST 2: Limit at Infinity ---
func_str_2 = "1 / x"
x0_2 = 'oo' # SymPy notation for infinity
exact_2 = 0
result_2 = symbolic_limit(func_str_2, x0_2, side="both")
assert result_2 == exact_2, f"Error (Test 2): 1/x at oo. Got {result_2}, expected {exact_2}"
print(f"Test 2 (Limit at Infinity): Result = {result_2}")


# --- TEST 3: Discontinuity (Limit Does Not Exist) ---
# Limit of 1/x as x -> 0 (from both sides)
func_str_3 = "1 / x"
x0_3 = 0
result_3 = symbolic_limit(func_str_3, x0_3, side="both")
# SymPy should return NaN if left and right limits differ
assert result_3 is sympy.nan, f"Error (Test 3): 1/x at 0. Got {result_3}, expected NaN"
print(f"Test 3 (Limit DNE): Result = {result_3}")


# --- TEST 4: One-Sided Limit (Right) ---
func_str_4 = "1 / x"
x0_4 = 0
result_4 = symbolic_limit(func_str_4, x0_4, side="right")
# The limit should be positive infinity
assert result_4 == sympy.oo, f"Error (Test 4): 1/x right limit. Got {result_4}, expected oo"
print(f"Test 4 (Right Limit): Result = {result_4}")


print("\nsymbolic_limit testing completed successfully.")


# Function used for numerical tests
def f_x(x):
    # Function: x^2 + 1
    return x**2 + 1

# --- TEST 5: Finite Limit (Side 'both') ---
x0_5 = 2.0
# Exact limit is 2^2 + 1 = 5.0
exact_5 = 5.0
result_5 = numerical_limit(f_x, x0_5, side="both", h=1e-6)
assert_close(result_5, exact_5, tol=1e-4)
print(f"Test 5 (Numerical, both): Result = {result_5}, Expected ≈ {exact_5}")


# --- TEST 6: One-Sided Limit ('left') ---
x0_6 = 1.0
# Exact limit is 1^2 + 1 = 2.0
exact_6 = 2.0
result_6 = numerical_limit(f_x, x0_6, side="left", h=1e-6)
assert_close(result_6, exact_6, tol=1e-4)
print(f"Test 6 (Numerical, left): Result = {result_6}, Expected ≈ {exact_6}")


# --- TEST 7: Limit DNE (Jump Discontinuity) ---
def f_jump(x):
    # Jump discontinuity at x=0: 1 for x>0, -1 for x<=0
    return 1 if x > 0 else -1

x0_7 = 0
result_7 = numerical_limit(f_jump, x0_7, side="both", h=1e-6)
# The difference between 1 and -1 is 2. Since 2 is much larger than the tolerance (1e-4), it should return None.
assert result_7 is None, f"Error (Test 7): Jump DNE. Got {result_7}, expected None"
print(f"Test 7 (Numerical, DNE): Result = {result_7}")


# --- TEST 8: Input Validation (h <= 0) ---
try:
    numerical_limit(f_x, 1.0, h=0)
except ValueError as e:
    print(f"Test 8 (Input Error): Caught expected error: {e}")

print("\nnumerical_limit testing completed successfully.")



# --- TEST 9: Standard Limit (Removable Discontinuity) ---
print("Visualization 1: sin(x)/x as x -> 0 (Removable Discontinuity)")
# The limit is 1. The plot should show a hole at x=0.
visualize_limit("sin(x)/x", x0=0, h_window=1.5, h_approx=1e-6)


# --- TEST 10: Infinite Discontinuity (Vertical Asymptote) ---
print("\nVisualization 2: 1/x as x -> 0 (Infinite Discontinuity)")
# The function blows up at x=0. The plot should show the asymptote and points approaching it.
visualize_limit("1/x", x0=0, h_window=0.5, h_approx=1e-6)


print("\nvisualize_limit testing completed successfully. Check the generated plots above.")