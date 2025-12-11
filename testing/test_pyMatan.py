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
from pyMatan.series import (
    d_alembert_test,
    cauchy_root_test,
    integral_test,
    leibniz_test,
    partial_sum,
    taylor_polynomial,
    taylor_remainder_lagrange,
    fourier_coefficients,
    fourier_partial_sum,
    plot_taylor_approximations
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


def a_geo(n):
    return (1/2)**n

def a_harmonic(n):
    return 1/n

def a_identity(n):
    return n


# --- TEST 1: D'Alembert Ratio Test ---
# For geometric series a_n = (1/2)^n, ratio limit is L = 1/2.
expected_ratio = 0.5
ratio_result = d_alembert_test(a_geo, n_start=80, steps=80)
# Moderate tolerance due to averaging
assert_close(ratio_result, expected_ratio, tol=1e-2)
print(f"Test 1 (D'Alembert): result = {ratio_result}, expected = {expected_ratio}")


# --- TEST 2: Cauchy Root Test ---
# For geometric series a_n = (1/2)^n, root limit is L = 1/2.
expected_root = 0.5
root_result = cauchy_root_test(a_geo, n_start=80, steps=80)
# Same tolerance as ratio test
assert_close(root_result, expected_root, tol=1e-2)
print(f"Test 2 (Cauchy Root): result = {root_result}, expected = {expected_root}")


# --- TEST 3: Integral Test ---
# For ∫[1,∞] 1/x^2 dx, the exact value is 1.
expected_int = 1.0
int_result = integral_test("1/x**2", a=1, upper=300, n=20000)
# Numerical integral requires moderate tolerance
assert_close(int_result, expected_int, tol=1e-2)
print(f"Test 3 (Integral Test): result = {int_result}, expected = {expected_int}")


# --- TEST 4: Leibniz Alternating Test ---
# For a_n = 1/n, the alternating series satisfies Leibniz conditions.
leib_result = leibniz_test(a_harmonic)
assert leib_result is True, "Error: Leibniz test should return True."
print(f"Test 4 (Leibniz): result = {leib_result}")


# --- TEST 5: Partial Sum ---
# Σ n from 1..5 = 15 exactly.
expected_sum = 15.0
sum_result = partial_sum(a_identity, 5)
assert_close(sum_result, expected_sum)
print(f"Test 5 (Partial Sum): result = {sum_result}, expected = {expected_sum}")


# --- TEST 6: Taylor Polynomial ---
# For sin(x), Taylor P3 at x0=0 is x - x^3/6.
x = sympy.Symbol("x")
expected_poly = x - x**3/6
poly_result = taylor_polynomial("sin(x)", x0=0, degree=3)
assert sympy.simplify(poly_result - expected_poly) == 0
print(f"Test 6 (Taylor P3): result = {poly_result}, expected = {expected_poly}")


# --- TEST 7: Taylor Remainder ---
# For exp(x) at x=1, R2(x) ≈ e/6 (Lagrange form approximation).
expected_rem = sympy.E / 6
rem_result = taylor_remainder_lagrange("exp(x)", x=1, x0=0, degree=2)
# Moderate tolerance (symbolic remainder approximation)
assert_close(float(rem_result), float(expected_rem), tol=1e-1)
print(f"Test 7 (Taylor Remainder): result = {rem_result}, expected ≈ {expected_rem}")


# --- TEST 8: Fourier Coefficients ---
# For f(x)=1 on [-L, L]:  a0=2,  a_n=0,  b_n=0.
a0, a_list, b_list = fourier_coefficients("1", L=sympy.pi, n_terms=3)
assert_close(float(a0), 2.0)
assert all(float(a) == 0 for a in a_list)
assert all(float(b) == 0 for b in b_list)
print(f"Test 8 (Fourier Coeffs): a0={a0}, a_n={a_list}, b_n={b_list}")


# --- TEST 9: Fourier Partial Sum ---
# For constant function 1, the Fourier partial sum satisfies S_n(x)=1.
S9 = fourier_partial_sum("1", L=sympy.pi, n_terms=5)
val_0 = float(S9.subs(x, 0))
assert_close(val_0, 1.0)
print(f"Test 9 (Fourier Partial Sum): S_n(0) = {val_0}, expected = 1.0")


# --- TEST 10: Taylor Visualization ---
# Only checks that the plotting function executes without error.
plot_taylor_approximations("sin(x)", 0, degrees=(1,2,4), x_min=-2, x_max=2)
print("Test 10 (Visualization): completed.")
print("\nSeries function testing completed successfully.")