import sympy
import numpy as np
from typing import Callable, Tuple

def parse_function(func_str: str, variable: str = 'x') -> Tuple[sympy.Expr, Callable]:
    
    try:
        x = sympy.symbols(variable)
    except Exception as e:
        raise ValueError(f"Failed to create a symbolic object for variable '{variable}': {e}")


    try:
        local_dict = {
            'x': x,
            'pi': sympy.pi, 
            'exp': sympy.exp,
            'log': sympy.log,
            'sin': sympy.sin,
            'cos': sympy.cos,
            'tan': sympy.tan,
            'sqrt': sympy.sqrt,
        }
        
        sympy_expr = sympy.sympify(func_str, locals=local_dict)
        
        if not isinstance(sympy_expr, (sympy.Expr, sympy.Number)):
             raise ValueError("SymPy failed to recognize the string as a valid mathematical expression.")

    except (sympy.SympifyError, Exception) as e:
        raise ValueError(f"SymPy parsing error for string '{func_str}': {e}")


    try:
        np_func = sympy.lambdify(x, sympy_expr, modules=['numpy', 'scipy'])
        
    except Exception as e:
        raise ValueError(f"Error creating numerical function (lambdify): {e}")


    return sympy_expr, np_func


print("--- Testing parse_function ---")
test_str = "2 * x**3 - cos(x) + pi"
    
try:
    symbolic_f, numeric_f = parse_function(test_str)

    print(f"\n✅ Input String: '{test_str}'")

    print("\n[SymPy Expression]:")
    print(f"  Type: {type(symbolic_f)}")
    print(f"  Representation: {symbolic_f}")
        
    test_value = 1.0
    result = numeric_f(test_value)
        
    expected = 2 * (1**3) - np.cos(1) + np.pi 
        
    print("\n[Numerical Function (NumPy)]:")
    print(f"  Type: {type(numeric_f)}")
    print(f"  f({test_value}) = {result:.6f}")
    print(f"  Expected (NumPy): {expected:.6f}")
        
    assert np.isclose(result, expected), "Numerical calculation does not match expected value!"
    print("\n✅ Numerical Calculation Test Passed.")
        
except ValueError as e:
    print(f"\n❌ Critical Parsing Error: {e}")