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

