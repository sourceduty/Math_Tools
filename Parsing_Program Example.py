# Parsing_Program Example
# https://chatgpt.com/g/g-67cc981656b8819196c22b67c9fbbb8c-sourceduty-math

import sympy as sp
from sympy.parsing.sympy_parser import parse_expr

# ===== Sourceduty-inspired framework modules =====

def factorchain_parser(expr):
    factored = sp.factor(expr)
    return {"factored_form": factored}

def contripot_transform(expr):
    terms = expr.as_ordered_terms()
    energies = [sp.integrate(term**2, ('x', 0, 1)) for term in terms if term.has(sp.Symbol('x'))]
    return {"signal_energy": energies}

def quadexpo_detect(expr):
    degree = sp.Poly(expr).degree()
    classification = "quadratic" if degree == 2 else "exponential" if expr.has(sp.exp) else "other"
    return {"degree": degree, "classification": classification}

def truthvar_analysis(expr):
    vars_used = expr.free_symbols
    logical_map = {str(var): True for var in vars_used}
    return {"truth_vars": logical_map}

def joint_driver_weight(expr):
    weights = {}
    for symbol in expr.free_symbols:
        count = str(expr).count(str(symbol))
        weights[str(symbol)] = count
    return {"weights": weights}

def universal_organize(results):
    return {key: val for module in results for key, val in module.items()}

# ===== Main Parser Function =====

def sourceduty_parser(expression_str):
    x = sp.Symbol('x')
    expr = parse_expr(expression_str, evaluate=False)

    results = []
    results.append(factorchain_parser(expr))
    results.append(contripot_transform(expr))
    results.append(quadexpo_detect(expr))
    results.append(truthvar_analysis(expr))
    results.append(joint_driver_weight(expr))

    final_result = universal_organize(results)
    return final_result

# Example usage
if __name__ == "__main__":
    expr_input = "3*x**2 + 6*x + 9"
    parsed = sourceduty_parser(expr_input)
    for key, val in parsed.items():
        print(f"{key}: {val}")
