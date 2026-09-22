import torch
import math
def compute_chain_rule_gradient(functions: list[str], x: float) -> float:
    """
    Compute derivative of composite functions using chain rule.
    
    Args:
        functions: List of function names (applied right to left)
                  Available: 'square', 'sin', 'exp', 'log'
        x: Point at which to evaluate derivative
    
    Returns:
        Derivative value at x
    
    Example:
        ['sin', 'square'] represents sin(x²)
        ['exp', 'sin', 'square'] represents exp(sin(x²))
    """
    # Your code here
    derivatives = {
        'square': lambda v: 2 * v,
        'sin': lambda v: math.cos(v),
        'exp': lambda v: math.exp(v),
        'log': lambda v: 1 / v,
    }
    forward = {
        'square': lambda v: v ** 2,
        'sin': lambda v: math.sin(v),
        'exp': lambda v: math.exp(v),
        'log': lambda v: math.log(v),
    }

    value = x
    grad = 1.0

    for name in reversed(functions):
        grad *= derivatives[name](value)
        value = forward[name](value)

    return grad