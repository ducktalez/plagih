import numpy as np
import sympy as sp
import pandas as pd

def evaluate_sympy_expression(expr, df):
    """
    Evaluate a SymPy expression on a pandas DataFrame using columns corresponding to the symbols in the expression.

    Args:
        expr (sympy.Expr): The SymPy expression to evaluate.
        df (pd.DataFrame): The DataFrame containing columns matching the symbols in the expression.

    Returns:
        np.ndarray: The evaluated results as a NumPy array.
    """
    # Extract symbols from the expression
    symbols = sorted(expr.free_symbols, key=lambda s: str(s))  # Sort for consistent ordering

    # Create a lambdified function
    lambdified_func = sp.lambdify(symbols, expr, modules="numpy")

    # Map symbols to DataFrame columns
    args = [df[str(symbol)].to_numpy() for symbol in symbols]

    # Evaluate the expression
    return lambdified_func(*args)


# Example usage
if __name__ == "__main__":
    # Define a SymPy expression
    x, y, z = sp.symbols('x y z')
    expr = x**2 + sp.sin(y) + sp.log(z)

    # Create a DataFrame
    df = pd.DataFrame({
        'x': [1, 2, 3],
        'y': [0.5, 1.0, 1.5],
        'z': [2.718, 7.389, 20.085]
    })

    # Evaluate the expression and store the result in a new column
    df['result'] = evaluate_sympy_expression(expr, df)

    print(df)