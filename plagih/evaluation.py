import pandas as pd

from plagih.trees import PowRounded, Round_Dummy
from plagih.util import *
import sympy
from sympy.utilities.exceptions import ignore_warnings
import warnings
import numpy as np


# custom_functions = {
#     # 'Min': np.minimum.reduce,  # this IS false
#     # 'Max': np.maximum.reduce,  # sfeh not so nice...
#     'Min': np.min,  # sfeh this works (?)
#     'Max': np.max,
#     # 'Min': np.minimum,
#     # 'Max': np.maximum,
#     # sympy.minimum: np.minimum,  # is this even an option?
#     # sympy.maximum: np.maximum,
#     'Round_Dummy': np.round,
# }


def eval_predict_df(sy_expr: sympy.Basic, df: pd.DataFrame, symbol_list):
    """
    """
    # func = sympy.lambdify(symbol_list, sy_expr, modules=[custom_functions, 'numpy'])
    func = sympy.lambdify(symbol_list, sy_expr, modules=['numpy'])

    with warnings.catch_warnings():
        with ignore_warnings(RuntimeWarning):  # often in ITE-terms? When math errors occur
            with ignore_warnings(DeprecationWarning):  # something 'like use "**" instead of "Pow"'
                df_results = df.apply(lambda row: func(*[row[str(var)] for var in symbol_list]), axis=1)

    return df_results

def evaluate_sympy_expression(expression, df, symbols):
    """
    Does NOT work!

    Try to evaluate this:
    - expr = sympy.Min(2, 2 * cartPos)
    """
    np_input = [df[str(name)].to_numpy() for name in symbols]
    func = sympy.lambdify(tuple(symbols), expression, modules='numpy')

    with warnings.catch_warnings():
        with ignore_warnings(RuntimeWarning):  # often in ITE-terms? When math errors occur
            with ignore_warnings(DeprecationWarning):  # something 'like use "**" instead of "Pow"'
                result = func(*np_input)

    return result

def eval_sympyLoop(expr, df):

    cartVel, cartPos = sympy.symbols('cartVel cartPos')
    ex = sympy.sympify(str(expr))
    f = sympy.lambdify([cartVel, cartPos], ex, 'numpy')
    cartVel = np.array(df['cartVel'])
    cartPos = np.array(df['cartPos'])
    action = np.array(df['action'])
    raw_results = f(cartVel, cartPos)
    results = np.round(np.clip(raw_results, 0, 2), 0)
    #
    # if not return_results:
    #     fitness = np.sqrt(np.mean((results-action)**2))
    #     return np.round(fitness, FLOAT_PRECISION)
    # else:
    #     return results

    return results

# def eval_tensorflow(expr, df):
#     pass


if __name__ == '__main__':
    import pandas as pd
    df = pd.read_csv(Path(__file__).parent.parent.absolute() / f'benchmarks/mc/gp_files/samples200.csv').astype('float32')
    symbols = sympy.symbols(['cartVel', 'cartPos'], real=True, imaginary=False)
    cartPos, cartVel = symbols[0], symbols[1]
    ex = '0.00162*cartPos*cartVel/(cartPos + 4.23)'
    # expr = sympy.Mul(symbols[0], (2, sympy.Add(1, symbols[1])))
    expr = sympy.Min(2, sympy.Add(1, symbols[1]))
    expr = sympy.Min(cartVel, sympy.Add(2, symbols[1]))
    # sfeh
    variable_names = [str(var) for var in (cartPos, cartVel)]
    # func = sympy.lambdify(tuple(symbols), expr, modules=[custom_functions, 'numpy'])
    # raw_results = df.apply(lambda row: func(row['cartPos'], row['cartVel']), axis=1)

    result = evaluate_sympy_expression(expr, df, [cartVel, cartPos])
    print('ssdfg', result)
