import pandas as pd

from plagih.sympy_extras import plagih_sympify
from plagih.tree_labels import Round_Dummy, PowRounded
from plagih.util import *
import sympy
from sympy.utilities.exceptions import ignore_warnings
import warnings
import numpy as np


custom_functions = {
    # 'Min': np.minimum.reduce,  # todo, might be false.
    # 'Max': np.maximum.reduce,
    'Min': np.min,  # todo, might be false.
    'Max': np.max,
    'Round_Dummy': np.round,
}


def eval_predict_df(sy_expr: sympy.Basic, df: pd.DataFrame, variable_names, normalize_numpy):
    """
    Returns the fitness (float)

    """
    # a, b = sympy.symbols('cartVel cartPos')
    # cartVels = df['cartVel']
    # cartPoss = df['cartPos']
    cartPos, cartVel = sympy.symbols('cartPos cartVel')
    variable_names = [str(var) for var in (cartPos, cartVel)]  # sfeh
    func = sympy.lambdify(tuple([cartPos, cartVel]), sy_expr, modules=[custom_functions, 'numpy'])

    # x = expr.evalf(subs={symbols[0]: 1.234, symbols[1]: 2.3456})

    # SUCK FYMPY
    # "ValueError: setting an array element with a sequence.
    #   The requested array has an inhomogeneous shape after 1 dimensions.
    #   The detected shape was (2,) + inhomogeneous part."

    # x = expr.evalf(subs={'cartVel': cartVels, 'cartPos': cartPoss})
    # raw_results = df.apply(lambda row: func(row['cartPos'], row['cartVel']), axis=1)
    with warnings.catch_warnings():
        with ignore_warnings(RuntimeWarning):  # often in ITE-terms? When math errors occur
            with ignore_warnings(DeprecationWarning):  # something like use "**" instead of "Pow"
                df_results = df.apply(lambda row: func(row['cartPos'], row['cartVel']), axis=1)

                # np_input = [df[str(symbol)].to_numpy() for symbol in symbols]
                # np_results = func(*np_input)
                # # df_results = df.apply(lambda row: func(), axis=1)
                # # df_results = df.apply(lambda row: func(row['cartPos'], row['cartVel']), axis=1)
                # # raw_results = df.apply(lambda row: func(*[row[var] for var in variable_names]), axis=1)

    if normalize_numpy is not None:  # clip and round result
        df_results = normalize_numpy(df_results)

    # todo numpy evaluation

    # df['result'] = df_results
    return df_results

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

def eval_tensorflow(expr, df):
    pass



if __name__ == '__main__':
    import pandas as pd
    df = pd.read_csv(Path(__file__).parent.parent.absolute() / f'benchmarks/mc/gp_files/samples200.csv').astype('float32')
    symbols = sympy.symbols(['cartVel', 'cartPos'], real=True, imaginary=False)
    ex = '0.00162*cartPos*cartVel/(cartPos + 4.23)'
    # expr = sympy.Mul(symbols[0], (2, sympy.Add(1, symbols[1])))
    expr = PowRounded.symfun(2, sympy.Add(1, symbols[1]))
    # sfeh
    cartPos, cartVel = sympy.symbols('cartPos cartVel')
    variable_names = [str(var) for var in (cartPos, cartVel)]
    func = sympy.lambdify(tuple(symbols), expr, modules=[custom_functions, 'numpy'])

    func = sympy.lambdify(tuple(symbols), expr, modules=[custom_functions, 'numpy'])
    raw_results = df.apply(lambda row: func(row['cartPos'], row['cartVel']), axis=1)
    print('ssdfg', raw_results)