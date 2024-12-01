import pandas

from plagih.sympy_extras import plagih_sympify
from plagih.tree_labels import Round_Dummy, PowRounded
from plagih.util import *
import sympy
from sympy.utilities.exceptions import ignore_warnings
import warnings
import numpy as np


custom_functions = {
    'Min': np.minimum.reduce,
    'Max': np.maximum.reduce,
    'Round_Dummy': Round_Dummy,
}


def eval_predict(expr, df: pandas.DataFrame, variable_names, normalize_numpy):
    """
    Returns the fitness (float)

    def eval_sym_experimental(self, expr, return_results=False):
        # Not stable and only working with mountaincar

        _inputs = self.data_dict

        cartVel, cartPos = sympy.symbols('cartVel cartPos')
        ex = sympy.sympify(str(expr))
        f = sympy.lambdify([cartVel, cartPos], ex, 'numpy')
        cartVel = np.array(_inputs['cartVel'])
        cartPos = np.array(_inputs['cartPos'])
        action = np.array(_inputs['action'])
        raw_results = f(cartVel, cartPos)
        results = np.round(np.clip(raw_results, 0, 2), 0)

        if not return_results:
            fitness = np.sqrt(np.mean((results-action)**2))
            return np.round(fitness, FLOAT_PRECISION)
        else:
            return results
    """
    # a, b = sympy.symbols('cartVel cartPos')
    # cartVels = df['cartVel']
    # cartPoss = df['cartPos']
    # print(f'HEsdfgRE TODO: {expr}')
    cartPos, cartVel = sympy.symbols('cartPos cartVel')
    variable_names = [str(var) for var in (cartPos, cartVel)]
    func = sympy.lambdify(tuple([cartPos, cartVel]), expr, modules=[custom_functions, 'numpy'])

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
                df['result'] = df.apply(lambda row: func(row['cartPos'], row['cartVel']), axis=1)
                # raw_results = df.apply(lambda row: func(*[row[var] for var in variable_names]), axis=1)

                #     TODO YO WTF
                # if normalize_numpy is not None:  # clip and round result
                #     # raw_results = normalize_numpy(raw_results)
                #     raw_results = lambda x: pd.round(np.clip(raw_results, 0, 2), 0)


    if normalize_numpy is not None:  # clip and round result
        # raw_results = normalize_numpy(raw_results)
        try:
            raw_results = lambda x: np.round(np.clip(raw_results, 0, 2), 0)
        except Exception as TODO:
            raw_results = lambda x: np.round(np.clip(raw_results, 0, 2), 0)

    return df


if __name__ == '__main__':
    import pandas as pd
    df = pd.read_csv(Path(__file__).parent.parent.absolute() / f'benchmarks/mc/gp_files/samples200.csv').astype('float32')
    symbols = sympy.symbols(['cartVel', 'cartPos'], real=True, imaginary=False)
    ex = '0.00162*cartPos*cartVel/(cartPos + 4.23)'
    # expr = sympy.Mul(symbols[0], (2, sympy.Add(1, symbols[1])))
    expr = PowRounded.symfun(2, sympy.Add(1, symbols[1]))
    print(f'HERE TODO22: {expr}')
    cartPos, cartVel = sympy.symbols('cartPos cartVel')
    variable_names = [str(var) for var in (cartPos, cartVel)]
    func = sympy.lambdify(tuple(symbols), expr, modules=[custom_functions, 'numpy'])

    func = sympy.lambdify(tuple(symbols), expr, modules=[custom_functions, 'numpy'])
    raw_results = df.apply(lambda row: func(row['cartPos'], row['cartVel']), axis=1)
    print('ssdfg', raw_results)