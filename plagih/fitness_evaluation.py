import pandas as pd
from plagih.tree_labels import Round_Dummy, PowRounded
from plagih.util import *
import sympy
from sympy.utilities.exceptions import ignore_warnings
import warnings
import numpy as np


custom_functions = {
    # # todo when symbols are used
    # 'Min': np.minimum.reduce,
    # 'Max': np.maximum.reduce,
    'Min': np.min,
    'Max': np.max,
    'Round_Dummy': np.round,
}


def eval_predict(expr, df: pd.DataFrame, symbols, normalize_numpy):
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
    func = sympy.lambdify(tuple(symbols), expr, modules=[custom_functions, 'numpy'])

    # x = expr.evalf(subs={symbols[0]: 1.234, symbols[1]: 2.3456})

    # SUCK FYMPY
    # "ValueError: setting an array element with a sequence.
    #   The requested array has an inhomogeneous shape after 1 dimensions.
    #   The detected shape was (2,) + inhomogeneous part."

    print(f'asddsa: {expr}')

    with warnings.catch_warnings():
        with ignore_warnings(RuntimeWarning):  # often in ITE-terms? When math errors occur
            with ignore_warnings(DeprecationWarning):  # something like use "**" instead of "Pow"

                np_input = [df[str(symbol)].to_numpy() for symbol in symbols]
                np_results = func(*np_input)
                # df_results = df.apply(lambda row: func(), axis=1)
                # df_results = df.apply(lambda row: func(row['cartPos'], row['cartVel']), axis=1)
                # raw_results = df.apply(lambda row: func(*[row[var] for var in variable_names]), axis=1)

    if normalize_numpy is not None:  # clip and round result
        np_results = normalize_numpy(np_results)

    return np_results


if __name__ == '__main__':
    import pandas as pd
    df = pd.read_csv(Path(__file__).parent.parent.absolute() / f'benchmarks/mc/gp_files/samples200.csv').astype('float32')
    symbols = sympy.symbols(['cartVel', 'cartPos'], real=True, imaginary=False)
    ex = '0.00162*cartPos*cartVel/(cartPos + 4.23)'

    # sfeh
    # cartPos, cartVel = sympy.symbols('cartPos cartVel')
    # variable_names = [str(var) for var in (cartPos, cartVel)]
    # func = sympy.lambdify(tuple(symbols), expr, modules=[custom_functions, 'numpy'])

    # ex = '0.00162*cartVel*cartVel/(cartVel + 4.23)'

    symbols = sympy.symbols(['cartVel', 'cartPos'], real=True, imaginary=False)
    cartVel, cartPos = symbols[0], symbols[1]
    ex = '1000*cartPos'
    expr = sympy.Mul(1000, cartPos)
    ex = '1000*cartPos+cartVel'
    expr = sympy.Add(sympy.Mul(1000, cartPos), cartVel)
    ex = '1000*cartVel+cartPos'
    expr = sympy.Add(sympy.Mul(1000, cartVel), cartPos)
    # expr = PowRounded.symfun(2, sympy.Add(1, symbols[1]))
    # expr = sympy.Mul(2, sympy.Add(1, symbols[1]))
    # raw_results = df.apply(lambda row: func(np_input), axis=1)

    expr = sympy.Min(cartVel, 2)

    symbols = sympy.symbols(['cartVel', 'cartPos'], real=True, imaginary=False)

    custom_functions = {
        'Min': min
    }
    # expr = sympy.Mul(2, sympy.Add(1, symbols[1]))
    func = sympy.lambdify(tuple(symbols), expr, modules=[custom_functions, 'numpy'])
    np_input = [df[str(symbol)].to_numpy() for symbol in symbols]
    raw_results = func(*np_input)
    print('raw_results:', raw_results)
    print(f'sum: {np.sum(raw_results)}')