from plagih.util import *
import sympy


custom_functions = {
    # # 'Min': np.minimum.reduce,  # this IS false
    # # 'Max': np.maximum.reduce,  # sfeh not so nice...
    # 'Min': np.min,  # sfeh this works (?)
    # 'Max': np.max,
    # # 'Min': np.minimum,
    # # 'Max': np.maximum,
    # # sympy.minimum: np.minimum,  # is this even an option?
    # # sympy.maximum: np.maximum,
    # 'RoundDummy': np.round,  todo check this
    'RoundDummy': lambda a: int(round(a)),
}


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
