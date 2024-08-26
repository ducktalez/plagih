from plagih.util import *
import pandas as pd
import sympy
from sympy.utilities.exceptions import ignore_warnings
import warnings
from sklearn.model_selection import train_test_split
import numpy as np


def eval_regression_sym_experimental(expr, df):
    """
    Returns the fitness (float)
    todo
        - This is seemingly faster than TF?... -> There is no reason why it should be faster with TF on processors
        - Not stable and only working with mountaincar

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

    a, b = sympy.symbols('cartVel cartPos')
    cartVels = df['cartVel']
    cartPoss = df['cartPos']
    f = sympy.lambdify([a, b], expr, 'numpy')
    
    #     x = expr.evalf(subs={'cartVel': cartVels, 'cartPos': cartPoss, 'action': actions})
    with warnings.catch_warnings():
        with ignore_warnings(RuntimeWarning):  # often in ITE-terms? When math errors occur
            with ignore_warnings(DeprecationWarning):  # something like use "**" instead of "Pow"
                raw_results = f(cartVels, cartPoss)

    results = np.round(np.clip(raw_results, 0, 2), 0)    # sfeh:data-specific! sanitize_results

    return results


def regression_error(yy_hat, yy):

    fitness = np.sqrt(np.mean((yy - yy_hat) ** 2))  # discuss: np.square vs. **2: should be mainly irrelevant
    fitness = round(fitness, FLOAT_PRECISION)
    return fitness
