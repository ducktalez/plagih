import matplotlib.pyplot as plt
import numpy as np
from sympy import sympify

algo_raw = 'if(1<0) then (0) else (2)'
algo_raw = 'a * b + 5 ** 4 - 2 + 3'

algo_sym = sympify(algo_raw)

print(type(algo_sym))