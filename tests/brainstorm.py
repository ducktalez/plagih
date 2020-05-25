import numpy as np
import matplotlib.pyplot as plt
import re

ert = """
_123asd_123
v e08u4jnrf_
dfgdfg123
gfg45g_123f4_r34"""

x = re.split('_\d+$', '123_234_ff_23')
if x:
    print(x)


