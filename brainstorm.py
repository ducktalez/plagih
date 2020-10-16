"""
#!/usr/bin/env bash
pdflatex agents_trees.tex
find . -name "visualisation/*.tex" -exec pdflatex {} \;
find . -name "visualisation/*.pdf" -exec pdftoppm {} {} -png \;
a*b * Ifte ** a*b*c *I *Ifte
"""

# from multiprocessing import Pool
#
#
# def f(x):
#     return x*x
#
#
# if __name__ == '__main__':
#     with Pool(5) as p:
#         x = p.map(f, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
#
#     print(x)

import matplotlib.pyplot as plt
import numpy as np
t = np.arange(0.0, 2.0, 0.2)
s = 1 + np.sin(2 * np.pi * t)
rc_params = {'text.usetex': True, 'figure.autolayout': True, 'font.size': 11, 'figure.figsize': (2.8, 2.1),
                 # 'lines.markersize': 5,
                 'xtick.labelsize': 8,
                 'ytick.labelsize': 8,
                'axes.xmargin': 0,
                'axes.ymargin': 0,
             }
with plt.rc_context(rc=rc_params):
    fig, ax = plt.subplots()
    ax.plot(t, s, marker='x', label='random values (idk)')
    ax.set(xlabel='some label (time)', ylabel='voltage (mV)')
    ax.legend(loc='lower left')

    fig.savefig(f"test-{str(rc_params.values())}.pdf", backend='pgf')
    fig.savefig(f"test-{str(rc_params.values())}.png", dpi=100)
    plt.show()

