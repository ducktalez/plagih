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

from plagih.file_interaction import *

t = np.arange(-1.2, 0.6, 0.1)
s = 0.1 * np.sin(2 * np.pi * t)


# rc_params = {'text.usetex': True, 'figure.autolayout': True, 'font.size': 11, 'figure.figsize': (3.6, 2.7), 'backend': 'pgf',
#                  # 'lines.markersize': 5,
#              'axes.labelpad': 0.5,  # padding axis-ticks to axis title
#              'xtick.labelsize': 8, 'xtick.major.size': 1.5, 'xtick.major.pad': 0.8,
#              'ytick.labelsize': 8, 'ytick.major.size': 1.5, 'ytick.major.pad': 0.8,
#              'savefig.pad_inches': 0,
#              # 'xtick.minor.size': 9,
#                 #  'ytick.labelsize': 8,
#                 # 'axes.xmargin': 0,
#                 # 'axes.ymargin': 0,
#              }
with plt.rc_context(rc=pyplot_rc_tex):
    fig, ax = plt.subplots()
    ax.plot(t, s, marker='x', label='random values (idk)')
    ax.set(xlabel='some label (time)', ylabel='voltage (mV)')
    ax.legend(loc='lower left')

    plt.yticks(MTC_XTICKS[0], MTC_XTICKS[1])
    plt.xticks(MTC_YTICKS[0], MTC_YTICKS[0])
    # fig.xticks((0, 0.5, 1), (r'\bf{0}', r'\bf{.5}', r'\bf{1}'), color='k', size=20)

    # fig.savefig(f"test-{str(rc_params.values())}.pdf", backend='pgf')
    fig.savefig(f"test-{str(pyplot_rc_tex.values())}.png", dpi=200)
    print(plt.rc_context)
    plt.show()
