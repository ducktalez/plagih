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

# import matplotlib.pyplot as plt
# import numpy as np
#
# from plagih.file_interaction import *
#
# t = np.arange(-1.2, 0.6, 0.1)
# s = 0.1 * np.sin(2 * np.pi * t)
#
#
# with plt.rc_context(rc=pyplot_rc_tex):
#     fig, ax = plt.subplots()
#     ax.plot(t, s, marker='x', label='random values (idk)')
#     ax.set(xlabel='some label (time)', ylabel='voltage (mV)')
#     ax.legend(loc='lower left')
#
#     # plt.yticks(MTC_XTICKS[0], MTC_XTICKS[1])
#     # plt.xticks(MTC_YTICKS[0], MTC_YTICKS[0])
#     # fig.xticks((0, 0.5, 1), (r'\bf{0}', r'\bf{.5}', r'\bf{1}'), color='k', size=20)
#
#     # fig.savefig(f"test-{str(rc_params.values())}.pdf", backend='pgf')
#     fig.savefig(f"test-{str(pyplot_rc_tex.values())}.png")
#     print(plt.rc_context)
#     plt.show()

import multiprocessing as mp
import numpy as np
import copy
import random
import time


class Nu:

    def __init__(self, a, x):
        self.a = a
        self.x = x

    def e1(self):
        self.x += 50

    def e2(self):
        for j in range(100):
            self.x += j


def mp_dummy(arg, **kwarg):
    return Nu.e2(arg, **kwarg)


class Huehue(object):

    def __init__(self):

        self.xlist = [Nu('a', 1), Nu('b', 2), Nu('c', 3), Nu('d', 4), Nu('e', 7), Nu('f', 8), Nu('g', 9), Nu('h', 11), Nu('i', 23)]

    def choose_one(self):
        x = random.choice(self.xlist)
        xnew = copy.deepcopy(x)
        return xnew

    def evolve_list(self, n):
        evolve_list = [self.choose_one() for _ in range(n)]
        return evolve_list

    def new_p(self, n=10, v=1):
        new = []

        if v == 1:
            for _ in range(n):
                x = self.choose_one()
                x.e2()
                new.append(x)
        else:
            evolve_list = self.evolve_list(n)
            if v == 2:
                for x in evolve_list:
                    x.e2()
                    new.append(x)

            elif v == 3:
                for x in evolve_list:
                    x = mp_dummy(x)
                    new.append(x)
            else:
                mp.Process()
                with mp.Pool(min(mp.cpu_count(), 8)) as p:
                    if v == 4:
                        new = p.starmap_async(mp_dummy, evolve_list)

                    elif v == 5:
                        new = p.map_async(mp_dummy, evolve_list)

                    elif v == 6:
                        new = p.map_async(mp_dummy, evolve_list)

                    else:
                        raise
        self.xlist = new
        # try:
        #     print([[nu.a, nu.x] for nu in new[998:1000]])
        # except:
        #     pass

    def mp_dummy(self, arg, **kwarg):
        return Nu.e2(arg, kwarg)

    def mp_new_x(self, x):
        x = x.e2()
        return x


# if __name__ == '__main__':
#     # t = [time.perf_counter()]
#     for v in [1, 2, 3, 4, 5]:
#         hue = Huehue()
#         t0 = time.perf_counter()
#         hue.new_p(n=10000, v=v)
#         t1 = time.perf_counter()
#         print(f'{v} took: {t1- t0:.4f}')
#
#     # print(f'time : \nnew_a {t[2] - t[1]:.4f}\nnew_b {t[3] - t[2]:.4f}\nnew_c {t[4] - t[3]:.4f}\nall: {t}')

from multiprocessing import Process, Queue

def f(q):
    q.put([42, None, 'hello'])

if __name__ == '__main__':
    q = Queue()
    p = Process(target=f, args=(q,))
    p.start()
    print(q.get())    # prints "[42, None, 'hello']"
    p.join()
