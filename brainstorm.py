import multiprocessing as mp
import numpy as np

funa = lambda a: a * 2
funb = lambda a: a + 5

alist = [1, 2, 3, 4, 5, 6, 7, 8]
queue_list = [funa, funa, funa, funa, funb, funb, funb, funb]


def fun(f):
    x = np.random.choice(alist)
    return f(x)


if __name__ == '__main__':
    # x = list(map(fun, queue_list))
    # print(x)
    with mp.Pool(processes=4) as pool:
        lel = iter(queue_list)

        L = pool.map_async(fun, iter(queue_list))

    print(L)
