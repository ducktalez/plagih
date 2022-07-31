import sympy
import matplotlib.pyplot as plt
import numpy as np
import sys

np.set_printoptions(precision=2, threshold=sys.maxsize, suppress=True)


def help_running_sum(alist):
    """
    With leading zero? nope
    """
    return [b-a for a, b in zip(alist[1:], alist[:-1])]


def binearspass():
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127]
    binprimes = [bin(x)[2:] for x in primes]
    primes2 = [[0, 1], [2, 3], [5, 7], [11, 13], [17, 19, 23, 29, 31], [37, 41, 43, 47, 53, 59, 61], [67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127]]
    bins2 = [[bin(x) for x in xs] for xs in primes2]
    bins_test = [int(x[:-1], 2) for x in binprimes]
    print(bins_test)


def altering_sum_plus_minus(pmax=1000):
    """
    Altering sum of primes; +2, -3, +5, -7
    """
    ppp = list(sympy.primerange(0, pmax))
    pp_count = [sympy.primepi(x) for x in range(pmax)]

    altsum = 0
    p_altsum = []
    fuk = 1
    for p in ppp:
        altsum += p * fuk
        fuk *= -1
        p_altsum.append(altsum)

    p_even = p_altsum[0::2]
    p_odd = p_altsum[1::2]

    p_altsum_abs = [abs(x) for x in p_altsum]
    p_altsum_mean = [sum(p_altsum[0:x]) / len(p_altsum[0:x]) for x in range(1, len(p_altsum))]

    altsum_even_mean = np.array([(sum(p_even[0:x][::2]) - sum(p_even[1:x][::2])) / len(p_even[0:x]) for x in range(1, len(p_even))])
    altsum_odd_mean = np.array([(sum(p_odd[0:x][::2]) - sum(p_odd[1:x][::2])) / len(p_odd[0:x]) for x in range(1, len(p_odd))])

    lul_remerge = [*zip(altsum_even_mean, altsum_odd_mean)]
    lul_remerge = np.array([item for t in lul_remerge for item in t])

    # print(np.array(p_altsum_mean))
    t = p_altsum_mean
    hmmmm = np.array([abs(abs(j) - abs(i)) for i, j in zip(t[:-1], t[1:])])
    p_diff = [abs(p_altsum_abs[x] - p_altsum_abs[x - 1]) for x in range(1, len(p_altsum_abs))]

    """another test"""
    asdasd = help_running_sum(ppp)
    # print(asdasd)

    # print(p_altsum)
    # print(p_odd)
    # print(p_diff)

    with plt.rc_context(rc={}):
        fig, ax = plt.subplots()
        # ax.plot(np.array(ppp[1:]), asdasd)
        print(p_altsum)
        ax.plot(p_altsum)
        # ax.plot([abs(x) for x in p_altsum])
        # ax.plot([p_diff[x] for x in p_diff])
        # ax.plot(p_diff)
        # # ax.plot(hmmmm)
        # ax.plot(lul_remerge)
        # plt.plot(p_altsum[1::2])
        ax.grid(True, linestyle='-.')
        plt.show()


def plottingspass(nmax=100):
    approx_prime_count = [(x/(sympy.log(x))) for x in range(2, nmax)]
    with plt.rc_context(rc={}):
        fig, ax = plt.subplots()
        ax.plot(approx_prime_count)
        ax.grid(True, linestyle='-.')
        plt.show()


if __name__ == '__main__':
    altering_sum_plus_minus()
