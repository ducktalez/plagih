import sympy
import matplotlib.pyplot as plt
import numpy as np
import sys

np.set_printoptions(precision=2, threshold=sys.maxsize, suppress=True)


def altering_sum_plus_minus(pmax=100):
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
    # print(lul_remerge)
    print(ppp)
    print(p_altsum_abs)

    # print(np.array(p_altsum_mean))
    t = p_altsum_mean
    hmmmm = np.array([abs(abs(j) - abs(i)) for i, j in zip(t[:-1], t[1:])])
    p_diff = [abs(p_altsum_abs[x] - p_altsum_abs[x - 1]) for x in range(1, len(p_altsum_abs))]

    # print(p_altsum)
    # print(p_odd)
    # print(p_diff)
    with plt.rc_context(rc={}):
        fig, ax = plt.subplots()
        # plt.plot([abs(x) for x in p_altsum])
        # plt.plot([p_diff[x] for x in p_diff])
        ax.step(range(pmax), pp_count)
        ax.step(range(pmax), range(pmax))
        ax.plot(p_altsum_mean)
        # ax.plot(hmmmm)
        # plt.plot(lul_remerge)
        # plt.plot(p_altsum[1::2])
        ax.grid(True, linestyle='-.')
        plt.show()


if __name__ == '__main__':
    altering_sum_plus_minus()
