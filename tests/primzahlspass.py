import sympy
import matplotlib.pyplot as plt
import numpy as np
import sys
np.set_printoptions(precision=2, threshold=sys.maxsize, suppress=True)

def altering_sum_plus_minus():
    """
    Altering sum of primes; +2, -3, +5, -7
    """
    ps = list(sympy.primerange(0, 1000))
    altsum = 0
    p_altsum = []
    fuk = 1
    for p in ps:
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
    lul_remerge = [item for t in lul_remerge for item in t]
    print(lul_remerge)

    print(np.array(p_altsum_mean))

    p_diff = [abs(p_altsum_abs[x] - p_altsum_abs[x - 1]) for x in range(1, len(p_altsum_abs))]

    # print(p_altsum)
    # print(p_odd)
    # print(p_diff)
    # plt.plot([abs(x) for x in p_altsum])
    plt.plot([p_diff[x] for x in p_diff])
    plt.plot(p_altsum_mean)
    plt.plot(lul_remerge)
    # plt.plot(p_altsum[1::2])
    plt.figure()
    plt.show()


if __name__ == '__main__':
    altering_sum_plus_minus()
