from multiprocessing import Pool


def f(x):
    return x*x


if __name__ == '__main__':
    with Pool(5) as p:
        x = p.map(f, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])

    print(x)
