


if __name__ == '__main__':
    print(f'Testing the plot style')
    import matplotlib.pyplot as plt
    import numpy as np
    x = range(10)
    y = np.sin(np.arange(10)/10) + np.arange(10)/10
    with plt.rc_context(rc=pyplot_rc_two_column):
        fig, ax = plt.subplots()
        ax.plot(x, y, marker='x', label='random values (idk)')
        ax.set(xlabel='some label', ylabel='some other value')
        ax.legend(loc='lower left')
        plt.show()