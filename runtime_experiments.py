import time

class WhatsBetter:

    def __init__(self, repeats):

        a = time.perf_counter()
        self.x = 0
        self.a_rw(repeats)
        b = time.perf_counter()
        self.x = 0
        self.b_rw(repeats, self.x)
        c = time.perf_counter()
        print('Read+Write: \tinstance: {:.7f}s \tlocal: {:.7f}s \tbetter x{:.2f}'.format(b-a, c-b, (b-a)/(c-b)))

        a = time.perf_counter()
        self.x = 0
        self.a_r(repeats)
        b = time.perf_counter()
        self.x = 0
        self.b_r(repeats, self.x)
        c = time.perf_counter()
        print('Read      : \tinstance: {:.7f}s \tlocal: {:.7f}s \tbetter x{:.2f}'.format(b-a, c-b, (b-a)/(c-b)))

        a = time.perf_counter()
        self.x = 0
        self.a_w(repeats)
        b = time.perf_counter()
        self.x = 0
        self.b_w(repeats, self.x)
        c = time.perf_counter()
        print('     Write: \tinstance: {:.7f}s \tlocal: {:.7f}s \tbetter x{:.2f}'.format(b-a, c-b, (b-a)/(c-b)))
        print('\n')

    def a_rw(self, repeats):
        for i in range(1, repeats):
            self.x += 1

    def b_rw(self, repeats, x):
        for i in range(1, repeats):
            x += 1
        return x

    def a_r(self, repeats):
        for i in range(1, repeats):
            _ = self.x

    def b_r(self, repeats, x):
        for i in range(1, repeats):
            _ = x
    def a_w(self, repeats):
        for i in range(1, repeats):
            self.x = 1

    def b_w(self, repeats, x):
        for i in range(1, repeats):
            x = 1
        return x

WhatsBetter(100)
WhatsBetter(10000000)
WhatsBetter(10000000)
