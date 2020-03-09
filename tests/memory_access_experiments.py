"""
Python has 'weird' memory usage. Let me help you.
> a = 5
1. object 5 is created (if not already existing)
2. Reference to this object is created


"""

a = 1
b = a
c = b
print('a=1;b=1;c=2 ==>', a, b, c)
b = 2
print('a=1;b=1;c=2 ==>', a, b, c)

import sys

# Where is a stored?
a = 45
print(id(a))

# But wait- the 45 was already there
print('id(45) ==>', id(45))

# This is because 45 is already often referred to in the python core
print('sys.getrefcount(45) ==>', sys.getrefcount(45), '\n')

# Have a look how often 1 is referenced...
print('sys.getrefcount(1)  ==>', sys.getrefcount(1))

print('\033[36m\033[39m')

a = 0
print('b = a  ==>', sys.getrefcount(a))

# deepcopy does not change anything
import copy
b = copy.deepcopy(1)
print(id(1), id(copy.deepcopy(1)))

import numpy as np

np_a = np.array([1, 2, 3])
print('id array 1  ==>', id(np.array([1, 2, 3])), sys.getrefcount(np.array([1, 2, 3])))
print('id array 2  ==>', id(np.array([1, 2, 3])), sys.getrefcount(np.array([1, 2, 3])))
np_b = np.array([1, 2, 3])
print('id array 2  ==>', id(np_a), sys.getrefcount(np_a))
print('id array 3  ==>', id(np.array([1, 2, 3])), sys.getrefcount(np.array([1, 2, 3])))
#
# a = np.array([1, 2, 3])
# b = np.array([1, 2, 3])
# print(sys.getrefcount(a))



def print_stuff(a, b):
    print(sys.getrefcount(a), sys.getrefcount(b))
    print(id(a), '-', id(b), 'is a ref' if id(a) == id(b) else 'are separate values')


def test_2(x, y):
    print('\nTest 1')
    print(sys.getrefcount(x), sys.getrefcount(y))
    print(id(x), '-', id(y), 'is a ref' if id(x) == id(y) else 'are separate values')
    x = 1
    y = 2
    print('\nTest 2')
    print(sys.getrefcount(a), sys.getrefcount(y))
    print(id(x), '-', id(y), 'is a ref' if id(x) == id(y) else 'are separate values')


def test_1(a):
    print(sys.getrefcount(a))


test_1(45)

import time


class WhatsBetter:

    def __init__(self):
        pass

    def test_me(self):
        repeats = 10000000
        time_1 = time.perf_counter()
        self.x = 0
        self.fun_a(repeats)
        time_2 = time.perf_counter()
        x = 0
        x = self.fun_b(repeats, x)
        time_3 = time.perf_counter()
        print('Time: {:6.4f} vs. {:6.4f}'.format(time_2 - time_1, time_3 - time_2))

    def fun_a(self, repeats):
        for i in range(1, repeats):
            self.x = self.x + 1
        return

    def fun_b(self, repeats, x):
        for i in range(1, repeats):
            x = x + 1
        return x


test = WhatsBetter()
test.test_me()
