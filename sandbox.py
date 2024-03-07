import numpy as np

T = np.zeros((4, 4, 4))
print(T.shape == (4, 4, 4))

print(T.size)
print(T[np.unravel_index(6, T.shape)])

class Foo:
    def __init__(self):
        self.gradients = []

    def bar(self):
        print('hey')


foo = Foo()
print(hasattr(foo, 'bar'))

print(np.log(1))