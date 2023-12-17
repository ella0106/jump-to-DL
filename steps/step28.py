if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from dezero import *
from dezero.utils import plot_dot_graph
import math

def rosenbrock(x0, x1):
    y = 100 * (x1 - x0 ** 2) ** 2 + (1 - x0) ** 2
    return y

x0 = Variable(np.array(0.0))
x1 = Variable(np.array(2.0))
y = rosenbrock(x0, x1)
lr = 0.001
iters = 10000

for i in range(iters):
    print(x0, x1)

    y = rosenbrock(x0 ,x1)

    x0.cleargrad()
    x1.cleargrad()
    y.backward()

    x0.data -= lr * x0.grad
    x1.data -= lr * x1.grad

x0.name = 'x0'
x1.name = 'x1'
y.name = 'y'

print(x0.grad)
print(x1.grad)
# plot_dot_graph(y, verbose=True, to_file='sin.png')