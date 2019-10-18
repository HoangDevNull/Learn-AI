from __future__ import division
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation

import math
import numpy as np
import matplotlib.pyplot as plt


def cost1a(x):
    return x**3 - 2*(x**2) + 3*x + 4

# min x -> infinitive
def grad1a(x):
    return 3*(x**2) - 4*x + 3


def cost(x):
    return np.sin(x) - np.cos(x) + np.tan(x)


def grad(x):
    return np.cos(x) + np.sin(x) + 1/(np.cos(x))**2


def cost1c(x):
    return x**4 + 2*(x**(0.5))


def grad1c(x):
    return 4*(x**3) + 1/(x**(0.5))


def cost1d(x):
    return (np.cos(x)/np.sin(x)) - 3*x + 2


def grad1d(x):
    return -1/(np.sin(x))**2 - 3


def cost2a(x):
    return -x**3 - 3*x**2 - 4*x + 1

# infinity


def grad2a(x):
    return -3*x**2 - 6*x - 4


def cost2b(x):
    return np.sin(2*x) + np.cos(x)


def grad2b(x):
    return 2*np.cos(2*x) - np.sin(x)


def cost2c(x):
    return np.sqrt(x) - 2*x


def grad2c(x):
    return 1/(2*np.sqrt(x)) - 2


def myGD1(x0, eta=0.1):
    x = [x0]
    for it in range(100):
        x_new = x[-1] - eta*grad(x[-1])
        if abs(grad(x_new)) < 1e-3:
            break
        x.append(x_new)
    x = np.asarray(x)
    return (x, it)


def GD_newton(x0):
    x = [x0]
    for it in range(100):
        if abs(cost(x[-1])) < 1e-6 or abs(grad(x[-1])) < 1e-6:
            break
        x_new = x[-1] - 3*cost(x[-1])/grad(x[-1])
        print(x_new, cost(x[-1]), grad(x[-1]))
        x.append(x_new)
    return (x, it)


def Momentum(x0, eta=0.1, gamma=0.9):
    v = [0]
    x = [x0]
    for it in range(100):
        print(x[-1])
        g = grad(x[-1])
        if abs(g) < 1e-3:
            break
        v_new = gamma*v[-1] + eta*g
        x_new = x[-1] - v_new
        v.append(v_new)
        x.append(x_new)
    return (np.asarray(x), v, it)


def GD_NAG(x0, eta=0.1, gamma=0.9):
    x = [x0]
    v = [0]
    for it in range(100):
        v_new = gamma*v[-1] + eta*grad(x[-1] - gamma*v[-1])
        x_new = x[-1] - v_new
        print(x_new)
        if abs(grad(x[-1])) < 1e-6:
            break
        x.append(x_new)
        v.append(v_new)

    return (np.asarray(x), v, it)

# plt chart show


def plot_fn(fn, xmin=-5, xmax=5, xaxis=True, opts='b-'):
    x = np.linspace(xmin, xmax, 1000)
    y = fn(x)
    ymin = np.min(y) - .5;
    ymax = np.max(y) + .5
    plt.axis([xmin, xmax, ymin, ymax])
    if xaxis:
        x0 = np.linspace(xmin, xmax, 2)
        plt.plot([xmin, xmax], [0, 0], 'k')
    plt.plot(x, y, opts)


# Animation


def viz_alg_1d_2(x, cost, filename='nomomentum1d.gif'):
    #     x = x.asarray()
    it = len(x)
    y = cost(x)
    xmin, xmax = np.min(x), np.max(x)
    ymin, ymax = np.min(y), np.max(y)
    xmin, xmax = -4, 6
    ymin, ymax = -12, 25
    x0 = np.linspace(xmin-1, xmax+1, 1000)
    y0 = cost(x0)

    fig, ax = plt.subplots(figsize=(4, 4))

    def update(i):
        ani = plt.cla()
        plt.axis([-4, 6, -13, 26])
        plt.plot(x0, y0)
        plt.axis([xmin, xmax, ymin, ymax])
        ani = plt.title('$f(x) =sin(x) - cos(x) + tan(X); \eta = 0.1$')
        if i == 0:
            ani = plt.plot(x[i], y[i], 'ro', markersize=7)
        else:
            ani = plt.plot(x[i-1], y[i-1], 'ok', markersize=7)
            ani = plt.plot(x[i-1:i+1], y[i-1:i+1], 'k-')
            ani = plt.plot(x[i], y[i], 'ro', markersize=7)
        label = 'GD with NAG: iter %d/%d' % (i, it)
        ax.set_xlabel(label)
        return ani, ax

    anim = FuncAnimation(fig, update, frames=np.arange(0, it), interval=200)
    anim.save(filename, dpi=100, writer='imagemagick')
    plt.show()

# x = np.asarray(x)
# (x, it) = myGD1(5, 0.1)


# (x, v, it) = Momentum(10, 0.1, .9)
(x, v, it) = GD_NAG(2, 0.01, 0.9)
viz_alg_1d_2(x, cost)


# plt.show()
