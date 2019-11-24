## Doc lai du global_features va labels tu file da luu
## Chay mo hinh SVC

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
import mahotas


from cvxopt import matrix, solvers
COLORS = ['red', 'blue']

X = np.loadtxt("feature-datas.txt")
y = np.loadtxt("label-datas.txt")

N = X.shape[0]
X0 = X[0:80, :2]  # we only take the Sepal two features.
X1 = X[80:160, :2]
y = np.reshape(y,(-1,N))


# build K
V = np.concatenate((X0.T, -X1.T), axis = 1)
K = matrix(V.T.dot(V).astype('d')) # see definition of V, K near eq (8)
p = matrix(-np.ones((2*N, 1)).astype('d')) # all-one vector
# build A, b, G, h
G = matrix(-np.eye(2*N).astype('d')) # for all lambda_n >= 0
h = matrix(np.zeros((2*N, 1)).astype('d'))
A = matrix(y) # the equality constrain is actually y^T lambda = 0
b = matrix(np.zeros((1, 1)).astype('d'))

print(K.typecode,'K')
print(p.typecode,'p')
print(G.typecode,'G')
print(h.typecode,'h')
print(A.typecode,'A')
print(b.typecode,'b')


solvers.options['show_progress'] = False
sol = solvers.qp(K, p, G, h, A, b)

l = np.array(sol['x'])

epsilon = 1e-6 # just a small number, greater than 1e-9
S = np.where(l > epsilon)[0]

VS = V[:, S]
XS = X[:, S]
yS = y[:,S]
lS = l[S]

# calculate w and b
w = VS.dot(lS)
bias = np.mean(yS - w.T.dot(XS))

# w = np.sum(l * y[:, None] * X, axis = 0)

# cond = (l > 1e-4).reshape(-1)
# b = y[cond] - np.dot(X[cond], w)
# bias = b[0]

print('w = ', w.T)
print('b = ', bias)


