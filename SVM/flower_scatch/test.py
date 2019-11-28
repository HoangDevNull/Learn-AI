## Doc lai du global_features va labels tu file da luu
## Chay mo hinh SVC

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import mahotas


from cvxopt import matrix, solvers

X = np.loadtxt("feature-datas.txt")
y = np.loadtxt("label-datas.txt")


model = SVC(kernel = 'linear', C = 1e5)
model.fit(X, y)


N = X.shape[0]
V = X.T * y

K = matrix(V.T.dot(V))
K = matrix(K, K.size, 'd')

p = matrix(-np.ones((N, 1)))
G = matrix(-np.eye(N))
h = matrix(np.zeros(N))
A = matrix(y.reshape(1, -1))
b = matrix(np.zeros(1)) 

solvers.options['show_progress'] = False
sol = solvers.qp(K, p, G, h, A, b)

l = np.array(sol['x'])
print(l.size, 'l size')
# print('lambda = ', l)


epsilon = 1e-6
S = np.where(l > epsilon)[0]
VS = V.T[:, S]
XS = X[:, S]
y = np.matrix(y)
yS = y[:, S]
lS = l[S]
# calculate w and b
w = VS.dot(lS)
b = np.mean(yS.T - w.T.dot(XS))

print('w = ', w.T)
print('b = ', b)


from sklearn.preprocessing import MinMaxScaler
bins = 8
# feature-descriptor-1: Hu Moments
def fd_hu_moments(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    return feature


# feature-descriptor-2: Haralick Texture
def fd_haralick(image):
    # convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # compute the haralick texture feature vector
    haralick = mahotas.features.haralick(gray).mean(axis=0)
    # return the result
    return haralick

# feature-descriptor-3: Color Histogram
def fd_histogram(image, mask=None):
    # convert the image to HSV color-space
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # compute the color histogram
    hist  = cv2.calcHist([image], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    # normalize the histogram
    cv2.normalize(hist, hist)
    # return the histogram
    return hist.flatten()


# predict data 

import cv2
import matplotlib.pyplot as plt

fixed_size       = tuple((500, 500))

image = cv2.imread('2.jpg')

# resize the image
image = cv2.resize(image, fixed_size)

    ####################################
    # Global Feature extraction
    ####################################
fv_hu_moments = fd_hu_moments(image)
fv_haralick   = fd_haralick(image)
fv_histogram  = fd_histogram(image)

    ###################################
    # Concatenate global features
    ###################################
global_feature = np.hstack([fv_histogram, fv_haralick, fv_hu_moments])

# scale features in the range (-1-1)
scaler = MinMaxScaler(feature_range=(-1, 1))
rescaled_feature = scaler.fit_transform(global_feature.reshape(1, -1))

x = rescaled_feature.reshape(1,-1)

print(x.shape, 'x shape ')
print(w.shape[0], 'wshape')

ws = 0
xs = 0

for i in range(0, w.shape[0] -1): 
    ws += (w[i] * w[i])
    xs += (w[i] * x[0][i])


mypredict = (xs + b) / np.sqrt(ws)
print(mypredict,'my predict')


w1 = model.coef_
b1 = model.intercept_
print('w1 = ', w1)
print('b1 = ', b1)
print(model.predict(global_feature.reshape(1,-1))[0], 'svc')