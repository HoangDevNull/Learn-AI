## Doc lai du global_features va labels tu file da luu
## Chay mo hinh SVC

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import mahotas

## Doc lai du global_features va labels tu file da luu
## Chay mo hinh SVC

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
import mahotas

from cvxopt import matrix, solvers

X = np.loadtxt("feature-datas.txt")
y = np.loadtxt("label-datas.txt")
N = X.shape[0]
y = y.reshape(-1,1) * 1.
X_dash = y * X
V = np.dot(X_dash , X_dash.T) * 1.



K = matrix(V)
p = matrix(-np.ones((N, 1)))
G = matrix(-np.eye(N))
h = matrix(np.zeros(N))
A = matrix(y.reshape(1, -1))
b = matrix(np.zeros(1))

solvers.options['show_progress'] = False
sol = solvers.qp(K, p, G, h, A, b)

l = np.array(sol['x'])

w = ((y * l).T @ X).reshape(-1,1)

S = (l > 1e-4).flatten()

b = y[S] - np.dot(X[S], w)

print('lambda = ',l[l > 1e-4])
print('w = ', w.flatten())
print('b = ', b[0])

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

image = cv2.imread('1.jpg')

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

# scale features in the range (0-1)
scaler = MinMaxScaler(feature_range=(0, 1))
rescaled_feature = scaler.fit_transform(global_feature.reshape(1, -1))


# predict label of test image
x = rescaled_feature.reshape(1,-1)

print(x)

mypredict = (w[0] * x[0] + w[1] * x[0] + b) / np.sqrt((w[0] * w[0] + w[1]*w[1]))

print('mypredict = ' , mypredict)


# label = 'hoa 1'

# if prediction==1.0:
#     label ='tulip'
# elif prediction==2.0:
#     label ='windflower'

# print (label)
 

# # show predicted label on image
# cv2.putText(image, str(label), (20,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 3)

# # display the output image
# plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
# plt.show()