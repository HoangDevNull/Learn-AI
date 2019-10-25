import cv2
from sklearn.externals import joblib
import numpy as np

thetas = np.loadtxt('./build-from-scratch/v3/theta.txt')

image = cv2.imread("./numbers.jpg")

font = cv2.FONT_HERSHEY_SIMPLEX


def sigmoid(s):
    return 1/(1 + np.exp(-s))

# Check if the webcam is opened correctly
grayImg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

grayImg = cv2.GaussianBlur(grayImg, (5, 5), 0)
_, fr_th = cv2.threshold(grayImg, 155, 255, cv2.THRESH_BINARY_INV)
_, ctrs, _ = cv2.findContours(
    fr_th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
rects = [cv2.boundingRect(ctr) for ctr in ctrs]
for rect in rects:
        # cat ra hinh anh moi so
        # dua ve hinh nah 28x28
    cv2.rectangle(image, (rect[0], rect[1]),
                  (rect[0]+rect[2], rect[1]+rect[3]), (0, 0, 255), 3)

    leng = int(rect[3]*1.6)
    pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
    pt2 = int(rect[0] + rect[2] // 2 - leng // 2)

    roi = fr_th[pt1:pt1 + leng, pt2:pt2 + leng]
    roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
    roi = cv2.dilate(roi, (3, 3))
    number = np.array([roi]).reshape(1, 28*28)

    one = np.ones((number.shape[0], 1))
    number = np.concatenate((number,one), axis = 1)
    predict = sigmoid(np.dot(number, thetas.T))
    print('prediction',str(int(predict[0])))

    

    image = cv2.putText(image, str(int(predict[0])),
                        (rect[0], rect[1]),
                        font,
                        1,
                        (0, 0, 255),
                        2)

cv2.imshow('image', image)

c = cv2.waitKey(0)

cv2.destroyAllWindows()
