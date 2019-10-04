import cv2
from sklearn.externals import joblib
import numpy as np


model = joblib.load("digits.pkl")

cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")
while True:
    ret, frame = cap.read()

    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    grayFrame = cv2.GaussianBlur(grayFrame, (5, 5), 0)
    _, fr_th = cv2.threshold(grayFrame, 155, 255, cv2.THRESH_BINARY_INV)
    _, ctrs, _ = cv2.findContours(
        fr_th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects = [cv2.boundingRect(ctr) for ctr in ctrs]
    for rect in rects:
        # cat ra hinh anh moi so
        # dua ve hinh nah 28x28

        leng = int(rect[3]*1.6)
        pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
        pt2 = int(rect[0] + rect[2] // 2 - leng // 2)

        roi = fr_th[pt1:pt1 + leng, pt2:pt2 + leng]
        roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
        roi = cv2.dilate(roi, (3, 3))
        number = np.array([roi]).reshape(1, 28*28)
        predict = model.predict(number)

        print(predict)

    cv2.imshow('video', frame)

    c = cv2.waitKey(1)
    if c == 13:
        break

cap.release()
cv2.destroyAllWindows()
