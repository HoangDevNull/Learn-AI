import cv2
import numpy as np

# Vietnamese 
# Cam thay` quy rung lac qua'. @@. x no' nhay~ lien tuc

cap = cv2.VideoCapture('pingping.mp4')

rightPoint, leftPoint = 0, 0
aMove, bMove = False, False

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

middleX = int(w/2)

font = cv2.FONT_HERSHEY_SIMPLEX

while(True):
    ret, frame = cap.read()
    blur = cv2.GaussianBlur(frame, (11, 11), 0)

    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    lower = np.array([25, 50, 50])
    upper = np.array([35, 255, 255])

    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    image, contours, hierarchy = cv2.findContours(
        mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    frame = cv2.line(frame, (middleX, 0), (middleX, h), (0, 0, 255), 2)

    
    if(len(contours) > 0):
        cnt = contours[0]
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        center = (int(x), int(y))
        radius = int(radius)

        frame = cv2.circle(frame, center, radius, (0, 0, 255), 2)
        
        # bMove true => ball move from A to B
        # bMove false => ball move from B to A
        # last second of ball move, we got x = 468 may be we can get the middle of table is 450 
        # But i'll choose the middle of screen to be the point which ball can move over and make player can increase our point
        if(x < middleX):
            if(bMove == False):
                rightPoint = rightPoint + 1
                aMove = False
                bMove = True
                print(leftPoint, rightPoint, sep=" --- ")

        if(x > middleX):
            if(bMove == True):
                leftPoint = leftPoint + 1
                bMove = False
                aMove = True
                print(leftPoint, rightPoint, sep=" --- ")

    point = repr(leftPoint) + "/" + repr(rightPoint)

    frame = cv2.putText(frame, point,
                        (10, 30),
                        font,
                        1,
                        (0, 0, 255),
                        2)
    cv2.imshow('mask', frame)

    # wait for enter key to break
    c = cv2.waitKey(1)
    if c == 13:
        break

cap.release()
cv2.destroyAllWindows()
