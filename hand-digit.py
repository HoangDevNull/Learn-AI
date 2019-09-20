import cv2

cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

while True:
    ret, frame = cap.read()
    # frame = cv2.resize(frame, None, fx=0.5, fy=0.5,
    #                    interpolation=cv2.INTER_AREA)

    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 

    grayFrame = cv2.GaussianBlur(grayFrame, (5, 5), 0)
    _, fr_th = cv2.threshold(grayFrame, 155, 255, cv2.THRESH_BINARY_INV)
    _, ctrs, _ = cv2.findContours(fr_th,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    rects = [cv2.boundingRect(ctr) for ctr in ctrs]
    for rect in rects:
        cv2.rectangle(frame, (rect[0], rect[1]),
                    (rect[0]+rect[2], rect[1]+rect[3]), (0, 0, 255), 3)

    cv2.imshow('video', frame)

    c = cv2.waitKey(1)
    if c == 13:
        break

cap.release()
cv2.destroyAllWindows()
