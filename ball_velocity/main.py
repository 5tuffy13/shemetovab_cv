import cv2
import numpy as np
import time
cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)
points = [(0,0), (0,0)]
capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)
capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
capture.set(cv2.CAP_PROP_EXPOSURE, 150)

hsv_color = np.array([60, 165, 150])
lower = np.array([40, 40, 40])
upper = np.array([100, 255, 255])
prev_time = time.time()
while capture.isOpened():
    ret, frame = capture.read()
    curr_time = time.time()
    blurred = cv2.GaussianBlur(frame,(7,7),0)
    hsv = cv2.cvtColor (blurred, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    speed = 0
    if len(contours) > 0:
        contour = max(contours, key=cv2.contourArea)
        (x,y), radius = cv2.minEnclosingCircle(contour)
        x,y, radius = int(x), int(y), int(radius)
        points.append((x,y))
        cv2.circle(frame, (x,y), radius, (0,255,0),2)
        if len(points) > 20:
            points.pop(0)
        dt = curr_time - prev_time
        prev_time = curr_time
        p1 = points[-1]
        p2 = points[-2]
        ds = ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)**0.5
        pxl2m = radius/0.04
        speed = ds / pxl2m / dt
    cv2.putText(frame,f"Speed = {speed:.2f} m/s",(10,30),cv2.FONT_HERSHEY_SIMPLEX,1.0,(255,0,0))
    for i, p in enumerate(points):  
        cv2.circle(frame,p,(i+1)*2, (0,255 * ((i+1)/20),0),2)




    cv2.imshow("Mask", mask)
    key = chr(cv2.waitKey(1) & 0xFF)
    if key == "q":
        break

    cv2.imshow("Camera", frame)

capture.release()
cv2.destroyAllWindows()