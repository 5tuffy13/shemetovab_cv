import cv2


image = cv2.imread("./contours/arrow.png")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
arrow = contours[0]

cv2.drawContours(image, contours, 0, (255, 0, 0), 3)

print(f"area = {cv2.contourArea(arrow)}")
print(f"perimetr = {cv2.arcLength(arrow, False)}")

moments = cv2.moments(arrow)
print(moments)

centroid = (int(moments['m10'] / moments['m00']), int(moments['m01'] / moments['m00']))

cv2.circle(image, centroid, 4, (0, 255, 0), 4)

eps = 0.001 * cv2.arcLength(arrow, True)
approx = cv2.approxPolyDP(arrow, eps, True)
for p in approx:
    cv2.circle(image, tuple(*p), 6, (0, 255, 0), 0)

hull = cv2.convexHull(arrow)
for i in range(1, len(hull)):
    cv2.line(image, tuple(*hull[i - 1]), tuple(*hull[i]), (0, 255, 255), 2)
cv2.line(image, tuple(*hull[- 1]), tuple(*hull[i]), (0, 255, 255), 2)

x,y,w,h = cv2.boundingRect(arrow)
cv2.rectangle(image,(x,y),(x+w,y+h), (0,255,0), 2)

rect = cv2.minAreaRect(arrow)
box = cv2.boxPoints(rect)
import numpy as np
box = box.astype("int32")
cv2.drawContours(image,[box],0,(0,255,0),2)

cv2.namedWindow("Arrow", cv2.WINDOW_GUI_NORMAL)
cv2.imshow("Arrow", image)
cv2.waitKey()
cv2.destroyAllWindows()
