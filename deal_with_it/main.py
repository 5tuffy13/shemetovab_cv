import cv2
import numpy as np

cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)

capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)
capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
capture.set(cv2.CAP_PROP_EXPOSURE, 400)


# Загрузка изображения с очками
glasses = cv2.imread("deal_with_it/deal-with-it.png")



def censor (image, size=(7,7)):
    result = np.zeros_like(image)
    step_y = result.shape[0] // size[0]
    step_x = result.shape[1] // size[1]      
    for y in range(0, image.shape[0], step_y):
        for x in range(0,image.shape[1],step_x):
            result[y:y+step_y, x:x+step_x] = np.mean(image[y:y+step_y, x:x+step_x])
    return result

def glassing(frame, eye1, eye2, glasses_img):
    pass

points = [(0,0), (0,0)]


face_cascade = cv2.CascadeClassifier("./deal_with_it/haarcascade-frontalface-default.xml")
eye_cascade = cv2.CascadeClassifier("./deal_with_it/haarcascade-eye.xml")

while capture.isOpened():
    ret, frame = capture.read()
    blurred = cv2.GaussianBlur(frame, (7,7), 0)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

    faces = eye_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=10)
    
    # for (x, y, w, h) in faces:
    #     cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    #     new_w = int(w*1.5)
    #     new_h = int(h*1.5)
    #     x -= (new_w - w) // 2
    #     y -= (new_h - h) // 2
    #     roi = frame[y:y+new_h,x:x+new_w]
    #     roi = frame[y:y*2,x:x*2]
    #     try:
    #         censored = censor(roi)
    #         glassed = glassing(roi)
    #         frame[y:y+new_h,x:x+new_w] = glassed # or censored
    #         frame[y:y*2,x:x*2] = glassed
    #     except ValueError as e:
    #         pass
    key = chr(cv2.waitKey(1) & 0xFF)
    if key == "q":
        break
    cv2.imshow("Camera", frame)

capture.release()
cv2.destroyAllWindows()



