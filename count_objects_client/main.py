import cv2
import numpy as np
import zmq

address = "84.237.21.36"
port = 6002

context = zmq.Context()
socket = context.socket(zmq.SUB)
socket.setsockopt(zmq.SUBSCRIBE, b"")
socket.connect(f"tcp://{address}:{port}")

cv2.namedWindow("Client", cv2.WINDOW_GUI_NORMAL)
count = 0


def is_cube(contour):
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = w / float(h)

    if 0.75 <= aspect_ratio <= 1.3 and w > 30 and h > 30:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
        return len(approx) == 4
    return False


def is_sphere(contour):
    (x, y), radius = cv2.minEnclosingCircle(contour)
    circle_area = np.pi * radius ** 2
    fill_ratio = area / circle_area if circle_area > 0 else 0
    return fill_ratio > 0.7

while True:
    message = socket.recv()
    frame = cv2.imdecode(np.frombuffer(message, np.uint8), -1)
    

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

 
    blurred = cv2.GaussianBlur(hsv, (7, 7), 0)


    s_channel = blurred[:, :, 1]
    v_channel = blurred[:, :, 2] 

 
    _, thresh = cv2.threshold(v_channel, 150, 255, cv2.THRESH_BINARY)



    kernel = np.ones((7, 7), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_ERODE, kernel, iterations=6)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, kernel, iterations=2)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    figure_count = 0
    cube_count = 0
    sphere_count = 0

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 300: 
            continue

        if is_cube(contour):
            cube_count += 1
            figure_count += 1
            cv2.drawContours(frame, [contour], -1, (0, 0, 255), 2)
        elif is_sphere(contour):
            sphere_count += 1
            figure_count += 1
            cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
        else:
            figure_count += 1
            cv2.drawContours(frame, [contour], -1, (255, 255, 0), 2)
    count += 1
    cv2.putText(frame, f"frame: {count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(frame, f"squares: {cube_count}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(frame, f"balls  : {sphere_count}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"figures  : {figure_count}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.imshow("Client", frame)

    key = chr(cv2.waitKey(1) & 0xFF)
    if key == "q":
        break

cv2.destroyAllWindows()
socket.close()
context.term()