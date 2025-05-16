import cv2
import numpy as np
import os

def find_image_in_video(video_path, target_image_path, threshold=0.3):
    target = cv2.imread(target_image_path, cv2.IMREAD_GRAYSCALE)
    cap = cv2.VideoCapture(video_path)
    os.makedirs("./pictures/out", exist_ok=True)
    match_count = 0
    frame_idx = 0

    while cap.grab():
        _, frame = cap.retrieve()
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        result = cv2.matchTemplate(gray_frame, target, cv2.TM_CCOEFF_NORMED)
        if np.max(result) >= threshold:
            match_count += 1
            cv2.imwrite(f"./pictures/out/frame_{frame_idx:06d}.png", frame)
        frame_idx += 1

    cap.release()
    print(f"Целевое изображение найдено в {match_count} кадрах.")
    return match_count


find_image_in_video("./pictures/output.avi", "./pictures/origin.png")
