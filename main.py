import cv2
import numpy as np
import pandas as pd
import time
from collections import deque

import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks import python

# -------- MODEL PATH --------
model_path = "face_landmarker.task"

# -------- MEDIAPIPE SETUP --------
BaseOptions = python.BaseOptions
FaceLandmarker = vision.FaceLandmarker
FaceLandmarkerOptions = vision.FaceLandmarkerOptions
VisionRunningMode = vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO,
    num_faces=1
)

landmarker = FaceLandmarker.create_from_options(options)

# -------- EYE LANDMARKS --------
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

def eye_aspect_ratio(landmarks, eye_indices, w, h):
    pts = []
    for idx in eye_indices:
        x = int(landmarks[idx].x * w)
        y = int(landmarks[idx].y * h)
        pts.append((x, y))

    A = np.linalg.norm(np.array(pts[1]) - np.array(pts[5]))
    B = np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))
    C = np.linalg.norm(np.array(pts[0]) - np.array(pts[3]))

    if C == 0:
        return 0

    return (A + B) / (2.0 * C)

EAR_THRESHOLD = 0.22
BLINK_COOLDOWN = 0.3

blink_count = 0
last_blink_time = 0
blink_history = deque(maxlen=120)

data_log = []
start_time = time.time()

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot access webcam")
    exit()

timestamp = 0

print("Press ESC to exit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    result = landmarker.detect_for_video(mp_image, timestamp)
    timestamp += 1

    if result.face_landmarks:
        landmarks = result.face_landmarks[0]

        left_ear = eye_aspect_ratio(landmarks, LEFT_EYE, w, h)
        right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE, w, h)
        ear = (left_ear + right_ear) / 2

        current_time = time.time()

        if ear < EAR_THRESHOLD and (current_time - last_blink_time) > BLINK_COOLDOWN:
            blink_count += 1
            last_blink_time = current_time
            blink_history.append(1)
        else:
            blink_history.append(0)

        blink_rate = sum(blink_history)

        fatigue_score = (blink_rate * 0.6) + ((0.3 - ear) * 50)
        fatigue_score = np.clip(fatigue_score, 0, 100)

        if fatigue_score < 30:
            fatigue_label = "Low"
        elif fatigue_score < 60:
            fatigue_label = "Medium"
        else:
            fatigue_label = "High"

        data_log.append({
            "time": time.time() - start_time,
            "EAR": ear,
            "blink_rate": blink_rate,
            "fatigue_score": fatigue_score
        })

        cv2.putText(frame, f"EAR: {ear:.2f}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        cv2.putText(frame, f"Blinks: {blink_count}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

        cv2.putText(frame, f"Fatigue: {fatigue_label}", (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)

    cv2.imshow("Fatigue Monitor", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

df = pd.DataFrame(data_log)
df.to_csv("fatigue_data.csv", index=False)

print("Data saved")
