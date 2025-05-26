import cv2
import tensorflow as tf
import os
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

# Load model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "utils", "eye_fatigue_model.h5")
model = tf.keras.models.load_model(MODEL_PATH)

# Constants
IMG_SIZE = 128
BLINK_THRESHOLD = 0.2
CLOSURE_DURATION_THRESHOLD = 5

# Mediapipe setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
drawing_utils = mp.solutions.drawing_utils

LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]

def calculate_ear(landmarks, left_eye_indices, right_eye_indices):
    def aspect_ratio(eye):
        return np.linalg.norm(eye[1] - eye[5]) / (2 * np.linalg.norm(eye[0] - eye[3]))
    left_eye = np.array([[landmarks[i].x, landmarks[i].y] for i in left_eye_indices])
    right_eye = np.array([[landmarks[i].x, landmarks[i].y] for i in right_eye_indices])
    return aspect_ratio(left_eye), aspect_ratio(right_eye)

def crop_eye_image(frame, landmarks, indices):
    h, w = frame.shape[:2]
    points = np.array([[int(landmarks[i].x * w), int(landmarks[i].y * h)] for i in indices])
    x, y, w_eye, h_eye = cv2.boundingRect(points)
    eye = frame[y:y+h_eye, x:x+w_eye]
    if eye.size == 0:
        return None
    eye = cv2.resize(eye, (IMG_SIZE, IMG_SIZE))
    eye = eye / 255.0
    return eye

cap = cv2.VideoCapture(0)
closure_start_time = None

print("Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            drawing_utils.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS)
            landmarks = face_landmarks.landmark
            left_ear, right_ear = calculate_ear(landmarks, LEFT_EYE_INDICES, RIGHT_EYE_INDICES)
            avg_ear = (left_ear + right_ear) / 2

            # Crop and prepare eye image for model
            left_eye_img = crop_eye_image(frame, landmarks, LEFT_EYE_INDICES)
            right_eye_img = crop_eye_image(frame, landmarks, RIGHT_EYE_INDICES)

            if left_eye_img is not None and right_eye_img is not None:
                combined = np.hstack((left_eye_img, right_eye_img))  # side-by-side
                combined = cv2.resize(combined, (IMG_SIZE, IMG_SIZE))
                input_img = np.expand_dims(combined, axis=0)  # [1, 128, 128, 3]
                pred = model.predict(input_img)[0][0]

                label = "FATIGUED" if pred > 0.5 else "ALERT"
                color = (0, 0, 255) if pred > 0.5 else (0, 255, 0)
                cv2.putText(frame, f"Model: {label} ({pred:.2f})", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            # EAR-based closure duration check
            if avg_ear < BLINK_THRESHOLD:
                if closure_start_time is None:
                    closure_start_time = cv2.getTickCount()
            else:
                closure_start_time = None

            if closure_start_time is not None:
                duration = (cv2.getTickCount() - closure_start_time) / cv2.getTickFrequency()
                if duration > CLOSURE_DURATION_THRESHOLD:
                    cv2.putText(frame, "EYES CLOSED TOO LONG!", (30, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Eye Fatigue Live", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
