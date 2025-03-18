# eye_fatigue_live.py
import cv2
import tensorflow as tf
import numpy as np
import mediapipe as mp
# import os
from tensorflow.keras.models import load_model

# os.environ['TF_ENABLE_ONEDNN_OPTS']=0

# Load Pre-trained Model
model = load_model('eye_fatigue_model.h5')

# Parameters for live analysis
IMG_SIZE = 128
BLINK_THRESHOLD = 0.2  # For eye aspect rati
CLOSURE_DURATION_THRESHOLD = 5  # Seconds of eye closure for fatigue detection

# Mediapipe Initialization for Face and Eye Detection
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
drawing_utils = mp.solutions.drawing_utils

# Function to Calculate Eye Aspect Ratio (EAR)
def calculate_ear(landmarks, left_eye_indices, right_eye_indices):
    def aspect_ratio(eye):
        return np.linalg.norm(eye[1] - eye[5]) / (2 * np.linalg.norm(eye[0] - eye[3]))
    
    left_eye = np.array([[landmarks[i].x, landmarks[i].y] for i in left_eye_indices])
    right_eye = np.array([[landmarks[i].x, landmarks[i].y] for i in right_eye_indices])
    return aspect_ratio(left_eye), aspect_ratio(right_eye)

# Define Left and Right Eye Indices
LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]

# Start Webcam Capture
cap = cv2.VideoCapture(0)

# Variables to Track Eye Closure Duration
closure_start_time = None

print("Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Flip frame for a mirror view
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Detect facial landmarks
    results = face_mesh.process(rgb_frame)
    if results.multi_face_landmarks:
        for landmarks in results.multi_face_landmarks:
            drawing_utils.draw_landmarks(
                frame, landmarks, mp_face_mesh.FACEMESH_CONTOURS
            )
            
            # Calculate EAR for both eyes
            landmarks_list = landmarks.landmark
            left_ear, right_ear = calculate_ear(landmarks_list, LEFT_EYE_INDICES, RIGHT_EYE_INDICES)
            avg_ear = (left_ear + right_ear) / 2
            
            # Check if eyes are closed
            if avg_ear < BLINK_THRESHOLD:
                if closure_start_time is None:
                    closure_start_time = cv2.getTickCount()
            else:
                closure_start_time = None
            
            # Calculate Closure Duration
            if closure_start_time is not None:
                closure_duration = (cv2.getTickCount() - closure_start_time) / cv2.getTickFrequency()
                if closure_duration > CLOSURE_DURATION_THRESHOLD:
                    cv2.putText(frame, "FATIGUE DETECTED: !", (50, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                closure_duration = 0
            
            # # Show EAR on the frame
            # cv2.putText(frame, f"EAR: {avg_ear:.2f}", (50, 50),
            #             cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    
    # Display frame
    cv2.imshow('Eye Fatigue Detection', frame)
    
    # Quit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release Resources
cap.release()
cv2.destroyAllWindows()
