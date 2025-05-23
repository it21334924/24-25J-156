from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import cv2
import tensorflow as tf
import numpy as np
import mediapipe as mp
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for communication with frontend

# Load pre-trained model
model = tf.keras.models.load_model('eye_fatigue_rnn_model.h5')

# Mediapipe initialization
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# Define eye landmarks
LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]

# Eye Aspect Ratio (EAR) calculation
def calculate_ear(landmarks, left_eye_indices, right_eye_indices):
    def aspect_ratio(eye):
        return np.linalg.norm(eye[1] - eye[5]) / (2 * np.linalg.norm(eye[0] - eye[3]))

    left_eye = np.array([[landmarks[i].x, landmarks[i].y] for i in left_eye_indices])
    right_eye = np.array([[landmarks[i].x, landmarks[i].y] for i in right_eye_indices])
    return aspect_ratio(left_eye), aspect_ratio(right_eye)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/eye_fatigue')
def eye_fatigue():
    return render_template('index.html')

@app.route('/color_test')
def color_test():
    return render_template('color_test.html')

@app.route('/analyze_frame', methods=['POST'])
def analyze_frame():
    try:
        file = request.files['frame'].read()
        nparr = np.frombuffer(file, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            left_ear, right_ear = calculate_ear(landmarks, LEFT_EYE_INDICES, RIGHT_EYE_INDICES)
            avg_ear = (left_ear + right_ear) / 2
            
           
            
            return jsonify({
                'avg_ear': float(avg_ear),
                'left_ear': float(left_ear),
                'right_ear': float(right_ear)
            })
        else:
            return jsonify({'error': 'No face detected'}), 400

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)