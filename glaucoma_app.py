import cv2
import numpy as np
from flask import Flask, Response, render_template, request, jsonify
from webcam_detection import PupilDetector, RealTimeGlaucomaDetection
import os
from werkzeug.utils import secure_filename
import math

app = Flask(__name__)

# Initialize detector and state
detector = RealTimeGlaucomaDetection(model_path='24-25J-156/models/best_glaucoma_model.keras')
detection_active = False
last_features = {'pupil_radius': 0, 'pupil_area': 0, 'pupil_circularity': 0}
uploaded_video_path = None
recent_frames = []  # Buffer for recent frames with detection data
questionnaire_data = None
MAX_FRAMES = 10  # Store up to 10 recent frames

def generate_frames(source):
    global detection_active, last_features, uploaded_video_path, recent_frames
    if detection_active:
        return
    
    detection_active = True
    recent_frames = []  # Reset frame buffer
    if isinstance(source, int):
        cap = cv2.VideoCapture(source)  # webcam
    else:
        cap = cv2.VideoCapture(source)  # video file
    
    if not cap.isOpened():
        print(f"Error: Could not open source: {source}")
        detection_active = False
        return
    
    while detection_active:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Resize full frame to 720x480
        frame = cv2.resize(frame, (720, 480))
        
        # Detect eyes
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        eyes = detector.pupil_detector.eye_cascade.detectMultiScale(gray, 1.3, 5)
        frame_data = {'frame': frame.copy(), 'prob': 0.0, 'pupil_detected': False}
        
        if len(eyes) == 0:
            cv2.putText(frame, "No eyes detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        for (ex, ey, ew, eh) in eyes:
            eye_frame = frame[ey:ey+eh, ex:ex+ew]
            if eye_frame.size == 0:
                continue
            processed_eye, result, prob, features = detector.analyze_eye_frame(eye_frame)
            frame[ey:ey+eh, ex:ex+ew] = processed_eye
            cv2.rectangle(frame, (ex, ey), (ex+ew, ey+eh), (255, 0, 0), 2)
            if features['pupil_detected']:
                pupil_info = f"Pupil R: {features['pupil_radius']} {'Glaucoma' if prob > 0.5 else 'Healthy'} ({prob:.2f})"
                cv2.putText(frame, pupil_info, (ex, ey-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                last_features = {
                    'pupil_radius': features['pupil_radius'],
                    'pupil_area': features['pupil_area'],
                    'pupil_circularity': round(features['circularity'], 3) if 'circularity' in features else 0
                }
                frame_data['prob'] = prob
                frame_data['pupil_detected'] = True
                frame_data['features'] = last_features.copy()
        
        # Store frame in buffer
        if frame_data['pupil_detected']:
            recent_frames.append(frame_data)
            if len(recent_frames) > MAX_FRAMES:
                recent_frames.pop(0)
        
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    cap.release()
    detection_active = False
    if uploaded_video_path and os.path.exists(uploaded_video_path):
        os.remove(uploaded_video_path)
        uploaded_video_path = None

@app.route('/')
def index():
    return render_template('gDetection_page.html')

@app.route('/video_feed')
def video_feed():
    global detection_active, uploaded_video_path
    if detection_active:
        return jsonify({'error': 'Another detection is already active'}), 400
    source_type = request.args.get('type', 'webcam')
    if source_type == 'webcam':
        source = 0
    elif source_type == 'uploaded' and uploaded_video_path:
        source = uploaded_video_path
    else:
        return jsonify({'error': 'Invalid source type'}), 400
    return Response(generate_frames(source), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop_feed', methods=['POST'])
def stop_feed():
    global detection_active, uploaded_video_path, recent_frames
    detection_active = False
    recent_frames = []
    if uploaded_video_path and os.path.exists(uploaded_video_path):
        os.remove(uploaded_video_path)
        uploaded_video_path = None
    return jsonify({'status': 'stopped'})

@app.route('/pupil_dynamics')
def pupil_dynamics():
    global last_features
    return jsonify(last_features)

@app.route('/analyze_video', methods=['POST'])
def analyze_video():
    global detection_active, uploaded_video_path
    if detection_active:
        return jsonify({'error': 'Another detection is already active'}), 400
    
    file = request.files['video']
    filename = secure_filename(file.filename)
    uploaded_video_path = os.path.join('uploads', filename)
    os.makedirs('uploads', exist_ok=True)
    file.save(uploaded_video_path)
    
    return jsonify({'status': 'started'})

@app.route('/submit_questionnaire', methods=['POST'])
def submit_questionnaire():
    global questionnaire_data
    data = request.form
    questionnaire_data = {
        'age': data.get('age'),
        'family_history': data.get('family_history'),
        'symptoms': data.get('symptoms'),
        'iop_history': data.get('iop_history'),
        'medical_history': data.get('medical_history')
    }
    return jsonify({'status': 'questionnaire submitted'})

@app.route('/compare_predict', methods=['GET'])
def compare_predict():
    global recent_frames, last_features, questionnaire_data
    if not recent_frames or questionnaire_data is None:
        return jsonify({'error': 'No detection data or questionnaire submitted'}), 400
    
    # Select frame with highest pupil detection confidence
    best_frame_data = max(recent_frames, key=lambda x: x['prob'] if x['pupil_detected'] else 0)
    frame = best_frame_data['frame']
    eye_prob = best_frame_data['prob']
    features = best_frame_data.get('features', last_features)
    
    # Process frame for prediction
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    eyes = detector.pupil_detector.eye_cascade.detectMultiScale(gray, 1.3, 5)
    if len(eyes) > 0:
        ex, ey, ew, eh = eyes[0]
        eye_frame = frame[ey:ey+eh, ex:ex+ew]
        if eye_frame.size > 0:
            _, result, prob, _ = detector.analyze_eye_frame(eye_frame)
            eye_prob = prob if prob is not None else eye_prob
    
    # Questionnaire-based risk score
    score = 0
    age = int(questionnaire_data['age']) if questionnaire_data['age'] else 0
    score += 0.05 * max(0, age - 40)  # +0.05 per year over 40
    if questionnaire_data['family_history'] == 'Yes':
        score += 2
    symptoms = questionnaire_data['symptoms'].split(', ') if questionnaire_data['symptoms'] != 'None' else []
    score += 0.5 * len(symptoms)
    iop = float(questionnaire_data['iop_history']) if questionnaire_data['iop_history'] else 0
    if iop > 30:
        score += 2
    elif iop > 21:
        score += 1
    
    # Pupil features risk
    if features['pupil_circularity'] < 0.8 and features['pupil_circularity'] > 0:
        score += 1
    if features['pupil_radius'] < 10 or features['pupil_radius'] > 50:
        score += 1
    
    # Combine with eye detection probability
    score += 3 * eye_prob  # Weight eye detection heavily
    # Apply sigmoid to get glaucoma probability
    glaucoma_prob = 1 / (1 + math.exp(-score))
    healthy_prob = 1 - glaucoma_prob
    
    return jsonify({
        'glaucoma_prob': round(glaucoma_prob * 100),
        'healthy_prob': round(healthy_prob * 100)
    })

if __name__ == '__main__':
    app.run(debug=True)