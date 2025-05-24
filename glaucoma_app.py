import cv2
import numpy as np
from flask import Flask, Response, render_template, request, jsonify
from webcam_detection import PupilDetector, RealTimeGlaucomaDetection
import os
from werkzeug.utils import secure_filename
import math
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__, static_folder='app/static', template_folder='app/templates')

# Initialize detector and state
detector = RealTimeGlaucomaDetection(model_path='models/best_glaucoma_model.keras')
detection_active = False
last_features = {'pupil_radius': 0, 'pupil_area': 0, 'pupil_circularity': 0}
uploaded_video_path = None
recent_frames = []  # Buffer for recent frames with detection data
questionnaire_data = None
MAX_FRAMES = 10  # Store up to 10 recent frames
webcam_recording_path = None  # Path for saved webcam footage
webcam_recording_done = False  # Flag for recording completion

def generate_frames(source):
    global detection_active, last_features, uploaded_video_path, recent_frames, webcam_recording_path, webcam_recording_done
    if detection_active:
        logging.warning("Attempted to start detection while another is active")
        return
    
    detection_active = True
    recent_frames = []
    webcam_recording_done = False
    frame_count = 0
    max_frames_to_record = 300  # 10s at 30 FPS
    
    # Initialize video writer for webcam
    video_writer = None
    if isinstance(source, int):
        webcam_recording_path = os.path.join('uploads', 'webcam_recording.avi')
        os.makedirs('uploads', exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_writer = cv2.VideoWriter(webcam_recording_path, fourcc, 30.0, (720, 480))
        logging.info(f"Started recording webcam to {webcam_recording_path}")
    
    if isinstance(source, int):
        cap = cv2.VideoCapture(source)  # webcam
    else:
        cap = cv2.VideoCapture(source)  # video file
    
    if not cap.isOpened():
        logging.error(f"Could not open source: {source}")
        detection_active = False
        if video_writer:
            video_writer.release()
            logging.info("VideoWriter released due to source error")
        return
    
    while detection_active:
        ret, frame = cap.read()
        if not ret:
            logging.warning(f"Failed to read frame from source: {source}")
            break
        
        # Resize full frame to 720x480
        frame = cv2.resize(frame, (720, 480))
        
        # Write frame to video for webcam
        if video_writer and frame_count < max_frames_to_record:
            video_writer.write(frame)
            frame_count += 1
            logging.debug(f"Wrote frame {frame_count} to webcam video")
            if frame_count >= max_frames_to_record:
                video_writer.release()
                video_writer = None
                webcam_recording_done = True
                logging.info(f"Completed recording {frame_count} frames to {webcam_recording_path}")
        
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
    if video_writer:
        video_writer.release()
        logging.info("VideoWriter released at end of stream")
    if uploaded_video_path and os.path.exists(uploaded_video_path):
        os.remove(uploaded_video_path)
        uploaded_video_path = None
        logging.info(f"Removed uploaded video: {uploaded_video_path}")
    logging.info(f"generate_frames ended. Webcam recording path: {webcam_recording_path}, done: {webcam_recording_done}")

@app.route('/')
def index():
    return render_template('gDetection.html')

@app.route('/video_feed')
def video_feed():
    global detection_active, uploaded_video_path
    if detection_active:
        logging.warning("Video feed requested while detection active")
        return jsonify({'error': 'Another detection is already active'}), 400
    source_type = request.args.get('type', 'webcam')
    if source_type == 'webcam':
        source = 0
    elif source_type == 'uploaded' and uploaded_video_path:
        source = uploaded_video_path
    else:
        logging.error(f"Invalid source type: {source_type}")
        return jsonify({'error': 'Invalid source type'}), 400
    return Response(generate_frames(source), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop_feed', methods=['POST'])
def stop_feed():
    global detection_active, uploaded_video_path, recent_frames, webcam_recording_path, webcam_recording_done
    detection_active = False
    recent_frames = []
    if uploaded_video_path and os.path.exists(uploaded_video_path):
        os.remove(uploaded_video_path)
        uploaded_video_path = None
        logging.info(f"Removed uploaded video: {uploaded_video_path}")
    # Mark webcam footage as ready for analysis
    if webcam_recording_path and os.path.exists(webcam_recording_path):
        webcam_recording_done = True
        logging.info(f"Stop feed: Webcam footage ready at {webcam_recording_path}")
        return jsonify({'status': 'stopped', 'webcam_ready': True})
    logging.info("Stop feed: No webcam footage available")
    return jsonify({'status': 'stopped', 'webcam_ready': False})

@app.route('/pupil_dynamics')
def pupil_dynamics():
    global last_features
    return jsonify(last_features)

@app.route('/analyze_video', methods=['POST'])
def analyze_video():
    global detection_active, uploaded_video_path
    if detection_active:
        logging.warning("Analyze video requested while detection active")
        return jsonify({'error': 'Another detection is already active'}), 400
    
    file = request.files['video']
    filename = secure_filename(file.filename)
    uploaded_video_path = os.path.join('Uploads', filename)
    os.makedirs('uploads', exist_ok=True)
    file.save(uploaded_video_path)
    logging.info(f"Saved uploaded video: {uploaded_video_path}")
    
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
        'medical_history': data.get('medical_history'),
        'questionnaire_score': float(data.get('questionnaire_score', 0))
    }
    logging.info("Questionnaire submitted")
    return jsonify({'status': 'questionnaire submitted'})

@app.route('/compare_predict', methods=['GET'])
def compare_predict():
    global recent_frames, last_features, questionnaire_data, webcam_recording_path, webcam_recording_done
    logging.info(f"compare_predict: webcam_recording_done={webcam_recording_done}, path={webcam_recording_path}")
    if not questionnaire_data:
        logging.error("No questionnaire submitted")
        return jsonify({'error': 'No questionnaire submitted'}), 400
    
    # For webcam, use saved footage if recording is done
    if webcam_recording_done and webcam_recording_path and os.path.exists(webcam_recording_path):
        logging.info(f"Processing webcam footage: {webcam_recording_path}")
        cap = cv2.VideoCapture(webcam_recording_path)
        if not cap.isOpened():
            logging.error(f"Failed to open webcam footage: {webcam_recording_path}")
            return jsonify({'error': 'Failed to open webcam footage'}), 400
        temp_frames = []
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            frame = cv2.resize(frame, (720, 480))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            eyes = detector.pupil_detector.eye_cascade.detectMultiScale(gray, 1.3, 5)
            frame_data = {'frame': frame.copy(), 'prob': 0.0, 'pupil_detected': False}
            for (ex, ey, ew, eh) in eyes:
                eye_frame = frame[ey:ey+eh, ex:ex+ew]
                if eye_frame.size == 0:
                    continue
                _, result, prob, features = detector.analyze_eye_frame(eye_frame)
                if features['pupil_detected']:
                    frame_data['prob'] = prob
                    frame_data['pupil_detected'] = True
                    frame_data['features'] = {
                        'pupil_radius': features['pupil_radius'],
                        'pupil_area': features['pupil_area'],
                        'pupil_circularity': round(features['circularity'], 3) if 'circularity' in features else 0
                    }
            if frame_data['pupil_detected']:
                temp_frames.append(frame_data)
        cap.release()
        logging.info(f"Processed {frame_count} frames, {len(temp_frames)} with pupil detected")
        if not temp_frames:
            logging.error("No pupil detected in webcam footage")
            return jsonify({'error': 'No pupil detected in webcam footage'}), 400
        recent_frames = temp_frames
    
    if not recent_frames:
        logging.error("No detection data available")
        return jsonify({'error': 'No detection data available'}), 400
    
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
    
    # Eye detection score
    eye_score = eye_prob
    if features['pupil_circularity'] < 0.8 and features['pupil_circularity'] > 0:
        eye_score += 0.2
    if features['pupil_radius'] < 10 or features['pupil_radius'] > 50:
        eye_score += 0.2
    
    # Combine scores (50% questionnaire, 50% eye detection)
    q_score = questionnaire_data['questionnaire_score']
    combined_score = 0.5 * q_score + 0.5 * eye_score
    
    # Scale to cap at 90% glaucoma probability
    max_score = 2  # Maps to ~88% via sigmoid
    scaled_score = min(max(combined_score * 4 - 2, -max_score), max_score)  # Linear scaling to [-2, 2]
    glaucoma_prob = 1 / (1 + math.exp(-scaled_score))
    healthy_prob = 1 - glaucoma_prob
    
    # Clean up webcam footage after prediction
    if webcam_recording_path and os.path.exists(webcam_recording_path):
        os.remove(webcam_recording_path)
        logging.info(f"Removed webcam footage: {webcam_recording_path}")
        webcam_recording_path = None
        webcam_recording_done = False
    
    logging.info(f"Prediction: glaucoma_prob={glaucoma_prob*100:.1f}%, healthy_prob={healthy_prob*100:.1f}%")
    return jsonify({
        'glaucoma_prob': round(glaucoma_prob * 100),
        'healthy_prob': round(healthy_prob * 100)
    })

if __name__ == '__main__':
    app.run(debug=True)