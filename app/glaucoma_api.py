import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request, APIRouter
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from app.glaucoma_detection import PupilDetector, RealTimeGlaucomaDetection
import os
from werkzeug.utils import secure_filename
import math
import logging
import asyncio
from typing import Optional

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


glaucoma_router = APIRouter()

app = FastAPI(title="Glaucoma Detection API", version="1.0.0")

# Mount static files and templates
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

# Initialize detector and state
detector = RealTimeGlaucomaDetection(model_path='app/utils/best_glaucoma_model.keras')
detection_active = False
last_features = {'pupil_radius': 0, 'pupil_area': 0, 'pupil_circularity': 0}
uploaded_video_path = None
recent_frames = []  # Buffer for recent frames with detection data
questionnaire_data = None
MAX_FRAMES = 10  # Store up to 10 recent frames with detection data

async def generate_frames(source):
    global detection_active, last_features, uploaded_video_path, recent_frames
    if detection_active:
        logging.warning("Attempted to start detection while another is active")
        return
    
    detection_active = True
    recent_frames = []  # Reset recent_frames for each detection session
    
    if isinstance(source, int):
        cap = cv2.VideoCapture(source)  # Webcam
    else:
        cap = cv2.VideoCapture(source)  # Video file
    
    if not cap.isOpened():
        logging.error(f"Could not open source: {source}")
        detection_active = False
        return
    
    try:
        while detection_active:
            ret, frame = cap.read()
            if not ret:
                logging.warning(f"Failed to read frame from source: {source}")
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
            
            # Store frame in buffer if pupil detected
            if frame_data['pupil_detected']:
                recent_frames.append(frame_data)
                if len(recent_frames) > MAX_FRAMES:
                    recent_frames.pop(0)  # Keep only the most recent MAX_FRAMES
            
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            # Add small delay to prevent blocking
            await asyncio.sleep(0.01)
    
    finally:
        cap.release()
        detection_active = False
        if uploaded_video_path and os.path.exists(uploaded_video_path):
            os.remove(uploaded_video_path)
            uploaded_video_path = None
            logging.info(f"Removed uploaded video: {uploaded_video_path}")
        logging.info("generate_frames ended.")

@glaucoma_router.get("/glaucomaPage", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("gDetection.html", {"request": request})

@glaucoma_router.get("/video_feed")
async def video_feed(request: Request, type: str = "webcam"):
    global detection_active, uploaded_video_path
    if detection_active:
        logging.warning("Video feed requested while detection active")
        raise HTTPException(status_code=400, detail="Another detection is already active")
    
    if type == "webcam":
        source = 0
    elif type == "uploaded" and uploaded_video_path:
        source = uploaded_video_path
    else:
        logging.error(f"Invalid source type: {type}")
        raise HTTPException(status_code=400, detail="Invalid source type")
    
    return StreamingResponse(
        generate_frames(source), 
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@glaucoma_router.post("/stop_feed")
async def stop_feed():
    global detection_active, uploaded_video_path, recent_frames
    detection_active = False
    has_data = len(recent_frames) > 0  # Check if there's data for prediction
    if uploaded_video_path and os.path.exists(uploaded_video_path):
        os.remove(uploaded_video_path)
        uploaded_video_path = None
        logging.info(f"Removed uploaded video: {uploaded_video_path}")
    logging.info(f"Stop feed: has_data={has_data}")
    return JSONResponse(content={'status': 'stopped', 'has_data': has_data})

@glaucoma_router.get("/pupil_dynamics")
async def pupil_dynamics():
    global last_features
    return JSONResponse(content=last_features)

@glaucoma_router.post("/analyze_video")
async def analyze_video(video: UploadFile = File(...)):
    global detection_active, uploaded_video_path
    if detection_active:
        logging.warning("Analyze video requested while detection active")
        raise HTTPException(status_code=400, detail="Another detection is already active")
    
    filename = secure_filename(video.filename)
    uploaded_video_path = os.path.join('g_Uploads', filename)
    os.makedirs('uploads', exist_ok=True)
    
    # Read and save the uploaded file
    content = await video.read()
    with open(uploaded_video_path, 'wb') as f:
        f.write(content)
    
    logging.info(f"Saved uploaded video: {uploaded_video_path}")
    return JSONResponse(content={'status': 'started'})

@glaucoma_router.post("/submit_questionnaire")
async def submit_questionnaire(
    age: int = Form(...),
    family_history: str = Form(...),
    symptoms: str = Form(...),
    iop_history: Optional[float] = Form(None),
    medical_history: Optional[str] = Form(None),
    questionnaire_score: float = Form(...)
):
    global questionnaire_data
    questionnaire_data = {
        'age': age,
        'family_history': family_history,
        'symptoms': symptoms,
        'iop_history': iop_history,
        'medical_history': medical_history,
        'questionnaire_score': questionnaire_score
    }
    logging.info("Questionnaire submitted")
    return JSONResponse(content={'status': 'questionnaire submitted'})

@glaucoma_router.get("/compare_predict")
async def compare_predict():
    global recent_frames, last_features, questionnaire_data
    if not questionnaire_data:
        logging.error("No questionnaire submitted")
        raise HTTPException(status_code=400, detail="No questionnaire submitted")
    
    if not recent_frames:
        logging.error("No detection data available")
        raise HTTPException(status_code=400, detail="No detection data available")
    
    # Select frame with highest pupil detection confidence
    best_frame_data = max(recent_frames, key=lambda x: x['prob'] if x['pupil_detected'] else 0)
    eye_prob = best_frame_data['prob']
    features = best_frame_data.get('features', last_features)
    
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
    
    logging.info(f"Prediction: glaucoma_prob={glaucoma_prob*100:.1f}%, healthy_prob={healthy_prob*100:.1f}%")
    return JSONResponse(content={
        'glaucoma_prob': round(glaucoma_prob * 100),
        'healthy_prob': round(healthy_prob * 100)
    })

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(
        "glaucoma_api:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="debug",
        access_log=True
    )