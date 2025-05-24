from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse, HTMLResponse
import os
import numpy as np
import cv2
import tensorflow as tf
import mediapipe as mp

fatigue_router = APIRouter()

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "utils", "eye_fatigue_model.h5")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")  # ✅ Use templates folder

# Load model
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
model = tf.keras.models.load_model(MODEL_PATH)

# Mediapipe setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]

def calculate_ear(landmarks, left_eye_indices, right_eye_indices):
    def aspect_ratio(eye):
        return np.linalg.norm(eye[1] - eye[5]) / (2 * np.linalg.norm(eye[0] - eye[3]))
    left_eye = np.array([[landmarks[i].x, landmarks[i].y] for i in left_eye_indices])
    right_eye = np.array([[landmarks[i].x, landmarks[i].y] for i in right_eye_indices])
    return aspect_ratio(left_eye), aspect_ratio(right_eye)

# ✅ Serve HTML from templates
@fatigue_router.get("/", response_class=HTMLResponse)
async def index():
    path = os.path.join(TEMPLATES_DIR, "index.html")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    return HTMLResponse("<h1>Index not found</h1>", status_code=404)

@fatigue_router.get("/eye_fatigue", response_class=HTMLResponse)
async def eye_fatigue():
    path = os.path.join(TEMPLATES_DIR, "index.html")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    return HTMLResponse("<h1>Eye Fatigue page not found</h1>", status_code=404)

@fatigue_router.get("/color_test", response_class=HTMLResponse)
async def color_test():
    path = os.path.join(TEMPLATES_DIR, "color_test.html")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    return HTMLResponse("<h1>Color Test page not found</h1>", status_code=404)

# ✅ Analyze uploaded image
@fatigue_router.post("/analyze_frame")
async def analyze_frame(frame: UploadFile = File(...)):
    try:
        contents = await frame.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(frame_np, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            left_ear, right_ear = calculate_ear(landmarks, LEFT_EYE_INDICES, RIGHT_EYE_INDICES)
            avg_ear = (left_ear + right_ear) / 2

            resized = cv2.resize(rgb, (128, 128))
            img_array = np.expand_dims(resized / 255.0, axis=0)
            prediction = model.predict(img_array)[0][0]

            return {
                "avg_ear": float(avg_ear),
                "left_ear": float(left_ear),
                "right_ear": float(right_ear),
                "fatigue_score": float(prediction)
            }
        else:
            return JSONResponse(content={"error": "No face detected"}, status_code=400)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
