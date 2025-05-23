from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List
import tensorflow as tf
import numpy as np
import io
import os
import cv2
from PIL import Image

cataract_router = APIRouter()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "utils", "rnn_cnn1_model.h5")
model = None

# Load the model once
@cataract_router.on_event("startup")
async def load_model():
    global model
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None

class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    probabilities: dict

def preprocess_image(image):
    image = image.resize((150, 150))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

def detect_eye(image: np.ndarray) -> bool:
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray, 1.1, 3)
    return len(eyes) > 0

@cataract_router.get("/health")
async def health_check():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy"}

@cataract_router.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        if image.mode == "RGBA":
            image = image.convert("RGB")

        nparr = np.frombuffer(contents, np.uint8)
        img_cv2 = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if not detect_eye(img_cv2):
            return {
                "prediction": "No eye-related features detected",
                "confidence": 0.0,
                "probabilities": {"Normal": 0.0, "Cataract": 0.0}
            }

        processed_image = preprocess_image(image)
        predictions = model.predict(processed_image)[0]

        classes = ["Cataract", "Normal"]
        predicted_class = classes[np.argmax(predictions)]
        confidence = float(np.max(predictions) * 100)

        return {
            "prediction": predicted_class,
            "confidence": confidence,
            "probabilities": {classes[i]: float(predictions[i] * 100) for i in range(len(classes))}
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@cataract_router.post("/batch-predict")
async def batch_predict(files: List[UploadFile] = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    results = []

    for file in files:
        try:
            contents = await file.read()
            image = Image.open(io.BytesIO(contents))
            if image.mode == "RGBA":
                image = image.convert("RGB")
            processed_image = preprocess_image(image)
            predictions = model.predict(processed_image)[0]
            classes = ["Cataract", "Normal"]
            predicted_class = classes[np.argmax(predictions)]
            confidence = float(np.max(predictions) * 100)
            results.append({
                "filename": file.filename,
                "prediction": predicted_class,
                "confidence": confidence,
                "probabilities": {
                    classes[i]: float(predictions[i] * 100) for i in range(len(classes))
                }
            })
        except Exception as e:
            results.append({
                "filename": file.filename,
                "error": str(e)
            })

    return {"results": results}

@cataract_router.get("/model-info")
async def model_info():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        return {
            "input_shape": model.input_shape[1:],
            "output_shape": model.output_shape[1:],
            "model_type": type(model).__name__,
            "classes": ["Normal", "Cataract"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting model info: {str(e)}")

@cataract_router.get("/", response_class=HTMLResponse)
async def get_index():
    with open("app/static/index.html") as f:
        return HTMLResponse(content=f.read())

@cataract_router.post("/debug/test-eye-detection")
async def debug_eye_detection(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return {"error": "Invalid image"}

    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray, 1.1, 3)
    eye_detected = len(eyes) > 0

    return {"eye_detected": eye_detected}
