from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import uvicorn
import numpy as np
from PIL import Image
import io
from pydantic import BaseModel
from typing import List, Optional
import os
import cv2


# Create FastAPI app
app = FastAPI(
    title="Cataract Classification API",
    description="API for classifying eye images as Cataract or Normal",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define model path - update this to your model's path
MODEL_PATH = "rnn_cnn1_model.h5"

# Load model at startup
model = None

@app.on_event("startup")
async def load_model():
    global model
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None

# Define response model
class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    probabilities: dict

# Preprocess image function
def preprocess_image(image):
    # Resize to match model input size
    image = image.resize((150, 150))
    # Convert to array and normalize
    image_array = np.array(image) / 255.0
    # Ensure correct shape with batch dimension
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

# Health check endpoint
@app.get("/health")
async def health_check():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy"}

# Eye detection function
def detect_eye(image: np.ndarray) -> bool:
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray, 1.1, 3)
    return len(eyes) > 0

# Prediction endpoint
@app.post("/predict", response_model=PredictionResponse)
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
            result = {
                "prediction": "No eye-related features detected",
                "confidence": 0.0,
                "probabilities": {"Normal": 0.0, "Cataract": 0.0}
            }
            return result
        
        processed_image = preprocess_image(image)
        predictions = model.predict(processed_image)[0]
        
        classes = ["Cataract", "Normal"]
        predicted_class = classes[np.argmax(predictions)]
        confidence = float(np.max(predictions) * 100)
        
        result = {
            "prediction": predicted_class,
            "confidence": confidence,
            "probabilities": {classes[i]: float(predictions[i] * 100) for i in range(len(classes))}
        }
        
        return result
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# Batch prediction endpoint
@app.post("/batch-predict")
async def batch_predict(files: List[UploadFile] = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    results = []
    
    for file in files:
        try:
            # Read image
            contents = await file.read()
            image = Image.open(io.BytesIO(contents))
            
            # Convert RGBA to RGB if necessary
            if image.mode == "RGBA":
                image = image.convert("RGB")
            
            # Preprocess image
            processed_image = preprocess_image(image)
            
            # Make prediction
            predictions = model.predict(processed_image)[0]
            
            # Process results
            classes = ["Cataract", "Normal"]
            predicted_class = classes[np.argmax(predictions)]
            confidence = float(np.max(predictions) * 100)
            
            # Add to results
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

# Optional: Get model information
@app.get("/model-info")
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
    
@app.get("/", response_class=HTMLResponse)
async def get_index():
    with open("static/index.html") as f:
        return HTMLResponse(content=f.read())


@app.post("/debug/test-eye-detection")
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
    
    _, buffer = cv2.imencode('.jpg', img)
    return {"eye_detected": eye_detected}

# Run server if executed as script
if __name__ == "__main__":
    # You can change host and port as needed
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)