import io
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, HTMLResponse
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array

# Load the trained model
model = tf.keras.models.load_model('my_model.h5')

app = FastAPI()


@app.get("/", response_class=HTMLResponse)
async def get_index():
    with open("static/index.html") as f:
        return HTMLResponse(content=f.read())

# Define the image size expected by the model
image_size = (150, 150)  # Adjust this based on your model's input size

# Image preprocessing function
def preprocess_image(image: Image.Image):
    # Resize the image to the target size expected by the model (150x150)
    image = image.resize(image_size)
    
    # Convert the image to a numpy array
    image = img_to_array(image)
    
    # Rescale the image (normalize to [0, 1])
    image = image / 255.0
    
    # Add an extra dimension for batch size (for prediction)
    image = np.expand_dims(image, axis=0)
    
    return image

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Read the image from the uploaded file
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))
    
    # Preprocess the image (resize + rescale)
    preprocessed_image = preprocess_image(image)
    
    # Make a prediction using the model
    prediction = model.predict(preprocessed_image)
    
    predicted_class = np.argmax(prediction, axis=1)
    
    if int(predicted_class[0]) == 0:
        return JSONResponse(content={"prediction": "Cataract"})
    else:
        return JSONResponse(content={"prediction": "Healthy"})
    
    

