from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import json
import os
from typing import List
import logging
from contextlib import asynccontextmanager
import gdown   # <-- NEW: for downloading from Google Drive

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for model and classes
model = None
plant_classes = None

# Google Drive link (replace with your file ID)
MODEL_PATH = "plant_species_CNN.h5"
GDRIVE_URL = "https://drive.google.com/uc?id=1rKl_jRCbTixMHIOct95pLyta1Q5tPgrk"  # <-- UPDATE THIS


def download_model():
    """Download model from Google Drive if not exists"""
    if not os.path.exists(MODEL_PATH):
        logger.info("Model not found locally. Downloading from Google Drive...")
        gdown.download(GDRIVE_URL, MODEL_PATH, quiet=False)
        logger.info("Model downloaded successfully.")

def load_model():
    """Load the trained model and plant classes"""
    global model, plant_classes
    
    try:
        # Ensure model exists (download if missing)
        download_model()
        
        # Load the model
        model = tf.keras.models.load_model(MODEL_PATH)
        logger.info("Model loaded successfully")
        
        # Load plant classes
        classes_path = "plant_classes.json"
        if os.path.exists(classes_path):
            with open(classes_path, 'r') as f:
                plant_classes = json.load(f)
            logger.info("Plant classes loaded successfully")
        else:
            logger.error(f"Classes file not found at {classes_path}")
            raise FileNotFoundError(f"Classes file not found at {classes_path}")
            
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

# Lifespan context manager to replace deprecated on_event
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting up Plant Classification API...")
    load_model()
    logger.info("API startup complete")
    yield
    # Shutdown (if needed)
    logger.info("Shutting down Plant Classification API...")

# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Plant Species Classification API",
    description="A FastAPI service for classifying plant species using ResNet50",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Preprocess image for model prediction"""
    try:
        image = Image.open(io.BytesIO(image_bytes))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image = image.resize((224, 224))
        image_array = np.array(image).astype('float32') / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        return image_array
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")

@app.get("/")
async def root():
    return {"message": "Plant Species Classification API", "version": "1.0.0", "status": "running"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "classes_loaded": plant_classes is not None
    }

@app.get("/classes")
async def get_classes():
    if plant_classes is None:
        raise HTTPException(status_code=500, detail="Plant classes not loaded")
    return {"classes": plant_classes, "count": len(plant_classes)}

@app.post("/predict")
async def predict_plant(file: UploadFile = File(...)):
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    try:
        image_bytes = await file.read()
        processed_image = preprocess_image(image_bytes)
        predictions = model.predict(processed_image, verbose=0)
        predicted_class_index = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_index])
        predicted_class = plant_classes[predicted_class_index]
        return {"predicted_class": predicted_class, "confidence": confidence, "filename": file.filename}
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict-base64")
async def predict_plant_base64(request: dict):
    try:
        if 'image' not in request:
            raise HTTPException(status_code=400, detail="Missing 'image' field in request")
        import base64
        image_data = base64.b64decode(request['image'])
        processed_image = preprocess_image(image_data)
        predictions = model.predict(processed_image, verbose=0)
        predicted_class_index = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_index])
        predicted_class = plant_classes[predicted_class_index]
        return {"predicted_class": predicted_class, "confidence": confidence}
    except Exception as e:
        logger.error(f"Base64 prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict-batch")
async def predict_batch(files: List[UploadFile] = File(...)):
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 images allowed per batch")
    results = []
    for file in files:
        try:
            if not file.content_type.startswith('image/'):
                results.append({"filename": file.filename, "error": "File must be an image"})
                continue
            image_bytes = await file.read()
            processed_image = preprocess_image(image_bytes)
            predictions = model.predict(processed_image, verbose=0)
            predicted_class_index = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class_index])
            predicted_class = plant_classes[predicted_class_index]
            results.append({"filename": file.filename, "predicted_class": predicted_class, "confidence": confidence})
        except Exception as e:
            results.append({"filename": file.filename, "error": str(e)})
    return {"results": results}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)





