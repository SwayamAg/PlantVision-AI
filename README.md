# 🌿 Plant Species Classification API

A machine learning-powered web API for identifying different types of house plants from images. This project uses a Convolutional Neural Network (CNN) based on ResNet50 architecture trained on TensorFlow to classify plant images into 10 different categories.

## 🌱 Supported Plant Types

The model can identify the following plant types:
- **Aloe Vera** (Aloe vera)
- **Areca Palm** (Dypsis lutescens)
- **Chinese evergreen** (Aglaonema)
- **English Ivy** (Hedera helix)
- **Jade plant** (Crassula ovata)
- **Money Tree** (Pachira aquatica)
- **Parlor Palm** (Chamaedorea elegans)
- **Rubber Plant** (Ficus elastica)
- **Snake plant** (Sanseviera)
- **ZZ Plant** (Zamioculcas zamiifolia)

## ✨ Features

- **FastAPI-based REST API** with automatic documentation
- **Real-time image classification** with confidence scores
- **Multiple input formats** - file uploads and base64 encoded images
- **CORS support** for web applications
- **Health check endpoint** for monitoring
- **Comprehensive error handling** and logging
- **Railway deployment ready** with Procfile

## 🚀 Quick Start

###  **Live Demo**
Your API is now deployed and ready to use!
- **API Base URL**: `http://web-production-f233.up.railway.app`
- **Interactive Documentation**: `http://web-production-f233.up.railway.app/docs`
- **Health Check**: `http://web-production-f233.up.railway.app/health`

### 📥 **Model File Download**
- **Google Drive Link**: [Download plant_species_Model.h5](https://drive.google.com/uc?id=1rKl_jRCbTixMHIOct95pLyta1Q5tPgrk)
- **File Size**: ~295MB
- **Required for**: Local development and testing

### Prerequisites

- Python 3.9.5
- TensorFlow 2.10.0
- FastAPI 0.109.0+
- Other dependencies (see `requirements.txt`)

### Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd Plant_CNN
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Model File Setup**
   - **For Local Development**: Download the model file from [Google Drive](https://drive.google.com/uc?id=1rKl_jRCbTixMHIOct95pLyta1Q5tPgrk) and place `plant_species_Model.h5` in the root directory
   - **For Deployment**: The model is automatically downloaded from Google Drive during deployment
   - This is a trained ResNet50 model for plant classification (~295MB)

### Running the API Locally

1. **Start the FastAPI server**
   ```bash
   python app.py
   ```

2. **Access the API**
   
   **Local Development:**
   - API will be available at: `http://localhost:8000`
   - Interactive documentation: `http://localhost:8000/docs`
   - Alternative docs: `http://localhost:8000/redoc`
   - Health check: `http://localhost:8000/health`
   
   **Live Deployment:**
   - API is available at: `http://web-production-f233.up.railway.app`
   - Interactive documentation: `http://web-production-f233.up.railway.app/docs`
   - Health check: `http://web-production-f233.up.railway.app/health`

## 📖 API Documentation

### Endpoints

#### 1. **GET /** - Root endpoint
Returns API information and available classes.
```json
{
  "message": "Plant Species Classification API",
  "classes": ["Aloe Vera", "Areca Palm", ...],
  "total_classes": 10
}
```

#### 2. **GET /health** - Health check
Returns the status of the API and model.
```json
{
  "status": "healthy",
  "model_loaded": true,
  "classes_loaded": true,
  "total_classes": 10
}
```

#### 3. **GET /classes** - Get available classes
Returns list of supported plant classes.
```json
{
  "classes": ["Aloe Vera", "Areca Palm", ...],
  "count": 10
}
```

#### 4. **POST /predict** - Image classification (file upload)
Upload an image file to get plant classification.

**Request:**
- Method: `POST`
- Content-Type: `multipart/form-data`
- Body: Image file

**Response:**
```json
{
  "predicted_class": "Aloe Vera",
  "confidence": 0.95
}
```

#### 5. **POST /predict-base64** - Image classification (base64)
Send base64 encoded image for plant classification.

**Request:**
- Method: `POST`
- Content-Type: `application/json`
- Body: `{"image": "base64_encoded_string"}`

**Response:**
```json
{
  "predicted_class": "Aloe Vera",
  "confidence": 0.95
}
```

### Example Usage

#### Using curl (Live API)
```bash
# File upload
curl -X POST "http://web-production-f233.up.railway.app/predict" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@your_image.jpg"

# Base64 image
curl -X POST "http://web-production-f233.up.railway.app/predict-base64" \
     -H "accept: application/json" \
     -H "Content-Type: application/json" \
     -d '{"image": "base64_encoded_image_string"}'
```

#### Using Python requests (Live API)
```python
import requests

# File upload
with open('your_image.jpg', 'rb') as f:
    files = {'file': f}
    response = requests.post('http://web-production-f233.up.railway.app/predict', files=files)
    result = response.json()
    print(f"Predicted: {result['predicted_class']}")
    print(f"Confidence: {result['confidence']}")

# Base64 image
import base64
with open('your_image.jpg', 'rb') as f:
    image_data = base64.b64encode(f.read()).decode('utf-8')
    response = requests.post('http://web-production-f233.up.railway.app/predict-base64', 
                           json={'image': image_data})
    result = response.json()
    print(f"Predicted: {result['predicted_class']}")
    print(f"Confidence: {result['confidence']}")
```

#### Using curl (Local API)
```bash
# File upload
curl -X POST "http://localhost:8000/predict" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@your_image.jpg"

# Base64 image
curl -X POST "http://localhost:8000/predict-base64" \
     -H "accept: application/json" \
     -H "Content-Type: application/json" \
     -d '{"image": "base64_encoded_image_string"}'
```

#### Using Python requests (Local API)
```python
import requests

# File upload
with open('your_image.jpg', 'rb') as f:
    files = {'file': f}
    response = requests.post('http://localhost:8000/predict', files=files)
    result = response.json()
    print(f"Predicted: {result['predicted_class']}")
    print(f"Confidence: {result['confidence']}")

# Base64 image
import base64
with open('your_image.jpg', 'rb') as f:
    image_data = base64.b64encode(f.read()).decode('utf-8')
    response = requests.post('http://localhost:8000/predict-base64', 
                           json={'image': image_data})
    result = response.json()
    print(f"Predicted: {result['predicted_class']}")
    print(f"Confidence: {result['confidence']}")
```

#### Using JavaScript
```javascript
// File upload
const formData = new FormData();
formData.append('file', fileInput.files[0]);

fetch('http://web-production-f233.up.railway.app/predict', {
    method: 'POST',
    body: formData
})
.then(response => response.json())
.then(data => {
    console.log(`Predicted: ${data.predicted_class}`);
    console.log(`Confidence: ${data.confidence}`);
});

// Base64 image
const reader = new FileReader();
reader.onload = function() {
    const base64Image = reader.result.split(',')[1];
    fetch('http://web-production-f233.up.railway.app/predict-base64', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({image: base64Image})
    })
    .then(response => response.json())
    .then(data => {
        console.log(`Predicted: ${data.predicted_class}`);
        console.log(`Confidence: ${data.confidence}`);
    });
};
reader.readAsDataURL(fileInput.files[0]);
```

## 📊 Local Testing

Use the included `test_api.py` script for quick local testing:

```bash
# Test with a specific image
python test_api.py path/to/your/image.jpg

# Or modify the IMAGE_PATH variable in test_api.py
```

### Testing with Sample Images

You can test the API with any plant image. Here are some suggestions:
- **Aloe Vera**: Images of aloe vera plants
- **Snake Plant**: Sansevieria trifasciata images
- **Jade Plant**: Crassula ovata images
- **Money Tree**: Pachira aquatica images

**Note**: For best results, use clear, well-lit images of the plant's leaves and overall structure.

## 🚀 Deployment

### Railway Deployment

This project is configured for Railway deployment with automatic model download:

1. **Connect your GitHub repository to Railway**
2. **Railway will automatically detect the Python project**
3. **The model file is automatically downloaded from Google Drive during deployment**
4. **Deploy with one click**

### Model File Management

Due to the large size of the trained model (~295MB), it's not included in the repository. Instead:

- **Local Development**: Download `plant_species_Model.h5` from [Google Drive](https://drive.google.com/uc?id=1rKl_jRCbTixMHIOct95pLyta1Q5tPgrk) and place it in the root directory
- **Deployment**: The FastAPI app automatically downloads the model from Google Drive during startup
- **Model Source**: The model is hosted on Google Drive for easy access and deployment

### Environment Variables

The API uses the following environment variables:
- `PORT`: Port number (automatically set by Railway)
- `MODEL_PATH`: Path to the model file (default: `plant_species_Model.h5`)
- `CLASSES_PATH`: Path to classes JSON file (default: `plant_classes.json`)
- `GOOGLE_DRIVE_MODEL_ID`: Google Drive file ID for model download (default: `1rKl_jRCbTixMHIOct95pLyta1Q5tPgrk`)

## 📁 Project Structure

```
Plant_CNN/
├── app.py                           # FastAPI application with CORS support
├── requirements.txt                 # Python dependencies and versions
├── Procfile                         # Railway deployment config
├── runtime.txt                      # Python runtime version (3.9.5)
├── plant_classes.json              # Class mapping configuration
├── plant-species-classification-resnet50-eda.ipynb # Jupyter notebook for EDA
├── README.md                       # Project documentation
├── DATA/                           # Training data directory
└── plant_species_Model.h5            # Trained ResNet50 model (download from Google Drive)
```

**Note**: The `plant_species_Model.h5` model file (~295MB) is not included in the repository due to size constraints. Download it from Google Drive for local development.

## 🔧 Model Information

- **Architecture**: ResNet50 (Convolutional Neural Network)
- **Framework**: TensorFlow 2.10.0
- **Input Size**: 224x224 pixels
- **Classes**: 10 house plant types
- **Training**: Custom dataset with train/validation/test splits
- **Python Version**: 3.9.5 (specifically configured)

## 🛠️ Development

### Adding New Plant Types

1. Update the `plant_classes.json` file
2. Retrain the model with new data
3. Update the model file `plant_species_Model.h5`
4. Upload the new model to Google Drive and update the file ID

### Model Training

The training process is documented in `plant-species-classification-resnet50-eda.ipynb`. To retrain:

1. Prepare your dataset in the `DATA/` directory
2. Run the Jupyter notebook
3. Replace `plant_species_Model.h5` with the new model
4. Update the Google Drive link in the code and documentation

### Code Structure

- **`app.py`**: Main FastAPI application with endpoints and model loading
- **`plant_classes.json`**: Class mapping configuration
- **`requirements.txt`**: Python dependencies
- **`Procfile`**: Railway deployment configuration
- **`runtime.txt`**: Python version specification

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

##  Acknowledgments

- ResNet50 architecture for image classification
- FastAPI team for the excellent web framework
- Railway for seamless deployment
- The plant dataset contributors

## 🔧 Troubleshooting

### Python Version Issues
If you encounter compatibility issues, ensure you're using Python 3.9.5:
```bash
python --version
```

If you need to install Python 3.9.5:
- **Windows**: Download from [python.org](https://www.python.org/downloads/release/python-395/)
- **macOS**: Use `pyenv install 3.9.5`
- **Linux**: Use your package manager or `pyenv install 3.9.5`

### Dependency Issues
If you encounter dependency conflicts:
1. Create a fresh virtual environment: `python -m venv venv`
2. Activate it: `source venv/bin/activate` (Linux/macOS) or `venv\Scripts\activate` (Windows)
3. Install dependencies: `pip install -r requirements.txt`

### Model Download Issues
If the model fails to download during deployment:
1. Check your internet connection
2. Verify the Google Drive link is accessible
3. Ensure the file ID is correct: `1rKl_jRCbTixMHIOct95pLyta1Q5tPgrk`
4. Check Railway logs for download errors

### API Performance
- **Model Loading**: The model takes ~30-60 seconds to load on first startup
- **Prediction Time**: Each prediction takes 1-3 seconds depending on image size
- **Memory Usage**: The model requires ~500MB RAM when loaded

## 📞 Support

For issues and questions:
- Create an issue in the repository
- Check the API documentation at `/docs` when running locally
- Review the health endpoint at `/health` for system status
- Check Railway deployment logs for troubleshooting

### Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| Model not loading | Check Google Drive link accessibility |
| Slow predictions | Normal for first few requests (model warming up) |
| Memory errors | Ensure sufficient RAM (500MB+ recommended) |
| CORS errors | API supports CORS for web applications |
| File upload errors | Check image format (JPEG, PNG supported) |

---

## 👨‍💻 Made by Swayam

- **LinkedIn**: [Swayam Agarwal](https://www.linkedin.com/in/swayam-agarwal/)
- **GitHub**: [SwayamAg](https://github.com/SwayamAg)
- **Repository**: [Plant Classification API](https://github.com/SwayamAg/Plant-Identification)

---

**Note**: 
- For local development: Download the `plant_species_Model.h5` model file from Google Drive and place it in the root directory
- For deployment: The model is automatically downloaded from Google Drive during startup
- Model file size: ~295MB (not included in repository due to size constraints)
