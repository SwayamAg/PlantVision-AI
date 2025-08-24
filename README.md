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
Your API is ready for deployment!
- **API Base URL**: `https://your-railway-app.railway.app`
- **Interactive Documentation**: `https://your-railway-app.railway.app/docs`
- **Health Check**: `https://your-railway-app.railway.app/health`

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

3. **Ensure the model file is present**
   - The `plant_species_CNN.h5` model file should be in the root directory
   - This is a trained ResNet50 model for plant classification

### Running the API Locally

1. **Start the FastAPI server**
   ```bash
   python app.py
   ```

2. **Access the API**
   - API will be available at: `http://localhost:8000`
   - Interactive documentation: `http://localhost:8000/docs`
   - Alternative docs: `http://localhost:8000/redoc`
   - Health check: `http://localhost:8000/health`

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
curl -X POST "https://your-railway-app.railway.app/predict" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@your_image.jpg"

# Base64 image
curl -X POST "https://your-railway-app.railway.app/predict-base64" \
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
    response = requests.post('https://your-railway-app.railway.app/predict', files=files)
    result = response.json()
    print(f"Predicted: {result['predicted_class']}")
    print(f"Confidence: {result['confidence']}")

# Base64 image
import base64
with open('your_image.jpg', 'rb') as f:
    image_data = base64.b64encode(f.read()).decode('utf-8')
    response = requests.post('https://your-railway-app.railway.app/predict-base64', 
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

fetch('https://your-railway-app.railway.app/predict', {
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
    fetch('https://your-railway-app.railway.app/predict-base64', {
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

## 🚀 Deployment

### Railway Deployment

This project is configured for Railway deployment:

1. **Connect your GitHub repository to Railway**
2. **Railway will automatically detect the Python project**
3. **Deploy with one click**

### Environment Variables

The API uses the following environment variables:
- `PORT`: Port number (automatically set by Railway)
- `MODEL_PATH`: Path to the model file (default: `plant_species_CNN.h5`)
- `CLASSES_PATH`: Path to classes JSON file (default: `plant_classes.json`)

## 📁 Project Structure

```
Plant_CNN/
├── app.py                           # FastAPI application with CORS support
├── requirements.txt                 # Python dependencies and versions
├── Procfile                         # Railway deployment config
├── runtime.txt                      # Python runtime version (3.9.5)
├── plant_species_CNN.h5            # Trained ResNet50 model
├── plant_classes.json              # Class mapping configuration
├── plant-species-classification-resnet50-eda.ipynb # Jupyter notebook for EDA
├── README.md                       # Project documentation
└── DATA/                           # Training data directory
```

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
3. Update the model file `plant_species_CNN.h5`

### Model Training

The training process is documented in `plant-species-classification-resnet50-eda.ipynb`. To retrain:

1. Prepare your dataset in the `DATA/` directory
2. Run the Jupyter notebook
3. Replace `plant_species_CNN.h5` with the new model

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

## 📞 Support

For issues and questions:
- Create an issue in the repository
- Check the API documentation at `/docs` when running locally
- Review the health endpoint at `/health` for system status

---

## 👨‍💻 Made by Swayam

- **LinkedIn**: [Swayam Agarwal](https://www.linkedin.com/in/swayam-agarwal/)
- **GitHub**: [SwayamAg](https://github.com/SwayamAg)
- **Repository**: [Plant Classification API](https://github.com/SwayamAg/Plant-Identification)

---

**Note**: Make sure the `plant_species_CNN.h5` model file is present in the root directory for the API to function properly.
