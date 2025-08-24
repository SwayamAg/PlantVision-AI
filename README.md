# Plant Classification API

A FastAPI-based REST API for classifying house plants using a ResNet50 deep learning model. This API can identify 10 different types of house plants with high accuracy.

## 🌿 Supported Plant Classes

1. Aloe Vera
2. Areca Palm (Dypsis lutescens)
3. Chinese evergreen (Aglaonema)
4. English Ivy (Hedera helix)
5. Jade plant (Crassula ovata)
6. Money Tree (Pachira aquatica)
7. Parlor Palm (Chamaedorea elegans)
8. Rubber Plant (Ficus elastica)
9. Snake plant (Sanseviera)
10. ZZ Plant (Zamioculcas zamiifolia)

## 🚀 Features

- **Image Classification**: Upload images to get plant species predictions
- **Multiple Input Formats**: Support for file uploads and base64 encoded images
- **Confidence Scores**: Get prediction confidence for the top class
- **Single Prediction**: Returns only the most likely plant species
- **RESTful API**: Clean, documented API endpoints
- **CORS Enabled**: Ready for web applications
- **Health Checks**: Monitor API status and model loading

## 🛠️ Tech Stack

- **FastAPI**: Modern, fast web framework for building APIs
- **TensorFlow**: Deep learning framework for model inference
- **ResNet50**: Pre-trained CNN architecture for image classification
- **Pillow**: Image processing library
- **Uvicorn**: ASGI server for production deployment

## 📋 Prerequisites

- Python 3.9.5
- TensorFlow 2.10.0
- FastAPI 0.109.0+

> **Note**: This project is specifically configured for Python 3.9.5. The dependency versions have been carefully selected to ensure compatibility with this Python version.

## 🚀 Quick Start

### Local Development

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd Plant_CNN
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Check your environment** (recommended)
   ```bash
   python check_environment.py
   ```

4. **Run the application**
   ```bash
   python app.py
   ```

5. **Access the API**
   - API Documentation: http://localhost:8000/docs
   - Alternative Docs: http://localhost:8000/redoc
   - Health Check: http://localhost:8000/health

### Railway Deployment

1. **Connect your GitHub repository to Railway**
2. **Railway will automatically detect the Python project**
3. **Deploy with one click**

## 📚 API Endpoints

### Base URL
```
https://your-railway-app.railway.app
```

### Endpoints

#### `GET /`
- **Description**: Root endpoint with API information
- **Response**: API status and available classes

#### `GET /health`
- **Description**: Health check endpoint
- **Response**: Model loading status and class count

#### `GET /classes`
- **Description**: Get list of available plant classes
- **Response**: Array of class names and count

#### `POST /predict`
- **Description**: Predict plant species from uploaded image
- **Input**: Image file (multipart/form-data)
- **Response**: Prediction results with confidence scores

#### `POST /predict-base64`
- **Description**: Predict plant species from base64 encoded image
- **Input**: JSON with base64 image data
- **Response**: Prediction results with confidence scores

## 📖 API Usage Examples

### Using cURL

#### File Upload
```bash
curl -X POST "https://your-railway-app.railway.app/predict" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@plant_image.jpg"
```

#### Base64 Image
```bash
curl -X POST "https://your-railway-app.railway.app/predict-base64" \
     -H "accept: application/json" \
     -H "Content-Type: application/json" \
     -d '{"image": "base64_encoded_image_string"}'
```

### Using Python

```python
import requests

# File upload
with open('plant_image.jpg', 'rb') as f:
    files = {'file': f}
    response = requests.post('https://your-railway-app.railway.app/predict', files=files)
    print(response.json())

# Base64 image
import base64
with open('plant_image.jpg', 'rb') as f:
    image_data = base64.b64encode(f.read()).decode('utf-8')
    response = requests.post('https://your-railway-app.railway.app/predict-base64', 
                           json={'image': image_data})
    print(response.json())
```

### Using JavaScript

```javascript
// File upload
const formData = new FormData();
formData.append('file', fileInput.files[0]);

fetch('https://your-railway-app.railway.app/predict', {
    method: 'POST',
    body: formData
})
.then(response => response.json())
.then(data => console.log(data));

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
    .then(data => console.log(data));
};
reader.readAsDataURL(fileInput.files[0]);
```

## 📊 Response Format

### Successful Prediction Response
```json
{
  "predicted_class": "Aloe Vera",
  "confidence": 0.95
}
```

## 🔧 Configuration

### Environment Variables
- `PORT`: Port number (automatically set by Railway)
- `MODEL_PATH`: Path to the model file (default: `resnet_model_final.h5`)
- `CLASSES_PATH`: Path to classes JSON file (default: `plant_classes.json`)

### Python Version
- **Runtime**: Python 3.9.5 (specified in `runtime.txt`)
- **Compatibility**: All dependencies are tested and compatible with Python 3.9.5

## 📁 Project Structure

```
Plant_CNN/
├── app.py                 # Main FastAPI application
├── requirements.txt       # Python dependencies
├── Procfile              # Railway deployment configuration
├── runtime.txt           # Python version specification
├── README.md             # Project documentation
├── check_environment.py  # Environment compatibility checker
├── test_api.py           # API testing script
├── resnet_model_final.h5 # Trained model file
├── plant_classes.json    # Class names mapping
├── DATA/                 # Training data directory
└── archive/              # Additional data directory
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- ResNet50 architecture for image classification
- FastAPI for the excellent web framework
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

If you encounter any issues or have questions, please open an issue on GitHub or contact the maintainers.

---

**Happy Plant Classification! 🌱**
