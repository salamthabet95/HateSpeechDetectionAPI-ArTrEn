# Hate Speech Detection API

Flask-based REST API for multilingual hate speech detection in Arabic, Turkish, and English using machine learning.

## Overview

This API serves trained machine learning models for real-time hate speech detection. It provides intelligent text analysis with confidence scoring, risk assessment, and language detection capabilities.

## Features

- **Multilingual Detection**: Arabic, Turkish, and English support
- **Automatic Language Detection**: Uses langdetect library
- **Advanced Feature Extraction**: 75+ linguistic and statistical features
- **Risk Assessment**: 5-level risk categorization with color coding
- **Sentiment Analysis**: VADER sentiment scoring integration
- **Real-time Processing**: Optimized for mobile app integration
- **CORS Support**: Cross-origin requests enabled

## Quick Start

### Prerequisites

- Python 3.7+
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone https://github.com/salamthabet95/HateSpeechDetectionAPI-ArTrEn.git
cd HateSpeechDetectionAPI-ArTrEn
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Ensure the trained model file `intelligent_multilingual_detector.pkl` is in the project directory

4. Start the server:
```bash
python app.py
```

The API will be available at `http://localhost:5000`

## API Endpoints

### Health Check
```
GET /health
```

Response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "langdetect_available": true,
  "textstat_available": true,
  "vader_available": true
}
```

### Text Analysis
```
POST /predict
Content-Type: application/json
```

Request:
```json
{
  "text": "Text to analyze"
}
```

Response:
```json
{
  "success": true,
  "prediction": "Not Hate Speech",
  "confidence": 0.856,
  "hate_probability": 0.144,
  "detected_language": "arabic",
  "risk_level": "SAFE",
  "risk_color": "#00AA44",
  "risk_message": "Safe - No Significant Risk",
  "analysis": {
    "text_length": 25,
    "word_count": 4,
    "sentence_count": 1,
    "sentiment_score": 0.0,
    "lexical_diversity": 1.0,
    "toxicity_indicators": {
      "profanity": 0,
      "violence": 0,
      "hate_words": 0,
      "personal_attacks": false
    },
    "language_features": {
      "arabic_ratio": 0.84,
      "turkish_ratio": 0.0,
      "is_multilingual": false
    }
  }
}
```

### Supported Languages
```
GET /languages
```

## Risk Assessment Levels

| Level | Probability Range | Color | Description |
|-------|------------------|-------|-------------|
| SAFE | < 0.4 | #00AA44 | No significant risk detected |
| LOW | 0.4-0.6 | #FFAA00 | Minor concerning patterns |
| MEDIUM | 0.6-0.8 | #FF8800 | Potentially problematic content |
| HIGH | 0.8-0.9 | #FF4444 | Likely hate speech |
| CRITICAL | ≥ 0.9 | #FF0000 | Immediate action required |

## Dependencies

```
Flask==2.3.3
flask-cors==4.0.0
scikit-learn==1.3.0
pandas==2.0.3
numpy==1.24.3
langdetect==1.0.9
textstat==0.7.3
vaderSentiment==3.3.2
```

## Configuration

### Environment Variables
```bash
export FLASK_ENV=development
export FLASK_DEBUG=1
export MODEL_PATH=intelligent_multilingual_detector.pkl
```

### Custom Configuration
Modify `app.py` for:
- Custom port: `app.run(port=8080)`
- Host binding: `app.run(host='0.0.0.0')`
- Debug mode: `app.run(debug=False)`

## Testing

Test the API with curl:
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world"}'
```

## Performance

- **Response Time**: < 200ms average
- **Throughput**: 100+ requests/minute
- **Memory Usage**: ~150MB base + model size

## Deployment

### Local Development
```bash
python app.py
```

### Production with Gunicorn
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### Docker
```bash
docker build -t hate-speech-api .
docker run -p 5000:5000 hate-speech-api
```

## Network Configuration

For mobile app integration:
- **Local testing**: `http://localhost:5000`
- **Android Emulator**: `http://10.0.2.2:5000`
- **Device testing**: Use your server's IP address

## Error Handling

Common response codes:
- `200`: Success
- `400`: Invalid input format
- `500`: Model not loaded or server error

## Client Applications

This API is designed to work with mobile applications. See the Android client:
[HateSpeechDetectionAndroidApp](https://github.com/salamthabet95/HateSpeechDetectionAndroidApp)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License.

## Developer

**Salam Thabet**
- GitHub: [@salamthabet95](https://github.com/salamthabet95)
- Email: salam.th.d@gmail.com

## Support

For API-related issues, please open an issue on this repository.
