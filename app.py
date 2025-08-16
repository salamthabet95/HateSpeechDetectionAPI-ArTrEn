from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import pandas as pd
import numpy as np
import os
import re
from collections import Counter
from sklearn.base import BaseEstimator, TransformerMixin

# Import intelligent components (you need to include these)
try:
    from langdetect import detect, detect_langs

    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False
    print("⚠️  langdetect not available. Using fallback detection.")

try:
    from textstat import flesch_reading_ease, flesch_kincaid_grade, automated_readability_index

    TEXTSTAT_AVAILABLE = True
except ImportError:
    TEXTSTAT_AVAILABLE = False
    print("⚠️  textstat not available. Readability features disabled.")

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False
    print("⚠️  VADER not available. Sentiment features disabled.")


# Add this to your app.py file, right after the imports and before load_intelligent_model()

class MultilingualTfidfExtractor(BaseEstimator, TransformerMixin):
    """Custom TF-IDF vectorizer optimized for multilingual text"""

    def __init__(self, max_features=2500, ngram_range=(1, 2)):
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.lang_detector = AdvancedLanguageDetector()
        self.vectorizers = {}
        self.feature_names = []

    def fit(self, X, y=None):
        # Detect languages for all texts
        languages = [self.lang_detector.detect_language(text) for text in X]
        lang_counts = Counter(languages)

        print(f"   TF-IDF language distribution:")
        for lang, count in lang_counts.most_common():
            print(f"     {lang}: {count} texts")

        # Group texts by language
        lang_texts = {}
        for text, lang in zip(X, languages):
            if lang not in lang_texts:
                lang_texts[lang] = []
            lang_texts[lang].append(text)

        # Create language-specific vectorizers
        for lang, texts in lang_texts.items():
            if len(texts) >= 5:  # Minimum texts for meaningful TF-IDF
                features_per_lang = min(self.max_features // len(lang_texts), 1000)

                from sklearn.feature_extraction.text import TfidfVectorizer

                if lang == 'arabic':
                    vectorizer = TfidfVectorizer(
                        max_features=features_per_lang,
                        ngram_range=self.ngram_range,
                        min_df=2,
                        max_df=0.9,
                        sublinear_tf=True,
                        token_pattern=r'\b\w{2,}\b',
                        analyzer='word',
                        lowercase=True
                    )
                elif lang == 'turkish':
                    vectorizer = TfidfVectorizer(
                        max_features=features_per_lang,
                        ngram_range=self.ngram_range,
                        min_df=2,
                        max_df=0.9,
                        sublinear_tf=True,
                        token_pattern=r'\b\w{2,}\b',
                        analyzer='word',
                        lowercase=True
                    )
                else:  # English and others
                    vectorizer = TfidfVectorizer(
                        max_features=features_per_lang,
                        ngram_range=self.ngram_range,
                        min_df=2,
                        max_df=0.9,
                        sublinear_tf=True,
                        strip_accents='unicode',
                        token_pattern=r'\b\w{2,}\b',
                        analyzer='word',
                        lowercase=True
                    )

                try:
                    vectorizer.fit(texts)
                    self.vectorizers[lang] = vectorizer
                    print(f"     ✅ Created {lang} vectorizer with {len(vectorizer.get_feature_names_out())} features")
                except Exception as e:
                    print(f"     ❌ Failed to create {lang} vectorizer: {e}")

        return self

    def transform(self, X):
        all_features = []

        for text in X:
            lang = self.lang_detector.detect_language(text)

            # Find appropriate vectorizer
            vectorizer = None
            if lang in self.vectorizers:
                vectorizer = self.vectorizers[lang]
            elif 'english' in self.vectorizers:
                vectorizer = self.vectorizers['english']
            elif len(self.vectorizers) > 0:
                vectorizer = list(self.vectorizers.values())[0]

            if vectorizer:
                try:
                    features = vectorizer.transform([text]).toarray()[0]
                    # Pad or truncate to ensure consistent size
                    if len(features) < self.max_features:
                        padded = np.zeros(self.max_features)
                        padded[:len(features)] = features
                        features = padded
                    else:
                        features = features[:self.max_features]
                    all_features.append(features)
                except:
                    all_features.append(np.zeros(self.max_features))
            else:
                all_features.append(np.zeros(self.max_features))

        return np.array(all_features)


# Also add this class if it's missing
class TrueMultilingualDetector:
    """True multilingual hate speech detector - dummy class for compatibility"""

    def __init__(self):
        self.feature_extractor = IntelligentFeatureExtractor()
        self.tfidf_extractor = MultilingualTfidfExtractor(max_features=2000)
        self.lang_detector = AdvancedLanguageDetector()
        self.models = {}
        self.final_ensemble = None
        self.is_trained = False


# Import Counter if not already imported
from collections import Counter
class AdvancedLanguageDetector:
    """Professional language detection using langdetect library"""

    def __init__(self):
        self.supported_languages = ['en', 'ar', 'tr']
        self.language_names = {'en': 'english', 'ar': 'arabic', 'tr': 'turkish'}

    def detect_language(self, text):
        """Detect language using langdetect library"""
        if not LANGDETECT_AVAILABLE:
            return self._fallback_detection(text)

        if pd.isna(text) or str(text).strip() == "":
            return 'unknown'

        try:
            # Clean text for better detection
            clean_text = re.sub(r'http[s]?://\S+', '', str(text))
            clean_text = re.sub(r'@\w+', '', clean_text)
            clean_text = re.sub(r'#\w+', '', clean_text)
            clean_text = clean_text.strip()

            if len(clean_text) < 3:
                return 'unknown'

            # Detect with confidence
            detected = detect(clean_text)

            # Map to our supported languages
            if detected in self.supported_languages:
                return self.language_names[detected]
            else:
                return 'english'  # Default fallback

        except Exception as e:
            return self._fallback_detection(text)

    def _fallback_detection(self, text):
        """Fallback language detection when langdetect is not available"""
        if pd.isna(text):
            return 'unknown'

        text = str(text)

        # Simple character-based detection
        arabic_chars = len(re.findall(r'[أبتثجحخدذرزسشصضطظعغفقكلمنهوي]', text))
        turkish_chars = len(re.findall(r'[çğıöşüÇĞIÖŞÜ]', text))
        english_chars = len(re.findall(r'[a-zA-Z]', text))

        total_chars = arabic_chars + turkish_chars + english_chars

        if total_chars == 0:
            return 'unknown'

        if arabic_chars / total_chars > 0.3:
            return 'arabic'
        elif turkish_chars / total_chars > 0.1:
            return 'turkish'
        else:
            return 'english'


class IntelligentFeatureExtractor(BaseEstimator, TransformerMixin):
    """Advanced feature extractor without manual word lists"""

    def __init__(self):
        self.lang_detector = AdvancedLanguageDetector()

        # Initialize sentiment analyzer if available
        if VADER_AVAILABLE:
            self.sentiment_analyzer = SentimentIntensityAnalyzer()

        # Regex patterns for advanced text analysis
        self.patterns = {
            'urls': re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'),
            'mentions': re.compile(r'@\w+'),
            'hashtags': re.compile(r'#\w+'),
            'repeated_chars': re.compile(r'(.)\1{2,}'),
            'caps_words': re.compile(r'\b[A-Z]{2,}\b'),
            'numbers': re.compile(r'\d+'),
            'emojis': re.compile(
                r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]'),
            'profanity_indicators': re.compile(r'\b\w*(?:fuck|shit|damn|hell|bitch|ass)\w*\b', re.IGNORECASE),
            'violence_indicators': re.compile(r'\b\w*(?:kill|die|death|murder|shoot|stab|hurt)\w*\b', re.IGNORECASE),
            'hate_indicators': re.compile(r'\b\w*(?:hate|stupid|idiot|ugly|disgusting|pathetic)\w*\b', re.IGNORECASE)
        }

    def extract_comprehensive_features(self, text):
        """Extract 75 intelligent features"""
        if pd.isna(text):
            return np.zeros(75)

        original_text = str(text)
        detected_lang = self.lang_detector.detect_language(original_text)
        text_lower = original_text.lower()
        words = text_lower.split()

        features = []

        # Basic statistics (15 features)
        features.extend([
            len(original_text),  # character count
            len(words),  # word count
            len(re.split(r'[.!?]+', text_lower)),  # sentence count
            len(original_text) / max(len(words), 1),  # avg word length
            len(words) / max(len(re.split(r'[.!?]+', text_lower)), 1),  # avg words per sentence
            len(set(words)) / max(len(words), 1),  # lexical diversity
            len([w for w in words if len(w) > 6]) / max(len(words), 1),  # long words ratio
            len([w for w in words if len(w) <= 3]) / max(len(words), 1),  # short words ratio
            len([w for w in words if w.isdigit()]) / max(len(words), 1),  # numeric words ratio
            sum(len(w) for w in words) / max(len(words), 1),  # actual avg word length
            original_text.count(' ') / max(len(original_text), 1),  # space density
            len([w for w in words if any(c.isdigit() for c in w)]) / max(len(words), 1),  # mixed alphanumeric ratio
            len(set(original_text.replace(' ', ''))) / max(len(original_text.replace(' ', '')), 1),
            # character diversity
            len([w for w in words if len(w) > 8]),  # very long words
            len([w for w in words if len(w) <= 2])  # very short words
        ])

        # Linguistic patterns (20 features)
        features.extend([
            len(self.patterns['urls'].findall(original_text)),
            len(self.patterns['mentions'].findall(original_text)),
            len(self.patterns['hashtags'].findall(original_text)),
            len(self.patterns['numbers'].findall(original_text)),
            len(self.patterns['repeated_chars'].findall(original_text)),
            len(self.patterns['caps_words'].findall(original_text)),
            original_text.count('!'),
            original_text.count('?'),
            original_text.count('.'),
            original_text.count(','),
            original_text.count('...'),
            len(self.patterns['emojis'].findall(original_text)),
            sum(1 for c in original_text if c.isupper()) / max(len(original_text), 1),
            original_text.count('!') / max(len(original_text), 1),
            original_text.count('?') / max(len(original_text), 1),
            len([c for c in original_text if not c.isalnum() and c != ' ']) / max(len(original_text), 1),
            len(re.findall(r'\s{2,}', original_text)),
            len(re.findall(r'(.)\1', text_lower)),
            text_lower.count('lol') + text_lower.count('haha'),
            len(re.findall(r'\b\w{1,2}\b', text_lower))
        ])

        # Language features (10 features)
        arabic_chars = len(re.findall(r'[\u0600-\u06FF]', original_text))
        turkish_chars = len(re.findall(r'[çğıöşüÇĞIÖŞÜ]', original_text))
        latin_chars = len(re.findall(r'[a-zA-Z]', original_text))

        features.extend([
            arabic_chars / max(len(original_text), 1),
            turkish_chars / max(len(original_text), 1),
            latin_chars / max(len(original_text), 1),
            1 if detected_lang == 'arabic' else 0,
            1 if detected_lang == 'turkish' else 0,
            1 if detected_lang == 'english' else 0,
            1 if arabic_chars > 0 and latin_chars > 0 else 0,  # code-switching
            1 if turkish_chars > 0 and latin_chars > 0 else 0,
            1 if len([c for c in [arabic_chars > 0, turkish_chars > 0, latin_chars > 0] if c]) > 1 else 0,
            (arabic_chars + turkish_chars + latin_chars) / max(len(original_text), 1)
        ])

        # Sentiment features (10 features)
        if VADER_AVAILABLE:
            try:
                scores = self.sentiment_analyzer.polarity_scores(original_text)
                sentiment_features = [scores['compound'], scores['pos'], scores['neu'], scores['neg']]
            except:
                sentiment_features = [0, 0, 0, 0]
        else:
            sentiment_features = [0, 0, 0, 0]

        # Simple sentiment indicators
        positive_indicators = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'love']
        negative_indicators = ['bad', 'terrible', 'awful', 'hate', 'dislike', 'angry']

        pos_count = sum(1 for word in positive_indicators if word in text_lower)
        neg_count = sum(1 for word in negative_indicators if word in text_lower)

        sentiment_features.extend([
            pos_count, neg_count,
            pos_count / max(len(words), 1),
            neg_count / max(len(words), 1),
            1 if pos_count > neg_count else 0,
            1 if neg_count > pos_count else 0
        ])

        features.extend(sentiment_features)

        # Readability features (8 features)
        if TEXTSTAT_AVAILABLE and len(original_text.strip()) > 10:
            try:
                readability_features = [
                    flesch_reading_ease(original_text),
                    flesch_kincaid_grade(original_text),
                    automated_readability_index(original_text)
                ]
            except:
                readability_features = [0, 0, 0]
        else:
            readability_features = [0, 0, 0]

        # Simple readability proxies
        sentences = re.split(r'[.!?]+', original_text)
        sentences = [s.strip() for s in sentences if s.strip()]

        readability_features.extend([
            len([w for w in words if len(w) > 2]) / max(len(words), 1),  # complex words ratio
            len(sentences) / max(len(original_text), 1) * 100,  # sentence density
            len([w for w in words if len(w) > 6]) / max(len(words), 1),  # long words ratio
            len(words) / max(len(sentences), 1),  # avg words per sentence
            len([w for w in words if len(w) > 8]) / max(len(words), 1)  # very long words ratio
        ])

        features.extend(readability_features)

        # Toxicity indicators (12 features)
        features.extend([
            len(self.patterns['profanity_indicators'].findall(text_lower)),
            len(self.patterns['violence_indicators'].findall(text_lower)),
            len(self.patterns['hate_indicators'].findall(text_lower)),
            1 if any(word in text_lower for word in ['kill', 'die', 'death']) else 0,
            1 if any(word in text_lower for word in ['you', 'your', 'u']) else 0,
            text_lower.count('!!!') + text_lower.count('???'),
            len(re.findall(r'[A-Z]{3,}', original_text)),
            1 if re.search(r'\b(go|fuck|get)\s+(away|off|out)\b', text_lower) else 0,
            len(re.findall(r'\b\w*(?:stupid|dumb|idiot)\w*\b', text_lower, re.IGNORECASE)),
            len(re.findall(r'\b\w*(?:ugly|fat|gross)\w*\b', text_lower, re.IGNORECASE)),
            1 if len(re.findall(r'(.)\1{3,}', text_lower)) > 0 else 0,
            len([w for w in text_lower.split() if len(w) > 10 and w.count(w[0]) > len(w) * 0.4])
        ])

        return np.array(features[:75])

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.array([self.extract_comprehensive_features(text) for text in X])


# Flask app setup
app = Flask(__name__)
CORS(app)

# Global variables
detector = None
model_info = {}


def load_intelligent_model():
    """Load the intelligent multilingual detector"""
    global detector, model_info

    try:
        model_path = "intelligent_multilingual_detector.pkl"

        if not os.path.exists(model_path):
            print(f"❌ Model file not found: {model_path}")
            print("💡 Make sure you have 'intelligent_multilingual_detector.pkl' in the same directory")
            return False

        print("📁 Loading intelligent multilingual model...")
        with open(model_path, 'rb') as f:
            model_package = pickle.load(f)

        # Create a simple detector class for API use
        class SimpleDetector:
            def __init__(self, model_package):
                self.ensemble = model_package['ensemble']
                self.feature_pipeline = model_package['feature_pipeline']
                self.feature_extractor = model_package['feature_extractor']
                self.tfidf_extractor = model_package['tfidf_extractor']
                self.lang_detector = model_package['lang_detector']
                self.feature_selector = model_package.get('feature_selector')
                self.models = model_package['models']
                self.is_trained = True

            def predict_intelligent(self, text):
                try:
                    # Language detection
                    detected_lang = self.lang_detector.detect_language(text)

                    # Feature extraction
                    X_features = self.feature_pipeline.transform([text])

                    # Apply feature selection if used during training
                    if self.feature_selector:
                        X_features = self.feature_selector.transform(X_features)

                    # Prediction
                    prediction = self.ensemble.predict(X_features)[0]
                    probabilities = self.ensemble.predict_proba(X_features)[0]

                    # Extract manual features for analysis
                    manual_features = self.feature_extractor.extract_comprehensive_features(text)

                    return {
                        "text": text,
                        "detected_language": detected_lang,
                        "prediction": "Hate Speech" if prediction == 1 else "Not Hate Speech",
                        "confidence": float(max(probabilities)),
                        "hate_probability": float(probabilities[1]) if len(probabilities) > 1 else 0.0,
                        "not_hate_probability": float(probabilities[0]),
                        "intelligent_analysis": {
                            "text_length": int(manual_features[0]),
                            "word_count": int(manual_features[1]),
                            "sentence_count": int(manual_features[2]),
                            "lexical_diversity": float(manual_features[5]),
                            "arabic_char_density": float(manual_features[25]),
                            "turkish_char_density": float(manual_features[26]),
                            "is_multilingual": bool(manual_features[33]),
                            "sentiment_compound": float(manual_features[35]),
                            "positive_score": float(manual_features[36]),
                            "negative_score": float(manual_features[38]),
                            "profanity_indicators": int(manual_features[63]),
                            "violence_indicators": int(manual_features[64]),
                            "hate_indicators": int(manual_features[65]),
                            "personal_attacks": bool(manual_features[67])
                        },
                        "success": True
                    }
                except Exception as e:
                    return {"text": text, "error": str(e), "success": False}

        detector = SimpleDetector(model_package)
        model_info = model_package.get('model_info', {})

        print("✅ Intelligent model loaded successfully!")
        print(f"🔧 Model version: {model_info.get('version', 'unknown')}")
        print(f"🌍 Languages: {model_info.get('languages', ['unknown'])}")
        print(
            f"📊 Features: {model_info.get('manual_features', 0)} manual + {model_info.get('tfidf_features', 0)} TF-IDF")
        return True

    except Exception as e:
        print(f"❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return False


def get_risk_assessment(hate_probability):
    """Enhanced risk assessment"""
    if hate_probability >= 0.9:
        return {
            "level": "CRITICAL",
            "color": "#FF0000",
            "message": "Very High Risk - Immediate Action Required",
            "description": "This content shows strong indicators of hate speech"
        }
    elif hate_probability >= 0.8:
        return {
            "level": "HIGH",
            "color": "#FF4444",
            "message": "High Risk - Likely Hate Speech",
            "description": "This content is very likely to be hate speech"
        }
    elif hate_probability >= 0.6:
        return {
            "level": "MEDIUM",
            "color": "#FF8800",
            "message": "Medium Risk - Potentially Problematic",
            "description": "This content may contain problematic language"
        }
    elif hate_probability >= 0.4:
        return {
            "level": "LOW",
            "color": "#FFAA00",
            "message": "Low Risk - Minor Concern",
            "description": "This content shows minor concerning patterns"
        }
    else:
        return {
            "level": "SAFE",
            "color": "#00AA44",
            "message": "Safe - No Significant Risk",
            "description": "This content appears to be safe"
        }


@app.route('/', methods=['GET'])
def home():
    """API status endpoint"""
    return jsonify({
        'status': 'online',
        'message': 'Intelligent Multilingual Hate Speech Detection API',
        'version': model_info.get('version', '3.0'),
        'model_loaded': detector is not None,
        'model_type': model_info.get('type', 'intelligent_multilingual'),
        'supported_languages': model_info.get('languages', ['english', 'arabic', 'turkish']),
        'features': {
            'manual_features': model_info.get('manual_features', 75),
            'tfidf_features': model_info.get('tfidf_features', 2000),
            'total_features': model_info.get('manual_features', 75) + model_info.get('tfidf_features', 2000)
        },
        'dependencies': model_info.get('dependencies', {}),
        'server': 'Flask Development Server'
    })


@app.route('/health', methods=['GET'])
def health():
    """Detailed health check"""
    return jsonify({
        "status": "healthy",
        "model_loaded": detector is not None,
        "model_ready": detector is not None and detector.is_trained,
        "langdetect_available": LANGDETECT_AVAILABLE,
        "textstat_available": TEXTSTAT_AVAILABLE,
        "vader_available": VADER_AVAILABLE,
        "timestamp": pd.Timestamp.now().isoformat()
    })


@app.route('/predict', methods=['POST'])
def predict():
    """Main intelligent prediction endpoint"""
    try:
        # Check if model is loaded
        if detector is None:
            return jsonify({
                'success': False,
                'error': 'Intelligent model not loaded. Please restart the server.',
                'prediction': 'Error',
                'confidence': 0.0
            }), 500

        # Get request data
        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'error': 'No JSON data provided',
                'prediction': 'Error',
                'confidence': 0.0
            }), 400

        # Extract text
        text = data.get('text', '').strip()
        if not text:
            return jsonify({
                'success': False,
                'error': 'No text provided',
                'prediction': 'Error',
                'confidence': 0.0
            }), 400

        # Validate text length
        if len(text) < 3:
            return jsonify({
                'success': False,
                'error': 'Text too short (minimum 3 characters)',
                'prediction': 'Error',
                'confidence': 0.0
            }), 400

        if len(text) > 1000:
            return jsonify({
                'success': False,
                'error': 'Text too long (maximum 1000 characters)',
                'prediction': 'Error',
                'confidence': 0.0
            }), 400

        # Make intelligent prediction
        print(f"🔍 Analyzing: {text[:50]}...")
        result = detector.predict_intelligent(text)

        if not result['success']:
            return jsonify({
                'success': False,
                'error': result.get('error', 'Unknown prediction error'),
                'prediction': 'Error',
                'confidence': 0.0
            }), 500

        # Get risk assessment
        risk_assessment = get_risk_assessment(result['hate_probability'])

        # Prepare enhanced response for Android
        response = {
            'success': True,
            'prediction': result['prediction'],
            'confidence': result['confidence'],
            'hate_probability': result['hate_probability'],
            'detected_language': result['detected_language'],
            'risk_level': risk_assessment['level'],
            'risk_color': risk_assessment['color'],
            'risk_message': risk_assessment['message'],

            # Analysis details for Android UI
            'analysis': {
                'text_length': result['intelligent_analysis']['text_length'],
                'word_count': result['intelligent_analysis']['word_count'],
                'sentence_count': result['intelligent_analysis']['sentence_count'],
                'sentiment_score': result['intelligent_analysis']['sentiment_compound'],
                'lexical_diversity': result['intelligent_analysis']['lexical_diversity'],
                'toxicity_indicators': {
                    'profanity': result['intelligent_analysis']['profanity_indicators'],
                    'violence': result['intelligent_analysis']['violence_indicators'],
                    'hate_words': result['intelligent_analysis']['hate_indicators'],
                    'personal_attacks': result['intelligent_analysis']['personal_attacks']
                },
                'language_features': {
                    'arabic_ratio': result['intelligent_analysis']['arabic_char_density'],
                    'turkish_ratio': result['intelligent_analysis']['turkish_char_density'],
                    'is_multilingual': result['intelligent_analysis']['is_multilingual']
                }
            },

            'model_info': {
                'version': model_info.get('version', '3.0'),
                'type': 'intelligent_multilingual',
                'processing_time': 'optimized'
            },

            'timestamp': pd.Timestamp.now().isoformat()
        }

        print(f"✅ Analysis complete: {result['prediction']} ({result['confidence']:.3f})")
        return jsonify(response)

    except Exception as e:
        error_msg = f"Server error: {str(e)}"
        print(f"❌ {error_msg}")
        import traceback
        traceback.print_exc()

        return jsonify({
            'success': False,
            'error': error_msg,
            'prediction': 'Error',
            'confidence': 0.0
        }), 500


@app.route('/languages', methods=['GET'])
def supported_languages():
    """Get supported languages with details"""
    return jsonify({
        "languages": [
            {"code": "en", "name": "English", "native": "English"},
            {"code": "ar", "name": "Arabic", "native": "العربية"},
            {"code": "tr", "name": "Turkish", "native": "Türkçe"}
        ],
        "auto_detection": True,
        "detection_method": "langdetect" if LANGDETECT_AVAILABLE else "fallback"
    })


@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'success': False,
        'error': 'Endpoint not found',
        'available_endpoints': {
            'GET /': 'API status and info',
            'GET /health': 'Detailed health check',
            'POST /predict': 'Intelligent hate speech detection',
            'GET /languages': 'Supported languages info'
        }
    }), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'success': False,
        'error': 'Internal server error',
        'message': 'Please check server logs for details'
    }), 500


if __name__ == '__main__':
    print("🧠 Starting Intelligent Multilingual API Server")
    print("=" * 65)

    # Check dependencies
    print("📋 Checking dependencies...")
    deps = {
        'langdetect': LANGDETECT_AVAILABLE,
        'textstat': TEXTSTAT_AVAILABLE,
        'vaderSentiment': VADER_AVAILABLE
    }

    for dep, available in deps.items():
        status = "✅" if available else "❌"
        print(f"   {status} {dep}")

    # Load intelligent model
    if load_intelligent_model():
        print("\n📡 API Endpoints:")
        print("   GET  /          - API status and model info")
        print("   GET  /health    - Detailed health check")
        print("   POST /predict   - Intelligent hate speech detection")
        print("   GET  /languages - Supported languages")

        print(f"\n🌐 Server URLs:")
        print(f"   Local: http://localhost:5000")
        print(f"   Android Emulator: http://10.0.2.2:5000")

        print(f"\n🚀 Starting Flask server...")
        print(f"   Press Ctrl+C to stop")
        print("-" * 65)

        # Run Flask app
        app.run(
            host='0.0.0.0',  # Allow external connections
            port=5000,
            debug=True  # Enable debug mode for development
        )
    else:
        print("❌ Failed to load intelligent model!")
        print("💡 Make sure 'intelligent_multilingual_detector.pkl' is in the same directory")
        input("Press Enter to exit...")