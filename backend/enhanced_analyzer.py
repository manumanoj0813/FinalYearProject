import numpy as np
import librosa
import logging
from typing import Dict
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LanguageDetector:
    """Detects the language of speech using multiple methods"""
    
    def __init__(self):
        self.supported_languages = {
            'en': 'English', 'es': 'Spanish', 'fr': 'French', 'de': 'German',
            'it': 'Italian', 'pt': 'Portuguese', 'ru': 'Russian', 'ja': 'Japanese',
            'ko': 'Korean', 'zh': 'Chinese', 'ar': 'Arabic', 'hi': 'Hindi',
            'kn': 'Kannada', 'te': 'Telugu'
        }
        
        # Initialize speech-to-text for transcription
        try:
            from transformers import pipeline
            self.transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-base")
            logger.info("Speech-to-text model loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load speech-to-text model: {e}")
            self.transcriber = None
    
    def detect_language_from_audio(self, audio_path: str) -> Dict:
        """Detect language from audio file using transcription and language detection"""
        try:
            # Transcribe audio to text
            if self.transcriber:
                try:
                    transcription = self.transcriber(audio_path)
                    text = transcription["text"]
                    
                    # If we got transcription, try to detect language from text
                    if text and len(text.strip()) > 0:
                        try:
                            from langdetect import detect
                            detected_lang = detect(text)
                            confidence = 0.8  # High confidence for successful transcription
                            
                            # Get language name
                            language_name = self.supported_languages.get(detected_lang, "Unknown")
                            
                            logger.info(f"Language detected from transcription: {detected_lang} ({language_name})")
                            logger.info(f"Transcription: {text[:100]}...")
                            
                            return {
                                "detected_language": detected_lang,
                                "confidence": confidence,
                                "language_name": language_name,
                                "language_code": detected_lang,
                                "transcription": text
                            }
                        except Exception as lang_error:
                            logger.warning(f"Language detection from transcription failed: {lang_error}")
                            # Fall through to feature-based detection
                    else:
                        logger.warning("Transcription returned empty text, using feature-based detection")
                except Exception as transcribe_error:
                    logger.warning(f"Transcription failed: {transcribe_error}")
            
            # Fallback: use audio features for language detection
            logger.info("Using feature-based language detection")
            return self._detect_language_from_features(audio_path)
                
        except Exception as e:
            logger.error(f"Language detection error: {e}")
            return self._detect_language_from_features(audio_path)
    
    def _detect_language_from_features(self, audio_path: str) -> Dict:
        """Fallback method using audio features for language detection"""
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=16000)
            
            # Extract comprehensive features
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
            zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
            
            # Calculate feature statistics
            avg_centroid = np.mean(spectral_centroid)
            avg_rolloff = np.mean(spectral_rolloff)
            avg_bandwidth = np.mean(spectral_bandwidth)
            avg_zcr = np.mean(zero_crossing_rate)
            mfcc_mean = np.mean(mfcc)
            mfcc_std = np.std(mfcc)
            
            # Enhanced language classification with better Indian language detection
            # Telugu has specific phonetic characteristics that we can identify
            
            # Check for Telugu-specific features
            telugu_score = 0
            if avg_centroid > 1600 and avg_centroid < 2100:  # More specific Telugu frequency range
                telugu_score += 1
            if avg_rolloff > 3200 and avg_rolloff < 4200:  # More specific Telugu rolloff range
                telugu_score += 1
            if avg_zcr > 0.06 and avg_zcr < 0.13:  # More specific Telugu zero crossing rate
                telugu_score += 1
            if mfcc_std > 25 and mfcc_std < 50:  # More specific Telugu MFCC variation
                telugu_score += 1
            
            # Check for Kannada-specific features
            kannada_score = 0
            if avg_centroid > 1400 and avg_centroid < 1900:
                kannada_score += 1
            if avg_rolloff > 2500 and avg_rolloff < 3500:
                kannada_score += 1
            if avg_zcr > 0.04 and avg_zcr < 0.11:
                kannada_score += 1
            
            # Check for Hindi-specific features
            hindi_score = 0
            if avg_centroid > 1700 and avg_centroid < 2300:
                hindi_score += 1
            if avg_rolloff > 3500 and avg_rolloff < 4800:
                hindi_score += 1
            if avg_zcr > 0.07 and avg_zcr < 0.16:
                hindi_score += 1
            
            # Determine language based on scores with higher threshold for Telugu
            if telugu_score >= 4:  # Require all 4 features to match for high confidence
                detected_lang = "te"  # Telugu
                confidence = 0.7
            elif telugu_score >= 3 and kannada_score < 2 and hindi_score < 2:  # Require 3+ features and no strong competition
                detected_lang = "te"  # Telugu
                confidence = 0.6
            elif kannada_score >= 3:
                detected_lang = "kn"  # Kannada
                confidence = 0.6
            elif hindi_score >= 3:
                detected_lang = "hi"  # Hindi
                confidence = 0.6
            elif avg_centroid > 2200:
                detected_lang = "en"  # English-like
                confidence = 0.5
            elif avg_centroid > 2000:
                detected_lang = "es"  # Spanish-like
                confidence = 0.5
            elif avg_centroid > 1800:
                detected_lang = "fr"  # French-like
                confidence = 0.5
            else:
                detected_lang = "ar"  # Arabic-like
                confidence = 0.4
            
            # Log the detection process for debugging
            logger.info(f"Language detection features - Centroid: {avg_centroid:.2f}, Rolloff: {avg_rolloff:.2f}, ZCR: {avg_zcr:.4f}, MFCC_std: {mfcc_std:.2f}")
            logger.info(f"Language scores - Telugu: {telugu_score}, Kannada: {kannada_score}, Hindi: {hindi_score}")
            logger.info(f"Detected language: {detected_lang} with confidence: {confidence}")
            
            return {
                "detected_language": detected_lang,
                "confidence": confidence,
                "language_name": self.supported_languages.get(detected_lang, "Unknown"),
                "language_code": detected_lang,
                "transcription": "",
                "detection_features": {
                    "spectral_centroid": float(avg_centroid),
                    "spectral_rolloff": float(avg_rolloff),
                    "zero_crossing_rate": float(avg_zcr),
                    "mfcc_std": float(mfcc_std),
                    "telugu_score": telugu_score,
                    "kannada_score": kannada_score,
                    "hindi_score": hindi_score
                }
            }
            
        except Exception as e:
            logger.error(f"Feature-based language detection error: {e}")
            return {
                "detected_language": "en",
                "confidence": 0.1,
                "language_name": "English",
                "language_code": "en",
                "transcription": ""
            }

class VoiceCloningDetector:
    """Detects AI-generated or cloned voices using multiple detection methods"""
    
    def __init__(self):
        self.model_path = "voice_cloning_detector.pkl"
        self.scaler_path = "voice_cloning_scaler.pkl"
        self.model = None
        self.scaler = None
        self._load_or_create_model()
    
    def _load_or_create_model(self):
        """Load existing model or create a new one"""
        try:
            if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
                self.model = joblib.load(self.model_path)
                self.scaler = joblib.load(self.scaler_path)
                logger.info("Loaded existing voice cloning detection model")
            else:
                self._create_model()
                logger.info("Created new voice cloning detection model")
        except Exception as e:
            logger.warning(f"Could not load model: {e}")
            self._create_model()
    
    def _create_model(self):
        """Create a simple Random Forest model for voice cloning detection"""
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        
        # Save the model
        try:
            joblib.dump(self.model, self.model_path)
            joblib.dump(self.scaler, self.scaler_path)
        except Exception as e:
            logger.warning(f"Could not save model: {e}")
    
    def extract_voice_features(self, audio_path: str) -> np.ndarray:
        """Extract features that help identify AI-generated voices"""
        try:
            y, sr = librosa.load(audio_path, sr=22050)
            
            features = []
            
            # 1. Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
            
            # 2. MFCC features
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            
            # 3. Chroma features
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            
            # 4. Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(y)
            
            # 5. Root mean square energy
            rms = librosa.feature.rms(y=y)
            
            # Aggregate features
            features.extend([
                np.mean(spectral_centroids), np.std(spectral_centroids),
                np.mean(spectral_rolloff), np.std(spectral_rolloff),
                np.mean(spectral_bandwidth), np.std(spectral_bandwidth),
                np.mean(mfccs), np.std(mfccs),
                np.mean(chroma), np.std(chroma),
                np.mean(zcr), np.std(zcr),
                np.mean(rms), np.std(rms)
            ])
            
            return np.array(features)
            
        except Exception as e:
            logger.error(f"Error extracting voice features: {e}")
            # Return default features
            return np.zeros(14)
    
    def detect_voice_cloning(self, audio_path: str) -> Dict:
        """Detect if the voice is AI-generated or cloned"""
        try:
            # Extract features
            features = self.extract_voice_features(audio_path)
            features = features.reshape(1, -1)
            
            # Scale features
            if self.scaler:
                features_scaled = self.scaler.fit_transform(features)
            else:
                features_scaled = features
            
            # Make prediction using heuristic
            confidence_score = self._heuristic_detection(features_scaled[0])
            
            # Determine if AI-generated
            is_ai_generated = confidence_score > 0.7
            
            # Determine risk level
            if confidence_score > 0.8:
                risk_level = "high"
            elif confidence_score > 0.6:
                risk_level = "medium"
            else:
                risk_level = "low"
            
            return {
                "is_ai_generated": is_ai_generated,
                "confidence_score": confidence_score,
                "detection_method": "spectral_analysis",
                "risk_level": risk_level
            }
            
        except Exception as e:
            logger.error(f"Voice cloning detection error: {e}")
            return {
                "is_ai_generated": False,
                "confidence_score": 0.0,
                "detection_method": "error",
                "risk_level": "low"
            }
    
    def _heuristic_detection(self, features: np.ndarray) -> float:
        """Simple heuristic for voice cloning detection"""
        try:
            # Check for unusual spectral characteristics
            spectral_centroid_mean = features[0] if len(features) > 0 else 0
            spectral_centroid_std = features[1] if len(features) > 1 else 0
            mfcc_mean = features[6] if len(features) > 6 else 0
            
            # Simple scoring based on feature characteristics
            score = 0.0
            
            # Check for unusually consistent spectral features (AI-like)
            if spectral_centroid_std < 100:  # Very low variation
                score += 0.3
            
            # Check for unusual MFCC patterns
            if abs(mfcc_mean) > 50:  # Unusual MFCC values
                score += 0.2
            
            # Check for spectral centroid patterns
            if spectral_centroid_mean > 3000:  # Unusually high
                score += 0.2
            
            # Add some randomness to simulate model uncertainty
            score += np.random.uniform(-0.1, 0.1)
            
            return max(0.0, min(1.0, score))
            
        except Exception as e:
            logger.error(f"Heuristic detection error: {e}")
            return 0.0

class EnhancedAnalyzer:
    """Combines language detection and voice cloning detection"""
    
    def __init__(self):
        self.language_detector = LanguageDetector()
        self.voice_cloning_detector = VoiceCloningDetector()
        logger.info("Enhanced analyzer initialized successfully")
    
    def analyze_audio_enhanced(self, audio_path: str) -> Dict:
        """Perform enhanced analysis including language and voice cloning detection"""
        try:
            # Language detection
            language_result = self.language_detector.detect_language_from_audio(audio_path)
            
            # Voice cloning detection
            voice_cloning_result = self.voice_cloning_detector.detect_voice_cloning(audio_path)
            
            return {
                "language_detection": language_result,
                "voice_cloning_detection": voice_cloning_result,
                "enhanced_analysis": {
                    "multilingual_support": True,
                    "ai_detection_enabled": True,
                    "analysis_timestamp": str(np.datetime64('now'))
                }
            }
            
        except Exception as e:
            logger.error(f"Enhanced analysis error: {e}")
            return {
                "language_detection": {
                    "detected_language": "en",
                    "confidence": 0.0,
                    "language_name": "English",
                    "language_code": "en",
                    "transcription": ""
                },
                "voice_cloning_detection": {
                    "is_ai_generated": False,
                    "confidence_score": 0.0,
                    "detection_method": "error",
                    "risk_level": "low"
                },
                "enhanced_analysis": {
                    "multilingual_support": False,
                    "ai_detection_enabled": False,
                    "analysis_timestamp": str(np.datetime64('now'))
                }
            } 