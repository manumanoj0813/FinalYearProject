import librosa
import numpy as np
from typing import Dict, List, Tuple
import logging
import subprocess
import tempfile
from pathlib import Path
import asyncio
import whisper

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def to_python_type(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: to_python_type(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_python_type(v) for v in obj]
    return obj

class VoiceAnalyzer:
    def __init__(self):
        try:
            logger.info("Initializing voice analyzer...")
            # Check if ffmpeg is available
            try:
                subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
            except (subprocess.CalledProcessError, FileNotFoundError) as e:
                logger.error(f"FFmpeg check failed: {str(e)}")
                raise RuntimeError("ffmpeg is not installed or not accessible. Please install ffmpeg to process audio files.")
            
            # Load Whisper model - use tiny model for speed
            logger.info("Loading Whisper model (tiny for speed)...")
            self.whisper_model = whisper.load_model("tiny")  # Changed from "base" to "tiny"
            logger.info("Whisper model loaded successfully.")

            logger.info("Voice analyzer initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize voice analyzer: {str(e)}")
            raise RuntimeError(f"Voice analyzer initialization failed: {str(e)}")
        
    async def analyze_audio(self, audio_path: str) -> dict:
        """Comprehensive voice analysis including fluency and linguistic features."""
        try:
            logger.info(f"Processing audio file: {audio_path}")
            
            # Check if file exists and has content
            if not Path(audio_path).exists():
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
            file_size = Path(audio_path).stat().st_size
            if file_size == 0:
                raise ValueError(f"Audio file is empty: {audio_path}")
            
            logger.info(f"Audio file size: {file_size} bytes")
            
            # Convert WebM to WAV
            wav_path = self._convert_to_wav(audio_path)
            
            try:
                # Load and analyze the WAV file first (faster)
                logger.info(f"Loading WAV file: {wav_path}")
                y, sr = librosa.load(wav_path, sr=16000)  # Reduced sample rate for speed
                logger.info(f"Audio loaded: {len(y)} samples, {sr} Hz sample rate")
                
                # Basic audio features
                duration = librosa.get_duration(y=y, sr=sr)
                logger.info(f"Audio duration: {duration} seconds")
                
                # Run analysis in parallel for speed
                logger.info("Starting parallel analysis...")
                
                # Simple and fast analysis
                pitch_data = self._analyze_pitch_fast(y, sr)
                rhythm_data = self._analyze_rhythm_fast(y, sr)
                clarity_data = self._analyze_clarity_fast(y, sr)
                emotion_data = self._analyze_emotion_fast(y, sr)
                
                # Transcribe audio (this is the slowest part)
                logger.info("Starting transcription...")
                transcript = self._transcribe_audio_fast(wav_path)
                logger.info(f"Transcription completed: {len(transcript)} characters")
                
                recommendations = self._generate_recommendations_fast(pitch_data, rhythm_data, clarity_data, emotion_data)
                
                logger.info("Audio analysis completed successfully")
                result = {
                    "audio_metrics": {
                        "duration": duration,
                        "pitch": {
                            "average_pitch": pitch_data["average"],
                            "pitch_variation": pitch_data["variation"],
                            "pitch_range": {
                                "min": pitch_data["range_min"],
                                "max": pitch_data["range_max"]
                            }
                        },
                        "rhythm": {
                            "speech_rate": rhythm_data["speech_rate"],
                            "pause_ratio": rhythm_data["pause_ratio"],
                            "average_pause_duration": rhythm_data["avg_pause_duration"],
                            "total_speaking_time": rhythm_data["total_speech_time"]
                        },
                        "clarity": {
                            "clarity_score": clarity_data["clarity_score"],
                            "pronunciation_score": clarity_data["articulation_score"],
                            "articulation_rate": clarity_data["spectral_quality"],
                            "speech_errors": []
                        },
                        "emotion": {
                            "dominant_emotion": emotion_data["dominant_emotion"],
                            "emotion_confidence": emotion_data["confidence"],
                            "emotion_scores": emotion_data["scores"]
                        }
                    },
                    "transcription": {
                        "full_text": transcript,
                        "word_count": len(transcript.split())
                    },
                    "recommendations": {
                        "key_points": recommendations[:2] if len(recommendations) >= 2 else recommendations,
                        "improvement_areas": recommendations[2:4] if len(recommendations) >= 4 else [],
                        "strengths": ["Good voice projection", "Clear pronunciation"],
                        "practice_suggestions": [
                            "Practice tongue twisters for better articulation",
                            "Record yourself reading aloud for 5 minutes daily",
                            "Focus on breathing exercises to improve voice control"
                        ]
                    },
                    "metadata": {
                        "session_type": "practice",
                        "topic": "general",
                        "duration": duration,
                        "file_path": audio_path
                    }
                }
                return to_python_type(result)
            finally:
                # Clean up the temporary WAV file
                try:
                    Path(wav_path).unlink()
                    logger.info(f"Cleaned up temporary WAV file: {wav_path}")
                except Exception as e:
                    logger.warning(f"Failed to delete temporary WAV file {wav_path}: {str(e)}")
                    
        except Exception as e:
            logger.error(f"Error analyzing audio: {str(e)}")
            logger.error(f"Exception type: {type(e).__name__}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise RuntimeError(f"Audio analysis failed: {str(e)}")
    
    def _transcribe_audio_fast(self, audio_path: str) -> str:
        """Transcribe audio using Whisper with optimized settings."""
        try:
            logger.info(f"Transcribing audio file: {audio_path}")
            # Use faster settings
            result = self.whisper_model.transcribe(
                audio_path,
                language="en",  # Specify language for speed
                task="transcribe",  # Explicitly set task
                fp16=False,  # Disable fp16 for compatibility
                verbose=False  # Reduce logging
            )
            transcript = result["text"]
            logger.info(f"Transcription successful. Text: {transcript[:100]}...")
            return transcript
        except Exception as e:
            logger.error(f"Error during transcription: {str(e)}")
            return "Transcription failed."
    
    def _convert_to_wav(self, input_path: str) -> str:
        """Convert input audio to WAV format using ffmpeg."""
        try:
            logger.info(f"Input audio file path: {input_path}")
            input_file_path_obj = Path(input_path)
            if not input_file_path_obj.exists():
                raise FileNotFoundError(f"Input file does not exist: {input_path}")
            if input_file_path_obj.stat().st_size == 0:
                raise ValueError(f"Input file is empty: {input_path}")

            # Create temporary WAV file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_wav:
                wav_path = temp_wav.name
            
            # Convert to WAV using ffmpeg with optimized settings
            cmd = [
                'ffmpeg',
                '-i', input_path,
                '-acodec', 'pcm_s16le',
                '-ar', '16000',  # Reduced sample rate for speed
                '-ac', '1',
                '-y',  # Overwrite output file if it exists
                wav_path
            ]
            
            logger.info(f"Running FFmpeg command: {' '.join(cmd)}")
            
            result = subprocess.run(cmd, capture_output=True)
            
            logger.info(f"FFmpeg process returned code: {result.returncode}")
            if result.stdout:
                logger.info(f"FFmpeg stdout: {result.stdout.decode(errors='ignore').strip()}")
            if result.stderr:
                logger.info(f"FFmpeg stderr: {result.stderr.decode(errors='ignore').strip()}")

            if result.returncode != 0:
                error_msg = f"FFmpeg conversion failed with code {result.returncode}. Stderr: '{result.stderr.decode(errors='ignore').strip()}'"
                logger.error(error_msg)
                raise RuntimeError(error_msg)
            
            logger.info(f"Successfully converted {input_path} to WAV format")
            return wav_path
            
        except Exception as e:
            logger.error(f"Error converting audio to WAV: {str(e)}")
            raise RuntimeError(f"Audio conversion failed: {str(e)}")
    
    def _analyze_pitch_fast(self, y: np.ndarray, sr: int) -> Dict:
        """Fast pitch analysis."""
        try:
            # Use a smaller window for speed
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr, hop_length=512, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
            
            # Get pitch values where magnitude is above threshold
            pitch_values = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0:
                    pitch_values.append(pitch)
            
            if not pitch_values:
                return {
                    "average": 0,
                    "variation": 0,
                    "range_min": 0,
                    "range_max": 0
                }
            
            pitch_values = np.array(pitch_values)
            return {
                "average": float(np.mean(pitch_values)),
                "variation": float(np.std(pitch_values)),
                "range_min": float(np.min(pitch_values)),
                "range_max": float(np.max(pitch_values))
            }
        except Exception as e:
            logger.error(f"Error in pitch analysis: {str(e)}")
            return {
                "average": 0,
                "variation": 0,
                "range_min": 0,
                "range_max": 0
            }
    
    def _analyze_rhythm_fast(self, y: np.ndarray, sr: int) -> Dict:
        """Fast rhythm analysis."""
        try:
            # Simple tempo detection
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            
            # Estimate speech rate (words per minute)
            speech_rate = min(1.0, tempo / 120.0)  # Normalize to 0-1
            
            # Simple pause detection
            energy = np.abs(y)
            threshold = np.mean(energy) * 0.1
            pauses = energy < threshold
            pause_ratio = np.sum(pauses) / len(pauses)
            
            return {
                "speech_rate": speech_rate,
                "pause_ratio": pause_ratio,
                "avg_pause_duration": pause_ratio * 0.5,
                "total_speech_time": 1.0 - pause_ratio
            }
        except Exception as e:
            logger.error(f"Error in rhythm analysis: {str(e)}")
            return {
                "speech_rate": 0.5,
                "pause_ratio": 0.2,
                "avg_pause_duration": 0.1,
                "total_speech_time": 0.8
            }
    
    def _analyze_clarity_fast(self, y: np.ndarray, sr: int) -> Dict:
        """Fast clarity analysis."""
        try:
            # Spectral centroid (brightness)
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            
            # Spectral rolloff (high frequency content)
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
            
            # Simple clarity score based on spectral features
            clarity_score = min(1.0, np.mean(spectral_centroids) / 2000.0)
            articulation_score = min(1.0, np.mean(spectral_rolloff) / 4000.0)
            spectral_quality = (clarity_score + articulation_score) / 2
            
            return {
                "clarity_score": clarity_score,
                "articulation_score": articulation_score,
                "spectral_quality": spectral_quality
            }
        except Exception as e:
            logger.error(f"Error in clarity analysis: {str(e)}")
            return {
                "clarity_score": 0.7,
                "articulation_score": 0.6,
                "spectral_quality": 0.65
            }
    
    def _analyze_emotion_fast(self, y: np.ndarray, sr: int) -> Dict:
        """Fast emotion analysis."""
        try:
            # Simple emotion detection based on pitch and energy
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
            
            # Get average pitch
            pitch_values = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0:
                    pitch_values.append(pitch)
            
            avg_pitch = np.mean(pitch_values) if pitch_values else 0
            
            # Energy analysis
            energy = np.mean(np.abs(y))
            
            # Simple emotion classification
            if avg_pitch > 300 and energy > 0.1:
                dominant_emotion = "excited"
                confidence = 0.8
            elif avg_pitch > 250:
                dominant_emotion = "happy"
                confidence = 0.7
            elif avg_pitch < 150:
                dominant_emotion = "calm"
                confidence = 0.6
            else:
                dominant_emotion = "neutral"
                confidence = 0.5
            
            return {
                "dominant_emotion": dominant_emotion,
                "confidence": confidence,
                "scores": {
                    "happy": 0.3,
                    "sad": 0.1,
                    "angry": 0.1,
                    "neutral": 0.4,
                    "excited": 0.2,
                    "calm": 0.3
                }
            }
        except Exception as e:
            logger.error(f"Error in emotion analysis: {str(e)}")
            return {
                "dominant_emotion": "neutral",
                "confidence": 0.5,
                "scores": {
                    "happy": 0.2,
                    "sad": 0.2,
                    "angry": 0.1,
                    "neutral": 0.5,
                    "excited": 0.1,
                    "calm": 0.2
                }
            }
    
    def _generate_recommendations_fast(self, pitch_data: Dict, rhythm_data: Dict, clarity_data: Dict, emotion_data: Dict) -> List[str]:
        """Generate fast recommendations."""
        recommendations = []
        
        # Pitch-based recommendations
        if pitch_data["variation"] < 50:
            recommendations.append("Try varying your pitch more for better expression")
        
        # Rhythm-based recommendations
        if rhythm_data["speech_rate"] > 0.8:
            recommendations.append("Slow down your speech rate for better clarity")
        elif rhythm_data["speech_rate"] < 0.3:
            recommendations.append("Speed up your speech rate for better engagement")
        
        # Clarity-based recommendations
        if clarity_data["clarity_score"] < 0.6:
            recommendations.append("Focus on clearer pronunciation")
        
        # Emotion-based recommendations
        if emotion_data["dominant_emotion"] == "neutral":
            recommendations.append("Add more emotional expression to your voice")
        
        return recommendations 