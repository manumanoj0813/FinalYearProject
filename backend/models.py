from typing import Optional
from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
from typing import Dict, List
from pydantic import BaseModel, Field
from bson import ObjectId

Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    preferred_language = Column(String, default="en")  # New field for language preference
    
    # Relationships
    recordings = relationship("Recording", back_populates="user")
    practice_sessions = relationship("PracticeSession", back_populates="user")

class Recording(Base):
    __tablename__ = "recordings"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    file_path = Column(String)
    duration = Column(Float)
    language = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Analysis results
    pitch_metrics = Column(JSON)
    rhythm_metrics = Column(JSON)
    clarity_metrics = Column(JSON)
    emotion_metrics = Column(JSON)
    recommendations = Column(JSON)
    
    # New fields for enhanced features
    detected_language = Column(String)  # Auto-detected language
    voice_cloning_score = Column(Float)  # AI voice detection score
    transcription = Column(String)  # Speech-to-text result
    
    # Relationships
    user = relationship("User", back_populates="recordings")
    practice_session = relationship("PracticeSession", back_populates="recordings")

class PracticeSession(Base):
    __tablename__ = "practice_sessions"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    session_type = Column(String)  # e.g., "interview", "presentation", "reading"
    topic = Column(String)
    start_time = Column(DateTime, default=datetime.utcnow)
    end_time = Column(DateTime)
    
    # Session metrics
    average_clarity = Column(Float)
    average_confidence = Column(Float)
    average_speech_rate = Column(Float)
    dominant_emotion = Column(String)
    improvement_areas = Column(JSON)
    
    # New fields
    language = Column(String)  # Session language
    voice_cloning_detected = Column(String, default="human")  # human/ai/uncertain
    
    # Relationships
    user = relationship("User", back_populates="practice_sessions")
    recordings = relationship("Recording", back_populates="practice_session")

class ProgressMetrics(Base):
    __tablename__ = "progress_metrics"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    metric_date = Column(DateTime, default=datetime.utcnow)
    
    # Aggregated metrics
    clarity_trend = Column(Float)
    confidence_trend = Column(Float)
    speech_rate_trend = Column(Float)
    emotion_expression_score = Column(Float)
    vocabulary_score = Column(Float)
    overall_improvement = Column(Float)
    
    # Goals and achievements
    current_goals = Column(JSON)
    completed_goals = Column(JSON)
    badges_earned = Column(JSON)
    
    # New fields for language-specific metrics
    language_metrics = Column(JSON)  # Metrics per language

class PyObjectId(ObjectId):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v, field_info):
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid objectid")
        return ObjectId(v)

    @classmethod
    def __get_pydantic_json_schema__(cls, core_schema, handler):
        json_schema = handler(core_schema)
        json_schema.update(
            type="string",
            examples=["5f9f1b9b9c9d1c0b8c8b8c8b"],
        )
        return json_schema

class UserDB(BaseModel):
    id: Optional[PyObjectId] = Field(default=None, alias="_id")
    username: str
    email: str
    hashed_password: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    preferred_language: str = Field(default="en")

    class Config:
        json_encoders = {ObjectId: str}
        arbitrary_types_allowed = True
        
class RecordingDB(BaseModel):
    id: Optional[PyObjectId] = Field(default=None, alias="_id")
    user_id: PyObjectId
    file_path: str
    session_type: str
    topic: str
    analysis_result: dict
    analysis_summary: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    detected_language: Optional[str] = None
    voice_cloning_score: Optional[float] = None
    transcription: Optional[str] = None

    class Config:
        json_encoders = {ObjectId: str}
        arbitrary_types_allowed = True

class PracticeSessionDB(BaseModel):
    id: Optional[PyObjectId] = Field(default=None, alias="_id")
    user_id: PyObjectId
    recording_id: PyObjectId
    session_type: str
    topic: str
    analysis_result: dict
    created_at: datetime = Field(default_factory=datetime.utcnow)
    language: Optional[str] = None
    voice_cloning_detected: str = Field(default="human")

    class Config:
        json_encoders = {ObjectId: str}
        arbitrary_types_allowed = True

class ProgressMetricsDB(BaseModel):
    id: Optional[PyObjectId] = Field(default=None, alias="_id")
    user_id: PyObjectId
    metric_date: datetime = Field(default_factory=datetime.utcnow)
    clarity_trend: float
    confidence_trend: float
    speech_rate_trend: float
    emotion_expression_score: float
    vocabulary_score: float
    overall_improvement: float
    current_goals: List[str]
    completed_goals: List[str]
    badges_earned: List[dict]
    language_metrics: Optional[Dict[str, dict]] = None

    class Config:
        json_encoders = {ObjectId: str}
        arbitrary_types_allowed = True

# New models for enhanced features
class LanguageDetectionResult(BaseModel):
    detected_language: str
    confidence: float
    language_name: str
    language_code: str

class VoiceCloningDetectionResult(BaseModel):
    is_ai_generated: bool
    confidence_score: float
    detection_method: str
    risk_level: str  # low, medium, high

class ExportRequest(BaseModel):
    format: str  # "pdf" or "csv"
    date_range: Optional[Dict[str, str]] = None
    include_transcriptions: bool = True
    include_voice_cloning: bool = True 