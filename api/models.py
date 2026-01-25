"""
Pydantic models for API request/response validation
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from datetime import datetime


class FrameAnalysis(BaseModel):
    """Analysis result for a single person in a frame"""
    person_id: int
    safety_score: float = Field(..., ge=0, le=100)
    risk_level: str
    dangers: List[str]
    missing_ppe: List[str]
    position: Dict[str, int]


class FrameAnalysisResponse(BaseModel):
    """Response for single frame analysis"""
    timestamp: datetime
    people_detected: int
    analysis: List[FrameAnalysis]


class VideoProcessingRequest(BaseModel):
    """Request for video processing"""
    filename: str
    size_mb: float


class JobStatus(BaseModel):
    """Status of a processing job"""
    job_id: str
    status: str  # 'queued', 'processing', 'completed', 'failed'
    filename: str
    created_at: datetime
    progress: float = Field(0, ge=0, le=100)
    error: Optional[str] = None


class VideoProcessingResponse(BaseModel):
    """Response for video upload"""
    job_id: str
    status: str
    message: str
    file_info: VideoProcessingRequest


class JobListResponse(BaseModel):
    """Response for job listing"""
    total: int
    jobs: List[JobStatus]


class ProcessingStats(BaseModel):
    """Statistics from video processing"""
    total_frames: int
    frames_processed: int
    avg_score: float
    min_score: Optional[float]
    max_score: Optional[float]
    risk_distribution: Dict[str, int]


class JobDetailResponse(BaseModel):
    """Detailed job information"""
    job_id: str
    status: str
    filename: str
    input_path: str
    output_path: str
    created_at: datetime
    progress: float
    stats: Optional[ProcessingStats]
    error: Optional[str]


class HealthCheckResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: datetime
    service: str


class StatisticsResponse(BaseModel):
    """Overall API statistics"""
    total_jobs: int
    completed: int
    processing: int
    failed: int
    average_safety_score: float
    uploads_dir: str
    outputs_dir: str


class DangerInfo(BaseModel):
    """Information about detected dangers"""
    danger_type: str
    severity: str
    description: str


class SafetyReport(BaseModel):
    """Complete safety report for a video"""
    job_id: str
    video_filename: str
    processing_time_seconds: float
    average_safety_score: float
    total_frames: int
    people_detected: int
    danger_summary: Dict[str, int]
    risk_distribution: Dict[str, int]
