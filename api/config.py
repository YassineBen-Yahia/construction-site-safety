"""
Configuration for FastAPI application
"""

import os
from pathlib import Path

# API Configuration
API_VERSION = "1.0.0"
API_TITLE = "Construction Site Safety Monitoring API"
API_DESCRIPTION = "Real-time safety monitoring system for construction sites using computer vision"

# File paths
BASE_DIR = Path(__file__).parent.parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
LOGS_DIR = BASE_DIR / "logs"

# File size limits (in bytes)
MAX_VIDEO_SIZE = 500 * 1024 * 1024  # 500MB
MAX_IMAGE_SIZE = 50 * 1024 * 1024   # 50MB

# Allowed file extensions
ALLOWED_VIDEO_EXTENSIONS = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.webm']
ALLOWED_IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.webp', '.bmp']

# Processing configuration
VIDEO_PROCESSING_TIMEOUT = 3600  # 1 hour in seconds
FRAME_BATCH_SIZE = 30  # Process frames in batches for display updates

# Database/Storage configuration
KEEP_PROCESSED_FILES = True  # Keep processed videos after completion
AUTO_DELETE_AFTER_DAYS = 7  # Auto-delete old files after N days

# Model paths
POSE_MODEL_PATH = BASE_DIR / "models" / "pose_estimation" / "yolov8n-pose.pt"
PPE_MODEL_PATH = BASE_DIR / "models" / "ppe" / "best.pt"

# Create directories if they don't exist
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# CORS settings
CORS_ORIGINS = ["*"]
CORS_CREDENTIALS = True
CORS_METHODS = ["*"]
CORS_HEADERS = ["*"]

# Logging configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Safety scoring thresholds
SAFETY_SCORE_THRESHOLDS = {
    "LOW": (80, 100),
    "MODERATE": (60, 79),
    "HIGH": (40, 59),
    "CRITICAL": (0, 39)
}

# Risk colors (BGR format for OpenCV)
RISK_COLORS = {
    "LOW": (0, 255, 0),        # Green
    "MODERATE": (0, 165, 255), # Orange
    "HIGH": (0, 69, 255),      # Red-Orange
    "CRITICAL": (0, 0, 255)    # Red
}
