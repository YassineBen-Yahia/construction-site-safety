"""
FastAPI application for Construction Site Safety Monitoring
Provides endpoints for video processing, real-time monitoring, and safety scoring
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import os
import uuid
from datetime import datetime
import logging

from src.inference.inference import process_video
from src.inference.danger_scoring import score_safety

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Construction Site Safety Monitoring API",
    description="Real-time safety monitoring system for construction sites",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create directories for uploads and outputs
UPLOAD_DIR = Path(__file__).parent.parent / "uploads"
OUTPUT_DIR = Path(__file__).parent.parent / "outputs"
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Store processing jobs
processing_jobs = {}


@app.get("/health", tags=["Health"])
async def health_check():
    """
    Health check endpoint to verify API is running
    """
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "Construction Site Safety Monitoring API"
    }


@app.post("/api/v1/process-video", tags=["Video Processing"])
async def process_video_endpoint(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
):
    """
    Upload and process a video file for safety monitoring
    
    Parameters:
    - file: Video file (mp4, avi, mov, etc.)
    
    Returns:
    - job_id: Unique identifier for the processing job
    - status: Current status of the job
    - file_info: Information about the uploaded file
    """
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        # Check file size (limit to 500MB)
        file_size = 0
        contents = await file.read()
        file_size = len(contents)
        
        if file_size > 500 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="File too large (max 500MB)")
        
        # Generate unique job ID
        job_id = str(uuid.uuid4())
        
        # Save uploaded file
        file_ext = Path(file.filename).suffix
        input_path = UPLOAD_DIR / f"{job_id}_input{file_ext}"
        output_path = OUTPUT_DIR / f"{job_id}_output.mp4"
        
        with open(input_path, "wb") as f:
            f.write(contents)
        
        # Create job record
        processing_jobs[job_id] = {
            "status": "processing",
            "filename": file.filename,
            "input_path": str(input_path),
            "output_path": str(output_path),
            "created_at": datetime.now().isoformat(),
            "progress": 0,
            "stats": None,
            "error": None
        }
        
        # Add background task to process video
        if background_tasks:
            background_tasks.add_task(
                process_video_background,
                job_id,
                str(input_path),
                str(output_path)
            )
        
        return {
            "job_id": job_id,
            "status": "queued",
            "message": "Video queued for processing",
            "file_info": {
                "filename": file.filename,
                "size_mb": round(file_size / (1024 * 1024), 2)
            }
        }
    
    except Exception as e:
        logger.error(f"Error uploading video: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error uploading file: {str(e)}")


@app.get("/api/v1/job/{job_id}", tags=["Job Management"])
async def get_job_status(job_id: str):
    """
    Get the status of a processing job
    
    Parameters:
    - job_id: The unique identifier of the job
    
    Returns:
    - Job status and results
    """
    if job_id not in processing_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return processing_jobs[job_id]


@app.get("/api/v1/download/{job_id}", tags=["Download"])
async def download_output(job_id: str):
    """
    Download the processed video output
    
    Parameters:
    - job_id: The unique identifier of the job
    
    Returns:
    - The processed video file
    """
    if job_id not in processing_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = processing_jobs[job_id]
    
    if job["status"] != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Job not ready for download. Current status: {job['status']}"
        )
    
    output_path = Path(job["output_path"])
    
    if not output_path.exists():
        raise HTTPException(status_code=404, detail="Output file not found")
    
    return FileResponse(
        path=output_path,
        filename=f"safety_monitoring_{job_id}.mp4",
        media_type="video/mp4"
    )


@app.get("/api/v1/jobs", tags=["Job Management"])
async def list_jobs(status: str = None):
    """
    List all processing jobs
    
    Parameters:
    - status: Optional filter by status (processing, completed, failed)
    
    Returns:
    - List of jobs
    """
    jobs = processing_jobs
    
    if status:
        jobs = {k: v for k, v in jobs.items() if v["status"] == status}
    
    return {
        "total": len(jobs),
        "jobs": [
            {
                "job_id": k,
                "status": v["status"],
                "filename": v["filename"],
                "created_at": v["created_at"],
                "progress": v["progress"]
            }
            for k, v in jobs.items()
        ]
    }


@app.delete("/api/v1/job/{job_id}", tags=["Job Management"])
async def delete_job(job_id: str):
    """
    Delete a processing job and its files
    
    Parameters:
    - job_id: The unique identifier of the job
    
    Returns:
    - Success message
    """
    if job_id not in processing_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = processing_jobs[job_id]
    
    # Delete input and output files
    try:
        if Path(job["input_path"]).exists():
            Path(job["input_path"]).unlink()
        if Path(job["output_path"]).exists():
            Path(job["output_path"]).unlink()
    except Exception as e:
        logger.error(f"Error deleting files: {str(e)}")
    
    # Remove job record
    del processing_jobs[job_id]
    
    return {"message": f"Job {job_id} deleted successfully"}


@app.get("/api/v1/stats", tags=["Statistics"])
async def get_statistics():
    """
    Get overall statistics for all processed videos
    
    Returns:
    - Statistics about processing jobs
    """
    completed_jobs = [v for v in processing_jobs.values() if v["status"] == "completed"]
    
    total_jobs = len(processing_jobs)
    completed_count = len(completed_jobs)
    avg_score = 0
    
    if completed_jobs:
        scores = [v["stats"]["avg_score"] for v in completed_jobs if v["stats"]]
        if scores:
            avg_score = sum(scores) / len(scores)
    
    return {
        "total_jobs": total_jobs,
        "completed": completed_count,
        "processing": len([v for v in processing_jobs.values() if v["status"] == "processing"]),
        "failed": len([v for v in processing_jobs.values() if v["status"] == "failed"]),
        "average_safety_score": round(avg_score, 2),
        "uploads_dir": str(UPLOAD_DIR),
        "outputs_dir": str(OUTPUT_DIR)
    }


@app.post("/api/v1/analyze-frame", tags=["Analysis"])
async def analyze_frame(
    file: UploadFile = File(...)
):
    """
    Analyze a single image frame for safety risks
    
    Parameters:
    - file: Image file (jpg, png, etc.)
    
    Returns:
    - Safety analysis for the frame
    """
    try:
        import cv2
        import numpy as np
        from src.analysis.detection_pipeline import process_frame
        
        contents = await file.read()
        
        # Convert bytes to numpy array and decode image
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Process frame
        dangers_list = process_frame(frame)
        
        # Analyze each person
        analysis_results = []
        for danger_info in dangers_list:
            person_box = danger_info['person_box']
            detected_dangers = danger_info['dangers']
            ppe = danger_info['ppe']
            
            detection_output = {
                'dangers': detected_dangers,
                'ppe': ppe
            }
            
            score, _, risk_level = score_safety(detection_output)
            
            analysis_results.append({
                "person_id": len(analysis_results),
                "safety_score": score,
                "risk_level": risk_level,
                "dangers": detected_dangers,
                "missing_ppe": ppe,
                "position": {
                    "x1": int(person_box[0]),
                    "y1": int(person_box[1]),
                    "x2": int(person_box[2]),
                    "y2": int(person_box[3])
                }
            })
        
        return {
            "timestamp": datetime.now().isoformat(),
            "people_detected": len(analysis_results),
            "analysis": analysis_results
        }
    
    except Exception as e:
        logger.error(f"Error analyzing frame: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error analyzing frame: {str(e)}")


@app.get("/docs", tags=["Documentation"])
async def swagger_docs():
    """Swagger UI documentation"""
    pass


# Background task function
def process_video_background(job_id: str, input_path: str, output_path: str):
    """Process video in background"""
    try:
        logger.info(f"Starting video processing for job {job_id}")
        processing_jobs[job_id]["status"] = "processing"
        
        stats = process_video(input_path, output_path, display=False)
        
        processing_jobs[job_id]["status"] = "completed"
        processing_jobs[job_id]["stats"] = stats
        processing_jobs[job_id]["completed_at"] = datetime.now().isoformat()
        
        logger.info(f"Video processing completed for job {job_id}")
    
    except Exception as e:
        logger.error(f"Error processing video {job_id}: {str(e)}")
        processing_jobs[job_id]["status"] = "failed"
        processing_jobs[job_id]["error"] = str(e)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
