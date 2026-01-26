FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (including OpenCV requirements)
RUN apt-get update && apt-get install -y \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglib2.0-0 \
    libgl1 \
    libgtk2.0-dev \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU version FIRST (before copying files for better caching)
RUN pip install --no-cache-dir \
    torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu

# Copy only requirements files first (better Docker layer caching)
COPY api/requirements.txt /app/api/requirements.txt
COPY requirements.txt /app/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir --timeout=1000 -r api/requirements.txt && \
    pip install --no-cache-dir --timeout=1000 -r requirements.txt

# Copy rest of project files
COPY . /app/

# Create necessary directories
RUN mkdir -p uploads outputs logs models/ppe models/pose_estimation

# Expose port
EXPOSE 8000

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV LOG_LEVEL=INFO

# Run the API
CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]