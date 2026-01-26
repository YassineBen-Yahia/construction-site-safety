FROM python:3.11-slim


#RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy project files    
COPY . /app/

# Install Python dependencies
RUN pip install --no-cache-dir -r api/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Create necessary directories
RUN mkdir -p uploads outputs logs models/ppe models/pose_estimation

# Expose port
EXPOSE 8000

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV LOG_LEVEL=INFO

# Run the API
CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
