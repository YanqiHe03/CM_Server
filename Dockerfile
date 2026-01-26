# RunPod Serverless Dockerfile for Complimentary Machine VLM
# Base image with CUDA and Python support

FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/runpod-volume/huggingface \
    TRANSFORMERS_CACHE=/runpod-volume/huggingface

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better Docker layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy handler code
COPY handler.py .

# Pre-download model during build (optional - reduces cold start time)
# Uncomment and replace with your HF model ID if you want to bake the model into the image
# ARG HF_MODEL_ID=your-username/cm-gallery-vlm
# ARG HF_TOKEN
# RUN python -c "from transformers import AutoModelForVision2Seq, AutoProcessor; \
#     AutoModelForVision2Seq.from_pretrained('${HF_MODEL_ID}', trust_remote_code=True, token='${HF_TOKEN}'); \
#     AutoProcessor.from_pretrained('${HF_MODEL_ID}', trust_remote_code=True, token='${HF_TOKEN}')"

# Start the handler
CMD ["python", "-u", "handler.py"]
