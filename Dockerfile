# Use lightweight Python 3.11 base image
FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Install system dependencies needed by some Python packages
# (build-essential is needed for twofish to compile on Linux)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (separate layer for caching)
# This way, code changes don't trigger reinstalling all packages
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy source code (changes most often — comes last for caching)
COPY src/ ./src/

# Default command — overridden by docker-compose for each pipeline
CMD ["python", "-m", "src.feature_pipeline", "--mode", "incremental"]