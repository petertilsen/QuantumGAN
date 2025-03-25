FROM python:3.9-slim

WORKDIR /app

# Install system dependencies required for TensorFlow Quantum
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY quantum_gan.py train.py ./

# Create directory for output files
RUN mkdir -p /app/output

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Command to run when container starts
CMD ["python", "train.py"]
