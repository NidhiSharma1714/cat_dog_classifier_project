# # ----------------------------
# # Stage 1: Builder
# # ----------------------------
# FROM python:3.10-slim AS builder

# WORKDIR /app

# # Install build dependencies
# RUN apt-get update && apt-get install -y \
#     build-essential \
#     libjpeg62-turbo-dev \
#     zlib1g-dev \
#     curl \
#     && rm -rf /var/lib/apt/lists/*

# # Copy requirements
# COPY requirements.txt .

# # Install Python dependencies (including tflite-runtime)
# RUN pip install --no-cache-dir -r requirements.txt

# # ----------------------------
# # Stage 2: Runtime
# # ----------------------------
# FROM python:3.10-slim

# WORKDIR /app

# # Minimal runtime libraries
# RUN apt-get update && apt-get install -y \
#     libjpeg62-turbo-dev \
#     zlib1g-dev \
#     && rm -rf /var/lib/apt/lists/*

# # Copy installed packages from builder
# COPY --from=builder /usr/local /usr/local

# # Copy app and model
# COPY app.py predict.py ./ 
# COPY cat_dog_classifier.tflite ./

# EXPOSE 8000

# CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
# Use a slim Python base
# Stage 1: Build dependencies
# Stage 1: build environment
FROM python:3.10-slim AS builder

WORKDIR /app

# System deps for Pillow
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libjpeg-dev zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Stage 2: minimal runtime
FROM python:3.10-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy app code and TFLite model
COPY app.py .
COPY predict.py .
COPY cat_dog_classifier.tflite .

# Expose port
EXPOSE 8000

# Run FastAPI app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

