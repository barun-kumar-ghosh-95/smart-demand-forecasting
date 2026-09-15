# Production Dockerfile for Smart Demand Forecasting Platform
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency specifications
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Set Python path
ENV PYTHONPATH=/app

# Expose FastAPI and Streamlit ports
EXPOSE 8000 8501

# Default command launches FastAPI service
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
