FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create data directory
RUN mkdir -p data

# Expose port for Hugging Face Spaces
EXPOSE 7860

# Run with gunicorn for production
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--timeout", "120", "--workers", "2", "app:app"]