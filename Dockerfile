# Use Python 3.11 slim image as base
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        libpq-dev \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install poetry==1.7.1

# Copy poetry files
COPY pyproject.toml poetry.lock ./

# Export dependencies to requirements.txt and install with pip
RUN poetry export -f requirements.txt --output requirements.txt --without dev \
    && pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/

# Add src directory to Python path
ENV PYTHONPATH="/app/src:$PYTHONPATH"

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser
RUN chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["python", "-m", "uvicorn", "chives_backend.main:app", "--host", "0.0.0.0", "--port", "8000"]