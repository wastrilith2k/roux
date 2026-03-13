# Companion Code Executor - Sandboxed Python execution container
FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .
COPY sandbox.py .

# Copy tool modules (also mounted as volume for dev)
COPY companion_tools/ ./companion_tools/

# Create non-root user for security
RUN useradd -m -s /bin/bash executor
RUN chown -R executor:executor /app

# Don't run as root
USER executor

# Expose port
EXPOSE 5001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5001/health || exit 1

# Run the Flask app
CMD ["python", "app.py"]
