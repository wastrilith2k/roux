# Companion Code Executor - Sandboxed Python execution + persistent browser
FROM python:3.12-slim

WORKDIR /app

# Install system dependencies (including Playwright browser deps)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Playwright and Chromium browser
RUN pip install --no-cache-dir playwright==1.49.0
RUN playwright install chromium
RUN playwright install-deps chromium

# Copy application code
COPY app.py .
COPY sandbox.py .
COPY browser_manager.py .

# Copy tool modules (also mounted as volume for dev)
COPY tools/ ./tools/

# Expose port
EXPOSE 5001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:5001/health || exit 1

# Run the Flask app (threaded for concurrent /execute and /browser/* handling)
CMD ["python", "app.py"]
