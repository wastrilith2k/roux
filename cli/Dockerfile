FROM python:3.11-slim

WORKDIR /app

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy CLI code
COPY . /app/

# Make sure script is executable
RUN chmod +x /app/companion_cli.py

# Default command - run the CLI
# Can be overridden with docker run
CMD ["python3", "/app/companion_cli.py"]
