# ============================================================================
# Stage 1: Base system with Node.js and system packages
# This layer changes very rarely (only when we need new system packages)
# ============================================================================
FROM python:3.12-slim AS base-system

# Disable Python output buffering for real-time logs
ENV PYTHONUNBUFFERED=1

# Force gRPC to use the asyncio eventloop for compatibility with eventlet
ENV GRPC_PYTHON_DISABLE_LIBC_COMPATIBILITY=1
ENV GRPC_ENABLE_FORK_SUPPORT=1

# Install Node.js and npm for MCP servers that require npx/uvx
RUN apt-get update && apt-get install -y \
    nodejs \
    npm \
    curl \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install uv (modern Python package manager) for uvx command
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /app


# ============================================================================
# Stage 2: Python dependencies
# ============================================================================
FROM base-system AS deps

COPY requirements-base.txt .
RUN pip install --no-cache-dir -r requirements-base.txt


# ============================================================================
# Stage 3: Final runtime image
# Only this stage rebuilds when code changes
# ============================================================================
FROM deps AS runtime

# Copy the entire src/ directory to preserve structure
COPY src/ /app/src/

# Copy prompts directory
COPY prompts/ /app/prompts/

# Note: data/ directory is mounted as a volume, not copied

# Config files
COPY mcp_config.json .

# Copy Alembic configuration and migrations
COPY alembic.ini .
COPY migrations/ migrations/

# Expose port for web interface
EXPOSE 5000

# Command to start the script is defined in docker-compose.yaml
