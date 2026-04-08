FROM python:3.11-slim

WORKDIR /app

# System dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl git && \
    rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Application code
COPY legal_env/ /app/legal_env/
COPY inference.py /app/inference.py
COPY openenv.yaml /app/openenv.yaml
COPY README.md /app/README.md

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Default: run the FastAPI server
CMD ["uvicorn", "legal_env.server.app:app", "--host", "0.0.0.0", "--port", "8000"]