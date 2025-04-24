# Base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:${PATH}"

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies using uv
RUN uv pip install --no-cache -r requirements.txt

# Copy project files
COPY . .

# Install the package
RUN pip install -e .

# Expose ports
EXPOSE 8000 8501

# Create volume for data
VOLUME /app/data

# Create volume for models
VOLUME /app/models

# Create entrypoint script
RUN echo '#!/bin/bash\n\
if [ "$1" = "api" ]; then\n\
    exec uvicorn api.main:app --host 0.0.0.0 --port 8000\n\
elif [ "$1" = "dashboard" ]; then\n\
    exec streamlit run dashboard/app.py --server.port 8501 --server.address 0.0.0.0\n\
elif [ "$1" = "train" ]; then\n\
    exec python -m src.cli train\n\
elif [ "$1" = "evaluate" ]; then\n\
    exec python -m src.cli evaluate\n\
else\n\
    exec "$@"\n\
fi' > /app/entrypoint.sh && chmod +x /app/entrypoint.sh

# Set entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]

# Default command
CMD ["api"]
