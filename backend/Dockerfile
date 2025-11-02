FROM python:3.11-slim

WORKDIR /app

# Copy requirements first for better caching
COPY backend/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the backend application
COPY backend/ .

# Expose port
EXPOSE $PORT

# Start command
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]