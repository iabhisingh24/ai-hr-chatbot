FROM python:3.11-slim

WORKDIR /app

# Copy only requirements first for better caching
COPY backend/requirements.txt .

# Install dependencies efficiently
RUN pip install --no-cache-dir -r requirements.txt

# Copy only the main application file (guaranteed to exist)
COPY backend/main.py .

# Create necessary directories at runtime
RUN mkdir -p uploads docs

EXPOSE $PORT

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]