FROM python:3.11-slim

WORKDIR /app

# Copy only requirements first for better caching
COPY backend/requirements.txt .

# Install dependencies efficiently
RUN pip install --no-cache-dir -r requirements.txt

# Copy only necessary backend files
COPY backend/main.py .
COPY backend/docs/ ./docs/
# Only copy essential directories
COPY backend/uploads/ ./uploads/

EXPOSE $PORT

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]