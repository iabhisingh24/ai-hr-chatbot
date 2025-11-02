FROM python:3.11-slim

WORKDIR /app

# Copy only requirements first for better caching
COPY backend/requirements.txt .

# Install dependencies efficiently
RUN pip install --no-cache-dir -r requirements.txt

# Copy main application file
COPY backend/main.py .

# Copy docs directory (if it exists)
COPY backend/docs/ ./docs/

# Create uploads directory if it doesn't exist (but don't copy content)
RUN mkdir -p uploads

EXPOSE $PORT

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]