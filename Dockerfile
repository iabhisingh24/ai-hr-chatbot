# Multi-stage build to reduce final image size
FROM python:3.11-slim as builder

WORKDIR /app

# Copy only requirements first for better caching
COPY backend/requirements.txt .

# Install dependencies
RUN pip install --user -r requirements.txt

# Final stage
FROM python:3.11-slim

WORKDIR /app

# Copy only the installed packages from builder stage
COPY --from=builder /root/.local /root/.local
COPY backend/ .

# Make sure Python finds the user-installed packages
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONPATH=/app

# Create necessary directories
RUN mkdir -p uploads

EXPOSE $PORT

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]