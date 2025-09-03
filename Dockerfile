FROM python:3.10-slim

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt /app/requirements.txt

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY main.py /app/main.py

# Expose port
EXPOSE 8080

# Run with gunicorn (Fly.io standard)
CMD ["gunicorn", "-b", "0.0.0.0:8080", "main:app"]
