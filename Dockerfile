FROM python:3.10-slim

WORKDIR /app

COPY main.py /app/main.py

# Install dependencies
RUN pip install --no-cache-dir flask tensorflow reportlab pypdf pillow markdown gunicorn

# Expose port
EXPOSE 8080

# Run with gunicorn (Fly.io standard)
CMD ["gunicorn", "-b", "0.0.0.0:8080", "main:app"]
