FROM python:3.11-slim

WORKDIR /app

# System deps:
# - tesseract-ocr + language packs: OCR for scanned PDFs/images
# - libgl1: Pillow/PyMuPDF native deps in some environments
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        tesseract-ocr \
        tesseract-ocr-eng \
        tesseract-ocr-tur \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt /app/requirements.txt
RUN python -m pip install -U pip \
    && python -m pip install --no-cache-dir -r /app/requirements.txt

# App source
COPY . /app

EXPOSE 8000

# Chainlit default port is 8000; bind to 0.0.0.0 for Docker.
CMD ["python", "-m", "chainlit", "run", "app.py", "-w", "--host", "0.0.0.0", "--port", "8000"]

