FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

ENV OPENBLAS_CORETYPE=ARMV8

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p /app/detections

ENV UDEV=1

CMD ["python", "src/avengers_assemble.py"]
