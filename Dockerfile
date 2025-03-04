FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir numpy opencv-python-headless depthai depthai-sdk pyserial pynmea2 

ENV OPENBLAS_CORETYPE=ARMV8

WORKDIR /app

COPY . .

RUN mkdir -p /app/detections

ENV UDEV=1

CMD ["python", "avengers_assemble.py"]
