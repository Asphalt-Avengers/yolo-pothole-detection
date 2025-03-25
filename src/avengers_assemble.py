#!/usr/bin/env python3

import cv2
import depthai as dai
import numpy as np
import os
import json
import uuid
import threading
import time
import serial
import pynmea2
import signal
import atexit
from gps import get_gps_coordinates  # We still keep the existing helper for fallback or reference

# --- Global GPS variables ---
current_gps = (43.473092, -80.539608)  # Initial/default coordinates (E7 LMAO)
gps_lock = threading.Lock()
gps_stop_event = threading.Event()

def gps_polling_thread(port="/dev/ttyAMA0", baudrate=9600, timeout=0.5):
    """
    Continuously reads from the GPS module and updates global coordinates.
    If the GPS module is not connected or no valid signal is received, the coordinates remain at their default.
    """
    global current_gps
    try:
        ser = serial.Serial(port, baudrate=baudrate, timeout=timeout)
        print("GPS thread: Serial port opened.")
    except serial.SerialException as e:
        print(f"GPS thread: Serial error on startup: {e}")
        return

    while not gps_stop_event.is_set():
        try:
            newdata = ser.readline().decode("utf-8", errors="ignore")
        except Exception as e:
            print("GPS thread: Error reading from serial:", e)
            time.sleep(0.5)
            continue

        if newdata and newdata.startswith("$GPRMC"):
            try:
                newmsg = pynmea2.parse(newdata)
                lat, lon = newmsg.latitude, newmsg.longitude
                with gps_lock:
                    current_gps = (lat, lon)
            except pynmea2.ParseError:
                continue
        time.sleep(0.1)

# --- Configuration ---
MODEL_PATH = 'models/blobs/yolov8s-416.blob'
CONFIG_PATH = 'models/config/yolov8s-416.json'
OUTPUT_DIR = "detections"

print("Setting up OAK-D...")

with open(CONFIG_PATH) as f:
    config = json.load(f)
nnConfig = config.get("nn_config", {})
W, H = map(int, nnConfig.get("input_size").split('x'))

metadata = nnConfig.get("NN_specific_metadata", {})
confidenceThreshold = metadata.get("confidence_threshold", 0.75)
classes = metadata.get("classes", 80)
coordinates = metadata.get("coordinates", 4)
anchors = metadata.get("anchors", [])
anchorMasks = metadata.get("anchor_masks", {})
iouThreshold = metadata.get("iou_threshold", 0.5)

labels = config.get("mappings", {}).get("labels", [])

os.makedirs(OUTPUT_DIR, exist_ok=True)

pipeline = dai.Pipeline()
camRgb = pipeline.create(dai.node.ColorCamera)
detectionNetwork = pipeline.create(dai.node.YoloDetectionNetwork)
xoutRgb = pipeline.create(dai.node.XLinkOut)
nnOut = pipeline.create(dai.node.XLinkOut)

xoutRgb.setStreamName("rgb")
nnOut.setStreamName("nn")

camRgb.setPreviewSize(W, H)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
camRgb.setInterleaved(False)
camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
camRgb.setFps(5)

detectionNetwork.setConfidenceThreshold(confidenceThreshold)
detectionNetwork.setNumClasses(classes)
detectionNetwork.setCoordinateSize(coordinates)
detectionNetwork.setAnchors(anchors)
detectionNetwork.setAnchorMasks(anchorMasks)
detectionNetwork.setIouThreshold(iouThreshold)
detectionNetwork.setBlobPath(MODEL_PATH)

camRgb.preview.link(detectionNetwork.input)
detectionNetwork.passthrough.link(xoutRgb.input)
detectionNetwork.out.link(nnOut.input)

def frameNorm(frame, bbox):
    normVals = np.full(len(bbox), frame.shape[0])
    normVals[::2] = frame.shape[1]
    return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)

def annotateFrame(frame, detections):
    for det in detections:
        bbox = frameNorm(frame, (det.xmin, det.ymin, det.xmax, det.ymax))
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
        label_text = f"{labels[det.label]}: {det.confidence:.2f}" if labels else f"{det.label}: {det.confidence:.2f}"
        cv2.putText(frame, label_text, (bbox[0], bbox[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    return frame

def saveDetections(frame, detections, lat, lon, frame_id):
    detection_dir = os.path.join(OUTPUT_DIR, f"detection_{frame_id}")
    os.makedirs(detection_dir, exist_ok=True)
    image_path = os.path.join(detection_dir, "frame.jpg")
    json_path = os.path.join(detection_dir, "metadata.json")
    cv2.imwrite(image_path, frame)
    detection_data = {
        "latitude": lat,
        "longitude": lon,
        "potholes": [
            {
                "label": det.label,
                "confidence": det.confidence,
                "box": {
                    "x_min": det.xmin,
                    "y_min": det.ymin,
                    "x_max": det.xmax,
                    "y_max": det.ymax
                }
            } for det in detections
        ]
    }
    with open(json_path, "w") as f:
        json.dump(detection_data, f, indent=2)
    print(f"Saved detection: {image_path} and {json_path}")

# --- Resource Cleanup ---
video_writer = None  # Global video writer variable
# Create a unique filename for the video using a timestamp
video_filename = f"realtime_output_{time.strftime('%Y%m%d_%H%M%S')}.mp4"

def cleanup():
    global video_writer
    print("Cleaning up resources...")
    if video_writer is not None:
        video_writer.release()
        print("Video writer released.")
    cv2.destroyAllWindows()
    gps_stop_event.set()

atexit.register(cleanup)

def signal_handler(sig, frame):
    print(f"Received signal {sig}. Exiting gracefully...")
    cleanup()
    exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

gps_thread = threading.Thread(target=gps_polling_thread, daemon=True)
gps_thread.start()

# --- Main Loop ---
try:
    with dai.Device(pipeline) as device:
        qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
        qDet = device.getOutputQueue(name="nn", maxSize=4, blocking=False)
        frame_id = uuid.uuid4()
        print("Device is running...")

        while True:
            inRgb = qRgb.get()
            inDet = qDet.get()
            frame = inRgb.getCvFrame()
            
            if video_writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                video_writer = cv2.VideoWriter(video_filename, fourcc, 30, (frame.shape[1], frame.shape[0]))
                print(f"Video writer initialized with file: {video_filename}")

            video_writer.write(frame)
            detections = inDet.detections

            if detections:
                with gps_lock:
                    lat, lon = current_gps
                annotated_frame = annotateFrame(frame.copy(), detections)
                saveDetections(annotated_frame, detections, lat, lon, frame_id)
                frame_id = uuid.uuid4()

            time.sleep(0.01)
except Exception as e:
    print("Exception encountered:", e)
finally:
    cleanup()
