#!/usr/bin/env python3

import cv2
import depthai as dai
import numpy as np
import os
import json
import uuid
from gps import get_gps_coordinates

# Configuration
MODEL_PATH = 'best_openvino_2022.1_6shave_m.blob'
CONFIG_PATH = 'best_m.json'
OUTPUT_DIR = "detections"

print("Setting up OAK-D...")

# Parse config
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

# Parse labels
labels = config.get("mappings", {}).get("labels", [])

# Prepare output directories
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Create pipeline
pipeline = dai.Pipeline()

# Define nodes
camRgb = pipeline.create(dai.node.ColorCamera)
detectionNetwork = pipeline.create(dai.node.YoloDetectionNetwork)
xoutRgb = pipeline.create(dai.node.XLinkOut)
nnOut = pipeline.create(dai.node.XLinkOut)

xoutRgb.setStreamName("rgb")
nnOut.setStreamName("nn")

# Camera settings
camRgb.setPreviewSize(W, H)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
camRgb.setInterleaved(False)
camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
camRgb.setFps(30)

# Network settings
detectionNetwork.setConfidenceThreshold(confidenceThreshold)
detectionNetwork.setNumClasses(classes)
detectionNetwork.setCoordinateSize(coordinates)
detectionNetwork.setAnchors(anchors)
detectionNetwork.setAnchorMasks(anchorMasks)
detectionNetwork.setIouThreshold(iouThreshold)
detectionNetwork.setBlobPath(MODEL_PATH)

# Linking
camRgb.preview.link(detectionNetwork.input)
detectionNetwork.passthrough.link(xoutRgb.input)
detectionNetwork.out.link(nnOut.input)

# Normalize bounding box
def frameNorm(frame, bbox):
    normVals = np.full(len(bbox), frame.shape[0])
    normVals[::2] = frame.shape[1]
    return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)

# Annotate frame
def annotateFrame(frame, detections):
    for det in detections:
        bbox = frameNorm(frame, (det.xmin, det.ymin, det.xmax, det.ymax))
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
        label = f"{labels[det.label]}: {det.confidence:.2f}" if labels else f"{det.label}: {det.confidence:.2f}"
        cv2.putText(frame, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    return frame

# Save predictions
def saveDetections(frame, detections, lat, lon, frame_id):
    detection_dir = f"{OUTPUT_DIR}/detection_{frame_id}"
    image_path = f"{detection_dir}/frame.jpg"
    json_path = f"{detection_dir}/metadata.json"

    # Save image
    cv2.imwrite(image_path, frame)

    # Save detection data as JSON
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

# Run pipeline
with dai.Device(pipeline) as device:
    qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    qDet = device.getOutputQueue(name="nn", maxSize=4, blocking=False)

    frame_id = uuid.uuid4()
    print("Device is running...")
    while True:
        inRgb = qRgb.get()
        inDet = qDet.get()

        frame = inRgb.getCvFrame()
        detections = inDet.detections
        lat, lon = get_gps_coordinates()

        if detections:
            annotated_frame = annotateFrame(frame, detections)
            saveDetections(annotated_frame, detections, lat, lon, frame_id)
            frame_id = uuid.uuid4()
