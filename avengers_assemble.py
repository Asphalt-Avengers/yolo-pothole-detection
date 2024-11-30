#!/usr/bin/env python3

import cv2
import depthai as dai
import numpy as np
import os
import json

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
def saveDetections(frame, detections, frame_count):
    image_path = f"{OUTPUT_DIR}/detection_{frame_count}.jpg"
    txt_path = f"{OUTPUT_DIR}/detection_{frame_count}.txt"

    # Save image
    cv2.imwrite(image_path, frame)

    # Save detections
    with open(txt_path, "w") as f:
        for det in detections:
            bbox = (det.xmin, det.ymin, det.xmax, det.ymax)
            f.write(f"{det.label} {det.confidence:.6f} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n")

    print(f"Saved detection: {image_path} and {txt_path}")

# Run pipeline
with dai.Device(pipeline) as device:
    qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    qDet = device.getOutputQueue(name="nn", maxSize=4, blocking=False)

    frame_count = 0
    print("Device is running...")
    while True:
        inRgb = qRgb.get()
        inDet = qDet.get()

        frame = inRgb.getCvFrame()
        detections = inDet.detections

        if detections:
            annotated_frame = annotateFrame(frame, detections)
            saveDetections(annotated_frame, detections, frame_count)
            frame_count += 1
