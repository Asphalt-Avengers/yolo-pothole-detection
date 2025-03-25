# Asphalt Avengers: Detections

This repository contains a Python-based object detection pipeline designed to work with the **Luxonis Oak-D** camera. It is currently configured for pothole detection. The code performs real-time object detection using a YOLO-based model, annotates detected frames, and saves the results to a directory as images and text files. It is designed to run inside a Docker container with USB access to the camera and a mounted directory for saving outputs.

---

## Features

- **Real-time object detection** with Luxonis DepthAI.
- Supports YOLOv5 models in OpenVINO format. [Look here](https://colab.research.google.com/github/luxonis/depthai-ml-training/blob/master/colab-notebooks/YoloV5_training.ipynb) to see how to train/export a model.
- Saves annotated images and detections in YOLO format.
- Configurable via a JSON file for model metadata and parameters.
- Docker image available on [Docker Hub](https://hub.docker.com/r/lmcardle/asphalt-avengers-detections).

---

## How to Build and Run

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/asphalt-avengers.git
cd asphalt-avengers
```

### 2. Build the Docker Image
```bash
docker build -t asphalt-avengers .
```

### 3. Run the Docker Container
Run the container, ensuring USB access is granted and outputs are saved to a mounted directory:
```bash
docker run --rm -it --privileged -v ${pwd}/models:/app/models -v /dev/bus/usb:/dev/bus/usb -v $(pwd)/detections:/app/detections -v $(pwd)/src:/app/src asphalt-avengers
```

### 4. Output Files
The `detections/` directory will contain:
- Annotated frames as `.jpg` files.
- Corresponding detections in YOLO format as `.txt` files.

---

## Configuration

The detection pipeline is configured via the JSON file (`best_m.json`). Key parameters include:
- **`input_size`**: Resolution of the input image (e.g., `224x224`).
- **`confidence_threshold`**: Minimum confidence for detections.
- **`classes`**: Number of classes in the model.

Update the JSON file to modify the pipeline settings or switch to a different model.

---

## Notes

- Ensure the `MODEL_PATH` in `avengers_assemble.py` points to a valid `.blob` model.

---

## Author

- **Liam McArdle**  
  GitHub: [lmcardle](https://github.com/LiamMcArdle)
