# Real-Time Object Detection and Tracking System

## Description

This project performs real-time object detection and tracking using YOLOv8 and OpenCV. The system detects objects from a webcam feed and assigns unique tracking IDs using the SORT tracking algorithm.

## Features

* Real-time object detection
* Object tracking with unique IDs
* Webcam support
* Bounding box visualization
* YOLOv8 pre-trained model

## Requirements

* Python 3.12+
* OpenCV
* NumPy
* Ultralytics (YOLOv8)
* FilterPy
* SciPy

## Installation

1. Clone the repository:

```bash
git clone <repository-link>
cd object_tracking
```

2. Install dependencies:

```bash
pip install opencv-python numpy ultralytics filterpy scipy
```

## How to Run

Run the following command:

```bash
python yolov8_tracking.py
```

## Controls

* Press **Q** to exit the application.
* Press **Ctrl + C** in the terminal to stop the program.

## Project Files

* `yolov8_tracking.py` - Main application
* `sort.py` - Object tracking algorithm
* `README.md` - Project documentation

## Output

The system displays:

* Detected objects
* Bounding boxes
* Tracking IDs

Example:

ID 0 - Person
ID 1 - Bottle
ID 2 - Mobile Phone

## Developed By

Bhargav Deshpande
