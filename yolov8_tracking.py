import cv2
import numpy as np
from ultralytics import YOLO
from sort import Sort

# Load YOLOv8 model (pretrained)
model = YOLO("yolov8n.pt")

# Initialize tracker
tracker = Sort()

# Start video capture (0 = webcam, or file path)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]

    detections = []

    # Extract detections
    for box in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = box
        if score > 0.5:
            detections.append([x1, y1, x2, y2])

    tracks = tracker.update(detections)

    # Draw results
    for track_id, bbox in tracks:
        x1, y1, x2, y2 = map(int, bbox)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame,
                    f"ID {track_id}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2)

    cv2.imshow("YOLO + SORT Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()