import cv2
import numpy as np
from ultralytics import YOLO
import pathlib
import platform

if platform.system() != 'Windows':
    pathlib.WindowsPath = pathlib.PosixPath

model = YOLO('/Users/alessiomaggio/Msc Artificial Intelligence/Semester 2/Group software project/Test/Website/venv/best.pt')

def analyse_eyes(filepath):
    img = cv2.imread(filepath)  # Read directly from disk using the filepath

    if img is None:
        return "Failed to load image."

    results = model(img)

    detection = []
    for r in results:
        for box in r.boxes:
            detection.append({
                "label": model.names[int(box.cls)],
                "confidence": float(box.conf),
                "bbox": box.xyxy.tolist()
            })
    return detection  # Return the data, not a rendered template